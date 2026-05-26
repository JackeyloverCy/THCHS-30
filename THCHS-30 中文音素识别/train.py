"""
THCHS30 中文语音识别
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

# ==================== 全局配置 ====================
TRAIN_DIR = r"D:\PythonProjects\data_thchs30\train"
TEST_DIR  = r"D:\PythonProjects\data_thchs30\test"

# ---------- 快速测试参数 (正式训练请改为注释中的值) ----------
BATCH_SIZE = 16                      # 正式: 16 测试 8
NUM_EPOCHS = 50                     # 正式: 50 测试 10
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
SAMPLE_RATE = 16000
N_MELS = 80                         # 正式: 80 测试 40
HIDDEN_SIZE = 512                   # 正式: 512 测试 128
NUM_LAYERS = 3                      # 正式: 3  测试 1
DROPOUT = 0.5                       # 正式: 0.5 (单层时不用dropout) 测试0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOP_PATIENCE = 5
GRAD_CLIP = 5.0

MAX_TRAIN_SIZE = None               # 正式: None 测试1000
MAX_TEST_SIZE = None                 # 正式: None 测试200

SPEC_AUG_FREQ_MASK = 10              # 正式: 10 测试 5
SPEC_AUG_TIME_MASK = 20             # 正式: 20 测试 10

NUM_WORKERS = 0                     # Windows 下使用 0 避免多进程问题
# ========================================================

torch.backends.cudnn.benchmark = True
scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else None

# ==================== 工具函数 ====================
def remove_tone(pinyin_str):
    pinyin = re.sub(r'[1-5]', '', pinyin_str)
    return ' '.join(pinyin.split())

# ==================== 1. 数据解析 ====================
def parse_links(directory):
    samples = []
    for item in os.listdir(directory):
        if not item.endswith('.wav'):
            continue
        wav_path = os.path.join(directory, item)
        trn_path = wav_path + '.trn'

        real_wav = wav_path
        try:
            with open(wav_path, 'r', encoding='utf-8') as f:
                content = f.read(200).strip()
            if content.startswith('../') or content.startswith('..\\'):
                real_wav = os.path.normpath(os.path.join(directory, content))
        except (UnicodeDecodeError, FileNotFoundError):
            pass

        if not os.path.exists(trn_path):
            continue
        real_trn = trn_path
        try:
            with open(trn_path, 'r', encoding='utf-8') as f:
                content = f.read(200).strip()
            if content.startswith('../') or content.startswith('..\\'):
                real_trn = os.path.normpath(os.path.join(directory, content))
        except (UnicodeDecodeError, FileNotFoundError):
            pass

        if not os.path.exists(real_trn):
            continue

        with open(real_trn, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')

        pinyin = ''
        hanzi = ''
        for line in lines:
            line = line.strip()
            if re.search(r'[1-5]', line):
                pinyin = line
            else:
                if not hanzi:
                    hanzi = line
        if not pinyin and len(lines) > 0:
            pinyin = lines[0].strip()
            if len(lines) > 1:
                hanzi = lines[1].strip()

        pinyin_clean = remove_tone(pinyin)
        samples.append((real_wav, pinyin_clean, hanzi))
    return samples

# ==================== 主程序保护 ====================
if __name__ == '__main__':
    train_samples = parse_links(TRAIN_DIR)
    test_samples  = parse_links(TEST_DIR)

    if MAX_TRAIN_SIZE:
        train_samples = train_samples[:MAX_TRAIN_SIZE]
    if MAX_TEST_SIZE:
        test_samples = test_samples[:MAX_TEST_SIZE]

    print(f"训练样本数: {len(train_samples)}, 测试样本数: {len(test_samples)}")
    print("示例标签（前60字符）:", train_samples[0][1][:60])

    # ==================== 2. 构建音素字典 ====================
    def build_vocab(label_lines):
        vocab = {'<pad>': 0}
        idx = 1
        for line in label_lines:
            for token in line.strip().split():
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        return vocab

    all_labels = [s[1] for s in train_samples]
    vocab = build_vocab(all_labels)
    print(f"音素类别数（含 blank）: {len(vocab)}")
    vocab_inv = {v: k for k, v in vocab.items()}

    # ==================== 3. Dataset ====================
    class THCHS30Dataset(Dataset):
        def __init__(self, samples, vocab, sr=SAMPLE_RATE, n_mels=N_MELS, augment=False):
            self.samples = samples
            self.vocab = vocab
            self.sr = sr
            self.augment = augment
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_mels=n_mels,
                n_fft=400, hop_length=160, win_length=400
            )
            self.valid_indices = []
            for idx, (wav_path, label_str, _) in enumerate(samples):
                if self._check_valid(wav_path, label_str):
                    self.valid_indices.append(idx)
            print(f"有效样本数 (标签长度 < 特征长度): {len(self.valid_indices)}")

        def _check_valid(self, wav_path, label_str):
            try:
                info = sf.info(wav_path)
                feat_len = int(info.duration * 100)
                label_len = len(label_str.split())
                return label_len + 5 <= feat_len
            except:
                return False

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            real_idx = self.valid_indices[idx]
            wav_path, label_str, _ = self.samples[real_idx]

            waveform_np, fs = sf.read(wav_path)
            if waveform_np.ndim > 1:
                waveform_np = waveform_np.mean(axis=1)
            waveform = torch.from_numpy(waveform_np).float()
            if fs != self.sr:
                waveform = torchaudio.functional.resample(waveform.unsqueeze(0), fs, self.sr).squeeze(0)

            mel = self.mel_transform(waveform)
            log_mel = torch.log(mel + 1e-9)
            features = log_mel.transpose(0, 1)

            if self.augment:
                features = self._spec_augment(features)

            label_indices = [self.vocab.get(tok, 0) for tok in label_str.split()]
            label_tensor = torch.tensor(label_indices, dtype=torch.long)
            return features, label_tensor

        def _spec_augment(self, features):
            freq_mask = torchaudio.transforms.FrequencyMasking(SPEC_AUG_FREQ_MASK)
            time_mask = torchaudio.transforms.TimeMasking(SPEC_AUG_TIME_MASK)
            features = features.transpose(0, 1)
            features = freq_mask(features)
            features = time_mask(features)
            features = features.transpose(0, 1)
            return features

    def collate_fn(batch):
        features, labels = zip(*batch)
        feat_lens = torch.tensor([f.size(0) for f in features])
        max_feat_len = feat_lens.max()
        label_lens = torch.tensor([l.size(0) for l in labels])
        max_label_len = label_lens.max()

        padded_feat = torch.zeros(len(features), max_feat_len, features[0].size(1))
        for i, f in enumerate(features):
            padded_feat[i, :f.size(0), :] = f

        padded_labels = torch.zeros(len(labels), max_label_len, dtype=torch.long)
        for i, l in enumerate(labels):
            padded_labels[i, :l.size(0)] = l

        return padded_feat, feat_lens, padded_labels, label_lens

    train_dataset = THCHS30Dataset(train_samples, vocab, augment=True)
    test_dataset  = THCHS30Dataset(test_samples, vocab, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )

    # ==================== 4. 模型 ====================
    class CTCASRModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=False, dropout=0.5):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                                num_layers=num_layers, dropout=dropout if num_layers>1 else 0.0,
                                bidirectional=bidirectional, batch_first=True)
            lstm_out = hidden_dim * (2 if bidirectional else 1)
            self.fc = nn.Linear(lstm_out, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out)

    # ==================== 5. 工具函数 ====================
    def decode_predictions(log_probs, vocab_inv):
        _, idx = torch.max(log_probs, dim=2)
        preds = []
        for seq in idx:
            decoded = []
            prev = -1
            for i in seq:
                if i != prev and i != 0:
                    decoded.append(vocab_inv.get(i.item(), '<unk>'))
                prev = i
            preds.append(' '.join(decoded) if decoded else '')
        return preds

    def token_error_rate(pred_strs, target_strs):
        total_err, total_len = 0, 0
        for p, t in zip(pred_strs, target_strs):
            p_tok = p.split()
            t_tok = t.split()
            m, n = len(p_tok), len(t_tok)
            if n == 0:
                if m == 0: continue
                total_err += m; continue
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m+1): dp[i][0] = i
            for j in range(n+1): dp[0][j] = j
            for i in range(1, m+1):
                for j in range(1, n+1):
                    cost = 0 if p_tok[i-1] == t_tok[j-1] else 1
                    dp[i][j] = min(dp[i-1][j-1] + cost, dp[i-1][j] + 1, dp[i][j-1] + 1)
            total_err += dp[m][n]
            total_len += n
        return total_err / total_len if total_len > 0 else 1.0

    def train_one_epoch(model, loader, optimizer, ctc_loss, clip=5.0):
        model.train()
        total_loss = 0
        for feat, feat_len, label, label_len in tqdm(loader, desc='Train'):
            feat, label = feat.to(DEVICE), label.to(DEVICE)
            feat_len, label_len = feat_len.to(DEVICE), label_len.to(DEVICE)

            optimizer.zero_grad()
            with autocast('cuda') if DEVICE.type == 'cuda' else torch.no_grad():
                logits = model(feat)
                log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
                loss = ctc_loss(log_probs, label, feat_len, label_len)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(model, loader, ctc_loss, vocab_inv):
        model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for feat, feat_len, label, label_len in tqdm(loader, desc='Eval'):
                feat, label = feat.to(DEVICE), label.to(DEVICE)
                feat_len, label_len = feat_len.to(DEVICE), label_len.to(DEVICE)

                logits = model(feat)
                log_probs = F.log_softmax(logits, dim=2)
                loss = ctc_loss(log_probs.permute(1,0,2), label, feat_len, label_len)
                total_loss += loss.item()

                pred_strs = decode_predictions(log_probs, vocab_inv)
                all_preds.extend(pred_strs)
                for l in label:
                    target_str = ' '.join([vocab_inv.get(idx.item(), '<unk>') for idx in l if idx != 0])
                    all_targets.append(target_str)

        avg_loss = total_loss / len(loader)
        cer = token_error_rate(all_preds, all_targets)
        if all_preds and all_targets:
            print(f"预测示例: {all_preds[0][:60]}...")
            print(f"真实示例: {all_targets[0][:60]}...")
        return avg_loss, cer, all_preds, all_targets

    def build_confusion_matrix(pred_strings, target_strings, top_n=20):
        pred_tokens = [t for s in pred_strings for t in s.split()]
        true_tokens = [t for s in target_strings for t in s.split()]
        if not pred_tokens or not true_tokens:
            print("警告：预测或真实音节列表为空，无法生成混淆矩阵。")
            return None, None
        counter = Counter(pred_tokens + true_tokens)
        common = [tok for tok, _ in counter.most_common(top_n)]
        filt_pred, filt_true = [], []
        for p, t in zip(pred_tokens, true_tokens):
            if p in common and t in common:
                filt_pred.append(p)
                filt_true.append(t)
        if not filt_pred:
            print("警告：无常见音节匹配，混淆矩阵为空。")
            return None, None
        cm = confusion_matrix(filt_true, filt_pred, labels=common)
        return cm, common

    # ==================== 6. 对比训练 ====================
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # 单向 LSTM
    print("\n========== 训练基准模型 (单向 LSTM) ==========")
    base_model = CTCASRModel(N_MELS, HIDDEN_SIZE, NUM_LAYERS, len(vocab), bidirectional=False, dropout=DROPOUT).to(DEVICE)
    optim_base = torch.optim.Adam(base_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_base, mode='min', patience=3, factor=0.5)

    hist_base = {'train_loss':[], 'test_loss':[], 'cer':[]}
    best_loss_base = float('inf')
    early_stop_base = 0
    for epoch in range(1, NUM_EPOCHS+1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(base_model, train_loader, optim_base, ctc_loss_fn, GRAD_CLIP)
        test_loss, cer, _, _ = evaluate(base_model, test_loader, ctc_loss_fn, vocab_inv)
        scheduler_base.step(test_loss)
        hist_base['train_loss'].append(train_loss)
        hist_base['test_loss'].append(test_loss)
        hist_base['cer'].append(cer)
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, TER: {cer:.4f}")

        if test_loss < best_loss_base:
            best_loss_base = test_loss
            early_stop_base = 0
            torch.save(base_model.state_dict(), 'best_base_model.pth')
        else:
            early_stop_base += 1
            if early_stop_base >= EARLY_STOP_PATIENCE:
                print(f"单向 LSTM 早停于 epoch {epoch}")
                break

    # 双向 LSTM
    print("\n========== 训练改进模型 (双向 LSTM) ==========")
    impr_model = CTCASRModel(N_MELS, HIDDEN_SIZE, NUM_LAYERS, len(vocab), bidirectional=True, dropout=DROPOUT).to(DEVICE)
    optim_impr = torch.optim.Adam(impr_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_impr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_impr, mode='min', patience=3, factor=0.5)

    hist_impr = {'train_loss':[], 'test_loss':[], 'cer':[]}
    best_loss_impr = float('inf')
    early_stop_impr = 0
    for epoch in range(1, NUM_EPOCHS+1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(impr_model, train_loader, optim_impr, ctc_loss_fn, GRAD_CLIP)
        test_loss, cer, _, _ = evaluate(impr_model, test_loader, ctc_loss_fn, vocab_inv)
        scheduler_impr.step(test_loss)
        hist_impr['train_loss'].append(train_loss)
        hist_impr['test_loss'].append(test_loss)
        hist_impr['cer'].append(cer)
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, TER: {cer:.4f}")

        if test_loss < best_loss_impr:
            best_loss_impr = test_loss
            early_stop_impr = 0
            torch.save(impr_model.state_dict(), 'best_impr_model.pth')
        else:
            early_stop_impr += 1
            if early_stop_impr >= EARLY_STOP_PATIENCE:
                print(f"双向 LSTM 早停于 epoch {epoch}")
                break

    print("\n加载最佳双向 LSTM 模型...")
    impr_model.load_state_dict(torch.load('best_impr_model.pth', weights_only=True))

    # ==================== 7. 可视化 ====================
    def plot_curves(hist_base, hist_impr):
        epochs_base = range(1, len(hist_base['train_loss'])+1)
        epochs_impr = range(1, len(hist_impr['train_loss'])+1)
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(epochs_base, hist_base['train_loss'], 'b-o', label='Base Train')
        plt.plot(epochs_base, hist_base['test_loss'], 'b--s', label='Base Test')
        plt.plot(epochs_impr, hist_impr['train_loss'], 'r-o', label='BiLSTM Train')
        plt.plot(epochs_impr, hist_impr['test_loss'], 'r--s', label='BiLSTM Test')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves')
        plt.legend(); plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(epochs_base, hist_base['cer'], 'b-o', label='Base TER')
        plt.plot(epochs_impr, hist_impr['cer'], 'r-o', label='BiLSTM TER')
        plt.xlabel('Epoch'); plt.ylabel('Token Error Rate'); plt.title('Test TER')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

    plot_curves(hist_base, hist_impr)

    print("\n========== 生成混淆矩阵 ==========")
    _, _, preds, targets = evaluate(impr_model, test_loader, ctc_loss_fn, vocab_inv)
    cm, labels = build_confusion_matrix(preds, targets, top_n=15)
    if cm is not None:
        print(f"混淆矩阵非零元素: {np.count_nonzero(cm)}/{cm.size}")
        print(f"对角线值: {np.diagonal(cm)}")
        plt.figure(figsize=(12,10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')
        plt.title('Confusion Matrix (Top 15 phonemes)')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
    else:
        print("混淆矩阵生成失败，请检查预测结果。")

    print("\n优化训练完成！图表已保存。")