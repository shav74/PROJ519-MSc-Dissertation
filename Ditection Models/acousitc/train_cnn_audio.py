# train_cnn_audio.py
# Tiny CNN on log-mel spectrograms (binary: drone present?).
# Expects: A3_Acoustic/data/{train,val}/{pos,neg}/*.wav

import argparse, math, random
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ----------------- Audio utils -----------------
def read_segment(path, sr_target, start_s, dur_s):
    """Read [start_s, start_s+dur_s) seconds and resample to sr_target."""
    info = sf.info(path)
    # read in original sr for the exact region, then resample
    start = int(start_s * info.samplerate)
    frames = int(dur_s * info.samplerate)
    x, sr = sf.read(path, start=start, frames=frames, always_2d=False)
    if x.ndim > 1: x = x.mean(axis=1)
    if sr != sr_target:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    # pad if shorter (edge windows)
    need = int(dur_s * sr) - len(x)
    if need > 0: x = np.pad(x, (0, need))
    return x.astype(np.float32), sr

def logmel(y, sr, n_mels=64, n_fft=1024, hop=320, fmin=50, fmax=8000):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    X = np.log(S + 1e-6).astype(np.float32)
    return X  # [n_mels, T]

# simple SpecAugment-ish masks on numpy array
def time_mask(x, max_w=16):
    if x.shape[1] < 2: return x
    w = random.randint(0, min(max_w, x.shape[1]//4) or 0)
    if w > 0:
        t0 = random.randint(0, x.shape[1]-w)
        x[:, t0:t0+w] = x.mean()
    return x

def freq_mask(x, max_w=8):
    if x.shape[0] < 2: return x
    w = random.randint(0, min(max_w, x.shape[0]//4) or 0)
    if w > 0:
        f0 = random.randint(0, x.shape[0]-w)
        x[f0:f0+w, :] = x.mean()
    return x

# ----------------- Dataset -----------------
class WindowIndex:
    """Holds (path, start_s, label) entries for all windows in a split."""
    def __init__(self, root, split, sr_probe=32000, win_s=1.0, hop_s=0.25):
        self.entries = []
        base = Path(root) / split
        for lab in ["pos", "neg"]:
            for wav in sorted((base/lab).rglob("*.wav")):
                info = sf.info(str(wav))
                dur = info.frames / info.samplerate
                n = 1 + max(0, math.floor((dur - win_s) / hop_s))
                for i in range(n):
                    self.entries.append((wav.as_posix(), i*hop_s, 1 if lab=="pos" else 0))
        random.shuffle(self.entries)

class MelWindowDataset(Dataset):
    def __init__(self, index: WindowIndex, sr=32000, win_s=1.0,
                 n_mels=64, n_fft=1024, hop=320, augment=False):
        self.idx = index.entries
        self.sr = sr; self.win = win_s
        self.n_mels = n_mels; self.n_fft = n_fft; self.hop = hop
        self.augment = augment

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        path, t0, lab = self.idx[i]
        y, sr = read_segment(path, self.sr, t0, self.win)

        # random gain (±6 dB) for robustness
        if self.augment:
            g_db = random.uniform(-6, 6); y = (10**(g_db/20.0)) * y

        X = logmel(y, sr, self.n_mels, self.n_fft, self.hop)
        if self.augment:
            if random.random() < 0.5: X = time_mask(X)
            if random.random() < 0.5: X = freq_mask(X)

        # normalize per-example
        mu, sigma = X.mean(), X.std() + 1e-6
        X = (X - mu) / sigma

        # to tensor [1, n_mels, T]
        X = torch.from_numpy(X)[None, ...]
        y = torch.tensor([float(lab)], dtype=torch.float32)
        return X, y

# ----------------- Model -----------------
class TinyCNN(nn.Module):
    def __init__(self, in_ch=1, n_mels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),    nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.Dropout(0.1),

            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.Dropout(0.15),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)  # logits

# ----------------- Train -----------------
def main():
    ap = argparse.ArgumentParser(description="Train tiny CNN on log-mel (binary).")
    ap.add_argument("--datadir", default="A3_Acoustic/data")
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="A3_Acoustic/weights/cnn_mel.pth")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Build indices
    train_idx = WindowIndex(args.datadir, "train", args.sr, args.win, args.hop)
    val_idx   = WindowIndex(args.datadir, "val",   args.sr, args.win, args.hop)

    # Balance sampler if needed
    y_train = [lab for _,_,lab in train_idx.entries]
    class_counts = np.bincount(y_train, minlength=2) + 1e-6
    weights = [1.0/class_counts[lab] for lab in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = MelWindowDataset(train_idx, args.sr, args.win, args.n_mels, augment=True)
    val_ds   = MelWindowDataset(val_idx,   args.sr, args.win, args.n_mels, augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # Model
    model = TinyCNN().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()
    best_loss, best_ep = 1e9, -1

    for ep in range(1, args.epochs+1):
        model.train(); tr_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(X)
        tr_loss /= len(train_dl.dataset)

        # val
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(args.device), y.to(args.device)
                logits = model(X)
                loss = crit(logits, y)
                va_loss += loss.item() * len(X)
        va_loss /= max(1, len(val_dl.dataset))

        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best_loss - 1e-4:
            best_loss, best_ep = va_loss, ep
            torch.save({"state_dict": model.state_dict(),
                        "sr": args.sr, "n_mels": args.n_mels}, args.out)
    print(f"Saved best @ epoch {best_ep} to {args.out}")

if __name__ == "__main__":
    main()
