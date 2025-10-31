# run_cnn_audio.py
# Score windows with the trained CNN and write per-window CSV.
# Expects: A3_Acoustic/data/test/{pos,neg}/*.wav

import argparse, csv, math
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn

# ------- must match training -------
def read_segment(path, sr_target, start_s, dur_s):
    info = sf.info(path)
    start = int(start_s * info.samplerate)
    frames = int(dur_s * info.samplerate)
    x, sr = sf.read(path, start=start, frames=frames, always_2d=False)
    if x.ndim > 1: x = x.mean(axis=1)
    if sr != sr_target:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    need = int(dur_s * sr) - len(x)
    if need > 0: x = np.pad(x, (0, need))
    return x.astype(np.float32), sr

def logmel(y, sr, n_mels=64, n_fft=1024, hop=320, fmin=50, fmax=8000):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    X = np.log(S + 1e-6).astype(np.float32)
    mu, sigma = X.mean(), X.std() + 1e-6
    X = (X - mu) / sigma
    return X  # [n_mels, T]

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.Dropout(0.1),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.Dropout(0.15),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Linear(64,1)
    def forward(self, x):
        return self.head(self.net(x).flatten(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="A3_Acoustic/data/test")
    ap.add_argument("--weights", default="A3_Acoustic/weights/cnn_mel.pth")
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--out", default="A3_Acoustic/results/cnn_scores.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    model = TinyCNN().to(args.device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    rows = []
    for lab in ["pos", "neg"]:
        for wav in sorted((Path(args.datadir)/lab).rglob("*.wav")):
            info = sf.info(str(wav))
            dur = info.frames / info.samplerate
            n = 1 + max(0, math.floor((dur - args.win) / args.hop))
            for i in range(n):
                t0 = i*args.hop
                y, sr = read_segment(str(wav), args.sr, t0, args.win)
                X = logmel(y, sr)  # [n_mels, T]
                x_t = torch.from_numpy(X)[None,None,...].to(args.device)
                with torch.no_grad():
                    logit = model(x_t)
                    prob = torch.sigmoid(logit).item()
                rows.append({
                    "file": wav.as_posix(),
                    "label": lab,
                    "start_s": round(t0, 3),
                    "end_s": round(t0 + args.win, 3),
                    "score": round(prob, 4)
                })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} windows)")

if __name__ == "__main__":
    main()
