import argparse, os
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, iirnotch, resample_poly
import csv

def load_mono_resampled(path, target_sr=32000):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        # high-quality polyphase resample
        g = np.gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        x = resample_poly(x, up, down)
        sr = target_sr
    return x.astype(np.float32), sr

def bandpass_80_2500(x, sr):
    sos = butter(4, [80, 2500], btype="bandpass", fs=sr, output="sos")
    y = sosfilt(sos, x)
    # notch 50 Hz mains (UK) just in case
    b, a = iirnotch(w0=50/(sr/2), Q=30)
    return sosfilt(np.array([b,0,0,0, a,0,0,0]).reshape(1,8), y) if False else y  # notch optional

def window_indices(n, sr, win_s=1.0, hop_s=0.25):
    w = int(win_s*sr); h = int(hop_s*sr)
    for start in range(0, max(1, n - w + 1), h):
        yield start, start + w

def harmonic_stack_score(x, sr, fmin=80, fmax=400, n_harm=8):
    """
    Returns (score, f0_hz, hsnr_db):
      - score in [0,1] (compressed from dB)
      - f0_hz best fundamental
      - hsnr_db harmonic-stack SNR in dB
    """
    # power spectrum of the whole 1-s window (Hann)
    N = len(x)
    win = np.hanning(N)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(N, d=1/sr)
    mag = np.abs(X)

    # candidate f0 grid (1 Hz steps is fine for 1-s windows)
    f0s = np.arange(fmin, fmax+1, 1.0)
    best_sum = 0.0; best_f0 = 0.0; best_noise = 1e-6
    for f0 in f0s:
        harm_bins = []
        noise_bins = []
        for k in range(1, n_harm+1):
            fk = k*f0
            if fk > f[-1]: break
            # ±5% tolerance band for each harmonic
            lo, hi = fk*0.95, fk*1.05
            mask = (f >= lo) & (f <= hi)
            if not mask.any(): continue
            harm_bins.append(mag[mask].mean())

            # local noise ring just outside ±8–12% (avoid tonal)
            lo2, hi2 = fk*0.88, fk*1.12
            mask2 = (f >= lo2) & (f <= hi2) & ~mask
            if mask2.any():
                noise_bins.append(mag[mask2].mean())
        if not harm_bins:
            continue
        H = np.sum(harm_bins)
        Nn = np.mean(noise_bins) * len(harm_bins) if noise_bins else 1e-6
        if H > best_sum:
            best_sum, best_f0, best_noise = H, f0, max(Nn, 1e-6)

    hsnr_db = 10*np.log10( (best_sum + 1e-9) / (best_noise + 1e-9) )
    # compress to 0..1 without tuning constants: 0 dB -> 0, 20 dB -> ~1
    score = float(np.clip(hsnr_db/20.0, 0.0, 1.0))
    return score, best_f0, float(hsnr_db)

def process_dir(audiodir, out_csv, sr=32000, win_s=1.0, hop_s=0.25):
    rows = []
    for label in ["pos", "neg"]:
        for wav in sorted(Path(audiodir, label).rglob("*.wav")):
            x, _sr = load_mono_resampled(str(wav), target_sr=sr)
            x = bandpass_80_2500(x, sr)
            # simple anti-wind highpass already in bandpass
            for s, e in window_indices(len(x), sr, win_s, hop_s):
                seg = x[s:e]
                if len(seg) < int(win_s*sr):  # pad (rare)
                    seg = np.pad(seg, (0, int(win_s*sr)-len(seg)))
                score, f0, hdb = harmonic_stack_score(seg, sr)
                rows.append({
                    "file": wav.as_posix(),
                    "label": label,               # 'pos' / 'neg'
                    "start_s": s/sr,
                    "end_s": e/sr,
                    "score": round(score, 4),
                    "hsnr_db": round(hdb, 2),
                    "f0_hz": round(f0, 1)
                })

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} windows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audiodir", default="A3_Acoustic/data/test")
    ap.add_argument("--out", default="A3_Acoustic/results/dsp_scores.csv")
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--fmin", type=float, default=80)
    ap.add_argument("--fmax", type=float, default=400)
    ap.add_argument("--harmonics", type=int, default=8)
    args = ap.parse_args()

    # Note: fmin/fmax/harmonics passed via globals inside harmonic_stack_score if you want to expose them—kept minimal here.
    process_dir(args.audiodir, args.out, sr=args.sr, win_s=args.win, hop_s=args.hop)
