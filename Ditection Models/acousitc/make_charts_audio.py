import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(a, title, path):
    if len(a)==0: return
    plt.figure()
    plt.hist(a, bins=30)
    plt.xlabel("Score (harmonic stack)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="A3_Acoustic/results/dsp_scores.csv")
    ap.add_argument("--calib",  default="A3_Acoustic/results/dsp_calibration.json")
    ap.add_argument("--outdir", default="A3_Acoustic/charts_dsp")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores)
    pos = df[df["label"]=="pos"]["score"].to_numpy()
    neg = df[df["label"]=="neg"]["score"].to_numpy()

    plot_hist(pos, f"DSP — Scores on positives (n={len(pos)})", outdir/"dsp_hist_pos.png")
    plot_hist(neg, f"DSP — Scores on negatives (n={len(neg)})", outdir/"dsp_hist_neg.png")

    # Optional vertical threshold line if calibration file exists
    if Path(args.calib).exists():
        import json
        thr = json.loads(Path(args.calib).read_text())["chosen_threshold"]
        for p in ["dsp_hist_pos.png","dsp_hist_neg.png"]:
            fig = plt.figure()
            # re-plot quickly with threshold line
            data = pos if "pos" in p else neg
            plt.hist(data, bins=30)
            plt.axvline(thr, linestyle="--")
            plt.xlabel("Score (harmonic stack)")
            plt.ylabel("Count")
            plt.title(("Positives" if "pos" in p else "Negatives") + f" — thr={thr:.3f}")
            plt.tight_layout(); plt.savefig(outdir/("thr_"+p), dpi=200); plt.close()

    # Quick text summary
    med_pos = float(np.median(pos)) if len(pos) else 0.0
    med_neg = float(np.median(neg)) if len(neg) else 0.0
    print({"median_pos": med_pos, "median_neg": med_neg, "n_pos": int(len(pos)), "n_neg": int(len(neg))})
