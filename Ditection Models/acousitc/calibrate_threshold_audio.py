import argparse, json
import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="A3_Acoustic/results/dsp_scores.csv")
    ap.add_argument("--target_fpr", type=float, default=0.05)
    ap.add_argument("--out", default="A3_Acoustic/results/dsp_calibration.json")
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    neg = df[df["label"]=="neg"]["score"].to_numpy()
    pos = df[df["label"]=="pos"]["score"].to_numpy()

    if len(neg)==0 or len(pos)==0:
        raise SystemExit("Need both pos and neg windows in the CSV.")

    thr = float(np.quantile(neg, 1 - args.target_fpr))
    tpr = float((pos >= thr).mean())
    fpr = float((neg >= thr).mean())
    out = {"chosen_threshold": thr, "tpr_at_thr": tpr, "fpr_at_thr": fpr,
           "n_pos_windows": int(len(pos)), "n_neg_windows": int(len(neg))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
