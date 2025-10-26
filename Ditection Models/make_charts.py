# make_charts.py
# Build clean charts from per-image score CSVs + manifest.
# - Uses matplotlib only (no seaborn), one chart per figure, no custom colors.
# - Robust to Windows/Unix path differences and messy drone_present values.
#
# Usage (Windows examples):
#   python make_charts.py --manifest data\meta\manifest.csv --inputs results\yoloworld_scores.csv results\drone_yolo_scores.csv --outdir charts
#
# Tip: run your model score scripts with --conf 0.001 so weak detections aren't clipped.

import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_scores(path: Path) -> pd.DataFrame:
    """Load a per-image score CSV and normalize paths."""
    df = pd.read_csv(path)
    if "model" not in df.columns:
        df["model"] = path.stem
    # Normalize to POSIX-style paths for reliable joins across OSes
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)
    # Coerce score
    df["max_conf"] = pd.to_numeric(df.get("max_conf", 0.0), errors="coerce").fillna(0.0)
    # Keep the essentials
    return df[["path", "model", "max_conf"]]

def normalize_pos_neg(man: pd.DataFrame) -> pd.DataFrame:
    """Create boolean __is_pos/__is_neg flags from drone_present and (as fallback) source."""
    man = man.copy()
    # Normalize path
    man["path"] = man["path"].astype(str).str.replace("\\", "/", regex=False)

    # Initialize flags
    man["__is_pos"] = False
    man["__is_neg"] = False

    # From drone_present if present
    if "drone_present" in man.columns:
        s = man["drone_present"].astype(str).str.strip().str.lower()
        # numeric parse handles "1", "0", "1.0", "0.0"
        snum = pd.to_numeric(s, errors="coerce")
        is_pos_num = (snum == 1).fillna(False)
        is_neg_num = (snum == 0).fillna(False)
        # textual fallbacks
        is_pos_txt = s.isin({"1", "true", "yes", "y", "pos", "positive"})
        is_neg_txt = s.isin({"0", "false", "no", "n", "neg", "negative"})
        man["__is_pos"] = man["__is_pos"] | is_pos_num | is_pos_txt
        man["__is_neg"] = man["__is_neg"] | is_neg_num | is_neg_txt

    # Fallback from source, if present
    if "source" in man.columns:
        src = man["source"].astype(str).str.strip().str.lower()
        man.loc[src == "dronevsbird_neg", "__is_neg"] = True
        man.loc[src == "dronevsbird_drone", "__is_pos"] = True
        # Do NOT auto-mark antiuav_vis positives; they can include blank/occluded frames.

    # Ensure size_bin exists
    if "size_bin" not in man.columns:
        man["size_bin"] = ""

    # Optional appearance/background defaults
    for col in ["appearance_color", "appearance_shape", "background"]:
        if col not in man.columns:
            man[col] = ""

    return man

def plot_hist(scores, title, outfile):
    if len(scores) == 0:
        return
    plt.figure()
    plt.hist(scores, bins=30)  # default color/style
    plt.xlabel("Score (max_conf)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def plot_box_by_size(df_pos: pd.DataFrame, title, outfile):
    if "size_bin" not in df_pos.columns or df_pos.empty:
        return
    order = [b for b in ["Tiny", "Small", "Medium", "Large"] if b in df_pos["size_bin"].unique()]
    if not order:
        return
    data = [df_pos.loc[df_pos["size_bin"] == b, "max_conf"].values for b in order]
    if sum(len(d) > 0 for d in data) == 0:
        return
    plt.figure()
    plt.boxplot(data, labels=order, showfliers=False)
    plt.xlabel("size_bin")
    plt.ylabel("Score (max_conf)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def plot_median_bar_by_size(df_pos: pd.DataFrame, title, outfile):
    if "size_bin" not in df_pos.columns or df_pos.empty:
        return
    order = [b for b in ["Tiny", "Small", "Medium", "Large"] if b in df_pos["size_bin"].unique()]
    if not order:
        return
    med = df_pos.groupby("size_bin")["max_conf"].median().reindex(order)
    plt.figure()
    plt.bar(med.index, med.values)  # default style
    plt.xlabel("size_bin")
    plt.ylabel("Median score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Make charts from per-image score CSVs + manifest.")
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more per-image score CSVs (e.g., yoloworld_scores.csv, drone_yolo_scores.csv)")
    ap.add_argument("--outdir", default="charts", help="Folder to save PNGs and summary CSV")
    ap.add_argument("--det_thr", type=float, default=0.25, help="Reference detection threshold for summary")
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)

    # Load & normalize manifest
    man_raw = pd.read_csv(args.manifest)
    if "path" not in man_raw.columns:
        raise SystemExit("manifest.csv must have a 'path' column.")
    man = normalize_pos_neg(man_raw)

    summary_rows = []

    for inp in args.inputs:
        inp_path = Path(inp)
        df = read_scores(inp_path)

        # Join with manifest (left join on path)
        m = df.merge(man[["path", "__is_pos", "__is_neg", "size_bin",
                          "appearance_color", "appearance_shape", "background"]],
                     on="path", how="left")

        model_name = str(m["model"].iloc[0]) if len(m) else inp_path.stem

        # Helpful debug
        matched = m["__is_pos"].notna().sum()
        print(f"[{model_name}] rows: {len(m)} | matched-with-manifest: {matched}")
        print(f"  positives matched: {int(m['__is_pos'].sum())} | negatives matched: {int(m['__is_neg'].sum())}")

        # Positives/negatives splits
        df_pos = m[m["__is_pos"]].copy()
        df_neg = m[m["__is_neg"]].copy()

        # Charts
        if len(df_pos):
            plot_hist(df_pos["max_conf"].values,
                      f"{model_name} — Scores on positives (n={len(df_pos)})",
                      outdir / f"{model_name}_hist_pos.png")
            plot_box_by_size(df_pos,
                             f"{model_name} — Score by size_bin (positives)",
                             outdir / f"{model_name}_box_size.png")
            plot_median_bar_by_size(df_pos,
                                    f"{model_name} — Median score by size_bin (positives)",
                                    outdir / f"{model_name}_median_by_size.png")
        if len(df_neg):
            plot_hist(df_neg["max_conf"].values,
                      f"{model_name} — Scores on negatives (n={len(df_neg)})",
                      outdir / f"{model_name}_hist_neg.png")

        # Summary table entries
        pos_scores = df_pos["max_conf"].values if len(df_pos) else np.array([])
        neg_scores = df_neg["max_conf"].values if len(df_neg) else np.array([])
        pos_med = float(np.median(pos_scores)) if pos_scores.size else 0.0
        pos_mean= float(np.mean(pos_scores)) if pos_scores.size else 0.0
        pos_rate= float((pos_scores >= args.det_thr).mean()) if pos_scores.size else 0.0

        neg_med = float(np.median(neg_scores)) if neg_scores.size else 0.0
        neg_mean= float(np.mean(neg_scores)) if neg_scores.size else 0.0
        neg_fpr = float((neg_scores >= args.det_thr).mean()) if neg_scores.size else 0.0

        summary_rows.append({
            "model": model_name,
            "pos_n": int(len(df_pos)), "pos_median": pos_med, "pos_mean": pos_mean, f"pos_det_rate@{args.det_thr}": pos_rate,
            "neg_n": int(len(df_neg)), "neg_median": neg_med, "neg_mean": neg_mean, f"neg_fpr@{args.det_thr}": neg_fpr
        })

        # Per-size medians CSV (positives only)
        if len(df_pos):
            per_size = (df_pos.groupby("size_bin")["max_conf"]
                        .agg(median="median", mean="mean", count="count")
                        .reindex(["Tiny", "Small", "Medium", "Large"]))
            per_size.to_csv(outdir / f"{model_name}_per_size.csv")

    # Save overall summary
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(outdir / "summary_scores.csv", index=False)

    print(f"Done. Charts in: {outdir}")

if __name__ == "__main__":
    main()

# python make_charts.py --manifest data/meta/manifest.csv --inputs results/yoloworld_scores.csv results/drone_yolo_scores.csv --outdir charts
