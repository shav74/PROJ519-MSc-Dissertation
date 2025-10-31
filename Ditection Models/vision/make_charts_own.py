# make_charts.py
# Build clean charts from per-image score CSVs + manifest.
# - Uses matplotlib only (no seaborn), one chart per figure, no custom colors.
# - Robust to Windows/Unix path differences and messy drone_present values.
# - Fixed to handle NaN values in boolean masks
#
# Usage (Windows examples):
#   python make_charts.py --manifest data\meta\manifest.csv --inputs results\yoloworld_scores.csv results\drone_yolo_scores.csv --outdir charts

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

    # Handle different possible column names for path
    path_col = None
    for col in ["path", "frame", "image", "filename"]:
        if col in df.columns:
            path_col = col
            break

    if path_col is None:
        # Use first column as path
        path_col = df.columns[0]

    # Rename to 'path' for consistency
    if path_col != "path":
        df = df.rename(columns={path_col: "path"})

    # Normalize to POSIX-style paths for reliable joins across OSes
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)

    # Coerce score - handle different column names
    score_col = None
    for col in ["max_conf", "confidence", "score", "conf"]:
        if col in df.columns:
            score_col = col
            break

    if score_col and score_col != "max_conf":
        df["max_conf"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    elif "max_conf" not in df.columns:
        df["max_conf"] = 0.0
    else:
        df["max_conf"] = pd.to_numeric(df["max_conf"], errors="coerce").fillna(0.0)

    # Keep the essentials
    return df[["path", "model", "max_conf"]]


def normalize_pos_neg(man: pd.DataFrame) -> pd.DataFrame:
    """Create boolean __is_pos/__is_neg flags from drone_present and (as fallback) source."""
    man = man.copy()

    # Handle different path column names
    path_col = None
    for col in ["path", "frame", "image", "filename"]:
        if col in man.columns:
            path_col = col
            break

    if path_col is None:
        path_col = man.columns[0]

    if path_col != "path":
        man = man.rename(columns={path_col: "path"})

    # Normalize path
    man["path"] = man["path"].astype(str).str.replace("\\", "/", regex=False)

    # Initialize flags - IMPORTANT: use False instead of NaN
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
    else:
        # If no drone_present column, assume all are positives
        print("Warning: No 'drone_present' column found. Assuming all frames contain drones.")
        man["__is_pos"] = True
        man["__is_neg"] = False

    # Fallback from source, if present
    if "source" in man.columns:
        src = man["source"].astype(str).str.strip().str.lower()
        man.loc[src == "dronevsbird_neg", "__is_neg"] = True
        man.loc[src == "dronevsbird_neg", "__is_pos"] = False
        man.loc[src == "dronevsbird_drone", "__is_pos"] = True
        man.loc[src == "dronevsbird_drone", "__is_neg"] = False

    # Ensure size_bin exists
    if "size_bin" not in man.columns:
        man["size_bin"] = "Unknown"

    # Optional appearance/background defaults
    for col in ["appearance_color", "appearance_shape", "background"]:
        if col not in man.columns:
            man[col] = ""

    return man


def plot_hist(scores, title, outfile):
    if len(scores) == 0:
        return
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Score (max_conf)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
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
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=order, showfliers=False)
    plt.xlabel("size_bin", fontsize=12)
    plt.ylabel("Score (max_conf)", fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()


def plot_median_bar_by_size(df_pos: pd.DataFrame, title, outfile):
    if "size_bin" not in df_pos.columns or df_pos.empty:
        return
    order = [b for b in ["Tiny", "Small", "Medium", "Large"] if b in df_pos["size_bin"].unique()]
    if not order:
        return
    med = df_pos.groupby("size_bin")["max_conf"].median().reindex(order)
    plt.figure(figsize=(8, 6))
    plt.bar(med.index, med.values, edgecolor='black', alpha=0.7)
    plt.xlabel("size_bin", fontsize=12)
    plt.ylabel("Median score", fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Make charts from per-image score CSVs + manifest.")
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more per-image score CSVs (e.g., yoloworld_scores.csv, drone_yolo_scores.csv)")
    ap.add_argument("--outdir", default="charts", help="Folder to save PNGs and summary CSV")
    ap.add_argument("--det_thr", type=float, default=0.25, help="Reference detection threshold for summary")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Load & normalize manifest
    man_raw = pd.read_csv(args.manifest)
    man = normalize_pos_neg(man_raw)

    print(f"\nManifest loaded: {len(man)} rows")
    print(f"  Positives (drone present): {man['__is_pos'].sum()}")
    print(f"  Negatives (no drone): {man['__is_neg'].sum()}")

    summary_rows = []

    for inp in args.inputs:
        inp_path = Path(inp)
        if not inp_path.exists():
            print(f"Warning: {inp} not found, skipping...")
            continue

        df = read_scores(inp_path)

        # Join with manifest (left join on path)
        m = df.merge(man[["path", "__is_pos", "__is_neg", "size_bin",
                          "appearance_color", "appearance_shape", "background"]],
                     on="path", how="left")

        model_name = str(m["model"].iloc[0]) if len(m) else inp_path.stem

        # Fill NaN values in boolean columns with False
        m["__is_pos"] = m["__is_pos"].fillna(False)
        m["__is_neg"] = m["__is_neg"].fillna(False)

        # Helpful debug
        matched = (~m["__is_pos"].isna()).sum()
        print(f"\n[{model_name}]")
        print(f"  Total rows: {len(m)}")
        print(f"  Matched with manifest: {matched}")
        print(f"  Positives: {int(m['__is_pos'].sum())}")
        print(f"  Negatives: {int(m['__is_neg'].sum())}")

        # Positives/negatives splits - now safe with no NaN
        df_pos = m[m["__is_pos"] == True].copy()
        df_neg = m[m["__is_neg"] == True].copy()

        # Charts
        if len(df_pos) > 0:
            plot_hist(df_pos["max_conf"].values,
                      f"{model_name} — Scores on positives (n={len(df_pos)})",
                      outdir / f"{model_name}_hist_pos.png")
            plot_box_by_size(df_pos,
                             f"{model_name} — Score by size_bin (positives)",
                             outdir / f"{model_name}_box_size.png")
            plot_median_bar_by_size(df_pos,
                                    f"{model_name} — Median score by size_bin (positives)",
                                    outdir / f"{model_name}_median_by_size.png")
        else:
            print(f"  Warning: No positive samples found for {model_name}")

        if len(df_neg) > 0:
            plot_hist(df_neg["max_conf"].values,
                      f"{model_name} — Scores on negatives (n={len(df_neg)})",
                      outdir / f"{model_name}_hist_neg.png")

        # Summary table entries
        pos_scores = df_pos["max_conf"].values if len(df_pos) else np.array([])
        neg_scores = df_neg["max_conf"].values if len(df_neg) else np.array([])
        pos_med = float(np.median(pos_scores)) if pos_scores.size else 0.0
        pos_mean = float(np.mean(pos_scores)) if pos_scores.size else 0.0
        pos_rate = float((pos_scores >= args.det_thr).mean()) if pos_scores.size else 0.0

        neg_med = float(np.median(neg_scores)) if neg_scores.size else 0.0
        neg_mean = float(np.mean(neg_scores)) if neg_scores.size else 0.0
        neg_fpr = float((neg_scores >= args.det_thr).mean()) if neg_scores.size else 0.0

        summary_rows.append({
            "model": model_name,
            "pos_n": int(len(df_pos)), "pos_median": pos_med, "pos_mean": pos_mean,
            f"pos_det_rate@{args.det_thr}": pos_rate,
            "neg_n": int(len(df_neg)), "neg_median": neg_med, "neg_mean": neg_mean, f"neg_fpr@{args.det_thr}": neg_fpr
        })

        # Per-size medians CSV (positives only)
        if len(df_pos) > 0:
            per_size = (df_pos.groupby("size_bin")["max_conf"]
                        .agg(median="median", mean="mean", count="count")
                        .reindex(["Tiny", "Small", "Medium", "Large"]))
            per_size.to_csv(outdir / f"{model_name}_per_size.csv")

    # Save overall summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(outdir / "summary_scores.csv", index=False)
        print(f"\n✓ Summary saved to: {outdir / 'summary_scores.csv'}")
        print("\nSummary:")
        print(summary_df.to_string(index=False))

    print(f"\n✓ Done. Charts saved in: {outdir}")


if __name__ == "__main__":
    main()

# python make_charts.py --manifest data/meta/manifest_own.csv --inputs results/yoloworld_scores_own.csv results/drone_yolo_scores_own.csv --outdir charts/own
