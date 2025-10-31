import argparse, csv, json
from pathlib import Path
import numpy as np

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def to_map(rows):
    return {r["path"]: float(r.get("max_conf", 0.0)) for r in rows}

def main():
    ap = argparse.ArgumentParser(description="Compare two per-image score CSVs (e.g., YOLO-World vs Drone-YOLO).")
    ap.add_argument("--a", required=True, help="CSV A (e.g., results/yoloworld_scores.csv)")
    ap.add_argument("--b", required=True, help="CSV B (e.g., results/drone_yolo_scores.csv)")
    ap.add_argument("--out", default="results/compare/side_by_side.csv")
    ap.add_argument("--manifest", default="", help="Optional manifest.csv to add size_bin/appearance columns")
    args = ap.parse_args()

    A = read_csv(args.a); B = read_csv(args.b)
    Amap, Bmap = to_map(A), to_map(B)

    # join on common paths
    common = sorted(set(Amap.keys()) & set(Bmap.keys()))
    rows = []
    for p in common:
        sa, sb = Amap[p], Bmap[p]
        rows.append({"path": p, "score_A": sa, "score_B": sb, "delta_B_minus_A": round(sb - sa, 4)})

    # optional: enrich with manifest fields
    if args.manifest and Path(args.manifest).exists():
        man = {r["path"]: r for r in csv.DictReader(open(args.manifest, "r", encoding="utf-8"))}
        for r in rows:
            m = man.get(r["path"], {})
            r["size_bin"] = m.get("size_bin", "")
            r["appearance_color"] = m.get("appearance_color", "")
            r["appearance_shape"] = m.get("appearance_shape", "")
            r["background"] = m.get("background", "")

    # stats
    sA = np.array([r["score_A"] for r in rows], float)
    sB = np.array([r["score_B"] for r in rows], float)
    delta = sB - sA
    summary = {
        "n_common": len(rows),
        "A_median": float(np.median(sA)) if len(rows) else 0.0,
        "B_median": float(np.median(sB)) if len(rows) else 0.0,
        "delta_median(B-A)": float(np.median(delta)) if len(rows) else 0.0,
        "A_mean": float(np.mean(sA)) if len(rows) else 0.0,
        "B_mean": float(np.mean(sB)) if len(rows) else 0.0,
        "delta_mean(B-A)": float(np.mean(delta)) if len(rows) else 0.0,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["path","score_A","score_B","delta_B_minus_A"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
