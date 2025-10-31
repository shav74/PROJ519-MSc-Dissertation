import argparse, csv, json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

# ---------- IO helpers ----------
def read_manifest(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_csv(rows: List[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def exists_image(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False

# ---------- scoring ----------
def per_image_rows_to_scores(rows: List[dict]) -> Dict[str, float]:
    """Map image path -> score (max_conf)."""
    return {r["path"]: float(r.get("max_conf", 0.0)) for r in rows}

def summarize_bucket(rows: List[dict], score_map: Dict[str,float], key: str):
    groups = defaultdict(list)
    for r in rows:
        v = (r.get(key) or "").strip()
        if v: groups[v].append(r)
    out = []
    for k, grp in sorted(groups.items()):
        scores = [score_map.get(g["path"], 0.0) for g in grp]
        hits   = sum(int(score_map.get(g["path"], 0.0) >= 0.25) for g in grp)  # rate at 0.25 for reference
        tot    = len(grp)
        out.append({
            key: k,
            "n": tot,
            "median_score": float(np.median(scores)) if tot else 0.0,
            "mean_score": float(np.mean(scores)) if tot else 0.0,
            "det_rate@0.25": hits / tot if tot else 0.0
        })
    return out

def overall_summary(pos_rows: List[dict], neg_rows: List[dict], score_map: Dict[str,float]):
    pos_scores = [score_map.get(r["path"], 0.0) for r in pos_rows]
    neg_scores = [score_map.get(r["path"], 0.0) for r in neg_rows]
    pos_det = sum(s >= 0.25 for s in pos_scores); neg_fp = sum(s >= 0.25 for s in neg_scores)
    return {
        "positives_total": len(pos_rows),
        "positives_median_score": float(np.median(pos_scores)) if pos_scores else 0.0,
        "positives_mean_score": float(np.mean(pos_scores)) if pos_scores else 0.0,
        "positives_det_rate@0.25": (pos_det / len(pos_rows)) if pos_rows else 0.0,
        "negatives_total": len(neg_rows),
        "negatives_median_score": float(np.median(neg_scores)) if neg_scores else 0.0,
        "negatives_mean_score": float(np.mean(neg_scores)) if neg_scores else 0.0,
        "negatives_fpr@0.25": (neg_fp / len(neg_rows)) if neg_rows else 0.0,
    }

# ---------- inference runners ----------
def run_yoloworld(img_paths: List[str], prompts: List[str], conf: float):
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(prompts)
    out = []
    for p in tqdm(img_paths, desc="YOLO-World"):
        r = model.predict(p, conf=conf, verbose=False)[0]
        hit = int(len(r.boxes) > 0)
        max_conf = float(r.boxes.conf.max().cpu().numpy()) if hit else 0.0
        out.append({"path": p, "hit": hit, "max_conf": round(max_conf,4), "model": "YOLO-World"})
    return out

def run_drone_specific(img_paths: List[str], weights_path: str, conf: float):
    if not weights_path or not Path(weights_path).exists():
        print("[warn] Drone-specific weights not found; skipping.")
        return []
    model = YOLO(weights_path)
    out = []
    for p in tqdm(img_paths, desc="Drone-specific YOLO"):
        r = model.predict(p, conf=conf, verbose=False)[0]
        hit = int(len(r.boxes) > 0)
        max_conf = float(r.boxes.conf.max().cpu().numpy()) if hit else 0.0
        out.append({"path": p, "hit": hit, "max_conf": round(max_conf,4), "model": "Drone-YOLO"})
    return out

def run_generic(img_paths: List[str], weights_path: str, conf: float):
    """Optional generic detector (e.g., yolo11s.pt or yolov8s.pt) as a control."""
    if not weights_path or not Path(weights_path).exists():
        print("[info] Generic weights not provided; skipping generic detector.")
        return []
    model = YOLO(weights_path)
    out = []
    for p in tqdm(img_paths, desc="Generic YOLO"):
        r = model.predict(p, conf=conf, verbose=False)[0]
        hit = int(len(r.boxes) > 0)
        max_conf = float(r.boxes.conf.max().cpu().numpy()) if hit else 0.0
        out.append({"path": p, "hit": hit, "max_conf": round(max_conf,4), "model": "Generic-YOLO"})
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Compare YOLO-World vs Drone-specific YOLO (+ optional generic).")
    ap.add_argument("--manifest", default="data/meta/manifest.csv")
    ap.add_argument("--conf", type=float, default=0.001, help="Low conf to capture weak detections as scores")
    ap.add_argument("--prompts", default="drone,quadcopter,uav")
    ap.add_argument("--weights_drone", default="weights/drone_yolo.pt", help="Drone-trained checkpoint")
    ap.add_argument("--weights_generic", default="", help="Optional generic model checkpoint (control)")
    ap.add_argument("--outdir", default="results_comp")
    args = ap.parse_args()

    rows = read_manifest(Path(args.manifest))
    rows = [r for r in rows if exists_image(r["path"])]
    if not rows:
        print("No images found from manifest paths."); sys.exit(1)

    # Positives/Negatives from manifest
    pos_rows = [r for r in rows if (r.get("drone_present") or "") == "1"]
    neg_rows = [r for r in rows if (r.get("drone_present") or "") == "0"]
    all_img_paths = sorted({r["path"] for r in rows})

    print(f"Images total: {len(all_img_paths)} | Positives: {len(pos_rows)} | Negatives: {len(neg_rows)}")

    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]

    # Run models
    preds_world  = run_yoloworld(all_img_paths, prompts, args.conf)
    preds_drone  = run_drone_specific(all_img_paths, args.weights_drone, args.conf)
    preds_generic= run_generic(all_img_paths, args.weights_generic, args.conf)

    # Save per-image predictions
    outdir = Path(args.outdir)
    save_csv(preds_world,   outdir / "per_image_yoloworld.csv")
    if preds_drone:   save_csv(preds_drone,   outdir / "per_image_drone_yolo.csv")
    if preds_generic: save_csv(preds_generic, outdir / "per_image_generic.csv")

    # Build summaries
    summaries = []
    def model_summary(name, preds):
        smap = per_image_rows_to_scores(preds)
        ov   = overall_summary(pos_rows, neg_rows, smap)
        by_size = summarize_bucket(pos_rows, smap, "size_bin") if any((r.get("size_bin") or "").strip() for r in pos_rows) else []
        by_color= summarize_bucket(pos_rows, smap, "appearance_color") if any((r.get("appearance_color") or "").strip() for r in pos_rows) else []
        by_shape= summarize_bucket(pos_rows, smap, "appearance_shape") if any((r.get("appearance_shape") or "").strip() for r in pos_rows) else []
        by_back = summarize_bucket(pos_rows, smap, "background") if any((r.get("background") or "").strip() for r in pos_rows) else []
        return {
            "model": name,
            "overall": ov,
            "by_size_bin": by_size,
            "by_appearance_color": by_color,
            "by_appearance_shape": by_shape,
            "by_background": by_back
        }

    summaries.append(model_summary("YOLO-World", preds_world))
    if preds_drone:
        summaries.append(model_summary("Drone-specific YOLO", preds_drone))
    if preds_generic:
        summaries.append(model_summary("Generic YOLO", preds_generic))

    save_json(summaries, outdir / "summary.json")

    # Also emit CSV tables for easy copy-paste
    tables = []
    for s in summaries:
        name = s["model"]
        ov = s["overall"]
        tables.append({
            "model": name,
            "pos_n": ov["positives_total"],
            "pos_median_score": ov["positives_median_score"],
            "pos_mean_score": ov["positives_mean_score"],
            "pos_det_rate@0.25": ov["positives_det_rate@0.25"],
            "neg_n": ov["negatives_total"],
            "neg_median_score": ov["negatives_median_score"],
            "neg_mean_score": ov["negatives_mean_score"],
            "neg_fpr@0.25": ov["negatives_fpr@0.25"],
        })
    save_csv(tables, outdir / "overall_table.csv")

    def dump_bucket(listdict, path, model_name):
        rows = []
        for d in listdict:
            row = {"model": model_name}; row.update(d); rows.append(row)
        save_csv(rows, path)

    for s in summaries:
        nm = s["model"].replace(" ", "_")
        if s["by_size_bin"]:          dump_bucket(s["by_size_bin"],          outdir / f"{nm}_by_size.csv", nm)
        if s["by_appearance_color"]:  dump_bucket(s["by_appearance_color"],  outdir / f"{nm}_by_color.csv", nm)
        if s["by_appearance_shape"]:  dump_bucket(s["by_appearance_shape"],  outdir / f"{nm}_by_shape.csv", nm)
        if s["by_background"]:        dump_bucket(s["by_background"],        outdir / f"{nm}_by_background.csv", nm)

    print("\nSaved:")
    print(f"- {outdir/'per_image_yoloworld.csv'}")
    if preds_drone:   print(f"- {outdir/'per_image_drone_yolo.csv'}")
    if preds_generic: print(f"- {outdir/'per_image_generic.csv'}")
    print(f"- {outdir/'overall_table.csv'}")
    print(f"- {outdir/'summary.json'}")
    print("Plus per-bucket CSVs (size/color/shape/background) where applicable.")

if __name__ == "__main__":
    main()
