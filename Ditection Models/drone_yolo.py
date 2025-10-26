import argparse, csv, os
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

def list_images(root, recursive=True):
    exts = (".jpg", ".jpeg", ".png")
    if recursive:
        return sorted([str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts])
    return sorted([str(p) for p in Path(root).iterdir() if p.suffix.lower() in exts])

def main():
    ap = argparse.ArgumentParser(description="Drone-specific YOLO on a folder; writes per-image scores (max_conf).")
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--weights", required=True, help="Path to your drone-trained checkpoint (.pt)")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--out", default="results/drone_yolo_scores.csv")
    ap.add_argument("--no_recursive", action="store_true")
    args = ap.parse_args()

    paths = list_images(args.imgdir, recursive=not args.no_recursive)
    if not paths:
        print("No images found.")
        return

    import numpy as np
    # ...
    model = YOLO(args.weights)

    # figure out which class IDs correspond to drone-like labels
    name_map = getattr(model, "names", {}) or {}
    drone_labels = {"drone", "uav", "quadcopter"}
    drone_ids = [i for i, n in name_map.items() if str(n).strip().lower() in drone_labels]

    # fallback: if model has exactly 1 class, assume it's "drone"
    if not drone_ids and len(name_map) == 1:
        drone_ids = [list(name_map.keys())[0]]

    print("Model classes:", name_map)
    print("Using class IDs as 'drone':", drone_ids)

    rows = []
    for p in tqdm(paths, desc="Drone-YOLO"):
        r = model.predict(p, conf=args.conf, verbose=False)[0]
        hit = 0;
        max_conf = 0.0
        if r.boxes is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            if drone_ids:
                mask = np.isin(cls, np.array(drone_ids))
                if mask.any():
                    hit = 1
                    max_conf = float(conf[mask].max())
            else:
                # If we still don't know the class, be strict: treat as no detection.
                hit = 0
                max_conf = 0.0

        rows.append({
            "path": Path(p).as_posix(),
            "model": "drone_yolo",
            "hit": hit,
            "max_conf": round(max_conf, 4)
        })

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","model","hit","max_conf"])
        w.writeheader(); w.writerows(rows)

    hits = sum(r["hit"] for r in rows)
    print(f"Images: {len(rows)} | Hits@{args.conf}: {hits} ({hits/len(rows):.3f})")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()


# python compare_scores.py --a results\yoloworld_scores.csv --b results\drone_yolo_scores.csv `
#    --manifest data\meta\manifest.csv --out results\compare\side_by_side.csv
