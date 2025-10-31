import argparse, csv, os, numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

def list_images(root, recursive=True):
    exts = (".jpg", ".jpeg", ".png")
    p = Path(root)
    if recursive:
        return sorted([str(x) for x in p.rglob("*") if x.suffix.lower() in exts])
    return sorted([str(x) for x in p.iterdir() if x.suffix.lower() in exts])

def norm(s): return str(s).strip().lower()

def main():
    ap = argparse.ArgumentParser(description="Drone-specific YOLO → per-image scores")
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--weights", required=True, help=".pt checkpoint")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--out", default="results/drone_yolo_scores.csv")
    ap.add_argument("--no_recursive", action="store_true")
    ap.add_argument("--drone_labels", default="drone,uav,quadcopter",
                    help="Comma list of class names to treat as 'drone'")
    ap.add_argument("--list_classes", action="store_true", help="Print model class map and exit")
    ap.add_argument("--fallback_any", action="store_true",
                    help="If no matching labels found, treat ANY detected class as drone (debug)")
    args = ap.parse_args()

    paths = list_images(args.imgdir, recursive=not args.no_recursive)
    if not paths:
        print("No images found."); return

    model = YOLO(args.weights)
    name_map = getattr(model, "names", {}) or {}
    print("Model classes:", name_map)

    if args.list_classes:
        return  # just listing class names

    wanted = {norm(x) for x in args.drone_labels.split(",") if x.strip()}
    drone_ids = [i for i, n in name_map.items() if norm(n) in wanted]

    # sensible fallback: single-class model ⇒ assume it's drone
    if not drone_ids and len(name_map) == 1:
        drone_ids = [list(name_map.keys())[0]]

    if not drone_ids and not args.fallback_any:
        print("[WARN] No matching 'drone' labels found. "
              "Use --drone_labels to set names or --fallback_any to ignore class filtering.")
    if args.fallback_any:
        print("[DEBUG] --fallback_any enabled: any detection counts as 'drone'.")

    rows = []
    for p in tqdm(paths, desc="Drone-YOLO"):
        r = model.predict(p, conf=args.conf, verbose=False)[0]
        hit = 0; max_conf = 0.0
        if r.boxes is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            if drone_ids:
                m = np.isin(cls, np.array(drone_ids))
                if m.any(): hit = 1; max_conf = float(conf[m].max())
            elif args.fallback_any:
                hit = 1; max_conf = float(conf.max())
            # else: leave hit=0,max_conf=0.0

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
