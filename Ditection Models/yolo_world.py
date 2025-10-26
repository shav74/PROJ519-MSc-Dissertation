import argparse, csv, glob, os
from pathlib import Path
from ultralytics import YOLOWorld
from tqdm import tqdm

def list_images(root, recursive=True):
    exts = (".jpg", ".jpeg", ".png")
    if recursive:
        return sorted([str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts])
    return sorted([str(p) for p in Path(root).iterdir() if p.suffix.lower() in exts])

def main():
    ap = argparse.ArgumentParser(description="YOLO-World on a folder; writes per-image scores (max_conf).")
    ap.add_argument("--imgdir", required=True, help="Folder with images")
    ap.add_argument("--conf", type=float, default=0.001, help="Low conf to keep weak detections for scoring")
    ap.add_argument("--prompts", default="drone,quadcopter,uav", help="Comma list of class prompts")
    ap.add_argument("--out", default="results_yoloworld.csv", help="Output CSV path")
    ap.add_argument("--no_recursive", action="store_true", help="Do not scan subfolders")
    args = ap.parse_args()

    paths = list_images(args.imgdir, recursive=not args.no_recursive)
    if not paths:
        print("No images found.")
        return

    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]

    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(prompts)

    rows = []
    for p in tqdm(paths, desc="YOLO-World"):
        r = model.predict(p, conf=args.conf, verbose=False)[0]
        hit = int(len(r.boxes) > 0)
        max_conf = float(r.boxes.conf.max().cpu().numpy()) if hit else 0.0
        rows.append({"path": p, "model": "yoloworld", "hit": hit, "max_conf": round(max_conf, 4)})

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","model","hit","max_conf"])
        w.writeheader(); w.writerows(rows)

    # quick console summary
    hits = sum(r["hit"] for r in rows)
    print(f"Images: {len(rows)} | Hits@{args.conf}: {hits} ({hits/len(rows):.3f})")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()

#python yolo_world.py --imgdir data/images_eval --conf 0.001 --out results/yoloworld_scores.cs