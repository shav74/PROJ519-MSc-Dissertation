import argparse, csv, os, shutil
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLOWorld
from tqdm import tqdm


def list_images(root, recursive=True):
    exts = (".jpg", ".jpeg", ".png")
    p = Path(root)
    if recursive:
        return sorted([str(x) for x in p.rglob("*") if x.suffix.lower() in exts])
    return sorted([str(x) for x in p.iterdir() if x.suffix.lower() in exts])


def draw_boxes(img, boxes_xyxy, conf, cls, name_map):
    h, w = img.shape[:2]
    t = max(2, (h + w) // 600)
    for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, conf, cls):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        x1 = max(0, min(x1, w - 1));
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1));
        y2 = max(0, min(y2, h - 1))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), t)
        label = f"{name_map.get(int(k), str(int(k)))} {float(c):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, t)
        ytxt = max(th + t, y1 - 5)
        cv2.rectangle(img, (x1, ytxt - th - t), (x1 + tw + t * 2, ytxt + t // 2), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + t, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), t, cv2.LINE_AA)
    return img


def safe_copy(src, dst_dir):
    dst = Path(dst_dir) / Path(src).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description="YOLO-World: annotate & split hits/misses.")
    ap.add_argument("--imgdir", required=True, help="Folder with images")
    ap.add_argument("--outdir", default="results/yw_inspect")
    ap.add_argument("--prompts", default="drone,quadcopter,uav")
    ap.add_argument("--conf", type=float, default=0.25, help="threshold that defines a 'hit'")
    ap.add_argument("--save_crops", action="store_true", help="Also save per-detection crops")
    ap.add_argument("--no_recursive", action="store_true")
    args = ap.parse_args()

    paths = list_images(args.imgdir, recursive=not args.no_recursive)
    if not paths:
        print("No images found.")
        return

    print(f"Found {len(paths)} images")

    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    print(f"Loading YOLO-World with prompts: {prompts}")
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(prompts)
    name_map = getattr(model, "names", {}) or {}

    out = Path(args.outdir)
    anno_hits = out / "annotated" / "hits"
    anno_miss = out / "annotated" / "misses"
    split_hits = out / "split" / "hits"
    split_miss = out / "split" / "misses"
    crops_dir = out / "crops"

    rows = []
    for p in tqdm(paths, desc="YOLO-World"):
        r = model.predict(p, conf=args.conf, verbose=False)[0]
        hit = int(r.boxes is not None and len(r.boxes) > 0)
        max_conf = float(r.boxes.conf.max().cpu().numpy()) if hit else 0.0
        n_det = int(len(r.boxes)) if hit else 0

        # Save annotated
        img = cv2.imread(p)
        if img is None:
            continue
        if hit:
            boxes = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            img_anno = draw_boxes(img.copy(), boxes, conf, cls, name_map)
            anno_path = anno_hits / Path(p).name
            anno_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(anno_path), img_anno)

            # optional crops
            if args.save_crops:
                for i, ((x1, y1, x2, y2), c, k) in enumerate(zip(boxes, conf, cls)):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size:
                        lab = name_map.get(int(k), str(int(k)))
                        crop_out = crops_dir / lab
                        crop_out.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(crop_out / f"{Path(p).stem}_{i}_{lab}_{c:.2f}.jpg"), crop)
            # copy original to hits
            safe_copy(p, split_hits)
        else:
            anno_path = anno_miss / Path(p).name
            anno_path.parent.mkdir(parents=True, exist_ok=True)
            # Write a tiny header saying "NO DET"
            h, w = img.shape[:2]
            t = max(2, (h + w) // 600)
            cv2.putText(img, "NO DET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), t, cv2.LINE_AA)
            cv2.imwrite(str(anno_path), img)
            # copy original to misses
            safe_copy(p, split_miss)

        rows.append({
            "path": Path(p).as_posix(),
            "model": "yoloworld",
            "hit": hit,
            "n_detections": n_det,
            "max_conf": round(max_conf, 4)
        })

    # write per-image CSV
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "model", "hit", "n_detections", "max_conf"])
        w.writeheader()
        w.writerows(rows)

    hits = sum(r["hit"] for r in rows)
    print(f"\n✓ Done. Hits: {hits}/{len(rows)}  (thr={args.conf})")
    print(f"Annotated: {anno_hits} / {anno_miss}")
    print(f"Split originals: {split_hits} / {split_miss}")
    if args.save_crops:
        print(f"Crops: {crops_dir}")


if __name__ == "__main__":
    main()
