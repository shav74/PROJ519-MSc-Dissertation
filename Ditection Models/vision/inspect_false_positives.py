# inspect_false_positives.py
import argparse, os, numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgdir", required=True)                 # folder with negatives (e.g., data/images_eval/dronevsbird/neg)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="results/fp_inspect")
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = getattr(model, "names", {}) or {}
    drone_labels = {"drone", "uav", "quadcopter"}
    drone_ids = [i for i,n in names.items() if str(n).lower() in drone_labels]
    if not drone_ids and len(names)==1:
        drone_ids = [list(names.keys())[0]]
    print("classes:", names, " drone_ids:", drone_ids)

    exts = {".jpg",".jpeg",".png"}
    paths = [p for p in Path(args.imgdir).rglob("*") if p.suffix.lower() in exts]
    scores = []
    for p in paths:
        r = model.predict(str(p), conf=args.conf, verbose=False)[0]
        s = 0.0
        if r.boxes is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            if drone_ids:
                m = np.isin(cls, np.array(drone_ids))
                if m.any(): s = float(conf[m].max())
        scores.append((s, p, r))

    scores.sort(reverse=True, key=lambda t: t[0])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for i, (s, p, r) in enumerate(scores[:args.topk]):
        img = cv2.imread(str(p))
        if r.boxes is not None and len(r.boxes):
            for xyxy, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy().astype(float)):
                if cls in drone_ids:
                    x1,y1,x2,y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, f"{names.get(cls,cls)} {conf:.2f}", (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imwrite(str(out / f"{i:02d}_{s:.2f}_{p.name}"), img)
    print(f"Saved top-{args.topk} FPs to {out}")

if __name__ == "__main__":
    main()
