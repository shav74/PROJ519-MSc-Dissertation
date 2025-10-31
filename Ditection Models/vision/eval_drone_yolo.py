# eval_video_drone_yolo.py
import argparse, csv, cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--conf", type=float, default=0.001)
ap.add_argument("--fps", type=float, default=2.0)
ap.add_argument("--out", default="results/drone_yolo_video_scores.csv")
args = ap.parse_args()

model = YOLO(args.weights)
names = getattr(model, "names", {}) or {}
drone_labels = {"drone","uav","quadcopter"}
drone_ids = [i for i,n in names.items() if str(n).strip().lower() in drone_labels]
if not drone_ids and len(names)==1:
    drone_ids=[list(names.keys())[0]]
print("classes:", names, " drone_ids:", drone_ids)

cap = cv2.VideoCapture(args.video)
native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = max(1, int(round(native_fps/args.fps)))

rows=[]; i=0
while True:
    ok, frame = cap.read()
    if not ok: break
    if i % step == 0:
        t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        r = model.predict(frame, conf=args.conf, verbose=False)[0]
        score=0.0; hit=0
        if r.boxes is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            if drone_ids:
                m = np.isin(cls, np.array(drone_ids))
                if m.any():
                    score=float(conf[m].max()); hit=1
        rows.append({"time_s": round(t,2), "frame": i, "hit": hit, "max_conf": round(score,4)})
    i+=1

import os; os.makedirs("results", exist_ok=True)
with open(args.out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["time_s","frame","hit","max_conf"])
    w.writeheader(); w.writerows(rows)
print("Saved", args.out, "rows:", len(rows))
