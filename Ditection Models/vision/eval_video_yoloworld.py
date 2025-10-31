# eval_video_yoloworld.py
import argparse, csv, cv2, time
from ultralytics import YOLOWorld

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--conf", type=float, default=0.001)
ap.add_argument("--fps", type=float, default=2.0)   # sample every ~0.5 s
ap.add_argument("--out", default="results/yoloworld_video_scores.csv")
ap.add_argument("--prompts", default="drone,quadcopter,uav")
args = ap.parse_args()

model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes([s.strip() for s in args.prompts.split(",") if s.strip()])

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
        if r.boxes and len(r.boxes):
            score = float(r.boxes.conf.max().cpu().numpy())
            hit=1
        else:
            score=0.0; hit=0
        rows.append({"time_s": round(t,2), "frame": i, "hit": hit, "max_conf": round(score,4)})
    i+=1

with open(args.out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["time_s","frame","hit","max_conf"])
    w.writeheader(); w.writerows(rows)
print("Saved", args.out, "rows:", len(rows))


# python eval_video_yoloworld.py --video data/test_videos/VID-20251031-WA0001.mp4 --fps 2 --conf 0.001 --out results/T01_yw.csv