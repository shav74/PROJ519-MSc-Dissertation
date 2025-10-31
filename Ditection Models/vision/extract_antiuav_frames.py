import argparse, os
from pathlib import Path
import cv2

def extract_from_sequence(seq_dir: Path, fps_out: float, limit: int):
    vid = seq_dir / "visible.mp4"
    if not vid.exists():
        print(f"[skip] no visible.mp4 in {seq_dir}")
        return 0
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        print(f"[warn] cannot open {vid}")
        return 0

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps_in / max(1e-6, fps_out))))  # e.g., 25fps -> stride 25 for 1 fps
    saved = 0
    frame_idx = -1  # 0-based counter of *read* frames (original index is frame_idx+1)

    print(f"[seq] {seq_dir.name} | fps_in≈{fps_in:.2f} | stride={stride} | target={limit} frames")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        # save with ORIGINAL frame number (1-based) => matches Anti-UAV JSON indexing better
        out_name = f"{frame_idx+1:06d}.jpg"
        out_path = seq_dir / out_name
        cv2.imwrite(str(out_path), frame)
        saved += 1
        if limit and saved >= limit:
            break
    cap.release()
    print(f"[done] {seq_dir.name}: saved {saved} frames")
    return saved

def main():
    ap = argparse.ArgumentParser(description="Extract RGB frames from Anti-UAV visible.mp4, preserving original frame index.")
    ap.add_argument("--root", default="data/images_eval/antiuav_vis", help="Root folder containing Anti-UAV VIS sequences")
    ap.add_argument("--fps", type=float, default=1.0, help="Sampling rate (frames per second) to export (approx.)")
    ap.add_argument("--limit", type=int, default=50, help="Max frames to save per sequence")
    args = ap.parse_args()

    root = Path(args.root)
    seqs = [p for p in root.iterdir() if p.is_dir()]
    total = 0
    for seq in seqs:
        total += extract_from_sequence(seq, fps_out=args.fps, limit=args.limit)
    print(f"\n[summary] total frames saved across sequences: {total}")

if __name__ == "__main__":
    main()
