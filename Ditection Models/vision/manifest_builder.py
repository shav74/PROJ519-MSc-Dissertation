# Build a manifest CSV for your RGB-only drone detectability study.
# Auto-fills most columns from folder structure + available labels,
# and creates a smaller subset CSV for manual tags (colour/shape/background).
#
# Expected layout:
# data/
#   images_eval/
#     antiuav_vis/
#       <seqA>/  000001.jpg, 000002.jpg, ...
#       <seqB>/  ...
#     dronevsbird/
#       drone/   *.jpg
#       neg/     *.jpg
#
# Optional Anti-UAV annotations inside each sequence folder:
#   visible_gt.txt | groundtruth.txt | visible.txt | gt.txt  (x,y,w,h[,visible])
#   or JSON like visible.json / gt.json with {"frame/index":..., "bbox":[x,y,w,h], "visible":1}
#
# Outputs:
#   data/meta/manifest.csv
#   data/meta/manifest_to_tag.csv

import os, csv, json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from PIL import Image
import numpy as np

ROOT = Path("..")
IM_ROOT = ROOT / "data" / "images_eval"
META_ROOT = ROOT / "data" / "meta"
META_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------

def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

def brightness_stats(img: Image.Image) -> Tuple[float, float]:
    g = img.convert("L")
    arr = np.array(g, dtype=np.uint8)
    return float(arr.mean()), float(arr.std())

def exposure_issue(img: Image.Image) -> str:
    g = img.convert("L")
    arr = np.array(g, dtype=np.uint8)
    total = arr.size
    if total == 0:
        return "none"
    over = (arr >= 245).sum() / total
    under = (arr <= 10).sum() / total
    if over > 0.45: return "over"
    if under > 0.45: return "under"
    return "none"

def backlit_flag(img: Image.Image) -> str:
    g = img.convert("L")
    arr = np.array(g, dtype=np.float32)
    if arr.size == 0: return "no"
    H = arr.shape[0]
    top = arr[:max(1, H//4), :].mean()
    rest = arr[max(1, H//4):, :].mean()
    return "yes" if (top - rest) >= 40.0 else "no"

def guess_lighting(mean: float, std: float, exp_issue: str) -> str:
    if exp_issue == "over": return "day"
    if exp_issue == "under": return "night-ish"
    if std < 28 and mean > 120: return "hazy"
    if 90 <= mean <= 140: return "cloudy"
    return "day"

def size_bin_from_h(h_px: Optional[float]) -> str:
    if h_px is None: return ""
    try:
        h = float(h_px)
    except Exception:
        return ""
    if h < 16: return "Tiny"
    if h < 32: return "Small"
    if h < 96: return "Medium"
    return "Large"

# ---------- Anti-UAV readers ----------

def find_antiuav_ann_file(seq_dir: Path) -> Optional[Path]:
    cand_names = [
        "visible_gt.txt","groundtruth.txt","visible.txt","gt.txt",
        "visible.json","annotations.json","gt.json","visible_gt.json"
    ]
    for name in cand_names:
        p = seq_dir / name
        if p.exists(): return p
    parent = seq_dir.parent
    for name in cand_names:
        p = parent / name
        if p.exists(): return p
    for p in list(seq_dir.glob("*.json")) + list(seq_dir.glob("*.txt")):
        if any(k in p.stem.lower() for k in ["vis","gt","ground"]): return p
    return None

def parse_antiuav_txt(txt_path: Path) -> Dict[int, Tuple[float,float,float,float,Optional[int]]]:
    out = {}
    with open(txt_path, "r", encoding="utf-8", newline="") as f:
        i = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 4: continue
            i += 1
            try:
                x,y,w,h = map(float, parts[:4])
                vis = None
                if len(parts) >= 5:
                    try: vis = int(float(parts[4]))
                    except: vis = None
                out[i] = (x,y,w,h,vis)
            except: continue
    return out

def parse_antiuav_json(json_path: Path) -> Dict[int, Tuple[float, float, float, float, Optional[int]]]:
    """
    Support common Anti-UAV JSON layouts:
      A) top-level lists:  {"gt_rect": [[x,y,w,h], ...], "visible": [0/1,...]}
      B) frame objects:    {"frames":[{"index":N, "bbox":[x,y,w,h], "visible":1}, ...]}
      C) generic           {"annotations":[{"frame_id":N, "bbox":[x,y,w,h], "visible":1}, ...]}
    Returns: dict frame_idx(1-based) -> (x, y, w, h, visible_flag or None)
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out = {}

    # ---- Pattern A: top-level lists (most Anti-UAV visible.json files)
    if isinstance(data, dict) and isinstance(data.get("gt_rect"), list):
        rects = data["gt_rect"]
        vis_list = data.get("visible", None)  # sometimes present
        for i, rect in enumerate(rects):
            if not rect or len(rect) < 4:
                continue
            x, y, w, h = rect[:4]
            # invalid or missing boxes often appear as zeros/negatives
            vflag = None
            if vis_list is not None and i < len(vis_list):
                try:
                    vflag = int(bool(vis_list[i]))
                except Exception:
                    vflag = None
            # if width/height are non-positive, mark as not visible
            if (w is None or h is None) or (float(w) <= 0 or float(h) <= 0):
                vflag = 0 if vflag is None else vflag
            # Use 1-based frame index to match 000123.jpg naming
            out[i + 1] = (float(x), float(y), float(w), float(h), vflag)
        return out

    # ---- Pattern B: frames list
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        for entry in data["frames"]:
            idx = int(entry.get("index", 0) or entry.get("frame", 0) or 0)
            bbox = entry.get("bbox") or entry.get("gt_rect") or entry.get("rect")
            vis = entry.get("visible")
            if not bbox or len(bbox) < 4 or idx <= 0:
                continue
            x, y, w, h = map(float, bbox[:4])
            vflag = None
            if vis is not None:
                try: vflag = int(bool(vis))
                except Exception: vflag = None
            out[idx] = (x, y, w, h, vflag)
        return out

    # ---- Pattern C: annotations list
    if isinstance(data, dict) and isinstance(data.get("annotations"), list):
        for entry in data["annotations"]:
            idx = int(entry.get("frame_id", 0) or entry.get("frame", 0) or 0)
            bbox = entry.get("bbox") or entry.get("gt_rect") or entry.get("rect")
            vis = entry.get("visible")
            if not bbox or len(bbox) < 4 or idx <= 0:
                continue
            x, y, w, h = map(float, bbox[:4])
            vflag = None
            if vis is not None:
                try: vflag = int(bool(vis))
                except Exception: vflag = None
            out[idx] = (x, y, w, h, vflag)
        return out

    return out


def load_antiuav_ann(seq_dir: Path):
    ann_path = find_antiuav_ann_file(seq_dir)
    if not ann_path: return {}
    if ann_path.suffix.lower() == ".json":
        return parse_antiuav_json(ann_path)
    return parse_antiuav_txt(ann_path)

# ---------- Drone-vs-Bird YOLO label reader (optional) ----------

def find_yolo_label(img_path: Path) -> Optional[Path]:
    p1 = img_path.with_suffix(".txt")
    if p1.exists(): return p1
    parent = img_path.parent
    labels_dir = parent.parent / "labels" if parent.name.lower() != "labels" else parent
    p2 = labels_dir / (img_path.stem + ".txt")
    if p2.exists(): return p2
    return None

def yolo_h_px_from_label(img_path: Path, lab_path: Path) -> Optional[float]:
    try:
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
    except Exception:
        return None
    try:
        lines = [ln.strip() for ln in lab_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines: return None
        max_h = 0.0
        for ln in lines:
            parts = ln.replace(",", " ").split()
            if len(parts) < 5: continue
            try:
                h_norm = float(parts[4])
            except:
                continue
            h_px = h_norm * H
            if h_px > max_h: max_h = h_px
        return max_h if max_h > 0 else None
    except Exception:
        return None

# ---------- Build manifest ----------

rows: List[dict] = []

# A) Anti-UAV VIS sequences
antiuav_root = IM_ROOT / "antiuav_vis"
if antiuav_root.exists():
    for seq_dir in sorted([p for p in antiuav_root.iterdir() if p.is_dir()]):
        ann = load_antiuav_ann(seq_dir)
        img_paths = list_images(seq_dir)
        for img_p in img_paths:
            # frame index from filename digits (e.g., 000123.jpg)
            try:
                stem = img_p.stem
                digits = "".join([c for c in stem if c.isdigit()])
                frame_idx = int(digits) if digits else None
            except:
                frame_idx = None

            gt_h = None; vis = None
            if frame_idx and frame_idx in ann:
                x,y,w,h,vis = ann[frame_idx]
                gt_h = float(h)

            size_bin = size_bin_from_h(gt_h)

            # image-level heuristics
            try:
                im = Image.open(img_p).convert("RGB")
                mean, std = brightness_stats(im)
                exp_issue = exposure_issue(im)
                backlit = backlit_flag(im)
                lighting = guess_lighting(mean, std, exp_issue)
            except Exception:
                exp_issue = "none"; backlit = "no"; lighting = ""

            rows.append({
                "path": img_p.as_posix(),
                "source": "antiuav_vis",
                "drone_present": "1" if gt_h is not None and (vis is None or int(vis) != 0) else "",
                "gt_box_h_px": f"{gt_h:.2f}" if gt_h is not None else "",
                "size_bin": size_bin,
                "lighting": lighting,
                "background": "",            # hand-tag later
                "backlit": backlit,
                "exposure_issue": exp_issue,
                "appearance_color": "",      # hand-tag later
                "appearance_shape": "",      # hand-tag later
                "notes": ""
            })

# B) Drone-vs-Bird images
dvb_root = IM_ROOT / "dronevsbird"
if dvb_root.exists():
    for sub, present in [("drone","1"), ("neg","0")]:
        subdir = dvb_root / sub
        if not subdir.exists(): continue
        for img_p in list_images(subdir):
            gt_h = None
            lab = find_yolo_label(img_p)
            if lab:
                gt_h = yolo_h_px_from_label(img_p, lab)
            size_bin = size_bin_from_h(gt_h)
            try:
                im = Image.open(img_p).convert("RGB")
                mean, std = brightness_stats(im)
                exp_issue = exposure_issue(im)
                backlit = backlit_flag(im)
                lighting = guess_lighting(mean, std, exp_issue)
            except Exception:
                exp_issue = "none"; backlit = "no"; lighting = ""
            rows.append({
                "path": img_p.as_posix(),
                "source": f"dronevsbird_{sub}",
                "drone_present": present,
                "gt_box_h_px": f"{gt_h:.2f}" if gt_h is not None else "",
                "size_bin": size_bin,
                "lighting": lighting,
                "background": "",            # hand-tag optional
                "backlit": backlit,
                "exposure_issue": exp_issue,
                "appearance_color": "",      # hand-tag optional
                "appearance_shape": "",      # hand-tag optional
                "notes": ""
            })

# Write manifest
manifest_path = META_ROOT / "manifest.csv"
fieldnames = [
    "path","source","drone_present","gt_box_h_px","size_bin",
    "lighting","background","backlit","exposure_issue",
    "appearance_color","appearance_shape","notes"
]

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows: w.writerow(r)

# Build a balanced subset to tag (≈200 rows total)
subset = []
cap_total = 200
caps = {
    ("antiuav_vis",        "Tiny"):   40,
    ("antiuav_vis",        "Small"):  30,
    ("antiuav_vis",        "Medium"): 20,
    ("antiuav_vis",        "Large"):  10,
    ("dronevsbird_drone",  "Tiny"):   20,   # if you add YOLO labels later
    ("dronevsbird_drone",  "Small"):  30,
    ("dronevsbird_drone",  "Medium"): 30,
    ("dronevsbird_drone",  "Large"):  20,
    ("dronevsbird_drone",  "Unknown"):30,   # no labels → size unknown
}
counts = {k: 0 for k in caps}

def key_for(r):
    src = r["source"]
    bin_ = r.get("size_bin", "") or "Unknown"
    return (src, bin_)

# positives only
pos_rows = [r for r in rows if r.get("drone_present") == "1"]

# pass 1: fill by per-(source, bin) caps
for r in pos_rows:
    k = key_for(r)
    if k in counts and counts[k] < caps[k]:
        subset.append(r)
        counts[k] += 1
    if len(subset) >= cap_total:
        break

# pass 2: top up with any remaining positives (to reach cap_total)
if len(subset) < cap_total:
    used = {s["path"] for s in subset}
    for r in pos_rows:
        if r["path"] in used:
            continue
        subset.append(r)
        if len(subset) >= cap_total:
            break


per_bin_limit = {"Tiny": 80, "Small": 60, "Medium": 40, "Large": 20}
per_bin_count = {k: 0 for k in per_bin_limit}

# First pass: positives with known size_bin
for r in rows:
    if r["drone_present"] != "1":
        continue
    bin_ = r["size_bin"]
    if bin_ in per_bin_limit and per_bin_count[bin_] < per_bin_limit[bin_]:
        subset.append(r); per_bin_count[bin_] += 1

# Fallback: if still too small (<150), top up with positives of unknown size
if len(subset) < 150:
    for r in rows:
        if r["drone_present"] == "1" and (r["size_bin"] == "" or r["size_bin"] is None):
            subset.append(r)
            if len(subset) >= 200:
                break

for r in rows:
    if r["drone_present"] != "1": continue
    bin_ = r["size_bin"]
    if bin_ in per_bin_limit and per_bin_count[bin_] < per_bin_limit[bin_]:
        subset.append(r); per_bin_count[bin_] += 1

subset_path = META_ROOT / "manifest_to_tag.csv"
with open(subset_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in subset: w.writerow(r)

print(f"Wrote manifest with {len(rows)} rows -> {manifest_path}")
print(f"Wrote subset to tag with {len(subset)} rows -> {subset_path}")
