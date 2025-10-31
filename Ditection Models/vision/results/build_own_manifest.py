# build_own_manifest.py
from pathlib import Path
import csv, re
root = Path('../data/images_eval/own')
rows=[]
for folder in root.iterdir():
    if not folder.is_dir(): continue
    take = folder.name                                  # e.g., T01_camo-dark_skyline
    m = re.match(r"(T\d+)_(.+)", take); tid = m.group(1) if m else take
    mod = m.group(2).replace("_"," ") if m else "unknown"
    for img in folder.rglob("*.jpg"):
        rows.append({
          "path": img.as_posix(),
          "source": "own",
          "take_id": tid,
          "mod": mod,
          "drone_present": "1",       # whole take is a positive
          "size_bin": ""              # unknown (ok for scores)
        })
Path("data/meta").mkdir(parents=True, exist_ok=True)
with open("../data/meta/manifest_own.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("Wrote data/meta/manifest_own.csv with", len(rows), "rows")
