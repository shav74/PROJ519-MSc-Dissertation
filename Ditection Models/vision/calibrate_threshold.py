# calibrate_threshold.py
import argparse, pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)     # e.g., results/drone_yolo_scores.csv
ap.add_argument("--manifest", required=True)   # data/meta/manifest.csv
ap.add_argument("--target_fpr", type=float, default=0.05)
args = ap.parse_args()

df = pd.read_csv(args.scores)
df["path"] = df["path"].astype(str).str.replace("\\","/", regex=False)
man = pd.read_csv(args.manifest)
man["path"] = man["path"].astype(str).str.replace("\\","/", regex=False)
# robust pos/neg flags
s = man.get("drone_present", pd.Series([""]*len(man))).astype(str).str.strip().str.lower()
pos = (pd.to_numeric(s, errors="coerce")==1) | s.isin({"1","true","yes","pos","positive"})
neg = (pd.to_numeric(s, errors="coerce")==0) | s.isin({"0","false","no","neg","negative"})
m = df.merge(man[["path"]], on="path", how="inner")  # just align paths
m = m.merge(pd.DataFrame({"path":man["path"], "__is_pos":pos, "__is_neg":neg}), on="path", how="left")

neg_scores = m.loc[m["__is_neg"], "max_conf"].to_numpy()
thr = float(np.quantile(neg_scores, 1 - args.target_fpr)) if len(neg_scores) else 0.25

pos_scores = m.loc[m["__is_pos"], "max_conf"].to_numpy()
tpr = float((pos_scores >= thr).mean()) if len(pos_scores) else 0.0
fpr = float((neg_scores >= thr).mean()) if len(neg_scores) else 0.0

print({"chosen_threshold": thr, "tpr_at_thr": tpr, "fpr_at_thr": fpr, "n_pos": int((m['__is_pos']).sum()), "n_neg": int((m['__is_neg']).sum())})
