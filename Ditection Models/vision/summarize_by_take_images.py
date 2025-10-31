# summarize_by_take.py
import pandas as pd, sys
scores = pd.read_csv(sys.argv[1]); scores["path"]=scores["path"].str.replace("\\","/",regex=False)
man    = pd.read_csv("data/meta/manifest_own.csv"); man["path"]=man["path"].str.replace("\\","/",regex=False)
df = scores.merge(man[["path","take_id","mod"]], on="path", how="left")
g = df.groupby(["model","take_id","mod"])["max_conf"]
out = g.agg(median="median", mean="mean", p75=lambda x: x.quantile(0.75), n="count").reset_index()
out.to_csv("results/own_takes_summary.csv", index=False)
print(out.head(20))
