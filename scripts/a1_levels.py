"""Descriptive companion to scripts/a1_analyze.py: the levels behind the pre-registered contrasts (mean
within-environment Spearman, selection differential at 10 %, pooled Pearson per method on the same common
cells and scored environments) and the epoch each run reported. Writes results/a1/{levels.csv, epochs.csv}."""
import json

import numpy as np
import pandas as pd

from dartgxe.eval.metrics import per_env
from dartgxe.paths import RESULTS

SEEDS = [42, 1, 2]
P = RESULTS / "predictions" / "g2f"
rows = []
for sc in ("F2024m", "F2022m"):
    preds = [pd.read_parquet(P / sc / f"GEBiFormer_{pr}" / f"seed{s}.parquet") for pr in ("own", "own_clean", "fair") for s in SEEDS]
    reml = pd.read_parquet(sorted((P / sc / "B1r_gblup_reml").glob("seed*.parquet"))[0])
    preds += [reml.assign(seed=s, method="B1r_gblup_reml") for s in SEEDS]
    allp = pd.concat(preds, ignore_index=True)
    allp["key"] = allp["env"].astype(str) + "|" + allp["genotype"].astype(str)
    nm = allp.groupby("key")["method"].nunique()
    allp = allp[allp["key"].isin(nm[nm == allp["method"].nunique()].index)].drop(columns="key")
    pe = per_env(allp)
    scored = set(pe.loc[(pe.method == "GEBiFormer_fair") & (pe.metric == "spearman"), "env"])
    for (m, s), d in allp[allp.env.isin(scored)].groupby(["method", "seed"]):
        q = pe[(pe.method == m) & (pe.seed == s)]
        rows.append({"scenario": sc, "method": m, "seed": s,
                     "within_spearman": q.loc[q.metric == "spearman", "value"].mean(),
                     "sel_diff_f10": q.loc[q.metric == "sel_diff_f10", "value"].mean(),
                     "pooled_pearson": float(np.corrcoef(d.prediction, d.observed)[0, 1])})
L = pd.DataFrame(rows)
L.to_csv(RESULTS / "a1" / "levels.csv", index=False)
print(L.groupby(["scenario", "method"])[["within_spearman", "sel_diff_f10", "pooled_pearson"]].agg(["mean", "std"]).round(3).to_string())
ep = []
for f in sorted((RESULTS / "a1" / "runs").glob("*.json")):
    d = json.load(open(f))
    ep.append({"scenario": d["scenario"], "protocol": d["protocol"], "seed": d["seed"], "reported_epoch": d["reported_epoch"],
               "epochs_run": d["log"]["epochs_run"], "stopped_early": d["log"]["stopped_early"],
               "tune_best_epoch": (d["tune_log"] or {}).get("best_epoch"), "tune_epochs_run": (d["tune_log"] or {}).get("epochs_run")})
E = pd.DataFrame(ep).sort_values(["scenario", "protocol", "seed"])
E.to_csv(RESULTS / "a1" / "epochs.csv", index=False)
print(E.to_string(index=False))
