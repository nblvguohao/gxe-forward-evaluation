"""E1 (docs/prereg_sequel_wave2a_2026-09-28.md §5): per-epoch decomposition of the test-year predictions.

For every run and epoch (definitions as scripts/w1b_tstar.py): m_j, t_j and s_ij per environment; t, a, v_b,
k, S; t* = k v_b / a; r_max; pooled r; within-environment Spearman (environments with >= 25 cells);
phi = (a^2/v_b) / (a^2/v_b + k^2), the share of r_max^2 that comes from environment means. a <= 0 -> t*,
phi undefined. GE-BiFormer runs also get the pooled Huber loss (their selection criterion).

Own-type runs (GE-BiFormer own and own_clean, GEFormer own): e_P = argmax pooled r, e_S = argmax within
Spearman. E1-1: phi(e_P) > phi(e_S). E1-2: |log(t/t*)|(e_P) < |log(t/t*)|(e_S). One-sided sign tests over
runs with e_P != e_S and both values defined. Descriptive; nothing here selects a reported method."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

from dartgxe.paths import RESULTS

OUT = RESULTS / "e1"
SRC = {"GEBiFormer": RESULTS / "a1" / "epochs", "GEFormer": RESULTS / "a1" / "geformer_epochs"}
EXPECT = {"GEBiFormer": [(sc, pr, s) for sc in ("F2024m", "F2022m") for pr in ("own", "own_clean", "fair") for s in (42, 1, 2)],
          "GEFormer": [(sc, pr, s) for sc in ("F2024m", "F2022m") for pr in ("own", "fair") for s in (147, 1, 2)]}
missing = [(m, *k) for m, ks in EXPECT.items() for k in ks if not (SRC[m] / f"{k[0]}_{k[1]}_seed{k[2]}.parquet").exists()]
if missing:
    sys.exit(f"E1 incomplete, {len(missing)} runs missing, e.g. {missing[:3]}")


def huber(r, d=1.0):
    a = np.abs(r)
    return float(np.mean(np.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))))


def decompose(g: pd.DataFrame) -> dict:
    y, yh, env = g["observed"].to_numpy(float), g["prediction"].to_numpy(float), g["env"].to_numpy()
    df = pd.DataFrame({"env": env, "y": y, "p": yh})
    grp = df.groupby("env")
    m = grp["p"].transform("mean").to_numpy()
    tj = grp["p"].transform(lambda v: v.std(ddof=0)).to_numpy()
    s = np.where(tj > 0, (yh - m) / np.where(tj > 0, tj, 1), 0.0)
    df["s"] = s
    n = grp.size()
    covw = df.groupby("env").apply(lambda q: float(np.mean((q.s - q.s.mean()) * (q.y - q.y.mean()))), include_groups=False)
    k = float((covw * n).sum() / n.sum())
    t = float(pd.Series(tj, index=env).groupby(level=0).first().reindex(n.index).mul(n).sum() / n.sum())
    a = float(np.mean((m - m.mean()) * (y - y.mean())))
    vb, S = float(m.var()), float(y.var())
    ok = a > 0 and vb > 0
    between = a * a / vb if ok else np.nan
    sp = [np.corrcoef(rankdata(q.p), rankdata(q.y))[0, 1] for _, q in grp if len(q) >= 25]
    return {"t": t, "a": a, "v_b": vb, "k": k, "S": S, "t_star": k * vb / a if ok else np.nan,
            "r_max": float(np.sqrt((between + k * k) / S)) if ok else np.nan,
            "phi": between / (between + k * k) if ok else np.nan,
            "pooled_r": float(np.corrcoef(yh, y)[0, 1]), "within_spearman": float(np.nanmean(sp)),
            "pooled_huber": huber(yh - y)}


rows = []
for model, keys in EXPECT.items():
    for sc, pr, s in keys:
        d = pd.read_parquet(SRC[model] / f"{sc}_{pr}_seed{s}.parquet")
        for ep, g in d.groupby("epoch"):
            rows.append({"model": model, "scenario": sc, "protocol": pr, "seed": s, "epoch": int(ep), **decompose(g)})
        print(model, sc, pr, s, "done", flush=True)
E = pd.DataFrame(rows)
E["abs_log_t_tstar"] = np.abs(np.log(E["t"] / E["t_star"]))
OUT.mkdir(parents=True, exist_ok=True)
E.to_parquet(OUT / "epochs.parquet", index=False)

runs = []
for (model, sc, pr, s), g in E.groupby(["model", "scenario", "protocol", "seed"]):
    g = g.set_index("epoch")
    rec = {"model": model, "scenario": sc, "protocol": pr, "seed": s, "epochs": len(g),
           "e_P": int(g["pooled_r"].idxmax()), "e_S": int(g["within_spearman"].idxmax())}
    if model == "GEBiFormer":
        rec["e_L"] = int(g["pooled_huber"].idxmin())
        rec["phi_e_L"] = float(g.loc[rec["e_L"], "phi"])
    for tag in ("e_P", "e_S"):
        rec[f"phi_{tag}"] = float(g.loc[rec[tag], "phi"])
        rec[f"abslog_{tag}"] = float(g.loc[rec[tag], "abs_log_t_tstar"])
        rec[f"within_{tag}"] = float(g.loc[rec[tag], "within_spearman"])
        rec[f"pooled_{tag}"] = float(g.loc[rec[tag], "pooled_r"])
    runs.append(rec)
R = pd.DataFrame(runs)
R.to_csv(OUT / "runs.csv", index=False)
own = R[R["protocol"].isin(["own", "own_clean"])]
verd = {"own_type_runs": len(own), "runs_eP_eq_eS": int((own.e_P == own.e_S).sum())}
for name, col_P, col_S, better in (("E1_1_phi", "phi_e_P", "phi_e_S", "gt"), ("E1_2_abslog", "abslog_e_P", "abslog_e_S", "lt")):
    u = own[(own.e_P != own.e_S) & own[col_P].notna() & own[col_S].notna()]
    wins = int((u[col_P] > u[col_S]).sum() if better == "gt" else (u[col_P] < u[col_S]).sum())
    n = len(u)
    verd[name] = {"n_used": n, "in_direction": wins, "p_one_sided": binomtest(wins, n, 0.5, alternative="greater").pvalue if n else None}
# consistency with gate 2 (prereg §1): the rerun's reported epoch vs the gate-2 prediction file
cons = []
for sc in ("F2024m", "F2022m"):
    for pr in ("own", "fair"):
        for s in (147, 1, 2):
            d = pd.read_parquet(SRC["GEFormer"] / f"{sc}_{pr}_seed{s}.parquet")
            ep = int(R[(R.model == "GEFormer") & (R.scenario == sc) & (R.protocol == pr) & (R.seed == s)]["e_P"].iloc[0]) \
                if pr == "own" else int(d["epoch"].max())
            new = d[d.epoch == ep][["env", "genotype", "prediction"]]
            old = pd.read_parquet(RESULTS / "predictions" / "g2f" / sc / f"GEFormer_{pr}" / f"seed{s}.parquet")
            j = old.merge(new, on=["env", "genotype"], suffixes=("_gate2", "_rerun"))
            cons.append({"scenario": sc, "protocol": pr, "seed": s, "epoch": ep, "cells": len(j),
                         "corr": float(np.corrcoef(j.prediction_gate2, j.prediction_rerun)[0, 1]),
                         "max_abs_diff": float((j.prediction_gate2 - j.prediction_rerun).abs().max())})
pd.DataFrame(cons).to_csv(OUT / "geformer_rerun_vs_gate2.csv", index=False)
verd["geformer_rerun_vs_gate2_min_corr"] = min(c["corr"] for c in cons)
json.dump(verd, open(OUT / "verdict.json", "w"), indent=1, default=float)
print(R.round(3).to_string(index=False))
print(json.dumps(verd, indent=1, default=float))
