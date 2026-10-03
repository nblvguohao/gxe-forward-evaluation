"""A1 analysis (docs/prereg_sequel_wave2a_2026-09-28.md §3-4). Refuses to run until all 18 GE-BiFormer runs
exist. GEFormer rows come from gate 2 and are not recomputed here.

O = own - fair; O_sel = own_clean - fair; O_scale = own - own_clean; C_fair = fair - REML-GBLUP.
Within-environment metrics: two-level bootstrap (env x seed) via paired_delta, verdict with 3/sqrt(N),
N = test cells in the scored environments. Pooled Pearson: the same two-level resampling (environments,
then seeds), pooled r per seed on the resampled environments, averaged over the resampled seeds.
REML-GBLUP is deterministic; its predictions are paired with every GE-BiFormer seed."""
import json
import sys

import numpy as np
import pandas as pd

from dartgxe.eval.compare import paired_delta, resolution, verdict
from dartgxe.eval.metrics import MIN_N, env_seed_matrix, per_env
from dartgxe.paths import RESULTS

SEEDS = [42, 1, 2]
PROTO = ["own", "own_clean", "fair"]
P = RESULTS / "predictions" / "g2f"
OUT = RESULTS / "a1"
B = 2000
CONTRASTS = [("O", "GEBiFormer_own", "GEBiFormer_fair"), ("O_sel", "GEBiFormer_own_clean", "GEBiFormer_fair"),
             ("O_scale", "GEBiFormer_own", "GEBiFormer_own_clean"), ("C_fair", "GEBiFormer_fair", "B1r_gblup_reml")]

missing = [(sc, pr, s) for sc in ("F2024m", "F2022m") for pr in PROTO for s in SEEDS
           if not (P / sc / f"GEBiFormer_{pr}" / f"seed{s}.parquet").exists()]
if missing:
    sys.exit(f"A1 incomplete, {len(missing)} runs missing, e.g. {missing[:3]}; refusing to score")


def pooled_r(d):
    return float(np.corrcoef(d["prediction"], d["observed"])[0, 1])


rows, verd = [], {}
for sc in ("F2024m", "F2022m"):
    preds = [pd.read_parquet(P / sc / f"GEBiFormer_{pr}" / f"seed{s}.parquet") for pr in PROTO for s in SEEDS]
    reml = pd.read_parquet(sorted((P / sc / "B1r_gblup_reml").glob("seed*.parquet"))[0])
    preds += [reml.assign(seed=s, method="B1r_gblup_reml") for s in SEEDS]
    allp = pd.concat(preds, ignore_index=True)
    key = allp[["env", "genotype"]].astype(str).agg("|".join, axis=1)
    allp["key"] = key
    n_methods = allp.groupby("key")["method"].nunique()
    allp = allp[allp["key"].isin(n_methods[n_methods == allp["method"].nunique()].index)]  # common cells
    dup = allp.duplicated(["method", "seed", "key"]).sum()
    assert dup == 0, f"{dup} duplicated cells"
    pe = per_env(allp.drop(columns="key"))
    scored = pe[(pe["method"] == "GEBiFormer_fair") & (pe["seed"] == SEEDS[0]) & (pe["metric"] == "spearman")]
    N = int(scored["n"].sum())
    thr = resolution(N)
    envs = sorted(scored["env"])
    for metric in ("spearman", "sel_diff_f10"):
        mat = env_seed_matrix(pe, metric)
        for name, a, b in CONTRASTS:
            r = paired_delta(mat, a, b, B=B)
            rows.append({"scenario": sc, "contrast": name, "metric": metric, "N": N, "resolution": thr, **r,
                         "verdict": verdict(r["delta"], r["ci2_lo"], r["ci2_hi"], thr, r["seeds_same_sign"])})
    # pooled Pearson on the scored environments, two-level bootstrap
    sub = allp[allp["env"].isin(envs)]
    by = {(m, s): {e: g for e, g in d.groupby("env")} for (m, s), d in sub.groupby(["method", "seed"])}
    point = {m: np.mean([pooled_r(sub[(sub.method == m) & (sub.seed == s)]) for s in SEEDS]) for m in sub["method"].unique()}
    rng = np.random.default_rng(0)
    bs = {name: [] for name, _, _ in CONTRASTS}
    for _ in range(B):
        es = rng.choice(envs, len(envs))
        ss = rng.choice(SEEDS, len(SEEDS))
        val = {m: np.mean([pooled_r(pd.concat([by[(m, s)][e] for e in es])) for s in ss]) for m in point}
        for name, a, b in CONTRASTS:
            bs[name].append(val[a] - val[b])
    for name, a, b in CONTRASTS:
        lo, hi = np.quantile(bs[name], [0.025, 0.975])
        rows.append({"scenario": sc, "contrast": name, "metric": "pooled_pearson", "N": N, "resolution": thr,
                     "a": a, "b": b, "delta": point[a] - point[b], "ci2_lo": lo, "ci2_hi": hi, "n_env": len(envs),
                     "n_seed": len(SEEDS), "verdict": "inflated" if lo > 0 else ("deflated" if hi < 0 else "no detectable change")})
    verd[sc] = {"pooled_pearson": {m: point[m] for m in point}, "n_env": len(envs), "N": N, "resolution": thr}

T = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
T.to_csv(OUT / "summary.csv", index=False)
g = T.set_index(["scenario", "contrast", "metric"])["delta"]
verd["prediction_1_O_pooled_positive_both"] = bool(all(g[(sc, "O", "pooled_pearson")] > 0 for sc in ("F2024m", "F2022m")))
verd["prediction_2_scale_smaller_than_sel_both"] = bool(all(abs(g[(sc, "O_scale", "pooled_pearson")]) < abs(g[(sc, "O_sel", "pooled_pearson")])
                                                        for sc in ("F2024m", "F2022m")))
json.dump(verd, open(OUT / "verdict.json", "w"), indent=1, default=float)
print(T[["scenario", "contrast", "metric", "delta", "ci2_lo", "ci2_hi", "verdict"]].round(4).to_string(index=False))
print(json.dumps(verd, indent=1, default=float))
