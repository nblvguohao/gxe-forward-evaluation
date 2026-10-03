"""Descriptive summary for the evaluation paper (plan v2 §10.3, figures 4-5): every method and every selection rule
against cell_reml on the 48 forward target years already scored in the pre-registered waves (31 = wave 2b/2c/2d on
G2F/NUST/URSN; 17 = wave 2e on ESWYT/GEM_IA/MU_SOY). Nothing here is confirmatory; all inputs were seen before.

Methods (per environment, genotypes shared with cell_reml): the ten two-stage methods (b3, b3d), cell_reml, the G x E
learners with historical-mean and observed ECs (b3b; G2F and URSN only), the within-env-loss MLPs (b3c; seed mean;
31 years). Rules: the pre-registered rule tables (b3b library 'main', b3d), each rule minus R_CELL in the same table.
Year effect = mean over environments of (a - cell_reml); SE by environment bootstrap; DerSimonian-Laird pooling with
the zero-SE floor, separately for the 31 original years, the 17 independent years and all 48."""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from dartgxe.forward.pool import dersimonian_laird, floor_se
from dartgxe.paths import RESULTS

OUT = RESULTS / "summary48"
OUT.mkdir(parents=True, exist_ok=True)
B = 2000
rng = np.random.default_rng(20260930)
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
cnt = pd.read_csv(RESULTS / "b3d" / "count" / "per_year.csv")
T_OLD = {d: sorted(int(y) for y in g["year"]) for d, g in w1c[(w1c["rule"] == "main") & w1c["qualifies"]].groupby("dataset")}
T_NEW = {d: sorted(int(y) for y in g["year"]) for d, g in cnt[(cnt["rule"] == "main") & cnt["qualifies"]].groupby("dataset")}
COLS = ["env", "genotype", "y", "method", "pred"]


def panels(d, y, new):
    if new:
        return pd.read_parquet(RESULTS / "b3d" / "forward" / f"{d}_{y}.parquet", columns=COLS)
    parts = [pd.read_parquet(RESULTS / k / "forward" / f"{d}_{y}.parquet", columns=COLS) for k in ("b3", "b3b")]
    for m in ("dl_g", "dl_ge"):
        seeds = [RESULTS / "b3c" / "pred" / f"{d}_{y}_{m}_seed{s}.parquet" for s in range(3)]
        if all(f.exists() for f in seeds):
            s = pd.concat([pd.read_parquet(f, columns=["env", "genotype", "y", "pred"]) for f in seeds])
            parts.append(s.groupby(["env", "genotype"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
                         .assign(method=m))
    return pd.concat(parts, ignore_index=True)


def scores(y, p):
    sp = np.corrcoef(rankdata(p), rankdata(y))[0, 1] if np.std(p) > 0 and np.std(y) > 0 else np.nan
    k = max(1, int(round(0.1 * len(y))))
    top = np.argsort(-p, kind="mergesort")[:k]
    yz = (y - y.mean()) / y.std() if y.std() > 0 else np.zeros_like(y)
    return sp, float(yz[top].mean())


rows = []
for new, T in ((False, T_OLD), (True, T_NEW)):
    for d, years in T.items():
        for Y in years:
            P = panels(d, Y, new)
            W = P.pivot_table(index=["env", "genotype"], columns="method", values="pred")
            yv = P.drop_duplicates(["env", "genotype"]).set_index(["env", "genotype"])["y"]
            for e, g in W.groupby(level=0):
                for m in W.columns.drop("cell_reml"):
                    ok = g[[m, "cell_reml"]].dropna()
                    if len(ok) < 3:
                        continue
                    yy = yv.reindex(ok.index).to_numpy(float)
                    sa, ga = scores(yy, ok[m].to_numpy(float))
                    sb, gb = scores(yy, ok["cell_reml"].to_numpy(float))
                    rows.append({"set": "new17" if new else "orig31", "dataset": d, "target": Y, "env": e,
                                 "n": len(ok), "item": m, "kind": "method", "d_spearman": sa - sb,
                                 "d_sel_diff_f10": ga - gb, "spearman": sa})
            print(d, Y, "done", flush=True)
rules_old = pd.read_parquet(RESULTS / "b3b" / "rules_per_env.parquet")
rules_old = rules_old[(rules_old["library"] == "main") & (rules_old["scope"] == "all")].assign(set="orig31")
rules_new = pd.read_parquet(RESULTS / "b3d" / "rules_per_env.parquet")
rules_new = rules_new[rules_new["scope"] == "all"].assign(set="new17")
for R in (rules_old, rules_new):
    base = R[R["rule"] == "R_CELL"].set_index(["dataset", "target", "env"])
    for r in ("R_STK_FW", "R_FW", "R_CV", "R_EQ", "R_STK_CV", "Oracle"):
        a = R[R["rule"] == r].set_index(["dataset", "target", "env"])
        j = a.join(base, rsuffix="_b", how="inner")
        for (d, Y, e), x in j.iterrows():
            rows.append({"set": x["set"], "dataset": d, "target": Y, "env": e, "n": int(x["n"]), "item": r, "kind": "rule",
                         "d_spearman": x["spearman"] - x["spearman_b"],
                         "d_sel_diff_f10": x["sel_diff_f10"] - x["sel_diff_f10_b"], "spearman": x["spearman"]})
E = pd.DataFrame(rows)
E.to_parquet(OUT / "per_env.parquet", index=False)

ye_rows = []
for (s, d, Y, item, kind), g in E.groupby(["set", "dataset", "target", "item", "kind"]):
    for metric in ("d_spearman", "d_sel_diff_f10"):
        v = g[metric].dropna().to_numpy()
        if len(v) == 0:
            continue
        bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        ye_rows.append({"set": s, "dataset": d, "target": Y, "item": item, "kind": kind, "metric": metric[2:],
                        "d": float(v.mean()), "se": float(bs.std(ddof=1)), "n_env": len(v)})
YE = pd.DataFrame(ye_rows)
YE.to_csv(OUT / "year_effects.csv", index=False)
pooled = []
for (item, kind, metric), g in YE.groupby(["item", "kind", "metric"]):
    for scope, sub in (("orig31", g[g["set"] == "orig31"]), ("new17", g[g["set"] == "new17"]), ("all48", g)):
        if sub.empty:
            continue
        r = dersimonian_laird(floor_se(sub))
        pooled.append({"item": item, "kind": kind, "metric": metric, "scope": scope, "k": r["k"], "est": r["est"],
                       "lo": r["lo"], "hi": r["hi"], "tau2": r["tau2"],
                       "datasets": ",".join(sorted(sub["dataset"].unique()))})
Pd = pd.DataFrame(pooled).sort_values(["kind", "metric", "scope", "est"])
Pd.to_csv(OUT / "pooled.csv", index=False)
json.dump({"n_env_rows": int(len(E)), "years_orig": int(sum(map(len, T_OLD.values()))),
           "years_new": int(sum(map(len, T_NEW.values())))}, open(OUT / "meta.json", "w"), indent=1)
print(Pd[Pd["metric"] == "spearman"].round(4).to_string(index=False))
print("SUMMARY48_DONE")
