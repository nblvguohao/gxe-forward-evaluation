"""Wave 2d analysis (docs/prereg_sequel_wave2d_2026-09-28.md §3-4). Refuses to run until every prediction file
exists. dl_* predictions are averaged over seeds per cell; comparators: cell_reml (wave 2c panels) and the
two-stage mlp (wave 2b panels). Year effects with environment-bootstrap SE, random-effects pooling with the
zero-SE floor, Holm over H5/H6."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from dartgxe.forward.data import LOADERS
from dartgxe.forward.pool import dersimonian_laird, floor_se, stratified_unweighted
from dartgxe.paths import RESULTS

B = 2000
IN = RESULTS / "b3c"
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
qual = w1c[(w1c["rule"] == "main") & w1c["qualifies"]]
TARGETS = {d: sorted(int(y) for y in g["year"]) for d, g in qual.groupby("dataset")}
HAS_EC = {"G2F": True, "URSN": True, "NUST": False}
need = [(d, Y, m, s) for d, ys in TARGETS.items() for Y in ys for m in (["dl_g", "dl_ge"] if HAS_EC[d] else ["dl_g"]) for s in (0, 1, 2)]
missing = [k for k in need if not (IN / "pred" / f"{k[0]}_{k[1]}_{k[2]}_seed{k[3]}.parquet").exists()]
if missing:
    sys.exit(f"panels incomplete ({len(missing)}), e.g. {missing[:3]}; refusing to score")


def spear(a, b):
    a, b = rankdata(a), rankdata(b)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def seldiff(y, s):
    k = max(1, int(round(0.1 * len(y))))
    sd = y.std()
    top = np.argsort(-s, kind="mergesort")[:k]
    return float((y[top].mean() - y.mean()) / sd) if sd > 0 else np.nan


rows = []
for d, ys in TARGETS.items():
    seen_all = LOADERS[d]().cells
    for Y in ys:
        parts = []
        for m in (["dl_g", "dl_ge"] if HAS_EC[d] else ["dl_g"]):
            p = pd.concat([pd.read_parquet(IN / "pred" / f"{d}_{Y}_{m}_seed{s}.parquet") for s in (0, 1, 2)])
            parts.append(p.groupby(["env", "genotype"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean")).assign(method=m))
        cr = pd.read_parquet(RESULTS / "b3b" / "forward" / f"{d}_{Y}.parquet")
        parts.append(cr[cr["method"] == "cell_reml"][["env", "genotype", "y", "pred", "method"]])
        tw = pd.read_parquet(RESULTS / "b3" / "forward" / f"{d}_{Y}.parquet")
        parts.append(tw[tw["method"] == "mlp"][["env", "genotype", "y", "pred", "method"]])
        allp = pd.concat(parts, ignore_index=True)
        w = allp.pivot_table(index=["env", "genotype"], columns="method", values="pred").dropna()
        yv = allp.drop_duplicates(["env", "genotype"]).set_index(["env", "genotype"])["y"].reindex(w.index)
        seen = set(seen_all[seen_all["year"] < Y]["genotype"])
        for e, P in w.groupby(level=0):
            y = yv.loc[P.index].to_numpy(float)
            g = P.index.get_level_values(1).to_numpy()
            for scope, keep in (("all", np.ones(len(y), bool)), ("new_only", ~np.isin(g, list(seen)))):
                if keep.sum() < (10 if scope == "new_only" else 1):
                    continue
                for m in P.columns:
                    s = P[m].to_numpy(float)[keep]
                    rows.append({"dataset": d, "target": Y, "scope": scope, "env": e, "n": int(keep.sum()), "method": m,
                                 "spearman": spear(s, y[keep]), "sel_diff_f10": seldiff(y[keep], s)})
        print(d, Y, "done", flush=True)
R = pd.DataFrame(rows)
R.to_parquet(IN / "per_env.parquet", index=False)
rng = np.random.default_rng(20260930)


def pooled(D, a, b, metric, name, sub):
    out = []
    for (d, Y), g in D.groupby(["dataset", "target"]):
        A = g[g["method"] == a].set_index("env")[metric]
        Bv = g[g["method"] == b].set_index("env")[metric]
        diff = (A - Bv).dropna().to_numpy()
        if len(diff) == 0:
            continue
        bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
        out.append({"dataset": d, "target": Y, "d": float(diff.mean()), "se": float(bs.std(ddof=1))})
    ye = floor_se(pd.DataFrame(out))
    res = dersimonian_laird(ye)
    res.update({"contrast": name, "a": a, "b": b, "metric": metric, "subset": sub, "unweighted": stratified_unweighted(ye, rng),
                "p": float(2 * norm.sf(abs(res["est"] / res["se"])))})
    return res, ye.assign(contrast=name, metric=metric, subset=sub)


allg, newg = R[R["scope"] == "all"], R[R["scope"] == "new_only"]
gu = allg[allg["dataset"].isin(["G2F", "URSN"])]
SPECS = [("H5", "dl_g", "cell_reml", allg, "all"), ("H6", "dl_g", "cell_reml", newg, "new_only"),
         ("dl_ge_vs_cell_G2F+URSN", "dl_ge", "cell_reml", gu, "G2F+URSN"),
         ("dl_g_vs_twostage_mlp", "dl_g", "mlp", allg, "all")]
P, Yl = [], []
for name, a, b, D, sub in SPECS:
    for metric in ("spearman", "sel_diff_f10"):
        res, ye = pooled(D, a, b, metric, name, sub)
        P.append(res)
        Yl.append(ye)
    if name in ("H5", "H6"):
        for dsn in TARGETS:
            res, _ = pooled(D[D["dataset"] == dsn], a, b, "spearman", name, dsn)
            P.append(res)
P = pd.DataFrame(P)
P.to_json(IN / "pooled.json", orient="records", indent=1)
pd.concat(Yl).to_csv(IN / "year_effects.csv", index=False)
N = int(allg[allg["method"] == "cell_reml"]["n"].sum())
N_new = int(newg[newg["method"] == "cell_reml"]["n"].sum())
conf = P[P["contrast"].isin(["H5", "H6"]) & (P["metric"] == "spearman") & P["subset"].isin(["all", "new_only"])].sort_values("p")
verd = {"N_all": N, "N_new": N_new, "holm": []}
for step, (_, r) in enumerate(conf.iterrows()):
    level = 0.975 if step == 0 else 0.95
    z = norm.ppf(1 - (1 - level) / 2)
    lo, hi = r["est"] - z * r["se"], r["est"] + z * r["se"]
    thr = 3 / np.sqrt(N if r["contrast"] == "H5" else N_new)
    excl = lo > 0 or hi < 0
    v = "better" if (excl and r["est"] >= thr) else ("worse" if (excl and r["est"] <= -thr) else ("detectable, below resolution" if excl else "tied"))
    verd["holm"].append({"contrast": r["contrast"], "est": r["est"], "level": level, "lo": lo, "hi": hi, "p": r["p"], "resolution": thr, "verdict": v})
    if not excl:
        for _, r2 in list(conf.iterrows())[step + 1:]:
            verd["holm"].append({"contrast": r2["contrast"], "est": r2["est"], "p": r2["p"], "verdict": "not tested (Holm stopped)"})
        break
verd["GO"] = any(h.get("verdict") == "better" for h in verd["holm"])
metas = [json.load(open(f)) for f in sorted((IN / "meta").glob("*_seed*.json"))]
verd["e_star_median"] = float(np.median([m["e_star"] for m in metas]))
verd["config_counts"] = pd.Series([m["config_index"] for m in metas]).value_counts().to_dict()
json.dump(verd, open(IN / "verdict.json", "w"), indent=1, default=float)
print(P[["contrast", "subset", "metric", "est", "lo", "hi", "k"]].round(4).to_string(index=False))
print(json.dumps(verd, indent=1, default=float))
print("B3C_DONE")
