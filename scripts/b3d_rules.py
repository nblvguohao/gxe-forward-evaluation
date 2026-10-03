"""Wave 2e rules and pooled comparisons (docs/prereg_sequel_wave2e_2026-09-29.md §3-5). Refuses to run until every
panel exists. Library per dataset: the wave 2b two-stage methods + cell_reml. Rules and scoring are the wave 2c
ones (scripts/b3b_rules.py) unchanged: R_STK_FW = NNLS stacking (LAM = 0.05) of within-environment standardised
predictions, weights fitted on the forward history (all predictable years before the target year).

Confirmatory (one test, 95%): H7 = R_STK_FW - R_CELL, selection differential of the top 10 % (sel_diff_f10, in
within-environment SD units), all scored genotypes, random-effects pooling over the target years of the three
datasets with the zero-SE floor. GO (plan v2 §9.3): CI excludes 0 and estimate >= 3/sqrt(N)."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import norm, rankdata

from dartgxe.forward.data import LOADERS, NEW
from dartgxe.forward.panel import METHODS as B3M
from dartgxe.forward.pool import dersimonian_laird, floor_se, stratified_unweighted
from dartgxe.paths import RESULTS

B = 2000
LAM = 0.05
IN = RESULTS / "b3d"
cnt = pd.read_csv(IN / "count" / "per_year.csv")
qual = cnt[(cnt["rule"] == "main") & cnt["qualifies"]]
TARGETS = {d: sorted(int(y) for y in g["year"]) for d, g in qual.groupby("dataset")}
METHODS = list(B3M) + ["cell_reml"]
summ, missing = {}, []
for d in TARGETS:
    f = IN / "meta" / f"summary_{d}.json"
    if not f.exists():
        missing.append(f"summary {d}")
        continue
    summ[d] = json.load(open(f))
    missing += [f"forward {d} {y}" for y in summ[d]["predictable_years"] if not (IN / "forward" / f"{d}_{y}.parquet").exists()]
    missing += [f"cv {d} {t}" for t in TARGETS[d] if not (IN / "cv" / f"{d}_{t}.parquet").exists()]
if missing:
    sys.exit(f"panels incomplete ({len(missing)}), e.g. {missing[:3]}; refusing to score")


def load(kind, d, y):
    return pd.read_parquet(IN / kind / f"{d}_{y}.parquet", columns=["env", "genotype", "y", "method", "pred"])


def envs_of(df, methods):
    w = df[df["method"].isin(methods)].pivot_table(index=["env", "genotype"], columns="method", values="pred")[methods]
    w = w.dropna()
    y = df.drop_duplicates(["env", "genotype"]).set_index(["env", "genotype"])["y"].reindex(w.index)
    out = {}
    for e, P in w.groupby(level=0):
        yy, Pm = y.loc[P.index].to_numpy(float), P.to_numpy(float)
        sd = Pm.std(0)
        Z = np.where(sd > 0, (Pm - Pm.mean(0)) / np.where(sd > 0, sd, 1), 0.0)
        ysd = yy.std()
        out[e] = {"y": yy, "yz": (yy - yy.mean()) / ysd if ysd > 0 else np.zeros_like(yy), "P": Pm, "Z": Z,
                  "k": max(1, int(round(0.1 * len(yy)))), "genotype": P.index.get_level_values(1).to_numpy()}
    return out


def pear(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def score(envs, m):
    return np.nanmean([[pear(v["P"][:, j], v["y"]) for j in range(m)] for v in envs.values()], axis=0)


def stack(envs, m):
    c, S = np.zeros(m), np.zeros((m, m))
    for v in envs.values():
        n = len(v["y"])
        c += v["Z"].T @ v["yz"] / n
        S += v["Z"].T @ v["Z"] / n
    c, S = c / len(envs), S / len(envs)
    L = np.linalg.cholesky(S + LAM * np.eye(m))
    w, _ = nnls(L.T, np.linalg.solve(L, c))
    return w if w.sum() > 0 else np.ones(m) / m


def evaluate(v, s):
    sp = pear(rankdata(s), rankdata(v["y"]))
    top = np.argsort(-np.where(np.isfinite(s), s, -np.inf), kind="mergesort")[:v["k"]]
    return sp, float(v["yz"][top].mean())


def subset(v, keep):
    y = v["y"][keep]
    ysd = y.std()
    return {**v, "y": y, "yz": (y - y.mean()) / ysd if ysd > 0 else np.zeros_like(y), "P": v["P"][keep], "Z": v["Z"][keep],
            "k": max(1, int(round(0.1 * keep.sum()))), "genotype": v["genotype"][keep]}


rows, weights = [], []
m = len(METHODS)
for d, targets in TARGETS.items():
    cells_all = LOADERS[d]().cells
    pred_years = summ[d]["predictable_years"]
    fw_all = {y: load("forward", d, y) for y in pred_years}
    for Y in targets:
        tgt = envs_of(fw_all[Y], METHODS)
        cvh = envs_of(load("cv", d, Y), METHODS)
        fwh = envs_of(pd.concat([fw_all[y] for y in pred_years if y < Y], ignore_index=True), METHODS)
        seen = set(cells_all[cells_all["year"] < Y]["genotype"])
        s_cv, s_fw = score(cvh, m), score(fwh, m)
        w_cv, w_fw = stack(cvh, m), stack(fwh, m)
        per_method = np.array([[evaluate(v, v["P"][:, j])[0] for j in range(m)] for v in tgt.values()])
        rules = {"R_CV": ("single", int(np.nanargmax(s_cv))), "R_FW": ("single", int(np.nanargmax(s_fw))),
                 "R_EQ": ("w", np.ones(m) / m), "R_STK_CV": ("w", w_cv), "R_STK_FW": ("w", w_fw),
                 "Oracle": ("single", int(np.nanargmax(np.nanmean(per_method, 0)))),
                 "R_CELL": ("single", METHODS.index("cell_reml")), "M:reml": ("single", METHODS.index("reml"))}
        weights.append({"dataset": d, "target": Y, "cv_pick": METHODS[rules["R_CV"][1]],
                        "fw_pick": METHODS[rules["R_FW"][1]], "oracle": METHODS[rules["Oracle"][1]],
                        "n_fw_history_envs": len(fwh), **{f"w_fw_{x}": float(v) for x, v in zip(METHODS, w_fw)}})
        for scope in ("all", "new_only"):
            for e, v in tgt.items():
                if scope == "new_only":
                    keep = ~np.isin(v["genotype"], list(seen))
                    if keep.sum() < 10:
                        continue
                    v = subset(v, keep)
                for r, (kind, arg) in rules.items():
                    s = v["P"][:, arg] if kind == "single" else np.nan_to_num(v["Z"]) @ arg
                    sp, gn = evaluate(v, s)
                    rows.append({"dataset": d, "target": Y, "scope": scope, "env": e, "n": len(v["y"]), "rule": r,
                                 "spearman": sp, "sel_diff_f10": gn})
        print(d, Y, "done", flush=True)

R = pd.DataFrame(rows)
R.to_parquet(IN / "rules_per_env.parquet", index=False)
Wt = pd.DataFrame(weights)
Wt.to_csv(IN / "choices_weights.csv", index=False)
rng = np.random.default_rng(20260929)


def year_effects(D, a, b, metric):
    out = []
    for (d, Y), g in D.groupby(["dataset", "target"]):
        A = g[g["rule"] == a].set_index("env")[metric]
        Bv = g[g["rule"] == b].set_index("env")[metric]
        diff = (A - Bv).dropna().to_numpy()
        if len(diff) == 0:
            continue
        bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
        out.append({"dataset": d, "target": Y, "d": float(diff.mean()), "se": float(bs.std(ddof=1)), "n_env": len(diff)})
    return pd.DataFrame(out)


def pooled(D, a, b, metric, name, sub):
    ye = year_effects(D, a, b, metric)
    if ye.empty:
        return None, ye
    ye = floor_se(ye)
    res = dersimonian_laird(ye)
    res.update({"contrast": name, "a": a, "b": b, "metric": metric, "subset": sub,
                "unweighted": stratified_unweighted(ye, rng), "p": float(2 * norm.sf(abs(res["est"] / res["se"])))})
    return res, ye.assign(contrast=name, metric=metric, subset=sub)


allg, newg = R[R["scope"] == "all"], R[R["scope"] == "new_only"]
SPECS = [("H7", "R_STK_FW", "R_CELL", allg, "all"), ("H7_new", "R_STK_FW", "R_CELL", newg, "new_only"),
         ("FW_vs_CELL", "R_FW", "R_CELL", allg, "all"), ("CV_vs_CELL", "R_CV", "R_CELL", allg, "all"),
         ("EQ_vs_CELL", "R_EQ", "R_CELL", allg, "all"), ("STK_CV_vs_CELL", "R_STK_CV", "R_CELL", allg, "all"),
         ("Oracle_vs_CELL", "Oracle", "R_CELL", allg, "all"), ("cell_vs_two_stage_reml", "R_CELL", "M:reml", allg, "all")]
P, Y = [], []
for name, a, b, D, sub in SPECS:
    for metric in ("sel_diff_f10", "spearman"):
        res, ye = pooled(D, a, b, metric, name, sub)
        if res:
            P.append(res)
            Y.append(ye)
    if name in ("H7", "H7_new"):
        for dsn in TARGETS:
            for metric in ("sel_diff_f10", "spearman"):
                res, _ = pooled(D[D["dataset"] == dsn], a, b, metric, name, dsn)
                if res:
                    P.append(res)
P = pd.DataFrame(P)
YE = pd.concat(Y)
P.to_json(IN / "pooled.json", orient="records", indent=1)
YE.to_csv(IN / "year_effects.csv", index=False)

# descriptive: the 31 wave 2c years (already seen) together with the new years, same contrast and metric
old = pd.read_csv(RESULTS / "b3b" / "year_effects.csv")
old = old[(old["contrast"] == "H3") & (old["metric"] == "sel_diff_f10")][["dataset", "target", "d", "se", "n_env"]]
new = YE[(YE["contrast"] == "H7") & (YE["metric"] == "sel_diff_f10")][["dataset", "target", "d", "se", "n_env"]]
comb = dersimonian_laird(floor_se(pd.concat([old, new], ignore_index=True)))

N = int(allg[allg["rule"] == "R_CELL"]["n"].sum())
N_new = int(newg[newg["rule"] == "R_CELL"]["n"].sum())
h = P[(P["contrast"] == "H7") & (P["metric"] == "sel_diff_f10") & (P["subset"] == "all")].iloc[0]
z = norm.ppf(0.975)
lo, hi = h["est"] - z * h["se"], h["est"] + z * h["se"]
thr = 3 / np.sqrt(N)
excl = lo > 0 or hi < 0
verdict = "better" if (lo > 0 and h["est"] >= thr) else ("worse" if hi < 0 else ("detectable, below threshold" if excl else "tied"))
verd = {"N_cells_all": N, "N_cells_new": N_new, "K_years": int(h["k"]),
        "H7": {"est": h["est"], "lo": lo, "hi": hi, "p": h["p"], "threshold": thr, "verdict": verdict},
        "GO": bool(lo > 0 and h["est"] >= thr),
        "combined_with_wave2c_31_years_descriptive": comb,
        "fw_pick_counts": Wt["fw_pick"].value_counts().to_dict(),
        "stack_weight_share_cell_reml": float((Wt["w_fw_cell_reml"] / Wt[[f"w_fw_{x}" for x in METHODS]].sum(1)).mean())}
json.dump(verd, open(IN / "verdict.json", "w"), indent=1, default=float)
print(P[["contrast", "subset", "metric", "est", "lo", "hi", "k"]].round(4).to_string(index=False))
print(json.dumps(verd, indent=1, default=float))
print("B3D_RULES_DONE")
