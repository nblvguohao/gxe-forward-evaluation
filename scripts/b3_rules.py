"""Wave 2b rules and pooled comparisons (docs/prereg_sequel_wave2b_2026-09-28.md §3-6). Refuses to run until
every forward and CV panel exists.

Per target year Y of each dataset: method scores on the CV history and on the forward history (mean within-env
Pearson), the rules R_REML, R_CV, R_FW, R_EQ, R_STK_CV, R_STK_FW, R_FW_lambda and the oracle, evaluated once on
Y's scorable environments (within-env Spearman; 10 % selection differential). Year effects d_Y with an
environment bootstrap SE; DerSimonian-Laird random-effects pooling; Holm over H1 and H2."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import kendalltau, norm, rankdata

from dartgxe.forward.data import LOADERS
from dartgxe.forward.panel import METHODS
from dartgxe.paths import RESULTS

B = 2000
LAM = 0.05  # atlas stack_select.py
LAMBDA_FAMILY = ["reml_x0.1", "reml", "reml_x10", "reml_x100"]
IN = RESULTS / "b3"
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
qual = w1c[(w1c["rule"] == "main") & w1c["qualifies"]]
TARGETS = {d: sorted(int(y) for y in g["year"]) for d, g in qual.groupby("dataset")}
summ = {d: json.load(open(IN / "meta" / f"summary_{d}.json")) for d in TARGETS if (IN / "meta" / f"summary_{d}.json").exists()}
missing = [d for d in TARGETS if d not in summ]
missing += [f"forward {d} {y}" for d, s in summ.items() for y in s["predictable_years"] if not (IN / "forward" / f"{d}_{y}.parquet").exists()]
missing += [f"cv {d} {t}" for d, ts in TARGETS.items() for t in ts if not (IN / "cv" / f"{d}_{t}.parquet").exists()]
if missing:
    sys.exit(f"panels incomplete ({len(missing)}), e.g. {missing[:3]}; refusing to score")


def envs_of(df):
    """dict env -> {y, yz, P (n x methods), Z}; methods in METHODS order."""
    out = {}
    w = df.pivot_table(index=["env", "genotype"], columns="method", values="pred")[METHODS]
    y = df.drop_duplicates(["env", "genotype"]).set_index(["env", "genotype"])["y"].reindex(w.index)
    for e, P in w.groupby(level=0):
        yy = y.loc[P.index].to_numpy(float)
        Pm = P.to_numpy(float)
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


def score(envs):
    """Mean within-env Pearson per method (the atlas leader criterion)."""
    return np.nanmean([[pear(v["P"][:, j], v["y"]) for j in range(len(METHODS))] for v in envs.values()], axis=0)


def stack(envs):
    m = len(METHODS)
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


def first_max(x, idx=None):
    idx = list(range(len(x))) if idx is None else idx
    vals = np.array([x[i] for i in idx])
    return idx[int(np.nanargmax(vals))]


rows, choice, e5 = [], [], []
for d, targets in TARGETS.items():
    pred_years = summ[d]["predictable_years"]
    fw_all = {y: pd.read_parquet(IN / "forward" / f"{d}_{y}.parquet") for y in pred_years}
    cells_all = LOADERS[d]().cells
    qual_years = set(TARGETS[d])
    for Y in targets:
        tgt = envs_of(fw_all[Y])
        cvh = envs_of(pd.read_parquet(IN / "cv" / f"{d}_{Y}.parquet"))
        hist_years = [y for y in pred_years if y < Y]
        seen = set(cells_all[cells_all["year"] < Y]["genotype"])  # any record in any earlier year (W1c definition)
        for variant, hy in (("main", hist_years), ("hist_qualifying_only", [y for y in hist_years if y in qual_years])):
            if not hy:
                continue
            fwh = envs_of(pd.concat([fw_all[y] for y in hy], ignore_index=True))
            s_cv, s_fw = score(cvh), score(fwh)
            j_cv, j_fw = first_max(s_cv), first_max(s_fw)
            j_lam = first_max(s_fw, [METHODS.index(m) for m in LAMBDA_FAMILY])
            w_cv, w_fw = stack(cvh), stack(fwh)
            rules = {"R_REML": ("single", METHODS.index("reml")), "R_CV": ("single", j_cv), "R_FW": ("single", j_fw),
                     "R_FW_lambda": ("single", j_lam), "R_EQ": ("w", np.ones(len(METHODS)) / len(METHODS)),
                     "R_STK_CV": ("w", w_cv), "R_STK_FW": ("w", w_fw)}
            per_method = np.array([[evaluate(v, v["P"][:, j])[0] for j in range(len(METHODS))] for v in tgt.values()])
            rules["Oracle"] = ("single", int(np.nanargmax(np.nanmean(per_method, 0))))
            choice.append({"dataset": d, "target": Y, "variant": variant, "cv_pick": METHODS[j_cv], "fw_pick": METHODS[j_fw],
                           "fw_lambda_pick": METHODS[j_lam], "oracle": METHODS[rules["Oracle"][1]],
                           **{f"w_stk_fw_{m}": float(x) for m, x in zip(METHODS, w_fw)},
                           **{f"w_stk_cv_{m}": float(x) for m, x in zip(METHODS, w_cv)}})
            for scope in (("all",) if variant != "main" else ("all", "new_only")):
                for e, v in tgt.items():
                    if scope == "new_only":
                        keep = ~np.isin(v["genotype"], list(seen))
                        if keep.sum() < 10:
                            continue
                        y = v["y"][keep]
                        ysd = y.std()
                        v = {**v, "y": y, "yz": (y - y.mean()) / ysd if ysd > 0 else np.zeros_like(y),
                             "P": v["P"][keep], "Z": v["Z"][keep], "k": max(1, int(round(0.1 * keep.sum())))}
                    for r, (kind, arg) in rules.items():
                        s = v["P"][:, arg] if kind == "single" else np.nan_to_num(v["Z"]) @ arg
                        sp, gn = evaluate(v, s)
                        rows.append({"dataset": d, "target": Y, "variant": variant, "scope": scope, "env": e,
                                     "n": len(v["y"]), "rule": r, "spearman": sp, "sel_diff_f10": gn})
            if variant == "main":
                tgt_rank = np.nanmean(per_method, 0)
                n_cells = sum(len(v["y"]) for v in tgt.values())
                thr = 3 / np.sqrt(n_cells)
                diffs = [abs(tgt_rank[i] - tgt_rank[j]) for i in range(len(METHODS)) for j in range(i + 1, len(METHODS))]
                e5.append({"dataset": d, "target": Y, "kendall_tau_cv_vs_target": kendalltau(s_cv, tgt_rank)[0],
                           "kendall_tau_fw_vs_target": kendalltau(s_fw, tgt_rank)[0], "n_cells": n_cells, "resolution": thr,
                           "share_pairs_below_resolution": float(np.mean(np.array(diffs) < thr))})
        print(d, Y, "done", flush=True)

R = pd.DataFrame(rows)
R.to_parquet(IN / "rules_per_env.parquet", index=False)
pd.DataFrame(choice).to_csv(IN / "choices.csv", index=False)
E5 = pd.DataFrame(e5)
E5.to_csv(IN / "e5.csv", index=False)


def year_effects(R, a, b, metric, rng):
    out = []
    for (d, Y), g in R.groupby(["dataset", "target"]):
        w = g.pivot_table(index="env", columns="rule", values=metric)
        diff = (w[a] - w[b]).dropna().to_numpy()
        if len(diff) == 0:
            continue
        bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
        out.append({"dataset": d, "target": Y, "d": float(diff.mean()), "se": float(bs.std(ddof=1)), "n_env": len(diff)})
    return pd.DataFrame(out)


def floor_se(ye):
    """Amendment 2026-09-28: a year where both rules pick the same method (or the same selected lines) has
    SE_Y = 0; floor it at the median positive SE_Y of the same dataset (all datasets if none is positive)."""
    ye = ye.copy()
    pos_all = ye.loc[ye["se"] > 0, "se"]
    ye["se_zero"] = ye["se"] <= 0
    for d, g in ye.groupby("dataset"):
        pos = g.loc[g["se"] > 0, "se"]
        fl = pos.median() if len(pos) else (pos_all.median() if len(pos_all) else np.nan)
        ye.loc[g.index, "se"] = np.maximum(g["se"], fl)
    return ye


def dersimonian_laird(ye, level):
    n_zero = int(ye["se_zero"].sum()) if "se_zero" in ye else 0
    d, v = ye["d"].to_numpy(), ye["se"].to_numpy() ** 2
    w = 1 / v
    mu = (w * d).sum() / w.sum()
    Q = (w * (d - mu) ** 2).sum()
    tau2 = max(0.0, (Q - (len(d) - 1)) / (w.sum() - (w * w).sum() / w.sum())) if len(d) > 1 else 0.0
    ws = 1 / (v + tau2)
    est, se = (ws * d).sum() / ws.sum(), 1 / np.sqrt(ws.sum())
    z = norm.ppf(1 - (1 - level) / 2)
    return {"est": float(est), "se": float(se), "lo": float(est - z * se), "hi": float(est + z * se), "tau2": float(tau2),
            "p": float(2 * norm.sf(abs(est / se))), "k": int(len(d)), "n_years_se_zero_floored": n_zero,
            "weight_share": (pd.Series(ws, index=ye["dataset"].to_numpy()).groupby(level=0).sum() / ws.sum()).round(3).to_dict()}


def stratified_unweighted(ye, rng):
    by = {d: g["d"].to_numpy() for d, g in ye.groupby("dataset")}
    est = float(np.mean(ye["d"]))
    bs = [np.mean(np.concatenate([x[rng.integers(0, len(x), len(x))] for x in by.values()])) for _ in range(B)]
    return {"est": est, "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}


main = R[(R["variant"] == "main") & (R["scope"] == "all")]
N = int(main[main["rule"] == "R_REML"]["n"].sum())
thr = 3 / np.sqrt(N)
CONF = [("H1", "R_FW", "R_CV"), ("H2", "R_STK_FW", "R_FW")]
DESC = [("REML_vs_FW", "R_FW", "R_REML"), ("REML_vs_CV", "R_CV", "R_REML"), ("REML_vs_STK_FW", "R_STK_FW", "R_REML"),
        ("REML_vs_EQ", "R_EQ", "R_REML"), ("lambda_FW_vs_REML", "R_FW_lambda", "R_REML"), ("STK_CV_vs_EQ", "R_STK_CV", "R_EQ"),
        ("FW_vs_Oracle", "R_FW", "Oracle"), ("STK_FW_vs_Oracle", "R_STK_FW", "Oracle")]
rng = np.random.default_rng(20260928)
pooled, yearly = [], []
subsets = {"main": main, "G2F+NUST": main[main["dataset"].isin(["G2F", "NUST"])],
           "new_only": R[(R["variant"] == "main") & (R["scope"] == "new_only")],
           "hist_qualifying_only": R[(R["variant"] == "hist_qualifying_only") & (R["scope"] == "all")]}
for name, a, b in CONF + DESC:
    for metric in ("spearman", "sel_diff_f10"):
        for sub, D in subsets.items():
            if sub != "main" and not (name in ("H1", "H2") or metric == "spearman"):
                continue
            ye = year_effects(D, a, b, metric, rng)
            if ye.empty:
                continue
            ye = floor_se(ye)
            res = dersimonian_laird(ye, 0.95)
            res.update({"contrast": name, "a": a, "b": b, "metric": metric, "subset": sub,
                        "unweighted": stratified_unweighted(ye, rng)})
            pooled.append(res)
            if sub == "main":
                yearly.append(ye.assign(contrast=name, metric=metric))
P = pd.DataFrame(pooled)
P.to_json(IN / "pooled.json", orient="records", indent=1)
pd.concat(yearly).to_csv(IN / "year_effects.csv", index=False)

# Holm over H1, H2 (main subset, Spearman): smaller p first at 97.5 %, the other at 95 %
conf = P[(P["contrast"].isin(["H1", "H2"])) & (P["metric"] == "spearman") & (P["subset"] == "main")].sort_values("p")
verd = {"N_cells": N, "resolution": thr, "holm": []}
for step, (_, r) in enumerate(conf.iterrows()):
    level = 0.975 if step == 0 else 0.95
    z = norm.ppf(1 - (1 - level) / 2)
    lo, hi = r["est"] - z * r["se"], r["est"] + z * r["se"]
    excl = lo > 0 or hi < 0
    v = "better" if excl and abs(r["est"]) >= thr else ("detectable, below resolution" if excl else "tied")
    verd["holm"].append({"contrast": r["contrast"], "a": r["a"], "b": r["b"], "est": r["est"], "level": level,
                         "lo": lo, "hi": hi, "p": r["p"], "verdict": v})
    if not excl:  # Holm stops: later hypotheses are not rejected
        for _, r2 in list(conf.iterrows())[step + 1:]:
            verd["holm"].append({"contrast": r2["contrast"], "est": r2["est"], "verdict": "not tested (Holm stopped)",
                                 "p": r2["p"]})
        break
verd["prediction_H1_FW_ge_CV"] = bool(conf.set_index("contrast").loc["H1", "est"] >= 0)
verd["prediction_H2_tied"] = [h for h in verd["holm"] if h["contrast"] == "H2"][0]["verdict"] in ("tied", "not tested (Holm stopped)")
ch = pd.DataFrame(choice)
chm = ch[ch["variant"] == "main"]
verd["years_cv_pick_eq_fw_pick"] = {d: f"{int((g.cv_pick == g.fw_pick).sum())}/{len(g)}" for d, g in chm.groupby("dataset")}
verd["years_fw_pick_eq_oracle"] = {d: f"{int((g.fw_pick == g.oracle).sum())}/{len(g)}" for d, g in chm.groupby("dataset")}
verd["years_cv_pick_eq_oracle"] = {d: f"{int((g.cv_pick == g.oracle).sum())}/{len(g)}" for d, g in chm.groupby("dataset")}
verd["e5_mean_tau_cv"] = float(E5["kendall_tau_cv_vs_target"].mean())
verd["e5_mean_tau_fw"] = float(E5["kendall_tau_fw_vs_target"].mean())
verd["e5_mean_share_pairs_below_resolution"] = float(E5["share_pairs_below_resolution"].mean())
json.dump(verd, open(IN / "verdict.json", "w"), indent=1, default=float)
print(P[P["subset"] == "main"][["contrast", "metric", "est", "lo", "hi", "k", "tau2"]].round(4).to_string(index=False))
print(json.dumps(verd, indent=1, default=float))
