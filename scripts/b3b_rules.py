"""Wave 2c rules and pooled comparisons (docs/prereg_sequel_wave2c_2026-09-28.md §3-5). Merges the wave 2b
two-stage panels with the wave 2c cell-level / G x E panels; refuses to run until every panel exists.

Libraries per dataset: main = 10 two-stage methods + cell_reml + (rn_ridge, gxe_gbm with historical-mean ECs where
ECs exist); real = main with the G x E learners using observed target-year ECs (upper bound); b3lib = the 10
two-stage methods. Rules as wave 2b with the default R_CELL = cell_reml. Confirmatory: H3 = R_STK_FW - R_CELL
(all genotypes), H4 = the same on new genotypes; Holm; random-effects pooling with the zero-SE floor."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import norm, rankdata

from dartgxe.forward.data import LOADERS
from dartgxe.forward.panel import METHODS as B3M
from dartgxe.forward.pool import dersimonian_laird, floor_se, stratified_unweighted
from dartgxe.paths import RESULTS

B = 2000
LAM = 0.05
IN2, IN3 = RESULTS / "b3", RESULTS / "b3b"
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
qual = w1c[(w1c["rule"] == "main") & w1c["qualifies"]]
TARGETS = {d: sorted(int(y) for y in g["year"]) for d, g in qual.groupby("dataset")}
summ = {}
missing = []
for d in TARGETS:
    f = IN3 / "meta" / f"summary_{d}.json"
    if not f.exists():
        missing.append(f"summary {d}")
        continue
    summ[d] = json.load(open(f))
    missing += [f"{k} {d} {y}" for k in ("forward",) for y in summ[d]["predictable_years"]
                if not ((IN2 / k / f"{d}_{y}.parquet").exists() and (IN3 / k / f"{d}_{y}.parquet").exists())]
    missing += [f"cv {d} {t}" for t in TARGETS[d] if not ((IN2 / "cv" / f"{d}_{t}.parquet").exists() and (IN3 / "cv" / f"{d}_{t}.parquet").exists())]
if missing:
    sys.exit(f"panels incomplete ({len(missing)}), e.g. {missing[:3]}; refusing to score")


def libraries(d):
    gxe = summ[d]["has_ec"]
    main = B3M + ["cell_reml"] + (["rn_ridge", "gxe_gbm"] if gxe else [])
    libs = {"main": main, "b3lib": list(B3M)}
    if gxe:
        libs["real"] = B3M + ["cell_reml", "rn_ridge_real", "gxe_gbm_real"]
    return libs


def load(kind, d, y, lib_name):
    a = pd.read_parquet(IN2 / kind / f"{d}_{y}.parquet", columns=["env", "genotype", "y", "method", "pred"])
    b = pd.read_parquet(IN3 / kind / f"{d}_{y}.parquet", columns=["env", "genotype", "y", "method", "pred"])
    if kind == "cv" and lib_name == "real":  # CV environments are observed: the observed-EC learners are the same fits
        b = pd.concat([b, b[b["method"].isin(["rn_ridge", "gxe_gbm"])].assign(method=lambda x: x["method"] + "_real")])
    return pd.concat([a, b], ignore_index=True)


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
for d, targets in TARGETS.items():
    cells_all = LOADERS[d]().cells
    pred_years = summ[d]["predictable_years"]
    for lib_name, methods in libraries(d).items():
        m = len(methods)
        fw_all = {y: load("forward", d, y, lib_name) for y in pred_years}
        for Y in targets:
            tgt = envs_of(fw_all[Y], methods)
            cvh = envs_of(load("cv", d, Y, lib_name), methods)
            fwh = envs_of(pd.concat([fw_all[y] for y in pred_years if y < Y], ignore_index=True), methods)
            seen = set(cells_all[cells_all["year"] < Y]["genotype"])
            s_cv, s_fw = score(cvh, m), score(fwh, m)
            w_cv, w_fw = stack(cvh, m), stack(fwh, m)
            per_method = np.array([[evaluate(v, v["P"][:, j])[0] for j in range(m)] for v in tgt.values()])
            rules = {"R_CV": ("single", int(np.nanargmax(s_cv))), "R_FW": ("single", int(np.nanargmax(s_fw))),
                     "R_EQ": ("w", np.ones(m) / m), "R_STK_CV": ("w", w_cv), "R_STK_FW": ("w", w_fw),
                     "Oracle": ("single", int(np.nanargmax(np.nanmean(per_method, 0))))}
            for j, name in enumerate(methods):
                if name in ("cell_reml", "reml", "rn_ridge", "gxe_gbm", "rn_ridge_real", "gxe_gbm_real"):
                    rules[f"M:{name}"] = ("single", j)
            if "cell_reml" in methods:
                rules["R_CELL"] = ("single", methods.index("cell_reml"))
            weights.append({"dataset": d, "target": Y, "library": lib_name, "cv_pick": methods[rules["R_CV"][1]],
                            "fw_pick": methods[rules["R_FW"][1]], "oracle": methods[rules["Oracle"][1]],
                            **{f"w_fw_{x}": float(v) for x, v in zip(methods, w_fw)}})
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
                        rows.append({"dataset": d, "target": Y, "library": lib_name, "scope": scope, "env": e,
                                     "n": len(v["y"]), "rule": r, "spearman": sp, "sel_diff_f10": gn})
            print(d, lib_name, Y, "done", flush=True)

R = pd.DataFrame(rows)
R.to_parquet(IN3 / "rules_per_env.parquet", index=False)
Wt = pd.DataFrame(weights)
Wt.to_csv(IN3 / "choices_weights.csv", index=False)
rng = np.random.default_rng(20260929)


def year_effects(D, a, b, metric):
    """a, b = (library, rule); d_Y = mean over environments of a - b."""
    out = []
    for (d, Y), g in D.groupby(["dataset", "target"]):
        A = g[(g["library"] == a[0]) & (g["rule"] == a[1])].set_index("env")[metric]
        Bv = g[(g["library"] == b[0]) & (g["rule"] == b[1])].set_index("env")[metric]
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
    res.update({"contrast": name, "a": "/".join(a), "b": "/".join(b), "metric": metric, "subset": sub,
                "unweighted": stratified_unweighted(ye, rng), "p": float(2 * norm.sf(abs(res["est"] / res["se"])))})
    return res, ye.assign(contrast=name, metric=metric, subset=sub)


allg, newg = R[R["scope"] == "all"], R[R["scope"] == "new_only"]
gu = ["G2F", "URSN"]
SPECS = [("H3", ("main", "R_STK_FW"), ("main", "R_CELL"), allg, "all"),
         ("H4", ("main", "R_STK_FW"), ("main", "R_CELL"), newg, "new_only"),
         ("FW_vs_CELL", ("main", "R_FW"), ("main", "R_CELL"), allg, "all"),
         ("CV_vs_CELL", ("main", "R_CV"), ("main", "R_CELL"), allg, "all"),
         ("EQ_vs_CELL", ("main", "R_EQ"), ("main", "R_CELL"), allg, "all"),
         ("STK_CV_vs_CELL", ("main", "R_STK_CV"), ("main", "R_CELL"), allg, "all"),
         ("Oracle_vs_CELL", ("main", "Oracle"), ("main", "R_CELL"), allg, "all"),
         ("new_learners_value", ("main", "R_STK_FW"), ("b3lib", "R_STK_FW"), allg, "all"),
         ("new_learners_value_G2F+URSN", ("main", "R_STK_FW"), ("b3lib", "R_STK_FW"), allg[allg["dataset"].isin(gu)], "G2F+URSN"),
         ("H3_realEC_G2F+URSN", ("real", "R_STK_FW"), ("real", "R_CELL"), allg[allg["dataset"].isin(gu)], "G2F+URSN"),
         ("H3_G2F+URSN", ("main", "R_STK_FW"), ("main", "R_CELL"), allg[allg["dataset"].isin(gu)], "G2F+URSN"),
         ("rn_ridge_vs_cell", ("main", "M:rn_ridge"), ("main", "M:cell_reml"), allg[allg["dataset"].isin(gu)], "G2F+URSN"),
         ("gxe_gbm_vs_cell", ("main", "M:gxe_gbm"), ("main", "M:cell_reml"), allg[allg["dataset"].isin(gu)], "G2F+URSN"),
         ("rn_ridge_real_vs_cell", ("real", "M:rn_ridge_real"), ("real", "M:cell_reml"), allg[allg["dataset"].isin(gu)], "G2F+URSN"),
         ("gxe_gbm_real_vs_cell", ("real", "M:gxe_gbm_real"), ("real", "M:cell_reml"), allg[allg["dataset"].isin(gu)], "G2F+URSN"),
         ("cell_vs_two_stage_reml", ("main", "M:cell_reml"), ("main", "M:reml"), allg, "all")]
P, Y = [], []
for name, a, b, D, sub in SPECS:
    for metric in ("spearman", "sel_diff_f10"):
        if metric == "sel_diff_f10" and name not in ("H3", "H4", "FW_vs_CELL", "EQ_vs_CELL"):
            continue
        res, ye = pooled(D, a, b, metric, name, sub)
        if res:
            P.append(res)
            Y.append(ye)
    if name in ("H3", "H4"):
        for dsn in TARGETS:
            res, _ = pooled(D[D["dataset"] == dsn], a, b, "spearman", name, dsn)
            if res:
                P.append(res)
P = pd.DataFrame(P)
P.to_json(IN3 / "pooled.json", orient="records", indent=1)
pd.concat(Y).to_csv(IN3 / "year_effects.csv", index=False)

N = int(allg[(allg["library"] == "main") & (allg["rule"] == "R_CELL")]["n"].sum())
N_new = int(newg[(newg["library"] == "main") & (newg["rule"] == "R_CELL")]["n"].sum())
conf = P[P["contrast"].isin(["H3", "H4"]) & (P["metric"] == "spearman") & P["subset"].isin(["all", "new_only"])].sort_values("p")
verd = {"N_cells_all": N, "N_cells_new": N_new, "holm": []}
for step, (_, r) in enumerate(conf.iterrows()):
    level = 0.975 if step == 0 else 0.95
    z = norm.ppf(1 - (1 - level) / 2)
    lo, hi = r["est"] - z * r["se"], r["est"] + z * r["se"]
    thr = 3 / np.sqrt(N if r["contrast"] == "H3" else N_new)
    excl = lo > 0 or hi < 0
    v = "better" if excl and abs(r["est"]) >= thr else ("detectable, below resolution" if excl else "tied")
    verd["holm"].append({"contrast": r["contrast"], "est": r["est"], "level": level, "lo": lo, "hi": hi, "p": r["p"],
                         "resolution": thr, "verdict": v})
    if not excl:
        for _, r2 in list(conf.iterrows())[step + 1:]:
            verd["holm"].append({"contrast": r2["contrast"], "est": r2["est"], "p": r2["p"], "verdict": "not tested (Holm stopped)"})
        break
hv = {h["contrast"]: h["verdict"] for h in verd["holm"]}
verd["prediction_H3_tied"] = hv.get("H3") in ("tied", "not tested (Holm stopped)")
verd["prediction_H4_tied"] = hv.get("H4") in ("tied", "not tested (Holm stopped)")
wm = Wt[Wt["library"] == "main"]
gcols = [c for c in wm.columns if c in ("w_fw_rn_ridge", "w_fw_gxe_gbm")]
tot = wm[[c for c in wm.columns if c.startswith("w_fw_")]].sum(1)
verd["stack_weight_share_gxe_learners_G2F_URSN"] = float((wm[gcols].sum(1) / tot)[wm["dataset"].isin(gu)].mean()) if gcols else None
verd["stack_weight_share_cell_reml"] = float((wm["w_fw_cell_reml"] / tot).mean())
verd["fw_pick_counts_main"] = wm["fw_pick"].value_counts().to_dict()
json.dump(verd, open(IN3 / "verdict.json", "w"), indent=1, default=float)
print(P[["contrast", "subset", "metric", "est", "lo", "hi", "k"]].round(4).to_string(index=False))
print(json.dumps(verd, indent=1, default=float))
print("B3B_RULES_DONE")
