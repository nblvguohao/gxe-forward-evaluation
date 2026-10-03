"""Descriptive diagnostics for docs/ideas_gblup_improvement_2026-09-29.md. Nothing here is confirmatory: every input
(results/summary48/per_env.parquet, results/benchmark/{cells,predictions}_*.parquet) has been seen before, and no
new model is fitted. Runs locally (numpy/pandas/pyarrow), about one minute.

Part A (method panels, 48 forward years, library = the pre-registered 'main' library: ten two-stage methods +
cell_reml (+ rn_ridge, gxe_gbm with historical-mean ECs on G2F/URSN)):
  A1 which method the post-hoc Oracle picks per year, and its class;
  A2 how much of the naive Oracle gain (+0.019) is the expected maximum of noise: (i) cross-fitted Oracle (pick the
     best method on a random half of the year's scored environments, score it on the other half; 200 splits, both
     directions), (ii) the naive Oracle under a null in which every method equals cell_reml in expectation (method
     columns centred within year, environments bootstrapped);
  A3 nonlinear vs linear two-stage methods (rf/gbm/knn/mlp minus two-stage reml, same stage-1 input) by dataset;
  A4 whether the year-level gain of the best alternative tracks year features (environments, lines per
     environment, new-line share, history size, cell_reml level).
Part B (phenotypes only, benchmark cells, all years):
  B1 same-year genotype-mean ceiling per scored environment: within-env Spearman between y and the genotype's mean
     z-score over the OTHER scored environments of the same year (not deployable; ceiling for any model of a
     genotype main effect), next to cell_reml;
  B2 pairwise environment correlations over shared genotypes by (same/different year) x (same/different location)
     and by year gap -> repeatable G x location, G x year, temporal decay;
  B3 spread of environment 'consensus' (B1 computed for every environment) -> room for environment weighting;
  B4 pedigree structure: G2F parents of target hybrids seen in training; GEM_IA families of target lines seen."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dartgxe.forward.pool import dersimonian_laird, floor_se, stratified_unweighted  # noqa: E402

RES = ROOT / "results"
OUT = RES / "ideas_diag"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260929)
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
TWO_STAGE = ["reml_x0.1", "reml", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp"]
MAIN = TWO_STAGE + ["cell_reml", "rn_ridge", "gxe_gbm"]
CLASS = {"reml_x0.1": "linear_2stage", "reml": "linear_2stage", "reml_x10": "linear_2stage",
         "reml_x100": "linear_2stage", "ridge_pc20": "linear_2stage", "rf": "nonlinear_2stage",
         "gbm": "nonlinear_2stage", "knn10": "nonlinear_2stage", "knn30": "nonlinear_2stage",
         "mlp": "nonlinear_2stage", "cell_reml": "cell_reml", "rn_ridge": "gxe_linear", "gxe_gbm": "gxe_nonlinear"}
N_SPLIT, B_SE, B_NULL = 200, 200, 1000


def pool(ye: pd.DataFrame) -> dict:
    r = dersimonian_laird(floor_se(ye))
    return {k: r[k] for k in ("est", "lo", "hi", "tau2", "k")}


# ------------------------------------------------------------------------------------------------ Part A
E = pd.read_parquet(RES / "summary48" / "per_env.parquet")
E = E[(E["kind"] == "method") & E["item"].isin(MAIN)]
cellref = E.groupby(["dataset", "target", "env"])[["spearman", "d_spearman"]].first()
cell_level = (cellref["spearman"] - cellref["d_spearman"]).rename("cell_spearman")


def year_matrix(g: pd.DataFrame, metric: str) -> pd.DataFrame:
    D = g.pivot_table(index="env", columns="item", values=metric)
    D["cell_reml"] = 0.0
    return D.dropna(axis=1, how="any").dropna(axis=0, how="any")


def cross_fit(D: np.ndarray, rng, n_split=N_SPLIT) -> float:
    n = len(D)
    vals = []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2:]
        for s, t in ((a, b), (b, a)):
            vals.append(D[t, int(np.argmax(D[s].mean(0)))].mean())
    return float(np.mean(vals))


def null_max(D: np.ndarray, rng, B=B_NULL) -> float:
    D0 = D - D.mean(0)
    idx = rng.integers(0, len(D), (B, len(D)))
    return float(np.mean([D0[i].mean(0).max() for i in idx]))


rows, picks = [], []
for (s, d, Y), g in E.groupby(["set", "dataset", "target"]):
    for metric in ("d_spearman", "d_sel_diff_f10"):
        Dm = year_matrix(g, metric)
        cols, D = list(Dm.columns), Dm.to_numpy(float)
        mean = D.mean(0)
        j = int(np.argmax(mean))
        naive = float(mean[j])
        cf = cross_fit(D, rng)
        nb = null_max(D, rng)
        # SEs by environment bootstrap (cross-fit with fewer splits inside each replicate)
        idx = rng.integers(0, len(D), (B_SE, len(D)))
        boot_max = np.array([D[i].mean(0).max() for i in idx])
        se_naive = float(np.std(boot_max, ddof=1))
        bias_corrected = 2 * naive - float(boot_max.mean())  # bootstrap bias correction of the max
        se_cf = float(np.std([cross_fit(D[i], rng, 20) for i in idx], ddof=1))
        rows.append({"set": s, "dataset": d, "target": Y, "metric": metric[2:], "n_env": len(D), "n_methods": len(cols),
                     "naive": naive, "se_naive": se_naive, "bias_corrected": bias_corrected, "cross_fit": cf, "se_cross_fit": se_cf, "null_max": nb,
                     "naive_minus_null": naive - nb})
        if metric == "d_spearman":
            order = np.argsort(-mean)
            picks.append({"set": s, "dataset": d, "target": Y, "oracle": cols[j], "class": CLASS.get(cols[j], "?"),
                          "oracle_d": naive, "cell_rank": int(np.where(np.array(cols)[order] == "cell_reml")[0][0]) + 1,
                          "n_methods": len(cols), "second": cols[order[1]], "second_d": float(mean[order[1]])})
YR = pd.DataFrame(rows)
PK = pd.DataFrame(picks)
YR.to_csv(OUT / "oracle_years.csv", index=False)

pooled = []
for metric, g in YR.groupby("metric"):
    for scope, sub in (("orig31", g[g["set"] == "orig31"]), ("new17", g[g["set"] == "new17"]), ("all48", g)):
        for q, se in (("naive", "se_naive"), ("cross_fit", "se_cross_fit")):
            r = pool(sub.rename(columns={q: "d", se: "se"})[["dataset", "d", "se"]])
            pooled.append({"metric": metric, "scope": scope, "quantity": q, **r})
        for q in ("naive", "bias_corrected", "cross_fit"):
            r = stratified_unweighted(sub.rename(columns={q: "d"})[["dataset", "d"]], rng)
            pooled.append({"metric": metric, "scope": scope, "quantity": q + "_unweighted", **r, "k": len(sub)})
        pooled.append({"metric": metric, "scope": scope, "quantity": "null_max_mean", "est": float(sub["null_max"].mean()),
                       "k": len(sub)})
        pooled.append({"metric": metric, "scope": scope, "quantity": "naive_minus_null_mean",
                       "est": float(sub["naive_minus_null"].mean()), "k": len(sub)})
    for d_, sub in g.groupby("dataset"):
        r = pool(sub.rename(columns={"cross_fit": "d", "se_cross_fit": "se"})[["dataset", "d", "se"]])
        pooled.append({"metric": metric, "scope": d_, "quantity": "cross_fit", **r})
        pooled.append({"metric": metric, "scope": d_, "quantity": "naive_mean", "est": float(sub["naive"].mean()), "k": len(sub)})
        pooled.append({"metric": metric, "scope": d_, "quantity": "null_max_mean", "est": float(sub["null_max"].mean()),
                       "k": len(sub)})
PO = pd.DataFrame(pooled)
PO.to_csv(OUT / "oracle_pooled.csv", index=False)

# A1: pick table vs the pre-registered Oracle choices (consistency check)
pre = pd.concat([pd.read_csv(RES / "b3b" / "choices_weights.csv").query("library == 'main'")[["dataset", "target", "oracle"]],
                 pd.read_csv(RES / "b3d" / "choices_weights.csv")[["dataset", "target", "oracle"]]])
PK = PK.merge(pre.rename(columns={"oracle": "oracle_prereg"}), on=["dataset", "target"], how="left")
PK.to_csv(OUT / "oracle_picks.csv", index=False)

# A3: nonlinear minus linear two-stage (same stage-1 genotype means)
nl_rows = []
W = E.pivot_table(index=["set", "dataset", "target", "env"], columns="item", values="d_spearman")
for m in ("rf", "gbm", "knn10", "knn30", "mlp", "ridge_pc20", "reml_x10"):
    x = (W[m] - W["reml"]).dropna().rename("d").reset_index()
    ye = x.groupby(["set", "dataset", "target"])["d"].agg(["mean", "count", "std"]).reset_index()
    ye["se"] = ye["std"] / np.sqrt(ye["count"])
    ye = ye.rename(columns={"mean": "d"})
    for scope, sub in [("all48", ye)] + list(ye.groupby("dataset")):
        r = pool(sub[["dataset", "d", "se"]].fillna({"se": 0}))
        nl_rows.append({"contrast": f"{m} - reml(2stage)", "scope": scope, **r,
                        "years_positive": int((sub["d"] > 0).sum())})
NL = pd.DataFrame(nl_rows)
NL.to_csv(OUT / "nonlinear_vs_linear.csv", index=False)

# ------------------------------------------------------------------------------------------------ Part B
splits = json.load(open(RES / "benchmark" / "splits.json"))


def location(ds: str, env: str):
    base = env.rsplit("_", 1)[0]
    if ds == "G2F":
        m = re.match(r"^([A-Z]{2}H\d)", base)
        return m.group(1) if m else base
    if ds in ("NUST", "URSN", "ESWYT"):
        return base
    return None  # GEM_IA: Loc letter is unique per location-year; MU_SOY: fields of one station


def zscore(c: pd.DataFrame) -> pd.Series:
    g = c.groupby("env")["y"]
    sd = g.transform("std").replace(0, np.nan)
    return (c["y"] - g.transform("mean")) / sd


def spear(a, b):
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


ceil_rows, cons_rows, pair_rows, struct_rows = [], [], [], []
for ds in DS:
    C = pd.read_parquet(RES / "benchmark" / f"cells_{ds}.parquet")
    C["z"] = zscore(C)
    C["loc"] = [location(ds, e) for e in C["env"]]
    P = pd.read_parquet(RES / "benchmark" / f"predictions_{ds}.parquet")
    P = P[P["method"] == "cell_reml"].set_index(["env", "genotype"])["pred"]
    min_n = splits[ds]["min_genotypes"]
    nenv = C.groupby("env")["genotype"].nunique()
    big = set(nenv[nenv >= min_n].index)
    first_year = C.groupby("genotype")["year"].min()
    # B1/B3: same-year leave-one-environment-out genotype mean, for every environment with >= min_n lines
    for Y, cy in C[C["env"].isin(big)].groupby("year"):
        Z = cy.pivot_table(index="genotype", columns="env", values="z")
        S, N = Z.fillna(0).sum(1), Z.notna().sum(1)
        for e in Z.columns:
            ze = Z[e].dropna()
            other_n = N.loc[ze.index] - 1
            ok = other_n > 0
            gm = (S.loc[ze.index] - ze)[ok] / other_n[ok]
            yv = cy[cy["env"] == e].set_index("genotype")["y"].loc[gm.index]
            cons_rows.append({"dataset": ds, "year": Y, "env": e, "n": int(ok.sum()), "n_env_year": Z.shape[1],
                              "consensus": spear(yv.to_numpy(), gm.to_numpy())})
    targets = splits[ds]["target_years"]
    for Y in targets:
        envs = splits[ds]["scorable_environments"][str(Y)]
        hist = C[C["year"] < Y]
        seen_g, seen_loc = set(hist["genotype"]), set(hist["loc"].dropna())
        cy = C[(C["year"] == Y) & C["env"].isin(envs)]
        Z = cy.pivot_table(index="genotype", columns="env", values="z")
        S, N = Z.fillna(0).sum(1), Z.notna().sum(1)
        for e in envs:
            ce = cy[cy["env"] == e].set_index("genotype")
            ze = Z[e].dropna()
            other_n = N.loc[ze.index] - 1
            ok = other_n > 0
            gm = (S.loc[ze.index] - ze)[ok] / other_n[ok]
            pr = P.reindex(pd.MultiIndex.from_product([[e], ce.index])).to_numpy()
            has = ~np.isnan(pr)
            ceil_rows.append({"dataset": ds, "target": Y, "env": e, "n": len(ce),
                              "new_share": float((~ce.index.isin(seen_g)).mean()),
                              "loc_seen": (location(ds, e) in seen_loc) if location(ds, e) else np.nan,
                              "gm_ceiling": spear(ce.loc[gm.index, "y"].to_numpy(), gm.to_numpy()),
                              "cell_reml": spear(ce["y"].to_numpy()[has], pr[has]),
                              "cell_reml_same_set": spear(ce.loc[gm.index, "y"].to_numpy(),
                                                          P.reindex(pd.MultiIndex.from_product([[e], gm.index])).to_numpy())})
        # B4 pedigree structure
        tg = sorted(set(cy["genotype"]) - seen_g)
        if ds == "G2F":
            par = lambda h: h.split("/") if "/" in h else [h, None]
            seen_par = {p for h in seen_g for p in par(h) if p}
            p1 = np.mean([par(h)[0] in seen_par for h in tg]) if tg else np.nan
            p2 = np.mean([(par(h)[1] in seen_par) if par(h)[1] else False for h in tg]) if tg else np.nan
            both = np.mean([all(p in seen_par for p in par(h) if p) and par(h)[1] is not None for h in tg]) if tg else np.nan
            struct_rows.append({"dataset": ds, "target": Y, "new_lines": len(tg), "parent1_seen": p1, "parent2_seen": p2,
                                "both_parents_seen": both})
        elif ds == "GEM_IA":
            fam = lambda h: h.split(":")[0]
            seen_f = {fam(h) for h in seen_g}
            struct_rows.append({"dataset": ds, "target": Y, "new_lines": len(tg),
                                "family_seen": float(np.mean([fam(h) in seen_f for h in tg])) if tg else np.nan,
                                "n_families_target": len({fam(h) for h in tg})})
    # B2 pairwise environment correlations over shared genotypes (environments with >= min_n lines, all years)
    Cb = C[C["env"].isin(big)]
    Zm = Cb.pivot_table(index="genotype", columns="env", values="z")
    envs = list(Zm.columns)
    M = Zm.notna().to_numpy(float)
    X = Zm.fillna(0).to_numpy(float)
    n = M.T @ M
    Sx = X.T @ M                      # Sx[i, j] = sum of env i over genotypes shared with env j
    Sxx = (X * X).T @ M
    Sxy = X.T @ X
    with np.errstate(invalid="ignore", divide="ignore"):
        num = n * Sxy - Sx * Sx.T
        den = np.sqrt((n * Sxx - Sx ** 2) * (n * Sxx.T - Sx.T ** 2))
        R = num / den
    yr = Cb.drop_duplicates("env").set_index("env").loc[envs, "year"].to_numpy()
    lc = Cb.drop_duplicates("env").set_index("env").loc[envs, "loc"].to_numpy()
    iu = np.triu_indices(len(envs), 1)
    min_shared = 8 if ds == "URSN" else 10
    sel = n[iu] >= min_shared
    pr = pd.DataFrame({"r": R[iu][sel], "n": n[iu][sel], "gap": np.abs(yr[iu[0]] - yr[iu[1]])[sel],
                       "same_loc": [(a == b) if (a is not None and b is not None) else np.nan
                                    for a, b in zip(lc[iu[0]][sel], lc[iu[1]][sel])]})
    pr["dataset"] = ds
    pair_rows.append(pr.dropna(subset=["r"]))

CE = pd.DataFrame(ceil_rows)
CE.to_csv(OUT / "ceiling_envs.csv", index=False)
CO = pd.DataFrame(cons_rows)
CO.to_csv(OUT / "consensus_envs.csv", index=False)
PR = pd.concat(pair_rows, ignore_index=True)
ST = pd.DataFrame(struct_rows)
ST.to_csv(OUT / "pedigree_structure.csv", index=False)


def wmean(x):
    return float(np.average(x["r"], weights=x["n"] - 3)) if len(x) else np.nan


pair_sum = []
for ds, g in PR.groupby("dataset"):
    for label, sub in (("same_year_diff_loc", g[(g["gap"] == 0) & (g["same_loc"] == False)]),  # noqa: E712
                       ("same_year_any_loc", g[g["gap"] == 0]),
                       ("diff_year_same_loc", g[(g["gap"] > 0) & (g["same_loc"] == True)]),  # noqa: E712
                       ("diff_year_diff_loc", g[(g["gap"] > 0) & (g["same_loc"] == False)]),  # noqa: E712
                       ("diff_year_any_loc", g[g["gap"] > 0])):
        pair_sum.append({"dataset": ds, "class": label, "pairs": len(sub), "median_shared": float(sub["n"].median()) if len(sub) else np.nan,
                         "r_weighted": wmean(sub)})
    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 9), (10, 99)):
        sub = g[(g["gap"] >= lo) & (g["gap"] <= hi)]
        pair_sum.append({"dataset": ds, "class": f"gap_{lo}_{hi}", "pairs": len(sub),
                         "median_shared": float(sub["n"].median()) if len(sub) else np.nan, "r_weighted": wmean(sub)})
PS = pd.DataFrame(pair_sum)
PS.to_csv(OUT / "pairwise_env_corr.csv", index=False)

# A4: year features vs gains
yf = CE.groupby(["dataset", "target"]).agg(n_env=("env", "size"), median_n=("n", "median"), new_share=("new_share", "mean"),
                                            cell_level=("cell_reml", "mean"), ceiling=("gm_ceiling", "mean"),
                                            loc_seen=("loc_seen", "mean")).reset_index()
C_hist = {ds: pd.read_parquet(RES / "benchmark" / f"cells_{ds}.parquet") for ds in DS}
yf["history_envs"] = [C_hist[d][C_hist[d]["year"] < Y]["env"].nunique() for d, Y in zip(yf["dataset"], yf["target"])]
ys = YR[YR["metric"] == "spearman"].drop(columns=["n_env"]).merge(yf, on=["dataset", "target"])
ys = ys.merge(PK[["dataset", "target", "oracle", "class", "cell_rank"]], on=["dataset", "target"])
ys.to_csv(OUT / "year_features.csv", index=False)
cor = []
for y_ in ("naive", "cross_fit"):
    for x_ in ("n_env", "median_n", "new_share", "history_envs", "cell_level", "ceiling"):
        r, p = spearmanr(ys[x_], ys[y_], nan_policy="omit")
        cor.append({"gain": y_, "feature": x_, "spearman_rho": float(r), "p_descriptive": float(p), "k": len(ys)})
pd.DataFrame(cor).to_csv(OUT / "gain_vs_features.csv", index=False)

# environment level: d(rf), d(gbm) vs environment size and cell_reml level, by dataset
env_rows = []
Wf = E.pivot_table(index=["dataset", "target", "env"], columns="item", values="d_spearman").join(cell_level)
Wf = Wf.join(E.groupby(["dataset", "target", "env"])["n"].first())
for ds, g in Wf.groupby(level=0):
    for m in ("rf", "gbm", "knn10"):
        ok = g[[m, "n", "cell_spearman"]].dropna()
        env_rows.append({"dataset": ds, "method": m, "envs": len(ok),
                         "rho_d_vs_n": float(spearmanr(ok[m], ok["n"])[0]),
                         "rho_d_vs_cell_level": float(spearmanr(ok[m], ok["cell_spearman"])[0])})
pd.DataFrame(env_rows).to_csv(OUT / "env_gain_vs_size.csv", index=False)

# ------------------------------------------------------------------------------------------------ printout
pd.set_option("display.width", 200)
print("== Oracle pooled (Spearman / sel_diff)")
print(PO.round(4).to_string(index=False))
print("== Oracle picks (count by dataset x class)")
print(pd.crosstab(PK["dataset"], PK["class"]))
print(pd.crosstab(PK["dataset"], PK["oracle"]))
print("prereg Oracle agrees:", float((PK["oracle"] == PK["oracle_prereg"]).mean()))
print(PK.to_string(index=False))
print("== nonlinear vs linear 2-stage")
print(NL.round(4).to_string(index=False))
print("== ceiling vs cell_reml by dataset")
cs = CE.groupby("dataset")[["gm_ceiling", "cell_reml_same_set", "cell_reml", "new_share", "loc_seen", "n"]].mean()
print(cs.round(3))
print(CE.groupby(["dataset", "target"])[["gm_ceiling", "cell_reml_same_set", "new_share", "loc_seen"]].mean().round(3).to_string())
print("== consensus spread")
print(CO.groupby("dataset")["consensus"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3))
print(CO.groupby("dataset")["consensus"].apply(lambda s: float((s < 0.1).mean())).round(3))
print("== pairwise")
print(PS.round(3).to_string(index=False))
print("== pedigree")
print(ST.round(3).to_string(index=False))
print("== gain vs features")
print(pd.DataFrame(cor).round(3).to_string(index=False))
print(pd.DataFrame(env_rows).round(3).to_string(index=False))
json.dump({"inputs": ["results/summary48/per_env.parquet", "results/benchmark/*"], "seed": 20260929,
           "n_split": N_SPLIT, "b_se": B_SE, "b_null": B_NULL, "years": int(len(PK))}, open(OUT / "meta.json", "w"), indent=1)
print("IDEAS_DIAG_DONE")
