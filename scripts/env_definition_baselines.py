"""Marker-free reference predictors for NUST and URSN, and the URSN minimum-size sensitivity (2026-10-04, post hoc).

NUST: each line in a target environment is predicted by the historical mean yield (years before the target) of its
maturity group at that location; if the location has no history for that group, by the group mean over all locations.
The maturity group comes from the line's trial in the target environment (UT/PT code, e.g. PTIIIA -> III), which is fixed
before the season. Within an environment the prediction is constant within a maturity group, so the reference is only
defined in environments with at least two maturity groups among the scored lines; methods are compared on those.
URSN: each check cultivar is predicted by its historical mean resistance (100 - DIS, years before the target); all
non-check lines share one value. Because the non-check lines are tied, the score depends on that value and on how ties
are ranked: average ranks with the non-check value set to the historical mean of non-check lines (main), set above or
below every check, and the expectation under random tie-breaking.
Also: benchmark contrasts with URSN environments of at least 12 lines (the specification) instead of 10.
Run from the repository root (GP_RAW = raw data root):  python3 scripts/env_definition_baselines.py"""
import json, os, re, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, "src")
from dartgxe.forward.pool import dersimonian_laird, floor_se
warnings.filterwarnings("ignore")
RAW = os.environ.get("GP_RAW", "<local>/gp_project/data/raw")
S = json.load(open("results/benchmark/splits.json")); OUT = "results/env_definition_check"
rng = np.random.default_rng(20261005); B = 2000
norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
LIB = ["cell_reml", "reml", "reml_x0.1", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp", "dl_g", "rn_ridge", "gxe_gbm"]
def sp(y, p): return np.nan if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0 else float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])
def preds(ds): return pd.read_parquet(f"results/benchmark/predictions_{ds}.parquet").pivot_table(index=["env", "genotype"], columns="method", values="pred")

# ---------------- NUST
ph = pd.read_csv(f"{RAW}/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz", low_memory=False)
ph = ph[ph.Phenotype == "YieldBuA"].copy(); ph["y"] = pd.to_numeric(ph.Value, errors="coerce")
ph["year"] = pd.to_numeric(ph.Experiment.str.extract(r"_(\d{4})$")[0], errors="coerce"); ph = ph.dropna(subset=["y", "year"]); ph["year"] = ph.year.astype(int)
ph["env"] = ph.Location.astype(str) + "_" + ph.year.astype(str); ph["genotype"] = ph.GermplasmId.map(norm)
ph["mg"] = ph.Experiment.str.extract(r"^(?:UT|PT)(00|0|IV|III|II|I)")[0]
P = preds("NUST"); cells = pd.read_parquet("results/benchmark/cells_NUST.parquet")
rows = []
for Ys, envs in S["NUST"]["scorable_environments"].items():
    Y = int(Ys); hist = ph[ph.year < Y]
    loc_mg = hist.groupby(["Location", "mg"]).y.mean(); mg_all = hist.groupby("mg").y.mean()
    tgt = ph[(ph.year == Y) & ph.env.isin(envs)].drop_duplicates(["env", "genotype"])[["env", "Location", "genotype", "mg"]]
    c = cells[cells.env.isin(envs)].merge(tgt, on=["env", "genotype"], how="left")
    c["base"] = [loc_mg.get((l, m), mg_all.get(m, np.nan)) for l, m in zip(c.Location, c.mg)]
    for e, g in c.groupby("env"):
        g = g.dropna(subset=["base"]); n_mg = g.mg.nunique()
        y = g.y.to_numpy(float); pr = P.reindex(pd.MultiIndex.from_arrays([g.env, g.genotype]))
        r = {"dataset": "NUST", "target": Y, "env": e, "n": len(g), "n_mg": n_mg, "baseline": sp(y, g.base.to_numpy(float)) if n_mg >= 2 else np.nan}
        for m in LIB:
            if m in pr: r[m] = sp(y, pr[m].to_numpy(float))
        rows.append(r)
EN = pd.DataFrame(rows)

# ---------------- URSN
u = pd.read_csv(f"{RAW}/ursn/data/URSN_cleaned_table_phenot_1995_2024.tsv", sep="\t").dropna(subset=["DIS"])
u["y"] = 100.0 - u.DIS.astype(float); u["line"] = u.line.astype(str); chk = set(u.loc[u.isCheck == True, "line"])
P = preds("URSN"); cells = pd.read_parquet("results/benchmark/cells_URSN.parquet")
rows = []
for Ys, envs in S["URSN"]["scorable_environments"].items():
    Y = int(Ys); hist = u[u.year < Y]
    cm = hist[hist.line.isin(chk)].groupby("line").y.mean(); nm = hist[~hist.line.isin(chk)].y.mean()
    for e, g in cells[cells.env.isin(envs) & (cells.year == Y)].groupby("env"):
        isc = g.genotype.isin(chk).to_numpy(); y = g.y.to_numpy(float)
        cv = g.genotype.map(cm).to_numpy(float)
        if np.isnan(cv[isc]).any(): cv[isc & np.isnan(cv)] = nm      # a check without history is treated like a non-check
        def with_value(v):
            p = np.where(isc, cv, v); return sp(y, p)
        rnd = []
        for _ in range(200):                                          # random tie-breaking among the non-checks
            p = np.where(isc, cv, nm) + np.where(isc, 0, rng.uniform(-1e-9, 1e-9, len(y))); rnd.append(sp(y, p))
        pr = P.reindex(pd.MultiIndex.from_arrays([g.env, g.genotype]))
        r = {"dataset": "URSN", "target": Y, "env": e, "n": len(g), "check_share": isc.mean(),
             "baseline": with_value(nm), "baseline_above": with_value(np.nanmax(cv[isc]) + 1 if isc.any() else nm),
             "baseline_below": with_value(np.nanmin(cv[isc]) - 1 if isc.any() else nm), "baseline_random": float(np.nanmean(rnd))}
        for m in LIB:
            if m in pr: r[m] = sp(y, pr[m].to_numpy(float))
        rows.append(r)
EU = pd.DataFrame(rows)

def pooled(E, a, b="cell_reml"):
    ye = []
    for (d, Y), g in E.groupby(["dataset", "target"]):
        v = (g[a] - g[b]).dropna().to_numpy()
        if len(v): ye.append({"dataset": d, "d": v.mean(), "se": v[rng.integers(0, len(v), (B, len(v)))].mean(1).std(ddof=1)})
    r = dersimonian_laird(floor_se(pd.DataFrame(ye))); return r["est"], r["lo"], r["hi"], r["k"]
summ = []
ENm = EN[EN.n_mg >= 2]
summ.append({"dataset": "NUST", "environments": f"{len(ENm)} of {len(EN)} with >= 2 maturity groups", "reference": "location x maturity-group history",
             "reference_mean": ENm.baseline.mean(), **{f"{m}_mean": ENm[m].mean() for m in LIB if m in ENm},
             "ref_minus_cell_reml": pooled(ENm, "baseline")})
summ.append({"dataset": "NUST", "environments": f"all {len(EN)} (undefined counted as 0)", "reference": "same",
             "reference_mean": EN.baseline.fillna(0).mean(), **{f"{m}_mean": EN[m].mean() for m in LIB if m in EN}})
for k in ("baseline", "baseline_above", "baseline_below", "baseline_random"):
    summ.append({"dataset": "URSN", "environments": f"all {len(EU)}", "reference": f"check history ({k})", "reference_mean": EU[k].mean(),
                 **{f"{m}_mean": EU[m].mean() for m in LIB if m in EU}, "ref_minus_cell_reml": pooled(EU, k)})
T = pd.DataFrame(summ); os.makedirs(OUT, exist_ok=True); T.to_csv(f"{OUT}/marker_free_reference.csv", index=False)
pd.concat([EN, EU]).to_csv(f"{OUT}/marker_free_reference_per_env.csv", index=False)
pd.set_option("display.width", 250)
print(T[["dataset", "environments", "reference", "reference_mean", "cell_reml_mean", "mlp_mean", "ref_minus_cell_reml"]].round(3).to_string(index=False))
lib10 = ["reml", "reml_x0.1", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp"]
print("NUST multi-MG envs, library range:", round(ENm[lib10].mean().min(), 3), "-", round(ENm[lib10].mean().max(), 3), "| methods below the reference:", [m for m in lib10 if ENm[m].mean() < ENm.baseline.mean()])
print("URSN library range:", round(EU[lib10].mean().min(), 3), "-", round(EU[lib10].mean().max(), 3), "| check share mean", round(EU.check_share.mean(), 3))

# ---------------- URSN minimum size 12: benchmark contrasts
Y48 = pd.read_csv("results/summary48/year_effects.csv"); Y48 = Y48[(Y48.metric == "spearman") & (Y48.kind == "method")]
EU12 = EU[EU.n >= 12]; res = []
for m in LIB[1:]:
    other = Y48[(Y48.item == m) & (Y48.dataset != "URSN")][["dataset", "d", "se"]]
    ye = [other]
    for Y, g in EU12.groupby("target"):
        if m not in g: continue
        v = (g[m] - g.cell_reml).dropna().to_numpy()
        if len(v): ye.append(pd.DataFrame([{"dataset": "URSN", "d": v.mean(), "se": v[rng.integers(0, len(v), (B, len(v)))].mean(1).std(ddof=1)}]))
    r = dersimonian_laird(floor_se(pd.concat(ye, ignore_index=True)))
    v = "resolved loss" if r["hi"] < 0 and -r["est"] >= 0.0087 else ("tied" if r["lo"] <= 0 <= r["hi"] else "detectable")
    res.append({"item": m, "est": r["est"], "lo": r["lo"], "hi": r["hi"], "k": r["k"], "verdict": v})
R = pd.DataFrame(res); R.to_csv(f"{OUT}/ursn_min12_pooled.csv", index=False)
print("\nURSN >= 12 lines:", len(EU12), "of", len(EU), "environments"); print(R.round(4).to_string(index=False))
