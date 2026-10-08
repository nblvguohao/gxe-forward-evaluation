"""Post hoc sensitivity of the sparse-testing result (M x E GBLUP minus main-effect GBLUP) to the NUST and URSN
environment definitions (2026-10-04). Rescores the stored per-cell sparse panels (no refit): NUST with yield centred
within trial (trial means over the scored cells of each environment), or trial x location x year as the unit (n >= 10);
URSN without check cultivars, in the environments scored originally (>= 10 lines including checks; >= 3 non-check lines). Other datasets keep their year effects from results/headroom_sparse.
Same per-environment metric, seed averaging, bootstrap and pooling as scripts/headroom_sparse.py. Not pre-registered.

Env: PANELS = directory with the NUST_* and URSN_* sparse panels; GP_RAW = raw data root."""
import os, re, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, "src")
from dartgxe.forward.pool import dersimonian_laird, floor_se
warnings.filterwarnings("ignore")
PAN, RAW = os.environ["PANELS"], os.environ.get("GP_RAW", "<local>/gp_project/data/raw")
B, rng = 2000, np.random.default_rng(20261004)
norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
def sp(y, p): return np.nan if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0 else float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])
ph = pd.read_csv(f"{RAW}/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz", low_memory=False)
ph = ph[ph.Phenotype == "YieldBuA"].copy(); ph["y"] = pd.to_numeric(ph.Value, errors="coerce")
ph["year"] = pd.to_numeric(ph.Experiment.str.extract(r"_(\d{4})$")[0], errors="coerce"); ph = ph.dropna(subset=["y", "year"])
ph["env"] = ph.Location.astype(str) + "_" + ph.year.astype(int).astype(str); ph["genotype"] = ph.GermplasmId.map(norm)
ph["trial"] = ph.Experiment.str.replace(r"_\d{4}$", "", regex=True)
rec = ph.groupby(["env", "trial", "genotype"], as_index=False).y.mean()
u = pd.read_csv(f"{RAW}/ursn/data/URSN_cleaned_table_phenot_1995_2024.tsv", sep="\t")
chk = set(u.loc[u.isCheck == True, "line"].astype(str))

def units(ds, P, variant):
    """Returns a frame with unit, genotype, y and m0/m1 for the scored cells under a variant."""
    if variant == "orig": return P.assign(unit=P.env), 25 if ds == "NUST" else 10
    if ds == "URSN":   # same environments as the original scoring (>= 10 lines incl. checks), checks removed
        keep = P.groupby("env").genotype.transform("nunique") >= 10
        return P[keep & ~P.genotype.isin(chk)].assign(unit=P.env), 3
    r = rec.merge(P[["env", "genotype", "m0", "m1"]], on=["env", "genotype"])     # scored cells, one row per trial
    if variant == "detrend":
        r["yd"] = r.y - r.groupby(["env", "trial"]).y.transform("mean")
        g = r.groupby(["env", "genotype"], as_index=False).agg(y=("yd", "mean"), m0=("m0", "first"), m1=("m1", "first"))
        return g.assign(unit=g.env), 25
    return r.assign(unit=r.env + "|" + r.trial), 10                                     # trial_env

out = []
for ds in ("NUST", "URSN"):
    for f in ("0.25", "0.5"):
        fn = {"0.25": "f25", "0.5": "f50"}[f]
        for path in sorted(glob.glob(f"{PAN}/{ds}_*_{fn}_s*.parquet")):
            Y, s = int(path.split("_")[-3]), int(path.split("_s")[-1].split(".")[0])
            P = pd.read_parquet(path)
            for variant in (("orig", "detrend", "trial_env") if ds == "NUST" else ("orig", "nocheck")):
                U, mn = units(ds, P, variant)
                for e, g in U.groupby("unit"):
                    if g.genotype.nunique() < mn: continue
                    y = g.y.to_numpy(float)
                    out.append({"dataset": ds, "fraction": float(f), "target": Y, "seed": s, "variant": variant, "unit": e, "n": len(g),
                                "d": sp(y, g.m1.to_numpy(float)) - sp(y, g.m0.to_numpy(float))})
E = pd.DataFrame(out)
YE = []
for (ds, f, v, Y), g in E.groupby(["dataset", "fraction", "variant", "target"]):
    w = g.groupby("unit").d.mean().dropna().to_numpy()
    if len(w): YE.append({"dataset": ds, "fraction": f, "variant": v, "target": Y, "d": w.mean(), "se": w[rng.integers(0, len(w), (B, len(w)))].mean(1).std(ddof=1)})
YE = pd.DataFrame(YE)
base = pd.read_csv("results/headroom_sparse/year_effects.csv"); base = base[(base.contrast == "m1-m0") & (base.metric == "sp")]
po = pd.read_csv("results/headroom_sparse/pooled.csv"); res = {f: float(po[(po.fraction == f) & (po.contrast == "m1-m0") & (po.metric == "sp") & (po.range == "all48")].resolution.iloc[0]) for f in (0.25, 0.5)}
rows = []
for f in (0.25, 0.5):
    chkd = YE[(YE.fraction == f) & (YE.variant == "orig")].merge(base[base.fraction == f], on=["dataset", "target"], suffixes=("", "_b"))
    print(f"fraction {f}: orig vs stored year effects, max |d diff| {np.abs(chkd.d - chkd.d_b).max():.2e} over {len(chkd)} years")
    other = base[(base.fraction == f) & ~base.dataset.isin(["NUST", "URSN"])][["dataset", "d", "se"]]
    for nv, uv in (("orig", "orig"), ("detrend", "orig"), ("detrend", "nocheck"), ("trial_env", "nocheck")):
        ye = pd.concat([other, YE[(YE.fraction == f) & (YE.dataset == "NUST") & (YE.variant == nv)][["dataset", "d", "se"]],
                        YE[(YE.fraction == f) & (YE.dataset == "URSN") & (YE.variant == uv)][["dataset", "d", "se"]]], ignore_index=True)
        r = dersimonian_laird(floor_se(ye))
        nus = YE[(YE.fraction == f) & (YE.dataset == "NUST") & (YE.variant == nv)]; rn = dersimonian_laird(floor_se(nus[["dataset", "d", "se"]].reset_index(drop=True)))
        rows.append({"fraction": f, "NUST": nv, "URSN": uv, "est": r["est"], "lo": r["lo"], "hi": r["hi"], "k": r["k"], "resolution_orig": res[f],
                     "NUST_alone_est": rn["est"], "NUST_alone_lo": rn["lo"], "NUST_alone_hi": rn["hi"]})
T = pd.DataFrame(rows); os.makedirs("results/env_definition_check", exist_ok=True); T.to_csv("results/env_definition_check/sparse_by_variant.csv", index=False)
pd.set_option("display.width", 250); print(T.round(4).to_string(index=False))
