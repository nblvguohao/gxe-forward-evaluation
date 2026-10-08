"""Value of the target-year phenotypes in NUST under the post hoc environment definitions (2026-10-08; extends
docs/analysis_note_sparse_value_2026-10-08.md part A to the definitions of Supplementary Note S4). No refit.

Same cells, masks and models as scripts/sparse_value_of_phenotypes.py (m0, m1 from the sparse panels; ref = ia_g from the
I-A panels), rescored as in scripts/env_definition_sparse.py: NUST yield centred on the trial mean within each
environment ('detrend', environments with >= 25 lines), or trial x location x year as the scoring unit ('trial_env',
>= 10 lines), on every environment of the panel as in scripts/env_definition_sparse.py; 'orig' must reproduce results/sparse_value (NUST, all lines). Two masks averaged per unit, unit bootstrap
(B = 2,000, seed 20261008), SE floor and DerSimonian-Laird pooling; judged against NUST's own resolution in the
pre-specified scoring (as Supplementary Table S11).

Env: SPARSE_PANELS, IA_PANELS as in sparse_value_of_phenotypes.py; GP_RAW = raw data root (nust/ table).
Run from the repository root  ->  results/sparse_value/nust_posthoc.csv"""
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "src")
from dartgxe.forward.pool import dersimonian_laird, floor_se, hartung_knapp  # noqa: E402

PAN, IAP = os.environ["SPARSE_PANELS"], os.environ["IA_PANELS"]
RAW = os.environ.get("GP_RAW", "<local>/gp_project/data/raw")
B, SEED = 2000, 20261008
norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())


def sp(y, p):
    return np.nan if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0 else float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


def verdict(est, lo, hi, res):
    if lo > 0 and est >= res:
        return "resolved gain"
    if hi < 0 and -est >= res:
        return "resolved loss"
    if lo > 0 or hi < 0:
        return "detectable, below resolution"
    return "tied"


ph = pd.read_csv(f"{RAW}/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz", low_memory=False)
ph = ph[ph.Phenotype == "YieldBuA"].copy(); ph["y"] = pd.to_numeric(ph.Value, errors="coerce")
ph["year"] = pd.to_numeric(ph.Experiment.str.extract(r"_(\d{4})$")[0], errors="coerce"); ph = ph.dropna(subset=["y", "year"])
ph["env"] = ph.Location.astype(str) + "_" + ph.year.astype(int).astype(str); ph["genotype"] = ph.GermplasmId.map(norm)
ph["trial"] = ph.Experiment.str.replace(r"_\d{4}$", "", regex=True)
rec = ph.groupby(["env", "trial", "genotype"], as_index=False).y.mean()


def units(P, variant):
    cols = ["env", "genotype", "m0", "m1", "ref"]
    if variant == "orig":
        return P.assign(unit=P.env), 25
    r = rec.merge(P[cols], on=["env", "genotype"])
    if variant == "detrend":
        r["yd"] = r.y - r.groupby(["env", "trial"]).y.transform("mean")
        g = r.groupby(["env", "genotype"], as_index=False).agg(y=("yd", "mean"), m0=("m0", "first"), m1=("m1", "first"), ref=("ref", "first"))
        return g.assign(unit=g.env), 25
    return r.assign(unit=r.env + "|" + r.trial), 10


def main():
    E = pd.read_parquet("results/headroom_sparse/per_env_seed.parquet")
    E = E[E["dataset"] == "NUST"]
    out = []
    for path in sorted(glob.glob(f"{PAN}/NUST_*_f*_s*.parquet")):
        m = re.match(r"NUST_(\d{4})_f(\d+)_s(\d)\.parquet$", Path(path).name)
        Y, fr, s = int(m.group(1)), int(m.group(2)) / 100, int(m.group(3))
        keep = set(E[(E["target"] == Y) & (E["fraction"] == fr) & (E["seed"] == s)]["env"])
        P_all = pd.read_parquet(path)
        P = P_all[P_all["env"].isin(keep)].copy()
        ia = pd.read_parquet(f"{IAP}/NUST_{Y}.parquet").drop_duplicates(["env", "genotype"]).set_index(["env", "genotype"])
        P["ref"] = ia["ia_g"].reindex(pd.MultiIndex.from_frame(P[["env", "genotype"]])).to_numpy()
        assert P["ref"].notna().all()
        # post hoc definitions use every environment of the panel, as scripts/env_definition_sparse.py (Table S11, Note S4)
        P_all = P_all.copy()
        P_all["ref"] = ia["ia_g"].reindex(pd.MultiIndex.from_frame(P_all[["env", "genotype"]])).to_numpy()
        for variant in ("orig", "detrend", "trial_env"):
            U, mn = units(P if variant == "orig" else P_all.dropna(subset=["ref"]), variant)
            for u, g in U.groupby("unit"):
                if g.genotype.nunique() < mn:
                    continue
                y = g.y.to_numpy(float)
                out.append({"fraction": fr, "target": Y, "seed": s, "variant": variant, "unit": u, "n": len(g),
                            "m0-ref": sp(y, g.m0.to_numpy(float)) - sp(y, g.ref.to_numpy(float)),
                            "m1-m0": sp(y, g.m1.to_numpy(float)) - sp(y, g.m0.to_numpy(float))})
    D = pd.DataFrame(out)
    rng = np.random.default_rng(SEED)
    res = {0.25: 0.0204, 0.5: 0.0290}  # NUST's own resolution in the pre-specified scoring (Table S15)
    rows = []
    for (fr, v), Dv in D.groupby(["fraction", "variant"]):
        for c in ("m0-ref", "m1-m0"):
            ye = []
            for Y, g in Dv.groupby("target"):
                w = g.groupby("unit")[c].mean().dropna().to_numpy()
                if len(w):
                    se = float(w[rng.integers(0, len(w), (B, len(w)))].mean(1).std(ddof=1))
                    ye.append({"dataset": "NUST", "d": float(w.mean()), "se": se if se > 1e-12 else 0.0})
            h = hartung_knapp(floor_se(pd.DataFrame(ye)))
            rows.append({"fraction": fr, "variant": v, "contrast": c, "k": h["k"], "resolution": res[fr], "est": h["est"], "lo": h["lo"], "hi": h["hi"],
                         "hk_lo": h["hk_lo"], "hk_hi": h["hk_hi"], "verdict": verdict(h["est"], h["lo"], h["hi"], res[fr])})
    T = pd.DataFrame(rows)
    T.to_csv("results/sparse_value/nust_posthoc.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
