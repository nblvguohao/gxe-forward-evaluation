"""Value of target-year phenotypes (post hoc, added before submission on 2026-10-08 after the mock reviews).

On exactly the cells scored in sparse testing, compares the two sparse-testing models, which were fitted with 25 % or
50 % of the target-year cells phenotyped (main-effect GBLUP m0 and marker x environment GBLUP m1), with the forward
cell-level GBLUP of the benchmark (cell_reml), which uses no phenotype of the target year. Nothing is refitted: the
stored sparse panels (per-cell m0, m1) and the stored benchmark predictions (cell_reml) are rescored.

Same per-environment metric (within-environment Spearman), seed averaging (two masks), environment bootstrap
(B = 2,000), SE floor and DerSimonian-Laird pooling as scripts/headroom_sparse.py; resolution 3/sqrt(N) with N the
cells scored with mask 0. Environments are those of results/headroom_sparse/per_env_seed.parquet.

Contrasts: m0 - ctrl (value of the target-year phenotypes with a main-effect model), m1 - m0 (G x E given them) and m1 - ctrl
(their sum); the bootstrap draws the same environments for all three. Reported for all lines, for new lines only (first year in
the cells of the dataset = target year or later) and old lines only, each in environments with at least 10 such lines; and,
besides all target years, on the target years common to the 25 % and 50 % analyses.
Reference (docs/analysis_note_sparse_value_2026-10-08.md, A2): `ia_g` of the I-A panels, the same main-effect GBLUP as m0
(fit_ia, kernel (1 - w) K_A + w I, w by profile REML on W_GRID, environment fixed effects) fitted to the years before the
target year only, i.e. with 0 % of the target year phenotyped. cell_reml is reported as a second, descriptive reference.

Env: SPARSE_PANELS = directory with the sparse panels {DATASET}_{YEAR}_f{25|50}_s{0|1}.parquet (4090:
scratch_ideas_2026-09-29/headroom_sparse/panels/); IA_PANELS = directory with the I-A panels {DATASET}_{YEAR}.parquet (4090:
scratch_ideas_2026-09-29/headroom_ia_ib/panels/).
Run from the repository root:  SPARSE_PANELS=... python3 scripts/sparse_value_of_phenotypes.py  ->  results/sparse_value/"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "src")
from dartgxe.forward.pool import dersimonian_laird, floor_se, hartung_knapp  # noqa: E402

R, OUT = Path("results"), Path("results/sparse_value")
PAN = os.environ["SPARSE_PANELS"]
IAP = os.environ["IA_PANELS"]
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
B, SEED = 2000, 20261008


def sp(y, p):
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


def verdict(est, lo, hi, res):
    if lo > 0 and est >= res:
        return "resolved gain"
    if hi < 0 and -est >= res:
        return "resolved loss"
    if lo > 0 or hi < 0:
        return "detectable, below resolution"
    return "tied"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = pd.read_parquet(R / "headroom_sparse/per_env_seed.parquet")
    first = {d: pd.read_parquet(R / f"benchmark/cells_{d}.parquet").groupby("genotype")["year"].min() for d in DS}
    pred = {d: pd.read_parquet(R / f"benchmark/predictions_{d}.parquet").query("method == 'cell_reml'").set_index(["env", "genotype"])["pred"]
            for d in DS}
    rows, missing = [], []
    for f in sorted(glob.glob(f"{PAN}/*.parquet")):
        m = re.match(r"(.+)_(\d{4})_f(\d+)_s(\d)\.parquet$", Path(f).name)
        ds, Y, fr, s = m.group(1), int(m.group(2)), int(m.group(3)) / 100, int(m.group(4))
        keep = E[(E["dataset"] == ds) & (E["target"] == Y) & (E["fraction"] == fr) & (E["seed"] == s)]
        if keep.empty:
            continue
        P = pd.read_parquet(f)
        P = P[P["env"].isin(set(keep["env"]))].copy()
        key = pd.MultiIndex.from_frame(P[["env", "genotype"]])
        P["ctrl"] = pred[ds].reindex(key).to_numpy()
        ia = pd.read_parquet(f"{IAP}/{ds}_{Y}.parquet").drop_duplicates(["env", "genotype"]).set_index(["env", "genotype"])
        P["ref"] = ia["ia_g"].reindex(key).to_numpy()
        ia_ctrl = ia["ctrl"].reindex(key).to_numpy()
        ok = ~np.isnan(ia_ctrl) & ~P["ctrl"].isna().to_numpy()
        agree = [sp(g["c"].to_numpy(), g["i"].to_numpy()) for _, g in pd.DataFrame({"e": P["env"].to_numpy()[ok], "c": P["ctrl"].to_numpy()[ok], "i": ia_ctrl[ok]}).groupby("e")]
        missing.append({"panel": Path(f).name, "cells": len(P), "without_cell_reml": int(P["ctrl"].isna().sum()),
                        "without_ia_g": int(P["ref"].isna().sum()), "ia_ctrl_vs_cell_reml_median_sp": float(np.nanmedian(agree)) if agree else np.nan})
        P = P.dropna(subset=["ctrl", "ref", "y"])
        n_ref = keep.set_index("env")["n"]
        P["new"] = (P["genotype"].map(first[ds]) >= Y).to_numpy()
        for e, g0 in P.groupby("env"):
            for scope, g in (("all", g0), ("new", g0[g0["new"]]), ("old", g0[~g0["new"]])):
                if scope != "all" and len(g) < 10:
                    continue
                y = g["y"].to_numpy(float)
                rows.append({"dataset": ds, "target": Y, "fraction": fr, "seed": s, "env": e, "scope": scope, "n": len(g),
                             "n_sparse": int(n_ref.get(e, -1)), "sp_m0": sp(y, g["m0"].to_numpy(float)), "sp_m1": sp(y, g["m1"].to_numpy(float)),
                             "sp_ctrl": sp(y, g["ctrl"].to_numpy(float)), "sp_ref": sp(y, g["ref"].to_numpy(float))})
    D = pd.DataFrame(rows)
    D.to_parquet(OUT / "per_env_seed.parquet", index=False)
    pd.DataFrame(missing).to_csv(OUT / "coverage.csv", index=False)
    rng = np.random.default_rng(SEED)
    ye = []
    for (fr, scope), Df in D.groupby(["fraction", "scope"]):
        W = Df.groupby(["dataset", "target", "env"])[["sp_m0", "sp_m1", "sp_ctrl", "sp_ref"]].mean().dropna().reset_index()
        for (d, Y), g in W.groupby(["dataset", "target"]):
            idx = rng.integers(0, len(g), (B, len(g)))  # the same resampled environments for the three contrasts
            for name, (a, b) in {"m0-ref": ("sp_m0", "sp_ref"), "m1-ref": ("sp_m1", "sp_ref"), "m1-m0": ("sp_m1", "sp_m0"),
                                 "m0-ctrl": ("sp_m0", "sp_ctrl"), "m1-ctrl": ("sp_m1", "sp_ctrl"), "ref-ctrl": ("sp_ref", "sp_ctrl")}.items():
                v = (g[a] - g[b]).to_numpy()
                ye.append({"fraction": fr, "scope": scope, "contrast": name, "dataset": d, "target": Y, "d": float(v.mean()),
                           "se": float(v[idx].mean(1).std(ddof=1)), "n_env": len(v)})
    YE = pd.DataFrame(ye)
    YE.loc[YE["se"] < 1e-12, "se"] = 0.0  # a single environment gives a bootstrap SE of ~1e-17; treat as zero (then floored)
    YE.to_csv(OUT / "year_effects.csv", index=False)
    pooled = []
    yrs = {f: set(map(tuple, YE[(YE["fraction"] == f) & (YE["scope"] == "all")][["dataset", "target"]].drop_duplicates().to_numpy())) for f in YE["fraction"].unique()}
    common = set.intersection(*yrs.values())
    for (fr, scope, name), g in YE.groupby(["fraction", "scope", "contrast"]):
        cells = D[(D["fraction"] == fr) & (D["seed"] == 0) & (D["scope"] == scope)].groupby("dataset")["n"].sum()
        gc = g[[(d, t) in common for d, t in zip(g["dataset"], g["target"])]]
        cells_c = D[(D["fraction"] == fr) & (D["seed"] == 0) & (D["scope"] == scope) &
                    [(d, t) in common for d, t in zip(D["dataset"], D["target"])]]["n"].sum()
        gm = g[g["n_env"] > 1]  # without years scored in a single environment (their SE is borrowed through the floor)
        cells_m = D[(D["fraction"] == fr) & (D["seed"] == 0) & (D["scope"] == scope)].groupby(["dataset", "target"]).filter(lambda x: x["env"].nunique() > 1)["n"].sum()
        for rng_name, dsets, gg, nn in [("all", DS, g, None), ("common_years", DS, gc, cells_c), ("multi_env_years", DS, gm, cells_m)] + [(d, [d], g, None) for d in DS]:
            sub = gg[gg["dataset"].isin(dsets)][["dataset", "d", "se"]].reset_index(drop=True)
            if sub.empty:
                continue
            h = hartung_knapp(floor_se(sub))
            res = 3 / np.sqrt(nn if nn is not None else cells.reindex(dsets).dropna().sum())
            pooled.append({"fraction": fr, "scope": scope, "contrast": name, "range": rng_name, "k": h["k"], "resolution": res, "est": h["est"], "lo": h["lo"],
                           "hi": h["hi"], "tau2": h["tau2"], "hk_lo": h["hk_lo"], "hk_hi": h["hk_hi"], "verdict": verdict(h["est"], h["lo"], h["hi"], res)})
    PO = pd.DataFrame(pooled)
    PO.to_csv(OUT / "pooled.csv", index=False)
    lvl = D[D["scope"] == "all"].groupby(["fraction"])[["sp_m0", "sp_m1", "sp_ctrl", "sp_ref"]].mean()
    cov = pd.DataFrame(missing)
    assert cov["without_ia_g"].sum() == 0, "ia_g does not cover every scored cell"
    assert cov["ia_ctrl_vs_cell_reml_median_sp"].min() >= 0.999, "I-A control disagrees with cell_reml"
    json.dump({"panels": int(len(missing)), "cells_without_cell_reml": int(sum(m["without_cell_reml"] for m in missing)),
               "mean_levels": lvl.round(4).to_dict(), "common_years": len(common), "seed": SEED, "B": B}, open(OUT / "meta.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print(PO.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
