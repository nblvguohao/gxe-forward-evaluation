"""Descriptive facts behind the resolution threshold 3/sqrt(N) (Supplementary Note S2 of the TCJ manuscript).

Nothing is fitted or re-scored. From the stored per-environment table of the forward benchmark this reports, per dataset
and overall:
- environments E, scored cells N, and the range of genotypes per environment n;
- the ratio of the standard deviation of an unweighted mean of E within-environment correlations, each with variance
  1/n_e, to 1/sqrt(N): sqrt(N * sum(1/n_e)) / E. It is 1 when all environments have the same size and larger otherwise;
- the factor (1 - rho^2) * sqrt(1.060) in the standard deviation of a Spearman correlation, (1 - rho^2) * sqrt(1.060 / (n - 3)),
  from var(atanh r_S) ~ 1.060 / (n - 3) (Fieller, Hartley and Pearson 1957, Biometrika 44:470-481, eq. 10, p. 472), at the
  mean within-environment Spearman correlation of cell_reml.
And, for the primary pooled contrasts of the manuscript, the standard error implied by the reported interval divided by
1/sqrt(N).

Run from the repository root:  python3 scripts/resolution_scale_facts.py  ->  results/resolution_note/"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

R, OUT = Path("results"), Path("results/resolution_note")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(R / "summary48/per_env.parquet")
    env = d.drop_duplicates(["dataset", "target", "env"])[["dataset", "target", "env", "n"]]
    ref = d.assign(ref=d["spearman"] - d["d_spearman"]).groupby(["dataset", "target", "env"])["ref"].agg(["min", "max"])
    assert float((ref["max"] - ref["min"]).abs().max()) < 1e-9, "reference value differs between items"
    rho = ref["min"].rename("rho_ref").reset_index()
    env = env.merge(rho, on=["dataset", "target", "env"])
    rows = []
    for name, g in list(env.groupby("dataset")) + [("All six datasets", env)]:
        n = g["n"].to_numpy(float)
        E, N = len(n), n.sum()
        r = float(g["rho_ref"].mean())
        rows.append({"dataset": name, "environments": E, "cells": int(N), "n_min": int(n.min()), "n_median": float(np.median(n)),
                     "n_max": int(n.max()), "ratio_unequal_sizes": float(np.sqrt(N * (1 / n).sum()) / E),
                     "mean_rho_cell_reml": r, "fhp_factor": float((1 - r * r) * np.sqrt(1.060))})
    T = pd.DataFrame(rows)
    # standard deviation of one environment's Spearman correlation relative to 1/sqrt(n), at the dataset's mean correlation
    fac = T.set_index("dataset")["fhp_factor"]
    sd_rel = fac.loc[env["dataset"]].to_numpy() * np.sqrt(env["n"].to_numpy(float) / (env["n"].to_numpy(float) - 3))
    T["sd_over_inv_sqrt_n_min"] = [float(sd_rel[(env["dataset"] == d).to_numpy()].min()) if d in fac.index[:-1] else float(sd_rel.min()) for d in T["dataset"]]
    T["sd_over_inv_sqrt_n_max"] = [float(sd_rel[(env["dataset"] == d).to_numpy()].max()) if d in fac.index[:-1] else float(sd_rel.max()) for d in T["dataset"]]
    T.to_csv(OUT / "by_dataset.csv", index=False)
    # the same with each environment's own observed correlation of cell_reml
    r_env = env["rho_ref"].to_numpy(float)
    n_env = env["n"].to_numpy(float)
    rel_env = (1 - r_env ** 2) * np.sqrt(1.060 * n_env / (n_env - 3))
    own_rho = {"environments": int(len(env)), "share_outside_20pct": float(np.mean((rel_env < 0.8) | (rel_env > 1.2))),
               "min": float(rel_env.min()), "max": float(rel_env.max())}

    # Standard errors implied by the reported intervals of the primary pooled contrasts, relative to 1/sqrt(N)
    t = pd.read_csv(R / "floor_sensitivity/table.csv")
    t["se_implied"] = (t["hi_plan"] - t["lo_plan"]) / (2 * norm.ppf(1 - (1 - t["level"]) / 2))
    # 1/sqrt(N) with each contrast's own N. Table S10 gives the covariate learners (16 target years) and the
    # within-environment-loss network (31) the resolution of all 48 years; their own N is taken from summary48.
    item_of = {"Covariate LightGBM": "gxe_gbm", "Covariate reaction-norm ridge": "rn_ridge", "Within-environment-loss network": "dl_g"}
    cells = d.groupby("item")["n"].sum()
    own = {f"Benchmark: {k} − cell-level GBLUP": float(cells[v]) for k, v in item_of.items()}
    t["sd_scale"] = [1 / np.sqrt(own[c]) if c in own else r / 3 for c, r in zip(t["contrast"], t["resolution"])]
    t["se_over_scale"] = t["se_implied"] / t["sd_scale"]
    S = t[["contrast", "target_years", "level", "resolution", "se_implied", "sd_scale", "se_over_scale"]]
    S.to_csv(OUT / "implied_se.csv", index=False)
    summary = {"by_dataset": T.to_dict("records"), "own_rho_per_environment": own_rho,
               "implied_se_over_scale": {"min": float(S["se_over_scale"].min()), "median": float(S["se_over_scale"].median()),
                                         "max": float(S["se_over_scale"].max()), "contrasts": int(len(S))}}
    json.dump(summary, open(OUT / "facts.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    print(S.round(3).to_string(index=False))
    print(json.dumps(summary["implied_se_over_scale"], indent=1))


if __name__ == "__main__":
    main()
