"""Other-trait cells for the I-B holdout (docs/prereg_holdout_ib_2026-09-29.md §2). Cell construction repeats
dartgxe.forward.data.load_nust / load_ursn exactly, for another trait; the target-year count repeats
scripts/b3d_count.py (the W1c rules) exactly."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

NUST_TRAITS = ["Height", "Maturity", "SeedSize", "Lodging", "Protein", "Oil", "SeedQuality"]
URSN_TRAITS = ["VSK"]
EXCLUDED = {"YieldRank": "function of yield within the trial", "DescriptiveCode": "free-text code"}
MIN_RELAXED, NEW_SHARE, MIN_ENVS, MIN_PRIOR_YEARS = 10, 0.5, 5, 2


def norm(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def nust_trait_cells(ph: pd.DataFrame, trait: str, ids) -> pd.DataFrame:
    """ph = raw NUST plot table (Experiment, Location, GermplasmId, Phenotype, Value); ids = genotyped line ids."""
    if trait in EXCLUDED or trait not in NUST_TRAITS:
        raise ValueError(f"{trait} is not a pre-registered holdout trait")
    d = ph[ph["Phenotype"] == trait].copy()
    d["y"] = pd.to_numeric(d["Value"], errors="coerce")
    d["year"] = pd.to_numeric(d["Experiment"].str.extract(r"_(\d{4})$")[0], errors="coerce")
    d = d.dropna(subset=["y", "year"])
    d["year"] = d["year"].astype(int)
    d["env"] = d["Location"].astype(str) + "_" + d["year"].astype(str)
    d["genotype"] = d["GermplasmId"].map(norm)
    d = d[d["genotype"].isin(set(ids))]
    return d.groupby(["env", "year", "genotype"], as_index=False)["y"].mean()


def ursn_trait_cells(u: pd.DataFrame, trait: str, ids) -> pd.DataFrame:
    if trait not in URSN_TRAITS:
        raise ValueError(f"{trait} is not a pre-registered holdout trait")
    d = u.dropna(subset=[trait])
    cells = pd.DataFrame({"env": d["env_code"].astype(str), "year": d["year"].astype(int),
                          "genotype": d["line"].astype(str), "y": d[trait].astype(float)})
    cells = cells[cells["genotype"].isin(set(ids))]
    return cells.groupby(["env", "year", "genotype"], as_index=False)["y"].mean()


def count_targets(cells: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """W1c rules, main and relaxed thresholds (same as scripts/b3d_count.py::count)."""
    cells = cells.assign(year=cells["year"].astype(int))
    per_env = cells.groupby(["year", "env"])["genotype"].nunique().rename("n").reset_index()
    first_year = cells.groupby("genotype")["year"].min()
    rows = []
    for tag, thr in (("main", min_n), ("relaxed", MIN_RELAXED)):
        big = per_env[per_env["n"] >= thr]
        years_with_big = sorted(big["year"].unique())
        for yr in sorted(cells["year"].unique()):
            envs = big[big["year"] == yr]["env"]
            lines = cells[(cells["year"] == yr) & cells["env"].isin(envs)]["genotype"].unique()
            prior = [y for y in years_with_big if y < yr]
            new = float((first_year.loc[lines] >= yr).mean()) if len(lines) else float("nan")
            n_cells = int(big[big["year"] == yr]["n"].sum())
            ok = len(prior) >= MIN_PRIOR_YEARS and len(envs) >= MIN_ENVS and len(lines) > 0 and new >= NEW_SHARE
            rows.append({"rule": tag, "min_n": thr, "year": int(yr), "envs_total": int((per_env["year"] == yr).sum()),
                         "envs_ge_min": int(len(envs)), "median_n": float(per_env[per_env["year"] == yr]["n"].median()),
                         "lines": int(len(lines)), "cells": n_cells,
                         "resolution": round(3 / n_cells ** 0.5, 3) if n_cells else float("nan"),
                         "new_share": round(new, 3), "prior_years": len(prior), "qualifies": bool(ok)})
    return pd.DataFrame(rows)
