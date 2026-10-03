"""W1c (docs/prereg_sequel_wave1_2026-09-28.md §4 and the W1c operating rules appended before any
count): how many qualifying forward years each multi-environment dataset offers. Counts sample
structure only (environments, genotyped lines, new-line share per year); no method is fitted and no
phenotype value is used except to decide whether a cell exists.

Inputs: G2F from the project data root (DARTGXE_DATA); the other datasets from a read-only copy of
gp_project/data (GP_DATA, SHA256SUMS alongside). Output: results/w1c/{per_year.csv, verdict.json,
run_meta.json}."""
import gzip
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

import pandas as pd

from dartgxe.paths import DATA, RESULTS

GP = Path(os.environ["GP_DATA"])
OUT = RESULTS / "w1c"
OUT.mkdir(parents=True, exist_ok=True)
MIN_MAIN = {"URSN": 10}  # everything else 25 (§4)
MIN_RELAXED = 10
NEW_SHARE = 0.5
MIN_ENVS = 5
MIN_PRIOR_YEARS = 2
K_RULE = 12


def vcf_ids(path, norm=lambda s: s):
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                return {norm(s) for s in line.rstrip("\n").split("\t")[9:]}
    raise ValueError(f"no #CHROM header in {path}")


def g2f():
    p = pd.read_parquet(DATA / "processed" / "g2f" / "pheno.parquet", columns=["env", "year", "genotype", "y", "trait"])
    traits = p["trait"].unique()
    assert len(traits) == 1, traits  # the builder keeps grain yield only
    p = p.dropna(subset=["y"])  # builder drops hybrids without genotype calls
    return p.rename(columns={"genotype": "k"})[["env", "year", "k"]]


def nust():
    norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())  # atlas.load_nust
    ids = vcf_ids(GP / "raw/nust/NUST_geno_Wm82.a2_v1_Filt_KNNimp.vcf.gz", norm)
    ph = pd.read_csv(GP / "raw/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz", low_memory=False)
    ph = ph[ph.Phenotype == "YieldBuA"].copy()
    ph["y"] = pd.to_numeric(ph.Value, errors="coerce")
    ph = ph.dropna(subset=["y"])
    ph["year"] = pd.to_numeric(ph.Experiment.str.extract(r"_(\d{4})$")[0], errors="coerce")
    ph = ph.dropna(subset=["year"])
    ph["env"] = ph.Location.astype(str) + "_" + ph.year.astype(int).astype(str)
    ph["k"] = ph.GermplasmId.map(norm)
    ph = ph[ph.k.isin(ids)]
    return ph[["env", "year", "k"]].drop_duplicates()


def ursn():
    # header has no name for the row-name column, so read it as panel_ursn.py does
    g = pd.read_csv(GP / "raw/ursn/data/URSN_3K_1968-2024_formatted_imputed_005mafFilt_gdose.tsv", sep="\t", index_col=0)
    ids = set(g.index.astype(str))
    ph = pd.read_csv(GP / "raw/ursn/data/URSN_cleaned_table_phenot_1995_2024.tsv", sep="\t").dropna(subset=["DIS"])
    ph = ph.rename(columns={"env_code": "env", "line": "k"})
    ph["k"] = ph.k.astype(str)
    return ph[ph.k.isin(ids)][["env", "year", "k"]].drop_duplicates()


def vef():
    ids = vcf_ids(GP / "raw/vef/VEF_Pvulgaris_genotypic_data.vcf.gz")
    ph = pd.read_csv(GP / "raw/vef/VEF_BLUE_data.csv")[["Trial", "Line", "Yd_BLUE"]].dropna()
    ph = ph.rename(columns={"Trial": "env", "Line": "k"})
    ph["year"] = 2000 + ph.env.str.slice(3, 5).astype(int)  # Pal13C_drt -> 2013
    return ph[ph.k.isin(ids)][["env", "year", "k"]].drop_duplicates()


def processed(name):
    """phenotype.parquet: genotype_id, environment_id, year, trait_id, phenotype_value (SoyNAM plot level)."""
    d = GP / "processed" / name
    ph = pd.read_parquet(d / "phenotype.parquet", columns=["genotype_id", "environment_id", "year", "trait_id", "phenotype_value"])
    ph = ph[ph.trait_id == "yield"].dropna(subset=["phenotype_value"])
    ids = set(pd.read_parquet(d / "genotype.parquet", columns=["genotype_id"]).genotype_id.astype(str).unique())
    ph = ph.rename(columns={"environment_id": "env", "genotype_id": "k"})
    ph["k"] = ph.k.astype(str)
    return ph[ph.k.isin(ids)][["env", "year", "k"]].drop_duplicates()


def soynam():
    return processed("soynam")


def drops():
    return processed("drops")


def count(name, cells):
    cells = cells.assign(year=cells.year.astype(int))
    per_env = cells.groupby(["year", "env"]).k.nunique().rename("n").reset_index()
    first_year = cells.groupby("k").year.min()  # first year with any record, any environment size
    rows = []
    for tag, thr in (("main", MIN_MAIN.get(name, 25)), ("relaxed", MIN_RELAXED)):
        big = per_env[per_env.n >= thr]
        years_with_big = sorted(big.year.unique())
        for yr in sorted(cells.year.unique()):
            envs = big[big.year == yr].env
            lines = cells[(cells.year == yr) & cells.env.isin(envs)].k.unique()
            prior = [y for y in years_with_big if y < yr]
            new = float((first_year.loc[lines] >= yr).mean()) if len(lines) else float("nan")
            n_cells = int(big[big.year == yr].n.sum())  # descriptive: cells in the qualifying environments
            ok = len(prior) >= MIN_PRIOR_YEARS and len(envs) >= MIN_ENVS and len(lines) > 0 and new >= NEW_SHARE
            rows.append({"dataset": name, "rule": tag, "min_n": thr, "year": yr,
                         "envs_total": int((per_env.year == yr).sum()), "envs_ge_min": int(len(envs)),
                         "median_n": float(per_env[per_env.year == yr].n.median()),
                         "lines": int(len(lines)), "cells": n_cells,
                         "resolution": round(3 / n_cells ** 0.5, 3) if n_cells else float("nan"),
                         "new_share": round(new, 3), "prior_years": len(prior),
                         "qualifies": bool(ok)})
    return rows


def cubic_structure():
    cols = pd.read_csv(GP / "raw/cubic/TableS12_phenotypes.csv", nrows=0).columns
    sites = sorted({c.rsplit("_", 1)[1] for c in cols if "_" in c})
    return {"columns": len(cols), "site_suffixes": sites,
            "year_column": any("year" in c.lower() for c in cols),
            "note": "one value per hybrid per site; no year field, so no forward year can be formed"}


t0 = time.time()
rows = []
for name, fn in (("G2F", g2f), ("NUST", nust), ("URSN", ursn), ("SoyNAM", soynam), ("DROPS", drops), ("VEF", vef)):
    rows += count(name, fn())
    print(name, "done", round(time.time() - t0), "s", flush=True)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "per_year.csv", index=False)

main = tab[(tab.rule == "main") & tab.qualifies]
relaxed = tab[(tab.rule == "relaxed") & tab.qualifies]
K = int(len(main))
verdict = {
    "K_main": K,
    "K_relaxed_sensitivity": int(len(relaxed)),
    "by_dataset_main": {d: sorted(int(y) for y in g.year) for d, g in main.groupby("dataset")},
    "by_dataset_relaxed": {d: sorted(int(y) for y in g.year) for d, g in relaxed.groupby("dataset")},
    "CUBIC": cubic_structure(),
    "rule": f"K >= {K_RULE} -> pre-register B3 and E5; otherwise atlas stacking reported under CV1 only",
    "decision": "pre-register B3/E5" if K >= K_RULE else "no B3/E5",
}
json.dump(verdict, open(OUT / "verdict.json", "w"), indent=1)
commit = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                            capture_output=True, text=True).stdout.strip()
json.dump({"git_commit": commit, "data_root": str(DATA), "gp_data": str(GP), "host": socket.gethostname(),
           "seconds": round(time.time() - t0)}, open(OUT / "run_meta.json", "w"), indent=1)
print(json.dumps(verdict, indent=1))
print("W1C_DONE")
