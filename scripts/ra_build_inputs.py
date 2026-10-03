"""RA (docs/prereg_reanalysis_published_2026-09-30.md §2.1): build competition-format inputs for a target year Y
from the G2F 2024 release, with the 2022-release file names the CLAC script reads.

Training_Data = every table restricted to years < Y (phenotypes, metadata, soil, weather, EC); the VCF holds
genotypes only (no labels) and is linked unchanged. Testing_Data = the Y environments' metadata, soil, weather and
EC, and a submission template listing the benchmark scored cells (env, hybrid) of Y with an empty yield column.
No phenotype of year Y or later is written anywhere.

  python scripts/ra_build_inputs.py 2016 2018 2020 2022 2024

Environment: RAW (data/raw/g2f2024), BENCH (benchmark dir with cells_G2F.parquet, splits.json), OUT (ra root)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

RAW, BENCH, OUT = Path(os.environ["RAW"]), Path(os.environ["BENCH"]), Path(os.environ["OUT"])
TRAIT = "1_Training_Trait_Data_2014_2023.csv"
META_TR, META_TE = "2_Training_Meta_Data_2014_2023.csv", "2_Testing_Meta_Data_2024.csv"
SOIL_TR, SOIL_TE = "3_Training_Soil_Data_2015_2023.csv", "3_Testing_Soil_Data_2024.csv"
WEAT_TR, WEAT_TE = "4_Training_Weather_Data_2014_2023_seasons_only.csv", "4_Testing_Weather_Data_2024_seasons_only.csv"
EC_TR, EC_TE = "6_Training_EC_Data_2014_2023.csv", "6_Testing_EC_Data_2024.csv"
VCF = "5_Genotype_Data_All_2014_2025_Hybrids.vcf"
# names the CLAC script reads (2022 release)
N = {"trait": "1_Training_Trait_Data_2014_2021.csv", "meta": "2_Training_Meta_Data_2014_2021.csv",
     "soil": "3_Training_Soil_Data_2015_2021.csv", "weat": "4_Training_Weather_Data_2014_2021.csv",
     "ec": "6_Training_EC_Data_2014_2021.csv", "vcf": "5_Genotype_Data_All_Years.vcf",
     "tpl": "1_Submission_Template_2022.csv", "tmeta": "2_Testing_Meta_Data_2022.csv",
     "tsoil": "3_Testing_Soil_Data_2022.csv", "tweat": "4_Testing_Weather_Data_2022.csv",
     "tec": "6_Testing_EC_Data_2022.csv"}


def env_year(env: pd.Series) -> pd.Series:
    return env.astype(str).str.rsplit("_", n=1).str[-1].astype(int)


def both(tr: str, te: str) -> pd.DataFrame:
    a = pd.read_csv(RAW / tr, low_memory=False)
    b = pd.read_csv(RAW / te, low_memory=False) if (RAW / te).exists() else a.iloc[:0]
    return pd.concat([a, b], ignore_index=True)


def build(Y: int, trait: pd.DataFrame, meta: pd.DataFrame, soil: pd.DataFrame, weat: pd.DataFrame, ec: pd.DataFrame,
          cells: pd.DataFrame, spl: dict) -> dict:
    d = OUT / f"Y{Y}"
    tr, te = d / "Training_Data", d / "Testing_Data"
    tr.mkdir(parents=True, exist_ok=True)
    te.mkdir(parents=True, exist_ok=True)
    # training: years < Y only
    t = trait[trait["Year"] < Y]
    assert t["Year"].max() < Y
    t.to_csv(tr / N["trait"], index=False)
    meta[meta["Year"] < Y].to_csv(tr / N["meta"], index=False)
    soil[soil["Year"] < Y].to_csv(tr / N["soil"], index=False)
    weat[env_year(weat["Env"]) < Y].to_csv(tr / N["weat"], index=False)
    # RA patch 3: the 2024-release ECs contain missing values (environments without a weather/APSIM run, a few partial
    # columns) that the 2022-release files did not. Drop all-missing environments, then any column still missing in
    # the training-year or target environments. Uses EC values only (no yield); recorded in build_info.json.
    envs_t = spl["scorable_environments"].get(str(Y), [])
    ec_use = ec[(env_year(ec["Env"]) < Y) | ec["Env"].isin(envs_t)]
    feat = [c for c in ec.columns if c != "Env"]
    ec_use = ec_use[ec_use[feat].notna().any(axis=1)]
    keep = [c for c in feat if ec_use[c].notna().all()]
    ec = ec[["Env"] + keep]
    ec[env_year(ec["Env"]) < Y].dropna().to_csv(tr / N["ec"], index=False)
    # the 2022 release VCF had no '##' meta lines and CLAC reads it with read_tsv: strip them (genotypes unchanged)
    stripped = OUT / "vcf_no_meta.vcf"
    if not stripped.exists():
        with open(RAW / VCF) as fin, open(stripped, "w") as fout:
            for line in fin:
                if not line.startswith("##"):
                    fout.write(line)
    if (tr / N["vcf"]).is_symlink() or (tr / N["vcf"]).exists():
        (tr / N["vcf"]).unlink()
    os.symlink(stripped, tr / N["vcf"])
    # testing: Y environments, no phenotype
    if str(Y) in spl["scorable_environments"]:
        envs = spl["scorable_environments"][str(Y)]
    else:   # feasibility rehearsal year (never scored): environments with >= 25 genotyped cells
        n = cells[cells["year"] == Y].groupby("env").size()
        envs = sorted(n[n >= 25].index)
    tpl = cells[(cells["year"] == Y) & cells["env"].isin(envs)][["env", "genotype"]]
    tpl = tpl.rename(columns={"env": "Env", "genotype": "Hybrid"}).assign(Yield_Mg_ha="")
    tpl.to_csv(te / N["tpl"], index=False)
    m = meta[meta["Env"].isin(envs)].copy()
    m["Irrigated"] = m["Treatment"].fillna("").str.contains("Irr").map({True: "yes", False: "no"})   # CLAC rule
    m.to_csv(te / N["tmeta"], index=False)
    soil[soil["Env"].isin(envs)].to_csv(te / N["tsoil"], index=False)
    weat[weat["Env"].isin(envs)].to_csv(te / N["tweat"], index=False)
    ec[ec["Env"].isin(envs)].dropna().to_csv(te / N["tec"], index=False)
    info = {"target": Y, "train_plots": len(t), "train_years": sorted(t["Year"].unique().tolist()),
            "test_envs": len(envs), "test_cells": len(tpl), "test_envs_with_ec": int(ec["Env"].isin(envs).sum()),
            "test_envs_with_meta": int(len(m)), "ec_columns_kept": len(keep), "ec_columns_dropped": len(feat) - len(keep)}
    json.dump(info, open(d / "build_info.json", "w"), indent=1)
    return info


def main(years):
    trait = pd.read_csv(RAW / TRAIT, low_memory=False)
    meta = both(META_TR, META_TE)
    soil = both(SOIL_TR, SOIL_TE)
    weat = both(WEAT_TR, WEAT_TE)
    ec = both(EC_TR, EC_TE)
    cells = pd.read_parquet(BENCH / "cells_G2F.parquet")
    spl = json.load(open(BENCH / "splits.json"))["G2F"]
    for Y in years:
        print(build(int(Y), trait, meta, soil, weat, ec, cells, spl), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
