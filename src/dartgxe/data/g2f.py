"""Build the unified G2F tables (spec 01 §1.2) from the 2024-competition release.

Inputs (until the raw CyVerse CSVs are downloaded by hand, spec 01 §1.1.1):
  * phenotype + environment: the processed parquet already on the 4090
    (``source_dataset == g2f_competition_2024``), plot level;
  * genotype: the raw VCF ``5_Genotype_Data_All_2014_2025_Hybrids.vcf`` (GT only).
    The processed genotype table is NOT used: its missing calls were already
    filled with an all-hybrid mean, which would put test hybrids' genotypes into
    the imputation (the leakage rule requires imputation fitted on the training fold).

Outputs in ``data/processed/g2f/``:
  pheno_plot.parquet  env, year, location, genotype, y, trait, rep, block, plot
  pheno.parquet       one row per genotype x environment cell (mean of plots):
                      env, year, location, genotype, y, n_plots, trait, parent1, parent2, tester
  geno.npy            int8 alt-allele counts 0/1/2, -1 = missing; rows follow geno_ids.txt
  geno_ids.txt, markers.tsv
  env_ec.parquet      env, lat, lon, ec_missing, <654 APSIM EC columns (build_report.json n_ec_columns)>
  build_report.json   counts, drops and input hashes
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EC_STAGES = ("GerEme", "EmeEnJ", "EnJFlo", "FloFla", "FlaFlw", "FlwStG", "StGEnG", "EnGMat", "MatHar")
# Stage-wise EC columns end in "_p<Stage>" or, for soil layers, "_p<Stage>_<layer>";
# "LL__<layer>" (lower limit of plant-available water per soil layer) is static per environment.
EC_PATTERN = re.compile(r"(_p(" + "|".join(EC_STAGES) + r")(_\d+)?$)|(^LL__\d+$)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_vcf_gt(path: Path) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Parse a GT-only VCF into an (n_samples x n_markers) int8 matrix of alt-allele counts."""
    ids: list[str] = []
    rows, markers = [], []
    code = {"0/0": 0, "0/1": 1, "1/0": 1, "1/1": 2, "0|0": 0, "0|1": 1, "1|0": 1, "1|1": 2}
    with open(path) as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                ids = line.rstrip("\n").split("\t")[9:]
                continue
            parts = line.rstrip("\n").split("\t")
            fmt = parts[8].split(":")
            gi = fmt.index("GT")
            gts = [s.split(":")[gi] for s in parts[9:]]
            rows.append(np.array([code.get(g, -1) for g in gts], dtype=np.int8))
            markers.append((parts[2], parts[0], int(parts[1]), parts[3], parts[4]))
    geno = np.stack(rows, axis=1)  # samples x markers
    mk = pd.DataFrame(markers, columns=["marker", "chrom", "pos", "ref", "alt"])
    return geno, ids, mk


def geno_from_processed(path: Path) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Recover raw calls from the processed long genotype table.

    That table stores dosage on a 0/0.5/1 scale and filled missing calls with a
    marker mean over all hybrids (a fractional value). Values in {0, 0.5, 1} are
    calls (-> 0/1/2); anything else was a missing call (-> -1), so imputation can be
    redone on the training fold only. A missing call whose all-hybrid mean happened
    to be exactly 0.5 or 1.0 cannot be told apart; `verify_against_vcf` measures this.
    """
    g = pd.read_parquet(path, columns=["genotype_id", "marker_id", "allele_dosage"])
    wide = g.pivot(index="genotype_id", columns="marker_id", values="allele_dosage")
    mk = pd.DataFrame({"marker": wide.columns})
    mk["chrom"] = mk["marker"].str.extract(r"^S(\d+)_")[0].astype(int)
    mk["pos"] = mk["marker"].str.extract(r"_(\d+)$")[0].astype(int)
    mk = mk.sort_values(["chrom", "pos"]).reset_index(drop=True)
    v = wide[mk["marker"]].to_numpy(dtype=np.float64)
    out = np.full(v.shape, -1, dtype=np.int8)
    for val, code in ((0.0, 0), (0.5, 1), (1.0, 2)):
        out[np.isclose(v, val, atol=1e-6)] = code
    return out, [str(i) for i in wide.index], mk


def verify_against_vcf(geno: np.ndarray, ids: list[str], markers: pd.DataFrame, vcf: Path) -> dict:
    """Compare recovered calls with whatever complete marker lines a (possibly partial) raw VCF holds.

    Coding note: the VCF counts the ALT allele; the processed table may count the other
    allele, so each marker is compared under both orientations and the better one kept.
    """
    col = {m: j for j, m in enumerate(markers["marker"])}
    row = {h: i for i, h in enumerate(ids)}
    code = {"0/0": 0, "0/1": 1, "1/0": 1, "1/1": 2}
    n_mk = agree = total = miss_match = miss_total = 0
    vcf_ids: list[str] = []
    with open(vcf, errors="replace") as f:
        for line in f:
            if line.startswith("#CHROM"):
                vcf_ids = line.rstrip("\n").split("\t")[9:]
                continue
            if line.startswith("#") or not line.endswith("\n"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9 + len(vcf_ids) or parts[2] not in col:
                continue
            j = col[parts[2]]
            raw = np.array([code.get(s.split(":")[0], -1) for s in parts[9:]], dtype=np.int8)
            idx = np.array([row.get(h, -1) for h in vcf_ids])
            ok = idx >= 0
            rec = geno[idx[ok], j]
            r = raw[ok]
            called = (r >= 0) & (rec >= 0)
            same = max((rec[called] == r[called]).sum(), (rec[called] == 2 - r[called]).sum())
            agree += same
            total += called.sum()
            miss_match += ((r < 0) & (rec < 0)).sum()
            miss_total += (r < 0).sum()
            n_mk += 1
    return {"markers_checked": n_mk, "call_agreement": float(agree / max(total, 1)),
            "raw_missing_recovered_as_missing": float(miss_match / max(miss_total, 1)),
            "raw_missing_calls": int(miss_total)}


def parse_parents(genotype: pd.Series) -> pd.DataFrame:
    # G2F hybrid names are "PARENT1/PARENT2"; parent2 is the tester (LH195, LH244, PHZ51, ...
    # dominate parent2 in the 2014-2023 data, checked 2026-09-25). A name without "/" keeps
    # itself as parent1 and gets no tester.
    parts = genotype.str.split("/", n=1, expand=True)
    p1 = parts[0]
    p2 = parts[1] if parts.shape[1] > 1 else pd.Series([None] * len(genotype), index=genotype.index)
    return pd.DataFrame({"parent1": p1, "parent2": p2, "tester": p2})


def build(pheno_src: Path, env_src: Path, vcf: Path | None, out: Path, geno_src: Path | None = None,
          vcf_partial: Path | None = None) -> dict:
    """Use the raw VCF when given; otherwise recover calls from the processed genotype table."""
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"inputs": {}}
    srcs = [("phenotype", pheno_src), ("environment", env_src), ("vcf", vcf), ("genotype_processed", geno_src)]
    for name, path in srcs:
        if path is not None:
            report["inputs"][name] = {"path": str(path), "sha256": sha256(path)}

    # ---- genotype ------------------------------------------------------------------------
    if vcf is not None:
        geno, ids, markers = read_vcf_gt(vcf)
        report["geno_source"] = "raw VCF"
    else:
        geno, ids, markers = geno_from_processed(geno_src)
        report["geno_source"] = "processed table, missing calls recovered from fractional imputed values"
        if vcf_partial is not None:
            report["geno_check_vs_raw_vcf"] = verify_against_vcf(geno, ids, markers, vcf_partial)
    report["geno"] = {
        "n_hybrids": len(ids),
        "n_markers": int(geno.shape[1]),
        "missing_rate": float((geno < 0).mean()),
        "dup_ids": int(len(ids) - len(set(ids))),
    }
    np.save(out / "geno.npy", geno)
    (out / "geno_ids.txt").write_text("\n".join(ids) + "\n")
    markers.to_csv(out / "markers.tsv", sep="\t", index=False)

    # ---- phenotype -----------------------------------------------------------------------
    p = pd.read_parquet(pheno_src)
    assert (p["trait_id"] == "yield").all(), "only grain yield is expected in this release"
    plot = pd.DataFrame(
        {
            "env": p["environment_id"].astype(str),
            "year": p["year"].astype(int),
            "location": p["location_id"].astype(str),
            "genotype": p["genotype_id"].astype(str),
            "y": p["phenotype_value"].astype(float),
            "trait": "yield_Mg_ha",
            "rep": p["replicate_id"],
            "block": p["block_id"],
            "plot": p["plot_id"],
        }
    )
    n_plot_all = len(plot)
    plot = plot[np.isfinite(plot["y"])]
    report["pheno"] = {"plots_in_source": n_plot_all, "plots_finite_y": len(plot)}

    genotyped = set(ids)
    miss = sorted(set(plot["genotype"]) - genotyped)
    report["pheno"]["hybrids_without_genotype"] = len(miss)
    report["pheno"]["plots_dropped_no_genotype"] = int(plot["genotype"].isin(miss).sum())
    plot = plot[~plot["genotype"].isin(miss)].reset_index(drop=True)
    plot.to_parquet(out / "pheno_plot.parquet", index=False)

    cell = (
        plot.groupby(["env", "year", "location", "genotype", "trait"], as_index=False)
        .agg(y=("y", "mean"), n_plots=("y", "size"))
    )
    cell = pd.concat([cell, parse_parents(cell["genotype"])], axis=1)
    cell = cell[["env", "year", "location", "genotype", "y", "n_plots", "trait", "parent1", "parent2", "tester"]]
    cell.to_parquet(out / "pheno.parquet", index=False)
    per_env = cell.groupby("env").size()
    report["pheno"].update(
        {
            "cells": len(cell),
            "envs": int(cell["env"].nunique()),
            "hybrids": int(cell["genotype"].nunique()),
            "years": {int(k): int(v) for k, v in cell.groupby("year").size().items()},
            "hybrids_per_env_median": float(per_env.median()),
            "envs_below_25": int((per_env < 25).sum()),
        }
    )

    # ---- environment covariates (official APSIM EC, spec 01 §1.3) --------------------------
    e = pd.read_parquet(env_src)
    ec_cols = [c for c in e.columns if EC_PATTERN.search(c)]
    env = pd.DataFrame({"env": e["environment_id"].astype(str)})
    env["lat"] = pd.to_numeric(e["weather_station_latitude_in_decimal_numbers_not_dms"], errors="coerce")
    env["lon"] = pd.to_numeric(e["weather_station_longitude_in_decimal_numbers_not_dms"], errors="coerce")
    ec = e[ec_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    env["ec_missing"] = ec.isna().all(axis=1).astype("int8")
    env = pd.concat([env, ec.reset_index(drop=True)], axis=1)
    env = env[env["env"].isin(set(cell["env"]))].reset_index(drop=True)
    env.to_parquet(out / "env_ec.parquet", index=False)
    report["env"] = {
        "n_env": len(env),
        "n_ec_columns": len(ec_cols),
        "envs_with_ec": int((env["ec_missing"] == 0).sum()),
        "apsim_simulated_yield_columns": [c for c in ec_cols if c.startswith("yield_")],
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------ v2: build from the raw CyVerse files
def build_raw(raw: Path, out: Path) -> dict:
    """Build every table from the G2F 2024-competition raw files (doi:10.25739/78mn-4394):
    1_Training_Trait_Data_2014_2023.csv (plots), 7_Testing_Observed_Values.csv (2024, one value per
    env x hybrid, as released), 5_Genotype_Data_All_2014_2025_Hybrids.vcf, 6_*_EC_Data_*.csv,
    4_*_Weather_Data_*_full_year.csv (daily NASA POWER). Same outputs as `build`, plus env_daily.parquet.
    Note: 2014-2023 cell values are plot means; 2024 values are the organisers' released values
    (their aggregation is not documented in the files); only within-environment ranks of the 2024
    test year are used, so the difference cannot leak across years."""
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "trait": raw / "1_Training_Trait_Data_2014_2023.csv",
        "obs2024": raw / "7_Testing_Observed_Values.csv",
        "vcf": raw / "5_Genotype_Data_All_2014_2025_Hybrids.vcf",
        "ec_train": raw / "6_Training_EC_Data_2014_2023.csv",
        "ec_test": raw / "6_Testing_EC_Data_2024.csv",
        "w_train": raw / "4_Training_Weather_Data_2014_2023_full_year.csv",
        "w_test": raw / "4_Testing_Weather_Data_2024_full_year.csv",
    }
    report: dict = {"inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in files.items()}}

    geno, ids, markers = read_vcf_gt(files["vcf"])
    np.save(out / "geno.npy", geno)
    (out / "geno_ids.txt").write_text("\n".join(ids) + "\n")
    markers.to_csv(out / "markers.tsv", sep="\t", index=False)
    report["geno"] = {"n_hybrids": len(ids), "n_markers": int(geno.shape[1]), "missing_rate": float((geno < 0).mean())}

    t = pd.read_csv(files["trait"], low_memory=False)
    plot = pd.DataFrame({"env": t["Env"].astype(str), "year": t["Year"].astype(int),
                         "location": t["Field_Location"].astype(str), "genotype": t["Hybrid"].astype(str),
                         "y": pd.to_numeric(t["Yield_Mg_ha"], errors="coerce"), "trait": "yield_Mg_ha",
                         "rep": t["Replicate"], "block": t["Block"], "plot": t["Plot"],
                         "date_planted": t["Date_Planted"]})
    report["pheno"] = {"plots_raw": len(plot)}
    plot = plot[np.isfinite(plot["y"])]
    genotyped = set(ids)
    report["pheno"]["plots_finite_y"] = len(plot)
    report["pheno"]["plots_dropped_no_genotype"] = int((~plot["genotype"].isin(genotyped)).sum())
    plot = plot[plot["genotype"].isin(genotyped)].reset_index(drop=True)
    plot.to_parquet(out / "pheno_plot.parquet", index=False)
    cell = (plot.groupby(["env", "year", "location", "genotype", "trait"], as_index=False)
            .agg(y=("y", "mean"), n_plots=("y", "size")))
    o = pd.read_csv(files["obs2024"])
    o = o[np.isfinite(pd.to_numeric(o["Yield_Mg_ha"], errors="coerce"))]
    c24 = pd.DataFrame({"env": o["Env"].astype(str), "year": 2024,
                        "location": o["Env"].astype(str).str.rsplit("_", n=1).str[0], "genotype": o["Hybrid"].astype(str),
                        "trait": "yield_Mg_ha", "y": o["Yield_Mg_ha"].astype(float), "n_plots": np.nan})
    report["pheno"]["cells_2024_released"] = len(c24)
    report["pheno"]["cells_2024_no_genotype"] = int((~c24["genotype"].isin(genotyped)).sum())
    c24 = c24[c24["genotype"].isin(genotyped)]
    cell = pd.concat([cell, c24], ignore_index=True)
    cell = pd.concat([cell, parse_parents(cell["genotype"])], axis=1)
    cell = cell[["env", "year", "location", "genotype", "y", "n_plots", "trait", "parent1", "parent2", "tester"]]
    cell.to_parquet(out / "pheno.parquet", index=False)
    per_env = cell.groupby("env").size()
    report["pheno"].update({"cells": len(cell), "envs": int(cell["env"].nunique()), "hybrids": int(cell["genotype"].nunique()),
                            "years": {int(k): int(v) for k, v in cell.groupby("year").size().items()},
                            "envs_below_25": int((per_env < 25).sum())})

    ec = pd.concat([pd.read_csv(files["ec_train"]), pd.read_csv(files["ec_test"])], ignore_index=True)
    ec_cols = [c for c in ec.columns if EC_PATTERN.search(c)]
    env = pd.DataFrame({"env": sorted(cell["env"].unique())})
    e2 = ec.rename(columns={"Env": "env"})[["env", *ec_cols]]
    env = env.merge(e2, on="env", how="left")
    env[ec_cols] = env[ec_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    env.insert(1, "ec_missing", env[ec_cols].isna().all(axis=1).astype("int8"))
    env.insert(1, "lat", np.nan)
    env.insert(2, "lon", np.nan)
    env.to_parquet(out / "env_ec.parquet", index=False)
    report["env"] = {"n_env": len(env), "n_ec_columns": len(ec_cols), "envs_with_ec": int((env["ec_missing"] == 0).sum())}

    w = pd.concat([pd.read_csv(files["w_train"]), pd.read_csv(files["w_test"])], ignore_index=True)
    w = w.rename(columns={"Env": "env", "Date": "date"})
    w["date"] = pd.to_datetime(w["date"].astype(str), format="%Y%m%d")
    w = w[w["env"].isin(set(cell["env"]))]
    w.to_parquet(out / "env_daily.parquet", index=False)
    report["weather"] = {"envs_with_daily": int(w["env"].nunique()), "variables": [c for c in w.columns if c not in ("env", "date")],
                         "days_per_env_median": float(w.groupby("env").size().median())}
    (out / "build_report.json").write_text(json.dumps(report, indent=1))
    return report
