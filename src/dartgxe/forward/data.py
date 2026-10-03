"""Datasets for the forward panels: cells (env, year, genotype, y) and a marker table (genotype x marker, float,
NaN = missing). G2F from the project data root; NUST and URSN from the read-only copy of gp_project/data
(GP_DATA, SHA256SUMS alongside) with the atlas definitions (gp_project/analysis/atlas/atlas.py).

Marker codes differ by source and are handled per dataset: G2F geno.npy is 0/1/2 with -1 = missing;
NUST is a VCF (0/0, 0/1, 1/1; ./. = missing); URSN dosage is -1/0/1 with no missing codes (already imputed),
so its -1 is a genotype, not a missing value.

Wave 2e (independent confirmation data, NEWDATA = the read-only download of 2026-09-29, SHA256SUMS alongside):
ESWYT (CIMMYT, Juliana et al. 2020) HapMap nucleotides (first listed allele 0, second 2, IUPAC heterozygote 1,
N = missing); MU_SOY (Vieira et al. 2022) 0/1/2; GEM_IA (Rogers et al. 2022) has only a genomic relationship
matrix, so its "markers" are the kernel embedding U·sqrt(Λ) of that matrix with markers.attrs["no_scale"] = True
(panel.features then centres without scaling, which keeps ridge on the embedding equal to GBLUP with that GRM)."""
from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dartgxe.features.fit import load_geno
from dartgxe.paths import processed

MIN_N = {"G2F": 25, "NUST": 25, "URSN": 10, "ESWYT": 25, "GEM_IA": 25, "MU_SOY": 25}  # W1c operating rules


@dataclass
class Dataset:
    name: str
    cells: pd.DataFrame      # env, year, genotype, y (genotyped lines only)
    markers: pd.DataFrame    # index genotype, float, NaN = missing
    min_n: int

    def scorable_envs(self, years=None) -> list[str]:
        c = self.cells if years is None else self.cells[self.cells["year"].isin(years)]
        n = c.groupby("env")["genotype"].nunique()
        return sorted(n[n >= self.min_n].index)


def _gp() -> Path:
    return Path(os.environ["GP_DATA"])


def load_g2f() -> Dataset:
    p = pd.read_parquet(processed("g2f") / "pheno.parquet", columns=["env", "year", "genotype", "y"]).dropna(subset=["y"])
    geno = load_geno()
    X = geno.X.astype(np.float32)
    X[X < 0] = np.nan
    m = pd.DataFrame(X, index=pd.Index(geno.ids, name="genotype"))
    p = p[p["genotype"].isin(m.index)].reset_index(drop=True)
    return Dataset("G2F", p.assign(year=p["year"].astype(int)), m, MIN_N["G2F"])


def load_nust() -> Dataset:
    norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())  # atlas.load_nust
    code = {"0/0": 0.0, "0/1": 1.0, "1/0": 1.0, "1/1": 2.0}
    rows, ids = [], None
    with gzip.open(_gp() / "raw/nust/NUST_geno_Wm82.a2_v1_Filt_KNNimp.vcf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            f = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                ids = [norm(s) for s in f[9:]]
                continue
            rows.append([code.get(g.split(":")[0], np.nan) for g in f[9:]])
    X = np.asarray(rows, dtype=np.float32).T
    keep = ~pd.Index(ids).duplicated()
    m = pd.DataFrame(X[keep], index=pd.Index(np.array(ids)[keep], name="genotype"))
    ph = pd.read_csv(_gp() / "raw/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz", low_memory=False)
    ph = ph[ph["Phenotype"] == "YieldBuA"].copy()
    ph["y"] = pd.to_numeric(ph["Value"], errors="coerce")
    ph["year"] = pd.to_numeric(ph["Experiment"].str.extract(r"_(\d{4})$")[0], errors="coerce")
    ph = ph.dropna(subset=["y", "year"])
    ph["year"] = ph["year"].astype(int)
    ph["env"] = ph["Location"].astype(str) + "_" + ph["year"].astype(str)
    ph["genotype"] = ph["GermplasmId"].map(norm)
    ph = ph[ph["genotype"].isin(m.index)]
    cells = ph.groupby(["env", "year", "genotype"], as_index=False)["y"].mean()
    return Dataset("NUST", cells, m, MIN_N["NUST"])


def load_ursn() -> Dataset:
    g = pd.read_csv(_gp() / "raw/ursn/data/URSN_3K_1968-2024_formatted_imputed_005mafFilt_gdose.tsv", sep="\t", index_col=0)
    m = g.astype(np.float32)
    m.index = m.index.astype(str).rename("genotype")
    ph = pd.read_csv(_gp() / "raw/ursn/data/URSN_cleaned_table_phenot_1995_2024.tsv", sep="\t").dropna(subset=["DIS"])
    cells = pd.DataFrame({"env": ph["env_code"].astype(str), "year": ph["year"].astype(int),
                          "genotype": ph["line"].astype(str), "y": 100.0 - ph["DIS"].astype(float)})  # resistance
    cells = cells[cells["genotype"].isin(m.index)]
    cells = cells.groupby(["env", "year", "genotype"], as_index=False)["y"].mean()
    return Dataset("URSN", cells, m, MIN_N["URSN"])


def _nd() -> Path:
    return Path(os.environ["NEWDATA"])


HETS = {"R", "Y", "S", "W", "K", "M"}


def hapmap_codes(calls: np.ndarray, alleles: pd.Series) -> np.ndarray:
    """markers x samples nucleotide calls -> 0 (first allele of 'A/G'), 2 (second), 1 (IUPAC heterozygote), NaN."""
    a1 = alleles.str[0].to_numpy()[:, None]
    a2 = alleles.str[-1].to_numpy()[:, None]
    X = np.full(calls.shape, np.nan, dtype=np.float32)
    X[calls == a1] = 0.0
    X[calls == a2] = 2.0
    X[np.isin(calls, list(HETS))] = 1.0
    return X


def load_eswyt() -> Dataset:
    """ESWYT 24th-37th (Table S3: grain-yield BLUE per line and site). Environment = site x cycle; year = the
    harvest year of the cycle ('2003-2004' -> 2004)."""
    ph = pd.read_csv(_nd() / "eswyt/TableS3_ESWYT_BLUEs.csv")
    ph = ph.melt(id_vars=["Gid", "ESWYT", "Cycle"], var_name="site", value_name="y")
    ph["y"] = pd.to_numeric(ph["y"], errors="coerce")
    ph = ph.dropna(subset=["y"])
    ph["year"] = ph["Cycle"].str.slice(0, 4).astype(int) + 1
    ph["env"] = ph["site"].astype(str) + "_" + ph["year"].astype(str)
    ph["genotype"] = ph["Gid"].astype(str)
    hmp = _nd() / "eswyt/Genotyping_data_24k_14k.hmp.txt"
    cols = pd.read_csv(hmp, sep="\t", nrows=0).columns
    want = [c for c in cols[11:] if c in set(ph["genotype"])]
    g = pd.read_csv(hmp, sep="\t", usecols=["alleles"] + want, dtype=str)
    X = hapmap_codes(g[want].to_numpy(), g["alleles"])
    m = pd.DataFrame(X.T, index=pd.Index(want, name="genotype"))
    cells = ph[ph["genotype"].isin(m.index)].groupby(["env", "year", "genotype"], as_index=False)["y"].mean()
    return Dataset("ESWYT", cells, m, MIN_N["ESWYT"])


def load_mu_soy() -> Dataset:
    """MU-FDREEC advanced yield trials 2017-2021 (Y.csv: environment = year + field; yield bu/ac)."""
    m = pd.read_csv(_nd() / "mu_soy/SNPs.csv", index_col=0).astype(np.float32)
    m.index = m.index.astype(str).rename("genotype")
    ph = pd.read_csv(_nd() / "mu_soy/Y.csv").dropna(subset=["yield"])
    cells = pd.DataFrame({"env": ph["Env"].astype(str), "year": ph["year"].astype(int),
                          "genotype": ph["name"].astype(str), "y": ph["yield"].astype(float)})
    cells = cells[cells["genotype"].isin(m.index)]
    cells = cells.groupby(["env", "year", "genotype"], as_index=False)["y"].mean()
    return Dataset("MU_SOY", cells, m, MIN_N["MU_SOY"])


GEM_DROP = ["CML373/PHJ40//GEMS-0162", "KO679Y/GEMS-0030//4676A", "KO679Y/GEMS-0030//794"]  # File S6 drops these


def gem_grm_key(s: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"^(CTU|08)_", "", str(s))).upper()


def gem_line_key(s: str) -> str:
    """'Converted Pedigree' line part -> GRM key (the '%.' and '@.@.0034.@.' selection-history forms differ)."""
    s = gem_grm_key(s).replace(":%.", ":")
    return re.sub(r":@\.@\.\d+\.@\.//", "//", s)


def load_gem_ia() -> Dataset:
    """GEM Iowa topcross trials 2014-2019 (File S1, plot level). Commercial and GEM checks and the three
    families File S6 drops are removed; line = 'Converted Pedigree' before '+tester'; environment = Loc
    (each Loc is one location-year); year = 2000 + the first two digits of Experiment; cell = mean plot yield
    of the line in the environment (bu/ac). Markers: kernel embedding of the GRM (File S3) over the lines kept."""
    d = pd.read_csv(_nd() / "gem_maize/File_S1_IA_GEM_data.csv", low_memory=False)
    d = d[d["Check"].isna() & ~d["Origin"].isin(GEM_DROP)].dropna(subset=["Yield (bu/ac)"])
    d = d.assign(genotype=d["Converted Pedigree"].str.rsplit("+", n=1).str[0].map(gem_line_key),
                 year=2000 + d["Experiment"].astype(str).str.slice(0, 2).astype(int))
    d["env"] = d["Loc"].astype(str) + "_" + d["year"].astype(str)
    G = pd.read_csv(_nd() / "gem_maize/File_S3_GEM_GRM.txt", sep=r"\s+")
    G.index = [gem_grm_key(s) for s in G.index]
    G.columns = [gem_grm_key(s) for s in G.columns]
    ids = sorted(set(d["genotype"]) & set(G.index))
    K = G.loc[ids, ids].to_numpy(np.float64)
    w, U = np.linalg.eigh((K + K.T) / 2)
    keep = w > 1e-8 * w.max()
    E = U[:, keep] * np.sqrt(w[keep])
    m = pd.DataFrame(E.astype(np.float32), index=pd.Index(ids, name="genotype"))
    m.attrs["no_scale"] = True
    cells = d[d["genotype"].isin(ids)].groupby(["env", "year", "genotype"], as_index=False)["Yield (bu/ac)"].mean()
    cells = cells.rename(columns={"Yield (bu/ac)": "y"})
    return Dataset("GEM_IA", cells, m, MIN_N["GEM_IA"])


LOADERS = {"G2F": load_g2f, "NUST": load_nust, "URSN": load_ursn,
           "ESWYT": load_eswyt, "GEM_IA": load_gem_ia, "MU_SOY": load_mu_soy}
NEW = ["ESWYT", "GEM_IA", "MU_SOY"]  # wave 2e


# ---------------------------------------------------------------- environmental covariates (wave 2c)
def location(env: str) -> str:
    """G2F 'DEH1_2022' -> 'DEH1'; URSN 'BRK_95' -> 'BRK' (its loc_code)."""
    return str(env).rsplit("_", 1)[0]


def load_ec(name: str) -> pd.DataFrame | None:
    """Environment x covariate table (numeric; NaN = missing), index env; None when the dataset has none here.
    G2F: official APSIM ECs of data_v2 (environments flagged ec_missing are dropped); URSN: the stage-wise
    NASA/soil covariates of its enviromics folder; NUST: none (only location names, no coordinates)."""
    if name == "G2F":
        ec = pd.read_parquet(processed("g2f") / "env_ec.parquet")
        ec = ec[ec["ec_missing"] == 0].drop(columns=["lat", "lon", "ec_missing"]).set_index("env")
    elif name == "URSN":
        ec = pd.read_csv(_gp() / "raw/ursn/data/enviromics/envt_covariates_stages_NASA_URSN.tsv", sep="\t")
        ec = ec.drop(columns=["loc_code"]).set_index("env_code")
    else:
        return None
    ec.index = ec.index.astype(str).rename("env")
    return ec.apply(pd.to_numeric, errors="coerce").astype(np.float64)
