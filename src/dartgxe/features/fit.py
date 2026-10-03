"""Feature transforms fitted on the training part only (leakage rule: fitted on training data only).

Every fitted object keeps `fit_ids` (the genotypes or environments it was fitted on) so tests can
assert that they all belong to the training role of the fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dartgxe.paths import processed


@dataclass
class GenoData:
    X: np.ndarray            # int8, n_hybrid x p, -1 missing
    ids: list[str]
    index: dict = field(init=False)

    def __post_init__(self):
        self.index = {h: i for i, h in enumerate(self.ids)}


def load_geno(dataset: str = "g2f") -> GenoData:
    d = processed(dataset)
    return GenoData(np.load(d / "geno.npy"), (d / "geno_ids.txt").read_text().split())


@dataclass
class GenoScaler:
    """Mode imputation + centring/scaling, fitted on training hybrids."""
    fit_ids: list[str]
    mode: np.ndarray
    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, geno: GenoData, ids) -> "GenoScaler":
        ids = sorted(set(ids))
        X = geno.X[[geno.index[h] for h in ids]]
        counts = np.stack([(X == k).sum(0) for k in (0, 1, 2)])
        mode = counts.argmax(0).astype(np.float32)
        Xf = np.where(X < 0, mode, X).astype(np.float32)
        mean, sd = Xf.mean(0), Xf.std(0)
        sd[sd < 1e-6] = 1.0  # monomorphic in training: contributes zeros
        return cls(ids, mode, mean, sd)

    def transform(self, geno: GenoData, ids) -> np.ndarray:
        X = geno.X[[geno.index[h] for h in ids]]
        Xf = np.where(X < 0, self.mode, X).astype(np.float32)
        return (Xf - self.mean) / self.sd


@dataclass
class GenoPCA:
    fit_ids: list[str]
    components: np.ndarray  # p x k
    scale: np.ndarray       # singular values / sqrt(n)

    @classmethod
    def fit(cls, Z_train: np.ndarray, ids, k: int = 100) -> "GenoPCA":
        U, S, Vt = np.linalg.svd(Z_train - Z_train.mean(0), full_matrices=False)
        return cls(list(ids), Vt[:k].T.astype(np.float32), (S[:k] / np.sqrt(len(Z_train))).astype(np.float32))

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components


@dataclass
class EnvScaler:
    """EC: impute a missing environment by its location's mean over training envs of other years,
    then by the training median; add a missing indicator; standardise on training envs."""
    fit_ids: list[str]
    cols: list[str]
    loc_mean: pd.DataFrame
    median: pd.Series
    mean: pd.Series
    sd: pd.Series

    @staticmethod
    def location(env: pd.Series) -> pd.Series:
        return env.str.rsplit("_", n=1).str[0]

    @classmethod
    def fit(cls, ec: pd.DataFrame, env_ids, drop_cols=()) -> "EnvScaler":
        cols = [c for c in ec.columns if c not in ("env", "lat", "lon", "ec_missing", *drop_cols)]
        tr = ec[ec["env"].isin(set(env_ids))].copy()
        tr["loc"] = cls.location(tr["env"])
        loc_mean = tr.groupby("loc")[cols].mean()
        median = tr[cols].median()
        filled = cls._fill(tr, cols, loc_mean, median)
        mean, sd = filled.mean(), filled.std().replace(0, 1.0).fillna(1.0)
        return cls(sorted(set(env_ids)), cols, loc_mean, median, mean, sd)

    @staticmethod
    def _fill(df, cols, loc_mean, median):
        x = df[cols].copy()
        loc = EnvScaler.location(df["env"])
        lm = loc_mean.reindex(loc.to_numpy())
        lm.index = x.index
        return x.fillna(lm).fillna(median).fillna(0.0)

    def transform(self, ec: pd.DataFrame, env_ids) -> np.ndarray:
        df = ec.set_index("env").loc[list(env_ids)].reset_index()
        x = self._fill(df, self.cols, self.loc_mean, self.median)
        z = ((x - self.mean) / self.sd).to_numpy(np.float32)
        return np.concatenate([z, df[["ec_missing"]].to_numpy(np.float32)], 1)


def load_ec(dataset: str = "g2f") -> pd.DataFrame:
    return pd.read_parquet(processed(dataset) / "env_ec.parquet")
