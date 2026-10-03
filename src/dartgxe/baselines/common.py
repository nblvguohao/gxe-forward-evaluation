"""Shared plumbing for baselines: fold assembly, validation scoring, prediction files, run metadata."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dartgxe.eval.metrics import per_env
from dartgxe.paths import REPO, RESULTS, processed, splits_dir

VAL_MIN_N = 10  # tuning only; reported metrics use MIN_N = 25


@dataclass
class Fold:
    scenario: str
    fold: int
    cells: pd.DataFrame  # env, year, genotype, y, role

    def rows(self, *roles) -> pd.DataFrame:
        return self.cells[self.cells["role"].isin(roles)].reset_index(drop=True)

    @property
    def forward(self) -> bool:
        return (self.cells["role"] == "refit_extra").any() or self.scenario.startswith("F")

    def fit_roles(self, final: bool) -> tuple[str, ...]:
        """Tuning always fits on train. The final fit adds val (+refit_extra) only in forward scenarios."""
        return ("train", "val", "refit_extra") if (final and self.forward) else ("train",)


def load_folds(scenario: str, dataset: str = "g2f"):
    cells = pd.read_parquet(processed(dataset) / "pheno.parquet")[["env", "year", "genotype", "y"]]
    sp = pd.read_parquet(splits_dir(dataset) / f"{scenario}.parquet")
    for f, s in sp.groupby("fold"):
        c = cells.merge(s[["env", "genotype", "role"]], on=["env", "genotype"], how="inner")
        yield Fold(scenario, int(f), c[c["role"] != "drop"].reset_index(drop=True))


def val_score(rows: pd.DataFrame, pred: np.ndarray) -> float:
    p = pd.DataFrame({"method": "x", "seed": 0, "fold": 0, "env": rows["env"].to_numpy(),
                      "genotype": rows["genotype"].to_numpy(), "observed": rows["y"].to_numpy(), "prediction": pred})
    pe = per_env(p, min_n=VAL_MIN_N)
    return float(pe.loc[pe["metric"] == "spearman", "value"].mean()) if len(pe) else float("nan")


def pred_frame(rows: pd.DataFrame, pred: np.ndarray, method: str, scenario: str, fold: int, seed: int) -> pd.DataFrame:
    return pd.DataFrame({"env": rows["env"].to_numpy(), "genotype": rows["genotype"].to_numpy(),
                         "observed": rows["y"].to_numpy(), "prediction": pred.astype(float),
                         "method": method, "scenario": scenario, "fold": fold, "seed": seed})


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        s = subprocess.check_output(["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "status", "--porcelain",
                                     "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL)
        return bool(s.strip())
    except Exception:
        return True


def write_predictions(frames: list[pd.DataFrame], method: str, scenario: str, seeds, meta: dict,
                      dataset: str = "g2f") -> None:
    out = RESULTS / "predictions" / dataset / scenario / method
    out.mkdir(parents=True, exist_ok=True)
    allp = pd.concat(frames, ignore_index=True)
    for s in seeds:
        d = allp[allp["seed"] == s] if (allp["seed"] == s).any() else allp.assign(seed=s)
        d.to_parquet(out / f"seed{s}.parquet", index=False)
    meta = {"method": method, "scenario": scenario, "dataset": dataset, "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "host": platform.node(), "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


def within_center(values: np.ndarray, env: np.ndarray) -> tuple[np.ndarray, dict]:
    """Subtract per-environment means (fixed environment effects, Frisch-Waugh-Lovell)."""
    df = pd.DataFrame(values if values.ndim == 2 else values[:, None])
    means = df.groupby(env).transform("mean").to_numpy()
    out = (df.to_numpy() - means)
    return (out if values.ndim == 2 else out[:, 0]), None
