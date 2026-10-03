"""Split generators (spec 01 §1.4). Seed 2026.

Every split is a long table with one row per (fold, cell):
    env, genotype, year, fold, role
role ∈ {train, val, test, refit_extra, drop}.
  * CV scenarios: models are fitted on `train`, early-stopped / tuned on `val`, scored on `test`
    (no refit; the same protocol for every method).
  * Forward scenarios: tuned with train -> val, then refitted on train ∪ val ∪ refit_extra with the
    selected hyperparameters (and the selected epoch count), scored once on `test`.
  * `drop`: cells that belong to no role in that fold (CV00 cross cells).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 2026
K = 5


def _groups(keys: np.ndarray, k: int, rng: np.random.Generator) -> dict:
    """Random balanced assignment of unique keys to k groups."""
    keys = np.array(sorted(set(keys)))
    perm = rng.permutation(len(keys))
    return dict(zip(keys[perm], np.arange(len(keys)) % k))


def _frame(cells: pd.DataFrame, fold: int, role: np.ndarray) -> pd.DataFrame:
    out = cells[["env", "genotype", "year"]].copy()
    out["fold"] = fold
    out["role"] = role
    return out


def cv_by(cells: pd.DataFrame, key: str, seed: int = SEED) -> pd.DataFrame:
    """CV1 (key=genotype), CV1-parent (key=parent1), CV0 (key=env): GroupKFold on `key`,
    inner validation = 1/5 of the training groups, drawn the same way."""
    rng = np.random.default_rng(seed)
    g = cells[key].map(_groups(cells[key].to_numpy(), K, rng)).to_numpy()
    parts = []
    for f in range(K):
        test = g == f
        tr_keys = cells.loc[~test, key].to_numpy()
        inner = _groups(tr_keys, K, np.random.default_rng(seed + 100 + f))
        val = (~test) & (cells[key].map(inner).fillna(-1).to_numpy() == 0)
        role = np.where(test, "test", np.where(val, "val", "train"))
        parts.append(_frame(cells, f, role))
    return pd.concat(parts, ignore_index=True)


def cv00(cells: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """New hybrids x new environments. Fold f test = hybrid group f x env group f;
    train = hybrid not in f and env not in f; cross cells dropped. Inner validation repeats the
    same construction inside the training block (inner block 0)."""
    rng = np.random.default_rng(seed)
    hg = cells["genotype"].map(_groups(cells["genotype"].to_numpy(), K, rng)).to_numpy()
    eg = cells["env"].map(_groups(cells["env"].to_numpy(), K, rng)).to_numpy()
    parts = []
    for f in range(K):
        test = (hg == f) & (eg == f)
        block = (hg != f) & (eg != f)
        irng = np.random.default_rng(seed + 100 + f)
        ih = cells["genotype"].map(_groups(cells.loc[block, "genotype"].to_numpy(), K, irng)).fillna(-1).to_numpy()
        ie = cells["env"].map(_groups(cells.loc[block, "env"].to_numpy(), K, irng)).fillna(-1).to_numpy()
        val = block & (ih == 0) & (ie == 0)
        train = block & (ih != 0) & (ie != 0)
        role = np.select([test, val, train], ["test", "val", "train"], default="drop")
        parts.append(_frame(cells, f, role))
    return pd.concat(parts, ignore_index=True)


def forward(cells: pd.DataFrame, train_end: int, val_year: int, refit_end: int, test_year: int) -> pd.DataFrame:
    y = cells["year"].to_numpy()
    assert train_end < val_year <= refit_end < test_year
    role = np.select(
        [y <= train_end, y == val_year, (y > train_end) & (y <= refit_end) & (y != val_year), y == test_year],
        ["train", "val", "refit_extra", "test"],
        default="drop",
    )
    return _frame(cells, 0, role)


# Forward scenarios. G2F hybrids rotate every two years (spec 01 §1.1.1), so the "m" versions and
# the FY families keep the validation year of the same kind as the test year.
FORWARD = {
    "F2022": (2020, 2021, 2021, 2022),   # original kit definition (validation year of repeat kind)
    "F2022m": (2019, 2020, 2021, 2022),
    "F2024": (2022, 2023, 2023, 2024),   # 2024 observed values released on CyVerse (7_Testing_Observed_Values.csv)
    "F2024m": (2021, 2022, 2023, 2024),  # needs 2024 observed values
    **{f"FYnew{t}": (t - 3, t - 2, t - 1, t) for t in (2018, 2020, 2022)},
    **{f"FYrep{t}": (t - 3, t - 2, t - 1, t) for t in (2019, 2021, 2023)},
    # shortened window (no data before 2014); added 2026-09-26 for docs/prestudy_nn_vs_gblup
    "FYnew2016s": (2014, 2015, 2015, 2016),
}


def make_all(cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {
        "CV1": cv_by(cells, "genotype"),
        "CV1parent": cv_by(cells, "parent1"),
        "CV0": cv_by(cells, "env"),
        "CV00": cv00(cells),
    }
    years = set(cells["year"])
    for name, (a, b, c, t) in FORWARD.items():
        if t in years:
            out[name] = forward(cells, a, b, c, t)
    return out


def novelty(split: pd.DataFrame, role: str, reference_roles: tuple[str, ...]) -> float:
    """Share of `role` cells whose genotype never appears in `reference_roles` (same fold)."""
    ref = set(split.loc[split["role"].isin(reference_roles), "genotype"])
    r = split.loc[split["role"] == role, "genotype"]
    return float((~r.isin(ref)).mean()) if len(r) else float("nan")
