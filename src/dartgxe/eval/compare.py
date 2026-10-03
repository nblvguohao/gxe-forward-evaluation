"""Paired comparisons, two-level bootstrap and resolution verdicts (spec 03 §3.3-3.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

B = 2000
RESOLUTION_C = {0.05: 4.3, 0.10: 3.0, 0.20: 2.1}
PRACTICAL = (1.5, 4.2)  # TCJ manuscript: real panels need 1.5-4.2x the Gaussian floor


def paired_delta(mat: pd.DataFrame, a: str, b: str, env_year: pd.Series | None = None,
                 B: int = B, seed: int = 0, level: float = 0.95) -> dict:
    """mat: index (method, env), columns seeds (from metrics.env_seed_matrix).
    Δ_{e,s} = m_{e,s}(a) − m_{e,s}(b), paired on seed. Envs missing either method are dropped.
    Returns the point estimate, single-level env bootstrap CI (seeds averaged first), the two-level
    (env, then seed) CI, seed SD and per-seed sign agreement. With env_year given, the outer
    resampling unit is the year and environments are resampled within year."""
    A, Bm = mat.loc[a], mat.loc[b]
    seeds = sorted(set(A.columns) & set(Bm.columns))
    envs = A.index.intersection(Bm.index)
    D = (A.loc[envs, seeds] - Bm.loc[envs, seeds]).dropna()
    d = D.to_numpy()
    n_e, n_s = d.shape
    est = float(d.mean())
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2
    env_mean = d.mean(1)

    if env_year is None:
        idx = rng.integers(0, n_e, (B, n_e))
        one = env_mean[idx].mean(1)
        sidx = rng.integers(0, n_s, (B, n_s))
        two = np.array([d[idx[i]][:, sidx[i]].mean() for i in range(B)])
    else:
        yr = env_year.reindex(D.index).to_numpy()
        years = np.unique(yr)
        by = [np.flatnonzero(yr == y) for y in years]
        one, two = np.empty(B), np.empty(B)
        for i in range(B):
            pick = np.concatenate([rng.choice(by[j], len(by[j])) for j in rng.integers(0, len(years), len(years))])
            one[i] = env_mean[pick].mean()
            two[i] = d[pick][:, rng.integers(0, n_s, n_s)].mean()
    per_seed = d.mean(0)
    return {
        "a": a, "b": b, "delta": est, "n_env": int(n_e), "n_seed": int(n_s),
        "ci1_lo": float(np.quantile(one, alpha)), "ci1_hi": float(np.quantile(one, 1 - alpha)),
        "ci2_lo": float(np.quantile(two, alpha)), "ci2_hi": float(np.quantile(two, 1 - alpha)),
        "seed_sd": float(per_seed.std(ddof=1)) if n_s > 1 else np.nan,
        "seeds_same_sign": bool(np.all(np.sign(per_seed) == np.sign(est))) if est != 0 else False,
        "level": level,
    }


def resolution(N: int, f: float = 0.10) -> float:
    return RESOLUTION_C[f] / np.sqrt(N)


def verdict(delta: float, lo: float, hi: float, thr: float, seeds_same_sign: bool = True) -> str:
    excl0 = lo > 0 or hi < 0
    if excl0 and abs(delta) >= thr and seeds_same_sign:
        return "resolved"
    if excl0:
        return "detectable"
    return "tied"


def holm_levels(k: int, alpha: float = 0.05) -> list[float]:
    """Confidence levels for the ordered Holm steps: 1 - alpha/k, 1 - alpha/(k-1), ..."""
    return [1 - alpha / (k - i) for i in range(k)]
