"""Within-environment metrics (spec 03 §3.1), computed per (method, seed, fold, env)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

MIN_N = 25


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _topk(yhat: np.ndarray, k: int) -> np.ndarray:
    # stable order on -yhat: ties are broken by position only, so any strictly monotone
    # transform of yhat (which preserves ties and order) selects the same set
    return np.argsort(-yhat, kind="stable")[:k]


def env_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    n = len(y)
    out = {"n": n, "spearman": _pearson(rankdata(y), rankdata(yhat)), "pearson": _pearson(y, yhat)}
    sd = y.std(ddof=1)
    for f in (0.05, 0.10, 0.20):
        k = max(1, int(round(f * n)))
        sel = _topk(yhat, k)
        out[f"sel_diff_f{int(f * 100):02d}"] = float((y[sel].mean() - y.mean()) / sd) if sd > 0 else np.nan
    k = max(1, int(round(0.10 * n)))
    sel, best = _topk(yhat, k), _topk(y, k)
    out["topk_hit_f10"] = len(set(sel) & set(best)) / k
    gain = y - y.min()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = (gain[best] * disc).sum()
    out["ndcg_f10"] = float((gain[sel] * disc).sum() / idcg) if idcg > 0 else np.nan
    r = yhat - y
    out["rmse"] = float(np.sqrt((r * r).mean()))
    out["bias"] = float(r.mean())
    v = yhat.var()
    out["calib_slope"] = float(np.cov(yhat, y, bias=True)[0, 1] / v) if v > 0 else np.nan
    return out


PRIMARY = ["spearman", "pearson"]
DECISION = ["sel_diff_f10", "topk_hit_f10", "ndcg_f10"]


def per_env(pred: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    """pred columns: method, seed, fold, env, genotype, observed, prediction.
    Returns long table: method, seed, fold, env, n, metric, value (envs with n >= min_n only)."""
    rows = []
    for (m, s, f, e), g in pred.groupby(["method", "seed", "fold", "env"], sort=False):
        if len(g) < min_n:
            continue
        met = env_metrics(g["observed"].to_numpy(float), g["prediction"].to_numpy(float))
        n = met.pop("n")
        rows.extend((m, s, f, e, n, k, v) for k, v in met.items())
    return pd.DataFrame(rows, columns=["method", "seed", "fold", "env", "n", "metric", "value"])


def pooled_pearson(pred: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic only (spec 03 §3.1): across environments, per (method, seed, fold)."""
    return (pred.groupby(["method", "seed", "fold"])
            .apply(lambda g: _pearson(g["observed"].to_numpy(float), g["prediction"].to_numpy(float)), include_groups=False)
            .rename("pooled_pearson_r").reset_index())


def env_seed_matrix(pe: pd.DataFrame, metric: str) -> pd.DataFrame:
    """m_{e,s}: fold-weighted (by n) mean within (method, seed, env). Returns index (method, env), columns seed."""
    d = pe[(pe["metric"] == metric) & pe["value"].notna()].copy()  # undefined values stay undefined,
    d["wv"] = d["value"] * d["n"]                                    # never summed as 0 (bug fixed 2026-09-26)
    g = d.groupby(["method", "env", "seed"])[["wv", "n"]].sum()
    m = (g["wv"] / g["n"]).rename("value").reset_index()
    return m.pivot_table(index=["method", "env"], columns="seed", values="value")
