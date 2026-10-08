"""Year-level pooling used after wave 2b (same definitions as scripts/b3_rules.py, which keeps its own copy so
its reported numbers stay reproducible): DerSimonian-Laird random effects with the zero-SE floor of the
2026-09-28 amendment, and the dataset-stratified unweighted year bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def floor_se(ye: pd.DataFrame) -> pd.DataFrame:
    ye = ye.copy()
    pos_all = ye.loc[ye["se"] > 0, "se"]
    ye["se_zero"] = ye["se"] <= 0
    for _, g in ye.groupby("dataset"):
        pos = g.loc[g["se"] > 0, "se"]
        fl = pos.median() if len(pos) else (pos_all.median() if len(pos_all) else np.nan)
        ye.loc[g.index, "se"] = np.maximum(g["se"], fl)
    return ye


def dersimonian_laird(ye: pd.DataFrame, level: float = 0.95) -> dict:
    d, v = ye["d"].to_numpy(float), ye["se"].to_numpy(float) ** 2
    w = 1 / v
    mu = (w * d).sum() / w.sum()
    Q = (w * (d - mu) ** 2).sum()
    tau2 = max(0.0, (Q - (len(d) - 1)) / (w.sum() - (w * w).sum() / w.sum())) if len(d) > 1 else 0.0
    ws = 1 / (v + tau2)
    est, se = (ws * d).sum() / ws.sum(), 1 / np.sqrt(ws.sum())
    z = norm.ppf(1 - (1 - level) / 2)
    return {"est": float(est), "se": float(se), "lo": float(est - z * se), "hi": float(est + z * se), "tau2": float(tau2),
            "k": int(len(d)), "n_years_se_zero_floored": int(ye["se_zero"].sum()) if "se_zero" in ye else 0,
            "weight_share": (pd.Series(ws, index=ye["dataset"].to_numpy()).groupby(level=0).sum() / ws.sum()).round(3).to_dict()}


def stratified_unweighted(ye: pd.DataFrame, rng, B: int = 2000) -> dict:
    by = {d: g["d"].to_numpy(float) for d, g in ye.groupby("dataset")}
    bs = [np.mean(np.concatenate([x[rng.integers(0, len(x), len(x))] for x in by.values()])) for _ in range(B)]
    return {"est": float(ye["d"].mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}


def hartung_knapp(ye: pd.DataFrame, level: float = 0.95) -> dict:
    """Hartung-Knapp(-Sidik-Jonkman) interval around the DerSimonian-Laird estimate: same weights 1/(v + tau2), variance
    sum w (d - est)^2 / ((k - 1) sum w), t quantile with k - 1 degrees of freedom. Sensitivity analysis only (TCJ revision,
    2026-10-08); the pre-specified pooling is dersimonian_laird."""
    from scipy.stats import t as tdist
    r = dersimonian_laird(ye, level=level)
    d, v = ye["d"].to_numpy(float), ye["se"].to_numpy(float) ** 2
    k = len(d)
    if k < 2:
        return {**r, "hk_lo": float("nan"), "hk_hi": float("nan"), "hk_se": float("nan")}
    w = 1 / (v + r["tau2"])
    se = float(np.sqrt((w * (d - r["est"]) ** 2).sum() / ((k - 1) * w.sum())))
    q = float(tdist.ppf(1 - (1 - level) / 2, k - 1))
    return {**r, "hk_se": se, "hk_lo": r["est"] - q * se, "hk_hi": r["est"] + q * se}
