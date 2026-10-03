"""Verdicts for W1a (docs/prereg_sequel_wave1_2026-09-28.md §2) from results/w1a/curves.parquet.

Prediction 1: S(λ_REML) >= S(λ_S) - 3/sqrt(N) in >= 4 of 5 scenarios (also reported at 1.5 x 3/sqrt(N)).
Prediction 2: SD(log10 λ_M) / SD(log10 λ_S) > 1 across scenarios, with the lower bound of the 95% CI
from a within-scenario environment bootstrap (2,000 draws) > 1; 'not decidable' if >= 2 scenarios
have λ_M at a grid end. S = n-weighted within-env Spearman, M = n-weighted within-env MSE."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/w1a")
d = pd.read_parquet(root / "curves.parquet")
grid = d[d.tag != "reml"]
ntags = grid.tag.nunique()
rng = np.random.default_rng(20260928)


def curves(g, w=None):
    """n-weighted S and M per grid tag; w = bootstrap multiplicity per env."""
    g = g.assign(w=g.n * (1 if w is None else g.env.map(w)))
    agg = g.assign(ws=g.w * g.spearman, wm=g.w * g.mse_within).groupby("tag")[["ws", "wm", "w"]].sum()
    return (agg.ws / agg.w), (agg.wm / agg.w)


out, lam = {}, {"S": {}, "M": {}}
for sc, g in grid.groupby("scenario"):
    S, M = curves(g)
    tS, tM = S.idxmax(), M.idxmin()
    lamS = float(g.loc[g.tag == tS, "lambda_rel"].iloc[0]); lamM = float(g.loc[g.tag == tM, "lambda_rel"].iloc[0])
    r = d[(d.scenario == sc) & (d.tag == "reml")]
    s_reml = float((r.n * r.spearman).sum() / r.n.sum())
    N = int(r.n.sum())
    res = 3 / np.sqrt(N)
    out[sc] = {"N": N, "n_env": int(r.env.nunique()), "resolution": res, "S_max": float(S.max()), "lambda_S": lamS,
               "S_reml": s_reml, "lambda_reml": float(r.lambda_rel.iloc[0]), "gap": float(S.max() - s_reml),
               "in_plateau": bool(s_reml >= S.max() - res), "in_plateau_x1.5": bool(s_reml >= S.max() - 1.5 * res),
               "lambda_M": lamM, "lambda_S_at_edge": tS in ("g00", f"g{ntags-1:02d}"),
               "lambda_M_at_edge": tM in ("g00", f"g{ntags-1:02d}"),
               "plateau_lambda_range": [float(g.loc[g.tag == t, "lambda_rel"].iloc[0]) for t in (S[S >= S.max() - res].index.min(), S[S >= S.max() - res].index.max())]}
    lam["S"][sc], lam["M"][sc] = np.log10(lamS), np.log10(lamM)

k = sum(v["in_plateau"] for v in out.values())
ratio = np.std(list(lam["M"].values()), ddof=1) / np.std(list(lam["S"].values()), ddof=1)
boot = []
for _ in range(2000):
    ls, lm = [], []
    for sc, g in grid.groupby("scenario"):
        envs = g.env.unique()
        w = pd.Series(rng.choice(envs, len(envs))).value_counts()
        S, M = curves(g[g.env.isin(w.index)], w)
        ls.append(np.log10(float(g.loc[g.tag == S.idxmax(), "lambda_rel"].iloc[0])))
        lm.append(np.log10(float(g.loc[g.tag == M.idxmin(), "lambda_rel"].iloc[0])))
    sdS = np.std(ls, ddof=1)
    boot.append(np.std(lm, ddof=1) / sdS if sdS > 0 else np.inf)
lo, hi = np.percentile(boot, [2.5, 97.5])
edgesM = sum(v["lambda_M_at_edge"] for v in out.values())
verdict = {"prediction1": {"k_in_plateau": k, "of": len(out), "pass": k >= 4,
                           "k_in_plateau_x1.5": sum(v["in_plateau_x1.5"] for v in out.values())},
           "prediction2": {"sd_log_lambda_M": float(np.std(list(lam["M"].values()), ddof=1)),
                           "sd_log_lambda_S": float(np.std(list(lam["S"].values()), ddof=1)),
                           "ratio": float(ratio), "ci95": [float(lo), float(hi)], "lambda_M_at_edge": edgesM,
                           "verdict": "not decidable" if edgesM >= 2 else ("pass" if ratio > 1 and lo > 1 else "fail")},
           "per_scenario": out}
json.dump(verdict, open(root / "verdict.json", "w"), indent=1)
print(json.dumps(verdict, indent=1))
