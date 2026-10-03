"""W1b (docs/prereg_sequel_wave1_2026-09-28.md §3): within-environment scale t and the pooled-Pearson
optimum t* = k·v_b/a on the gate-2 GEFormer final predictions (own vs fair protocol, 3 seeds, 2 scenarios).

Decomposition of a prediction: m_j = mean prediction in environment j; t_j = SD of predictions within j;
s_ij = (ŷ_ij − m_j)/t_j. t = n-weighted mean of t_j; a = Cov(m, y); v_b = Var(m); S = Var(y) over cells;
k = n-weighted mean of the within-environment covariance Cov_j(s, y). Descriptive only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

base = Path(sys.argv[1])  # .../results_v2/predictions/g2f
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rows = []
for sc in ("F2024m", "F2022m"):
    for proto in ("own", "fair"):
        for f in sorted((base / sc / f"GEFormer_{proto}").glob("seed*.parquet")):
            p = pd.read_parquet(f)
            y, yh = p.observed.to_numpy(float), p.prediction.to_numpy(float)
            g = p.groupby("env")
            m = g.prediction.transform("mean").to_numpy(float)
            tj = g.prediction.transform(lambda v: v.std(ddof=0)).to_numpy(float)
            s = np.where(tj > 0, (yh - m) / np.where(tj > 0, tj, 1), 0.0)
            dfw = pd.DataFrame({"env": p.env, "s": s, "y": y})
            n = dfw.groupby("env").size()
            covw = dfw.groupby("env").apply(lambda q: float(np.mean((q.s - q.s.mean()) * (q.y - q.y.mean()))), include_groups=False)
            k = float((covw * n).sum() / n.sum())
            t = float(pd.Series(tj).groupby(p.env.to_numpy()).first().mul(n).sum() / n.sum())
            a = float(np.mean((m - m.mean()) * (y - y.mean()))); vb = float(m.var()); S = float(y.var())
            tstar = k * vb / a if a > 0 else np.nan
            rmax = float(np.sqrt((a * a / vb + k * k) / S)) if a > 0 else np.nan
            sp = float(np.mean([spearmanr(q.prediction, q.observed)[0] for _, q in g if len(q) >= 25]))
            rows.append({"scenario": sc, "protocol": proto, "seed": f.stem, "t": t, "t_star": tstar,
                         "log_t_over_tstar": float(np.log(t / tstar)) if a > 0 else np.nan, "a": a, "v_b": vb, "k": k,
                         "pooled_r": float(np.corrcoef(yh, y)[0, 1]), "r_max_formula": rmax, "within_spearman": sp})
T = pd.DataFrame(rows); T.to_csv(out / "tstar.csv", index=False)
summ = T.assign(abs_log=T.log_t_over_tstar.abs()).groupby(["scenario", "protocol"])[["t", "t_star", "abs_log", "pooled_r", "r_max_formula", "within_spearman"]].mean()
print(T.round(4).to_string()); print(summ.round(4).to_string())
verd = {}
for sc in ("F2024m", "F2022m"):
    o, fa = summ.loc[(sc, "own")], summ.loc[(sc, "fair")]
    verd[sc] = {"own_closer_to_tstar": bool(o.abs_log < fa.abs_log), "own_t_smaller": bool(o.t < fa.t)}
verd["supported"] = all(v["own_closer_to_tstar"] and v["own_t_smaller"] for v in verd.values())
json.dump(verd, open(out / "verdict.json", "w"), indent=1); print(json.dumps(verd, indent=1))
