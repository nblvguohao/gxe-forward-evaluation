"""WP4 sensitivity for W1b: gate-2 GEFormer final predictions (own vs fair) with the exact weighted decomposition
(u = sqrt(sum w_j t_j^2), k = sum w_j Cov_j(p, y) / u, u* = k v_b / a). Usage: wp4_w1b_exact.py <predictions/g2f> <out>."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

base, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = []
for sc in ("F2024m", "F2022m"):
    for proto in ("own", "fair"):
        for f in sorted((base / sc / f"GEFormer_{proto}").glob("seed*.parquet")):
            p = pd.read_parquet(f)
            g = p.groupby("env")
            w = g.size() / len(p)
            tj = g.prediction.std(ddof=0)
            cov = g.apply(lambda q: float(np.mean((q.prediction - q.prediction.mean()) * (q.observed - q.observed.mean()))), include_groups=False)
            u = float(np.sqrt((w * tj ** 2).sum())); k = float((w * cov).sum() / u)
            m = g.prediction.transform("mean").to_numpy(float); y = p.observed.to_numpy(float)
            a = float(np.mean((m - m.mean()) * (y - y.mean()))); vb = float(m.var())
            us = k * vb / a if a > 0 else np.nan
            rows.append({"scenario": sc, "protocol": proto, "seed": f.stem, "u": u, "u_star": us,
                         "abs_log_u_ustar": abs(np.log(u / us)) if a > 0 else np.nan})
T = pd.DataFrame(rows); T.to_csv(out / "w1b_exact.csv", index=False)
s = T.groupby(["scenario", "protocol"])[["u", "u_star", "abs_log_u_ustar"]].mean()
verd = {sc: {"own_closer_to_ustar": bool(s.loc[(sc, "own"), "abs_log_u_ustar"] < s.loc[(sc, "fair"), "abs_log_u_ustar"]),
             "own_u_smaller": bool(s.loc[(sc, "own"), "u"] < s.loc[(sc, "fair"), "u"])} for sc in ("F2024m", "F2022m")}
json.dump(verd, open(out / "w1b_exact_verdict.json", "w"), indent=1)
print(s.round(4).to_string()); print(json.dumps(verd, indent=1))
