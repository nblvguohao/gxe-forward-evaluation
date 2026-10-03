"""WP4 sensitivity for E1 (docs/wp4_tstar_proposition_2026-09-30.md §4): recompute the E1 statistics with the exact
weighted decomposition instead of the approximation used by scripts/w1b_tstar.py and scripts/e1_epochs.py.

Approximation used before: t = n-weighted mean of t_j (within-env SD of predictions), k = n-weighted mean of
Cov_j(s, y) with s standardised within env. Exact (proposition): u = sqrt(sum_j w_j t_j^2), k = sum_j w_j t_j Cov_j(s, y)
/ u; u* = k v_b / a; phi = (a^2/v_b) / (a^2/v_b + k^2). a, v_b, S are unchanged (cell-weighted in both). The selected
epochs e_P (max pooled r) and e_S (max within-env Spearman) do not depend on the decomposition and are read from
results/e1/runs.csv. Descriptive re-analysis of a pre-registered test; the original verdict stays on record."""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from dartgxe.paths import RESULTS

SRC = {"GEBiFormer": RESULTS / "a1" / "epochs", "GEFormer": RESULTS / "a1" / "geformer_epochs"}
R = pd.read_csv(RESULTS / "e1" / "runs.csv")
own = R[R["protocol"].isin(["own", "own_clean"]) & (R["e_P"] != R["e_S"])]


def exact(g):
    y, p, env = g["observed"].to_numpy(float), g["prediction"].to_numpy(float), g["env"].to_numpy()
    df = pd.DataFrame({"env": env, "y": y, "p": p})
    grp = df.groupby("env")
    n = grp.size()
    w = n / n.sum()
    tj = grp["p"].std(ddof=0)
    cov_py = grp.apply(lambda q: float(np.mean((q.p - q.p.mean()) * (q.y - q.y.mean()))), include_groups=False)
    u = float(np.sqrt((w * tj ** 2).sum()))
    k = float((w * cov_py).sum() / u)          # = sum w_j t_j Cov_j(s, y) / u
    m = grp["p"].transform("mean").to_numpy()
    a = float(np.mean((m - m.mean()) * (y - y.mean())))
    vb, S = float(m.var()), float(y.var())
    ok = a > 0 and vb > 0
    between = a * a / vb if ok else np.nan
    return {"u": u, "k": k, "u_star": k * vb / a if ok else np.nan,
            "phi": between / (between + k * k) if ok else np.nan,
            "r_formula": (a + k * u) / np.sqrt((vb + u * u) * S), "pooled_r": float(np.corrcoef(p, y)[0, 1]),
            "cv_tj": float(tj.std(ddof=0) / tj.mean()) if tj.mean() > 0 else np.nan}


rows, check = [], []
for _, r in own.iterrows():
    d = pd.read_parquet(SRC[r["model"]] / f"{r['scenario']}_{r['protocol']}_seed{r['seed']}.parquet")
    rec = {k: r[k] for k in ("model", "scenario", "protocol", "seed", "e_P", "e_S")}
    for tag in ("e_P", "e_S"):
        x = exact(d[d["epoch"] == r[tag]])
        check.append(abs(x["r_formula"] - x["pooled_r"]))
        rec.update({f"phi_{tag}": x["phi"], f"abslog_{tag}": abs(np.log(x["u"] / x["u_star"])) if x["u_star"] > 0 else np.nan,
                    f"cv_tj_{tag}": x["cv_tj"]})
    rows.append(rec)
X = pd.DataFrame(rows)
out = {"identity_max_abs_error": float(max(check)), "runs": len(X)}
for name, cp, cs, better in (("E1_1_phi", "phi_e_P", "phi_e_S", "gt"), ("E1_2_abslog", "abslog_e_P", "abslog_e_S", "lt")):
    u = X[X[cp].notna() & X[cs].notna()]
    wins = int((u[cp] > u[cs]).sum() if better == "gt" else (u[cp] < u[cs]).sum())
    out[name] = {"n_used": len(u), "in_direction": wins, "p_one_sided": binomtest(wins, len(u), 0.5, alternative="greater").pvalue}
out["median_cv_of_tj"] = float(pd.concat([X["cv_tj_e_P"], X["cv_tj_e_S"]]).median())
(RESULTS / "wp4").mkdir(exist_ok=True)
X.to_csv(RESULTS / "wp4" / "e1_exact_runs.csv", index=False)
json.dump(out, open(RESULTS / "wp4" / "e1_exact_verdict.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print(json.dumps(json.load(open(RESULTS / "e1" / "verdict.json")), indent=1)[:1200])
