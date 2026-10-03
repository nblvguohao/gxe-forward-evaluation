"""Figure data for Fig. 2: exact per-epoch decomposition of pooled Pearson (docs/wp4_tstar_proposition_2026-09-30.md)
for the GEFormer 'own' runs of the paired reruns, and the GE-BiFormer 'own' runs. Writes RESULTS/figdata/fig2_epochs.csv.
Environments with >= 25 cells enter the within-environment Spearman (as scripts/e1_epochs.py)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from dartgxe.paths import RESULTS

SRC = {"GEFormer": RESULTS / "a1" / "geformer_epochs", "GEBiFormer": RESULTS / "a1" / "epochs"}
RUNS = {"GEFormer": [("F2024m", "own", s) for s in (147, 1, 2)] + [("F2022m", "own", s) for s in (147, 1, 2)],
        "GEBiFormer": [("F2024m", "own", s) for s in (42, 1, 2)] + [("F2022m", "own", s) for s in (42, 1, 2)]}


def exact(g):
    y, p, env = g["observed"].to_numpy(float), g["prediction"].to_numpy(float), g["env"].to_numpy()
    df = pd.DataFrame({"env": env, "y": y, "p": p})
    grp = df.groupby("env")
    n = grp.size()
    w = n / n.sum()
    tj = grp["p"].std(ddof=0)
    cov = grp.apply(lambda q: float(np.mean((q.p - q.p.mean()) * (q.y - q.y.mean()))), include_groups=False)
    u = float(np.sqrt((w * tj ** 2).sum()))
    k = float((w * cov).sum() / u) if u > 0 else 0.0
    m = grp["p"].transform("mean").to_numpy()
    a = float(np.mean((m - m.mean()) * (y - y.mean())))
    vb, S = float(m.var()), float(y.var())
    sp = [np.corrcoef(rankdata(q.p), rankdata(q.y))[0, 1] for _, q in grp if len(q) >= 25]
    ok = a > 0 and vb > 0
    return {"a": a, "v_b": vb, "k": k, "u": u, "S": S, "u_star": k * vb / a if ok else np.nan,
            "phi": (a * a / vb) / (a * a / vb + k * k) if ok else np.nan, "pooled_r": float(np.corrcoef(p, y)[0, 1]),
            "within_spearman": float(np.nanmean(sp))}


rows = []
for model, runs in RUNS.items():
    for sc, pr, seed in runs:
        f = SRC[model] / f"{sc}_{pr}_seed{seed}.parquet"
        if not f.exists():
            print("missing", f)
            continue
        d = pd.read_parquet(f)
        for ep, g in d.groupby("epoch"):
            rows.append({"model": model, "scenario": sc, "protocol": pr, "seed": seed, "epoch": int(ep), **exact(g)})
        print(model, sc, seed, "done", flush=True)
out = RESULTS / "figdata"
out.mkdir(exist_ok=True)
E = pd.DataFrame(rows)
E.to_csv(out / "fig2_epochs.csv", index=False)
print(len(E), "rows; FIG2_EXPORT_DONE")
