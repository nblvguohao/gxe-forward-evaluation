"""Across-environment (target-population) metric for the forward benchmark (post hoc, 2026-10-08;
docs/analysis_note_sparse_value_2026-10-08.md, part B).

For each target year: per line, the mean observed and the mean predicted value over the scored environments of that year
(lines in at least 2 scored environments); Spearman correlation between the two; contrast = method minus cell_reml.
Year standard errors by a line bootstrap (B = 2,000, seed 20261008); SE floor and DerSimonian-Laird pooling as in the
benchmark, with Hartung-Knapp; resolution 3/sqrt(L), L = lines summed over the pooled years. Stored predictions only.

Run from the repository root:  python3 scripts/tpe_metric.py  ->  results/tpe_metric/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "src")
from dartgxe.forward.pool import floor_se, hartung_knapp  # noqa: E402

R, OUT = Path("results"), Path("results/tpe_metric")
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
B, SEED = 2000, 20261008


def spearman_rows(Y, P):
    """Spearman per bootstrap row: Y, P are (B, L) arrays."""
    ry = np.apply_along_axis(rankdata, 1, Y)
    rp = np.apply_along_axis(rankdata, 1, P)
    ry -= ry.mean(1, keepdims=True); rp -= rp.mean(1, keepdims=True)
    return (ry * rp).sum(1) / np.sqrt((ry ** 2).sum(1) * (rp ** 2).sum(1))


def verdict(est, lo, hi, res):
    if lo > 0 and est >= res:
        return "resolved gain"
    if hi < 0 and -est >= res:
        return "resolved loss"
    if lo > 0 or hi < 0:
        return "detectable, below resolution"
    return "tied"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pe = pd.read_parquet(R / "summary48/per_env.parquet")
    rows = []
    for d in DS:
        cells = pd.read_parquet(R / f"benchmark/cells_{d}.parquet")
        pred = pd.read_parquet(R / f"benchmark/predictions_{d}.parquet")
        ctrl = pred[pred["method"] == "cell_reml"].set_index(["env", "genotype"])["pred"]
        for m in sorted(set(pred["method"]) - {"cell_reml", "dl_ge"}):
            pm = pred[pred["method"] == m].set_index(["env", "genotype"])["pred"]
            item_envs = pe[(pe["dataset"] == d) & (pe["item"] == m)][["target", "env"]]
            for Y, ev in item_envs.groupby("target"):
                c = cells[(cells["year"] == Y) & cells["env"].isin(set(ev["env"]))].copy()
                key = pd.MultiIndex.from_frame(c[["env", "genotype"]])
                c["pm"], c["pc"] = pm.reindex(key).to_numpy(), ctrl.reindex(key).to_numpy()
                c = c.dropna(subset=["pm", "pc", "y"])
                g = c.groupby("genotype").agg(y=("y", "mean"), pm=("pm", "mean"), pc=("pc", "mean"), k=("env", "nunique"))
                g = g[g["k"] >= 2]
                if len(g) < 10:
                    continue
                y, a, b = g["y"].to_numpy(), g["pm"].to_numpy(), g["pc"].to_numpy()
                idx = rng.integers(0, len(g), (B, len(g)))
                dbs = spearman_rows(y[idx], a[idx]) - spearman_rows(y[idx], b[idx])
                sm = spearman_rows(y[None], a[None])[0]; sc = spearman_rows(y[None], b[None])[0]
                if not np.isfinite(sm - sc) or np.isfinite(dbs).sum() < B // 2:   # constant predictions (e.g. gbm in two URSN years)
                    continue
                rows.append({"dataset": d, "item": m, "target": Y, "lines": len(g), "sp_method": sm, "sp_ctrl": sc,
                             "d": sm - sc, "se": float(np.nanstd(dbs, ddof=1)), "boot_finite": int(np.isfinite(dbs).sum())})
    YE = pd.DataFrame(rows)
    YE.to_csv(OUT / "year_effects.csv", index=False)
    pooled = []
    for m, g in YE.groupby("item"):
        for rng_name, sub in [("all", g)] + [(d, g[g["dataset"] == d]) for d in DS]:
            if sub.empty:
                continue
            h = hartung_knapp(floor_se(sub[["dataset", "d", "se"]].reset_index(drop=True)))
            res = 3 / np.sqrt(sub["lines"].sum())
            pooled.append({"item": m, "range": rng_name, "k": h["k"], "lines": int(sub["lines"].sum()), "resolution": res, "est": h["est"],
                           "lo": h["lo"], "hi": h["hi"], "hk_lo": h["hk_lo"], "hk_hi": h["hk_hi"], "verdict": verdict(h["est"], h["lo"], h["hi"], res)})
    PO = pd.DataFrame(pooled)
    PO.to_csv(OUT / "pooled.csv", index=False)
    pd.set_option("display.width", 200)
    print(PO[PO["range"] == "all"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
