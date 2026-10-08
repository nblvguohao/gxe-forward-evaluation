"""Sensitivity of the verdicts to the constant of the resolution threshold, and the threshold of single target years (post hoc,
2026-10-08; no model is refitted and nothing is rescored).

1. verdicts.csv: every pooled contrast of results/revision_tcj/hk_sensitivity.csv (Supplementary Table S14) judged again with the
   threshold c/sqrt(N), c = 2, 3, 4, N the scored cells behind the stored resolution 3/sqrt(N); estimate and interval (at the
   stated Holm level) unchanged. c = 3 must reproduce the stored DerSimonian-Laird verdict.
2. per_year.csv: 3/sqrt(N_y) for single target years of the forward benchmark, N_y the cells scored in year y (two-stage GBLUP
   against cell-level GBLUP, results/summary48/per_env.parquet), summarised by dataset.
Run from the repository root:  python3 scripts/threshold_sensitivity.py  ->  results/threshold_sensitivity/"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results"); OUT = R / "threshold_sensitivity"; OUT.mkdir(parents=True, exist_ok=True)


def verdict(est, lo, hi, res):
    if lo > 0 and est >= res:
        return "resolved gain"
    if hi < 0 and -est >= res:
        return "resolved loss"
    if lo > 0 or hi < 0:
        return "detectable, below resolution"
    return "tied"


def main():
    H = pd.read_csv(R / "revision_tcj/hk_sensitivity.csv")
    rows = []
    for r in H.itertuples():
        N = (3 / r.resolution) ** 2
        v = {c: verdict(r.est, r.dl_lo, r.dl_hi, c / np.sqrt(N)) for c in (2, 3, 4)}
        assert v[3] == r.dl_verdict, (r.contrast, v[3], r.dl_verdict)
        rows.append({"contrast": r.contrast, "target_years": r.target_years, "level": r.level, "est": r.est, "lo": r.dl_lo, "hi": r.dl_hi,
                     "resolution_c2": 2 / np.sqrt(N), "resolution_c3": r.resolution, "resolution_c4": 4 / np.sqrt(N),
                     "verdict_c2": v[2], "verdict_c3": v[3], "verdict_c4": v[4]})
    V = pd.DataFrame(rows)
    V.to_csv(OUT / "verdicts.csv", index=False)
    pe = pd.read_parquet(R / "summary48/per_env.parquet")
    pe = pe[pe["item"] == "reml"]
    ny = pe.groupby(["dataset", "target"])["n"].sum().reset_index()
    ny["threshold"] = 3 / np.sqrt(ny["n"])
    S = ny.groupby("dataset").agg(target_years=("target", "size"), cells_min=("n", "min"), cells_max=("n", "max"),
                                  threshold_min=("threshold", "min"), threshold_median=("threshold", "median"),
                                  threshold_max=("threshold", "max")).reset_index()
    S.to_csv(OUT / "per_year.csv", index=False)
    pd.set_option("display.width", 220)
    print(V[["contrast", "verdict_c2", "verdict_c3", "verdict_c4"]].to_string(index=False))
    print("changed vs c=3: c=2", int((V.verdict_c2 != V.verdict_c3).sum()), "c=4", int((V.verdict_c4 != V.verdict_c3).sum()), "of", len(V))
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
