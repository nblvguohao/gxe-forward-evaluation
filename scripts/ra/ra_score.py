"""RA scoring (docs/prereg_reanalysis_published_2026-09-30.md §3-4 and the 2026-10-02 addendum). Run once, after every
pre-registered prediction file exists.

Methods (prediction files, env x genotype):
  clac       ra/Y<Y>/CLACsubmission5.csv (+ MODEL_A.csv, MODEL_B.csv descriptive)    Y in 2016..2024 (even)
  lc_real    ra/lc/pred_Y<Y>_real_a0.9.parquet                                       Y in 2022, 2024
  lc_hist    ra/lc/pred_Y2024_hist_a0.9.parquet (descriptive)
  flgbm_gae  ra/flgbm/pred_Y<Y>.parquet (+ flgbm_g descriptive)                       Y in 2022, 2024
Control: benchmark cell_reml (results_v2/benchmark/predictions_G2F.parquet). Observed y joined here only.

Per method: common cells with cell_reml in scored environments (>= 25 genotypes), within-environment Spearman
(primary), Pearson, top-10% selection differential, and pooled Pearson (the methods' own metric); year effect = mean
environment difference, SE by environment bootstrap (B = 2000, seed 20260930); DerSimonian-Laird with the zero-SE
floor; Hartung-Knapp reported; resolution 3/sqrt(N); Holm over {clac, lc_real, flgbm_gae} (two-sided)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dartgxe.forward.pool import dersimonian_laird, floor_se  # noqa: E402

RA, BENCH, OUT = Path(os.environ["RA"]), Path(os.environ["BENCH"]), Path(os.environ["OUT"])
OUT.mkdir(parents=True, exist_ok=True)
SPL = json.load(open(BENCH / "splits.json"))["G2F"]["scorable_environments"]
B, SEED, MIN_N = 2000, 20260930, 25
# Lopez-Cruz 2024 (real and hist) were OOM-killed at the pre-registered 48 GB cap -> "not reproducible on this
# machine" (addendum C rule); lc_real is scored on 2022 only and lc_hist is not available.
YEARS = {"clac": [2016, 2018, 2020, 2022, 2024], "lc_real": [2022], "flgbm_gae": [2022, 2024],
         "clac_A": [2016, 2018, 2020, 2022, 2024], "clac_B": [2016, 2018, 2020, 2022, 2024],
         "flgbm_g": [2022, 2024]}
HOLM = ["clac", "lc_real", "flgbm_gae"]


def load_method(m: str, Y: int) -> pd.DataFrame:
    if m.startswith("clac"):
        f = {"clac": "CLACsubmission5.csv", "clac_A": "MODEL_A.csv", "clac_B": "MODEL_B.csv"}[m]
        d = pd.read_csv(RA / f"Y{Y}" / f)
        return d.rename(columns={"Env": "env", "Hybrid": "genotype", "Yield_Mg_ha": "pred"})[["env", "genotype", "pred"]]
    if m.startswith("lc"):
        d = pd.read_parquet(RA / "lc" / f"pred_Y{Y}_{m.split('_')[1]}_a0.9.parquet")
        return d[["env", "genotype", "pred"]]
    d = pd.read_parquet(RA / "flgbm" / f"pred_Y{Y}.parquet")
    return d.rename(columns={m: "pred"})[["env", "genotype", "pred"]]


def spearman(y, p):
    return np.nan if len(y) < 3 or np.std(y) == 0 else (0.0 if np.std(p) == 0 else float(np.corrcoef(rankdata(y), rankdata(p))[0, 1]))


def pearson(y, p):
    return np.nan if len(y) < 3 or np.std(y) == 0 else (0.0 if np.std(p) == 0 else float(np.corrcoef(y, p)[0, 1]))


def sel_diff(y, p):
    k = max(1, int(round(0.1 * len(y))))
    top = np.argsort(-p, kind="mergesort")[:k]
    return float(((y - y.mean()) / y.std())[top].mean()) if y.std() > 0 else np.nan


def hk(ye):
    d, v = ye["d"].to_numpy(float), ye["se"].to_numpy(float) ** 2
    if len(d) < 2:
        return {"hk_lo": np.nan, "hk_hi": np.nan}
    tau2 = dersimonian_laird(floor_se(ye))["tau2"]
    w = 1 / (v + tau2)
    mu = (w * d).sum() / w.sum()
    se = np.sqrt((w * (d - mu) ** 2).sum() / (len(d) - 1) / w.sum())
    q = tdist.ppf(0.975, len(d) - 1)
    return {"hk_lo": float(mu - q * se), "hk_hi": float(mu + q * se)}


def main():
    cells = pd.read_parquet(BENCH / "cells_G2F.parquet")
    yv = cells.set_index(["env", "genotype"])["y"]
    bench = pd.read_parquet(BENCH / "predictions_G2F.parquet")
    ctrl = bench[bench["method"] == "cell_reml"].set_index(["env", "genotype"])["pred"]
    first = cells.groupby("genotype")["year"].min()
    rng = np.random.default_rng(SEED)
    env_rows, ye_rows, n_cells = [], [], {}
    for m, years in YEARS.items():
        n_cells[m] = 0
        for Y in years:
            d = load_method(m, Y)
            d = d[d["env"].isin(SPL[str(Y)])].dropna(subset=["pred"])
            key = pd.MultiIndex.from_frame(d[["env", "genotype"]])
            d["ctrl"] = ctrl.reindex(key).to_numpy()
            d["y"] = yv.reindex(key).to_numpy()
            d = d.dropna(subset=["ctrl", "y"])
            d["new"] = d["genotype"].map(first).ge(Y).to_numpy()
            n_env = d.groupby("env").size()
            d = d[d["env"].isin(n_env[n_env >= MIN_N].index)]
            n_cells[m] += len(d)
            yy, pp, cc = d["y"].to_numpy(float), d["pred"].to_numpy(float), d["ctrl"].to_numpy(float)
            pooled = {"pooled_m": pearson(yy, pp), "pooled_c": pearson(yy, cc)}
            rows = []
            for e, g in d.groupby("env"):
                y, p, c = g["y"].to_numpy(float), g["pred"].to_numpy(float), g["ctrl"].to_numpy(float)
                r = {"method": m, "target": Y, "env": e, "n": len(g), "sp": spearman(y, p) - spearman(y, c),
                     "pe": pearson(y, p) - pearson(y, c), "sd": sel_diff(y, p) - sel_diff(y, c),
                     "sp_m": spearman(y, p), "sp_c": spearman(y, c)}
                gn = g[g["new"]]
                if len(gn) >= MIN_N:
                    r["sp_new"] = spearman(gn["y"].to_numpy(float), gn["pred"].to_numpy(float)) - spearman(gn["y"].to_numpy(float), gn["ctrl"].to_numpy(float))
                go = g[~g["new"]]
                if len(go) >= MIN_N:
                    r["sp_old"] = spearman(go["y"].to_numpy(float), go["pred"].to_numpy(float)) - spearman(go["y"].to_numpy(float), go["ctrl"].to_numpy(float))
                rows.append(r)
            E = pd.DataFrame(rows)
            env_rows.append(E)
            for metric in ("sp", "pe", "sd", "sp_new", "sp_old"):
                if metric not in E:
                    continue
                v = E[metric].dropna().to_numpy()
                if len(v) == 0:
                    continue
                bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
                ye_rows.append({"method": m, "metric": metric, "dataset": "G2F", "target": Y, "d": float(v.mean()),
                                "se": float(bs.std(ddof=1)), "n_env": len(v), **pooled})
    E = pd.concat(env_rows, ignore_index=True)
    E.to_parquet(OUT / "per_env.parquet", index=False)
    YE = pd.DataFrame(ye_rows)
    YE.to_csv(OUT / "year_effects.csv", index=False)
    PO = []
    for (m, metric), g in YE.groupby(["method", "metric"]):
        r = dersimonian_laird(floor_se(g[["dataset", "d", "se"]]))
        PO.append({"method": m, "metric": metric, "est": r["est"], "se": r["se"], "lo": r["lo"], "hi": r["hi"], "tau2": r["tau2"],
                   "k": r["k"], "N": n_cells[m], "resolution": 3 / np.sqrt(n_cells[m]), **hk(g),
                   "pooled_pearson_method": float(g["pooled_m"].mean()), "pooled_pearson_cell_reml": float(g["pooled_c"].mean())})
    PO = pd.DataFrame(PO)
    PO.to_csv(OUT / "pooled.csv", index=False)
    main_ = PO[(PO["metric"] == "sp") & PO["method"].isin(HOLM)].copy()
    main_["p"] = 2 * norm.sf(np.abs(main_["est"] / main_["se"]))
    main_ = main_.sort_values("p").reset_index(drop=True)
    levels = [1 - 0.05 / 3, 1 - 0.05 / 2, 0.95]
    verdict, stop = {}, False
    for i, row in main_.iterrows():
        g = YE[(YE["method"] == row["method"]) & (YE["metric"] == "sp")]
        ci = dersimonian_laird(floor_se(g[["dataset", "d", "se"]]), level=levels[i])
        rej = (not stop) and (ci["lo"] > 0 or ci["hi"] < 0)
        if not rej:
            stop = True
        res = row["resolution"]
        if rej and row["est"] >= res and ci["lo"] > 0:
            word = "优于 cell_reml"
        elif rej and row["est"] <= -res and ci["hi"] < 0:
            word = "不如 cell_reml"
        elif rej:
            word = "统计上可检测，但低于选择分辨率"
        else:
            word = "未分辨"
        verdict[row["method"]] = {"est": row["est"], "holm_level": levels[i], "ci": [ci["lo"], ci["hi"]], "p": row["p"],
                                  "resolution": res, "N": int(row["N"]), "k": int(row["k"]), "judgement": word}
    json.dump({"verdict": verdict, "note": "re-evaluation by this project; see prereg addendum for fidelity"},
              open(OUT / "verdict.json", "w"), indent=1, ensure_ascii=False, default=float)
    pd.set_option("display.width", 250)
    print(PO.round(4).to_string(index=False))
    print(json.dumps(verdict, indent=1, ensure_ascii=False, default=float))
    print("RA_SCORE_DONE")


if __name__ == "__main__":
    main()
