"""Score the CLAC decomposition (docs/prereg_clac_decomposition_2026-10-02.md section 3). Run once, after all ablation
files exist. Same cells, metric, bootstrap and pooling as scripts/ra/ra_score.py.

Env: DECOMP (ra_decomp dir), BENCH (benchmark dir with cells_G2F.parquet, predictions_G2F.parquet, splits.json), OUT."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dartgxe.forward.pool import dersimonian_laird, floor_se  # noqa: E402

DEC, BENCH, OUT = Path(os.environ["DECOMP"]), Path(os.environ["BENCH"]), Path(os.environ["OUT"])
OUT.mkdir(parents=True, exist_ok=True)
SPL = json.load(open(BENCH / "splits.json"))["G2F"]["scorable_environments"]
YEARS, B, SEED, MIN_N = [2016, 2018, 2020, 2022, 2024], 2000, 20261002, 25
IDS = ["A_full", "D1_equal_weights", "D2_no_spatial", "D3_no_cleaning", "D4_linear_kernel", "D5_single_main_effect"]


def sp(y, p):
    return np.nan if len(y) < 3 or np.std(y) == 0 else (0.0 if np.std(p) == 0 else float(np.corrcoef(rankdata(y), rankdata(p))[0, 1]))


def main():
    cells = pd.read_parquet(BENCH / "cells_G2F.parquet")
    yv = cells.set_index(["env", "genotype"])["y"]
    bench = pd.read_parquet(BENCH / "predictions_G2F.parquet")
    ctrl = bench[bench["method"] == "cell_reml"].set_index(["env", "genotype"])["pred"]
    rng = np.random.default_rng(SEED)
    rows, ncell = [], {}
    for Y in YEARS:
        P = None
        for i in IDS:
            d = pd.read_csv(DEC / f"Y{Y}" / f"decomp_{i}.csv").rename(columns={"Env": "env", "Hybrid": "genotype", "pred": i})
            P = d if P is None else P.merge(d, on=["env", "genotype"], how="inner")
        P = P[P["env"].isin(SPL[str(Y)])]
        key = pd.MultiIndex.from_frame(P[["env", "genotype"]])
        P["ctrl"] = ctrl.reindex(key).to_numpy(); P["y"] = yv.reindex(key).to_numpy()
        P = P.dropna()
        n = P.groupby("env").size(); P = P[P["env"].isin(n[n >= MIN_N].index)]
        ncell[Y] = len(P)
        for e, g in P.groupby("env"):
            y = g["y"].to_numpy(float)
            r = {"target": Y, "env": e, "n": len(g), "ctrl": sp(y, g["ctrl"].to_numpy(float))}
            for i in IDS:
                r[i] = sp(y, g[i].to_numpy(float))
            rows.append(r)
    E = pd.DataFrame(rows); E.to_csv(OUT / "per_env.csv", index=False)
    N = sum(ncell.values()); res = 3 / np.sqrt(N)
    contrasts = {f"loss_{i}": ("A_full", i) for i in IDS[1:]}
    contrasts.update({f"{i}_vs_cell_reml": (i, "ctrl") for i in IDS})
    YE, PO = [], []
    for name, (a, b) in contrasts.items():
        for Y, g in E.groupby("target"):
            v = (g[a] - g[b]).dropna().to_numpy()
            bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
            YE.append({"contrast": name, "dataset": "G2F", "target": Y, "d": float(v.mean()), "se": float(bs.std(ddof=1)), "n_env": len(v)})
    YE = pd.DataFrame(YE); YE.to_csv(OUT / "year_effects.csv", index=False)
    for name, g in YE.groupby("contrast"):
        r = dersimonian_laird(floor_se(g[["dataset", "d", "se"]]))
        PO.append({"contrast": name, "est": r["est"], "se": r["se"], "lo": r["lo"], "hi": r["hi"], "tau2": r["tau2"], "k": r["k"],
                   "N": N, "resolution": res, "years_positive": int((g["d"] > 0).sum())})
    PO = pd.DataFrame(PO); PO.to_csv(OUT / "pooled.csv", index=False)
    prim = PO[PO["contrast"].str.startswith("loss_")].copy()
    prim["p"] = 2 * norm.sf(np.abs(prim["est"] / prim["se"]))
    prim = prim.sort_values("p").reset_index(drop=True)
    verdict, stop = {}, False
    for k, row in prim.iterrows():
        level = 1 - 0.05 / (len(prim) - k)
        g = YE[YE["contrast"] == row["contrast"]]
        ci = dersimonian_laird(floor_se(g[["dataset", "d", "se"]]), level=level)
        rej = (not stop) and (ci["lo"] > 0 or ci["hi"] < 0)
        if not rej:
            stop = True
        if rej and row["est"] >= res and ci["lo"] > 0:
            word = "carries a resolved share of the gain"
        elif rej:
            word = "detectable, below resolution"
        else:
            word = "no detectable share"
        verdict[row["contrast"]] = {"est": row["est"], "holm_level": level, "ci": [ci["lo"], ci["hi"]], "p": row["p"],
                                    "resolution": res, "N": N, "judgement": word}
    json.dump({"verdict": verdict, "n_cells_by_year": ncell}, open(OUT / "verdict.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print(PO.round(4).to_string(index=False)); print(json.dumps(verdict, indent=1, default=float)); print("DECOMP_SCORE_DONE")


if __name__ == "__main__":
    main()
