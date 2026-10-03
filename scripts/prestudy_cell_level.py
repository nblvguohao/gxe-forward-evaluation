"""Pre-study 2 (docs/prestudy_cell_level_2026-09-28.md): cell-level marker ridge with environment fixed effects on
the 31 forward target years. Q1: cross-fitted headroom of per-year shrinkage calibration (grid REML x 10^k,
k in [-2, 4], 25 points). Q2: cell-level REML vs two-stage REML (pre-study 1 curves, k = 0). Upper bounds and
descriptive contrasts only; nothing here selects a reported method."""
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from dartgxe.baselines.linear import Ridge
from dartgxe.forward.data import LOADERS
from dartgxe.forward.panel import features
from dartgxe.forward.pool import dersimonian_laird, floor_se, stratified_unweighted
from dartgxe.paths import RESULTS

K = np.linspace(-2, 4, 25)
MULT = 10.0 ** K
I1 = int(np.argmin(np.abs(K)))
R_SPLIT, B_BOOT, R_BOOT = 200, 200, 50
GO_THRESHOLD = 0.03
OUT = RESULTS / "prestudy_cell_level"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260929)
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
qual = w1c[(w1c["rule"] == "main") & w1c["qualifies"]]
TARGETS = {d: sorted(int(y) for y in g["year"]) for d, g in qual.groupby("dataset")}
two = pd.read_parquet(RESULTS / "prestudy_headroom" / "curves.parquet")
two = two[np.isclose(two["k"], 0.0)].set_index(["dataset", "target", "env"])["spearman"]


def spearman(a, b):
    a, b = rankdata(a), rankdata(b)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def crossfit(S, R, rng):
    n = S.shape[0]
    if n < 2:
        return np.nan
    gains = []
    for _ in range(R):
        p = rng.permutation(n)
        A, Bh = p[: n // 2], p[n // 2:]
        for X, Y in ((A, Bh), (Bh, A)):
            m = int(np.nanargmax(np.nanmean(S[X], 0)))
            gains.append(np.nanmean(S[Y, m] - S[Y, I1]))
    return float(np.nanmean(gains))


t0 = time.time()
curves, q1, q2 = [], [], []
for d, targets in TARGETS.items():
    ds = LOADERS[d]()
    for Y in targets:
        train = ds.cells[ds.cells["year"] < Y]
        train = train[train["genotype"].isin(ds.markers.index)]
        envs = ds.scorable_envs([Y])
        test = ds.cells[(ds.cells["year"] == Y) & ds.cells["env"].isin(envs)]
        fit_ids = sorted(set(train["genotype"]))
        test_ids = sorted(set(test["genotype"]))
        Zf, Zt, _, _ = features(ds.markers, fit_ids, test_ids)
        fpos = {h: i for i, h in enumerate(fit_ids)}
        F = Zf[[fpos[h] for h in train["genotype"]]]
        r = Ridge(F, train["env"].to_numpy(), train["y"].to_numpy(np.float64))
        rm = r.reml()
        P = np.stack([Zt @ r.coef(rm["lambda_rel"] * m) for m in MULT], axis=1)
        del F, r
        pos = {h: i for i, h in enumerate(test_ids)}
        S, diff = [], []
        for e in envs:
            rows = test[test["env"] == e]
            idx = [pos[h] for h in rows["genotype"]]
            s = [spearman(P[idx, j], rows["y"].to_numpy(float)) for j in range(len(MULT))]
            S.append(s)
            curves += [{"dataset": d, "target": Y, "env": e, "n": len(rows), "k": float(k), "spearman": v} for k, v in zip(K, s)]
            diff.append(s[I1] - two.get((d, Y, e), np.nan))
        S, diff = np.asarray(S), np.asarray(diff)
        mc = np.nanmean(S, 0)
        cf = crossfit(S, R_SPLIT, rng)
        boot = [crossfit(S[rng.integers(0, len(S), len(S))], R_BOOT, rng) for _ in range(B_BOOT)]
        q1.append({"dataset": d, "target": Y, "n_env": len(S), "n_train_cells": len(train), "reml_lambda_rel": rm["lambda_rel"],
                   "reml_at_grid_edge": rm["at_grid_edge"], "S_reml_cell": float(mc[I1]),
                   "naive_gain": float(np.nanmax(mc) - mc[I1]), "naive_log10_mult": float(K[int(np.nanargmax(mc))]),
                   "d": cf, "se": float(np.nanstd(boot, ddof=1))})
        dd = diff[np.isfinite(diff)]
        bs = dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)
        q2.append({"dataset": d, "target": Y, "d": float(dd.mean()), "se": float(bs.std(ddof=1)), "n_env": len(dd)})
        print(d, Y, "cell REML %.4f  naive %.4f  crossfit %.4f  cell-two %.4f" % (mc[I1], q1[-1]["naive_gain"], cf, q2[-1]["d"]),
              round(time.time() - t0), "s", flush=True)

pd.DataFrame(curves).to_parquet(OUT / "curves.parquet", index=False)
Q1, Q2 = pd.DataFrame(q1), pd.DataFrame(q2)
Q1.to_csv(OUT / "years.csv", index=False)
Q2.to_csv(OUT / "cell_vs_two_stage.csv", index=False)
y1, y2 = floor_se(Q1[["dataset", "target", "d", "se"]]), floor_se(Q2[["dataset", "target", "d", "se"]])
res = {"Q1_all": dersimonian_laird(y1), "Q1_unweighted": stratified_unweighted(y1, rng),
       "Q1_by_dataset": {d: dersimonian_laird(y1[y1["dataset"] == d]) for d in TARGETS},
       "Q1_naive_mean_by_dataset": Q1.groupby("dataset")["naive_gain"].mean().round(4).to_dict(),
       "Q2_all": dersimonian_laird(y2), "Q2_unweighted": stratified_unweighted(y2, rng),
       "Q2_by_dataset": {d: dersimonian_laird(y2[y2["dataset"] == d]) for d in TARGETS},
       "consistency_g2f_cell_reml_env_mean": Q1[Q1["dataset"] == "G2F"].set_index("target")["S_reml_cell"].round(4).to_dict(),
       "w1a_reference_env_mean": {2016: 0.241, 2018: 0.120, 2020: 0.041, 2022: 0.223, 2024: 0.224}}
res["Q1_decision"] = "GO" if (res["Q1_all"]["est"] >= GO_THRESHOLD and res["Q1_all"]["lo"] > 0) else "NO-GO"
res["Q2_reading"] = "specification differs across crops" if (res["Q2_all"]["lo"] > 0 or res["Q2_all"]["hi"] < 0) else "no consistent difference"
res["git_commit"] = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                                       capture_output=True, text=True).stdout.strip()
res["seconds"] = round(time.time() - t0)
json.dump(res, open(OUT / "verdict.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
print("CELL_LEVEL_DONE")
