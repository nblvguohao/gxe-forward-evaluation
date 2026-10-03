"""Pre-study (docs/prestudy_shrink_headroom_2026-09-28.md): how much could per-year shrinkage calibration gain
over REML at best? Two-stage marker ridge (as wave 2b) with lambda_rel = REML x 10^k, k in 25 steps from -2 to 4,
on the 31 forward target years. Naive oracle (max over the grid on the same environments) and cross-fitted
oracle (choose on half of the year's environments, score on the other half, 200 splits); environment bootstrap
SE; random-effects pooling. Upper bounds only: nothing here selects a reported method."""
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from dartgxe.baselines.linear import Ridge
from dartgxe.forward.data import LOADERS
from dartgxe.forward.panel import features, stage1
from dartgxe.forward.pool import dersimonian_laird, floor_se, stratified_unweighted
from dartgxe.paths import RESULTS

K = np.linspace(-2, 4, 25)
MULT = 10.0 ** K
I1 = int(np.argmin(np.abs(K)))  # the REML point (k = 0)
R_SPLIT, B_BOOT, R_BOOT = 200, 200, 50
GO_THRESHOLD = 0.03
OUT = RESULTS / "prestudy_headroom"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260928)
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
qual = w1c[(w1c["rule"] == "main") & w1c["qualifies"]]
TARGETS = {d: sorted(int(y) for y in g["year"]) for d, g in qual.groupby("dataset")}


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
curves, years = [], []
for d, targets in TARGETS.items():
    ds = LOADERS[d]()
    for Y in targets:
        train = ds.cells[ds.cells["year"] < Y]
        envs = ds.scorable_envs([Y])
        test = ds.cells[(ds.cells["year"] == Y) & ds.cells["env"].isin(envs)]
        g = stage1(train)
        fit_ids = sorted(set(g.index) & set(ds.markers.index))
        test_ids = sorted(set(test["genotype"]))
        Zf, Zt, _, _ = features(ds.markers, fit_ids, test_ids)
        r = Ridge(Zf, np.zeros(len(fit_ids)), g.loc[fit_ids].to_numpy(np.float64))
        rm = r.reml()
        P = np.stack([Zt @ r.coef(rm["lambda_rel"] * m) for m in MULT], axis=1)
        pos = {h: i for i, h in enumerate(test_ids)}
        S = []
        for e in envs:
            rows = test[test["env"] == e]
            idx = [pos[h] for h in rows["genotype"]]
            s = [spearman(P[idx, j], rows["y"].to_numpy(float)) for j in range(len(MULT))]
            S.append(s)
            curves += [{"dataset": d, "target": Y, "env": e, "n": len(rows), "k": float(k), "mult": float(m), "spearman": v}
                       for k, m, v in zip(K, MULT, s)]
        S = np.asarray(S)
        mean_curve = np.nanmean(S, 0)
        cf = crossfit(S, R_SPLIT, rng)
        boot = [crossfit(S[rng.integers(0, len(S), len(S))], R_BOOT, rng) for _ in range(B_BOOT)]
        years.append({"dataset": d, "target": Y, "n_env": len(S), "reml_lambda_rel": rm["lambda_rel"], "reml_at_grid_edge": rm["at_grid_edge"],
                      "S_reml": float(mean_curve[I1]), "naive_gain": float(np.nanmax(mean_curve) - mean_curve[I1]),
                      "naive_log10_mult": float(K[int(np.nanargmax(mean_curve))]), "d": cf, "se": float(np.nanstd(boot, ddof=1))})
        print(d, Y, "naive %.4f crossfit %.4f" % (years[-1]["naive_gain"], cf), round(time.time() - t0), "s", flush=True)

C = pd.DataFrame(curves)
C.to_parquet(OUT / "curves.parquet", index=False)
Yr = pd.DataFrame(years)
Yr.to_csv(OUT / "years.csv", index=False)
ye = floor_se(Yr[["dataset", "target", "d", "se"]])
res = {"all": dersimonian_laird(ye), "all_unweighted": stratified_unweighted(ye, rng)}
for d in TARGETS:
    sub = ye[ye["dataset"] == d]
    res[d] = dersimonian_laird(sub)
res["G2F+NUST"] = dersimonian_laird(ye[ye["dataset"].isin(["G2F", "NUST"])])
res["naive_mean_by_dataset"] = Yr.groupby("dataset")["naive_gain"].mean().round(4).to_dict()
res["crossfit_mean_by_dataset"] = Yr.groupby("dataset")["d"].mean().round(4).to_dict()
res["naive_log10_mult_by_dataset"] = Yr.groupby("dataset")["naive_log10_mult"].describe()[["min", "50%", "max"]].round(2).to_dict("index")
res["rule"] = f"GO if pooled cross-fitted gain >= {GO_THRESHOLD} and its 95% CI lower bound > 0"
res["decision"] = "GO" if (res["all"]["est"] >= GO_THRESHOLD and res["all"]["lo"] > 0) else "NO-GO"
res["git_commit"] = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                                       capture_output=True, text=True).stdout.strip()
res["seconds"] = round(time.time() - t0)
json.dump(res, open(OUT / "verdict.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
print("HEADROOM_DONE")
