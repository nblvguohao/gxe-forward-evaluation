"""W1a (docs/prereg_sequel_wave1_2026-09-28.md §2): REML shrinkage vs the full λ curve on the five
new-hybrid forward scenarios. Writes per-(scenario, env, λ) within-environment Spearman and
within-environment MSE (prediction and observation both centred within the environment), plus the
same for the REML λ. Verdicts are computed separately by scripts/w1a_analyze.py."""
import json
import os
import socket
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dartgxe.baselines.common import load_folds
from dartgxe.baselines.linear import Ridge, _geno_feats
from dartgxe.eval.metrics import MIN_N
from dartgxe.features.fit import load_geno
from dartgxe.paths import DATA, RESULTS

SCEN = ["FYnew2016s", "FYnew2018", "FYnew2020", "F2022m", "F2024m"]
GRID = np.logspace(-4, 3, 36)  # identical to gate 1
OUT = RESULTS / "w1a"
OUT.mkdir(parents=True, exist_ok=True)


def per_env(te, pred, tag, lam, sc):
    d = pd.DataFrame({"env": te["env"].to_numpy(), "y": te["y"].to_numpy(float), "p": pred})
    rows = []
    for e, g in d.groupby("env", sort=False):
        if len(g) < MIN_N:
            continue
        yc, pc = g.y - g.y.mean(), g.p - g.p.mean()
        rows.append((sc, e, len(g), tag, lam, spearmanr(g.p, g.y)[0], float(((pc - yc) ** 2).mean())))
    return rows


t0 = time.time()
geno = load_geno()
rows, reml = [], []
for sc in SCEN:
    for fold in load_folds(sc):
        fr = fold.rows(*fold.fit_roles(True))
        _, pos, Z = _geno_feats(geno, fr, fold.cells["genotype"])
        r = Ridge(Z[[pos[h] for h in fr["genotype"]]], fr["env"].to_numpy(), fr["y"].to_numpy())
        rm = r.reml()
        reml.append({"scenario": sc, "fold": fold.fold, **rm})
        te = fold.rows("test")
        Zt = Z[[pos[h] for h in te["genotype"]]]
        rows += per_env(te, Zt @ r.coef(rm["lambda_rel"]), "reml", rm["lambda_rel"], sc)
        for i, l in enumerate(GRID):
            rows += per_env(te, Zt @ r.coef(l), f"g{i:02d}", float(l), sc)
    print(sc, "done", round(time.time() - t0), "s", flush=True)

pd.DataFrame(rows, columns=["scenario", "env", "n", "tag", "lambda_rel", "spearman", "mse_within"]).to_parquet(OUT / "curves.parquet", index=False)
pd.DataFrame(reml).to_csv(OUT / "reml.csv", index=False)
# the run hosts receive the code by rsync, so their .git is stale: the launcher passes the local commit
commit = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                            capture_output=True, text=True).stdout.strip()
json.dump({"git_commit": commit, "data_root": str(DATA), "host": socket.gethostname(), "grid": GRID.tolist(),
           "min_n": MIN_N, "seconds": round(time.time() - t0)}, open(OUT / "run_meta.json", "w"), indent=1)
print("W1A_DONE")
