"""Gate 1 (docs/gate1_reml_2026-09-26.md): REML vs validation-selected vs test-optimal shrinkage."""
import json

import numpy as np
import pandas as pd

from dartgxe.baselines.common import load_folds
from dartgxe.baselines.linear import Ridge, _geno_feats
from dartgxe.eval.metrics import per_env
from dartgxe.features.fit import load_ec, load_geno
from dartgxe.paths import RESULTS

MAIN = ["FYnew2016s", "FYnew2018", "FYnew2020", "F2022m"]
DESC = ["CV00", "CV1"]
GRID = np.logspace(-4, 3, 36)
OUT = RESULTS / "gate1"
OUT.mkdir(parents=True, exist_ok=True)


def score(rows, pred, fold):
    p = pd.DataFrame({"method": "x", "seed": 0, "fold": fold, "env": rows["env"].to_numpy(),
                      "genotype": rows["genotype"].to_numpy(), "observed": rows["y"].to_numpy(), "prediction": pred})
    v = per_env(p).query("metric == 'spearman'")
    return v[["env", "n", "value"]]


def pooled(parts):
    d = pd.concat(parts)
    return {"n_weighted": float((d.value * d.n).sum() / d.n.sum()), "env_mean": float(d.groupby("env").value.mean().mean())}


geno, ec = load_geno(), load_ec()
res, reml_rows = {}, []
for sc in MAIN + DESC:
    parts = {"reml": [], **{f"g{i}": [] for i in range(len(GRID))}}
    for fold in load_folds(sc):
        fr = fold.rows(*fold.fit_roles(True))
        _, pos, Z = _geno_feats(geno, fr, fold.cells["genotype"])
        r = Ridge(Z[[pos[h] for h in fr["genotype"]]], fr["env"].to_numpy(), fr["y"].to_numpy())
        rm = r.reml()
        reml_rows.append({"scenario": sc, "fold": fold.fold, **rm})
        te = fold.rows("test")
        Zt = Z[[pos[h] for h in te["genotype"]]]
        parts["reml"].append(score(te, Zt @ r.coef(rm["lambda_rel"]), fold.fold))
        for i, l in enumerate(GRID):
            parts[f"g{i}"].append(score(te, Zt @ r.coef(l), fold.fold))
    grid_scores = {i: pooled(parts[f"g{i}"]) for i in range(len(GRID))}
    best_i = max(grid_scores, key=lambda i: grid_scores[i]["n_weighted"])
    out = {"reml": pooled(parts["reml"]), "optimal": {**grid_scores[best_i], "lambda_rel": float(GRID[best_i])}}
    # validation-selected B1 and the networks, same metric
    for m in ["B1_gblup", "S1_twohead_mse_noec", "S1_twohead_mse"]:
        base = RESULTS / "predictions" / "g2f" / sc / m
        files = sorted(base.glob("seed*.parquet")) or sorted((base / "parts").glob("seed*_fold*.parquet"))
        if not files:
            continue
        p = pd.concat([pd.read_parquet(f) for f in files])
        v = per_env(p).query("metric == 'spearman'")
        # average over seeds within (fold, env) first, then pool like the others
        v = v.groupby(["fold", "env"]).agg(n=("n", "first"), value=("value", "mean")).reset_index()
        out[m] = pooled([v])
    res[sc] = out
    print(sc, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}), flush=True)

pd.DataFrame(reml_rows).to_csv(OUT / "reml_estimates.csv", index=False)
json.dump(res, open(OUT / "gate1_scores.json", "w"), indent=1)
g = {sc: res[sc]["optimal"]["n_weighted"] - res[sc]["reml"]["n_weighted"] for sc in MAIN}
gv = {sc: res[sc]["optimal"]["n_weighted"] - res[sc]["B1_gblup"]["n_weighted"] for sc in MAIN}
verdict = {"gap_reml": g, "gap_val": gv, "mean_gap_reml": float(np.mean(list(g.values()))),
           "years_gap_reml_gt_0.005": int(sum(x > 0.005 for x in g.values())),
           "pass": bool(np.mean(list(g.values())) >= 0.02 and sum(x > 0.005 for x in g.values()) >= 3)}
json.dump(verdict, open(OUT / "gate1_verdict.json", "w"), indent=1)
print("VERDICT", json.dumps(verdict), flush=True)
print(pd.DataFrame(reml_rows).groupby("scenario")[["lambda_rel", "h2", "at_grid_edge"]].agg(["mean", "min", "max"]).round(4).to_string())
print("GATE1_DONE")
