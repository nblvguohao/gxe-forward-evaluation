"""Score a submission to the G x E forward-year benchmark (standalone: numpy and pandas only).

Submission: CSV or parquet with columns dataset, env, year, genotype, pred -- one prediction for every scorable cell
of the target years you enter (a dataset may be left out entirely; a dataset you enter must be complete). Rules
(README): predictions for target year Y may use phenotypes of years < Y only; genotypes of every year may be used.

For every scorable environment: within-environment Spearman and top-10 % selection differential (in SD of y), for the
submission and for the reference cell-level GBLUP (cell_reml). Year effect = mean over environments of the
difference; SE by environment bootstrap; DerSimonian-Laird pooling over target years with the zero-SE floor (SE of a
year with zero SE set to the median positive SE of its dataset). Reported for the 31 original years, the 17
independent years and all entered years, plus the resolution 3/sqrt(N).

Usage: python score.py submission.csv [--data DIR] [--out result.json]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

B, SEED, FRAC, COVER = 2000, 20261001, 0.10, 0.95


def ranks(a):
    return pd.Series(a).rank(method="average").to_numpy()


def env_scores(y, p):
    if len(y) < 3 or np.std(p) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    sp = np.corrcoef(ranks(p), ranks(y))[0, 1]
    top = np.argsort(-p, kind="mergesort")[:max(1, int(round(FRAC * len(y))))]
    return float(sp), float(((y - y.mean()) / y.std())[top].mean())


def floor_se(ye):
    ye = ye.copy()
    for d, g in ye.groupby("dataset"):
        pos = g.loc[g["se"] > 0, "se"]
        fl = pos.median() if len(pos) else ye.loc[ye["se"] > 0, "se"].median()
        ye.loc[g.index, "se"] = np.maximum(g["se"], fl)
    return ye


def dl(ye):
    d, v = ye["d"].to_numpy(float), ye["se"].to_numpy(float) ** 2
    w = 1 / v
    mu = (w * d).sum() / w.sum()
    Q = (w * (d - mu) ** 2).sum()
    tau2 = max(0.0, (Q - (len(d) - 1)) / (w.sum() - (w * w).sum() / w.sum())) if len(d) > 1 else 0.0
    ws = 1 / (v + tau2)
    est, se = (ws * d).sum() / ws.sum(), 1 / np.sqrt(ws.sum())
    return {"est": float(est), "lo": float(est - 1.96 * se), "hi": float(est + 1.96 * se), "tau2": float(tau2), "years": len(d)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--out", default="score.json")
    a = ap.parse_args(argv)
    data = Path(a.data)
    splits = json.load(open(data / "splits.json"))
    sub = pd.read_parquet(a.submission) if a.submission.endswith(".parquet") else pd.read_csv(a.submission)
    miss = {"dataset", "env", "year", "genotype", "pred"} - set(sub.columns)
    if miss:
        raise SystemExit(f"submission lacks columns {sorted(miss)}")
    sub = sub.astype({"dataset": str, "env": str, "genotype": str, "year": int})
    dup = sub.duplicated(["dataset", "env", "year", "genotype"])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate (dataset, env, year, genotype) rows in the submission; one prediction per cell")
    rng = np.random.default_rng(SEED)
    rows, cover = [], {}
    for d in sorted(set(sub["dataset"])):
        if d not in splits:
            raise SystemExit(f"unknown dataset {d!r}; known: {sorted(splits)}")
        sp = splits[d]
        cells = pd.read_parquet(data / f"cells_{d}.parquet").astype({"env": str, "genotype": str})
        ref = pd.read_parquet(data / f"predictions_{d}.parquet").astype({"env": str, "genotype": str})
        ref = ref[ref["method"] == "cell_reml"][["env", "year", "genotype", "pred"]].rename(columns={"pred": "ref"})
        s = sub[sub["dataset"] == d][["env", "year", "genotype", "pred"]]
        for Y in sp["target_years"]:
            envs = sp["scorable_environments"][str(Y)]
            c = cells[(cells["year"] == Y) & cells["env"].isin(envs)]
            m = c.merge(s, on=["env", "year", "genotype"], how="left").merge(ref, on=["env", "year", "genotype"], how="left")
            share = float(m["pred"].notna().mean())
            cover[f"{d}_{Y}"] = share
            if share < COVER:
                raise SystemExit(f"{d} {Y}: only {share:.1%} of the {len(c)} scorable cells predicted (need {COVER:.0%})")
            for e, g in m.dropna(subset=["pred", "ref"]).groupby("env"):
                y = g["y"].to_numpy(float)
                sa, ga = env_scores(y, g["pred"].to_numpy(float))
                sb, gb = env_scores(y, g["ref"].to_numpy(float))
                rows.append({"dataset": d, "set": sp["set"], "year": Y, "env": e, "n": len(g), "spearman": sa,
                             "sel_diff": ga, "d_spearman": sa - sb, "d_sel_diff": ga - gb})
    E = pd.DataFrame(rows)
    out = {"coverage_min": min(cover.values()), "cells_scored": int(E["n"].sum()),
           "resolution_3_over_sqrt_N": float(3 / np.sqrt(E["n"].sum())),
           "mean_within_spearman": float(E["spearman"].mean()), "vs_cell_gblup": {}}
    for metric in ("spearman", "sel_diff"):
        ye, skipped = [], []
        for (d, s_, Y), g in E.groupby(["dataset", "set", "year"]):
            v = g[f"d_{metric}"].dropna().to_numpy()
            if len(v) == 0:  # e.g. constant predictions in every environment: the metric is undefined for this year
                skipped.append(f"{d}_{Y}")
                continue
            bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
            ye.append({"dataset": d, "set": s_, "year": Y, "d": float(v.mean()), "se": float(bs.std(ddof=1))})
        if not ye:
            raise SystemExit(f"no target year has a defined {metric} difference; check the submission (constant or missing predictions?)")
        out.setdefault("years_without_defined_metric", {})[metric] = skipped
        ye = floor_se(pd.DataFrame(ye))
        res = {"all": dl(ye)}
        for s_ in ("original", "independent"):
            if (ye["set"] == s_).any():
                res[s_] = dl(ye[ye["set"] == s_])
        for d, g in ye.groupby("dataset"):
            res[d] = dl(g)
        out["vs_cell_gblup"][metric] = res
    json.dump(out, open(a.out, "w"), indent=1)
    r = out["vs_cell_gblup"]["spearman"]["all"]
    print(f"{out['cells_scored']:,} cells in {r['years']} target years; within-environment Spearman vs cell-level GBLUP "
          f"{r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] (resolution {out['resolution_3_over_sqrt_N']:.4f})")
    return out


if __name__ == "__main__":
    main()
