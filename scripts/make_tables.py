"""Tables from prediction files (spec 03 §3.7). Never hand-edit the outputs.

python scripts/make_tables.py --scenario CV1 --ref S1_twohead_mse [--methods ...]
python scripts/make_tables.py --family FYnew2018 FYnew2020 F2022m --name FYnew --ref ...   (year-clustered)
"""
import argparse

import numpy as np
import pandas as pd

from dartgxe.eval.compare import PRACTICAL, paired_delta, resolution, verdict
from dartgxe.eval.metrics import DECISION, PRIMARY, env_seed_matrix, per_env, pooled_pearson
from dartgxe.paths import RESULTS

P = RESULTS / "predictions" / "g2f"


def load(scenario, methods=None):
    frames = []
    for mdir in sorted((P / scenario).iterdir()):
        if methods and mdir.name not in methods:
            continue
        files = sorted(mdir.glob("seed*.parquet")) or sorted((mdir / "parts").glob("seed*_fold*.parquet"))
        frames += [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+")
    ap.add_argument("--family", nargs="+", help="pool these scenarios with a year-clustered bootstrap")
    ap.add_argument("--name")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--confirm", nargs="*", default=[], help="methods whose comparison with --ref is confirmatory")
    ap.add_argument("--level", type=float, default=0.95)
    a = ap.parse_args()
    scen = a.family or a.scenario
    name = a.name or "_".join(scen)
    pred = pd.concat([load(s, a.methods).assign(scenario=s) for s in scen], ignore_index=True)
    if a.family:
        # pool by loss: each year keeps the hyperparameters selected on its own validation year
        pred["method"] = pred["method"].str.replace(r"^(S1_listnet|S1_soft_spearman)_[te][0-9.]+$", r"\1_sel", regex=True)
    pe = per_env(pred)
    out = RESULTS / "metrics" / "g2f" / name
    out.mkdir(parents=True, exist_ok=True)
    pe.to_parquet(out / "per_env.parquet", index=False)

    # N: test cells entering within-env metrics, counted once (one method, one seed)
    one = pe[(pe["method"] == a.ref) & (pe["metric"] == "spearman")]
    one = one[one["seed"] == one["seed"].min()]
    N = int(one["n"].sum())
    thr = resolution(N)
    env_year = pd.Series({e: int(e.rsplit("_", 1)[1]) for e in pe["env"].unique()}) if a.family else None

    rows = []
    for metric in PRIMARY + DECISION:
        mat = env_seed_matrix(pe, metric)
        for m in sorted(mat.index.get_level_values(0).unique()):
            if mat.loc[m].notna().sum().sum() == 0:
                continue
            mv = mat.loc[m].mean(axis=1).dropna()
            rng = np.random.default_rng(0)
            bs = [mv.to_numpy()[rng.integers(0, len(mv), len(mv))].mean() for _ in range(2000)]
            row = {"metric": metric, "method": m, "mean": mv.mean(), "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975),
                   "n_env": len(mv), "n_seed": mat.loc[m].shape[1]}
            if m != a.ref:
                lvl = a.level
                r = paired_delta(mat, m, a.ref, env_year=env_year, level=lvl)
                row.update({k: r[k] for k in ("delta", "ci1_lo", "ci1_hi", "ci2_lo", "ci2_hi", "seed_sd", "seeds_same_sign")})
                if metric in PRIMARY:
                    row["verdict"] = verdict(r["delta"], r["ci2_lo"], r["ci2_hi"], thr, r["seeds_same_sign"])
                    for mult in PRACTICAL:
                        row[f"verdict_x{mult}"] = verdict(r["delta"], r["ci2_lo"], r["ci2_hi"], thr * mult, r["seeds_same_sign"])
                    row["confirmatory"] = m in a.confirm and metric == "spearman"
            rows.append(row)
    tab = pd.DataFrame(rows)
    pp = pooled_pearson(pred).groupby("method")["pooled_pearson_r"].mean()
    tab["pooled_pearson_r_diag"] = tab["method"].map(pp)
    tdir = RESULTS / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(tdir / f"g2f_{name}_main.csv", index=False)
    print(f"{name}: N = {N}, 3/sqrt(N) = {thr:.4f}, practical x1.5 = {thr*1.5:.4f}, x4.2 = {thr*4.2:.4f}, ref = {a.ref}")
    show = tab[tab["metric"].isin(["spearman", "pearson", "sel_diff_f10"])]
    cols = ["metric", "method", "mean", "delta", "ci2_lo", "ci2_hi", "seed_sd", "verdict", "verdict_x1.5", "verdict_x4.2", "pooled_pearson_r_diag"]
    print(show[[c for c in cols if c in show]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
