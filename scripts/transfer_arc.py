"""Transfer test of the arc-cosine kernel (docs/prereg_transfer_arc_kernel_2026-10-02.md).

  python scripts/transfer_arc.py panels G2F NUST ...   # one parquet + meta json per target year (skips done ones)
  python scripts/transfer_arc.py analyze                # refuses to run until all 48 panels exist and the control passes

Environment: BENCH = benchmark package dir (splits.json, predictions_*.parquet), OUT = output dir,
TRANSFER_COMMIT = git commit of this code, plus the data variables of dartgxe.forward.data."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from dartgxe.forward import cellarc
from dartgxe.forward.data import LOADERS
from dartgxe.forward.pool import dersimonian_laird, floor_se

BENCH = Path(os.environ["BENCH"])
OUT = Path(os.environ["OUT"])
SPL = json.load(open(BENCH / "splits.json"))
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
ORIG, NEW = ["G2F", "NUST", "URSN"], ["ESWYT", "GEM_IA", "MU_SOY"]
B, SEED = 2000, 20261002


def spearman(y, p):
    if len(y) < 3 or np.std(p) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


def sel_diff(y, p):
    k = max(1, int(round(0.1 * len(y))))
    top = np.argsort(-p, kind="mergesort")[:k]
    return float(((y - y.mean()) / y.std())[top].mean()) if y.std() > 0 else np.nan


def run_meta():
    import torch
    return {"commit": os.environ.get("TRANSFER_COMMIT", "unknown"), "host": platform.node(),
            "device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
            "python": platform.python_version(), "numpy": np.__version__, "threads": os.environ.get("OMP_NUM_THREADS"),
            "config": {"MIN_EV": cellarc.MIN_EV, "kernel": "arc-cosine order 1"}}


# ------------------------------------------------------------------------------------------------ panels
def panels(names):
    (OUT / "panels").mkdir(parents=True, exist_ok=True)
    for name in names:
        ds = None
        bench = pd.read_parquet(BENCH / f"predictions_{name}.parquet")
        bench = bench[bench["method"] == "cell_reml"].set_index(["env", "genotype"])["pred"]
        for Y in SPL[name]["target_years"]:
            f = OUT / "panels" / f"{name}_{Y}.parquet"
            if f.exists():
                continue
            ds = ds or LOADERS[name]()
            t0, start = time.time(), dt.datetime.now().isoformat(timespec="seconds")
            out, meta = cellarc.panel_year(ds, Y, SPL[name]["scorable_environments"][str(Y)])
            out["bench_cell_reml"] = bench.reindex(pd.MultiIndex.from_frame(out[["env", "genotype"]])).to_numpy()
            cons = []
            for _, g in out.groupby("env"):
                ok = g[["ctrl", "bench_cell_reml"]].dropna()
                cons.append(spearman(ok["ctrl"].to_numpy(), ok["bench_cell_reml"].to_numpy()))
            meta.update(run_meta(), dataset=name, target=Y, start=start, end=dt.datetime.now().isoformat(timespec="seconds"),
                        seconds=round(time.time() - t0, 1), consistency_median=float(np.nanmedian(cons)),
                        consistency_min=float(np.nanmin(cons)), n_cells=int(len(out)),
                        n_bench_missing=int(out["bench_cell_reml"].isna().sum()))
            out.to_parquet(f, index=False)
            json.dump(meta, open(f.with_suffix(".json"), "w"), indent=1, default=str)
            # no accuracy is printed here: scoring happens once, in analyze()
            print(name, Y, f"{meta['seconds']}s", "rank", meta["arc"]["rank"], "n_fit", meta["arc"]["n_fit"], "cons",
                  round(meta["consistency_median"], 5), round(meta["consistency_min"], 5), flush=True)


# ------------------------------------------------------------------------------------------------ analysis
def per_env(P: pd.DataFrame, first_year: pd.Series, name: str, Y: int) -> list[dict]:
    rows = []
    P = P.assign(new=~P["genotype"].map(first_year.lt(Y)).fillna(False).astype(bool))
    for e, g in P.groupby("env"):
        for scope, gg in (("all", g), ("new", g[g["new"]])):
            if scope == "new" and len(gg) < 10:
                continue
            y, c, a = gg["y"].to_numpy(float), gg["ctrl"].to_numpy(float), gg["cell_arc"].to_numpy(float)
            rows.append({"dataset": name, "target": Y, "env": e, "scope": scope, "n": len(gg),
                         "spearman_ctrl": spearman(y, c), "spearman_arc": spearman(y, a),
                         "d_spearman": spearman(y, a) - spearman(y, c), "d_sel_diff": sel_diff(y, a) - sel_diff(y, c)})
    return rows


def year_effects(E: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for key, g in E.groupby(["dataset", "target", "scope"]):
        for m in ("d_spearman", "d_sel_diff"):
            v = g[m].dropna().to_numpy()
            if not len(v):
                continue
            bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
            rows.append(dict(zip(["dataset", "target", "scope"], key), metric=m[2:], d=float(v.mean()),
                             se=float(bs.std(ddof=1)), n_env=len(v)))
    return pd.DataFrame(rows)


def judge(est, lo, hi, res):
    if lo > 0 and est >= res:
        return "resolved gain"
    if hi < 0 and -est >= res:
        return "resolved loss"
    if lo > 0 or hi < 0:
        return "detectable, below resolution"
    return "tied"


def analyze():
    need = [(d, Y) for d in DS for Y in SPL[d]["target_years"]]
    missing = [f"{d}_{Y}" for d, Y in need if not (OUT / "panels" / f"{d}_{Y}.parquet").exists()]
    if missing:
        sys.exit(f"panels missing: {missing}")
    rng = np.random.default_rng(SEED)
    metas = {f"{d}_{Y}": json.load(open(OUT / "panels" / f"{d}_{Y}.json")) for d, Y in need}
    cons = pd.DataFrame([{"panel": k, "median": m["consistency_median"], "min": m["consistency_min"],
                          "bench_missing": m["n_bench_missing"]} for k, m in metas.items()])
    ok = bool((cons["median"] >= 0.999).all() and (cons["min"] >= 0.99).all())
    json.dump({"pass": ok, "rule": "per target year: median >= 0.999 and min >= 0.99 (prereg §3)",
               "worst_median": float(cons["median"].min()), "worst_min": float(cons["min"].min()),
               "per_panel": cons.to_dict("records")}, open(OUT / "consistency.json", "w"), indent=1)
    if not ok:
        sys.exit("control check failed: see consistency.json (prereg §3: fix the implementation first)")
    pd.DataFrame([{"panel": k, "n_cells": m["n_cells"], "n_fit": m["arc"]["n_fit"], "rank": m["arc"]["rank"],
                   "h2_arc": m["arc"]["h2"], "arc_at_grid_edge": m["arc"]["at_grid_edge"], "seconds": m["seconds"],
                   "commit": m["commit"], "device": m["device"], "start": m["start"], "end": m["end"]}
                  for k, m in metas.items()]).to_csv(OUT / "runs.csv", index=False)

    rows = []
    for d in DS:
        first = LOADERS[d]().cells.groupby("genotype")["year"].min()
        for Y in SPL[d]["target_years"]:
            rows += per_env(pd.read_parquet(OUT / "panels" / f"{d}_{Y}.parquet"), first, d, Y)
    E = pd.DataFrame(rows)
    E.to_parquet(OUT / "per_env.parquet", index=False)
    YE = year_effects(E, rng)
    YE.to_csv(OUT / "year_effects.csv", index=False)
    N = pd.Series({k.rsplit("_", 1)[0]: 0 for k in metas})
    for k, m in metas.items():
        N[k.rsplit("_", 1)[0]] += m["n_cells"]

    P_rows = []

    def add(range_name, dsets, metric="spearman", scope="all"):
        sub = YE[(YE["metric"] == metric) & (YE["scope"] == scope) & YE["dataset"].isin(dsets)]
        if sub.empty:
            return None
        r = dersimonian_laird(floor_se(sub[["dataset", "d", "se"]]))
        res = 3 / np.sqrt(N[list(dsets)].sum())
        row = {"range": range_name, "metric": metric, "scope": scope, "N": int(N[list(dsets)].sum()), "resolution": res,
               **{k: r[k] for k in ("est", "se", "lo", "hi", "tau2", "k")}, "years_positive": int((sub["d"] > 0).sum())}
        if metric == "spearman":
            row["judgement"] = judge(r["est"], r["lo"], r["hi"], res)
        P_rows.append(row)
        return row

    for metric in ("spearman", "sel_diff"):
        for scope in ("all", "new"):
            add("all48", DS, metric, scope)
            add("orig31", ORIG, metric, scope)
            add("new17", NEW, metric, scope)
            for d in DS:
                add(d, [d], metric, scope)
    PO = pd.DataFrame(P_rows)
    PO.to_csv(OUT / "pooled.csv", index=False)
    lodo = []
    for d in DS:
        sub = YE[(YE["metric"] == "spearman") & (YE["scope"] == "all") & (YE["dataset"] != d)]
        r = dersimonian_laird(floor_se(sub[["dataset", "d", "se"]]))
        lodo.append({"left_out": d, **{k: r[k] for k in ("est", "lo", "hi", "k")}})
    LO = pd.DataFrame(lodo)
    LO.to_csv(OUT / "lodo.csv", index=False)

    def get(rng_):
        return PO[(PO["range"] == rng_) & (PO["metric"] == "spearman") & (PO["scope"] == "all")].iloc[0].to_dict()

    h1, g2f = get("all48"), get("G2F")
    if h1["judgement"] == "resolved gain":
        outcome = "H1 resolved gain"
    elif h1["judgement"] == "resolved loss":
        outcome = "H1 resolved loss"
    elif g2f["judgement"] == "resolved gain":
        outcome = "H1 not resolved, G2F alone a resolved gain (descriptive)"
    else:
        outcome = "H1 not resolved, G2F alone not a resolved gain"
    json.dump({"H1": h1, "G2F_descriptive": g2f, "outcome": outcome, "lodo_min": float(LO["est"].min()),
               "lodo_max": float(LO["est"].max())}, open(OUT / "verdict.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(PO.round(4).to_string(index=False))
    print(LO.round(4).to_string(index=False))
    print(outcome)
    print("TRANSFER_ANALYZE_DONE")


if __name__ == "__main__":
    if sys.argv[1] == "panels":
        panels(sys.argv[2:] or DS)
    elif sys.argv[1] == "analyze":
        analyze()
