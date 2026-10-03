"""Headroom checks I-A / I-B (docs/prereg_headroom_ia_ib_2026-09-29.md). Descriptive, not confirmatory.

  python scripts/headroom_ia_ib.py panels G2F NUST ...   # one parquet + meta json per target year (skips done ones)
  python scripts/headroom_ia_ib.py analyze                # refuses to run until all 48 panels exist

Environment: BENCH = benchmark package dir (splits.json, predictions_*.parquet), OUT = output dir,
HEADROOM_COMMIT = git commit of this code, plus the data variables of dartgxe.forward.data."""
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

from dartgxe.forward import cellspec
from dartgxe.forward.data import LOADERS
from dartgxe.forward.pool import dersimonian_laird, floor_se

BENCH = Path(os.environ["BENCH"])
OUT = Path(os.environ["OUT"])
SPL = json.load(open(BENCH / "splits.json"))
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
LOC_DS = ["G2F", "NUST", "URSN", "ESWYT"]
ORIG, NEW = ["G2F", "NUST", "URSN"], ["ESWYT", "GEM_IA", "MU_SOY"]
B, SEED = 2000, 20260929


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
    return {"commit": os.environ.get("HEADROOM_COMMIT", "unknown"), "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "config": {"W_GRID": cellspec.W_GRID, "C2_GRID": cellspec.C2_GRID, "K_GL": cellspec.K_GL,
                       "MIN_LOC_YEARS": cellspec.MIN_LOC_YEARS, "MIN_LOC_CELLS": cellspec.MIN_LOC_CELLS}}


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
            t0 = time.time()
            start = dt.datetime.now().isoformat(timespec="seconds")
            out, meta = cellspec.panel_year(ds, Y, SPL[name]["scorable_environments"][str(Y)])
            out["bench_cell_reml"] = bench.reindex(pd.MultiIndex.from_frame(out[["env", "genotype"]])).to_numpy()
            cons = []
            for e, g in out.groupby("env"):
                ok = g[["ctrl", "bench_cell_reml"]].dropna()
                cons.append(spearman(ok["ctrl"].to_numpy(), ok["bench_cell_reml"].to_numpy()))
            meta.update(run_meta(), dataset=name, target=Y, start=start, end=dt.datetime.now().isoformat(timespec="seconds"),
                        seconds=round(time.time() - t0, 1), consistency_median=float(np.nanmedian(cons)),
                        consistency_min=float(np.nanmin(cons)), n_cells=int(len(out)),
                        n_bench_missing=int(out["bench_cell_reml"].isna().sum()))
            out.to_parquet(f, index=False)
            json.dump(meta, open(f.with_suffix(".json"), "w"), indent=1, default=str)
            print(name, Y, f"{meta['seconds']}s", "omega", meta["ia"]["omega_main"], "c2",
                  meta.get("ib_main", {}).get("c2_main"), "cons", round(meta["consistency_median"], 5),
                  round(meta["consistency_min"], 5), flush=True)


# ------------------------------------------------------------------------------------------------ analysis
def per_env(P: pd.DataFrame, first_year: pd.Series, name: str, Y: int, variants) -> list[dict]:
    rows = []
    P = P.assign(new=~P["genotype"].map(first_year.lt(Y)).fillna(False).astype(bool))
    for e, g in P.groupby("env"):
        for scope, gg in (("all", g), ("new", g[g["new"]])):
            if scope == "new" and len(gg) < 10:
                continue
            y = gg["y"].to_numpy(float)
            s0, d0 = spearman(y, gg["ctrl"].to_numpy(float)), sel_diff(y, gg["ctrl"].to_numpy(float))
            for v in variants:
                if v not in gg:
                    continue
                p = gg[v].to_numpy(float)
                rows.append({"dataset": name, "target": Y, "env": e, "scope": scope, "n": len(gg), "variant": v,
                             "active": bool(gg["ib_active"].all()) if "ib_active" in gg else np.nan,
                             "d_spearman": spearman(y, p) - s0, "d_sel_diff": sel_diff(y, p) - d0,
                             "spearman_ctrl": s0})
    return rows


def year_effects(E: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for key, g in E.groupby(["dataset", "target", "scope", "variant"]):
        for m in ("d_spearman", "d_sel_diff"):
            v = g[m].dropna().to_numpy()
            if not len(v):
                continue
            bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
            rows.append(dict(zip(["dataset", "target", "scope", "variant"], key), metric=m[2:], d=float(v.mean()),
                             se=float(bs.std(ddof=1)), n_env=len(v)))
    return pd.DataFrame(rows)


def pooled(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    r = dersimonian_laird(floor_se(sub))
    return {k: r[k] for k in ("est", "lo", "hi", "tau2", "k")}


def cross_fit(D: np.ndarray, rng, n_split=200) -> float:
    n, vals = len(D), []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2:]
        for s, t in ((a, b), (b, a)):
            vals.append(D[t, int(np.argmax(D[s].mean(0)))].mean())
    return float(np.mean(vals))


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
    json.dump({"pass": ok, "rule": "per target year: median >= 0.999 and min >= 0.99 (prereg §4)",
               "worst_median": float(cons["median"].min()), "worst_min": float(cons["min"].min()),
               "per_panel": cons.to_dict("records")}, open(OUT / "consistency.json", "w"), indent=1)
    if not ok:
        sys.exit("consistency check failed: see consistency.json (prereg §4: fix the implementation first)")

    choices = []
    for k, m in metas.items():
        choices.append({"panel": k, "omega_main": m["ia"]["omega_main"], "omega_g": m["ia"]["omega_g"],
                        "s_min": m["ia"]["s_min"], "s_max": m["ia"]["s_max"], "ia_any_at_edge": m["ia"]["any_at_edge"],
                        "c2_main": m.get("ib_main", {}).get("c2_main"), "n_active": m.get("ib_main", {}).get("n_active"),
                        "c2_region": m.get("ib_region", {}).get("c2_main"),
                        "ib_any_at_edge": m.get("ib_main", {}).get("any_at_edge"), "seconds": m["seconds"],
                        "n_cells": m["n_cells"]})
    pd.DataFrame(choices).to_csv(OUT / "choices.csv", index=False)

    rows, grids = [], {}
    for d in DS:
        panels_d = {Y: pd.read_parquet(OUT / "panels" / f"{d}_{Y}.parquet") for Y in SPL[d]["target_years"]}
        # first year each genotype was phenotyped (for the 'new genotype' scope): from the loader's cells
        first = LOADERS[d]().cells.groupby("genotype")["year"].min()
        for Y, P in panels_d.items():
            variants = [c for c in P.columns if c.startswith(("ia_", "iaw_", "ib_main", "ib_region", "ibc_"))]
            rows += per_env(P, first, d, Y, variants)
    E = pd.DataFrame(rows)
    E.to_parquet(OUT / "per_env.parquet", index=False)
    YE = year_effects(E, rng)
    YE.to_csv(OUT / "year_effects.csv", index=False)
    n_cells = pd.DataFrame(choices).assign(dataset=lambda x: x["panel"].str.rsplit("_", n=1).str[0])
    N = n_cells.groupby("dataset")["n_cells"].sum()

    def res(dsets):
        return 3 / np.sqrt(N[list(dsets)].sum())

    P_rows = []

    def add(variant, scope_name, dsets, metric="spearman", scope="all", years=None):
        sub = YE[(YE["variant"] == variant) & (YE["metric"] == metric) & (YE["scope"] == scope) & YE["dataset"].isin(dsets)]
        r = pooled(sub[["dataset", "d", "se"]])
        if r:
            P_rows.append({"variant": variant, "range": scope_name, "metric": metric, "scope": scope,
                           "resolution": res(dsets), **r})
        return r

    ia_vars = ["ia_main", "ia_g", "ia_het", "ia_std"]
    for v in ia_vars:
        for metric in ("spearman", "sel_diff"):
            for scope in ("all", "new"):
                add(v, "all48", DS, metric, scope)
                add(v, "orig31", ORIG, metric, scope)
                add(v, "new17", NEW, metric, scope)
                for d in DS:
                    add(v, d, [d], metric, scope)
    for v in ["ib_main", "ib_region"]:
        dsets = LOC_DS if v == "ib_main" else ["G2F", "NUST", "ESWYT"]
        for metric in ("spearman", "sel_diff"):
            for scope in ("all", "new"):
                add(v, f"loc{len(dsets)}", dsets, metric, scope)
                for d in dsets:
                    add(v, d, [d], metric, scope)
    PO = pd.DataFrame(P_rows)
    PO.to_csv(OUT / "pooled.csv", index=False)

    # leave one dataset out (decision variants, Spearman, all genotypes)
    lodo = []
    for v, dsets in (("ia_main", DS), ("ib_main", LOC_DS)):
        for d in dsets:
            rest = [x for x in dsets if x != d]
            sub = YE[(YE["variant"] == v) & (YE["metric"] == "spearman") & (YE["scope"] == "all") & YE["dataset"].isin(rest)]
            lodo.append({"variant": v, "left_out": d, **pooled(sub[["dataset", "d", "se"]])})
    LO = pd.DataFrame(lodo)
    LO.to_csv(OUT / "lodo.csv", index=False)

    # active vs non-active target environments for ib_main
    act = []
    for flag in (True, False):
        e = E[(E["variant"] == "ib_main") & (E["scope"] == "all") & (E["active"] == flag)]
        ye = year_effects(e.assign(variant=f"ib_main_active_{flag}"), rng)
        ye = ye[ye["metric"] == "spearman"]
        act.append({"active": flag, "n_env": len(e), **pooled(ye[["dataset", "d", "se"]])})
    pd.DataFrame(act).to_csv(OUT / "ib_active_split.csv", index=False)

    # cross-fitted upper bounds over the grids (Spearman, all genotypes)
    cf_rows = []
    for fam, prefix, base, dsets in (("ia_omega", "iaw_", "iaw_0", DS), ("ib_c2", "ibc_", None, LOC_DS)):
        for (d, Y), g in E[(E["scope"] == "all") & E["dataset"].isin(dsets)].groupby(["dataset", "target"]):
            W = g[g["variant"].str.startswith(prefix)].pivot_table(index="env", columns="variant", values="d_spearman")
            if base is not None:
                W = W.sub(W[base], axis=0)          # relative to omega = 0 with the same weights (= ia_het)
            else:
                W["ibc_0"] = 0.0                    # c2 = 0 is the control
            W = W.dropna()
            D = W.to_numpy(float)
            est = cross_fit(D, rng)
            idx = rng.integers(0, len(D), (200, len(D)))
            se = float(np.std([cross_fit(D[i], rng, 20) for i in idx], ddof=1))
            cf_rows.append({"family": fam, "dataset": d, "target": Y, "d": est, "se": se, "n_env": len(D)})
    CF = pd.DataFrame(cf_rows)
    CF.to_csv(OUT / "crossfit_years.csv", index=False)
    cfp = []
    for fam, g in CF.groupby("family"):
        cfp.append({"family": fam, "range": "all", **pooled(g[["dataset", "d", "se"]])})
        for d, gg in g.groupby("dataset"):
            cfp.append({"family": fam, "range": d, **pooled(gg[["dataset", "d", "se"]])})
    pd.DataFrame(cfp).to_csv(OUT / "crossfit_pooled.csv", index=False)

    # verdict (prereg §3)
    def get(v, rng_):
        r = PO[(PO["variant"] == v) & (PO["range"] == rng_) & (PO["metric"] == "spearman") & (PO["scope"] == "all")]
        return r.iloc[0].to_dict()

    ia48, ia31, ia17 = get("ia_main", "all48"), get("ia_main", "orig31"), get("ia_main", "new17")
    ia_lodo = LO[LO["variant"] == "ia_main"]["est"]
    ia_go = (ia48["est"] >= ia48["resolution"] and ia48["lo"] > 0 and bool((ia_lodo > 0).all())
             and not (ia31["est"] >= ia31["resolution"] and ia17["est"] < 0))
    ib43, nust = get("ib_main", "loc4"), get("ib_main", "NUST")
    ib_lodo = LO[LO["variant"] == "ib_main"]["est"]
    ib_go_general = ib43["est"] >= ib43["resolution"] and ib43["lo"] > 0 and bool((ib_lodo > 0).all())
    ib_go_nust = nust["est"] >= nust["resolution"] and nust["lo"] > 0
    verdict = {"I-A": {"GO": bool(ia_go), "all48": ia48, "orig31": ia31, "new17": ia17,
                       "lodo_min": float(ia_lodo.min())},
               "I-B": {"GO_general": bool(ib_go_general), "GO_NUST": bool(ib_go_nust), "loc43": ib43, "NUST": nust,
                       "lodo_min": float(ib_lodo.min())},
               "note": "descriptive headroom check; GO = worth a confirmatory pre-registration on new data"}
    json.dump(verdict, open(OUT / "verdict.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(PO[(PO["metric"] == "spearman")].round(4).to_string(index=False))
    print(LO.round(4).to_string(index=False))
    print(pd.DataFrame(act).round(4).to_string(index=False))
    print(pd.DataFrame(cfp).round(4).to_string(index=False))
    print(json.dumps(verdict, indent=1, default=float))
    print("HEADROOM_ANALYZE_DONE")


if __name__ == "__main__":
    if sys.argv[1] == "panels":
        panels(sys.argv[2:] or DS)
    elif sys.argv[1] == "analyze":
        analyze()
