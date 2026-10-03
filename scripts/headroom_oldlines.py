"""OL headroom check (docs/prereg_headroom_oldlines_2026-09-30.md). Descriptive, not confirmatory.

  python scripts/headroom_oldlines.py panels G2F NUST MU_SOY ...   # one parquet + json per qualifying (dataset, year)
  python scripts/headroom_oldlines.py analyze        # refuses to run until every qualifying panel exists, the qualifying
                                                     # set equals the structure count, and the consistency check passes

Environment: BENCH (benchmark dir: splits.json, predictions_*.parquet), OUT, COUNT_CSV (results/count_oldlines/
per_year.csv), HEADROOM_COMMIT, plus dartgxe.forward.data variables."""
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
from scipy.stats import norm, rankdata

from dartgxe.forward import cellspec, oldlines
from dartgxe.forward.data import LOADERS
from dartgxe.forward.pool import dersimonian_laird, floor_se

BENCH, OUT, COUNT = Path(os.environ["BENCH"]), Path(os.environ["OUT"]), Path(os.environ["COUNT_CSV"])
SPL = json.load(open(BENCH / "splits.json"))
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
LOC_DS = ["G2F", "NUST", "URSN", "ESWYT"]
ORIG, NEW = ["G2F", "NUST", "URSN"], ["ESWYT", "GEM_IA", "MU_SOY"]
B, SEED = 2000, 20260930
MODELS = ["ol0", "ol1", "ol2", "ol3", "ol4"]
HYP = {"H1": ("ol2", "ol1"), "H2": ("ol3", "ol1")}
DESC = {"ol1-ol0": ("ol1", "ol0"), "ol2-ol0": ("ol2", "ol0"), "ol4-ol1": ("ol4", "ol1"), "ol4-ol0": ("ol4", "ol0"),
        "ol3-ol0": ("ol3", "ol0")}


def run_meta():
    import torch
    return {"commit": os.environ.get("HEADROOM_COMMIT", "unknown"), "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version(),
            "config": {"W_GRID": cellspec.W_GRID, "C2_GRID": cellspec.C2_GRID, "K_GL": cellspec.K_GL,
                       "MIN_LOC_YEARS": cellspec.MIN_LOC_YEARS, "MIN_LOC_CELLS": cellspec.MIN_LOC_CELLS,
                       "MIN_ENVS": oldlines.MIN_ENVS, "EM_ITER": oldlines.EM_ITER, "EM_TOL": oldlines.EM_TOL}}


def spearman(y, p):
    if len(y) < 3 or np.std(p) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


def pearson(y, p):
    if len(y) < 3 or np.std(p) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def sel_diff(y, p):
    k = max(1, int(round(0.1 * len(y))))
    top = np.argsort(-p, kind="mergesort")[:k]
    return float(((y - y.mean()) / y.std())[top].mean()) if y.std() > 0 else np.nan


# ------------------------------------------------------------------------------------------------ panels
def panels(names):
    (OUT / "panels").mkdir(parents=True, exist_ok=True)
    for d in names:
        ds = None
        bench = pd.read_parquet(BENCH / f"predictions_{d}.parquet")
        bench = bench[bench["method"] == "cell_reml"].set_index(["env", "genotype"])["pred"]
        for Y in SPL[d]["target_years"]:
            fp = OUT / "panels" / f"{d}_{Y}.parquet"
            skip = OUT / "panels" / f"{d}_{Y}.skip.json"
            if fp.exists() or skip.exists():
                continue
            ds = ds or LOADERS[d]()
            envs = SPL[d]["scorable_environments"][str(Y)]
            scored, qualifies, ok_envs = oldlines.old_line_targets(ds, Y, envs)
            if not qualifies:
                json.dump({"dataset": d, "target": Y, "qualifies": False, "ok_envs": ok_envs, "n_scored": len(scored)},
                          open(skip, "w"), indent=1)
                print(d, Y, "not qualifying", len(ok_envs), "envs", flush=True)
                continue
            t0, start = time.time(), dt.datetime.now().isoformat(timespec="seconds")
            out, meta = oldlines.oldlines_year(ds, Y, envs)
            out["bench_cell_reml"] = bench.reindex(pd.MultiIndex.from_frame(out[["env", "genotype"]])).to_numpy()
            cons = []
            for e, g in out.groupby("env"):
                ok = g[["ol0", "bench_cell_reml"]].dropna()
                cons.append(spearman(ok["ol0"].to_numpy(), ok["bench_cell_reml"].to_numpy()))
            meta.update(run_meta(), dataset=d, target=Y, start=start, end=dt.datetime.now().isoformat(timespec="seconds"),
                        seconds=round(time.time() - t0, 1), consistency_median=float(np.nanmedian(cons)),
                        consistency_min=float(np.nanmin(cons)), n_bench_missing=int(out["bench_cell_reml"].isna().sum()))
            out.to_parquet(fp, index=False)
            json.dump(meta, open(fp.with_suffix(".json"), "w"), indent=1, default=str)
            print(d, Y, f"{meta['seconds']}s", "omega", meta["omega"], "c2", meta.get("c2"), "cells", len(out),
                  "cons", round(meta["consistency_median"], 5), round(meta["consistency_min"], 5), flush=True)


# ------------------------------------------------------------------------------------------------ analysis
def pooled(ye, level=0.95):
    if ye.empty:
        return {}
    r = dersimonian_laird(floor_se(ye), level=level)
    return {k: r[k] for k in ("est", "se", "lo", "hi", "tau2", "k")}


def analyze():
    count = pd.read_csv(COUNT)
    want = sorted((r.dataset, int(r.year)) for r in count[count["qualifies"]].itertuples())
    have, skipped, missing = [], [], []
    for d in DS:
        for Y in SPL[d]["target_years"]:
            fp = OUT / "panels" / f"{d}_{Y}.parquet"
            if fp.exists():
                have.append((d, Y))
            elif (OUT / "panels" / f"{d}_{Y}.skip.json").exists():
                skipped.append((d, Y))
            else:
                missing.append((d, Y))
    if missing:
        sys.exit(f"panels missing ({len(missing)}): {missing[:10]}")
    if sorted(have) != want:
        sys.exit(f"qualifying set differs from the structure count: panels {sorted(have)} vs count {want}")
    metas = {f"{d}_{Y}": json.load(open(OUT / "panels" / f"{d}_{Y}.json")) for d, Y in have}
    cons = []
    for (d, Y) in have:
        g = pd.read_parquet(OUT / "panels" / f"{d}_{Y}.parquet")
        for e, gg in g.groupby("env"):
            ok = gg[["ol0", "bench_cell_reml"]].dropna()
            cons.append({"panel": f"{d}_{Y}", "env": e, "rho": spearman(ok["ol0"].to_numpy(), ok["bench_cell_reml"].to_numpy()),
                         "n": len(gg), "bench_missing": int(gg["bench_cell_reml"].isna().sum())})
    C = pd.DataFrame(cons)
    consistency = {"median": float(C["rho"].median()), "min": float(C["rho"].min()),
                   "bench_missing": int(C["bench_missing"].sum()), "n_env": len(C)}
    consistency["pass"] = bool(consistency["median"] >= 0.999 and consistency["min"] >= 0.99)
    json.dump({**consistency, "per_env": C.to_dict("records")}, open(OUT / "consistency.json", "w"), indent=1, default=float)
    if not consistency["pass"]:
        sys.exit(f"consistency check failed (prereg §5: fix the implementation first): {consistency}")
    choices = pd.DataFrame([{"panel": k, "omega": m["omega"], "c2": m.get("c2"), "n_active_locations": m.get("n_active_locations"),
                             "ol3_s2v": m.get("ol3_em", {}).get("s2v"), "ol3_s2e": m.get("ol3_em", {}).get("s2e"),
                             "ol3_em_converged": m.get("ol3_em", {}).get("converged"),
                             "ol4_s2v": m.get("ol4_em", {}).get("s2v"), "ol4_em_converged": m.get("ol4_em", {}).get("converged"),
                             "share_active": m.get("share_scored_active_location"),
                             "share_line_loc_history": m.get("share_scored_with_line_location_history"),
                             "any_at_edge": bool(m.get("any_at_edge_omega") or m.get("any_at_edge_c2")),
                             "n_fit": m["n_fit"], "n_scored": m["n_scored"], "seconds": m["seconds"], "gpu": m["gpu"]}
                            for k, m in metas.items()])
    choices.to_csv(OUT / "choices.csv", index=False)
    rows = []
    for d, Y in have:
        P = pd.read_parquet(OUT / "panels" / f"{d}_{Y}.parquet")
        min_n = SPL[d]["min_genotypes"]
        for e, g in P.groupby("env"):
            if len(g) < min_n:
                continue
            y = g["y"].to_numpy(float)
            r = {"dataset": d, "target": Y, "env": e, "n": len(g)}
            for m in MODELS:
                if m in g:
                    p = g[m].to_numpy(float)
                    r[f"sp_{m}"], r[f"pe_{m}"], r[f"sd_{m}"] = spearman(y, p), pearson(y, p), sel_diff(y, p)
            rows.append(r)
    E = pd.DataFrame(rows)
    E.to_parquet(OUT / "per_env.parquet", index=False)
    rng = np.random.default_rng(SEED)
    ye_rows = []
    for name, (a, b) in {**HYP, **DESC}.items():
        for metric in ("sp", "pe", "sd"):
            if f"{metric}_{a}" not in E:
                continue
            env = E.assign(d=E[f"{metric}_{a}"] - E[f"{metric}_{b}"]).dropna(subset=["d"])
            for (d, Y), g in env.groupby(["dataset", "target"]):
                v = g["d"].to_numpy()
                bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
                ye_rows.append({"contrast": name, "metric": metric, "dataset": d, "target": Y, "d": float(v.mean()),
                                "se": float(bs.std(ddof=1)), "n_env": len(v), "n_cells": int(g["n"].sum())})
    YE = pd.DataFrame(ye_rows)
    YE.to_csv(OUT / "year_effects.csv", index=False)
    present = sorted(E["dataset"].unique())
    ranges = [("all", present), ("loc", [d for d in present if d in LOC_DS]), ("orig", [d for d in present if d in ORIG]),
              ("new", [d for d in present if d in NEW])] + [(d, [d]) for d in present]
    Pr = []
    for (name, metric), g in YE.groupby(["contrast", "metric"]):
        for rname, dsets in ranges:
            sub = g[g["dataset"].isin(dsets)]
            if sub.empty:
                continue
            r = pooled(sub[["dataset", "d", "se"]])
            Pr.append({"contrast": name, "metric": metric, "range": rname, "N": int(sub["n_cells"].sum()),
                       "resolution": 3 / np.sqrt(sub["n_cells"].sum()), **r})
    PO = pd.DataFrame(Pr)
    PO.to_csv(OUT / "pooled.csv", index=False)
    # decision (prereg §4): H1, H2 on the location datasets, Spearman, Holm (97.5% then 95%)
    main = PO[(PO["metric"] == "sp") & (PO["range"] == "loc") & PO["contrast"].isin(list(HYP))].copy()
    main = main.assign(p=2 * norm.sf(np.abs(main["est"] / main["se"]))).sort_values("p").reset_index(drop=True)
    levels = [1 - 0.05 / 2, 0.95]
    verdict, stop = {}, False
    for i, row in main.iterrows():
        h = row["contrast"]
        g = YE[(YE["contrast"] == h) & (YE["metric"] == "sp") & YE["dataset"].isin(LOC_DS)]
        ci = pooled(g[["dataset", "d", "se"]], level=levels[i])
        dsets = sorted(g["dataset"].unique())
        lodo = {d: pooled(g[g["dataset"] != d][["dataset", "d", "se"]]).get("est") for d in dsets}
        single = {d: pooled(g[g["dataset"] == d][["dataset", "d", "se"]]).get("est") for d in dsets}
        robust = all(x is not None and x > 0 for x in lodo.values()) and (len(dsets) > 2 or all(x > 0 for x in single.values()))
        go = (not stop) and ci["lo"] > 0 and row["est"] >= row["resolution"] and robust
        if not ci["lo"] > 0:
            stop = True                                    # Holm: later hypotheses are not rejected once one fails
        verdict[h] = {"contrast": f"{HYP[h][0]} - {HYP[h][1]}", "GO": bool(go), "est": row["est"], "p": row["p"],
                      "holm_level": levels[i], "ci": [ci["lo"], ci["hi"]], "resolution": row["resolution"], "N": row["N"],
                      "lodo": lodo, "single_dataset": single, "k": row["k"]}
    json.dump({"decisions": verdict, "consistency": consistency, "units": [f"{d}_{Y}" for d, Y in have],
               "skipped": [f"{d}_{Y}" for d, Y in skipped],
               "note": "descriptive headroom check; GO = worth a confirmatory pre-registration on data not used here"},
              open(OUT / "verdict.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(PO[(PO["metric"] == "sp") & PO["range"].isin(["all", "loc", "G2F", "NUST", "MU_SOY"])].round(4).to_string(index=False))
    print(choices.round(4).to_string(index=False))
    print(json.dumps(verdict, indent=1, default=float))
    print("OLDLINES_ANALYZE_DONE")


if __name__ == "__main__":
    if sys.argv[1] == "panels":
        panels(sys.argv[2:] or DS)
    elif sys.argv[1] == "analyze":
        analyze()
