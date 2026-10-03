"""SP headroom check (docs/prereg_headroom_sparse_2026-09-29.md). Descriptive, not confirmatory.

  python scripts/headroom_sparse.py panels G2F NUST ...   # one parquet + json per (dataset, year, f, seed)
  python scripts/headroom_sparse.py analyze                # refuses to run until every panel exists

Environment: BENCH (benchmark dir with splits.json), OUT, HEADROOM_COMMIT, plus dartgxe.forward.data variables."""
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

from dartgxe.forward import sparse
from dartgxe.forward.data import LOADERS
from dartgxe.forward.pool import dersimonian_laird, floor_se

BENCH, OUT = Path(os.environ["BENCH"]), Path(os.environ["OUT"])
SPL = json.load(open(BENCH / "splits.json"))
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
ORIG, NEW = ["G2F", "NUST", "URSN"], ["ESWYT", "GEM_IA", "MU_SOY"]
B, SEED = 2000, 20260930


def tag(d, y, f, s):
    return f"{d}_{y}_f{int(round(f * 100))}_s{s}"


def run_meta():
    import torch
    return {"commit": os.environ.get("HEADROOM_COMMIT", "unknown"), "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version(),
            "config": {"W_GRID": sparse.W_GRID, "C2_GRID": sparse.C2_GRID, "RHO_GRID": sparse.RHO_GRID,
                       "K_GL": sparse.K_GL, "FRACTIONS": sparse.FRACTIONS, "SEEDS": sparse.SEEDS}}


def panels(names):
    (OUT / "panels").mkdir(parents=True, exist_ok=True)
    for d in names:
        ds = None
        for Y in SPL[d]["target_years"]:
            for f in sparse.FRACTIONS:
                for s in sparse.SEEDS:
                    fp = OUT / "panels" / f"{tag(d, Y, f, s)}.parquet"
                    if fp.exists():
                        continue
                    ds = ds or LOADERS[d]()
                    t0, start = time.time(), dt.datetime.now().isoformat(timespec="seconds")
                    out, meta = sparse.sparse_year(ds, Y, SPL[d]["scorable_environments"][str(Y)], f, s)
                    meta.update(run_meta(), dataset=d, target=Y, fraction=f, seed=s, start=start,
                                end=dt.datetime.now().isoformat(timespec="seconds"), seconds=round(time.time() - t0, 1))
                    out.to_parquet(fp, index=False)
                    json.dump(meta, open(fp.with_suffix(".json"), "w"), indent=1, default=str)
                    print(d, Y, f, s, f"{meta['seconds']}s", "omega", meta["omega"], "c2", meta["c2"], "rho", meta["rho"],
                          "envs", meta["n_target_envs"], flush=True)


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


def pooled(ye):
    if ye.empty:
        return {}
    r = dersimonian_laird(floor_se(ye))
    return {k: r[k] for k in ("est", "lo", "hi", "tau2", "k")}


def analyze():
    need = [(d, Y, f, s) for d in DS for Y in SPL[d]["target_years"] for f in sparse.FRACTIONS for s in sparse.SEEDS]
    missing = [tag(*n) for n in need if not (OUT / "panels" / f"{tag(*n)}.parquet").exists()]
    if missing:
        sys.exit(f"panels missing ({len(missing)}): {missing[:10]}")
    rng = np.random.default_rng(SEED)
    rows, metas = [], []
    for d, Y, f, s in need:
        P = pd.read_parquet(OUT / "panels" / f"{tag(d, Y, f, s)}.parquet")
        metas.append(json.load(open(OUT / "panels" / f"{tag(d, Y, f, s)}.json")))
        min_n = SPL[d]["min_genotypes"]
        for e, g in P.groupby("env"):
            if len(g) < min_n:
                continue
            y = g["y"].to_numpy(float)
            r = {"dataset": d, "target": Y, "fraction": f, "seed": s, "env": e, "n": len(g)}
            for m in ("m0", "m1", "m2"):
                p = g[m].to_numpy(float)
                r[f"sp_{m}"], r[f"pe_{m}"], r[f"sd_{m}"] = spearman(y, p), pearson(y, p), sel_diff(y, p)
            rows.append(r)
    E = pd.DataFrame(rows)
    E.to_parquet(OUT / "per_env_seed.parquet", index=False)
    M = pd.DataFrame(metas)
    M[["dataset", "target", "fraction", "seed", "omega", "c2", "rho", "k", "n_target_envs", "n_observed", "n_scored",
       "any_at_edge", "envs_without_observed", "seconds"]].to_csv(OUT / "choices.csv", index=False)
    N = {f: int(E[(E["fraction"] == f) & (E["seed"] == 0)]["n"].sum()) for f in sparse.FRACTIONS}
    contrasts = {"m2-m1": ("m2", "m1"), "m1-m0": ("m1", "m0"), "m2-m0": ("m2", "m0")}
    ye_rows = []
    for f in sparse.FRACTIONS:
        Ef = E[E["fraction"] == f]
        for name, (a, b) in contrasts.items():
            for metric in ("sp", "pe", "sd"):
                env = Ef.assign(d=Ef[f"{metric}_{a}"] - Ef[f"{metric}_{b}"]).groupby(["dataset", "target", "env"])["d"].mean()
                env = env.dropna().reset_index()
                for (d, Y), g in env.groupby(["dataset", "target"]):
                    v = g["d"].to_numpy()
                    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
                    ye_rows.append({"fraction": f, "contrast": name, "metric": metric, "dataset": d, "target": Y,
                                    "d": float(v.mean()), "se": float(bs.std(ddof=1)), "n_env": len(v)})
    YE = pd.DataFrame(ye_rows)
    YE.to_csv(OUT / "year_effects.csv", index=False)
    Pr = []
    for (f, name, metric), g in YE.groupby(["fraction", "contrast", "metric"]):
        for rng_name, dsets in [("all48", DS), ("orig31", ORIG), ("new17", NEW)] + [(d, [d]) for d in DS]:
            sub = g[g["dataset"].isin(dsets)]
            r = pooled(sub[["dataset", "d", "se"]])
            if r:
                Pr.append({"fraction": f, "contrast": name, "metric": metric, "range": rng_name,
                           "resolution": 3 / np.sqrt(N[f]), **r})
    PO = pd.DataFrame(Pr)
    PO.to_csv(OUT / "pooled.csv", index=False)
    lodo = []
    g = YE[(YE["fraction"] == 0.25) & (YE["contrast"] == "m2-m1") & (YE["metric"] == "sp")]
    for d in sorted(g["dataset"].unique()):
        lodo.append({"left_out": d, **pooled(g[g["dataset"] != d][["dataset", "d", "se"]])})
    LO = pd.DataFrame(lodo)
    LO.to_csv(OUT / "lodo.csv", index=False)
    main = PO[(PO["fraction"] == 0.25) & (PO["contrast"] == "m2-m1") & (PO["metric"] == "sp") & (PO["range"] == "all48")].iloc[0]
    m10 = PO[(PO["fraction"] == 0.25) & (PO["contrast"] == "m1-m0") & (PO["metric"] == "sp") & (PO["range"] == "all48")].iloc[0]
    go = bool(main["est"] >= main["resolution"] and main["lo"] > 0 and (LO["est"] > 0).all())
    verdict = {"GO": go, "main_m2_m1": main.to_dict(), "m1_m0": m10.to_dict(), "lodo_min": float(LO["est"].min()),
               "N_f25": N[0.25], "N_f50": N[0.5],
               "note": "descriptive headroom check; GO = worth a confirmatory pre-registration on data not used here"}
    json.dump(verdict, open(OUT / "verdict.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(PO[(PO["metric"] == "sp")].round(4).to_string(index=False))
    print(LO.round(4).to_string(index=False))
    print(json.dumps(verdict, indent=1, default=float))
    print("SPARSE_ANALYZE_DONE")


if __name__ == "__main__":
    if sys.argv[1] == "panels":
        panels(sys.argv[2:] or DS)
    elif sys.argv[1] == "analyze":
        analyze()
