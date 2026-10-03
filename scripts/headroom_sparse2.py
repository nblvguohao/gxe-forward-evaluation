"""SP2 headroom check (docs/prereg_headroom_sparse2_2026-09-29.md). Descriptive, not confirmatory.

  python scripts/headroom_sparse2.py panels G2F NUST ...
  python scripts/headroom_sparse2.py analyze        # refuses to run until every panel exists and the consistency check passes

Environment: BENCH, OUT, SP_PANELS (SP panel dir, for the consistency check), HEADROOM_COMMIT, dartgxe.forward.data vars."""
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

from dartgxe.forward import sparse, sparse2
from dartgxe.forward.data import LOADERS, load_ec
from dartgxe.forward.pool import dersimonian_laird, floor_se

BENCH, OUT, SPP = Path(os.environ["BENCH"]), Path(os.environ["OUT"]), Path(os.environ["SP_PANELS"])
SPL = json.load(open(BENCH / "splits.json"))
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
EC_DS = ["G2F", "URSN"]
B, SEED = 2000, 20260930
METHODS = {"stk": DS, "dlres": DS, "ecmxe": EC_DS}


def tag(d, y, f, s):
    return f"{d}_{y}_f{int(round(f * 100))}_s{s}"


def run_meta():
    import torch
    return {"commit": os.environ.get("HEADROOM_COMMIT", "unknown"), "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version()}


def panels(names):
    (OUT / "panels").mkdir(parents=True, exist_ok=True)
    for d in names:
        ds, ec = None, None
        for Y in SPL[d]["target_years"]:
            for f in sparse.FRACTIONS:
                for s in sparse.SEEDS:
                    fp = OUT / "panels" / f"{tag(d, Y, f, s)}.parquet"
                    if fp.exists():
                        continue
                    if ds is None:
                        ds = LOADERS[d]()
                        ec = load_ec(d) if d in EC_DS else None
                    t0, start = time.time(), dt.datetime.now().isoformat(timespec="seconds")
                    out, meta = sparse2.sp2_year(ds, Y, SPL[d]["scorable_environments"][str(Y)], f, s, ec=ec)
                    sp = pd.read_parquet(SPP / f"{tag(d, Y, f, s)}.parquet")
                    m = out.merge(sp[["env", "genotype", "m0", "m1"]], on=["env", "genotype"], suffixes=("", "_sp"))
                    scale = float(np.abs(sp["m1"]).max()) or 1.0
                    meta["consistency"] = {"rows": [len(out), len(sp), len(m)],
                                           "max_rel_m0": float((m["m0"] - m["m0_sp"]).abs().max() / scale),
                                           "max_rel_m1": float((m["m1"] - m["m1_sp"]).abs().max() / scale)}
                    meta.update(run_meta(), dataset=d, target=Y, fraction=f, seed=s, start=start,
                                end=dt.datetime.now().isoformat(timespec="seconds"), seconds=round(time.time() - t0, 1))
                    out.to_parquet(fp, index=False)
                    json.dump(meta, open(fp.with_suffix(".json"), "w"), indent=1, default=str)
                    print(d, Y, f, s, f"{meta['seconds']}s", "w", meta["stack_w"], "ec_rho", meta.get("ec_rho"),
                          "cons", round(meta["consistency"]["max_rel_m1"], 12), flush=True)


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


def pooled(ye, level=0.95):
    if ye.empty:
        return {}
    r = dersimonian_laird(floor_se(ye), level=level)
    return {k: r[k] for k in ("est", "se", "lo", "hi", "tau2", "k")}


def analyze():
    need = [(d, Y, f, s) for d in DS for Y in SPL[d]["target_years"] for f in sparse.FRACTIONS for s in sparse.SEEDS]
    missing = [tag(*n) for n in need if not (OUT / "panels" / f"{tag(*n)}.parquet").exists()]
    if missing:
        sys.exit(f"panels missing ({len(missing)}): {missing[:8]}")
    metas = [json.load(open(OUT / "panels" / f"{tag(*n)}.json")) for n in need]
    worst = max(max(m["consistency"]["max_rel_m0"], m["consistency"]["max_rel_m1"]) for m in metas)
    rows_ok = all(m["consistency"]["rows"][0] == m["consistency"]["rows"][1] == m["consistency"]["rows"][2] for m in metas)
    cons = {"pass": bool(worst < 1e-8 and rows_ok), "worst_rel": worst, "rows_ok": rows_ok}
    json.dump(cons, open(OUT / "consistency.json", "w"), indent=1)
    if not cons["pass"]:
        sys.exit(f"consistency check failed: {cons}")
    rng = np.random.default_rng(SEED)
    rows = []
    for d, Y, f, s in need:
        P = pd.read_parquet(OUT / "panels" / f"{tag(d, Y, f, s)}.parquet")
        min_n = SPL[d]["min_genotypes"]
        for e, g in P.groupby("env"):
            if len(g) < min_n:
                continue
            y = g["y"].to_numpy(float)
            r = {"dataset": d, "target": Y, "fraction": f, "seed": s, "env": e, "n": len(g)}
            for m in ["m0", "m1", "gbm", "stk", "dlres"] + (["ecmxe"] if "ecmxe" in g else []):
                p = g[m].to_numpy(float)
                r[f"sp_{m}"], r[f"pe_{m}"], r[f"sd_{m}"] = spearman(y, p), pearson(y, p), sel_diff(y, p)
            rows.append(r)
    E = pd.DataFrame(rows)
    E.to_parquet(OUT / "per_env_seed.parquet", index=False)
    ye_rows = []
    for f in sparse.FRACTIONS:
        Ef = E[E["fraction"] == f]
        for m in ["stk", "dlres", "ecmxe", "gbm"]:
            for metric in ("sp", "pe", "sd"):
                col = f"{metric}_{m}"
                if col not in Ef:
                    continue
                env = Ef.assign(d=Ef[col] - Ef[f"{metric}_m1"]).groupby(["dataset", "target", "env"])["d"].mean()
                env = env.dropna().reset_index()
                for (d, Y), g in env.groupby(["dataset", "target"]):
                    v = g["d"].to_numpy()
                    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
                    ye_rows.append({"fraction": f, "method": m, "metric": metric, "dataset": d, "target": Y,
                                    "d": float(v.mean()), "se": float(bs.std(ddof=1)), "n_env": len(v)})
    YE = pd.DataFrame(ye_rows)
    YE.to_csv(OUT / "year_effects.csv", index=False)
    Nc = E[(E["fraction"] == 0.25) & (E["seed"] == 0)].groupby("dataset")["n"].sum()
    Pr = []
    for (f, m, metric), g in YE.groupby(["fraction", "method", "metric"]):
        scope = METHODS.get(m, DS)
        for rng_name, dsets in [("main", scope)] + [(d, [d]) for d in scope]:
            sub = g[g["dataset"].isin(dsets)]
            r = pooled(sub[["dataset", "d", "se"]])
            if r:
                Pr.append({"fraction": f, "method": m, "metric": metric, "range": rng_name,
                           "resolution": 3 / np.sqrt(Nc[list(dsets)].sum()), **r})
    PO = pd.DataFrame(Pr)
    PO.to_csv(OUT / "pooled.csv", index=False)
    # decision (prereg §4): Holm across the three methods, f = 0.25, Spearman
    main = PO[(PO["fraction"] == 0.25) & (PO["metric"] == "sp") & (PO["range"] == "main") & PO["method"].isin(list(METHODS))]
    main = main.assign(p=2 * norm.sf(np.abs(main["est"] / main["se"]))).sort_values("p").reset_index(drop=True)
    levels = [1 - 0.05 / 3, 1 - 0.05 / 2, 0.95]
    verdict, stop = {}, False
    for i, row in main.iterrows():
        m = row["method"]
        g = YE[(YE["fraction"] == 0.25) & (YE["method"] == m) & (YE["metric"] == "sp") & YE["dataset"].isin(METHODS[m])]
        ci = pooled(g[["dataset", "d", "se"]], level=levels[i])
        lodo = [pooled(g[g["dataset"] != d][["dataset", "d", "se"]]).get("est") for d in sorted(g["dataset"].unique())]
        go = (not stop) and ci["lo"] > 0 and row["est"] >= row["resolution"] and all(x is not None and x > 0 for x in lodo)
        if not (ci["lo"] > 0):
            stop = True  # Holm: later hypotheses are not rejected once one fails
        verdict[m] = {"GO": bool(go), "est": row["est"], "p": row["p"], "holm_level": levels[i], "ci": [ci["lo"], ci["hi"]],
                      "resolution": row["resolution"], "lodo": lodo, "k": row["k"]}
    json.dump({"decisions": verdict, "consistency": cons,
               "note": "descriptive headroom check; GO = worth a confirmatory pre-registration on data not used here"},
              open(OUT / "verdict.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(PO[(PO["metric"] == "sp") & (PO["range"] == "main")].round(4).to_string(index=False))
    print(json.dumps(verdict, indent=1, default=float))
    print("SPARSE2_ANALYZE_DONE")


if __name__ == "__main__":
    if sys.argv[1] == "panels":
        panels(sys.argv[2:] or DS)
    elif sys.argv[1] == "analyze":
        analyze()
