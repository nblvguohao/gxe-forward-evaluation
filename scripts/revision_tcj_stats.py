"""Statistics added in the TCJ revision (2026-10-08); nothing is fitted or re-scored, everything comes from stored year
effects and per-environment tables.

1. hk_sensitivity.csv: for the 31 primary pooled contrasts (scripts/floor_sensitivity.py, Table S10), the
   pre-specified DerSimonian-Laird estimate and interval (SE floor of the analysis plans), the between-year variance tau2,
   and the Hartung-Knapp interval with the same weights; verdicts under both.
2. per_dataset.csv: sparse testing (25 %, 50 %) and the forward-benchmark contrasts against cell-level GBLUP, pooled
   within each dataset, judged against each dataset's own resolution 3/sqrt(N_d).
3. without_eswyt_ursn.csv: the same contrasts pooled over G2F, NUST, GEM_IA and MU_SOY only.
6. without_nust.csv: benchmark contrasts pooled without NUST (which datasets drive the pooled losses).
7. kernel_without_eswyt_ursn.csv: the kernel transfer contrast pooled without ESWYT and URSN.
8. floor_variants.csv: the two sparse-testing gains with the floor of the plans, with a floor on zero SEs only, and without any
   floor after dropping the target years whose SE is zero.
5. clac_new_hybrids.json: scored cells of new hybrids in the CLAC re-run (definition of scripts/ra/ra_score.py: a hybrid is new
   if its first year in the G2F cells is the target year or later; environments with at least 25 new hybrids), and the
   resolution of the new-hybrid contrast. Every environment's cell count is checked against results/reanalysis/per_env.parquet.
4. env_shrink.csv: for each alternative environment definition, how much the loss of each two-stage method against
   cell-level GBLUP shrinks relative to the pre-specified scoring (results/env_definition_check/pooled_by_variant.csv).

Run from the repository root:  python3 scripts/revision_tcj_stats.py  ->  results/revision_tcj/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from dartgxe.forward.pool import dersimonian_laird, floor_se, hartung_knapp  # noqa: E402
from floor_sensitivity import contrast_inputs, verdict  # noqa: E402

R, OUT = Path("results"), Path("results/revision_tcj")
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
BENCH = (("reml", "Two-stage GBLUP"), ("reml_x0.1", "Two-stage ridge, shrinkage x0.1"), ("reml_x10", "Two-stage ridge, shrinkage x10"),
         ("reml_x100", "Two-stage ridge, shrinkage x100"), ("ridge_pc20", "Ridge on 20 PCs"), ("rf", "Random forest"),
         ("gbm", "Gradient boosting"), ("knn10", "kNN, k = 10"), ("knn30", "kNN, k = 30"), ("mlp", "Multilayer perceptron"),
         ("gxe_gbm", "Covariate LightGBM"), ("rn_ridge", "Covariate reaction-norm ridge"), ("dl_g", "Within-environment-loss network"),
         ("R_STK_FW", "Stacking on the forward history"))


def pool(ye, level=0.95):
    return dersimonian_laird(floor_se(ye[["dataset", "d", "se"]].reset_index(drop=True)), level=level)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. Hartung-Knapp sensitivity
    rows = []
    for label, ye, res, level in contrast_inputs():
        h = hartung_knapp(floor_se(ye), level=level)
        rows.append({"contrast": label, "target_years": h["k"], "level": level, "resolution": res, "est": h["est"],
                     "dl_lo": h["lo"], "dl_hi": h["hi"], "dl_verdict": verdict(h["est"], h["lo"], h["hi"], res), "tau2": h["tau2"],
                     "hk_lo": h["hk_lo"], "hk_hi": h["hk_hi"],
                     "hk_verdict": verdict(h["est"], h["hk_lo"], h["hk_hi"], res) if h["k"] > 1 else "n/a (one target year)"})
    H = pd.DataFrame(rows)
    H["verdict_changed"] = (H["dl_verdict"] != H["hk_verdict"]) & (H["target_years"] > 1)
    H.to_csv(OUT / "hk_sensitivity.csv", index=False)

    # cells per dataset: benchmark (summary48) and sparse testing (seed 0, as in scripts/headroom_sparse.py)
    pe = pd.read_parquet(R / "summary48/per_env.parquet")
    sp = pd.read_parquet(R / "headroom_sparse/per_env_seed.parquet")
    rows, rows2 = [], []
    ys = pd.read_csv(R / "headroom_sparse/year_effects.csv")
    for f in (0.25, 0.5):
        y = ys[(ys["fraction"] == f) & (ys["contrast"] == "m1-m0") & (ys["metric"] == "sp")]
        cells = sp[(sp["fraction"] == f) & (sp["seed"] == 0)].groupby("dataset")["n"].sum()
        for d in DS:
            g = y[y["dataset"] == d]
            if len(g) == 0:
                continue
            r, res = pool(g), 3 / np.sqrt(cells[d])
            rows.append({"analysis": f"Sparse {int(100 * f)} %: M×E − main-effect GBLUP", "dataset": d, "target_years": r["k"],
                         "cells": int(cells[d]), "resolution": res, "est": r["est"], "lo": r["lo"], "hi": r["hi"],
                         "verdict": verdict(r["est"], r["lo"], r["hi"], res)})
        g = y[~y["dataset"].isin(["ESWYT", "URSN"])]
        r, res = pool(g), 3 / np.sqrt(cells.drop(["ESWYT", "URSN"], errors="ignore").sum())
        rows2.append({"analysis": f"Sparse {int(100 * f)} %: M×E − main-effect GBLUP", "target_years": r["k"], "resolution": res,
                      "est": r["est"], "lo": r["lo"], "hi": r["hi"], "verdict": verdict(r["est"], r["lo"], r["hi"], res)})
    yb = pd.read_csv(R / "summary48/year_effects.csv")
    yb = yb[yb["metric"] == "spearman"]
    for item, nm in BENCH:
        y = yb[yb["item"] == item]
        cells = pe[pe["item"] == item].groupby("dataset")["n"].sum()
        for d in DS:
            g = y[y["dataset"] == d]
            if len(g) == 0:
                continue
            r, res = pool(g), 3 / np.sqrt(cells[d])
            rows.append({"analysis": f"Benchmark: {nm} − cell-level GBLUP", "dataset": d, "target_years": r["k"], "cells": int(cells[d]),
                         "resolution": res, "est": r["est"], "lo": r["lo"], "hi": r["hi"], "verdict": verdict(r["est"], r["lo"], r["hi"], res)})
        g = y[~y["dataset"].isin(["ESWYT", "URSN"])]
        if len(g):
            r, res = pool(g), 3 / np.sqrt(cells.drop(["ESWYT", "URSN"], errors="ignore").sum())
            rows2.append({"analysis": f"Benchmark: {nm} − cell-level GBLUP", "target_years": r["k"], "resolution": res,
                          "est": r["est"], "lo": r["lo"], "hi": r["hi"], "verdict": verdict(r["est"], r["lo"], r["hi"], res)})
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "per_dataset.csv", index=False)
    W = pd.DataFrame(rows2)
    W.to_csv(OUT / "without_eswyt_ursn.csv", index=False)
    pv = pd.read_csv(R / "env_definition_check/pooled_by_variant.csv")
    lib = [k for k, _ in BENCH[:10]]
    base = pv[pv["variant"] == "NUST:orig / URSN:orig"].set_index("item")["est"]
    sh = []
    for v, g in pv[pv["variant"] != "NUST:orig / URSN:orig"].groupby("variant"):
        x = g.set_index("item")["est"]
        for m in lib:
            sh.append({"variant": v, "item": m, "est_orig": base[m], "est_variant": x[m], "shrink_pct": 100 * (1 - x[m] / base[m])})
    pd.DataFrame(sh).to_csv(OUT / "env_shrink.csv", index=False)
    rows3 = []
    for item, nm in BENCH:
        y = yb[(yb["item"] == item) & (yb["dataset"] != "NUST")]
        cells = pe[(pe["item"] == item) & (pe["dataset"] != "NUST")]["n"].sum()
        if len(y):
            r, res = pool(y), 3 / np.sqrt(cells)
            full = pool(yb[yb["item"] == item])
            rows3.append({"analysis": f"Benchmark: {nm} − cell-level GBLUP", "est_all": full["est"], "est_without_nust": r["est"], "lo": r["lo"],
                          "hi": r["hi"], "resolution": res, "verdict": verdict(r["est"], r["lo"], r["hi"], res),
                          "loss_smaller_without_nust": bool(abs(r["est"]) < abs(full["est"]))})
    pd.DataFrame(rows3).to_csv(OUT / "without_nust.csv", index=False)
    kt = pd.read_csv(R / "transfer_arc/year_effects.csv")
    kt = kt[(kt["metric"] == "spearman") & (kt["scope"] == "all")]
    pek = pe[pe["item"] == "reml"].groupby("dataset")["n"].sum()  # same cells as the benchmark
    g = kt[~kt["dataset"].isin(["ESWYT", "URSN"])]
    r, res = pool(g), 3 / np.sqrt(pek.drop(["ESWYT", "URSN"]).sum())
    pd.DataFrame([{"analysis": "Kernel transfer: cell_arc − cell_reml", "target_years": r["k"], "resolution": res, "est": r["est"], "lo": r["lo"],
                   "hi": r["hi"], "verdict": verdict(r["est"], r["lo"], r["hi"], res)}]).to_csv(OUT / "kernel_without_eswyt_ursn.csv", index=False)
    fv = []
    for f in (0.25, 0.5):
        y = ys[(ys["fraction"] == f) & (ys["contrast"] == "m1-m0") & (ys["metric"] == "sp")][["dataset", "d", "se"]].reset_index(drop=True)
        a = pool(y)
        from floor_sensitivity import zero_only
        b = dersimonian_laird(zero_only(y))
        c0 = dersimonian_laird(y[y["se"] > 0].reset_index(drop=True))
        fv.append({"fraction": f, "est_plan_floor": a["est"], "est_zero_floor": b["est"], "lo_zero_floor": b["lo"], "hi_zero_floor": b["hi"],
                   "est_no_floor_zero_dropped": c0["est"], "lo_no_floor_zero_dropped": c0["lo"], "hi_no_floor_zero_dropped": c0["hi"],
                   "years_dropped": int((y["se"] <= 0).sum())})
    pd.DataFrame(fv).to_csv(OUT / "floor_variants.csv", index=False)
    # New-hybrid part: needs the per-cell G2F benchmark table, which is built from the public data and not distributed;
    # without it the stored results/revision_tcj/clac_new_hybrids.json is kept.
    if (R / "benchmark/cells_G2F.parquet").exists():
        c = pd.read_parquet(R / "benchmark/cells_G2F.parquet")
        first = c.groupby("genotype")["year"].min()
        pe_ra = pd.read_parquet(R / "reanalysis/per_env.parquet")
        pe_ra = pe_ra[pe_ra["method"] == "clac"]
        n_new, envs, mismatch = 0, 0, 0
        for r in pe_ra.itertuples():
            g = c[c["env"] == r.env]
            mismatch += int(len(g) != r.n)
            k = int((g["genotype"].map(first) >= r.target).sum())
            if k >= 25 and pd.notna(r.sp_new):
                n_new += k; envs += 1
        import json
        json.dump({"new_hybrid_cells": n_new, "environments": envs, "resolution": 3 / np.sqrt(n_new), "cell_count_mismatches": mismatch},
                  open(OUT / "clac_new_hybrids.json", "w"), indent=1)
    else:
        print("results/benchmark/cells_G2F.parquet not found: clac_new_hybrids.json not recomputed")
    pd.set_option("display.width", 250)
    print(H[["contrast", "target_years", "level", "resolution", "est", "dl_lo", "dl_hi", "dl_verdict", "tau2", "hk_lo", "hk_hi", "hk_verdict"]].round(4).to_string(index=False))
    print(P.round(4).to_string(index=False))
    print(W.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
