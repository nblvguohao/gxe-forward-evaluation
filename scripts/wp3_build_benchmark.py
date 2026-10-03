"""WP3 (plan v2 §10.4): package the forward-year benchmark. For each of the six datasets writes
  cells_<D>.parquet        env, year, genotype, y  (genotyped cells of every year; derived from the public releases)
  predictions_<D>.parquet  env, year, genotype, method, pred  (reference predictions on the scorable cells of every
                           target year: the two-stage library, cell_reml, G x E learners, within-env-loss MLPs)
  splits.json              per dataset: min genotypes per scorable environment, target years, and for every target year
                           the scorable environments
into RESULTS/benchmark/, plus SHA256SUMS. Uses the loaders and panels of the pre-registered waves unchanged."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dartgxe.forward.data import LOADERS
from dartgxe.paths import RESULTS

OUT = RESULTS / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
cnt = pd.read_csv(RESULTS / "b3d" / "count" / "per_year.csv")
TARGETS = {d: sorted(int(y) for y in g["year"]) for tab in (w1c, cnt)
           for d, g in tab[(tab["rule"] == "main") & tab["qualifies"]].groupby("dataset")}
NEW = {"ESWYT", "GEM_IA", "MU_SOY"}
COLS = ["env", "genotype", "method", "pred"]
splits = {}
for d, targets in TARGETS.items():
    ds = LOADERS[d]()
    cells = ds.cells[["env", "year", "genotype", "y"]].copy()
    cells["y"] = cells["y"].astype(np.float64)
    cells.sort_values(["year", "env", "genotype"]).to_parquet(OUT / f"cells_{d}.parquet", index=False)
    parts, sp = [], {}
    for Y in targets:
        envs = ds.scorable_envs([Y])
        sp[str(Y)] = envs
        if d in NEW:
            P = pd.read_parquet(RESULTS / "b3d" / "forward" / f"{d}_{Y}.parquet", columns=COLS)
        else:
            P = pd.concat([pd.read_parquet(RESULTS / k / "forward" / f"{d}_{Y}.parquet", columns=COLS) for k in ("b3", "b3b")])
            for m in ("dl_g", "dl_ge"):
                fs = [RESULTS / "b3c" / "pred" / f"{d}_{Y}_{m}_seed{s}.parquet" for s in range(3)]
                if all(f.exists() for f in fs):
                    s = pd.concat([pd.read_parquet(f, columns=["env", "genotype", "pred"]) for f in fs])
                    parts.append(s.groupby(["env", "genotype"], as_index=False)["pred"].mean().assign(method=m, year=Y))
        P = P[~P["method"].str.endswith("_real")]  # observed-EC variants use target-year weather: upper bound only
        assert set(P["env"]) <= set(envs), (d, Y)
        parts.append(P.assign(year=Y))
    pred = pd.concat(parts, ignore_index=True)[["env", "year", "genotype", "method", "pred"]]
    pred["pred"] = pred["pred"].astype(np.float32)
    pred.sort_values(["year", "env", "method", "genotype"]).to_parquet(OUT / f"predictions_{d}.parquet", index=False)
    splits[d] = {"min_genotypes": int(ds.min_n), "target_years": targets, "scorable_environments": sp,
                 "methods": sorted(pred["method"].unique()),
                 "set": "independent" if d in NEW else "original"}
    print(d, len(cells), "cells;", len(pred), "predictions;", len(targets), "target years", flush=True)
json.dump(splits, open(OUT / "splits.json", "w"), indent=1)
with open(OUT / "SHA256SUMS", "w") as fh:
    for f in sorted(OUT.glob("*")):
        if f.name != "SHA256SUMS":
            fh.write(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n")
print("WP3_BUILD_DONE")
