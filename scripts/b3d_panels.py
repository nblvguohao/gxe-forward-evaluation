"""Wave 2e panels (docs/prereg_sequel_wave2e_2026-09-29.md): on each independent dataset, forward predictions for
every predictable year and a 5-fold leave-genotypes-out CV history for every target year, for the wave 2b
two-stage library (panel.METHODS) plus cell_reml (no ECs on these data, so no G x E learners). The code paths
are the unchanged wave 2b / 2c functions. Resumable. No metric is computed here; scripts/b3d_rules.py scores
after every panel exists."""
import argparse
import json
import os
import subprocess
import time

import pandas as pd

from dartgxe.forward.cellgxe import cv_history_cell, forward_year_cell
from dartgxe.forward.data import LOADERS, NEW
from dartgxe.forward.panel import cv_history, forward_year
from dartgxe.paths import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=NEW)
a = ap.parse_args()
OUT = RESULTS / "b3d"
for d in ("forward", "cv", "meta"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
cnt = pd.read_csv(OUT / "count" / "per_year.csv")
targets = sorted(int(y) for y in cnt[(cnt["rule"] == "main") & cnt["qualifies"] & (cnt["dataset"] == a.dataset)]["year"])
commit = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                            capture_output=True, text=True).stdout.strip()
t0 = time.time()
ds = LOADERS[a.dataset]()
years = sorted(ds.cells["year"].unique())
predictable = [int(y) for y in years if ds.scorable_envs([y]) and any(v < y for v in years)
               and ds.cells[ds.cells["year"] < y]["genotype"].nunique() >= 5]
missing_targets = [t for t in targets if t not in predictable]
if missing_targets:
    raise RuntimeError(f"targets not predictable: {missing_targets}")
for y in predictable:
    f = OUT / "forward" / f"{a.dataset}_{y}.parquet"
    if f.exists():
        continue
    d2, m2 = forward_year(ds, y)
    dc, mc = forward_year_cell(ds, y, None)
    pd.concat([d2, dc], ignore_index=True).to_parquet(f, index=False)
    json.dump({"two_stage": {k: v for k, v in m2.items() if k != "fit_ids"}, "cell": mc, "git_commit": commit},
              open(OUT / "meta" / f"forward_{a.dataset}_{y}.json", "w"), indent=1, default=float)
    print("forward", a.dataset, y, len(d2) + len(dc), round(time.time() - t0), "s", flush=True)
for t in targets:
    f = OUT / "cv" / f"{a.dataset}_{t}.parquet"
    if f.exists():
        continue
    d2, m2 = cv_history(ds, t)
    dc, mc = cv_history_cell(ds, t, None)
    pd.concat([d2, dc], ignore_index=True).to_parquet(f, index=False)
    json.dump({"two_stage": m2, "cell": mc, "git_commit": commit},
              open(OUT / "meta" / f"cv_{a.dataset}_{t}.json", "w"), indent=1, default=float)
    print("cv", a.dataset, t, len(d2) + len(dc), round(time.time() - t0), "s", flush=True)
json.dump({"dataset": a.dataset, "targets": targets, "predictable_years": predictable, "n_cells": len(ds.cells),
           "n_genotyped": int(ds.markers.shape[0]), "n_marker_columns": int(ds.markers.shape[1]),
           "kernel_embedding": bool(ds.markers.attrs.get("no_scale", False)), "git_commit": commit,
           "seconds": round(time.time() - t0)}, open(OUT / "meta" / f"summary_{a.dataset}.json", "w"), indent=1)
print("B3D_PANELS_DONE", a.dataset)
