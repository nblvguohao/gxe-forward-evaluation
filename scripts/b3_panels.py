"""Wave 2b panels (docs/prereg_sequel_wave2b_2026-09-28.md §1-3): forward predictions for every predictable
year of each dataset, and a 5-fold leave-genotypes-out CV history for every target year. Resumable: existing
files are skipped. No metric is computed here; scripts/b3_rules.py scores after every panel exists."""
import argparse
import json
import os
import subprocess
import time

import pandas as pd

from dartgxe.forward.data import LOADERS
from dartgxe.forward.panel import cv_history, forward_year
from dartgxe.paths import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(LOADERS))
a = ap.parse_args()
OUT = RESULTS / "b3"
for d in ("forward", "cv", "meta"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
targets = sorted(int(y) for y in w1c[(w1c["rule"] == "main") & w1c["qualifies"] & (w1c["dataset"] == a.dataset)]["year"])
commit = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                            capture_output=True, text=True).stdout.strip()
t0 = time.time()
ds = LOADERS[a.dataset]()
years = sorted(ds.cells["year"].unique())
predictable = [y for y in years if ds.scorable_envs([y]) and any(v < y for v in years)
               and ds.cells[ds.cells["year"] < y]["genotype"].nunique() >= 5]
summary = {"dataset": a.dataset, "targets": targets, "predictable_years": [int(y) for y in predictable],
           "n_cells": len(ds.cells), "n_genotyped": int(ds.markers.shape[0]), "n_markers": int(ds.markers.shape[1]),
           "git_commit": commit}
missing_targets = [t for t in targets if t not in predictable]
if missing_targets:
    raise RuntimeError(f"targets not predictable: {missing_targets}")
for y in predictable:
    f = OUT / "forward" / f"{a.dataset}_{y}.parquet"
    if f.exists():
        continue
    d, meta = forward_year(ds, int(y))
    d.to_parquet(f, index=False)
    json.dump({k: v for k, v in meta.items() if k != "fit_ids"} | {"git_commit": commit},
              open(OUT / "meta" / f"forward_{a.dataset}_{y}.json", "w"), indent=1, default=float)
    print("forward", a.dataset, y, len(d), round(time.time() - t0), "s", flush=True)
for t in targets:
    f = OUT / "cv" / f"{a.dataset}_{t}.parquet"
    if f.exists():
        continue
    d, metas = cv_history(ds, t)
    d.to_parquet(f, index=False)
    json.dump({"folds": metas, "git_commit": commit}, open(OUT / "meta" / f"cv_{a.dataset}_{t}.json", "w"), indent=1, default=float)
    print("cv", a.dataset, t, len(d), round(time.time() - t0), "s", flush=True)
summary["seconds"] = round(time.time() - t0)
json.dump(summary, open(OUT / "meta" / f"summary_{a.dataset}.json", "w"), indent=1, default=float)
print("B3_PANELS_DONE", a.dataset)
