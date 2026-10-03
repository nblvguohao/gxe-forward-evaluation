"""Wave 2c panels (docs/prereg_sequel_wave2c_2026-09-28.md): cell-level and G x E learner predictions for every
predictable year (forward; historical-mean and observed ECs) and every target year's CV history (same folds as
wave 2b). Resumable. No metric is computed here; scripts/b3b_rules.py scores after every panel exists."""
import argparse
import json
import os
import subprocess
import time

import pandas as pd

from dartgxe.forward.cellgxe import cv_history_cell, forward_year_cell
from dartgxe.forward.data import LOADERS, load_ec
from dartgxe.paths import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(LOADERS))
a = ap.parse_args()
OUT = RESULTS / "b3b"
for d in ("forward", "cv", "meta"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
targets = sorted(int(y) for y in w1c[(w1c["rule"] == "main") & w1c["qualifies"] & (w1c["dataset"] == a.dataset)]["year"])
summ_b3 = json.load(open(RESULTS / "b3" / "meta" / f"summary_{a.dataset}.json"))
predictable = summ_b3["predictable_years"]  # identical set to wave 2b
commit = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                            capture_output=True, text=True).stdout.strip()
t0 = time.time()
ds = LOADERS[a.dataset]()
ec = load_ec(a.dataset)
for y in predictable:
    f = OUT / "forward" / f"{a.dataset}_{y}.parquet"
    if f.exists():
        continue
    d, metas = forward_year_cell(ds, int(y), ec)
    d.to_parquet(f, index=False)
    json.dump(metas | {"git_commit": commit}, open(OUT / "meta" / f"forward_{a.dataset}_{y}.json", "w"), indent=1, default=float)
    print("forward", a.dataset, y, len(d), round(time.time() - t0), "s", flush=True)
for t in targets:
    f = OUT / "cv" / f"{a.dataset}_{t}.parquet"
    if f.exists():
        continue
    d, metas = cv_history_cell(ds, t, ec)
    d.to_parquet(f, index=False)
    json.dump({"folds": metas, "git_commit": commit}, open(OUT / "meta" / f"cv_{a.dataset}_{t}.json", "w"), indent=1, default=float)
    print("cv", a.dataset, t, len(d), round(time.time() - t0), "s", flush=True)
json.dump({"dataset": a.dataset, "targets": targets, "predictable_years": predictable, "has_ec": ec is not None,
           "n_ec_envs": 0 if ec is None else int(ec.shape[0]), "git_commit": commit, "seconds": round(time.time() - t0)},
          open(OUT / "meta" / f"summary_{a.dataset}.json", "w"), indent=1)
print("B3B_PANELS_DONE", a.dataset)
