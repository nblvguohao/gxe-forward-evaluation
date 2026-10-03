"""Wave 2d panels (docs/prereg_sequel_wave2d_2026-09-28.md): within-environment-loss MLP predictions for every
target year, seeds {0, 1, 2}; dl_g everywhere, dl_ge where ECs exist. Resumable per (dataset, year, method,
seed). No metric on the target year is computed here; scripts/b3c_analyze.py scores after every file exists."""
import argparse
import json
import os
import subprocess
import time

import pandas as pd

from dartgxe.forward.data import LOADERS, load_ec
from dartgxe.forward.dl import tune_and_refit
from dartgxe.paths import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(LOADERS))
ap.add_argument("--seeds", default="0,1,2")
a = ap.parse_args()
OUT = RESULTS / "b3c"
(OUT / "pred").mkdir(parents=True, exist_ok=True)
(OUT / "meta").mkdir(parents=True, exist_ok=True)
w1c = pd.read_csv(RESULTS / "w1c" / "per_year.csv")
targets = sorted(int(y) for y in w1c[(w1c["rule"] == "main") & w1c["qualifies"] & (w1c["dataset"] == a.dataset)]["year"])
predictable = json.load(open(RESULTS / "b3" / "meta" / f"summary_{a.dataset}.json"))["predictable_years"]
commit = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                            capture_output=True, text=True).stdout.strip()
ds = LOADERS[a.dataset]()
ec = load_ec(a.dataset)
t0 = time.time()
for Y in targets:
    V = max(y for y in predictable if y < Y)
    for method, e in (("dl_g", None), ("dl_ge", ec)):
        if method == "dl_ge" and ec is None:
            continue
        for seed in (int(s) for s in a.seeds.split(",")):
            f = OUT / "pred" / f"{a.dataset}_{Y}_{method}_seed{seed}.parquet"
            if f.exists():
                continue
            out, meta = tune_and_refit(ds, V, Y, seed, e)
            out.assign(dataset=a.dataset, method=method, seed=seed).to_parquet(f, index=False)
            json.dump(meta | {"method": method, "dataset": a.dataset, "git_commit": commit},
                      open(OUT / "meta" / f"{a.dataset}_{Y}_{method}_seed{seed}.json", "w"), indent=1, default=float)
            print(a.dataset, Y, method, seed, "V", V, "cfg", meta["config_index"], "E*", meta["e_star"],
                  round(time.time() - t0), "s", flush=True)
json.dump({"dataset": a.dataset, "targets": targets, "seconds": round(time.time() - t0), "git_commit": commit},
          open(OUT / "meta" / f"summary_{a.dataset}.json", "w"), indent=1)
print("B3C_PANELS_DONE", a.dataset)
