"""Wave 2e structure count (plan v2 §9.3 direction 1; before the wave 2e pre-registration and before any method
is fitted on these data): qualifying forward years of the independent datasets under the unchanged W1c rules
(docs/prereg_sequel_wave1_2026-09-28.md §4 and its operating rules). Counts sample structure only; phenotype
values are used only to decide whether a cell exists. Output: results/b3d/count/{per_year.csv, verdict.json,
run_meta.json}."""
import json
import os
import socket
import subprocess
import time

import pandas as pd

from dartgxe.forward.data import LOADERS, NEW
from dartgxe.paths import RESULTS

OUT = RESULTS / "b3d" / "count"
OUT.mkdir(parents=True, exist_ok=True)
MIN_RELAXED, NEW_SHARE, MIN_ENVS, MIN_PRIOR_YEARS = 10, 0.5, 5, 2


def count(ds):
    cells = ds.cells.assign(year=ds.cells["year"].astype(int))
    per_env = cells.groupby(["year", "env"])["genotype"].nunique().rename("n").reset_index()
    first_year = cells.groupby("genotype")["year"].min()
    rows = []
    for tag, thr in (("main", ds.min_n), ("relaxed", MIN_RELAXED)):
        big = per_env[per_env["n"] >= thr]
        years_with_big = sorted(big["year"].unique())
        for yr in sorted(cells["year"].unique()):
            envs = big[big["year"] == yr]["env"]
            lines = cells[(cells["year"] == yr) & cells["env"].isin(envs)]["genotype"].unique()
            prior = [y for y in years_with_big if y < yr]
            new = float((first_year.loc[lines] >= yr).mean()) if len(lines) else float("nan")
            n_cells = int(big[big["year"] == yr]["n"].sum())
            ok = len(prior) >= MIN_PRIOR_YEARS and len(envs) >= MIN_ENVS and len(lines) > 0 and new >= NEW_SHARE
            rows.append({"dataset": ds.name, "rule": tag, "min_n": thr, "year": int(yr),
                         "envs_total": int((per_env["year"] == yr).sum()), "envs_ge_min": int(len(envs)),
                         "median_n": float(per_env[per_env["year"] == yr]["n"].median()),
                         "lines": int(len(lines)), "cells": n_cells,
                         "resolution": round(3 / n_cells ** 0.5, 3) if n_cells else float("nan"),
                         "new_share": round(new, 3), "prior_years": len(prior), "qualifies": bool(ok)})
    return rows


t0 = time.time()
rows, info = [], {}
for name in NEW:
    ds = LOADERS[name]()
    rows += count(ds)
    info[name] = {"cells": int(len(ds.cells)), "genotypes_with_cells": int(ds.cells["genotype"].nunique()),
                  "genotyped": int(ds.markers.shape[0]), "marker_columns": int(ds.markers.shape[1]),
                  "missing_share": float(ds.markers.isna().to_numpy().mean()),
                  "kernel_embedding": bool(ds.markers.attrs.get("no_scale", False))}
    print(name, info[name], round(time.time() - t0), "s", flush=True)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "per_year.csv", index=False)
main = tab[(tab["rule"] == "main") & tab["qualifies"]]
verdict = {"K_main": int(len(main)),
           "by_dataset_main": {d: sorted(int(y) for y in g["year"]) for d, g in main.groupby("dataset")},
           "N_cells_main": int(main["cells"].sum()),
           "K_relaxed_sensitivity": int(((tab["rule"] == "relaxed") & tab["qualifies"]).sum()),
           "datasets": info}
json.dump(verdict, open(OUT / "verdict.json", "w"), indent=1)
commit = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                            capture_output=True, text=True).stdout.strip()
json.dump({"git_commit": commit, "newdata": os.environ.get("NEWDATA"), "host": socket.gethostname(),
           "seconds": round(time.time() - t0)}, open(OUT / "run_meta.json", "w"), indent=1)
print(tab[tab["rule"] == "main"].to_string(index=False))
print(json.dumps(verdict, indent=1))
print("B3D_COUNT_DONE")
