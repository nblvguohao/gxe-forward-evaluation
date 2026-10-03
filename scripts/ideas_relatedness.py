"""Genotype-only structure diagnostic for docs/ideas_gblup_improvement_2026-09-29.md (descriptive; no phenotype of
any target year is read beyond the list of lines in its scored environments, which is known before the trial).

For every forward target year Y (benchmark splits): training lines = lines with a phenotype in years < Y; target lines
= lines of Y's scored environments that never appeared before. Markers standardised on the training lines
(panel.features, as the method library). Genomic relationship target x training = Zt Zf' / p, divided by the mean
self-relationship of the training lines so that 1 ~ 'as related as a line to itself' in every dataset.
Per target year:
  - mean over target lines of the max relationship to any training line, by the training line's first year gap
    (1, 2, 3-4, >=5 years before Y);
  - share of each target line's 20 nearest training lines that come from the last 2 years, next to the share of
    training lines that come from those years (enrichment > 1 = close relatives are recent);
  - mean relationship of target lines to the training centroid (0 by construction) is not informative, so the mean
    of the top-20 relationships is reported instead.
Runs on the 4090 (CPU), outputs results/ideas_diag/relatedness.csv."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dartgxe.forward.data import LOADERS
from dartgxe.forward.panel import features

OUT = Path(os.environ.get("IDEAS_OUT", "."))
SPL = json.load(open(os.environ["SPLITS"]))
BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 999)]
rows = []
for name in sys.argv[1:] or list(LOADERS):
    ds = LOADERS[name]()
    first = ds.cells.groupby("genotype")["year"].min()
    for Y in SPL[name]["target_years"]:
        envs = SPL[name]["scorable_environments"][str(Y)]
        train = ds.cells[ds.cells["year"] < Y]
        fit_ids = sorted(set(train["genotype"]) & set(ds.markers.index))
        tgt = sorted((set(ds.cells.loc[(ds.cells["year"] == Y) & ds.cells["env"].isin(envs), "genotype"]) - set(fit_ids))
                     & set(ds.markers.index))
        if not tgt:
            continue
        Zf, Zt, _, _ = features(ds.markers, fit_ids, tgt)
        p = Zf.shape[1]
        self_rel = float((Zf * Zf).sum(1).mean() / p)
        K = (Zt @ Zf.T) / p / self_rel
        fy = first.loc[fit_ids].to_numpy()
        row = {"dataset": name, "target": Y, "n_train_lines": len(fit_ids), "n_target_new": len(tgt), "n_markers": p}
        for lo, hi in BUCKETS:
            m = (Y - fy >= lo) & (Y - fy <= hi)
            row[f"lines_gap_{lo}_{hi}"] = int(m.sum())
            row[f"maxrel_gap_{lo}_{hi}"] = float(K[:, m].max(1).mean()) if m.any() else np.nan
        k = min(20, K.shape[1])
        top = np.argpartition(-K, k - 1, axis=1)[:, :k]
        recent = (Y - fy) <= 2
        row["top20_mean_rel"] = float(np.take_along_axis(K, top, 1).mean())
        row["top20_share_recent2"] = float(recent[top].mean())
        row["train_share_recent2"] = float(recent.mean())
        row["recent_enrichment"] = row["top20_share_recent2"] / row["train_share_recent2"] if recent.any() else np.nan
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "relatedness.csv", index=False)
print("RELATEDNESS_DONE")
