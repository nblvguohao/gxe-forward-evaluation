"""How robust is the verdict 'within-environment-loss network is a resolved loss' under NUST trial x location x year units with
the original URSN scoring (Section 3.6)? Re-pools the stored per-unit scores (results/env_definition_check/per_unit.csv)
exactly as scripts/env_definition_sensitivity.py, with 20 bootstrap seeds for the NUST and URSN year SEs (other datasets
keep their stored year effects). Post hoc. Run from the repository root  ->  results/env_definition_check/network_boundary_seeds.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from dartgxe.forward.pool import dersimonian_laird, floor_se  # noqa: E402

B, RES = 2000, 0.0087
U = pd.read_csv("results/env_definition_check/per_unit.csv")
Y48 = pd.read_csv("results/summary48/year_effects.csv")
Y48 = Y48[(Y48.metric == "spearman") & (Y48.kind == "method") & (Y48.item == "dl_g")]


def yearfx(E, rng):
    out = []
    for (d, Y), g in E.groupby(["dataset", "target"]):
        v = (g["dl_g"] - g["cell_reml"]).dropna().to_numpy()
        if len(v):
            out.append({"dataset": d, "d": v.mean(), "se": v[rng.integers(0, len(v), (B, len(v)))].mean(1).std(ddof=1)})
    return pd.DataFrame(out)


rows = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    ye = pd.concat([Y48[~Y48.dataset.isin(["NUST", "URSN"])][["dataset", "d", "se"]],
                    yearfx(U[(U.dataset == "NUST") & (U.variant == "trial_env")], rng),
                    yearfx(U[(U.dataset == "URSN") & (U.variant == "orig")], rng)], ignore_index=True)
    r = dersimonian_laird(floor_se(ye.reset_index(drop=True)))
    v = "resolved loss" if r["hi"] < 0 and -r["est"] >= RES else ("tied" if r["hi"] >= 0 else "detectable")
    rows.append({"seed": seed, "est": r["est"], "lo": r["lo"], "hi": r["hi"], "verdict": v})
T = pd.DataFrame(rows)
T.to_csv("results/env_definition_check/network_boundary_seeds.csv", index=False)
print(T.round(5).to_string(index=False)); print(T.verdict.value_counts().to_dict())
