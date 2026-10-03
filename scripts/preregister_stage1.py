"""Stage-1 pre-registration numbers (spec 00 stage-0 exit, spec 03 §3.3): MDE of the confirmatory
comparisons rank* vs twohead_mse on CV1 and CV00, computed WITHOUT any stage-1 test prediction.

Environment-level variance of a paired Δ is proxied by two closely related baselines (B3 vs B1,
which are not part of any confirmatory test); seed variance comes from the validation-only selection
runs. Holm over the two confirmatory comparisons: the first step is tested at α = 0.025.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from dartgxe.eval.compare import resolution
from dartgxe.eval.metrics import env_seed_matrix, per_env
from dartgxe.paths import RESULTS

SEEDS = 5
out = {}
sel = json.load(open(RESULTS / "stage1" / "selection.json"))
tab = pd.read_csv(RESULTS / "stage1" / "selection_table.csv")
for sc in ("CV1", "CV00"):
    frames = [pd.read_parquet(RESULTS / "predictions" / "g2f" / sc / m / "seed0.parquet") for m in ("B1_gblup", "B3_ecrn")]
    pe = per_env(pd.concat(frames))
    mat = env_seed_matrix(pe, "spearman")
    d = (mat.loc["B3_ecrn"][0] - mat.loc["B1_gblup"][0]).dropna()
    n_env, sd_env = len(d), float(d.std(ddof=1))
    N = int(pe[(pe.method == "B1_gblup") & (pe.metric == "spearman")]["n"].sum())
    t = tab[tab.scenario == sc].set_index("method")
    rs = sel[sc]["rank_star"]
    seed_sd = float(np.sqrt((t.loc[rs, "val_seed_sd"] ** 2 + t.loc["S1_twohead_mse", "val_seed_sd"] ** 2) / 2))
    # SE of the mean paired Δ: environments (independent) + seed noise of a difference of seed means
    se = float(np.sqrt(sd_env**2 / n_env + 2 * seed_sd**2 / SEEDS))
    z = norm.ppf(1 - 0.025 / 2) + norm.ppf(0.80)
    out[sc] = {"rank_star": rs, "N_test_cells": N, "resolution_3_over_sqrtN": round(resolution(N), 4),
               "practical_x1.5": round(1.5 * resolution(N), 4), "practical_x4.2": round(4.2 * resolution(N), 4),
               "env_sd_of_paired_delta_proxy_B3_minus_B1": round(sd_env, 4), "n_env": n_env,
               "seed_sd_val": round(seed_sd, 4), "se_delta": round(se, 4), "mde_80_holm_first_step": round(z * se, 4),
               "validation_delta_rankstar_minus_twohead": round(float(t.loc[rs, "val"] - t.loc["S1_twohead_mse", "val"]), 4)}
(RESULTS / "stage1" / "preregistration.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
