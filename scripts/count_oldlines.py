"""Structure count for the OL headroom check (docs/prereg_headroom_oldlines_2026-09-30.md §6 step 2). No model is
fitted; phenotype values are used only to decide whether a cell exists. Inputs: results/benchmark/{cells_*, splits.json}.
Output: results/count_oldlines/{per_year.csv, summary.json}."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH, OUT = ROOT / "results" / "benchmark", ROOT / "results" / "count_oldlines"
OUT.mkdir(parents=True, exist_ok=True)
SPL = json.load(open(BENCH / "splits.json"))
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
LOC_DS = ["G2F", "NUST", "URSN", "ESWYT"]
MIN_ENVS = 5


def location(dataset, env):
    """Same mapping as cellspec.location (I-B prereg §2.2), level 'location'."""
    base = str(env).rsplit("_", 1)[0]
    if dataset == "G2F":
        m = re.match(r"^[A-Z]{2}[HS]\d", base)
        return m.group(0) if m else base
    if dataset == "NUST":
        name, state = base.rsplit("_", 1)
        return re.sub(r"[^A-Za-z]", "", name) + "_" + state
    if dataset in ("URSN", "ESWYT"):
        return base
    return None


rows = []
for d in DS:
    C = pd.read_parquet(BENCH / f"cells_{d}.parquet")
    min_n = SPL[d]["min_genotypes"]
    first = C.groupby("genotype")["year"].min()
    if d in LOC_DS:
        C["loc"] = [location(d, e) for e in C["env"]]
    for Y in SPL[d]["target_years"]:
        envs = SPL[d]["scorable_environments"][str(Y)]
        T = C[(C["year"] == Y) & C["env"].isin(envs)]
        old = T[T["genotype"].map(first) < Y]
        per_env = old.groupby("env")["genotype"].nunique().reindex(envs).fillna(0).astype(int)
        ok_envs = per_env[per_env >= min_n]
        n_cells = int(per_env[per_env >= min_n].sum())
        row = {"dataset": d, "year": Y, "scored_envs": len(envs), "lines_in_year": int(T["genotype"].nunique()),
               "old_lines": int(old["genotype"].nunique()), "old_share": float(old["genotype"].nunique() / max(T["genotype"].nunique(), 1)),
               "median_old_per_env": float(per_env.median()), "envs_ok": int(len(ok_envs)), "cells_ok": n_cells,
               "qualifies": bool(len(ok_envs) >= MIN_ENVS)}
        if d in LOC_DS:
            hist = C[C["year"] < Y]
            seen_pairs = set(zip(hist["genotype"], hist["loc"]))
            sub = old[old["env"].isin(ok_envs.index)]
            sub_loc = [location(d, e) for e in sub["env"]]
            has = np.array([(g, l) in seen_pairs for g, l in zip(sub["genotype"], sub_loc)])
            row["share_cells_with_same_loc_history"] = float(has.mean()) if len(has) else np.nan
        rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(OUT / "per_year.csv", index=False)
q = T[T["qualifies"]]
N = int(q["cells_ok"].sum())
Nloc = int(q[q["dataset"].isin(LOC_DS)]["cells_ok"].sum())
summary = {"units_qualifying": int(len(q)), "units_by_dataset": q.groupby("dataset").size().to_dict(),
           "units_qualifying_loc_datasets": int(q["dataset"].isin(LOC_DS).sum()),
           "N_old_cells_all": N, "resolution_all": 3 / np.sqrt(N) if N else None,
           "N_old_cells_loc": Nloc, "resolution_loc": 3 / np.sqrt(Nloc) if Nloc else None,
           "stop_rule": "stop if units < 15 or location-dataset units < 10 (prereg §6.3)",
           "decision": "proceed" if (len(q) >= 15 and q["dataset"].isin(LOC_DS).sum() >= 10) else "stop: insufficient power"}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(T.round(3).to_string(index=False))
print(json.dumps(summary, indent=1, default=float))
