"""Structure count for SX (docs/prereg_headroom_secondary_2026-09-30.md §6). No model is fitted and no yield value is
read: yield cells are used as keys (env, year, genotype) only; the raw table is read through the whitelisted reader,
plus the presence-only and calendar readers for the two diagnostics.

Environment: DARTGXE_ROOT, DARTGXE_DATA, BENCH, OUT (results/count_secondary). Output: OUT/{per_year.csv, calib.csv,
dup_pairs.csv, exclusion_groups.json, presence.csv, calendar.csv, ny.csv, power.json, summary.json}."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dartgxe.forward import cellspec, secondary
from dartgxe.paths import processed

BENCH, OUT = Path(os.environ["BENCH"]), Path(os.environ["OUT"])
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[1]
SPL = json.load(open(BENCH / "splits.json"))["G2F"]
TARGETS = [2016, 2018, 2020, 2022]
MIN_N = 25
region = lambda e: cellspec.location("G2F", e, "region")

keys = pd.read_parquet(processed("g2f") / "pheno.parquet", columns=["env", "year", "genotype"])
keys["year"] = keys["year"].astype(int)
first = keys.groupby("genotype")["year"].min()
plots = secondary.load_notes()
silk = secondary.note_cells(plots, "silk")

# ---- 3. duplicates and exclusion groups (all years 2014-2023, same-state pairs)
pairs = secondary.duplicate_pairs(plots, region)
pairs.to_csv(OUT / "dup_pairs.csv", index=False)
groups = secondary.exclusion_groups(pairs, sorted(plots["Env"].unique()))
multi = {e: g for e, g in groups.items() if len(g) > 1}
json.dump(groups, open(OUT / "exclusion_groups.json", "w"), indent=1)
uniq = {"env_experiment_plot_unique": bool(not plots.dropna(subset=["Plot"]).duplicated(["Env", "Experiment", "Plot"]).any()),
        "env_experiment_range_pass_unique": bool(not plots.dropna(subset=["Range", "Pass"]).duplicated(["Env", "Experiment", "Range", "Pass"]).any())}
dup_counts = {"n_dup_plot_keys": int(plots.dropna(subset=["Plot"]).duplicated(["Env", "Experiment", "Plot"]).sum()),
              "n_dup_range_pass": int(plots.dropna(subset=["Range", "Pass"]).duplicated(["Env", "Experiment", "Range", "Pass"]).sum())}


def coverage(cells, year):
    x, n = secondary.loeo(silk, year, cells, groups)
    return n


# ---- 1, 2, 7. scored cells, LOEO coverage, new-hybrid scope
rows = []
for Y in TARGETS:
    envs = SPL["scorable_environments"][str(Y)]
    c = keys[(keys["year"] == Y) & keys["env"].isin(envs)].reset_index(drop=True)
    per_env = c.groupby("env").size()
    c = c[c["env"].isin(per_env[per_env >= MIN_N].index)].reset_index(drop=True)
    n = coverage(c, Y)
    new = c["genotype"].map(first).ge(Y).to_numpy()
    newenv = c[new].groupby("env").size()
    ok_new = newenv[newenv >= MIN_N].index
    rows.append({"year": Y, "scored_envs": int(c["env"].nunique()), "N": len(c), "share_cells_loeo_ge2": float((n >= 2).mean()),
                 "share_cells_loeo_ge1": float((n >= 1).mean()), "envs_in_multi_groups": int(sum(e in multi for e in c["env"].unique())),
                 "new_scope_envs": len(ok_new), "new_scope_cells": int(c[new & c["env"].isin(ok_new).to_numpy()].shape[0])})
PY = pd.DataFrame(rows)
PY.to_csv(OUT / "per_year.csv", index=False)

# ---- 2. calibration data: new-at-t cells, t = 2015 .. 2021, envs with >= 25 new-at-t hybrids
cal = []
for t in range(2015, 2022):
    c = keys[keys["year"] == t].reset_index(drop=True)
    c = c[c["genotype"].map(first).ge(t).to_numpy()]
    per = c.groupby("env").size()
    c = c[c["env"].isin(per[per >= MIN_N].index)].reset_index(drop=True)
    n = coverage(c, t) if len(c) else np.array([])
    cal.append({"t": t, "envs": int(c["env"].nunique()), "cells": len(c), "share_loeo_ge2": float((n >= 2).mean()) if len(n) else np.nan})
CAL = pd.DataFrame(cal)
CAL.to_csv(OUT / "calib.csv", index=False)
per_target = {Y: {"envs": int(CAL[CAL["t"] < Y]["envs"].sum()), "cells": int(CAL[CAL["t"] < Y]["cells"].sum())} for Y in TARGETS}

# ---- 4. note presence vs yield presence (presence only)
pres = secondary.load_yield_presence()
pres["Year"] = plots["Year"].to_numpy()
pres["has_silk"] = plots["Silk_DAP_days"].notna().to_numpy()
pres["has_height"] = plots["Plant_Height_cm"].notna().to_numpy()
P = pres.groupby(["Year", "has_yield"])[["has_silk", "has_height"]].mean().reset_index()
P.to_csv(OUT / "presence.csv", index=False)

# ---- 5. calendar
cal_raw = secondary.load_calendar()
cal_raw["planted"] = pd.to_datetime(cal_raw["Date_Planted"], errors="coerce")
cal_raw["harvested"] = pd.to_datetime(cal_raw["Date_Harvested"], errors="coerce")
cal_raw["Year"] = cal_raw["Year"].astype(int)
ms = plots.groupby("Env")["Silk_DAP_days"].median()
E = cal_raw.groupby("Env").agg(year=("Year", "first"), planted=("planted", "median"), harvested=("harvested", "median"))
E["silk_date"] = E["planted"] + pd.to_timedelta(ms.reindex(E.index), unit="D")
calrows = []
for Y in TARGETS:
    e = E[E["year"] == Y]
    last_silk = e["silk_date"].max()
    envs = SPL["scorable_environments"][str(Y)]
    sc = e[e.index.isin(envs)]
    early = sc["harvested"] < last_silk
    ncell = keys[(keys["year"] == Y) & keys["env"].isin(envs)].groupby("env").size()
    calrows.append({"year": Y, "last_median_silk_date": str(last_silk.date()) if pd.notna(last_silk) else None,
                    "scored_envs_with_harvest_date": int(sc["harvested"].notna().sum()),
                    "share_scored_envs_harvested_before": float(early.mean()),
                    "share_scored_cells_harvested_before": float(ncell.reindex(sc.index[early]).sum() / ncell.sum())})
CR = pd.DataFrame(calrows)
CR.to_csv(OUT / "calendar.csv", index=False)

# ---- 6. NY scoring set
ny = []
for Y in TARGETS:
    envs = SPL["scorable_environments"][str(Y)]
    hy = set(keys[(keys["year"] == Y) & keys["env"].isin(envs)]["genotype"])
    c = keys[(keys["year"] == Y + 1) & keys["genotype"].isin(hy)]
    per = c.groupby("env").size()
    ok = per[per >= MIN_N]
    ny.append({"year": Y, "ny_year": Y + 1, "envs": len(ok), "cells": int(ok.sum())})
NY = pd.DataFrame(ny)
NY.to_csv(OUT / "ny.csv", index=False)

# ---- 8. power proxies from already-seen results
ce = pd.read_csv(ROOT / "results/ideas_diag/ceiling_envs.csv")
ce = ce[(ce["dataset"] == "G2F") & ce["target"].astype(str).isin([str(y) for y in TARGETS])]
ce["d"] = ce["gm_ceiling"] - ce["cell_reml_same_set"]
s48 = pd.read_parquet(ROOT / "results/summary48/per_env.parquet")
s48 = s48[(s48["dataset"] == "G2F") & (s48["item"] == "rn_ridge") & s48["target"].astype(int).isin(TARGETS)]


def mde(df, col, tcol):
    g = df.groupby(tcol)[col]
    sd_env = float(g.std().mean())
    tau = float(g.mean().std())
    n_env = g.size()
    se_y = sd_env / np.sqrt(n_env)
    se = 1 / np.sqrt((1 / (se_y ** 2 + tau ** 2)).sum())
    return {"sd_env": sd_env, "sd_year": tau, "pooled_se": float(se), "mde80": float(2.80 * se)}


power = {"upper_ceiling_gap": mde(ce, "d", "target"), "lower_rn_ridge": mde(s48.assign(target=s48["target"].astype(int)), "d_spearman", "target")}
json.dump(power, open(OUT / "power.json", "w"), indent=1)

N = int(PY["N"].sum())
cov_ok = int((PY["share_cells_loeo_ge2"] >= 0.8).sum())
cal_ok = sum(1 for Y in TARGETS if CAL[CAL["t"] < Y]["envs"].sum() >= 5)
summary = {"N": N, "resolution": 3 / np.sqrt(N), "coverage_years_ok": cov_ok, "calibration_years_ok": cal_ok,
           "calibration_per_target": per_target, "n_multi_env_groups": len({tuple(g) for g in multi.values()}),
           "multi_groups": sorted({tuple(g) for g in multi.values()}), **uniq, **dup_counts,
           "decision": "proceed" if cov_ok >= 3 and cal_ok >= 3 else "stop: coverage insufficient"}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=str)
pd.set_option("display.width", 220)
for T in (PY, CAL, P, CR, NY):
    print(T.round(3).to_string(index=False))
print(json.dumps(power, indent=1))
print(json.dumps(summary, indent=1, default=str))
