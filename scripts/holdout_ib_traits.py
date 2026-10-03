"""I-B holdout on other traits (docs/prereg_holdout_ib_2026-09-29.md).
  python scripts/holdout_ib_traits.py count                      # structure count only (done, bc385e0)
  python scripts/holdout_ib_traits.py precheck                   # the panel path reproduces the existing yield panels exactly
  python scripts/holdout_ib_traits.py panels NUST:Height ...     # one panel per (trait, target year), control + ib_main (+ ib_region)
  python scripts/holdout_ib_traits.py analyze                    # refuses to run until every panel exists; scores once
Environment: GP_DATA (NUST/URSN raw), OUT, W1C, YIELD_PANELS (dir of results/headroom_ia_ib panels, precheck only),
HEADROOM_COMMIT, plus the variables of dartgxe.forward.data."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from dartgxe.forward import cellspec
from dartgxe.forward import traitcells as tc
from dartgxe.forward.data import LOADERS, Dataset
from dartgxe.forward.pool import dersimonian_laird, floor_se

OUT = Path(os.environ["OUT"])
GP = Path(os.environ["GP_DATA"])
MIN_TARGET_YEARS = 5


def count():
    t0 = time.time()
    (OUT / "count").mkdir(parents=True, exist_ok=True)
    nust, ursn = LOADERS["NUST"](), LOADERS["URSN"]()
    ph = pd.read_csv(GP / "raw/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz", low_memory=False)
    ur = pd.read_csv(GP / "raw/ursn/data/URSN_cleaned_table_phenot_1995_2024.tsv", sep="\t")
    # the count must reproduce the yield-based target years already in use (W1c): same code path, same rules
    w1c = pd.read_csv(Path(os.environ["W1C"]))
    checks = {}
    for name, ds in (("NUST", nust), ("URSN", ursn)):
        t = tc.count_targets(ds.cells, ds.min_n)
        mine = sorted(int(y) for y in t[(t["rule"] == "main") & t["qualifies"]]["year"])
        ref = sorted(int(y) for y in w1c[(w1c["dataset"] == name) & (w1c["rule"] == "main") & w1c["qualifies"]]["year"])
        checks[name] = {"yield_target_years": mine, "w1c_target_years": ref, "same": mine == ref}
    rows, cells_by = [], {}
    specs = [("NUST", tr, nust) for tr in tc.NUST_TRAITS] + [("URSN", tr, ursn) for tr in tc.URSN_TRAITS]
    for name, trait, ds in specs:
        c = (tc.nust_trait_cells(ph, trait, ds.markers.index) if name == "NUST"
             else tc.ursn_trait_cells(ur, trait, ds.markers.index))
        t = tc.count_targets(c, ds.min_n)
        t.insert(0, "trait", trait)
        t.insert(0, "dataset", name)
        rows.append(t)
        cells_by[(name, trait)] = c
        m = t[(t["rule"] == "main") & t["qualifies"]]
        print(name, trait, "records", len(c), "cells", "envs", c["env"].nunique(), "main target years", len(m),
              "relaxed", int(((t["rule"] == "relaxed") & t["qualifies"]).sum()), flush=True)
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(OUT / "count" / "per_trait_year.csv", index=False)

    included = {}
    for (name, trait), g in tab[(tab["rule"] == "main") & tab["qualifies"]].groupby(["dataset", "trait"]):
        if len(g) >= MIN_TARGET_YEARS:
            included[f"{name}:{trait}"] = sorted(int(y) for y in g["year"])
    excluded = {f"{n}:{t}": int(((tab["dataset"] == n) & (tab["trait"] == t) & (tab["rule"] == "main") & tab["qualifies"]).sum())
                for n, t, _ in specs if f"{n}:{t}" not in included}
    # units (dataset x target year) and distinct scored cells over the included traits
    units, union_cells = {}, {}
    for key, years in included.items():
        name, trait = key.split(":")
        min_n = LOADERSMIN[name]
        c = cells_by[(name, trait)]
        n_env = c.groupby(["year", "env"])["genotype"].nunique().rename("n").reset_index()
        for y in years:
            ok = set(n_env[(n_env["year"] == y) & (n_env["n"] >= min_n)]["env"])
            sub = c[(c["year"] == y) & c["env"].isin(ok)]
            union_cells.setdefault((name, y), set()).update(zip(sub["env"], sub["genotype"]))
            units.setdefault((name, y), []).append(trait)
    n_cells = {f"{n}_{y}": len(s) for (n, y), s in union_cells.items()}
    N = sum(n_cells.values())
    verdict = {"included_traits_main": included, "excluded_traits_n_target_years": excluded, "n_units": len(units),
               "units": {f"{n}_{y}": v for (n, y), v in sorted(units.items())}, "distinct_cells_per_unit": n_cells,
               "N_distinct_cells": N, "resolution_3_over_sqrtN": (3 / np.sqrt(N)) if N else None,
               "units_by_dataset": {d: sum(1 for (n, _) in units if n == d) for d in ("NUST", "URSN")},
               "decision": "proceed" if len(units) >= 5 else "stop: fewer than 5 units (prereg §3, §6.4)",
               "checks": checks, "host": socket.gethostname(), "seconds": round(time.time() - t0)}
    json.dump(verdict, open(OUT / "count" / "included.json", "w"), indent=1)
    json.dump(checks, open(OUT / "count" / "checks.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print(json.dumps(verdict, indent=1))
    print(tab[(tab["rule"] == "main")].pivot(index="year", columns=["dataset", "trait"], values="qualifies").fillna(False).astype(int).to_string())
    print("HOLDOUT_COUNT_DONE")


LOADERSMIN = {"NUST": 25, "URSN": 10}
FROZEN_BLOB = "4ca8090345cc73aca5fde9b9ba31483031fdf8a3"   # git blob of src/dartgxe/forward/cellspec.py (prereg §1)
B, SEED = 2000, 20260930


def assert_frozen():
    b = Path(cellspec.__file__).read_bytes()
    got = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
    assert got == FROZEN_BLOB, f"cellspec.py is not the frozen version: {got}"


def trait_dataset(name: str, trait: str, base: Dataset, raw) -> Dataset:
    cells = (tc.nust_trait_cells(raw, trait, base.markers.index) if name == "NUST"
             else tc.ursn_trait_cells(raw, trait, base.markers.index))
    return Dataset(f"{name}:{trait}", cells, base.markers, base.min_n)


def panel(ds: Dataset, name: str, Y: int, region: bool):
    """Control (fit_ia with omega = 0, s = 1, as in cellspec.ia_family) and ib_main [+ ib_region] for year Y, trained on
    the years before Y only (same steps as cellspec.panel_year, without the I-A family)."""
    envs = ds.scorable_envs([Y])
    train = ds.cells[ds.cells["year"] < Y]
    test = ds.cells[(ds.cells["year"] == Y) & ds.cells["env"].isin(envs)]
    assert train["year"].max() < Y and set(test["year"]) == {Y}
    P = cellspec.Prep(train, ds.markers, test)
    out = P.test[["env", "year", "genotype", "y"]].copy()
    ti = out["genotype"].map({g: i for i, g in enumerate(P.test_ids)}).to_numpy()
    out["ctrl"] = cellspec.fit_ia(P, np.ones(P.n_env), P.yc, 0.0)["u_t"][ti]
    ib, meta_ib = cellspec.ib_family(P, name, "location")
    out["ib_main"] = ib["main"]
    out["ib_active"] = ib["active_cell"]
    out["ib_c0"] = ib["grid"][0.0]
    metas = {"n_fit": len(P.fit_ids), "n_train_cells": len(P.train), "ib_main": meta_ib}
    if region and cellspec.location(name, str(out["env"].iloc[0]), "region") is not None:
        ibr, meta_r = cellspec.ib_family(P, name, "region")
        out["ib_region"] = ibr["main"]
        metas["ib_region"] = meta_r
    return out, metas


def run_meta():
    import torch
    return {"commit": os.environ.get("HEADROOM_COMMIT", "unknown"), "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version(), "cellspec_blob": FROZEN_BLOB,
            "config": {"C2_GRID": cellspec.C2_GRID, "K_GL": cellspec.K_GL, "MIN_LOC_YEARS": cellspec.MIN_LOC_YEARS,
                       "MIN_LOC_CELLS": cellspec.MIN_LOC_CELLS}}


def load_all():
    nust, ursn = LOADERS["NUST"](), LOADERS["URSN"]()
    ph = pd.read_csv(GP / "raw/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz", low_memory=False)
    ur = pd.read_csv(GP / "raw/ursn/data/URSN_cleaned_table_phenot_1995_2024.tsv", sep="\t")
    return {"NUST": (nust, ph), "URSN": (ursn, ur)}


def spearman(y, p):
    if len(y) < 3 or np.std(p) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


def precheck():
    """The panel path above must reproduce the ctrl / ib_main columns of the existing yield panels exactly."""
    assert_frozen()
    data = load_all()
    res = {}
    for name, years in (("NUST", (2005, 2012)), ("URSN", (2003,))):
        ds, _ = data[name]
        for Y in years:
            out, _ = panel(ds, name, Y, region=False)
            ref = pd.read_parquet(Path(os.environ["YIELD_PANELS"]) / f"{name}_{Y}.parquet")
            m = out.merge(ref[["env", "genotype", "ctrl", "ib_main", "ib_active"]], on=["env", "genotype"], suffixes=("", "_ref"))
            assert len(m) == len(out) == len(ref)
            sd = float(ref["ctrl"].std())
            res[f"{name}_{Y}"] = {"max_rel_ctrl": float((m["ctrl"] - m["ctrl_ref"]).abs().max() / sd),
                                  "max_rel_ib_main": float((m["ib_main"] - m["ib_main_ref"]).abs().max() / sd),
                                  "active_same": bool((m["ib_active"] == m["ib_active_ref"]).all()),
                                  "ctrl_vs_ib_c0": float((m["ctrl"] - m["ib_c0"]).abs().max() / sd)}
            print(name, Y, res[f"{name}_{Y}"], flush=True)
    ok = all(v["max_rel_ctrl"] < 1e-6 and v["max_rel_ib_main"] < 1e-6 and v["active_same"] and v["ctrl_vs_ib_c0"] < 1e-6
             for v in res.values())
    json.dump({"pass": ok, "cases": res}, open(OUT / "precheck.json", "w"), indent=1)
    print("PRECHECK", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


def panels(specs):
    assert_frozen()
    inc = json.load(open(OUT / "count" / "included.json"))["included_traits_main"]
    data = load_all()
    (OUT / "panels").mkdir(parents=True, exist_ok=True)
    for spec in specs:
        name, trait = spec.split(":")
        base, raw = data[name]
        ds = trait_dataset(name, trait, base, raw)
        for Y in inc[spec]:
            f = OUT / "panels" / f"{name}_{trait}_{Y}.parquet"
            if f.exists():
                continue
            t0, start = time.time(), dt.datetime.now().isoformat(timespec="seconds")
            out, meta = panel(ds, name, Y, region=(name == "NUST"))
            meta.update(run_meta(), dataset=name, trait=trait, target=Y, start=start,
                        end=dt.datetime.now().isoformat(timespec="seconds"), seconds=round(time.time() - t0, 1),
                        n_cells=int(len(out)), ctrl_vs_c0=float((out["ctrl"] - out["ib_c0"]).abs().max() / out["ctrl"].std()))
            out.to_parquet(f, index=False)
            json.dump(meta, open(f.with_suffix(".json"), "w"), indent=1, default=str)
            print(name, trait, Y, f"{meta['seconds']}s", "c2", meta["ib_main"]["c2_main"], "active", meta["ib_main"]["n_active"],
                  flush=True)


def analyze():
    inc = json.load(open(OUT / "count" / "included.json"))
    need = [(k.split(":")[0], k.split(":")[1], y) for k, ys in inc["included_traits_main"].items() for y in ys]
    missing = [f"{n}_{t}_{y}" for n, t, y in need if not (OUT / "panels" / f"{n}_{t}_{y}.parquet").exists()]
    if missing:
        sys.exit(f"panels missing: {missing}")
    assert json.load(open(OUT / "precheck.json"))["pass"], "precheck did not pass"
    rng = np.random.default_rng(SEED)
    rows, cell_sets = [], {}
    for n, t, y in need:
        P = pd.read_parquet(OUT / "panels" / f"{n}_{t}_{y}.parquet")
        cell_sets.setdefault((n, y), set()).update(zip(P["env"], P["genotype"]))
        for e, g in P.groupby("env"):
            yv = g["y"].to_numpy(float)
            s0 = spearman(yv, g["ctrl"].to_numpy(float))
            rows.append({"dataset": n, "trait": t, "target": y, "env": e, "n": len(g), "spearman_ctrl": s0,
                         "d": spearman(yv, g["ib_main"].to_numpy(float)) - s0,
                         "d_region": (spearman(yv, g["ib_region"].to_numpy(float)) - s0) if "ib_region" in g else np.nan,
                         "active": bool(g["ib_active"].all())})
    E = pd.DataFrame(rows)
    E.to_parquet(OUT / "per_env_trait.parquet", index=False)
    N = sum(len(s) for s in cell_sets.values())
    assert N == inc["N_distinct_cells"], (N, inc["N_distinct_cells"])
    res = 3 / np.sqrt(N)

    def unit_effects(df, col="d"):
        """Prereg §3: trait-average per environment, then the year effect and its environment-bootstrap SE."""
        env = df.groupby(["dataset", "target", "env"])[col].mean().dropna().reset_index()
        out = []
        for (n, y), g in env.groupby(["dataset", "target"]):
            v = g[col].to_numpy()
            bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
            out.append({"dataset": n, "target": y, "d": float(v.mean()), "se": float(bs.std(ddof=1)), "n_env": len(v)})
        return pd.DataFrame(out)

    def pooled(ye):
        if ye.empty:
            return {}
        r = dersimonian_laird(floor_se(ye))
        return {k: r[k] for k in ("est", "lo", "hi", "tau2", "k")}

    UE = unit_effects(E)
    UE.to_csv(OUT / "unit_effects.csv", index=False)
    prim = pooled(UE)
    verdict_txt = ("优于" if prim["est"] >= res and prim["lo"] > 0 else
                   "统计上可检测，但低于选择分辨率" if prim["lo"] > 0 else "并列")
    verdict = {"H_hold": {**prim, "resolution": res, "N": N, "n_units": len(UE), "judgement": verdict_txt},
               "by_dataset": {d: pooled(UE[UE["dataset"] == d]) for d in ("NUST", "URSN")}}
    # descriptive checks (prereg §4)
    desc = []
    for (n, t), g in E.groupby(["dataset", "trait"]):
        ye = unit_effects(g.assign(dataset=n))
        desc.append({"check": "per_trait", "key": f"{n}:{t}", **pooled(ye)})
    for (n, t) in sorted(set(zip(E["dataset"], E["trait"]))):
        ye = unit_effects(E[~((E["dataset"] == n) & (E["trait"] == t))])
        desc.append({"check": "leave_one_trait_out", "key": f"{n}:{t}", **pooled(ye)})
    for flag in (True, False):
        ye = unit_effects(E[E["active"] == flag])
        desc.append({"check": "active_environments" if flag else "non_active_environments", "key": "all", "n_env": int(E["active"].eq(flag).sum()),
                     **pooled(ye)})
    reg = E.dropna(subset=["d_region"])
    desc.append({"check": "ib_region", "key": "NUST", **pooled(unit_effects(reg, "d_region"))})
    desc.append({"check": "control_level_spearman", "key": "all", "est": float(E["spearman_ctrl"].mean())})
    D = pd.DataFrame(desc)
    D.to_csv(OUT / "descriptive.csv", index=False)
    metas = [json.load(open(f)) for f in sorted((OUT / "panels").glob("*.json"))]
    verdict["c2_main"] = {"median": float(np.median([m["ib_main"]["c2_main"] for m in metas])),
                          "share_zero": float(np.mean([m["ib_main"]["c2_main"] == 0 for m in metas])),
                          "share_at_grid_top": float(np.mean([m["ib_main"]["c2_main"] == max(cellspec.C2_GRID) for m in metas]))}
    verdict["ctrl_vs_c0_max"] = float(max(m["ctrl_vs_c0"] for m in metas))
    json.dump(verdict, open(OUT / "verdict.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(json.dumps(verdict, indent=1, default=float))
    print(D.round(4).to_string(index=False))
    print(UE.round(4).to_string(index=False))
    print("HOLDOUT_ANALYZE_DONE")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "count":
        count()
    elif cmd == "precheck":
        precheck()
    elif cmd == "panels":
        panels(sys.argv[2:])
    elif cmd == "analyze":
        analyze()
