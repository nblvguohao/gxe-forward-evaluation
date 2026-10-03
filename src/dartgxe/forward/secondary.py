"""SX headroom check: forward + same-season pre-harvest notes (docs/prereg_headroom_secondary_2026-09-30.md).

This part: whitelisted note reader, plot -> cell -> within-environment z, leave-own-exclusion-group-out (LOEO)
covariates. No yield value is read here: the notes reader asks pandas for the whitelisted columns only; the diagnostic
readers (yield presence, calendar) are separate functions that never feed a covariate."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["Env", "Year", "Field_Location", "Experiment", "Replicate", "Block", "Plot", "Range", "Pass", "Hybrid"]
NOTES = ["Pollen_DAP_days", "Silk_DAP_days", "Plant_Height_cm", "Ear_Height_cm"]
WHITELIST = KEYS + NOTES
FORBIDDEN = ["Yield_Mg_ha", "Grain_Moisture", "Twt_kg_m3", "Stand_Count_plants", "Root_Lodging_plants",
             "Stalk_Lodging_plants", "Date_Harvested"]
RANGES = {"silk": (30.0, 120.0), "pollen": (30.0, 120.0), "asi": (-20.0, 30.0), "height": (50.0, 450.0),
          "ear": (10.0, 300.0)}
COLS = {"silk": "Silk_DAP_days", "pollen": "Pollen_DAP_days", "height": "Plant_Height_cm", "ear": "Ear_Height_cm"}
MIN_RECORDS, CLIP = 10, 4.0


def raw_path() -> Path:
    return Path(os.environ["DARTGXE_ROOT"]) / "data/raw/g2f2024/1_Training_Trait_Data_2014_2023.csv"


def load_notes(path=None) -> pd.DataFrame:
    """Plot table with the whitelisted columns only (keys as str, notes numeric)."""
    d = pd.read_csv(path or raw_path(), usecols=WHITELIST, dtype={c: str for c in KEYS}, low_memory=False)
    assert set(d.columns) == set(WHITELIST) and not set(d.columns) & set(FORBIDDEN)
    d["Year"] = d["Year"].astype(int)
    for c in NOTES:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def load_yield_presence(path=None) -> pd.DataFrame:
    """Diagnostic only (prereg §6.4): keys + whether a yield value exists. The value itself is dropped at once."""
    d = pd.read_csv(path or raw_path(), usecols=["Env", "Experiment", "Plot", "Hybrid", "Yield_Mg_ha"], dtype=str)
    d["has_yield"] = pd.to_numeric(d.pop("Yield_Mg_ha"), errors="coerce").notna()
    return d


def load_calendar(path=None) -> pd.DataFrame:
    """Diagnostic only (prereg §6.5): planting and harvest dates per plot."""
    return pd.read_csv(path or raw_path(), usecols=["Env", "Year", "Hybrid", "Date_Planted", "Date_Harvested"], dtype=str)


def plot_values(plots: pd.DataFrame, trait: str) -> pd.Series:
    """Per-plot value of a trait; out-of-range -> NaN (ASI from in-range silk and pollen of the same plot)."""
    if trait == "asi":
        s = plots["Silk_DAP_days"].where(plots["Silk_DAP_days"].between(*RANGES["silk"]))
        p = plots["Pollen_DAP_days"].where(plots["Pollen_DAP_days"].between(*RANGES["pollen"]))
        v = s - p
    else:
        v = plots[COLS[trait]]
    return v.where(v.between(*RANGES[trait]))


def note_cells(plots: pd.DataFrame, trait: str) -> pd.DataFrame:
    """Cell (env, genotype) mean over plots with a valid value, then within-environment z over all hybrids with a
    record (genotyped or not), clipped to [-4, 4]. Environments with < 10 records or SD 0 do not supply the trait."""
    v = plot_values(plots, trait)
    c = (pd.DataFrame({"env": plots["Env"], "year": plots["Year"], "genotype": plots["Hybrid"], "v": v})
         .dropna(subset=["v"]).groupby(["env", "year", "genotype"], as_index=False)["v"].mean())
    g = c.groupby("env")["v"]
    n, mu, sd = g.transform("size"), g.transform("mean"), g.transform("std")
    c = c[(n >= MIN_RECORDS) & (sd > 0)].copy()
    c["z"] = ((c["v"] - mu[c.index]) / sd[c.index]).clip(-CLIP, CLIP)
    return c[["env", "year", "genotype", "z"]].reset_index(drop=True)


def year_matrix(z: pd.DataFrame, year: int):
    """Hybrid x environment z matrix of one year (NaN = no record)."""
    m = z[z["year"] == year].pivot(index="genotype", columns="env", values="z")
    return m


def loeo(z: pd.DataFrame, year: int, cells: pd.DataFrame, groups: dict | None = None, exclude_extra=None):
    """LOEO covariate for cells (env, genotype) of `year`: mean z of the hybrid over the year's environments outside
    G(env) (groups: env -> set of envs, default {env}); exclude_extra(env) -> further envs to drop (e.g. same state).
    Returns (x, n_envs_used); x = 0 where no environment is left."""
    M = year_matrix(z, year)
    envs = list(M.columns)
    col = {e: i for i, e in enumerate(envs)}
    V = M.to_numpy()
    present = ~np.isnan(V)
    Vz = np.where(present, V, 0.0)
    row = {g: i for i, g in enumerate(M.index)}
    x = np.zeros(len(cells))
    n = np.zeros(len(cells), dtype=int)
    for e, idx in cells.groupby("env").groups.items():
        drop = set((groups or {}).get(e, {e})) | {e}
        if exclude_extra is not None:
            drop |= set(exclude_extra(e))
        keep = np.ones(len(envs), dtype=bool)
        keep[[col[d] for d in drop if d in col]] = False
        s = Vz[:, keep].sum(1)                          # summed over the kept environments only (bitwise free of j)
        c = present[:, keep].sum(1)
        ri = np.array([row.get(g, -1) for g in cells.loc[idx, "genotype"]])
        ok = ri >= 0
        cc = np.where(ok, c[np.clip(ri, 0, None)], 0)
        ss = np.where(ok, s[np.clip(ri, 0, None)], 0.0)
        pos = cells.index.get_indexer(idx)
        x[pos] = np.where(cc > 0, ss / np.maximum(cc, 1), 0.0)
        n[pos] = cc
    return x, n


def all_env_mean(z: pd.DataFrame, year: int, genotypes) -> np.ndarray:
    """Mean z of each hybrid over every environment of `year` (NY covariate); 0 if none."""
    m = year_matrix(z, year).mean(axis=1)
    return m.reindex(list(genotypes)).fillna(0.0).to_numpy()


# ------------------------------------------------------------------------------------------------ exclusion groups
def duplicate_pairs(plots: pd.DataFrame, region_of) -> pd.DataFrame:
    """Same-year environment pairs in the same state: share of shared hybrids with an identical plot-level
    (pollen, silk, height) tuple, and whether the (Plot, Range, Pass, Hybrid) layouts are identical."""
    p = plots.dropna(subset=["Pollen_DAP_days", "Silk_DAP_days", "Plant_Height_cm"]).copy()
    p["tup"] = list(zip(p["Pollen_DAP_days"], p["Silk_DAP_days"], p["Plant_Height_cm"]))
    tup = p.groupby(["Env", "Hybrid"])["tup"].apply(set)
    q = plots[plots["Plot"].notna()].fillna({"Range": "", "Pass": ""})
    lay = q.assign(k=list(zip(q["Plot"], q["Range"], q["Pass"], q["Hybrid"]))).groupby("Env")["k"].apply(frozenset)
    rows = []
    for y, g in plots.groupby("Year"):
        envs = sorted(g["Env"].unique())
        for i, a in enumerate(envs):
            for b in envs[i + 1:]:
                if region_of(a) != region_of(b):
                    continue
                ha = set(g.loc[g["Env"] == a, "Hybrid"]); hb = set(g.loc[g["Env"] == b, "Hybrid"])
                shared = ha & hb
                same = [h for h in shared if (a, h) in tup.index and (b, h) in tup.index and tup[(a, h)] & tup[(b, h)]]
                comparable = [h for h in shared if (a, h) in tup.index and (b, h) in tup.index]
                rows.append({"year": int(y), "env_a": a, "env_b": b, "shared": len(shared), "comparable": len(comparable),
                             "identical_share": len(same) / len(comparable) if comparable else np.nan,
                             "layout_identical": bool(a in lay.index and b in lay.index and len(lay[a]) > 0 and lay[a] == lay[b])})
    return pd.DataFrame(rows)


def exclusion_groups(pairs: pd.DataFrame, envs, share=0.20) -> dict:
    """Union-find over flagged pairs: env -> sorted list of its group (itself included)."""
    parent = {e: e for e in envs}

    def find(e):
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    flag = pairs[(pairs["identical_share"] > share) | pairs["layout_identical"]] if len(pairs) else pairs
    for a, b in zip(flag.get("env_a", []), flag.get("env_b", [])):
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups = {}
    for e in envs:
        groups.setdefault(find(e), []).append(e)
    return {e: sorted(groups[find(e)]) for e in envs}


# ------------------------------------------------------------------------------------------------ models (§3)
from dartgxe.forward.cellspec import Prep, fit_ia  # noqa: E402

TRAITS_MAIN, TRAITS_H = ["silk"], ["silk", "asi", "height", "ear"]
T_FIRST = 2015
MIN_NEW = 25


def ghat(ds, t: int, extra: pd.DataFrame | None = None) -> tuple[pd.Series, dict]:
    """cell_reml fitted on the years before t, predicting the genotypes of year t (and of `extra`)."""
    train = ds.cells[ds.cells["year"] < t]
    test = ds.cells[ds.cells["year"] == t][["env", "year", "genotype"]]
    if extra is not None:
        test = pd.concat([test, extra[["env", "year", "genotype"]]], ignore_index=True)
    assert train["year"].max() < t
    P = Prep(train, ds.markers, test.assign(y=0.0))
    f = fit_ia(P, np.ones(P.n_env), P.yc, 0.0)
    return pd.Series(f["u_t"], index=P.test_ids), {"delta": f["delta"], "at_edge": f["at_edge"], "n_fit": len(P.fit_ids)}


def xhat(ds, z: pd.DataFrame, t: int, genotypes) -> tuple[pd.Series, dict]:
    """Marker prediction of a note (z cells of the years before t as response) for `genotypes`; no year-t note."""
    train = z[z["year"] < t].rename(columns={"z": "y"})
    g = sorted(set(genotypes))
    P = Prep(train, ds.markers, pd.DataFrame({"env": "x", "year": t, "genotype": g, "y": 0.0}))
    f = fit_ia(P, np.ones(P.n_env), P.yc, 0.0)
    return pd.Series(f["u_t"], index=P.test_ids).reindex(g).fillna(0.0), {"delta": f["delta"]}


def centre(M: np.ndarray, env: np.ndarray) -> np.ndarray:
    _, e = np.unique(env, return_inverse=True)
    n = np.bincount(e).astype(float)
    out = np.empty_like(M, dtype=float)
    for k in range(M.shape[1]):
        out[:, k] = M[:, k] - (np.bincount(e, M[:, k]) / n)[e]
    return out


def ols(y, X, env):
    """Within-environment OLS without intercept; columns with ~zero variance are dropped (coefficient 0)."""
    Xc, yc = centre(np.asarray(X, float), env), centre(np.asarray(y, float)[:, None], env)[:, 0]
    keep = Xc.var(0) > 1e-12
    coef = np.zeros(Xc.shape[1])
    coef[keep] = np.linalg.lstsq(Xc[:, keep], yc, rcond=None)[0]
    return coef, keep


def first_year(ds) -> pd.Series:
    return ds.cells.groupby("genotype")["year"].min()


def sx_year(ds, Y: int, envs, notes: dict, ny_envs=None, groups=None):
    """All SX models for target Y. notes: trait -> z cells (secondary.note_cells). Returns (panel without y, NY panel
    without y, meta). Uses yields of years < Y only; Y's notes only via LOEO / all-environment means."""
    first = first_year(ds)
    region = lambda e: cellspec_location(e)
    tgt = ds.cells[(ds.cells["year"] == Y) & ds.cells["env"].isin(list(envs))][["env", "year", "genotype"]]
    tgt = tgt[tgt["genotype"].isin(ds.markers.index)].reset_index(drop=True)
    ny = pd.DataFrame(columns=["env", "year", "genotype"])
    if ny_envs is not None:
        hy = set(tgt["genotype"])
        ny = ds.cells[(ds.cells["year"] == Y + 1) & ds.cells["env"].isin(list(ny_envs)) & ds.cells["genotype"].isin(hy)]
        ny = ny[["env", "year", "genotype"]].reset_index(drop=True)
    gY, mY = ghat(ds, Y)
    meta = {"target": Y, "delta_Y": mY["delta"], "n_fit_Y": mY["n_fit"], "calib": {}}
    # ---- calibration rows (years T_FIRST .. Y-1, new-at-t cells in envs with >= MIN_NEW of them)
    rows = []
    for t in range(T_FIRST, Y):
        c = ds.cells[ds.cells["year"] == t]
        c = c[c["genotype"].map(first).ge(t).to_numpy() & c["genotype"].isin(ds.markers.index).to_numpy()]
        per = c.groupby("env").size()
        c = c[c["env"].isin(per[per >= MIN_NEW].index)].reset_index(drop=True)
        if c.empty:
            continue
        g_t, m_t = ghat(ds, t)
        r = c[["env", "year", "genotype", "y"]].copy()
        r["g"] = g_t.reindex(r["genotype"]).to_numpy()
        for m, z in notes.items():
            r[f"x_{m}"] = loeo(z, t, r, groups)[0]
            r[f"n_{m}"] = all_env_mean(z, t, r["genotype"])      # own-environment (naive) covariate
        r["xh_silk"] = xhat(ds, notes["silk"], t, r["genotype"])[0].reindex(r["genotype"]).to_numpy()
        rows.append(r)
        meta["calib"][t] = {"envs": int(r["env"].nunique()), "cells": len(r), "delta": m_t["delta"]}
    C = pd.concat(rows, ignore_index=True)
    assert C["year"].max() < Y and not C["g"].isna().any()
    # naive own-environment covariate must include env j itself: all_env_mean over the year's environments
    fits = {}
    spec = {"s1": [f"x_{m}" for m in TRAITS_MAIN], "s1h": [f"x_{m}" for m in TRAITS_H],
            "s1g": ["xh_silk"], "s1n": [f"n_{m}" for m in TRAITS_MAIN]}
    for name, cols in spec.items():
        coef, keep = ols(C["y"], C[["g"] + cols].to_numpy(), C["env"].to_numpy())
        fits[name] = {"a": float(coef[0]), "b": [float(v) for v in coef[1:]], "cols": cols, "kept": keep.tolist()}
    meta["fits"] = fits
    meta["calib_cells"], meta["calib_envs"] = len(C), int(C.groupby(["year", "env"]).ngroups)
    # ---- target covariates
    T = tgt.copy()
    T["g"] = gY.reindex(T["genotype"]).to_numpy()
    for m, z in notes.items():
        T[f"x_{m}"], T[f"k_{m}"] = loeo(z, Y, T, groups)
        T[f"n_{m}"] = all_env_mean(z, Y, T["genotype"])
    T["xh_silk"] = xhat(ds, notes["silk"], Y, T["genotype"])[0].reindex(T["genotype"]).to_numpy()
    reg = {e: region(e) for e in set(ds.cells.loc[ds.cells["year"] == Y, "env"]) | set(notes["silk"].loc[notes["silk"]["year"] == Y, "env"])}
    same_state = lambda e: [f for f, r in reg.items() if r == reg.get(e)]
    T["x_silk_lor"] = loeo(notes["silk"], Y, T, groups, exclude_extra=same_state)[0]
    T["s0"] = T["g"]
    for name, f in fits.items():
        T[name] = f["a"] * T["g"] + T[f["cols"]].to_numpy() @ np.array(f["b"])
    f1 = fits["s1"]
    T["s1_lor"] = f1["a"] * T["g"] + T[["x_silk_lor"]].to_numpy() @ np.array(f1["b"])
    T["new"] = T["genotype"].map(first).ge(Y).to_numpy()
    # ---- NY cells: Y+1 yields of Y's scored hybrids, same g (years < Y), Y's all-environment silk mean
    N = ny.copy()
    if len(N):
        N["g"] = gY.reindex(N["genotype"]).to_numpy()
        N["xbar_silk"] = all_env_mean(notes["silk"], Y, N["genotype"])
        N["s0"] = N["g"]
        N["s1"] = f1["a"] * N["g"] + N[["xbar_silk"]].to_numpy() @ np.array(f1["b"])
    meta.update({"n_target": len(T), "n_ny": len(N), "share_target_no_loeo": float((T["k_silk"] == 0).mean())})
    return T, N, meta


def cellspec_location(env):
    from dartgxe.forward import cellspec
    return cellspec.location("G2F", env, "region")
