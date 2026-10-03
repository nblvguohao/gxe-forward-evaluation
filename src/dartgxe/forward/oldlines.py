"""OL headroom check: old lines x new year (docs/prereg_headroom_oldlines_2026-09-30.md).

Training = every genotyped cell of the years before Y (no target-year phenotype). Scored = cells of old lines (lines
with a cell before Y) in the target year's scored environments that have >= min_n old lines; a target year is used
when at least 5 such environments exist. Models (cell-level, environment fixed effects, genotype-level solver):

OL0  cell_reml (fit_ia with w = 0).
OL1  main effect with the genotype's own history: K(w) = (1 - w) K_A + w I, w by profile REML (= I-A's ia_g).
OL2  OL1's main block (w fixed) + marker G x location blocks as in I-B (k = 100, active locations: >= 2 years and
     >= 50 cells, I-B location mapping), c2 by profile REML (c2 = 0 is OL1).
OL3  OL1 + shrunken line x location mean of OL1's training residuals (one-way random effect with zero mean, variance
     components by EM; with no fixed effect in this model ML and REML coincide).
OL4  the same correction applied to OL2.

Imports the frozen cellspec (blob 4ca8090) and sparse without modifying them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dartgxe.forward import cellspec, sparse
from dartgxe.forward.cellspec import C2_GRID, K_GL, MIN_LOC_CELLS, MIN_LOC_YEARS, W_GRID, Prep, fit_ia, reml_solve

MIN_ENVS = 5
EM_ITER, EM_TOL = 200, 1e-10


# ------------------------------------------------------------------------------------------------ structure
def old_line_targets(ds, year: int, envs):
    """Scored cells (old lines in qualifying environments), whether the year qualifies, and the qualifying
    environments. Uses only which cells exist, never phenotype values."""
    first = ds.cells.groupby("genotype")["year"].min()
    tgt = ds.cells[(ds.cells["year"] == year) & ds.cells["env"].isin(list(envs))
                   & ds.cells["genotype"].isin(ds.markers.index)]
    old = tgt[tgt["genotype"].map(first) < year]
    n = old.groupby("env")["genotype"].nunique()
    ok_envs = sorted(n[n >= ds.min_n].index)
    scored = old[old["env"].isin(ok_envs)].reset_index(drop=True)
    return scored, len(ok_envs) >= MIN_ENVS, ok_envs


# ------------------------------------------------------------------------------------------------ one-way random effect
def em_oneway(r: np.ndarray, groups: np.ndarray) -> dict:
    """Variance components of r = v_group + e (zero mean) by EM, start s2v = s2e = var(r)/2, at most EM_ITER steps."""
    _, g = np.unique(groups, return_inverse=True)
    n = np.bincount(g).astype(float)
    S = np.bincount(g, r)
    N = len(r)
    s2v = s2e = max(float(np.var(r)) / 2, 1e-12)
    converged, it = False, 0
    for it in range(1, EM_ITER + 1):
        denom = n * s2v + s2e
        m = s2v * S / denom                              # posterior means of v
        v = s2v * s2e / denom                            # posterior variances of v
        s2v_new = float(np.mean(m ** 2 + v))
        s2e_new = float((np.sum((r - m[g]) ** 2) + np.sum(n * v)) / N)
        step = max(abs(s2v_new - s2v) / max(s2v, 1e-12), abs(s2e_new - s2e) / max(s2e, 1e-12))
        s2v, s2e = max(s2v_new, 0.0), max(s2e_new, 1e-12)
        if step < EM_TOL:
            converged = True
            break
    return {"s2v": s2v, "s2e": s2e, "iterations": it, "converged": converged}


def shrunken_group_means(r: np.ndarray, groups: np.ndarray, s2v: float, s2e: float) -> dict:
    codes, g = np.unique(groups, return_inverse=True)
    n = np.bincount(g).astype(float)
    mean = np.bincount(g, r) / n
    return dict(zip(codes, n * s2v / (n * s2v + s2e) * mean))


def env_centred(x: np.ndarray, ej: np.ndarray) -> np.ndarray:
    nj = np.bincount(ej).astype(float)
    return x - (np.bincount(ej, x) / nj)[ej]


def pair_keys(genotypes, locations) -> np.ndarray:
    return np.array([f"{g}|{l}" for g, l in zip(genotypes, locations)], dtype=object)


def line_location_correction(P: Prep, fitted_train: np.ndarray, tr_loc, sc_geno, sc_loc):
    """OL3/OL4 step: shrunken line x location means of the training residuals y_c - centred(fitted), looked up for
    the scored cells (0 for a line never tested at that location)."""
    r = P.yc - env_centred(fitted_train, P.ej)
    grp = pair_keys(P.train["genotype"], tr_loc)
    vc = em_oneway(r, grp)
    eff = shrunken_group_means(r, grp, vc["s2v"], vc["s2e"])
    add = np.array([eff.get(k, 0.0) for k in pair_keys(sc_geno, sc_loc)])
    return add, vc


# ------------------------------------------------------------------------------------------------ OL2
def gxl_basis(P: Prep, dataset: str):
    """Pk (training genotypes, K_A^(k) = Pk Pk'), active locations and each training environment's location."""
    keep0 = P.lam > 1e-8 * P.lam.max()
    k = int(min(K_GL, len(P.fit_ids) - 1, keep0.sum()))
    order = np.argsort(-P.lam)[:k]
    Pk = P.U[:, order] * np.sqrt(P.lam[order])
    env_loc = np.array([cellspec.location(dataset, e, "location") for e in P.envs], dtype=object)
    yrs = P.train.groupby("env")["year"].first().reindex(P.envs).to_numpy()
    agg = pd.DataFrame({"loc": env_loc, "year": yrs, "n": P.nj}).groupby("loc").agg(
        years=("year", "nunique"), cells=("n", "sum"))
    active = sorted(agg.index[(agg["years"] >= MIN_LOC_YEARS) & (agg["cells"] >= MIN_LOC_CELLS)])
    return Pk, active, env_loc


def gxl_pieces(P: Prep, T0: np.ndarray, Pk: np.ndarray, active, env_loc):
    A, C, R = [], [], []
    for loc in active:
        wm = (env_loc == loc).astype(float)             # environments are nested in locations
        A.append(P.cross(Pk, Pk, wm))
        C.append(P.cross(T0, Pk, wm))
        R.append(Pk.T @ P.rvec(wm, P.yc))
    return np.stack(A), np.stack(C), np.stack(R)


def fit_ol2(P: Prep, dataset: str, omega: float, grid=C2_GRID) -> dict:
    T0 = sparse.main_block(P, omega)
    main_pieces = P.pieces(T0, np.ones(P.n_env), P.yc)
    Pk, active, env_loc = gxl_basis(P, dataset)
    L = len(active)
    pieces = gxl_pieces(P, T0, Pk, active, env_loc) if L else None
    fits = {}
    for c2 in grid:
        if c2 == 0 or L == 0:
            r = reml_solve(main_pieces[0], main_pieces[1], main_pieces[2], P.n_resid)
            fits[c2] = {"a0": r["coef"], "G": np.zeros((Pk.shape[1], max(L, 1))), "m2ll": r["m2ll"],
                        "at_edge": r["at_edge"]}
            continue
        Afull, rfull = sparse.assemble(main_pieces[0], main_pieces[1], *pieces, np.eye(L), c2)
        r = reml_solve(Afull, rfull, main_pieces[2], P.n_resid)
        m0 = T0.shape[1]
        fits[c2] = {"a0": r["coef"][:m0], "G": r["coef"][m0:].reshape(Pk.shape[1], L) * np.sqrt(c2),
                    "m2ll": r["m2ll"], "at_edge": r["at_edge"]}
    c2 = min(grid, key=lambda c: fits[c]["m2ll"])
    return {"T0": T0, "Pk": Pk, "active": active, "fits": fits, "c2": c2, "main_pieces": main_pieces,
            "pieces": pieces}


def gxl_values(ol2: dict, G: np.ndarray, gi: np.ndarray, locs) -> np.ndarray:
    """Marker G x location part for cells of training genotypes gi at locations locs (0 at inactive locations)."""
    if not ol2["active"]:
        return np.zeros(len(gi))
    lidx = {l: i for i, l in enumerate(ol2["active"])}
    li = np.array([lidx.get(l, -1) for l in locs])
    val = np.einsum("ca,ca->c", ol2["Pk"][gi], G[:, np.clip(li, 0, None)].T)
    return np.where(li >= 0, val, 0.0)


# ------------------------------------------------------------------------------------------------ one target year
def oldlines_year(ds, year: int, envs):
    scored, qualifies, ok_envs = old_line_targets(ds, year, envs)
    train = ds.cells[ds.cells["year"] < year]
    assert train["year"].max() < year and set(scored["year"]) <= {year}
    P = Prep(train, ds.markers, scored)
    pos = {g: i for i, g in enumerate(P.fit_ids)}
    assert len(P.test) == len(scored) and set(P.test["genotype"]) <= set(pos), "old lines must be training genotypes"
    gi_s = P.test["genotype"].map(pos).to_numpy()
    ones = np.ones(P.n_env)
    fits0 = {om: fit_ia(P, ones, P.yc, om) for om in W_GRID}
    om = min(W_GRID, key=lambda w: fits0[w]["m2ll"])
    out = P.test[["env", "year", "genotype", "y"]].copy()
    out["ol0"] = fits0[0.0]["u_f"][gi_s]
    out["ol1"] = fits0[om]["u_f"][gi_s]
    meta = {"omega": om, "n_fit": len(P.fit_ids), "n_train_cells": len(P.train), "n_scored": len(out),
            "ok_envs": ok_envs, "qualifies": bool(qualifies), "n_markers": P.n_markers,
            "m2ll_omega": {f"{w:g}": fits0[w]["m2ll"] for w in W_GRID},
            "any_at_edge_omega": bool(any(f["at_edge"] for f in fits0.values()))}
    if cellspec.location(ds.name, str(out["env"].iloc[0]), "location") is None:
        return out, meta                               # no location that repeats across years: OL0/OL1 only
    tr_loc = np.array([cellspec.location(ds.name, e, "location") for e in P.envs], dtype=object)[P.ej]
    sc_loc = np.array([cellspec.location(ds.name, e, "location") for e in out["env"]], dtype=object)
    ol2 = fit_ol2(P, ds.name, om)
    f2 = ol2["fits"][ol2["c2"]]
    u2_f = ol2["T0"] @ f2["a0"]
    out["ol2"] = u2_f[gi_s] + gxl_values(ol2, f2["G"], gi_s, sc_loc)
    add1, vc1 = line_location_correction(P, fits0[om]["u_f"][P.gi], tr_loc, out["genotype"], sc_loc)
    out["ol3"] = out["ol1"].to_numpy() + add1
    fit2_train = u2_f[P.gi] + gxl_values(ol2, f2["G"], P.gi, tr_loc)
    add2, vc2 = line_location_correction(P, fit2_train, tr_loc, out["genotype"], sc_loc)
    out["ol4"] = out["ol2"].to_numpy() + add2
    seen_pairs = set(pair_keys(P.train["genotype"], tr_loc))
    meta.update({"c2": ol2["c2"], "k": ol2["Pk"].shape[1], "n_active_locations": len(ol2["active"]),
                 "m2ll_c2": {f"{c:g}": ol2["fits"][c]["m2ll"] for c in C2_GRID},
                 "any_at_edge_c2": bool(any(f["at_edge"] for f in ol2["fits"].values())),
                 "ol3_em": vc1, "ol4_em": vc2,
                 "share_scored_active_location": float(np.mean([l in set(ol2["active"]) for l in sc_loc])),
                 "share_scored_with_line_location_history": float(
                     np.mean([k in seen_pairs for k in pair_keys(out["genotype"], sc_loc)]))})
    return out, meta
