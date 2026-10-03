"""SP headroom check: within-target-year sparse testing (docs/prereg_headroom_sparse_2026-09-29.md).

Target year Y: every genotyped cell of Y's scored environments is either observed (enters training with all earlier
years) or scored. Models, all cell-level with environment fixed effects and solved at the genotype level:

M0  main-effect GBLUP, kernel (1 - w) K_A + w I, w by profile REML (as I-A's ia_g).
M1  M0's main effect (w fixed) + target-environment-specific marker effects, independent across environments
    (M x E GBLUP restricted to the target year), c2 by profile REML.
M2  G x E covariance K_A^(k) (x) E_rho, E_rho = (1 - rho) I + rho R_hat, R_hat = correlation of M1's
    environment-specific genetic values over the training genotypes; c2 fixed at M1's, rho by profile REML.

Imports the frozen cellspec (blob 4ca8090) without changing it."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import torch

from dartgxe.forward import cellspec
from dartgxe.forward.cellspec import C2_GRID, K_GL, W_GRID, Prep, fit_ia, reml_solve

RHO_GRID = [0.25, 0.5, 0.75, 1.0]
FRACTIONS = [0.25, 0.5]
SEEDS = [0, 1]


# ------------------------------------------------------------------------------------------------ masking
def sparse_mask(cells: pd.DataFrame, f: float, seed: int, dataset: str, year: int) -> np.ndarray:
    """Boolean 'observed' per row of `cells` (one row per env x genotype of the target year). Reads env and genotype
    only. Quota per genotype: floor(f * n_g + 0.5) clipped to [1, n_g - 1] (n_g = 1: observed, never scored);
    environments are filled least-observed-share first, ties broken at random."""
    rng = np.random.default_rng([int(seed), int(year), int(round(f * 1000)), zlib.crc32(dataset.encode())])
    env = cells["env"].to_numpy()
    envs, ei = np.unique(env, return_inverse=True)
    total = np.bincount(ei, minlength=len(envs)).astype(float)
    filled = np.zeros(len(envs))
    obs = np.zeros(len(cells), dtype=bool)
    groups = pd.Series(np.arange(len(cells))).groupby(cells["genotype"].to_numpy()).apply(np.array)
    order = rng.permutation(np.array(groups.index, dtype=object))
    for g in order:
        rows = groups[g]
        n = len(rows)
        if n == 1:
            obs[rows] = True
            continue
        q = int(min(max(np.floor(f * n + 0.5), 1), n - 1))
        share = filled[ei[rows]] / total[ei[rows]]
        pick = rows[np.lexsort((rng.random(n), share))[:q]]
        obs[pick] = True
        np.add.at(filled, ei[pick], 1.0)
    return obs


# ------------------------------------------------------------------------------------------------ models
def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64), device=cellspec.DEV)


def main_block(P: Prep, omega: float):
    s = np.sqrt((1 - omega) * P.lam + omega)
    keep = s > 1e-8 * s.max()
    return P.U[:, keep] * s[keep]


def gxe_pieces(P: Prep, T0: np.ndarray, Pk: np.ndarray, envs: list[str]):
    """Per target environment j (training cells only): A_j = X_j'X_j, C_j = T0[g]'X_j, r_j = X_j' y_c with X_j the
    environment-centred rows of Pk. Equal to Pk' M_j Pk, T0' M_j Pk, Pk' D_j' y_c."""
    code = {e: i for i, e in enumerate(P.envs)}
    A, C, R = [], [], []
    for e in envs:
        rows = np.where(P.ej == code[e])[0]
        g = P.gi[rows]
        X = Pk[g] - Pk[g].mean(0)
        A.append(X.T @ X)
        C.append(T0[g].T @ X)
        R.append(X.T @ P.yc[rows])
    return np.stack(A), np.stack(C), np.stack(R)


def assemble(A00, r0, A, C, R, L, c2):
    """Normal equations for [main | c * (Pk (x) L_j)] with a-major ordering of the G x E coefficients."""
    m0, k, m = A00.shape[0], A.shape[1], L.shape[1]
    At, Ct, Rt, Lt = _t(A), _t(C), _t(R), _t(L)
    c = float(np.sqrt(c2))
    Agg = torch.einsum("jac,jb,jd->abcd", At, Lt, Lt).reshape(k * m, k * m) * c2
    Amg = torch.einsum("jta,jb->tab", Ct, Lt).reshape(m0, k * m) * c
    rg = torch.einsum("ja,jb->ab", Rt, Lt).reshape(k * m) * c
    top = torch.cat([_t(A00), Amg], 1)
    bot = torch.cat([Amg.T, Agg], 1)
    return torch.cat([top, bot], 0).cpu().numpy(), np.concatenate([r0, rg.cpu().numpy()])


def fit_gxe(P, T0, Pk, envs, pieces, main_pieces, L, c2):
    A00, r0, yy = main_pieces
    if c2 == 0:
        r = reml_solve(A00, r0, yy, P.n_resid)
        return {"a0": r["coef"], "G": np.zeros((Pk.shape[1], L.shape[1])), "m2ll": r["m2ll"], "delta": r["delta"],
                "at_edge": r["at_edge"]}
    A, C, R = pieces
    Afull, rfull = assemble(A00, r0, A, C, R, L, c2)
    r = reml_solve(Afull, rfull, yy, P.n_resid)
    m0 = A00.shape[0]
    G = r["coef"][m0:].reshape(Pk.shape[1], L.shape[1]) * np.sqrt(c2)   # effect of Pk (x) L_j
    return {"a0": r["coef"][:m0], "G": G, "m2ll": r["m2ll"], "delta": r["delta"], "at_edge": r["at_edge"]}


def env_factor(Rhat: np.ndarray, rho: float) -> np.ndarray:
    E = (1 - rho) * np.eye(len(Rhat)) + rho * Rhat
    d, V = np.linalg.eigh((E + E.T) / 2)
    return V * np.sqrt(np.clip(d, 0, None))


def corr_or_identity(M: np.ndarray) -> np.ndarray:
    sd = M.std(0)
    ok = sd > 1e-12
    R = np.eye(M.shape[1])
    if ok.sum() >= 2:
        R[np.ix_(ok, ok)] = np.corrcoef(M[:, ok].T)
    return R


def sparse_year(ds, year: int, envs, f: float, seed: int):
    """M0, M1, M2 predictions for the scored cells of `year` under mask (f, seed)."""
    hist = ds.cells[ds.cells["year"] < year]
    tgt = ds.cells[(ds.cells["year"] == year) & ds.cells["env"].isin(list(envs))
                   & ds.cells["genotype"].isin(ds.markers.index)].reset_index(drop=True)
    obs = sparse_mask(tgt[["env", "genotype"]], f, seed, ds.name, year)
    train = pd.concat([hist, tgt[obs]], ignore_index=True)
    test = tgt[~obs].reset_index(drop=True)
    assert hist["year"].max() < year and set(test["year"]) <= {year}
    P = Prep(train, ds.markers, test)
    pos = {g: i for i, g in enumerate(P.fit_ids)}
    assert set(P.test["genotype"]) <= set(pos), "every scored genotype must have an observed cell"
    ones = np.ones(P.n_env)
    # M0
    fits0 = {om: fit_ia(P, ones, P.yc, om) for om in W_GRID}
    om = min(W_GRID, key=lambda w: fits0[w]["m2ll"])
    T0 = main_block(P, om)
    main_pieces = P.pieces(T0, ones, P.yc)
    # G x E basis
    keep0 = P.lam > 1e-8 * P.lam.max()
    k = int(min(K_GL, len(P.fit_ids) - 1, keep0.sum()))
    order = np.argsort(-P.lam)[:k]
    Pk = P.U[:, order] * np.sqrt(P.lam[order])
    tenvs = sorted(set(tgt.loc[obs, "env"]))
    pieces = gxe_pieces(P, T0, Pk, tenvs)
    I = np.eye(len(tenvs))
    fits1 = {c2: fit_gxe(P, T0, Pk, tenvs, pieces, main_pieces, I, c2) for c2 in C2_GRID}
    c2 = min(C2_GRID, key=lambda c: fits1[c]["m2ll"])
    f1 = fits1[c2]
    Rhat = corr_or_identity(Pk @ f1["G"]) if c2 > 0 else np.eye(len(tenvs))
    fits2 = {}
    if c2 > 0:   # prereg §2: rho in {0.25, 0.5, 0.75, 1.0} (rho = 0 is M1 and is not in M2's grid)
        for rho in RHO_GRID:
            fits2[rho] = fit_gxe(P, T0, Pk, tenvs, pieces, main_pieces, env_factor(Rhat, rho), c2)
        rho = min(fits2, key=lambda r: fits2[r]["m2ll"])
    else:        # no G x E term selected: M2 = M1 = M0
        fits2[0.0] = f1
        rho = 0.0
    L2 = env_factor(Rhat, rho) if rho > 0 else I
    # predictions for scored cells (all scored genotypes are training genotypes)
    gi = P.test["genotype"].map(pos).to_numpy()
    jidx = {e: i for i, e in enumerate(tenvs)}
    ji = P.test["env"].map(jidx).to_numpy()
    has = ~pd.isna(ji)
    jj = np.where(has, ji, 0).astype(int)

    def pred(fit, L):
        main = (T0 @ fit["a0"])[gi]
        gx = np.einsum("ca,ab,cb->c", Pk[gi], fit["G"], L[jj])
        return main + np.where(has, gx, 0.0)

    out = P.test[["env", "year", "genotype", "y"]].copy()
    out["m0"] = pred({"a0": fits1[0.0]["a0"], "G": np.zeros((k, len(tenvs)))}, I)
    out["m1"] = pred(f1, I)
    out["m2"] = pred(fits2[rho], L2)
    meta = {"omega": om, "c2": c2, "rho": rho, "k": k, "n_target_envs": len(tenvs), "n_observed": int(obs.sum()),
            "n_scored": int((~obs).sum()), "n_fit": len(P.fit_ids), "n_train_cells": len(P.train),
            "m2ll_c2": {f"{c:g}": fits1[c]["m2ll"] for c in C2_GRID},
            "m2ll_rho": {f"{r:g}": fits2[r]["m2ll"] for r in fits2},
            "any_at_edge": bool(any(x["at_edge"] for x in list(fits1.values()) + list(fits2.values()))),
            "envs_without_observed": int((~has).sum())}
    return out, meta
