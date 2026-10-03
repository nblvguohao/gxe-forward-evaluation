"""SP2 headroom check (docs/prereg_headroom_sparse2_2026-09-29.md): on top of M x E GBLUP (M1) in within-year sparse
testing, (C1) decision stacking of M0, M1 and gbm_env with weights fitted on inner-CV out-of-fold predictions,
(C2) an MLP on M1's out-of-fold residuals, (C3) M1 with an environmental-covariate kernel among target environments.
Reuses sparse / cellspec / cellgxe without modifying them."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls
from sklearn.neural_network import MLPRegressor

from dartgxe.forward import cellgxe, sparse
from dartgxe.forward.cellspec import C2_GRID, K_GL, W_GRID, Prep, fit_ia
from dartgxe.forward.panel import features

N_FOLDS = 5
STACK_LAMBDA = 0.05
MLP_SEEDS = [0, 1, 2]


# ------------------------------------------------------------------------------------------------ REML models
def fit_core(train, markers, test, omega_grid=W_GRID, Omega_envs=None):
    """M0, M1 (and C3 when Omega_envs = (envs, Omega) is given) fitted on `train`, predicting every row of `test`
    (test genotypes may be unseen; they are predicted from markers)."""
    P = Prep(train, markers, test)
    ones = np.ones(P.n_env)
    fits0 = {om: fit_ia(P, ones, P.yc, om) for om in omega_grid}
    om = min(omega_grid, key=lambda w: fits0[w]["m2ll"])
    s = np.sqrt((1 - om) * P.lam + om)
    keep = s > 1e-8 * s.max()
    T0 = P.U[:, keep] * s[keep]
    Uinv_basis = P.U[:, keep] / s[keep]                     # U S^-1
    main_pieces = P.pieces(T0, ones, P.yc)
    keep0 = P.lam > 1e-8 * P.lam.max()
    k = int(min(K_GL, len(P.fit_ids) - 1, keep0.sum()))
    order = np.argsort(-P.lam)[:k]
    Pk = P.U[:, order] * np.sqrt(P.lam[order])
    Pk_t = P.Ktf @ (P.U[:, order] / np.sqrt(P.lam[order]))  # Nystrom (equals Pk rows for training genotypes)
    tgt_year = test["year"].iloc[0] if len(test) else None
    tenvs = sorted(set(train.loc[train["year"] == tgt_year, "env"])) if tgt_year is not None else []
    pieces = sparse.gxe_pieces(P, T0, Pk, tenvs)
    I = np.eye(len(tenvs))
    fits1 = {c2: sparse.fit_gxe(P, T0, Pk, tenvs, pieces, main_pieces, I, c2) for c2 in C2_GRID}
    c2 = min(C2_GRID, key=lambda c: fits1[c]["m2ll"])
    tpos = {g: i for i, g in enumerate(P.test_ids)}
    ti = P.test["genotype"].map(tpos).to_numpy()
    jidx = {e: i for i, e in enumerate(tenvs)}
    jj = P.test["env"].map(jidx)
    has = jj.notna().to_numpy()
    jj = jj.fillna(0).astype(int).to_numpy()

    def main_pred(a0):
        w = Uinv_basis @ a0
        u = (1 - om) * (P.Ktf @ w)
        if om > 0:
            u = u + om * np.where(P.seen, w[np.clip(P.seen_idx, 0, None)], 0.0)
        return u[ti]

    def pred(fit, L):
        gx = np.einsum("ca,ab,cb->c", Pk_t[ti], fit["G"], L[jj]) if len(tenvs) else 0.0
        return main_pred(fit["a0"]) + np.where(has, gx, 0.0)

    out = P.test[["env", "year", "genotype", "y"]].copy()
    out["m0"] = pred(fits1[0.0], I)
    out["m1"] = pred(fits1[c2], I)
    meta = {"omega": om, "c2": c2, "k": k, "n_tenvs": len(tenvs)}
    if Omega_envs is not None and c2 > 0 and len(tenvs):
        envs_o, Om = Omega_envs
        pos = {e: i for i, e in enumerate(envs_o)}
        sel = [pos[e] for e in tenvs]
        Om_t = Om[np.ix_(sel, sel)]
        fits3 = {rho: sparse.fit_gxe(P, T0, Pk, tenvs, pieces, main_pieces, sparse.env_factor(Om_t, rho), c2)
                 for rho in sparse.RHO_GRID}
        rho = min(fits3, key=lambda r: fits3[r]["m2ll"])
        out["ecmxe"] = pred(fits3[rho], sparse.env_factor(Om_t, rho))
        meta["ec_rho"] = rho
    elif Omega_envs is not None:
        out["ecmxe"] = out["m1"]
        meta["ec_rho"] = None
    return out.reset_index(drop=True), meta


# ------------------------------------------------------------------------------------------------ helpers
def cv_folds(obs_cells: pd.DataFrame, dataset: str, year: int, f: float, seed: int) -> np.ndarray:
    """Fold id per observed target cell, stratified by environment (reads env only)."""
    rng = np.random.default_rng([int(seed), int(year), int(round(f * 1000)), zlib.crc32(dataset.encode()),
                                 zlib.crc32(b"cv")])
    fold = np.zeros(len(obs_cells), dtype=int)
    for _, idx in obs_cells.groupby("env").indices.items():
        fold[idx] = rng.permutation(len(idx)) % N_FOLDS
    return fold


def ec_kernel(ec, fit_envs, target_envs):
    """Linear kernel of 5 EC PCs among target environments (observed ECs), PCs fitted on training envs."""
    W, _ = cellgxe.ec_pcs(ec, fit_envs, target_envs, "real")
    Z = np.stack([W[e] for e in target_envs])
    Om = Z @ Z.T
    md = np.mean(np.diag(Om))
    return (Om / md) if md > 0 else np.zeros_like(Om), {e: W[e] for e in target_envs}


def env_features(cells, markers, fit_ids, tenvs_all, ecvec=None, n_pc=80):
    ids = sorted(set(cells["genotype"]))
    _, _, Pf, Po = features(markers, fit_ids, ids)
    pos = {g: i for i, g in enumerate(ids)}
    G = Po[[pos[g] for g in cells["genotype"]]][:, :n_pc]
    eidx = {e: i for i, e in enumerate(tenvs_all)}
    H = np.zeros((len(cells), len(tenvs_all)))
    H[np.arange(len(cells)), cells["env"].map(eidx).to_numpy()] = 1.0
    parts = [G, H]
    if ecvec is not None:
        parts.append(np.stack([ecvec[e] for e in cells["env"]]))
    return np.hstack(parts)


def centred(y, env):
    s = pd.Series(y)
    return (s - s.groupby(np.asarray(env)).transform("mean")).to_numpy()


def env_standardise(p, env):
    s = pd.Series(p)
    g = s.groupby(np.asarray(env))
    sd = g.transform("std").replace(0, np.nan)
    return ((s - g.transform("mean")) / sd).fillna(0.0).to_numpy()


def stack_weights(Z: np.ndarray, y: np.ndarray, env) -> np.ndarray:
    """atlas stack_weights: per environment standardise predictions and y; c = mean Z'y/n, S = mean Z'Z/n;
    minimise w'(S + lambda I) w - 2 c'w with w >= 0 (NNLS via Cholesky); all-zero -> equal weights."""
    env = np.asarray(env)
    cs, Ss = [], []
    for e in pd.unique(env):
        m = env == e
        if m.sum() < 3:
            continue
        ye = y[m]
        if ye.std() == 0:
            continue
        Ze = Z[m]
        sd = Ze.std(0)
        sd[sd == 0] = 1.0
        Ze = (Ze - Ze.mean(0)) / sd
        yz = (ye - ye.mean()) / ye.std()
        cs.append(Ze.T @ yz / m.sum())
        Ss.append(Ze.T @ Ze / m.sum())
    c, S = np.mean(cs, 0), np.mean(Ss, 0)
    R = cholesky(S + STACK_LAMBDA * np.eye(len(c)), lower=False)
    w, _ = nnls(R, solve_triangular(R, c, trans="T"))
    return w if w.sum() > 0 else np.ones(len(c)) / len(c)


# ------------------------------------------------------------------------------------------------ one unit
def sp2_year(ds, year: int, envs, f: float, seed: int, ec=None):
    hist = ds.cells[ds.cells["year"] < year]
    tgt = ds.cells[(ds.cells["year"] == year) & ds.cells["env"].isin(list(envs))
                   & ds.cells["genotype"].isin(ds.markers.index)].reset_index(drop=True)
    obs = sparse.sparse_mask(tgt[["env", "genotype"]], f, seed, ds.name, year)
    obs_cells = tgt[obs].reset_index(drop=True)
    test = tgt[~obs].reset_index(drop=True)
    assert hist["year"].max() < year and set(test["year"]) <= {year}
    tenvs_all = sorted(set(obs_cells["env"]))
    train_full = pd.concat([hist, obs_cells], ignore_index=True)
    fit_envs = sorted(set(train_full["env"]))
    Om = ecvec = None
    if ec is not None:
        Om, ecvec = ec_kernel(ec, fit_envs, tenvs_all)
    # full fit
    full, meta = fit_core(train_full, ds.markers, test, Omega_envs=(tenvs_all, Om) if Om is not None else None)
    # inner CV on observed target cells
    fold = cv_folds(obs_cells, ds.name, year, f, seed)
    oof = pd.DataFrame({"m0": np.nan, "m1": np.nan, "gbm": np.nan}, index=obs_cells.index)
    fold_meta = []
    import lightgbm as lgb

    fit_ids_full = sorted(set(train_full["genotype"]) & set(ds.markers.index))
    for k in range(N_FOLDS):
        tr_t, te_t = obs_cells[fold != k], obs_cells[fold == k]
        if te_t.empty:
            continue
        pr, fm = fit_core(pd.concat([hist, tr_t], ignore_index=True), ds.markers, te_t)
        oof.loc[te_t.index, "m0"] = pr["m0"].to_numpy()
        oof.loc[te_t.index, "m1"] = pr["m1"].to_numpy()
        fit_ids_k = sorted(set(pd.concat([hist, tr_t])["genotype"]) & set(ds.markers.index))
        Xtr = env_features(tr_t, ds.markers, fit_ids_k, tenvs_all, ecvec, n_pc=20)
        Xte = env_features(te_t, ds.markers, fit_ids_k, tenvs_all, ecvec, n_pc=20)
        gbm = lgb.LGBMRegressor(num_leaves=31, n_estimators=300, learning_rate=0.05, min_child_samples=20,
                                random_state=0, n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)
        gbm.fit(Xtr, centred(tr_t["y"].to_numpy(float), tr_t["env"]))
        oof.loc[te_t.index, "gbm"] = gbm.predict(Xte)
        fold_meta.append(fm | {"fold": k})
    # gbm_env on all observed cells -> scored cells
    Xtr = env_features(obs_cells, ds.markers, fit_ids_full, tenvs_all, ecvec, n_pc=20)
    Xte = env_features(test, ds.markers, fit_ids_full, tenvs_all, ecvec, n_pc=20)
    gbm = lgb.LGBMRegressor(num_leaves=31, n_estimators=300, learning_rate=0.05, min_child_samples=20,
                            random_state=0, n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)
    gbm.fit(Xtr, centred(obs_cells["y"].to_numpy(float), obs_cells["env"]))
    full["gbm"] = gbm.predict(Xte)
    # C1 stacking
    y_obs = obs_cells["y"].to_numpy(float)
    w = stack_weights(oof[["m0", "m1", "gbm"]].to_numpy(), y_obs, obs_cells["env"])
    Zt = np.column_stack([env_standardise(full[m].to_numpy(), full["env"]) for m in ("m0", "m1", "gbm")])
    full["stk"] = Zt @ w
    # C2 residual MLP on M1's out-of-fold residuals
    r = centred(y_obs, obs_cells["env"]) - centred(oof["m1"].to_numpy(), obs_cells["env"])
    Xtr = env_features(obs_cells, ds.markers, fit_ids_full, tenvs_all, ecvec, n_pc=80)
    Xte = env_features(test, ds.markers, fit_ids_full, tenvs_all, ecvec, n_pc=80)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    nets = []
    for s in MLP_SEEDS:
        net = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, validation_fraction=0.2,
                           n_iter_no_change=20, max_iter=300, random_state=s)
        nets.append(net.fit((Xtr - mu) / sd, r).predict((Xte - mu) / sd))
    full["dlres"] = full["m1"].to_numpy() + np.mean(nets, 0)
    meta.update({"stack_w": dict(zip(["m0", "m1", "gbm"], map(float, w))), "folds": fold_meta,
                 "n_observed": int(obs.sum()), "n_scored": int((~obs).sum()),
                 "oof_cor_m1": float(np.corrcoef(y_obs, oof["m1"])[0, 1]) if len(y_obs) > 2 else None})
    return full, meta
