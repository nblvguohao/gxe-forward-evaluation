"""Cell-level and G x E learners for wave 2c (docs/prereg_sequel_wave2c_2026-09-28.md §1-2).

cell_reml : marker ridge on every training cell, environment fixed effects, REML lambda (as pre-study 2).
rn_ridge  : reaction-norm ridge on [80 genotype PCs, 20 genotype PCs x 5 EC PCs], environment fixed effects, REML.
gxe_gbm   : LightGBM on [20 genotype PCs, 5 EC PCs], target centred within environment.

Genotype standardisation/PCA are fitted on the training genotypes (panel.features); EC standardisation/PCA on the
training environments. A training environment uses its observed ECs. A target environment uses, by `mode`,
its observed ECs ('real') or the mean ECs of the same location over the training environments ('hist'; the
mean of all training environments when the location is new). Environments without ECs get the training mean
(zero after standardisation), i.e. no G x E information."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dartgxe.baselines.linear import Ridge
from dartgxe.forward.data import location
from dartgxe.forward.panel import N_FOLDS, features, long

CELL = ["cell_reml"]
GXE = ["rn_ridge", "gxe_gbm"]
N_GPC_MAIN, N_GPC_INT, N_ECPC = 80, 20, 5


def ec_pcs(ec: pd.DataFrame, fit_envs, target_envs, mode: str):
    """Returns (dict env -> EC PC vector for fit_envs and target_envs, meta). Fitted on fit_envs only."""
    fit_obs = [e for e in fit_envs if e in ec.index]
    E = ec.loc[fit_obs]
    mu, sd = E.mean(), E.std(ddof=0)
    keep = mu.notna() & (sd > 0)
    Zfit = ((E.loc[:, keep] - mu[keep]) / sd[keep]).fillna(0.0)
    q = max(1, min(N_ECPC, Zfit.shape[0] - 1, Zfit.shape[1]))
    pca = PCA(n_components=q, random_state=0).fit(Zfit.to_numpy())

    def z(rows: pd.DataFrame) -> np.ndarray:
        return pca.transform(((rows.loc[:, keep] - mu[keep]) / sd[keep]).fillna(0.0).to_numpy())

    zero = np.zeros(q)
    out = {}
    for e in fit_envs:
        out[e] = z(ec.loc[[e]])[0] if e in ec.index else zero
    loc_fit = pd.Series({e: location(e) for e in fit_obs})
    for e in target_envs:
        if e in out:
            continue
        if mode == "real":
            out[e] = z(ec.loc[[e]])[0] if e in ec.index else zero
        else:  # historical mean of the same location over the training environments
            same = loc_fit[loc_fit == location(e)].index.tolist()
            rows = ec.loc[same] if same else ec.loc[fit_obs]
            out[e] = z(rows.mean().to_frame().T)[0]
    return out, {"n_fit_envs_with_ec": len(fit_obs), "n_ec_cols": int(keep.sum()), "n_ec_pc": int(q),
                 "fit_envs": list(fit_obs)}


def fit_predict_cell(train: pd.DataFrame, markers: pd.DataFrame, test: pd.DataFrame, ec=None, mode="hist",
                     seed: int = 0):
    """Predict every cell of `test` (env, genotype) from `train` cells. Returns (DataFrame with columns env,
    genotype and one column per method, meta). G x E learners only when `ec` is given."""
    train = train[train["genotype"].isin(markers.index)]
    fit_ids = sorted(set(train["genotype"]))
    test_ids = sorted(set(test["genotype"]))
    Zf, Zt, Pf, Pt = features(markers, fit_ids, test_ids)
    fpos, tpos = {h: i for i, h in enumerate(fit_ids)}, {h: i for i, h in enumerate(test_ids)}
    fi = np.array([fpos[h] for h in train["genotype"]])
    ti = np.array([tpos[h] for h in test["genotype"]])
    y, env = train["y"].to_numpy(np.float64), train["env"].to_numpy()
    out = test[["env", "genotype"]].reset_index(drop=True).copy()
    r = Ridge(Zf[fi], env, y)
    rm = r.reml()
    out["cell_reml"] = Zt[ti] @ r.coef(rm["lambda_rel"])
    del r
    meta = {"fit_ids": fit_ids, "n_fit": len(fit_ids), "n_train_cells": len(train), "cell_reml_lambda_rel": rm["lambda_rel"],
            "cell_reml_at_grid_edge": rm["at_grid_edge"]}
    if ec is not None:
        W, em = ec_pcs(ec, sorted(set(train["env"])), sorted(set(test["env"])), mode)
        Wf = np.stack([W[e] for e in train["env"]])
        Wt = np.stack([W[e] for e in test["env"]])
        k_main, k_int = min(N_GPC_MAIN, Pf.shape[1]), min(N_GPC_INT, Pf.shape[1])

        def design(P, idx, Wx):
            G = P[idx]
            inter = (G[:, :k_int, None] * Wx[:, None, :]).reshape(len(idx), -1)
            return np.c_[G[:, :k_main], inter]

        Df, Dt = design(Pf, fi, Wf), design(Pt, ti, Wt)
        m, s = Df.mean(0), Df.std(0)
        s[s < 1e-12] = 1.0
        rr = Ridge((Df - m) / s, env, y)
        rrm = rr.reml()
        out["rn_ridge"] = ((Dt - m) / s) @ rr.coef(rrm["lambda_rel"])
        del rr
        import lightgbm as lgb
        yc = y - pd.Series(y).groupby(env).transform("mean").to_numpy()
        Xf, Xt = np.c_[Pf[fi][:, :k_int], Wf], np.c_[Pt[ti][:, :k_int], Wt]
        gbm = lgb.LGBMRegressor(num_leaves=31, n_estimators=300, learning_rate=0.05, min_child_samples=50,
                                random_state=seed, n_jobs=8, verbose=-1)
        out["gxe_gbm"] = gbm.fit(Xf, yc).predict(Xt)
        meta.update({"rn_ridge_lambda_rel": rrm["lambda_rel"], "ec": {k: v for k, v in em.items() if k != "fit_envs"},
                     "ec_fit_envs": em["fit_envs"]})
    return out, meta


def to_long(out: pd.DataFrame, cells: pd.DataFrame, methods, suffix="", **tags) -> pd.DataFrame:
    d = cells[["env", "year", "genotype", "y"]].reset_index(drop=True)
    parts = []
    for mth in methods:
        p = d.copy()
        p["method"] = mth + suffix
        p["pred"] = out[mth].to_numpy()
        parts.append(p)
    res = pd.concat(parts, ignore_index=True)
    for k, v in tags.items():
        res[k] = v
    return res


def forward_year_cell(ds, year: int, ec=None, seed: int = 0):
    """cell_reml, plus rn_ridge / gxe_gbm with historical-mean ECs and (suffix _real) observed ECs."""
    train = ds.cells[ds.cells["year"] < year]
    envs = ds.scorable_envs([year])
    test = ds.cells[(ds.cells["year"] == year) & ds.cells["env"].isin(envs)].reset_index(drop=True)
    parts, metas = [], {}
    out, meta = fit_predict_cell(train, ds.markers, test, ec, "hist", seed)
    assert train["year"].max() < year
    parts.append(to_long(out, test, CELL + (GXE if ec is not None else []), dataset=ds.name, target=year, kind="forward"))
    metas["hist"] = {k: v for k, v in meta.items() if k not in ("fit_ids",)}
    if ec is not None:
        assert set(meta["ec_fit_envs"]) <= set(train["env"])
        out_r, meta_r = fit_predict_cell(train, ds.markers, test, ec, "real", seed)
        parts.append(to_long(out_r, test, GXE, suffix="_real", dataset=ds.name, target=year, kind="forward"))
        metas["real"] = {k: v for k, v in meta_r.items() if k not in ("fit_ids", "ec_fit_envs")}
    return pd.concat(parts, ignore_index=True), metas


def cv_folds(ds, target: int) -> pd.Series:
    """Identical to the fold assignment inside panel.cv_history (tests check it)."""
    train = ds.cells[ds.cells["year"] < target]
    geno = np.array(sorted(set(train["genotype"])))
    rng = np.random.default_rng([target, sum(map(ord, ds.name))])
    return pd.Series(rng.permutation(np.arange(len(geno)) % N_FOLDS), index=geno)


def cv_history_cell(ds, target: int, ec=None, seed: int = 0):
    """5-fold leave-genotypes-out CV within the years before `target` (training environments are observed,
    so G x E learners use their observed ECs)."""
    train = ds.cells[ds.cells["year"] < target]
    fold = cv_folds(ds, target)
    envs = ds.scorable_envs(sorted(set(train["year"])))
    parts, metas = [], []
    for f in range(N_FOLDS):
        held = set(fold.index[fold == f])
        fit_cells = train[~train["genotype"].isin(held)]
        score = train[train["genotype"].isin(held) & train["env"].isin(envs)].reset_index(drop=True)
        if score.empty:
            continue
        out, meta = fit_predict_cell(fit_cells, ds.markers, score, ec, "real", seed)
        assert not (set(meta["fit_ids"]) & held)
        metas.append({k: v for k, v in meta.items() if k not in ("fit_ids", "ec_fit_envs")} | {"fold": f})
        parts.append(to_long(out, score, CELL + (GXE if ec is not None else []), dataset=ds.name, target=target,
                             kind="cv", fold=f))
    return pd.concat(parts, ignore_index=True), metas
