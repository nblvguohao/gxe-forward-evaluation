"""Two-stage genotype-only method library and the forward / CV histories (prereg wave 2b §1-3).

Stage 1: each training cell minus the mean of the training cells in its environment, averaged per genotype.
Stage 2: the ten methods below on those genotype means. Standardisation, imputation and PCA are fitted on the
training genotypes only (leakage rule: fitted on training data only); `fit_predict` returns the ids it fitted on so tests can check it."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge as SkRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from dartgxe.baselines.linear import Ridge

METHODS = ["reml_x0.1", "reml", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp"]
NPC = 80
N_FOLDS = 5


def stage1(cells: pd.DataFrame) -> pd.Series:
    yc = cells["y"] - cells.groupby("env")["y"].transform("mean")
    return yc.groupby(cells["genotype"].to_numpy()).mean()


def features(markers: pd.DataFrame, fit_ids, other_ids):
    """Standardised markers and PCs; means, SDs, imputation and PCA from fit_ids only. A kernel embedding
    (markers.attrs["no_scale"], data.load_gem_ia) is centred but not scaled, so that Z Z' stays the GRM."""
    F = markers.loc[list(fit_ids)].to_numpy(np.float64)
    O = markers.loc[list(other_ids)].to_numpy(np.float64)
    mu = np.nanmean(F, axis=0)
    F = np.where(np.isnan(F), mu, F)
    O = np.where(np.isnan(O), mu, O)
    sd = F.std(axis=0)
    ok = np.isfinite(mu) & (sd > 0)
    if markers.attrs.get("no_scale"):
        sd = np.ones_like(sd)
    Zf, Zo = (F[:, ok] - mu[ok]) / sd[ok], (O[:, ok] - mu[ok]) / sd[ok]
    n_pc = max(1, min(NPC, Zf.shape[0] - 1, Zf.shape[1]))
    pca = PCA(n_components=n_pc, random_state=0).fit(Zf)
    return Zf, Zo, pca.transform(Zf), pca.transform(Zo)


def fit_predict(cells: pd.DataFrame, markers: pd.DataFrame, test_ids, seed: int = 0):
    """Fit the ten methods on `cells` (training only); predict `test_ids`. Returns (DataFrame test_id x method,
    meta) with meta["fit_ids"] = the genotypes every transform and model was fitted on."""
    g = stage1(cells)
    fit_ids = sorted(set(g.index) & set(markers.index))
    test_ids = list(test_ids)
    y = g.loc[fit_ids].to_numpy(np.float64)
    Zf, Zt, Pf, Pt = features(markers, fit_ids, test_ids)
    out = {}
    r = Ridge(Zf, np.zeros(len(fit_ids)), y)
    rm = r.reml()
    for tag, mult in (("reml_x0.1", 0.1), ("reml", 1.0), ("reml_x10", 10.0), ("reml_x100", 100.0)):
        out[tag] = Zt @ r.coef(rm["lambda_rel"] * mult)
    k20 = min(20, Pf.shape[1])
    out["ridge_pc20"] = SkRidge(alpha=1.0).fit(Pf[:, :k20], y).predict(Pt[:, :k20])
    n = len(fit_ids)
    models = {"rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=8, random_state=seed),
              "gbm": HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06, random_state=seed + 1),
              "knn10": KNeighborsRegressor(n_neighbors=min(10, n)),
              "knn30": KNeighborsRegressor(n_neighbors=min(30, n)),
              "mlp": MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=500, random_state=seed + 2,
                                  early_stopping=n >= 20)}
    for name, md in models.items():
        out[name] = md.fit(Pf, y).predict(Pt)
    meta = {"fit_ids": fit_ids, "n_fit": n, "n_pc": int(Pf.shape[1]), "n_markers_used": int(Zf.shape[1]),
            "reml_lambda_rel": rm["lambda_rel"], "reml_h2": rm["h2"], "reml_at_grid_edge": rm["at_grid_edge"]}
    return pd.DataFrame(out, index=pd.Index(test_ids, name="genotype"))[METHODS], meta


def long(cells: pd.DataFrame, pred: pd.DataFrame, **tags) -> pd.DataFrame:
    """One row per (cell, method): the genotype's prediction for every cell of that genotype."""
    p = pred.reset_index().melt(id_vars="genotype", var_name="method", value_name="pred")
    d = cells[["env", "year", "genotype", "y"]].merge(p, on="genotype", how="inner")
    for k, v in tags.items():
        d[k] = v
    return d


def forward_year(ds, year: int, seed: int = 0):
    """Predict the scorable environments of `year` from all years before it."""
    train = ds.cells[ds.cells["year"] < year]
    envs = ds.scorable_envs([year])
    test = ds.cells[(ds.cells["year"] == year) & ds.cells["env"].isin(envs)]
    pred, meta = fit_predict(train, ds.markers, sorted(set(test["genotype"])), seed)
    assert not (set(train["year"]) & {year}) and train["year"].max() < year
    return long(test, pred, dataset=ds.name, target=year, kind="forward"), meta


def cv_history(ds, target: int, seed: int = 0):
    """5-fold leave-genotypes-out CV within the years before `target`; out-of-fold predictions for the cells of
    the scorable environments of those years. Each fold's stage 1 and transforms use the other folds only."""
    train = ds.cells[ds.cells["year"] < target]
    geno = np.array(sorted(set(train["genotype"])))
    rng = np.random.default_rng([target, sum(map(ord, ds.name))])  # fixed per (dataset, target year)
    fold = pd.Series(rng.permutation(np.arange(len(geno)) % N_FOLDS), index=geno)
    envs = ds.scorable_envs(sorted(set(train["year"])))
    parts, metas = [], []
    for f in range(N_FOLDS):
        held = set(fold.index[fold == f])
        fit_cells = train[~train["genotype"].isin(held)]
        score_cells = train[train["genotype"].isin(held) & train["env"].isin(envs)]
        if score_cells.empty:
            continue
        pred, meta = fit_predict(fit_cells, ds.markers, sorted(set(score_cells["genotype"])), seed)
        assert not (set(meta["fit_ids"]) & held)
        meta["fold"] = f
        metas.append({k: v for k, v in meta.items() if k != "fit_ids"})
        parts.append(long(score_cells, pred, dataset=ds.name, target=target, kind="cv", fold=f))
    return pd.concat(parts, ignore_index=True), metas
