"""B4 LightGBM G+E (spec 02): 100 genotype PCs + standardised EC, target y, 20 Optuna trials scored by
validation mean Spearman; each trial early-stops on validation l2. Final fits for seeds 0..4 with the
chosen parameters (refit on train ∪ val ∪ refit_extra in forward scenarios, rounds fixed to the tuned
best iteration)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna

from dartgxe.baselines.common import Fold, val_score
from dartgxe.features.fit import EnvScaler, GenoData, GenoPCA, GenoScaler

THREADS = 8


def features(fold: Fold, roles, geno: GenoData, ec, k: int = 100):
    fr = fold.rows(*roles)
    gs = GenoScaler.fit(geno, fr["genotype"].unique())
    hyb = sorted(fold.cells["genotype"].unique())
    Z = gs.transform(geno, hyb)
    hpos = {h: i for i, h in enumerate(hyb)}
    tr_ids = fr["genotype"].unique()
    pca = GenoPCA.fit(Z[[hpos[h] for h in tr_ids]], tr_ids, k)
    U = pca.transform(Z)
    es = EnvScaler.fit(ec, fr["env"].unique())
    envs = sorted(fold.cells["env"].unique())
    E = es.transform(ec, envs)
    epos = {e: i for i, e in enumerate(envs)}

    def X(rows):
        return np.c_[U[[hpos[h] for h in rows["genotype"]]], E[[epos[e] for e in rows["env"]]]].astype(np.float32)
    return fr, X


def fit_b4(fold: Fold, geno: GenoData, ec, seeds=range(5), n_trials: int = 20):
    fr, X = features(fold, ("train",), geno, ec)
    va = fold.rows("val")
    Xtr, ytr, Xva, yva = X(fr), fr["y"].to_numpy(), X(va), va["y"].to_numpy()
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)

    def params(trial):
        return {"objective": "regression", "verbose": -1, "num_threads": THREADS, "seed": 0,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 500, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0), "bagging_freq": 1,
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True)}

    def objective(trial):
        p = params(trial)
        b = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        trial.set_user_attr("best_iter", b.best_iteration)
        return val_score(va, b.predict(Xva, num_iteration=b.best_iteration))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2026))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    p = {**params(best), "seed": 0}
    n_round = max(best.user_attrs["best_iter"], 1)
    if fold.forward:
        fr, X = features(fold, ("train", "val", "refit_extra"), geno, ec)
    te = fold.rows("test")
    Xfit, yfit, Xte = X(fr), fr["y"].to_numpy(), X(te)
    preds = {}
    for s in seeds:
        b = lgb.train({**p, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}, lgb.Dataset(Xfit, yfit), num_boost_round=n_round)
        preds[s] = b.predict(Xte)
    return te, preds, {"best_params": best.params, "best_iter": n_round, "val_spearman": best.value, "trials": n_trials}
