"""Gate 2 (docs/gate2_geformer_2026-09-26.md): GEFormer under its own protocol vs a fair protocol.

own  : fit on all pre-test years; Optuna and the reported epoch are chosen on the TEST year (pooled Pearson),
       i.e. the public train.py practice transposed to forward prediction.
fair : fit on train years; Optuna and the epoch E* are chosen on the VALIDATION year (pooled Pearson);
       refit on all pre-test years for E* epochs; the test year is scored once.

--role tune  : one Optuna worker (several may share a study through a journal file)
--role final : retrain with the study's best parameters for one seed and write test predictions
"""
import argparse
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
import torch

from dartgxe.baselines.common import git_commit, git_dirty, load_folds
from dartgxe.eval.metrics import per_env
from dartgxe.external.geformer import (SEARCH, Data, env_tensors, epoch_train, import_geformer, make_args,
                                       planting_dates, pooled_pearson, predict)
from dartgxe.features.fit import load_geno
from dartgxe.paths import RESULTS, processed

EPOCHS = 100
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True)
ap.add_argument("--protocol", required=True, choices=["own", "fair"])
ap.add_argument("--role", required=True, choices=["tune", "final"])
ap.add_argument("--n_trials", type=int, default=1)
ap.add_argument("--seed", type=int, default=147)
ap.add_argument("--required_trials", type=int, default=8)  # the pre-registered budget  # GEFormer's train.py uses setup_seed(147)
a = ap.parse_args()
dev = "cuda"
OUT = RESULTS / "gate2" / f"{a.scenario}_{a.protocol}"
OUT.mkdir(parents=True, exist_ok=True)

fold = next(load_folds(a.scenario))
geno = load_geno()
envs_all = sorted(fold.cells["env"].unique())
daily = pd.read_parquet(processed("g2f") / "env_daily.parquet")
W, M, how, envs = env_tensors(daily, envs_all, planting_dates(os.environ["G2F_RAW"]))
keep = set(envs)
pre = fold.rows("train", "val", "refit_extra")
pre = pre[pre["env"].isin(keep)]
tr, va, te = (fold.rows(r) for r in ("train", "val", "test"))
tr, va, te = (d[d["env"].isin(keep)].reset_index(drop=True) for d in (tr, va, te))
fit = pre if a.protocol == "own" else tr
sel = te if a.protocol == "own" else va
nan_envs = {e for e, v in how.items() if v == "nan_in_window"}


def check(rows, what):
    bad = sorted(set(rows["env"]) & nan_envs)
    if bad:
        raise RuntimeError(f"{what} uses {len(bad)} environments whose weather window contains NaN, e.g. {bad[:3]}")


check(fit, "fit set")
check(sel, "selection set")
if a.role == "final":
    check(te, "test set")
hyb = sorted(fold.cells["genotype"].unique())
hpos = {h: i for i, h in enumerate(hyb)}
X = geno.X[[geno.index[h] for h in hyb]].astype(np.float32)


def impute(fit_rows):
    ids = sorted(set(fit_rows["genotype"]))
    F = X[[hpos[h] for h in ids]]
    mode = np.stack([(F == k).sum(0) for k in (0, 1, 2)]).argmax(0)
    return np.where(X < 0, mode, X)


epos = {e: i for i, e in enumerate(envs)}
GEF = import_geformer()


def idx(rows):
    return (torch.tensor([hpos[h] for h in rows["genotype"]], device=dev),
            torch.tensor([epos[e] for e in rows["env"]], device=dev),
            torch.tensor(rows["y"].to_numpy(np.float32), device=dev))


def run(hp, fit_rows, sel_rows, epochs, seed, keep_best_preds=None):
    """Train; after every epoch score sel_rows by pooled Pearson. Returns curve, best epoch,
    and (optionally) predictions on keep_best_preds rows at the best epoch."""
    torch.manual_seed(seed); np.random.seed(seed)
    data = Data(impute(fit_rows), W, M, dev)
    gi, ei, y = idx(fit_rows)
    net = GEF(make_args(W.shape[2], hp), X.shape[1], W.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), hp["lr"])
    rng = np.random.default_rng(seed)
    sgi, sei, sy = idx(sel_rows) if sel_rows is not None else (None, None, None)
    curve, best, best_ep, best_pred = [], -np.inf, 0, None
    for ep in range(1, epochs + 1):
        epoch_train(net, data, gi, ei, y, hp["batch"], opt, rng)
        if sel_rows is not None:
            p = predict(net, data, sgi, sei)
            r = pooled_pearson(p, sel_rows["y"].to_numpy())
            curve.append(r)
            if np.isfinite(r) and r > best:
                best, best_ep = r, ep
                if keep_best_preds is not None:
                    best_pred = p
    return net, data, curve, best, best_ep, best_pred


storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(OUT / "study.journal")))
study = optuna.create_study(study_name=f"{a.scenario}_{a.protocol}", storage=storage, direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=2026 + os.getpid() % 1000), load_if_exists=True)

if a.role == "tune":
    def objective(trial):
        hp = {"batch": trial.suggest_int("batch", *SEARCH["batch"]), "lr": trial.suggest_float("lr", *SEARCH["lr"], log=True),
              "dropout": trial.suggest_float("dropout", *SEARCH["dropout"]), "depth": trial.suggest_int("depth", *SEARCH["depth"]),
              "neurons1": trial.suggest_int("neurons1", *SEARCH["neurons1"]), "neurons2": trial.suggest_int("neurons2", *SEARCH["neurons2"])}
        t0 = time.time()
        _, _, curve, best, best_ep, _ = run(hp, fit, sel, EPOCHS, a.seed)
        trial.set_user_attr("best_epoch", best_ep)
        trial.set_user_attr("curve", [float(c) for c in curve])
        trial.set_user_attr("seconds", round(time.time() - t0))
        if not np.isfinite(best):
            raise RuntimeError("selection metric is not finite in every epoch; refusing to record a sentinel")
        return best
    study.optimize(objective, n_trials=a.n_trials)
else:
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    if len(done) < a.required_trials:
        raise RuntimeError(f"study has {len(done)} complete trials, {a.required_trials} required before the final refit")
    bt = study.best_trial
    hp = bt.params
    t0 = time.time()
    if a.protocol == "own":
        # train.py final loop: retrain with the best parameters, report the max-Pearson epoch on the evaluated fold
        net, data, curve, best, best_ep, pred = run(hp, pre, te, EPOCHS, a.seed, keep_best_preds=True)
    else:
        E = max(int(bt.user_attrs["best_epoch"]), 1)
        net, data, _, _, _, _ = run(hp, pre, None, E, a.seed)
        gi, ei, _ = idx(te)
        pred, best_ep, curve = predict(net, data, gi, ei), E, []
    p = pd.DataFrame({"env": te["env"], "genotype": te["genotype"], "observed": te["y"], "prediction": pred,
                      "method": f"GEFormer_{a.protocol}", "scenario": a.scenario, "fold": 0, "seed": a.seed})
    pdir = RESULTS / "predictions" / "g2f" / a.scenario / f"GEFormer_{a.protocol}"
    pdir.mkdir(parents=True, exist_ok=True)
    p.to_parquet(pdir / f"seed{a.seed}.parquet", index=False)
    pe = per_env(p).query("metric == 'spearman'")
    meta = {"scenario": a.scenario, "protocol": a.protocol, "seed": a.seed, "best_params": hp, "best_trial_value": bt.value,
            "n_trials": len([t for t in study.trials if t.state.name == "COMPLETE"]), "reported_epoch": best_ep,
            "test_pooled_pearson": pooled_pearson(pred, te["y"].to_numpy()), "test_within_env_spearman": float(pe["value"].mean()),
            "test_curve_pooled_pearson": [float(c) for c in curve], "envs_dropped_no_weather": [e for e, v in how.items() if v == "no_weather"],
            "git_commit": git_commit(), "git_dirty": git_dirty(), "gpu": torch.cuda.get_device_name(0), "seconds": round(time.time() - t0)}
    (OUT / f"final_seed{a.seed}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("scenario", "protocol", "seed", "reported_epoch", "test_pooled_pearson", "test_within_env_spearman")}))
