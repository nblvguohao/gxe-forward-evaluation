"""A1 (docs/prereg_sequel_wave2a_2026-09-28.md §2): one GE-BiFormer run = (scenario, protocol, seed).

own       : fit on all pre-test years; scalers fitted on every hybrid and environment of the scenario, test
            year included (their loader); scheduler and early stopping on the TEST-year Huber loss; the
            reported prediction is the epoch with the lowest test loss (their script's best_test_loss).
own_clean : as own, scalers fitted on the fit set only.
fair      : tune on train with the validation year as monitor -> E* and the learning rate of every epoch;
            refit on all pre-test years for E* epochs replaying those rates; test predicted after epoch E*.

Every protocol also stores the test-year prediction after every epoch (E1 only). No metric is computed
here: scoring is done by scripts/a1_analyze.py after all runs have finished."""
import argparse
import json
import os
import socket
import time

import numpy as np
import pandas as pd
import torch

from dartgxe.baselines.common import git_commit, git_dirty, load_folds
from dartgxe.external.gebiformer import env_features, geno_features, import_gebiformer, predict, tensors, train
from dartgxe.features.fit import load_geno
from dartgxe.paths import RESULTS, processed

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True, choices=["F2024m", "F2022m"])
ap.add_argument("--protocol", required=True, choices=["own", "own_clean", "fair"])
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
dev = "cuda"
t0 = time.time()
method = f"GEBiFormer_{a.protocol}"
cfg, Model, ES = import_gebiformer()
if os.environ.get("A1_SMOKE_EPOCHS"):  # smoke test only, run with a scratch DARTGXE_RESULTS
    cfg = {**cfg, "epochs": int(os.environ["A1_SMOKE_EPOCHS"])}
geno = load_geno()
ec = pd.read_parquet(processed("g2f") / "env_ec.parquet")
fold = next(load_folds(a.scenario))
tr, va, te = (fold.rows(r) for r in ("train", "val", "test"))
pre = fold.rows("train", "val", "refit_extra")
ids = lambda d: sorted(set(d["genotype"]))
envs = lambda d: sorted(set(d["env"]))
rng = np.random.default_rng(a.seed)  # their random fill of missing calls, seeded here
per_epoch = {}


def keep(tst):
    def on_epoch(ep, model):
        per_epoch[ep] = predict(model, tst[0], tst[1])
    return on_epoch


tune_log = None
if a.protocol in ("own", "own_clean"):
    all_h, all_e = ids(fold.cells), envs(fold.cells)
    fit_h, fit_e = (all_h, all_e) if a.protocol == "own" else (ids(pre), envs(pre))
    G = geno_features(geno, all_h, fit_h, rng)
    E = env_features(ec, all_e, fit_e)
    r_fit, *fit = tensors(pre, G, E, dev)
    r_te, *tst = tensors(te, G, E, dev)
    _, log = train(cfg, Model, ES, tuple(fit), tuple(tst), seed=a.seed, on_epoch=keep(tst))
    reported = log["best_epoch"]
else:
    tv = pd.concat([tr, va])
    G1 = geno_features(geno, ids(tv), ids(tr), rng)  # no test-year hybrid or environment enters tuning
    E1 = env_features(ec, envs(tv), envs(tr))
    _, *fit1 = tensors(tr, G1, E1, dev)
    _, *mon = tensors(va, G1, E1, dev)
    _, tune_log = train(cfg, Model, ES, tuple(fit1), tuple(mon), seed=a.seed)
    e_star = max(int(tune_log["best_epoch"]), 1)
    G = geno_features(geno, sorted(set(ids(pre)) | set(ids(te))), ids(pre), rng)
    E = env_features(ec, sorted(set(envs(pre)) | set(envs(te))), envs(pre))
    r_fit, *fit = tensors(pre, G, E, dev)
    r_te, *tst = tensors(te, G, E, dev)
    _, log = train(cfg, Model, ES, tuple(fit), None, seed=a.seed, epochs=e_star,
                   lr_schedule=tune_log["lr"][:e_star], on_epoch=keep(tst))
    reported = e_star

base = {"env": r_te["env"].to_numpy(), "genotype": r_te["genotype"].to_numpy(), "observed": r_te["y"].to_numpy(float)}
p = pd.DataFrame({**base, "prediction": per_epoch[reported].astype(float), "method": method,
                  "scenario": a.scenario, "fold": 0, "seed": a.seed})
pdir = RESULTS / "predictions" / "g2f" / a.scenario / method
pdir.mkdir(parents=True, exist_ok=True)
p.to_parquet(pdir / f"seed{a.seed}.parquet", index=False)
edir = RESULTS / "a1" / "epochs"
edir.mkdir(parents=True, exist_ok=True)
pd.concat([pd.DataFrame({**base, "epoch": ep, "prediction": v.astype(np.float32)}) for ep, v in per_epoch.items()]) \
    .to_parquet(edir / f"{a.scenario}_{a.protocol}_seed{a.seed}.parquet", index=False)
dropped = sorted(set(te["env"]) - set(r_te["env"]))
meta = {"scenario": a.scenario, "protocol": a.protocol, "seed": a.seed, "reported_epoch": int(reported),
        "log": {k: v for k, v in log.items()}, "tune_log": tune_log,
        "cells": {"fit": len(r_fit), "test_scored_input": len(r_te), "test_total": len(te)},
        "test_envs_dropped_no_ec": dropped, "gpu": torch.cuda.get_device_name(0), "host": socket.gethostname(),
        "git_commit": os.environ.get("DARTGXE_COMMIT") or git_commit(), "git_dirty": git_dirty(), "seconds": round(time.time() - t0)}
rdir = RESULTS / "a1" / "runs"
rdir.mkdir(parents=True, exist_ok=True)
(rdir / f"{a.scenario}_{a.protocol}_seed{a.seed}.json").write_text(json.dumps(meta, indent=1, default=float))
print("A1_RUN_DONE", a.scenario, a.protocol, a.seed, "reported_epoch", reported, "epochs_run", log["epochs_run"])
