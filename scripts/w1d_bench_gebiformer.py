"""W1d (docs/prereg_sequel_wave1_2026-09-28.md §5): can GE-BiFormer run on the forward splits, and at what
GPU cost? Training-year rows only: one short training on `train`, monitored on the validation year (its
metrics may be looked at, §5). No test-year hybrid or environment enters any transform, and the test year
is never scored. Writes results/w1d/gebiformer_bench.json."""
import json
import os
import socket
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from dartgxe.baselines.common import load_folds
from dartgxe.eval.metrics import MIN_N
from dartgxe.external.gebiformer import env_features, geno_features, import_gebiformer, predict, tensors, train
from dartgxe.features.fit import load_geno
from dartgxe.paths import RESULTS, processed

SHORT_EPOCHS = int(os.environ.get("W1D_EPOCHS", "30"))
SEEDS = 3
RUNS_PER_SEED = {"own": 1, "fair": 2}  # own: one run on all pre-test years; fair: tune on train->val, then refit
EPOCH_CAP = 100  # their config.py
TIMEBOX_GPU_H = 72

OUT = RESULTS / "w1d"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
t_start = time.time()
cfg, Model, ES = import_gebiformer()
geno = load_geno()
ec = pd.read_parquet(processed("g2f") / "env_ec.parquet")
res = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "their_config": cfg, "short_epochs": SHORT_EPOCHS, "scenarios": {}}
for sc in ("F2024m", "F2022m"):
    fold = next(load_folds(sc))
    tr, va = fold.rows("train"), fold.rows("val")
    final = fold.rows("train", "val", "refit_extra")
    rng = np.random.default_rng(0)
    G = geno_features(geno, sorted(set(final["genotype"])), sorted(set(tr["genotype"])), rng)
    E = env_features(ec, sorted(set(final["env"])), sorted(set(tr["env"])))
    r_tr, *fit = tensors(tr, G, E, dev)
    r_va, *mon = tensors(va, G, E, dev)
    r_fi, *_ = tensors(final, G, E, dev)
    torch.cuda.reset_peak_memory_stats()
    model, log = train(cfg, Model, ES, tuple(fit), tuple(mon), seed=147, max_epochs=SHORT_EPOCHS)
    peak = torch.cuda.max_memory_allocated() / 2**20
    d = pd.DataFrame({"env": r_va["env"], "y": r_va["y"], "p": predict(model, mon[0], mon[1])})
    rho = [spearmanr(g["p"], g["y"])[0] for _, g in d.groupby("env") if len(g) >= MIN_N]
    s_ep = float(np.median(log["epoch_seconds"][1:])) if len(log["epoch_seconds"]) > 1 else float(log["epoch_seconds"][0])
    s_final_ep = s_ep * len(r_fi) / len(r_tr)  # scaled by cells; includes the monitor pass, so conservative
    gpu_h = SEEDS * sum(RUNS_PER_SEED.values()) * EPOCH_CAP * s_final_ep / 3600
    res["scenarios"][sc] = {
        "years": {role: sorted(int(y) for y in fold.rows(role)["year"].unique()) for role in ("train", "val", "refit_extra")},
        "cells": {"train": len(r_tr), "val": len(r_va), "final_fit": len(r_fi)},
        "cells_dropped_no_ec": {"train": len(tr) - len(r_tr), "val": len(va) - len(r_va), "final_fit": len(final) - len(r_fi)},
        "n_snp_features": int(fit[0].shape[1]), "n_env_features": int(fit[1].shape[1]),
        "params": int(sum(p.numel() for p in model.parameters())),
        "epochs_run": log["epochs_run"], "stopped_early": log["stopped_early"], "best_epoch_on_val": log["best_epoch"],
        "val_loss_first_last": [log["monitor_loss"][0], log["monitor_loss"][-1]],
        "median_s_per_epoch_train": round(s_ep, 2), "est_s_per_epoch_final_fit": round(s_final_ep, 2),
        "peak_mem_mb": round(peak),
        "val_within_env_spearman_mean": round(float(np.nanmean(rho)), 4), "val_envs_scored": len(rho),
        "est_gpu_h_budget": round(gpu_h, 1),
    }
    print(sc, json.dumps(res["scenarios"][sc]), flush=True)
res["est_gpu_h_total"] = round(sum(v["est_gpu_h_budget"] for v in res["scenarios"].values()), 1)
res["budget"] = {"seeds": SEEDS, "runs_per_seed": RUNS_PER_SEED, "epoch_cap": EPOCH_CAP, "timebox_gpu_h": TIMEBOX_GPU_H}
res["feasible"] = res["est_gpu_h_total"] <= TIMEBOX_GPU_H
res["git_commit"] = os.environ.get("DARTGXE_COMMIT") or subprocess.run(
    ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
res["seconds"] = round(time.time() - t_start)
json.dump(res, open(OUT / "gebiformer_bench.json", "w"), indent=1, default=float)
print("W1D_BENCH_DONE", json.dumps({k: res[k] for k in ("est_gpu_h_total", "feasible")}))
