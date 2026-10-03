"""LSTM-GNN feasibility (docs/prereg_sequel_wave2a_2026-09-28.md, section appended before any LSTM-GNN run):
architecture C on the forward splits, training-year rows only. Transforms are fitted on the training years;
the validation year is monitored; no test-year hybrid or environment is used and nothing is scored on it.

The training step is their own Arch_C_train.train_epoch, run on a random subset of TIMED_BATCHES batches to time
it. If their batch size (32) does not fit in memory, batches of 8 and 4 are timed and memory is extrapolated.
Writes results/w1d/lstmgnn_bench.json."""
import gc
import json
import math
import os
import socket
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Subset

from dartgxe.baselines.common import load_folds
from dartgxe.external.lstmgnn import (env_vectors, geno_pcs, import_lstmgnn, marker_table, preprocess_markers, robust,
                                      trait_frame)
from dartgxe.features.fit import load_geno
from dartgxe.paths import RESULTS, processed

TIMED_BATCHES = int(os.environ.get("W1D_BATCHES", "40"))
SEEDS, RUNS_PER_SEED, EPOCH_CAP, TIMEBOX_GPU_H = 3, 3, 100, 72
dev = torch.device("cuda")
OUT = RESULTS / "w1d"
OUT.mkdir(parents=True, exist_ok=True)
t_start = time.time()
ppd, evl, arc = import_lstmgnn()
geno = load_geno()
daily = pd.read_parquet(processed("g2f") / "env_daily.parquet")


def timed(ds, batch, n_batches, model, opt, scaler):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(len(ds), batch * (n_batches + 2)), replace=False)
    loader = arc.make_loader(Subset(ds, idx.tolist()), batch, 4, True, dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated() / 2**20
    try:
        arc.train_epoch(model, arc.make_loader(Subset(ds, idx[:batch * 2].tolist()), batch, 4, True, dev), opt, scaler, dev, 0.8)  # warm-up
        torch.cuda.synchronize()
        t = time.perf_counter()
        arc.train_epoch(model, loader, opt, scaler, dev, 0.8)
        torch.cuda.synchronize()
        n = math.ceil(len(idx) / batch)
        return {"batch": batch, "s_per_batch": (time.perf_counter() - t) / n, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "oom": False,
                "mem_at_start_mb": mem0}
    except torch.cuda.OutOfMemoryError:
        oom_peak = torch.cuda.max_memory_allocated() / 2**20
    gc.collect()  # outside the except block, so the traceback no longer pins the failed step's tensors
    torch.cuda.empty_cache()
    return {"batch": batch, "oom": True, "peak_mem_mb": oom_peak}


res = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "gpu_mem_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
       "torch": torch.__version__, "timed_batches": TIMED_BATCHES, "scenarios": {}}
for sc in ("F2024m", "F2022m"):
    fold = next(load_folds(sc))
    tr, va = fold.rows("train"), fold.rows("val")
    final = fold.rows("train", "val", "refit_extra")
    t0 = time.time()
    fit_h = sorted(set(tr["genotype"]))
    hyb = sorted(set(fit_h) | set(va["genotype"]))
    M = preprocess_markers(ppd, marker_table(geno, hyb), fit_h, seed=42)
    G, expl = geno_pcs(M, fit_h, 548, 42)
    fit_e = sorted(set(tr["env"]))
    Eraw, rec = env_vectors(evl, daily, fit_e, sorted(set(fit_e) | set(va["env"])), 42, "cuda")
    E = robust(Eraw, [e for e in fit_e if e in Eraw])
    prep_s = time.time() - t0
    tt, tv = trait_frame(tr), trait_frame(va)
    tt = tt[tt.Hybrid.isin(G.keys()) & tt.Env.isin(E.keys())].reset_index(drop=True)
    tv = tv[tv.Hybrid.isin(G.keys()) & tv.Env.isin(E.keys())].reset_index(drop=True)
    ys = RobustScaler().fit(tt[["Yield_Mg_ha"]].to_numpy(np.float32))
    gk, ek = list(G), list(E)
    GR, ER = np.stack([G[h] for h in gk]), np.stack([E[e] for e in ek])
    g2i, e2i = {h: i for i, h in enumerate(gk)}, {e: i for i, e in enumerate(ek)}
    arc.set_seed(42)
    ds = arc.GxEDataset(tt, GR, ER, g2i, e2i, ys, 10)
    trials = []
    for b in [int(x) for x in os.environ.get("W1D_SIZES", "4,8,16,32").split(",")]:  # ascending, stop at the first out-of-memory
        model = arc.GxEGAT(128, 8, 0.25, 30).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        r = timed(ds, b, TIMED_BATCHES, model, opt, arc.grad_scaler(dev))
        r["params"] = int(sum(p.numel() for p in model.parameters()))
        trials.append(r)
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
        if r["oom"]:
            break
    ok = [r for r in trials if not r["oom"]]
    base = max(ok, key=lambda r: r["batch"]) if ok else None  # the largest batch that fits
    n_final = int(final["genotype"].isin(G.keys()).sum())  # approximate: final-fit hybrids outside train+val are not featurised here
    if base:
        s_epoch_final = base["s_per_batch"] * math.ceil(len(final) / base["batch"])
        gpu_h = SEEDS * RUNS_PER_SEED * EPOCH_CAP * s_epoch_final / 3600
    else:
        s_epoch_final, gpu_h = None, None
    mem32 = None
    small = [r for r in ok if r["batch"] < 32]
    if len(small) >= 2:  # linear in batch size through the two largest batches that fit
        (b1, m1), (b2, m2) = sorted((r["batch"], r["peak_mem_mb"]) for r in small)[-2:]
        mem32 = m1 + (m2 - m1) / (b2 - b1) * (32 - b1)
    res["scenarios"][sc] = {
        "cells": {"train": len(tt), "val": len(tv), "final_fit_total": len(final)},
        "markers_after_their_filters": int(M.shape[1]), "pcs": int(GR.shape[1]), "pca_explained": round(expl, 4),
        "envs_with_vectors": len(E), "lstm_best_reconstruction_mse": rec, "prep_seconds": round(prep_s),
        "timing_trials": trials, "est_s_per_epoch_final_fit": s_epoch_final,
        "est_gpu_h_budget_at_timed_batch": gpu_h, "extrapolated_peak_mem_mb_batch32": mem32,
    }
    print(sc, json.dumps(res["scenarios"][sc], default=float), flush=True)
tot = [v["est_gpu_h_budget_at_timed_batch"] for v in res["scenarios"].values()]
res["est_gpu_h_total"] = sum(tot) if all(t is not None for t in tot) else None
res["budget"] = {"seeds": SEEDS, "runs_per_seed": RUNS_PER_SEED, "epoch_cap": EPOCH_CAP, "timebox_gpu_h": TIMEBOX_GPU_H}
res["git_commit"] = os.environ.get("DARTGXE_COMMIT") or subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                                                       capture_output=True, text=True).stdout.strip()
res["seconds"] = round(time.time() - t_start)
json.dump(res, open(OUT / os.environ.get("W1D_OUT", "lstmgnn_bench.json"), "w"), indent=1, default=float)
print("LSTMGNN_BENCH_DONE", json.dumps({"est_gpu_h_total": res["est_gpu_h_total"]}, default=float))
