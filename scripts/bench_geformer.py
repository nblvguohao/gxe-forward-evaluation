"""Time GEFormer per epoch on a real G2F training set, to fix the gate-2 budget before any result."""
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

from dartgxe.baselines.common import load_folds
from dartgxe.external.geformer import Data, env_tensors, import_geformer, make_args, planting_dates, epoch_train
from dartgxe.features.fit import load_geno
from dartgxe.paths import processed

dev = "cuda"
torch.manual_seed(0)
fold = next(load_folds("F2024m"))
fr = fold.rows("train", "val", "refit_extra")
geno = load_geno()
hyb = sorted(fold.cells["genotype"].unique())
X = geno.X[[geno.index[h] for h in hyb]].astype(np.float32)
fit = X[[hyb.index(h) for h in sorted(set(fr["genotype"]))]]
mode = np.stack([(fit == k).sum(0) for k in (0, 1, 2)]).argmax(0)
X = np.where(X < 0, mode, X)
envs = sorted(fold.cells["env"].unique())
daily = pd.read_parquet(processed("g2f") / "env_daily.parquet")
W, M, how, envs = env_tensors(daily, envs, planting_dates(os.environ["G2F_RAW"]))
fr = fr[fr["env"].isin(set(envs))]
data = Data(X, W, M, dev)
gi = torch.tensor([hyb.index(h) for h in fr["genotype"]], device=dev)
ei = torch.tensor([envs.index(e) for e in fr["env"]], device=dev)
y = torch.tensor(fr["y"].to_numpy(np.float32), device=dev)
GEF = import_geformer()
res = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "n_train_cells": len(y),
       "window_placement": pd.Series(how).value_counts().to_dict()}
for name, hp in {"default": dict(batch=64, lr=5e-4, dropout=0.3, depth=2, neurons1=256, neurons2=32),
                 "slowest": dict(batch=16, lr=5e-4, dropout=0.3, depth=6, neurons1=512, neurons2=128)}.items():
    net = GEF(make_args(W.shape[2], hp), X.shape[1], W.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), hp["lr"])
    n_b = 200
    idx = np.arange(n_b * hp["batch"]) % len(y)
    sub = torch.tensor(idx, device=dev)
    rng = np.random.default_rng(0)
    epoch_train(net, data, gi[sub[:hp["batch"] * 5]], ei[sub[:hp["batch"] * 5]], y[sub[:hp["batch"] * 5]], hp["batch"], opt, rng)  # warm-up
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    epoch_train(net, data, gi[sub], ei[sub], y[sub], hp["batch"], opt, rng)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / n_b
    ep = dt * len(y) / hp["batch"]
    res[name] = {"s_per_batch": round(dt, 4), "est_s_per_epoch": round(ep, 1), "est_h_per_trial_100ep": round(ep * 100 / 3600, 2),
                 "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20), "params": sum(p.numel() for p in net.parameters())}
print(json.dumps(res, indent=1))
