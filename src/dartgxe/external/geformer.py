"""Adapter for GEFormer (Yao et al. 2025, Mol Plant; github.com/Deep-Breeding/GEFormer, commit c99448a,
code from GEFormerV1.0_update/code, used unmodified by import; the repository declares no licence, so
it is not redistributed here — point GEFORMER_CODE at a local checkout).

Faithful to the public training script (train.py) in everything except where the evaluation protocol
is the object of study (docs/gate2_*.md):
  * genotype: 0/1/2 integers as float (missing calls filled with the training-fold mode), no scaling;
  * phenotype: raw yield, MSE loss, Adam, 100 epochs, batch size / lr / dropout / depth / neurons1 /
    neurons2 from the same Optuna search space;
  * environment: daily weather, each environment standardised on its own window (their `handle_env`),
    time marks = month, day, weekday (their `time_features(freq='d')`).
Window: 150 days from planting (all G2F envs have >= 195 days of weather after planting).
"""
from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

WINDOW = 150
SEARCH = {"batch": (16, 128), "lr": (1e-7, 1e-2), "dropout": (0.2, 0.6), "depth": (1, 6),
          "neurons1": (128, 512), "neurons2": (1, 128)}


def import_geformer():
    code = os.environ.get("GEFORMER_CODE", "<workstation>/dart-gxe/third_party/GEFormer/GEFormerV1.0_update/code")
    if code not in sys.path:
        sys.path.insert(0, code)
    from GEFormer import GEFormer  # noqa: E402
    return GEFormer


def planting_dates(raw_dir) -> pd.Series:
    t = pd.read_csv(f"{raw_dir}/1_Training_Trait_Data_2014_2023.csv", usecols=["Env", "Date_Planted"], low_memory=False)
    m = pd.read_csv(f"{raw_dir}/2_Testing_Meta_Data_2024.csv", usecols=["Env", "Date_Planted"])
    d = pd.concat([t, m]).dropna()
    d["Date_Planted"] = pd.to_datetime(d["Date_Planted"], format="mixed", errors="coerce")
    return d.dropna().groupby("Env")["Date_Planted"].agg(lambda s: s.mode().iloc[0])


def env_tensors(daily: pd.DataFrame, envs, plant: pd.Series):
    """Returns weather (n_kept, WINDOW, n_var) standardised within each env, marks (n_kept, WINDOW, 3),
    how each env's window was placed, and the kept env list (envs without daily weather are skipped)."""
    vars_ = [c for c in daily.columns if c not in ("env", "date")]
    loc = pd.Series({e: e.rsplit("_", 1)[0] for e in envs})
    doy_by_loc = {}
    for e, d in plant.items():
        doy_by_loc.setdefault(e.rsplit("_", 1)[0], []).append(d.dayofyear)
    W, M, how, kept = [], [], {}, []
    g = dict(tuple(daily.groupby("env")))
    for e in envs:
        w = g.get(e)
        if w is None:  # GEFormer cannot use an environment without daily weather; the caller drops its cells
            how[e] = "no_weather"
            continue
        w = w.sort_values("date")
        if e in plant.index:
            start, how[e] = plant[e], "planting"
        else:
            yr = int(e.rsplit("_", 1)[1])
            doys = doy_by_loc.get(loc[e]) or [d.dayofyear for d in plant.values]
            start, how[e] = pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=int(np.median(doys)) - 1), "location_median_doy"
        seg = w[(w["date"] >= start)].head(WINDOW)
        if len(seg) < WINDOW:
            raise ValueError(f"{e}: only {len(seg)} days after {start.date()}")
        x = seg[vars_].to_numpy(np.float64)
        if np.isnan(x).any():
            # the released 2024 test weather is complete only to 2024-07-01; a NaN window would
            # propagate through the per-env standardisation to NaN predictions (bug found 2026-09-27)
            how[e] = "nan_in_window"
        x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)  # StandardScaler on this env only
        W.append(x.astype(np.float32))
        dt = seg["date"]
        M.append(np.c_[dt.dt.month, dt.dt.day, dt.dt.weekday].astype(np.float32))
        kept.append(e)
    return np.stack(W), np.stack(M), how, kept


def make_args(n_var: int, hp: dict):
    return SimpleNamespace(enc_in=n_var, c_out=n_var, dropout=hp["dropout"], depth=hp["depth"],
                           neurons1=hp["neurons1"], neurons2=hp["neurons2"])


class Data:
    """GPU-resident genotype, weather and marks; rows are indexed by (hybrid index, env index)."""

    def __init__(self, G: np.ndarray, W: np.ndarray, M: np.ndarray, dev: str):
        self.G = torch.tensor(G, dtype=torch.float32, device=dev)
        self.W = torch.tensor(W, device=dev)
        self.M = torch.tensor(M, device=dev)

    def batch(self, gi, ei):
        # their collate turns the per-SNP list into (snp_len, B); GEFormer.forward transposes it back
        return self.G[gi].T.contiguous(), self.W[ei], self.M[ei]


def epoch_train(net, data, gi, ei, y, bs, opt, rng):
    net.train()
    order = rng.permutation(len(y))
    lossf = torch.nn.MSELoss()
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        x, w, m = data.batch(gi[b], ei[b])
        pred = net(x, w, m).flatten()
        loss = lossf(pred, y[b])
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def predict(net, data, gi, ei, bs=512):
    net.eval()
    out = []
    for s in range(0, len(gi), bs):
        x, w, m = data.batch(gi[s:s + bs], ei[s:s + bs])
        out.append(net(x, w, m).flatten().float().cpu())
    return torch.cat(out).numpy()


def pooled_pearson(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")
