"""Wave 2d learners (docs/prereg_sequel_wave2d_2026-09-28.md): an MLP trained with a within-environment loss.

dl_g  : genotype PCs (80) -> score.  dl_ge : genotype PCs + 5 EC PCs (historical-mean ECs for the environments
being predicted, as wave 2c), concatenated in the first layer.
Loss: mean squared error after centring target and prediction within each environment (equal to the
within-environment MSED up to a constant). Every batch holds ENVS_PER_BATCH complete environments.
Tuning: train on the years before the validation year V, choose (configuration, epoch) by V's mean
within-environment Spearman (admissible metric); refit on every year before the target year for E* epochs.
Implementation details fixed here (not in the pre-registration): 4 environments per batch; a configuration's
tuning run stops after 50 epochs without improvement (the pre-registration caps at 200)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata

from dartgxe.forward.cellgxe import ec_pcs
from dartgxe.forward.panel import features

GRID = [{"h": h, "p": p, "wd": wd} for h in (64, 256) for p in (0.1, 0.3) for wd in (1e-4, 1e-2)]
MAX_EPOCHS, PATIENCE, LR, ENVS_PER_BATCH = 200, 50, 1e-3, 4


class MLP(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def within_env_mse(pred, y, eid, n_env):
    """MSE after centring prediction and target within each environment (eid = 0..n_env-1 in the batch)."""
    cnt = torch.zeros(n_env, device=y.device).index_add_(0, eid, torch.ones_like(y))
    mp = torch.zeros(n_env, device=y.device).index_add_(0, eid, pred) / cnt
    my = torch.zeros(n_env, device=y.device).index_add_(0, eid, y) / cnt
    return (((pred - mp[eid]) - (y - my[eid])) ** 2).mean()


def env_spearman(env, y, p):
    out = []
    for e in pd.unique(env):
        m = env == e
        if m.sum() < 3:
            continue
        a, b = rankdata(y[m]), rankdata(p[m])
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        out.append((a * b).sum() / d if d > 0 else np.nan)
    return float(np.nanmean(out)) if out else np.nan


class Design:
    """Features for a fit set and a set of cells to predict; transforms fitted on the fit set only."""

    def __init__(self, fit: pd.DataFrame, pred: pd.DataFrame, markers: pd.DataFrame, ec=None):
        fit = fit[fit["genotype"].isin(markers.index)]
        self.fit, self.pred = fit.reset_index(drop=True), pred.reset_index(drop=True)
        fit_ids = sorted(set(fit["genotype"]))
        pred_ids = sorted(set(pred["genotype"]))
        _, _, Pf, Pp = features(markers, fit_ids, pred_ids)
        fpos, ppos = {h: i for i, h in enumerate(fit_ids)}, {h: i for i, h in enumerate(pred_ids)}
        Xf = Pf[[fpos[h] for h in self.fit["genotype"]]]
        Xp = Pp[[ppos[h] for h in self.pred["genotype"]]]
        if ec is not None:
            W, _ = ec_pcs(ec, sorted(set(self.fit["env"])), sorted(set(self.pred["env"])), "hist")
            Xf = np.c_[Xf, np.stack([W[e] for e in self.fit["env"]])]
            Xp = np.c_[Xp, np.stack([W[e] for e in self.pred["env"]])]
        mu, sd = Xf.mean(0), Xf.std(0)
        sd[sd < 1e-12] = 1.0
        self.Xf, self.Xp = ((Xf - mu) / sd).astype(np.float32), ((Xp - mu) / sd).astype(np.float32)
        self.yf = self.fit["y"].to_numpy(np.float32)
        codes, self.envs = pd.factorize(self.fit["env"])
        self.ef = codes.astype(np.int64)


def train(des: Design, cfg: dict, seed: int, epochs: int, val=None, device="cuda"):
    """Train for up to `epochs`; with `val` (Design-compatible pred set with y) record the validation
    within-env Spearman after every epoch and stop after PATIENCE epochs without improvement.
    Returns (model, curve)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.tensor(des.Xf, device=device)
    y = torch.tensor(des.yf, device=device)
    e = torch.tensor(des.ef, device=device)
    n_env = len(des.envs)
    rows_of = [np.flatnonzero(des.ef == k) for k in range(n_env)]
    model = MLP(X.shape[1], cfg["h"], cfg["p"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg["wd"])
    curve, best, since = [], -np.inf, 0
    Xv = torch.tensor(des.Xp, device=device) if val is not None else None
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n_env)
        for i in range(0, n_env, ENVS_PER_BATCH):
            ks = order[i:i + ENVS_PER_BATCH]
            idx = np.concatenate([rows_of[k] for k in ks])
            local = torch.tensor(np.repeat(np.arange(len(ks)), [len(rows_of[k]) for k in ks]), device=device)
            ii = torch.tensor(idx, device=device)
            loss = within_env_mse(model(X[ii]), y[ii], local, len(ks))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is not None:
            s = env_spearman(val["env"].to_numpy(), val["y"].to_numpy(float), predict(model, Xv))
            curve.append(s)
            if np.isfinite(s) and s > best:
                best, since = s, 0
            else:
                since += 1
                if since >= PATIENCE:
                    break
    return model, curve


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(X).float().cpu().numpy()


def tune_and_refit(ds, V: int, Y: int, seed: int, ec=None, device="cuda"):
    """Pre-registration §2: tune on the years before V scored on V's scorable environments, refit on the years
    before Y, predict Y's scorable environments. Returns (predictions, meta)."""
    cells, markers = ds.cells, ds.markers
    tune_fit = cells[cells["year"] < V]
    vcells = cells[(cells["year"] == V) & cells["env"].isin(ds.scorable_envs([V]))]
    dv = Design(tune_fit, vcells, markers, ec)
    vrows = dv.pred
    best = (-np.inf, None, None)
    curves = {}
    for ci, cfg in enumerate(GRID):
        _, curve = train(dv, cfg, seed, MAX_EPOCHS, val=vrows, device=device)
        curves[ci] = curve
        if curve and np.nanmax(curve) > best[0]:
            best = (float(np.nanmax(curve)), ci, int(np.nanargmax(curve)) + 1)
    assert tune_fit["year"].max() < V
    _, ci, e_star = best
    refit = cells[cells["year"] < Y]
    target = cells[(cells["year"] == Y) & cells["env"].isin(ds.scorable_envs([Y]))]
    dy = Design(refit, target, markers, ec)
    model, _ = train(dy, GRID[ci], seed, e_star, val=None, device=device)
    assert refit["year"].max() < Y
    out = dy.pred[["env", "year", "genotype", "y"]].copy()
    out["pred"] = predict(model, torch.tensor(dy.Xp, device=device))
    meta = {"V": int(V), "Y": int(Y), "seed": seed, "config": GRID[ci], "config_index": ci, "e_star": e_star,
            "val_best_spearman": best[0], "val_curves_len": {k: len(v) for k, v in curves.items()},
            "n_tune_cells": len(dv.fit), "n_refit_cells": len(dy.fit), "n_target_cells": len(out)}
    return out, meta
