"""Adapter for GE-BiFormer (Zhou et al. 2026, Brief Bioinform, doi:10.1093/bib/bbag437;
github.com/Zhoushuchang-lab/GE-BiFormer, commit 6563b12, MIT licence). The model, its config and its
EarlyStopping are imported unmodified from `algorithm/` (point GEBIFORMER_CODE at a checkout).

Faithful to the public generalisation script (code/run_generalization_experiments.py, which reuses
algorithm/train.py) in everything except where the evaluation protocol is the object of study:
  * genotype: 0/1/2 with missing calls (-1) drawn uniformly from {0, 1, 2} (their preprocess_snp_matrix),
    minus 1, then sklearn RobustScaler. Their loader fits the scaler on every hybrid in genotype.tsv,
    test hybrids included; `geno_features(scaler_ids=...)` chooses the fit set.
  * environment: one covariate vector per environment. Their Environment_data.csv is not public; we use
    the official APSIM ECs in data_v2 (env_ec.parquet). Missing values are filled with the column mean and
    the matrix is standardised (StandardScaler); their loader computes both over every environment in the
    file; `env_features(scaler_envs=...)` chooses the fit set. Environments without any EC are absent, as
    they would be from their file. Not replicated: their loader also drops the first data row of the
    covariate file (`iloc[1:, 1:]`); we build the matrix directly.
  * target raw yield; HuberLoss; AdamW(lr 3e-4, weight decay 1e-5); ReduceLROnPlateau(0.5, patience 20,
    min 1e-6); their EarlyStopping (patience 30, min_delta 1e-4; it restores the best weights only when it
    triggers); batch 64 with shuffling; gradient clipping 1.0; loss = Huber + 0.01 x MoE auxiliary loss +
    clustering entropy x 0.2 + diversity x 0.1; 100 epochs. All values are read from their config.py,
    which is what the code imports (config.json, which says 300 epochs, is not read by any script).
  * `train()` steps the scheduler and the early stopper on the loss of the `monitor` set. The public
    script passes the TEST set here and reports the best test loss; the fair protocol passes a validation
    year instead.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def import_gebiformer():
    code = os.environ.get("GEBIFORMER_CODE", "<workstation>/dart-gxe/third_party/GE-BiFormer/algorithm")
    if code not in sys.path:
        sys.path.insert(0, code)
    import config as gcfg  # noqa: E402  (their module)
    from model import GeneEnvAttentionModelWithMoE  # noqa: E402
    from utils import EarlyStopping  # noqa: E402
    return gcfg.config, GeneEnvAttentionModelWithMoE, EarlyStopping


def geno_features(geno, hybrids: list[str], scaler_ids, rng: np.random.Generator) -> dict[str, np.ndarray]:
    from sklearn.preprocessing import RobustScaler
    X = geno.X[[geno.index[h] for h in hybrids]].astype(np.float32)
    miss = X < 0
    X[miss] = rng.choice([0, 1, 2], size=int(miss.sum()))
    X = X - 1.0
    pos = {h: i for i, h in enumerate(hybrids)}
    sc = RobustScaler().fit(X[[pos[h] for h in scaler_ids]])
    Z = sc.transform(X).astype(np.float32)
    return {h: Z[i] for i, h in enumerate(hybrids)}


def env_features(ec: pd.DataFrame, envs, scaler_envs) -> dict[str, np.ndarray]:
    from sklearn.preprocessing import StandardScaler
    ec = ec[ec["ec_missing"] == 0].set_index("env")
    envs = [e for e in envs if e in ec.index]
    fit = [e for e in scaler_envs if e in ec.index]
    cols = [c for c in ec.columns if c not in ("lat", "lon", "ec_missing")]
    E = ec.loc[envs, cols].astype(float)
    mu = E.loc[fit].mean()
    cols = [c for c in cols if np.isfinite(mu[c])]  # a column never observed in the fit set carries nothing
    E = E[cols].fillna(mu[cols])
    sc = StandardScaler().fit(E.loc[fit])
    Z = sc.transform(E).astype(np.float32)
    return {e: Z[i] for i, e in enumerate(envs)}


def tensors(rows: pd.DataFrame, G: dict, E: dict, device) -> tuple:
    """Rows whose hybrid or environment has no features are dropped (their loader skips them too)."""
    keep = rows["genotype"].isin(G.keys()) & rows["env"].isin(E.keys())
    r = rows[keep].reset_index(drop=True)
    snp = torch.tensor(np.stack([G[h] for h in r["genotype"]]), device=device)
    env = torch.tensor(np.stack([E[e] for e in r["env"]]), device=device)
    y = torch.tensor(r["y"].to_numpy(np.float32), device=device)
    return r, snp, env, y


@torch.no_grad()
def predict(model, snp, env, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = [model(snp[i:i + batch], env[i:i + batch], hard_clustering=True)[0].reshape(-1) for i in range(0, len(snp), batch)]
    return torch.cat(out).float().cpu().numpy()


def train(cfg, Model, EarlyStopping, fit: tuple, monitor: tuple | None, seed: int, epochs: int | None = None,
          on_epoch=None, max_epochs: int | None = None, lr_schedule: list[float] | None = None):
    """fit/monitor = (snp, env, y) tensors. Returns (model, log). `epochs` fixes the number of epochs (no
    early stopping, used by a refit); otherwise their scheduler and early stopper run on `monitor`.
    `lr_schedule` (with `epochs`) replays the learning rate used in each epoch of a tuning run.
    `on_epoch(epoch, model)` is called after every epoch (e.g. to store predictions); `max_epochs` caps
    the loop for timing runs. log["lr"][i] is the learning rate used during epoch i + 1."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    snp, env, y = fit
    n_feat_snp, n_feat_env = snp.shape[1], env.shape[1]
    model = Model(n_feat_snp, n_feat_env, num_traits=1).to(snp.device)
    crit = nn.HuberLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                            betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg["lr_reduce_factor"],
                                                       patience=cfg["lr_reduce_patience"], min_lr=cfg["min_lr"])
    stopper = EarlyStopping(patience=cfg["early_patience"], min_delta=cfg["early_min_delta"], verbose=False)
    n_ep = epochs if epochs is not None else cfg["epochs"]
    if max_epochs is not None:
        n_ep = min(n_ep, max_epochs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    log = {"monitor_loss": [], "lr": [], "epoch_seconds": [], "stopped_early": False}
    best, best_epoch = float("inf"), 0
    for ep in range(n_ep):
        t0 = time.perf_counter()
        if lr_schedule is not None:
            for g in opt.param_groups:
                g["lr"] = lr_schedule[ep]
        model.train()
        perm = torch.randperm(len(y), generator=gen).to(snp.device)
        for i in range(0, len(y), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            preds, aux = model(snp[b], env[b], hard_clustering=False)
            loss = crit(preds, y[b].squeeze()) + cfg["aux_loss_coef"] * aux
            info = model.get_clustering_info()
            if info is not None:
                loss = loss + cfg.get("clustering_entropy_weight", 0.05) * info.get("entropy", torch.tensor(0.0)) \
                    + cfg.get("clustering_diversity_weight", 0.01) * info.get("diversity", torch.tensor(0.0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        log["lr"].append(opt.param_groups[0]["lr"])
        if on_epoch is not None:
            on_epoch(ep + 1, model)
        if monitor is not None and epochs is None:
            m_snp, m_env, m_y = monitor
            p = torch.tensor(predict(model, m_snp, m_env), device=m_y.device)
            ml = float(crit(p, m_y).item())
            log["monitor_loss"].append(ml)
            sched.step(ml)
            if ml < best - cfg["early_min_delta"]:
                best, best_epoch = ml, ep + 1
            if snp.is_cuda:
                torch.cuda.synchronize()
            log["epoch_seconds"].append(time.perf_counter() - t0)
            if stopper.step(ml, model):
                log["stopped_early"] = True
                break
        else:
            if snp.is_cuda:
                torch.cuda.synchronize()
            log["epoch_seconds"].append(time.perf_counter() - t0)
    log["best_epoch"] = best_epoch
    log["epochs_run"] = len(log["epoch_seconds"])
    return model, log
