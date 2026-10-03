"""One stage-1 run = (scenario, fold, loss config, seed). spec 04 §4.3-4.4.

CV scenarios: train on role=train, early-stop on val mean Spearman (patience 10), predict test with
the best-validation weights. Forward scenarios: the same on train -> val gives the best epoch E*;
then refit from scratch (same seed) on train ∪ val ∪ refit_extra for E* epochs and predict test.
Feature scalers are always fitted on the rows the network is trained on.
"""
from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from dartgxe.baselines.common import Fold, git_commit, git_dirty
from dartgxe.features.fit import EnvScaler, GenoData, GenoScaler
from dartgxe.losses.rank import LOSSES
from dartgxe.models.dart import DartStage1

DEV = "cuda"


@dataclass
class Config:
    loss: str                 # mse | twohead_mse | msed | listnet | soft_spearman | lambda_at_k
    tau: float = 1.0          # listnet
    eps: float = 1.0          # soft_spearman
    K: int = 4                # environments per batch
    max_per_env: int = 512
    lr: float = 1e-3
    wd: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    bf16: bool = True
    use_ec: bool = True       # ablations (docs/prestudy_nn_vs_gblup_2026-09-26.md)
    linear_g: bool = False
    target: str = "z"         # "z" (within-env standardised) | "center" (within-env centred, global SD units)

    @property
    def name(self) -> str:
        extra = {"listnet": f"_t{self.tau:g}", "soft_spearman": f"_e{self.eps:g}"}.get(self.loss, "")
        abl = ("_noec" if not self.use_ec else "") + ("_lin" if self.linear_g else "") + ("_ctr" if self.target != "z" else "")
        return f"S1_{self.loss}{extra}{abl}"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tensors:
    """GPU-resident features for one fold; scalers fitted on `fit_rows`."""

    def __init__(self, fold: Fold, fit_rows: pd.DataFrame, geno: GenoData, ec: pd.DataFrame):
        self.gs = GenoScaler.fit(geno, fit_rows["genotype"].unique())
        self.es = EnvScaler.fit(ec, fit_rows["env"].unique())
        self.hyb = sorted(fold.cells["genotype"].unique())
        self.envs = sorted(fold.cells["env"].unique())
        self.hpos = {h: i for i, h in enumerate(self.hyb)}
        self.epos = {e: i for i, e in enumerate(self.envs)}
        self.X = torch.tensor(self.gs.transform(geno, self.hyb), device=DEV)
        self.E = torch.tensor(self.es.transform(ec, self.envs), device=DEV)
        # global y standardisation from fit rows (only used by `mse` and the μ head; monotone)
        self.ym, self.ysd = float(fit_rows["y"].mean()), float(fit_rows["y"].std())

    def groups(self, rows: pd.DataFrame) -> list[dict]:
        out = []
        for e, g in rows.groupby("env", sort=True):
            y = g["y"].to_numpy(np.float32)
            sd = y.std() if len(y) > 1 and y.std() > 0 else 1.0
            out.append({"env": e, "ei": self.epos[e], "gi": np.array([self.hpos[h] for h in g["genotype"]]),
                        "y": y, "z": (y - y.mean()) / sd, "zc": (y - y.mean()) / self.ysd, "ystd": (y - self.ym) / self.ysd,
                        "ybar": (y.mean() - self.ym) / self.ysd, "idx": g.index.to_numpy()})
        return out


def pad(groups, rng: np.random.Generator | None, max_per_env: int, target: str = "z"):
    L = max(min(len(g["gi"]), max_per_env) for g in groups)
    K = len(groups)
    gi = np.zeros((K, L), np.int64)
    z = np.zeros((K, L), np.float32)
    ys = np.zeros((K, L), np.float32)
    mask = np.zeros((K, L), bool)
    sel = []
    for k, g in enumerate(groups):
        n = len(g["gi"])
        take = rng.choice(n, max_per_env, replace=False) if (rng is not None and n > max_per_env) else np.arange(n)
        m = len(take)
        gi[k, :m], z[k, :m], ys[k, :m], mask[k, :m] = g["gi"][take], g[target if target == "z" else "zc"][take], g["ystd"][take], True
        sel.append(take)
    t = lambda a: torch.tensor(a, device=DEV)
    return t(gi), t(z), t(ys), t(mask), torch.tensor([g["ei"] for g in groups], device=DEV), \
        torch.tensor([g["ybar"] for g in groups], device=DEV, dtype=torch.float32), sel


def loss_fn(cfg: Config, model, T: Tensors, batch):
    gi, z, ys, mask, ei, ybar, _ = batch
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
        s, mu = model(T.X[gi], T.E[ei])
    s = s.float()
    if cfg.loss == "mse":
        d = (s - ys) * mask
        return (d * d).sum() / mask.sum()
    l_rank = LOSSES[cfg.loss](s, z, mask, tau=cfg.tau, eps=cfg.eps)
    return l_rank + ((mu.float() - ybar) ** 2).mean()


@torch.no_grad()
def score(model, T: Tensors, groups, chunk: int = 16):
    """Returns per-group raw scores s (numpy) and μ̂ (or None)."""
    model.eval()
    s_out, mu_out = [], []
    for i in range(0, len(groups), chunk):
        gs = groups[i:i + chunk]
        gi, _, _, mask, ei, _, _ = pad(gs, None, 10**9)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s, mu = model(T.X[gi], T.E[ei])
        s = s.float().cpu().numpy()
        for k, g in enumerate(gs):
            s_out.append(s[k, :len(g["gi"])])
            mu_out.append(None if mu is None else float(mu[k].float()))
    model.train()
    return s_out, mu_out


def val_spearman(s_list, groups, min_n: int = 10) -> float:
    from scipy.stats import spearmanr
    v = [spearmanr(s, g["y"])[0] for s, g in zip(s_list, groups) if len(g["y"]) >= min_n]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def train(cfg: Config, T: Tensors, fit_groups, val_groups, seed: int, epochs: int | None = None):
    """epochs=None: early stopping on val; else a fixed number of epochs (forward refit)."""
    set_seed(seed)
    model = DartStage1(T.X.shape[1], T.E.shape[1], two_head=cfg.loss != "mse", use_ec=cfg.use_ec,
                       linear_g=cfg.linear_g).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_ep = epochs or cfg.max_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    rng = np.random.default_rng(seed)
    best, best_ep, best_state, bad, hist = -np.inf, 0, None, 0, []
    for ep in range(1, n_ep + 1):
        order = rng.permutation(len(fit_groups))
        for i in range(0, len(order), cfg.K):
            b = pad([fit_groups[j] for j in order[i:i + cfg.K]], rng, cfg.max_per_env, cfg.target)
            loss = loss_fn(cfg, model, T, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if epochs is None:
            v = val_spearman(score(model, T, val_groups)[0], val_groups)
            hist.append(v)
            if v > best:
                best, best_ep, bad = v, ep, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_spearman": best if epochs is None else None, "best_epoch": best_ep if epochs is None else epochs,
                   "epochs_run": ep, "val_history": hist}


def calib_a(model, T, groups) -> float:
    """Global within-env least-squares scale a > 0 (only affects RMSE-type diagnostics)."""
    s_list, _ = score(model, T, groups)
    num = den = 0.0
    for s, g in zip(s_list, groups):
        sc, yc = s - s.mean(), g["y"] - g["y"].mean()
        num += float((sc * yc).sum())
        den += float((sc * sc).sum())
    return max(num / den, 1e-6) if den > 0 else 1.0


def predict(cfg: Config, model, T: Tensors, test_groups, a: float) -> dict:
    s_list, mu_list = score(model, T, test_groups)
    out = {}
    for s, mu, g in zip(s_list, mu_list, test_groups):
        if cfg.loss == "mse":
            yhat = s * T.ysd + T.ym
        else:
            yhat = (mu * T.ysd + T.ym) + a * (s - s.mean())  # ŷ = μ̂_e + a(s − s̄_e), a in y units per unit s
        out.update(dict(zip(g["idx"], yhat)))
    return out


def run(cfg: Config, fold: Fold, geno: GenoData, ec: pd.DataFrame, seed: int, out_dir, meta_dir,
        predict_test: bool = True) -> dict:
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    tr, va, te = fold.rows("train"), fold.rows("val"), fold.rows("test")
    T = Tensors(fold, tr, geno, ec)
    model, info = train(cfg, T, T.groups(tr), T.groups(va), seed)
    a = calib_a(model, T, T.groups(va))
    if not predict_test:  # hyperparameter selection: validation only, the test set is never touched
        meta = {"config": asdict(cfg), "method": cfg.name, "scenario": fold.scenario, "fold": fold.fold, "seed": seed,
                **info, "seconds": round(time.time() - t0, 1), "git_commit": git_commit(), "git_dirty": git_dirty(),
                "gpu": torch.cuda.get_device_name(0)}
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
        return meta
    if fold.forward:
        fit = fold.rows("train", "val", "refit_extra")
        T = Tensors(fold, fit, geno, ec)
        model, _ = train(cfg, T, T.groups(fit), [], seed, epochs=max(info["best_epoch"], 1))
    te_groups = T.groups(te)
    pred = predict(cfg, model, T, te_groups, a)
    p = te.assign(prediction=te.index.map(pred)).rename(columns={"y": "observed"})
    p = p[["env", "genotype", "observed", "prediction"]].assign(method=cfg.name, scenario=fold.scenario,
                                                                 fold=fold.fold, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    p.to_parquet(out_dir / f"seed{seed}_fold{fold.fold}.parquet", index=False)
    meta = {"config": asdict(cfg), "method": cfg.name, "scenario": fold.scenario, "fold": fold.fold, "seed": seed,
            "calib_a": a, **{k: v for k, v in info.items()}, "git_commit": git_commit(), "git_dirty": git_dirty(),
            "host": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda": torch.version.cuda, "driver": _driver(), "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def _driver() -> str:
    try:
        import subprocess
        return subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).split()[0]
    except Exception:
        return "unknown"
