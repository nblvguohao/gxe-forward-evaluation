"""Hardware benchmark used to decide where the formal runs go (4090 vs amax H200).

GPU part: the fixed stage-1 network of spec 04 §4.1 on synthetic data with the real
G2F shapes (2,425 SNPs, 644 EC columns, K = 4 environments x 512 hybrids per batch,
bf16 autocast, AdamW). Reports training steps/s and the implied time per run
(100 epochs x ~55 steps + one validation pass per epoch). A 4x wider variant is
included as a proxy for the heavier stage-2 variants.

CPU part: X'X for 100k x 2,425 (the B1/B2 ridge bottleneck) and LightGBM on
108k x 744 (B4), at a fixed thread count.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn


def net(width: int, p: int = 2425, q: int = 644) -> nn.Module:
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.Sequential(nn.Linear(p, 256 * width), nn.LayerNorm(256 * width), nn.GELU(), nn.Dropout(0.2), nn.Linear(256 * width, 64 * width))
            self.e = nn.Sequential(nn.Linear(q, 64 * width), nn.GELU(), nn.Linear(64 * width, 64 * width))
            self.head = nn.Sequential(nn.Linear(192 * width, 64 * width), nn.GELU(), nn.Linear(64 * width, 1))
            self.mu = nn.Sequential(nn.Linear(q, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, xg, xe):
            hg, he = self.g(xg), self.e(xe)
            he = he.unsqueeze(1).expand(-1, hg.shape[1], -1)
            return self.head(torch.cat([hg, he, hg * he], -1)).squeeze(-1), self.mu(xe).squeeze(-1)

    return Net()


def gpu_bench(width: int, steps: int = 300, K: int = 4, n: int = 512) -> dict:
    dev = "cuda"
    torch.manual_seed(0)
    X = torch.randn(5899, 2425, device=dev)
    E = torch.randn(272, 644, device=dev)
    y = torch.randn(272, n, device=dev)
    m = net(width).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)

    def step():
        ei = torch.randint(0, 272, (K,), device=dev)
        gi = torch.randint(0, 5899, (K, n), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s, mu = m(X[gi], E[ei])
        s = s.float()
        z = y[ei]
        # ListNet within each environment (the heaviest of the planned losses besides pairwise ones)
        loss = -(torch.softmax(z, -1) * torch.log_softmax(s, -1)).sum(-1).mean() + ((mu.float() - z.mean(-1)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(30):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / steps
    # validation pass: every environment, 512 hybrids, no grad
    m.eval()
    t = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for e0 in range(0, 272, 16):
            ei = torch.arange(e0, min(e0 + 16, 272), device=dev)
            gi = torch.randint(0, 5899, (len(ei), n), device=dev)
            m(X[gi], E[ei])
    torch.cuda.synchronize()
    val = time.perf_counter() - t
    m.train()
    run_s = 100 * (55 * dt + val)
    return {"width": width, "ms_per_step": round(dt * 1e3, 3), "val_pass_s": round(val, 3),
            "est_run_s_100ep": round(run_s, 1), "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def cpu_bench(threads: int) -> dict:
    out = {"threads": threads}
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100_000, 2425))
    t = time.perf_counter()
    X.T @ X
    out["xtx_100k_x_2425_s"] = round(time.perf_counter() - t, 2)
    try:
        import lightgbm as lgb

        Xl = rng.standard_normal((108_000, 744)).astype(np.float32)
        yl = Xl[:, :5].sum(1) + rng.standard_normal(108_000)
        t = time.perf_counter()
        lgb.train({"objective": "regression", "num_threads": threads, "verbose": -1, "learning_rate": 0.05,
                   "num_leaves": 63, "feature_fraction": 0.5}, lgb.Dataset(Xl, yl), num_boost_round=300)
        out["lgbm_108k_x_744_300rounds_s"] = round(time.perf_counter() - t, 2)
    except ImportError:
        out["lgbm"] = "not installed"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda": torch.version.cuda, "cpu_count": os.cpu_count()}
    res["gpu_bench"] = [gpu_bench(1), gpu_bench(4)]
    res["cpu_bench"] = cpu_bench(a.threads)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
