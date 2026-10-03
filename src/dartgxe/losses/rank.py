"""Stage-1 losses (spec 04 §4.2). Inputs are padded per-environment batches:
    s    (K, L) model scores          z    (K, L) within-environment standardised targets
    mask (K, L) bool, True where a real cell sits
Each loss is computed within each environment and averaged over the K environments.
All losses except `mse` are invariant to adding a constant to s within an environment.
"""
from __future__ import annotations

import torch

NEG = -1e9


def _n(mask):
    return mask.sum(1).clamp(min=1).float()


def twohead_mse(s, z, mask, **_):
    d = (s - z) * mask
    return ((d * d).sum(1) / _n(mask)).mean()


def msed(s, z, mask, **_):
    """Piepho 1998 / Eckhoff 2026. Σ_{i<j}((s_i−s_j)−(z_i−z_j))² = n Σ (d_i−d̄)², d = s−z;
    divided by the number of pairs n(n−1)/2, i.e. 2/(n−1) Σ (d_i−d̄)²."""
    n = _n(mask)
    d = (s - z) * mask
    dbar = d.sum(1) / n
    dev = (d - dbar[:, None]) * mask
    return (2.0 / (n - 1).clamp(min=1) * (dev * dev).sum(1)).mean()


def listnet(s, z, mask, tau: float = 1.0, **_):
    P = torch.softmax(torch.where(mask, z / tau, NEG), 1)
    logQ = torch.log_softmax(torch.where(mask, s, NEG), 1)
    return -((P * logQ) * mask).sum(1).mean()


def _masked_pearson(a, b, mask):
    n = _n(mask)
    am = (a * mask).sum(1) / n
    bm = (b * mask).sum(1) / n
    ac, bc = (a - am[:, None]) * mask, (b - bm[:, None]) * mask
    return (ac * bc).sum(1) / ((ac * ac).sum(1).sqrt() * (bc * bc).sum(1).sqrt() + 1e-12)


def _hard_rank(x, mask):
    x = torch.where(mask, x, torch.full_like(x, float("inf")))
    return x.argsort(1).argsort(1).float() + 1.0


def soft_spearman(s, z, mask, eps: float = 1.0, **_):
    """Pure-PyTorch soft rank r̃_i = 1 + Σ_{j≠i} σ((s_i − s_j)/ε) (Qin et al. 2010 smooth rank),
    used instead of torchsort (spec 09 §9.2 fallback). L = 1 − Pearson(r̃, rank(z))."""
    pair = torch.sigmoid((s[:, :, None] - s[:, None, :]) / eps) * (mask[:, :, None] & mask[:, None, :])
    r = 0.5 + pair.sum(2)  # includes σ(0)=0.5 for j=i
    return (1.0 - _masked_pearson(r, _hard_rank(z, mask), mask)).mean()


def lambda_at_k(s, z, mask, frac: float = 0.10, **_):
    """LambdaRank-style pairwise loss truncated at k = round(0.10 n) (Burges 2010):
    Σ_{l_i > l_j} |ΔNDCG@k_ij| · log(1 + exp(−(s_i − s_j))), labels l = floor(4 · within-env percentile)
    (0–4), gain 2^l − 1, ranks from the current scores (no gradient through the weights).
    Implementation note: the kit named LambdaLoss/allRank; allRank's source could not be fetched from
    the 4090 (GitHub unreachable), so the classic LambdaRank weighting is used and named as such."""
    K, L = s.shape
    n = _n(mask)
    rz = _hard_rank(z, mask)                      # 1 = lowest
    pct = (rz - 1) / n[:, None]
    lab = torch.clamp(torch.floor(4 * pct), 0, 4) * mask
    gain = (2.0 ** lab - 1) * mask
    k = torch.clamp(torch.round(frac * n), min=1)
    with torch.no_grad():
        pos = _hard_rank(-s.detach(), mask)       # 1 = top by current score
        disc = torch.where(pos <= k[:, None], 1.0 / torch.log2(pos + 1), torch.zeros_like(pos)) * mask
        ideal_pos = _hard_rank(-z, mask)
        idisc = torch.where(ideal_pos <= k[:, None], 1.0 / torch.log2(ideal_pos + 1), torch.zeros_like(pos)) * mask
        idcg = (gain * idisc).sum(1).clamp(min=1e-9)
        w = (gain[:, :, None] - gain[:, None, :]).abs() * (disc[:, :, None] - disc[:, None, :]).abs() / idcg[:, None, None]
        valid = (lab[:, :, None] > lab[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        w = w * valid
    diff = s[:, :, None] - s[:, None, :]
    pair_loss = torch.nn.functional.softplus(-diff)
    return ((w * pair_loss).sum((1, 2)) / w.sum((1, 2)).clamp(min=1e-9)).mean()


LOSSES = {"twohead_mse": twohead_mse, "msed": msed, "listnet": listnet,
          "soft_spearman": soft_spearman, "lambda_at_k": lambda_at_k}
RANKING = ["msed", "listnet", "soft_spearman", "lambda_at_k"]
