"""prereg v7 H1/H2 helpers (settings S1_G2F_LOYO, S2_G2F_LOLO, S3_BRIWECS_LOYO; protocols P1/P2/P3).
Requires v5_lib.py exec'd first (ceris_*, per_env_r, boot helpers, zfit, pearson)."""
import numpy as np
import pandas as pd

V7_GRID = np.array([1.0, 3.162278, 10.0, 31.622777, 100.0, 316.227766, 1000.0, 3162.27766, 10000.0, 31622.776602,
                    100000.0, 316227.766017, 1000000.0, 3162277.660168, 10000000.0])
SEED = 20260928


def centered_gram(X, y, rows, chunk=8192):
    """(X-xbar)'(X-xbar), (X-xbar)'(y-ybar) over the given rows (column means taken on those rows)."""
    p = X.shape[1]; G = np.zeros((p, p)); s = np.zeros(p); xy = np.zeros(p); ys = 0.0
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]; B = X[r]; yy = y[r]
        G += B.T @ B; s += B.sum(0); xy += B.T @ yy; ys += yy.sum()
    m = len(rows); xbar = s / m; ybar = ys / m
    return G - m * np.outer(xbar, xbar), xy - m * xbar * ybar, xbar, ybar


def path_predict(Gc, bc, Xev, lams):
    d, V = np.linalg.eigh(Gc); d = np.clip(d, 0, None); Vb = V.T @ bc; XV = Xev @ V
    return np.column_stack([XV @ (Vb / (d + l)) for l in lams])


def block_paths(X, y, tr_rows, ev_X, blocks, lams):
    """For each named column block (slice), prediction path on ev_X. Centred ridge with unpenalised intercept
    (intercept omitted from predictions: within-env r is invariant to it)."""
    out = {}
    Gc, bc, _, _ = centered_gram(X, y, tr_rows)
    for name, sl in blocks.items():
        out[name] = path_predict(Gc[sl, sl], bc[sl], ev_X[:, sl], lams)
    return out


def ceris_candidates_w(daily_by_env, envs, windows, nvars):
    """generalised CERIS candidate matrix: window [d1, d2] inclusive, env-level mean; NaN if beyond record."""
    M = np.full((len(envs), nvars * len(windows)), np.nan)
    for i, e in enumerate(envs):
        A = daily_by_env[e]; P = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, 0)])
        vals = []
        for vi in range(nvars):
            for (d1, d2) in windows:
                vals.append((P[d2 + 1, vi] - P[d1, vi]) / (d2 - d1 + 1) if d2 + 1 <= A.shape[0] and d1 >= 0 else np.nan)
        M[i] = vals
    return M


def ols_predict(x_tr, y_tr, x_te):
    A = np.column_stack([np.ones_like(x_tr), x_tr]); c = np.linalg.lstsq(A, y_tr, rcond=None)[0]
    return c[0] + c[1] * x_te


def boot_p_one_sided(d, n_boot=2000, seed=SEED):
    """bootstrap one-sided p for mean(d) > 0: (1 + #boot_means <= 0) / (B + 1)."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(1); return float((1 + np.sum(bm <= 0)) / (n_boot + 1))


def boot_p_envmean(obs, pa, pb, n_boot=2000, seed=SEED):
    obs, pa, pb = (np.asarray(x, float) for x in (obs, pa, pb))
    idx = np.random.default_rng(seed).integers(0, len(obs), size=(n_boot, len(obs)))
    def rr(p, o):
        pc = p - p.mean(1, keepdims=True); oc = o - o.mean(1, keepdims=True)
        return (pc * oc).sum(1) / np.sqrt((pc ** 2).sum(1) * (oc ** 2).sum(1))
    diff = rr(pa[idx], obs[idx]) - rr(pb[idx], obs[idx])
    return float((1 + np.sum(diff <= 0)) / (n_boot + 1))


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj.tolist()
