"""prereg v5 (round 5): C pilot (leakage-route optimism) + L4 pilot (EC dissimilarity as applicability signal).
Shared helpers; dataset drivers are v5_g2f.py and v5_briwecs.py. Requires numpy/pandas/scipy."""
import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.stats import rankdata

SEED = 20260928


# ---------------------------------------------------------------- ridge on a Gram matrix
def gram_chunked(X, cols=None, chunk=8192, y=None):
    """X'X (and X'y) for column subset `cols` (slice or None), computed in row chunks."""
    Xs = X if cols is None else X[:, cols]
    p = Xs.shape[1]; G = np.zeros((p, p)); b = None if y is None else np.zeros(p)
    for i in range(0, X.shape[0], chunk):
        B = Xs[i:i + chunk]; G += B.T @ B
        if y is not None: b += B.T @ y[i:i + chunk]
    return G, b


def ridge_path_preds(G, b, Xevs, lams):
    """Eigen path: returns {name: (n_ev x n_lam) predictions} for each eval matrix in Xevs (dict)."""
    d, V = np.linalg.eigh(G); d = np.clip(d, 0, None); Vb = V.T @ b
    out = {}
    for k, Xe in Xevs.items():
        XV = Xe @ V
        out[k] = np.column_stack([XV @ (Vb / (d + l)) for l in lams])
    return out


def ridge_solve(G, b, lam):
    c = sla.cho_factor(G + lam * np.eye(G.shape[0]), check_finite=False)
    return sla.cho_solve(c, b, check_finite=False)


def per_env_r(env, y, P, envs):
    """P (n x k) -> DataFrame envs x k of within-env Pearson."""
    rows = []
    for e in envs:
        m = env == e; yy = y[m] - y[m].mean(); PP = P[m] - P[m].mean(0)
        den = np.sqrt((yy ** 2).sum() * (PP ** 2).sum(0))
        rows.append(np.where(den > 0, (yy @ PP) / np.where(den > 0, den, 1), np.nan))
    return pd.DataFrame(np.vstack(rows), index=list(envs))


# ---------------------------------------------------------------- env-mean model
def zfit(A):
    mu = A.mean(0); sd = A.std(0); sd[~(sd > 1e-12)] = 1.0
    return lambda B: (B - mu) / sd


def env_ridge_fit_predict(Etr, ytr, Eev, lam):
    z = zfit(Etr); Z = z(Etr); ybar = ytr.mean()
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (ytr - ybar))
    return z(Eev) @ beta + ybar


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- bootstrap
def boot_mean_diff(d, n_boot=2000, seed=SEED):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(1)
    return dict(mean=float(d.mean()), ci_low=float(np.percentile(bm, 2.5)), ci_high=float(np.percentile(bm, 97.5)), n_env=int(len(d)))


def boot_envmean_r_diff(obs, pred_a, pred_b, n_boot=2000, seed=SEED):
    """Paired env bootstrap of r(pred_a, obs) - r(pred_b, obs) across envs."""
    obs, pa, pb = (np.asarray(x, float) for x in (obs, pred_a, pred_b))
    idx = np.random.default_rng(seed).integers(0, len(obs), size=(n_boot, len(obs)))
    def rr(p, o):
        pc = p - p.mean(1, keepdims=True); oc = o - o.mean(1, keepdims=True)
        return (pc * oc).sum(1) / np.sqrt((pc ** 2).sum(1) * (oc ** 2).sum(1))
    diff = rr(pa[idx], obs[idx]) - rr(pb[idx], obs[idx])
    return dict(mean=pearson(pa, obs) - pearson(pb, obs), r_a=pearson(pa, obs), r_b=pearson(pb, obs),
                ci_low=float(np.nanpercentile(diff, 2.5)), ci_high=float(np.nanpercentile(diff, 97.5)), n_env=int(len(obs)))


# ---------------------------------------------------------------- C2: CERIS-style window search
CERIS_VARS = ["T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT", "VPD"]
CERIS_WINDOWS = [(d1, d2) for d1 in range(0, 141, 5) for d2 in range(d1 + 10, d1 + 61, 5)]


def ceris_candidates(daily_by_env, envs):
    """daily_by_env[env] = (n_days x 7) array from planting day 0. Window [d1, d2] inclusive, env-level mean.
    Returns (n_env x n_candidates) matrix and list of (var, d1, d2)."""
    labels = [(v, d1, d2) for v in CERIS_VARS for (d1, d2) in CERIS_WINDOWS]
    M = np.full((len(envs), len(labels)), np.nan)
    for i, e in enumerate(envs):
        A = daily_by_env[e]; P = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, 0)])
        vals = []
        for vi in range(len(CERIS_VARS)):
            for (d1, d2) in CERIS_WINDOWS:
                vals.append((P[d2 + 1, vi] - P[d1, vi]) / (d2 - d1 + 1) if d2 + 1 <= A.shape[0] else np.nan)
        M[i] = vals
    return M, labels


def ceris_select(M, y):
    """argmax |Pearson| across candidates (first maximum on ties)."""
    Mc = M - np.nanmean(M, 0); yc = y - y.mean()
    r = (Mc * yc[:, None]).sum(0) / np.sqrt((Mc ** 2).sum(0) * (yc ** 2).sum())
    k = int(np.nanargmax(np.abs(r)))
    return k, float(r[k])


# ---------------------------------------------------------------- L4
def mahal_knn(Etr_env, Ete_env, k=5, ridge=1e-3):
    """Standardize on training envs; pooled training covariance + ridge*I; mean distance to k nearest training envs."""
    z = zfit(Etr_env); Ztr = z(Etr_env); Zte = z(Ete_env)
    S = np.cov(Ztr, rowvar=False) + ridge * np.eye(Ztr.shape[1]); Si = np.linalg.inv(S)
    D = np.array([[np.sqrt(max((x - t) @ Si @ (x - t), 0)) for t in Ztr] for x in Zte])
    return np.sort(D, 1)[:, :k].mean(1)


def spearman_perm(x, y, strata, n_perm=10000, seed=SEED):
    """Spearman rho and permutation p (x shuffled within strata). Two-sided and one-sided (rho_perm <= rho_obs)."""
    x = np.asarray(x, float); y = np.asarray(y, float); strata = np.asarray(strata)
    ok = np.isfinite(x) & np.isfinite(y); x, y, strata = x[ok], y[ok], strata[ok]
    ry = rankdata(y); ry = (ry - ry.mean()) / np.sqrt(((ry - ry.mean()) ** 2).sum())
    def rho(xx):
        rx = rankdata(xx); rx = (rx - rx.mean()) / np.sqrt(((rx - rx.mean()) ** 2).sum()); return float(rx @ ry)
    obs = rho(x); rng = np.random.default_rng(seed); groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    perm = np.empty(n_perm)
    for b in range(n_perm):
        xp = x.copy()
        for g in groups: xp[g] = x[rng.permutation(g)]
        perm[b] = rho(xp)
    return dict(rho=obs, p_two_sided=float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (n_perm + 1)),
                p_one_sided_neg=float((1 + np.sum(perm <= obs + 1e-12)) / (n_perm + 1)), n=int(len(x)), n_perm=n_perm)
