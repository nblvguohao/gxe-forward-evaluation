"""Closed-form baselines B0, B1, B2, B3, B5 (spec 02).

All within-environment rankings come from models with environment fixed effects, fitted by
centring y and features within environment (Frisch-Waugh-Lovell). The environment mean needed for
RMSE-type diagnostics comes from `EnvMean` (training mean for known environments, a ridge on EC
for new ones). Tuning budget: 20 grid points per method, scored by validation mean Spearman.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from dartgxe.baselines.common import Fold, val_score
from dartgxe.features.fit import EnvScaler, GenoData, GenoPCA, GenoScaler

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def codes(env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, c = np.unique(env, return_inverse=True)
    return u, c


def center(M: np.ndarray, c: np.ndarray, n_groups: int) -> np.ndarray:
    """Subtract group means (c = group code per row) using a sparse indicator product."""
    M2 = M if M.ndim == 2 else M[:, None]
    ind = sp.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))), shape=(n_groups, len(c)))
    cnt = np.bincount(c, minlength=n_groups)[:, None]
    out = M2 - (np.asarray(ind @ M2) / cnt)[c]
    return out if M.ndim == 2 else out[:, 0]


class Ridge:
    """Ridge on within-env-centred features with a shared eigendecomposition for a λ grid (GPU, float64)."""

    def __init__(self, F: np.ndarray, env: np.ndarray, y: np.ndarray):
        u, c = codes(env)
        Fc = torch.tensor(center(F.astype(np.float64), c, len(u)), device=DEV)
        yc = torch.tensor(center(y.astype(np.float64), c, len(u)), device=DEV)
        G = Fc.T @ Fc
        self.d, self.V = torch.linalg.eigh(G)
        self.b = self.V.T @ (Fc.T @ yc)
        self.scale = float(torch.trace(G)) / G.shape[0]
        self.yy = float((yc * yc).sum())
        self.n_resid = len(y) - len(u)  # REML degrees of freedom after absorbing environment effects

    def reml(self) -> dict:
        """REML for y_c = F_c β + e, β ~ N(0, σ²_β I), e ~ N(0, σ²_e I), on the within-environment
        contrasts (dimension n − n_env). Profile likelihood in δ = σ²_β/σ²_e:
        −2ℓ(δ) ∝ n′ log Q(δ) + Σ log(1 + δ d_i), Q(δ) = Σ c_i²/(1 + δ d_i) + (‖y_c‖² − Σ c_i²),
        c_i² = b_i²/d_i. Returns δ, λ = 1/δ (absolute and relative to the mean eigenvalue) and h²."""
        d = self.d.cpu().numpy()
        b = self.b.cpu().numpy()
        keep = d > 1e-8 * d.max()
        d, c2 = d[keep], (b[keep] ** 2) / d[keep]
        rest = max(self.yy - c2.sum(), 1e-12)
        n = self.n_resid

        def f(ld):
            dl = np.exp(ld)
            return n * np.log((c2 / (1 + dl * d)).sum() + rest) + np.log1p(dl * d).sum()

        grid = np.linspace(np.log(1e-6), np.log(1e4), 400)
        vals = np.array([f(g) for g in grid])
        i = int(vals.argmin())
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        for _ in range(60):  # golden-section refinement
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            if f(m1) < f(m2):
                hi = m2
            else:
                lo = m1
        delta = float(np.exp((lo + hi) / 2))
        lam = 1.0 / delta
        s2e = ((c2 / (1 + delta * d)).sum() + rest) / n
        # share of within-env variance explained by the marker term (h² of the fitted model)
        vg = delta * s2e * self.scale * len(self.d) / max(n, 1)
        return {"delta": delta, "lambda": lam, "lambda_rel": lam / self.scale, "s2e": s2e,
                "h2": vg / (vg + s2e), "at_grid_edge": i in (0, len(grid) - 1)}

    def coef(self, lam_rel: float) -> np.ndarray:
        lam = lam_rel * self.scale
        return (self.V @ (self.b / (self.d + lam))).cpu().numpy()


class EnvMean:
    """Environment mean: observed training mean if the environment was trained on; otherwise a
    ridge on standardised EC fitted to training environment means (λ by leave-one-out)."""

    def __init__(self, fit_rows: pd.DataFrame, resid: np.ndarray, ec: pd.DataFrame, scaler: EnvScaler):
        m = pd.Series(resid).groupby(fit_rows["env"].to_numpy()).mean()
        self.known = m
        X = scaler.transform(ec, m.index)
        X = np.c_[np.ones(len(X)), X]
        yv = m.to_numpy()
        best = (np.inf, None)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        for lam in np.logspace(-1, 4, 12):
            H = (U * (S**2 / (S**2 + lam))) @ U.T
            r = (yv - H @ yv) / (1 - np.diag(H))
            if (r**2).mean() < best[0]:
                best = ((r**2).mean(), lam)
        lam = best[1]
        self.w = Vt.T @ ((S / (S**2 + lam)) * (U.T @ yv))
        self.ec, self.scaler = ec, scaler

    def __call__(self, envs: np.ndarray) -> np.ndarray:
        u = pd.unique(envs)
        new = [e for e in u if e not in self.known.index]
        vals = self.known.to_dict()
        if new:
            X = np.c_[np.ones(len(new)), self.scaler.transform(self.ec, new)]
            vals.update(dict(zip(new, X @ self.w)))
        return pd.Series(envs).map(vals).to_numpy()


def _geno_feats(geno: GenoData, fit_rows: pd.DataFrame, all_ids) -> tuple[GenoScaler, dict]:
    sc = GenoScaler.fit(geno, fit_rows["genotype"].unique())
    ids = sorted(set(all_ids))
    Z = sc.transform(geno, ids)
    return sc, dict(zip(ids, range(len(ids)))), Z


LAMS = np.logspace(-3, 2, 20)  # relative to mean eigenvalue: 20 grid points = the tuning budget


def fit_b1(fold: Fold, geno: GenoData, ec: pd.DataFrame):
    """B1 GBLUP-equivalent: ridge on standardised markers + environment fixed effects.
    Returns predictions for B0, B1 and B5 (B5 = B1's genetic values + EC environment mean; its
    within-environment ranking is identical to B1's by construction)."""
    out, info = {}, {}

    def fit(roles, lam=None):
        fr = fold.rows(*roles)
        sc, pos, Z = _geno_feats(geno, fr, fold.cells["genotype"])
        F = Z[[pos[g] for g in fr["genotype"]]]
        r = Ridge(F, fr["env"].to_numpy(), fr["y"].to_numpy())
        return fr, pos, Z, r, sc

    fr, pos, Z, r, _ = fit(fold.fit_roles(False))
    va = fold.rows("val")
    Fv = Z[[pos[g] for g in va["genotype"]]]
    scores = [val_score(va, Fv @ r.coef(l)) for l in LAMS]
    lam = float(LAMS[int(np.nanargmax(scores))])
    info["B1"] = {"lambda_rel": lam, "val_spearman": float(np.nanmax(scores)), "grid": len(LAMS)}
    if fold.forward:
        fr, pos, Z, r, _ = fit(fold.fit_roles(True))
    beta = r.coef(lam)
    te = fold.rows("test")
    g_te = Z[[pos[g] for g in te["genotype"]]] @ beta
    g_fr = Z[[pos[g] for g in fr["genotype"]]] @ beta
    esc = EnvScaler.fit(ec, fr["env"].unique())
    resid = fr["y"].to_numpy() - g_fr
    envs_te = te["env"].to_numpy()
    known = pd.Series(resid).groupby(fr["env"].to_numpy()).mean()
    grand = float(resid.mean())
    mu_b1 = pd.Series(envs_te).map(known).fillna(grand).to_numpy()
    mu_ec = EnvMean(fr, resid, ec, esc)(envs_te)
    out["B1_gblup"] = mu_b1 + g_te
    out["B5_twostage"] = mu_ec + g_te
    out["B0_envmean"] = mu_ec
    return te, out, info, lam


def fit_b3(fold: Fold, geno: GenoData, ec: pd.DataFrame, k: int = 100, m: int = 10):
    """B3 EC reaction norm: all markers (main) + low-rank U_g ⊗ U_e (interaction), separate penalties.
    Spec 02 listed U_g (k=100) for the main effect too; that discards hybrid identity beyond 100 PCs
    and lost 0.11 validation Spearman to B1 on FYrep2023 (2026-09-26), so the main effect uses the
    full marker set, as in Jarquín et al. 2014 (G + G∘Ω)."""
    ratios = np.array([0.1, 1.0, 10.0, 100.0])       # λ_int / λ_main
    lams = np.logspace(-3, 1, 5)                       # 5 x 4 = 20 grid points

    def design(roles):
        fr = fold.rows(*roles)
        sc, pos, Z = _geno_feats(geno, fr, fold.cells["genotype"])
        tr_ids = fr["genotype"].unique()
        pca = GenoPCA.fit(Z[[pos[g] for g in tr_ids]], tr_ids, k)
        Ug = pca.transform(Z) / pca.scale  # unit-variance PCs
        esc = EnvScaler.fit(ec, fr["env"].unique())
        envs = sorted(fold.cells["env"].unique())
        Xe = esc.transform(ec, envs)
        tr_envs = sorted(fr["env"].unique())
        Xtr = Xe[[envs.index(e) for e in tr_envs]]
        mu = Xtr.mean(0)
        _, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
        Ue = (Xe - mu) @ Vt[:m].T / (S[:m] / np.sqrt(len(Xtr)))
        epos = {e: i for i, e in enumerate(envs)}

        def feats(rows, ratio):
            # main effect on all standardised markers (the full G kernel of Jarquín 2014);
            # interaction on the low-rank U_g ⊗ U_e (the G∘Ω kernel approximation of spec 02)
            ix = [pos[g] for g in rows["genotype"]]
            G = Ug[ix]
            E = Ue[[epos[e] for e in rows["env"]]]
            GE = (G[:, :, None] * E[:, None, :]).reshape(len(rows), -1) * np.sqrt(Z.shape[1] / (k * m * ratio))
            return np.c_[Z[ix], GE]
        return fr, feats

    fr, feats = design(fold.fit_roles(False))
    va = fold.rows("val")
    best = (-np.inf, None, None)
    for ratio in ratios:
        r = Ridge(feats(fr, ratio), fr["env"].to_numpy(), fr["y"].to_numpy())
        Fv = feats(va, ratio)
        for lam in lams:
            s = val_score(va, Fv @ r.coef(lam))
            if s > best[0]:
                best = (s, ratio, lam)
    _, ratio, lam = best
    if fold.forward:
        fr, feats = design(fold.fit_roles(True))
    r = Ridge(feats(fr, ratio), fr["env"].to_numpy(), fr["y"].to_numpy())
    beta = r.coef(lam)
    te = fold.rows("test")
    g_fr = feats(fr, ratio) @ beta
    known = pd.Series(fr["y"].to_numpy() - g_fr).groupby(fr["env"].to_numpy()).mean()
    mu = pd.Series(te["env"].to_numpy()).map(known).fillna(float((fr["y"].to_numpy() - g_fr).mean())).to_numpy()
    pred = mu + feats(te, ratio) @ beta
    return te, {"B3_ecrn": pred}, {"B3_ecrn": {"lambda_rel": lam, "ratio_int_main": ratio,
                                              "val_spearman": best[0], "grid": 20, "k": k, "m": m}}


def fit_b2(fold: Fold, geno: GenoData, ec: pd.DataFrame, lam_main: float, iters: int = 6):
    """B2 M×E (Lopez-Cruz 2015): main marker effect + environment-specific marker effects.
    Backfitting between the main ridge and per-environment kernel ridges (K = ZZ'/p).
    Grid: λ_main ∈ {¼, ½, 1, 2, 4}·(B1's choice) × λ_env ∈ 4 values = 20 points."""
    grid_main = lam_main * np.array([0.25, 0.5, 1, 2, 4])
    grid_env = np.array([0.1, 1.0, 10.0, 100.0])

    def fit(roles, lm, le):
        fr = fold.rows(*roles)
        sc, pos, Z = _geno_feats(geno, fr, fold.cells["genotype"])
        F = Z[[pos[g] for g in fr["genotype"]]]
        u, c = codes(fr["env"].to_numpy())
        y = center(fr["y"].to_numpy(np.float64), c, len(u))
        Fc = center(F.astype(np.float64), c, len(u))
        Ft = torch.tensor(Fc, device=DEV)
        G = Ft.T @ Ft
        d, V = torch.linalg.eigh(G)
        lam = lm * float(torch.trace(G)) / G.shape[0]
        p = F.shape[1]
        idx = [np.flatnonzero(c == j) for j in range(len(u))]
        Zt = torch.tensor(Z.astype(np.float64), device=DEV)
        rows_pos = np.array([pos[g] for g in fr["genotype"]])
        alphas = [None] * len(u)
        uhat = np.zeros_like(y)
        for _ in range(iters):
            b = V.T @ (Ft.T @ torch.tensor(y - uhat, device=DEV))
            beta = V @ (b / (d + lam))
            resid = y - (Ft @ beta).cpu().numpy()
            for j, ix in enumerate(idx):
                Zj = Zt[rows_pos[ix]]
                K = (Zj @ Zj.T) / p
                rj = torch.tensor(resid[ix] - resid[ix].mean(), device=DEV)
                a = torch.linalg.solve(K + le * torch.eye(len(ix), device=DEV, dtype=K.dtype), rj)
                alphas[j] = (rows_pos[ix], a)
                uhat[ix] = (K @ a).cpu().numpy()
        return fr, pos, Zt, beta, dict(zip(u, alphas)), p

    def predict(model, rows):
        fr, pos, Zt, beta, alphas, p = model
        rp = np.array([pos[g] for g in rows["genotype"]])
        g = (Zt[rp] @ beta).cpu().numpy()
        out = g.copy()
        envs = rows["env"].to_numpy()
        for e in np.unique(envs):
            if e not in alphas:
                continue  # new environment: no environment-specific effect
            ix = np.flatnonzero(envs == e)
            trp, a = alphas[e]
            out[ix] += ((Zt[rp[ix]] @ Zt[trp].T / p) @ a).cpu().numpy()
        return out

    va = fold.rows("val")
    best = (-np.inf, None, None)
    for lm in grid_main:
        for le in grid_env:
            s = val_score(va, predict(fit(fold.fit_roles(False), lm, le), va))
            if s > best[0]:
                best = (s, lm, le)
    _, lm, le = best
    model = fit(fold.fit_roles(True), lm, le)
    te = fold.rows("test")
    pred = predict(model, te)
    fr = model[0]
    resid = fr["y"].to_numpy() - predict(model, fr)
    known = pd.Series(resid).groupby(fr["env"].to_numpy()).mean()
    mu = pd.Series(te["env"].to_numpy()).map(known).fillna(float(resid.mean())).to_numpy()
    return te, {"B2_mxe": mu + pred}, {"B2_mxe": {"lambda_main_rel": lm, "lambda_env": le, "val_spearman": best[0], "grid": 20}}


def fit_b1z(fold: Fold, geno: GenoData, ec: pd.DataFrame):
    """Ablation L1: B1 with the target standardised within environment (each environment weighted
    equally), docs/prestudy_nn_vs_gblup_2026-09-26.md. Same 20-point λ grid."""
    def zfit(fr):
        g = fr.groupby("env")["y"]
        return ((fr["y"] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)).to_numpy()

    def fit(roles):
        fr = fold.rows(*roles)
        sc, pos, Z = _geno_feats(geno, fr, fold.cells["genotype"])
        r = Ridge(Z[[pos[g] for g in fr["genotype"]]], fr["env"].to_numpy(), zfit(fr))
        return fr, pos, Z, r

    fr, pos, Z, r = fit(fold.fit_roles(False))
    va = fold.rows("val")
    Fv = Z[[pos[g] for g in va["genotype"]]]
    scores = [val_score(va, Fv @ r.coef(l)) for l in LAMS]
    lam = float(LAMS[int(np.nanargmax(scores))])
    if fold.forward:
        fr, pos, Z, r = fit(fold.fit_roles(True))
    te = fold.rows("test")
    g_te = Z[[pos[g] for g in te["genotype"]]] @ r.coef(lam)
    return te, {"L1_gblup_z": g_te}, {"L1_gblup_z": {"lambda_rel": lam, "val_spearman": float(np.nanmax(scores)), "grid": len(LAMS)}}


def fit_b1_reml(fold: Fold, geno: GenoData, ec: pd.DataFrame):
    """B1 with the standard REML shrinkage (no validation-set tuning): fitted on the final fitting set
    (forward: train ∪ val ∪ refit_extra; CV: train). Gate 1 showed this beats validation-year tuning."""
    fr = fold.rows(*fold.fit_roles(True))
    _, pos, Z = _geno_feats(geno, fr, fold.cells["genotype"])
    r = Ridge(Z[[pos[g] for g in fr["genotype"]]], fr["env"].to_numpy(), fr["y"].to_numpy())
    rm = r.reml()
    beta = r.coef(rm["lambda_rel"])
    te = fold.rows("test")
    g_fr = Z[[pos[g] for g in fr["genotype"]]] @ beta
    known = pd.Series(fr["y"].to_numpy() - g_fr).groupby(fr["env"].to_numpy()).mean()
    mu = pd.Series(te["env"].to_numpy()).map(known).fillna(float((fr["y"].to_numpy() - g_fr).mean())).to_numpy()
    return te, {"B1r_gblup_reml": mu + Z[[pos[g] for g in te["genotype"]]] @ beta}, {"B1r_gblup_reml": rm}
