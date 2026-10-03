"""Headroom checks I-A and I-B (docs/prereg_headroom_ia_ib_2026-09-29.md §2).

Both are cell-level genomic models with environment fixed effects (absorbed by centring within environment, as
cell_reml), solved at the genotype level:

I-A  u ~ N(0, s2u K(w)), K(w) = (1 - w) K_A + w I; residual variance of environment j = s2e * s_j^2.
     w = 0 and s = 1 is cell_reml. w and the overall shrinkage are chosen by profile REML on training data.
I-B  u_il = a_i + b_il, a ~ N(0, s2G K_A), b_.l ~ N(0, s2GL K_A^(k)) independently per active location l;
     c2 = s2GL / s2G by profile REML. c2 = 0 is cell_reml.

K_A = Z Z' / p with Z the markers standardised on the training genotypes (panel.features), divided by its mean
diagonal; K_A = U diag(lam) U'. Features for the main effect are D_c U S (D_c = environment-centred cell x genotype
incidence), so ridge on them with one penalty is GBLUP with kernel U S^2 U'. All normal-equation pieces are built
from the sparse environment x genotype incidence, never from the dense cell x genotype matrix.

Profile REML is the same criterion as dartgxe.baselines.linear.Ridge.reml (restricted to the within-environment
contrasts): -2l(delta) = n' log Q(delta) + sum log(1 + delta d_k); here the minimised value is also returned so
that models with different covariance structures (w, c2) fitted to the same data can be compared."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch

from dartgxe.forward.panel import features

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
C2_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
K_GL = 100
MIN_LOC_YEARS, MIN_LOC_CELLS = 2, 50
LOG_DELTA = np.linspace(np.log(1e-8), np.log(1e8), 600)


# ------------------------------------------------------------------------------------------------ locations
def location(dataset: str, env: str, level: str = "location"):
    """Prereg §2.2 mapping from the environment name only. None = no location that repeats across years."""
    base = str(env).rsplit("_", 1)[0]
    if dataset == "G2F":
        m = re.match(r"^[A-Z]{2}[HS]\d", base)
        loc = m.group(0) if m else base
        return loc[:2] if level == "region" else loc
    if dataset == "NUST":
        name, state = base.rsplit("_", 1)
        return state if level == "region" else re.sub(r"[^A-Za-z]", "", name) + "_" + state
    if dataset == "URSN":
        return None if level == "region" else base
    if dataset == "ESWYT":
        if level == "region":
            return "South Africa" if base.startswith("South Africa") else base.split(" ")[0]
        return base
    return None


# ------------------------------------------------------------------------------------------------ REML
def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64), device=DEV)


def reml_solve(A: np.ndarray, rhs: np.ndarray, yy: float, n_resid: int) -> dict:
    """Ridge with normal matrix A = F'F, F'y = rhs, y'y = yy on n_resid within-environment contrasts.
    Returns delta (= prior variance / residual variance), the coefficients at the REML optimum and -2l."""
    At = _t(A)
    d, V = torch.linalg.eigh((At + At.T) / 2)
    b = V.T @ _t(rhs)
    d, b, V = d.cpu().numpy(), b.cpu().numpy(), V
    keep = d > 1e-10 * max(d.max(), 1e-300)
    dk, bk = d[keep], b[keep]
    c2 = bk ** 2 / dk
    rest = max(yy - c2.sum(), 1e-12 * max(yy, 1e-300))

    def f(ld):
        dl = np.exp(ld)
        return n_resid * np.log((c2 / (1 + dl * dk)).sum() + rest) + np.log1p(dl * dk).sum()

    vals = np.array([f(g) for g in LOG_DELTA])
    i = int(vals.argmin())
    lo, hi = LOG_DELTA[max(i - 1, 0)], LOG_DELTA[min(i + 1, len(LOG_DELTA) - 1)]
    for _ in range(80):
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    ld = (lo + hi) / 2
    delta = float(np.exp(ld))
    lam = 1.0 / delta
    coef = (V[:, _t(keep).bool()] @ _t(bk / (dk + lam))).cpu().numpy()
    return {"delta": delta, "m2ll": float(f(ld)), "coef": coef, "at_edge": i in (0, len(LOG_DELTA) - 1),
            "rank": int(keep.sum())}


# ------------------------------------------------------------------------------------------------ data prep
class Prep:
    """Training cells, kernel and incidence for one forward target (all fitted on training data only)."""

    def __init__(self, train: pd.DataFrame, markers: pd.DataFrame, test: pd.DataFrame):
        train = train[train["genotype"].isin(markers.index)].reset_index(drop=True)
        test = test[test["genotype"].isin(markers.index)].reset_index(drop=True)
        self.train, self.test = train, test
        self.fit_ids = sorted(set(train["genotype"]))
        self.test_ids = sorted(set(test["genotype"]))
        Zf, Zt, _, _ = features(markers, self.fit_ids, self.test_ids)
        p = Zf.shape[1]
        Zf_t, Zt_t = _t(Zf), _t(Zt)
        Kff = (Zf_t @ Zf_t.T) / p
        c0 = float(torch.diagonal(Kff).mean())
        Kff = Kff / c0
        self.Ktf = ((Zt_t @ Zf_t.T) / p / c0).cpu().numpy()
        lam, U = torch.linalg.eigh((Kff + Kff.T) / 2)
        self.lam = np.clip(lam.cpu().numpy(), 0, None)
        self.U = U.cpu().numpy()
        self.n_markers, self.c0 = p, c0
        pos = {g: i for i, g in enumerate(self.fit_ids)}
        self.gi = train["genotype"].map(pos).to_numpy()
        self.envs, self.ej = np.unique(train["env"].to_numpy(), return_inverse=True)
        self.n_env = len(self.envs)
        self.nj = np.bincount(self.ej, minlength=self.n_env).astype(float)
        y = train["y"].to_numpy(np.float64)
        self.yc = y - (np.bincount(self.ej, y, self.n_env) / self.nj)[self.ej]
        B = np.zeros((self.n_env, len(self.fit_ids)))
        B[self.ej, self.gi] = 1.0
        self.B = _t(B)
        self.n_resid = len(train) - self.n_env
        tpos = {g: i for i, g in enumerate(self.fit_ids)}
        self.seen = np.array([g in tpos for g in self.test_ids])
        self.seen_idx = np.array([tpos.get(g, -1) for g in self.test_ids])

    # normal-equation pieces for weights w_j (per environment) and centred response yc (per cell)
    def cross(self, T1: np.ndarray, T2: np.ndarray, w: np.ndarray) -> np.ndarray:
        """T1' M_w T2 with M_w = sum_j w_j [diag(d_j) - d_j d_j' / n_j] (d_j = genotypes of environment j)."""
        a, b, wt = _t(T1), _t(T2), _t(w)
        dw = self.B.T @ wt
        return (a.T @ (dw[:, None] * b) - (self.B @ a).T @ ((wt / _t(self.nj))[:, None] * (self.B @ b))).cpu().numpy()

    def rvec(self, w: np.ndarray, yc: np.ndarray) -> np.ndarray:
        """D' W yc: per training genotype, the weighted sum of its centred cells."""
        return np.bincount(self.gi, w[self.ej] * yc, minlength=len(self.fit_ids))

    def pieces(self, T: np.ndarray, w: np.ndarray, yc: np.ndarray):
        """A = T' M_w T, rhs = T' D' W yc, y'Wy."""
        return self.cross(T, T, w), T.T @ self.rvec(w, yc), float((w[self.ej] * yc * yc).sum())


# ------------------------------------------------------------------------------------------------ I-A
def fit_ia(P: Prep, w_env: np.ndarray, yc: np.ndarray, omega: float) -> dict:
    s = np.sqrt((1 - omega) * P.lam + omega)
    keep = s > 1e-8 * s.max()
    T = P.U[:, keep] * s[keep]
    A, rhs, yy = P.pieces(T, w_env, yc)
    r = reml_solve(A, rhs, yy, P.n_resid)
    a = r["coef"]
    u_f = T @ a
    Uinv = P.U[:, keep] @ (a / s[keep])                        # U S^-1 a
    u_t = (1 - omega) * (P.Ktf @ Uinv)
    if omega > 0:
        u_t = u_t + omega * np.where(P.seen, Uinv[np.clip(P.seen_idx, 0, None)], 0.0)
    return {"u_f": u_f, "u_t": u_t, "m2ll": r["m2ll"], "delta": r["delta"], "at_edge": r["at_edge"]}


def env_scales(P: Prep, u_f: np.ndarray) -> np.ndarray:
    """Prereg §2.1: one-step relative residual SD per training environment from the control fit."""
    fit = u_f[P.gi]
    fit_c = fit - (np.bincount(P.ej, fit, P.n_env) / P.nj)[P.ej]
    rss = np.bincount(P.ej, (P.yc - fit_c) ** 2, P.n_env)
    sig = np.sqrt(rss / np.maximum(P.nj - 1, 1))
    s = sig / np.median(sig[P.nj >= 3])
    lo, hi = np.percentile(s[P.nj >= 3], [5, 95])
    s = np.clip(s, lo, hi)
    s[P.nj < 3] = 1.0
    return s


def ia_family(P: Prep) -> tuple[dict, dict]:
    """Control, ia_main, ia_g, ia_het, ia_std and the weighted omega grid. Returns (test predictions, meta)."""
    ones = np.ones(P.n_env)
    ctrl = fit_ia(P, ones, P.yc, 0.0)
    s = env_scales(P, ctrl["u_f"])
    w_het = 1.0 / s ** 2
    grid_w = {om: fit_ia(P, w_het, P.yc, om) for om in W_GRID}
    grid_u = {om: (ctrl if om == 0 else fit_ia(P, ones, P.yc, om)) for om in W_GRID}
    om_main = min(W_GRID, key=lambda om: grid_w[om]["m2ll"])
    om_g = min(W_GRID, key=lambda om: grid_u[om]["m2ll"])
    sd = np.sqrt(np.bincount(P.ej, P.yc ** 2, P.n_env) / np.maximum(P.nj - 1, 1))
    ystd = np.where(sd[P.ej] > 0, P.yc / np.where(sd > 0, sd, 1.0)[P.ej], 0.0)
    std = fit_ia(P, ones, ystd, 0.0)
    preds = {"ctrl": ctrl["u_t"], "ia_main": grid_w[om_main]["u_t"], "ia_g": grid_u[om_g]["u_t"],
             "ia_het": grid_w[0.0]["u_t"], "ia_std": std["u_t"]}
    for om in W_GRID:
        preds[f"iaw_{om:g}"] = grid_w[om]["u_t"]
    meta = {"omega_main": om_main, "omega_g": om_g, "s_min": float(s.min()), "s_max": float(s.max()),
            "m2ll_w": {f"{om:g}": grid_w[om]["m2ll"] for om in W_GRID},
            "m2ll_u": {f"{om:g}": grid_u[om]["m2ll"] for om in W_GRID},
            "ctrl_delta": ctrl["delta"], "ctrl_at_edge": ctrl["at_edge"],
            "any_at_edge": bool(any(g["at_edge"] for g in list(grid_w.values()) + list(grid_u.values())))}
    return preds, meta


# ------------------------------------------------------------------------------------------------ I-B
def ib_family(P: Prep, dataset: str, level: str, grid=C2_GRID) -> tuple[dict, dict]:
    """G x location model for every c2 in grid (c2 = 0 is the control). Returns predictions for the REML choice
    ('main') and every grid point, plus meta (active locations, chosen c2)."""
    keep0 = P.lam > 1e-8 * P.lam.max()
    T0 = P.U[:, keep0] * np.sqrt(P.lam[keep0])
    k = int(min(K_GL, len(P.fit_ids) - 1, keep0.sum()))
    order = np.argsort(-P.lam)[:k]
    Pk = P.U[:, order] * np.sqrt(P.lam[order])                          # training genotypes, K_A^(k) = Pk Pk'
    Pt = P.Ktf @ (P.U[:, order] / np.sqrt(P.lam[order]))                # test genotypes (Nystrom)
    env_loc = np.array([location(dataset, e, level) for e in P.envs], dtype=object)
    yrs = P.train.groupby("env")["year"].first().reindex(P.envs).to_numpy()
    loc_df = pd.DataFrame({"loc": env_loc, "year": yrs, "n": P.nj}).dropna(subset=["loc"])
    agg = loc_df.groupby("loc").agg(years=("year", "nunique"), cells=("n", "sum"))
    active = sorted(agg.index[(agg["years"] >= MIN_LOC_YEARS) & (agg["cells"] >= MIN_LOC_CELLS)])
    ones = np.ones(P.n_env)
    A00, r0, yy = P.pieces(T0, ones, P.yc)
    m0, L = T0.shape[1], len(active)
    dim = m0 + k * L
    A1 = np.zeros((dim, dim))
    rhs1 = np.zeros(dim)
    A1[:m0, :m0], rhs1[:m0] = A00, r0
    for li, loc in enumerate(active):
        wm = np.where(env_loc == loc, 1.0, 0.0)            # environments are nested in locations
        sl = slice(m0 + li * k, m0 + (li + 1) * k)
        A1[:m0, sl] = P.cross(T0, Pk, wm)
        A1[sl, :m0] = A1[:m0, sl].T
        A1[sl, sl] = P.cross(Pk, Pk, wm)
        rhs1[sl] = Pk.T @ P.rvec(wm, P.yc)
    fits = {}
    for c2 in grid:
        if c2 == 0 or L == 0:
            r = reml_solve(A00, r0, yy, P.n_resid)
            a0, al = r["coef"], np.zeros((L, k))
        else:
            cs = np.r_[np.ones(m0), np.full(k * L, np.sqrt(c2))]
            r = reml_solve(cs[:, None] * A1 * cs[None, :], cs * rhs1, yy, P.n_resid)
            a = r["coef"] * cs
            a0, al = a[:m0], a[m0:].reshape(L, k)
        fits[c2] = {"a0": a0, "al": al, "m2ll": r["m2ll"], "delta": r["delta"], "at_edge": r["at_edge"]}
    c2_main = min(grid, key=lambda c: fits[c]["m2ll"])
    main_t = P.Ktf @ (P.U[:, keep0] / np.sqrt(P.lam[keep0]))            # test genotypes x m0 (maps a0 -> u)
    # predictions per test cell (location of the target environment decides the G x L part)
    tg = {g: i for i, g in enumerate(P.test_ids)}
    ti = P.test["genotype"].map(tg).to_numpy()
    tloc = np.array([location(dataset, e, level) for e in P.test["env"]], dtype=object)
    lidx = {loc: i for i, loc in enumerate(active)}
    tli = np.array([lidx.get(l, -1) for l in tloc])
    preds, mains = {}, {}
    for c2, f in fits.items():
        u = (main_t @ f["a0"])[ti]
        mains[c2] = u
        if L and c2 > 0:
            gl = np.einsum("ck,ck->c", Pt[ti], f["al"][np.clip(tli, 0, None)])
            u = u + np.where(tli >= 0, gl, 0.0)
        preds[c2] = u
    meta = {"level": level, "k": k, "n_active": L, "active": active, "c2_main": c2_main,
            "m2ll": {f"{c:g}": fits[c]["m2ll"] for c in grid}, "any_at_edge": bool(any(f["at_edge"] for f in fits.values())),
            "test_env_active": sorted(set(P.test["env"][tli >= 0]))}
    return {"main": preds[c2_main], "grid": preds, "grid_main_only": mains, "active_cell": tli >= 0}, meta


# ------------------------------------------------------------------------------------------------ one target year
def panel_year(ds, year: int, envs) -> tuple[pd.DataFrame, dict]:
    """All I-A and I-B variants for the scored cells of `year`, trained on the years before it only."""
    train = ds.cells[ds.cells["year"] < year]
    test = ds.cells[(ds.cells["year"] == year) & ds.cells["env"].isin(list(envs))]
    assert train["year"].max() < year and set(test["year"]) == {year}
    P = Prep(train, ds.markers, test)
    out = P.test[["env", "year", "genotype", "y"]].copy()
    ti = out["genotype"].map({g: i for i, g in enumerate(P.test_ids)}).to_numpy()
    ia, meta_ia = ia_family(P)
    for k, v in ia.items():
        out[k] = v[ti]
    metas = {"n_fit": len(P.fit_ids), "n_train_cells": len(P.train), "n_markers": P.n_markers, "ia": meta_ia}
    for level, name in (("location", "ib_main"), ("region", "ib_region")):
        if location(ds.name, str(out["env"].iloc[0]), level) is None:
            continue
        ib, meta_ib = ib_family(P, ds.name, level)
        out[name] = ib["main"]
        if level == "location":
            out["ib_active"] = ib["active_cell"]
            for c2, v in ib["grid"].items():
                if c2 > 0:
                    out[f"ibc_{c2:g}"] = v
        metas[name] = meta_ib
    return out, metas
