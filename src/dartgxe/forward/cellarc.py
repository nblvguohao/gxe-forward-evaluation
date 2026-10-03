"""cell_arc: cell_reml with an arc-cosine kernel (docs/prereg_transfer_arc_kernel_2026-10-02.md §2).

The only change from cell_reml is the genomic kernel. Standardised markers Z come from panel.features (constants from
the training genotypes only); G = Z Z' / c with c the mean diagonal over the training genotypes; K is the order-1
arc-cosine kernel of G (Cho and Saul 2009; bWGR::EigenARC). The kernel enters the ridge of baselines.linear.Ridge
through the embedding Φ_f = U Λ^½ (K_ff = U Λ U'), Φ_t = K_tf U Λ^-½, so the fit (environment fixed effects, REML) is the
one cell_reml uses. `kernel="linear"` passes G itself through the same embedding (positive control: equals cell_reml)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from dartgxe.baselines.linear import DEV, Ridge, center, codes
from dartgxe.forward.panel import features

MIN_EV = 1e-8  # eigenvalues kept: above MIN_EV times the largest (the threshold of CLAC's K2X)


class AggRidge(Ridge):
    """The estimator of Ridge(Phi[gi], env, y), computed without the cells x features matrix.

    With N the genotype x environment count matrix of the training cells, the within-environment-centred cross-products are
    F_c'F_c = Phi'(D_g - N D_e^-1 N')Phi and F_c'y_c = Phi's, s_g = sum of the centred phenotypes of genotype g. REML and
    the coefficients then come from the methods of Ridge unchanged. Memory is O(n_genotypes^2) instead of
    O(n_cells x n_genotypes); tests check equality with the dense computation."""

    def __init__(self, Phi: np.ndarray, gi: np.ndarray, env: np.ndarray, y: np.ndarray):
        u, c = codes(env)
        yc = center(y.astype(np.float64), c, len(u))
        N = sp.csr_matrix((np.ones(len(gi)), (gi, c)), shape=(Phi.shape[0], len(u)))
        cnt_g, cnt_e = np.asarray(N.sum(1)).ravel(), np.asarray(N.sum(0)).ravel()
        NP = np.asarray(N.T @ Phi)  # environment sums of the features
        G = Phi.T @ (cnt_g[:, None] * Phi) - NP.T @ (NP / cnt_e[:, None])
        G = torch.tensor((G + G.T) / 2, device=DEV)
        self.d, self.V = torch.linalg.eigh(G)
        self.b = self.V.T @ torch.tensor(Phi.T @ np.bincount(gi, weights=yc, minlength=Phi.shape[0]), device=DEV)
        self.scale = float(torch.trace(G)) / G.shape[0]
        self.yy = float((yc * yc).sum())
        self.n_resid = len(y) - len(u)


def arc_cosine(G: np.ndarray, na: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """Order-1 arc-cosine kernel from inner products G (a x b) and the norms na (a), nb (b) of the two sides."""
    nn = np.outer(na, nb)
    c = np.divide(G, nn, out=np.zeros_like(G, dtype=np.float64), where=nn > 0)
    np.clip(c, -1.0, 1.0, out=c)
    th = np.arccos(c)
    return nn / np.pi * (np.sin(th) + (np.pi - th) * c)


def embed(Kff: np.ndarray, Ktf: np.ndarray):
    """Features whose inner products reproduce the kernel: Φ_f Φ_f' = K_ff (up to dropped eigenvalues), Φ_t Φ_f' = K_tf."""
    d, U = np.linalg.eigh((Kff + Kff.T) / 2)
    keep = d > MIN_EV * d.max()
    d, U = d[keep], U[:, keep]
    return U * np.sqrt(d), (Ktf @ U) / np.sqrt(d), int(keep.sum())


def kernel_features(markers: pd.DataFrame, fit_ids, test_ids, kernel: str = "arc"):
    Zf, Zt, _, _ = features(markers, fit_ids, test_ids)
    c = float((Zf ** 2).sum(1).mean())
    Gff, Gtf = Zf @ Zf.T / c, Zt @ Zf.T / c
    if kernel == "arc":
        nf, nt = np.sqrt(np.diag(Gff)), np.sqrt((Zt ** 2).sum(1) / c)
        Kff, Ktf = arc_cosine(Gff, nf, nf), arc_cosine(Gtf, nt, nf)
        Kff[np.diag_indices_from(Kff)] = nf ** 2  # exact diagonal (θ = 0)
    elif kernel == "linear":
        Kff, Ktf = Gff, Gtf
    else:
        raise ValueError(kernel)
    Pf, Pt, rank = embed(Kff, Ktf)
    pos = {g: i for i, g in enumerate(fit_ids)}
    for j, g in enumerate(test_ids):  # a target genotype that was also trained on gets its training row
        if g in pos:
            Pt[j] = Pf[pos[g]]
    return Pf, Pt, {"gram_scale": c, "rank": rank, "n_fit": len(fit_ids), "n_test": len(test_ids)}


def fit_predict_arc(train: pd.DataFrame, markers: pd.DataFrame, test: pd.DataFrame, kernel: str = "arc"):
    """Predict every cell of `test` (env, genotype) from the `train` cells. Returns (prediction array aligned with
    test rows, meta). meta["fit_ids"] lists the genotypes every constant was computed on."""
    train = train[train["genotype"].isin(markers.index)]
    fit_ids, test_ids = sorted(set(train["genotype"])), sorted(set(test["genotype"]))
    Pf, Pt, meta = kernel_features(markers, fit_ids, test_ids, kernel)
    fpos, tpos = {h: i for i, h in enumerate(fit_ids)}, {h: i for i, h in enumerate(test_ids)}
    fi = np.array([fpos[h] for h in train["genotype"]])
    ti = np.array([tpos[h] for h in test["genotype"]])
    r = AggRidge(Pf, fi, train["env"].to_numpy(), train["y"].to_numpy(np.float64))
    rm = r.reml()
    pred = Pt[ti] @ r.coef(rm["lambda_rel"])
    meta.update(fit_ids=fit_ids, n_train_cells=len(train), lambda_rel=rm["lambda_rel"], h2=rm["h2"],
                at_grid_edge=rm["at_grid_edge"], kernel=kernel)
    return pred, meta


def panel_year(ds, year: int, envs):
    """cell_arc and the refitted control (cell_reml) for the scored cells of `year`, trained on earlier years only."""
    from dartgxe.forward.cellgxe import fit_predict_cell
    train = ds.cells[ds.cells["year"] < year]
    test = ds.cells[(ds.cells["year"] == year) & ds.cells["env"].isin(list(envs))].reset_index(drop=True)
    test = test[test["genotype"].isin(ds.markers.index)].reset_index(drop=True)
    assert train["year"].max() < year and set(test["year"]) == {year}
    out = test[["env", "year", "genotype", "y"]].copy()
    ctrl, mc = fit_predict_cell(train, ds.markers, test)
    out["ctrl"] = ctrl["cell_reml"].to_numpy()
    out["cell_arc"], ma = fit_predict_arc(train, ds.markers, test, "arc")
    assert ma["fit_ids"] == mc["fit_ids"]
    return out, {"ctrl": {k: v for k, v in mc.items() if k != "fit_ids"}, "arc": {k: v for k, v in ma.items() if k != "fit_ids"}}
