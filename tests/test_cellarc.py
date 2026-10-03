"""Checks of docs/prereg_transfer_arc_kernel_2026-10-02.md §5, on synthetic data (no target year of the benchmark)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dartgxe.baselines.linear import Ridge
from dartgxe.forward.cellarc import AggRidge, arc_cosine, embed, fit_predict_arc, kernel_features
from dartgxe.forward.cellgxe import fit_predict_cell

# reference: bWGR 2.2.18, EigenARC(X, TRUE, 1) on the 6 x 10 matrix below (4090, 2026-10-02)
X_REF = np.array([[0, 2, 0, 2, 0, 1, 1, 2, 1, 1], [2, 1, 0, 0, 1, 0, 1, 1, 2, 0], [0, 1, 1, 2, 0, 2, 1, 0, 2, 1],
                  [1, 2, 1, 0, 0, 0, 2, 0, 1, 1], [0, 2, 1, 0, 1, 2, 1, 2, 1, 1], [2, 0, 1, 0, 1, 1, 0, 1, 1, 1]], float)
K_REF = np.array([[1.000003, 0.118035, 0.461897, 0.172762, 0.398180, 0.072115],
                  [0.118035, 1.000003, 0.090957, 0.370986, 0.109642, 0.556578],
                  [0.461897, 0.090957, 1.140355, 0.221967, 0.240383, 0.187336],
                  [0.172762, 0.370986, 0.221967, 0.929828, 0.165560, 0.197213],
                  [0.398180, 0.109642, 0.240383, 0.165560, 0.859652, 0.215166],
                  [0.072115, 0.556578, 0.187336, 0.197213, 0.215166, 1.070179]])


def toy(seed=0, n_geno=120, p=300, years=(1, 2, 3, 4), envs_per_year=4):
    rng = np.random.default_rng(seed)
    M = rng.integers(0, 3, (n_geno, p)).astype(float)
    ids = [f"g{i:03d}" for i in range(n_geno)]
    markers = pd.DataFrame(M, index=ids)
    b = rng.normal(0, 1, p) / np.sqrt(p)
    g = (M - M.mean(0)) @ b + 0.5 * np.tanh((M - 1)[:, :20].sum(1))
    rows = []
    for yr in years:
        pool = ids[: 80] if yr < years[-1] else ids[40:]  # the last year has 40 new genotypes
        for e in range(envs_per_year):
            for i in rng.choice(len(pool), 50, replace=False):
                h = pool[i]
                rows.append((f"E{yr}_{e}", yr, h, g[ids.index(h)] + rng.normal(0, 1) + e))
    return pd.DataFrame(rows, columns=["env", "year", "genotype", "y"]), markers


def test_kernel_matches_bwgr_reference():
    Z = X_REF - X_REF.mean(0)
    G = Z @ Z.T
    G = G / np.diag(G).mean()
    n = np.sqrt(np.diag(G))
    assert np.abs(arc_cosine(G, n, n) - K_REF).max() < 2e-4


def test_embedding_reproduces_kernel():
    cells, markers = toy()
    fit_ids, test_ids = sorted(set(cells[cells.year < 4].genotype)), sorted(set(cells[cells.year == 4].genotype))
    Pf, Pt, meta = kernel_features(markers, fit_ids, test_ids, "arc")
    from dartgxe.forward.panel import features
    Zf, Zt, _, _ = features(markers, fit_ids, test_ids)
    c = (Zf ** 2).sum(1).mean()
    nf, nt = np.sqrt((Zf ** 2).sum(1) / c), np.sqrt((Zt ** 2).sum(1) / c)
    Kff, Ktf = arc_cosine(Zf @ Zf.T / c, nf, nf), arc_cosine(Zt @ Zf.T / c, nt, nf)
    assert np.abs(Pf @ Pf.T - Kff).max() < 1e-6
    assert np.abs(Pt @ Pf.T - Ktf).max() < 1e-6
    assert meta["n_fit"] == len(fit_ids)


def test_linear_kernel_through_embedding_equals_cell_reml():
    cells, markers = toy(1)
    train, test = cells[cells.year < 4], cells[cells.year == 4].reset_index(drop=True)
    ctrl, _ = fit_predict_cell(train, markers, test)
    lin, meta = fit_predict_arc(train, markers, test, "linear")
    assert np.corrcoef(ctrl["cell_reml"], lin)[0, 1] > 1 - 1e-6
    for _, g in test.assign(a=ctrl["cell_reml"].to_numpy(), b=lin).groupby("env"):
        assert spearmanr(g["a"], g["b"])[0] > 1 - 1e-6


def test_no_target_year_phenotype_enters():
    cells, markers = toy(2)
    train, test = cells[cells.year < 4], cells[cells.year == 4].reset_index(drop=True)
    p1, m1 = fit_predict_arc(train, markers, test)
    junk = test.assign(y=np.random.default_rng(9).normal(1e6, 1e6, len(test)))
    p2, m2 = fit_predict_arc(train, markers, junk)
    assert np.array_equal(p1, p2)
    assert set(m1["fit_ids"]) == set(train["genotype"]) and m1["gram_scale"] == m2["gram_scale"]


def test_arc_differs_from_linear():
    cells, markers = toy(3)
    train, test = cells[cells.year < 4], cells[cells.year == 4].reset_index(drop=True)
    a, _ = fit_predict_arc(train, markers, test, "arc")
    b, _ = fit_predict_arc(train, markers, test, "linear")
    assert np.corrcoef(a, b)[0, 1] < 1 - 1e-6


def test_aggregated_ridge_equals_dense_ridge():
    cells, markers = toy(4)
    train = cells[cells.year < 4]
    fit_ids = sorted(set(train["genotype"]))
    Pf, _, _ = kernel_features(markers, fit_ids, fit_ids, "arc")
    gi = np.array([fit_ids.index(h) for h in train["genotype"]])
    env, y = train["env"].to_numpy(), train["y"].to_numpy(float)
    dense, agg = Ridge(Pf[gi], env, y), AggRidge(Pf, gi, env, y)
    rd, ra = dense.reml(), agg.reml()
    assert abs(rd["lambda_rel"] - ra["lambda_rel"]) < 1e-6 * rd["lambda_rel"]
    assert np.abs(dense.coef(rd["lambda_rel"]) - agg.coef(ra["lambda_rel"])).max() < 1e-7
    assert dense.n_resid == agg.n_resid and abs(dense.yy - agg.yy) < 1e-9
