"""Checks for the SP headroom check (docs/prereg_headroom_sparse_2026-09-29.md §5)."""
import numpy as np
import pandas as pd

from dartgxe.forward import sparse
from dartgxe.forward.cellspec import Prep, fit_ia
from dartgxe.forward.data import Dataset


def toy(n_geno=90, years=(2001, 2002, 2003), n_env=6, p=50, seed=0, gxe=1.0):
    rng = np.random.default_rng(seed)
    ids = [f"g{i}" for i in range(n_geno)]
    M = pd.DataFrame(rng.integers(0, 3, (n_geno, p)).astype(float), index=pd.Index(ids, name="genotype"))
    beta = rng.normal(size=p)
    load = rng.normal(size=(n_env, 2))
    fac = rng.normal(size=(2, p))
    rows = []
    for y in years:
        lines = rng.choice(ids, 50, replace=False)
        for e in range(n_env):
            for g in lines:
                x = M.loc[g].to_numpy()
                gx = gxe * x @ (load[e] @ fac) * 0.3
                rows.append((f"E{e}_{y}", y, g, float(x @ beta + gx + rng.normal() + e)))
    return Dataset("TOY", pd.DataFrame(rows, columns=["env", "year", "genotype", "y"]), M, 25)


def target(ds, y):
    c = ds.cells[ds.cells["year"] == y]
    return c.reset_index(drop=True), sorted(set(c["env"]))


def test_mask_quota_balance_reproducible_and_phenotype_free():
    ds = toy()
    c, _ = target(ds, 2003)
    o1 = sparse.sparse_mask(c[["env", "genotype"]], 0.25, 0, "TOY", 2003)
    o2 = sparse.sparse_mask(c[["env", "genotype"]], 0.25, 0, "TOY", 2003)
    assert (o1 == o2).all()
    o3 = sparse.sparse_mask(c[["env", "genotype"]], 0.25, 1, "TOY", 2003)
    assert (o1 != o3).any()
    n = c.groupby("genotype").size()
    q = c[o1].groupby("genotype").size().reindex(n.index).fillna(0)
    expect = np.clip(np.floor(0.25 * n + 0.5), 1, n - 1)
    assert (q == expect).all()
    share = c.assign(o=o1).groupby("env")["o"].mean()
    assert share.max() - share.min() < 0.1
    # a genotype seen in one environment only is observed and never scored
    c1 = pd.concat([c, pd.DataFrame({"env": ["E0_2003"], "year": [2003], "genotype": ["solo"], "y": [0.0]})], ignore_index=True)
    o = sparse.sparse_mask(c1[["env", "genotype"]], 0.5, 0, "TOY", 2003)
    assert o[len(c1) - 1]


def test_scored_phenotypes_do_not_leak():
    ds = toy()
    _, envs = target(ds, 2003)
    out1, meta1 = sparse.sparse_year(ds, 2003, envs, 0.25, 0)
    c = ds.cells.copy()
    obs = sparse.sparse_mask(c[c["year"] == 2003][["env", "genotype"]].reset_index(drop=True), 0.25, 0, "TOY", 2003)
    idx = c.index[c["year"] == 2003][~obs]
    c.loc[idx, "y"] = 1e6 * np.random.default_rng(9).normal(size=len(idx))
    out2, meta2 = sparse.sparse_year(Dataset("TOY", c, ds.markers, 25), 2003, envs, 0.25, 0)
    for m in ("m0", "m1", "m2"):
        np.testing.assert_array_equal(out1[m].to_numpy(), out2[m].to_numpy())
    assert (meta1["omega"], meta1["c2"], meta1["rho"]) == (meta2["omega"], meta2["c2"], meta2["rho"])


def test_assembly_equals_explicit_features_and_nesting():
    ds = toy(n_geno=60, n_env=4, p=30, seed=2)
    c, envs = target(ds, 2003)
    obs = sparse.sparse_mask(c[["env", "genotype"]], 0.5, 0, "TOY", 2003)
    train = pd.concat([ds.cells[ds.cells["year"] < 2003], c[obs]], ignore_index=True)
    P = Prep(train, ds.markers, c[~obs])
    T0 = sparse.main_block(P, 0.1)
    k = 10
    order = np.argsort(-P.lam)[:k]
    Pk = P.U[:, order] * np.sqrt(P.lam[order])
    tenvs = sorted(set(c.loc[obs, "env"]))
    A00, r0, yy = P.pieces(T0, np.ones(P.n_env), P.yc)
    A, C, R = sparse.gxe_pieces(P, T0, Pk, tenvs)
    rng = np.random.default_rng(0)
    Rh = np.corrcoef(rng.normal(size=(20, len(tenvs))).T)
    for L in (np.eye(len(tenvs)), sparse.env_factor(Rh, 0.6)):
        c2 = 0.7
        Afull, rfull = sparse.assemble(A00, r0, A, C, R, L, c2)
        # explicit design: env-centred [T0 | sqrt(c2) * kron(Pk_i, L_j)] on the training cells
        code = {e: i for i, e in enumerate(tenvs)}
        rows = []
        for r in range(len(P.train)):
            e = P.envs[P.ej[r]]
            g = np.zeros(k * len(tenvs)) if e not in code else np.sqrt(c2) * np.kron(Pk[P.gi[r]], L[code[e]])
            rows.append(np.r_[T0[P.gi[r]], g])
        F = np.array(rows)
        mu = pd.DataFrame(F).groupby(P.ej).transform("mean").to_numpy()
        Fc = F - mu
        np.testing.assert_allclose(Afull, Fc.T @ Fc, rtol=1e-8, atol=1e-8 * np.abs(Afull).max())
        np.testing.assert_allclose(rfull, Fc.T @ P.yc, rtol=1e-8, atol=1e-8 * np.abs(rfull).max())
    # M1 with c2 = 0 is M0 (= fit_ia at the same omega)
    f0 = sparse.fit_gxe(P, T0, Pk, tenvs, (A, C, R), (A00, r0, yy), np.eye(len(tenvs)), 0.0)
    ia = fit_ia(P, np.ones(P.n_env), P.yc, 0.1)
    np.testing.assert_allclose(T0 @ f0["a0"], ia["u_f"], rtol=1e-6, atol=1e-8 * np.abs(ia["u_f"]).max())
    # E_rho with rho -> 0 is the identity, so M2 at rho = 0 is M1
    np.testing.assert_allclose(sparse.env_factor(Rh, 0.0) @ sparse.env_factor(Rh, 0.0).T, np.eye(len(tenvs)), atol=1e-10)


def test_sparse_year_trains_only_on_history_and_observed_cells(monkeypatch):
    ds = toy()
    _, envs = target(ds, 2003)
    seen = {}
    real = sparse.Prep

    class Spy(real):
        def __init__(self, train, markers, test):
            seen["train_years"], seen["test_years"] = set(train["year"]), set(test["year"])
            seen["train_2003"] = train[train["year"] == 2003][["env", "genotype"]].apply(tuple, axis=1).tolist()
            seen["test"] = test[["env", "genotype"]].apply(tuple, axis=1).tolist()
            super().__init__(train, markers, test)

    monkeypatch.setattr(sparse, "Prep", Spy)
    out, meta = sparse.sparse_year(ds, 2003, envs, 0.25, 0)
    assert seen["train_years"] == {2001, 2002, 2003} and seen["test_years"] == {2003}
    assert not set(seen["train_2003"]) & set(seen["test"])
    assert len(out) == meta["n_scored"] and meta["envs_without_observed"] == 0


def test_learned_correlation_is_picked_up_on_toy_data():
    ds = toy(gxe=3.0, seed=5)
    _, envs = target(ds, 2003)
    _, meta = sparse.sparse_year(ds, 2003, envs, 0.25, 0)
    assert meta["c2"] > 0 and meta["rho"] >= 0.25
