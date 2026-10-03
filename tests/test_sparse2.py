"""Checks for SP2 (docs/prereg_headroom_sparse2_2026-09-29.md §5)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dartgxe.forward import sparse, sparse2
from dartgxe.forward.data import Dataset


def toy(n_geno=80, years=(2001, 2002, 2003), n_env=6, p=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"g{i}" for i in range(n_geno)]
    M = pd.DataFrame(rng.integers(0, 3, (n_geno, p)).astype(float), index=pd.Index(ids, name="genotype"))
    beta, fac, load = rng.normal(size=p), rng.normal(size=(2, p)), rng.normal(size=(n_env, 2))
    rows = []
    for y in years:
        for e in range(n_env):
            for g in rng.choice(ids, 45, replace=False) if y < 2003 else ids[:50]:
                x = M.loc[g].to_numpy()
                rows.append((f"L{e}_{y}", y, g, float(x @ beta + 0.4 * x @ (load[e] @ fac) + rng.normal() + e)))
    cells = pd.DataFrame(rows, columns=["env", "year", "genotype", "y"])
    ec = pd.DataFrame(rng.normal(size=(len(set(cells["env"])), 8)), index=sorted(set(cells["env"])))
    ec.index.name = "env"
    return Dataset("TOY", cells, M, 10), ec


def envs_of(ds, y):
    return sorted(set(ds.cells.loc[ds.cells["year"] == y, "env"]))


def test_folds_reproducible_stratified():
    ds, _ = toy()
    c = ds.cells[ds.cells["year"] == 2003].reset_index(drop=True)
    f1 = sparse2.cv_folds(c, "TOY", 2003, 0.25, 0)
    f2 = sparse2.cv_folds(c, "TOY", 2003, 0.25, 0)
    assert (f1 == f2).all()
    for _, g in c.assign(f=f1).groupby("env"):
        cnt = np.bincount(g["f"], minlength=5)
        assert cnt.max() - cnt.min() <= 1


def test_full_fit_matches_sp_models():
    ds, _ = toy()
    envs = envs_of(ds, 2003)
    sp_out, _ = sparse.sparse_year(ds, 2003, envs, 0.25, 0)
    full, _ = sparse2.sp2_year(ds, 2003, envs, 0.25, 0)
    for m in ("m0", "m1"):
        np.testing.assert_allclose(full[m].to_numpy(), sp_out[m].to_numpy(), rtol=1e-8,
                                   atol=1e-8 * np.abs(sp_out[m]).max())


def test_scored_phenotypes_do_not_leak():
    ds, ec = toy()
    envs = envs_of(ds, 2003)
    a, ma = sparse2.sp2_year(ds, 2003, envs, 0.25, 0, ec=ec)
    c = ds.cells.copy()
    tgt = c[(c["year"] == 2003)]
    obs = sparse.sparse_mask(tgt[["env", "genotype"]].reset_index(drop=True), 0.25, 0, "TOY", 2003)
    c.loc[tgt.index[~obs], "y"] = 1e6 * np.random.default_rng(3).normal(size=(~obs).sum())
    b, mb = sparse2.sp2_year(Dataset("TOY", c, ds.markers, 10), 2003, envs, 0.25, 0, ec=ec)
    for m in ("m0", "m1", "gbm", "stk", "dlres", "ecmxe"):
        np.testing.assert_array_equal(a[m].to_numpy(), b[m].to_numpy())
    assert ma["stack_w"] == mb["stack_w"] and ma["ec_rho"] == mb["ec_rho"]


def test_identity_ec_kernel_is_m1():
    ds, _ = toy()
    envs = envs_of(ds, 2003)
    tgt = ds.cells[ds.cells["year"] == 2003].reset_index(drop=True)
    obs = sparse.sparse_mask(tgt[["env", "genotype"]], 0.25, 0, "TOY", 2003)
    train = pd.concat([ds.cells[ds.cells["year"] < 2003], tgt[obs]], ignore_index=True)
    te = sorted(set(tgt.loc[obs, "env"]))
    out, _ = sparse2.fit_core(train, ds.markers, tgt[~obs].reset_index(drop=True), Omega_envs=(te, np.eye(len(te))))
    np.testing.assert_allclose(out["ecmxe"], out["m1"], rtol=1e-8, atol=1e-10)


def test_stack_weights_solve_the_constrained_problem():
    rng = np.random.default_rng(1)
    env = np.repeat(np.arange(6), 40)
    y = rng.normal(size=240)
    Z = np.column_stack([y + rng.normal(size=240), 0.5 * y + rng.normal(size=240), rng.normal(size=240)])
    w = sparse2.stack_weights(Z, y, env)
    cs, Ss = [], []
    for e in range(6):
        m = env == e
        Ze = (Z[m] - Z[m].mean(0)) / Z[m].std(0)
        yz = (y[m] - y[m].mean()) / y[m].std()
        cs.append(Ze.T @ yz / m.sum())
        Ss.append(Ze.T @ Ze / m.sum())
    c, S = np.mean(cs, 0), np.mean(Ss, 0) + 0.05 * np.eye(3)
    r = minimize(lambda v: v @ S @ v - 2 * c @ v, np.ones(3) / 3, bounds=[(0, None)] * 3, method="L-BFGS-B",
                 options={"ftol": 1e-14, "gtol": 1e-12})
    np.testing.assert_allclose(w, r.x, atol=1e-5)
