"""Checks for the OL headroom check (docs/prereg_headroom_oldlines_2026-09-30.md §5)."""
import numpy as np
import pandas as pd

from dartgxe.forward import cellspec, oldlines, sparse
from dartgxe.forward.cellspec import Prep, fit_ia
from dartgxe.forward.data import Dataset

PRED = ["ol0", "ol1", "ol2", "ol3", "ol4"]


def toy(n_geno=80, years=(2001, 2002, 2003, 2004), n_loc=6, p=60, seed=0, gxl=1.0, lxl=1.0):
    """URSN-style names (location = env name without the year). Marker G x location (gxl) and a non-marker
    line x location effect that repeats across years (lxl)."""
    rng = np.random.default_rng(seed)
    ids = [f"g{i}" for i in range(n_geno)]
    M = pd.DataFrame(rng.integers(0, 3, (n_geno, p)).astype(float), index=pd.Index(ids, name="genotype"))
    beta = rng.normal(size=p)
    bl = rng.normal(size=(n_loc, p)) * gxl
    own = rng.normal(size=(n_geno, n_loc)) * lxl * 3
    rows = []
    for y in years:
        lines = rng.choice(n_geno, 45, replace=False)
        for l in range(n_loc):
            for i in lines:
                x = M.iloc[i].to_numpy()
                rows.append((f"L{l}_{y}", y, ids[i], float(x @ beta + 0.5 * x @ bl[l] + own[i, l] + rng.normal() + 3 * l)))
    return Dataset("URSN", pd.DataFrame(rows, columns=["env", "year", "genotype", "y"]), M, 10)


def envs_of(ds, y):
    return sorted(set(ds.cells.loc[ds.cells["year"] == y, "env"]))


def test_old_lines_scored_cells_and_training_years(monkeypatch):
    ds = toy()
    # a line with history but no markers, and a line first seen in the target year
    extra = pd.DataFrame({"env": ["L0_2002", "L0_2004", "L1_2004"], "year": [2002, 2004, 2004],
                          "genotype": ["nomark", "nomark", "g_new"], "y": [0.0, 0.0, 0.0]})
    M = pd.concat([ds.markers, pd.DataFrame([ds.markers.iloc[0].to_numpy()], index=pd.Index(["g_new"], name="genotype"))])
    ds = Dataset("URSN", pd.concat([ds.cells, extra], ignore_index=True), M, 10)
    seen = {}
    real = oldlines.Prep

    class Spy(real):
        def __init__(self, train, markers, test):
            seen["train_years"], seen["test_years"] = set(train["year"]), set(test["year"])
            super().__init__(train, markers, test)

    monkeypatch.setattr(oldlines, "Prep", Spy)
    out, meta = oldlines.oldlines_year(ds, 2004, envs_of(ds, 2004))
    assert seen["train_years"] == {2001, 2002, 2003} and seen["test_years"] == {2004}
    hist = set(ds.cells.loc[ds.cells["year"] < 2004, "genotype"]) & set(ds.markers.index)
    assert set(out["genotype"]) <= hist and "g_new" not in set(out["genotype"]) and "nomark" not in set(out["genotype"])
    expect = ds.cells[(ds.cells["year"] == 2004) & ds.cells["genotype"].isin(hist)]
    assert len(out) == len(expect) == meta["n_scored"] and meta["qualifies"]
    # qualification needs MIN_ENVS environments with >= min_n old lines
    few = Dataset("URSN", ds.cells, ds.markers, 10_000)
    _, q, ok = oldlines.old_line_targets(few, 2004, envs_of(ds, 2004))
    assert not q and ok == []


def test_target_and_later_phenotypes_do_not_leak():
    ds = toy(years=(2001, 2002, 2003, 2004))
    out1, meta1 = oldlines.oldlines_year(ds, 2003, envs_of(ds, 2003))
    c = ds.cells.copy()
    late = c["year"] >= 2003
    c.loc[late, "y"] = 1e6 * np.random.default_rng(9).normal(size=int(late.sum()))
    out2, meta2 = oldlines.oldlines_year(Dataset("URSN", c, ds.markers, 10), 2003, envs_of(ds, 2003))
    for m in PRED:
        np.testing.assert_array_equal(out1[m].to_numpy(), out2[m].to_numpy())
    for key in ("omega", "c2", "ol3_em", "ol4_em"):
        assert meta1[key] == meta2[key]


def test_nesting_and_benchmark_path():
    ds = toy(seed=1)
    y = 2004
    out, meta = oldlines.oldlines_year(ds, y, envs_of(ds, y))
    train = ds.cells[ds.cells["year"] < y]
    # OL0 equals the cell_reml prediction made the benchmark way (all target cells as test, Nystrom path)
    Pb = Prep(train, ds.markers, ds.cells[ds.cells["year"] == y])
    ub = fit_ia(Pb, np.ones(Pb.n_env), Pb.yc, 0.0)["u_t"]
    bench = pd.Series(ub, index=Pb.test_ids).reindex(out["genotype"]).to_numpy()
    np.testing.assert_allclose(out["ol0"].to_numpy(), bench, rtol=0, atol=1e-8 * np.abs(bench).max())
    # ... and OL1 the I-A ia_g prediction at the chosen omega
    ug = fit_ia(Pb, np.ones(Pb.n_env), Pb.yc, meta["omega"])["u_t"]
    np.testing.assert_allclose(out["ol1"].to_numpy(), pd.Series(ug, index=Pb.test_ids).reindex(out["genotype"]).to_numpy(),
                               rtol=0, atol=1e-8 * np.abs(ug).max())
    # OL2 at c2 = 0 is OL1 (any omega)
    P = Prep(train, ds.markers, out)
    for om in (0.0, 0.3):
        ol2 = oldlines.fit_ol2(P, "URSN", om)
        u0 = ol2["T0"] @ ol2["fits"][0.0]["a0"]
        ia = fit_ia(P, np.ones(P.n_env), P.yc, om)["u_f"]
        np.testing.assert_allclose(u0, ia, rtol=0, atol=1e-8 * np.abs(ia).max())
        assert np.all(ol2["fits"][0.0]["G"] == 0)
    # OL3 with s2v -> 0 is OL1
    r = np.random.default_rng(0).normal(size=50)
    grp = np.array([f"a{i % 7}" for i in range(50)])
    assert all(v == 0 for v in oldlines.shrunken_group_means(r, grp, 0.0, 1.0).values())


def test_em_matches_closed_form_on_balanced_data():
    rng = np.random.default_rng(3)
    G, n = 400, 4
    v = rng.normal(scale=1.0, size=G)
    grp = np.repeat(np.arange(G), n)
    r = v[grp] + rng.normal(size=G * n)
    vc = oldlines.em_oneway(r, grp.astype(str))
    means = r.reshape(G, n).mean(1)
    s2e = ((r.reshape(G, n) - means[:, None]) ** 2).sum() / (G * (n - 1))
    s2v = (means ** 2).mean() - s2e / n                 # ML with known zero mean (balanced, interior optimum)
    assert vc["converged"] and s2v > 0
    assert abs(vc["s2e"] - s2e) / s2e < 1e-4 and abs(vc["s2v"] - s2v) / s2v < 1e-4
    # shrinkage uses those components: n s2v / (n s2v + s2e) times the group mean
    eff = oldlines.shrunken_group_means(r, grp.astype(str), vc["s2v"], vc["s2e"])
    np.testing.assert_allclose(eff["0"], n * vc["s2v"] / (n * vc["s2v"] + vc["s2e"]) * means[0], rtol=1e-12)


def test_ol2_system_equals_explicit_features_and_inactive_locations_get_main_only():
    ds = toy(n_geno=60, p=30, seed=2, n_loc=4)
    y = 2004
    scored, _, _ = oldlines.old_line_targets(ds, y, envs_of(ds, y))
    P = Prep(ds.cells[ds.cells["year"] < y], ds.markers, scored)
    om, c2 = 0.2, 0.75                                  # a C2_GRID point
    ol2 = oldlines.fit_ol2(P, "URSN", om)
    T0, Pk, active = ol2["T0"], ol2["Pk"], ol2["active"]
    k, L = Pk.shape[1], len(active)
    assert L == 4
    A00, r0, _ = ol2["main_pieces"]
    Afull, rfull = sparse.assemble(A00, r0, *ol2["pieces"], np.eye(L), c2)
    code = {l: i for i, l in enumerate(active)}
    rows = []
    for i in range(len(P.train)):
        loc = cellspec.location("URSN", P.envs[P.ej[i]])
        g = np.sqrt(c2) * np.kron(Pk[P.gi[i]], np.eye(L)[code[loc]])
        rows.append(np.r_[T0[P.gi[i]], g])
    F = np.array(rows)
    Fc = F - pd.DataFrame(F).groupby(P.ej).transform("mean").to_numpy()
    np.testing.assert_allclose(Afull, Fc.T @ Fc, rtol=1e-8, atol=1e-8 * np.abs(Afull).max())
    np.testing.assert_allclose(rfull, Fc.T @ P.yc, rtol=1e-8, atol=1e-8 * np.abs(rfull).max())
    # the G x L values used for prediction are the explicit features times the coefficients
    f = ol2["fits"][c2]
    gx = oldlines.gxl_values(ol2, f["G"], P.gi, [cellspec.location("URSN", e) for e in P.envs[P.ej]])
    coef = np.r_[f["a0"], f["G"].reshape(-1) / np.sqrt(c2)]
    np.testing.assert_allclose(T0[P.gi] @ f["a0"] + gx, F @ coef, rtol=1e-8, atol=1e-8)
    # a location never seen in training contributes nothing beyond the main effect
    assert np.all(oldlines.gxl_values(ol2, f["G"], P.gi[:5], ["NEW"] * 5) == 0)


def test_reml_picks_up_gxl_and_line_by_location_on_toy_data():
    ds = toy(seed=4, gxl=2.0, lxl=1.5)
    y = 2004
    out, meta = oldlines.oldlines_year(ds, y, envs_of(ds, y))
    assert meta["c2"] > 0 and meta["ol3_em"]["s2v"] > 0 and meta["share_scored_with_line_location_history"] > 0.5
    sp = lambda m: out.groupby("env")[["y", m]].apply(lambda g: g["y"].rank().corr(g[m].rank())).mean()
    assert sp("ol3") > sp("ol1")
