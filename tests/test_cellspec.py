"""Checks for the headroom models I-A / I-B (docs/prereg_headroom_ia_ib_2026-09-29.md §4)."""
import numpy as np
import pandas as pd

from dartgxe.baselines.linear import Ridge
from dartgxe.forward import cellspec
from dartgxe.forward.data import Dataset
from dartgxe.forward.panel import features


def toy(name="URSN", n_geno=80, years=(2001, 2002, 2003, 2004), n_loc=4, p=60, seed=0, gxl=1.0):
    rng = np.random.default_rng(seed)
    ids = [f"g{i}" for i in range(n_geno)]
    M = pd.DataFrame(rng.integers(0, 3, (n_geno, p)).astype(float), index=pd.Index(ids, name="genotype"))
    beta = rng.normal(size=p)
    bl = rng.normal(size=(n_loc, p)) * gxl
    rows = []
    for y in years:
        lines = rng.choice(ids, 40, replace=False)
        for l in range(n_loc):
            sd = 1.0 + l  # heteroscedastic environments
            for g in lines:
                x = M.loc[g].to_numpy()
                rows.append((f"L{l}_{y}", y, g, float(x @ beta + 0.5 * x @ bl[l] + sd * rng.normal() + 3 * l)))
    return Dataset(name, pd.DataFrame(rows, columns=["env", "year", "genotype", "y"]), M, 10)


def test_omega_zero_equals_marker_ridge_at_the_same_lambda():
    ds = toy()
    train, test = ds.cells[ds.cells["year"] < 2004], ds.cells[ds.cells["year"] == 2004]
    P = cellspec.Prep(train, ds.markers, test)
    ours = cellspec.fit_ia(P, np.ones(P.n_env), P.yc, 0.0)
    Zf, Zt, _, _ = features(ds.markers, P.fit_ids, P.test_ids)
    r = Ridge(Zf[P.gi], P.train["env"].to_numpy(), P.train["y"].to_numpy())
    lam_abs = (1.0 / ours["delta"]) * P.n_markers * P.c0
    theirs = Zt @ r.coef(lam_abs / r.scale)
    assert np.max(np.abs(ours["u_t"] - theirs)) / np.std(theirs) < 1e-6
    # and the REML optimum is the same model (criterion invariant to the parametrisation)
    rm = r.reml()
    assert abs(np.log(rm["lambda"]) - np.log(lam_abs)) < 1e-3


def test_ib_c2_zero_is_the_ia_control_and_gxl_only_in_active_locations():
    ds = toy()
    envs = sorted(set(ds.cells.loc[ds.cells["year"] == 2004, "env"]))
    out, meta = cellspec.panel_year(ds, 2004, envs)
    P = cellspec.Prep(ds.cells[ds.cells["year"] < 2004], ds.markers, ds.cells[ds.cells["year"] == 2004])
    ib, _ = cellspec.ib_family(P, "URSN", "location")
    assert np.max(np.abs(ib["grid"][0.0] - out["ctrl"].to_numpy())) < 1e-6 * out["ctrl"].std()
    assert meta["ib_main"]["n_active"] == 4 and out["ib_active"].all()
    # a target environment at a location never seen before gets the (jointly fitted) main effect only
    ds2 = toy()
    c = ds2.cells.copy()
    c.loc[(c["year"] == 2004) & (c["env"] == "L3_2004"), "env"] = "NEW_2004"
    ds2 = Dataset("URSN", c, ds2.markers, 10)
    P2 = cellspec.Prep(c[c["year"] < 2004], ds2.markers, c[c["year"] == 2004])
    ib2, _ = cellspec.ib_family(P2, "URSN", "location")
    new = (P2.test["env"] == "NEW_2004").to_numpy()
    assert new.any() and not ib2["active_cell"][new].any() and ib2["active_cell"][~new].all()
    for c2, v in ib2["grid"].items():
        np.testing.assert_allclose(v[new], ib2["grid_main_only"][c2][new], rtol=0, atol=1e-9)
        if c2 > 0:
            assert np.abs(v[~new] - ib2["grid_main_only"][c2][~new]).max() > 1e-6


def test_panel_year_trains_only_on_earlier_years(monkeypatch):
    ds = toy()
    seen = {}
    real = cellspec.Prep

    class Spy(real):
        def __init__(self, train, markers, test):
            seen["train_years"], seen["test_years"] = set(train["year"]), set(test["year"])
            super().__init__(train, markers, test)

    monkeypatch.setattr(cellspec, "Prep", Spy)
    cellspec.panel_year(ds, 2003, sorted(set(ds.cells.loc[ds.cells["year"] == 2003, "env"])))
    assert seen["train_years"] == {2001, 2002} and seen["test_years"] == {2003}


def test_reml_picks_up_gxl_on_toy_data():
    ds = toy(gxl=2.0, seed=3)
    envs = sorted(set(ds.cells.loc[ds.cells["year"] == 2004, "env"]))
    _, meta = cellspec.panel_year(ds, 2004, envs)
    assert meta["ib_main"]["c2_main"] > 0


def test_env_scales_track_residual_sd_on_toy_data():
    ds = toy(gxl=0.0, seed=4, years=tuple(range(2001, 2008)))
    P = cellspec.Prep(ds.cells[ds.cells["year"] < 2007], ds.markers, ds.cells[ds.cells["year"] == 2007])
    ctrl = cellspec.fit_ia(P, np.ones(P.n_env), P.yc, 0.0)
    s = cellspec.env_scales(P, ctrl["u_f"])
    loc = np.array([int(e[1]) for e in P.envs])          # environment L{l}: residual SD 1 + l
    means = [s[loc == l].mean() for l in range(4)]
    assert all(np.diff(means) > 0) and means[-1] / means[0] > 1.5


def test_location_mapping_examples():
    L = cellspec.location
    assert L("G2F", "IAH1a_2014") == L("G2F", "IAH1c_2014") == "IAH1"
    assert L("G2F", "TXH1-Dry_2018") == L("G2F", "TXH1-Late_2018") == "TXH1"
    assert L("G2F", "MOH1_1_2018") == L("G2F", "MOH1_2_2020") == "MOH1"
    assert L("G2F", "NYS1_2016") == "NYS1" and L("G2F", "IAH1a_2014", "region") == "IA"
    assert L("NUST", "Saginaw_County_MI_2005") == L("NUST", "SaginawCounty_MI_2006") == "SaginawCounty_MI"
    assert L("NUST", "Boone_IA_2001") != L("NUST", "BooneCounty_IA_2001")
    assert L("NUST", "Arlington_WI_1993", "region") == "WI"
    assert L("URSN", "BRK_95") == "BRK" and L("URSN", "BRK_95", "region") is None
    assert L("ESWYT", "South Africa Pannar_2010", "region") == "South Africa"
    assert L("ESWYT", "India New Delhi_2010") == "India New Delhi" and L("ESWYT", "India New Delhi_2010", "region") == "India"
    assert L("GEM_IA", "A_2014") is None and L("MU_SOY", "2017_FLD_12_4") is None
