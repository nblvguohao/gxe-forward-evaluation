"""Leakage checks for the wave-2b forward panels (docs/prereg_sequel_wave2b_2026-09-28.md §7)."""
import numpy as np
import pandas as pd
import pytest

from dartgxe.forward import panel
from dartgxe.forward.data import Dataset


def toy(n_geno=60, years=(2001, 2002, 2003, 2004), envs_per_year=3, p=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"g{i}" for i in range(n_geno)]
    M = pd.DataFrame(rng.integers(0, 3, (n_geno, p)).astype(float), index=pd.Index(ids, name="genotype"))
    M.iloc[0, 0] = np.nan
    beta = rng.normal(size=p)
    rows = []
    for y in years:
        lines = rng.choice(ids, 30, replace=False)
        for e in range(envs_per_year):
            for g in lines:
                rows.append((f"E{y}_{e}", y, g, float(M.loc[g].fillna(1).to_numpy() @ beta + rng.normal() + e)))
    cells = pd.DataFrame(rows, columns=["env", "year", "genotype", "y"])
    return Dataset("TOY", cells, M, 25)


def test_forward_year_trains_only_on_earlier_years(monkeypatch):
    ds = toy()
    seen = {}
    real = panel.fit_predict

    def spy(cells, markers, test_ids, seed=0):
        seen["years"] = set(cells["year"])
        return real(cells, markers, test_ids, seed)

    monkeypatch.setattr(panel, "fit_predict", spy)
    out, _ = panel.forward_year(ds, 2003)
    assert seen["years"] == {2001, 2002}
    assert set(out["year"]) == {2003}
    assert set(out["method"]) == set(panel.METHODS)


def test_cv_history_holds_out_genotypes_and_target_year():
    ds = toy()
    out, metas = panel.cv_history(ds, 2004)
    assert out["year"].max() < 2004
    # every genotype is predicted in exactly one fold
    assert out.groupby("genotype")["fold"].nunique().max() == 1
    assert len(metas) == panel.N_FOLDS


def test_transforms_fitted_on_training_genotypes_only():
    ds = toy()
    fit_ids = list(ds.markers.index[:40])
    test_ids = list(ds.markers.index[40:])
    shifted = ds.markers.copy()
    shifted.loc[test_ids] += 100.0  # test genotypes far away; must not move the fitted centring
    Zf1, _, Pf1, _ = panel.features(ds.markers, fit_ids, test_ids)
    Zf2, _, Pf2, _ = panel.features(shifted, fit_ids, test_ids)
    np.testing.assert_allclose(Zf1, Zf2)
    np.testing.assert_allclose(np.abs(Pf1), np.abs(Pf2), atol=1e-8)


def test_stage1_centres_within_environment():
    cells = pd.DataFrame({"env": ["a", "a", "b", "b"], "year": 1, "genotype": ["x", "y", "x", "y"], "y": [10, 12, 100, 104]})
    g = panel.stage1(cells)
    assert g["x"] == pytest.approx(-1.5) and g["y"] == pytest.approx(1.5)


# ---------------------------------------------------------------- wave 2c (cell-level and G x E learners)
from dartgxe.forward import cellgxe  # noqa: E402


def toy_loc(seed=1):
    ds = toy(seed=seed)
    ds.cells["env"] = ds.cells["env"].str.replace(r"^E(\d{4})_(\d)$", r"L\2_\1", regex=True)  # location_year
    rng = np.random.default_rng(seed)
    envs = sorted(ds.cells["env"].unique())
    ec = pd.DataFrame(rng.normal(size=(len(envs), 8)), index=pd.Index(envs, name="env"))
    return ds, ec


def test_cell_cv_folds_match_two_stage_folds():
    ds = toy()
    out, _ = panel.cv_history(ds, 2004)
    m = out.drop_duplicates("genotype").set_index("genotype")["fold"]
    f = cellgxe.cv_folds(ds, 2004)
    assert (f.loc[m.index] == m).all()


def test_hist_ec_ignores_target_environment_values():
    ds, ec = toy_loc()
    fit_envs = sorted(ds.cells[ds.cells["year"] < 2004]["env"].unique())
    tgt = sorted(ds.cells[ds.cells["year"] == 2004]["env"].unique())
    w1, _ = cellgxe.ec_pcs(ec, fit_envs, tgt, "hist")
    ec2 = ec.copy()
    ec2.loc[tgt] += 1000.0  # the target year's own ECs must not matter
    w2, _ = cellgxe.ec_pcs(ec2, fit_envs, tgt, "hist")
    for e in fit_envs + tgt:
        np.testing.assert_allclose(w1[e], w2[e])
    w3, _ = cellgxe.ec_pcs(ec2, fit_envs, tgt, "real")
    assert not np.allclose(w3[tgt[0]], w1[tgt[0]])


def test_forward_year_cell_trains_only_on_earlier_years(monkeypatch):
    ds, ec = toy_loc()
    seen = []
    real = cellgxe.fit_predict_cell

    def spy(train, markers, test, ec=None, mode="hist", seed=0):
        seen.append((set(train["year"]), set(test["year"]), mode))
        return real(train, markers, test, ec, mode, seed)

    monkeypatch.setattr(cellgxe, "fit_predict_cell", spy)
    out, _ = cellgxe.forward_year_cell(ds, 2003, ec)
    assert all(tr == {2001, 2002} and te == {2003} for tr, te, _ in seen)
    assert {m for m in out["method"]} == {"cell_reml", "rn_ridge", "gxe_gbm", "rn_ridge_real", "gxe_gbm_real"}


# ---------------------------------------------------------------- wave 2d (within-environment-loss MLP)
import torch  # noqa: E402

from dartgxe.forward import dl  # noqa: E402


def test_within_env_loss_ignores_environment_shifts():
    g = torch.Generator().manual_seed(0)
    pred, y = torch.randn(12, generator=g), torch.randn(12, generator=g)
    eid = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    shift_y = y + torch.tensor([5.0] * 4 + [-3.0] * 4 + [100.0] * 4)
    shift_p = pred + torch.tensor([-7.0] * 4 + [2.0] * 4 + [0.5] * 4)
    a = dl.within_env_mse(pred, y, eid, 3)
    b = dl.within_env_mse(shift_p, shift_y, eid, 3)
    assert torch.allclose(a, b, atol=1e-5)


def test_tune_and_refit_respects_years(monkeypatch):
    ds = toy(n_geno=80, years=(2001, 2002, 2003, 2004, 2005))
    seen = []
    real_design = dl.Design

    def spy(fit, pred, markers, ec=None):
        seen.append((set(fit["year"]), set(pred["year"])))
        return real_design(fit, pred, markers, ec)

    monkeypatch.setattr(dl, "Design", spy)
    monkeypatch.setattr(dl, "GRID", dl.GRID[:1])
    monkeypatch.setattr(dl, "MAX_EPOCHS", 3)
    out, meta = dl.tune_and_refit(ds, V=2004, Y=2005, seed=0, device="cpu")
    (tune_fit, tune_pred), (refit_fit, refit_pred) = seen
    assert max(tune_fit) < 2004 and tune_pred == {2004}
    assert max(refit_fit) < 2005 and refit_pred == {2005}
    assert set(out["year"]) == {2005} and 1 <= meta["e_star"] <= 3


def test_hapmap_codes():
    from dartgxe.forward.data import hapmap_codes
    calls = np.array([["A", "G", "R", "N"], ["C", "C", "T", "Y"]], dtype=object)
    X = hapmap_codes(calls, pd.Series(["A/G", "C/T"]))
    assert np.array_equal(X[:, :3], [[0, 2, 1], [0, 0, 2]])
    assert np.isnan(X[0, 3]) and X[1, 3] == 1


def test_kernel_embedding_keeps_the_grm():
    """GRM-only data (GEM_IA): features() must centre without scaling, so ridge on the embedding is GBLUP."""
    rng = np.random.default_rng(1)
    W = rng.normal(size=(40, 300))
    K = W @ W.T / 300
    w, U = np.linalg.eigh(K)
    keep = w > 1e-8 * w.max()
    E = pd.DataFrame(U[:, keep] * np.sqrt(w[keep]), index=pd.Index([f"g{i}" for i in range(40)], name="genotype"))
    E.attrs["no_scale"] = True
    fit = [f"g{i}" for i in range(30)]
    Zf, Zo, _, _ = panel.features(E, fit, ["g35", "g36"])
    Kc = K[:30, :30] - K[:30, :30].mean(0) - K[:30, :30].mean(1)[:, None] + K[:30, :30].mean()
    assert np.allclose(Zf @ Zf.T, Kc, atol=1e-8)
    E.attrs = {}
    Zs, _, _, _ = panel.features(E, fit, ["g35"])
    assert np.allclose(Zs.std(0), 1.0)


def test_gem_line_keys_match_grm_names():
    from dartgxe.forward.data import gem_grm_key, gem_line_key
    assert gem_line_key("KO679Y/GEMS-0115//GEMS-0162:0007.0001.") == gem_grm_key("CTU_KO679Y/GEMS-0115//GEMS-0162:0007.0001.")
    assert gem_line_key("PHB47/CML373//GEMS-0162:%.0002.0001.") == gem_grm_key("CTU_PHB47/CML373//GEMS-0162:0002.0001.")
    assert gem_line_key("GEMN-0140/GEMN-0097:@.@.0034.@.//GEMN-0205:0002.0001.") == \
        gem_grm_key("CTU_GEMN-0140/GEMN-0097//GEMN-0205:0002.0001.")
    assert gem_line_key("3IIH6 //GEMN-0173/KO679Y:0072.0001.") == gem_grm_key("CTU_3IIH6//GEMN-0173/KO679Y:0072.0001.")
