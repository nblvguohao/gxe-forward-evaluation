"""Checks for the SX headroom check (docs/prereg_headroom_secondary_2026-09-30.md §7)."""
import numpy as np
import pandas as pd

from dartgxe.forward import secondary
from dartgxe.forward.data import Dataset

ENVS = ["IAH1", "IAH2", "NEH1", "TXH1", "WIH1"]


def toy(seed=0, years=range(2014, 2019), n_new=40, p=60):
    rng = np.random.default_rng(seed)
    ids = [f"g{y}_{i}" for y in years for i in range(n_new)]
    M = pd.DataFrame(rng.integers(0, 3, (len(ids), p)).astype(float), index=pd.Index(ids, name="genotype"))
    beta, gam = rng.normal(size=p), rng.normal(size=p)
    flw = pd.Series(M.to_numpy() @ gam / 5 + rng.normal(size=len(ids)), index=ids)     # flowering (genetic + non-marker)
    rows, notes = [], []
    for y in years:
        lines = [f"g{y}_{i}" for i in range(n_new)] + ([f"g{y - 1}_{i}" for i in range(8)] if y > years[0] else [])
        yeff = {g: rng.normal() * 0.5 for g in lines}                                     # line x year, seen by notes
        for e in ENVS:
            env = f"{e}_{y}"
            for g in lines:
                x = M.loc[g].to_numpy()
                rows.append((env, y, g, float(x @ beta / 5 + 0.8 * flw[g] + yeff[g] + rng.normal())))
                notes.append((env, y, g, float(flw[g] + yeff[g] + 0.5 * rng.normal())))
    cells = pd.DataFrame(rows, columns=["env", "year", "genotype", "y"])
    z = pd.DataFrame(notes, columns=["env", "year", "genotype", "v"])
    z["z"] = z.groupby("env")["v"].transform(lambda v: (v - v.mean()) / v.std())
    z = z.drop(columns="v")
    nt = {m: z.copy() for m in secondary.TRAITS_H}
    return Dataset("G2F", cells, M, 25), nt


def envs_of(ds, y):
    return sorted(set(ds.cells.loc[ds.cells["year"] == y, "env"]))


def test_reader_reads_whitelist_only(tmp_path):
    cols = secondary.WHITELIST + secondary.FORBIDDEN + ["Hybrid_orig_name", "Date_Planted"]
    row = {c: "1" for c in cols}
    row.update({"Env": "IAH1_2016", "Year": "2016", "Hybrid": "A/B", "Yield_Mg_ha": "9.9", "Silk_DAP_days": "70"})
    f = tmp_path / "t.csv"
    pd.DataFrame([row, row]).to_csv(f, index=False)
    d = secondary.load_notes(f)
    assert set(d.columns) == set(secondary.WHITELIST) and not set(d.columns) & set(secondary.FORBIDDEN)


def test_note_cells_range_zscore_and_min_records():
    n = 12
    plots = pd.DataFrame({"Env": ["E_2016"] * n + ["F_2016"] * 5, "Year": 2016, "Hybrid": [f"h{i}" for i in range(n)] + [f"h{i}" for i in range(5)],
                          "Silk_DAP_days": list(np.linspace(60, 80, n - 1)) + [500.0] + [70.0] * 5,
                          "Pollen_DAP_days": 65.0, "Plant_Height_cm": 200.0, "Ear_Height_cm": 100.0})
    z = secondary.note_cells(plots, "silk")
    assert set(z["env"]) == {"E_2016"} and len(z) == n - 1                 # out of range dropped; F has < 10 records
    assert abs(z["z"].mean()) < 1e-12 and abs(z["z"].std() - 1) < 1e-12


def test_loeo_excludes_own_environment_and_its_group():
    ds, notes = toy()
    z = notes["silk"]
    cells = ds.cells[ds.cells["year"] == 2018][["env", "year", "genotype"]].reset_index(drop=True)
    x0, n0 = secondary.loeo(z, 2018, cells)
    z2 = z.copy()
    z2.loc[z2["env"] == "IAH1_2018", "z"] += 100.0
    x1, _ = secondary.loeo(z2, 2018, cells)
    own = (cells["env"] == "IAH1_2018").to_numpy()
    np.testing.assert_array_equal(x0[own], x1[own])                        # env j never enters its own x
    assert np.all(np.abs(x1[~own] - x0[~own]) > 1)                          # positive control: other envs do see it
    assert np.all(n0 == len(ENVS) - 1)
    # an exclusion group removes its partners too
    g = {"IAH1_2018": ["IAH1_2018", "IAH2_2018"], "IAH2_2018": ["IAH1_2018", "IAH2_2018"]}
    z3 = z.copy()
    z3.loc[z3["env"] == "IAH2_2018", "z"] += 100.0
    xa, na = secondary.loeo(z, 2018, cells, g)
    xb, _ = secondary.loeo(z3, 2018, cells, g)
    np.testing.assert_array_equal(xa[own], xb[own])
    assert np.all(na[own] == len(ENVS) - 2)
    # the naive own-environment covariate does include env j (it is the all-environment mean)
    xn = secondary.all_env_mean(z2, 2018, cells["genotype"])
    assert np.all(np.abs(xn[own] - secondary.all_env_mean(z, 2018, cells["genotype"])[own]) > 1)


def run(ds, notes, Y=2018):
    ny = sorted(set(ds.cells.loc[ds.cells["year"] == Y + 1, "env"])) if (ds.cells["year"] == Y + 1).any() else None
    return secondary.sx_year(ds, Y, envs_of(ds, Y), notes, ny_envs=ny)


def test_target_and_later_yields_do_not_leak():
    ds, notes = toy(years=range(2014, 2020))
    T1, N1, m1 = run(ds, notes)
    c = ds.cells.copy()
    late = c["year"] >= 2018
    c.loc[late, "y"] = 1e6 * np.random.default_rng(1).normal(size=int(late.sum()))
    T2, N2, m2 = run(Dataset("G2F", c, ds.markers, 25), notes)
    for col in ["s0", "s1", "s1h", "s1g", "s1n", "s1_lor"]:
        np.testing.assert_array_equal(T1[col].to_numpy(), T2[col].to_numpy())
    np.testing.assert_array_equal(N1["s1"].to_numpy(), N2["s1"].to_numpy())
    assert m1["fits"] == m2["fits"] and m1["delta_Y"] == m2["delta_Y"]
    assert "y" not in T1 and "y" not in N1 and len(N1) > 0


def test_target_year_notes_do_not_affect_s0_and_s1g():
    ds, notes = toy()
    T1, _, m1 = run(ds, notes)
    n2 = {m: z.assign(z=np.where(z["year"] == 2018, z["z"] + np.random.default_rng(3).normal(size=len(z)) * 5, z["z"]))
          for m, z in notes.items()}
    T2, _, m2 = run(ds, n2)
    for col in ["s0", "s1g"]:
        np.testing.assert_array_equal(T1[col].to_numpy(), T2[col].to_numpy())
    assert m1["fits"]["s1"] == m2["fits"]["s1"]                             # calibration uses years < Y only
    assert np.abs(T1["s1"] - T2["s1"]).max() > 0                            # but target covariates do use them


def test_calibration_is_within_environment_ols_and_b_zero_gives_s0_ranks():
    rng = np.random.default_rng(0)
    env = np.repeat(["a", "b", "c"], 30)
    X = rng.normal(size=(90, 2))
    y = 2 * X[:, 0] - X[:, 1] + np.repeat([5.0, -3.0, 1.0], 30) + 0.1 * rng.normal(size=90)
    coef, keep = secondary.ols(y, X, env)
    D = pd.get_dummies(env).to_numpy(float)
    ref = np.linalg.lstsq(np.c_[X, D], y, rcond=None)[0][:2]
    np.testing.assert_allclose(coef, ref, rtol=1e-10)
    ds, notes = toy(seed=2)
    T, _, meta = run(ds, notes)
    assert meta["fits"]["s1"]["a"] > 0 and meta["fits"]["s1"]["b"][0] > 0   # the toy's notes carry signal
    a = meta["fits"]["s1"]["a"]
    for _, g in T.groupby("env"):
        np.testing.assert_array_equal(np.argsort(a * g["g"].to_numpy()), np.argsort(g["s0"].to_numpy()))


def test_calibration_uses_only_earlier_years_and_new_lines():
    ds, notes = toy()
    _, _, meta = run(ds, notes)
    assert set(meta["calib"]) == {2015, 2016, 2017}
    assert all(v["cells"] == len(ENVS) * 40 for v in meta["calib"].values())   # the 8 repeated old lines are excluded
