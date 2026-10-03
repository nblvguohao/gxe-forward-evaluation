"""Checks for the holdout trait cells (docs/prereg_holdout_ib_2026-09-29.md §2, §6)."""
import numpy as np
import pandas as pd
import pytest

from dartgxe.forward import traitcells as tc


def nust_toy():
    rows = []
    for yr in (2001, 2002, 2003, 2004):
        for loc in "ABCDEF":
            for g in range(30):
                gid = f"g-{yr if g >= 15 else 0}_{g}"          # 15 lines persist, 15 are new each year
                for ph_, v in (("Height", 80 + g), ("Maturity", 120 + g), ("YieldBuA", 50 + g), ("YieldRank", g)):
                    rows.append((f"Trial_{yr}", loc, gid, ph_, str(v)))
    return pd.DataFrame(rows, columns=["Experiment", "Location", "GermplasmId", "Phenotype", "Value"])


def test_cells_contain_only_the_requested_trait_and_genotyped_lines():
    ph = nust_toy()
    ids = {tc.norm(g) for g in ph["GermplasmId"].unique()} - {tc.norm("g-0_3")}
    c = tc.nust_trait_cells(ph, "Height", ids)
    assert tc.norm("g-0_3") not in set(c["genotype"])
    # height = 80 + g: the mean of the cell for a persistent line must be its own value
    row = c[(c["env"] == "A_2001") & (c["genotype"] == tc.norm("g-0_4"))]
    assert float(row["y"].iloc[0]) == 84.0
    assert set(c["year"]) == {2001, 2002, 2003, 2004}


def test_yield_rank_and_unregistered_traits_are_refused():
    ph = nust_toy()
    for t in ("YieldRank", "YieldBuA", "DescriptiveCode"):
        with pytest.raises(ValueError):
            tc.nust_trait_cells(ph, t, [])
    with pytest.raises(ValueError):
        tc.ursn_trait_cells(pd.DataFrame(), "DIS", [])


def test_count_matches_hand_count():
    ph = nust_toy()
    ids = {tc.norm(g) for g in ph["GermplasmId"].unique()}
    c = tc.nust_trait_cells(ph, "Maturity", ids)
    t = tc.count_targets(c, 25)
    m = t[t["rule"] == "main"].set_index("year")
    assert m.loc[2001, "prior_years"] == 0 and not m.loc[2001, "qualifies"]
    assert m.loc[2002, "prior_years"] == 1 and not m.loc[2002, "qualifies"]
    # 6 environments per year with 30 lines each (>= 25); half of the 30 lines are new every year -> new share 0.5
    assert m.loc[2003, "envs_ge_min"] == 6 and m.loc[2003, "prior_years"] == 2
    assert m.loc[2003, "new_share"] == 0.5 and bool(m.loc[2003, "qualifies"])
    assert m.loc[2003, "cells"] == 6 * 30
    assert m.loc[2004, "qualifies"]


def test_ursn_cells_use_env_code_and_selected_trait():
    u = pd.DataFrame({"line": ["a", "b", "a"], "year": [2000, 2000, 2001], "env_code": ["X_00", "X_00", "X_01"],
                      "DIS": [10, 20, 30], "VSK": [1.0, np.nan, 3.0]})
    c = tc.ursn_trait_cells(u, "VSK", ["a", "b"])
    assert list(c["env"]) == ["X_00", "X_01"] and list(c["y"]) == [1.0, 3.0]
