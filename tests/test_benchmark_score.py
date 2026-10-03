"""Edge cases of benchmark/score.py (WP3): constant predictions, missing coverage, duplicate cells, unknown dataset."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
import score  # noqa: E402


@pytest.fixture()
def bench(tmp_path):
    rng = np.random.default_rng(0)
    rows, ref = [], []
    envs = {}
    for Y in (2003, 2004, 2005):
        envs[str(Y)] = [f"E{Y}_{j}" for j in range(6)]
        for e in envs[str(Y)]:
            g = np.arange(30)
            y = rng.normal(size=30) + 0.5 * g / 30
            for gi, yi in zip(g, y):
                rows.append((e, Y, f"g{gi}", yi))
                ref.append((e, Y, f"g{gi}", "cell_reml", yi + rng.normal(scale=1.0)))
    pd.DataFrame(rows, columns=["env", "year", "genotype", "y"]).to_parquet(tmp_path / "cells_T.parquet")
    pd.DataFrame(ref, columns=["env", "year", "genotype", "method", "pred"]).to_parquet(tmp_path / "predictions_T.parquet")
    json.dump({"T": {"min_genotypes": 10, "target_years": [2003, 2004, 2005], "scorable_environments": envs,
                     "methods": ["cell_reml"], "set": "original"}}, open(tmp_path / "splits.json", "w"))
    cells = pd.DataFrame(rows, columns=["env", "year", "genotype", "y"])
    return tmp_path, cells


def submission(cells, pred_fn):
    s = cells[["env", "year", "genotype"]].copy()
    s["dataset"] = "T"
    s["pred"] = pred_fn(cells)
    return s


def run(tmp, sub, name="s.csv"):
    f = tmp / name
    sub.to_csv(f, index=False)
    return score.main([str(f), "--data", str(tmp), "--out", str(tmp / "out.json")])


def test_constant_year_is_skipped_and_reported(bench):
    tmp, cells = bench
    rng = np.random.default_rng(1)
    sub = submission(cells, lambda c: np.where(c["year"] == 2004, 1.0, c["y"] + rng.normal(size=len(c))))
    out = run(tmp, sub)
    assert out["years_without_defined_metric"]["spearman"] == ["T_2004"]
    est = out["vs_cell_gblup"]["spearman"]["all"]
    assert np.isfinite(est["est"]) and est["years"] == 2


def test_all_years_constant_is_an_error(bench):
    tmp, cells = bench
    with pytest.raises(SystemExit, match="no target year"):
        run(tmp, submission(cells, lambda c: 1.0))


def test_missing_coverage_is_refused(bench):
    tmp, cells = bench
    sub = submission(cells, lambda c: c["y"])
    with pytest.raises(SystemExit, match="scorable cells predicted"):
        run(tmp, sub.iloc[::2])


def test_unknown_dataset_is_refused(bench):
    tmp, cells = bench
    sub = submission(cells, lambda c: c["y"]).assign(dataset="Z")
    with pytest.raises(SystemExit, match="unknown dataset"):
        run(tmp, sub)


def test_duplicate_cells_are_refused(bench):
    tmp, cells = bench
    sub = submission(cells, lambda c: c["y"])
    with pytest.raises(SystemExit, match="duplicate"):
        run(tmp, pd.concat([sub, sub.iloc[:5]]))


def test_perfect_submission_beats_reference(bench):
    tmp, cells = bench
    out = run(tmp, submission(cells, lambda c: c["y"]))
    r = out["vs_cell_gblup"]["spearman"]["all"]
    assert r["est"] > 0.3 and r["lo"] > 0
