"""spec 01 §1.5. Runs on the real G2F splits when they exist, and always on a synthetic design."""
import numpy as np
import pandas as pd
import pytest

from dartgxe.paths import splits_dir
from dartgxe.splits.core import FORWARD, cv00, cv_by, forward, make_all, novelty


def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2014, 2024):
        hyb = [f"H{(year - 2014) // 2}_{i}/T{i % 3}" for i in range(60)]  # two-year rotation like G2F
        for loc in range(6):
            for h in rng.choice(hyb, 40, replace=False):
                rows.append((f"L{loc}_{year}", year, h))
    c = pd.DataFrame(rows, columns=["env", "year", "genotype"])
    c["parent1"] = c["genotype"].str.split("/").str[0]
    return c


def real_or_none(name):
    p = splits_dir("g2f") / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def check_common(sp: pd.DataFrame):
    # each cell has exactly one role per fold
    assert not sp.duplicated(["fold", "env", "genotype"]).any()
    assert set(sp["role"]) <= {"train", "val", "test", "refit_extra", "drop"}


def check_disjoint(sp: pd.DataFrame, key: str):
    for f, s in sp.groupby("fold"):
        test = set(s.loc[s["role"] == "test", key])
        for r in ("train", "val"):
            assert not (test & set(s.loc[s["role"] == r, key])), (f, r, key)
        # inner validation must not share the key with inner training either
        assert not (set(s.loc[s["role"] == "val", key]) & set(s.loc[s["role"] == "train", key])), (f, key)


@pytest.fixture(params=["synthetic", "real"])
def splits(request):
    if request.param == "synthetic":
        c = synthetic()
        return make_all(c), c
    sp = {n: real_or_none(n) for n in ["CV1", "CV1parent", "CV0", "CV00", *FORWARD]}
    sp = {k: v for k, v in sp.items() if v is not None}
    if not sp:
        pytest.skip("real splits not built")
    return sp, None


def test_cv1(splits):
    sp, _ = splits
    check_common(sp["CV1"])
    check_disjoint(sp["CV1"], "genotype")
    # every cell is a test cell exactly once across folds
    t = sp["CV1"][sp["CV1"]["role"] == "test"]
    assert not t.duplicated(["env", "genotype"]).any()


def test_cv1_parent(splits):
    sp, c = splits
    s = sp["CV1parent"]
    check_common(s)
    p1 = s["genotype"].str.split("/").str[0]
    check_disjoint(s.assign(parent1=p1), "parent1")


def test_cv0(splits):
    sp, _ = splits
    check_common(sp["CV0"])
    check_disjoint(sp["CV0"], "env")


def test_cv00(splits):
    sp, _ = splits
    s = sp["CV00"]
    check_common(s)
    check_disjoint(s, "env")
    check_disjoint(s, "genotype")


def test_forward(splits):
    sp, _ = splits
    for name, (a, b, c, t) in FORWARD.items():
        if name not in sp:
            continue
        s = sp[name]
        check_common(s)
        yr = lambda r: set(s.loc[s["role"] == r, "year"])
        assert yr("test") == {t}
        assert max(yr("train")) <= a and yr("val") == {b}
        assert t not in yr("train") | yr("val") | yr("refit_extra")
        assert max(yr("val")) < t


def test_same_kind_validation(splits):
    """spec 01 §1.5: validation and test years are of the same kind (new vs repeated hybrids)."""
    sp, c = splits
    for name in sp:
        if not (name.startswith("FYnew") or name.startswith("FYrep") or name.endswith("m")):
            continue
        if c is not None and name == "FYnew2016s":
            continue  # synthetic rotation pairs 2014/2015 exactly; real G2F overlap is ~0.28 (checked on real splits)
        s = sp[name]
        v_new = novelty(s, "val", ("train",))
        t_new = novelty(s, "test", ("train", "val", "refit_extra"))
        if name.startswith("FYrep"):
            assert v_new <= 0.3 and t_new <= 0.3, (name, v_new, t_new)
        else:
            assert v_new >= 0.7 and t_new >= 0.7, (name, v_new, t_new)


def test_forward_rejects_bad_years():
    with pytest.raises(AssertionError):
        forward(synthetic(), 2020, 2020, 2021, 2022)
