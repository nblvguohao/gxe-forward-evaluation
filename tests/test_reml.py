"""REML recovers a known variance ratio on simulated data with environment fixed effects."""
import numpy as np
import pytest

from dartgxe.baselines.linear import Ridge


@pytest.mark.parametrize("delta_true", [0.002, 0.02])
def test_reml_recovers_ratio(delta_true):
    rng = np.random.default_rng(0)
    n, p, n_env = 6000, 200, 30
    F = rng.standard_normal((n, p)).astype(np.float32)
    env = rng.integers(0, n_env, n).astype(str)
    s2e = 1.0
    beta = rng.normal(scale=np.sqrt(delta_true * s2e), size=p)
    y = F @ beta + rng.normal(scale=np.sqrt(s2e), size=n) + rng.normal(scale=5, size=n_env)[env.astype(int)]
    r = Ridge(F, env, y).reml()
    assert not r["at_grid_edge"]
    assert abs(np.log(r["delta"] / delta_true)) < np.log(1.5), r
    assert abs(r["s2e"] - s2e) < 0.1
