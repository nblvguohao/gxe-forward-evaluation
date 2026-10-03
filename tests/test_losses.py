"""spec 04 / prompt p04: MSED identity, shift invariance, finite gradients."""
import itertools

import pytest
import torch

from dartgxe.losses.rank import LOSSES, msed

torch.manual_seed(0)


def batch(K=3, L=40):
    s = torch.randn(K, L, dtype=torch.float64)
    z = torch.randn(K, L, dtype=torch.float64)
    mask = torch.ones(K, L, dtype=torch.bool)
    mask[1, 30:] = False
    mask[2, 12:] = False
    return s, z, mask


def test_msed_matches_naive_pairs():
    s, z, mask = batch()
    per = []
    for e in range(s.shape[0]):
        idx = torch.nonzero(mask[e]).flatten().tolist()
        tot = sum(((s[e, i] - s[e, j]) - (z[e, i] - z[e, j])) ** 2 for i, j in itertools.combinations(idx, 2))
        n = len(idx)
        per.append(tot / (n * (n - 1) / 2))
    assert torch.allclose(msed(s, z, mask), torch.stack(per).mean(), atol=1e-10)


@pytest.mark.parametrize("name", list(LOSSES))
def test_shift_invariance(name):
    s, z, mask = batch()
    shift = torch.randn(s.shape[0], 1, dtype=s.dtype) * 5
    a = LOSSES[name](s, z, mask)
    b = LOSSES[name](s + shift, z, mask)
    if name == "twohead_mse":
        # twohead_mse is a squared error on z: it is NOT shift-invariant, which is exactly why the
        # final prediction subtracts the within-environment mean of s (spec 04 §4.1)
        assert not torch.allclose(a, b)
    else:
        assert torch.allclose(a, b, atol=1e-8), name


@pytest.mark.parametrize("name", list(LOSSES))
def test_finite_grad_and_padding_ignored(name):
    s, z, mask = batch()
    s = s.clone().requires_grad_(True)
    LOSSES[name](s, z, mask).backward()
    assert torch.isfinite(s.grad).all()
    assert (s.grad[~mask] == 0).all(), name


def test_perfect_scores_are_good():
    s, z, mask = batch()
    for name, f in LOSSES.items():
        assert f(z.clone(), z, mask) <= f(-z.clone(), z, mask), name
