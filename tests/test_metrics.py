"""spec 03 §3.6."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from dartgxe.eval.compare import paired_delta, verdict
from dartgxe.eval.metrics import env_metrics, env_seed_matrix, per_env

RNG = np.random.default_rng(1)


def make_pred(n_env=3, n=30, noise=1.0, method="m", seed=0, fold=0):
    rows = []
    for e in range(n_env):
        y = RNG.normal(size=n) + e * 10
        yhat = y + RNG.normal(scale=noise, size=n)
        rows += [(method, seed, fold, f"E{e}", f"G{i}", y[i], yhat[i]) for i in range(n)]
    return pd.DataFrame(rows, columns=["method", "seed", "fold", "env", "genotype", "observed", "prediction"])


def by_metric(pe):
    return pe.groupby("metric")["value"].mean()


def test_monotone_invariance():
    p = make_pred()
    base = by_metric(per_env(p))
    q = p.copy()
    # random strictly increasing transform, different per environment
    for e, g in q.groupby("env"):
        a, b = RNG.uniform(0.5, 2), RNG.uniform(-5, 5)
        q.loc[g.index, "prediction"] = np.exp(a * g["prediction"] / 10) * 3 + b
    new = by_metric(per_env(q))
    for m in ["spearman", "sel_diff_f05", "sel_diff_f10", "sel_diff_f20", "topk_hit_f10", "ndcg_f10"]:
        assert abs(base[m] - new[m]) < 1e-9, m


def test_affine_invariance_pearson():
    p = make_pred()
    q = p.copy()
    for e, g in q.groupby("env"):
        q.loc[g.index, "prediction"] = RNG.uniform(0.1, 5) * g["prediction"] + RNG.uniform(-9, 9)
    assert abs(by_metric(per_env(p))["pearson"] - by_metric(per_env(q))["pearson"]) < 1e-9


def test_reversal_flips_spearman():
    p = make_pred()
    q = p.assign(prediction=-p["prediction"])
    assert abs(by_metric(per_env(p))["spearman"] + by_metric(per_env(q))["spearman"]) < 1e-12


def test_matches_scipy():
    p = make_pred()
    pe = per_env(p)
    for e, g in p.groupby("env"):
        s = pe[(pe.env == e) & (pe.metric == "spearman")]["value"].item()
        r = pe[(pe.env == e) & (pe.metric == "pearson")]["value"].item()
        assert abs(s - spearmanr(g.observed, g.prediction)[0]) < 1e-12
        assert abs(r - pearsonr(g.observed, g.prediction)[0]) < 1e-12


def test_small_env_excluded():
    p = make_pred(n=20)
    assert per_env(p).empty


def test_fold_offsets_do_not_leak_into_ranking():
    """CV1: one env scored by two folds whose models differ by a constant offset."""
    p0 = make_pred(n_env=1, n=40, fold=0)
    p1 = make_pred(n_env=1, n=40, fold=1)
    p1["genotype"] = p1["genotype"] + "b"
    p1["prediction"] += 100.0  # fold-1 model predicts everything 100 higher
    both = pd.concat([p0, p1])
    pe = per_env(both)
    within = pe[pe.metric == "spearman"]["value"].mean()
    concat = spearmanr(both.observed, both.prediction)[0]
    base = (spearmanr(p0.observed, p0.prediction)[0] + spearmanr(p1.observed, p1.prediction)[0]) / 2
    assert abs(within - base) < 1e-12
    assert abs(concat - base) > 0.05  # pooling across folds would have been contaminated


def test_shuffle_null():
    p = make_pred(n_env=40, n=50)
    p["observed"] = p.groupby("env")["observed"].transform(lambda s: RNG.permutation(s.to_numpy()))
    v = per_env(p).query("metric == 'spearman'")["value"]
    assert abs(v.mean()) < 3 * v.std(ddof=1) / np.sqrt(len(v))


def test_paired_delta_two_level():
    pa = pd.concat([make_pred(n_env=30, n=40, noise=0.5, method="A", seed=s) for s in range(3)])
    pb = pa.assign(method="B", prediction=pa["prediction"] + RNG.normal(scale=3.0, size=len(pa)))
    mat = env_seed_matrix(per_env(pd.concat([pa, pb])), "spearman")
    r = paired_delta(mat, "A", "B", B=500)
    assert r["delta"] > 0 and r["ci2_lo"] > 0 and r["seeds_same_sign"]
    assert verdict(r["delta"], r["ci2_lo"], r["ci2_hi"], thr=0.01) == "resolved"
    assert verdict(0.001, 0.0005, 0.002, thr=0.01) == "detectable"
    assert verdict(0.02, -0.01, 0.05, thr=0.01) == "tied"


def test_undefined_env_metric_is_not_zero():
    """A method constant within an environment has an undefined Spearman there; it must drop out,
    not enter the mean as 0 (bug found 2026-09-26 with B0_envmean)."""
    p = make_pred(n_env=3, n=30)
    const = p.assign(method="C", prediction=p.groupby("env")["observed"].transform("mean"))
    mat = env_seed_matrix(per_env(pd.concat([p, const])), "spearman")
    assert "C" not in mat.index.get_level_values(0)
    assert "m" in mat.index.get_level_values(0)
