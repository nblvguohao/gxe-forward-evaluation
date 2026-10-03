"""Unit tests for gpverdict-audit (hand-checkable cases)."""
import numpy as np
import pandas as pd
import pytest

from gpverdict_audit.core import (against_baseline, audit, check_overlap, decompose, env_scores, load, optimism)


def world(n_env=12, n_g=40, seed=0):
    rng = np.random.default_rng(seed)
    env = np.repeat([f"E{j}" for j in range(n_env)], n_g)
    geno = np.tile([f"g{i}" for i in range(n_g)], n_env)
    mu = rng.normal(0, 3, n_env)
    gv = rng.normal(size=n_g)
    y = np.repeat(mu, n_g) + np.tile(gv, n_env) + rng.normal(size=n_env * n_g)
    return env, geno, y, np.repeat(mu, n_g), np.tile(gv, n_env), rng


def frame(env, geno, y, preds):
    return pd.concat([pd.DataFrame({"environment": env, "genotype": geno, "observed": y, "predicted": p, "candidate": c})
                      for c, p in preds.items()], ignore_index=True)


def test_decompose_identity_unequal_sizes():
    rng = np.random.default_rng(1)
    n = rng.integers(5, 60, 7)
    env = np.repeat(np.arange(7), n)
    y = rng.normal(0, 2, 7)[env] + rng.normal(size=len(env)) * rng.uniform(0.5, 3, 7)[env]
    p = 0.5 * y + rng.normal(size=len(env))
    d = decompose(y, p, env)
    r = (d["a"] + d["k"] * d["u"]) / np.sqrt((d["v_b"] + d["u"] ** 2) * d["S"])
    assert abs(r - d["pooled_pearson"]) < 1e-12


def test_env_scores_perfect_and_reversed():
    y = np.arange(20.0)
    assert env_scores(y, y)[0] == pytest.approx(1.0)
    assert env_scores(y, -y)[0] == pytest.approx(-1.0)
    assert env_scores(y, y)[2] > 0 > env_scores(y, -y)[2]


def test_overlap_fail_and_pass():
    env, geno, y, mu, gv, _ = world()
    df = load(frame(env, geno, y, {"a": gv}))
    sel = df.copy()
    sel["set"] = "selection"
    assert check_overlap(pd.concat([sel, df]), {})["verdict"] == "FAIL"
    sel2 = sel[sel["Env"].isin(["E0", "E1"])]
    rep2 = df[~df["Env"].isin(["E0", "E1"])]
    assert check_overlap(pd.concat([sel2, rep2]), {})["verdict"] == "PASS"
    assert check_overlap(df, {})["verdict"] == "UNKNOWN"
    assert check_overlap(df, {"selected_on_report": True})["verdict"] == "FAIL"


def test_pooled_criterion_prefers_env_mean_candidate():
    """Candidate 'means' copies environment means with weak ranking; 'rank' ranks well without means."""
    env, geno, y, mu, gv, rng = world()
    preds = {"means": mu + 0.1 * gv + rng.normal(0, 0.5, len(y)), "rank": gv}
    res = audit(frame(env, geno, y, preds), {"criterion": "pooled_pearson", "chosen": "means"}, reps=20)
    assert res["A2_criterion"]["picks"]["pooled_pearson"] == "means"
    assert res["A2_criterion"]["picks"]["within_spearman"] == "rank"
    assert res["A2_criterion"]["verdict"] == "WARN"
    assert res["A3_source"]["verdict"] == "WARN"


def test_optimism_one_candidate():
    """One candidate: no choice, so o_b = metric(resample) - metric(original) averages near zero (bootstrap noise)."""
    env, geno, y, mu, gv, _ = world()
    df = load(frame(env, geno, y, {"only": gv}))
    r = optimism(df, "within_spearman", "only", reps=200)
    assert abs(r["optimism"]) < 0.01 and r["verdict"] == "PASS" and r["pick_stability"] == 1.0


def test_optimism_detects_selection_among_noise():
    """Many pure-noise candidates: picking the best on the reported data is optimistic; the deployable estimate is
    near zero."""
    env, geno, y, mu, gv, rng = world(n_env=20)
    preds = {f"c{i}": rng.normal(size=len(y)) for i in range(40)}
    df = load(frame(env, geno, y, preds))
    from gpverdict_audit.core import candidate_table, pick
    C, _ = candidate_table(df)
    best = pick(C, "within_spearman")
    r = optimism(df, "within_spearman", best, reps=100)
    assert r["optimism"] > 0.01 and abs(r["deploy_estimate"]) < abs(r["reported"])


def test_subset_pooled_pearson_matches_direct():
    env, geno, y, mu, gv, rng = world()
    df = load(frame(env, geno, y, {"a": mu + gv + rng.normal(size=len(y))}))
    from gpverdict_audit.core import env_sums, subset_metric
    S = env_sums(df)
    envs = ["E1", "E3", "E4", "E7"]
    sub = df[df["Env"].isin(envs)]
    assert abs(subset_metric(S, envs, "pooled_pearson")["a"] - np.corrcoef(sub["p"], sub["y"])[0, 1]) < 1e-10
    assert abs(subset_metric(S, envs, "pooled_mse")["a"] - np.mean((sub["p"] - sub["y"]) ** 2)) < 1e-10


def test_baseline_tied_with_itself():
    env, geno, y, mu, gv, _ = world()
    df = load(frame(env, geno, y, {"m": gv}))
    base = load(pd.DataFrame({"environment": env, "genotype": geno, "observed": y, "predicted": gv}), "baseline")
    r = against_baseline(df, "m", base, B=50)
    assert r["d_within_spearman"] == 0 and r["verdict"] == "tied"


def test_html_report_renders_all_checks():
    from gpverdict_audit import render_html, to_json
    env, geno, y, mu, gv, rng = world()
    res = audit(frame(env, geno, y, {"means": mu + rng.normal(0, 0.5, len(y)), "rank": gv}),
                {"criterion": "pooled_pearson", "chosen": "means", "selected_on_report": True}, reps=10)
    page = render_html(res)
    for key in ("A1", "A2", "A3", "A4", "<svg", "FAIL", "deployable estimate"):
        assert key in page
    assert "NaN" not in to_json(res)
