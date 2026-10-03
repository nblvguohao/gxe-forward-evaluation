"""Tuning audit for genomic prediction across environments (design: dart-gxe docs/wp2_audit_module_design_2026-09-30.md).

Given the predictions of every candidate an author chose among (hyperparameter settings, epochs or models), the
split description and optionally a strong baseline, the audit answers:
  A1 was the choice made on the data that are reported?
  A2 is the selection criterion admissible for within-environment selection, and where does its pick rank?
  A3 does the pooled score of the pick come from environment means (phi) or from ranking (u / u*)?
  A4 how much does choosing on the reported data inflate the reported value (optimism bootstrap)?
  A5 is the lead over a strong baseline resolvable (two-level bootstrap, 3/sqrt(N))?
Dependencies: numpy and pandas only (runs under Pyodide)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ALIASES = {"environment": "Env", "env": "Env", "genotype": "k", "k": "k", "line": "k", "observed": "y", "y": "y",
           "predicted": "p", "prediction": "p", "p": "p", "candidate": "cand", "config": "cand", "epoch": "cand",
           "method": "cand", "set": "set", "year": "year"}
HIGHER = {"pooled_pearson": True, "pooled_mse": False, "pooled_huber": False, "within_spearman": True,
          "within_pearson": True}
POOLED = {"pooled_pearson", "pooled_mse", "pooled_huber"}


def load(data, name="candidates"):
    """CSV path or DataFrame -> columns Env, k, y, p, cand, set (default 'report'), optional year."""
    df = pd.read_csv(data) if isinstance(data, (str, bytes)) or hasattr(data, "read") else data.copy()
    df = df.rename(columns={c: ALIASES[c.strip().lower()] for c in df.columns if c.strip().lower() in ALIASES})
    need = ["Env", "k", "y", "p"] + (["cand"] if name == "candidates" else [])
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: missing column(s) {miss}; expected environment, genotype, observed, predicted"
                         + (", candidate" if name == "candidates" else ""))
    if "set" not in df.columns:
        df["set"] = "report"
    keep = [c for c in ("Env", "k", "y", "p", "cand", "set", "year") if c in df.columns]
    df = df[keep].dropna(subset=[c for c in keep if c != "year"])
    for c in ("Env", "k", "cand", "set"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    df["y"] = df["y"].astype(float)
    df["p"] = df["p"].astype(float)
    bad = set(df["set"]) - {"selection", "report"}
    if bad:
        raise ValueError(f"{name}: column 'set' must be 'selection' or 'report', found {sorted(bad)}")
    return df.reset_index(drop=True)


def _ranks(a):
    a = np.asarray(a, dtype=float)
    o = np.argsort(a, kind="mergesort")
    s = a[o]
    edge = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    starts, ends = edge[:-1], edge[1:]
    r = np.empty(len(a))
    r[o] = np.repeat((starts + ends - 1) / 2.0 + 1.0, ends - starts)
    return r


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def n_select(n, frac):
    return max(1, int(round(frac * n)))


def env_scores(y, p, frac=0.10):
    """Within-environment Spearman, Pearson and top-frac selection differential (in SD of y)."""
    if len(y) < 3 or np.unique(p).size < 2 or y.std() == 0:
        return np.nan, np.nan, np.nan
    top = np.argsort(-p, kind="stable")[:n_select(len(y), frac)]
    return _corr(_ranks(y), _ranks(p)), _corr(y, p), float((y[top].mean() - y.mean()) / y.std())


def within_table(df, frac=0.10, min_n=10):
    """One row per candidate and environment (environments with >= min_n genotypes)."""
    rows = []
    for (c, e), g in df.groupby(["cand", "Env"], sort=True):
        if len(g) < min_n:
            continue
        sp, pe, sd = env_scores(g["y"].to_numpy(), g["p"].to_numpy(), frac)
        rows.append({"cand": c, "Env": e, "n": len(g), "spearman": sp, "pearson": pe, "sel_diff": sd})
    return pd.DataFrame(rows)


def decompose(y, p, env):
    """Exact weighted decomposition of pooled Pearson (WP4 proposition): r = (a + k u) / sqrt((v_b + u^2) S)."""
    d = pd.DataFrame({"env": env, "y": y, "p": p})
    g = d.groupby("env")
    w = g.size() / len(d)
    m = g["p"].transform("mean").to_numpy()
    a = float(np.mean((m - m.mean()) * (y - y.mean())))
    vb, S = float(m.var()), float(y.var())
    tj = g["p"].std(ddof=0)
    cov = g.apply(lambda q: float(np.mean((q.p - q.p.mean()) * (q.y - q.y.mean()))), include_groups=False)
    u = float(np.sqrt((w * tj ** 2).sum()))
    k = float((w * cov).sum() / u) if u > 0 else 0.0
    ok = a > 0 and vb > 0
    between = a * a / vb if ok else np.nan
    return {"a": a, "v_b": vb, "k": k, "u": u, "S": S, "u_star": k * vb / a if ok else np.nan,
            "phi": between / (between + k * k) if ok else np.nan,
            "r_max": float(np.sqrt((between + k * k) / S)) if ok else np.nan,
            "pooled_pearson": _corr(y, p)}


def _huber(e, delta=1.0):
    a = np.abs(e)
    return float(np.mean(np.where(a <= delta, 0.5 * e * e, delta * (a - 0.5 * delta))))


def candidate_table(df, frac=0.10, min_n=10):
    """Per candidate: pooled Pearson, MSE, Huber; mean within-env Spearman, Pearson, selection differential; phi, u/u*."""
    W = within_table(df, frac, min_n)
    envs = set(W["Env"])
    rows = []
    for c, g in df[df["Env"].isin(envs)].groupby("cand", sort=True):
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        dec = decompose(y, p, g["Env"].to_numpy())
        w = W[W["cand"] == c]
        rows.append({"cand": c, "cells": len(g), "envs": int(w["Env"].nunique()), "pooled_pearson": dec["pooled_pearson"],
                     "pooled_mse": float(np.mean((p - y) ** 2)), "pooled_huber": _huber(p - y),
                     "within_spearman": float(w["spearman"].mean()), "within_pearson": float(w["pearson"].mean()),
                     "within_sel_diff": float(w["sel_diff"].mean()), "phi": dec["phi"], "u": dec["u"],
                     "u_star": dec["u_star"],
                     "log_u_over_ustar": float(np.log(dec["u"] / dec["u_star"])) if dec["u_star"] and dec["u_star"] > 0 else np.nan})
    return pd.DataFrame(rows).set_index("cand"), W


def pick(C, criterion):
    col = C[criterion]
    return str(col.idxmax() if HIGHER[criterion] else col.idxmin())


def resolution(n_cells, z=3.0):
    return z / np.sqrt(n_cells) if n_cells > 0 else np.nan


# ------------------------------------------------------------------------------------------------ checks
def check_overlap(df, manifest):
    """A1: does the selection set share cells, environments or years with the report set?"""
    sel, rep = df[df["set"] == "selection"], df[df["set"] == "report"]
    out = {"selection_rows": int(len(sel)), "report_rows": int(len(rep))}
    m_sel = manifest.get("selection", {}) or {}
    m_rep = manifest.get("report", {}) or {}
    if len(sel):
        cs, cr = set(zip(sel["Env"], sel["k"])), set(zip(rep["Env"], rep["k"]))
        es, er = set(sel["Env"]), set(rep["Env"])
        out.update({"shared_cells": len(cs & cr), "shared_environments": len(es & er),
                    "shared_genotypes": len(set(sel["k"]) & set(rep["k"]))})
        if "year" in df.columns:
            out["shared_years"] = len(set(sel["year"].dropna()) & set(rep["year"].dropna()))
        source = "data"
    else:
        out["note"] = "only report predictions supplied; overlap judged from the manifest"
        source = "manifest"
    for key in ("environments", "years", "genotypes"):
        a, b = set(map(str, m_sel.get(key, []) or [])), set(map(str, m_rep.get(key, []) or []))
        if a and b:
            out[f"manifest_shared_{key}"] = len(a & b)
    same_flag = bool(manifest.get("selected_on_report", False))
    shared = [out.get("shared_cells", 0), out.get("shared_environments", 0), out.get("shared_years", 0),
              out.get("manifest_shared_environments", 0), out.get("manifest_shared_years", 0)]
    if same_flag or any(v > 0 for v in shared):
        out["verdict"] = "FAIL"
        out["message"] = ("The choice among candidates used the reported environments or years; the reported value is "
                          "not a deployment estimate.")
    elif source == "manifest" and not (m_sel and m_rep):
        out["verdict"] = "UNKNOWN"
        out["message"] = "No selection data and no split description: cannot tell whether the choice used the reported data."
    else:
        out["verdict"] = "PASS"
        out["message"] = "Selection and report data do not share environments, years or cells."
        if out.get("shared_genotypes", 0) or out.get("manifest_shared_genotypes", 0):
            out["message"] += " Genotypes are shared (known-genotype scenario); state this when reporting."
    return out


def check_criterion(C, criterion, chosen, n_cells):
    """A2: which candidate each criterion picks; rank and regret of the author's pick on within-env Spearman."""
    picks = {c: pick(C, c) for c in HIGHER}
    ws = C["within_spearman"]
    rank = int((ws > ws.loc[chosen]).sum() + 1)
    regret = float(ws.max() - ws.loc[chosen])
    tau = {}
    for c in POOLED:
        a = C[c] if HIGHER[c] else -C[c]
        tau[c] = _corr(_ranks(a.to_numpy()), _ranks(ws.to_numpy()))
    res = resolution(n_cells)
    out = {"criterion": criterion, "chosen": chosen, "picks": picks, "chosen_rank_within_spearman": rank,
           "candidates": int(len(C)), "regret_within_spearman": regret, "resolution": res,
           "rank_agreement_with_within_spearman": tau}
    if criterion in POOLED and regret > res:
        out["verdict"] = "WARN"
        out["message"] = (f"The pooled criterion picked a candidate ranked {rank} of {len(C)} on within-environment "
                          f"Spearman, {regret:.3f} below the best (resolution {res:.3f}).")
    elif criterion not in HIGHER:
        out["verdict"] = "UNKNOWN"
        out["message"] = "Criterion not recognised; only within-environment scores are reported."
    else:
        out["verdict"] = "PASS"
        out["message"] = "The criterion is within-environment, or its pick is within resolution of the best."
    return out


def check_source(C, chosen, criterion="pooled_pearson", phi_warn=0.9):
    """A3: share of the pooled score that environment means can explain, and the scale relative to u*. Warns only when
    the author selected (and so presumably reports) by a pooled criterion."""
    r = C.loc[chosen]
    out = {"phi": r["phi"], "u": r["u"], "u_star": r["u_star"], "log_u_over_ustar": r["log_u_over_ustar"],
           "median_phi_all_candidates": float(C["phi"].median())}
    if criterion not in POOLED:
        out["verdict"] = "PASS"
        out["message"] = f"Selection used a within-environment criterion; phi = {r['phi']:.2f} is shown for information only."
    elif np.isfinite(r["phi"]) and r["phi"] >= phi_warn:
        out["verdict"] = "WARN"
        out["message"] = (f"phi = {r['phi']:.2f}: at least {phi_warn:.0%} of the attainable pooled r^2 comes from "
                          "environment means, so the pooled score says little about ranking within environments.")
    else:
        out["verdict"] = "PASS"
        out["message"] = "Environment means do not dominate the pooled score."
    return out


def env_sums(df, frac=0.10, min_n=10):
    """Per candidate and scorable environment: sufficient statistics for pooled Pearson, MSE and Huber over any
    subset of environments, plus the within-environment Spearman."""
    rows = []
    for (c, e), g in df.groupby(["cand", "Env"], sort=True):
        if len(g) < min_n:
            continue
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        rows.append({"cand": c, "Env": e, "n": len(y), "sy": y.sum(), "sp": p.sum(), "syy": (y * y).sum(),
                     "spp": (p * p).sum(), "syp": (y * p).sum(), "sse": ((p - y) ** 2).sum(),
                     "shub": _huber(p - y) * len(y), "spearman": env_scores(y, p, frac)[0]})
    return pd.DataFrame(rows)


def weighted_metric(S, counts, criterion):
    """Criterion value of every candidate on a multiset of environments (counts: environment -> multiplicity)."""
    w = S["Env"].map(counts).fillna(0.0)
    T = S.assign(w=w)
    T = T[T["w"] > 0]
    if criterion == "within_spearman":
        return (T["spearman"] * T["w"]).groupby(T["cand"]).sum() / T.groupby("cand")["w"].sum()
    cols = ["n", "sy", "sp", "syy", "spp", "syp", "sse", "shub"]
    a = T[cols].mul(T["w"], axis=0).groupby(T["cand"]).sum()
    if criterion == "pooled_pearson":
        num = a["n"] * a["syp"] - a["sy"] * a["sp"]
        den = np.sqrt((a["n"] * a["syy"] - a["sy"] ** 2) * (a["n"] * a["spp"] - a["sp"] ** 2))
        return num / den
    if criterion == "pooled_mse":
        return a["sse"] / a["n"]
    if criterion == "pooled_huber":
        return a["shub"] / a["n"]
    raise ValueError(criterion)


def subset_metric(S, envs, criterion):
    """Criterion value of every candidate on a set of environments (from env_sums)."""
    return weighted_metric(S, pd.Series(1.0, index=list(envs)), criterion)


def optimism(df, criterion, chosen, frac=0.10, min_n=10, reps=200, seed=0):
    """A4 (Efron 1983 optimism bootstrap, amendment of 2026-09-30): resample the report environments with replacement
    (same number), pick by the author's criterion on the resample, and take o_b = metric(resample, pick) -
    metric(original, pick). The mean of o_b estimates how much selecting on the reported data inflates the reported
    value; deployable estimate = reported - optimism (sign reversed for losses)."""
    if criterion not in HIGHER:
        return {"verdict": "UNKNOWN", "message": f"criterion {criterion!r} not recognised; optimism not estimated"}
    if criterion == "within_pearson":
        criterion = "within_spearman"
    rep = df[df["set"] == "report"]
    S = env_sums(rep, frac, min_n)
    envs = np.array(sorted(S["Env"].unique()))
    if len(envs) < 4:
        return {"verdict": "UNKNOWN", "message": "fewer than 4 scorable report environments; optimism not estimated"}
    sign = 1.0 if HIGHER[criterion] else -1.0
    full = weighted_metric(S, pd.Series(1.0, index=envs), criterion)
    rng = np.random.default_rng(seed)
    o, picks = [], []
    for _ in range(reps):
        cnt = pd.Series(envs[rng.integers(0, len(envs), len(envs))]).value_counts().astype(float)
        mb = weighted_metric(S, cnt, criterion)
        c = str(mb.idxmax() if HIGHER[criterion] else mb.idxmin())
        o.append(sign * (mb[c] - full[c]))
        picks.append(c)
    o = np.array(o)
    opt = float(o.mean())
    reported = float(full[chosen])
    n_cells = int(S[S["cand"] == chosen]["n"].sum())
    res = resolution(n_cells)
    lo, hi = float(np.quantile(o, 0.025)), float(np.quantile(o, 0.975))
    out = {"metric": criterion, "reported": reported, "optimism": opt, "optimism_interval": [lo, hi],
           "deploy_estimate": reported - sign * opt, "resolution": res, "bootstrap_reps": reps,
           "pick_stability": float(pd.Series(picks).value_counts().iloc[0] / reps)}
    big = opt > res if criterion in ("pooled_pearson", "within_spearman") else lo > 0
    if big:
        out["verdict"] = "WARN"
        out["message"] = (f"Choosing among the candidates on the reported data inflates {criterion} by about {opt:.3f}; "
                          f"the deployable estimate is {out['deploy_estimate']:.3f} (reported {reported:.3f}).")
    else:
        out["verdict"] = "PASS"
        out["message"] = (f"Estimated inflation of {criterion} from choosing on the reported data is {opt:.3f}, within "
                          f"resolution; deployable estimate {out['deploy_estimate']:.3f}.")
    return out


def against_baseline(df, chosen, base, frac=0.10, min_n=10, B=500, seed=0):
    """A5: chosen candidate minus baseline on within-env Spearman and selection differential; two-level bootstrap."""
    rep = df[(df["set"] == "report") & (df["cand"] == chosen)][["Env", "k", "y", "p"]]
    m = rep.merge(base[["Env", "k", "p"]].rename(columns={"p": "b"}), on=["Env", "k"], how="inner")
    groups = [(g["y"].to_numpy(), g["p"].to_numpy(), g["b"].to_numpy()) for _, g in m.groupby("Env") if len(g) >= min_n]
    if not groups:
        return {"verdict": "UNKNOWN", "message": "no report cells shared with the baseline"}

    def diff(gs):
        s = [(env_scores(y, p, frac)[0] - env_scores(y, b, frac)[0], env_scores(y, p, frac)[2] - env_scores(y, b, frac)[2])
             for y, p, b in gs]
        s = np.array(s, dtype=float)
        return np.nanmean(s[:, 0]), np.nanmean(s[:, 1])

    est = diff(groups)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        gs = []
        for i in rng.integers(0, len(groups), len(groups)):
            y, p, b = groups[i]
            j = rng.integers(0, len(y), len(y))
            gs.append((y[j], p[j], b[j]))
        bs.append(diff(gs))
    bs = np.array(bs)
    n_cells = int(sum(len(g[0]) for g in groups))
    res = resolution(n_cells)
    lo, hi = np.nanquantile(bs[:, 0], [0.025, 0.975])
    verdict = ("better" if lo > 0 and est[0] >= res else "worse" if hi < 0 and -est[0] >= res else
               "detectable, below resolution" if (lo > 0 or hi < 0) else "tied")
    return {"d_within_spearman": float(est[0]), "interval": [float(lo), float(hi)],
            "d_sel_diff": float(est[1]), "interval_sel_diff": [float(x) for x in np.nanquantile(bs[:, 1], [0.025, 0.975])],
            "cells": n_cells, "environments": len(groups), "resolution": res, "verdict": verdict,
            "message": f"Chosen candidate minus baseline: {est[0]:+.3f} within-environment Spearman "
                       f"[{lo:+.3f}, {hi:+.3f}], resolution {res:.3f} -> {verdict}."}


def audit(candidates, manifest, baseline=None, frac=0.10, min_n=10, reps=200, B=500, seed=0):
    df = load(candidates)
    manifest = dict(manifest)
    criterion = manifest.get("criterion", "other")
    rep = df[df["set"] == "report"]
    if rep.empty:
        raise ValueError("no report rows: mark reported predictions with set = 'report' (default when the column is absent)")
    C, W = candidate_table(rep, frac, min_n)
    chosen = str(manifest.get("chosen") or (pick(C, criterion) if criterion in HIGHER else C.index[0]))
    if chosen not in C.index:
        raise ValueError(f"chosen candidate {chosen!r} has no scorable report predictions")
    n_cells = int(W[W["cand"] == chosen]["n"].sum())
    out = {"frac": frac, "min_genotypes": min_n, "report_cells_chosen": n_cells,
           "A1_overlap": check_overlap(df, manifest),
           "A2_criterion": check_criterion(C, criterion, chosen, n_cells),
           "A3_source": check_source(C, chosen, criterion),
           "A4_optimism": optimism(df, criterion, chosen, frac, min_n, reps, seed)}
    if baseline is not None:
        out["A5_baseline"] = against_baseline(df, chosen, load(baseline, "baseline"), frac, min_n, B, seed)
    out["candidates"] = C.reset_index().to_dict(orient="records")
    flags = [v.get("verdict") for k, v in out.items() if k.startswith("A") and k != "A5_baseline"]
    out["summary"] = ("Reported value is not a deployment estimate (selection used the reported data)."
                      if "FAIL" in flags else
                      "Reported value is optimistic or rests on an inadmissible criterion; report the cross-fitted "
                      "within-environment value." if "WARN" in flags else
                      "No tuning problem detected by these checks.")
    return out
