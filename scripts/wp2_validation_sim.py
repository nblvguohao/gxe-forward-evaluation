"""WP2 simulation validation of gpverdict-audit (docs/prereg_wp2_validation_2026-09-30.md). Four pipelines on paired
replicates; writes RESULTS/wp2/validation/{replicates.parquet, verdict.json}. Usage: wp2_validation_sim.py [n_rep] [procs]."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "gpverdict_audit"))
from gpverdict_audit.core import audit, candidate_table, load, pick  # noqa: E402

from dartgxe.paths import RESULTS  # noqa: E402

SEED = 20261001  # re-validation after the 2026-09-30 amendment (first run: 20260930)
OUT = RESULTS / "wp2" / "validation_v2"
J_D = 200


def gen_set(J, n, sig_e2, rng, tag):
    mu = rng.normal(0, np.sqrt(sig_e2), J)
    g = rng.normal(0, 1, n)
    ge = rng.normal(0, np.sqrt(0.5), (J, n))
    e = rng.normal(0, 1, (J, n))
    y = mu[:, None] + g[None, :] + ge + e
    z = (g[None, :] + ge) / np.sqrt(1.5)
    return {"mu": mu, "y": y, "z": z, "env": [f"{tag}E{j}" for j in range(J)], "geno": [f"{tag}g{i}" for i in range(n)]}


def preds(S, par, c, rng):
    J, n = S["y"].shape
    delta = rng.normal(0, par["s"][c], J)
    eps = rng.normal(0, 1, (J, n))
    return S["mu"][:, None] + delta[:, None] + par["t"][c] * (par["rho"][c] * S["z"] + np.sqrt(1 - par["rho"][c] ** 2) * eps)


def long(S, P, setname):
    J, n = S["y"].shape
    parts = []
    for c, p in P.items():
        parts.append(pd.DataFrame({"environment": np.repeat(S["env"], n), "genotype": np.tile(S["geno"], J),
                                   "observed": S["y"].ravel(), "predicted": p.ravel(), "candidate": f"c{c}",
                                   "set": setname}))
    return pd.concat(parts, ignore_index=True)


def metric_on(S, p, metric):
    if metric == "pooled_pearson":
        return float(np.corrcoef(p.ravel(), S["y"].ravel())[0, 1])
    r = [np.corrcoef(pd.Series(p[j]).rank(), pd.Series(S["y"][j]).rank())[0, 1] for j in range(len(p))]
    return float(np.mean(r))


def replicate(r, seed=None):
    rng = np.random.default_rng([SEED if seed is None else seed, r])
    J, n, sig_e2, K = int(rng.choice([8, 16, 32])), int(rng.choice([40, 100])), float(rng.choice([1, 4, 16])), int(rng.choice([10, 40]))
    structure = "tradeoff" if rng.random() < 0.5 else "independent"
    sig = np.sqrt(sig_e2)
    t = np.exp(rng.uniform(np.log(0.05), np.log(3), K))
    if structure == "independent":
        s = rng.uniform(0.05, 1.5, K) * sig
        rho = rng.uniform(0, 0.6, K)
    else:
        q = rng.uniform(0, 1, K)
        s = (0.05 + 1.45 * q) * sig
        rho = np.clip(0.6 * q + rng.normal(0, 0.05, K), 0, 0.6)
    par = {"s": s, "rho": rho, "t": t}
    Rq = rho * np.sqrt(1.5 / 2.5)
    S, R = gen_set(J, n, sig_e2, rng, "S"), gen_set(J, n, sig_e2, rng, "R")
    PS = {c: preds(S, par, c, rng) for c in range(K)}
    PR = {c: preds(R, par, c, rng) for c in range(K)}
    dS, dR = long(S, PS, "selection"), long(R, PR, "report")
    CS, _ = candidate_table(load(dS).assign(set="report"))
    CR, _ = candidate_table(load(dR))
    D = gen_set(J_D, n, sig_e2, rng, "D")
    rows = []
    for name, crit, where in (("P0", "within_spearman", "S"), ("P1", "within_spearman", "R"),
                              ("P2", "pooled_pearson", "S"), ("P3", "pooled_pearson", "R")):
        chosen = pick(CS if where == "S" else CR, crit)
        c = int(chosen[1:])
        if where == "S":
            cand = pd.concat([dS, dR], ignore_index=True)
            manifest = {"criterion": crit, "chosen": chosen, "selection": {"environments": S["env"]},
                        "report": {"environments": R["env"]}}
        else:
            cand = pd.concat([dR.assign(set="selection"), dR], ignore_index=True)
            manifest = {"criterion": crit, "chosen": chosen}
        res = audit(cand, manifest, reps=200, seed=r)  # B = 200 per the amendment
        a4 = res["A4_optimism"]
        T = metric_on(D, preds(D, par, c, rng), crit)
        rows.append({"rep": r, "pipeline": name, "J": J, "n": n, "sig_e2": sig_e2, "K": K, "structure": structure,
                     "chosen": chosen, "R_chosen": float(Rq[c]), "R_max": float(Rq.max()),
                     "true_regret": float(Rq.max() - Rq[c]), "A1": res["A1_overlap"]["verdict"],
                     "A2": res["A2_criterion"]["verdict"], "A3": res["A3_source"]["verdict"],
                     "phi": res["A3_source"]["phi"], "A4": a4["verdict"], "A4_reported": a4.get("reported"),
                     "A4_deploy": a4.get("deploy_estimate"), "A4_optimism": a4.get("optimism"), "T": T})
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan, np.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]


if __name__ == "__main__":
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    if n_rep <= 5:  # smoke test: different seed, separate output, never the pre-registered results
        SEED, OUT = SEED + 999, OUT.parent / "validation_smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(procs) as pool:
        rows = [x for rr in pool.starmap(replicate, [(i, SEED) for i in range(n_rep)]) for x in rr]
    X = pd.DataFrame(rows).sort_values(["rep", "pipeline"])
    X.to_parquet(OUT / "replicates.parquet", index=False)
    v = {}
    a1_fail = X[X.pipeline.isin(["P1", "P3"])]
    a1_clean = X[X.pipeline.isin(["P0", "P2"])]
    v["V1_A1_fail_rate_P1P3"] = wilson(int((a1_fail.A1 == "FAIL").sum()), len(a1_fail))
    v["V1_A1_nonpass_rate_P0P2"] = wilson(int((a1_clean.A1 != "PASS").sum()), len(a1_clean))
    v["V1_pass"] = v["V1_A1_fail_rate_P1P3"][0] >= 0.95 and v["V1_A1_nonpass_rate_P0P2"][0] <= 0.05
    pos = X[X.pipeline.isin(["P2", "P3"]) & (X.true_regret >= 0.05)]
    neg = X[X.true_regret < 0.02]
    v["V2_A2_sensitivity"] = wilson(int((pos.A2 == "WARN").sum()), len(pos))
    v["V2_A2_false_alarm"] = wilson(int((neg.A2 == "WARN").sum()), len(neg))
    v["V2_pass"] = v["V2_A2_sensitivity"][0] >= 0.80 and v["V2_A2_false_alarm"][0] <= 0.10
    lk = X[X.pipeline.isin(["P1", "P3"])].dropna(subset=["A4_deploy", "A4_reported", "T"])
    mae_cf, mae_rep = float((lk.A4_deploy - lk["T"]).abs().mean()), float((lk.A4_reported - lk["T"]).abs().mean())
    v["V3_mae_deploy_estimate"], v["V3_mae_reported"], v["V3_ratio"] = mae_cf, mae_rep, mae_cf / mae_rep
    v["V3_A4_warn_rate_P0P2"] = wilson(int((a1_clean.A4 == "WARN").sum()), len(a1_clean))
    v["V3_pass"] = v["V3_ratio"] <= 0.7 and v["V3_A4_warn_rate_P0P2"][0] <= 0.10
    v["all_pass"] = bool(v["V1_pass"] and v["V2_pass"] and v["V3_pass"])
    v["by_pipeline_rates"] = {p: {k: float((g[k] != "PASS").mean()) for k in ("A1", "A2", "A3", "A4")} for p, g in X.groupby("pipeline")}
    v["n_rep"] = n_rep
    json.dump(v, open(OUT / "verdict.json", "w"), indent=1, default=float)
    print(json.dumps(v, indent=1, default=float))
    print("WP2_VALIDATION_DONE")
