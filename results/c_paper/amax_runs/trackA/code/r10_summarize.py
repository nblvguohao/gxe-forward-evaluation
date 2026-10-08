"""R10 track A summariser (prereg v10, Part G, 2k models): G1 (D_noEC - MG_2k, D_full - MG_2k), G2 (D_full - D_noEC),
G3 (CERIS window-search leak) and G4 (tuning audit) on 2024 alone and on EXT3 = 2022 + 2023 (R9 cells, re-scored from
the saved per-env predictions, not refit) + 2024 (R10 fold). Env bootstrap stratified by year (2000, seed 20260928).
Writes R10_A_results.json (+ .sha256) and per-env CSVs into <final_dir>.
Usage: python r10_summarize.py <r9_code_dir> <runs_A_dir> <runs_trackA_dir> <final_dir>"""
import os, sys, json, pickle, time, hashlib
import numpy as np, pandas as pd
CODE9, RUNA, RUN, FIN = sys.argv[1:5]
for f in ("a_lib.py", "v5_lib.py", "v7_lib.py"):
    exec(open(os.path.join(CODE9, f)).read())
os.makedirs(FIN, exist_ok=True)
PRE10 = os.path.join(RUN, "prereg", "prereg_v10_final_confirmation.json")
PRE10_SHA = "dfc8e990551241c72d04e3d74f2526dca52209d677884bbcd88edfb6644e015b"
assert sha256_file(PRE10) == PRE10_SHA
pre = json.load(open(PRE10)); PG = pre["part_G_G2F_2024"]
R9J = os.path.join(RUNA, "final", "R9_A_results.json")
assert sha256_file(R9J) == "f946231b4707af79e398163027f07c9f141dbfcd8a32d5a900e69edc88403a4b"
R9 = json.load(open(R9J))
# Declared before any 2024 result is summarised (see also r10_deviations.json).
OVERALL_MAPPING = ("GO if D_noEC - MG_2k or D_full - MG_2k is USEFUL on EXT3 (an external enhancement exists); "
                   "WEAK_GO if neither is USEFUL but G2 is CONFIRMED; NO_GO if neither is USEFUL and G2 is not CONFIRMED. "
                   "NO_GO here is the outcome that supports C_final_no_external_gain for the track-A contrasts; the family claim "
                   "also needs track B's MG_98k - MG_2k.")
R = lambda x, k=6: None if x is None or not np.isfinite(x) else round(float(x), k)
MODELS, PROT = ["MG_2k", "D_noEC", "D_full"], ["P1", "P2", "P3"]
NITER = [50, 100, 200, 400, 800]; L = 19; B = 2000; YEARS = [2022, 2023, 2024]
V8_GRID = np.array(pre["tuning"]["grid"])
CELLF = {2022: os.path.join(RUNA, "out", "cells", "EXT_2022__2022.pkl"), 2023: os.path.join(RUNA, "out", "cells", "EXT_2023__2023.pkl"),
         2024: os.path.join(RUN, "out", "cells", "EXT_2024__2024.pkl")}
cells = {y: pickle.load(open(f, "rb")) for y, f in CELLF.items()}
cell_sha = {y: sha256_file(f) for y, f in CELLF.items()}


def per_env_table(c):
    """per-env r at each protocol's selected index, read from the cell's saved per-env path (float64, as R9 reported);
    checked against a recomputation from the saved float32 test predictions."""
    tk = c["test_keys"]; env, y, new = tk.Env.values, tk.y.values, tk.new_hybrid.values; tenv = list(c["test_envs"])
    cols, nh, chk = {}, {}, 0.0
    for m in MODELS:
        for p_ in PROT:
            k = c["sel"][m][p_]; s = c["test_path"][m].iloc[:, k].loc[tenv]; cols[f"{m}|{p_}"] = s
            pred = c["test_pred"][(m, p_)].astype(float)
            chk = max(chk, float(np.nanmax(np.abs(per_env_r(env, y, pred[:, None], tenv)[0].values - s.values))))
            ne = [e for e in tenv if (new & (env == e)).sum() >= 10]
            if ne and p_ == "P3":
                nh[f"{m}|{p_}"] = per_env_r(env[new], y[new], pred[new][:, None], ne)[0]
    W = pd.DataFrame(cols); NH = pd.DataFrame(nh)
    return W, NH, chk


W, NH, rescore = {}, {}, {}
for y in YEARS:
    W[y], NH[y], chk = per_env_table(cells[y]); rescore[y] = dict(max_abs_diff_saved_path_vs_recomputed_from_float32_preds=chk)
    assert chk < 1e-5
for y in (2022, 2023):
    csv = pd.read_csv(os.path.join(RUNA, "final", f"R9_A_EXT_{y}_per_env_r.csv"), index_col=0)
    d = float(np.nanmax(np.abs(W[y][csv.columns].loc[csv.index].values - csv.values)))
    rescore[y]["max_abs_diff_vs_R9_csv_6dp"] = d; rescore[y]["same_env_set_as_R9"] = bool(set(csv.index) == set(W[y].index)); assert d < 1e-6
NY = {y: len(W[y]) for y in YEARS}
rng = np.random.default_rng(SEED); IDX = {y: rng.integers(0, NY[y], size=(B, NY[y])) for y in YEARS}   # one draw set, reused (paired)


def sboot(dy, years=YEARS):
    """dy: {year: per-env values}; pooled equal-env-weight mean, stratified env bootstrap (percentile CI, one-sided p)."""
    d_all = np.concatenate([np.asarray(dy[y], float) for y in years])
    bm = np.concatenate([np.asarray(dy[y], float)[IDX[y]] for y in years], axis=1).mean(1)
    return dict(mean=R(d_all.mean()), ci_low=R(np.percentile(bm, 2.5)), ci_high=R(np.percentile(bm, 97.5)), n_env=int(len(d_all)),
                n_env_by_year={int(y): int(len(dy[y])) for y in years}, p_one_sided_gt_0=R((1 + np.sum(bm <= 0)) / (B + 1)))


def yboot(d):
    b = boot_mean_diff(np.asarray(d, float))
    return dict(mean=R(b["mean"]), ci_low=R(b["ci_low"]), ci_high=R(b["ci_high"]), n_env=b["n_env"], p_one_sided_gt_0=R(boot_p_one_sided(np.asarray(d, float))))


def contrast(a, b, T=W):
    dy = {y: (T[y][a] - T[y][b]).values for y in YEARS}
    per_year = {int(y): dict(**yboot(dy[y]), sign="+" if np.mean(dy[y]) > 0 else ("-" if np.mean(dy[y]) < 0 else "0")) for y in YEARS}
    return dict(quantity=f"r({a}) - r({b})", EXT3_pooled=sboot(dy), per_year=per_year,
                n_years_positive=int(sum(np.mean(dy[y]) > 0 for y in YEARS)), year_2024_alone=per_year[2024])


CON = {"D_noEC_minus_MG_2k": contrast("D_noEC|P3", "MG_2k|P3"), "D_full_minus_MG_2k": contrast("D_full|P3", "MG_2k|P3"),
       "D_full_minus_D_noEC": contrast("D_full|P3", "D_noEC|P3")}
useful = lambda q: bool(q["EXT3_pooled"]["mean"] >= 0.01 and q["EXT3_pooled"]["ci_low"] > 0 and q["n_years_positive"] >= 2)
G1 = {k: dict(**CON[k], USEFUL=useful(CON[k]), verdict="USEFUL" if useful(CON[k]) else "NOT_USEFUL") for k in ("D_noEC_minus_MG_2k", "D_full_minus_MG_2k")}
q2 = CON["D_full_minus_D_noEC"]; P2_ = q2["EXT3_pooled"]
g2_conf = bool(P2_["mean"] >= 0.005 and P2_["ci_low"] > 0 and q2["n_years_positive"] >= 2); g2_negl = bool(P2_["ci_high"] < 0.01)
G2 = dict(**q2, verdict="CONFIRMED" if g2_conf else ("NEGLIGIBLE" if g2_negl else "INCONCLUSIVE"))

# ---------------- G3: CERIS window-search leak (env-mean r, leaky - nested)
EM = {}
for y in YEARS:
    e = cells[y]["ceris_env_mean"].copy(); e["year"] = y; EM[y] = e
ceris_chk = {}
for y in (2022, 2023):
    csv = pd.read_csv(os.path.join(RUNA, "final", f"R9_A_ceris_EXT_{y}_env_means.csv"), index_col=0)
    ceris_chk[int(y)] = float(np.nanmax(np.abs(EM[y][["obs", "C2_nested", "C2_leaky"]].loc[csv.index].values - csv[["obs", "C2_nested", "C2_leaky"]].values)))
    assert ceris_chk[int(y)] < 1e-5
nan_drop = {int(y): sorted(EM[y].index[EM[y][["obs", "C2_nested", "C2_leaky"]].isna().any(axis=1)].tolist()) for y in YEARS}
EMc = {y: EM[y].dropna(subset=["obs", "C2_nested", "C2_leaky"]) for y in YEARS}


def rr(p, o):
    pc = p - p.mean(1, keepdims=True); oc = o - o.mean(1, keepdims=True)
    return (pc * oc).sum(1) / np.sqrt((pc ** 2).sum(1) * (oc ** 2).sum(1))


def ceris_boot(years, stratified):
    E = pd.concat([EMc[y] for y in years]); o, pn, pk = (E[c].values.astype(float) for c in ("obs", "C2_nested", "C2_leaky"))
    if stratified:
        r2 = np.random.default_rng(SEED); off = np.cumsum([0] + [len(EMc[y]) for y in years])[:-1]
        idx = np.concatenate([r2.integers(0, len(EMc[y]), size=(B, len(EMc[y]))) + off[i] for i, y in enumerate(years)], axis=1)
    else:
        idx = np.random.default_rng(SEED).integers(0, len(E), size=(B, len(E)))
    diff = rr(pk[idx], o[idx]) - rr(pn[idx], o[idx])
    r_n, r_k = pearson(pn, o), pearson(pk, o)
    return dict(n_env=int(len(E)), r_envmean_nested=R(r_n), r_envmean_leaky=R(r_k), optimism=R(r_k - r_n),
                ci=[R(np.nanpercentile(diff, 2.5)), R(np.nanpercentile(diff, 97.5))], p_one_sided_gt_0=R((1 + np.sum(diff <= 0)) / (B + 1)),
                bootstrap="stratified by year" if stratified else "unstratified (single year)")


G3 = dict(procedure=PG["G3_window_search_leak"]["procedure"], rule=PG["G3_window_search_leak"]["REPLICATED_EXTERNALLY"],
          EXT3_pooled=ceris_boot(YEARS, True), per_year={int(y): ceris_boot([y], False) for y in YEARS},
          selected={int(y): cells[y]["ceris_selected"] for y in YEARS}, envs_dropped_nan_prediction=nan_drop,
          ceris_candidates_finite_2024=cells[2024].get("ceris_candidates_finite"), rescore_check_max_abs_diff_vs_R9_csv=ceris_chk)
for y in YEARS: G3["per_year"][int(y)]["sign"] = "+" if G3["per_year"][int(y)]["optimism"] > 0 else ("-" if G3["per_year"][int(y)]["optimism"] < 0 else "0")
G3["year_2024_alone"] = G3["per_year"][2024]
pc = G3["EXT3_pooled"]; G3["verdict"] = "REPLICATED_EXTERNALLY" if (pc["optimism"] >= 0.05 and pc["ci"][0] > 0) else "NOT_REPLICATED_EXTERNALLY"

# ---------------- G4: tuning audit (descriptive)
G4 = {}
for m in MODELS:
    q = {}
    for a, b in (("P2", "P3"), ("P1", "P3")):
        dy = {y: (W[y][f"{m}|{a}"] - W[y][f"{m}|{b}"]).values for y in YEARS}
        q[f"{a}_minus_{b}"] = dict(EXT3_pooled=sboot(dy), per_year={int(y): yboot(dy[y]) for y in YEARS})
    q["chosen"] = {p_: {int(y): (float(V8_GRID[cells[y]["sel"][m][p_]]) if m == "MG_2k" else NITER[cells[y]["sel"][m][p_]]) for y in YEARS} for p_ in PROT}
    q["edge_P3"] = {int(y): bool(cells[y]["sel"][m]["P3"] in ((0, L - 1) if m == "MG_2k" else (0, len(NITER) - 1))) for y in YEARS}
    q["years_P2_below_P3"] = [int(y) for y in YEARS if W[y][f"{m}|P2"].mean() < W[y][f"{m}|P3"].mean() - 1e-12]
    G4[m] = q

# ---------------- r table, new-hybrid subset (post-hoc, descriptive), 2024 fold facts
rtab = {"EXT3_pooled": {k: R(np.concatenate([W[y][k].values for y in YEARS]).mean()) for k in W[2024].columns}}
for y in YEARS: rtab[str(y)] = {k: R(W[y][k].mean()) for k in W[y].columns}
nh = {}
for y in YEARS:
    T = NH[y]
    nh[str(y)] = dict(n_env=int(len(T))) if len(T) == 0 else dict(n_env=int(len(T)), r_P3={m: R(T[f"{m}|P3"].mean()) for m in MODELS},
                                                                  D_noEC_minus_MG_2k=yboot((T["D_noEC|P3"] - T["MG_2k|P3"]).values),
                                                                  D_full_minus_MG_2k=yboot((T["D_full|P3"] - T["MG_2k|P3"]).values),
                                                                  D_full_minus_D_noEC=yboot((T["D_full|P3"] - T["D_noEC|P3"]).values))
c24 = cells[2024]; tk24 = c24["test_keys"]
fold2024 = dict(n_train_records=int(c24["n_train_records"]), n_train_envs=int(c24["n_train_envs"]), n_test_records=int(c24["n_test_records"]),
                n_test_envs=len(c24["test_envs"]), test_envs=list(c24["test_envs"]), p1_key=c24["p1_key"], inner_keys=[int(k) for k in c24["inner_keys"]],
                maturity=c24["maturity"], stage1_cv_r2=R(c24["stage1_cv_r2"], 4),
                new_hybrid_fraction=R(float(tk24[tk24.Env.isin(c24["test_envs"])].new_hybrid.mean()), 4),
                ec_window_complete_fraction=c24.get("ec_window_complete_fraction"), computed_on=c24.get("computed_on"))
data24 = json.load(open(os.path.join(RUN, "out", "R10_A_data_2024.json")))
fam = not any(G1[k]["USEFUL"] for k in G1)
overall = "GO" if not fam else ("WEAK_GO" if G2["verdict"] == "CONFIRMED" else "NO_GO")
dev = json.load(open(os.path.join(RUN, "code", "r10_deviations.json")))
res = dict(track="R10_A_partG_2k_G1_G2_G3_G4", generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           prereg=dict(file="prereg_v10_final_confirmation.json", sha256=PRE10_SHA, frozen_at_utc=pre["frozen_at_utc"]),
           compute=dict(host="amax (x86_64, egp311)", note="2022/2023 folds from R9 (amax, 3 x 8 threads); 2024 fold on amax, 1 x 24 threads; every setting's folds on one machine",
                        computed_on={int(y): cells[y].get("computed_on") for y in YEARS}),
           inputs=dict(R9_A_results_sha256=sha256_file(R9J), cell_sha256={int(y): s for y, s in cell_sha.items()}, rescore_checks={int(y): v for y, v in rescore.items()}),
           data_2024=data24, fold_2024=fold2024, n_env_by_year={int(y): NY[y] for y in YEARS},
           r_mean=rtab, G1_enhancements_external=dict(rule=PG["G1_enhancements_external"]["USEFUL"], contrasts=G1,
                                                     trackA_part_of_family_claim_holds=fam,
                                                     note="family claim C_final_no_external_gain also requires track B's MG_98k - MG_2k"),
           G2_EC_increment=dict(rule_confirmed=PG["G2_EC_increment"]["CONFIRMED"], rule_negligible=PG["G2_EC_increment"]["NEGLIGIBLE"], **G2),
           G3_window_search_leak=G3, G4_tuning_audit=G4,
           POST_HOC_new_hybrid_subset=dict(label="POST-HOC, descriptive only: test envs with >= 10 test hybrids absent from the fold's training records", **nh),
           uncertainty_note=PG["uncertainty"], multiplicity="prereg v10 specifies no multiplicity correction; one-sided bootstrap p reported for information",
           overall_verdict=overall, overall_verdict_mapping_declared=OVERALL_MAPPING, deviations_from_prereg=dev,
           code_sha256={f: sha256_file(os.path.join(RUN, "code", f)) for f in sorted(os.listdir(os.path.join(RUN, "code"))) if f.endswith(".py")})
fo = os.path.join(FIN, "R10_A_results.json")
json.dump(res, open(fo, "w"), indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
open(fo + ".sha256", "w").write(f"{sha256_file(fo)}  R10_A_results.json\n")
pe = pd.concat([W[y].assign(year=y) for y in YEARS]); pe.index.name = "env"; pe.round(6).to_csv(os.path.join(FIN, "R10_A_EXT3_per_env_r.csv"))
pd.concat([EM[y] for y in YEARS]).round(6).to_csv(os.path.join(FIN, "R10_A_ceris_EXT3_env_means.csv"))
print(json.dumps(dict(G1={k: G1[k]["verdict"] for k in G1}, G2=G2["verdict"], G3=G3["verdict"], overall=overall,
                      pooled={k: CON[k]["EXT3_pooled"]["mean"] for k in CON}, per_year={k: {y: CON[k]["per_year"][y]["mean"] for y in YEARS} for k in CON})))
