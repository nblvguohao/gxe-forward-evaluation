"""R9 track A summariser (run on amax): H9a / H9b gates on S4_G2F_CV00 and EXT_pooled (EXT_2022, EXT_2023 reported),
secondary CERIS window-search leak on EXT_pooled, tuning audit, new-hybrid subset, data-consistency report.
Writes R9_A_results.json (+ .sha256) and per-env CSVs into <final_dir>.
Usage: python v9_summarize.py <code_dir> <run_dir> <final_dir>"""
import os, sys, json, pickle, time, glob, hashlib
import numpy as np, pandas as pd
CODE, RUN, FIN = sys.argv[1:4]
for f in ("a_lib.py", "v5_lib.py", "v7_lib.py"):
    exec(open(os.path.join(CODE, f)).read())
os.makedirs(FIN, exist_ok=True)
PRE9 = os.path.join(RUN, "prereg", "prereg_v9_mainline_external.json")
assert sha256_file(PRE9) == "6ed45e2bd64c88226fe72360e0b8405233cc84b03b1b604330eaa000dd99aa09"
pre9 = json.load(open(PRE9)); H = pre9["hypotheses"]
R = lambda x, k=6: None if x is None or not np.isfinite(x) else round(float(x), k)
MODELS, PROT = ["MG_2k", "D_noEC", "D_full"], ["P1", "P2", "P3"]
NITER = [50, 100, 200, 400, 800]; L = 19


def boot(d):
    b = boot_mean_diff(np.asarray(d, float))
    return dict(mean=R(b["mean"]), ci_low=R(b["ci_low"]), ci_high=R(b["ci_high"]), n_env=b["n_env"], p_one_sided_gt_0=R(boot_p_one_sided(np.asarray(d, float))))


cells = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(os.path.join(RUN, "out", "cells", "*.pkl")))]
by_set = {}
for c in cells: by_set.setdefault(c["setting"], []).append(c)
fold_list = {s: dict(n_folds_total=len(v), n_skipped=sum(c["skipped"] for c in v), skipped=[c["key"] for c in v if c["skipped"]],
                     run=[c["key"] for c in v if not c["skipped"]]) for s, v in by_set.items()}


def table(cs):
    rows, nh, edge, chk, chosen = [], [], {}, [], {}
    for c in cs:
        if c["skipped"]: continue
        tk = c["test_keys"]; env, y, new = tk.Env.values, tk.y.values, tk.new_hybrid.values; tenv = c["test_envs"]
        fold_mean = {}
        for m in MODELS:
            for p_ in PROT:
                k = c["sel"][m][p_]; s = c["test_path"][m].iloc[:, k]
                pred = c["test_pred"][(m, p_)].astype(float)
                assert np.nanmax(np.abs(per_env_r(env, y, pred[:, None], tenv)[0].values - s.loc[tenv].values)) < 1e-5
                for e, v in s.items(): rows.append(dict(setting=c["setting"], fold=c["key"], env=e, key=f"{m}|{p_}", r=v))
                ne = [e for e in tenv if (new & (env == e)).sum() >= 10]
                if ne and p_ == "P3":
                    for e, v in per_env_r(env[new], y[new], pred[new][:, None], ne)[0].items(): nh.append(dict(env=e, key=f"{m}|{p_}", r=v))
                fold_mean[f"{m}|{p_}"] = float(s.mean())
                is_edge = k in ((0, L - 1) if m == "MG_2k" else (0, len(NITER) - 1))
                edge.setdefault(f"{m}|{p_}", []).append(bool(is_edge))
                chosen.setdefault(f"{m}|{p_}", {})[c["key"]] = (float(V8_GRID[k]) if m == "MG_2k" else NITER[k])
        chk.append(dict(fold=c["key"], **{m: dict(P2_below_P3=fold_mean[f"{m}|P2"] < fold_mean[f"{m}|P3"] - 1e-12,
                                                   P2_below_P1=fold_mean[f"{m}|P2"] < fold_mean[f"{m}|P1"] - 1e-12) for m in MODELS}))
    pe = pd.DataFrame(rows); assert not pe.duplicated(["env", "key"]).any()
    W = pe.pivot(index="env", columns="key", values="r")
    NH = pd.DataFrame(nh).pivot(index="env", columns="key", values="r") if nh else pd.DataFrame()
    return W, NH, edge, chk, chosen


V8_GRID = np.array([0.01, 0.031623, 0.1, 0.316228, 1.0, 3.162278, 10.0, 31.622777, 100.0, 316.227766, 1000.0, 3162.27766,
                    10000.0, 31622.776602, 100000.0, 316227.766017, 1000000.0, 3162277.660168, 10000000.0])
SETS = {"S4_G2F_CV00": ["S4_G2F_CV00"], "EXT_2022": ["EXT_2022"], "EXT_2023": ["EXT_2023"], "EXT_pooled": ["EXT_2022", "EXT_2023"],
        "S1_G2F_LOYO": ["S1_G2F_LOYO"], "S2_G2F_LOLO": ["S2_G2F_LOLO"]}
out_sets, W_all = {}, {}
for sname, parts in SETS.items():
    cs = [c for p in parts for c in by_set.get(p, [])]
    if not cs or all(c["skipped"] for c in cs): continue
    W, NH, edge, chk, chosen = table(cs); W_all[sname] = W
    q = dict(n_env=int(len(W)), n_folds_run=sum(not c["skipped"] for c in cs),
             r={f"{m}|{p_}": R(W[f"{m}|{p_}"].mean()) for m in MODELS for p_ in PROT},
             H9a_D_noEC_minus_MG=boot((W["D_noEC|P3"] - W["MG_2k|P3"]).values),
             H9b_D_full_minus_D_noEC=boot((W["D_full|P3"] - W["D_noEC|P3"]).values),
             D_full_minus_MG=boot((W["D_full|P3"] - W["MG_2k|P3"]).values),
             audit={m: dict(loss_P2_minus_P1=boot((W[f"{m}|P2"] - W[f"{m}|P1"]).values), loss_P2_minus_P3=boot((W[f"{m}|P2"] - W[f"{m}|P3"]).values),
                            edge_fraction={p_: R(np.mean(edge[f"{m}|{p_}"]), 4) for p_ in PROT},
                            n_folds_P2_below_P3=int(sum(x[m]["P2_below_P3"] for x in chk)), n_folds_P2_below_P1=int(sum(x[m]["P2_below_P1"] for x in chk)))
                    for m in MODELS},
             chosen=chosen,
             model_ranking_P3=[m for m in sorted(MODELS, key=lambda m: -W[f"{m}|P3"].mean())])
    if len(NH):
        q["new_hybrid_subset"] = dict(rule="test envs with >= 10 test hybrids absent from the fold's training records", n_env=int(len(NH)),
                                      r_P3={m: R(NH[f"{m}|P3"].mean()) for m in MODELS},
                                      H9a=boot((NH["D_noEC|P3"] - NH["MG_2k|P3"]).values), H9b=boot((NH["D_full|P3"] - NH["D_noEC|P3"]).values))
    else:
        q["new_hybrid_subset"] = dict(n_env=0)
    q["maturity_lambda_edge_fraction"] = R(np.mean([c["maturity"]["lam_idx"] in (0, 14) for c in cs if not c["skipped"]]), 4)
    q["stage1_cv_r2_mean"] = R(np.mean([c["stage1_cv_r2"] for c in cs if not c["skipped"]]), 4)
    out_sets[sname] = q
    W.round(6).to_csv(os.path.join(FIN, f"R9_A_{sname}_per_env_r.csv"))

# ---------------- gates
def rule_a(s): d = out_sets[s]["H9a_D_noEC_minus_MG"]; return bool(d["mean"] >= 0.01 and d["ci_low"] > 0)
def rule_b_conf(s): d = out_sets[s]["H9b_D_full_minus_D_noEC"]; return bool(d["mean"] >= 0.005 and d["ci_low"] > 0)
def rule_b_negl(s): d = out_sets[s]["H9b_D_full_minus_D_noEC"]; return bool(d["ci_high"] < 0.01)
G = [s for s in ["S4_G2F_CV00", "EXT_pooled"] if s in out_sets]
COMPLETE = len(G) == 2
ha = {s: rule_a(s) for s in G}; hb_c = {s: rule_b_conf(s) for s in G}; hb_n = {s: rule_b_negl(s) for s in G}
v_a = ("CONFIRMED" if all(ha.values()) else "NOT_CONFIRMED") if COMPLETE else "INCOMPLETE"
v_b = ("CONFIRMED" if all(hb_c.values()) else ("NEGLIGIBLE" if all(hb_n.values()) else "INCONCLUSIVE")) if COMPLETE else "INCOMPLETE"
p_iut = {"H9a": max(out_sets[s]["H9a_D_noEC_minus_MG"]["p_one_sided_gt_0"] for s in G),
         "H9b": max(out_sets[s]["H9b_D_full_minus_D_noEC"]["p_one_sided_gt_0"] for s in G)}
holm2 = dict(zip(p_iut, holm(list(p_iut.values()))))
overall = ("GO" if v_a == "CONFIRMED" else ("WEAK_GO" if sum(ha.values()) == 1 else "NO_GO")) if COMPLETE else "BLOCKED"

# ---------------- secondary: CERIS window-search leak on EXT
ceris = {}
ems = {c["setting"]: c for c in cells if c["setting"].startswith("EXT") and not c["skipped"]}
if ems:
    for name, parts in (("EXT_pooled", ["EXT_2022", "EXT_2023"]), ("EXT_2022", ["EXT_2022"]), ("EXT_2023", ["EXT_2023"])):
        EM = pd.concat([ems[p]["ceris_env_mean"] for p in parts if p in ems])
        b = boot_envmean_r_diff(EM.obs.values, EM.C2_leaky.values, EM.C2_nested.values)
        ceris[name] = dict(n_env=int(len(EM)), r_envmean_nested=R(b["r_b"]), r_envmean_leaky=R(b["r_a"]), optimism=R(b["mean"]),
                           ci=[R(b["ci_low"]), R(b["ci_high"])], p_one_sided_gt_0=R(boot_p_envmean(EM.obs.values, EM.C2_leaky.values, EM.C2_nested.values)))
        EM.round(6).to_csv(os.path.join(FIN, f"R9_A_ceris_{name}_env_means.csv"))
    ceris["selected"] = {p: ems[p]["ceris_selected"] for p in ems}

cons = json.load(open(os.path.join(RUN, "out", "R9_A_data_consistency.json")))
dev = json.load(open(os.path.join(CODE, "v9_deviations.json")))
res = dict(track="R9_A_mainline_external_H9a_H9b", generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           prereg=dict(file="prereg_v9_mainline_external.json", sha256="6ed45e2bd64c88226fe72360e0b8405233cc84b03b1b604330eaa000dd99aa09", frozen_at_utc=pre9["frozen_at_utc"]),
           compute=dict(host="amax (x86_64, egp311)", note="all folds of every setting computed on the same machine", computed_on=sorted({json.dumps(c.get("computed_on", {}), sort_keys=True) for c in cells if not c["skipped"]})),
           data_consistency=cons, data_consistency_plot_level=json.load(open(os.path.join(RUN, "out", "R9_A_yield_diff.json"))),
           data_consistency_note="data_consistency uses exact equality after sorting on labels (hybrid relabelling misaligns rows); data_consistency_plot_level matches plots on field position and is the authoritative comparison",
           setting_roles={"S4_G2F_CV00": "gate", "EXT_pooled": "gate (primary external)", "EXT_2022": "reported", "EXT_2023": "reported", "S1_G2F_LOYO": "REPLICATION", "S2_G2F_LOLO": "REPLICATION"},
           data_cache_sha256=sha256_file(os.path.join(RUN, "out", "v9_data_cache.pkl")),
           code_sha256={os.path.basename(f): sha256_file(f) for f in sorted(glob.glob(os.path.join(CODE, "*.py")))},
           folds=fold_list, settings=out_sets,
           H9a=dict(quantity=H["H9a_nonlinear_genetic_stage"]["quantity"], rule=H["H9a_nonlinear_genetic_stage"]["CONFIRMED"],
                    per_setting={s: dict(**out_sets[s]["H9a_D_noEC_minus_MG"], meets_rule=ha[s]) for s in G},
                    reported={s: out_sets[s]["H9a_D_noEC_minus_MG"] for s in ("EXT_2022", "EXT_2023") if s in out_sets}, verdict=v_a),
           H9b=dict(quantity=H["H9b_EC_increment"]["quantity"], rule_confirmed=H["H9b_EC_increment"]["CONFIRMED"], rule_negligible=H["H9b_EC_increment"]["NEGLIGIBLE"],
                    per_setting={s: dict(**out_sets[s]["H9b_D_full_minus_D_noEC"], meets_confirmed=hb_c[s], meets_negligible=hb_n[s]) for s in G},
                    reported={s: out_sets[s]["H9b_D_full_minus_D_noEC"] for s in ("EXT_2022", "EXT_2023") if s in out_sets}, verdict=v_b),
           multiplicity=dict(note="per-hypothesis p = max over the two gate settings of the one-sided env-bootstrap p (intersection-union); Holm over H9a and H9b only. The prereg's Holm across H9a-H9d needs track B's p-values.",
                             p_iut=p_iut, p_holm_H9a_H9b=holm2),
           secondary_ceris_window_search_leak_EXT=ceris,
           overall_verdict=overall,
           overall_verdict_mapping_declared="H9a CONFIRMED -> GO; H9a rule met in exactly one of S4/EXT_pooled -> WEAK_GO; neither -> NO_GO. H9b is reported separately (C paper). Declared before the summary was run.",
           deviations_from_prereg=dev)
fo = os.path.join(FIN, "R9_A_results.json")
json.dump(res, open(fo, "w"), indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
open(fo + ".sha256", "w").write(f"{sha256_file(fo)}  R9_A_results.json\n")
print(json.dumps(dict(H9a=v_a, H9b=v_b, overall=overall, gates={s: (out_sets[s]["H9a_D_noEC_minus_MG"], out_sets[s]["H9b_D_full_minus_D_noEC"]) for s in G})))
