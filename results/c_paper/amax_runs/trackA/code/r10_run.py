"""R10 track A, Part G step 2 (part G code path): one external fold EXT_2024 (train = all 2014-2023 records of the 2025
release, test = the 2024 envs with >= 30 genotyped hybrids). Models MG_2k, D_noEC(MG_2k), D_full(MG_2k) with P1/P2/P3
exactly as R9 (v9_lib.run_fold_v9, v8 grid, HGB v8 settings), CERIS nested/leaky env-mean predictions (R7 H1 / R9).
The fold helpers (records, maturity model, context, CERIS) are exec'd verbatim from R9's v9_run.py (sha256 checked).
Writes <out>/cells/EXT_2024__2024.pkl, the record-level MG_2k predictions for track B and per-env r.
Usage: python r10_run.py <r9_code_dir> <r10_cache.pkl> <prereg_v10.json> <out_dir> <threads> <mg2k_csv_path>"""
import os, sys
CODE, CACHE, PRE10, OUT, TH, MGCSV = sys.argv[1:7]
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[k] = TH
import json, pickle, time, re, gc, hashlib, platform
for f in ("a_lib.py", "a_models.py", "v5_lib.py", "v8_lib.py", "v9_lib.py"):
    exec(open(os.path.join(CODE, f)).read())
assert sha256_file(PRE10) == "dfc8e990551241c72d04e3d74f2526dca52209d677884bbcd88edfb6644e015b", "prereg v10 sha256 mismatch"
pre10 = json.load(open(PRE10))
GRID = V8_GRID; assert np.allclose(GRID, pre10["tuning"]["grid"])
R9_CODE_SHA = {"v9_run.py": "faf604c376e5d753ebd3946282434f213ad8e27242783248122d5c7f6f02412a",
               "v9_lib.py": "412e59a732b6d7f91d72c4a9c10594077e0b26d95434511026cd9e9b59645d0f",
               "v8_lib.py": "92e3bbed177e99f19f74308237f2f39f45598296d501b36f4a8aa530a349b822"}
assert sha256_file(os.path.join(CODE, "v9_run.py")) == R9_CODE_SHA["v9_run.py"] and sha256_file(os.path.join(CODE, "v9_lib.py")) == R9_CODE_SHA["v9_lib.py"]
assert sha256_file(os.path.join(CODE, "v8_lib.py")) == R9_CODE_SHA["v8_lib.py"]
os.makedirs(os.path.join(OUT, "cells"), exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), "[EXT_2024]", *a, flush=True)
D_ = pickle.load(open(CACHE, "rb"))
pl, he, seasons, daily, Dk, PCs, hyb_index = (D_[k] for k in ("pl", "he", "seasons", "daily", "Dk", "PCs", "hyb_index"))
del D_; gc.collect()
he["genotyped"] = he.Hybrid.isin(hyb_index)
src = open(os.path.join(CODE, "v9_run.py")).read()
exec(src[src.index("def incl_envs"):src.index("# ---------------------------------------------------------------- fold definitions")])
log("cache loaded", Dk.shape, len(he))
ENV_ALL = sorted(pl.Env.unique()); YEAR = {e: int(e[-4:]) for e in ENV_ALL}
T = list(range(2014, 2024)); assert sorted(set(YEAR.values())) == T
tr_e = [e for e in ENV_ALL if YEAR[e] in T]
te_e = sorted(he[he.Year == 2024].Env.unique())
inner = {int(v): [e for e in ENV_ALL if YEAR[e] == v] for v in T}; p1 = 2023
f = os.path.join(OUT, "cells", "EXT_2024__2024.pkl")
if os.path.exists(f):
    r = pickle.load(open(f, "rb")); log("done (resume)")
else:
    t0 = time.time(); log(f"{len(tr_e)} train envs, {len(te_e)} test envs; P1 {p1}")
    ctx, meta, R_tr, R_te = make_ctx(tr_e, te_e, inner, p1)
    r = run_fold_v9(ctx, GRID, log=log); r.update(meta); r.update(setting="EXT_2024", key="2024", skipped=False)
    r["ceris_env_mean"], r["ceris_selected"] = ceris_ext(R_tr, R_te, ctx["test_envs"])
    # descriptive: CERIS candidate availability (2024 weather ends 2024-11-10) and EC window completeness at the env anchors
    envs_n = sorted(R_tr.Env.unique()); Mn, _ = ceris_candidates(daily, envs_n)
    Ml, _ = ceris_candidates(daily, envs_n + list(ctx["test_envs"]))
    r["ceris_candidates_finite"] = dict(nested=int(np.isfinite(Mn).all(0).sum()), leaky=int(np.isfinite(Ml).all(0).sum()), total=int(Mn.shape[1]))
    pm = fit_maturity_mask(pl.Env.isin(set(tr_e)), hyb_index, [meta["maturity"]["lam"]])[0][0]
    anc_te = pd.Series(pm[np.array([hyb_index[h] for h in R_te.Hybrid])], index=R_te.index).groupby(R_te.Env).mean()
    ec_te, comp_te = ec_table(seasons, anc_te.index.values, anc_te.values)
    E_chk = pd.DataFrame(ec_te, index=anc_te.index).loc[R_te.Env].values
    assert np.allclose(np.nan_to_num(E_chk), np.nan_to_num(ctx["Fd_full"]("te", np.arange(len(R_te)))[:, :21]))
    anc_tr = pd.Series(pm[np.array([hyb_index[h] for h in R_tr.Hybrid])], index=R_tr.index).groupby(R_tr.Env).mean()
    _, comp_tr = ec_table(seasons, anc_tr.index.values, anc_tr.values)
    r["ec_window_complete_fraction"] = dict(test_2024={w: float(comp_te[:, i].mean()) for i, w in enumerate(WINDOWS)},
                                            train={w: float(comp_tr[:, i].mean()) for i, w in enumerate(WINDOWS)},
                                            test_2024_incomplete_envs={w: sorted(anc_te.index[~comp_te[:, i]].tolist()) for i, w in enumerate(WINDOWS)})
    r["computed_on"] = dict(machine=platform.machine(), python=platform.python_version(), numpy=np.__version__, threads=TH)
    pickle.dump(r, open(f + ".tmp", "wb")); os.replace(f + ".tmp", f)
    log(f"written ({time.time() - t0:.0f}s)")
# ---- record-level MG_2k predictions on the 2024 test envs (for track B's MG_98k - MG_2k) and per-env r
tk = r["test_keys"].copy(); tenv = r["test_envs"]
yl = he[(he.Year == 2024)].set_index(["Env", "Hybrid"]).yield_
tk["Yield_Mg_ha"] = yl.loc[list(zip(tk.Env, tk.Hybrid))].values
for p_ in ("P1", "P2", "P3"):
    tk[f"pred_MG_2k_{p_}"] = r["test_pred"][("MG_2k", p_)].astype(float)
tk = tk[tk.Env.isin(tenv)].rename(columns={"y": "y_env_centred"})
tk = tk[["Env", "Hybrid", "Yield_Mg_ha", "y_env_centred", "new_hybrid", "pred_MG_2k_P1", "pred_MG_2k_P2", "pred_MG_2k_P3"]]
os.makedirs(os.path.dirname(MGCSV), exist_ok=True)
tk.to_csv(MGCSV, index=False, float_format="%.8g")
pe = pd.DataFrame({p_: per_env_r(tk.Env.values, tk.y_env_centred.values, tk[f"pred_MG_2k_{p_}"].values[:, None], tenv)[0] for p_ in ("P1", "P2", "P3")})
pe.insert(0, "n", tk.groupby("Env").size().loc[pe.index].values); pe.index.name = "Env"
pe.to_csv(MGCSV.replace(".csv", "_r.csv"), float_format="%.8g")
log("MG_2k per-env predictions written", len(tk), "records", len(pe), "envs")
