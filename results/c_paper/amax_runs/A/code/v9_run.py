"""R9 track A driver: runs one shard of the outer folds of EXT_2022, EXT_2023, S4_G2F_CV00 (and optionally S1/S2
replication) from the v9 data cache, one checkpoint per fold (resumable). EXT folds also get the CERIS nested/leaky
env-mean predictions (secondary, as R7 H1).
Usage: python v9_run.py <code_dir> <cache.pkl> <prereg_v9.json> <v7_S2_raw.pkl> <out_dir> <shard i/n> <threads> <settings e.g. EXT,S4>"""
import os, sys
CODE, CACHE, PRE9, V7S2, OUT, SHARD, TH, SETS = sys.argv[1:9]
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[k] = TH
import json, pickle, time, re, gc, hashlib, platform
for f in ("a_lib.py", "a_models.py", "v5_lib.py", "v8_lib.py", "v9_lib.py"):
    exec(open(os.path.join(CODE, f)).read())
assert sha256_file(PRE9) == "6ed45e2bd64c88226fe72360e0b8405233cc84b03b1b604330eaa000dd99aa09"
pre9 = json.load(open(PRE9)); assert np.allclose(V8_GRID, [0.01, 0.031623, 0.1, 0.316228, 1.0, 3.162278, 10.0, 31.622777, 100.0, 316.227766, 1000.0,
                                                           3162.27766, 10000.0, 31622.776602, 100000.0, 316227.766017, 1000000.0, 3162277.660168, 10000000.0])
GRID = V8_GRID
SH_I, SH_N = (int(x) for x in SHARD.split("/"))
os.makedirs(os.path.join(OUT, "cells"), exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[shard {SH_I}/{SH_N}]", *a, flush=True)
D_ = pickle.load(open(CACHE, "rb"))
pl, he, seasons, daily, Dk, PCs, hyb_index = (D_[k] for k in ("pl", "he", "seasons", "daily", "Dk", "PCs", "hyb_index"))
del D_; gc.collect()
he["genotyped"] = he.Hybrid.isin(hyb_index)
log("cache loaded", Dk.shape, len(he))


def incl_envs(R, min_n=30):
    c = R.groupby("Env").size(); return sorted(c[c >= min_n].index)


def make_records_envs(envs):          # verbatim from v7_g2f.py
    r = he[he.Env.isin(envs) & he.yield_.notna() & he.genotyped & he.has_season].copy()
    r["y"] = r.yield_ - r.groupby("Env").yield_.transform("mean")
    return r.reset_index(drop=True)


def fit_maturity_mask(mask, hidx, lams):   # verbatim from v7_g2f.py
    t = pl[mask & pl.silk_gdd.notna() & pl.Hybrid.isin(hidx)]; tgt = t.groupby("Hybrid").silk_gdd.mean()
    rows = np.array([hidx[h] for h in tgt.index])
    X = Dk[rows]; mu_ = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Xs = (X - mu_) / sd; ybar = tgt.values.mean()
    betas = _ridge_path_svd(Xs, tgt.values - ybar, lams)
    Zall = (Dk - mu_) / sd
    return [Zall @ b + ybar for b in betas], len(tgt)


def make_ctx(train_envs, test_envs_all, inner, p1_key):
    """maturity model and records exactly as v7/v8 run_fold; SNP block = record-weighted z-scored 2k dosages."""
    tr_set = set(train_envs); V1 = set(inner[p1_key])
    preds_in, _ = fit_maturity_mask(pl.Env.isin(tr_set - V1), hyb_index, LAM_GRID)
    he_v = he[he.has_season & he.Env.isin(V1)]
    val = [env_r(he_v, p_, hyb_index).mean() for p_ in preds_in]; bi = int(np.nanargmax(val))
    pm = fit_maturity_mask(pl.Env.isin(tr_set), hyb_index, [LAM_GRID[bi]])[0][0]
    R_tr = make_records_envs(train_envs); R_te = make_records_envs(test_envs_all); test_envs = incl_envs(R_te)
    htr = np.array([hyb_index[h] for h in R_tr.Hybrid]); hte = np.array([hyb_index[h] for h in R_te.Hybrid])
    pm_tr = pm[htr]; anc_tr, ece_tr = env_level_ec(R_tr, pm_tr, seasons)
    anc_te = pd.Series(pm[hte], index=R_te.index).groupby(R_te.Env).mean()
    ec_, _ = ec_table(seasons, anc_te.index.values, anc_te.values); ece_te = pd.DataFrame(ec_, index=anc_te.index, columns=EC_NAMES)
    E_tr = ece_tr.loc[R_tr.Env].values; E_te = ece_te.loc[R_te.Env].values
    cnt = np.bincount(htr, minlength=Dk.shape[0]).astype(float); w = cnt / cnt.sum()
    smu = w @ Dk; ssd = np.sqrt(np.maximum(w @ (Dk ** 2) - smu ** 2, 0)); ssd[~(ssd > 1e-12)] = 1.0
    Dz = (Dk - smu) / ssd
    H = {"tr": htr, "te": hte}; EE = {"tr": E_tr, "te": E_te}
    groups = {k: np.where(R_tr.Env.isin(set(v)).values)[0] for k, v in inner.items()}
    groups = {k: v for k, v in groups.items() if len(v)}
    ctx = dict(Zb=lambda which, idx: Dz[H[which][idx]],
               Fd_noEC=lambda which, idx: np.hstack([PCs[H[which][idx]], pm[H[which][idx]][:, None]]),
               Fd_full=lambda which, idx: np.hstack([EE[which][idx], PCs[H[which][idx]], pm[H[which][idx]][:, None]]),
               y_tr=R_tr.y.values, env_tr=R_tr.Env.values, y_te=R_te.y.values, env_te=R_te.Env.values, test_envs=test_envs,
               groups=groups, p1_key=p1_key, inner_val_envs={k: incl_envs(R_tr.iloc[i]) for k, i in groups.items()})
    meta = dict(maturity=dict(lam=float(LAM_GRID[bi]), lam_idx=bi), n_train_records=len(R_tr), n_train_envs=int(R_tr.Env.nunique()),
                n_test_records=len(R_te), test_envs=test_envs, p1_key=p1_key, inner_keys=list(groups),
                test_keys=pd.DataFrame(dict(Env=R_te.Env.values, Hybrid=R_te.Hybrid.values, y=R_te.y.values,
                                            new_hybrid=~R_te.Hybrid.isin(set(R_tr.Hybrid)).values)))
    return ctx, meta, R_tr, R_te


def ceris_ext(R_tr, R_te, test_envs):
    """R7 H1 procedure: CERIS search on training envs (nested) or training + included test envs (leaky); OLS env-mean model."""
    em_tr = R_tr.groupby("Env").yield_.mean(); em_te = R_te.groupby("Env").yield_.mean().loc[test_envs]
    envs_n = list(em_tr.index); Mn, labels = ceris_candidates(daily, envs_n); kn, rn = ceris_select(Mn, em_tr.values)
    Ml, _ = ceris_candidates(daily, envs_n + list(em_te.index)); kl, rl = ceris_select(Ml, np.concatenate([em_tr.values, em_te.values]))
    all_envs = sorted(set(R_tr.Env) | set(R_te.Env)); Mall, _ = ceris_candidates(daily, all_envs)
    idx = {"nested": pd.Series(Mall[:, kn], index=all_envs), "leaky": pd.Series(Mall[:, kl], index=all_envs)}
    em = pd.DataFrame(dict(obs=em_te.values), index=test_envs)
    for v in ("nested", "leaky"):
        xi = idx[v].loc[em_tr.index].values; A = np.column_stack([np.ones_like(xi), xi]); coef = np.linalg.lstsq(A, em_tr.values, rcond=None)[0]
        em[f"C2_{v}"] = coef[0] + coef[1] * idx[v].loc[test_envs].values
    sel = {"nested": dict(var=labels[kn][0], d1=labels[kn][1], d2=labels[kn][2], r_train=rn),
           "leaky": dict(var=labels[kl][0], d1=labels[kl][1], d2=labels[kl][2], r_train_plus_test=rl)}
    return em, sel


# ---------------------------------------------------------------- fold definitions
loc_of = lambda e: re.sub(r"_\d{4}$", "", e)
ENV_ALL = sorted(pl.Env.unique()); YEAR = {e: int(e[-4:]) for e in ENV_ALL}
E1421 = [e for e in ENV_ALL if YEAR[e] <= 2021]
LOCS = sorted({loc_of(e) for e in E1421})
perm = np.random.default_rng(20260929).permutation(LOCS); FOLD_OF = {l: i % 5 for i, l in enumerate(perm)}
v7 = pickle.load(open(V7S2, "rb")); assert v7["fold_of"] == FOLD_OF, "location folds differ from v7/v8"
KF = {e: FOLD_OF[loc_of(e)] for e in E1421}
nearest = lambda v, T: sorted(T, key=lambda y: (abs(y - v), y))[0]
items = []
if "EXT" in SETS.split(","):
    for t, T in ((2022, list(range(2014, 2022))), (2023, list(range(2014, 2023)))):
        items.append((f"EXT_{t}", str(t), [e for e in ENV_ALL if YEAR[e] in T], [e for e in ENV_ALL if YEAR[e] == t],
                      {int(v): [e for e in ENV_ALL if YEAR[e] == v] for v in T}, int(T[-1])))
YRS = list(range(2014, 2022))
if "S4" in SETS.split(","):
    for k0 in range(5):
        for t in YRS:
            T = [y for y in YRS if y != t]
            items.append(("S4_G2F_CV00", f"{t}_f{k0}", [e for e in E1421 if YEAR[e] != t and KF[e] != k0], [e for e in E1421 if YEAR[e] == t and KF[e] == k0],
                          {int(v): [e for e in E1421 if YEAR[e] == v and KF[e] != k0] for v in T}, int(nearest(t, T))))
if "S1" in SETS.split(","):
    for t in YRS:
        T = [y for y in YRS if y != t]
        items.append(("S1_G2F_LOYO", str(t), [e for e in E1421 if YEAR[e] != t], [e for e in E1421 if YEAR[e] == t],
                      {int(v): [e for e in E1421 if YEAR[e] == v] for v in T}, int(nearest(t, T))))
if "S2" in SETS.split(","):
    for k0 in range(5):
        items.append(("S2_G2F_LOLO", str(k0), [e for e in E1421 if KF[e] != k0], [e for e in E1421 if KF[e] == k0],
                      {int(j): [e for e in E1421 if KF[e] == j] for j in range(5) if j != k0}, int((k0 + 1) % 5)))
mine = [it for i, it in enumerate(items) if i % SH_N == SH_I]
log(f"{len(items)} folds in total, {len(mine)} in this shard")
for setting, key, tr_e, te_e, inner, p1 in mine:
    f = os.path.join(OUT, "cells", f"{setting}__{key}.pkl")
    if os.path.exists(f): log(f"{setting} {key} done (resume)"); continue
    R_te0 = make_records_envs(te_e)
    if not incl_envs(R_te0):
        pickle.dump(dict(setting=setting, key=key, skipped=True, reason="no test env with >= 30 genotyped hybrids with yield",
                         n_test_envs_all=len(te_e)), open(f, "wb"))
        log(f"{setting} {key} skipped (no included test env)"); continue
    t0 = time.time(); log(f"{setting} {key}: {len(tr_e)} train envs, {len(te_e)} test envs; P1 {p1}")
    ctx, meta, R_tr, R_te = make_ctx(tr_e, te_e, inner, p1)
    r = run_fold_v9(ctx, GRID, log=log); r.update(meta); r.update(setting=setting, key=key, skipped=False)
    if setting.startswith("EXT"):
        r["ceris_env_mean"], r["ceris_selected"] = ceris_ext(R_tr, R_te, ctx["test_envs"])
    r["computed_on"] = dict(machine=platform.machine(), python=platform.python_version(), numpy=np.__version__, threads=TH)
    pickle.dump(r, open(f + ".tmp", "wb")); os.replace(f + ".tmp", f)
    log(f"{setting} {key} written ({time.time() - t0:.0f}s)")
    del ctx, R_tr, R_te; gc.collect()
log("shard done")
