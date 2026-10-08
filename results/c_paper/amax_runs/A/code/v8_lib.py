"""prereg v8 Q2 (EC-specification robustness) + Q3 (tuning audit) fold engine, shared by v8_g2f.py and v8_briwecs.py.
Design matrices are never materialised: raw (uncentred) Gram sums are accumulated from row chunks produced by a
dataset-specific builder; centring, column standardisation and block penalties are applied on the Gram.
Models: M_G, M0, M0plus (as R7), B_PC200xEC, C_SNPxEnvIndex, D_NL_env. Protocols P1/P2/P3 as v7.
Requires v5_lib.py exec'd first (per_env_r, env_ridge_fit_predict, pearson)."""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

V8_GRID = np.array([0.01, 0.031623, 0.1, 0.316228, 1.0, 3.162278, 10.0, 31.622777, 100.0, 316.227766, 1000.0, 3162.27766,
                    10000.0, 31622.776602, 100000.0, 316227.766017, 1000000.0, 3162277.660168, 10000000.0])
C_MULT = [0.1, 1.0, 10.0]                     # interaction-block penalty = lambda x c
NITER = [50, 100, 200, 400, 800]
HGB_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=20260928, early_stopping=False)
CHUNK = 3072


def rawsums(build, which, idx, y, chunk=CHUNK):
    G = s = xy = None; ys = 0.0; m = 0
    for i in range(0, len(idx), chunk):
        r = idx[i:i + chunk]; X = build(which, r); v = y[r]
        if G is None:
            p = X.shape[1]; G = np.zeros((p, p)); s = np.zeros(p); xy = np.zeros(p)
        G += X.T @ X; s += X.sum(0); xy += X.T @ v; ys += float(v.sum()); m += len(r)
    return dict(G=G, s=s, xy=xy, ys=ys, m=m)


def rs_snp(rs, ns):
    return dict(G=rs["G"][:ns, :ns].copy(), s=rs["s"][:ns].copy(), xy=rs["xy"][:ns].copy(), ys=rs["ys"], m=rs["m"])


def rs_minus(a, *bs):
    o = dict(G=a["G"].copy(), s=a["s"].copy(), xy=a["xy"].copy(), ys=a["ys"], m=a["m"])
    for b in bs:
        o["G"] -= b["G"]; o["s"] -= b["s"]; o["xy"] -= b["xy"]; o["ys"] -= b["ys"]; o["m"] -= b["m"]
    return o


def center_inplace(rs):
    """turns raw sums into centred Gram (in place on rs['G']); returns (Gc, bc, xbar, ybar)."""
    m = rs["m"]; xb = rs["s"] / m; yb = rs["ys"] / m
    G = rs["G"]; G -= m * np.outer(xb, xb)
    return G, rs["xy"] - m * xb * yb, xb, yb


def coef_path(Gc, bc, lams, dvec=None):
    """ridge path in the (optionally D-rescaled) space; returns coefficients in the input space (p x L)."""
    if dvec is not None:
        Gc = Gc * dvec[:, None]; Gc *= dvec[None, :]; bc = bc * dvec
    d, V = np.linalg.eigh(Gc); d = np.clip(d, 0, None); Vb = V.T @ bc
    B = V @ (Vb[:, None] / (d[:, None] + np.asarray(lams)[None, :]))
    return B if dvec is None else B * dvec[:, None]


def model_specs(ns, n30, npcec, nec):
    """column index sets (in the full B layout [SNP | PC200xEC | MxEC]) and block structure."""
    snp = np.arange(ns); pc30 = ns + np.arange(n30); pcall = ns + np.arange(npcec); mx = ns + npcec + np.arange(nec)
    return {"M_G": (snp, None), "M0": (np.concatenate([snp, pc30]), None), "M0plus": (np.concatenate([snp, pc30, mx]), None),
            "B": (np.concatenate([snp, pcall, mx]), ns)}      # second item: number of main (unblocked) columns for block models


def coef_stack(Gc, bc, xb, yb, sc, specs, grid):
    """For every model (and block multiplier) the raw-feature coefficient paths + intercepts.
    Returns dict name -> (cols, Craw (|cols| x K), a0 (K,)), K = L (single) or 3L (block, c-major)."""
    out = {}
    for name, (cols, nmain) in specs.items():
        G = Gc[np.ix_(cols, cols)]; G *= sc[cols][:, None]; G *= sc[cols][None, :]; b = bc[cols] * sc[cols]
        if nmain is None:
            Bs = [coef_path(G, b, grid)]
        else:
            Bs = []
            for c in C_MULT:
                dv = np.ones(len(cols)); dv[nmain:] = 1.0 / np.sqrt(c); Bs.append(coef_path(G, b, grid, dv))
        Bsc = np.hstack(Bs); Craw = Bsc * sc[cols][:, None]
        out[name] = (cols, Craw, yb - xb[cols] @ Craw)
        del G
    return out


def predict_stack(build, which, idx, cs, p, chunk=CHUNK):
    """one chunked pass: predictions (with intercept) for every model in cs."""
    names = list(cs); Cfull = np.zeros((p, sum(cs[n][1].shape[1] for n in names))); offs = {}; o = 0
    for n in names:
        cols, C, a0 = cs[n]; Cfull[cols, o:o + C.shape[1]] = C; offs[n] = (o, o + C.shape[1]); o += C.shape[1]
    P = np.vstack([build(which, idx[i:i + chunk]) @ Cfull for i in range(0, len(idx), chunk)])
    return {n: P[:, a:b] + cs[n][2][None, :] for n, (a, b) in offs.items()}


def c_rawsums(Zb, which, idx, y, w, snp_rs, chunk=CHUNK):
    """raw sums of [Z | w*Z] over rows idx, SNP part taken from snp_rs (same rows)."""
    ns = snp_rs["G"].shape[0]; G1 = np.zeros((ns, ns)); G2 = np.zeros((ns, ns)); sW = np.zeros(ns); xyW = np.zeros(ns)
    for i in range(0, len(idx), chunk):
        r = idx[i:i + chunk]; Z = Zb(which, r); Zw = Z * w[r][:, None]; v = y[r]
        G1 += Z.T @ Zw; G2 += Zw.T @ Zw; sW += Zw.sum(0); xyW += Zw.T @ v
    G = np.block([[snp_rs["G"], G1], [G1.T, G2]])
    return dict(G=G, s=np.concatenate([snp_rs["s"], sW]), xy=np.concatenate([snp_rs["xy"], xyW]), ys=snp_rs["ys"], m=snp_rs["m"])


def c_fit(rs, grid):
    ns = rs["G"].shape[0] // 2
    Gc, bc, xb, yb = center_inplace(rs)
    Bs = []
    for c in C_MULT:
        dv = np.ones(2 * ns); dv[ns:] = 1.0 / np.sqrt(c); Bs.append(coef_path(Gc, bc, grid, dv))
    C = np.hstack(Bs); return C, yb - xb @ C


def c_predict(Zb, which, idx, w, C, a0, chunk=CHUNK):
    ns = C.shape[0] // 2
    return np.vstack([Zb(which, idx[i:i + chunk]) @ C[:ns] + (Zb(which, idx[i:i + chunk]) * w[idx[i:i + chunk]][:, None]) @ C[ns:]
                      for i in range(0, len(idx), chunk)]) + a0[None, :]


def curve(env, y, P, envs):
    if not envs: return np.full(P.shape[1], np.nan)
    return per_env_r(env, y, P, envs).mean(0).values


def zfit_rows(v):
    mu = v.mean(); sd = v.std(); sd = sd if sd > 1e-12 else 1.0
    return lambda a: (a - mu) / sd


def hgb_staged(F_tr, r_tr, F_ev):
    m = HistGradientBoostingRegressor(max_iter=max(NITER), **HGB_PARAMS).fit(F_tr, r_tr)
    keep = {k - 1 for k in NITER}; out = []
    for i, p in enumerate(m.staged_predict(F_ev)):
        if i in keep: out.append(p)
    return np.column_stack(out)


def run_fold_v8(ctx, grid, log=print):
    """ctx keys: build(which, idx) -> raw B features; Zb(which, idx) -> SNP block; Fd(which, idx) -> HGB features;
    ns, n30, npcec, nec; y_tr, env_tr, y_te, env_te, test_envs; groups {key: idx}; inner_val_envs {key: envs}; p1_key;
    ece (DataFrame env x EC for training + test envs); em_tr (Series env -> env-mean yield of training envs);
    group_envs {key: training envs of the group}."""
    t0 = time.time(); L = len(grid)
    ns = ctx["ns"]; y_tr, env_tr, y_te, env_te = ctx["y_tr"], ctx["env_tr"], ctx["y_te"], ctx["env_te"]
    groups, keys, p1 = ctx["groups"], list(ctx["groups"]), ctx["p1_key"]
    n_tr, n_te = len(y_tr), len(y_te); all_tr = np.arange(n_tr); all_te = np.arange(n_te)
    specs = model_specs(ns, ctx["n30"], ctx["npcec"], ctx["nec"])
    build, Zb, Fd = ctx["build"], ctx["Zb"], ctx["Fd"]
    res = dict(inner_curves={}, test_path={}, sel={}, test_pred={})
    # ---- pass A: group raw sums -> full; SNP-block sums per group (for D nested cross-fitting and C)
    full = None; gsnp = {}
    for g in keys:
        rs = rawsums(build, "tr", groups[g], y_tr); gsnp[g] = rs_snp(rs, ns)
        if full is None: full = rs
        else:
            full["G"] += rs["G"]; full["s"] += rs["s"]; full["xy"] += rs["xy"]; full["ys"] += rs["ys"]; full["m"] += rs["m"]
        del rs
    p = full["G"].shape[0]; assert full["m"] == n_tr
    fsnp = rs_snp(full, ns)
    m = full["m"]; xb = full["s"] / m
    var = np.diag(full["G"]) / m - xb ** 2; sd = np.sqrt(np.maximum(var, 0)); sd[~(sd > 1e-12)] = 1.0
    sc = np.ones(p); sc[ns:] = 1.0 / sd[ns:]                            # z-score interaction columns on training rows
    log(f"   pass A {time.time() - t0:.0f}s p={p} n_tr={n_tr}")
    # ---- C: env-index ridge penalty by P3 inner folds (env level), outer and inner e_hat
    E = ctx["ece"]; em = ctx["em_tr"]; tr_envs = list(em.index)
    env_curves = {}
    for g in keys:
        ev = [e for e in ctx["group_envs"][g] if e in em.index]; ei = [e for e in tr_envs if e not in set(ev)]
        env_curves[g] = [pearson(env_ridge_fit_predict(E.loc[ei].values, em.loc[ei].values, E.loc[ev].values, l), em.loc[ev].values)
                         if len(ev) >= 3 else np.nan for l in grid]
    lam_env_idx = int(np.nanargmax(np.nanmean(np.array([env_curves[g] for g in keys], float), 0)))
    lam_env = grid[lam_env_idx]
    def ehat_rows(fit_envs, which_rows_envs):
        pr = env_ridge_fit_predict(E.loc[fit_envs].values, em.loc[fit_envs].values, E.loc[which_rows_envs].values, lam_env)
        return pd.Series(pr, index=which_rows_envs)
    all_envs_te = sorted(set(env_te))
    eh_o = ehat_rows(tr_envs, tr_envs + all_envs_te)
    w_tr_raw = eh_o.loc[env_tr].values; zw = zfit_rows(w_tr_raw); w_tr = zw(w_tr_raw); w_te = zw(eh_o.loc[env_te].values)
    res["C_env_index"] = dict(lam_env=float(lam_env), lam_env_idx=lam_env_idx, env_curve_mean=np.nanmean(np.array([env_curves[g] for g in keys], float), 0).round(6).tolist(),
                              r_envmean_test=float(pearson(eh_o.loc[ctx["test_envs"]].values, ctx["em_te"].loc[ctx["test_envs"]].values)))
    # ---- pass B: inner folds
    mg_val = {}; cf_mg = {}
    for g in keys:
        t1 = time.time(); idx_v = groups[g]; venv = ctx["inner_val_envs"][g]
        rs = rawsums(build, "tr", idx_v, y_tr)
        inn = rs_minus(full, rs); del rs
        Gc, bc, xbi, ybi = center_inplace(inn)
        cs = coef_stack(Gc, bc, xbi, ybi, sc, specs, grid); del inn, Gc
        P = predict_stack(build, "tr", idx_v, cs, p); del cs
        cur = {n: curve(env_tr[idx_v], y_tr[idx_v], P[n], venv) for n in P}
        mg_val[g] = P["M_G"]                                               # (n_v x L) with intercept
        # C inner: e_hat from env ridge on inner-training envs
        ev = [e for e in ctx["group_envs"][g] if e in em.index]; ei = [e for e in tr_envs if e not in set(ev)]
        eh = ehat_rows(ei, ei + ev); idx_in = np.concatenate([groups[k] for k in keys if k != g])
        wr = np.full(n_tr, np.nan); wr[idx_in] = eh.loc[env_tr[idx_in]].values; wr[idx_v] = eh.loc[env_tr[idx_v]].values
        zwi = zfit_rows(wr[idx_in]); wr = zwi(wr)
        crs = c_rawsums(Zb, "tr", idx_in, y_tr, wr, rs_minus(fsnp, gsnp[g]))
        Cc, a0c = c_fit(crs, grid); del crs
        cur["C"] = curve(env_tr[idx_v], y_tr[idx_v], c_predict(Zb, "tr", idx_v, wr, Cc, a0c), venv)
        res["inner_curves"][g] = {n: np.round(v, 6).tolist() for n, v in cur.items()}
        log(f"   inner {g}: {time.time() - t1:.0f}s")
    # ---- outer fit
    t1 = time.time()
    Gc, bc, xbo, ybo = center_inplace(full)
    cs = coef_stack(Gc, bc, xbo, ybo, sc, specs, grid); del Gc, full
    Pte = predict_stack(build, "te", all_te, cs, p); del cs
    crs = c_rawsums(Zb, "tr", all_tr, y_tr, w_tr, fsnp); Cc, a0c = c_fit(crs, grid); del crs
    Pte["C"] = c_predict(Zb, "te", all_te, w_te, Cc, a0c)
    tenv = ctx["test_envs"]
    for n, P in Pte.items():
        res["test_path"][n] = per_env_r(env_te, y_te, P, tenv)
    log(f"   outer {time.time() - t1:.0f}s")
    # ---- selection (flattened index; block models c-major)
    for n in ["M_G", "M0", "M0plus", "B", "C"]:
        c1 = np.array(res["inner_curves"][p1][n], float); c3 = np.nanmean(np.array([res["inner_curves"][g][n] for g in keys], float), 0)
        c2 = res["test_path"][n].mean(0).values
        res["sel"][n] = dict(P1=int(np.nanargmax(c1)), P2=int(np.nanargmax(c2)), P3=int(np.nanargmax(c3)))
        for pr, k in res["sel"][n].items(): res["test_pred"][(n, pr)] = Pte[n][:, k].astype(np.float32)
    # ---- D: two-stage, stage 1 = M_G at the protocol's lambda, stage 2 = HGB on cross-fitted residuals
    t1 = time.time()
    lam_idx = {pr: res["sel"]["M_G"][pr] for pr in ("P1", "P2", "P3")}
    need = sorted(set(lam_idx.values()))
    # nested cross-fitting predictions: for inner fold v, rows of u (u != v) predicted by M_G fit on train - {u, v}
    pair_pred = {}
    for i, u in enumerate(keys):
        for v in keys[i + 1:]:
            rsuv = rs_minus(fsnp, gsnp[u], gsnp[v]); Gc_, bc_, xb_, yb_ = center_inplace(rsuv)
            C_ = coef_path(Gc_, bc_, grid[need]); a_ = yb_ - xb_ @ C_
            for a, b in ((u, v), (v, u)):
                pair_pred[(a, b)] = Zb("tr", groups[a]) @ C_ + a_[None, :]    # rows of a, model without a and b
    col = {li: j for j, li in enumerate(need)}
    Ftr = Fd("tr", all_tr); Fte = Fd("te", all_te)
    dcur = {}
    for pr in ("P3", "P1"):
        li = lam_idx[pr]
        for v in (keys if pr == "P3" else [p1]):
            if (v, li) in dcur: continue
            idx_in = np.concatenate([groups[u] for u in keys if u != v])
            tgt = np.concatenate([y_tr[groups[u]] - pair_pred[(u, v)][:, col[li]] for u in keys if u != v])
            S = hgb_staged(Ftr[idx_in], tgt, Ftr[groups[v]])
            Pv = mg_val[v][:, [li]] + S
            dcur[(v, li)] = curve(env_tr[groups[v]], y_tr[groups[v]], Pv, ctx["inner_val_envs"][v])
    d_out = {}
    for li in need:
        cf = np.empty(n_tr)
        for u in keys: cf[groups[u]] = mg_val[u][:, li]
        S = hgb_staged(Ftr, y_tr - cf, Fte)
        d_out[li] = Pte["M_G"][:, [li]] + S
    dsel = {"P3": int(np.nanargmax(np.nanmean(np.array([dcur[(v, lam_idx["P3"])] for v in keys], float), 0))),
            "P1": int(np.nanargmax(dcur[(p1, lam_idx["P1"])]))}
    res["test_path"]["D"] = {li: per_env_r(env_te, y_te, d_out[li], tenv) for li in need}
    dsel["P2"] = int(np.nanargmax(res["test_path"]["D"][lam_idx["P2"]].mean(0).values))
    res["sel"]["D"] = dsel; res["D_stage1_lam_idx"] = lam_idx
    res["inner_curves_D"] = {f"{v}|{li}": np.round(c, 6).tolist() for (v, li), c in dcur.items()}
    for pr in ("P1", "P2", "P3"):
        res["test_pred"][("D", pr)] = d_out[lam_idx[pr]][:, dsel[pr]].astype(np.float32)
    log(f"   D {time.time() - t1:.0f}s; total {time.time() - t0:.0f}s")
    return res
