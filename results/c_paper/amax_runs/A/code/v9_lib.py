"""R9 track A engine (prereg v9 H9a/H9b): MG_2k ridge path over the v8 grid with P1/P2/P3 selection, and the two-stage
models D_noEC(MG_2k) / D_full(MG_2k): stage 1 = MG_2k at its P3 lambda (all protocols), stage 2 = HistGradientBoosting
(v8 settings) on stage-1 residuals cross-fitted over the P3 inner groups; n_iter chosen by the protocol (P3: mean over
inner folds with nested cross-fitting; P1: the P1 inner fold; P2: test envs, audit only).
Built from v8_lib.run_fold_v8 / v8_ablation.d_variants. Requires v5_lib.py and v8_lib.py exec'd first."""
import time
import numpy as np
import pandas as pd


def run_fold_v9(ctx, grid, log=print):
    t0 = time.time(); L = len(grid)
    Zb, y_tr, env_tr, y_te, env_te = ctx["Zb"], ctx["y_tr"], ctx["env_tr"], ctx["y_te"], ctx["env_te"]
    groups, keys, p1, tenv = ctx["groups"], list(ctx["groups"]), ctx["p1_key"], ctx["test_envs"]
    n_tr, n_te = len(y_tr), len(y_te); all_tr, all_te = np.arange(n_tr), np.arange(n_te)
    res = dict(sel={}, inner_curves={}, test_path={}, test_pred={})
    # ---- SNP raw sums per inner group
    gsnp = {g: rawsums(Zb, "tr", groups[g], y_tr) for g in keys}
    fsnp = rs_minus(gsnp[keys[0]])
    for g in keys[1:]:
        fsnp["G"] += gsnp[g]["G"]; fsnp["s"] += gsnp[g]["s"]; fsnp["xy"] += gsnp[g]["xy"]; fsnp["ys"] += gsnp[g]["ys"]; fsnp["m"] += gsnp[g]["m"]
    assert fsnp["m"] == n_tr
    def fit(rs, lams):
        Gc, bc, xb, yb = center_inplace(rs); C = coef_path(Gc, bc, lams); return C, yb - xb @ C
    # ---- MG_2k: inner paths (P1/P3) and outer path (P2)
    mg_val, cur = {}, {}
    for g in keys:
        C, a = fit(rs_minus(fsnp, gsnp[g]), grid); P = Zb("tr", groups[g]) @ C + a[None, :]
        mg_val[g] = P; cur[g] = curve(env_tr[groups[g]], y_tr[groups[g]], P, ctx["inner_val_envs"][g])
    C, a = fit(rs_minus(fsnp), grid); Pte = Zb("te", all_te) @ C + a[None, :]
    res["test_path"]["MG_2k"] = per_env_r(env_te, y_te, Pte, tenv)
    c3 = np.nanmean(np.array([cur[g] for g in keys], float), 0)
    res["sel"]["MG_2k"] = dict(P1=int(np.nanargmax(cur[p1])), P2=int(np.nanargmax(res["test_path"]["MG_2k"].mean(0).values)), P3=int(np.nanargmax(c3)))
    res["inner_curves"]["MG_2k"] = {str(g): np.round(cur[g], 6).tolist() for g in keys}
    for pr, k in res["sel"]["MG_2k"].items(): res["test_pred"][("MG_2k", pr)] = Pte[:, k].astype(np.float32)
    li3 = res["sel"]["MG_2k"]["P3"]; res["stage1_lam_idx"] = li3
    log(f"   MG done {time.time() - t0:.0f}s; lambda idx {res['sel']['MG_2k']}")
    # ---- nested cross-fitting predictions at the stage-1 lambda (rows of u from a fit without u and v)
    pair = {}
    for i, u in enumerate(keys):
        for v in keys[i + 1:]:
            C, a = fit(rs_minus(fsnp, gsnp[u], gsnp[v]), grid[[li3]])
            pair[(u, v)] = (Zb("tr", groups[u]) @ C + a)[:, 0]; pair[(v, u)] = (Zb("tr", groups[v]) @ C + a)[:, 0]
    cf = np.empty(n_tr)
    for u in keys: cf[groups[u]] = mg_val[u][:, li3]
    res["stage1_cv_r2"] = float(1 - np.var(y_tr - cf) / np.var(y_tr))
    # ---- stage 2 variants
    for name, Fd in (("D_noEC", ctx["Fd_noEC"]), ("D_full", ctx["Fd_full"])):
        t1 = time.time(); Ftr, Fte = Fd("tr", all_tr), Fd("te", all_te); dcur = {}
        for v in keys:
            idx_in = np.concatenate([groups[u] for u in keys if u != v])
            tgt = np.concatenate([y_tr[groups[u]] - pair[(u, v)] for u in keys if u != v])
            S = hgb_staged(Ftr[idx_in], tgt, Ftr[groups[v]])
            dcur[v] = curve(env_tr[groups[v]], y_tr[groups[v]], mg_val[v][:, [li3]] + S, ctx["inner_val_envs"][v])
        S = hgb_staged(Ftr, y_tr - cf, Fte); Pd = Pte[:, [li3]] + S
        res["test_path"][name] = per_env_r(env_te, y_te, Pd, tenv)
        res["sel"][name] = dict(P1=int(np.nanargmax(dcur[p1])), P2=int(np.nanargmax(res["test_path"][name].mean(0).values)),
                                P3=int(np.nanargmax(np.nanmean(np.array([dcur[v] for v in keys], float), 0))))
        res["inner_curves"][name] = {str(v): np.round(dcur[v], 6).tolist() for v in keys}
        for pr, k in res["sel"][name].items(): res["test_pred"][(name, pr)] = Pd[:, k].astype(np.float32)
        log(f"   {name} {time.time() - t1:.0f}s; n_iter idx {res['sel'][name]}")
    log(f"   fold total {time.time() - t0:.0f}s")
    return res
