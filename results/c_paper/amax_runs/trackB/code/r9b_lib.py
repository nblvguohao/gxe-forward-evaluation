"""prereg v9 track B engine (H9c, H9d; dense genotypes; GPU check). Requires a_lib.py, a_models.py, r6_lib.py exec'd.
Data are rebuilt from the G2F 2025 release (2014-2023) exactly along the round-1 code path (run_R6 step 1).
SNP-only ridge models (MG_2k, MG_98k) use hybrid-level features that are standardised once per OUTER fold on its
training records (as the v8 track-A engine, v8_lib.run_fold_v8); inner and pair fits reuse those features with an
intercept (centred Gram). D_noEC follows v8_lib.run_fold_v8's D exactly: stage 1 = MG at its P3 lambda; stage 2 =
HistGradientBoosting on nested cross-fitted stage-1 residuals, n_iter chosen by the same P3 inner folds; features
PC1..30 (same SNP set) + predicted maturity (v9: EC_env removed).
"""
import os, json, time, hashlib, re, gc
import numpy as np, pandas as pd
import scipy.linalg as sla
from sklearn.ensemble import HistGradientBoostingRegressor

V8_GRID = np.array([0.01, 0.031623, 0.1, 0.316228, 1.0, 3.162278, 10.0, 31.622777, 100.0, 316.227766, 1000.0, 3162.27766,
                    10000.0, 31622.776602, 100000.0, 316227.766017, 1000000.0, 3162277.660168, 10000000.0])
NITER = [50, 100, 200, 400, 800]
HGB_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=20260928, early_stopping=False)
YEARS_CV = list(range(2014, 2022))
loc_of = lambda e: re.sub(r"_\d{4}$", "", e)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ data (round-1 path on the 2025 release)
def build_data(T25, log=print):
    tr = pd.read_csv(os.path.join(T25, "1_Training_Trait_Data_2014_2023.csv"), low_memory=False)
    tr["Date_Planted_p"] = pd.to_datetime(tr["Date_Planted"], format="mixed")
    wx = pd.read_csv(os.path.join(T25, "4_Training_Weather_Data_2014_2023_full_year.csv"))
    wx["Date_p"] = pd.to_datetime(wx.Date.astype(str), format="%Y%m%d")
    pl = tr[["Env", "Year", "Hybrid", "Date_Planted_p", "Silk_DAP_days", "Pollen_DAP_days", "Yield_Mg_ha"]].copy()
    plant_med = pl.groupby("Env").Date_Planted_p.median().dt.normalize()
    seasons = build_env_season(wx, plant_med)
    pl["silk_gdd"] = np.nan
    for env, ii in pl.groupby("Env").indices.items():
        if env in seasons:
            pl.iloc[ii, pl.columns.get_loc("silk_gdd")] = silk_gdd_from_dap(seasons[env], pl.Silk_DAP_days.values[ii])
    samples, snp_ids, D = parse_vcf_dosage(os.path.join(T25, "5_Genotype_Data_All_2014_2025_Hybrids.vcf"))
    af = np.nanmean(D, axis=0) / 2; keep = np.minimum(af, 1 - af) >= 0.05
    Dk = D[:, keep].astype(np.float64); mu = np.nanmean(Dk, axis=0)
    ind = np.where(np.isnan(Dk)); Dk[ind] = np.take(mu, ind[1]); del D
    Gz = (Dk - Dk.mean(0)) / Dk.std(axis=0)
    U, S, Vt = np.linalg.svd(Gz, full_matrices=False); PCs = U[:, :30] * S[:30]; del U, Vt, Gz
    hyb_index = {h: i for i, h in enumerate(samples)}
    ph = (pl[pl.silk_gdd.notna()].groupby(["Env", "Hybrid"], as_index=False)
          .agg(silk_gdd=("silk_gdd", "mean"), silk_dap=("Silk_DAP_days", "mean")))
    yl = pl[pl.Yield_Mg_ha.notna()].groupby(["Env", "Year", "Hybrid"], as_index=False).agg(yield_=("Yield_Mg_ha", "mean"))
    he = yl.merge(ph, on=["Env", "Hybrid"], how="outer")
    he["Year"] = he.Year.fillna(he.Env.str[-4:].astype(int)).astype(int)
    he["has_season"] = he.Env.isin(seasons)
    par = tr[["Hybrid", "Hybrid_Parent1", "Hybrid_Parent2"]].dropna()
    par = (par.value_counts().reset_index().sort_values(["Hybrid", "count", "Hybrid_Parent1"], ascending=[True, False, True])
           .groupby("Hybrid").first())
    log(f"data: trait rows {len(tr)}, envs {tr.Env.nunique()}, seasons {len(seasons)}, 2k SNPs {Dk.shape[1]}, VCF hybrids {len(samples)}")
    return dict(tr=tr, wx=wx, pl=pl, seasons=seasons, samples=samples, Dk=Dk, PCs=PCs, hyb_index=hyb_index, he=he, par=par)


def agreement_2014_2021(d, LOCAL):
    """2014-2021 subset of the 2025 release vs the local files used in R1-R8."""
    loc = pd.read_csv(os.path.join(LOCAL, "1_Training_Trait_Data_2014_2021.csv"), low_memory=False)
    new = d["tr"][d["tr"].Year <= 2021]
    keys = ["Env", "Hybrid", "Replicate", "Block", "Plot"]
    a = loc[keys + ["Yield_Mg_ha", "Silk_DAP_days"]].copy(); b = new[keys + ["Yield_Mg_ha", "Silk_DAP_days"]].copy()
    for x in (a, b): x["Plot"] = x.Plot.astype(str); x["Replicate"] = x.Replicate.astype(str); x["Block"] = x.Block.astype(str)
    m = a.merge(b, on=keys, how="outer", suffixes=("_loc", "_new"), indicator=True)
    both = m[m._merge == "both"]
    yeq = np.isclose(both.Yield_Mg_ha_loc, both.Yield_Mg_ha_new, atol=1e-9, equal_nan=True)
    seq = np.isclose(both.Silk_DAP_days_loc, both.Silk_DAP_days_new, atol=1e-9, equal_nan=True)
    wl = pd.read_csv(os.path.join(LOCAL, "4_Training_Weather_Data_2014_2021.csv"))
    wn = d["wx"][d["wx"].Date.astype(str).str[:4].astype(int) <= 2021]
    vars_ = ["T2M_MAX", "T2M_MIN", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT"]
    ww = wl[["Env", "Date"] + vars_].merge(wn[["Env", "Date"] + vars_], on=["Env", "Date"], suffixes=("_l", "_n"))
    wdiff = {v: float(np.nanmax(np.abs(ww[v + "_l"] - ww[v + "_n"]))) if len(ww) else None for v in vars_}
    return dict(rows_local=int(len(loc)), rows_2025_subset=int(len(new)), envs_local=int(loc.Env.nunique()), envs_2025_subset=int(new.Env.nunique()),
                env_sets_equal=bool(set(loc.Env) == set(new.Env)), rows_matched_on_keys=int(len(both)),
                rows_only_local=int((m._merge == "left_only").sum()), rows_only_2025=int((m._merge == "right_only").sum()),
                yield_equal_frac=float(yeq.mean()) if len(both) else None, silk_dap_equal_frac=float(seq.mean()) if len(both) else None,
                yield_max_abs_diff=float(np.nanmax(np.abs(both.Yield_Mg_ha_loc - both.Yield_Mg_ha_new))) if len(both) else None,
                weather_rows_local=int(len(wl)), weather_rows_2025_subset=int(len(wn)), weather_rows_matched=int(len(ww)),
                weather_max_abs_diff=wdiff)


def recs(he, E):
    r = he[he.Env.isin(E) & he.yield_.notna() & he.genotyped & he.has_season].copy()
    r["y"] = r.yield_ - r.groupby("Env").yield_.transform("mean")
    return r.reset_index(drop=True)


def incl(R, min_n=30):
    c = R.groupby("Env").size(); return sorted(c[c >= min_n].index)


def nearest(v, T):
    return sorted(T, key=lambda y: (abs(y - v), y))[0]


def folds_S4(env_all_cv):
    """v8 S4: year y x location fold k (v7 S2 folds from the 2014-2021 env list)."""
    locs = sorted({loc_of(e) for e in env_all_cv})
    perm = np.random.default_rng(20260929).permutation(locs); fold_of = {l: i % 5 for i, l in enumerate(perm)}
    ey = {e: int(e[-4:]) for e in env_all_cv}; ef = {e: fold_of[loc_of(e)] for e in env_all_cv}
    F = []
    for k0 in range(5):
        for t in YEARS_CV:
            T = [y for y in YEARS_CV if y != t]
            tr = frozenset(e for e in env_all_cv if ey[e] != t and ef[e] != k0)
            te = frozenset(e for e in env_all_cv if ey[e] == t and ef[e] == k0)
            inner = {str(v): frozenset(e for e in tr if ey[e] == v) for v in T}
            F.append(dict(key=f"{t}_f{k0}", train=tr, test=te, inner=inner, p1=str(nearest(t, T))))
    return F, {str(k): sorted(l for l in locs if fold_of[l] == k) for k in range(5)}


def folds_EXT(env_all):
    ey = {e: int(e[-4:]) for e in env_all}; F = []
    for t in (2022, 2023):
        T = list(range(2014, t))
        tr = frozenset(e for e in env_all if ey[e] in T); te = frozenset(e for e in env_all if ey[e] == t)
        F.append(dict(key=f"EXT_{t}", train=tr, test=te, inner={str(v): frozenset(e for e in tr if ey[e] == v) for v in T},
                      p1=str(nearest(t, T))))
    return F


def maturity(d, hyb_idx_g, f, he_s):
    """round-1 rule: lambda (round-1 LAM_GRID) on the fold's P1 validation envs (fit on train minus them), refit on train."""
    pl = d["pl"]; V = f["inner"][f["p1"]]; Etr = f["train"] - V
    preds, _ = fit_maturity(pl[pl.Env.isin(Etr)], d["Dk"], hyb_idx_g, list(range(2014, 2024)), LAM_GRID)
    val = [env_r(he_s[he_s.Env.isin(V)], p, hyb_idx_g).mean() for p in preds]; bi = int(np.nanargmax(val))
    pm = fit_maturity(pl[pl.Env.isin(f["train"])], d["Dk"], hyb_idx_g, list(range(2014, 2024)), [LAM_GRID[bi]])[0][0]
    return pm, dict(lam=float(LAM_GRID[bi]), lam_idx=bi, val_r=float(val[bi]))


# ------------------------------------------------------------------ SNP-only ridge on hybrid-level features
class HybRidge:
    """rows = records; features = A[h]; fit subsets via hybrid counts (exact ridge with intercept)."""
    def __init__(self, A, h, y):
        self.A, self.h, self.y = A, h, y; self.m = A.shape[0]

    def sums(self, idx):
        c = np.bincount(self.h[idx], minlength=self.m).astype(float)
        s = np.bincount(self.h[idx], weights=self.y[idx], minlength=self.m)
        return c, s

    def fit(self, idx, lams, chol=False):
        c, s = self.sums(idx); n = c.sum(); A = self.A
        xb = (c @ A) / n; yb = s.sum() / n
        G = A.T @ (c[:, None] * A); G -= n * np.outer(xb, xb); b = A.T @ s - n * xb * yb
        if chol:
            B = np.column_stack([sla.cho_solve(sla.cho_factor(G + l * np.eye(len(b)), check_finite=False), b, check_finite=False) for l in lams])
        else:
            dd, V = np.linalg.eigh(G); dd = np.clip(dd, 0, None); Vb = V.T @ b
            B = V @ (Vb[:, None] / (dd[:, None] + np.asarray(lams)[None, :]))
        return A @ B + (yb - xb @ B)[None, :]          # per-hybrid predictions (m x L)


def env_curve(env, y, P, envs):
    if not envs: return np.full(P.shape[1], np.nan)
    return mean_env_r_multi(env, y, P, envs)


def hgb_staged(F_tr, r_tr, F_ev):
    m = HistGradientBoostingRegressor(max_iter=max(NITER), **HGB_PARAMS).fit(F_tr, r_tr)
    keep = {k - 1 for k in NITER}; out = []
    for i, p in enumerate(m.staged_predict(F_ev)):
        if i in keep: out.append(p)
    return np.column_stack(out)


def per_env_metrics_all(R, P, envs, new_mask, min_n=30):
    out = []; env = R.Env.values; y = R.y.values; keep = R.Env.isin(envs).values
    for sname, msk, mn in (("all", keep, min_n), ("new", keep & new_mask, 10)):
        if msk.sum() == 0: continue
        for j in range(P.shape[1]):
            met = within_env_metrics(env[msk], y[msk], P[msk, j]); met = met[met.n >= mn].reset_index()
            met["col"] = j; met["subset"] = sname; out.append(met)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_fold(R_tr, R_te, groups, p1, inner_envs, te_envs, models, grid, new_mask, log=print):
    """models: {name: dict(A=hybrid features (m x p), h_tr, h_te, Fd_tr, Fd_te, D=bool)}. Returns fold dict."""
    y_tr, env_tr = R_tr.y.values, R_tr.Env.values; keys = list(groups); all_tr = np.arange(len(R_tr))
    out = dict(models={}, n_train=len(R_tr), n_test=len(R_te))
    for name, spec in models.items():
        t0 = time.time(); hr = HybRidge(spec["A"], spec["h_tr"], y_tr)
        mg_val, cur = {}, {}
        for g in keys:
            idx_in = np.concatenate([groups[k] for k in keys if k != g])
            Ph = hr.fit(idx_in, grid); mg_val[g] = Ph[spec["h_tr"][groups[g]]]
            cur[g] = env_curve(env_tr[groups[g]], y_tr[groups[g]], mg_val[g], inner_envs[g])
        Ph = hr.fit(all_tr, grid); Pte = Ph[spec["h_te"]]
        tscore = env_curve(R_te.Env.values, R_te.y.values, Pte, te_envs)
        c3 = np.nanmean(np.array([cur[g] for g in keys], float), 0)
        sel = dict(P1=int(np.nanargmax(cur[p1])), P2=int(np.nanargmax(tscore)), P3=int(np.nanargmax(c3)))
        res = dict(inner_curves={g: np.round(cur[g], 7).tolist() for g in keys}, test_score=tscore.tolist(), sel=sel,
                   metrics=per_env_metrics_all(R_te, Pte, te_envs, new_mask))
        log(f"    {name}: ridge {time.time()-t0:.0f}s sel {sel}")
        if spec.get("D"):
            t1 = time.time(); li = sel["P3"]; lam = grid[li]
            pair = {}
            for i, u in enumerate(keys):
                for v in keys[i + 1:]:
                    idx_in = np.concatenate([groups[k] for k in keys if k not in (u, v)])
                    Ph = hr.fit(idx_in, [lam], chol=True)[:, 0]
                    pair[(u, v)] = Ph[spec["h_tr"][groups[u]]]; pair[(v, u)] = Ph[spec["h_tr"][groups[v]]]
            Ftr, Fte = spec["Fd_tr"], spec["Fd_te"]; dcur = {}
            for v in keys:
                us = [u for u in keys if u != v]
                idx_in = np.concatenate([groups[u] for u in us])
                tgt = np.concatenate([y_tr[groups[u]] - pair[(u, v)] for u in us])
                S = hgb_staged(Ftr[idx_in], tgt, Ftr[groups[v]])
                dcur[v] = env_curve(env_tr[groups[v]], y_tr[groups[v]], mg_val[v][:, [li]] + S, inner_envs[v])
            cf = np.empty(len(R_tr))
            for u in keys: cf[groups[u]] = mg_val[u][:, li]
            S = hgb_staged(Ftr, y_tr - cf, Fte); Pd = Pte[:, [li]] + S
            dts = env_curve(R_te.Env.values, R_te.y.values, Pd, te_envs)
            dsel = dict(P3=int(np.nanargmax(np.nanmean(np.array([dcur[v] for v in keys], float), 0))), P2=int(np.nanargmax(dts)))
            res["D"] = dict(stage1_lam_idx=li, stage1_lam=float(lam), inner_curves={v: np.round(dcur[v], 7).tolist() for v in keys},
                            test_score=dts.tolist(), sel=dsel, metrics=per_env_metrics_all(R_te, Pd, te_envs, new_mask))
            log(f"    D_noEC({name}): {time.time()-t1:.0f}s n_iter P3 {NITER[dsel['P3']]}")
        out["models"][name] = res
        del hr; gc.collect()
    return out
