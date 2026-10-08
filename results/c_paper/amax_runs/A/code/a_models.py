"""A2 (genomic phenology prediction) and A3 (forward-year reaction-norm ridge) engines.
Requires a_lib.py already exec'd. Expects in namespace: pl (plot table with silk_gdd), he (hybrid x env table),
Dk (imputed dosage, all genotyped hybrids), hyb_index, PCs (all hybrids x 30), seasons.
"""
import numpy as np, pandas as pd

LAM_GRID = 10.0 ** np.arange(-2, 5.0001, 0.5)   # 10^-2 .. 10^5, 15 values


def _ridge_path_svd(X, y, lams):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Uy = U.T @ y
    return [(Vt.T * (s / (s ** 2 + l))) @ Uy for l in lams]


def maturity_target(pl, years, hyb_index):
    t = pl[pl.Year.isin(years) & pl.silk_gdd.notna() & pl.Hybrid.isin(hyb_index)]
    return t.groupby("Hybrid").silk_gdd.mean()


def fit_maturity(pl, Dk, hyb_index, years, lams):
    """Ridge on SNPs (standardized on training hybrids). Returns list of prediction vectors for ALL genotyped hybrids."""
    tgt = maturity_target(pl, years, hyb_index)
    rows = np.array([hyb_index[h] for h in tgt.index])
    X = Dk[rows]; mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd; ybar = tgt.values.mean()
    betas = _ridge_path_svd(Xs, tgt.values - ybar, lams)
    Zall = (Dk - mu) / sd
    return [Zall @ b + ybar for b in betas], len(tgt)


def env_r(he_sub, pred_all, hyb_index, min_n=30, col="silk_gdd"):
    """Within-env Pearson of pred vs observed hybrid-mean `col` for envs with >= min_n genotyped hybrids observed."""
    d = he_sub[he_sub[col].notna() & he_sub.genotyped].copy()
    d["p"] = pred_all[[hyb_index[h] for h in d.Hybrid]]
    out = {}
    for e, g in d.groupby("Env"):
        if len(g) >= min_n and g[col].std() > 0:
            out[e] = float(np.corrcoef(g[col], g.p)[0, 1])
    return pd.Series(out, dtype=float)


# ------------------------------------------------------------------ A3 feature engine
def make_records(he, years):
    r = he[he.Year.isin(years) & he.yield_.notna() & he.genotyped & he.has_season].copy()
    r["y"] = r.yield_ - r.groupby("Env").yield_.transform("mean")   # env-centred hybrid-mean yield
    return r.reset_index(drop=True)


def env_level_ec(recs, pm, seasons):
    """anchor_e = mean predicted silking GDD of the hybrids (records) in env; EC_env (21) at that anchor."""
    anc = pd.Series(pm, index=recs.index).groupby(recs.Env).mean()
    ec, comp = ec_table(seasons, anc.index.values, anc.values)
    ecdf = pd.DataFrame(ec, index=anc.index, columns=EC_NAMES)
    return anc, ecdf


class ZStd:
    """z-score with training statistics; zero-variance columns -> sd 1."""
    def __init__(self, A):
        self.mu = np.nanmean(A, 0); self.sd = np.nanstd(A, 0); self.sd[~(self.sd > 0)] = 1.0
    def __call__(self, A):
        Z = (A - self.mu) / self.sd
        return np.where(np.isfinite(Z), Z, 0.0)


def interact(P, E):
    """row-wise outer product: (n,a) x (n,b) -> (n, a*b)"""
    return (P[:, :, None] * E[:, None, :]).reshape(P.shape[0], -1)


def mean_env_r_multi(env, y, P, envs_eval):
    """P: (n, k) predictions; returns (k,) mean within-env Pearson over envs_eval."""
    rs = []
    for e in envs_eval:
        m = env == e
        yy = y[m] - y[m].mean(); PP = P[m] - P[m].mean(0)
        den = np.sqrt((yy ** 2).sum() * (PP ** 2).sum(0))
        rs.append(np.where(den > 0, (yy @ PP) / np.where(den > 0, den, 1), np.nan))
    return np.nanmean(np.vstack(rs), 0)


def build_fold(R_tr, R_ev, pm_all, hyb_index, Dk, PCs, seasons):
    """Base (M0plus) design for train/eval records; returns dict with standardized matrices and helpers."""
    htr = np.array([hyb_index[h] for h in R_tr.Hybrid]); hev = np.array([hyb_index[h] for h in R_ev.Hybrid])
    pm_tr, pm_ev = pm_all[htr], pm_all[hev]
    anc_tr, ece_tr = env_level_ec(R_tr, pm_tr, seasons)
    anc_ev, ece_ev = env_level_ec(R_ev, pm_ev, seasons)
    E_tr = ece_tr.loc[R_tr.Env].values; E_ev = ece_ev.loc[R_ev.Env].values
    zE = ZStd(E_tr); zP = ZStd(PCs[htr]); zM = ZStd(pm_tr[:, None]); zS = ZStd(Dk[htr])
    def base(h, E, pm):
        Pz = zP(PCs[h]); Ez = zE(E); Mz = zM(pm[:, None])
        return np.hstack([zS(Dk[h]), interact(Pz, Ez), Mz * Ez]), Pz
    Xtr, Pz_tr = base(htr, E_tr, pm_tr); Xev, Pz_ev = base(hev, E_ev, pm_ev)
    zB = ZStd(Xtr); Xtr = zB(Xtr); Xev = zB(Xev)
    y = R_tr.y.values; yc = y - y.mean()
    return dict(Xtr=Xtr, Xev=Xev, yc=yc, htr=htr, hev=hev, pm_tr=pm_tr, pm_ev=pm_ev, E_tr=E_tr, E_ev=E_ev,
                Pz_tr=Pz_tr, Pz_ev=Pz_ev, n_snp=Dk.shape[1], n_pcec=PCs.shape[1] * 21,
                anc_tr=anc_tr, anc_ev=anc_ev, ece_tr=ece_tr, ece_ev=ece_ev)


def delta_block(F, R_tr, R_ev, anchor_tr, anchor_ev, seasons):
    """dEC main (21) + PC x dEC (630), standardized on training records. Also returns W3 completeness fraction."""
    ecij_tr, c_tr = ec_table(seasons, R_tr.Env.values, anchor_tr)
    ecij_ev, c_ev = ec_table(seasons, R_ev.Env.values, anchor_ev)
    d_tr = ecij_tr - F["E_tr"]; d_ev = ecij_ev - F["E_ev"]
    nan_frac = float(np.isnan(d_tr).mean() + np.isnan(d_ev).mean()) / 2
    zD = ZStd(d_tr)
    Dtr = zD(d_tr); Dev = zD(d_ev)
    Xd_tr = np.hstack([Dtr, interact(F["Pz_tr"], Dtr)]); Xd_ev = np.hstack([Dev, interact(F["Pz_ev"], Dev)])
    z2 = ZStd(Xd_tr)
    return z2(Xd_tr), z2(Xd_ev), nan_frac


def shuffle_within_env(values, env, seed):
    rng = np.random.default_rng(seed)
    out = values.copy()
    for e, ii in pd.Series(np.arange(len(env))).groupby(env).indices.items():
        out[ii] = values[rng.permutation(ii)]
    return out


def _variants_anchor(R_tr, R_ev, pm_tr, pm_ev, n_shuffle=10):
    """Anchors for the dEC_ij block for each M1 variant (train, eval)."""
    out = {"M1plus": (pm_tr, pm_ev)}
    env_all = np.concatenate([R_tr.Env.values, R_ev.Env.values]); pm_all = np.concatenate([pm_tr, pm_ev])
    ntr = len(R_tr)
    for s in range(1, n_shuffle + 1):
        sh = shuffle_within_env(pm_all, env_all, s)
        out[f"M1shuffle_s{s}"] = (sh[:ntr], sh[ntr:])
    otr = np.where(np.isfinite(R_tr.silk_gdd.values), R_tr.silk_gdd.values, pm_tr)
    oev = np.where(np.isfinite(R_ev.silk_gdd.values), R_ev.silk_gdd.values, pm_ev)
    out["M1oracle"] = (otr, oev)
    return out


def run_split(R_in_tr, R_val, R_tr, R_te, pm_inner, pm_outer, hyb_index, Dk, PCs, seasons, lam_grid,
              val_envs, n_shuffle=10, log=print):
    import scipy.linalg as sla, time
    res = {}
    # ---------------- inner fold: penalty selection on the last training year
    t0 = time.time()
    F = build_fold(R_in_tr, R_val, pm_inner, hyb_index, Dk, PCs, seasons)
    Gbb = F["Xtr"].T @ F["Xtr"]; bb = F["Xtr"].T @ F["yc"]
    yv = R_val.y.values; ev = R_val.Env.values
    nb0 = F["n_snp"] + F["n_pcec"]
    def select(G, b, Xev):
        d, V = np.linalg.eigh(G); d = np.clip(d, 0, None); Vb = V.T @ b; XV = Xev @ V
        P = np.column_stack([XV @ (Vb / (d + l)) for l in lam_grid])
        sc = mean_env_r_multi(ev, yv, P, val_envs)
        i = int(np.nanargmax(sc)); return i, sc
    sel = {}
    i, sc = select(Gbb[:nb0, :nb0], bb[:nb0], F["Xev"][:, :nb0]); sel["M0"] = (i, sc)
    i, sc = select(Gbb, bb, F["Xev"]); sel["M0plus"] = (i, sc)
    anch = _variants_anchor(R_in_tr, R_val, F["pm_tr"], F["pm_ev"], n_shuffle)
    for name, (atr, aev) in anch.items():
        Xd_tr, Xd_ev, _ = delta_block(F, R_in_tr, R_val, atr, aev, seasons)
        G = np.block([[Gbb, F["Xtr"].T @ Xd_tr], [Xd_tr.T @ F["Xtr"], Xd_tr.T @ Xd_tr]])
        b = np.concatenate([bb, Xd_tr.T @ F["yc"]])
        i, sc = select(G, b, np.hstack([F["Xev"], Xd_ev])); sel[name] = (i, sc)
    log(f"  inner done {time.time()-t0:.0f}s; lam idx: " + ", ".join(f"{k}:{v[0]}" for k, v in sel.items() if not k.startswith('M1shuffle_s') or k.endswith('s1')))
    del F, Gbb
    # ---------------- outer fold: refit on all training years, predict test year
    t0 = time.time()
    F = build_fold(R_tr, R_te, pm_outer, hyb_index, Dk, PCs, seasons)
    Gbb = F["Xtr"].T @ F["Xtr"]; bb = F["Xtr"].T @ F["yc"]
    def solve(G, b, lam):
        c = sla.cho_factor(G + lam * np.eye(G.shape[0]), lower=False, check_finite=False)
        return sla.cho_solve(c, b, check_finite=False)
    preds = {}
    preds["M0"] = F["Xev"][:, :nb0] @ solve(Gbb[:nb0, :nb0], bb[:nb0], lam_grid[sel["M0"][0]])
    preds["M0plus"] = F["Xev"] @ solve(Gbb, bb, lam_grid[sel["M0plus"][0]])
    anch = _variants_anchor(R_tr, R_te, F["pm_tr"], F["pm_ev"], n_shuffle)
    nanf = {}
    for name, (atr, aev) in anch.items():
        Xd_tr, Xd_ev, nanf[name] = delta_block(F, R_tr, R_te, atr, aev, seasons)
        G = np.block([[Gbb, F["Xtr"].T @ Xd_tr], [Xd_tr.T @ F["Xtr"], Xd_tr.T @ Xd_tr]])
        b = np.concatenate([bb, Xd_tr.T @ F["yc"]])
        preds[name] = np.hstack([F["Xev"], Xd_ev]) @ solve(G, b, lam_grid[sel[name][0]])
    log(f"  outer done {time.time()-t0:.0f}s")
    lam_sel = {k: float(lam_grid[v[0]]) for k, v in sel.items()}
    val_score = {k: float(v[1][v[0]]) for k, v in sel.items()}
    diag = dict(anchor_env_test=F["anc_ev"].to_dict(), ec_nan_frac=nanf, n_train=len(R_tr), n_test=len(R_te),
                n_inner_train=len(R_in_tr), n_val=len(R_val))
    return preds, lam_sel, val_score, diag


# ---- memory-efficient overrides (in-place standardisation; peak RSS kept < 6 GB) ----
def _col_stats(X, c0, c1, chunk=8192):
    n = X.shape[0]; s = np.zeros(c1 - c0); ss = np.zeros(c1 - c0)
    for i in range(0, n, chunk):
        B = X[i:i + chunk, c0:c1]; s += B.sum(0); ss += (B * B).sum(0)
    mu = s / n; var = np.maximum(ss / n - mu ** 2, 0); sd = np.sqrt(var); sd[~(sd > 1e-12)] = 1.0
    return mu, sd


def _std_inplace(X, c0, c1, mu, sd, chunk=8192):
    for i in range(0, X.shape[0], chunk):
        B = X[i:i + chunk, c0:c1]; B -= mu; B /= sd


def build_fold(R_tr, R_ev, pm_all, hyb_index, Dk, PCs, seasons):
    htr = np.array([hyb_index[h] for h in R_tr.Hybrid]); hev = np.array([hyb_index[h] for h in R_ev.Hybrid])
    pm_tr, pm_ev = pm_all[htr], pm_all[hev]
    anc_tr, ece_tr = env_level_ec(R_tr, pm_tr, seasons)
    anc_ev, ece_ev = env_level_ec(R_ev, pm_ev, seasons)
    E_tr = ece_tr.loc[R_tr.Env].values; E_ev = ece_ev.loc[R_ev.Env].values
    zE = ZStd(E_tr); zP = ZStd(PCs[htr]); zM = ZStd(pm_tr[:, None])
    # SNP standardisation from training-record-weighted hybrid counts (identical to column z-score on records)
    cnt = np.bincount(htr, minlength=Dk.shape[0]).astype(float); w = cnt / cnt.sum()
    smu = w @ Dk; ssd = np.sqrt(np.maximum(w @ (Dk ** 2) - smu ** 2, 0)); ssd[~(ssd > 1e-12)] = 1.0
    Dz = (Dk - smu) / ssd
    ns, npc = Dk.shape[1], PCs.shape[1] * 21
    def base(h, E, pm):
        X = np.empty((len(h), ns + npc + 21))
        X[:, :ns] = Dz[h]
        Pz = zP(PCs[h]); Ez = zE(E); Mz = zM(pm[:, None])
        X[:, ns:ns + npc] = interact(Pz, Ez); X[:, ns + npc:] = Mz * Ez
        return X, Pz
    Xtr, Pz_tr = base(htr, E_tr, pm_tr); Xev, Pz_ev = base(hev, E_ev, pm_ev)
    mu, sd = _col_stats(Xtr, ns, Xtr.shape[1])
    _std_inplace(Xtr, ns, Xtr.shape[1], mu, sd); _std_inplace(Xev, ns, Xev.shape[1], mu, sd)
    y = R_tr.y.values; yc = y - y.mean()
    return dict(Xtr=Xtr, Xev=Xev, yc=yc, htr=htr, hev=hev, pm_tr=pm_tr, pm_ev=pm_ev, E_tr=E_tr, E_ev=E_ev,
                Pz_tr=Pz_tr, Pz_ev=Pz_ev, n_snp=ns, n_pcec=npc,
                anc_tr=anc_tr, anc_ev=anc_ev, ece_tr=ece_tr, ece_ev=ece_ev)


def delta_block(F, R_tr, R_ev, anchor_tr, anchor_ev, seasons):
    ecij_tr, _ = ec_table(seasons, R_tr.Env.values, anchor_tr)
    ecij_ev, _ = ec_table(seasons, R_ev.Env.values, anchor_ev)
    d_tr = ecij_tr - F["E_tr"]; d_ev = ecij_ev - F["E_ev"]
    nan_frac = float((np.isnan(d_tr).sum() + np.isnan(d_ev).sum()) / (d_tr.size + d_ev.size))
    zD = ZStd(d_tr); Dtr = zD(d_tr); Dev = zD(d_ev)
    Xd_tr = np.hstack([Dtr, interact(F["Pz_tr"], Dtr)]); Xd_ev = np.hstack([Dev, interact(F["Pz_ev"], Dev)])
    mu, sd = _col_stats(Xd_tr, 0, Xd_tr.shape[1])
    _std_inplace(Xd_tr, 0, Xd_tr.shape[1], mu, sd); _std_inplace(Xd_ev, 0, Xd_ev.shape[1], mu, sd)
    return Xd_tr, Xd_ev, nan_frac
