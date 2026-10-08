"""Reusable helpers for prereg angle-A pilot (genotype-specific phenology-aligned exposure) on G2F 2014-2021.
All functions are deterministic; no simulated data. Written for prereg_AB_gonogo.json (sha256 df52f18a...).
"""
import os, hashlib, json
import numpy as np
import pandas as pd

WVARS = ["T2M_MAX", "T2M_MIN", "VPD", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT", "HOT32"]
WVAR_AGG = {"T2M_MAX": "mean", "T2M_MIN": "mean", "VPD": "mean", "PRECTOTCORR": "sum",
            "ALLSKY_SFC_SW_DWN": "mean", "GWETROOT": "mean", "HOT32": "sum"}
WINDOWS = {"W1_pre": (-400, -50), "W2_flowering": (-50, 150), "W3_grainfill": (150, 750)}
EC_NAMES = [f"{w}__{v}" for w in WINDOWS for v in WVARS]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def daily_gdd(tmax, tmin):
    """prereg: max(0, (min(Tmax,30)+max(Tmin,10))/2 - 10)."""
    return np.maximum(0.0, (np.minimum(tmax, 30.0) + np.maximum(tmin, 10.0)) / 2.0 - 10.0)


def tetens_vpd(t2m, rh):
    es = 0.6108 * np.exp(17.27 * t2m / (t2m + 237.3))
    return es * (1.0 - rh / 100.0)


def build_env_season(wx, plant_median):
    """For each env: daily arrays from env median planting date to Dec 31.
    Returns dict env -> dict(cum=cumGDD through end of day k (k=0 planting day), prev=cum before day k,
    P=prefix sums of each variable (length n+1), n=ndays)."""
    out = {}
    wx = wx.sort_values(["Env", "Date_p"])
    for env, g in wx.groupby("Env", sort=False):
        if env not in plant_median.index:
            continue
        d0 = plant_median.loc[env]
        g = g[g.Date_p >= d0]
        if len(g) == 0 or g.Date_p.iloc[0] != d0:
            continue
        gdd = daily_gdd(g.T2M_MAX.values, g.T2M_MIN.values)
        cum = np.cumsum(gdd)
        prev = cum - gdd
        vals = np.column_stack([
            g.T2M_MAX.values, g.T2M_MIN.values, tetens_vpd(g.T2M.values, g.RH2M.values),
            g.PRECTOTCORR.values, g.ALLSKY_SFC_SW_DWN.values, g.GWETROOT.values,
            (g.T2M_MAX.values > 32.0).astype(float)])
        P = np.vstack([np.zeros((1, vals.shape[1])), np.cumsum(vals, axis=0)])
        out[env] = dict(cum=cum, prev=prev, P=P, n=len(g), gdd=gdd)
    return out


def silk_gdd_from_dap(season, dap):
    """GDD accumulated over the first `dap` days starting at the env median planting day (days 0..dap-1)."""
    dap = np.asarray(dap, dtype=float)
    res = np.full(dap.shape, np.nan)
    ok = np.isfinite(dap) & (dap >= 1) & (dap <= season["n"])
    res[ok] = season["cum"][dap[ok].astype(int) - 1]
    return res


def ec_for_anchors(season, anchors):
    """Compute the 21 ECs for an array of anchor silking GDDs within one env.
    Day k is in window [S+lo, S+hi] if its accumulation interval (prev[k], cum[k]] overlaps it,
    i.e. cum[k] > S+lo and prev[k] < S+hi. Days start at the median planting day and end at Dec 31.
    Returns (n_anchor x 21) array and a boolean (n_anchor x 3) 'window reached upper bound before Dec 31'."""
    anchors = np.asarray(anchors, dtype=float)
    cum, prev, P = season["cum"], season["prev"], season["P"]
    nv = P.shape[1]
    out = np.full((len(anchors), 3 * nv), np.nan)
    complete = np.zeros((len(anchors), 3), dtype=bool)
    fin = np.isfinite(anchors)
    for wi, (lo, hi) in enumerate(WINDOWS.values()):
        a = anchors[fin]
        start = np.searchsorted(cum, a + lo, side="right")      # first k with cum[k] > S+lo
        end = np.searchsorted(prev, a + hi, side="left")        # first k with prev[k] >= S+hi -> exclusive end
        nd = end - start
        good = nd > 0
        S = (P[end] - P[start])
        blk = np.full((len(a), nv), np.nan)
        for vi, v in enumerate(WVARS):
            if WVAR_AGG[v] == "mean":
                blk[good, vi] = S[good, vi] / nd[good]
            else:
                blk[good, vi] = S[good, vi]
        out[np.where(fin)[0], wi * nv:(wi + 1) * nv] = blk
        complete[np.where(fin)[0], wi] = cum[-1] >= a + hi
    return out, complete


def ec_table(season_by_env, env_arr, anchor_arr):
    """Vectorised over records: env_arr (n,), anchor_arr (n,) -> (n,21) ECs, (n,3) completeness."""
    env_arr = np.asarray(env_arr)
    out = np.full((len(env_arr), 21), np.nan)
    comp = np.zeros((len(env_arr), 3), dtype=bool)
    idx = pd.Series(np.arange(len(env_arr))).groupby(env_arr).indices
    for env, ii in idx.items():
        if env not in season_by_env:
            continue
        o, c = ec_for_anchors(season_by_env[env], anchor_arr[ii])
        out[ii] = o
        comp[ii] = c
    return out, comp


def parse_vcf_dosage(path):
    """Return (sample_names, snp_ids, dosage float32 [n_samples x n_snps] with NaN missing).
    Dosage = count of ALT allele (0/1/2)."""
    samples, ids, rows = None, [], []
    code = {"0/0": 0.0, "0/1": 1.0, "1/0": 1.0, "1/1": 2.0, "0|0": 0.0, "0|1": 1.0, "1|0": 1.0, "1|1": 2.0}
    with open(path) as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            p = line.rstrip("\n").split("\t")
            ids.append(p[2])
            gts = [x.split(":")[0] for x in p[9:]]
            rows.append(np.array([code.get(g, np.nan) for g in gts], dtype=np.float32))
    return samples, ids, np.vstack(rows).T


def ridge_eig(G, b):
    """Eigendecomposition helper for ridge on a standardized Gram G = Xs'Xs, b = Xs'y."""
    d, V = np.linalg.eigh(G)
    d = np.clip(d, 0, None)
    return d, V, V.T @ b


def within_env_metrics(env, y, yhat, min_n=2, top_frac=0.10):
    """Per-env Pearson, Spearman, top-10% overlap."""
    from scipy.stats import pearsonr, spearmanr
    df = pd.DataFrame(dict(env=env, y=y, p=yhat))
    rows = []
    for e, g in df.groupby("env", sort=True):
        if len(g) < min_n or g.y.std() == 0 or g.p.std() == 0:
            rows.append((e, len(g), np.nan, np.nan, np.nan)); continue
        r = np.corrcoef(g.y, g.p)[0, 1]
        rs = spearmanr(g.y, g.p).correlation
        k = max(1, int(round(top_frac * len(g))))
        top_o = set(g.nlargest(k, "y").index); top_p = set(g.nlargest(k, "p").index)
        rows.append((e, len(g), r, rs, len(top_o & top_p) / k))
    return pd.DataFrame(rows, columns=["Env", "n", "pearson", "spearman", "top10_overlap"]).set_index("Env")


def paired_bootstrap(diff, n_boot=2000, seed=20260928):
    """diff: per-env differences (NaNs dropped). Percentile 95% CI of the mean."""
    d = np.asarray(diff, dtype=float); d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
