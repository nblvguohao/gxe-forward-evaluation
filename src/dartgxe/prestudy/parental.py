"""Pre-study (docs/prestudy_parental_2026-09-26.md): repeatability ceiling (C2) and the linear
parental-decomposition probe (C3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dartgxe.baselines.common import Fold, val_score
from dartgxe.baselines.linear import Ridge
from dartgxe.features.fit import EnvScaler, GenoData


# ---------------------------------------------------------------- C2: repeatability ceiling
def env_repeatability(plot: pd.DataFrame) -> pd.DataFrame:
    """One-way random model per environment (hybrid as the factor) on plot yields.
    σ²_g = (MSB − MSW)/n0 (n0: unbalanced-design coefficient), H = σ²_g /(σ²_g + σ²_ε / n_h),
    n_h = harmonic mean of plots per hybrid. Ceiling of within-env correlation with cell means = √H.
    Blocks/spatial trends are not removed, so σ²_ε is inflated and H is conservative (a lower bound
    on the ceiling)."""
    rows = []
    for e, g in plot.groupby("env"):
        c = g.groupby("genotype")["y"]
        n_i = c.size().to_numpy()
        if (n_i > 1).sum() < 10:
            rows.append({"env": e, "H": np.nan, "n_hyb": len(n_i), "mean_reps": n_i.mean()})
            continue
        N, a = n_i.sum(), len(n_i)
        grand = g["y"].mean()
        ssb = (n_i * (c.mean().to_numpy() - grand) ** 2).sum()
        ssw = ((g["y"] - c.transform("mean")) ** 2).sum()
        msb, msw = ssb / (a - 1), ssw / (N - a)
        n0 = (N - (n_i ** 2).sum() / N) / (a - 1)
        s2g = max((msb - msw) / n0, 0.0)
        nh = a / (1.0 / n_i).sum()
        H = s2g / (s2g + msw / nh) if s2g > 0 else 0.0
        rows.append({"env": e, "H": H, "ceiling": np.sqrt(H), "n_hyb": a, "mean_reps": n_i.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- parental genotypes
def infer_parents(geno: GenoData, min_hybrids: int = 3):
    """Hybrid dosage h ∈ {0,1,2} (alt count) = p1 + t with homozygous inbreds (alleles 0/1).
    Tester allele t_j: majority of its hybrids' informative calls (h=0 → t=0, h=2 → t=1).
    Parent-1 allele = h − t, clipped to {0,1}; averaged over all hybrids of that parent-1.
    Undetermined values are NaN (imputed later on the training fold)."""
    ids = geno.ids
    p1 = pd.Series([h.split("/", 1)[0] for h in ids])
    p2 = pd.Series([h.split("/", 1)[1] if "/" in h else "" for h in ids])
    X = geno.X.astype(np.float32)
    X[X < 0] = np.nan
    tester_geno = {}
    for t, idx in p2.groupby(p2).groups.items():
        H = X[list(idx)]
        n2, n0 = (H == 2).sum(0), (H == 0).sum(0)
        g = np.where(n2 > n0, 1.0, np.where(n0 > n2, 0.0, np.nan))
        if len(idx) < min_hybrids:
            g = np.where((n2 + n0) > 0, g, np.nan)
        tester_geno[t] = g
    T = np.stack([tester_geno[t] for t in p2])
    P = np.clip(X - T, 0, 1)
    p1_geno = pd.DataFrame(P).groupby(p1.to_numpy()).mean()  # nanmean over hybrids of the same parent-1
    return p1, p2, p1_geno, pd.DataFrame(tester_geno).T


class ParentFeatures:
    """Standardised parent-1 and tester allele matrices per hybrid, fitted on training hybrids."""

    def __init__(self, geno: GenoData, parents, fit_hybrids):
        p1, p2, p1g, tg = parents
        self.hidx = {h: i for i, h in enumerate(geno.ids)}
        A = p1g.loc[p1.to_numpy()].to_numpy(np.float32)
        B = tg.loc[p2.to_numpy()].to_numpy(np.float32)
        rows = [self.hidx[h] for h in sorted(set(fit_hybrids))]
        self.blocks = []
        for M in (A, B):
            fill = np.nanmean(M[rows], 0)
            fill = np.where(np.isfinite(fill), fill, 0.0)
            M = np.where(np.isnan(M), fill, M)
            mu, sd = M[rows].mean(0), M[rows].std(0)
            sd[sd < 1e-6] = 1.0
            self.blocks.append((M - mu) / sd)
        self.tester = p2.to_numpy()

    def get(self, hybrids):
        ix = [self.hidx[h] for h in hybrids]
        return self.blocks[0][ix], self.blocks[1][ix], self.tester[ix]


def fit_probe(fold: Fold, geno: GenoData, ec: pd.DataFrame, parents, m: int = 10):
    """P1: [parent-1 block, tester block] with separate penalties (5 λ x 4 ratios = 20 points).
    P2: P1's choice + tester-identity x EC-PC interaction block (5 λ_int x 4 scales = 20 points)."""
    ratios = [0.1, 1.0, 10.0, 100.0]
    lams = np.logspace(-3, 1, 5)

    def design(roles):
        fr = fold.rows(*roles)
        pf = ParentFeatures(geno, parents, fr["genotype"].unique())
        es = EnvScaler.fit(ec, fr["env"].unique())
        envs = sorted(fold.cells["env"].unique())
        Xe = es.transform(ec, envs)
        tr = [envs.index(e) for e in sorted(fr["env"].unique())]
        mu = Xe[tr].mean(0)
        _, S, Vt = np.linalg.svd(Xe[tr] - mu, full_matrices=False)
        Ue = (Xe - mu) @ Vt[:m].T / (S[:m] / np.sqrt(len(tr)))
        epos = {e: i for i, e in enumerate(envs)}
        testers = sorted(set(pf.tester[[pf.hidx[h] for h in fr["genotype"].unique()]]))
        tpos = {t: i for i, t in enumerate(testers)}

        def feats(rows, ratio, inter=None):
            A, B, tt = pf.get(rows["genotype"])
            F = [A, B / np.sqrt(ratio)]
            if inter is not None:
                oh = np.zeros((len(rows), len(testers)), np.float32)
                for i, t in enumerate(tt):
                    if t in tpos:
                        oh[i, tpos[t]] = 1.0  # unseen testers get no tester x E term
                E = Ue[[epos[e] for e in rows["env"]]]
                F.append((oh[:, :, None] * E[:, None, :]).reshape(len(rows), -1) * inter)
            return np.concatenate(F, 1)
        return fr, feats

    va = fold.rows("val")
    fr, feats = design(("train",))
    best1 = (-np.inf, None, None)
    for r in ratios:
        R = Ridge(feats(fr, r), fr["env"].to_numpy(), fr["y"].to_numpy())
        Fv = feats(va, r)
        for l in lams:
            s = val_score(va, Fv @ R.coef(l))
            if s > best1[0]:
                best1 = (s, r, l)
    _, r1, l1 = best1
    best2 = (-np.inf, None, None)
    for inter in (0.1, 0.3, 1.0, 3.0):
        R = Ridge(feats(fr, r1, inter), fr["env"].to_numpy(), fr["y"].to_numpy())
        Fv = feats(va, r1, inter)
        for l in lams:
            s = val_score(va, Fv @ R.coef(l))
            if s > best2[0]:
                best2 = (s, inter, l)
    _, i2, l2 = best2
    if fold.forward:
        fr, feats = design(("train", "val", "refit_extra"))
    te = fold.rows("test")
    out = {}
    for name, kw, lam in (("PP1_parental", {}, l1), ("PP2_parental_testerE", {"inter": i2}, l2)):
        R = Ridge(feats(fr, r1, **kw), fr["env"].to_numpy(), fr["y"].to_numpy())
        beta = R.coef(lam)
        g_fr = feats(fr, r1, **kw) @ beta
        known = pd.Series(fr["y"].to_numpy() - g_fr).groupby(fr["env"].to_numpy()).mean()
        mu_e = pd.Series(te["env"].to_numpy()).map(known).fillna(float((fr["y"].to_numpy() - g_fr).mean())).to_numpy()
        out[name] = mu_e + feats(te, r1, **kw) @ beta
    info = {"PP1_parental": {"ratio_tester_p1": r1, "lambda_rel": l1, "val_spearman": best1[0], "grid": 20},
            "PP2_parental_testerE": {"ratio_tester_p1": r1, "inter_scale": i2, "lambda_rel": l2, "val_spearman": best2[0], "grid": 20}}
    return te, out, info
