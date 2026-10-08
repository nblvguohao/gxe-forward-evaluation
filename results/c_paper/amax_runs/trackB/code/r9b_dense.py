"""prereg v9 dense genotypes: hybrid dosages at the P8 (Lima) 98k positions built as parent1 + parent2 inbred calls
from the G2F inbred 437k VCF (v5 coordinates), then the preregistered validity check against the P8 GENO matrix
(per-SNP dosage concordance median >= 0.98 and kinship correlation >= 0.95 on overlapping hybrids).
Hybrid dosage = (d_P1 + d_P2) / 2 with d = ALT-allele count of the inbred call (0/1/2), rounded to 0/1/2;
missing if either parent call is missing. Allele orientation vs P8 is harmonised per SNP on the overlap hybrids.
Usage (from runs/trackB): python code/r9b_dense.py <threads>
Writes cache/dense_G.npy (int8, -1 missing), cache/dense_meta.json, out/dense_validity.json.
"""
import os, sys, json, time, re
TH = sys.argv[1] if len(sys.argv) > 1 else "8"
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = TH
import numpy as np, pandas as pd
BASE = os.environ.get("R9B_BASE", "<server>/egp_gxe"); RUN = f"{BASE}/runs/trackB"
VCF = os.environ.get("R9B_VCF", f"{BASE}/data/g2f_inbred_437k/inbreds_G2F_2014-2023_437k.vcf")
KEY = f"{BASE}/data/g2f_inbred_437k/key_inbreds_G2F_2014-2023.txt"
T25 = f"{BASE}/data/g2f_2025/Training_data"
t0 = time.time()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
assert os.path.exists(VCF), "437k VCF not present"
MAP = pd.read_csv(f"{RUN}/cache/MAP.csv"); lm = json.load(open(f"{RUN}/cache/lima_meta.json"))
assert list(MAP.name) == lm["snps"]
want = {(str(c), int(p)): j for j, (c, p) in enumerate(zip(MAP.chr, MAP.pos))}
# ---- stream VCF, keep 98k positions
samples = None; rows = {}; ref_alt = {}; n_lines = 0; chrom_names = set()
code = {"0/0": 0, "0|0": 0, "0/1": 1, "1/0": 1, "0|1": 1, "1|0": 1, "1/1": 2, "1|1": 2}
with open(VCF) as fh:
    for line in fh:
        if line.startswith("##"): continue
        if line.startswith("#CHROM"):
            samples = line.rstrip("\n").split("\t")[9:]; continue
        n_lines += 1
        t1 = line.find("\t"); t2 = line.find("\t", t1 + 1); ch = line[:t1]; pos = int(line[t1 + 1:t2])
        chn = re.sub(r"^(chr|Chr)", "", ch); chrom_names.add(ch)
        j = want.get((chn, pos))
        if j is None: continue
        p = line.rstrip("\n").split("\t")
        if len(p[4].split(",")) > 1: continue                        # multi-allelic: skip
        g = np.fromiter((code.get(x.split(":", 1)[0], -1) for x in p[9:]), dtype=np.int8, count=len(p) - 9)
        rows[j] = g; ref_alt[j] = (p[3], p[4])
log(f"VCF lines {n_lines}, samples {len(samples)}, matched 98k positions {len(rows)} / {len(MAP)}; chrom names {sorted(chrom_names)[:12]}")
# ---- inbred name matching
tr = pd.read_csv(f"{T25}/1_Training_Trait_Data_2014_2023.csv", low_memory=False)
par = tr[["Hybrid", "Hybrid_Parent1", "Hybrid_Parent2"]].dropna()
par = (par.value_counts().reset_index().sort_values(["Hybrid", "count", "Hybrid_Parent1"], ascending=[True, False, True]).groupby("Hybrid").first())
key = pd.read_csv(KEY, sep="\t", dtype=str)
sidx = {s: i for i, s in enumerate(samples)}; sup = {s.upper(): i for i, s in enumerate(samples)}
alt = {}
for _, r in key.iterrows():
    a, b = str(r.get("Cultivar", "")).strip(), str(r.get("Alternative name", "")).strip()
    if a and b and b != "nan":
        alt[b.upper()] = a; alt[a.upper()] = a
def find(nm):
    if nm in sidx: return sidx[nm], "exact"
    if nm.upper() in sup: return sup[nm.upper()], "case"
    a = alt.get(nm.upper())
    if a is not None:
        if a in sidx: return sidx[a], "alt"
        if a.upper() in sup: return sup[a.upper()], "alt"
    return None, None
how = {}; hyb, pa, pb = [], [], []
for h, r in par.iterrows():
    ia, wa = find(str(r.Hybrid_Parent1)); ib, wb = find(str(r.Hybrid_Parent2))
    how[wa] = how.get(wa, 0) + 1; how[wb] = how.get(wb, 0) + 1
    if ia is not None and ib is not None: hyb.append(h); pa.append(ia); pb.append(ib)
log(f"hybrids in trait file {len(par)}; buildable {len(hyb)}; parent match kinds {how}")
# ---- build hybrid dosages (Lima SNP order)
p = len(MAP); H = np.full((len(hyb), p), -1, np.int8); pa = np.array(pa); pb = np.array(pb)
for j, g in rows.items():
    a = g[pa].astype(np.int16); b = g[pb].astype(np.int16); ok = (a >= 0) & (b >= 0)
    v = np.where(ok, np.rint((a + b) / 2.0), -1).astype(np.int8); H[:, j] = v
# ---- validity check vs P8 GENO
Lg = np.load(f"{RUN}/cache/lima_G.npy"); ln = lm["names"]; li = {h: i for i, h in enumerate(ln)}
ov = [(i, li[h]) for i, h in enumerate(hyb) if h in li]; bi = np.array([a for a, _ in ov]); lj = np.array([b for _, b in ov])
cols = np.array(sorted(rows)); Hb = H[bi][:, cols].astype(np.int16); Lb = Lg[lj][:, cols].astype(np.int16)
okm = (Hb >= 0) & (Lb >= 0)
agree = ((Hb == Lb) & okm).sum(0) / np.maximum(okm.sum(0), 1)
agree_f = (((2 - Hb) == Lb) & okm).sum(0) / np.maximum(okm.sum(0), 1)
flip = agree_f > agree; conc = np.where(flip, agree_f, agree); nobs = okm.sum(0)
use = nobs >= 50
Hb2 = np.where(flip[None, :], 2 - Hb, Hb); Hb2 = np.where(Hb >= 0, Hb2, -1)
# kinship on overlap hybrids, SNPs with MAF >= 0.05 in P8 and >= 50 joint calls; mean-imputed, standardised
def kin(M):
    X = M.astype(float); X[X < 0] = np.nan; mu = np.nanmean(X, 0); X = np.where(np.isnan(X), mu, X); sd = X.std(0); sd[sd < 1e-9] = 1
    Z = (X - mu) / sd; return Z @ Z.T / Z.shape[1]
Lf = Lb.astype(float); Lf[Lf < 0] = np.nan; maf = np.nanmean(Lf, 0) / 2; maf = np.minimum(maf, 1 - maf)
ks = use & (maf >= 0.05)
K1 = kin(Hb2[:, ks]); K2 = kin(Lb[:, ks]); iu = np.triu_indices(len(ov), 1)
kcor = float(np.corrcoef(K1[iu], K2[iu])[0, 1]); dcor = float(np.corrcoef(np.diag(K1), np.diag(K2))[0, 1])
med = float(np.median(conc[use]))
passed = bool(med >= 0.98 and kcor >= 0.95)
# orient the whole built matrix to P8 coding (so that EXT uses the same allele coding as the P8 matrix)
Hf = H.copy()
for k_, j in enumerate(cols):
    if flip[k_]:
        c_ = Hf[:, j]; Hf[:, j] = np.where(c_ >= 0, 2 - c_, -1)
np.save(f"{RUN}/cache/dense_G.npy", Hf)
res = dict(vcf_lines=n_lines, vcf_samples=len(samples), p8_snps=p, matched_positions=len(rows), matched_frac=len(rows) / p,
           hybrids_trait_2014_2023=int(len(par)), hybrids_built=len(hyb), parent_match_kinds={str(k): v for k, v in how.items()},
           overlap_hybrids_with_P8=len(ov), snps_evaluated=int(use.sum()), flipped_frac=float(flip[use].mean()),
           concordance_median=med, concordance_q10_q90=[float(np.quantile(conc[use], 0.1)), float(np.quantile(conc[use], 0.9))],
           kinship_offdiag_cor=kcor, kinship_diag_cor=dcor, kinship_n_snps=int(ks.sum()),
           thresholds=dict(concordance_median=0.98, kinship_cor=0.95), validity_pass=passed,
           built_hybrids_by_year={int(y): int(tr[(tr.Year == y) & tr.Hybrid.isin(set(hyb))].Hybrid.nunique()) for y in range(2014, 2024)},
           trait_hybrids_by_year={int(y): int(tr[tr.Year == y].Hybrid.nunique()) for y in range(2014, 2024)}, seconds=round(time.time() - t0))
json.dump(res, open(f"{RUN}/out/dense_validity.json", "w"), indent=1)
json.dump(dict(names=hyb, snps=lm["snps"], validity_pass=passed), open(f"{RUN}/cache/dense_meta.json", "w"))
log(json.dumps(res))
