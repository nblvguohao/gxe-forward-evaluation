"""prereg v10 track B on amax: Part G (MG_98k - MG_2k on G2F 2024; EXT3 re-scoring done locally) and Part W (ESWYT W1-W4).
Run from <server>/egp_gxe/runs/trackB.  Usage: python code/r10b.py <step> [threads]
steps: G_dense (parse repaired 437k VCF at the P8 98k positions -> inbred matrix; hybrid dosages 2014-2024)
       G_fit   (train 2014-2023, test 2024: MG_2k and MG_98k on the 98k-buildable population, P1/P2/P3)
       W_fit   (ESWYT: FORWARD and INTERNAL_LONO for test nurseries 34-37; MG and D_noEC(PC1..30), P1/P2/P3)
2024 data are read ONLY inside the G_dense / G_fit code paths (prereg v10 reporting rule).
Engine: r9b_lib.py (R9 track B; HybRidge, run_fold with the v8-lib D definition) + a_lib/a_models/r6_lib (round-1 code).
"""
import os, sys, json, pickle, time, gc, hashlib, re
STEP = sys.argv[1]; TH = sys.argv[2] if len(sys.argv) > 2 else "24"
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = TH
BASE = "<server>/egp_gxe"; RUN = f"{BASE}/runs/trackB"; CODE = f"{BASE}/egp_bundle/code"
T24 = f"{BASE}/data/g2f_2025/Testing_data_SEALED"; ESW = f"{BASE}/data/eswyt"
VCF = f"{BASE}/data/g2f_inbred_437k/inbreds_G2F_2014-2023_437k.vcf"   # repaired copy (moved here); sha asserted below
VCF_SHA = "66597e1b44018eab50bd3a59e3219f37732aa3e06bac3fb9fcc08e53caf731d7"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
assert sha(f"{RUN}/code/prereg_v10.json") == "dfc8e990551241c72d04e3d74f2526dca52209d677884bbcd88edfb6644e015b"
for f, hh in {"a_lib.py": "52ea0f3fb94e1492170cdc54ccdd39bbf9d96b2657de522b5322917c75144ab4",
              "a_models.py": "939864a6464ff1d2185f4d8900b78a5ae1f45d810a683c3329258047a067793f",
              "r6_lib.py": "7dc20471fc4aa6c242c9c46e81d5b2459e8137808cc5be55e200b798e44afb96"}.items():
    assert sha(f"{CODE}/{f}") == hh, f
    exec(open(f"{CODE}/{f}").read())
assert sha(f"{RUN}/code/r9b_lib.py") == "369c0674c16f0379b54086efd78e2d082fb558d88ab0e276e681c4076942f7a8"
exec(open(f"{RUN}/code/r9b_lib.py").read())
pre = json.load(open(f"{RUN}/code/prereg_v10.json")); GRID = np.array(pre["tuning"]["grid"], float)
assert np.allclose(GRID, V8_GRID)
os.makedirs(f"{RUN}/out10", exist_ok=True); os.makedirs(f"{RUN}/cache", exist_ok=True)
LOGF = open(f"{RUN}/logs/r10_{STEP}.log", "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


# ============================================================== Part G
if STEP in ("G_dense", "G_fit"):
    d = pickle.load(open(f"{RUN}/cache/data.pkl", "rb"))            # R9 prep (2025 release 2014-2023)
    tr, samples, hyb_index = d["tr"], d["samples"], d["hyb_index"]
    obs24 = pd.read_csv(f"{T24}/7_Testing_Observed_Values.csv")      # Env, Hybrid, Yield_Mg_ha (env x hybrid means)

if STEP == "G_dense":
    assert sha(VCF) == VCF_SHA, "437k VCF is not the repaired copy"
    lm = json.load(open(f"{RUN}/cache/lima_meta.json")); MAP = pd.read_csv(f"{RUN}/cache/MAP.csv"); assert list(MAP.name) == lm["snps"]
    want = {(str(c), int(p)): j for j, (c, p) in enumerate(zip(MAP.chr, MAP.pos))}
    code = {"0/0": 0, "0|0": 0, "0/1": 1, "1/0": 1, "0|1": 1, "1|0": 1, "1/1": 2, "1|1": 2}
    vcf_samples = None; I = None
    with open(VCF) as fh:
        for line in fh:
            if line.startswith("##"): continue
            if line.startswith("#CHROM"):
                vcf_samples = line.rstrip("\n").split("\t")[9:]; I = np.full((len(MAP), len(vcf_samples)), -1, np.int8); continue
            t1 = line.find("\t"); t2 = line.find("\t", t1 + 1)
            j = want.get((re.sub(r"^(chr|Chr)", "", line[:t1]), int(line[t1 + 1:t2])))
            if j is None: continue
            p = line.rstrip("\n").split("\t")
            if len(p[4].split(",")) > 1: continue
            I[j] = np.fromiter((code.get(x.split(":", 1)[0], -1) for x in p[9:]), dtype=np.int8, count=len(p) - 9)
    np.save(f"{RUN}/cache/inbred98k.npy", I); json.dump(vcf_samples, open(f"{RUN}/cache/inbred98k_samples.json", "w"))
    # parents: trait-file columns (modal orientation) for 2014-2023 hybrids; otherwise split of the hybrid name at '/'
    par = d["par"]; sidx = {s: i for i, s in enumerate(vcf_samples)}; sup = {s.upper(): i for i, s in enumerate(vcf_samples)}
    key = pd.read_csv(f"{BASE}/data/g2f_inbred_437k/key_inbreds_G2F_2014-2023.txt", sep="\t", dtype=str); alt = {}
    for _, r in key.iterrows():                                        # same alternative-name map as R9 r9b_dense.py
        a_, b_ = str(r.get("Cultivar", "")).strip(), str(r.get("Alternative name", "")).strip()
        if a_ and b_ and b_ != "nan": alt[b_.upper()] = a_; alt[a_.upper()] = a_
    def find(nm):
        nm = str(nm).strip()
        if nm in sidx: return sidx[nm]
        if nm.upper() in sup: return sup[nm.upper()]
        a_ = alt.get(nm.upper())
        if a_ is not None:
            if a_ in sidx: return sidx[a_]
            if a_.upper() in sup: return sup[a_.upper()]
        return None
    hybs = sorted(set(tr.Hybrid.dropna()) | set(obs24.Hybrid.dropna()))
    rows, names, src = [], [], {}
    for h in hybs:
        if h in par.index: a, b, s = par.loc[h, "Hybrid_Parent1"], par.loc[h, "Hybrid_Parent2"], "trait_columns"
        else:
            sp = h.split("/", 1); a, b, s = sp[0], (sp[1] if len(sp) > 1 else None), "name_split"
        ia, ib = find(a), (find(b) if b is not None else None)
        src[s] = src.get(s, 0) + 1
        if ia is not None and ib is not None: rows.append((ia, ib)); names.append(h)
    pa = np.array([r[0] for r in rows]); pb = np.array([r[1] for r in rows])
    H = np.full((len(names), len(MAP)), -1, np.int8)
    for j0 in range(0, len(MAP), 4096):
        a = I[j0:j0 + 4096][:, pa].astype(np.int16).T; b = I[j0:j0 + 4096][:, pb].astype(np.int16).T
        ok = (a >= 0) & (b >= 0); H[:, j0:j0 + 4096] = np.where(ok, np.rint((a + b) / 2.0), -1).astype(np.int8)
    # orient to P8 coding per SNP exactly as R9 (flip if 2-x agrees better with P8 on overlap hybrids; >= 50 joint calls)
    Lg = np.load(f"{RUN}/cache/lima_G.npy"); lix = {h: i for i, h in enumerate(lm["names"])}
    ov = [(i, lix[h]) for i, h in enumerate(names) if h in lix]; bi = np.array([a for a, _ in ov]); lj = np.array([b for _, b in ov])
    nflip = 0
    for j0 in range(0, len(MAP), 4096):
        Hb = H[bi, j0:j0 + 4096].astype(np.int16); Lb = Lg[lj, j0:j0 + 4096].astype(np.int16); okm = (Hb >= 0) & (Lb >= 0)
        ag = ((Hb == Lb) & okm).sum(0); agf = (((2 - Hb) == Lb) & okm).sum(0); fl = agf > ag
        for jj in np.where(fl)[0]:
            c_ = H[:, j0 + jj]; H[:, j0 + jj] = np.where(c_ >= 0, 2 - c_, -1); nflip += 1
    del Lg
    np.save(f"{RUN}/cache/dense10_G.npy", H); json.dump(dict(names=names), open(f"{RUN}/cache/dense10_meta.json", "w"))
    # consistency with the R9 dense matrix (2014-2023 hybrids)
    r9 = json.load(open(f"{RUN}/cache/dense_meta.json")); G9 = np.load(f"{RUN}/cache/dense_G.npy"); i10 = {h: i for i, h in enumerate(names)}
    common = [h for h in r9["names"] if h in i10]
    same = float(np.mean([np.array_equal(G9[k], H[i10[h]]) for k, h in enumerate(r9["names"]) if h in i10]))
    # coverage of 2024 test records (records with yield); genotyped in the 2k VCF as the reference denominator
    o = obs24[obs24.Yield_Mg_ha.notna()]
    built = set(names); vcf2k = set(samples)
    cov = dict(records_with_yield=int(len(o)), envs=int(o.Env.nunique()), hybrids=int(o.Hybrid.nunique()),
               frac_records_buildable=float(o.Hybrid.isin(built).mean()),
               frac_records_in_2k_vcf=float(o.Hybrid.isin(vcf2k).mean()),
               frac_records_buildable_and_2k=float((o.Hybrid.isin(built) & o.Hybrid.isin(vcf2k)).mean()),
               frac_2k_records_buildable=float(o[o.Hybrid.isin(vcf2k)].Hybrid.isin(built).mean()),
               hybrids_2024_buildable=int(o[o.Hybrid.isin(built)].Hybrid.nunique()), parent_source_counts=src,
               n_built_total=len(names), r9_dense_rows_identical_frac=same, r9_common=len(common), r9_names_not_rebuilt=len(set(r9['names']) - set(names)), n_flipped_snps=nflip,
               vcf_sha256=sha(VCF))
    json.dump(cov, open(f"{RUN}/out10/G_dense_coverage.json", "w"), indent=1); log("G_dense", json.dumps(cov))

if STEP == "G_fit":
    pl, he, seasons, Dk, PCs = d["pl"], d["he"], d["seasons"], d["Dk"], d["PCs"]
    dm = json.load(open(f"{RUN}/cache/dense10_meta.json")); Hd = np.load(f"{RUN}/cache/dense10_G.npy")
    keep = np.array([h in hyb_index for h in dm["names"]]); pop = [h for h in dm["names"] if h in hyb_index]; Hd = Hd[keep]
    popset = set(pop); HL = {h: i for i, h in enumerate(pop)}; DKR = np.array([hyb_index[h] for h in pop])
    he["genotyped"] = he.Hybrid.isin(popset)
    tr_envs = frozenset(he[he.Year <= 2023].Env.unique())
    R_tr = recs(he, tr_envs)
    o = obs24[obs24.Yield_Mg_ha.notna() & obs24.Hybrid.isin(popset)].copy()
    o["y"] = o.Yield_Mg_ha - o.groupby("Env").Yield_Mg_ha.transform("mean"); R_te = o.reset_index(drop=True)
    te_envs = incl(R_te)
    years = list(range(2014, 2024))
    groups = {str(v): np.where(R_tr.Year.values == v)[0] for v in years}; groups = {g: i for g, i in groups.items() if len(i)}
    inner_envs = {g: incl(R_tr.iloc[i]) for g, i in groups.items()}
    h_tr = np.array([HL[h] for h in R_tr.Hybrid]); h_te = np.array([HL[h] for h in R_te.Hybrid])
    cnt = np.bincount(h_tr, minlength=len(HL)).astype(float)
    A2 = snp_block_2k(Dk[DKR], cnt); L98, p98, s98 = kernel_features_98k(Hd, cnt, cnt > 0, p_ref=Dk.shape[1])
    models = {"MG_2k": dict(A=A2, h_tr=h_tr, h_te=h_te, D=False), "MG_98k": dict(A=L98, h_tr=h_tr, h_te=h_te, D=False)}
    new_mask = ~R_te.Hybrid.isin(set(pl[pl.Year <= 2023].Hybrid)).values
    log(f"G_fit: pop {len(pop)}, n_tr {len(R_tr)}, n_te {len(R_te)}, test envs {len(te_envs)}, p98 {p98}")
    res = run_fold(R_tr, R_te, groups, "2023", inner_envs, te_envs, models, GRID, new_mask, log=log)
    rows = []
    for name, x in res["models"].items():
        for prot, j in x["sel"].items():
            g = x["metrics"]; g = g[g.col == j].copy(); g["model"] = name; g["protocol"] = prot; g["fold"] = "EXT_2024"; rows.append(g)
    pd.concat(rows, ignore_index=True).to_csv(f"{RUN}/out10/G2024_selected_per_env.csv", index=False)
    meta = dict(pop=len(pop), n_train=len(R_tr), n_test=len(R_te), n_test_envs=len(te_envs), test_envs=te_envs, p98=p98, scale98=s98,
                sel={n: x["sel"] for n, x in res["models"].items()}, lam_edge={n: {p_: bool(j in (0, len(GRID) - 1)) for p_, j in x["sel"].items()} for n, x in res["models"].items()},
                inner_n_envs={g: len(v) for g, v in inner_envs.items()}, n_test_hybrids=int(R_te.Hybrid.nunique()),
                obs24_sha256=sha(f"{T24}/7_Testing_Observed_Values.csv"))
    json.dump(meta, open(f"{RUN}/out10/G2024_meta.json", "w"), indent=1, default=float); log("G_fit done")


# ============================================================== Part W (ESWYT)
if STEP == "W_fit":
    ph = pd.read_parquet(f"{ESW}/phenotype.parquet"); gl = pd.read_parquet(f"{ESW}/genotype.parquet")
    ph = ph[ph.trait_id == "yield"].copy()
    M = gl.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
    lines = list(M.index); X = M.values.astype(np.float64)            # NaN = missing
    li = {g: i for i, g in enumerate(lines)}
    rec = ph[ph.genotype_id.isin(li)].groupby(["environment_id", "year", "genotype_id"], as_index=False).phenotype_value.mean()
    rec = rec.rename(columns={"environment_id": "Env", "year": "Nursery", "genotype_id": "Hybrid", "phenotype_value": "yield_"})
    def wrecs(nurseries):
        r = rec[rec.Nursery.isin(nurseries)].copy(); r["y"] = r.yield_ - r.groupby("Env").yield_.transform("mean"); return r.reset_index(drop=True)
    # PC1..30 of the markers on all genotyped lines (no phenotypes): mean-imputed, unit variance
    mu = np.nanmean(X, 0); Xi = np.where(np.isnan(X), mu, X); sd = Xi.std(0); okc = sd > 1e-12
    Zp = (Xi[:, okc] - Xi[:, okc].mean(0)) / sd[okc]; U, S, _ = np.linalg.svd(Zp, full_matrices=False); PCw = U[:, :30] * S[:30]
    pve30 = float((S[:30] ** 2).sum() / (S ** 2).sum()); del Zp, U
    def feats(cnt):
        """record-weighted standardisation on the fold's training records; kernel features L (lines x r)."""
        w = cnt / cnt.sum(); obs = ~np.isnan(X); Xo = np.where(obs, X, 0.0)
        m_ = (w @ Xo) / np.maximum(w @ obs, 1e-300); Xf = np.where(obs, X, m_)
        s_ = np.sqrt(np.maximum(w @ (Xf * Xf) - m_ ** 2, 0)); ok = s_ > 1e-12
        Z = (Xf[:, ok] - m_[ok]) / s_[ok]; K = Z @ Z.T; dd, V = np.linalg.eigh(K); pos = dd > dd.max() * 1e-10
        return V[:, pos] * np.sqrt(dd[pos]), int(ok.sum())
    nurs = sorted(rec.Nursery.unique()); tests = [34, 35, 36, 37]
    rows, meta = [], dict(n_lines=len(lines), n_markers=int(X.shape[1]), missing_frac=float(np.isnan(X).mean()), pc_pve30=pve30,
                          n_records=int(len(rec)), n_envs=int(rec.Env.nunique()), nurseries=[int(n) for n in nurs],
                          lines_per_nursery={int(n): int(rec[rec.Nursery == n].Hybrid.nunique()) for n in nurs}, folds={})
    for k in tests:
        R_te = wrecs([k]); te_envs = incl(R_te)
        for prot in ("FORWARD", "INTERNAL_LONO"):
            trn = [n for n in nurs if n < k] if prot == "FORWARD" else [n for n in nurs if n != k]
            R_tr = wrecs(trn)
            groups = {str(n): np.where(R_tr.Nursery.values == n)[0] for n in trn}; groups = {g: i for g, i in groups.items() if len(i)}
            inner_envs = {g: incl(R_tr.iloc[i]) for g, i in groups.items()}
            p1 = str(max(trn))                                            # most recent training nursery
            h_tr = np.array([li[h] for h in R_tr.Hybrid]); h_te = np.array([li[h] for h in R_te.Hybrid])
            cnt = np.bincount(h_tr, minlength=len(lines)).astype(float)
            L, nmk = feats(cnt)
            models = {"MG": dict(A=L, h_tr=h_tr, h_te=h_te, Fd_tr=PCw[h_tr], Fd_te=PCw[h_te], D=True)}
            new_mask = ~R_te.Hybrid.isin(set(R_tr.Hybrid)).values
            t0 = time.time()
            res = run_fold(R_tr, R_te, groups, p1, inner_envs, te_envs, models, GRID, new_mask, log=log)
            x = res["models"]["MG"]
            for pr, j in x["sel"].items():
                g = x["metrics"]; g = g[g.col == j].copy(); g["model"] = "MG"; g["protocol"] = pr; rows.append(g.assign(fold=f"N{k}", scheme=prot))
            for pr, j in x["D"]["sel"].items():
                g = x["D"]["metrics"]; g = g[g.col == j].copy(); g["model"] = "D_noEC"; g["protocol"] = pr; rows.append(g.assign(fold=f"N{k}", scheme=prot))
            meta["folds"][f"N{k}|{prot}"] = dict(train_nurseries=[int(n) for n in trn], p1=p1, n_train=len(R_tr), n_test=len(R_te), n_test_envs=len(te_envs),
                                                 n_markers_used=nmk, sel=x["sel"], D_sel=x["D"]["sel"], D_stage1_lam_idx=x["D"]["stage1_lam_idx"],
                                                 inner_n_envs={g: len(v) for g, v in inner_envs.items()}, seconds=round(time.time() - t0))
            log(f"W N{k} {prot}: n_tr {len(R_tr)} test envs {len(te_envs)} sel {x['sel']} D {x['D']['sel']} {time.time()-t0:.0f}s")
    pd.concat(rows, ignore_index=True).to_csv(f"{RUN}/out10/W_selected_per_env.csv", index=False)
    meta["input_sha256"] = {f: sha(f"{ESW}/{f}") for f in ["phenotype.parquet", "genotype.parquet", "environment.parquet"]}
    json.dump(meta, open(f"{RUN}/out10/W_meta.json", "w"), indent=1, default=float); log("W_fit done")
log("end", STEP)
