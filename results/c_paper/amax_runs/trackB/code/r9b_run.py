"""prereg v9 track B CPU driver on amax (run from <server>/egp_gxe/runs/trackB).
Usage: python code/r9b_run.py <step> [shard i/n] [threads]
steps: prep (data from the 2025 release, agreement check, Lima 98k int8, cached) | S4 | EXT | S4assemble | EXTassemble
S4 = v8 S4_G2F_CV00 on the Lima common population (4,372 hybrids): MG_2k, MG_98k, D_noEC(MG_2k), D_noEC(MG_98k).
EXT = train 2014-2021 / test 2022 and train 2014-2022 / test 2023 on hybrids with 2k + dense (437k-built) genotypes.
"""
import os, sys, json, pickle, time, gc, zipfile
STEP = sys.argv[1]; SHARD = tuple(int(x) for x in sys.argv[2].split("/")) if len(sys.argv) > 2 else (0, 1)
TH = sys.argv[3] if len(sys.argv) > 3 else "24"
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[k] = TH
BASE = os.environ.get("R9B_BASE", "<server>/egp_gxe"); RUN = f"{BASE}/runs/trackB"; CODE = f"{BASE}/egp_bundle/code"
T25 = f"{BASE}/data/g2f_2025/Training_data"; LOCAL = f"{BASE}/egp_bundle/data/g2f_raw"
P8ZIP = f"{BASE}/egp_bundle/data/p8_figshare/P8_figshare_22776806_curated_data.zip"
for dd in ("cache", "ckpt/S4", "ckpt/EXT", "ckpt/EXT2k", "out", "logs", "tmp"): os.makedirs(f"{RUN}/{dd}", exist_ok=True)
import hashlib
def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
assert _sha(f"{RUN}/code/prereg_v9.json") == "6ed45e2bd64c88226fe72360e0b8405233cc84b03b1b604330eaa000dd99aa09"
CODE_SHA = {"a_lib.py": "52ea0f3fb94e1492170cdc54ccdd39bbf9d96b2657de522b5322917c75144ab4",
            "a_models.py": "939864a6464ff1d2185f4d8900b78a5ae1f45d810a683c3329258047a067793f",
            "r6_lib.py": "7dc20471fc4aa6c242c9c46e81d5b2459e8137808cc5be55e200b798e44afb96"}
for f, h in CODE_SHA.items():
    assert _sha(f"{CODE}/{f}") == h, f
    exec(open(f"{CODE}/{f}").read())
exec(open(f"{RUN}/code/r9b_lib.py").read())
LOGF = open(f"{RUN}/logs/run_{STEP}_{SHARD[0]}of{SHARD[1]}.log", "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
prereg = json.load(open(f"{RUN}/code/prereg_v9.json"))
GRID = V8_GRID
MODELS_MG = ["MG_2k", "MG_98k"]

# ------------------------------------------------------------------ prep / cache
DPK = f"{RUN}/cache/data.pkl"
if STEP == "prep":
    d = build_data(T25, log); pickle.dump(d, open(DPK + ".tmp", "wb")); os.replace(DPK + ".tmp", DPK)
    agr = agreement_2014_2021(d, LOCAL); json.dump(agr, open(f"{RUN}/out/agreement_2014_2021.json", "w"), indent=1)
    log("agreement", json.dumps(agr))
    lg = f"{RUN}/cache/lima_G.npy"
    if not os.path.exists(lg):
        names, blocks, nfrac = [], [], 0
        with zipfile.ZipFile(P8ZIP) as z:
            with z.open("data/GENO.csv") as fh:
                hdr = pd.read_csv(z.open("data/GENO.csv"), nrows=0).columns.tolist(); snps = hdr[1:]
                for ch in pd.read_csv(fh, chunksize=200, index_col=0, dtype={c_: np.float32 for c_ in snps}):
                    V = ch.values; ok = np.isfinite(V) & (np.abs(V - np.round(V)) < 1e-6) & (V >= 0) & (V <= 2)
                    blocks.append(np.where(ok, np.round(V), -1).astype(np.int8)); nfrac += int((~ok).sum())
                    names += ch.index.astype(str).tolist()
        G = np.vstack(blocks); np.save(lg, G)
        json.dump(dict(names=names, snps=snps, n_noninteger=nfrac, shape=list(G.shape)), open(f"{RUN}/cache/lima_meta.json", "w"))
        with zipfile.ZipFile(P8ZIP) as z:
            with z.open("data/MAP.csv") as fh, open(f"{RUN}/cache/MAP.csv", "wb") as o: o.write(fh.read())
    json.dump(dict(lima_G_npy=_sha(lg), P8_zip=_sha(P8ZIP), trait_2025=_sha(f"{T25}/1_Training_Trait_Data_2014_2023.csv"),
                   weather_2025=_sha(f"{T25}/4_Training_Weather_Data_2014_2023_full_year.csv"),
                   vcf_2k=_sha(f"{T25}/5_Genotype_Data_All_2014_2025_Hybrids.vcf"), prereg_v9=_sha(f"{RUN}/code/prereg_v9.json"),
                   r9b_lib=_sha(f"{RUN}/code/r9b_lib.py"), r9b_run=_sha(f"{RUN}/code/r9b_run.py"), **{k: v for k, v in CODE_SHA.items()}),
              open(f"{RUN}/out/inputs_sha256.json", "w"), indent=1)
    log("prep done"); sys.exit(0)

d = pickle.load(open(DPK, "rb"))
pl, he, seasons, Dk, PCs, hyb_index, samples = d["pl"], d["he"], d["seasons"], d["Dk"], d["PCs"], d["hyb_index"], d["samples"]
P2K = Dk.shape[1]


def setting_run(tag, folds, pop_names, D98, PCs98, use98=True):
    he["genotyped"] = he.Hybrid.isin(set(pop_names)); he_s = he[he.has_season]
    hyb_idx_g = {h: hyb_index[h] for h in pop_names}
    HL = {h: i for i, h in enumerate(pop_names)}; DKR = np.array([hyb_index[h] for h in pop_names])
    PCs2 = PCs[DKR]; Dk_l = Dk[DKR]
    for fi, f in enumerate(folds):
        if fi % SHARD[1] != SHARD[0]: continue
        ck = f"{RUN}/ckpt/{tag}/{f['key']}.pkl"
        if os.path.exists(ck): continue
        t0 = time.time()
        R_tr = recs(he, f["train"]); R_te = recs(he, f["test"]); te_envs = incl(R_te)
        if not te_envs:
            pickle.dump(dict(key=f["key"], skipped=True), open(ck, "wb")); log(f"[{tag}] {f['key']} skipped (no includable env)"); continue
        pm, mat = maturity(d, hyb_idx_g, f, he_s); pm_l = pm[DKR]
        groups = {g: np.where(R_tr.Env.isin(V).values)[0] for g, V in f["inner"].items()}
        groups = {g: v for g, v in groups.items() if len(v)}
        inner_envs = {g: incl(R_tr.iloc[i]) for g, i in groups.items()}
        h_tr = np.array([HL[h] for h in R_tr.Hybrid]); h_te = np.array([HL[h] for h in R_te.Hybrid])
        cnt = np.bincount(h_tr, minlength=len(HL)).astype(float)
        A2 = snp_block_2k(Dk_l, cnt); Fd2 = np.column_stack([PCs2, pm_l])
        models = {"MG_2k": dict(A=A2, h_tr=h_tr, h_te=h_te, Fd_tr=Fd2[h_tr], Fd_te=Fd2[h_te], D=True)}
        p98 = s98 = L98 = None
        if use98:
            L98, p98, s98 = kernel_features_98k(D98, cnt, cnt > 0, p_ref=P2K); Fd98 = np.column_stack([PCs98, pm_l])
            models["MG_98k"] = dict(A=L98, h_tr=h_tr, h_te=h_te, Fd_tr=Fd98[h_tr], Fd_te=Fd98[h_te], D=True)
        new_mask = ~R_te.Hybrid.isin(set(pl[pl.Env.isin(f["train"])].Hybrid)).values
        log(f"[{tag}] {f['key']}: n_tr {len(R_tr)} n_te {len(R_te)} test envs {len(te_envs)} groups {len(groups)} p98 {p98}")
        res = run_fold(R_tr, R_te, groups, f["p1"], inner_envs, te_envs, models, GRID, new_mask, log=log)
        res.update(key=f["key"], p1=f["p1"], maturity=mat, test_envs=te_envs, p98=p98, scale98=s98,
                   n_test_hybrids=int(R_te.Hybrid.nunique()), inner_n_envs={g: len(v) for g, v in inner_envs.items()})
        pickle.dump(res, open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
        del models, A2, L98, res; gc.collect()
        log(f"[{tag}] {f['key']} done {time.time()-t0:.0f}s")


def assemble(tag, folds):
    rows, meta = [], {}
    for f in folds:
        r = pickle.load(open(f"{RUN}/ckpt/{tag}/{f['key']}.pkl", "rb"))
        if r.get("skipped"): meta[f["key"]] = dict(skipped=True); continue
        m = dict(maturity=r["maturity"], p1=r["p1"], n_train=r["n_train"], n_test=r["n_test"], n_test_envs=len(r["test_envs"]),
                 n_test_hybrids=r["n_test_hybrids"], p98=r["p98"], scale98=r["scale98"], inner_n_envs=r["inner_n_envs"], sel={})
        for name, x in r["models"].items():
            m["sel"][name] = x["sel"]
            for prot, j in x["sel"].items():
                g = x["metrics"]; g = g[g.col == j].copy(); g["model"] = name; g["protocol"] = prot; g["fold"] = f["key"]; rows.append(g)
            if "D" in x:
                m["sel"][f"D_noEC({name})"] = dict(stage1_lam_idx=x["D"]["stage1_lam_idx"], **x["D"]["sel"])
                for prot, j in x["D"]["sel"].items():
                    g = x["D"]["metrics"]; g = g[g.col == j].copy(); g["model"] = f"D_noEC({name})"; g["protocol"] = prot; g["fold"] = f["key"]; rows.append(g)
        meta[f["key"]] = m
    pd.concat(rows, ignore_index=True).to_csv(f"{RUN}/out/{tag}_selected_per_env.csv", index=False)
    json.dump(meta, open(f"{RUN}/out/{tag}_fold_meta.json", "w"), indent=1, default=float)
    log(f"[{tag}] assembled {len(meta)} folds")


if STEP in ("S4", "S4assemble"):
    lm = json.load(open(f"{RUN}/cache/lima_meta.json")); pop = lm["names"]; assert all(h in hyb_index for h in pop)
    env_cv = sorted(pl[pl.Year <= 2021].Env.unique())
    folds, locf = folds_S4(env_cv)
    r8 = json.load(open(f"{RUN}/code/r8_location_folds.json"))
    json.dump(dict(location_folds=locf, equal_to_R8=(locf == r8)), open(f"{RUN}/out/S4_location_folds.json", "w"), indent=1)
    assert locf == r8, "location folds differ from R8"
    if STEP == "S4":
        D98 = np.load(f"{RUN}/cache/lima_G.npy"); PCs98 = pcs_98k(D98, 30)[0]
        setting_run("S4", folds, pop, D98, PCs98)
    else:
        assemble("S4", folds)
if STEP in ("EXT", "EXTassemble"):
    dm = json.load(open(f"{RUN}/cache/dense_meta.json")); assert dm["validity_pass"]
    pop = [h for h in dm["names"] if h in hyb_index]
    folds = folds_EXT(sorted(pl.Env.unique()))
    if STEP == "EXT":
        D98 = np.load(f"{RUN}/cache/dense_G.npy")
        keep = np.array([h in hyb_index for h in dm["names"]]); D98 = D98[keep]
        PCs98 = pcs_98k(D98, 30)[0]
        setting_run("EXT", folds, pop, D98, PCs98)
    else:
        assemble("EXT", folds)
if STEP in ("EXT2k", "EXT2kassemble"):
    # prereg fallback (dense genotypes not obtainable): EXT on the 2k SNP set only, round-1 population (all VCF hybrids)
    pop = list(samples); folds = folds_EXT(sorted(pl.Env.unique()))
    if STEP == "EXT2k": setting_run("EXT2k", folds, pop, None, None, use98=False)
    else: assemble("EXT2k", folds)
log("end", STEP, SHARD)
