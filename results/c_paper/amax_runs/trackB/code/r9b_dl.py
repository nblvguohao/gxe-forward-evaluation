"""prereg v9 secondary (no gate): GPU two-tower MLP, DL_full and DL_noEC, 5 seeds averaged, on S4 and EXT.
Architecture exactly as prereg v9: genotype tower Linear(p_SNP -> 512) + GELU + Dropout(0.3) + Linear(512 -> 128);
env tower Linear(21 EC_env + predicted phenology -> 64) + GELU; head MLP([128+64] -> 128 -> 1) (Linear-GELU-Linear).
DL_noEC removes the env tower (head input 128). Target = env-centred yield; AdamW lr 1e-3, wd 1e-4, batch 1024, max 60
epochs; early stopping (patience 10, best-epoch weights restored) on validation MSE of the inner validation year
(the training year nearest the test year, same as the P1 fold); the model is trained on the remaining training years.
SNP input = the 98k P8 set standardised on the fold's training records (MAF >= 0.05 on training hybrids, missing -> 0).
Usage (from runs/trackB): python code/r9b_dl.py <S4|EXT> <cpu threads>
"""
import os, sys, json, pickle, time, subprocess, gc
SET = sys.argv[1]; TH = sys.argv[2] if len(sys.argv) > 2 else "4"
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = TH
q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
gpus = sorted([tuple(int(x) for x in l.split(",")) for l in q.strip().splitlines()], key=lambda t: (t[1], t[2]))
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0][0])
BASE = os.environ.get("R9B_BASE", "<server>/egp_gxe"); RUN = f"{BASE}/runs/trackB"; CODE = f"{BASE}/egp_bundle/code"
for f in ["a_lib.py", "a_models.py", "r6_lib.py"]: exec(open(f"{CODE}/{f}").read())
exec(open(f"{RUN}/code/r9b_lib.py").read())
import torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True; torch.set_num_threads(int(TH))
dev = torch.device("cuda")
os.makedirs(f"{RUN}/ckpt/DL_{SET}", exist_ok=True)
LOGF = open(f"{RUN}/logs/dl_{SET}.log", "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
log(f"GPU loads {gpus}; using physical GPU {gpus[0][0]}")
for _ in range(720):
    if os.path.exists(f"{RUN}/cache/data.pkl") and os.path.exists(f"{RUN}/cache/lima_meta.json"): break
    time.sleep(30)
d = pickle.load(open(f"{RUN}/cache/data.pkl", "rb"))
pl, he, seasons, Dk, hyb_index = d["pl"], d["he"], d["seasons"], d["Dk"], d["hyb_index"]
SEEDS = [20260928 + s for s in range(5)]


class TwoTower(nn.Module):
    def __init__(self, p, full):
        super().__init__(); self.full = full
        self.g = nn.Sequential(nn.Linear(p, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 128))
        if full: self.e = nn.Sequential(nn.Linear(22, 64), nn.GELU())
        self.h = nn.Sequential(nn.Linear(128 + (64 if full else 0), 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, xg, xe=None):
        z = self.g(xg)
        if self.full: z = torch.cat([z, self.e(xe)], 1)
        return self.h(z).squeeze(1)


def train_one(Xg, E, h, y, fit, val, ev_h, ev_E, full, seed):
    torch.manual_seed(seed); np.random.seed(seed % (2 ** 32))
    net = TwoTower(Xg.shape[1], full).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ht = torch.as_tensor(h, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev); Et = torch.as_tensor(E, dtype=torch.float32, device=dev)
    fit_t = torch.as_tensor(fit, device=dev); val_t = torch.as_tensor(val, device=dev)
    def pred(idx_h, idx_E):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx_h), 8192):
                xg = Xg[idx_h[i:i + 8192]].float(); xe = idx_E[i:i + 8192] if full else None
                out.append(net(xg, xe))
        net.train(); return torch.cat(out)
    best, best_state, bad, best_ep = np.inf, None, 0, -1
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    for ep in range(60):
        perm = fit_t[torch.randperm(len(fit_t), device=dev, generator=gen)]
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]; xg = Xg[ht[b]].float(); xe = Et[b] if full else None
            loss = ((net(xg, xe) - yt[b]) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        vl = float(((pred(ht[val_t], Et[val_t]) - yt[val_t]) ** 2).mean())
        if vl < best - 1e-7: best, bad, best_ep = vl, 0, ep; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10: break
    net.load_state_dict(best_state)
    p = pred(torch.as_tensor(ev_h, device=dev), torch.as_tensor(ev_E, dtype=torch.float32, device=dev)).cpu().numpy()
    del net, opt; torch.cuda.empty_cache(); return p, best_ep, best


if SET == "S4":
    lm = json.load(open(f"{RUN}/cache/lima_meta.json")); pop = lm["names"]; D98 = np.load(f"{RUN}/cache/lima_G.npy")
    folds, _ = folds_S4(sorted(pl[pl.Year <= 2021].Env.unique()))
elif SET == "EXT2k":     # prereg fallback: 2k SNP input (genotype tower = linear(2k SNP -> 512)), all VCF hybrids
    pop = list(d["samples"]); D98 = Dk; folds = folds_EXT(sorted(pl.Env.unique()))
else:
    dm = json.load(open(f"{RUN}/cache/dense_meta.json")); assert dm["validity_pass"]
    keep = np.array([h in hyb_index for h in dm["names"]]); pop = [h for h in dm["names"] if h in hyb_index]
    D98 = np.load(f"{RUN}/cache/dense_G.npy")[keep]; folds = folds_EXT(sorted(pl.Env.unique()))
he["genotyped"] = he.Hybrid.isin(set(pop)); he_s = he[he.has_season]
hyb_idx_g = {h: hyb_index[h] for h in pop}; HL = {h: i for i, h in enumerate(pop)}; DKR = np.array([hyb_index[h] for h in pop])
Dg = torch.as_tensor(D98, device=dev)                     # int8 (98k) or float64 (2k) on GPU (m x p)
for f in folds:
    ck = f"{RUN}/ckpt/DL_{SET}/{f['key']}.pkl"
    if os.path.exists(ck): continue
    t0 = time.time()
    R_tr = recs(he, f["train"]); R_te = recs(he, f["test"]); te_envs = incl(R_te)
    if not te_envs: pickle.dump(dict(skipped=True), open(ck, "wb")); continue
    pm, mat = maturity(d, hyb_idx_g, f, he_s); pm_l = pm[DKR]
    h_tr = np.array([HL[h] for h in R_tr.Hybrid]); h_te = np.array([HL[h] for h in R_te.Hybrid])
    # EC_env at env-mean predicted maturity (as round-1 fold_base), z-scored on training records
    _, ece_tr = env_level_ec(R_tr, pm_l[h_tr], seasons); _, ece_te = env_level_ec(R_te, pm_l[h_te], seasons)
    E_tr = np.column_stack([ece_tr.loc[R_tr.Env].values, pm_l[h_tr]]); E_te = np.column_stack([ece_te.loc[R_te.Env].values, pm_l[h_te]])
    z = ZStd(E_tr); E_tr, E_te = z(E_tr), z(E_te)
    # SNP standardisation on training records
    cnt = torch.as_tensor(np.bincount(h_tr, minlength=len(HL)).astype(np.float64), device=dev); w = cnt / cnt.sum(); trm = cnt > 0
    X = Dg.double(); obs = X >= 0; Xo = torch.where(obs, X, torch.zeros_like(X))
    nt = obs[trm].sum(0); af = Xo[trm].sum(0) / torch.clamp(nt, min=1) / 2; keepc = (torch.minimum(af, 1 - af) >= 0.05) & (nt > 0)
    wo = w @ obs.double(); mu = (w @ Xo) / torch.clamp(wo, min=1e-300); X = torch.where(obs, X, mu[None, :])
    sd = torch.sqrt(torch.clamp(w @ (X * X) - mu ** 2, min=0)); keepc &= sd > 1e-12
    Xg = ((X[:, keepc] - mu[keepc]) / sd[keepc]).half(); del X, Xo, obs; torch.cuda.empty_cache()
    vset = f["inner"][f["p1"]]; val = np.where(R_tr.Env.isin(vset).values)[0]; fit = np.where(~R_tr.Env.isin(vset).values)[0]
    out = dict(key=f["key"], p1=f["p1"], n_snp=int(keepc.sum()), maturity=mat, test_envs=te_envs, variants={})
    new_mask = ~R_te.Hybrid.isin(set(pl[pl.Env.isin(f["train"])].Hybrid)).values
    for full in (True, False):
        preds, eps = [], []
        for s in SEEDS:
            p, be, bl = train_one(Xg, E_tr, h_tr, R_tr.y.values, fit, val, h_te, E_te, full, s); preds.append(p); eps.append(be)
        P = np.mean(preds, 0)[:, None]
        out["variants"]["DL_full" if full else "DL_noEC"] = dict(best_epochs=eps, metrics=per_env_metrics_all(R_te, P, te_envs, new_mask))
    del Xg; torch.cuda.empty_cache(); gc.collect()
    pickle.dump(out, open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
    log(f"[DL {SET}] {f['key']} done {time.time()-t0:.0f}s; epochs {out['variants']['DL_full']['best_epochs']} / {out['variants']['DL_noEC']['best_epochs']}")
rows = []
for f in folds:
    r = pickle.load(open(f"{RUN}/ckpt/DL_{SET}/{f['key']}.pkl", "rb"))
    if r.get("skipped"): continue
    for v, x in r["variants"].items():
        g = x["metrics"].copy(); g["model"] = v; g["protocol"] = "ES_P1year"; g["fold"] = f["key"]; rows.append(g)
pd.concat(rows, ignore_index=True).to_csv(f"{RUN}/out/DL_{SET}_per_env.csv", index=False)
log("DL end", SET)
