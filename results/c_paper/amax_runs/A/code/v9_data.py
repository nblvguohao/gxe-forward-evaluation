"""R9 track A, step 0: rebuild the G2F 2014-2023 pipeline from the 2025 competition release (round-1 code: plot table,
predicted-phenology inputs, 21 EC_env seasons, 2k SNP, PC1..30) and, BEFORE any model fit, report agreement of the
2014-2021 subset with the local 2014-2021 files used in R1-R8 (row counts, env sets, yield values, weather, derived
quantities). Writes <out>/v9_data_cache.pkl and <out>/R9_A_data_consistency.json.
Usage: python v9_data.py <code_dir> <new_training_dir> <old_g2f_raw_dir> <prereg_v9.json> <out_dir>"""
import os, sys, json, pickle, hashlib, time
CODE, NEW, OLD, PRE9, OUT = sys.argv[1], sys.argv[2].rstrip("/") + "/", sys.argv[3].rstrip("/") + "/", sys.argv[4], sys.argv[5]
os.makedirs(OUT, exist_ok=True)
for f in ("a_lib.py", "a_models.py", "v5_lib.py"):
    exec(open(os.path.join(CODE, f)).read())
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
assert sha256_file(PRE9) == "6ed45e2bd64c88226fe72360e0b8405233cc84b03b1b604330eaa000dd99aa09"
F_TR, F_WX, F_VCF = "1_Training_Trait_Data_2014_2023.csv", "4_Training_Weather_Data_2014_2023_full_year.csv", "5_Genotype_Data_All_2014_2025_Hybrids.vcf"
O_TR, O_WX, O_VCF = "1_Training_Trait_Data_2014_2021.csv", "4_Training_Weather_Data_2014_2021.csv", "5_Genotype_Data_All_2014_2025_Hybrids.vcf"
# ---- file integrity: new files against the release SHA256SUMS.txt; old files against the round-1 prereg values
sums = {}
for line in open(os.path.join(os.path.dirname(NEW.rstrip("/")), "SHA256SUMS.txt")):
    h, p = line.split(); sums[os.path.basename(p)] = h
integ = {}
for f in (F_TR, F_WX, F_VCF):
    s = sha256_file(NEW + f); integ["new/" + f] = dict(sha256=s, matches_SHA256SUMS=(s == sums.get(f)))
    assert s == sums[f], f
OLD_SHA = {O_TR: "502582195367608dda756d63aaf48d70e371a6468c3c9fa883d0367f9a383e72", O_WX: "ed53f534cf813fd821e82e1b791256ef41d350624a16155cc0c3c67154e404f2",
           O_VCF: "28f2e8c6c933539af4dfdd966df77f773dfdc925e6f3235ecb20d52b92acce6c"}
for f, h in OLD_SHA.items():
    s = sha256_file(OLD + f); integ["old/" + f] = dict(sha256=s, matches_round1_prereg=(s == h)); assert s == h, f
log("integrity ok")


def read_wx(path):
    wx = pd.read_csv(path)
    d = wx.Date.astype(str)
    wx["Date_p"] = pd.to_datetime(d, format="%Y%m%d") if d.str.fullmatch(r"\d{8}").all() else pd.to_datetime(d, format="mixed")
    return wx


def prep(tr, wx, Dk, hyb_index):
    """round-1 data preparation (verbatim from v7_g2f.py) on the given trait and weather tables."""
    tr = tr.copy(); tr["Date_Planted_p"] = pd.to_datetime(tr["Date_Planted"], format="mixed")
    pl = tr[["Env", "Year", "Hybrid", "Date_Planted_p", "Silk_DAP_days", "Pollen_DAP_days", "Yield_Mg_ha"]].copy()
    plant_med = pl.groupby("Env").Date_Planted_p.median().dt.normalize()
    seasons = build_env_season(wx, plant_med)
    pl["silk_gdd"] = np.nan
    for env, ii in pl.groupby("Env").indices.items():
        if env in seasons:
            pl.iloc[ii, pl.columns.get_loc("silk_gdd")] = silk_gdd_from_dap(seasons[env], pl.Silk_DAP_days.values[ii])
    ph = (pl[pl.silk_gdd.notna()].groupby(["Env", "Hybrid"], as_index=False).agg(silk_gdd=("silk_gdd", "mean"), silk_dap=("Silk_DAP_days", "mean")))
    yl = (pl[pl.Yield_Mg_ha.notna()].groupby(["Env", "Year", "Hybrid"], as_index=False).agg(yield_=("Yield_Mg_ha", "mean")))
    he = yl.merge(ph, on=["Env", "Hybrid"], how="outer")
    he["Year"] = he.Year.fillna(he.Env.str[-4:].astype(int)).astype(int)
    he["genotyped"] = he.Hybrid.isin(hyb_index); he["has_season"] = he.Env.isin(seasons)
    daily = {}
    for env, g in wx.sort_values(["Env", "Date_p"]).groupby("Env"):
        if env not in seasons: continue
        g = g[g.Date_p >= plant_med.loc[env]]
        daily[env] = np.column_stack([g.T2M_MAX.values, g.T2M_MIN.values, g.T2M.values, g.PRECTOTCORR.values,
                                      g.ALLSKY_SFC_SW_DWN.values, g.GWETROOT.values, tetens_vpd(g.T2M.values, g.RH2M.values)])
    return dict(pl=pl, plant_med=plant_med, seasons=seasons, he=he, daily=daily)


# ---- genotypes (identical file in both releases): 2k SNP (MAF >= 0.05), mean-imputed, PC1..30 over all genotyped hybrids
samples, snp_ids, D = parse_vcf_dosage(NEW + F_VCF)
af = np.nanmean(D, axis=0) / 2; keep = np.minimum(af, 1 - af) >= 0.05
Dk = D[:, keep].astype(np.float64); mu = np.nanmean(Dk, axis=0)
ind = np.where(np.isnan(Dk)); Dk[ind] = np.take(mu, ind[1]); del D
Gz = (Dk - Dk.mean(0)) / Dk.std(axis=0)
U, S, Vt = np.linalg.svd(Gz, full_matrices=False); PCs = U[:, :30] * S[:30]; del U, Gz, Vt
hyb_index = {h: i for i, h in enumerate(samples)}
log("genotypes", Dk.shape)
# ---- read tables
tn = pd.read_csv(NEW + F_TR, low_memory=False); to = pd.read_csv(OLD + O_TR, low_memory=False)
wn = read_wx(NEW + F_WX); wo = read_wx(OLD + O_WX)
rep = dict(integrity=integ, new_trait_columns=list(tn.columns), old_trait_columns=list(to.columns), new_weather_columns=list(wn.columns))
# ---- 2014-2021 agreement: trait
t14 = tn[tn.Year <= 2021].reset_index(drop=True)
common = [c for c in to.columns if c in t14.columns]
keys = [c for c in ["Env", "Hybrid", "Replicate", "Block", "Plot", "Range", "Pass", "Experiment", "Field_Location"] if c in common]
same_order = bool(len(t14) == len(to) and all(t14[c].astype(str).equals(to[c].astype(str)) for c in keys))
A = t14.sort_values(keys, kind="mergesort").reset_index(drop=True); B = to.sort_values(keys, kind="mergesort").reset_index(drop=True)
def eq(a, b):
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        a = a.astype(float).values; b = b.astype(float).values
        return int(((np.isnan(a) & np.isnan(b)) | (a == b)).sum()), float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0
    return int((a.astype(str).values == b.astype(str).values).sum()), None
col_eq = {}
if len(A) == len(B):
    for c in common:
        n_eq, mx = eq(A[c], B[c]); col_eq[c] = dict(n_equal=n_eq, n_rows=len(A), max_abs_diff=mx)
rep["trait_2014_2021"] = dict(n_rows_new=int(len(t14)), n_rows_old=int(len(to)), n_rows_new_total=int(len(tn)),
    rows_by_year_new={int(k): int(v) for k, v in tn.Year.value_counts().sort_index().items()},
    env_set_equal=bool(set(t14.Env) == set(to.Env)), n_env_new=int(t14.Env.nunique()), n_env_old=int(to.Env.nunique()),
    envs_only_new=sorted(set(t14.Env) - set(to.Env)), envs_only_old=sorted(set(to.Env) - set(t14.Env)),
    same_row_order=same_order, sort_keys=keys, common_columns=common, columns_only_new=[c for c in tn.columns if c not in to.columns],
    columns_only_old=[c for c in to.columns if c not in tn.columns], column_equality_after_sort=col_eq,
    yield_identical=bool(col_eq.get("Yield_Mg_ha", {}).get("n_equal") == len(to)))
# ---- 2014-2021 agreement: weather (columns used by the pipeline)
WC = ["T2M_MAX", "T2M_MIN", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT"]
m = wo[["Env", "Date"] + WC].merge(wn[["Env", "Date"] + WC], on=["Env", "Date"], how="outer", suffixes=("_old", "_new"), indicator=True)
m14 = m[m.Env.isin(set(wo.Env))]
wrep = dict(n_rows_old=int(len(wo)), n_rows_new_for_old_envs=int(wn.Env.isin(set(wo.Env)).sum()),
            n_old_rows_missing_in_new=int((m14._merge == "left_only").sum()), n_new_rows_not_in_old_for_old_envs=int((m14._merge == "right_only").sum()),
            env_set_old=int(wo.Env.nunique()), old_envs_missing_in_new=sorted(set(wo.Env) - set(wn.Env)),
            n_env_new_total=int(wn.Env.nunique()), n_env_new_by_year={int(k): int(v) for k, v in wn.Env.str[-4:].astype(int).value_counts().sort_index().items()})
bm = m14[m14._merge == "both"]
wrep["max_abs_diff_by_variable"] = {c: float(np.nanmax(np.abs(bm[c + "_old"].values - bm[c + "_new"].values))) for c in WC}
wrep["n_values_differing_by_variable"] = {c: int((~((bm[c + "_old"] == bm[c + "_new"]) | (bm[c + "_old"].isna() & bm[c + "_new"].isna()))).sum()) for c in WC}
rep["weather_2014_2021"] = wrep
log("tables compared")
# ---- pipelines: new release (2014-2023) and old local files (2014-2021) through identical code
Pn = prep(tn, wn, Dk, hyb_index); Po = prep(to, wo, Dk, hyb_index)
he_n, he_o = Pn["he"], Po["he"]
mm = he_o.merge(he_n, on=["Env", "Hybrid"], how="left", suffixes=("_o", "_n"), indicator=True)
env_o = sorted(Po["seasons"]); cum_diff = max(float(np.max(np.abs(Po["seasons"][e]["cum"] - Pn["seasons"][e]["cum"]))) if Po["seasons"][e]["n"] == Pn["seasons"][e]["n"] else np.inf for e in env_o)
Pdiff = max(float(np.nanmax(np.abs(Po["seasons"][e]["P"] - Pn["seasons"][e]["P"]))) if Po["seasons"][e]["n"] == Pn["seasons"][e]["n"] else np.inf for e in env_o)
rep["derived_2014_2021"] = dict(
    n_env_with_season_old=len(Po["seasons"]), n_env_with_season_new_2014_2021=int(sum(int(e[-4:]) <= 2021 for e in Pn["seasons"])),
    season_env_sets_equal=bool(set(Po["seasons"]) == {e for e in Pn["seasons"] if int(e[-4:]) <= 2021}),
    plant_median_equal=bool(Po["plant_med"].equals(Pn["plant_med"].loc[Po["plant_med"].index])),
    max_abs_diff_cum_gdd=cum_diff, max_abs_diff_weather_prefix_sums=Pdiff,
    n_hybrid_env_old=int(len(he_o)), n_hybrid_env_matched=int((mm._merge == "both").sum()),
    max_abs_diff_hybrid_mean_yield=float(np.nanmax(np.abs(mm.yield__o - mm.yield__n))),
    n_yield_mismatch=int((~((mm.yield__o == mm.yield__n) | (mm.yield__o.isna() & mm.yield__n.isna()))).sum()),
    max_abs_diff_silk_gdd=float(np.nanmax(np.abs(mm.silk_gdd_o - mm.silk_gdd_n))),
    n_silk_gdd_mismatch=int((~((mm.silk_gdd_o == mm.silk_gdd_n) | (mm.silk_gdd_o.isna() & mm.silk_gdd_n.isna()))).sum()))
# ---- counts per year (new release)
hs = he_n[he_n.yield_.notna() & he_n.genotyped & he_n.has_season]
cnt = hs.groupby("Env").size()
rep["counts_by_year_new"] = {int(y): dict(n_env_trait=int(Pn["pl"][Pn["pl"].Year == y].Env.nunique()),
                                         n_env_with_weather=int(sum(int(e[-4:]) == y for e in Pn["seasons"])),
                                         n_env_ge30_genotyped_with_yield=int(sum(int(e[-4:]) == y for e in cnt[cnt >= 30].index)),
                                         n_hybrids_with_yield=int(he_n[(he_n.Year == y) & he_n.yield_.notna()].Hybrid.nunique()),
                                         n_hybrids_genotyped_with_yield=int(hs[hs.Year == y].Hybrid.nunique()))
                             for y in sorted(he_n.Year.unique())}
rep["genotype"] = dict(vcf_identical_to_local=bool(sha256_file(NEW + F_VCF) == OLD_SHA[O_VCF]), n_hybrids=len(samples), n_snp_maf05=int(Dk.shape[1]))
rep["verdict"] = dict(
    trait_rows_equal=bool(rep["trait_2014_2021"]["n_rows_new"] == rep["trait_2014_2021"]["n_rows_old"]),
    env_set_equal=rep["trait_2014_2021"]["env_set_equal"], yield_identical=rep["trait_2014_2021"]["yield_identical"],
    weather_identical=bool(wrep["n_old_rows_missing_in_new"] == 0 and all(v == 0 for v in wrep["n_values_differing_by_variable"].values())),
    derived_identical=bool(rep["derived_2014_2021"]["n_yield_mismatch"] == 0 and rep["derived_2014_2021"]["n_silk_gdd_mismatch"] == 0 and cum_diff == 0.0))
json.dump(rep, open(os.path.join(OUT, "R9_A_data_consistency.json"), "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
pl = Pn["pl"]
pickle.dump(dict(pl=pl, plant_med=Pn["plant_med"], seasons=Pn["seasons"], he=he_n, daily=Pn["daily"], Dk=Dk, PCs=PCs, samples=samples,
                 hyb_index=hyb_index, new_files={k: v for k, v in integ.items() if k.startswith("new/")}),
            open(os.path.join(OUT, "v9_data_cache.pkl"), "wb"), protocol=4)
log("cache written", json.dumps(rep["verdict"]))
