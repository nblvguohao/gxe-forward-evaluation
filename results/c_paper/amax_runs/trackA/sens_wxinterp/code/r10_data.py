"""R10 track A, Part G step 1 (part G code path; 2024 data are read only here and in r10_run.py).
Extends the frozen R9 track A data cache (2014-2023 training plots of the 2025 release, sha256 e80c0a73...) with the
unsealed 2024 test year: env x hybrid mean yields (7_Testing_Observed_Values.csv), 2024 daily weather
(4_Testing_Weather_Data_2024_full_year.csv) and planting dates (2_Testing_Meta_Data_2024.csv). Season arrays, daily
CERIS arrays and hybrid-env records for 2024 are built with the round-1 functions used for the training years.
Writes <out>/r10_data_cache.pkl and <out>/R10_A_data_2024.json.
Usage: python r10_data.py <r9_code_dir> <r9_cache.pkl> <prereg_v10.json> <test_dir> <out_dir>"""
import os, sys, json, pickle, time
CODE, CACHE9, PRE10, TEST, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
for f in ("a_lib.py", "a_models.py", "v5_lib.py"):
    exec(open(os.path.join(CODE, f)).read())
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
assert sha256_file(PRE10) == "dfc8e990551241c72d04e3d74f2526dca52209d677884bbcd88edfb6644e015b", "prereg v10 sha256 mismatch"
CACHE9_SHA = "e80c0a7335dbea8578419bd1c52b0566b28df7400ed0a5902b8ab8a56869f4ee"   # R9_A_results.json data_cache_sha256
assert sha256_file(CACHE9) == CACHE9_SHA, "R9 cache changed"
F_OBS, F_WX, F_META = "7_Testing_Observed_Values.csv", "4_Testing_Weather_Data_2024_full_year.csv", "2_Testing_Meta_Data_2024.csv"
# ---- integrity against the release checksums shipped in the sealed folder
sums = {}
for line in open(os.path.join(TEST, "SHA256SUMS.txt")):
    parts = line.split()
    if len(parts) >= 2: sums[os.path.basename(parts[-1])] = parts[0]
integ = {}
for f in (F_OBS, F_WX, F_META):
    s = sha256_file(os.path.join(TEST, f)); integ[f] = dict(sha256=s, in_SHA256SUMS=f in sums, matches_SHA256SUMS=(s == sums.get(f)))
    if f in sums: assert s == sums[f], f
log("integrity", json.dumps({k: v["matches_SHA256SUMS"] for k, v in integ.items()}))
C = pickle.load(open(CACHE9, "rb"))
pl, he, seasons, daily, hyb_index = C["pl"], C["he"], C["seasons"], C["daily"], C["hyb_index"]
assert he.Year.max() == 2023 and pl.Year.max() == 2023


def read_wx(path):                  # verbatim from v9_data.py
    wx = pd.read_csv(path)
    d = wx.Date.astype(str)
    wx["Date_p"] = pd.to_datetime(d, format="%Y%m%d") if d.str.fullmatch(r"\d{8}").all() else pd.to_datetime(d, format="mixed")
    return wx


ov = pd.read_csv(os.path.join(TEST, F_OBS)); wx = read_wx(os.path.join(TEST, F_WX)); meta = pd.read_csv(os.path.join(TEST, F_META))
assert list(ov.columns) == ["Env", "Hybrid", "Yield_Mg_ha"] and not ov.duplicated(["Env", "Hybrid"]).any()
plant = meta[["Env", "Date_Planted"]].copy(); plant["d"] = pd.to_datetime(plant.Date_Planted, format="mixed")
plant_med24 = plant.groupby("Env").d.median().dt.normalize()
seasons24 = build_env_season(wx, plant_med24)
daily24 = {}
for env, g in wx.sort_values(["Env", "Date_p"]).groupby("Env"):          # as prep() in v9_data.py
    if env not in seasons24: continue
    g = g[g.Date_p >= plant_med24.loc[env]]
    daily24[env] = np.column_stack([g.T2M_MAX.values, g.T2M_MIN.values, g.T2M.values, g.PRECTOTCORR.values,
                                    g.ALLSKY_SFC_SW_DWN.values, g.GWETROOT.values, tetens_vpd(g.T2M.values, g.RH2M.values)])
he24 = pd.DataFrame(dict(Env=ov.Env.values, Year=2024, Hybrid=ov.Hybrid.values, yield_=ov.Yield_Mg_ha.values.astype(float),
                         silk_gdd=np.nan, silk_dap=np.nan))
he24["genotyped"] = he24.Hybrid.isin(hyb_index); he24["has_season"] = he24.Env.isin(seasons24)
he24 = he24[he.columns]
ok = he24[he24.yield_.notna() & he24.genotyped & he24.has_season]; cnt = ok.groupby("Env").size()
train_h = set(he[he.yield_.notna()].Hybrid)
sl24 = {e: int(s["n"]) for e, s in seasons24.items()}; sl_tr = [int(s["n"]) for s in seasons.values()]
max_d2 = max(d2 for _, d2 in CERIS_WINDOWS)
rep = dict(integrity=integ, files=dict(observed=F_OBS, weather=F_WX, meta=F_META),
           observed=dict(n_rows=int(len(ov)), n_env=int(ov.Env.nunique()), envs=sorted(ov.Env.unique()), n_yield_nan=int(ov.Yield_Mg_ha.isna().sum()),
                         n_rows_genotyped=int(he24.genotyped.sum()), n_hybrids=int(ov.Hybrid.nunique()),
                         n_hybrids_not_in_training_records=int(len(set(ov.Hybrid) - train_h))),
           envs_meta=sorted(meta.Env.unique()), envs_weather=sorted(wx.Env.unique()),
           envs_meta_without_observed=sorted(set(meta.Env) - set(ov.Env)), envs_observed_without_season=sorted(set(ov.Env) - set(seasons24)),
           planting_date=plant_med24.dt.strftime("%Y-%m-%d").to_dict(), weather_last_date=wx.groupby("Env").Date_p.max().dt.strftime("%Y-%m-%d").to_dict(),
           season_days_2024=sl24, season_days_training=dict(min=int(min(sl_tr)), median=float(np.median(sl_tr)), max=int(max(sl_tr))),
           ceris_max_window_end_day=int(max_d2), n_env_2024_season_shorter_than_ceris_max=int(sum(v < max_d2 + 1 for v in sl24.values())),
           records_per_env=cnt.to_dict(), n_env_ge30=int((cnt >= 30).sum()), envs_ge30=sorted(cnt[cnt >= 30].index), n_test_records_ge30=int(cnt[cnt >= 30].sum()))
he_all = pd.concat([he, he24], ignore_index=True)
seasons_all = dict(seasons); seasons_all.update(seasons24); daily_all = dict(daily); daily_all.update(daily24)
plant_all = pd.concat([C["plant_med"], plant_med24])
out = dict(C); out.update(he=he_all, seasons=seasons_all, daily=daily_all, plant_med=plant_all, test2024_files=integ, r9_cache_sha256=CACHE9_SHA)
pickle.dump(out, open(os.path.join(OUT, "r10_data_cache.pkl"), "wb"), protocol=4)
rep["cache_sha256"] = sha256_file(os.path.join(OUT, "r10_data_cache.pkl"))
json.dump(rep, open(os.path.join(OUT, "R10_A_data_2024.json"), "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
log("2024 records", len(he24), "envs >=30", rep["n_env_ge30"], "cache written")
