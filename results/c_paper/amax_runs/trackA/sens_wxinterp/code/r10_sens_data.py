"""R10 track A, POST-HOC sensitivity (declared after the primary summary exposed a defect in the released 2024 weather):
4_Testing_Weather_Data_2024_full_year.csv has T2M_MAX, T2M_MIN, T2M, RH2M and PRECTOTCORR missing on 2024-07-12..14 in
all 23 envs (and GWETROOT missing from 2024-07-02 on). In the primary run the missing days propagate through the
cumulative GDD and prefix sums, so every 2024 weather window after 2024-07-11 is undefined. This script fills only the
3-day interior gap of those five variables by linear interpolation in time within each env (pandas interpolate,
limit 3, inside only); GWETROOT is left missing (no substitute variable). It then rebuilds the 2024 seasons and daily
CERIS arrays with the same round-1 functions and writes a copy of the primary R10 cache with only those replaced.
Usage: python r10_sens_data.py <r9_code_dir> <primary_r10_cache.pkl> <prereg_v10.json> <test_dir> <out_dir>"""
import os, sys, json, pickle, time
CODE, CACHE, PRE10, TEST, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
for f in ("a_lib.py", "a_models.py", "v5_lib.py"):
    exec(open(os.path.join(CODE, f)).read())
assert sha256_file(PRE10) == "dfc8e990551241c72d04e3d74f2526dca52209d677884bbcd88edfb6644e015b"
PRIMARY_CACHE_SHA = "1ef929594e800de32313ef99365a710d1961c52387a41ccf60132895b25bbe46"
assert sha256_file(CACHE) == PRIMARY_CACHE_SHA
F_WX = "4_Testing_Weather_Data_2024_full_year.csv"
FILL = ["T2M_MAX", "T2M_MIN", "T2M", "RH2M", "PRECTOTCORR"]
C = pickle.load(open(CACHE, "rb"))
wx = pd.read_csv(os.path.join(TEST, F_WX)); wx["Date_p"] = pd.to_datetime(wx.Date.astype(str), format="%Y%m%d")
wx = wx.sort_values(["Env", "Date_p"]).reset_index(drop=True)
n_before = {v: int(wx[v].isna().sum()) for v in FILL + ["GWETROOT"]}
for v in FILL:
    wx[v] = wx.groupby("Env")[v].transform(lambda s: s.interpolate(method="linear", limit=3, limit_area="inside"))
n_after = {v: int(wx[v].isna().sum()) for v in FILL + ["GWETROOT"]}
he = C["he"]; e24 = sorted(he[he.Year == 2024].Env.unique())
plant24 = C["plant_med"].loc[[e for e in C["plant_med"].index if str(e).endswith("_2024")]]
seasons24 = build_env_season(wx, plant24)
daily24 = {}
for env, g in wx.groupby("Env"):
    if env not in seasons24: continue
    g = g[g.Date_p >= plant24.loc[env]]
    daily24[env] = np.column_stack([g.T2M_MAX.values, g.T2M_MIN.values, g.T2M.values, g.PRECTOTCORR.values,
                                    g.ALLSKY_SFC_SW_DWN.values, g.GWETROOT.values, tetens_vpd(g.T2M.values, g.RH2M.values)])
assert set(seasons24) == {e for e in C["seasons"] if e.endswith("_2024")}
C["seasons"].update(seasons24); C["daily"].update(daily24)
C["sensitivity"] = dict(name="wx_gap_interpolated", filled_variables=FILL, rule="linear interpolation in time within env, limit 3 days, interior only; GWETROOT left missing")
pickle.dump(C, open(os.path.join(OUT, "r10_data_cache_wxinterp.pkl"), "wb"), protocol=4)
rep = dict(primary_cache_sha256=PRIMARY_CACHE_SHA, rule=C["sensitivity"]["rule"], nan_before=n_before, nan_after=n_after,
           season_cum_gdd_nan_2024_after={e: int(np.isnan(seasons24[e]["cum"]).sum()) for e in e24 if e in seasons24},
           cache_sha256=sha256_file(os.path.join(OUT, "r10_data_cache_wxinterp.pkl")))
json.dump(rep, open(os.path.join(OUT, "R10_A_sens_wxinterp_data.json"), "w"), indent=1)
print(json.dumps(dict(nan_before=n_before, nan_after=n_after)))
