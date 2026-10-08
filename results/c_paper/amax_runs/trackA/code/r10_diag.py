"""R10 track A diagnostic (part G code path; descriptive only, no model fit): missing values in the 2024 weather
files vs the 2014-2023 training weather, NaN propagation into the season prefix sums, CERIS candidates and the 21 EC_env
of the 2024 test envs (at the fold's own maturity model). Usage: python r10_diag.py <r9_code_dir> <run_trackA_dir> <project_dir> <out.json>"""
import os, sys, json, pickle
CODE, RUN, PROJ, OUT = sys.argv[1:5]
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = "8"
for f in ("a_lib.py", "a_models.py", "v5_lib.py", "v8_lib.py"):
    exec(open(os.path.join(CODE, f)).read())
T = os.path.join(PROJ, "data/g2f_2025/Testing_data_SEALED"); TR = os.path.join(PROJ, "data/g2f_2025/Training_data")
V = ["T2M_MAX", "T2M_MIN", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT"]
rep = {}
def miss(df, name):
    df = df.copy(); df["Date_p"] = pd.to_datetime(df.Date.astype(str), format="%Y%m%d")
    q = dict(n_rows=int(len(df)), n_env=int(df.Env.nunique()), nan_by_var={v: int(df[v].isna().sum()) for v in V},
             le_minus900_by_var={v: int((df[v] <= -900).sum()) for v in V},
             envs_with_nan_by_var={v: int(df[df[v].isna()].Env.nunique()) for v in V})
    q["nan_dates_by_var"] = {v: dict(first=str(df[df[v].isna()].Date_p.min().date()) if df[v].isna().any() else None,
                                     last=str(df[df[v].isna()].Date_p.max().date()) if df[v].isna().any() else None) for v in V}
    q["nan_by_env_var"] = {e: {v: int(g[v].isna().sum()) for v in V if g[v].isna().any()} for e, g in df.groupby("Env") if g[V].isna().any().any()}
    q["nan_by_month_var"] = {v: {str(k): int(x) for k, x in df[df[v].isna()].groupby(df.Date_p.dt.month).size().items()} for v in V if df[v].isna().any()}
    rep[name] = q
miss(pd.read_csv(os.path.join(T, "4_Testing_Weather_Data_2024_full_year.csv")), "weather_2024_full_year")
miss(pd.read_csv(os.path.join(T, "4_Testing_Weather_Data_2024_seasons_only.csv")), "weather_2024_seasons_only")
wtr = pd.read_csv(os.path.join(TR, "4_Training_Weather_Data_2014_2023_full_year.csv"))
miss(wtr, "weather_training_2014_2023")
C = pickle.load(open(os.path.join(RUN, "out", "r10_data_cache.pkl"), "rb"))
seasons, daily, he = C["seasons"], C["daily"], C["he"]
e24 = sorted(he[he.Year == 2024].Env.unique()); etr = sorted(e for e in seasons if int(e[-4:]) <= 2023)
col7 = ["T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT", "VPD"]
rep["daily_nan_2024"] = {e: {c: int(np.isnan(daily[e][:, i]).sum()) for i, c in enumerate(col7) if np.isnan(daily[e][:, i]).any()} for e in e24 if e in daily}
rep["daily_first_nan_day_2024"] = {e: {c: int(np.where(np.isnan(daily[e][:, i]))[0][0]) for i, c in enumerate(col7) if np.isnan(daily[e][:, i]).any()} for e in e24 if e in daily}
rep["daily_nan_training_n_env"] = {c: int(sum(np.isnan(daily[e][:, i]).any() for e in etr if e in daily)) for i, c in enumerate(col7)}
rep["season_cum_gdd_nan_2024"] = {e: int(np.isnan(seasons[e]["cum"]).sum()) for e in e24 if e in seasons}
Mn, lab = ceris_candidates(daily, [e for e in etr if e in daily]); M4, _ = ceris_candidates(daily, e24)
fin4 = np.isfinite(M4)
rep["ceris_candidates"] = dict(total=int(M4.shape[1]), finite_all_training=int(np.isfinite(Mn).all(0).sum()), finite_all_2024=int(fin4.all(0).sum()),
                               finite_2024_by_var={v: int(fin4[:, [i for i, l in enumerate(lab) if l[0] == v]].all(0).sum()) for v in CERIS_VARS},
                               n_2024_envs_finite_T2M_80_100=int(fin4[:, lab.index(("T2M", 80, 100))].sum()))
json.dump(rep, open(OUT, "w"), indent=1, default=str); print("ok")
