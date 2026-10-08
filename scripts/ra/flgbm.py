"""RA §2.3 (with the 2026-09-30 addendum): Fernandes et al. 2024 LightGBM G(A)+E, rebuilt from the Fernandes et al. (2024) repository's
own functions (third_party/Maize_GxE_Prediction/src/preprocessing.py, imported unmodified) and settings
(create_datasets.py, run_g_or_gxe_model.py: EC SVD 15, lag-2 yield features, lat/lon bins, weather x latitude terms,
train-mean imputation; G features = additive kinship rows, SVD 100 on non-lag features, LightGBM max_depth=3,
random_state=seed, seeds 1..10, Field_Location categorical).

  python flgbm.py forward <Y>     # train years Y-2, Y-1 (BLUEs) at the target year's locations; predict Y's scored cells
  python flgbm.py cv0             # fidelity: the original CV0 (train 2019-2020, validate 2021, known hybrids)

Environment: RAW, BENCH, RA (ra root with flgbm/blues.csv, flgbm/kinship.parquet), TP (third_party root)."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold

RAW, BENCH, RA, TP = (Path(os.environ[k]) for k in ("RAW", "BENCH", "RA", "TP"))
sys.path.insert(0, str(TP / "Maize_GxE_Prediction" / "src"))
from preprocessing import (agg_yield, create_field_location, feat_eng_soil, feat_eng_target,  # noqa: E402
                           feat_eng_weather, lat_lon_to_bin, process_blues, process_metadata)

META_COLS = ["Env", "weather_station_lat", "weather_station_lon", "treatment_not_standard"]
LAT_BIN_STEP = 1.2
LON_BIN_STEP = LAT_BIN_STEP * 3
SEEDS = list(range(1, 11))
OUT = RA / "flgbm"


def load_tables():
    meta = pd.concat([process_metadata(str(RAW / "2_Training_Meta_Data_2014_2023.csv")),
                      process_metadata(str(RAW / "2_Testing_Meta_Data_2024.csv"))], ignore_index=True)
    trait = pd.read_csv(RAW / "1_Training_Trait_Data_2014_2023.csv", low_memory=False)
    trait = trait.merge(meta[META_COLS], on="Env", how="left")
    trait = create_field_location(trait)
    trait = agg_yield(trait)                                   # unadjusted means per env x hybrid
    weather = pd.concat([pd.read_csv(RAW / "4_Training_Weather_Data_2014_2023_full_year.csv"),
                         pd.read_csv(RAW / "4_Testing_Weather_Data_2024_full_year.csv")], ignore_index=True)
    soil = pd.concat([pd.read_csv(RAW / "3_Training_Soil_Data_2015_2023.csv"),
                      pd.read_csv(RAW / "3_Testing_Soil_Data_2024.csv")], ignore_index=True)
    ec = pd.concat([pd.read_csv(RAW / "6_Training_EC_Data_2014_2023.csv"),
                    pd.read_csv(RAW / "6_Testing_EC_Data_2024.csv")]).drop_duplicates("Env").set_index("Env")
    blues = pd.read_csv(OUT / "blues.csv")
    kin = pd.read_parquet(OUT / "kinship.parquet").set_index("Hybrid")
    return meta, trait, weather, soil, ec, blues, kin


def env_features(xtr, xte, meta, trait, weather, soil, ec, ref_tr, ref_te, seed):
    """create_datasets.py feature engineering for a training frame and a prediction frame (Env, Hybrid[, Yield])."""
    wf = feat_eng_weather(weather.copy())
    sf = feat_eng_soil(soil)
    out = []
    for x, ref in ((xtr, ref_tr), (xte, ref_te)):
        x = x.merge(wf, on="Env", how="left").merge(sf, on="Env", how="left")
        out.append(x)
    xtr, xte = out
    etr = ec[ec.index.isin(xtr["Env"])]
    ete = ec[ec.index.isin(xte["Env"])]
    svd = TruncatedSVD(n_components=15, n_iter=20, random_state=seed).fit(etr)
    cols = [f"EC_svd_comp{i}" for i in range(15)]
    xtr = xtr.merge(pd.DataFrame(svd.transform(etr), index=etr.index, columns=cols), left_on="Env", right_index=True, how="left")
    xte = xte.merge(pd.DataFrame(svd.transform(ete), index=ete.index, columns=cols), left_on="Env", right_index=True, how="left")
    xtr, xte = create_field_location(xtr), create_field_location(xte)
    xtr = xtr.merge(feat_eng_target(trait, ref_year=ref_tr, lag=2), on="Field_Location", how="left")
    xte = xte.merge(feat_eng_target(trait, ref_year=ref_te, lag=2), on="Field_Location", how="left")
    for d in (xtr, xte):
        d["T2M_std_spring_X_weather_station_lat"] = d["T2M_std_spring"] * d["weather_station_lat"]
        d["T2M_std_fall_X_weather_station_lat"] = d["T2M_std_fall"] * d["weather_station_lat"]
        d["T2M_min_fall_X_weather_station_lat"] = d["T2M_min_fall"] * d["weather_station_lat"]
        d["weather_station_lat"] = d["weather_station_lat"].apply(lambda v: lat_lon_to_bin(v, LAT_BIN_STEP))
        d["weather_station_lon"] = d["weather_station_lon"].apply(lambda v: lat_lon_to_bin(v, LON_BIN_STEP))
    xtr, xte = xtr.drop(columns="Field_Location"), xte.drop(columns="Field_Location")
    feat = [c for c in xtr.columns if c not in ("Env", "Hybrid", "Yield_Mg_ha", "Year")]
    for c in feat:
        m = xtr[c].mean()
        xtr[c] = xtr[c].fillna(m)
        xte[c] = xte[c].fillna(m) if c in xte else m
    return xtr, xte, feat


def fit_predict(xtr, xte, feat, kin, seed, use_e=True):
    """run_g_or_gxe_model.py --model=G --A [--E] --svd --lag_features (n_components=100)."""
    ind = list(dict.fromkeys(xtr["Hybrid"].tolist() + xte["Hybrid"].tolist()))
    A = kin.loc[ind, ind]
    A.columns = [f"{c}_A" for c in A.columns]
    lag = [c for c in feat if "yield_lag" in c]
    efeat = [c for c in feat if c not in lag]
    mats = []
    for x in (xtr, xte):
        M = A.loc[x["Hybrid"]].to_numpy()
        if use_e:
            M = np.hstack([M, x[efeat].to_numpy(float)])
        mats.append(M)
    svd = TruncatedSVD(n_components=100, random_state=seed).fit(mats[0])
    frames = []
    for x, M in zip((xtr, xte), mats):
        f = pd.DataFrame(svd.transform(M), columns=[f"svd{i}" for i in range(100)])
        f[lag] = x[lag].to_numpy(float)
        f["Field_Location"] = x["Env"].str.replace("(_).*", "", regex=True).astype("category").to_numpy()
        f["Field_Location"] = f["Field_Location"].astype("category")
        frames.append(f)
    model = lgbm.LGBMRegressor(random_state=seed, max_depth=3, verbose=-1, n_jobs=2)   # n_jobs: machine-share rule only
    model.fit(frames[0], xtr["Yield_Mg_ha"].to_numpy(float))
    return model.predict(frames[1])


def train_frame(blues, trait, meta, years, locs, kin):
    t = trait[trait["Year"].isin(years)].merge(blues, on=["Env", "Hybrid"], how="left")
    t["predicted.value"] = t["predicted.value"].fillna(t["Yield_Mg_ha"])
    t = process_blues(t)
    t = create_field_location(t)
    t = t[t["Hybrid"].isin(kin.index) & t["Field_Location"].isin(locs)].dropna(subset=["Yield_Mg_ha"])
    return t[["Env", "Hybrid", "Yield_Mg_ha"] + META_COLS[1:]].reset_index(drop=True)


def forward(Y: int):
    meta, trait, weather, soil, ec, blues, kin = load_tables()
    spl = json.load(open(BENCH / "splits.json"))["G2F"]["scorable_environments"][str(Y)]
    cells = pd.read_parquet(BENCH / "cells_G2F.parquet")
    te = cells[(cells["year"] == Y) & cells["env"].isin(spl)][["env", "genotype"]].rename(columns={"env": "Env", "genotype": "Hybrid"})
    te = te[te["Hybrid"].isin(kin.index)].merge(meta[META_COLS], on="Env", how="left").reset_index(drop=True)
    locs = set(te["Env"].str.replace("(_).*", "", regex=True))
    tr = train_frame(blues, trait[trait["Year"] < Y], meta, [Y - 2, Y - 1], locs, kin)
    assert tr["Env"].str[-4:].astype(int).max() < Y
    res = te[["Env", "Hybrid"]].copy()
    pe, pg = [], []
    for s in SEEDS:
        random.seed(s)
        xtr, xte, feat = env_features(tr.copy(), te.copy(), meta, trait[trait["Year"] < Y], weather, soil, ec, Y - 1, Y, s)
        pe.append(fit_predict(xtr, xte, feat, kin, s, use_e=True))
        pg.append(fit_predict(xtr, xte, feat, kin, s, use_e=False))
    res["flgbm_gae"], res["flgbm_g"] = np.mean(pe, 0), np.mean(pg, 0)
    res.rename(columns={"Env": "env", "Hybrid": "genotype"}).to_parquet(OUT / f"pred_Y{Y}.parquet", index=False)
    json.dump({"target": Y, "train_rows": len(tr), "train_years": [Y - 2, Y - 1], "test_rows": len(res),
               "locations": len(locs)}, open(OUT / f"meta_Y{Y}.json", "w"), indent=1)
    print("FLGBM_DONE", Y, len(tr), len(res))


MARKERS = None


def cell_reml_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """cell_reml (dartgxe.forward.cellspec, frozen) on the given training rows: environment fixed effects, marker
    GBLUP, REML; returns the genotype values for the test rows (constant within a genotype, so only within-environment
    ranks and the genotype part of pooled r are compared)."""
    global MARKERS
    from dartgxe.forward.cellspec import Prep, fit_ia
    if MARKERS is None:
        MARKERS = pd.read_parquet(RA / "lc_inputs" / "markers.parquet").set_index("genotype")
    tr = train.rename(columns={"Env": "env", "Hybrid": "genotype", "Yield_Mg_ha": "y"})[["env", "genotype", "y"]]
    tr = tr.assign(year=0)
    te = test.rename(columns={"Env": "env", "Hybrid": "genotype"})[["env", "genotype"]].assign(year=1, y=0.0)
    P = Prep(tr, MARKERS, te)
    u = pd.Series(fit_ia(P, np.ones(P.n_env), P.yc, 0.0)["u_t"], index=P.test_ids)
    return u.reindex(te["genotype"]).to_numpy(float)


def cv0():
    """The original CV0 (preprocessing.create_folds, cv=0, val_year=2021): train 2019-2020 at known locations
    (NYS1 removed), 5 GroupKFold folds of 2021 hybrids, each fold trained on a random 60% of the other hybrids plus the
    fold's hybrids (their earlier-year records). Pooled Pearson per fold, averaged over folds and seeds 1..10."""
    meta, trait, weather, soil, ec, blues, kin = load_tables()
    t = trait[trait["Year"].isin([2019, 2020, 2021])].merge(blues, on=["Env", "Hybrid"], how="left")
    t["predicted.value"] = t["predicted.value"].fillna(t["Yield_Mg_ha"])
    t = create_field_location(process_blues(t)).dropna(subset=["Yield_Mg_ha"])
    rows = []
    for s in SEEDS:
        tr0 = t[t["Year"].isin([2019, 2020])]
        va0 = t[t["Year"] == 2021]
        known = (set(tr0["Field_Location"]) & set(va0["Field_Location"])) - {"NYS1"}
        tr0 = tr0[tr0["Hybrid"].isin(kin.index) & tr0["Field_Location"].isin(known)].sample(frac=1, random_state=s).reset_index(drop=True)
        va0 = va0[va0["Hybrid"].isin(kin.index) & va0["Field_Location"].isin(known)].sample(frac=1, random_state=s).reset_index(drop=True)
        folds = np.zeros(len(va0), int)
        for i, (_, v) in enumerate(GroupKFold(n_splits=5).split(X=va0, groups=va0["Hybrid"])):
            folds[v] = i
        allh = set(tr0["Hybrid"]) | set(va0["Hybrid"])
        for f in range(5):
            xv = va0[folds == f]
            random.seed(s)
            cand = list(allh - set(xv["Hybrid"]))
            sel = random.choices(cand, k=int(len(cand) * 0.6))
            xt = tr0[tr0["Hybrid"].isin(sel + xv["Hybrid"].tolist())]
            keep = ["Env", "Hybrid", "Yield_Mg_ha"] + META_COLS[1:]
            xtr, xte, feat = env_features(xt[keep].copy(), xv[keep].copy(), meta, trait, weather, soil, ec, 2020, 2021, s)
            p = fit_predict(xtr, xte, feat, kin, s, use_e=True)
            y = xv["Yield_Mg_ha"].to_numpy(float)
            q = cell_reml_pred(xt, xv)               # addendum D: cell_reml on the same training rows and fold
            env = xv["Env"].to_numpy()
            r = {"seed": s, "fold": f, "n_val": len(xv)}
            for name, pr in (("lgbm", p), ("cell_reml", q)):
                d = pd.DataFrame({"e": env, "y": y, "p": pr})
                r[f"pooled_pearson_{name}"] = float(np.corrcoef(y, pr)[0, 1])
                r[f"within_pearson_{name}"] = float(d.groupby("e")[["y", "p"]].apply(lambda g: g["y"].corr(g["p"]) if len(g) > 2 else np.nan).mean())
                r[f"within_spearman_{name}"] = float(d.groupby("e")[["y", "p"]].apply(lambda g: g["y"].corr(g["p"], method="spearman") if len(g) > 2 else np.nan).mean())
            r["pooled_pearson"], r["within_env_pearson"] = r["pooled_pearson_lgbm"], r["within_pearson_lgbm"]
            rows.append(r)
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "cv0_fidelity.csv", index=False)
    print(R.drop(columns=["seed", "fold", "n_val"]).mean().round(4).to_string())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "forward":
        forward(int(sys.argv[2]))
    else:
        cv0()
