"""R9 track A step 0b: characterise how the 2014-2021 yields (and hybrid labels) in the 2025 release differ from the
local R1-R8 files. Plot-level match on the field-position keys; hybrid x env means; per-env agreement.
Usage: python v9_yield_diff.py <new_training_dir> <old_g2f_raw_dir> <out.json>"""
import sys, json
import numpy as np, pandas as pd
NEW, OLD, OUTF = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/", sys.argv[3]
tn = pd.read_csv(NEW + "1_Training_Trait_Data_2014_2023.csv", low_memory=False); tn = tn[tn.Year <= 2021]
to = pd.read_csv(OLD + "1_Training_Trait_Data_2014_2021.csv", low_memory=False)
q = lambda a: {f"q{int(p*100)}": float(np.nanquantile(a, p)) for p in (0.5, 0.9, 0.99, 1.0)}
out = {}
# plot level: match on field position within env
K = ["Env", "Replicate", "Block", "Plot", "Range", "Pass"]
dup_n, dup_o = int(tn.duplicated(K).sum()), int(to.duplicated(K).sum())
m = to.merge(tn, on=K, how="outer", suffixes=("_o", "_n"), indicator=True)
b = m[m._merge == "both"]
d = (b.Yield_Mg_ha_n - b.Yield_Mg_ha_o).values; ad = np.abs(d)
both_nan = b.Yield_Mg_ha_n.isna() & b.Yield_Mg_ha_o.isna()
out["plot_level"] = dict(keys=K, duplicated_keys_new=dup_n, duplicated_keys_old=dup_o, n_matched=int(len(b)), n_only_old=int((m._merge == "left_only").sum()),
    n_only_new=int((m._merge == "right_only").sum()), n_hybrid_label_changed=int((b.Hybrid_o != b.Hybrid_n).sum()),
    n_yield_both_nan=int(both_nan.sum()), n_yield_nan_only_one_side=int((b.Yield_Mg_ha_n.isna() ^ b.Yield_Mg_ha_o.isna()).sum()),
    n_yield_exact_equal=int((b.Yield_Mg_ha_n == b.Yield_Mg_ha_o).sum()), n_yield_absdiff_lt_1em6=int((ad < 1e-6).sum()),
    n_yield_absdiff_lt_0_01=int((ad < 0.01).sum()), n_yield_absdiff_ge_0_1=int((ad >= 0.1).sum()), n_yield_absdiff_ge_1=int((ad >= 1).sum()),
    absdiff_quantiles=q(ad), mean_signed_diff=float(np.nanmean(d)), ratio_new_over_old_quantiles=q((b.Yield_Mg_ha_n / b.Yield_Mg_ha_o).values),
    corr_plot_yield=float(pd.Series(b.Yield_Mg_ha_n.values).corr(pd.Series(b.Yield_Mg_ha_o.values))))
# per env
pe = []
for e, g in b.groupby("Env"):
    ok = g.Yield_Mg_ha_n.notna() & g.Yield_Mg_ha_o.notna()
    if ok.sum() < 3: continue
    x, y = g.Yield_Mg_ha_o[ok].values, g.Yield_Mg_ha_n[ok].values
    pe.append(dict(Env=e, n=int(ok.sum()), r=float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan,
                   mean_old=float(x.mean()), mean_new=float(y.mean()), max_absdiff=float(np.abs(y - x).max()), frac_equal=float(np.mean(np.abs(y - x) < 1e-6)),
                   slope=float(np.polyfit(x, y, 1)[0])))
pe = pd.DataFrame(pe)
out["per_env"] = dict(n_env=int(len(pe)), r_quantiles=q(pe.r.values), frac_equal_quantiles=q(pe.frac_equal.values), slope_quantiles=q(pe.slope.values),
    n_env_all_equal=int((pe.frac_equal == 1).sum()), n_env_r_below_0_99=int((pe.r < 0.99).sum()),
    envs_r_below_0_99=pe[pe.r < 0.99].sort_values("r")[["Env", "n", "r", "max_absdiff"]].round(4).to_dict("records"),
    by_year=pe.assign(Year=pe.Env.str[-4:]).groupby("Year").agg(mean_frac_equal=("frac_equal", "mean"), min_r=("r", "min")).round(4).to_dict("index"))
pe["shift"] = pe.mean_new - pe.mean_old
out["per_env"]["mean_shift_quantiles"] = q(np.abs(pe["shift"].values))
# phenology columns at plot level (tolerance 1e-9, NaN-aware)
for col in ("Silk_DAP_days", "Pollen_DAP_days"):
    x, y = b[col + "_o"].astype(float).values, b[col + "_n"].astype(float).values
    same = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9)
    out["plot_level"][f"n_{col}_differing"] = int((~same).sum())
out["plot_level"]["hybrid_label_change_examples"] = b.loc[b.Hybrid_o != b.Hybrid_n, ["Env", "Plot", "Hybrid_o", "Hybrid_n"]].head(15).to_dict("records")
# hybrid x env mean yields (as in the pipeline), tolerance comparison
mo = to[to.Yield_Mg_ha.notna()].groupby(["Env", "Hybrid"]).Yield_Mg_ha.mean(); mn = tn[tn.Yield_Mg_ha.notna()].groupby(["Env", "Hybrid"]).Yield_Mg_ha.mean()
j = pd.concat([mo.rename("o"), mn.rename("n")], axis=1)
both = j.dropna(); dd = (both.n - both.o).abs()
out["hybrid_env_mean_yield"] = dict(n_old=int(mo.size), n_new=int(mn.size), n_both=int(len(both)), n_only_old=int(j.n.isna().sum()), n_only_new=int(j.o.isna().sum()),
                                    n_absdiff_gt_1e_6=int((dd > 1e-6).sum()), max_absdiff=float(dd.max()), n_absdiff_gt_0_01=int((dd > 0.01).sum()))
# weather env sets (2014-2021)
wo = pd.read_csv(OLD + "4_Training_Weather_Data_2014_2021.csv", usecols=["Env"]); wn = pd.read_csv(NEW + "4_Training_Weather_Data_2014_2023_full_year.csv", usecols=["Env"])
en = {e for e in wn.Env.unique() if int(e[-4:]) <= 2021}
out["weather_envs_2014_2021"] = dict(n_old=int(wo.Env.nunique()), n_new=len(en), only_new=sorted(en - set(wo.Env)), only_old=sorted(set(wo.Env) - en))
json.dump(out, open(OUTF, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print(json.dumps(out["plot_level"]))
