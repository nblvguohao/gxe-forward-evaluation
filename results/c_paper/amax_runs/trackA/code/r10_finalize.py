"""R10 track A finaliser: builds the deliverable R10_A_results.json from
(1) the as-run primary summary (R10_A_results_asrun.json, sha256 checked; every primary number and verdict is copied unchanged),
(2) the 2024 weather missing-value diagnostic (R10_A_weather_diag.json), and
(3) the POST-HOC weather-gap sensitivity summary (same summariser on a 2024 fold refit with the 3-day gap interpolated).
It adds a gate-level note on which primary quantities used defective 2024 weather inputs and extends the deviations.
Usage: python r10_finalize.py <asrun.json> <weather_diag.json> <sens_results.json> <sens_data.json> <extra_deviations.json> <out_dir>"""
import os, sys, json, hashlib, time
ASRUN, DIAG, SENS, SENSD, XDEV, OUT = sys.argv[1:7]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ASRUN_SHA = "29455efcb2d9880f953cd5a4b6aa3f51265331a1d807e9dc12fc6cf6af7cc048"
assert sha(ASRUN) == ASRUN_SHA
P = json.load(open(ASRUN)); D = json.load(open(DIAG)); S = json.load(open(SENS)); SD = json.load(open(SENSD))
assert S["prereg"]["sha256"] == P["prereg"]["sha256"]
# MG_2k and D_noEC do not use weather: their 2024 mean r should be unchanged by the refit (recorded, not asserted)
refit_check = {k: abs(P["r_mean"]["2024"][k] - S["r_mean"]["2024"][k]) for k in ("MG_2k|P1", "MG_2k|P2", "MG_2k|P3", "D_noEC|P1", "D_noEC|P2", "D_noEC|P3")}
w = D["weather_2024_full_year"]
defect = dict(
    summary=("Released 2024 weather (both the full-year and seasons-only files) has T2M_MAX, T2M_MIN, T2M, RH2M and PRECTOTCORR "
             "missing on 2024-07-12..14 in all 23 envs and GWETROOT missing from 2024-07-02 to 2024-11-10; the 2014-2023 training "
             "weather has no missing values. In the primary run (R9 pipeline, no imputation) the gap propagates through cumulative GDD "
             "and prefix sums, so every 2024 window after 2024-07-11 is undefined."),
    nan_by_var_2024=w["nan_by_var"], nan_dates_by_var_2024=w["nan_dates_by_var"], envs_with_nan_by_var_2024=w["envs_with_nan_by_var"],
    nan_by_var_training=D["weather_training_2014_2023"]["nan_by_var"], season_cum_gdd_nan_days_2024=D["season_cum_gdd_nan_2024"],
    ceris_candidates=D["ceris_candidates"],
    affected_primary_quantities=["D_full 2024 predictions (EC_env windows after 2024-07-11 undefined) -> G1 D_full - MG_2k, G2 D_full - D_noEC",
                                 "CERIS 2024: nested-window predictions undefined in 20 of 22 envs -> G3 2024 alone and the 2024 part of EXT3"],
    unaffected_primary_quantities=["MG_2k and D_noEC (no weather inputs) -> G1 D_noEC - MG_2k; G4 for MG_2k and D_noEC; the MG_2k file for track B"])
keep = ["G1_enhancements_external", "G2_EC_increment", "G3_window_search_leak", "G4_tuning_audit", "r_mean", "fold_2024", "POST_HOC_new_hybrid_subset"]
sens = dict(label=("POST-HOC sensitivity, declared after the primary summary: 2024 fold refit with the 3-day interior gap of five "
                   "weather variables filled by linear interpolation (GWETROOT left missing); 2022/2023 unchanged; identical summariser. "
                   "It does not replace the primary gates."),
            data=SD, source_results_sha256=sha(SENS), **{k: S[k] for k in keep}, overall_verdict_under_same_mapping=S["overall_verdict"],
            refit_abs_diff_2024_mean_r_weather_free_models=refit_check)
res = dict(P)
res["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
res["asrun_primary_results"] = dict(file="R10_A_results_asrun.json", sha256=ASRUN_SHA,
                                    note="all primary blocks below are copied unchanged from this as-run summary")
res["data_defect_2024_weather"] = defect
res["POST_HOC_sensitivity_wx_gap_interpolated"] = sens
res["primary_vs_sensitivity"] = {
    "G1_D_noEC_minus_MG_2k": [P["G1_enhancements_external"]["contrasts"]["D_noEC_minus_MG_2k"]["verdict"], S["G1_enhancements_external"]["contrasts"]["D_noEC_minus_MG_2k"]["verdict"]],
    "G1_D_full_minus_MG_2k": [P["G1_enhancements_external"]["contrasts"]["D_full_minus_MG_2k"]["verdict"], S["G1_enhancements_external"]["contrasts"]["D_full_minus_MG_2k"]["verdict"]],
    "G2": [P["G2_EC_increment"]["verdict"], S["G2_EC_increment"]["verdict"]], "G3": [P["G3_window_search_leak"]["verdict"], S["G3_window_search_leak"]["verdict"]],
    "overall": [P["overall_verdict"], S["overall_verdict"]], "order": ["primary (as run)", "POST-HOC sensitivity"]}
res["deviations_from_prereg"] = P["deviations_from_prereg"] + json.load(open(XDEV))
fo = os.path.join(OUT, "R10_A_results.json")
json.dump(res, open(fo, "w"), indent=1, ensure_ascii=False)
open(fo + ".sha256", "w").write(f"{sha(fo)}  R10_A_results.json\n")
print(json.dumps(res["primary_vs_sensitivity"]))
