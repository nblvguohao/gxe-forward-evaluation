"""Crosswalk check for the integrated sequel (COMPAG): every number the outline uses -> the file it comes from.

Unlike scripts/check_numbers.py (tolerance-based, main-branch Methods/Results only), a row passes only if the value read
from the file, rounded to the number of decimals printed, equals the printed value. Sources live in three places:
  MAIN  this repository's results/ (main branch)
  A     results of the scenario-spectrum and selection-optimism analyses
        (8d43df8), so they sit in results/ too; override with $SEQ_SRC_A
  R     reaction-norm analysis records copied into results/rn_paper/ (from github.com/nblvguohao/kernel-gated-reaction-norm,
        see results/rn_paper/SOURCE.txt; merged into the sequel on the user's decision of 2026-09-30); read from results/
  C     C-paper results copied into results/c_paper/ (workspace/: local JSON, CSV, pre-registrations; amax_runs/: the
        R9/R10 result JSON and per-environment CSVs from amax:<server>/egp_gxe/runs); override with $SEQ_SRC_C
A source whose root is missing is reported SKIP, never PASS. Rows of kind 'summary' have no machine-readable
file (only a Markdown summary); they are listed as UNCHECKED so that they cannot be mistaken for checked numbers.
Run from the repository root:  python3 scripts/check_sequel_sources.py [--md]   (exit code 1 if any row FAILs)"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOTS = {"MAIN": Path("results"),
         "A": Path(os.environ.get("SEQ_SRC_A", "results")),
         "C": Path(os.environ.get("SEQ_SRC_C", "results/c_paper"))}
rows = []  # (id, claim, source, file, locator, printed, reader) ; reader(root) -> float


def csv_row(path, cond, col):
    def r(root):
        d = pd.read_csv(root / path)
        for k, v in cond.items():
            d = d[d[k].astype(str) == str(v)]
        if len(d) != 1:
            raise LookupError(f"{len(d)} rows match {cond}")
        return float(d.iloc[0][col])
    return r


def js(path, *keys):
    def r(root):
        x = json.load(open(root / path))
        for k in keys:
            x = x[k]  # str keys for dicts, int for lists
        return float(x)
    return r


def add(i, claim, src, path, loc, printed, reader=None):
    rows.append((i, claim, src, path, loc, printed, reader))


# ---------------- MAIN: paired reruns (gate2, A1), E1, audit tool, 48-year summary ----------------
g24, g22 = "tables/g2f_gate2_F2024m_vs_B1r_gblup_reml_main.csv", "tables/g2f_gate2_F2022m_vs_B1r_gblup_reml_main.csv"
for i, f, est, lo, hi, nenv in (("M1", g24, "-0.063", "-0.120", "-0.020", "22"), ("M2", g22, "+0.051", "+0.023", "+0.078", "27")):
    c = {"metric": "spearman", "method": "GEFormer_fair"}
    add(i, "GEFormer fair - REML, within-env Spearman", "MAIN", f, "delta", est, csv_row(f, c, "delta"))
    add(i, "  95% CI low (env x seed)", "MAIN", f, "ci2_lo", lo, csv_row(f, c, "ci2_lo"))
    add(i, "  95% CI high", "MAIN", f, "ci2_hi", hi, csv_row(f, c, "ci2_hi"))
    add(i, "  scored environments", "MAIN", f, "n_env", nenv, csv_row(f, c, "n_env"))
for i, sc, est, lo, hi in (("M3", "F2024m", "-0.084", "-0.163", "-0.019"), ("M3", "F2022m", "-0.058", "-0.142", "+0.004")):
    f, c = f"tables/g2f_gate2_{sc}_vs_GEFormer_fair_main.csv", {"metric": "spearman", "method": "GEFormer_own"}
    add(i, f"GEFormer own - fair, {sc}, within-env Spearman", "MAIN", f, "delta", est, csv_row(f, c, "delta"))
    add(i, "  CI low", "MAIN", f, "ci2_lo", lo, csv_row(f, c, "ci2_lo"))
    add(i, "  CI high", "MAIN", f, "ci2_hi", hi, csv_row(f, c, "ci2_hi"))
for sc, want in (("F2024m", "+0.202"), ("F2022m", "+0.200")):
    f = f"tables/g2f_gate2_{sc}_vs_GEFormer_fair_main.csv"
    own = csv_row(f, {"metric": "spearman", "method": "GEFormer_own"}, "pooled_pearson_r_diag")
    fair = csv_row(f, {"metric": "spearman", "method": "GEFormer_fair"}, "pooled_pearson_r_diag")
    add("M4", f"GEFormer own - fair, {sc}, pooled Pearson (point only; CI has no result file)", "MAIN", f,
        "pooled_pearson_r_diag own - fair", want, lambda root, o=own, a=fair: o(root) - a(root))
for sc, est, lo, hi, res in (("F2024m", "-0.231", "-0.352", "-0.120", "0.032"), ("F2022m", "-0.177", "-0.236", "-0.124", "0.030")):
    c = {"scenario": sc, "contrast": "C_fair", "metric": "spearman"}
    add("M5", f"GE-BiFormer fair - REML, {sc}", "MAIN", "a1/summary.csv", "delta", est, csv_row("a1/summary.csv", c, "delta"))
    add("M5", "  CI low", "MAIN", "a1/summary.csv", "ci2_lo", lo, csv_row("a1/summary.csv", c, "ci2_lo"))
    add("M5", "  CI high", "MAIN", "a1/summary.csv", "ci2_hi", hi, csv_row("a1/summary.csv", c, "ci2_hi"))
    add("M5", "  resolution 3/sqrt(N)", "MAIN", "a1/summary.csv", "resolution", res, csv_row("a1/summary.csv", c, "resolution"))
for sc, est, lo, hi in (("F2024m", "+0.170", "+0.068", "+0.263"), ("F2022m", "+0.021", "-0.135", "+0.144")):
    c = {"scenario": sc, "contrast": "O", "metric": "spearman"}
    add("M6", f"GE-BiFormer own - fair, {sc}", "MAIN", "a1/summary.csv", "delta", est, csv_row("a1/summary.csv", c, "delta"))
    add("M6", "  CI low", "MAIN", "a1/summary.csv", "ci2_lo", lo, csv_row("a1/summary.csv", c, "ci2_lo"))
    add("M6", "  CI high", "MAIN", "a1/summary.csv", "ci2_hi", hi, csv_row("a1/summary.csv", c, "ci2_hi"))
e1 = "wp4/e1_exact_verdict.json"
add("M7", "E1-1: pooled-Pearson epoch has larger phi (runs in direction)", "MAIN", e1, "E1_1_phi.in_direction", "18", js(e1, "E1_1_phi", "in_direction"))
add("M7", "E1-2 exact: scale closer to u* (runs in direction, of 16)", "MAIN", e1, "E1_2_abslog.in_direction", "12", js(e1, "E1_2_abslog", "in_direction"))
add("M7", "E1-2 exact one-sided p", "MAIN", e1, "E1_2_abslog.p_one_sided", "0.038", js(e1, "E1_2_abslog", "p_one_sided"))
for sc, want in (("F2024m", "0.077"), ("F2022m", "0.124")):
    f = f"wp2/case_gate2_{sc}_own.json"
    add("M8", f"Audit tool A4: pooled-Pearson optimism, GEFormer own {sc}", "MAIN", f, "A4_optimism.optimism", want, js(f, "A4_optimism", "optimism"))
s48 = "summary48/pooled.csv"
for item, est, lo, hi in (("Oracle", "+0.019", "+0.010", "+0.027"), ("R_STK_FW", "-0.002", "-0.012", "+0.007"),
                          ("rn_ridge", "-0.020", "-0.044", "+0.005"), ("gxe_gbm", "-0.036", "-0.072", "-0.000"),
                          ("dl_g", "-0.008", "-0.018", "+0.001"), ("rf", "-0.026", "-0.040", "-0.011")):
    c = {"item": item, "metric": "spearman", "scope": "all48"}
    add("M9", f"48-year summary: {item} - cell_reml", "MAIN", s48, "est", est, csv_row(s48, c, "est"))
    add("M9", "  CI low (DL)", "MAIN", s48, "lo", lo, csv_row(s48, c, "lo"))
    add("M9", "  CI high", "MAIN", s48, "hi", hi, csv_row(s48, c, "hi"))

# ---------------- A: scenario spectrum and selection optimism  ----------------
sp = "headroom_sparse/pooled.csv"
for frac, est, lo, hi, res in (("0.25", "+0.0187", "+0.0136", "+0.0238", "0.0103"), ("0.5", "+0.0349", "+0.0253", "+0.0445", "0.0140")):
    c = {"fraction": frac, "contrast": "m1-m0", "metric": "sp", "range": "all48"}
    add("A1", f"Sparse testing {frac}: MxE - main effect", "A", sp, "est", est, csv_row(sp, c, "est"))
    add("A1", "  CI low", "A", sp, "lo", lo, csv_row(sp, c, "lo"))
    add("A1", "  CI high", "A", sp, "hi", hi, csv_row(sp, c, "hi"))
    add("A1", "  resolution", "A", sp, "resolution", res, csv_row(sp, c, "resolution"))
add("A2", "Sparse 0.25: learned env correlation on top of MxE (M2-M1)", "A", "headroom_sparse/verdict.json", "main_m2_m1.est", "+0.0001",
    js("headroom_sparse/verdict.json", "main_m2_m1", "est"))
for k, want in (("stk", "-0.0028"), ("dlres", "-0.0014"), ("ecmxe", "-0.0012")):
    add("A2", f"Sparse 0.25: {k} on top of MxE (Holm interval in verdict)", "A", "headroom_sparse2/verdict.json", f"decisions.{k}.est", want,
        js("headroom_sparse2/verdict.json", "decisions", k, "est"))
ia = "headroom_ia_ib/verdict.json"
for key, sub, est, lo, hi, res in (("I-A", "all48", "+0.0055", "+0.0009", "+0.0100", "0.0087"), ("I-B", "loc43", "+0.0120", "+0.0057", "+0.0183", "0.0091")):
    add("A3", f"Forward {key} ({sub})", "A", ia, f"{key}.{sub}.est", est, js(ia, key, sub, "est"))
    add("A3", "  CI low", "A", ia, f"{key}.{sub}.lo", lo, js(ia, key, sub, "lo"))
    add("A3", "  CI high", "A", ia, f"{key}.{sub}.hi", hi, js(ia, key, sub, "hi"))
    add("A3", "  resolution", "A", ia, f"{key}.{sub}.resolution", res, js(ia, key, sub, "resolution"))
add("A4", "I-B held-out other traits", "A", "holdout_ib/verdict.json", "H_hold.est", "-0.0015", js("holdout_ib/verdict.json", "H_hold", "est"))
add("A5", "Old lines H2 (line x location history), Holm 97.5% in verdict", "A", "headroom_oldlines/verdict.json", "decisions.H2.est", "+0.0115",
    js("headroom_oldlines/verdict.json", "decisions", "H2", "est"))
add("A5", "Old lines H1 (marker G x location)", "A", "headroom_oldlines/verdict.json", "decisions.H1.est", "+0.0127",
    js("headroom_oldlines/verdict.json", "decisions", "H1", "est"))
add("A6", "Same-season silking of other environments (SX)", "A", "headroom_secondary/verdict.json", "H1.est", "+0.0005",
    js("headroom_secondary/verdict.json", "H1", "est"))
# A7: hindsight pick per year (same pick as M9's Oracle, library = ten two-stage + cell_reml + rn_ridge + gxe_gbm, complete
# cells only). 'naive' pools with the bootstrap SE of the maximum (so it differs from M9's +0.019, whose year SEs treat the
# pick as fixed); 'cross_fit' picks on half of the year's environments and scores on the other half (scripts/ideas_diag.py).
op = "ideas_diag/oracle_pooled.csv"
for q, est, lo, hi in (("naive", "+0.0129", "+0.0066", "+0.0192"), ("cross_fit", "-0.0037", "-0.0104", "+0.0030"),
                       ("naive_unweighted", "+0.033", "+0.020", "+0.048"), ("cross_fit_unweighted", "+0.009", "-0.005", "+0.024")):
    c = {"metric": "spearman", "scope": "all48", "quantity": q}
    add("A7", f"Hindsight pick - cell_reml, {q}", "A", op, f"quantity={q}", est, csv_row(op, c, "est"))
    add("A7", "  CI low", "A", op, "lo", lo, csv_row(op, c, "lo"))
    add("A7", "  CI high", "A", op, "hi", hi, csv_row(op, c, "hi"))

# ---------------- C: C-paper results (results/c_paper) ----------------
r7 = "workspace/handoff/r7_h1h2.json"
for s_, est, lo, hi, wi in (("S1_G2F_LOYO", "+0.127", "+0.074", "+0.179", "-0.0012"), ("S2_G2F_LOLO", "+0.088", "+0.038", "+0.139", "+0.0005"),
                            ("S3_BRIWECS_LOYO", "+0.239", "+0.046", "+0.437", "+0.0014")):
    kn = lambda k: (lambda root: float(json.load(open(root / r7))[0]["structured_output"]["key_numbers"][k]))
    add("C1", f"Window-search leakage, env-mean r inflation, {s_}", "C", r7, f"{s_}|H1_optimism", est, kn(f"{s_}|H1_optimism"))
    add("C1", "  CI low", "C", r7, f"{s_}|H1_ci_low", lo, kn(f"{s_}|H1_ci_low"))
    add("C1", "  CI high", "C", r7, f"{s_}|H1_ci_high", hi, kn(f"{s_}|H1_ci_high"))
    add("C1", "  within-env r change (leaky - nested)", "C", r7, f"{s_}|C2_within_env_leaky_minus_nested", wi, kn(f"{s_}|C2_within_env_leaky_minus_nested"))
r10, r9 = "amax_runs/trackA/final/R10_A_results.json", "amax_runs/A/final/R9_A_results.json"
g3 = ("G3_window_search_leak", "EXT3_pooled")
add("C2", "Leakage on external years 2022-24 (env-mean r)", "C", r10, "G3.EXT3_pooled.optimism", "+0.016", js(r10, *g3, "optimism"))
add("C2", "  CI low", "C", r10, "G3.EXT3_pooled.ci[0]", "-0.015", js(r10, *g3, "ci", 0))
add("C2", "  CI high", "C", r10, "G3.EXT3_pooled.ci[1]", "+0.056", js(r10, *g3, "ci", 1))
add("C2", "  environments with predictions (2024: 2 of 22)", "C", r10, "G3.EXT3_pooled.n_env", "55", js(r10, *g3, "n_env"))
cv = ("H9a", "per_setting", "S4_G2F_CV00")
add("C3", "Internal split CV00: non-linear stage - ridge (within-env Pearson)", "C", r9, "H9a.S4_G2F_CV00.mean", "+0.0195", js(r9, *cv, "mean"))
add("C3", "  CI low", "C", r9, "ci_low", "+0.0093", js(r9, *cv, "ci_low"))
add("C3", "  CI high", "C", r9, "ci_high", "+0.0299", js(r9, *cv, "ci_high"))
add("C3", "  environments", "C", r9, "n_env", "211", js(r9, *cv, "n_env"))
ex = ("G1_enhancements_external", "contrasts", "D_noEC_minus_MG_2k", "EXT3_pooled")
add("C3", "Same contrast on 3 external years", "C", r10, "G1.D_noEC_minus_MG_2k.EXT3_pooled.mean", "-0.0061", js(r10, *ex, "mean"))
add("C3", "  CI low", "C", r10, "ci_low", "-0.0100", js(r10, *ex, "ci_low"))
add("C3", "  CI high", "C", r10, "ci_high", "-0.0022", js(r10, *ex, "ci_high"))
tu = ("G4_tuning_audit", "MG_2k", "P2_minus_P3")
add("C4", "Tuning on the test year (P2) - inner CV (P3), ridge 2k, EXT3", "C", r10, "G4.MG_2k.P2_minus_P3.EXT3_pooled.mean", "+0.030", js(r10, *tu, "EXT3_pooled", "mean"))
add("C4", "  CI low", "C", r10, "ci_low", "+0.023", js(r10, *tu, "EXT3_pooled", "ci_low"))
add("C4", "  CI high", "C", r10, "ci_high", "+0.037", js(r10, *tu, "EXT3_pooled", "ci_high"))
add("C4", "  2022 alone", "C", r10, "per_year.2022.mean", "+0.075", js(r10, *tu, "per_year", "2022", "mean"))
add("C4", "Largest pooled model contrast: EC increment D_full - D_noEC", "C", r10, "G2_EC_increment.EXT3_pooled.mean", "+0.0093",
    js(r10, "G2_EC_increment", "EXT3_pooled", "mean"))


def n_cells(per_env_csv, envs_csv, env_col="env"):
    """Cells N = sum of per-environment genotype counts (track B tables, ridge 2k, P3, all lines) over the environments
    scored in the track-A/R9 table; resolution = 3/sqrt(N)."""
    def r(root):
        n = pd.concat([pd.read_csv(root / f) for f in per_env_csv])
        n = n[(n["model"] == "MG_2k") & (n["protocol"] == "P3") & (n["subset"] == "all")].drop_duplicates("Env")
        e = pd.read_csv(root / envs_csv)[[env_col]].merge(n[["Env", "n"]], left_on=env_col, right_on="Env", how="left")
        if e["n"].isna().any():
            raise LookupError(f"{int(e['n'].isna().sum())} environments without n")
        return float(e["n"].sum())
    return r


extN = n_cells(["amax_runs/trackB/out/EXT_selected_per_env.csv", "amax_runs/trackB/out10/G2024_selected_per_env.csv"],
               "amax_runs/trackA/final/R10_A_EXT3_per_env_r.csv")
s4N = n_cells(["amax_runs/trackB/out/S4_selected_per_env.csv"], "amax_runs/A/final/R9_A_S4_G2F_CV00_per_env_r.csv")
add("C6", "N cells, external 3 years (75 environments)", "C", "amax_runs/trackB/*_selected_per_env.csv", "sum n", "24995", extN)
add("C6", "  resolution 3/sqrt(N)", "C", "same", "3/sqrt(N)", "0.0190", lambda root: 3 / extN(root) ** 0.5)
add("C6", "N cells, CV00 (211 environments)", "C", "amax_runs/trackB/out/S4_selected_per_env.csv", "sum n", "82082", s4N)
add("C6", "  resolution 3/sqrt(N)", "C", "same", "3/sqrt(N)", "0.0105", lambda root: 3 / s4N(root) ** 0.5)


def eswyt(root):
    d = pd.read_csv(root / "workspace/R10_posthoc_ESWYT_repeatability.csv")
    return float(d[d["nursery"].isin([34, 35, 36, 37])]["mean_pairwise_r"].mean())


add("C5", "ESWYT between-environment correlation, test nurseries 34-37 (mean)", "C", "workspace/R10_posthoc_ESWYT_repeatability.csv",
    "mean_pairwise_r", "0.044", eswyt)

# ---------------- added for manuscript/tag_draft_v1 (numbers printed in the draft that were not yet checked) ----------------
for sc, own, fair in (("F2024m", "0.078", "0.162"), ("F2022m", "0.216", "0.274")):
    f = f"tables/g2f_gate2_{sc}_vs_GEFormer_fair_main.csv"
    add("M10", f"Table 2 level: GEFormer own, {sc}", "MAIN", f, "mean (spearman, GEFormer_own)", own, csv_row(f, {"metric": "spearman", "method": "GEFormer_own"}, "mean"))
    add("M10", f"Table 2 level: GEFormer fair, {sc}", "MAIN", f, "mean (spearman, GEFormer_fair)", fair, csv_row(f, {"metric": "spearman", "method": "GEFormer_fair"}, "mean"))


def a1_level(sc, method):
    def r(root):
        d = pd.read_csv(root / "a1/levels.csv")
        return float(d[(d["scenario"] == sc) & (d["method"] == method)]["within_spearman"].mean())
    return r


for sc, own, fair in (("F2024m", "0.167", "-0.003"), ("F2022m", "0.060", "0.040")):
    add("M10", f"Table 2 level: GE-BiFormer own, {sc} (mean of 3 seeds)", "MAIN", "a1/levels.csv", "within_spearman", own, a1_level(sc, "GEBiFormer_own"))
    add("M10", f"Table 2 level: GE-BiFormer fair, {sc}", "MAIN", "a1/levels.csv", "within_spearman", fair, a1_level(sc, "GEBiFormer_fair"))


def gef_res(sc):
    def r(root):
        d = pd.read_csv(root / "e1/geformer_rerun_vs_gate2.csv")
        return 3 / float(d[d["scenario"] == sc]["cells"].iloc[0]) ** 0.5
    return r


add("M10", "GEFormer resolution F2024m (3/sqrt cells)", "MAIN", "e1/geformer_rerun_vs_gate2.csv", "cells", "0.031", gef_res("F2024m"))
add("M10", "GEFormer resolution F2022m", "MAIN", "e1/geformer_rerun_vs_gate2.csv", "cells", "0.028", gef_res("F2022m"))
add("M10", "phi of GEFormer own pick, F2024m", "MAIN", "wp2/case_gate2_F2024m_own.json", "A3_source.phi", "0.95", js("wp2/case_gate2_F2024m_own.json", "A3_source", "phi"))
for item, est, lo, hi in (("reml", "-0.012", "-0.019", "-0.006"), ("gbm", "-0.041", "-0.058", "-0.024"), ("mlp", "-0.089", "-0.110", "-0.067"),
                          ("R_CV", "-0.010", "-0.019", "-0.001")):
    c = {"item": item, "metric": "spearman", "scope": "all48"}
    add("M9", f"48-year summary: {item} - cell_reml", "MAIN", s48, "est", est, csv_row(s48, c, "est"))
    add("M9", "  CI low", "MAIN", s48, "lo", lo, csv_row(s48, c, "lo"))
    add("M9", "  CI high", "MAIN", s48, "hi", hi, csv_row(s48, c, "hi"))


def picks(cls):
    def r(root):
        d = pd.read_csv(root / "ideas_diag/oracle_picks.csv")
        return float(d["class"].isin(cls).sum())
    return r


for cls, n in ((["cell_reml"], "20"), (["nonlinear_2stage"], "15"), (["linear_2stage"], "9"), (["gxe_linear", "gxe_nonlinear"], "4")):
    add("A7", f"Hindsight pick is {'/'.join(cls)} (years of 48)", "A", "ideas_diag/oracle_picks.csv", "class counts", n, picks(cls))
sv, s2v, hv, ov, xv = ("headroom_sparse/verdict.json", "headroom_sparse2/verdict.json", "holdout_ib/verdict.json",
                       "headroom_oldlines/verdict.json", "headroom_secondary/verdict.json")
add("A2", "  M2-M1 CI low", "A", sv, "main_m2_m1.lo", "-0.0029", js(sv, "main_m2_m1", "lo"))
add("A2", "  M2-M1 CI high", "A", sv, "main_m2_m1.hi", "+0.0030", js(sv, "main_m2_m1", "hi"))
for k, lo, hi in (("stk", "-0.0061", "+0.0006"), ("dlres", "-0.0047", "+0.0018"), ("ecmxe", "-0.0040", "+0.0017")):
    add("A2", f"  {k} Holm CI low", "A", s2v, f"decisions.{k}.ci[0]", lo, js(s2v, "decisions", k, "ci", 0))
    add("A2", f"  {k} Holm CI high", "A", s2v, f"decisions.{k}.ci[1]", hi, js(s2v, "decisions", k, "ci", 1))
add("A2", "  ecmxe resolution", "A", s2v, "decisions.ecmxe.resolution", "0.0150", js(s2v, "decisions", "ecmxe", "resolution"))
add("A2", "  stk Holm level", "A", s2v, "decisions.stk.holm_level", "0.983", js(s2v, "decisions", "stk", "holm_level"))
add("A2", "  dlres Holm level", "A", s2v, "decisions.dlres.holm_level", "0.975", js(s2v, "decisions", "dlres", "holm_level"))
c = {"fraction": "0.25", "method": "stk", "metric": "sp", "range": "main"}
add("A2", "  stk unadjusted 95% CI low", "A", "headroom_sparse2/pooled.csv", "lo", "-0.0055", csv_row("headroom_sparse2/pooled.csv", c, "lo"))
add("A2", "  stk unadjusted 95% CI high", "A", "headroom_sparse2/pooled.csv", "hi", "-0.00001", csv_row("headroom_sparse2/pooled.csv", c, "hi"))
add("A4", "  held-out CI low", "A", hv, "H_hold.lo", "-0.0036", js(hv, "H_hold", "lo"))
add("A4", "  held-out CI high", "A", hv, "H_hold.hi", "+0.0005", js(hv, "H_hold", "hi"))
add("A4", "  held-out resolution", "A", hv, "H_hold.resolution", "0.0164", js(hv, "H_hold", "resolution"))
add("A4", "  held-out units", "A", hv, "H_hold.n_units", "26", js(hv, "H_hold", "n_units"))
for k, lo, hi in (("H2", "+0.0035", "+0.0195"), ("H1", "-0.0040", "+0.0293")):
    add("A5", f"  {k} CI low", "A", ov, f"decisions.{k}.ci[0]", lo, js(ov, "decisions", k, "ci", 0))
    add("A5", f"  {k} CI high", "A", ov, f"decisions.{k}.ci[1]", hi, js(ov, "decisions", k, "ci", 1))
add("A5", "  H2 Holm level", "A", ov, "decisions.H2.holm_level", "0.975", js(ov, "decisions", "H2", "holm_level"))
add("A5", "  old-lines resolution", "A", ov, "decisions.H2.resolution", "0.0266", js(ov, "decisions", "H2", "resolution"))
add("A6", "  SX CI low", "A", xv, "H1.lo", "-0.0035", js(xv, "H1", "lo"))
add("A6", "  SX CI high", "A", xv, "H1.hi", "+0.0045", js(xv, "H1", "hi"))
add("A6", "  SX resolution", "A", xv, "resolution", "0.0143", js(xv, "resolution"))
add("A6", "  SX target years (G2F)", "A", xv, "H1.k", "4", js(xv, "H1", "k"))
add("A3", "  I-B target years", "A", ia, "I-B.loc43.k", "43", js(ia, "I-B", "loc43", "k"))
for ds, est, lo, hi in (("G2F", "+0.039", "+0.022", "+0.056"), ("GEM_IA", "+0.022", "+0.001", "+0.043"), ("NUST", "+0.028", "+0.019", "+0.037"),
                        ("MU_SOY", "+0.027", "+0.018", "+0.036"), ("ESWYT", "+0.009", "+0.004", "+0.015"), ("URSN", "+0.000", "-0.011", "+0.011")):
    c = {"fraction": "0.25", "contrast": "m1-m0", "metric": "sp", "range": ds}
    add("A1", f"Sparse 25%: MxE - main effect, {ds}", "A", sp, "est", est, csv_row(sp, c, "est"))
    add("A1", "  CI low", "A", sp, "lo", lo, csv_row(sp, c, "lo"))
    add("A1", "  CI high", "A", sp, "hi", hi, csv_row(sp, c, "hi"))
add("A1", "  sparse 50% target years", "A", sp, "k", "36", csv_row(sp, {"fraction": "0.5", "contrast": "m1-m0", "metric": "sp", "range": "all48"}, "k"))
add("A1", "  sparse 25% target years", "A", sp, "k", "46", csv_row(sp, {"fraction": "0.25", "contrast": "m1-m0", "metric": "sp", "range": "all48"}, "k"))

# ---------------- R: reaction-norm records (results/rn_paper), within-environment Spearman, model minus additive G+E ----------------
rs, rp = "rn_paper/ge_mean_shift_decomposition.json", "rn_paper/ge_mean_primary_evidence.json"
for cell, est, lo, hi, n, res in (("cv0_strict", "+0.0225", "+0.0151", "+0.0303", "106468", "0.0092"),
                                  ("cohort_matched_forward", "+0.0426", "+0.0228", "+0.0655", "31120", "0.0170"),
                                  ("cv00", "-0.0022", "-0.0148", "+0.0101", "21147", "0.0206"),
                                  ("forward_year", "-0.0386", "-0.0569", "-0.0193", "48364", "0.0136"),
                                  ("forward_year_v2", "+0.0028", "-0.0110", "+0.0166", "48364", "0.0136")):
    base = ("cells", cell, "vs_additive_g_e", "spearman")
    add("R1", f"Kernel-gated RN - additive G+E, {cell}", "MAIN", rs, f"cells.{cell}.vs_additive_g_e.spearman.estimate", est, js(rs, *base, "estimate"))
    add("R1", "  CI low", "MAIN", rs, "ci_low", lo, js(rs, *base, "ci_low"))
    add("R1", "  CI high", "MAIN", rs, "ci_high", hi, js(rs, *base, "ci_high"))
    add("R1", "  N cells", "MAIN", rs, f"cells.{cell}.n_prediction_rows", n, js(rs, "cells", cell, "n_prediction_rows"))
    add("R1", "  resolution 3/sqrt(N)", "MAIN", rs, "3/sqrt(n_prediction_rows)", res,
        lambda root, c=cell: 3 / js(rs, "cells", c, "n_prediction_rows")(root) ** 0.5)
tc = ("contrasts", "germplasm_turnover_in_forward_year", "spearman")
add("R1", "Turnover contrast (forward - cohort-matched)", "MAIN", rs, "contrasts.germplasm_turnover_in_forward_year.spearman", "-0.081", js(rs, *tc, "estimate"))
add("R1", "  CI low", "MAIN", rs, "ci_low", "-0.110", js(rs, *tc, "ci_low"))
add("R1", "  CI high", "MAIN", rs, "ci_high", "-0.053", js(rs, *tc, "ci_high"))
for state, comp, est, lo, hi, n, res in (("observed", "fw_vs_additive", "+0.0205", "+0.0128", "+0.0282", "106468", "0.0092"),
                                         ("preseason", "fw_vs_additive", "+0.0127", "+0.0041", "+0.0214", "81547", "0.0105"),
                                         ("inseason", "fw_vs_additive", "+0.0215", "+0.0130", "+0.0299", "85197", "0.0103"),
                                         ("observed", "kernel_gated_vs_fw", "+0.0020", "-0.0049", "+0.0094", "106468", "0.0092"),
                                         ("preseason", "kernel_gated_vs_fw", "-0.0032", "-0.0137", "+0.0071", "81547", "0.0105"),
                                         ("inseason", "kernel_gated_vs_fw", "+0.0011", "-0.0075", "+0.0103", "85197", "0.0103"),
                                         ("preseason", "kernel_gated_vs_additive", "+0.0096", "+0.0004", "+0.0184", "81547", "0.0105"),
                                         ("inseason", "kernel_gated_vs_additive", "+0.0226", "+0.0138", "+0.0315", "85197", "0.0103")):
    base = ("states", state, comp, "spearman")
    add("R2", f"{comp}, {state} (tested lines, new environments)", "MAIN", rp, f"states.{state}.{comp}.spearman.estimate", est, js(rp, *base, "estimate"))
    add("R2", "  CI low", "MAIN", rp, "ci_low", lo, js(rp, *base, "ci_low"))
    add("R2", "  CI high", "MAIN", rp, "ci_high", hi, js(rp, *base, "ci_high"))
    add("R2", "  N cells", "MAIN", rp, f"states.{state}.n_genotype_environment_units", n, js(rp, "states", state, "n_genotype_environment_units"))
    add("R2", "  resolution", "MAIN", rp, "3/sqrt(N)", res, lambda root, st=state: 3 / js(rp, "states", st, "n_genotype_environment_units")(root) ** 0.5)
for cell, n in (("cv0_strict", "266"), ("cohort_matched_forward", "82"), ("cv00", "258"), ("forward_year", "106")):
    add("R1", f"  environments, {cell}", "MAIN", rs, f"cells.{cell}.n_environments", n, js(rs, "cells", cell, "n_environments"))
for state, n in (("observed", "266"), ("preseason", "204"), ("inseason", "206")):
    add("R2", f"  environments, {state}", "MAIN", rp, f"states.{state}.n_environments", n, js(rp, "states", state, "n_environments"))
ol = "headroom_oldlines/pooled.csv"
c = {"contrast": "ol1-ol0", "metric": "sp", "range": "all"}
for col, v in (("est", "+0.0010"), ("lo", "-0.0031"), ("hi", "+0.0051"), ("resolution", "0.0257")):
    add("A5", f"OL1-OL0 (own history) {col}", "A", ol, f"ol1-ol0,sp,all:{col}", v, csv_row(ol, c, col))
for frac, est, lo, hi in (("0.25", "+0.027", "+0.014", "+0.039"), ("0.5", "+0.067", "+0.037", "+0.096")):
    c = {"fraction": frac, "contrast": "m1-m0", "metric": "sd", "range": "all48"}
    add("A1", f"Sparse {frac}: M×E - main effect, top-10% selection differential (SD units)", "A", sp, "est", est, csv_row(sp, c, "est"))
    add("A1", "  CI low", "A", sp, "lo", lo, csv_row(sp, c, "lo"))
    add("A1", "  CI high", "A", sp, "hi", hi, csv_row(sp, c, "hi"))


def sd_level(frac):
    def r(root):
        d = pd.read_parquet(root / "headroom_sparse/per_env_seed.parquet")
        g = d[d["fraction"].astype(str) == frac].groupby(["dataset", "target"])["sd_m0"].mean()
        return float(g.mean())
    return r


add("A1", "Main-effect GBLUP selection differential level, 25% (mean of year means)", "A", "headroom_sparse/per_env_seed.parquet", "sd_m0", "0.58", sd_level("0.25"))
add("A1", "Main-effect GBLUP selection differential level, 50%", "A", "headroom_sparse/per_env_seed.parquet", "sd_m0", "0.55", sd_level("0.5"))

# ---------------- D: reanalysis of published methods (results/reanalysis) ----------------
rv = "reanalysis/verdict.json"
for m, est, lo, hi, res, n, k, lvl in (("clac", "+0.060", "+0.037", "+0.082", "0.0132", "51666", "5", "0.975"),
                                      ("lc_real", "+0.019", "+0.003", "+0.036", "0.0296", "10283", "1", "0.95"),
                                      ("flgbm_gae", "-0.235", "-0.295", "-0.175", "0.0206", "21266", "2", "0.983")):
    add("D1", f"{m} - cell_reml, forward, within-env Spearman (Holm interval)", "MAIN", rv, f"verdict.{m}.est", est, js(rv, "verdict", m, "est"))
    add("D1", "  CI low", "MAIN", rv, "ci[0]", lo, js(rv, "verdict", m, "ci", 0))
    add("D1", "  CI high", "MAIN", rv, "ci[1]", hi, js(rv, "verdict", m, "ci", 1))
    add("D1", "  resolution", "MAIN", rv, "resolution", res, js(rv, "verdict", m, "resolution"))
    add("D1", "  N cells", "MAIN", rv, "N", n, js(rv, "verdict", m, "N"))
    add("D1", "  target years", "MAIN", rv, "k", k, js(rv, "verdict", m, "k"))
    add("D1", "  Holm level", "MAIN", rv, "holm_level", lvl, js(rv, "verdict", m, "holm_level"))


# ---------------- D3: CLAC ablations (results/clac_decomp, pre-registered 2026-10-02) ----------------
dp, dv = "clac_decomp/pooled.csv", "clac_decomp/verdict.json"
for con, est, lo, hi in (("A_full_vs_cell_reml", "+0.053", "+0.024", "+0.082"), ("D4_linear_kernel_vs_cell_reml", "+0.029", "+0.006", "+0.052")):
    add("D3", f"CLAC ablations: {con} (95%, descriptive)", "MAIN", dp, "est", est, csv_row(dp, {"contrast": con}, "est"))
    add("D3", "  CI low", "MAIN", dp, "lo", lo, csv_row(dp, {"contrast": con}, "lo"))
    add("D3", "  CI high", "MAIN", dp, "hi", hi, csv_row(dp, {"contrast": con}, "hi"))
for key, est, lo, hi, lev in (("loss_D4_linear_kernel", "+0.023", "+0.002", "+0.043", "0.99"), ("loss_D5_single_main_effect", "+0.032", "-0.020", "+0.085", "0.9875"),
                              ("loss_D1_equal_weights", "-0.001", "-0.017", "+0.015", "0.95")):
    add("D3", f"CLAC ablations: {key} (Holm interval)", "MAIN", dv, f"verdict.{key}.est", est, js(dv, "verdict", key, "est"))
    add("D3", "  Holm CI low", "MAIN", dv, "ci[0]", lo, js(dv, "verdict", key, "ci", 0))
    add("D3", "  Holm CI high", "MAIN", dv, "ci[1]", hi, js(dv, "verdict", key, "ci", 1))
    add("D3", "  Holm level", "MAIN", dv, "holm_level", lev, js(dv, "verdict", key, "holm_level"))
add("D3", "  ablations: resolution", "MAIN", dv, "resolution", "0.0130", js(dv, "verdict", "loss_D4_linear_kernel", "resolution"))
add("D3", "  ablations: scored cells", "MAIN", dv, "N", "53638", js(dv, "verdict", "loss_D4_linear_kernel", "N"))
add("D3", "  kernel loss: target years with a loss (of 5)", "MAIN", dp, "years_positive", "5", csv_row(dp, {"contrast": "loss_D4_linear_kernel"}, "years_positive"))


def decomp_stat(fn):
    def r(root):
        return fn(pd.read_csv(root / "clac_decomp/year_effects.csv"), pd.read_csv(root / "clac_decomp/pooled.csv"),
                  pd.read_csv(root / "clac_decomp/spatial_adjustment_counts.csv"), json.load(open(root / "clac_decomp/verdict.json"))["verdict"])
    return r


d5 = lambda y: y[y["contrast"] == "loss_D5_single_main_effect"]["d"]
add("D3", "  single-main-effect loss: smallest year effect", "MAIN", "clac_decomp/year_effects.csv", "min d", "-0.017", decomp_stat(lambda y, p, s, v: d5(y).min()))
add("D3", "  single-main-effect loss: largest year effect", "MAIN", "clac_decomp/year_effects.csv", "max d", "+0.103", decomp_stat(lambda y, p, s, v: d5(y).max()))
add("D3", "  no-spatial and no-cleaning losses: largest |estimate| (< 0.001)", "MAIN", dv, "max |est|", "0.0005",
    decomp_stat(lambda y, p, s, v: max(abs(v[k]["est"]) for k in ("loss_D2_no_spatial", "loss_D3_no_cleaning"))))
add("D3", "  no-spatial and no-cleaning losses: widest Holm limit (within 0.005)", "MAIN", dv, "max |ci|", "0.0047",
    decomp_stat(lambda y, p, s, v: max(abs(c) for k in ("loss_D2_no_spatial", "loss_D3_no_cleaning") for c in v[k]["ci"])))
add("D3", "  spatial adjustment applied: smallest share of training environments (%)", "MAIN", "clac_decomp/spatial_adjustment_counts.csv", "min", "42",
    decomp_stat(lambda y, p, s, v: 100 * (s["adjustment_successful"] / s["training_environments"]).min()))
add("D3", "  spatial adjustment applied: largest share (%)", "MAIN", "clac_decomp/spatial_adjustment_counts.csv", "max", "69",
    decomp_stat(lambda y, p, s, v: 100 * (s["adjustment_successful"] / s["training_environments"]).max()))
add("D3", "  kernel loss / resolution (Table 4)", "MAIN", dv, "est/resolution", "1.75",
    decomp_stat(lambda y, p, s, v: v["loss_D4_linear_kernel"]["est"] / v["loss_D4_linear_kernel"]["resolution"]))


# ---------------- D4: kernel transfer test (results/transfer_arc, pre-registered 2026-10-02) ----------------
tp = "transfer_arc/pooled.csv"
for rng_, est, lo, hi, res in (("all48", "+0.004", "-0.000", "+0.007", "0.0087"), ("G2F", "+0.034", "+0.021", "+0.047", "0.0130"),
                               ("NUST", "+0.004", "-0.001", "+0.009", None), ("URSN", "+0.003", "-0.008", "+0.014", None),
                               ("ESWYT", "-0.000", "-0.006", "+0.005", None), ("GEM_IA", "-0.006", "-0.032", "+0.021", None),
                               ("MU_SOY", "+0.006", "-0.010", "+0.021", None)):
    c = {"range": rng_, "metric": "spearman", "scope": "all"}
    add("D4", f"Kernel transfer: cell_arc - cell_reml, {rng_}", "MAIN", tp, "est", est, csv_row(tp, c, "est"))
    add("D4", "  CI low", "MAIN", tp, "lo", lo, csv_row(tp, c, "lo"))
    add("D4", "  CI high", "MAIN", tp, "hi", hi, csv_row(tp, c, "hi"))
    if res:
        add("D4", "  resolution", "MAIN", tp, "resolution", res, csv_row(tp, c, "resolution"))
c = {"range": "G2F", "metric": "sel_diff", "scope": "all"}
for col, v in (("est", "+0.083"), ("lo", "+0.049"), ("hi", "+0.116")):
    add("D4", f"  G2F selection differential {col}", "MAIN", tp, col, v, csv_row(tp, c, col))
add("D4", "  G2F target years with a gain (of 5)", "MAIN", tp, "years_positive", "5", csv_row(tp, {"range": "G2F", "metric": "spearman", "scope": "all"}, "years_positive"))
add("D4", "  target years pooled", "MAIN", tp, "k", "48", csv_row(tp, {"range": "all48", "metric": "spearman", "scope": "all"}, "k"))
add("D4", "  G2F kernel gain / resolution (Table 4)", "MAIN", tp, "est/resolution", "2.64",
    lambda root: csv_row(tp, {"range": "G2F", "metric": "spearman", "scope": "all"}, "est")(root) / csv_row(tp, {"range": "G2F", "metric": "spearman", "scope": "all"}, "resolution")(root))
add("D4", "  control agrees with the benchmark (worst within-environment Spearman)", "MAIN", "transfer_arc/consistency.json", "worst_min", "1.000", js("transfer_arc/consistency.json", "worst_min"))


add("C7", "Second G2F analysis: SNPs with MAF >= 0.05 (of the 2,425)", "C", "amax_runs/A/final/R9_A_results.json", "genotype.n_snp_maf05", "2355",
    lambda root: float(json.dumps(json.load(open(root / "amax_runs/A/final/R9_A_results.json"))).split('"n_snp_maf05": ')[1].split("}")[0].split(",")[0]))


# ---------------- D5: sensitivity checks after the independent recomputation (results/floor_sensitivity) ----------------
def floor_tab(fn):
    return lambda root: fn(pd.read_csv(root / "floor_sensitivity/table.csv"))


add("D5", "SE floor sensitivity: contrasts examined", "MAIN", "floor_sensitivity/table.csv", "rows", "31", floor_tab(len))
add("D5", "  verdicts changed", "MAIN", "floor_sensitivity/table.csv", "verdict_changed", "1", floor_tab(lambda T: T["verdict_changed"].sum()))
add("D5", "  largest change of an estimate", "MAIN", "floor_sensitivity/table.csv", "abs_change_est", "0.007", floor_tab(lambda T: T["abs_change_est"].max()))
six = "floor_sensitivity/ablations_without_six_environments.json"
for col, v in (("est", "+0.024"), ("lo", "+0.005"), ("hi", "+0.043")):
    add("D5", f"Ablations without six environments: loss linear kernel {col}", "MAIN", six, col, v, js(six, "contrasts", "loss_D4_linear_kernel", col))
add("D5", "  cells in the six environments (53,638 - 51,666)", "MAIN", six, "cells", "1972", lambda root: 53638 - json.load(open(root / six))["cells"])
add("D5", "  loss equal weights: upper Holm limit (just above the resolution)", "MAIN", dv, "ci[1]", "0.015", js(dv, "verdict", "loss_D1_equal_weights", "ci", 1))
add("D5", "  kernel transfer, all 48 years: estimate / resolution (less than half)", "MAIN", tp, "est/resolution", "0.42",
    lambda root: csv_row(tp, {"range": "all48", "metric": "spearman", "scope": "all"}, "est")(root) / csv_row(tp, {"range": "all48", "metric": "spearman", "scope": "all"}, "resolution")(root))


def clac_years_positive(root):
    d = pd.read_csv(root / "reanalysis/year_effects.csv")
    d = d[(d["method"] == "clac") & (d["metric"] == "sp")]
    return float((d["d"] > 0).sum())


add("D1", "CLAC target years with positive year effect (of 5)", "MAIN", "reanalysis/year_effects.csv", "method=clac, metric=sp, d>0", "5", clac_years_positive)
ck = "reanalysis/checks.json"
add("D2", "CLAC fidelity 2022: cell-wise Spearman vs original submission", "MAIN", ck, "clac_fidelity...cellwise_spearman", "0.955", js(ck, "clac_fidelity_2022_vs_CLACsubmission5", "cellwise_spearman"))
add("D2", "CLAC fidelity 2022: median within-environment agreement", "MAIN", ck, "within_env_spearman_median", "0.78", js(ck, "clac_fidelity_2022_vs_CLACsubmission5", "within_env_spearman_median"))
add("D2", "CLAC leakage check: max abs difference with junk yields", "MAIN", ck, "clac_leakage_2016...junk", "0", js(ck, "clac_leakage_2016", "original_vs_junk_yield_ge2016_max_abs_diff"))
add("D2", "Fernandes CV0 fidelity: our pooled Pearson (all threads)", "MAIN", ck, "ours_v1_all_threads", "0.327", js(ck, "fernandes_cv0_fidelity", "ours_v1_all_threads"))
add("D2", "Fernandes CV0 fidelity: published", "MAIN", ck, "paper_GA_E_pooled_pearson", "0.45", js(ck, "fernandes_cv0_fidelity", "paper_GA_E_pooled_pearson"))
c = {"method": "clac", "metric": "sp_new"}
add("D2", "CLAC - cell_reml, new hybrids only (95%)", "MAIN", "reanalysis/pooled.csv", "clac,sp_new:est", "+0.066", csv_row("reanalysis/pooled.csv", c, "est"))
add("D2", "  CI low", "MAIN", "reanalysis/pooled.csv", "lo", "+0.037", csv_row("reanalysis/pooled.csv", c, "lo"))
add("D2", "  CI high", "MAIN", "reanalysis/pooled.csv", "hi", "+0.095", csv_row("reanalysis/pooled.csv", c, "hi"))


def mg98_minus_2k(root):
    ext = pd.concat([pd.read_csv(root / "amax_runs/trackB/out/EXT_selected_per_env.csv"),
                     pd.read_csv(root / "amax_runs/trackB/out10/G2024_selected_per_env.csv")])
    ext = ext[(ext["subset"] == "all") & (ext["protocol"] == "P3")]
    w = ext.pivot_table(index="Env", columns="model", values="pearson")
    return float((w["MG_98k"] - w["MG_2k"]).dropna().mean())


add("C7", "98k-marker minus 2k-marker ridge, external years (unweighted mean over environments, within-env Pearson)", "C",
    "amax_runs/trackB/*_selected_per_env.csv", "MG_98k - MG_2k, P3", "-0.005", mg98_minus_2k)

def decimals(s):
    s = s.split()[0].lstrip("<=+ ")
    return len(s.split(".")[1]) if "." in s else 0


def main():
    out, bad = [], 0
    for i, claim, src, path, loc, printed, reader in rows:
        root = ROOTS[src]
        if reader is None:
            status, got = "UNCHECKED", ""
        elif root is None or not root.exists():
            status, got = "SKIP", ""
        else:
            try:
                v = reader(root)
                got = f"{v:.6g}"
                status = "PASS" if round(v, decimals(printed)) == round(float(printed.lstrip("+")), decimals(printed)) else "FAIL"
            except Exception as e:  # noqa: BLE001 - report, do not hide
                status, got = "ERROR", f"{type(e).__name__}: {e}"
        bad += status in ("FAIL", "ERROR")
        out.append((i, claim, src, path, loc, printed, got, status))
    if "--md" in sys.argv:
        print("| id | claim | src | file | locator | printed | file value | status |\n|---|---|---|---|---|---|---|---|")
        for r in out:
            print("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    else:
        for r in out:
            print(f"{r[7]:9s} {r[0]:3s} {r[2]:4s} {r[1][:70]:70s} printed {r[5]:>10s} file {r[6]}")
    n = {s: sum(r[7] == s for r in out) for s in ("PASS", "FAIL", "ERROR", "SKIP", "UNCHECKED")}
    print(f"\n{n}", file=sys.stderr)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
