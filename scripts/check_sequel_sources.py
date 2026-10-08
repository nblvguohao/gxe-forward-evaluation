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
# (earlier-draft-only rows M7, M8 removed 2026-10-08: not printed in the TCJ manuscript; their source files are not in the public release.)
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


# (earlier-draft-only rows M10 (GE-BiFormer levels) removed 2026-10-08: not printed in the TCJ manuscript; their source files are not in the public release.)


def gef_res(sc):
    def r(root):
        d = pd.read_csv(root / "e1/geformer_rerun_vs_gate2.csv")
        return 3 / float(d[d["scenario"] == sc]["cells"].iloc[0]) ** 0.5
    return r


add("M10", "GEFormer resolution F2024m (3/sqrt cells)", "MAIN", "e1/geformer_rerun_vs_gate2.csv", "cells", "0.031", gef_res("F2024m"))
add("M10", "GEFormer resolution F2022m", "MAIN", "e1/geformer_rerun_vs_gate2.csv", "cells", "0.028", gef_res("F2022m"))
# (earlier-draft-only row M10 (phi) removed 2026-10-08: not printed in the TCJ manuscript; their source files are not in the public release.)
for item, est, lo, hi in (("reml", "-0.012", "-0.019", "-0.006"), ("gbm", "-0.041", "-0.058", "-0.024"), ("mlp", "-0.089", "-0.110", "-0.067"),
                          ("R_CV", "-0.010", "-0.019", "-0.001")):
    c = {"item": item, "metric": "spearman", "scope": "all48"}
    add("M9", f"48-year summary: {item} - cell_reml", "MAIN", s48, "est", est, csv_row(s48, c, "est"))
    add("M9", "  CI low", "MAIN", s48, "lo", lo, csv_row(s48, c, "lo"))
    add("M9", "  CI high", "MAIN", s48, "hi", hi, csv_row(s48, c, "hi"))


# (earlier-draft-only rows A7 (pick counts) removed 2026-10-08: not printed in the TCJ manuscript; their source files are not in the public release.)
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
add("D3", "  kernel loss / resolution (Table S12)", "MAIN", dv, "est/resolution", "1.75",
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
add("D4", "  G2F kernel gain / resolution (Table S12)", "MAIN", tp, "est/resolution", "2.64",
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


# ---------------- D6: environment definitions of NUST and URSN (post hoc, results/env_definition_check) ----------------
ST, PV, SV, MF, U12 = ("env_definition_check/" + f for f in ("structure.json", "pooled_by_variant.csv", "sparse_by_variant.csv",
                                                              "marker_free_reference.csv", "ursn_min12_pooled.csv"))
for key, v in (("nust_envs_ge2_trials", "493"), ("nust_scored_envs", "499"), ("nust_median_trials", "4"), ("ursn_envs_worst_is_check", "47"),
               ("ursn_scored_envs", "58"), ("ursn_envs_ge12", "53")):
    add("D6", f"Structure: {key}", "MAIN", ST, key, v, js(ST, key))
add("D6", "Structure: NUST trial share of within-env variance (%)", "MAIN", ST, "x100", "28", lambda root: 100 * js(ST, "nust_trial_share_median")(root))
add("D6", "Structure: URSN check share (%)", "MAIN", ST, "x100", "29", lambda root: 100 * js(ST, "ursn_check_share_mean")(root))
for var, item, est, lo, hi in (("NUST:detrend / URSN:orig", "reml", "-0.005", "-0.011", "+0.002"), ("NUST:detrend / URSN:nocheck_n3", "reml", "-0.005", "-0.011", "+0.002"),
                               ("NUST:trial_env / URSN:nocheck_n3", "reml", "-0.011", "-0.018", "-0.004"),
                               ("NUST:detrend / URSN:orig", "gxe_gbm", "-0.037", "-0.072", "-0.001"), ("NUST:detrend / URSN:nocheck_n3", "gxe_gbm", "-0.036", "-0.079", "+0.007"),
                               ("NUST:trial_env / URSN:nocheck_n3", "gxe_gbm", "-0.037", "-0.080", "+0.006"),
                               ("NUST:detrend / URSN:orig", "dl_g", "-0.006", "-0.015", "+0.003"), ("NUST:detrend / URSN:nocheck_n3", "dl_g", "-0.006", "-0.017", "+0.005"),
                               ("NUST:trial_env / URSN:nocheck_n3", "dl_g", "-0.012", "-0.027", "+0.002")):
    c = {"variant": var, "item": item}
    for col, val in (("est", est), ("lo", lo), ("hi", hi)):
        add("D6", f"Table S11 {var} {item} {col}", "MAIN", PV, col, val, csv_row(PV, c, col))
for item, est, lo, hi in (("reml", "-0.012", "-0.019", "-0.005"), ("gxe_gbm", "-0.034", "-0.070", "+0.002"), ("dl_g", "-0.014", "-0.026", "-0.001")):
    for col, val in (("est", est), ("lo", lo), ("hi", hi)):
        add("D6", f"Table S11 URSN>=12 {item} {col}", "MAIN", U12, col, val, csv_row(U12, {"item": item}, col))
for f, nv, uv, est, lo, hi in ((0.25, "detrend", "orig", "+0.013", "+0.009", "+0.018"), (0.25, "detrend", "nocheck", "+0.015", "+0.010", "+0.019"),
                               (0.25, "trial_env", "nocheck", "+0.016", "+0.011", "+0.021"), (0.5, "detrend", "orig", "+0.025", "+0.018", "+0.032"),
                               (0.5, "detrend", "nocheck", "+0.025", "+0.018", "+0.033"), (0.5, "trial_env", "nocheck", "+0.022", "+0.013", "+0.031")):
    c = {"fraction": f, "NUST": nv, "URSN": uv}
    for col, val in (("est", est), ("lo", lo), ("hi", hi)):
        add("D6", f"Table S11 sparse {f} {nv}/{uv} {col}", "MAIN", SV, col, val, csv_row(SV, c, col))
add("D6", "Sparse 25%: smallest post hoc estimate (+0.0135)", "MAIN", SV, "est", "+0.0135", csv_row(SV, {"fraction": 0.25, "NUST": "detrend", "URSN": "orig"}, "est"))
add("D6", "Sparse 25%: largest post hoc estimate (+0.0156)", "MAIN", SV, "est", "+0.0156", csv_row(SV, {"fraction": 0.25, "NUST": "trial_env", "URSN": "nocheck"}, "est"))
for nv, v in (("detrend", "+0.012"), ("trial_env", "+0.014")):
    add("D6", f"NUST alone sparse 25% {nv}", "MAIN", SV, "NUST_alone_est", v, csv_row(SV, {"fraction": 0.25, "NUST": nv, "URSN": "nocheck"}, "NUST_alone_est"))
NU = {"dataset": "NUST", "reference": "location x maturity-group history"}
for col, v in (("reference_mean", "0.257"), ("cell_reml_mean", "0.343")):
    add("D6", f"Marker-free NUST {col}", "MAIN", MF, col, v, csv_row(MF, NU, col))
lib10 = ["reml", "reml_x0.1", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp"]
add("D6", "Marker-free NUST: lowest two-stage mean", "MAIN", MF, "min", "0.209", lambda root: min(csv_row(MF, NU, m + "_mean")(root) for m in lib10))
add("D6", "Marker-free NUST: highest two-stage mean", "MAIN", MF, "max", "0.302", lambda root: max(csv_row(MF, NU, m + "_mean")(root) for m in lib10))
add("D6", "Marker-free NUST: two-stage methods below the reference", "MAIN", MF, "count", "4",
    lambda root: sum(csv_row(MF, NU, m + "_mean")(root) < csv_row(MF, NU, "reference_mean")(root) for m in lib10))
UR = {"dataset": "URSN", "reference": "check history (baseline)"}
for col, v in (("reference_mean", "0.440"), ("cell_reml_mean", "0.480")):
    add("D6", f"Marker-free URSN {col}", "MAIN", MF, col, v, csv_row(MF, UR, col))
add("D6", "Marker-free URSN: lowest under other tie placements", "MAIN", MF, "reference_mean", "-0.076", csv_row(MF, {"dataset": "URSN", "reference": "check history (baseline_below)"}, "reference_mean"))
add("D6", "Marker-free URSN: lowest two-stage mean", "MAIN", MF, "min", "0.261", lambda root: min(csv_row(MF, UR, m + "_mean")(root) for m in lib10))
add("D6", "Marker-free URSN: highest two-stage mean", "MAIN", MF, "max", "0.493", lambda root: max(csv_row(MF, UR, m + "_mean")(root) for m in lib10))


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


# ---------------- D7: TCJ v9 (2026-10-08): basis of 3/sqrt(N) (Section 2.1, Note S2, Table S13) and Table S12 ratios ----------------
RN = "resolution_note/by_dataset.csv"
rn = lambda root: pd.read_csv(root / RN)
rn6 = lambda root: rn(root)[rn(root)["dataset"] != "All six datasets"]
add("D7", "Unequal environment sizes: largest ratio over the six datasets ('up to 1.37 times')", "MAIN", RN, "max ratio_unequal_sizes", "1.37",
    lambda root: rn6(root)["ratio_unequal_sizes"].max())
add("D7", "  smallest ratio (1.00, ESWYT)", "MAIN", RN, "min ratio_unequal_sizes", "1.00", lambda root: rn6(root)["ratio_unequal_sizes"].min())
add("D7", "Fieller-Hartley-Pearson factor (1-rho^2)sqrt(1.060) at the mean correlation: smallest over datasets", "MAIN", RN, "min fhp_factor", "0.79",
    lambda root: rn6(root)["fhp_factor"].min())
add("D7", "  largest", "MAIN", RN, "max fhp_factor", "1.02", lambda root: rn6(root)["fhp_factor"].max())
add("D7", "SD of one environment's Spearman / (1/sqrt n): largest deviation from 1 over all environments ('within about 20 %')", "MAIN", RN,
    "max |sd_over_inv_sqrt_n - 1|", "0.16", lambda root: max(1 - rn6(root)["sd_over_inv_sqrt_n_min"].min(), rn6(root)["sd_over_inv_sqrt_n_max"].max() - 1))
add("D7", "  mean cell_reml within-environment Spearman: smallest (ESWYT)", "MAIN", RN, "min mean_rho_cell_reml", "0.08", lambda root: rn6(root)["mean_rho_cell_reml"].min())
add("D7", "  mean cell_reml within-environment Spearman: largest (URSN)", "MAIN", RN, "max mean_rho_cell_reml", "0.48", lambda root: rn6(root)["mean_rho_cell_reml"].max())
IS = "resolution_note/implied_se.csv"
add("D7", "Implied SE / (1/sqrt N) over the 31 pooled contrasts: smallest", "MAIN", IS, "min", "0.02", lambda root: pd.read_csv(root / IS)["se_over_scale"].min())
add("D7", "  median", "MAIN", IS, "median", "1.05", lambda root: pd.read_csv(root / IS)["se_over_scale"].median())
add("D7", "  largest", "MAIN", IS, "max", "4.9", lambda root: pd.read_csv(root / IS)["se_over_scale"].max())
FT = "floor_sensitivity/table.csv"
for lab, contrast, v in (("CLAC - cell_reml", "CLAC − cell-level GBLUP", "4.51"), ("Sparse 25 %", "Sparse 25 %: M×E − main-effect GBLUP", "1.81"),
                         ("Sparse 50 %", "Sparse 50 %: M×E − main-effect GBLUP", "2.49")):
    add("D7", f"Table S12: {lab}, estimate / resolution", "MAIN", FT, "est_plan/resolution", v,
        lambda root, c=contrast: csv_row(FT, {"contrast": c}, "est_plan")(root) / csv_row(FT, {"contrast": c}, "resolution")(root))
for state, v in (("observed", "2.23"), ("inseason", "2.09"), ("preseason", "1.21")):
    add("D7", f"Table S12: FW vs additive, {state}, estimate / resolution", "MAIN", rp, "estimate/(3/sqrt N)", v,
        lambda root, st=state: js(rp, "states", st, "fw_vs_additive", "spearman", "estimate")(root)
        / (3 / js(rp, "states", st, "n_genotype_environment_units")(root) ** 0.5))
add("D7", "Table S12: GEFormer 2022, estimate / resolution", "MAIN", g22, "delta/(3/sqrt cells)", "1.83",
    lambda root: csv_row(g22, {"metric": "spearman", "method": "GEFormer_fair"}, "delta")(root) / gef_res("F2022m")(root))


# ---------------- D8: TCJ v10 revision (2026-10-08): Hartung-Knapp, per-dataset results, rescoring details ----------------
HK, PD, WO, ES, CN = ("revision_tcj/" + f for f in ("hk_sensitivity.csv", "per_dataset.csv", "without_eswyt_ursn.csv", "env_shrink.csv", "clac_new_hybrids.json"))
for c, col, v in (("CLAC ablation, loss: linear kernel", "hk_lo", "-0.018"), ("CLAC ablation, loss: linear kernel", "hk_hi", "+0.063"),
                  ("Kernel transfer: G2F", "hk_lo", "+0.016"), ("Kernel transfer: G2F", "hk_hi", "+0.052")):
    add("D8", f"HK {c} {col}", "MAIN", HK, col, v, csv_row(HK, {"contrast": c}, col))
add("D8", "HK: verdicts changed (two)", "MAIN", HK, "verdict_changed", "2", lambda root: float(pd.read_csv(root / HK)["verdict_changed"].sum()))
add("D8", "HK: covariate LightGBM tied", "MAIN", HK, "hk_hi>0", "1",
    lambda root: float(csv_row(HK, {"contrast": "Benchmark: Covariate LightGBM − cell-level GBLUP"}, "hk_hi")(root) > 0))
for col, v in (("est", "-0.0361"), ("lo", "-0.0717"), ("hi", "-0.0004")):
    add("D8", f"Covariate LightGBM 4 decimals {col}", "MAIN", "floor_sensitivity/table.csv", col + "_plan", v,
        csv_row("floor_sensitivity/table.csv", {"contrast": "Benchmark: Covariate LightGBM − cell-level GBLUP"}, col + "_plan"))
for d, v in (("G2F", "0.0151"), ("NUST", "0.0204")):
    add("D8", f"Sparse 25% own resolution {d}", "MAIN", PD, "resolution", v, csv_row(PD, {"analysis": "Sparse 25 %: M×E − main-effect GBLUP", "dataset": d}, "resolution"))
for f, v in (("25", "+0.029"), ("50", "+0.055")):
    add("D8", f"Sparse {f}% without ESWYT and URSN", "MAIN", WO, "est", v, csv_row(WO, {"analysis": f"Sparse {f} %: M×E − main-effect GBLUP"}, "est"))
SG = {"analysis": "Benchmark: Stacking on the forward history − cell-level GBLUP", "dataset": "G2F"}
for col, v in (("est", "+0.037"), ("lo", "+0.006"), ("hi", "+0.068"), ("resolution", "0.0130")):
    add("D8", f"G2F stacking {col}", "MAIN", PD, col, v, csv_row(PD, SG, col))
for col, v in (("est", "-0.075"), ("lo", "-0.141"), ("hi", "-0.009")):
    add("D8", f"GEM_IA stacking {col}", "MAIN", PD, col, v, csv_row(PD, {"analysis": "Benchmark: Stacking on the forward history − cell-level GBLUP", "dataset": "GEM_IA"}, col))
add("D8", "Stacking: datasets with a resolved gain (G2F only)", "MAIN", PD, "verdict", "1",
    lambda root: float((pd.read_csv(root / PD).query("analysis == 'Benchmark: Stacking on the forward history − cell-level GBLUP'")["verdict"] == "resolved gain").sum()))
lib_names = ["Two-stage GBLUP", "Two-stage ridge, shrinkage x0.1", "Two-stage ridge, shrinkage x10", "Two-stage ridge, shrinkage x100", "Ridge on 20 PCs",
             "Random forest", "Gradient boosting", "kNN, k = 10", "kNN, k = 30", "Multilayer perceptron"]
add("D8", "G2F: two-stage methods tied with cell_reml (six of ten)", "MAIN", PD, "verdict", "6",
    lambda root: float(sum(pd.read_csv(root / PD).set_index(["analysis", "dataset"]).loc[(f"Benchmark: {m} − cell-level GBLUP", "G2F"), "verdict"] == "tied" for m in lib_names)))
def sparse_drop(fr, fn):
    def r(root):
        sv = pd.read_csv(root / "env_definition_check/sparse_by_variant.csv")
        sv = sv[sv["fraction"] == fr]
        o = float(sv[(sv["NUST"] == "orig") & (sv["URSN"] == "orig")]["est"].iloc[0])
        return fn(100 * (1 - sv[sv["NUST"] != "orig"]["est"] / o))
    return r
add("D8", "Sparse: post hoc pooled gains smaller by at least (%; 25 %)", "MAIN", "env_definition_check/sparse_by_variant.csv", "min", "16", sparse_drop(0.25, min))
add("D8", "Sparse: post hoc pooled gains smaller by at most (%; 50 %)", "MAIN", "env_definition_check/sparse_by_variant.csv", "max", "38", sparse_drop(0.5, max))
def shr(v, fn, two=False):
    def r(root):
        e = pd.read_csv(root / ES)
        e = e[e["variant"] == v]
        return float(e[e["item"] == "reml"]["shrink_pct"].iloc[0]) if two else float(fn(e[e["item"] != "reml"]["shrink_pct"]))
    return r


add("D8", "Shrink, NUST detrend: nine methods min (%)", "MAIN", ES, "min", "14", shr("NUST:detrend / URSN:orig", min))
add("D8", "Shrink, NUST detrend: nine methods max (%)", "MAIN", ES, "max", "47", shr("NUST:detrend / URSN:orig", max))
add("D8", "Shrink, NUST detrend + URSN nocheck n3: nine methods min (%)", "MAIN", ES, "min", "33", shr("NUST:detrend / URSN:nocheck_n3", min))
add("D8", "Shrink, NUST detrend + URSN nocheck n3: nine methods max (%)", "MAIN", ES, "max", "44", shr("NUST:detrend / URSN:nocheck_n3", max))
add("D8", "Shrink, two-stage GBLUP (%)", "MAIN", ES, "reml", "63", shr("NUST:detrend / URSN:nocheck_n3", None, True))
for col, v in (("est", "-0.011"), ("lo", "-0.023"), ("hi", "-0.000")):
    add("D8", f"Network, NUST trial_env / URSN orig {col}", "MAIN", "env_definition_check/pooled_by_variant.csv", col, v,
        csv_row("env_definition_check/pooled_by_variant.csv", {"variant": "NUST:trial_env / URSN:orig", "item": "dl_g"}, col))
YS = "headroom_sparse/year_effects.csv"
zs = lambda f: (lambda root: float(((lambda y: (y["fraction"] == f) & (y["contrast"] == "m1-m0") & (y["metric"] == "sp") & (y["se"] <= 0))(pd.read_csv(root / YS))).sum()))
add("D8", "Zero-SE sparse years, 25 %", "MAIN", YS, "se==0", "8", zs(0.25))
add("D8", "Zero-SE sparse years, 50 %", "MAIN", YS, "se==0", "3", zs(0.5))
for c, v in (("Sparse 25 %: M×E − main-effect GBLUP", "0.016"), ("Sparse 50 %: M×E − main-effect GBLUP", "0.029")):
    add("D8", f"Floor on zero SEs only: {c}", "MAIN", "floor_sensitivity/table.csv", "est_zero_only", v, csv_row("floor_sensitivity/table.csv", {"contrast": c}, "est_zero_only"))
add("D8", "CLAC new hybrids: cells", "MAIN", CN, "new_hybrid_cells", "46627", js(CN, "new_hybrid_cells"))
add("D8", "CLAC new hybrids: resolution", "MAIN", CN, "resolution", "0.0139", js(CN, "resolution"))
add("D7", "SD / (1/sqrt n) at dataset mean rho: min over environments", "MAIN", RN, "min", "0.84", lambda root: rn6(root)["sd_over_inv_sqrt_n_min"].min())
add("D7", "SD / (1/sqrt n) at dataset mean rho: max over environments", "MAIN", RN, "max", "1.07", lambda root: rn6(root)["sd_over_inv_sqrt_n_max"].max())
add("D7", "SD / (1/sqrt n) with own rho: share of environments outside +-20 % (%)", "MAIN", "resolution_note/facts.json", "x100", "14",
    lambda root: 100 * js("resolution_note/facts.json", "own_rho_per_environment", "share_outside_20pct")(root))


# ---------------- D9: TCJ v12 (2026-10-08): second-round check ----------------
SVF = "env_definition_check/sparse_by_variant.csv"
nus = lambda f, nv, col: csv_row(SVF, {"fraction": f, "NUST": nv, "URSN": "nocheck"}, col)
add("D9", "NUST alone sparse 25 %, post hoc: smallest (+0.012)", "MAIN", SVF, "NUST_alone_est", "+0.012", nus(0.25, "detrend", "NUST_alone_est"))
add("D9", "NUST alone sparse 25 %, post hoc: largest (+0.014)", "MAIN", SVF, "NUST_alone_est", "+0.014", nus(0.25, "trial_env", "NUST_alone_est"))
for col, v in (("NUST_alone_est", "+0.031"), ("NUST_alone_lo", "+0.019"), ("NUST_alone_hi", "+0.043")):
    add("D9", f"NUST alone sparse 50 %, centred within trial {col}", "MAIN", SVF, col, v, nus(0.5, "detrend", col))
for col, v in (("NUST_alone_est", "+0.022"), ("NUST_alone_lo", "+0.001"), ("NUST_alone_hi", "+0.043")):
    add("D9", f"NUST alone sparse 50 %, trial units {col}", "MAIN", SVF, col, v, nus(0.5, "trial_env", col))
add("D9", "NUST own resolution, sparse 50 %", "MAIN", PD, "resolution", "0.0290", csv_row(PD, {"analysis": "Sparse 50 %: M×E − main-effect GBLUP", "dataset": "NUST"}, "resolution"))
add("D9", "Table 2: post hoc pooled sparse 25 % lower end (+0.013)", "MAIN", SVF, "est", "+0.013", csv_row(SVF, {"fraction": 0.25, "NUST": "detrend", "URSN": "orig"}, "est"))
YR = "reanalysis/year_effects.csv"
cy = lambda fn: (lambda root: float(fn(pd.read_csv(root / YR).query("method == 'clac' and metric == 'sp'")["d"])))
add("D9", "CLAC per year: smallest", "MAIN", YR, "min d", "+0.031", cy(min))
add("D9", "CLAC per year: largest", "MAIN", YR, "max d", "+0.080", cy(max))
FV = "revision_tcj/floor_variants.csv"
for f, col, v in ((0.25, "est_no_floor_zero_dropped", "0.019"), (0.5, "est_no_floor_zero_dropped", "0.034")):
    add("D9", f"No floor, zero-SE years dropped, sparse {f}", "MAIN", FV, col, v, csv_row(FV, {"fraction": f}, col))
for f, v in ((0.25, "14"), (0.5, "17")):
    add("D9", f"Zero-only floor: sparse {f} smaller by (%)", "MAIN", FV, "pct", v,
        lambda root, f=f: 100 * (1 - csv_row(FV, {"fraction": f}, "est_zero_floor")(root) / csv_row(FV, {"fraction": f}, "est_plan_floor")(root)))
KW = "revision_tcj/kernel_without_eswyt_ursn.csv"
for col, v in (("est", "+0.0064"), ("lo", "+0.0005"), ("hi", "+0.0123"), ("resolution", "0.0097")):
    add("D9", f"Kernel transfer without ESWYT and URSN {col}", "MAIN", KW, col, v, csv_row(KW, {"analysis": "Kernel transfer: cell_arc − cell_reml"}, col))
WN = "revision_tcj/without_nust.csv"
wn = lambda m: {"analysis": f"Benchmark: {m} − cell-level GBLUP"}
for m in ("Two-stage GBLUP", "Random forest"):
    add("D9", f"Without NUST, {m}: loss shrinks by more than half", "MAIN", WN, "ratio<0.5", "1",
        lambda root, m=m: float(abs(csv_row(WN, wn(m), "est_without_nust")(root)) < 0.5 * abs(csv_row(WN, wn(m), "est_all")(root))))
    add("D9", f"Without NUST, {m}: tied", "MAIN", WN, "verdict", "1", lambda root, m=m: float(pd.read_csv(root / WN).set_index("analysis").loc[wn(m)["analysis"], "verdict"] == "tied"))
for m in ("Ridge on 20 PCs", "kNN, k = 30"):
    add("D9", f"Without NUST, {m}: loss does not shrink", "MAIN", WN, "smaller", "0", lambda root, m=m: float(pd.read_csv(root / WN).set_index("analysis").loc[wn(m)["analysis"], "loss_smaller_without_nust"]))
add("D9", "MLP: resolved loss in all six datasets", "MAIN", PD, "verdict", "6",
    lambda root: float((pd.read_csv(root / PD).query("analysis == 'Benchmark: Multilayer perceptron − cell-level GBLUP'")["verdict"] == "resolved loss").sum()))


# ---------------- D10: across-environment metric (post hoc, results/tpe_metric; Table S18) ----------------
TP = "tpe_metric/pooled.csv"
tpa = lambda root: pd.read_csv(root / TP).query("range == 'all'").set_index("item")
lib10b = ["reml", "reml_x0.1", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp"]
add("D10", "Across-env: resolution, 48 years", "MAIN", TP, "resolution", "0.0297", lambda root: tpa(root).loc["reml", "resolution"])
add("D10", "Across-env: two-stage methods with a resolved loss (seven)", "MAIN", TP, "verdict", "7", lambda root: float((tpa(root).loc[lib10b, "verdict"] == "resolved loss").sum()))
for m in ("reml", "reml_x10", "knn30", "gxe_gbm"):
    add("D10", f"Across-env: {m} tied", "MAIN", TP, "verdict", "1", lambda root, m=m: float(tpa(root).loc[m, "verdict"] == "tied"))
add("D10", "Across-env: no method with a resolved gain", "MAIN", TP, "verdict", "0", lambda root: float((pd.read_csv(root / TP)["verdict"] == "resolved gain").query if False else (tpa(root)["verdict"] == "resolved gain").sum()))


# ---------------- D11: value of the target-year phenotypes (post hoc, results/sparse_value; Table S19) ----------------
SVP = "sparse_value/pooled.csv"
sv = lambda f, sc, c, rg, col: csv_row(SVP, {"fraction": f, "scope": sc, "contrast": c, "range": rg}, col)
for f, est, lo, hi, res in ((0.25, "+0.072", "+0.057", "+0.088", "0.0103"), (0.5, "+0.103", "+0.079", "+0.126", "0.0140")):
    for col, v in (("est", est), ("lo", lo), ("hi", hi), ("resolution", res)):
        add("D11", f"Phenotypes themselves {f} all lines {col}", "MAIN", SVP, col, v, sv(f, "all", "m0-ref", "all", col))
for f, v in ((0.25, "4"), (0.5, "3")):
    add("D11", f"Phenotype value / G×E increment, {f} (about)", "MAIN", SVP, "ratio", v,
        lambda root, f=f: sv(f, "all", "m0-ref", "all", "est")(root) / sv(f, "all", "m1-m0", "all", "est")(root))
add("D11", "Phenotype value resolved within datasets at 25 % (G2F, NUST, ESWYT, GEM_IA)", "MAIN", SVP, "verdict", "4",
    lambda root: float(sum(pd.read_csv(root / SVP).set_index(["fraction", "scope", "contrast", "range"]).loc[(0.25, "all", "m0-ref", d), "verdict"] == "resolved gain"
                           for d in ("G2F", "NUST", "ESWYT", "GEM_IA"))))
add("D11", "  and not in URSN or MU_SOY at 25 %", "MAIN", SVP, "verdict", "0",
    lambda root: float(sum(pd.read_csv(root / SVP).set_index(["fraction", "scope", "contrast", "range"]).loc[(0.25, "all", "m0-ref", d), "verdict"] == "resolved gain"
                           for d in ("URSN", "MU_SOY"))))
for col, v in (("est", "+0.095"), ("lo", "+0.074"), ("hi", "+0.116")):
    add("D11", f"Phenotypes themselves 25 % new lines {col}", "MAIN", SVP, col, v, sv(0.25, "new", "m0-ref", "all", col))
add("D11", "Phenotypes themselves 25 % old lines est", "MAIN", SVP, "est", "+0.021", sv(0.25, "old", "m0-ref", "all", "est"))
add("D11", "  old lines resolution", "MAIN", SVP, "resolution", "0.0268", sv(0.25, "old", "m0-ref", "all", "resolution"))
add("D11", "  old lines 25 % below resolution (detectable)", "MAIN", SVP, "verdict", "1", lambda root: float(pd.read_csv(root / SVP).set_index(["fraction", "scope", "contrast", "range"]).loc[(0.25, "old", "m0-ref", "all"), "verdict"] == "detectable, below resolution"))
add("D11", "  old lines 50 % tied", "MAIN", SVP, "verdict", "1", lambda root: float(pd.read_csv(root / SVP).set_index(["fraction", "scope", "contrast", "range"]).loc[(0.5, "old", "m0-ref", "all"), "verdict"] == "tied"))
for col, v in (("est", "+0.013"), ("lo", "+0.010"), ("hi", "+0.016")):
    add("D11", f"G×E increment 25 % on common years {col}", "MAIN", SVP, col, v, sv(0.25, "all", "m1-m0", "common_years", col))
add("D11", "Common years", "MAIN", "sparse_value/meta.json", "common_years", "36", js("sparse_value/meta.json", "common_years"))
add("D11", "Reference control vs cell_reml: min of per-panel medians", "MAIN", "sparse_value/coverage.csv", "min", "1.000",
    lambda root: pd.read_csv(root / "sparse_value/coverage.csv")["ia_ctrl_vs_cell_reml_median_sp"].min())

NP = "sparse_value/nust_posthoc.csv"
for v, est, lo, hi in (("detrend", "+0.041", "+0.029", "+0.053"), ("trial_env", "+0.052", "+0.034", "+0.070")):
    for col, val in (("est", est), ("lo", lo), ("hi", hi)):
        add("D11", f"NUST phenotype value 25 %, {v} {col}", "MAIN", NP, col, val, csv_row(NP, {"fraction": 0.25, "variant": v, "contrast": "m0-ref"}, col))
add("D11", "NUST phenotype value resolved under all post hoc definitions and fractions", "MAIN", NP, "verdict", "4",
    lambda root: float((pd.read_csv(root / NP).query("contrast == 'm0-ref' and variant != 'orig'")["verdict"] == "resolved gain").sum()))
add("D11", "NUST post hoc script reproduces the pre-specified NUST phenotype value (25 %, 3 decimals)", "MAIN", NP, "est", "+0.048",
    csv_row(NP, {"fraction": 0.25, "variant": "orig", "contrast": "m0-ref"}, "est"))

for col, v in (("est", "+0.023"), ("lo", "+0.008"), ("hi", "+0.038")):
    add("D11", f"Old lines 50 % without single-environment years {col}", "MAIN", SVP, col, v, sv(0.5, "old", "m0-ref", "multi_env_years", col))
NB = "env_definition_check/network_boundary_seeds.csv"
add("D11", "Network, NUST trial units / URSN orig: seeds with tied verdict (of 20)", "MAIN", NB, "tied", "15", lambda root: float((pd.read_csv(root / NB)["verdict"] == "tied").sum()))
add("D11", "  seeds run", "MAIN", NB, "rows", "20", lambda root: float(len(pd.read_csv(root / NB))))

# ---------------- D12: main-text numbers found unchecked in the public-release audit (2026-10-08) ----------------
add("D12", "Unequal environment sizes: ratio over all six datasets ('1.44 times')", "MAIN", "resolution_note/by_dataset.csv",
    "All six datasets ratio_unequal_sizes", "1.44", csv_row("resolution_note/by_dataset.csv", {"dataset": "All six datasets"}, "ratio_unequal_sizes"))
for k, want in (("stk", "98.3"), ("dlres", "97.5")):
    add("D12", f"Sparse 0.25: Holm interval level of {k} on top of MxE (%)", "A", "headroom_sparse2/verdict.json",
        f"decisions.{k}.holm_level x 100", want, lambda root, k=k: 100 * js("headroom_sparse2/verdict.json", "decisions", k, "holm_level")(root))
add("D12", "CLAC ablation, single main effect: Holm interval level (%)", "MAIN", "clac_decomp/verdict.json",
    "verdict.loss_D5_single_main_effect.holm_level x 100", "98.75",
    lambda root: 100 * js("clac_decomp/verdict.json", "verdict", "loss_D5_single_main_effect", "holm_level")(root))
add("D12", "Table 1: GxE increment (MxE - main effect), upper end of the pooled range, 50 %", "A", sp, "est (3 decimals)", "+0.035",
    csv_row(sp, {"fraction": "0.5", "contrast": "m1-m0", "metric": "sp", "range": "all48"}, "est"))
HKS = "revision_tcj/hk_sensitivity.csv"


def hk_not_wider(root):
    d = pd.read_csv(root / HKS)
    d = d[d["target_years"] > 1]
    return float(((d["hk_hi"] - d["hk_lo"]) <= (d["dl_hi"] - d["dl_lo"])).sum())


add("D12", "Section 2.6: contrasts (>= 2 years) whose Hartung-Knapp interval is not wider than DL ('wider for every contrast')",
    "MAIN", HKS, "count hk width <= dl width", "0", hk_not_wider)
for f, v in ((0.25, "3.9"), (0.5, "2.9")):   # cover letter
    add("D12", f"Cover letter: phenotype value / GxE increment on the same cells, {f}", "MAIN", SVP, "ratio", v,
        lambda root, f=f: sv(f, "all", "m0-ref", "all", "est")(root) / sv(f, "all", "m1-m0", "all", "est")(root))
TA = "transfer_arc/pooled.csv"
for col, v in (("est", "-0.006"), ("lo", "-0.032"), ("hi", "+0.021"), ("resolution", "0.036"), ("k", "4")):   # Section 4.2
    add("D12", f"Kernel transfer, GEM_IA ({col})", "MAIN", TA, col, v, csv_row(TA, {"range": "GEM_IA", "metric": "spearman", "scope": "all"}, col))

def decimals(s):
    s = s.split()[0].lstrip("<=+ ")
    return len(s.split(".")[1]) if "." in s else 0


# ---------------- D13: v14e (2026-10-08): threshold constant and single-year thresholds (results/threshold_sensitivity; Table S20, Section 4.3)
TS, TY = "threshold_sensitivity/verdicts.csv", "threshold_sensitivity/per_year.csv"
add("D13", "Section 4.3: pooled contrasts re-judged", "MAIN", TS, "rows", "31", lambda root: float(len(pd.read_csv(root / TS))))
for c in (2, 4):
    add("D13", f"Section 4.3: verdicts changed with constant {c} (vs 3)", "MAIN", TS, f"verdict_c{c} != verdict_c3", "0",
        lambda root, c=c: float((pd.read_csv(root / TS)[f"verdict_c{c}"] != pd.read_csv(root / TS)["verdict_c3"]).sum()))
add("D13", "Section 4.3: constant 3 reproduces the stored verdicts", "MAIN", TS, "verdict_c3 == dl_verdict", "31",
    lambda root: float((pd.read_csv(root / TS)["verdict_c3"].values == pd.read_csv(root / "revision_tcj/hk_sensitivity.csv")["dl_verdict"].values).sum()))
for d, col, v in (("G2F", "threshold_min", "0.025"), ("G2F", "threshold_max", "0.035"), ("NUST", "threshold_min", "0.051"), ("NUST", "threshold_max", "0.092")):
    add("D13", f"Section 4.3: single-year 3/sqrt(N), {d} {col}", "MAIN", TY, col, v, csv_row(TY, {"dataset": d}, col))
add("D13", "Section 4.3: smallest multiple among the other positive findings (FW, March forecasts)", "MAIN", RN, "preseason ratio", "1.2",
    lambda root: [r for r in rows if r[0] == "D7" and "preseason" in r[1]][0][6](root))


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
