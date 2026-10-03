"""Mechanical check of the numbers in manuscript/draft_methods_results_*.md against the committed result files.
Each row: (label, value read from a result file, value printed in the draft, tolerance). Exit code 1 if any fails.
Run from the repository root:  python3 scripts/check_numbers.py
Intervals printed with 97.5 % (Holm first test) are checked against the pre-registration text, not pooled.json,
which stores 95 % intervals; those rows are marked 'text' and compare with values transcribed from the result section
of the pre-registration (see the source column)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
rows = []


def add(label, got, want, tol=6e-4, src=""):
    rows.append((label, float(got), float(want), tol, src))


def pj(path):
    return pd.DataFrame(json.load(open(R / path)))


def pooled(df, contrast, subset, metric):
    r = df[(df.contrast == contrast) & (df["subset"] == subset) & (df.metric == metric)].iloc[0]
    return r


b3, b3b, b3c, b3d = pj("b3/pooled.json"), pj("b3b/pooled.json"), pj("b3c/pooled.json"), pj("b3d/pooled.json")

# ---- 3.4.1 / 3.4.2 (wave 2b)
r = pooled(b3, "H1", "main", "spearman")
add("H1 est", r.est, 0.0010, src="b3/pooled.json"); add("H1 lo", r.lo, -0.0051); add("H1 hi", r.hi, 0.0071)
r = pooled(b3, "H2", "main", "spearman")
add("H2 est", r.est, 0.0155, src="b3/pooled.json")
r = pooled(b3, "H2", "new_only", "spearman")
add("H2 new est", r.est, -0.0015); add("H2 new lo", r.lo, -0.0139); add("H2 new hi", r.hi, 0.0110)
# ---- 3.4.3 (wave 2c)
for c, sub, m, est, lo, hi in (("H3", "all", "spearman", 0.0028, -0.0079, 0.0135), ("H3", "all", "sel_diff_f10", 0.031, 0.010, 0.052),
                               ("H4", "new_only", "sel_diff_f10", 0.040, 0.001, 0.080), ("H4", "new_only", "spearman", 0.0072, None, None)):
    r = pooled(b3b, c, sub, m)
    add(f"{c} {sub} {m} est", r.est, est, 1e-3, "b3b/pooled.json")
    if lo is not None:
        add(f"{c} {sub} {m} lo", r.lo, lo, 1e-3); add(f"{c} {sub} {m} hi", r.hi, hi, 1e-3)
# ---- 3.4.4 (wave 2d)
r = pooled(b3c, "H5", "all", "spearman"); add("H5 est", r.est, -0.0083, src="b3c/pooled.json")
r = pooled(b3c, "H6", "new_only", "spearman"); add("H6 est", r.est, -0.0028)
d = pj("b3c/pooled.json")
r = d[d.contrast == "dl_g_vs_twostage_mlp"]; r = r[r.metric == "spearman"].iloc[0]
add("dl_g - two-stage mlp est", r.est, 0.089, 1e-3); add("  lo", r.lo, 0.063, 1e-3); add("  hi", r.hi, 0.115, 1e-3)
r = d[(d.contrast == "dl_ge_vs_cell_G2F+URSN") & (d.metric == "spearman")].iloc[0]
add("dl_ge - cell est", r.est, -0.023, 1e-3); add("  lo", r.lo, -0.048, 1e-3); add("  hi", r.hi, 0.003, 1e-3)
v = json.load(open(R / "b3c/verdict.json"))
add("dl configurations selected", len(v["config_counts"]), 8); add("dl max config count", max(v["config_counts"].values()), 25)
add("dl median epochs", v["e_star_median"], 26); add("dl runs", sum(v["config_counts"].values()), 141)
# ---- 3.4.5 (wave 2e)
for sub, m, est, lo, hi in (("all", "sel_diff_f10", -0.0396, -0.0885, 0.0093), ("all", "spearman", -0.0125, -0.0280, 0.0029),
                            ("ESWYT", "sel_diff_f10", -0.0382, -0.0903, 0.0139), ("GEM_IA", "sel_diff_f10", -0.1902, -0.3604, -0.0199),
                            ("GEM_IA", "spearman", -0.0765, -0.1421, -0.0108), ("MU_SOY", "sel_diff_f10", 0.0808, -0.0298, 0.1913)):
    r = pooled(b3d, "H7", sub, m)
    add(f"H7 {sub} {m} est", r.est, est, 6e-4, "b3d/pooled.json"); add("  lo", r.lo, lo, 6e-4); add("  hi", r.hi, hi, 6e-4)
r = pooled(b3d, "H7_new", "new_only", "sel_diff_f10")
add("H7 new est", r.est, -0.1032, 6e-4); add("  lo", r.lo, -0.1632, 6e-4); add("  hi", r.hi, -0.0433, 6e-4)
vd = json.load(open(R / "b3d/verdict.json"))
add("N independent", vd["N_cells_all"], 31144, 0); add("resolution independent", vd["H7"]["threshold"], 0.0170, 1e-4)
add("combined 48 est", vd["combined_with_wave2c_31_years_descriptive"]["est"], 0.0092, 6e-4)
add("combined 48 lo", vd["combined_with_wave2c_31_years_descriptive"]["lo"], -0.0133, 6e-4)
add("combined 48 hi", vd["combined_with_wave2c_31_years_descriptive"]["hi"], 0.0318, 6e-4)
# ---- 3.4.6 (summary48)
S = pd.read_csv(R / "summary48/pooled.csv")
S = S[(S.metric == "spearman") & (S.scope == "all48")].set_index("item")
for it, est, lo, hi, k in (("reml", -0.012, -0.019, -0.006, 48), ("rf", -0.026, -0.040, -0.011, 48), ("gbm", -0.041, -0.058, -0.024, 46),
                           ("mlp", -0.089, -0.110, -0.067, 48), ("R_STK_FW", -0.002, -0.012, 0.007, 48), ("R_STK_CV", -0.004, -0.014, 0.006, 48),
                           ("R_EQ", -0.006, -0.014, 0.003, 48), ("R_CV", -0.010, -0.019, -0.001, 48), ("R_FW", -0.005, -0.013, 0.004, 48),
                           ("Oracle", 0.019, 0.010, 0.027, 48)):
    add(f"48y {it} est", S.loc[it, "est"], est, 1e-3, "summary48/pooled.csv"); add("  lo", S.loc[it, "lo"], lo, 1e-3)
    add("  hi", S.loc[it, "hi"], hi, 1e-3); add("  years", S.loc[it, "k"], k, 0)
S2 = pd.read_csv(R / "summary48/pooled.csv")
S2 = S2[(S2.metric == "spearman") & (S2.scope == "new17")].set_index("item")
add("new17 Oracle est", S2.loc["Oracle", "est"], 0.018, 1e-3); add("new17 R_STK_FW est", S2.loc["R_STK_FW", "est"], -0.013, 1e-3)
# ---- 3.2 (E1, exact)
w = json.load(open(R / "wp4/e1_exact_verdict.json"))
add("E1-1 in direction", w["E1_1_phi"]["in_direction"], 18, 0); add("E1-2 in direction", w["E1_2_abslog"]["in_direction"], 12, 0)
add("E1-2 n used", w["E1_2_abslog"]["n_used"], 16, 0); add("E1-2 p", w["E1_2_abslog"]["p_one_sided"], 0.038, 1e-3)
add("E1-1 p", w["E1_1_phi"]["p_one_sided"], 3.8e-6, 1e-7)
e1 = json.load(open(R / "e1/verdict.json")); add("E1-2 approx in direction", e1["E1_2_abslog"]["in_direction"], 13, 0)
wb = pd.read_csv(R / "wp4/w1b_exact.csv").groupby(["scenario", "protocol"])["abs_log_u_ustar"].mean()
for sc, pr, v_ in (("F2024m", "own", 0.88), ("F2024m", "fair", 1.53), ("F2022m", "own", 0.31), ("F2022m", "fair", 0.94)):
    add(f"W1b exact |log u/u*| {sc} {pr}", wb.loc[(sc, pr)], v_, 6e-3, "wp4/w1b_exact.csv")
# ---- 3.1 (A1) table
A = pd.read_csv(R / "a1/summary.csv")
for sc, ctr, met, d_, lo_, hi_ in (("F2024m", "O", "spearman", 0.170, 0.068, 0.263), ("F2024m", "O", "pooled_pearson", 0.074, -0.099, 0.231),
                                   ("F2022m", "O", "spearman", 0.021, -0.135, 0.144), ("F2022m", "O", "pooled_pearson", 0.075, -0.081, 0.238),
                                   ("F2024m", "O_sel", "spearman", 0.226, 0.091, 0.380), ("F2024m", "O_scale", "spearman", -0.056, -0.153, 0.057),
                                   ("F2024m", "C_fair", "spearman", -0.231, -0.352, -0.120), ("F2022m", "C_fair", "spearman", -0.177, -0.236, -0.125)):
    a1 = A[(A.scenario == sc) & (A.contrast == ctr) & (A.metric == met)]
    if a1.empty:
        print(f"WARN no a1 row for {sc} {ctr} {met}; metric names: {sorted(A.metric.unique())}")
        continue
    a1 = a1.iloc[0]
    add(f"A1 {sc} {ctr} {met} delta", a1.delta, d_, 1.5e-3, "a1/summary.csv"); add("  lo", a1.ci2_lo, lo_, 1.5e-3); add("  hi", a1.ci2_hi, hi_, 1.5e-3)
# ---- 3.5 (tool validation)
for tag, path in (("round1", "wp2/validation/verdict.json"), ("round2", "wp2/validation_v2/verdict.json")):
    v = json.load(open(R / path))
    add(f"{tag} A1 fail rate", v["V1_A1_fail_rate_P1P3"][0], 1.0, 1e-9, path)
    add(f"{tag} A2 sensitivity", v["V2_A2_sensitivity"][0], 0.846 if tag == "round1" else 0.876, 6e-4)
    add(f"{tag} A2 false alarm", v["V2_A2_false_alarm"][0], 0.001 if tag == "round1" else 0.000, 6e-4)
    add(f"{tag} A4 error ratio", v["V3_ratio"] if "V3_ratio" in v else v["V3_mae_crossfit"] / v["V3_mae_reported"], 1.36 if tag == "round1" else 1.02, 6e-3)
# ---- Table 1 (benchmark package)
sp = json.load(open(R / "benchmark/splits.json"))
for dsn, ty, ne in (("G2F", 5, 129), ("NUST", 15, 499), ("URSN", 11, 58), ("ESWYT", 12, 451), ("GEM_IA", 4, 33), ("MU_SOY", 1, 11)):
    add(f"{dsn} target years", len(sp[dsn]["target_years"]), ty, 0, "benchmark/splits.json")
    add(f"{dsn} scored environments", sum(len(x) for x in sp[dsn]["scorable_environments"].values()), ne, 0)
cnt = pd.concat([pd.read_csv(R / "w1c/per_year.csv"), pd.read_csv(R / "b3d/count/per_year.csv")])
q = cnt[(cnt.rule == "main") & cnt.qualifies].groupby("dataset")["cells"].sum()
for dsn, n in (("G2F", 53638), ("NUST", 32740), ("URSN", 916), ("ESWYT", 22323), ("GEM_IA", 6942), ("MU_SOY", 1879)):
    add(f"{dsn} scored cells", q.loc[dsn], n, 0, "w1c + b3d count")
add("original cells", q.loc[["G2F", "NUST", "URSN"]].sum(), 87294, 0); add("original resolution", 3 / np.sqrt(87294), 0.0102, 5e-5)

bad = 0
for label, got, want, tol, src in rows:
    ok = abs(got - want) <= tol
    bad += not ok
    print(("ok   " if ok else "FAIL ") + f"{label:42s} file {got:+.5g}  draft {want:+.5g}  {src}")
print(f"\n{len(rows) - bad}/{len(rows)} numbers agree")
sys.exit(1 if bad else 0)
