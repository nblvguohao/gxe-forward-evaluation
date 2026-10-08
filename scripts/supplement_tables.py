"""Supplementary tables that are generated from the result files (Tables S3, S5-S7, S10, S11, S13-S15, S17-S19, and the
numbers in the sentences next to them). The manuscript builders (not in the public release) import these functions, so
the supplement and this script print the same tables.

Run from the repository root:  python3 scripts/supplement_tables.py [output directory, default results/supplement_tables]
Writes one Markdown file per table; compare them with the published Supplementary Information."""
import json
import sys
from pathlib import Path

import pandas as pd

R = Path("results")


def f(x, d=4):
    return f"{x:+.{d}f}".replace("-", "−")


def ci(lo, hi, d=4):
    return f"[{f(lo, d)}, {f(hi, d)}]"


def table_mde():
    m = pd.read_csv(R / "mde_ties/mde.csv")
    rows = ["| Contrast | Reference | Target years | Δ [interval] | Interval level | Minimum detectable effect | Resolution | Resolved gain excluded |",
            "|---|---|---|---|---|---|---|---|"]
    for r in m.itertuples():
        ref = "cell-level GBLUP" if r.reference == "cell_reml" else "M×E GBLUP"
        rows.append(f"| {r.contrast} | {ref} | {r.target_years} | {f(r.est)} {ci(r.lo, r.hi)} | {100 * r.level:.4g} % | {r.mde80:.4f} | "
                    f"{r.resolution:.4f} | {'yes' if r.resolved_gain_excluded else 'no'} |")
    return "\n".join(rows)


def table_ablation():
    v = json.load(open(R / "clac_decomp/verdict.json"))["verdict"]
    y = pd.read_csv(R / "clac_decomp/year_effects.csv").pivot(index="contrast", columns="target", values="d")
    names = {"loss_D4_linear_kernel": "Linear instead of arc-cosine kernel", "loss_D5_single_main_effect": "Single main-effect GBLUP",
             "loss_D1_equal_weights": "Equal weights over training environments", "loss_D3_no_cleaning": "No phenotype cleaning",
             "loss_D2_no_spatial": "No spatial adjustment"}
    yrs = [2016, 2018, 2020, 2022, 2024]
    rows = ["| Component removed or replaced | Loss [Holm interval] | Level | " + " | ".join(map(str, yrs)) + " | Verdict |",
            "|---|---|---|" + "---|" * len(yrs) + "---|"]
    for k, nm in names.items():
        x = v[k]
        rows.append(f"| {nm} | {f(x['est'])} {ci(*x['ci'])} | {100 * x['holm_level']:.4g} % | " + " | ".join(f(y.loc[k, t]) for t in yrs)
                    + f" | {x['judgement']} |")
    p = pd.read_csv(R / "clac_decomp/pooled.csv").set_index("contrast")
    desc = []
    for k, nm in (("A_full_vs_cell_reml", "full multivariate GBLUP"), ("D4_linear_kernel_vs_cell_reml", "with a linear kernel"),
                  ("D5_single_main_effect_vs_cell_reml", "single main-effect GBLUP"), ("D1_equal_weights_vs_cell_reml", "equal weights"),
                  ("D3_no_cleaning_vs_cell_reml", "no phenotype cleaning"), ("D2_no_spatial_vs_cell_reml", "no spatial adjustment")):
        desc.append(f"{nm} {f(p.loc[k, 'est'])} {ci(p.loc[k, 'lo'], p.loc[k, 'hi'])}")
    return "\n".join(rows), "; ".join(desc)


def table_spatial():
    s = pd.read_csv(R / "clac_decomp/spatial_adjustment_counts.csv")
    rows = ["| Target year | Training environments | Adjustment applied | No field layout | Algorithm failed | Negative mean | Implausible variance share |",
            "|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        rows.append(f"| {r.target} | {r.training_environments} | {r.adjustment_successful} ({100 * r.adjustment_successful / r.training_environments:.0f} %) | "
                    f"{r.no_spatial_information} | {r.algorithm_broke} | {r.negative_mean} | {r.odd_result} |")
    return "\n".join(rows)


def table_transfer():
    p = pd.read_csv(R / "transfer_arc/pooled.csv")
    lab = {"all48": "All six datasets", "orig31": "Original set (G2F, NUST, URSN)", "new17": "Independent set (ESWYT, GEM_IA, MU_SOY)", "G2F": "G2F maize",
           "NUST": "NUST soybean", "URSN": "URSN wheat", "ESWYT": "ESWYT wheat", "GEM_IA": "GEM_IA maize", "MU_SOY": "MU_SOY soybean"}
    rows = ["| Range | Target years | Scored cells | Δ Spearman, all genotypes [95 % CI] | Resolution | Verdict | Δ Spearman, new genotypes only [95 % CI] | "
            "Δ selection differential, all genotypes [95 % CI] |", "|---|---|---|---|---|---|---|---|"]
    g = lambda rng, metric, scope: p[(p["range"] == rng) & (p["metric"] == metric) & (p["scope"] == scope)].iloc[0]
    for k, nm in lab.items():
        a, n, s = g(k, "spearman", "all"), g(k, "spearman", "new"), g(k, "sel_diff", "all")
        rows.append(f"| {nm} | {int(a.k)} | {int(a.N):,} | {f(a.est)} {ci(a.lo, a.hi)} | {a.resolution:.4f} | {a.judgement} | "
                    f"{f(n.est)} {ci(n.lo, n.hi)} | {f(s.est, 3)} {ci(s.lo, s.hi, 3)} |")
    lo = pd.read_csv(R / "transfer_arc/lodo.csv")
    lod = "; ".join(f"without {r.left_out} {f(r.est)} {ci(r.lo, r.hi)}" for r in lo.itertuples())
    return "\n".join(rows), lod


def table_floor():
    T = pd.read_csv(R / "floor_sensitivity/table.csv")
    rows = ["| Contrast | Target years | Level | Resolution | Floor of the analysis plans: Δ [interval] | Verdict | Floor on zero standard errors only: Δ [interval] | Verdict | "
            "Years raised by the floor | Years with zero standard error |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        rows.append(f"| {r.contrast} | {r.target_years} | {100 * r.level:.4g} % | {r.resolution:.4f} | {f(r.est_plan)} {ci(r.lo_plan, r.hi_plan)} | {r.verdict_plan} | "
                    f"{f(r.est_zero_only)} {ci(r.lo_zero_only, r.hi_zero_only)} | {r.verdict_zero_only} | {r.years_raised_by_plan_floor} | {r.years_with_zero_se} |")
    return "\n".join(rows), int(T["verdict_changed"].sum()), len(T)


def six_env_note():
    j = json.load(open(R / "floor_sensitivity/ablations_without_six_environments.json"))
    c = j["contrasts"]
    nm = {"loss_D4_linear_kernel": "linear kernel", "loss_D5_single_main_effect": "single main effect", "loss_D1_equal_weights": "equal weights",
          "loss_D3_no_cleaning": "no phenotype cleaning", "loss_D2_no_spatial": "no spatial adjustment"}
    parts = "; ".join(f"{nm[k]} {f(c[k]['est'])} {ci(c[k]['lo'], c[k]['hi'])}" for k in nm)
    return (f"Six of the 129 scored environments (1,972 cells) have no final CLAC prediction, because the environment-mean model of the pipeline had no "
            f"covariates for them. Without these environments ({j['environments']} environments, {j['cells']:,} cells, resolution {j['resolution']:.4f}), "
            f"the losses at the same interval levels were: {parts}. The full multivariate GBLUP then exceeded cell-level GBLUP by "
            f"{f(c['A_full_vs_cell_reml']['est'])} {ci(c['A_full_vs_cell_reml']['lo'], c['A_full_vs_cell_reml']['hi'])}. No verdict changed.")


def ci3(est, lo, hi):
    return f"{f(est, 3)} {ci(lo, hi, 3)}"


VARIANTS = [("NUST:orig / URSN:orig", "Pre-specified"),
            ("NUST:detrend / URSN:orig", "NUST centred within trial"),
            ("NUST:detrend / URSN:nocheck_n3", "NUST centred within trial; URSN without checks (≥ 3 lines)"),
            ("NUST:detrend / URSN:nocheck_n8", "NUST centred within trial; URSN without checks (≥ 8 lines)"),
            ("NUST:trial_env / URSN:orig", "NUST trial × location × year (≥ 10 lines)"),
            ("NUST:trial_env / URSN:nocheck_n3", "NUST trial × location × year (≥ 10 lines); URSN without checks (≥ 3 lines)")]
SPARSE_KEY = {"NUST:orig / URSN:orig": ("orig", "orig"), "NUST:detrend / URSN:orig": ("detrend", "orig"),
              "NUST:detrend / URSN:nocheck_n3": ("detrend", "nocheck"), "NUST:trial_env / URSN:nocheck_n3": ("trial_env", "nocheck")}
LIB10 = ["reml", "reml_x0.1", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp"]


def table_s11():
    pv = pd.read_csv(R / "env_definition_check/pooled_by_variant.csv")
    sv = pd.read_csv(R / "env_definition_check/sparse_by_variant.csv")
    u12 = pd.read_csv(R / "env_definition_check/ursn_min12_pooled.csv").set_index("item")
    rows = ["| Scoring | Two-stage methods with a resolved loss (of 10) | Two-stage GBLUP | Covariate LightGBM | Within-environment-loss network | "
            "Sparse testing, 25 % | Sparse testing, 50 % | Sparse testing in NUST alone, 25 % (resolution 0.0204) | NUST alone, 50 % (resolution 0.0290) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for key, label in VARIANTS:
        g = pv[pv["variant"] == key].set_index("item")
        n = int((g.loc[LIB10, "verdict"] == "resolved loss").sum())
        cells = [ci3(g.loc[m, "est"], g.loc[m, "lo"], g.loc[m, "hi"]) for m in ("reml", "gxe_gbm", "dl_g")]
        if key == "NUST:orig / URSN:orig":  # the primary results of the main text, not the rescored copy
            ft = pd.read_csv(R / "floor_sensitivity/table.csv").set_index("contrast")
            names = ("Two-stage GBLUP", "Covariate LightGBM", "Within-environment-loss network")
            cells = [ci3(*ft.loc[f"Benchmark: {m} − cell-level GBLUP", ["est_plan", "lo_plan", "hi_plan"]]) for m in names]
            cells += [ci3(*ft.loc[f"Sparse {q} %: M×E − main-effect GBLUP", ["est_plan", "lo_plan", "hi_plan"]]) for q in (25, 50)]
            for fr in (0.25, 0.5):
                r = sv[(sv["fraction"] == fr) & (sv["NUST"] == "orig") & (sv["URSN"] == "orig")].iloc[0]
                cells.append(ci3(r["NUST_alone_est"], r["NUST_alone_lo"], r["NUST_alone_hi"]))
            rows.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
            continue
        for fr in (0.25, 0.5):
            if key in SPARSE_KEY:
                nv, uv = SPARSE_KEY[key]
                r = sv[(sv["fraction"] == fr) & (sv["NUST"] == nv) & (sv["URSN"] == uv)].iloc[0]
                cells.append(ci3(r["est"], r["lo"], r["hi"]))
            else:
                cells.append("—")
        for fr in (0.25, 0.5):
            if key in SPARSE_KEY:
                nv, uv = SPARSE_KEY[key]
                r = sv[(sv["fraction"] == fr) & (sv["NUST"] == nv) & (sv["URSN"] == uv)].iloc[0]
                cells.append(ci3(r["NUST_alone_est"], r["NUST_alone_lo"], r["NUST_alone_hi"]) if nv != "orig" or uv == "orig" else "—")
            else:
                cells.append("—")
        rows.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    n = int((u12.loc[LIB10, "verdict"] == "resolved loss").sum())
    rows.append(f"| URSN with at least 12 lines | {n} | " + " | ".join(ci3(u12.loc[m, "est"], u12.loc[m, "lo"], u12.loc[m, "hi"]) for m in ("reml", "gxe_gbm", "dl_g")) + " | — | — | — | — |")
    return "\n".join(rows)


def table_s14():
    H = pd.read_csv(R / "revision_tcj/hk_sensitivity.csv")
    rows = ["| Contrast | Target years | Level | Resolution | Estimate | DerSimonian–Laird interval | Verdict | τ² (×10⁻⁴) | Hartung–Knapp interval | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in H.itertuples():
        hk = ci(r.hk_lo, r.hk_hi) if r.target_years > 1 else "—"
        rows.append(f"| {r.contrast} | {r.target_years} | {100 * r.level:.4g} % | {r.resolution:.4f} | {f(r.est)} | {ci(r.dl_lo, r.dl_hi)} | "
                    f"{r.dl_verdict} | {1e4 * r.tau2:.2f} | {hk} | {r.hk_verdict} |")
    return "\n".join(rows), int(H["verdict_changed"].sum()), len(H)


def table_s17():
    y = pd.read_csv(R / "reanalysis/year_effects.csv")
    y = y[(y["method"] == "clac") & (y["metric"] == "sp")]
    rows = ["| Target year | Environments | Δ | Standard error |", "|---|---|---|---|"]
    for r in y.itertuples():
        rows.append(f"| {r.target} | {r.n_env} | {f(r.d)} | {r.se:.4f} |")
    return "\n".join(rows)


def table_s18():
    P = pd.read_csv(R / "tpe_metric/pooled.csv")
    P = P[P["range"] == "all"]
    W = pd.read_csv(R / "summary48/pooled.csv")
    W = W[(W["metric"] == "spearman") & (W["scope"] == "all48")].set_index("item")
    names = {"reml": "Two-stage GBLUP", "reml_x0.1": "Two-stage ridge, shrinkage ×0.1", "reml_x10": "Two-stage ridge, shrinkage ×10",
             "reml_x100": "Two-stage ridge, shrinkage ×100", "ridge_pc20": "Ridge on 20 PCs", "rf": "Random forest", "gbm": "Gradient boosting",
             "knn10": "kNN, k = 10", "knn30": "kNN, k = 30", "mlp": "Multilayer perceptron", "gxe_gbm": "Covariate LightGBM",
             "rn_ridge": "Covariate reaction-norm ridge", "dl_g": "Within-environment-loss network"}
    rows = ["| Method − cell-level GBLUP | Target years | Lines | Resolution 3/√L | Across environments: Δ [95 % CI] | Hartung–Knapp | Verdict | Within environments (main text): Δ [95 % CI] |",
            "|---|---|---|---|---|---|---|---|"]
    for k in names:
        r = P[P["item"] == k].iloc[0]
        w = W.loc[k]
        rows.append(f"| {names[k]} | {int(r['k'])} | {int(r['lines']):,} | {r['resolution']:.4f} | {f(r['est'])} {ci(r['lo'], r['hi'])} | "
                    f"{ci(r['hk_lo'], r['hk_hi'])} | {r['verdict']} | {f(w['est'])} {ci(w['lo'], w['hi'])} |")
    return "\n".join(rows)


def table_s19():
    P = pd.read_csv(R / "sparse_value/pooled.csv")
    lab = {"m0-ref": "Phenotypes themselves: main-effect GBLUP with minus without target-year phenotypes",
           "m1-m0": "G×E increment: M×E GBLUP minus main-effect GBLUP, both with the phenotypes",
           "m1-ref": "Total: M×E GBLUP with minus main-effect GBLUP without target-year phenotypes"}
    rows = ["| Contrast | Phenotyped | Lines | Range | Target years | Resolution | Estimate [95 % CI] | Hartung–Knapp | Verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for c in ("m0-ref", "m1-m0", "m1-ref"):
        for fr in (0.25, 0.5):
            for scope in ("all", "new", "old"):
                for rng_ in (["all", "common_years", "multi_env_years"] + (["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"] if scope == "all" else [])):
                    r = P[(P["contrast"] == c) & (P["fraction"] == fr) & (P["scope"] == scope) & (P["range"] == rng_)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    hk = ci(r["hk_lo"], r["hk_hi"]) if pd.notna(r["hk_lo"]) else "—"
                    rng_lab = {"common_years": "common years", "multi_env_years": "years with ≥ 2 environments"}.get(rng_, rng_)
                    rows.append(f"| {lab[c]} | {int(100 * fr)} % | {scope} | {rng_lab} | {int(r['k'])} | {r['resolution']:.4f} | "
                                f"{f(r['est'])} {ci(r['lo'], r['hi'])} | {hk} | {r['verdict']} |")
    N = pd.read_csv(R / "sparse_value/nust_posthoc.csv")
    names = {"orig": "pre-specified", "detrend": "centred within trial", "trial_env": "trial × location × year units"}
    for r in N[N["contrast"] == "m0-ref"].itertuples():
        rows.append(f"| {lab['m0-ref']}, NUST under {names[r.variant]} | {int(100 * r.fraction)} % | all | NUST | {r.k} | {r.resolution:.4f} | "
                    f"{f(r.est)} {ci(r.lo, r.hi)} | {ci(r.hk_lo, r.hk_hi)} | {r.verdict} |")
    return "\n".join(rows)


def table_s15():
    P = pd.read_csv(R / "revision_tcj/per_dataset.csv")
    W = pd.read_csv(R / "revision_tcj/without_eswyt_ursn.csv")
    rows = ["| Contrast | Dataset | Target years | Cells | Resolution | Estimate [95 % CI] | Verdict |", "|---|---|---|---|---|---|---|"]
    for r in P.itertuples():
        rows.append(f"| {r.analysis} | {r.dataset} | {r.target_years} | {r.cells:,} | {r.resolution:.4f} | {f(r.est)} {ci(r.lo, r.hi)} | {r.verdict} |")
    for r in W.itertuples():
        rows.append(f"| {r.analysis} | without ESWYT and URSN | {r.target_years} | — | {r.resolution:.4f} | {f(r.est)} {ci(r.lo, r.hi)} | {r.verdict} |")
    WN = pd.read_csv(R / "revision_tcj/without_nust.csv")
    for r in WN.itertuples():
        rows.append(f"| {r.analysis} | without NUST | — | — | {r.resolution:.4f} | {f(r.est_without_nust)} {ci(r.lo, r.hi)} | {r.verdict} |")
    return "\n".join(rows)


def table_s13():
    T = pd.read_csv(R / "resolution_note/by_dataset.csv")
    rows = ["| Dataset | Environments | Cells (N) | Lines per environment: min / median / max | Mean within-environment Spearman of cell-level GBLUP (ρ) | "
            "(1 − ρ²)(1.060)<sup>½</sup> | Ratio for unequal sizes, (N Σ 1/n<sub>e</sub>)<sup>½</sup>/E |", "|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        rows.append(f"| {r.dataset} | {r.environments:,} | {r.cells:,} | {r.n_min} / {r.n_median:g} / {r.n_max} | {r.mean_rho_cell_reml:.2f} | "
                    f"{r.fhp_factor:.2f} | {r.ratio_unequal_sizes:.2f} |")
    return "\n".join(rows)


def implied():
    S = pd.read_csv(R / "resolution_note/implied_se.csv")
    return S["se_over_scale"].min(), S["se_over_scale"].median(), S["se_over_scale"].max(), len(S)


def floor_no_floor_sentence():
    fv = pd.read_csv(R / "revision_tcj/floor_variants.csv").set_index("fraction")
    return (" Without any floor, after dropping the target years whose standard error is zero "
            f"({int(fv.loc[0.25, 'years_dropped'])} at 25 % and {int(fv.loc[0.5, 'years_dropped'])} at 50 %), the sparse-testing gains were "
            f"{f(fv.loc[0.25, 'est_no_floor_zero_dropped'])} {ci(fv.loc[0.25, 'lo_no_floor_zero_dropped'], fv.loc[0.25, 'hi_no_floor_zero_dropped'])} "
            f"and {f(fv.loc[0.5, 'est_no_floor_zero_dropped'])} {ci(fv.loc[0.5, 'lo_no_floor_zero_dropped'], fv.loc[0.5, 'hi_no_floor_zero_dropped'])}.")


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    abl, abl_desc = table_ablation()
    tr, lodo = table_transfer()
    fl, n_changed, n_all = table_floor()
    hk, hk_changed, hk_all = table_s14()
    lo, med, hi, k = implied()
    tables = {
        "S03_minimum_detectable_effects": table_mde(),
        "S05_clac_ablations": abl + "\n\nAgainst cell-level GBLUP (95 % intervals, descriptive): " + abl_desc + ".\n\n" + six_env_note(),
        "S06_spatial_adjustment": table_spatial(),
        "S07_kernel_transfer": tr + "\n\nLeaving one dataset out, the pooled estimate over the remaining target years was: " + lodo + ".",
        "S10_floor_sensitivity": fl + f"\n\nThe verdict changed in {n_changed} of {n_all} contrasts." + floor_no_floor_sentence(),
        "S11_environment_definitions": table_s11(),
        "S13_resolution_by_dataset": table_s13() + f"\n\nNote S2, condition 3: Over the {k} pooled contrasts of Supplementary Table S10, the standard error "
                                     f"implied by the reported interval ranged from {lo:.2f} to {hi:.1f} times 1/√N (median {med:.2f}).",
        "S14_hartung_knapp": hk + f"\n\nThe verdict changed in {hk_changed} of {hk_all} contrasts.",
        "S15_by_dataset": table_s15(),
        "S17_clac_by_year": table_s17(),
        "S18_across_environment_metric": table_s18(),
        "S19_value_of_phenotypes": table_s19(),
    }
    for name, body in tables.items():
        (out / f"{name}.md").write_text(f"# Table {name.split('_')[0]}\n\n{body}\n")
    print(f"wrote {len(tables)} tables to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else R / "supplement_tables")
