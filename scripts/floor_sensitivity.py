"""Sensitivity of the pooled contrasts to the standard-error floor, and of the CLAC ablations to six environments.

The analysis plans (amendment of 2026-09-28) set each year's standard error to max(SE, median positive SE of its dataset)
before DerSimonian-Laird pooling. This flattens the weights of years with small bootstrap standard errors. Here every
primary within-environment Spearman contrast of the manuscript is pooled again with a floor applied to zero standard
errors only, from the stored year effects. No model is fitted and nothing is re-scored.

Second part: the CLAC ablations were scored in 129 environments, 6 of which have no final CLAC prediction (the
pipeline's environment-mean model had no covariates there). The ablation contrasts are recomputed without these 6.

Run from the repository root:  python3 scripts/floor_sensitivity.py  ->  results/floor_sensitivity/"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from dartgxe.forward.pool import dersimonian_laird, floor_se  # noqa: E402

R, OUT = Path("results"), Path("results/floor_sensitivity")
NO_FINAL_CLAC = ["NCH1_2016", "ONH1_2016", "ONH2_2016", "GEH1_2020", "GEH1_2022", "ONH3_2024"]  # GM_models_PredVar is NA


def zero_only(ye: pd.DataFrame) -> pd.DataFrame:
    """Replace zero standard errors by the median positive SE of the dataset; leave all others as they are."""
    ye = ye.copy()
    pos_all = ye.loc[ye["se"] > 0, "se"]
    for _, g in ye.groupby("dataset"):
        pos = g.loc[g["se"] > 0, "se"]
        fl = pos.median() if len(pos) else pos_all.median()
        ye.loc[g.index[g["se"] <= 0], "se"] = fl
    return ye


def verdict(est, lo, hi, res):
    if lo > 0 and est >= res:
        return "resolved gain"
    if hi < 0 and -est >= res:
        return "resolved loss"
    if lo > 0 or hi < 0:
        return "detectable, below resolution"
    return "tied"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(label, ye, res, level=0.95):
        ye = ye[["dataset", "d", "se"]].reset_index(drop=True)
        a, b = dersimonian_laird(floor_se(ye), level=level), dersimonian_laird(zero_only(ye), level=level)
        rows.append({"contrast": label, "target_years": a["k"], "level": level, "resolution": res,
                     "est_plan": a["est"], "lo_plan": a["lo"], "hi_plan": a["hi"], "verdict_plan": verdict(a["est"], a["lo"], a["hi"], res),
                     "est_zero_only": b["est"], "lo_zero_only": b["lo"], "hi_zero_only": b["hi"],
                     "verdict_zero_only": verdict(b["est"], b["lo"], b["hi"], res),
                     "years_raised_by_plan_floor": int((floor_se(ye)["se"] > ye["se"]).sum()), "years_with_zero_se": int((ye["se"] <= 0).sum())})

    y = pd.read_csv(R / "summary48/year_effects.csv")
    for item, nm in (("reml", "Two-stage GBLUP"), ("rf", "Random forest"), ("gbm", "Gradient boosting"), ("mlp", "Multilayer perceptron"),
                     ("gxe_gbm", "Covariate LightGBM"), ("rn_ridge", "Covariate reaction-norm ridge"), ("dl_g", "Within-environment-loss network"),
                     ("R_STK_FW", "Stacking on the forward history"), ("Oracle", "Best method in hindsight")):
        add(f"Benchmark: {nm} − cell-level GBLUP", y[(y["item"] == item) & (y["metric"] == "spearman")], 0.0087)
    y = pd.read_csv(R / "headroom_sparse/year_effects.csv")
    p = pd.read_csv(R / "headroom_sparse/pooled.csv")
    for frac in (0.25, 0.5):
        res = float(p[(p["fraction"] == frac) & (p["contrast"] == "m1-m0") & (p["metric"] == "sp") & (p["range"] == "all48")]["resolution"].iloc[0])
        add(f"Sparse {int(100 * frac)} %: M×E − main-effect GBLUP", y[(y["fraction"] == frac) & (y["contrast"] == "m1-m0") & (y["metric"] == "sp")], res)
    res25 = float(p[(p["fraction"] == 0.25) & (p["contrast"] == "m1-m0") & (p["metric"] == "sp") & (p["range"] == "all48")]["resolution"].iloc[0])
    add("Sparse 25 %: learned environment correlation − M×E", y[(y["fraction"] == 0.25) & (y["contrast"] == "m2-m1") & (y["metric"] == "sp")], res25)
    y = pd.read_csv(R / "headroom_sparse2/year_effects.csv")
    v2 = json.load(open(R / "headroom_sparse2/verdict.json"))["decisions"]
    for k, nm in (("stk", "stacking with LightGBM"), ("dlres", "residual network"), ("ecmxe", "covariate environment correlation")):
        add(f"Sparse 25 %: {nm} − M×E", y[(y["fraction"] == 0.25) & (y["method"] == k) & (y["metric"] == "sp")], v2[k]["resolution"], v2[k]["holm_level"])
    y = pd.read_csv(R / "headroom_oldlines/year_effects.csv")
    vo = json.load(open(R / "headroom_oldlines/verdict.json"))["decisions"]
    po = pd.read_csv(R / "headroom_oldlines/pooled.csv")
    r0 = float(po[(po["contrast"] == "ol1-ol0") & (po["metric"] == "sp") & (po["range"] == "all")]["resolution"].iloc[0])
    add("Old lines: own-history residual", y[(y["contrast"] == "ol1-ol0") & (y["metric"] == "sp")], r0)
    add("Old lines: line × location history", y[(y["contrast"] == "H2") & (y["metric"] == "sp")], vo["H2"]["resolution"], vo["H2"]["holm_level"])
    y = pd.read_csv(R / "headroom_secondary/year_effects.csv")
    add("Same-season silking dates", y[y["contrast"] == "H1"], json.load(open(R / "headroom_secondary/verdict.json"))["resolution"])
    y = pd.read_csv(R / "reanalysis/year_effects.csv")
    vr = json.load(open(R / "reanalysis/verdict.json"))["verdict"]["clac"]
    add("CLAC − cell-level GBLUP", y[(y["method"] == "clac") & (y["metric"] == "sp")], vr["resolution"], vr["holm_level"])
    y = pd.read_csv(R / "clac_decomp/year_effects.csv")
    vd = json.load(open(R / "clac_decomp/verdict.json"))["verdict"]
    for k, nm in (("loss_D4_linear_kernel", "linear kernel"), ("loss_D5_single_main_effect", "single main effect"),
                  ("loss_D1_equal_weights", "equal weights"), ("loss_D3_no_cleaning", "no phenotype cleaning"), ("loss_D2_no_spatial", "no spatial adjustment")):
        add(f"CLAC ablation, loss: {nm}", y[y["contrast"] == k], vd[k]["resolution"], vd[k]["holm_level"])
    y = pd.read_csv(R / "transfer_arc/year_effects.csv")
    y = y[(y["metric"] == "spearman") & (y["scope"] == "all")]
    pt = pd.read_csv(R / "transfer_arc/pooled.csv")
    pt = pt[(pt["metric"] == "spearman") & (pt["scope"] == "all")].set_index("range")["resolution"]
    add("Kernel transfer: all six datasets", y, pt["all48"])
    for d in ("G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"):
        add(f"Kernel transfer: {d}", y[y["dataset"] == d], pt[d])
    T = pd.DataFrame(rows)
    T["verdict_changed"] = T["verdict_plan"] != T["verdict_zero_only"]
    T["abs_change_est"] = (T["est_plan"] - T["est_zero_only"]).abs()
    T.to_csv(OUT / "table.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[["contrast", "target_years", "est_plan", "lo_plan", "hi_plan", "verdict_plan", "est_zero_only", "lo_zero_only", "hi_zero_only",
             "verdict_zero_only", "years_raised_by_plan_floor"]].round(4).to_string(index=False))
    print("verdicts changed:", int(T["verdict_changed"].sum()), "of", len(T), "| largest change in estimate:", round(float(T["abs_change_est"].max()), 4))

    # ---- CLAC ablations without the six environments that have no final CLAC prediction
    E = pd.read_csv(R / "clac_decomp/per_env.csv")
    assert set(NO_FINAL_CLAC) <= set(E["env"])
    E2 = E[~E["env"].isin(NO_FINAL_CLAC)]
    rng = np.random.default_rng(20261002)
    N = int(E2["n"].sum())
    res = 3 / np.sqrt(N)
    out = {"environments": int(len(E2)), "cells": N, "resolution": res, "contrasts": {}}
    for k, (a, b) in {"loss_D4_linear_kernel": ("A_full", "D4_linear_kernel"), "loss_D5_single_main_effect": ("A_full", "D5_single_main_effect"),
                      "loss_D1_equal_weights": ("A_full", "D1_equal_weights"), "loss_D3_no_cleaning": ("A_full", "D3_no_cleaning"),
                      "loss_D2_no_spatial": ("A_full", "D2_no_spatial"), "A_full_vs_cell_reml": ("A_full", "ctrl")}.items():
        ye = []
        for Y, g in E2.groupby("target"):
            v = (g[a] - g[b]).to_numpy()
            ye.append({"dataset": "G2F", "d": float(v.mean()), "se": float(v[rng.integers(0, len(v), (2000, len(v)))].mean(1).std(ddof=1))})
        level = vd[k]["holm_level"] if k in vd else 0.95
        r = dersimonian_laird(floor_se(pd.DataFrame(ye)), level=level)
        out["contrasts"][k] = {"est": r["est"], "lo": r["lo"], "hi": r["hi"], "level": level, "verdict": verdict(r["est"], r["lo"], r["hi"], res)}
    json.dump(out, open(OUT / "ablations_without_six_environments.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
