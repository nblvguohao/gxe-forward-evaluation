"""Minimum detectable effects for the tied contrasts of the primary analysis (within-environment Spearman).

No model is fitted. For each contrast the pooled standard error is recovered from the stored interval,
SE = (hi - lo) / (2 z_level), and the minimum detectable effect at 80 % power and two-sided 5 % is (1.960 + 0.842) SE.
The table also states whether the upper interval limit is below the resolution threshold, i.e. whether a resolved
gain is excluded.

Run from the repository root:  python3 scripts/mde_ties.py  ->  results/mde_ties/mde.csv"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import norm

R = Path("results")
OUT = R / "mde_ties"
RES48 = 0.0087  # 3/sqrt(118,438), all 48 target years; subsets have fewer cells and hence a larger threshold


def csv_ci(path, cond):
    d = pd.read_csv(R / path)
    for k, v in cond.items():
        d = d[d[k].astype(str) == str(v)]
    assert len(d) == 1, (path, cond, len(d))
    r = d.iloc[0]
    return float(r["est"]), float(r["lo"]), float(r["hi"]), int(r["k"]), float(r["resolution"]) if "resolution" in d else RES48


def js_ci(path, *keys):
    x = json.load(open(R / path))
    for k in keys:
        x = x[k]
    lo, hi = (x["ci"] if "ci" in x else (x["lo"], x["hi"]))
    return float(x["est"]), float(lo), float(hi), int(x["k"]), float(x.get("holm_level", 0.95)), x.get("resolution")


def main():
    rows = []

    def put(label, reference, est, lo, hi, k, level, res):
        se = (hi - lo) / (2 * norm.ppf(0.5 + level / 2))
        rows.append({"contrast": label, "reference": reference, "target_years": k, "est": est, "lo": lo, "hi": hi,
                     "level": level, "se": se, "mde80": (norm.ppf(0.975) + norm.ppf(0.80)) * se, "resolution": res,
                     "resolved_gain_excluded": hi < res})

    for item, label in (("rn_ridge", "Covariate reaction-norm ridge"), ("dl_g", "Within-environment-loss network"),
                        ("R_STK_FW", "Stacking on the forward history")):
        est, lo, hi, k, res = csv_ci("summary48/pooled.csv", {"item": item, "metric": "spearman", "scope": "all48"})
        put(label, "cell_reml", est, lo, hi, k, 0.95, res)
    est, lo, hi, k, res = csv_ci("headroom_oldlines/pooled.csv", {"contrast": "ol1-ol0", "metric": "sp", "range": "all"})
    put("Own-history residual for old lines", "cell_reml", est, lo, hi, k, 0.95, res)
    sx = json.load(open(R / "headroom_secondary/verdict.json"))
    put("Same-season silking dates", "cell_reml", sx["H1"]["est"], sx["H1"]["lo"], sx["H1"]["hi"], sx["H1"]["k"], 0.95, sx["resolution"])
    v = json.load(open(R / "headroom_sparse/verdict.json"))["main_m2_m1"]
    put("Sparse 25 %: learned environment correlation", "MxE GBLUP", v["est"], v["lo"], v["hi"], v["k"], 0.95, v["resolution"])
    for key, label in (("stk", "Sparse 25 %: stacking with LightGBM"), ("dlres", "Sparse 25 %: residual network"),
                       ("ecmxe", "Sparse 25 %: covariate environment correlation")):
        est, lo, hi, k, level, res = js_ci("headroom_sparse2/verdict.json", "decisions", key)
        put(label, "MxE GBLUP", est, lo, hi, k, level, res)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "mde.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
