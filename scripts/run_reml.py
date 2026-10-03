"""REML-GBLUP predictions for the given scenarios (the standard-practice linear comparator of gate 2)."""
import argparse

from dartgxe.baselines.common import load_folds, pred_frame, write_predictions
from dartgxe.baselines.linear import fit_b1_reml
from dartgxe.features.fit import load_ec, load_geno

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", nargs="+", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[147, 1, 2])
a = ap.parse_args()
geno, ec = load_geno(), load_ec()
for sc in a.scenarios:
    frames, info = [], {}
    for fold in load_folds(sc):
        te, out, inf = fit_b1_reml(fold, geno, ec)
        frames += [pred_frame(te, out["B1r_gblup_reml"], "B1r_gblup_reml", sc, fold.fold, s) for s in a.seeds]
        info[f"fold{fold.fold}"] = inf["B1r_gblup_reml"]
        print(sc, fold.fold, {k: round(v, 4) if isinstance(v, float) else v for k, v in inf["B1r_gblup_reml"].items()}, flush=True)
    write_predictions(frames, "B1r_gblup_reml", sc, a.seeds, {"deterministic": True, "reml": info})
