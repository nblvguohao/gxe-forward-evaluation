"""B4 LightGBM for the given scenarios (CPU, 8 threads). Writes prediction files + meta.json."""
import argparse
import time

from dartgxe.baselines.common import load_folds, pred_frame, write_predictions
from dartgxe.baselines.lgbm import fit_b4
from dartgxe.features.fit import load_ec, load_geno

SEEDS = range(5)
ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", nargs="+", required=True)
a = ap.parse_args()
geno, ec = load_geno(), load_ec()
for sc in a.scenarios:
    frames, info, t0 = [], {}, time.time()
    for fold in load_folds(sc):
        te, preds, inf = fit_b4(fold, geno, ec, SEEDS)
        frames += [pred_frame(te, p, "B4_lgbm", sc, fold.fold, s) for s, p in preds.items()]
        info[f"fold{fold.fold}"] = inf
        print(sc, fold.fold, round(time.time() - t0), "s", round(inf["val_spearman"], 4), flush=True)
    write_predictions(frames, "B4_lgbm", sc, SEEDS, {"deterministic": False, "tuning": info, "seconds": round(time.time() - t0)})
