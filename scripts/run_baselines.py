"""Run B0-B3 and B5 (closed form) for the given scenarios; writes prediction files + meta.json."""
import argparse
import time

from dartgxe.baselines.common import load_folds, pred_frame, write_predictions
from dartgxe.baselines.linear import fit_b1, fit_b2, fit_b3
from dartgxe.features.fit import load_ec, load_geno

SEEDS = range(5)  # deterministic methods: the same predictions are written under every seed for pairing

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", nargs="+", required=True)
ap.add_argument("--methods", nargs="+", default=["B1", "B3", "B2"])
a = ap.parse_args()
geno, ec = load_geno(), load_ec()
for sc in a.scenarios:
    frames, info, t0 = {}, {}, time.time()
    for fold in load_folds(sc):
        tf = time.time()
        te, out, inf, lam = fit_b1(fold, geno, ec)
        if "B3" in a.methods:
            _, o3, i3 = fit_b3(fold, geno, ec); out.update(o3); inf.update(i3)
        if "B2" in a.methods:
            _, o2, i2 = fit_b2(fold, geno, ec, lam); out.update(o2); inf.update(i2)
        for m, p in out.items():
            frames.setdefault(m, []).extend(pred_frame(te, p, m, sc, fold.fold, s) for s in SEEDS)
            info.setdefault(m, {})[f"fold{fold.fold}"] = inf.get(m, inf.get("B1"))
        print(f"{sc} fold {fold.fold}: {time.time() - tf:.0f}s", {k: round(v.get('val_spearman', float('nan')), 4) for k, v in inf.items()}, flush=True)
    for m, fr in frames.items():
        write_predictions(fr, m, sc, SEEDS, {"deterministic": True, "tuning": info[m], "seconds": round(time.time() - t0)})
