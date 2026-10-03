"""Linear side of docs/prestudy_nn_vs_gblup_2026-09-26.md: L1 everywhere; B1/B3 on the new scenario."""
import argparse
import time

from dartgxe.baselines.common import load_folds, pred_frame, write_predictions
from dartgxe.baselines.linear import fit_b1, fit_b1z, fit_b3
from dartgxe.features.fit import load_ec, load_geno

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", nargs="+", required=True)
ap.add_argument("--with_baselines", nargs="*", default=[])
a = ap.parse_args()
geno, ec = load_geno(), load_ec()
for sc in a.scenarios:
    frames, info = {}, {}
    for fold in load_folds(sc):
        t0 = time.time()
        te, out, inf = fit_b1z(fold, geno, ec)
        if sc in a.with_baselines:
            _, o1, i1, _ = fit_b1(fold, geno, ec); out.update(o1); inf.update({k: i1["B1"] for k in o1})
            _, o3, i3 = fit_b3(fold, geno, ec); out.update(o3); inf.update(i3)
        for m, p in out.items():
            frames.setdefault(m, []).extend(pred_frame(te, p, m, sc, fold.fold, s) for s in range(5))
            info.setdefault(m, {})[f"fold{fold.fold}"] = inf[m]
        print(sc, fold.fold, round(time.time() - t0), "s", flush=True)
    for m, fr in frames.items():
        write_predictions(fr, m, sc, range(5), {"deterministic": True, "tuning": info[m], "ablation": True})
print("LINEAR_DONE", flush=True)
