"""Run one stage-1 job (JSON file or CLI flags). Idempotent: skips if meta.json exists."""
import argparse
import json

from dartgxe.baselines.common import load_folds
from dartgxe.features.fit import load_ec, load_geno
from dartgxe.paths import RESULTS
from dartgxe.train.stage1 import Config, run

ap = argparse.ArgumentParser()
ap.add_argument("--job")
ap.add_argument("--scenario")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--loss")
ap.add_argument("--tau", type=float, default=1.0)
ap.add_argument("--eps", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--phase", default="grid", choices=["select", "grid", "ablation"])
ap.add_argument("--use_ec", type=int, default=1)
ap.add_argument("--linear_g", type=int, default=0)
ap.add_argument("--target", default="z")
a = ap.parse_args()
if a.job:
    j = json.load(open(a.job))
    for k, v in j.items():
        setattr(a, k, v)
cfg = Config(loss=a.loss, tau=a.tau, eps=a.eps, use_ec=bool(a.use_ec), linear_g=bool(a.linear_g), target=a.target)
fold = next(f for f in load_folds(a.scenario) if f.fold == a.fold)
base = RESULTS / "stage1" / a.phase
meta_dir = base / "runs" / a.scenario / cfg.name / f"seed{a.seed}_fold{a.fold}"
if (meta_dir / "meta.json").exists():
    print("exists", meta_dir)
else:
    m = run(cfg, fold, load_geno(), load_ec(), a.seed, RESULTS / "predictions" / "g2f" / a.scenario / cfg.name / "parts",
            meta_dir, predict_test=a.phase in ("grid", "ablation"))
    print(json.dumps({k: m[k] for k in ("method", "scenario", "fold", "seed", "best_val_spearman", "best_epoch", "seconds")
                      if k in m}, default=float))
