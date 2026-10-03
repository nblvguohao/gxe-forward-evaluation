"""E1 (docs/prereg_sequel_wave2a_2026-09-28.md §1, §5): rerun GEFormer's gate-2 final refits with each
study's best parameters and store the test-year prediction after every epoch.

Same setup as scripts/gate2_geformer.py --role final: own = all pre-test years for 100 epochs; fair = all
pre-test years for E* epochs (E* = best epoch of the study's best trial on the validation year). Only the
per-epoch storage is new. Nothing is scored here; these runs do not replace the gate-2 numbers."""
import argparse
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
import torch

from dartgxe.baselines.common import git_commit, load_folds
from dartgxe.external.geformer import Data, env_tensors, epoch_train, import_geformer, make_args, planting_dates, predict
from dartgxe.features.fit import load_geno
from dartgxe.paths import RESULTS, processed

EPOCHS = 100
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True, choices=["F2024m", "F2022m"])
ap.add_argument("--protocol", required=True, choices=["own", "fair"])
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
dev = "cuda"
t0 = time.time()
fold = next(load_folds(a.scenario))
geno = load_geno()
daily = pd.read_parquet(processed("g2f") / "env_daily.parquet")
W, M, how, envs = env_tensors(daily, sorted(fold.cells["env"].unique()), planting_dates(os.environ["G2F_RAW"]))
keep = set(envs)
nan_envs = {e for e, v in how.items() if v == "nan_in_window"}
pre = fold.rows("train", "val", "refit_extra")
pre = pre[pre["env"].isin(keep)].reset_index(drop=True)
te = fold.rows("test")
te = te[te["env"].isin(keep)].reset_index(drop=True)
bad = sorted((set(pre["env"]) | set(te["env"])) & nan_envs)
if bad:
    raise RuntimeError(f"weather window contains NaN in {len(bad)} environments, e.g. {bad[:3]}")
hyb = sorted(fold.cells["genotype"].unique())
hpos = {h: i for i, h in enumerate(hyb)}
epos = {e: i for i, e in enumerate(envs)}
X = geno.X[[geno.index[h] for h in hyb]].astype(np.float32)
F = X[[hpos[h] for h in sorted(set(pre["genotype"]))]]
mode = np.stack([(F == k).sum(0) for k in (0, 1, 2)]).argmax(0)
Xi = np.where(X < 0, mode, X)  # training-fold mode, as gate 2


def idx(rows):
    return (torch.tensor([hpos[h] for h in rows["genotype"]], device=dev),
            torch.tensor([epos[e] for e in rows["env"]], device=dev),
            torch.tensor(rows["y"].to_numpy(np.float32), device=dev))


study_dir = RESULTS / "gate2" / f"{a.scenario}_{a.protocol}"
storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(study_dir / "study.journal")))
study = optuna.load_study(study_name=f"{a.scenario}_{a.protocol}", storage=storage)
done = [t for t in study.trials if t.state.name == "COMPLETE"]
if len(done) < 8:
    raise RuntimeError(f"study has {len(done)} complete trials; gate 2 required 8")
bt = study.best_trial
hp = bt.params
n_ep = EPOCHS if a.protocol == "own" else max(int(bt.user_attrs["best_epoch"]), 1)

torch.manual_seed(a.seed); np.random.seed(a.seed)
data = Data(Xi, W, M, dev)
gi, ei, y = idx(pre)
tgi, tei, _ = idx(te)
GEF = import_geformer()
net = GEF(make_args(W.shape[2], hp), X.shape[1], W.shape[1]).to(dev)
opt = torch.optim.Adam(net.parameters(), hp["lr"])
rng = np.random.default_rng(a.seed)
frames = []
for ep in range(1, n_ep + 1):
    epoch_train(net, data, gi, ei, y, hp["batch"], opt, rng)
    frames.append(pd.DataFrame({"env": te["env"], "genotype": te["genotype"], "observed": te["y"], "epoch": ep,
                                "prediction": np.asarray(predict(net, data, tgi, tei), dtype=np.float32)}))
out = RESULTS / "a1" / "geformer_epochs"
out.mkdir(parents=True, exist_ok=True)
pd.concat(frames).to_parquet(out / f"{a.scenario}_{a.protocol}_seed{a.seed}.parquet", index=False)
meta = {"scenario": a.scenario, "protocol": a.protocol, "seed": a.seed, "best_params": hp, "epochs": n_ep,
        "study_best_value": bt.value, "study_best_epoch": bt.user_attrs.get("best_epoch"),
        "git_commit": os.environ.get("DARTGXE_COMMIT") or git_commit(), "gpu": torch.cuda.get_device_name(0),
        "seconds": round(time.time() - t0)}
(out / f"{a.scenario}_{a.protocol}_seed{a.seed}.json").write_text(json.dumps(meta, indent=1, default=float))
print("E1_GEFORMER_DONE", a.scenario, a.protocol, a.seed, n_ep)
