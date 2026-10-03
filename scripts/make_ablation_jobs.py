"""Job files for the NN ablations (N0 only for the new scenario; N1, N2, N3 everywhere)."""
import json

import pandas as pd

from dartgxe.paths import ROOT, splits_dir

SC = ["FYnew2016s", "FYnew2018", "FYnew2020", "F2022m", "CV1", "CV00", "CV0"]
VARIANTS = {"N1": {"use_ec": 0}, "N2": {"linear_g": 1}, "N3": {"use_ec": 0, "target": "center"}}
pend = ROOT / "jobs" / "pending"
pend.mkdir(parents=True, exist_ok=True)
n = 0
for sc in SC:
    folds = sorted(pd.read_parquet(splits_dir("g2f") / f"{sc}.parquet", columns=["fold"])["fold"].unique())
    todo = dict(VARIANTS)
    if sc == "FYnew2016s":
        todo["N0"] = {}
    for v, kw in todo.items():
        for f in folds:
            for s in range(5):
                job = {"phase": "ablation", "scenario": sc, "fold": int(f), "seed": s, "loss": "twohead_mse", **kw}
                (pend / f"abl__{sc}__{v}__f{f}__s{s}.json").write_text(json.dumps(job)); n += 1
print(n, "ablation jobs")
