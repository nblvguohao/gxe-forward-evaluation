"""Write stage-1 job files to jobs/pending/ (spec 09 §9.3).

--phase select : every loss config, folds {0}, seeds {0,1}, validation only (spec 04 §4.5)
--phase grid   : the selected configs, every fold, seeds 0..4, test predictions
"""
import argparse
import json

from dartgxe.paths import ROOT, splits_dir

SCENARIOS = ["CV1", "CV0", "CV00", "F2022", "F2022m", "FYnew2018", "FYnew2020", "FYrep2019", "FYrep2021", "FYrep2023"]
# FYnew2022 is the same split as F2022m (train <=2019, val 2020, refit <=2021, test 2022); it is not run twice.
SELECT_CONFIGS = [{"loss": "mse"}, {"loss": "twohead_mse"}, {"loss": "msed"}, {"loss": "lambda_at_k"},
                  {"loss": "listnet", "tau": 0.5}, {"loss": "listnet", "tau": 1.0},
                  {"loss": "soft_spearman", "eps": 0.1}, {"loss": "soft_spearman", "eps": 1.0}]

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=["select", "grid"])
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
pend = ROOT / "jobs" / "pending"
pend.mkdir(parents=True, exist_ok=True)
n = 0
if a.phase == "select":
    for sc in SCENARIOS:
        for cfg in SELECT_CONFIGS:
            for seed in (0, 1):
                job = {"phase": "select", "scenario": sc, "fold": 0, "seed": seed, **cfg}
                name = f"select__{sc}__{cfg['loss']}_{cfg.get('tau', cfg.get('eps', ''))}__s{seed}.json"
                (pend / name).write_text(json.dumps(job)); n += 1
else:
    sel = json.load(open(ROOT / "results" / "stage1" / "selection.json"))
    import pandas as pd
    for sc in SCENARIOS:
        folds = sorted(pd.read_parquet(splits_dir("g2f") / f"{sc}.parquet", columns=["fold"])["fold"].unique())
        for cfg in sel[sc]["grid_configs"]:
            for f in folds:
                for seed in range(a.seeds):
                    job = {"phase": "grid", "scenario": sc, "fold": int(f), "seed": seed, **cfg}
                    name = f"grid__{sc}__{cfg['loss']}_{cfg.get('tau', cfg.get('eps', ''))}__f{f}__s{seed}.json"
                    (pend / name).write_text(json.dumps(job)); n += 1
print(n, "jobs written to", pend)
