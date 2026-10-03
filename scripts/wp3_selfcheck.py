"""WP3 self-check: score library methods as submissions with benchmark/score.py and compare with results/summary48
(same year effects; the pooled estimate can differ slightly because the bootstrap SEs, hence the weights, use other
random draws). Also checks that a submission missing cells is refused. Writes RESULTS/benchmark_check.json."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
import score  # noqa: E402

from dartgxe.paths import RESULTS  # noqa: E402

DATA = RESULTS / "benchmark"
splits = json.load(open(DATA / "splits.json"))
S48 = pd.read_csv(RESULTS / "summary48" / "pooled.csv")
out = {}
for method in ("reml", "rf", "gbm", "knn10", "mlp", "dl_g"):
    parts = []
    for d in splits:
        P = pd.read_parquet(DATA / f"predictions_{d}.parquet")
        P = P[P["method"] == method]
        if len(P):
            parts.append(P.drop(columns="method").assign(dataset=d))
    sub = pd.concat(parts, ignore_index=True)
    f = RESULTS / f"_sub_{method}.parquet"
    sub.to_parquet(f, index=False)
    res = score.main([str(f), "--data", str(DATA), "--out", str(RESULTS / f"_score_{method}.json")])
    r = res["vs_cell_gblup"]["spearman"]["all"]
    ref = S48[(S48["item"] == method) & (S48["metric"] == "spearman") & (S48["scope"] == "all48")].iloc[0]
    out[method] = {"benchmark_est": r["est"], "summary48_est": float(ref["est"]), "diff": r["est"] - float(ref["est"]),
                   "years": r["years"], "summary48_years": int(ref["k"])}
    print(method, out[method], flush=True)
d0 = next(iter(splits))
P = pd.read_parquet(DATA / f"predictions_{d0}.parquet")
P = P[P["method"] == "cell_reml"].drop(columns="method").assign(dataset=d0).iloc[::2]
P.to_parquet(RESULTS / "_sub_partial.parquet", index=False)
try:
    score.main([str(RESULTS / "_sub_partial.parquet"), "--data", str(DATA), "--out", "/dev/null"])
    out["refuses_partial"] = False
except SystemExit as e:
    out["refuses_partial"] = True
    out["refusal_message"] = str(e)
json.dump(out, open(RESULTS / "benchmark_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print("WP3_CHECK_DONE")
