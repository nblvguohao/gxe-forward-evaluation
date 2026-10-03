"""Per scenario, choose τ (listnet), ε (soft_spearman) and rank* from validation only (spec 04 §4.5).
Writes results/stage1/selection.json and selection_table.csv."""
import json

import pandas as pd

from dartgxe.losses.rank import RANKING
from dartgxe.paths import RESULTS

rows = []
for p in (RESULTS / "stage1" / "select" / "runs").glob("*/*/*/meta.json"):
    m = json.loads(p.read_text())
    c = m["config"]
    rows.append({"scenario": m["scenario"], "method": m["method"], "loss": c["loss"], "tau": c["tau"], "eps": c["eps"],
                 "seed": m["seed"], "val": m["best_val_spearman"], "best_epoch": m["best_epoch"], "seconds": m["seconds"]})
d = pd.DataFrame(rows)
tab = (d.groupby(["scenario", "method", "loss", "tau", "eps"])
       .agg(val=("val", "mean"), val_seed_sd=("val", "std"), n_seeds=("seed", "size"), epoch=("best_epoch", "mean"),
            seconds=("seconds", "mean")).reset_index())
tab.to_csv(RESULTS / "stage1" / "selection_table.csv", index=False)
sel = {}
for sc, t in tab.groupby("scenario"):
    t = t.set_index("method")
    ln = t[t["loss"] == "listnet"]["val"].idxmax()
    ss = t[t["loss"] == "soft_spearman"]["val"].idxmax()
    cands = {"msed": "S1_msed", "listnet": ln, "soft_spearman": ss, "lambda_at_k": "S1_lambda_at_k"}
    rank_star = max(cands.values(), key=lambda m: t.loc[m, "val"])
    cfgs = [{"loss": "mse"}, {"loss": "twohead_mse"}, {"loss": "msed"}, {"loss": "lambda_at_k"},
            {"loss": "listnet", "tau": float(t.loc[ln, "tau"])}, {"loss": "soft_spearman", "eps": float(t.loc[ss, "eps"])}]
    sel[sc] = {"rank_star": rank_star, "listnet": ln, "soft_spearman": ss, "grid_configs": cfgs,
               "val": {m: round(float(v), 4) for m, v in t["val"].items()}}
(RESULTS / "stage1" / "selection.json").write_text(json.dumps(sel, indent=1))
print(tab.pivot_table(index="method", columns="scenario", values="val").round(3).to_string())
print({sc: v["rank_star"] for sc, v in sel.items()})
