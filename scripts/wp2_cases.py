"""WP2 real cases (design §6), descriptive: gpverdict-audit on
 (1) GE-BiFormer epoch panels of wave 2a/A1 (own: epoch chosen by the lowest test-year Huber loss, its code's rule;
     own_clean: same with fit-set-only standardisation; fair: epoch fixed on the validation year, final model only),
     baseline = gate-2 REML-GBLUP (same seed number is not shared, so seed 1 of the REML run is used throughout);
 (2) B3 forward panels as a clean pipeline: candidates = the ten two-stage methods + cell_reml, selection set = the
     forward panels of the earlier predictable years, criterion = within-environment Spearman (rule R_FW), report =
     the target year, baseline = cell_reml. Expected: A1-A4 PASS.
Writes RESULTS/wp2/cases/*.json and cases_summary.json."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "gpverdict_audit"))
from gpverdict_audit import audit  # noqa: E402
from gpverdict_audit.core import candidate_table, load, pick  # noqa: E402

from dartgxe.paths import RESULTS  # noqa: E402

OUT = RESULTS / "wp2" / "cases"
OUT.mkdir(parents=True, exist_ok=True)
REN = {"env": "environment", "prediction": "predicted", "pred": "predicted", "y": "observed", "epoch": "candidate",
       "method": "candidate"}
summary = {}


def keep(name, res):
    json.dump(res, open(OUT / f"{name}.json", "w"), indent=1, default=float)
    summary[name] = {k: v.get("verdict") for k, v in res.items() if k.startswith("A")}
    summary[name]["A4_optimism_value"] = res["A4_optimism"].get("optimism")
    print(name, summary[name], flush=True)


# (1) GE-BiFormer
EPOCHS = {("F2022m", "fair", 1): 6, ("F2022m", "fair", 2): 1, ("F2022m", "fair", 42): 1, ("F2022m", "own", 1): 3,
          ("F2022m", "own", 2): 4, ("F2022m", "own", 42): 2, ("F2022m", "own_clean", 1): 5, ("F2022m", "own_clean", 2): 2,
          ("F2022m", "own_clean", 42): 3, ("F2024m", "fair", 1): 5, ("F2024m", "fair", 2): 2, ("F2024m", "fair", 42): 3,
          ("F2024m", "own", 1): 50, ("F2024m", "own", 2): 5, ("F2024m", "own", 42): 2, ("F2024m", "own_clean", 1): 10,
          ("F2024m", "own_clean", 2): 62, ("F2024m", "own_clean", 42): 24}  # results/a1/epochs.csv (reported_epoch)
for (sc, proto, seed), e_rep in EPOCHS.items():
    d = pd.read_parquet(RESULTS / "a1" / "epochs" / f"{sc}_{proto}_seed{seed}.parquet").rename(columns=REN)
    base = pd.read_parquet(RESULTS / "wp2" / "baselines" / sc / "seed1.parquet").rename(columns=REN).drop(columns=["candidate"], errors="ignore")
    if proto == "fair":
        d = d[d["candidate"] == d["candidate"].max()]
        manifest = {"criterion": "within_spearman", "chosen": str(int(d["candidate"].max())),
                    "selection": {"years": ["validation"]}, "report": {"years": ["test"]}}
    else:
        manifest = {"criterion": "pooled_huber", "chosen": str(e_rep), "selected_on_report": True}
    if manifest["chosen"] not in set(d["candidate"].astype(str)):
        raise SystemExit(f"{sc} {proto} {seed}: reported epoch {manifest['chosen']} not in the epoch panel")
    keep(f"gebiformer_{sc}_{proto}_seed{seed}", audit(d, manifest, base, reps=200, B=300, seed=seed))

# (2) B3 forward panels, clean pipeline
COLS = ["env", "genotype", "y", "method", "pred"]
for d_name, Y in (("G2F", 2022), ("NUST", 2015), ("URSN", 2014), ("NUST", 2019)):
    summ = json.load(open(RESULTS / "b3" / "meta" / f"summary_{d_name}.json"))
    years = [y for y in summ["predictable_years"] if y <= Y]

    def panel(y):
        return pd.concat([pd.read_parquet(RESULTS / k / "forward" / f"{d_name}_{y}.parquet", columns=COLS)
                          for k in ("b3", "b3b")], ignore_index=True).query("method != ['rn_ridge', 'gxe_gbm', 'rn_ridge_real', 'gxe_gbm_real']")

    sel = pd.concat([panel(y).assign(set="selection") for y in years if y < Y], ignore_index=True)
    rep = panel(Y).assign(set="report")
    cand = pd.concat([sel, rep], ignore_index=True).rename(columns=REN)
    base = rep[rep["method"] == "cell_reml"].rename(columns=REN).drop(columns=["candidate", "set"])
    Csel, _ = candidate_table(load(sel.rename(columns=REN)).assign(set="report"))
    chosen = pick(Csel, "within_spearman")
    manifest = {"criterion": "within_spearman", "chosen": chosen,
                "selection": {"years": [str(y) for y in years if y < Y]}, "report": {"years": [str(Y)]}}
    keep(f"b3_{d_name}_{Y}", audit(cand, manifest, base, reps=200, B=300))
json.dump(summary, open(OUT / "cases_summary.json", "w"), indent=1, default=float)
print("WP2_CASES_DONE")
