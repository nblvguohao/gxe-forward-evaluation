"""WP2 real case (design §6): gpverdict-audit on the gate-2 GEFormer epoch panels (own protocol: epoch chosen by pooled
Pearson on the test year; fair protocol: chosen on the validation year, so only its final epoch is reported), with
the REML-GBLUP of gate 2 as the baseline. Descriptive demonstration; writes RESULTS/wp2/case_gate2_*.json."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "gpverdict_audit"))
from gpverdict_audit import audit  # noqa: E402

from dartgxe.paths import RESULTS  # noqa: E402

EP = RESULTS / "a1" / "geformer_epochs"
PR = RESULTS / "predictions" / "g2f"
out = RESULTS / "wp2"
out.mkdir(parents=True, exist_ok=True)
summary = {}
for sc in ("F2024m", "F2022m"):
    for proto in ("own", "fair"):
        d = pd.read_parquet(EP / f"{sc}_{proto}_seed147.parquet")
        cand = d.rename(columns={"env": "environment", "prediction": "predicted", "epoch": "candidate"})
        if proto == "fair":  # the fair protocol fixes the epoch on the validation year; report its final model only
            cand = cand[cand["candidate"] == cand["candidate"].max()]
            manifest = {"criterion": "within_spearman", "chosen": str(int(cand["candidate"].max())),
                        "selection": {"years": ["validation"]}, "report": {"years": ["test"]}}
        else:
            manifest = {"criterion": "pooled_pearson", "selected_on_report": True}
        base = pd.read_parquet(PR / sc / "B1r_gblup_reml" / "seed147.parquet").rename(
            columns={"env": "environment", "prediction": "predicted"})
        res = audit(cand, manifest, base, reps=100, B=300)
        json.dump(res, open(out / f"case_gate2_{sc}_{proto}.json", "w"), indent=1, default=float)
        summary[f"{sc}_{proto}"] = {k: {"verdict": v.get("verdict"), "message": v.get("message")}
                                    for k, v in res.items() if k.startswith("A")}
        print(sc, proto)
        for k, v in summary[f"{sc}_{proto}"].items():
            print(f"  {k:13s} {v['verdict']:8s} {v['message']}")
json.dump(summary, open(out / "case_gate2_summary.json", "w"), indent=1)
