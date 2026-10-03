"""Command line: gpverdict-audit candidates.csv --manifest manifest.json [--baseline baseline.csv] --out audit.json"""
import argparse
import json

from .core import audit
from .report import render_html, to_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidates")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--min-genotypes", type=int, default=10)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="audit.html", help="audit.html or audit.json")
    a = ap.parse_args(argv)
    res = audit(a.candidates, json.load(open(a.manifest)), a.baseline, a.frac, a.min_genotypes, a.reps, seed=a.seed)
    open(a.out, "w").write(render_html(res) if a.out.endswith((".html", ".htm")) else to_json(res))
    for key in ("A1_overlap", "A2_criterion", "A3_source", "A4_optimism", "A5_baseline"):
        if key in res:
            print(f"{key:14s} {res[key]['verdict']:8s} {res[key]['message']}")
    print(res["summary"])


if __name__ == "__main__":
    main()
