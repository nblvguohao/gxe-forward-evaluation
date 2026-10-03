"""Build data/processed/g2f from the 2024-release sources (see src/dartgxe/data/g2f.py)."""
import argparse
import json
from pathlib import Path

from dartgxe.data.g2f import build
from dartgxe.paths import processed

SRC = "<workstation>/lgh_jobs/trustbio_gxe_sota_bench_20260921/data/processed/g2f"

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=SRC, help="dir with phenotype/environment/genotype parquet")
ap.add_argument("--vcf", default=None, help="complete raw VCF; if absent, calls are recovered from genotype.parquet")
ap.add_argument("--vcf-partial", default=None, help="a partial raw VCF used only to verify the recovery")
ap.add_argument("--raw", default=None, help="v2: directory with the raw CyVerse files; builds everything from them")
a = ap.parse_args()
if a.raw:
    from dartgxe.data.g2f import build_raw
    rep = build_raw(Path(a.raw), processed("g2f"))
    print(json.dumps({k: v for k, v in rep.items() if k != "inputs"}, indent=1))
    raise SystemExit
src = Path(a.src)
rep = build(src / "phenotype.parquet", src / "environment.parquet", Path(a.vcf) if a.vcf else None,
            processed("g2f"), geno_src=src / "genotype.parquet",
            vcf_partial=Path(a.vcf_partial) if a.vcf_partial else None)
print(json.dumps({k: v for k, v in rep.items() if k != "inputs"}, indent=1))
