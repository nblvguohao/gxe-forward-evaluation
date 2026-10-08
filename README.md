# gxe-forward-evaluation

Code and analysis plans for a forward evaluation of genomic prediction under genotype-by-environment interaction.

Lv G, Gu L. *Which information improves within-trial ranking in a new year? A forward evaluation of genomic prediction under
genotype-by-environment interaction in public maize, wheat and soybean trials.* Manuscript in preparation; not yet peer reviewed or published.

## Contents

- `src/`, `scripts/`, `tests/`, `packages/`: analysis code.
- `benchmark/score.py` and `results/benchmark/splits.json`: scoring script and target-year definitions of the forward benchmark
  (48 target years, six datasets). `results/benchmark/SHA256SUMS` lists the checksums of the benchmark files that
  `scripts/wp3_build_benchmark.py` builds from the public data.
- `results/`: summary tables only (pooled estimates, verdicts, year effects). `scripts/check_sequel_sources.py` checks every
  number reported in the manuscript against these tables.
- `docs/prereg_en/` and `results/c_paper/workspace/prereg/`: the analysis plans (pre-registrations). `docs/prereg_en/MANIFEST.md`
  lists the commit hash and sha256 of each original.

Phenotypes, genotypes, per-cell predictions and figures are not included. Per-cell predictions can be regenerated with the
scripts from the public data below, or obtained from the authors on request.

## Data

All input data are public. Please obtain them from the original sources and follow their licences:

| Dataset | Crop | Source |
|---|---|---|
| Genomes to Fields (G2F) | maize hybrids | https://doi.org/10.25739/78mn-4394 |
| Northern Uniform Soybean Tests (NUST) | soybean | SoyBase NUST portal (https://www.soybase.org/); genotypes: European Variation Archive PRJEB86055 |
| URSN | spring wheat | https://doi.org/10.5061/dryad.wstqjq2z0 |
| ESWYT | spring wheat | https://doi.org/10.6084/m9.figshare.12609368 |
| GEM_IA | maize topcrosses | https://doi.org/10.5281/zenodo.7150254 |
| MU_SOY | soybean | https://doi.org/10.5061/dryad.z8w9ghxf9 |
| BRIWECS | winter wheat | https://doi.org/10.6084/m9.figshare.27910269 |

## Licence

For academic and other non-commercial use only: Creative Commons Attribution-NonCommercial 4.0 International (see `LICENSE`).
If you use this material, please cite the article above.
