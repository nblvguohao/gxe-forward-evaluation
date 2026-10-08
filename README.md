# gxe-forward-evaluation

Code and analysis plans for a forward evaluation of genomic prediction under genotype-by-environment interaction.

Lv G, Gu L. *Genomic prediction for selection in a new year: a forward benchmark across 48 target years of maize, wheat and soybean trials.* Manuscript in preparation; not yet peer reviewed or published.

## Contents

- `src/`, `scripts/`, `tests/`: analysis code (forward benchmark, sparse testing, value of target-year phenotypes, re-runs of
  published methods, CLAC ablations, kernel transfer, sensitivity analyses). `src/dartgxe/forward/pool.py` holds the pooling
  (DerSimonian-Laird with a standard-error floor; Hartung-Knapp).
- `benchmark/score.py` and `results/benchmark/splits.json`: scoring script and target-year definitions of the forward benchmark
  (48 target years, six datasets). `results/benchmark/SHA256SUMS` lists the checksums of the benchmark files that
  `scripts/wp3_build_benchmark.py` builds from the public data.
- `results/`: summary tables (pooled estimates, verdicts, year effects) and per-environment scores; no phenotypes and no
  per-cell predictions. `python3 scripts/check_sequel_sources.py` checks the results reported in the main text (and the
  supplementary numbers it lists) against these tables, row by row (run from the repository root; exit code 1 if any row
  fails). `python3 scripts/supplement_tables.py` rebuilds the supplementary tables that are generated from these tables
  (Tables S3, S5-S7, S10, S11, S13-S15, S17-S20) as Markdown files in `results/supplement_tables/`.
- `results/figdata_tcj/`: source data of the figures; `scripts/make_figures_tcj.py` draws them (panel 1a also needs the
  benchmark cells built from the public data).
- `results/c_paper/amax_runs/*/code/`: code of the separately planned G2F analysis with external years (Section 3.5).
- `docs/prereg_en/` and `results/c_paper/workspace/prereg/`: the analysis plans (pre-registrations). `docs/prereg_en/MANIFEST.md`
  lists the commit hash and sha256 of each original.

Scripts that pool or rescore stored tables (for example `floor_sensitivity.py`, `revision_tcj_stats.py` except its new-hybrid
part, `resolution_scale_facts.py`, `mde_ties.py`, `network_boundary_seeds.py`, `threshold_sensitivity.py`, `supplement_tables.py`) run from the files in this
repository. Scripts
that fit models or score per-cell predictions need the public data below. Phenotypes, genotypes, per-cell predictions and
figures are not included; per-cell predictions can be regenerated with the scripts from the public data, or obtained from
the authors on request.

## Tests

`pytest tests` (packages as in `requirements.lock`). Seven test files (`test_cellarc`, `test_cellspec`, `test_oldlines`,
`test_reml`, `test_secondary`, `test_sparse`, `test_sparse2`) import PyTorch, and `test_forward` imports scikit-learn. On the
machine used to prepare this release, which had neither package, the other five files ran: 27 tests passed and 6 were
skipped; the eight files above were not run on this release.

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
