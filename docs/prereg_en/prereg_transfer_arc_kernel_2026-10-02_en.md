# Pre-registration: transfer test of the arc-cosine kernel to the multi-crop forward benchmark (2026-10-02)

Written in English and committed before the model is implemented or fitted. Later changes are appended with a date, never edited in place.

**Purpose.** To test, outside CLAC and outside G2F, the component identified by the CLAC ablations.

**Known before writing.**
- CLAC ablations (`results/clac_decomp/`, `docs/prereg_clac_decomposition_2026-10-02.md`, Appendix A): inside CLAC's multivariate GBLUP, replacing the arc-cosine kernel by a linear kernel lowered within-environment Spearman correlation by 0.023 (99 % Holm interval [+0.002, +0.043]; G2F, five target years). No other component carried a detectable share.
- All results of the forward benchmark (48 target years, six datasets), in which `cell_reml` is the reference.
- In the second G2F analysis, a non-linear genetic stage did not beat ridge regression on external years.
- Not known: any result of an arc-cosine kernel in the cell-level model, in any dataset. No such model has been fitted.

## 1. Question

Does replacing the linear kernel of cell-level GBLUP (`cell_reml`) by an arc-cosine kernel improve within-environment ranking of lines in a new year, across the six datasets of the forward benchmark?

## 2. Model

`cell_arc` is `cell_reml` with one change, the genomic kernel. Everything else is identical: training cells are all cells of years before the target year; environment fixed effects; one variance ratio chosen by REML; no tuned parameter.

1. **Markers.** Z is the standardised marker matrix that `cell_reml` uses (`dartgxe.forward.panel.features`): means, standard deviations and imputation come from the training genotypes only. For GEM_IA, Z is the embedding of its genomic relationship matrix, centred and not scaled, as in `cell_reml`.
2. **Linear Gram matrix.** G = Z Z′ / c over the training and target genotypes together, where c is the mean diagonal element over the training genotypes.
3. **Arc-cosine kernel of order 1** (Cho and Saul 2009; `bWGR::EigenARC`): K_ij = (n_i n_j / π) [sin θ_ij + (π − θ_ij) cos θ_ij], with n_i = √G_ii and θ_ij = arccos(G_ij / (n_i n_j)). On a test matrix this formula agrees with `bWGR::EigenARC` (version 2.2.18, used by CLAC) to within 2 × 10⁻⁴ per element; the remaining difference is a constant in the bWGR code.
4. **Embedding.** K_ff = U Λ U′ for the training genotypes, keeping eigenvalues above 10⁻⁸ times the largest (the threshold of CLAC's `K2X`). Features are Φ_f = U Λ^½ for training genotypes and Φ_t = K_tf U Λ^−½ for target genotypes. A target genotype that is also a training genotype gets its training row.
5. **Fit.** The ridge of `dartgxe.baselines.linear.Ridge` on Φ_f at the cell level with environment fixed effects and REML, exactly as `cell_reml` on Z. Predictions are Φ_t β.

No phenotype of the target year enters any step. Marker data of target-year genotypes enter only through K_tf and their own norms, as they enter `cell_reml` through Z.

Differences from CLAC that are deliberate: CLAC centres but does not scale markers, fits a multivariate model across environments, and uses cleaned phenotypes. Here only the kernel changes, so that the contrast with `cell_reml` has one cause.

## 3. Data and scoring

- **Target years.** The 48 target years of the benchmark (`results/benchmark/splits.json`): G2F 5, NUST 15, URSN 11, ESWYT 12, GEM_IA 4, MU_SOY 1. No rehearsal year is scored; code checks use training years only (Section 5).
- **Cells.** The scorable environments of each target year, as in the benchmark.
- **Control.** `cell_reml` is refitted in the same run. It must agree with the benchmark's `cell_reml` predictions (per target year: median within-environment Spearman at least 0.999, minimum at least 0.99); otherwise the implementation is fixed before anything is scored.
- **Metric.** Within-environment Spearman correlation; the top-10 % selection differential is secondary.
- **Year effect.** Mean environment difference with environment-bootstrap standard error (B = 2,000, seed 20261002).
- **Pooling.** DerSimonian–Laird over target years with the zero-SE floor; resolution 3/√N, N = scored cells.

## 4. Hypothesis, verdict and use

**Primary contrast (one, so no multiplicity correction).** H1: `cell_arc` − `cell_reml`, all 48 target years, all genotypes.

**Verdict.** Resolved gain if the estimate is at least 3/√N and the 95 % interval excludes 0. Detectable but below resolution if only the interval condition holds. Tied otherwise. A resolved loss is reported as such.

**Descriptive, reported whatever the outcome.** The same contrast in each dataset, in the original and the independent set, for new genotypes only, with each dataset left out in turn, and for the selection differential.

**Use of the result, fixed now.**

| Outcome | What the manuscript says |
|---|---|
| H1 resolved gain | The arc-cosine kernel improves ranking of new lines across the benchmark; cell-level GBLUP with this kernel becomes the recommended default, with the per-dataset results shown |
| H1 not resolved, G2F alone a resolved gain (descriptive) | The kernel's gain is specific to G2F maize hybrids; the statement is limited to that dataset |
| H1 not resolved, G2F alone not a resolved gain | The kernel's share inside CLAC does not transfer to the simple cell-level model; the manuscript says that the kernel helped only within the CLAC pipeline |
| H1 resolved loss | Reported as a loss; the recommendation stays with the linear kernel outside CLAC |

In every case the result is reported in the main text in one short paragraph and one table row. No further variant (other kernels, kernel order, marker scaling) will be run after seeing the result.

## 5. Checks before any target year is fitted

`tests/test_cellarc.py`:
1. The kernel function reproduces a reference matrix computed with `bWGR::EigenARC` (tolerance 2 × 10⁻⁴).
2. The embedding reproduces the kernel: Φ_f Φ_f′ = K_ff and Φ_t Φ_f′ = K_tf.
3. Positive control: with the linear Gram matrix passed through the same embedding and fit, predictions equal those of `cell_reml` (within-environment Spearman 1 up to numerical error).
4. No leakage: changing the phenotypes of the target year, or of any later year, does not change the predictions; standardisation constants and c come from training genotypes only.

## 6. Directional predictions (before any fit)

- G2F: positive, between +0.01 and +0.03.
- The other five datasets (inbred lines or lines per se): smaller, between 0 and +0.01.
- H1 pooled over 48 years: positive and detectable, but below the resolution of about 0.009.

## 7. Computation

- **Machine.** RTX 4090 workstation. Both GPUs are in use by other jobs, so the run uses the CPU (`CUDA_VISIBLE_DEVICES=""`), double precision, 8 threads, each process under `systemd-run --user --scope -p MemoryMax=40G`.
- **Code.** `src/dartgxe/forward/cellarc.py`, `scripts/transfer_arc.py` (panels, then analysis; the analysis refuses to run until all 48 panels exist and the control check passes), `tests/test_cellarc.py`. Only new files.
- **Output.** `results/transfer_arc/`: per-environment table, year effects, pooled estimates, verdict, and run metadata (commit, versions, start and end times).

---

## Appendix A (2026-10-02, after scoring): run notes and results

Nothing above this line was changed. Results are in `results/transfer_arc/`; the 48 panels stay on the 4090 (`scratch_ideas_2026-09-29/transfer_arc/out/panels/`).

**Run notes.**
- Tests (`tests/test_cellarc.py`) passed on the 4090 before any target year was fitted.
- First run (commit 6887fe9): the dense ridge on the cells × features matrix was killed by the 40 GB cap at G2F 2022, after three panels (G2F 2016, 2018, 2020) had been written. Nothing had been scored.
- Change (commit d68057e): the same ridge is computed from genotype-by-environment counts (`AggRidge`); the estimator, REML and the kernel are unchanged. A sixth test checks equality with the dense ridge. On the three panels of the first run, the two computations agree to within 4 × 10⁻⁷ for `cell_arc` (prediction SD about 0.5) and exactly for the control. All 48 panels were then produced with commit d68057e, on CPU, 8 threads; the analysis was run once.
- Control: the refitted `cell_reml` agrees with the benchmark in every environment of every target year (worst within-environment Spearman 1.000).
- REML was at a grid edge in no panel.

**Results (within-environment Spearman, all genotypes).**

| Range | Estimate [95 % CI] | Resolution | Verdict |
|---|---|---|---|
| H1: all 48 target years | +0.0036 [−0.0002, +0.0075] | 0.0087 | tied |
| G2F (5) | +0.0342 [+0.0214, +0.0470] | 0.0130 | resolved gain (descriptive); gain in 5 of 5 years |
| NUST (15) | +0.0041 [−0.0007, +0.0090] | 0.0166 | tied |
| URSN (11) | +0.0032 [−0.0078, +0.0142] | 0.0991 | tied |
| ESWYT (12) | −0.0004 [−0.0058, +0.0050] | 0.0201 | tied |
| GEM_IA (4) | −0.0057 [−0.0325, +0.0211] | 0.0360 | tied |
| MU_SOY (1) | +0.0056 [−0.0097, +0.0210] | 0.0692 | tied |
| Original set (31) | +0.0069 [+0.0019, +0.0119] | 0.0102 | detectable, below resolution |
| Independent set (17) | −0.0011 [−0.0069, +0.0047] | 0.0170 | tied |

Other descriptive results: new genotypes only, all 48 years +0.0064 [+0.0014, +0.0114] (detectable, below resolution); G2F selection differential +0.083 [+0.049, +0.116] SD; leaving one dataset out, the pooled estimate ranges from +0.0016 (without G2F) to +0.0056.

**Outcome (Section 4).** "H1 not resolved, G2F alone a resolved gain (descriptive)": the manuscript limits the kernel statement to G2F maize hybrids.

**Predictions against outcomes (Section 6).** G2F +0.01 to +0.03: the estimate (+0.034) is just above the range. Other datasets 0 to +0.01: right for NUST, URSN and MU_SOY; ESWYT and GEM_IA are slightly negative, all tied. Pooled: positive, detectable and below resolution was predicted; the estimate is positive and below resolution, but its interval includes zero.

**Not done.** No further variant was run.


---

## Appendix B (2026-10-03): corrections after an independent recomputation

The independent recomputation rebuilt the per-environment values from all 48 panels (1,181 environments, largest difference 10⁻¹⁶), confirmed the control against the benchmark and reproduced every verdict. Nothing above this line was changed.

1. **Correction to Section 2, step 4.** The remark "(the threshold of CLAC's `K2X`)" is wrong. The pipeline's own `K2X` keeps eigenvalues above 0.1 in absolute value; 10⁻⁸ is the default of `bWGR::K2X`. The model was run as specified in the plan (10⁻⁸ times the largest eigenvalue); only the remark is corrected.
2. **Pooling rule.** "The zero-SE floor" of Section 3 is the rule of the 2026-09-28 amendment, max(SE, median positive SE of the dataset), which raised 22 of the 48 year SEs for H1. With a floor on zero SEs only: H1 +0.0038 [−0.0001, +0.0077], G2F +0.0322 [+0.0184, +0.0459]; no verdict changes (`results/floor_sensitivity/table.csv`).
3. **H1 is at the boundary of "tied".** Its lower limit is −0.0002. Under other reasonable choices (25 genotypes per environment in URSN instead of the benchmark's 10, together with a floor on zero SEs only) the interval excludes zero and the verdict would read "detectable, below resolution". Under no choice examined is it a resolved gain. The manuscript states this.
4. **New genotypes only.** The `judgement` column of `pooled.csv` for scope "new" uses the resolution of all genotypes and is not valid; the Supplement reports these estimates without verdicts.
