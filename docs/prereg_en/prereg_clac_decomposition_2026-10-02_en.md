# Pre-registration: decomposition of the CLAC gain over cell-level GBLUP (2026-10-02)

Written in English and committed before any ablation is run or scored. Later changes are appended with a date, never edited in place.

**Purpose.** To find which part of CLAC carries its gain, before deciding how the CLAC result is reported.

**Known before writing.** The reanalysis results (`results/reanalysis/`, pre-registration `docs/prereg_reanalysis_published_2026-09-30.md`) have been seen:
- CLAC against `cell_reml`: +0.0596 within-environment Spearman, Holm 97.5 % interval [+0.0374, +0.0817], 5 target years, N = 51,666, resolution 0.0132.
- Its components, scored descriptively against `cell_reml`:
  - Model A (multi-environment GBLUP part): +0.055 [+0.023, +0.088];
  - Model B (single GBLUP with fixed effects): +0.014 [−0.017, +0.045].
- No ablation of Model A has ever been run.

## 1. What is decomposed

Within an environment, CLAC's prediction is 0.5 × (Model A + Model B). In Model A, the environment-mean model and the scaling by the predicted environment standard deviation are constant within an environment, so within-environment ranking depends only on the genomic part:

GPRED(target env) = Σ_t SI(t, target) · GEBV_t,

where GEBV_t is the genomic value for training environment t from a multivariate GBLUP (`MRR3`). That GBLUP is fitted to phenotypes that were outlier-filtered (|z| > 3 within environment set to missing) and spatially adjusted (`mkr2X` with secondary traits). It uses an arc-cosine kernel (`EigenARC`, reparametrised as X with XX′ = K). SI(t, target) weights training environment t by:
- same station: 2 × A;
- same state: 2 × A;
- relatedness: 0.1 × A;

where A is the expected accuracy between the training and target hybrid sets (`EigenAcc`).

## 2. Ablations

Each ablation changes one component of Model A's genomic part, reuses every other intermediate file of the existing CLAC run (`ra/Y<Y>/`), and is run for the five target years 2016, 2018, 2020, 2022 and 2024. Year 2023 is a rehearsal and is not scored.

| Id | Change | What it removes |
|---|---|---|
| A_full | none: GPRED from the existing run | — |
| D1 equal weights | SI(t, target) = 1 / n_train for all t | location, state and relatedness weighting of training environments |
| D2 no spatial adjustment | MRR3 refitted to outlier-filtered raw environment means of YLD (no `mkr2X` adjustment) | spatial adjustment using secondary traits |
| D3 no phenotype cleaning | MRR3 refitted to raw environment means of Yield_Mg_ha (no outlier filter, no spatial adjustment) | outlier filtering and spatial adjustment |
| D4 linear kernel | X rebuilt from a linear (VanRaden-type) kernel on the same QC'd markers (`gen.RData`); MRR3 refitted to the same adjusted phenotypes | the arc-cosine kernel |
| D5 single main effect | one GBLUP (arc-cosine X, `emRR` in bWGR) fitted to each hybrid's mean of within-environment-centred adjusted phenotypes over all training environments; prediction = that single GEBV | multi-environment structure and environment weighting together |

Other settings:
- **Unchanged inputs.** All ablations use the same input files (years before Y only), random seeds and patches as the reanalysis.
- **Scope.** No other variant will be run, and none will be added after seeing results.

## 3. Scoring (identical to the reanalysis)

- **Cells and metric.** Common cells with `cell_reml` in G2F scored environments with at least 25 genotypes; within-environment Spearman.
- **Year effect.** The mean environment difference, with environment-bootstrap SE (B = 2,000, seed 20261002).
- **Pooling.** DerSimonian–Laird over the five years, with the zero-SE floor; resolution 3/√N.
- **Contrasts.**
  - **Primary, for each D_k:** loss_k = A_full − D_k, i.e. how much ranking is lost when the component is removed.
  - **Descriptive:** each D_k against `cell_reml`.
- **Holm correction** over the five primary contrasts.
- **Verdict per component.** "Carries a resolved share of the gain" if loss_k ≥ 3/√N and its Holm interval excludes 0. "Detectable, below resolution" if only the interval condition holds. Otherwise "no detectable share".

## 4. Directional predictions (before any run)

| Ablation | Predicted loss_k |
|---|---|
| D5 single main effect | the largest: most of Model A's +0.055; it removes both the multi-environment structure and the weighting |
| D1 equal weights | resolved, +0.02 to +0.04; location and state weighting is a form of G×location information, and marker G×location (I-B) gave +0.012 in the forward benchmark |
| D2 no spatial adjustment | small to moderate, 0 to +0.02 |
| D3 no phenotype cleaning | at least that of D2 |
| D4 linear kernel | small, about 0 |

## 5. Use of the results

1. **If at least one component carries a resolved share.** CLAC goes into the main text together with the decomposition. The finding is interpreted in terms of the information each component uses: location-specific training data (weighting), cleaner phenotypes (spatial adjustment), or the kernel.
2. **If no component carries a resolved share alone but A_full is resolved against `cell_reml`.** CLAC stays in the main text as an observed gain whose source is distributed. The decomposition goes to the Supplement.
3. **In both cases.** All ablations are reported, including negative ones. The decision on main text versus Supplement is the authors'.

## 6. Computation

- **Machine.** RTX 4090 workstation, CPU only (R 4.x).
- **Resources.** Two BLAS threads; each process wrapped in `systemd-run --user --scope -p MemoryMax=40G`; one MRR3 fit at a time.
- **Files.** New working directory `ra_decomp/Y<Y>/` with read-only links to the existing intermediate files; the existing `ra/` directories are not modified.
- **Code.** `scripts/clac_decomp/ablate.R` (ablations) and `scripts/clac_decomp/score.py` (scoring). Results go to `results/clac_decomp/`.

---

## Appendix A (2026-10-02, after scoring): results, checks and notes

Nothing above this line was changed. Scoring was run once (`scripts/clac_decomp/score.py`); the output is in `results/clac_decomp/`.

**Checks before scoring (no target-year phenotype used).**
- A_full reproduces the genomic part of the existing run: within-environment Spearman correlation with `GM_models_PredVar.csv` is 1.000000 in every environment of all six years (including the rehearsal year 2023).
- Every ablation file has the same cells as A_full, with no missing values and no constant environment.
- D2 is not identical to A_full (largest absolute difference 0.02–0.12), but nearly so: within-environment Spearman correlation at least 0.9991. The reason is in the rebuild itself. The spatial adjustment of the original script was applied in 35 of 51, 45 of 108, 72 of 165, 95 of 217 and 120 of 272 training environments (targets 2016 to 2024); in the others the script's own fall-backs used the raw phenotype (`results/clac_decomp/spatial_adjustment_counts.csv`). Where applied, it removed little variance (2022: median 1 %, maximum 10 % of yield variance). D2 therefore tests a weak adjustment.

**Results (five target years, 53,638 common cells, resolution 0.0130).** The cell count differs from the 51,666 of the reanalysis because the common cells are defined over a different set of methods.

| Contrast | Estimate | Holm interval (level) | Verdict |
|---|---|---|---|
| loss D4, linear kernel | +0.0227 | [+0.0022, +0.0431] (99 %) | carries a resolved share of the gain; loss in 5 of 5 years |
| loss D5, single main effect | +0.0324 | [−0.0203, +0.0851] (98.75 %) | no detectable share; year effects −0.017 to +0.103 |
| loss D2, no spatial adjustment | −0.0000 | [−0.0002, +0.0002] (98.3 %) | no detectable share |
| loss D3, no phenotype cleaning | +0.0005 | [−0.0038, +0.0047] (97.5 %) | no detectable share |
| loss D1, equal weights | −0.0008 | [−0.0169, +0.0152] (95 %) | no detectable share |

Descriptive, against `cell_reml` (95 %): A_full +0.053 [+0.024, +0.082]; with a linear kernel +0.029 [+0.006, +0.052]; single main effect +0.020 [−0.008, +0.047].

**Predictions against outcomes (Section 4).**
- D5 largest: the point estimate is the largest, but the loss is not detectable.
- D1 resolved at +0.02 to +0.04: wrong; the loss is about zero.
- D2 and D3 small: right.
- D4 about zero: wrong; it is the only resolved loss.

**Consequence (Section 5, rule 1).** One component carries a resolved share, so CLAC and its ablations are reported in the main text; the interpretation is the kernel, not location-specific information. The decision on main text versus Supplement remains the authors'.

**Not done.** No variant was added after seeing the results.


---

## Appendix B (2026-10-03): corrections after an independent recomputation

An independent recomputation from the raw prediction files (separately written code) reproduced every per-environment value (129 environments, largest difference 10⁻¹⁶) and every verdict. It also found the following; nothing above this line was changed.

1. **Correction to Appendix A, first check.** A_full agrees with `GM_models_PredVar.csv` in 123 of the 129 scored environments, not in all of them. In six environments (NCH1_2016, ONH1_2016, ONH2_2016, GEH1_2020, GEH1_2022, ONH3_2024) that file has no value, because the environment-mean model of the pipeline had no covariates; the first check ignored missing values. The genomic part, which the ablations use, exists for these environments.
2. **Consequence for the cell set.** The ablations were scored on 53,638 cells, the CLAC reanalysis on 51,666; the difference of 1,972 cells is these six environments. The plan (Section 3) did not exclude them. Without them (`results/floor_sensitivity/ablations_without_six_environments.json`): loss D4 +0.0240, 99 % interval [+0.0048, +0.0431], resolution 0.0132; no verdict changes.
3. **Certainty of "no detectable share".** For D2 and D3 the Holm intervals exclude a share as large as the resolution. For D1 the upper limit (+0.0152) is just above the resolution (0.0130), and for D5 the interval is wide; for these two a share of that size is not excluded. The manuscript says so.
4. **Pooling rule.** "The zero-SE floor" of Section 3 is the rule of the 2026-09-28 amendment: a year's SE is set to max(SE, median positive SE of the dataset). It therefore raises every SE below the median, here 2 of 5 years per contrast, although no year had an SE of zero. With a floor on zero SEs only, loss D4 is +0.0226 [+0.0040, +0.0412] and no verdict changes (`results/floor_sensitivity/table.csv`).
5. **Checked and found in order.** D4 uses the same eigenvalue threshold (> 0.1) as the pipeline's own `K2X`.
