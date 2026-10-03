English translation of the pre-registration document prereg_reanalysis_published_2026-09-30.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

# Pre-registration RA: reanalysis of three published G×E prediction methods under true forward prediction (2026-09-30)

Written before any reanalysis run (the feasibility pilot also comes after this file is committed). Not changed after commit; changes are only appended as dated sections. Basis: on 2026-09-30 the user decided that [journal-planning remark omitted], and that this analysis is responsible for two reanalyses (user's words: "OK, go ahead"); the user agreed to download the three code bases ("Agreed") and to install an R environment ("OK").

**Nature**: an independent reanalysis of published methods. The outcomes (G2F 2016–2024 yield) have been seen many times in this project. The new content of this check is the performance of these three methods under true forward prediction, which **has never been run**. Conclusions are written as "under this protocol", not as a refutation of the original papers.

## 0. Known information, stated as is

- In the forward benchmark, `cell_reml` (cell-level REML-GBLUP) was not exceeded by any method, including this project's low-rank EC reaction norm `rn_ridge` (80 genotype PCs plus 20×5 interaction terms). GEFormer and GE-BiFormer have already been reanalysed under an equivalent pipeline (main: gate2, A1).
- **The code of the three methods has been read (2026-09-30); this is the basis of the design**:
  - CLAC (G2F 2022 winner): environmental covariates enter only the models for the environment mean and standard deviation. They are constant within an environment and **do not affect the within-environment ranking**. The within-environment ranking comes from two parts: (i) multivariate GBLUP genomic values from each training environment, weighted by location code, state and kinship-based predictability (Model A); (ii) a GBLUP main effect with environment fixed effects (Model B). The final prediction is the equal-weight average of the two.
  - Lopez-Cruz et al. 2023 (Nat Commun), model "SNP+EC+SNPxEC": BGLR BRR fitted to SNP principal components, EC principal components, and the G⊙W interaction obtained by tensor eigendecomposition (retaining 97.5% of the variance). The original paper fitted the northern and southern regions separately, evaluated by leave-one-year-out, and included future years in training.
  - Fernandes et al. 2024 (TAG), LightGBM G(A)+E: in the original paper, CV0 is training on 2020 and validation on 2021, and the test hybrids are tested hybrids. It reports the across-environment pooled Pearson (+7%, against FA). BLUE and FA depend on the commercial software asreml.
- MAIZE-HUB's `historical_ecov.zip` contains only soil-layer files and **lacks the main table of 207 covariates described in the README**. So the historical-EC version instead uses the location means of the official G2F APSIM EC over the training years (§2.2).
- The SHA256 sums and commit IDs of the downloaded items are recorded on the 4090 in `<workstation>/dart-gxe/third_party/SHA256SUMS_reanalysis_2026-09-30`. None of the three repositories has a licence file; they are only imported at run time and are not redistributed.

## 1. Scenario and data

- **Dataset**: G2F, using this project's benchmark cell table, scored environments and `cell_reml` predictions (`results/benchmark/`). Raw files are taken from the 4090 `data/raw/g2f2024/` (2024 competition release, including the 2014–2023 training tables, 2024 test observed values, official APSIM EC, weather and VCF).
- **True forward prediction**: predictions for target year Y use only phenotypes from years before Y; phenotypes of Y do not enter any fitting, tuning, early stopping or feature construction.
- **Target years**:
  - CLAC, Lopez-Cruz: Y ∈ {2016, 2018, 2020, 2022, 2024}, the same as the benchmark;
  - Fernandes: Y ∈ {2022, 2024}. Its pipeline hard-codes the competition file structure: 2022 uses the submission mode of the original repository (training 2019–2021), and 2024 uses the G2F_2024 repository of the same authors.
- **Scored cells**: cells within the benchmark scored environments for which both the method and `cell_reml` have predictions (common cells); environments have at least 25 genotypes. The N and 3/√N of each method are computed on the common cells and written into an appended section after the feasibility pilot and before any scoring.

## 2. Methods (run the original authors' code as far as possible; change only paths, years and necessary substitutions)

### 2.1 CLAC (R; `third_party/CLAC/Script_CLAC_model.R`)
- Training files are changed to the raw plot tables, metadata, EC and VCF of the years before Y; the test template is the scored cells of Y. The other steps follow the original script: 35,000 random markers (seed 123), quality control, arc-cosine kernel, per-environment spatial correction, multivariate GBLUP (MRR3), selection index, Model B, equal-weight average.
- Output: `clac` (final), `clac_A`, `clac_B` (descriptive).
- Necessary substitution: where the column names of the 2024 release differ from the 2022 version, only a column-name mapping is applied, recorded in the list of deviations.

### 2.2 Lopez-Cruz reaction norm (R; `1_get_model_components.R` and `5_LYO_CV_model.R` from MAIZE-HUB `analysis.zip`)
- Model "SNP+EC+SNPxEC", BGLR BRR, nIter = 12,000, burnIn = 4,000 (as in the original paper); training = years before Y.
- **Necessary deviations**:
  - EC uses the official G2F APSIM EC (the original authors' own APSIM EC only goes to 2021);
  - Markers use the 2024 release VCF, processed by the rules of the original `2_LD_prune_genotypes.R`; if it cannot be run, the 2,425 markers of this project's data_v2 are used instead, written into the list of deviations;
  - No split into northern and southern regions (the new locations of 2022–2024 have no region assignment in the original work); all environments are fitted together.
- Two versions:
  - `lc_real`: the target year uses the EC actually measured in that year (the original authors' approach);
  - `lc_hist`: the target year uses the EC mean of that location over the training years; new locations use the mean over all training environments.
- **Memory rule (fixed)**: the tensor eigendecomposition is first run retaining 97.5% of the variance. If memory exceeds the limit of 48 GB, change in turn to 95% and 90%. This choice looks only at memory on the training data, not at any outcome, and is recorded in the appended section. If 90% still exceeds the limit, the method is recorded as "cannot be reproduced on this machine" and reported as is.

### 2.3 Fernandes LightGBM G(A)+E (Python + R; `third_party/Maize_GxE_Prediction`, `third_party/G2F_2024`)
- Use the final submission pipeline of the repository, i.e. the predictions for the 2022 test set; 2024 uses the pipeline of the G2F_2024 repository. Model, feature engineering (weather, soil, 15-dimensional SVD of EC, lagged yield, latitude/longitude bins), hyperparameters and seeds all follow the original repository.
- **Necessary substitutions**:
  - BLUE changed from asreml to lme4 with the same model structure: fixed effects Hybrid + Replicate, random effects Rep:Block, Range, Pass;
  - The FA baseline is not run; the control is uniformly `cell_reml`;
  - Data files are constructed from the raw tables on the 4090 in the format of the 2022 and 2024 releases.
- Output: `flgbm_gae` (G(A)+E), and `flgbm_g` (G only, descriptive).

## 3. Metrics and pooling

- **Primary metric**: within-environment Spearman. Also reported: within-environment Pearson, top-10% selection differential, and **the metric used in each original paper** (across-environment pooled Pearson; Fernandes also RMSE), to show the difference between "performance under the original paper's metric" and "within-environment performance".
- **Year effect**: the mean Δ across the scored environments of that year; SE from an environment bootstrap (B = 2000, seed 20260930). Pooling uses DerSimonian–Laird, with the same zero-SE floor as before (`dartgxe.forward.pool`); the Hartung–Knapp interval is also reported.
- **Resolution**: 3/√N, N = number of common scored cells for that method.

## 4. Decision

- **Tested hypotheses**: H_m: m − `cell_reml` (within-environment Spearman, two-sided), m ∈ {`clac`, `lc_real`, `flgbm_gae`}, with Holm correction across the three (in order of increasing p, using 98.33%, 97.5% and 95% intervals).
- **"Better than `cell_reml`"** requires both: pooled Δ ≥ 3/√N, and lower bound of the Holm-level CI > 0.
- **"Worse than `cell_reml`"** requires both: pooled Δ ≤ −3/√N, and upper bound of the Holm-level CI < 0.
- Other cases use the wording of the project protocol document §2.2: CI excludes 0 but |Δ| < 3/√N, write "statistically detectable, but below selection resolution"; otherwise write "not resolved".
- **Descriptive comparisons that must be reported** (they do not change the decision):
  - `lc_hist` − `lc_real`, i.e. the cost of using historical EC only;
  - The decomposition of `clac` into `clac_A` and `clac_B`;
  - `flgbm_g` versus `flgbm_gae`;
  - New and old hybrids reported separately;
  - For each method, the difference from `cell_reml` under the original paper's metric (pooled Pearson), side by side with the difference in within-environment Spearman.
- After NO-GO or "not resolved", no new variants are run and parameters are not changed.

## 5. Fidelity and leakage checks (before any scoring)

1. **Fidelity** (comparison with the original authors' output, using only their own data and splits):
   - Fernandes: reproduce the pooled Pearson of G(A)+E under the original CV0 (training 2020, validation 2021); it must differ from the value reported in the original paper (about 0.45) by no more than 0.03. The original value has been verified only at the abstract level; the original paper must be opened and checked before citing.
   - CLAC: the per-cell Spearman between the 2022 predictions and its public submission file `CLACsubmission5.csv` ≥ 0.9. This requires downloading that file separately (588 KB, same repository); **the user's consent is obtained before downloading**. If not agreed, this item is skipped and reported as is.
   - Lopez-Cruz: there is no comparable public prediction file. Only the within-environment correlation obtained with the original LYO split on 2014–2021 is reported and compared in magnitude with the original "0.25–0.28" (descriptive, no threshold).
   - A method that fails fidelity is still reported, but its conclusion is downgraded to "as reproduced by this project", and the gap is stated.
2. **Leakage**:
   - Using a wrapper script, the yields of Y and later years are replaced with junk values and the run is repeated; all predictions of each method must be bitwise identical to the normal run (MCMC and LightGBM with fixed seeds). If parallel random numbers in R make results non-reproducible, re-run single-threaded before comparing.
   - Check and record the target-year information used by each method. Only these are allowed: environment metadata of Y, measured weather and EC (except for `lc_hist`), and markers of the test hybrids.
3. **Timing and resources**: first run a feasibility pilot on the training years (no prediction or scoring of Y), recording memory and run time. The results are committed as an appended section; only then do the main runs start.

## 6. Direction predictions (fixed in advance)

- `clac` − `cell_reml`: tie to slightly positive, −0.01 to +0.015. Basis: its G×E part is of the same kind as I-B (marker G×location); I-B gave about +0.012 in forward prediction, but the hold-out test was a tie. Subjective probability of "better than": 0.15.
- `lc_real` − `cell_reml`: tie to negative, −0.02 to +0.005; probability of "better than" 0.05. `lc_hist` ≤ `lc_real`.
- `flgbm_gae` − `cell_reml`: negative, −0.05 to 0; probability of "better than" 0.05. On pooled Pearson it may be no worse or even better; this is exactly the contrast to be revealed.

## 7. Implementation, computation and time

- New files only: `scripts/ra_*` (wrappers, data construction, scoring), `tests/test_ra.py` (leakage and common cells), and `results/reanalysis/`. The original code is not changed; steps that need substitution are written as separate wrapper scripts.
- Computation on the 4090 (R environment `<workstation>/dart-gxe/envs/r-reanalysis`); all processes have memory limits, and total machine usage does not exceed 60%.
- Estimates: CLAC several hours per target year (multivariate GBLUP); Lopez-Cruz depends on the memory rule, from several hours to one day per year; Fernandes about 1 hour per year. Total 3–5 days.

## 8. Use of the results

- Written, with the decisions and wording of §4, into the section "Reanalysis of published methods under true forward prediction" of the follow-up paper, alongside the equivalent-pipeline reanalyses of GEFormer and GE-BiFormer on main.
- If any method is "better than `cell_reml`", this is reported as is and checked against the conclusions of that method's original paper.

## Appendix (2026-10-02, after the feasibility pilot, before any reanalysis prediction for target years)

On 2026-10-02 the user agreed to the scope changes in this section (user's words: "OK"). The only things run so far are: a CLAC rehearsal on rehearsal year 2023 (not a scored year); a Lopez-Cruz memory test at the 2024 training size (eigendecomposition only, no fitting); and the fidelity check of Fernandes on the original CV0 split (training 2019–2020, validation 2021; 2021 is not a scored year of this check). **No reanalysis prediction has yet been generated for any scored year 2016–2024.**

### A. Facts found during implementation and necessary deviations

1. **VCF format**: the 2024 release VCF begins with `##` meta-information lines; CLAC reads it with `read_tsv` and gets only one column. The `##` lines are removed when building the inputs; genotypes are unchanged (`scripts/ra_build_inputs.py`).
2. **Number of markers**: the 2024 release VCF has only **2,425 markers** (the same as the benchmark `cell_reml`); the original authors worked with about 437,000 markers from the 2022 release. The CLAC step "randomly sample 35,000 markers" is changed to `min(35000, available)`, i.e. take all (patch 2, `scripts/ra/run_clac.sh`). The 437k file in the workspace is an inbred-line VCF, not hybrid genotypes, and is not used.
3. **Missing EC**: in the 2024 release EC, some environments have no EC rows, so the CLAC rehearsal failed at the eigendecomposition. When building the inputs, environments that are entirely missing are dropped, as well as columns that still have missing values in training-year or target-year environments (patch 3). Actual check: all 654 columns were kept in all 6 target years; only environments were dropped. This affects only CLAC's models for the environment mean and standard deviation, not the within-environment ranking (§0).
4. **Lopez-Cruz historical EC**: MAIZE-HUB `historical_ecov.zip` lacks the main table of 207 covariates; `lc_hist` uses, as in §2.2, the location means of the official APSIM EC over the training years.
5. **Fernandes has no "submission mode"**: the TAG repository produces predictions only for the 2021 CV0 validation fold; there is no "2022 submission pipeline" as described in §2.3. Instead, it is **rebuilt for true forward prediction using the functions and hyperparameters of the original repository** (`scripts/ra/flgbm.py`):
   - Training = BLUEs of the two years before target year Y (the window length of the original CV0), restricted to locations present in Y (the original "known locations" rule). The original CV0 sampling of "random 60% of hybrids + validation hybrids" is not done, because in forward prediction most target hybrids are new;
   - Features: from the original `preprocessing.py`, weather (using the full-year weather file), soil, 15-dimensional SVD of EC, 2-year lagged yield, latitude/longitude bins and interaction terms, imputation with training means; G(A) is the rows of the additive relationship matrix; non-lagged features undergo a 100-dimensional SVD; LightGBM `max_depth=3`; seeds 1–10, **predictions averaged**;
   - BLUE uses lme4 instead of asreml, with the same model structure as the original; 1 of 135 environments had a convergence warning; on failure, it falls back to plot means as in the original `process_blues`;
   - Relationship matrix: AGHmatrix VanRaden on the 2,425 markers (MAF ≥ 0.01), without the original vcftools/plink LD pruning;
   - Target years: 2022, 2024.
6. **Resources**: other jobs on the 4090 have long occupied substantial resources; all processes are limited to 2 threads (`OMP/OPENBLAS_NUM_THREADS=2`, LightGBM `n_jobs=2`). This does not affect the results.

### B. Fidelity check results (§5.1)

- **Fernandes**: in the original paper, the pooled Pearson of G(A)+E under CV0 is **0.45** (mean of 10 replicates, range 0.41–0.49; G(A) alone is also 0.45; FA is 0.41; verified in the original paper PMC11266441). This project's reproduction: per-fold pooled Pearson **0.327 ± 0.032** (mean of 5 folds × 10 seeds = 50 runs, range 0.239–0.400), within-environment Pearson 0.162. The difference is 0.12, which **fails** the tolerance of 0.03.
  - Known differences: markers (2,425 vs the original authors' pruned 2022-release markers), BLUE (lme4 vs asreml), versions of the EC and weather files, pooling convention (this project computes per fold; whether the original paper computed after combining the 5 folds is **not verified**).
  - Under §5.1, all conclusions for Fernandes are downgraded to "as reproduced by this project", and the gap is stated in the results.
- **CLAC**: the comparison of the 2022 reproduction with `CLACsubmission5.csv` is done after the main run (not yet run).

### C. Execution of the Lopez-Cruz memory rule and reduction in scope

- Retaining 97.5% of the variance: killed by the system at the 48 GB limit (ran for 3 h 36 min, maximum resident memory 50.3 GB).
- Estimated from eigenvalues only (no yield involved): the 2024 target year has 103,648 rows in total; retaining 97.5%, 95% and 90% requires 59,941, 33,597 and 15,367 components, and the interaction matrix is about 49.7, 27.8 and 11.9 GiB, respectively.
- The 95% interaction matrix alone is already about 27.8 GiB; with the copies made while BGLR runs it would exceed the 48 GB limit, so **90% is taken directly**. This choice is based only on memory, without looking at any outcome.
- **Reduction in scope** (for computational reasons: BGLR 12,000 iterations times about 15,000 columns; each fit takes several hours or more):
  - `lc_real`: Y ∈ {2022, 2024};
  - `lc_hist`: Y = 2024;
  - 2016, 2018 and 2020 are not run. Pooling for H_lc uses only these two years; N and 3/√N are computed on the common cells of these two years.

### D. Added descriptive comparisons (not in the Holm family; they do not change the decision)

To test "whether the gain appears only for tested hybrids and under the pooled metric", the following is fixed in advance:

- **Fernandes under two splits, against `cell_reml`**:
  - Original CV0 split: `cell_reml` is fitted on the same training rows (BLUEs) as LightGBM and predicts the same validation fold;
  - True forward prediction (2022, 2024): the benchmark `cell_reml` is used.
  - Under both splits, report the difference in pooled Pearson and in within-environment Pearson/Spearman, and the results for new and old hybrids separately.
- Reading fixed in advance: if under CV0 LightGBM is better than `cell_reml` on pooled Pearson, while it is not better on within-environment ranking in forward prediction, write "the gain reported in the original paper depends on tested hybrids and the pooled metric"; otherwise describe as is.

### E. Unchanged content

The tested hypotheses, Holm correction, wording rules for "better than / worse than / not resolved", computation of common cells and 3/√N, and no new variants after a NO-GO are all unchanged.

## Results (2026-10-02; 4090; code 674f806; `results/reanalysis/`; scoring executed only once)

**Process and checks**
- CLAC was run with 6 documented patches (`checks.json`). The 2023 rehearsal and all 5 target years were completed, 4–29 minutes per year, peak 1.7 GB.
- **CLAC fidelity**: the per-cell Spearman between the 2022 reproduction and the original submission `CLACsubmission5.csv` on 11,314 common cells is **0.955**, passing the ≥ 0.9 of §5.1. But this number comes mainly from agreement in environment means; **the median agreement of the within-environment ranking with the original submission is only 0.78** (26 environments, 0.73–0.83). The difference may come from the number of markers (2,425 vs the original 35,000 sampled), the bWGR version and the patches; not verified item by item.
- **CLAC leakage check**: for 2016, after replacing all yields of 2016 and later with junk values and re-running, predictions were bitwise identical to the original run (maximum difference 0); a second normal re-run was also bitwise identical (CLAC runs are deterministic).
- **Fernandes fidelity failed** (see Appendix B; 0.318 in the second run after changing the number of threads to 2).
- **Lopez-Cruz**: 2022 with measured EC completed (1 h 10 min, peak 42.1 GB). 2024 with measured EC and with historical EC were both killed under the pre-registered 48 GB limit (peak 50.3 GB). Under the rule of Appendix C, recorded as "cannot be reproduced on this machine": `lc_real` is scored only for 2022, and `lc_hist` has no result. The two earlier kills were due to my mistakenly setting a 40 GB limit; they were re-run with 48 GB.

**Decision (within-environment Spearman, relative to the benchmark `cell_reml`, common cells, Holm correction)**

| Method | Years | N | 3/√N | Δ | Holm-level CI | HK interval | Decision |
|---|---|---|---|---|---|---|---|
| **CLAC** | 5 | 51,666 | 0.0132 | **+0.0596** | [+0.0374, +0.0817] (97.5%) | [+0.031, +0.078] | **Better than `cell_reml`** |
| Lopez-Cruz (measured EC) | 1 (2022) | 10,283 | 0.0296 | +0.0191 | [+0.0026, +0.0356] (95%) | — | Statistically detectable, but below selection resolution |
| Fernandes LightGBM G(A)+E | 2 | 21,266 | 0.0206 | **−0.2351** | [−0.2953, −0.1749] (98.33%) | [−0.555, +0.085] | **Worse than `cell_reml`** |

**Descriptive (uncorrected)**
- **CLAC by year**: 2016 +0.080, 2018 +0.054, 2020 +0.031, 2022 +0.063, 2024 +0.075, all positive; 72% of 123 environments positive.
  - New hybrids +0.066 [+0.037, +0.095]; old hybrids +0.029 [−0.013, +0.070].
  - Within-environment Pearson +0.078, top-10% selection differential +0.103 [+0.062, +0.145].
  - Pooled Pearson: CLAC 0.481, `cell_reml` 0.102. `cell_reml` predicts only genotype values, not environment means, so pooled Pearson is not comparable; for illustration only.
- **CLAC decomposition**: Model A (multivariate GBLUP genomic values from each training environment, weighted by location, state and kinship) +0.055 [+0.023, +0.088], by year +0.022 to +0.115; Model B (GBLUP with environment fixed effects) +0.014 [−0.017, +0.045], −0.073 in 2024. The gain comes mainly from Model A.
- **Fernandes**: 2022 −0.261, 2024 −0.210; new hybrids −0.236, old hybrids −0.028. With G(A) only, −0.226, similar to G(A)+E. CV0 paired: pooled Pearson 0.318 vs `cell_reml` 0.419; within-environment Spearman 0.136 vs 0.296, lower in all 50 runs. Because fidelity failed, **this can only be written as "LightGBM as reproduced by this project"**, not as a refutation of the original paper.
- **Lopez-Cruz** has only 1 year, so there is no estimate of between-year variation and the conclusion is weak; the low-rank version `rn_ridge` in the benchmark can serve as a reference.

**Comparison with the predictions in §6**
- CLAC predicted −0.01 to +0.015, probability of "better than" 0.15: **actual +0.060; the prediction was wrong**.
- Lopez-Cruz predicted −0.02 to +0.005: actual +0.019, above the prediction.
- Fernandes predicted −0.05 to 0: actual −0.235, far below the prediction; this very likely comes partly from the quality of the reproduction.

**Reading**
1. **The statement "in the forward scenario no method exceeds cell-level REML-GBLUP" is overturned by CLAC.** The statistical pipeline of the G2F 2022 winner is resolvably better than `cell_reml` in all 5 true forward years and on new hybrids, and it exceeds 3/√N by a factor of 4.5 (0.060 / 0.0132).
2. CLAC is not a complex machine-learning model. Its gain comes from Model A, whose components include: per-environment spatial correction and outlier removal of training phenotypes, the arc-cosine kernel, per-environment multivariate GBLUP, and weighting by location, state and kinship. **Which component is responsible was not tested in this check**; a decomposition requires a new pre-registration.
3. The three methods differ in reproduction quality, and so in the strength of their conclusions: CLAC (per-cell 0.955, within-environment 0.78) > Lopez-Cruz (no comparable file, only 1 year) > Fernandes (fidelity failed).
4. No new variants are run after a NO-GO; the component decomposition of CLAC is a new question and requires a new pre-registration first.
