English translation of the pre-registration document prereg_sequel_wave2e_2026-09-29.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Sequel pre-registration, wave 2E (positive push, direction 1): can the decision ensemble beat cell-level REML on independent new data (2026-09-29)

Written before any method was fitted and before any prediction was produced on these 3 datasets. Not changed after commit; changes are only appended as dated sections. Basis: `docs/sequel_plan_v2_2026-09-28.md` §9.3. On 2026-09-28 the user agreed to the positive push ("agree to 1"). On 2026-09-29 the user approved downloading the 3 datasets ("download all 3").

## 0. Known information, stated as is

- **Wave 2c (B3b, same method, 31 forward years, already seen, cannot serve as confirmation again)**: R_STK_FW − R_CELL
  - within-environment Spearman +0.0028, tie;
  - selection differential (top 10%) +0.031 [+0.010, +0.052], a descriptive result only. Per-year means by dataset: G2F +0.079, NUST +0.027, URSN −0.020.
  - This selection-differential signal is exactly what this wave is meant to confirm on new data.
- **Direction 2 (wave 2d) has already been judged NO-GO** (ce10bf1). This was known when this file was written.
- **On the new data, only the sample structure has been examined; no method has been fitted**:
  - structure counts: `scripts/b3d_count.py`, code fe47ffe, results in §1.2;
  - before counting, to check file formats and ID correspondence, the number of environments per year, the number of lines per environment and the proportion of new lines were glanced at for each year.

## 1. Data

### 1.1 Sources and definitions (reading code `src/dartgxe/forward/data.py`, fe47ffe)

| Dataset | Crop | Source (licence) | Environment | Year | Trait (cell value) | Genotype |
|---|---|---|---|---|---|---|
| ESWYT | Spring wheat, CIMMYT international Elite Spring Wheat Yield Trial, 24th–37th nurseries | Juliana et al. 2020; figshare 12609368 + Frontiers Supplementary Table S3 (CC BY 4.0) | Location × nursery | Harvest year of the nursery (2003–04 → 2004) | Per-site grain yield BLUE (t/ha) | GBS HapMap (14,389 markers; of the two alleles listed for each locus, the first is coded 0 and the second 2; IUPAC heterozygotes coded 1; N coded missing) |
| GEM_IA | Maize GEM line topcrosses, Iowa program | Rogers et al. 2022; Zenodo 7150254 (CC BY 4.0) | Loc (each Loc is one location-year) | 2000 + first two digits of Experiment | Mean plot yield of the line in that environment (bu/ac) | **GRM only**: its kernel embedding U·√Λ is used as "markers", centred only and not scaled, so marker ridge regression is equivalent to GBLUP with this GRM (unit test `test_kernel_embedding_keeps_the_grm`) |
| MU_SOY | Soybean, University of Missouri MU-FDREEC advanced yield trials | Vieira et al. 2022; Zenodo 7259254 (CC0) | Env (year + field) | year | Yield (bu/ac, mean of three replicates) | SoySNP6K, 0/1/2 |

- **Processing of GEM_IA**:
  - commercial checks, GEM checks, and the 3 families removed by the original authors in File S6 were removed;
  - the line name is the part of `Converted Pedigree` before "+tester", and it is matched to GRM row names by normalisation rules (`gem_line_key`, with unit test);
  - 87% of plots with yield records can be matched to the GRM.
  - Within one environment, each line is crossed with only one tester (checked: all 8,078 line-environment combinations satisfy this). Different lines may have different testers; this is kept as is.
- **Use of the GRM**: the GRM was computed by the original authors from the markers of all lines (including target-year lines). It contains genotype information only. At deployment, new lines are genotyped before they enter trials, so this is not phenotype leakage. Centring and PCA after embedding are fitted on training genotypes only.
- **Unused parts, stated in advance**:
  - the NC program of GEM: only 3 years, and it is a different breeding program;
  - Obregón Stage 1–3 and SABWGPYT in the ESWYT data package: they belong to the same CIMMYT system as ESWYT, and they have few years.
- **Marker processing** is the same as for the existing datasets: means and standard deviations are estimated, and missing values imputed, on training genotypes only; markers without variation in the training set are removed; no other filtering.

### 1.2 Target years (W1c rules unchanged)

Conditions for a qualifying forward year:
- at least 2 earlier years each have one environment meeting the threshold;
- that year has ≥5 environments, each with ≥25 genotyped lines;
- proportion of new lines ≥0.5.

Count results (`results/b3d/count/`, 4090, code fe47ffe; the main definition and the relaxed definition ≥10 give the same result): **K = 17 target years, scored cells N ≈ 31,144** (count definition).

| Dataset | Target years | Environments per year | Lines per environment (median) | Proportion of new lines | Genotyped lines |
|---|---|---|---|---|---|
| ESWYT | 2006–2017 (12 years) | 22–45 | 49–50 | 0.66–0.94 | 626 (almost all in each nursery) |
| GEM_IA | 2016–2019 (4 years) | 7–10 | 219–341 | 1.00 | 1,528 |
| MU_SOY | 2020 (1 year) | 11 | 78 | 0.56 | 792 |

- The other MU_SOY years do not qualify: most lines are tested in 2 consecutive years; the proportion of new lines is 0.29 in 2019 and 0.11 in 2021. 2018 has only 1 earlier year.
- The marker missing rate is 17% for ESWYT and 1.6% for MU_SOY. GEM_IA is a GRM, with no missing values.
- The forward histories of the 3 datasets (the training panels of R_STK_FW) start in 2005, 2015 and 2018, respectively.

## 2. Method library and rules (same as wave 2c, unchanged)

- **Method library**: the 10 two-stage methods of wave 2b, plus `cell_reml` (cell-level marker ridge regression, environment fixed effects, REML λ). The new data have no environmental covariates, so there is no G×E learner. This is the same as the NUST method library in wave 2c.
- **Rules**: R_CV, R_FW, R_EQ, R_STK_CV, R_STK_FW, Oracle, R_CELL; code identical to `scripts/b3b_rules.py`.
- **R_STK_FW**:
  - within each environment, the predictions of each method are first standardised; then non-negative weights are fitted by NNLS, with ridge parameter LAM = 0.05;
  - the target is the covariance between forward predictions and observations (standardised within environment) in earlier years; the weights are fitted only on the forward panels of all predictable years before the target year;
  - **the weights are not re-tuned for the selection differential**, i.e. the method is exactly the same as in wave 2c.
- **Panels**: `scripts/b3d_panels.py` calls the unchanged functions of wave 2b/2c (`panel.forward_year`, `panel.cv_history`, `cellgxe.forward_year_cell`, `cellgxe.cv_history_cell`).
- **Scoring**: `scripts/b3d_rules.py`; it refuses to run when panels are incomplete.

## 3. Confirmatory comparison (1 comparison, two-sided 95%)

- **H7**: R_STK_FW − R_CELL
  - **Primary metric**: top-10% selection differential `sel_diff_f10`, in units of within-environment standard deviation: select the top round(0.1n) lines by prediction, and take the mean of their within-environment standardised observed values;
  - all scored genotypes;
  - pooling: per-year effects (environment bootstrap SE, B = 2000) are pooled with DerSimonian–Laird random effects; zero SEs are handled per the 2026-09-28 correction, with a floor equal to the median of the non-zero SEs within the same dataset.
- **GO criterion (plan v2 §9.3)**: lower CI bound > 0, and point estimate ≥ 3/√N (N = number of scored cells in the target years, i.e. the project-wide resolution rule).
- **MDE and power** (based on the wave 2c selection differential: median per-year SE 0.03–0.08, taken as 0.045 given the number of environments per year in the new data; between-year variance τ² = 0.00037):
  - pooled SE ≈ √((0.045² + 0.00037)/17) ≈ 0.012;
  - MDE (80% power, two-sided 5%) ≈ 0.033;
  - the GO threshold 3/√N ≈ 0.017 is below the ≈0.023 that "lower CI bound > 0" actually requires, so GO is in practice decided by the CI;
  - if the true effect equals the wave 2c value of +0.031, power ≈ 0.74; if it equals the per-year mean over the 31 wave 2c years, +0.019, power ≈ 0.36.
- **Directional prediction (written in advance)**:
  - the point estimate is positive, between 0 and +0.03;
  - no confidence about whether GO will be reached; subjective probability about 0.35.
  - Basis: the wave 2c value of +0.031 came mainly from G2F and NUST, while URSN was negative; the new data have fewer lines per environment (ESWYT about 40, so the top 10% is only 4 lines), and the selection differential is noisy.

## 4. Descriptive results (not part of the verdict)

- within-environment Spearman for H7;
- both metrics on new genotypes only (≥10 per environment);
- results by dataset;
- each of R_FW, R_CV, R_EQ, R_STK_CV and Oracle versus R_CELL;
- `cell_reml` versus two-stage REML;
- the H7 selection differential pooled with the 31 wave 2c years: descriptive only, because the 31 years have been seen;
- the share of `cell_reml` in the stacking weights; the distribution of the forward first-ranked method.

## 5. Use of results (written in advance)

- **GO**: the sequel is written as a positive paper on "decision-aligned forward ensembles". This wave is the confirmatory evidence; the 31 years of wave 2b/2c, the audit and the criteria serve as support.
- **NO-GO**: neither direction 1 nor direction 2 is achieved. Per plan v2 §9.3 (a)/(b), the user and the supervisor decide: move to sparse testing, or publish as an evaluation paper, with [reference to a separate unpublished study omitted] carrying the positive main line.
- Whatever the result, the two metrics, H7 and Spearman, are both reported side by side as they are.

## 6. Hard-rule check

- **Leakage prevention**:
  - forward panels are trained only on years before the target year (existing assertions and unit tests are kept);
  - marker transformations are fitted only on training genotypes;
  - stacking weights use only forward panels before the target year;
  - target-year phenotypes are used once, only at final scoring.
- **Reproducibility**:
  - input SHA256 sums are recorded on the 4090 at `<workstation>/dart-gxe/external_ro/newdata_2026-09-29/SHA256SUMS_*.txt`; sources and licences are in `PROVENANCE.md` in the same directory;
  - each panel records the git commit.
- **Compute**: 4090. Time box: complete before 2026-10-12.

## Results (2026-09-29; 4090; panel and scoring code 7e066a1)

Output: `results/b3d/{pooled.json, year_effects.csv, rules_per_env.parquet, choices_weights.csv, verdict.json, meta/}`; panel files are on the 4090 in `results_v2/b3d/{forward,cv}`. Panels for all 17 target years are complete, with no failures; the GEM_IA REML never landed on a grid endpoint. Scored cells: all, N = 31,144; new genotypes, N = 27,906.

**Confirmatory comparison (H7, R_STK_FW − R_CELL, top-10% selection differential, 95%)**

| | Estimate | 95% CI | Verdict | Prior prediction |
|---|---|---|---|---|
| H7 (17 years) | **−0.040** | [−0.088, +0.009] | Tie (negative point estimate) | Small positive value ✘ |

**GO criterion not met → direction 1 NO-GO.**

**Descriptive**
- Within-environment Spearman: H7 −0.013 [−0.028, +0.003], tie.
- By dataset (selection differential / Spearman):
  - ESWYT −0.038 / −0.008, tie;
  - GEM_IA −0.190 [−0.360, −0.020] / −0.077 [−0.142, −0.011], **the ensemble is clearly worse**;
  - MU_SOY +0.081 / −0.016, only 1 year.
- New genotypes only: selection differential −0.103 [−0.163, −0.043], the ensemble is worse; Spearman −0.015, tie.
- Other rules versus R_CELL (Spearman): R_FW −0.016, R_CV −0.011 [−0.021, −0.001], R_EQ −0.005, R_STK_CV −0.007. **No rule beat the fixed use of cell-level REML**.
- Oracle (best single method in hindsight) versus R_CELL: Spearman +0.018 [+0.007, +0.029], close to the +0.020 of wave 2c. A better single method exists, but no prior rule can pick it out.
- `cell_reml` versus two-stage REML: Spearman +0.002, selection differential +0.017, both ties (wave 2c: +0.020).
- **Mechanism (descriptive)**: when the forward history is short, the ensemble weights and the forward first-ranked method fall on unstable methods.
  - GEM_IA has a history of only 4–27 environments; the first-ranked methods were, in order, knn10, knn10, gbm, gbm, and almost all ensemble weight went to knn/gbm;
  - in the first 7 ESWYT years (history of 28–228 environments), the first-ranked method was gbm, rf or a REML multiple, and the weights were mainly on rf/gbm/mlp;
  - once the history reached ≥266 environments (2013–2017), `cell_reml` was first-ranked in 5/5 years, its weight rose to 25–41%, and the ensemble was roughly level with R_CELL.
- Pooled with the 31 wave 2c years (48 years, descriptive): selection differential +0.009 [−0.013, +0.032]. The +0.031 seen in wave 2c was not reproduced on independent data.
- Sanity check (not pre-registered): for GEM_IA, with `cell_reml` forward predictions, the predictive ability computed on line means was 2016 −0.10, 2017 0.00, 2018 0.44, 2019 0.29. The original authors' values, leaving one year out and training on all other years, were 0.35, 0.20, 0.49, 0.37. The last two years are close; the first two years are lower, which is consistent with our training on only 2–3 earlier years. No problems were found in data matching or code.

**Conclusion in one sentence**: on 3 crops and 17 independent forward years, the decision ensemble fitted on the forward results of earlier years did not beat fixed cell-level REML-GBLUP; the point estimate was negative, and it was even worse when the history was short. The selection-differential advantage seen in wave 2c was not reproduced.

**Direction 1 and direction 2 are both NO-GO → per plan v2 §9.3 (a)/(b), the decision is handed to the user and the supervisor.**
