English translation of the pre-registration document prereg_sequel_wave2b_2026-09-28.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Sequel pre-registration, wave 2 B: model selection and decision combination under deployment conditions (B3, E5) (2026-09-28)

Written before any forward panel or selection rule in this wave was run. Not changed after commit; amendments are only appended as dated sections. Basis: the W1c result (`docs/prereg_sequel_wave1_2026-09-28.md`, K = 31 ≥ 12, so B3/E5 are done by the advance rule); atlas decision stacking (`<lab-data>/analysis/atlas/stack_select.py`, read-only; on 2026-09-28 the user decided to merge it into the sequel); W1a (REML under-shrinks in forward years).

## 0. The question

Each year, a breeder has to pick a prediction method (or combination) for new lines and a new year. The sequel's question is: **which data should be used to pick it**:
- random cross-validation (CV, genotypes held out within the historical years);
- forward results of past years (each year is predicted by training on the years before it).

A second question: is a combination fitted for selection decisions (decision stacking) better than the single best method that was picked?

Known information, declared as is:
- In an evaluation of CV1 + random halves of environments, atlas stacking was +0.038 SD [0.016, 0.070] above "pick the top method" (22 combinations, selection-differential scale). **This is not a forward condition**.
- W1a: in forward years, the post-hoc optimal λ was larger than the REML λ in 5/5 years.
- W1c only counted sample structure. No method in this wave has been run on these 31 forward years.

## 1. Data and target years

- Datasets and traits:
  - G2F: yield, data_v2;
  - NUST: `YieldBuA`; environment = location-year; multiple trials within the same location-year are averaged by line (atlas loader);
  - URSN: resistance = 100 − DIS; higher is better (atlas direction).
- **Target years** = the 31 qualifying forward years of W1c (G2F 5, NUST 15, URSN 11). Scored environments are the environments of that year that reach the threshold (≥25 marker-typed genotypes; URSN ≥10). All marker-typed genotypes in these environments are scored. A sensitivity analysis uses only new genotypes (never appeared in any earlier year).
- **Historical years**: years before the target year that have at least 1 scored environment and at least 1 earlier year of data. A forward prediction is made for every historical year: train only on the years before it and predict that year.

## 2. Method library (10 methods, two-stage, genotype only)

A new environment has no training data from the same environment, so the atlas environment index cannot be used. This wave therefore uses only genotype main-effect models.
- Stage 1: from each training cell, subtract the mean of the training cells in that environment; then average by genotype across all training years.
- Stage 2: learn these genotype means with the methods below.
- Standardisation, PCA and GRM are fitted only on training lines (the project protocol document §2.1).

1–4. **Marker ridge regression, λ = REML λ × {0.1, 1, 10, 100}** (`reml_x0.1`, `reml`, `reml_x10`, `reml_x100`): all markers, standardised on the training lines; REML with `dartgxe.baselines.linear.Ridge`, with only one environment group. This family links to W1a: W1a found that REML under-shrinks in forward years.
5. `ridge_pc20`: ridge regression on the first 20 PCs, λ = 1.
6. `rf`: random forest, 300 trees, min_samples_leaf = 3, 80 PCs.
7. `gbm`: HistGradientBoosting, max_depth 4, 300 iterations, learning rate 0.06, 80 PCs.
8–9. `knn10`, `knn30`: k nearest neighbours on 80 PCs.
10. `mlp`: (48, 24), early_stopping = True, 80 PCs.

The tree and neural-network parameters follow atlas `panel_design.py`; random seeds are fixed. The units of replication are years and environments, not seeds.

## 3. Selection rules (the phenotypes of target year Y do not enter any rule)

Score: each method's **mean within-environment Pearson** on "historical data" (the quantity atlas uses to pick the top method).
- **CV history**: within all years before Y, 5-fold CV by genotype (the two stages are recomputed on the training folds); the out-of-fold predictions are put back into their environments and the score is computed.
- **Forward history**: the forward predictions of all historical years before Y, with environments weighted equally.

| Rule | Definition |
|---|---|
| R_REML | Always use `reml`, no selection (the default practice) |
| R_CV | The single method with the highest score on the CV history |
| R_FW | The single method with the highest score on the forward history |
| R_EQ | Equal-weight average of the 10 methods (each first standardised within environment) |
| R_STK_CV | Decision stacking, fitted on the environments of the CV history |
| R_STK_FW | Decision stacking, fitted on the environments of the forward history |
| Oracle | The best single method on the target year (only described, as an upper bound) |

Decision stacking follows atlas `stack_weights` exactly: within each environment, predictions and phenotypes are standardised; c = mean over environments of Z'y/n, S = mean over environments of Z'Z/n; minimise w'(S + 0.05 I)w − 2c'w subject to w ≥ 0 (atlas uses a Cholesky decomposition to turn this into an NNLS problem); if all w are 0, equal weights are used.

## 4. Metrics and summary

- Primary metric: **within-environment Spearman** in the scored environments of the target year (the primary metric of the project protocol document).
- Decision metric: realised 10% selection differential (atlas `gain`: within each environment, take the top k = max(1, round(0.1n)) by prediction and compute their mean standardised phenotype).
- Year-level effect: d_Y = mean over the scored environments of target year Y of (rule A − rule B); SE_Y from a within-year environment bootstrap (2,000 replicates).
- **Pooling**: random-effects pooling of d_Y over the K target years (DerSimonian–Laird, weights 1/(SE_Y² + τ²)). URSN years have large SEs and therefore naturally small weights.
- Sensitivity: G2F + NUST only; unweighted mean with a year bootstrap stratified by dataset; selection differential as the metric; new genotypes only; history restricted to qualifying years (years with new-genotype proportion ≥ 0.5).

## 5. Confirmatory comparisons (only 2; Holm correction; two-sided)

- **H1 (core of E5)**: R_FW − R_CV, i.e., is picking a method by the forward results of past years better than picking it by random CV?
- **H2 (core of B3)**: R_STK_FW − R_FW, i.e., is decision stacking better than the top method on the forward history?
- Verdict (the project protocol document §2.2): "A is better than B" requires that the CI at the corresponding Holm level (step 1: 97.5%, step 2: 95%) excludes 0, **and** that |Δ| ≥ 3/√N (N = number of scored cells over all target years). If only the former holds, write "statistically detectable, but below the selection resolution"; if neither holds, write "tied".
- **Directional predictions (fixed in advance)**:
  - H1: R_FW ≥ R_CV, with a small gap. Reason: the forward history has the same distribution as the deployment condition, but there are few historical years and much noise; CV has more data, but a different distribution.
  - H2: **tied**. Reasons: under random effects, a pooled ridge regression with a well-tuned λ is no worse than an ensemble of the same kind (Ramchandran & Mukherjee 2026); the variance of estimated weights often keeps complex combinations from beating simple approaches (Claeskens et al. 2016); the atlas +0.038 SD comes from CV1 and is expected to shrink under forward prediction.
- **MDE** (computed now, fixed in advance): the SD of method differences at the environment level is taken as G2F 0.072 (gate 2 amendment 1), NUST 0.176, URSN 0.255 (median of pairwise method differences on the atlas CV1 panels; no forward results used). The number of environments per year is taken from W1c. α = 0.025 two-sided (Holm step 1), power 80%.
  - With between-year SD σ_y = 0.02 / 0.037 / 0.05, the MDE pooled over 31 years is 0.022 / 0.030 / 0.037 (Spearman scale);
  - G2F + NUST only: 0.022 / 0.031 / 0.039; G2F only: 0.034 / 0.055 / 0.072;
  - Weights at σ_y = 0.037: NUST 0.62, G2F 0.30, URSN 0.08.
  - Reading: only gaps above about 0.03 can be resolved. If H2 is estimated at half the atlas effect, power is insufficient; a verdict of "tied" must also be reported as is.

## 6. Descriptive results (no confirmatory conclusions)

- R_REML vs each rule: under deployment conditions, is model selection worth doing?
- λ family: pick only within {`reml_x0.1`, `reml`, `reml_x10`, `reml_x100`} by the forward history, compared with always using `reml`. This tests whether the under-shrinkage found in W1a can be compensated by the forward results of past years.
- R_STK_CV vs R_EQ; the gap between each rule and the Oracle.
- **E5 ranking transfer**: for each target year, compute Kendall τ between the method ranking on the CV history and the method ranking on the target year; summarise the mean τ with a stratified bootstrap CI. Also report the proportion of pairwise method gaps in the target year that are below 3/√N_Y, i.e., how much of the ranking disagreement is only ties.
- Detailed tables for each dataset and each target year.

## 7. Hard-rule check, implementation and timing

- Leakage prevention:
  - the phenotypes of target year Y are used only once, at the final scoring;
  - all fitting (stage-1 means, standardisation, PCA, GRM, REML, stacking weights, selection of the top method) uses only data before Y;
  - the CV history holds out by genotype within the years before Y;
  - each year of the forward history uses only the years before it.
- When split logic is added or modified, unit tests are added that assert the conditions above.
- Scripts:
  - `scripts/b3_panels.py`: generates the forward predictions for each year of each dataset, and the CV out-of-fold predictions for each target year;
  - `scripts/b3_rules.py`: computes the rules, year-level effects, pooling and verdicts.
  - The analysis script refuses to run before all panels are complete.
- Compute: CPU, 4090 host; NUST/URSN data from the uploaded read-only copies with checked SHA256.
- Time box: panels completed before 2026-11-15, analysis completed before 2026-11-30; 12-15 is the hard deadline, applied by the W1c rule.

## Amendment (2026-09-28, before any corrected number was computed): pooling weights for zero-variance years

- **Finding**: in the first run (code 02537f9, output archived to `results_v2/b3/run1_degenerate_se/`), the pooled estimates of H1, of REML vs each single-method rule, and of all selection-differential contrasts were exactly 0, with CIs of ±0.0000.
- **Cause**: when two rules pick the same method in a target year (on the selection differential: when the same set of selected lines is picked in every environment, which is common in URSN where only 1 line is selected per environment), the differences in all environments of that year are 0, and the environment bootstrap gives SE_Y = 0. The DerSimonian–Laird weight 1/SE_Y² becomes 10¹² at the code floor of 1e-12. These years take all the weight and pin the pooled estimate at 0. This is a degenerate case not considered in §4, and it is unrelated to the results.
- **Handling**: all pooled results of the first run are **void, and no conclusions are drawn from them**; the H2 value is not used either.
- **Correction** (written before any corrected number was seen):
  - Within each (dataset, contrast, metric), SE_Y is set to max(SE_Y, s_floor), where s_floor is the median SE of the years in that group with SE_Y > 0. If all years in the group have SE_Y = 0, the median over years with SE_Y > 0 across all datasets is used instead.
  - That is, a year in which "the same method was picked" is treated as an observation with a difference of 0 and typical precision. It is neither given infinite weight nor dropped.
  - Also reported: the number of years with SE_Y = 0 in each contrast, and in how many target years the CV and forward routes picked the same method (descriptive; this is itself a result).
  - The pre-registered sensitivity analysis "unweighted mean with a year bootstrap stratified by dataset" is not affected by this problem and is reported as planned.
- Everything else (rules, metrics, Holm, verdict criteria) is unchanged.

## Results (2026-09-28; 4090 CPU; panel code f-series to a4d0029, rule analysis code a4d0029 including the correction; `results/b3/{pooled.json, year_effects.csv, choices.csv, e5.csv, verdict.json, rules_per_env.parquet, meta/}`)

Panels: 56 forward years (G2F 10, NUST 18, URSN 28), 31 CV histories, 6.45 million rows of predictions in total, no missing values. URSN training sets are very small (at least 15 lines). In 5 of the 28 URSN forward years, REML fell on a grid endpoint. MLP failed to converge once (500-iteration limit, following atlas). Scored cells in the target years N = 87,294, 3/√N = 0.0102.

**Confirmatory comparisons** (primary metric within-environment Spearman, random-effects pooling, Holm)

| Contrast | Estimate | CI | Holm level | Verdict | Advance prediction |
|---|---|---|---|---|---|
| H2: R_STK_FW − R_FW | **+0.0155** | [+0.0035, +0.0274] | 97.5% (p = 0.004) | **Better** (CI excludes 0 and ≥ 0.0102) | Tied → **prediction failed** |
| H1: R_FW − R_CV | +0.0010 | [−0.0051, +0.0071] | 95% | Tied | Forward ≥ CV, small gap → consistent |

- None of the 31 years used the SE floor for H2, so its conclusion is not affected by the zero-variance correction. For H1, 19 years used the floor (the two routes picked the same method).
- Sensitivity (H2, Spearman): G2F + NUST +0.011 [+0.002, +0.020]; history restricted to qualifying years +0.017 [+0.002, +0.031]; unweighted mean with a year bootstrap stratified by dataset +0.016 [+0.001, +0.031]. By year, the positive years are G2F 4/5, NUST 9/15, URSN 7/11.
- **Boundaries of H2**:
  - **It disappears when only new genotypes are scored** (−0.0015 [−0.014, +0.011]);
  - **It is not significant on the selection-differential metric** (+0.011 [−0.012, +0.034]).
  - So it can only be written: "On all scored genotypes of the forward years, decision stacking fitted on the forward results of past years gives a resolvably better within-environment ranking than the single best method picked from the same history; for new genotypes that never appeared before, and when measured by the 10% selection differential, no difference was detected".
- The sensitivity analyses of H1 are all tied in the weighted analysis. In the unweighted mean, new genotypes +0.028 [+0.008, +0.053] and selection differential +0.082 [+0.011, +0.159]; these come mainly from URSN (when unweighted, the 11 URSN years have the same weight as NUST). As specified in advance, the weighted main analysis is authoritative.

**Descriptive results**
- Years in which the two routes picked the same method: G2F 3/5, NUST 12/15, URSN 3/11. Years in which the best method of the target year (Oracle) was picked: for both forward and CV, G2F 1/5, NUST 8/15, URSN 0/11.
- In the target years, the proportion of pairwise method gaps below 3/√N_Y averages **0.80**. That is, picking a single "best method" is mostly a choice among tied methods.
- E5 ranking transfer: the mean Kendall τ between the CV-history ranking and the target-year ranking is 0.30; for the forward history it is 0.43.
- Compared with the default practice (always use REML-GBLUP): R_FW −0.0003, R_CV −0.0084, R_STK_FW +0.0092 [−0.0033, +0.0216], R_EQ +0.0045; none significant. That is, selection or combination gives no resolvable gain over "just use REML".
- λ family: picking the REML multiplier by the forward history, compared with always using REML: −0.0033 [−0.011, +0.005]. The under-shrinkage found in W1a cannot be compensated by "choosing the multiplier by the forward results of past years" (consistent with W1a, where "no fixed multiplier is favourable in every year").
- Gap to the Oracle: R_FW −0.020 [−0.030, −0.010]; R_STK_FW −0.0065 [−0.014, +0.001].
- Stacking weights are spread out: the median number of methods per year with weight > 0.01 is 7 (of 10).

**One-sentence conclusion**: across 31 forward years in 3 crops, there is no difference between picking a single model by random CV or by the forward results of past years (tied). Both rarely pick the best method of the year, because most methods are tied within the resolution. Decision stacking fitted on the forward results of past years is 0.015 above the single model picked, but this advantage does not hold for new genotypes and is not significant relative to just using REML-GBLUP.

Correction (2026-09-28, same day): "panel code f-series to ..." at the start of the results section above is a typo. It should read "panel code e0eece9 (data loading, method library, history construction and leakage tests)".
