English translation of the pre-registration document prereg_headroom_secondary_2026-09-30.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

# Headroom check pre-registration SX: forward + same-season pre-harvest records (G2F secondary traits, information ablation) (2026-09-30)

Written before any structural counting, any implementation and any fitting. Not changed after commit; changes are only appended as dated sections. Basis: the user's request on 2026-09-30, "do 1 and 2 together, first write the pre-registration for G2F secondary traits". The scenario is defined in `docs/candidate_data_secondary_stage_2026-09-30.md` §1: "new lines of the target year have flowering time and plant height measured before harvest; use a multi-trait model to predict the yield ranking".

Design process: 4 independent drafts (realism, validity/leakage, statistics/power, methods). After merging, they were given to 3 adversarial reviews (leakage, realism, statistics). This file is the version revised accordingly. The points raised in review that were adopted or rejected are listed in §11.

**Nature**: a descriptive headroom check, not a confirmatory test. GO only means "worth writing a confirmatory pre-registration, and preparing data that did not take part in this check".

## 0. Known information, stated as is

- **The outcomes have all been seen**: G2F yields for 2016, 2018, 2020 and 2022 have been scored in the forward benchmark (`cell_reml`, `rn_ridge` and others, 15 methods), in I-A/I-B, and in sparse testing SP/SP2. In the forward scenario no method exceeded `cell_reml`. I-B gave +0.012 on all data, but the hold-out test on other traits was a tie, so it is recorded as a selective finding.
- **Same-year ceiling (already seen)**: in `results/ideas_diag/ceiling_envs.csv`, prediction using the **yield** main effects from other environments in the same year (`gm_ceiling`) gives environment means of 0.498, 0.340, 0.220 and 0.414 for G2F 2016–2022; `cell_reml` on the same cells gives 0.231, 0.117, 0.032 and 0.239. This shows that same-year information has large headroom, but that is yield; this scenario does not use the yield of any environment in the target year.
- The panels of the **OL check** (`docs/prereg_headroom_oldlines_2026-09-30.md`) have been generated on the 4090 (code 40624af) and **have not yet been scored**; scoring takes place after this file is committed. OL and SX use different scored cells (OL scores old lines only); no joint correction is made for the two.
- **The values of the secondary traits have never been used**: days to silking, days to anthesis (pollen shed), plant height and ear height in the G2F raw table (`data/raw/g2f2024/1_Training_Trait_Data_2014_2023.csv`, ODC PDDL) have never been used in this project as predictors or outcomes. The only thing seen is a structural count (04fa213, presence/absence only): among plots with yield, coverage is 0.67–0.87 for silking, 0.72–0.87 for anthesis and 0.83–1.00 for plant height; in each target year 4–6 environments have no such records at all; the proportion of cells with ≥ 2 replicates is 0.88 in 2016 and 0.33–0.39 in 2018–2022.
- **2024 cannot be used**: `7_Testing_Observed_Values.csv` has only Env, Hybrid and Yield_Mg_ha (header verified).
- **There is no field description file**: `data/raw/g2f2024/` contains only the data tables and a 1.3 KB metadata JSON, with no description of measurement times or protocols. So **the measurement time of plant height and ear height cannot be verified** (it may be close to harvest). Therefore they enter only the descriptive analysis (§3), not the decision.
- **This is the 6th family of headroom checks on the same set of outcomes** (I-A, I-B, SP, SP2, OL, SX). Each family is tested at the 5% level, with no project-level multiplicity correction. So any GO of SX is only **an exploratory finding on outcomes already seen** (§8).
- **NUST is not included**: its Maturity is recorded at R8, and Height and Lodging are also recorded around maturity; none of these is pre-harvest information usable for decisions. In addition, these traits have already been used as outcomes in the I-B hold-out test. These two are the only reasons for exclusion; they are unrelated to the ceiling gap of each dataset.
- **Novelty boundary**: multi-trait genomic prediction with secondary traits is a published direction. The related literature (Rutkoski et al. 2016, Montesinos-López et al. 2019, Krause et al. 2019, Lopez-Cruz et al. 2020, Runcie and Cheng 2019) was **not opened and verified** in this analysis; it must be opened before being cited. The contribution of this check can only come from "strict evaluation under deployment conditions", not from the method itself.

## 1. Scenario and information set

- **SX = forward + same-season pre-harvest records**: new lines × new year; there is no yield at all for target year Y. What is additionally available is the pre-harvest records of **other environments** in Y.
- **This is an information ablation, not a real decision time point**: by the time the last silking date is recorded in northern environments, southern environments may already have been harvested (item 5 of §6 counts this proportion). This file and any later text do not write "valid for deployment" or "available before harvest at the breeding-programme level"; they write only "adding flowering records from other environments in the same season on top of the forward information".
- **Available information**:
  - Markers of all lines;
  - Yield and whitelisted records of all years before Y;
  - Whitelisted records of Y, but only from environments outside the "exclusion group" of the scored environment j (leave-one-environment-out, LOEO; exclusion groups: see item 3 of §6).
- **Whitelist** (the reader reads only these columns and keys): `Env, Year, Field_Location, Experiment, Replicate, Block, Plot, Range, Pass, Hybrid, Pollen_DAP_days, Silk_DAP_days, Plant_Height_cm, Ear_Height_cm`.
  - **Used for the decision**: only Silk_DAP_days (days to silking, q = 1). Flowering time is recorded in mid-season; its timing is unambiguous.
  - **Descriptive only**: ASI (silking − anthesis, computed per plot). It is highly correlated with silking; including it in the decision would add collinearity.
  - **Descriptive only**: Plant_Height_cm, Ear_Height_cm (measurement time cannot be verified).
  - **Never enter any feature**: Yield_Mg_ha, Grain_Moisture, Twt_kg_m3, Stand_Count_plants, Root/Stalk_Lodging_plants, Date_Harvested. Date_Planted and Date_Harvested are read only by a separate "calendar reader", used for the description in item 5 of §6, and are never joined to features.
- **Relation to the project protocol document §2.1**: the forward rule of §2.1 (no phenotype of the test year enters training or tuning) is unchanged for the forward scenario. Like SP and OL, SX is a separately defined scenario:
  - The yield of Y does not enter any fit;
  - The whitelisted records of Y become covariates only through a **fixed averaging rule**; no model is fitted and no parameter is chosen on the records of Y;
  - All numbers of SX are labelled SX and never labelled forward.
  - This branch does not modify the project protocol document. If merged, whether to add a sentence for SX to §2.1 is decided by the authors together with the user.
  - **User confirmation**: on 2026-09-30, before the structural counts, the user confirmed this information set in the conversation ("pre-harvest flowering records from other environments in the target year may be used; no yield of the target year is used").
- **Target years and scored cells**: Y ∈ {2016, 2018, 2020, 2022}; scored environments = the scored environments in `results/benchmark/splits.json`; scored cells = all lines with markers in these environments (both new and old lines). This is exactly the same as the forward benchmark, so the control (S0) can be compared directly. From `results/w1c/per_year.csv`, N ≈ 7,243 + 13,850 + 11,279 + 11,780 = 44,152, 3/√N ≈ 0.0143; the counts in §6 take precedence.
- **Outcome**: processed cell-mean yield (`pheno.parquet`, the mean over all plots), the same as in the benchmark.

## 2. Covariate construction (no yield involved)

1. **Plot → cell**: for each (Env, Hybrid), take the mean of the trait over all plots with a record (including plots with missing yield). Values outside the ranges are set to missing: flowering time 30–120 days, ASI −20 to +30 days, plant height 50–450 cm, ear height 10–300 cm. These constants were set from agronomic common sense; no distribution of any year was looked at before they were fixed.
2. **Within-environment standardisation**: within each environment e, compute the mean and standard deviation over all lines with a record of the trait (with or without markers), giving z; truncate to [−4, 4]. If an environment has fewer than 10 records of a trait, or a standard deviation of 0, that environment does not provide that trait.
3. **LOEO covariate**: for the cell of line i in environment j of year t, the covariate for trait m is
   x_ij^m = the **simple average** of z of line i over all environments in year t that are outside exclusion group G(j) and have that trait.
   - When there is no such environment, x = 0 (i.e. the environment mean), and the proportion of such cells is recorded.
   - No shrinkage and no model fitting, so the records of Y determine no parameter.
4. **Next-year covariate** (used only for the transfer test in §5): x̄_i^m = the average z of line i over all environments of Y.
5. **Naive own-environment covariate** (descriptive only): as in step 3, but including environment j itself. It brings in shared plot error (for example, a damaged plot with late silking and low yield). It is used to show "how much an evaluation without LOEO is inflated".
6. **Marker-predicted records** (F-class control, S1g in §3): for year t and trait m, take the cell z of all years before t as the response, fit with `cellspec.fit_ia(Prep(...), ω = 0)`, and predict the lines of year t, giving x̂_t^m. It uses no record of year t.

## 3. Models

The marker part of all models comes from the same object: **ĝ_t = the prediction of `cell_reml` fitted on all years before t, for the cells of year t** (u_t of `cellspec.Prep` + `cellspec.fit_ia(ω = 0)`, the frozen cellspec, blob 4ca8090). When t = Y, ĝ_Y is the control S0.

- **S0 = `cell_reml` (control)**: prediction = ĝ_Y. Consistency: within each scored environment, the Spearman with the benchmark `cell_reml` prediction has median ≥ 0.999 and minimum ≥ 0.99. If this fails, the implementation is checked first and no comparison is looked at.
- **S1 = rolling-calibration offset model (main model, trained in the deployment manner)**:
  - Calibration data: for each year t = 2015, …, Y − 1, the cells of "new lines of year t" (lines with no yield cell with markers before t), restricted to environments containing ≥ 25 such new lines. Each cell has ĝ_t and the LOEO covariate x = [silking] (LOEO computed over all environments of year t). Odd years are almost all old lines (`results/w1c/per_year.csv`: proportions of new lines in 2017, 2019 and 2021 are 0.04, 0.19 and 0.01), so in practice the calibration data come mainly from 2015 and earlier even years.
  - Fit: within each calibration environment, centre y, ĝ and x of these cells, then fit ordinary least squares y_c = a·ĝ_c + b′x_c (no intercept, no penalty, no parameter to tune).
  - Prediction: target cell (i, j) = a·ĝ_Y,i + b′x_ij. The within-environment ranking depends only on b/a; when b = 0 the ranking is the same as S0.
  - Reason for this form: 83–96% of the target year are new lines, whose ĝ comes only from markers. The calibration also uses "new lines of that year + ĝ from past data only + LOEO records", which matches the information structure at deployment. Joint REML (y = β′x + u) would let the training lines' own u absorb the between-line differences in x, so β would be underestimated and identified mainly by line × year variation; so it was not adopted (§11).
  - If a ≤ 0 in a target year, S1 for that year is recorded as "calibration failed" and judged NO-GO under §5.
  - If, for a target year, the calibration data have fewer than 5 environments each with ≥ 25 new-line cells, that year does not enter S1, and this is reported as is (2016 is expected to have only t = 2015 available, but 2015 has 27 environments with 0.72 new lines, which should suffice).
- **S1h (descriptive)**: as S1, with x = [silking, ASI, plant height, ear height].
- **S1@LOR (used for decision condition (j))**: as S1 (a and b unchanged), but the covariate of the target cell excludes all target-year environments in the same state as j (`cellspec.location(..., level='region')`). It tests whether the gain depends on same-state sister trials.
- **S1g (F-class control, used for decision condition (e))**: as S1, but x is replaced by the marker-predicted records x̂ (step 6 of §2); neither calibration nor target uses same-season records. S1g − S0 shows how much "historical secondary traits + markers" can bring on its own; this belongs to the forward scenario. If it crosses the resolution, it is a forward finding that needs a separate pre-registration and does not count as a gain of SX.
- **S1n (descriptive)**: as S1, but both calibration and target use the naive own-environment covariate (step 5 of §2).
- **S1@NY (transfer test, used for decision condition (f))**: the scored lines of Y are scored in the environments of year Y + 1; prediction = a·ĝ_Y,i + b′x̄_i (a, b are the calibration values for target Y); the control is ĝ_Y,i. Both use the same ĝ_Y (using only data before Y); the difference comes only from the records of Y. Scoring set: environments of year Y + 1 containing ≥ 25 "scored lines of Y"; only the cells of these lines are scored (counted first in item 6 of §6).
- Not done: joint REML, per-environment β, multi-trait GBLUP, LightGBM, neural networks; no post hoc new variants.

## 4. Metrics and pooling

- **Primary metric**: within-environment Spearman of each scored environment (≥ 25 lines). Descriptive: within-environment Pearson, top-10% selection differential. All models use the same set of environments. When a prediction is constant, Spearman is recorded as 0 and the environment is not dropped (S0 and S1 will not in practice be constant); the number of occurrences is reported.
- **Year effect**: the mean Δ across the scored environments of that year.
- **SE of the year effect (used for the decision)**: two-way bootstrap within year, B = 2000, seed 20260930:
  - In each resample, "environment clusters" (the exclusion groups of item 3 of §6; an environment without duplicates forms its own cluster) are sampled with replacement, and at the same time the lines of that year are sampled with replacement; **the same line resample is used for all environments of that year**;
  - Each environment computes Spearman on the sampled lines (duplicated lines are treated as ties);
  - All models and contrasts of the same year use the same set of resamples (paired).
  - Reason: x is a line-level covariate that is almost the same across environments within a year; resampling only environments would underestimate the SE. The SE from resampling environments only is also reported for reference.
- **Pooling**: DerSimonian–Laird, with the same zero-SE floor as before (`dartgxe.forward.pool`); this is the project convention. With k = 4 years the τ² estimate is very unstable, so the Hartung–Knapp (t₃) interval, Q and I² are also reported. **Inferential target**: conditional on these 4 G2F line cohorts, with environments and lines as sampling units. Whether this generalises to later cohorts is reflected only by leave-one-year-out and the transfer test (§5 (c)(f)).
- **Resolution**: 3/√N, N = number of scored cells used for the decision (fixed in item 1 of §6).

## 5. Decision (the only tested hypothesis H1 = S1 − S0, within-environment Spearman)

There is only one tested hypothesis, so no multiplicity correction. S1h, S1n, S1g − S0, and all Pearson and selection-differential results cannot trigger GO.

**GO requires all of the following**:
- (a) pooled Δ ≥ 3/√N;
- (b) lower bound of the DL 95% CI > 0 (year-effect SE from the two-way bootstrap);
- (c) leave-one-year-out: after removing any target year, all pooled point estimates > 0;
- (d) new lines: on new-line cells only (lines with no yield cell before Y; environments need ≥ 25 new lines), the pooled point estimate > 0;
- (e) value of same-season information: the pooled point estimate of S1 − S1g > 0, and ≥ 0.5 × (S1 − S0);
- (f) transfer: the one-sided 95% DL lower bound of pooled S1@NY − control > 0, and the transfer ratio R = (S1@NY − control)/(S1 − S0) (ratio of pooled point estimates) ≥ 0.5; the bootstrap CI of R is reported;
- (g) environmental-covariate baseline (the project protocol document §2.3): the pooled point estimate of S1 − `rn_ridge` (predictions in the benchmark package) > 0;
- (h) within-maturity ranking: within each environment, split lines into thirds by the LOEO silking covariate (a third counts only with ≥ 8 lines), average the within-third Spearman; the pooled point estimate of S1 − S0 > 0;
- (i) the consistency check and the shuffled-yield re-run (§7) both pass; a > 0 in every target year that enters S1;
- (j) same-state sister trials: the pooled point estimate of S1@LOR − S0 > 0.

**Fixed wording**:
- Only (b) is met: write "statistically detectable, but below selection resolution", record NO-GO.
- (a)–(d) are met but (e) is not: write "the gain can be obtained by a multi-trait model without same-season records (forward class); it is not a gain from same-season information", record NO-GO.
- (f) is not met: write "the gain does not transfer to the next year; it may be a year-specific or non-genetic (for example seed-lot) effect", record NO-GO. Whether the same set of G2F hybrids in two adjacent years came from the same seed lot is **not verified**; if so, (f) cannot exclude a seed-lot effect. This is a known blind spot.
- (h) is not met: write "the gain comes from recovering maturity differences between lines; ranking within maturity is not improved", record NO-GO.
- (j) is not met: write "the gain depends on same-state sister trials", record NO-GO.
- The DL interval excludes 0 but the HK interval includes 0: the GO text must add "GO depends on the interval conditional on these 4 cohorts; the interval with cohort as a random effect includes 0".
- All other cases: NO-GO.
- Errors found after scoring: may be corrected only if a pre-specified test or check should have found them; after correction both versions are reported, and the conclusion is labelled "post hoc correction", counting at most as exploratory.
- After a NO-GO, no new variants are run, and covariates, grids and parameters are not changed.

## 6. Structural counts (before any fitting; read only keys and whitelisted records, never any yield value)

`scripts/count_secondary.py` writes to `results/count_secondary/`. The results are committed as an appended section of this file; only then is the fitting code written.

1. Number of scored environments and scored cells N for each target year (fixes the N and 3/√N used for the decision).
2. **Coverage**:
   - Proportion of scored cells with a silking LOEO covariate coming from ≥ 2 other environments (per year);
   - The same proportion among new-line cells of the calibration years (2015–2021), and the number of calibration environments and cells available for each target year.
3. **Duplicate and sister-trial check (determines exclusion groups)**:
   - Assert that (Env, Experiment, Plot) is unique; for environments with Range/Pass, assert that (Env, Experiment, Range, Pass) is unique;
   - For each pair of environments in the same year at the same `cellspec.location` (location level) or the same state (`level='region'`), compute the proportion of shared lines whose per-plot (anthesis, silking, plant height) triplets are identical, and whether the plot layout (Plot/Range/Pass) is identical;
   - Environment pairs with an identical-triplet proportion > 20%, or an identical layout, are merged into the same exclusion group G(j). LOEO excludes the whole group, and the bootstrap uses the group as the cluster. The list of exclusion groups is committed with the counts.
4. **Whether missingness carries yield information** (presence/absence only): for each year, a per-plot cross-table of "record present/absent × yield present/absent"; the correlation, per line, between the number of records and the number of yield records in other environments. Reported only; the design is not changed.
5. **Calendar** (separate calendar reader): for each environment, silking date = Date_Planted + median days to silking, and Date_Harvested; report the proportion of scored environments and cells harvested before the silking date of the last environment of that year.
6. **Scoring set of the transfer test**: environments of year Y + 1 containing ≥ 25 scored lines of Y, and the number of cells N_NY.
7. **New-line scope**: number of scored environments with ≥ 25 new lines, and number of cells, per year.
8. **Power proxies** (using only results already seen; no new scoring):
   - Upper bound: the environment-level standard deviation and the between-year standard deviation of (`gm_ceiling` − `cell_reml_same_set`) for G2F 2016–2022 in `results/ideas_diag/ceiling_envs.csv`;
   - Lower bound: the same two standard deviations of `d_spearman` of `rn_ridge` for these 4 G2F years in `results/summary48/per_env.parquet`;
   - From these, compute the minimum detectable effect of (b) at 80% power. If the MDE under the upper-bound proxy is > 0.03, the appended section states "GO is almost impossible unless the true gain exceeds the MDE".
- **Stopping rule**: if fewer than 3 of the 4 target years have ≥ 80% of scored cells with the silking LOEO covariate of item 2, write "insufficient coverage" and do not continue fitting; also stop if fewer than 3 target years can enter S1.

## 7. Unit tests and consistency (before any comparison)

`tests/test_secondary.py`:
- **Whitelist**: the reader never reads yield, moisture, test weight, stand count, lodging or harvest date (check the columns actually read); panel files contain no y or any yield column; observed values are joined only at the scoring step.
- **LOEO**: x_ij contains no record from environment j or its exclusion group; positive control: after injecting own-environment records, this test must fail.
- **No leakage of target-year yield**: after replacing the yield of Y and all later years with junk values, the predictions and a, b of S0, S1, S1h, S1g, S1n, S1@NY and S1@LOR are bitwise identical.
- **Shuffled-yield re-run on real data** (before scoring, 4090): copy the yield table, replace the yield of each target year Y and later years with 1e6·N(0,1) (presence/absence unchanged), and re-run all panels. For each target year, the maximum relative difference of predictions, a, b and REML δ from the real run must be ≤ 1e-9; otherwise do not score and check the implementation first. The array hashes of both runs are written to verdict.json.
- **Records of Y do not affect S0 and S1g**: after perturbing all whitelisted records of Y, the predictions of S0 and S1g are bitwise identical.
- **Calibration**: (a, b) of S1 agree with the least-squares solution from a directly constructed design matrix on toy data; when b = 0 the ranking of S1 within each environment is the same as S0.
- **Exclusion groups**: no record from any environment in G(j) enters x_ij.
- Real data: S0 agrees with the benchmark `cell_reml` (§3).
- Run `pytest tests/test_splits.py` as required by the project protocol document §2.1.

## 8. Direction predictions and use of the results (fixed in advance)

- **Direction predictions**:
  - H1 (flowering time): 0 to +0.01; GO probability 0.1. Basis: flowering time has high heritability and can be predicted by markers; the increment of same-season records over markers may be very small; S1g may be close to S1.
  - S1h (adding plant height): +0.005 to +0.03, larger than H1.
  - S1n − S1 (own-environment inflation): positive, larger for the plant-height version.
  - Transfer ratio (S1@NY)/(S1 − S0): possibly < 0.5.
- **GO**: write a confirmatory pre-registration; confirmation must use data not used in this check. At present we have **no** such data: 2024 has no secondary traits; the licences of the various CIMMYT datasets are not verified, and Phenocart has different traits (NDVI, canopy temperature); any download first requires the user's consent. Until such data are obtained, a GO of SX is written in verdict.json, the ablation_log and the paper only as "an exploratory finding on outcomes already seen", and the "A is better than B" wording of the project protocol document §2.2 is not used.
- **NO-GO**: record as "adding flowering records from other environments in the same season on top of the forward information brought no resolvable gain (G2F, 4 new-line years)". Descriptive results of S1h, S1n and S1g are reported as usual, labelled "uncorrected, descriptive".
- **Use for the evaluation paper**: whatever the result, SX is reported as one step of the scenario spectrum (forward → old lines → SX → sparse testing), together with the existing scenarios. S1n − S1 is used to show "how much an evaluation of secondary traits without leave-one-environment-out is inflated", but it is a descriptive result and cannot be a conclusion in the title or abstract.

## 9. Implementation, computation and time

- New files only: `src/dartgxe/forward/secondary.py` (reusing `cellspec`, not modifying it), `scripts/count_secondary.py`, `scripts/headroom_secondary.py`, `tests/test_secondary.py`.
- Computation: 4090; about 10 fits of `cell_reml` size per target year (ĝ_t and x̂_t can be shared across target years); estimated total under 1 hour. Memory and thread limits as per project convention.
- Output: `results/headroom_secondary/`; record git commit, configuration, GPU, versions, start and end times. Scoring is executed only once.

## 10. Report template

- Write the decision first, then the description; the header of every descriptive table states "uncorrected, descriptive"; report the number of descriptive 95% CIs that exclude 0 and the expected number under the null hypothesis.
- "Proportion of the ceiling gap closed" is computed as the ratio of pooled estimates, with a paired bootstrap CI, and only on the same set of cells as in `ceiling_envs.csv`.

## 11. Trade-offs in the design review (record)

Adopted:
- The main model changed from "joint REML" to "rolling calibration trained in the deployment manner" (both the statistics and the realism reviews judged this a blocking issue);
- Plant height downgraded to descriptive (measurement time cannot be verified, and it is close to a proxy of yield);
- Added the F-class control S1g and condition (e);
- Main covariate is silking only (q = 1); ASI downgraded to descriptive;
- Transfer test made quantitative, (f) (one-sided lower bound and transfer ratio ≥ 0.5); within-maturity ranking condition (h); same-state exclusion condition (j);
- Shuffled-yield re-run on real data;
- Two-way bootstrap; report the HK interval and the inferential target;
- Duplicate/sister-trial check and exclusion groups;
- Missingness and calendar diagnostics;
- New-line scope uses the threshold of ≥ 25 lines (the project protocol document §2.2);
- `rn_ridge` condition (g);
- The "exploratory" clause and the statement that this is the 6th check family;
- The reasons for excluding NUST are only recording time and prior use as an outcome.

Not adopted:
- **Plot-level replicate split (H2)**: 3/√N_H2 ≥ 0.021 was known before fitting; only 33–39% of cells in 2018–2022 have replicates; and the field layout would need to be verified to ensure independent errors. So it is not done; own-environment inflation is described only with S1n.
- **Seed-lot diagnostic (residualising on stand count)**: stand count is not in the whitelist; seed lot is written into the wording of §5 as a known blind spot.
- **Year-effect SE taken as max(bootstrap, delete-one-environment jackknife)**: the jackknife needs about 3,000 covariate rebuilds; the two-way bootstrap already handles covariates shared across environments, so it is not added.
- **REML shrinkage of covariates (xblup)**: not adopted; a simple average is used instead, so that the records of the target year determine no parameter.
- **Per-environment β and per-location descriptive variants**: not done, to avoid forking paths.
- **Asking the user to reconfirm the information set before fitting**: the user explicitly requested this scenario; §1 of this file states the information set, which is public once committed, and the user can stop it at any time.

## Appendix (2026-09-30, structural counts completed, before any model code and fitting)

Code: `scripts/count_secondary.py` and the reading and covariate parts of `src/dartgxe/forward/secondary.py` (run on the 4090; no yield value is read; yield cells are used only as keys, and yield presence/absence is used only for item 4). Output `results/count_secondary/`.

1. **Scored cells**: 27 / 29 / 24 / 27 environments, N = 7,243 / 13,850 / 11,279 / 11,780, **total 44,152, 3/√N = 0.0143** (consistent with `results/w1c`); fixed as the N used for the decision.
2. **Coverage**: the proportion of scored cells with a silking LOEO covariate from ≥ 2 other environments is 0.999 / 0.991 / 1.000 / 0.996. Calibration data (new lines of year t, ≥ 25 per environment): 2015 27 environments 7,915 cells, 2016 27 / 5,590, 2018 29 / 12,862, 2019 26 / 3,034, 2020 24 / 10,543; 2017 and 2021 have no eligible environments. Calibration environments / cells available for each target year: 2016 27 / 7,915, 2018 54 / 13,505, 2020 109 / 29,401, 2022 133 / 39,944.
3. **Duplicates and sister trials**:
   - The pre-registered requirement "(Env, Experiment, Plot) is unique" **does not hold**: 60,497 rows involve duplicated keys, concentrated in 2014, 2016, 2017 and 2018. After inspecting keys and records, the reason is that plot numbers in these years restart within each replicate (Replicate); rows with the same key are different plots with different lines and different records; the number of rows with identical whitelisted columns is 0. After adding Replicate, 1,121 rows still have duplicated keys. These are all within a single environment and do not affect LOEO (LOEO works at the environment level); recorded as is.
   - Among the 174 environment pairs in the same year and the same state: of the 104 comparable pairs, the median proportion of shared lines with identical per-plot (anthesis, silking, plant height) is 0, and the maximum is 0.10 (NYH1_2017 and NYH3_2017, 1 of 10 lines); 0 pairs have an identical layout. **No environment pair met the merging condition; all exclusion groups are single environments** (`exclusion_groups.json`).
4. **Record presence and yield presence** (per plot): among plots with yield, the proportion with a silking record is 0.67–0.87; among plots with missing yield, it ranges from 0.01 (2017) to 0.74 (2022), clearly lower in early years. So missingness is associated with missing yield; under §6 this is only reported, and the design is not changed.
5. **Calendar**: the proportion of scored environments harvested before the silking date of the last environment of that year is 0 / 0.034 / 0.042 / 0 (cell proportions 0 / 0.029 / 0.022 / 0), all below 25%.
6. **Scoring set of the transfer test**: environments / cells in year Y + 1 containing ≥ 25 scored lines of Y: 2017 30 / 5,605, 2019 28 / 11,962, 2021 25 / 13,547, 2023 26 / 11,478.
7. **New-line scope**: in every year, all scored environments have ≥ 25 new lines; new-line cells 5,590 / 12,862 / 10,543 / 10,999.
8. **Power proxies** (using only results already seen): upper bound (same-year yield ceiling gap) environment-level SD 0.134, between-year SD 0.041, pooled SE ≈ 0.024, MDE at 80% power ≈ 0.068; lower bound (`rn_ridge` − `cell_reml`) 0.090, 0.038, SE ≈ 0.021, MDE ≈ 0.058. **Both are > 0.03, so as required by §6 it is stated: GO is almost impossible unless the true gain exceeds about 0.06.** Both proxies have large between-year differences; with 4 years, τ² dominates the pooled SE. If the SX gain is stable across years, the actual SE will be much smaller, but this can only be seen at scoring.

**Stopping rule: continue** (coverage ≥ 80% in all 4 target years; all 4 target years have ≥ 5 calibration environments).

## Results (2026-09-30; 4090; code b6e4919; `results/headroom_secondary/`; panels kept on the 4090 in `scratch_ideas_2026-09-29/headroom_secondary/`)

**Process**: the 7 tests of `tests/test_secondary.py` (including whitelist, LOEO positive control, no leakage of target-year yield, and target-year records not affecting S0 and S1g), together with `test_oldlines.py` and `test_splits.py`, 26 in total, all passed on the 4090. One problem was found and fixed during implementation: LOEO originally used "total minus own environment", which left the own-environment records as a rounding residue of about 1e-15; it was changed to sum only over the retained environments, which is bitwise independent (found by a test before commit). The main panels and the shuffled-yield panels took about 3 minutes each on the two RTX 4090s. **Both gates passed**: after shuffling all yields of Y and later years, the maximum relative difference of all predictions, a, b and δ was 0; the within-environment Spearman between S0 and the benchmark `cell_reml` had median 1.0 and minimum 0.99999994, with no missing values; N = 44,152, consistent with the counts; a > 0 in all 4 target years (0.26–0.37). Scoring was executed only once.

**Decision (H1 = S1 − S0, within-environment Spearman): NO-GO.**

| Condition | Result | Met? |
|---|---|---|
| (a) pooled Δ ≥ 0.0143 | **+0.0005** (year effects −0.0026, −0.0021, −0.0007, +0.0054) | No |
| (b) lower bound of DL 95% CI > 0 (two-way bootstrap SE) | [−0.0035, +0.0045]; HK [−0.0038, +0.0030] | No |
| (c) leave-one-year-out all > 0 | −0.0015 after removing 2022 | No |
| (d) new lines | +0.0003 | Yes |
| (e) S1 − S1g > 0 and ≥ 0.5×H1 | +0.0001 (< 0.5 × 0.0005) | No |
| (f) next-year transfer | NY +0.0041 [−0.0026, +0.0108], one-sided lower bound < 0 | No |
| (g) S1 − `rn_ridge` > 0 | +0.0184 | Yes |
| (h) within maturity thirds | +0.0002 | Yes |
| (i) gates and a > 0 | passed | Yes |
| (j) same-state exclusion | +0.0004 | Yes |

**Calibration coefficients (using only new-line cells before Y)**: b/a for silking in the target years is +0.21, −0.11, −0.04, +0.11; the sign is unstable and the magnitude is small. That is, in historical data, silking in other environments has almost no explanatory power for the yield ranking beyond ĝ.

**Descriptive (uncorrected)**
- **S1g − S0 (marker-predicted silking, forward class)**: +0.0009 [−0.0046, +0.0063].
- **S1n − S0 (naive own-environment covariate)**: −0.0012 [−0.0048, +0.0023]. Silking shows no "inflation without LOEO".
- **S1h − S0 (adding ASI, plant height, ear height)**: pooled +0.0152 [−0.0560, +0.0865], but extremely inconsistent between years: 2016 **−0.122**, 2018 +0.056, 2020 +0.049, 2022 +0.069 (τ² = 0.0049); the Pearson version is similar (2016 −0.114, others +0.06 to +0.08). The calibration for 2016 has only the year 2015, and its plant-height coefficient differs strongly from other years (ASI −0.45, plant height +0.22 vs plant height +0.36 to +0.50 in the other years). The measurement time of plant height cannot be verified (§0); it may partly be a proxy of same-season yield. This result can only be a description; it cannot be used to change the decision or to run more analyses.
- Top-10% selection differential of H1 −0.0027 [−0.022, +0.017]; within-environment Pearson −0.0011.
- Transfer ratio R = NY/H1 = 8.5 (the denominator is close to 0; meaningless).

**Comparison with the predictions in §8**: H1 predicted 0 to +0.01: +0.0005, holds (at the low end); GO probability 0.1 did not materialise. S1h predicted +0.005 to +0.03: pooled +0.015 lies within the range, but is inconsistent between years. S1n − S1 predicted positive: −0.0010, does not hold.

**Reading (under §8)**
1. **Adding silking records from other environments in the same season on top of the forward information brought no resolvable gain (G2F, 4 new-line years)**: point estimate +0.0005, far below 0.0143, and there is no statistically detectable signal either.
2. Plant-height-type traits have large descriptive positive values in 3 years and a large negative value in 2016. Because the measurement time cannot be verified and the calibration is unstable, this cannot be read as "secondary traits are useful". A test would need data with verifiable measurement times and a new pre-registration.
3. Combined with OL and the earlier forward and sparse-testing results: at present the only improvement that crosses the resolution comes from having some yield already available in the target year (M×E in sparse testing); same-season flowering records, the old lines' own history, and marker G × location did not cross the resolution.
