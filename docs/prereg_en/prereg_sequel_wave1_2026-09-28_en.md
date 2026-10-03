English translation of the pre-registration document prereg_sequel_wave1_2026-09-28.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Sequel pre-registration, wave 1 (2026-09-28)

Written before any run in this wave. Not changed after commit; amendments are only appended as dated sections. For background, see `docs/sequel_paths_2026-09-28.md` (path C) and `docs/sequel_elevation_2026-09-28.md` (E1, E2).

## 0. User decisions (2026-09-28)

1. Take path C: audit + criterion + tools as the backbone; REML plateau analysis; forward combination (B3) as a conditional item.
2. The atlas decision stacking (`<lab-data>/analysis/atlas/stack_select.py`, completed separately) is merged into the sequel. That directory is read-only. When needed, files are copied into this repository by commit, with the source noted.
3. The original GEFormer authors are not contacted. The wording restriction is unchanged: write only "the pipeline in the public code is ...".
4. The work of a parallel planning stream is merged into this analysis (see `docs/lit_scan_sequel_2026-09-28.md` section 8).

## 1. Overview of this wave

| ID | Content | Uses test-year phenotypes? | Nature |
|---|---|---|---|
| W1a | Plateau analysis of shrinkage selection (predictions 1 and 2 of E2) | Yes | Pre-registered test |
| W1b | Compute t/t\* on the existing final predictions of gate 2 (E1) | Yes (the predictions already exist; t was never computed) | Descriptive check, direction fixed in advance |
| W1c | Pre-study counting forward years (path B1) | No, only counts sample structure | Pre-study; decides whether to do B3 |
| W1d | Feasibility pre-study for GE-BiFormer and SINN (path A1) | No, only measures speed and whether the model runs on the forward splits | Pre-study; decides the scope of A1 |

## 2. W1a Plateau analysis of shrinkage selection

**Data and scenarios**: v2 data (`data_v2/`), 5 new-hybrid forward scenarios: FYnew2016s, FYnew2018, FYnew2020, F2022m, F2024m. Gate 1 used the first 4 scenarios on v1 data. This wave uses v2 throughout, and uses the gate-1 results for the first 4 scenarios on v1 as a consistency check.

**Model**: the same as gate 1, i.e., standardised all-marker ridge regression + environment fixed effects (`Ridge`, FWL). For each scenario (forward scenarios have only 1 fold): REML (`r.reml()`) is run on the final fitting set. In addition, the test year is predicted on a 36-point grid λ_rel ∈ logspace(−4, 3, 36).

**Recorded for each λ**:
- S(λ): test-year within-environment Spearman, pooled with weights by number of cells, with the same definition as gate 1;
- M(λ): test-year within-environment MSE. Within each environment, the respective environment means are subtracted from both predictions and observations; the result is then averaged with weights by number of cells. It is sensitive to the prediction scale and is used to test "scale attenuation affects only the MSE optimum".

**Definitions**:
- λ_S = argmax S, λ_M = argmin M;
- N = number of test cells; plateau P = {λ : S(λ) ≥ S(λ_S) − 3/√N}. A version with the ×1.5 practical resolution is also reported, as a sensitivity analysis;
- The REML λ is not on the grid. S(λ_REML) is computed directly from its own predictions, and the verdict is S(λ_REML) ≥ S(λ_S) − 3/√N.

**Prediction 1 (primary)**: the REML λ falls within the plateau in ≥4/5 scenarios.
- Pass: ≥4/5 → the conclusion is written as "the loss from REML shrinkage in deployment is below the selection resolution (k of 5 forward years)".
- Fail: ≤3/5 → reported as is. The sequel's REML recommendation is downgraded to "sufficient in x/5 years", and the years outside the plateau and their gaps are reported.
- Known information, declared as is: gate 1 on v1 already gave S(λ_S) − S(λ_REML) for 4 years: 0.031, 0.000, 0.059, 0.010. With 3/√N ≈ 0.026–0.03, the years 2016 and 2020 are expected to be possibly outside the plateau. So prediction 1 is **expected to fail** on v1, and F2024m is the only new information. This is written here before seeing the v2 results, to prevent changing the definition after the fact.

**Prediction 2 (secondary)**: the across-year SD of log10 λ_M is larger than the across-year SD of log10 λ_S.
- Criterion: the ratio SD(log λ_M)/SD(log λ_S) > 1, and the lower bound of the 95% CI of the ratio > 1. The CI comes from a "within-year environment-cluster bootstrap" (2,000 replicates; in each replicate, environments are resampled within each scenario and λ_S and λ_M are recomputed).
- Known information, declared as is: gate 1 already gave λ_S for 4 years on v1: 2.51, 0.1, 251, 1.0 (log10 SD ≈ 1.4). λ_M was never computed.
- When λ_S or λ_M falls on a grid endpoint, this is flagged separately. If λ_M falls on an endpoint in ≥2 scenarios, prediction 2 is recorded as "undecidable".
- This is a test of the first-order derivation in A5. Whatever the result, the derivation is not written as "proven".

**Power note**: Prediction 1 is a year-by-year descriptive verdict and involves no power. Prediction 2 has only 5 years. Sampling variation between years cannot be estimated, and the bootstrap reflects only within-year environment sampling. So even if prediction 2 passes, it is written only as "consistent across 5 forward years".

**Computation**: CPU, 4090 host, adapted from `scripts/gate1_reml.py`; new script `scripts/w1a_plateau.py`. Results are written to `results_v2/w1a/` and synced back to `results/w1a/` in this repository.

## 3. W1b t/t\* check (existing gate-2 predictions)

**Objects**: gate-2 final predictions, {F2024m, F2022m} × {own, fair} × seeds {147, 1, 2}, 12 in total.

**Computation**: For each prediction, following the definitions of E1, compute m_j (environment mean of the predictions), s_ij (centred within environment, then divided by the within-environment SD), t (cell-weighted mean of the within-environment prediction SDs), a = Cov(m, y), v_b = Var(m), k = weighted mean of the within-environment covariance Cov_w(s, y), t\* = k·v_b/a, and the ratio t/t\*. Compare the formula r_max with the observed pooled Pearson.

**Direction (fixed in advance)**: by the proposition, the own configuration, selected by pooled Pearson, should have a within-environment scale closer to t\*.
- Supported: in both scenarios, own's |log(t/t\*)| (mean over 3 seeds) is smaller than fair's; and own's t is smaller than fair's t.
- Not supported: any other case; reported as is.
- This is a descriptive check (only 3 seeds per group). No significance conclusion is drawn. It gives a direction for the trial-level test in A1 and cannot replace A1.
- If a ≤ 0 (the environment means of the predictions are negatively correlated with the actual ones), t\* is undefined and that prediction is marked "not applicable".

**Computation**: CPU. The predictions are on amax (`<server>/dart-gxe/results_v2/`) or on the 4090; new script `scripts/w1b_tstar.py`.

## 4. W1c Pre-study counting forward years (no method results are viewed)

**Data**: multi-environment data already in `<lab-data>` (read-only, read via `GP_DATA`): G2F, NUST soybean, URSN spring wheat, SoyNAM, DROPS, CUBIC, VEF.

**Statistics per year for each dataset**: number of environments; number of genotypes per environment; number of environments with ≥25 genotypes (a relaxed definition with ≥10 is reported separately); the proportion of that year's genotypes that never appeared in any earlier year ("new-genotype proportion").

**Qualifying forward year**: at least 2 years of data before that year; ≥5 environments in that year, each with ≥25 genotypes (relaxed to ≥10 for URSN, and noted); new-genotype proportion ≥0.5.

**Decision (fixed in advance)**:
- Total number of qualifying forward years K ≥ 12 → write pre-registrations for B3 (forward decision stacking) and E5 (CV ranking transfer), and start the runs no later than 2026-12-15;
- K < 12 → B3/E5 are not done. Atlas stacking is reported only as a supplementary result under CV1, with the statement "not a deployment condition".

## 5. W1d Feasibility pre-study for GE-BiFormer and SINN (no test-year results are viewed)

**Content**: write adapters that plug both models into the F2024m and F2022m splits. Do speed tests and one short training only within the training years (validation-year metrics may be viewed; the test year is not scored). Record the time per epoch, GPU memory, and the total GPU time estimated under the gate-2 budget (each scenario × each pipeline: 8 trials, 100 epochs, 3 seeds).

**Time box**: 1 week per model (until 2026-10-05). A model that does not run by the deadline, or whose estimated total GPU time is >72 hours, is removed from A1; only the code-audit facts are kept.

**If passed**: write a separate pre-registration for A1. Its content: own pipeline vs fair pipeline; saving the predictions of every trial for the trial-level t/t\* test; the MDE. MDE following gate 2 amendment 1: about 0.06 within-environment for a single (model, scenario). The pooled criterion uses the directional consistency of O across the 6 (model × scenario).

## 6. Hard-rule check

- Leakage prevention: REML in W1a uses only the final fitting set. Grid results serve only as the upper bound and the plateau, and are not used to select any reported method. W1b only recomputes descriptive quantities of existing predictions. W1c and W1d do not touch test-year phenotypes.
- Evaluation: the primary metric is within-environment Spearman. MSE is used only to test prediction 2, not to rank methods.
- Reproducibility: each script records the git commit, data root directory, host, and start and end times. Result file paths are written back to the results section of this file.

## Results (2026-09-28; W1a 4090 CPU/GPU1, code fe793b9, verdict script 5db9e6d; W1b amax CPU, code 5db9e6d)

Output: `results/w1a/{curves.parquet, reml.csv, run_meta.json, verdict.json}`, `results/w1b/{tstar.csv, verdict.json}`.

### W1a (v2 data, 5 new-hybrid forward scenarios)

| Scenario | N | 3/√N | S(λ_S) | λ_S | S(REML) | λ_REML | Gap | Within plateau | λ_M |
|---|---|---|---|---|---|---|---|---|---|
| FYnew2016s | 7,243 | 0.035 | 0.277 | 2.51 | 0.248 | 0.231 | 0.030 | Yes | 3.98 |
| FYnew2018 | 13,850 | 0.025 | 0.117 | 0.158 | 0.117 | 0.089 | 0.001 | Yes | 39.8 |
| FYnew2020 | 11,279 | 0.028 | 0.103 | 251 | 0.044 | 0.073 | 0.059 | No | 25.1 |
| F2022m | 11,780 | 0.028 | 0.240 | 0.631 | 0.229 | 0.057 | 0.011 | Yes | 0.398 |
| F2024m | 9,486 | 0.031 | 0.319 | 63.1 | 0.224 | 0.046 | **0.094** | No | 3.98 |

- **Prediction 1: failed** (3/5 within the plateau; also 3/5 under the ×1.5 practical resolution). Consistent with the advance statement, 2020, already known on v1, is still outside the plateau. The new information, F2024m, has a gap of 0.094, about 3 times the resolution. The mean gap over the 5 years is 0.039 (0.025 for the first 4 years on v1).
- **Prediction 2: failed, in the opposite direction.** SD(log10 λ_M) = 0.79, SD(log10 λ_S) = 1.35, ratio 0.59, 95% CI [0.40, 0.84]. The across-year variation of the MSE-optimal λ is **smaller** than that of the Spearman-optimal λ. The first-order derivation of A5 §1.2(b) (scale attenuation affects only the MSE optimum) is refuted by the data and withdrawn.
- **Exploratory observations (not in the pre-registration; no conclusions drawn)**:
  - In all 5 years, both λ_S and λ_M are larger than λ_REML (λ_S/λ_REML from 1.8 to about 3,400; λ_M/λ_REML from 7 to about 450). In these 5 forward years, REML shrinkage is insufficient in every year.
  - Multiplying the REML λ by a fixed factor: no factor is favourable in every year. ×10 is close to optimal in 2016 and 2022, but worse in 2018 and 2024. ×1000 is close to optimal in 2020 and 2024, but clearly worse in 2016 and 2022.
  - Therefore the explanation "REML is optimal in expectation under covariate shift" (Patil et al. Prop 3) is **not supported**. Systematic under-shrinkage is more consistent with regression shift (marker effects change between years, Thm 5 of the same paper), but this is only an interpretation and was not tested.

### W1b (gate-2 final predictions, t/t\*)

| Scenario | Pipeline | t | t\* | \|log(t/t\*)\| | Pooled r | Formula r_max | Within-environment Spearman |
|---|---|---|---|---|---|---|---|
| F2024m | own | 0.380 | 0.167 | 0.87 | 0.389 | 0.407 | 0.078 |
| F2024m | fair | 0.452 | 3.546 | 1.52 | 0.187 | 0.221 | 0.162 |
| F2022m | own | 0.840 | 0.761 | 0.28 | 0.363 | 0.366 | 0.216 |
| F2022m | fair | 0.893 | 2.703 | 0.99 | 0.162 | 0.239 | 0.274 |

(Means over 3 seeds. For F2022m fair seed 1, a < 0 and t\* is undefined; by the advance rule it is marked not applicable, and the mean uses only the other 2 seeds.)

- **By the direction fixed in advance: supported.** In both scenarios, own has the smaller t and is also closer to t\*. Descriptive, 3 seeds per group.
- **A decomposition outside the criterion that must be reported alongside**: pooled r = r_max × efficiency.
  - The gap between own and fair comes mainly from the environment-mean part of r_max: a²/v_b is 0.97–2.12 for own and 0.004–0.68 for fair (excluding the 1 prediction with a < 0).
  - Scale efficiency (pooled r / r_max): own 0.92–1.00, fair 0.48–0.99.
  - That is, for GEFormer, **the main route by which the public pipeline raises pooled Pearson is selecting, on the test year, the configuration that happens to fit the test-year environment means**. Pushing the scale towards t\* is a secondary route. The E1 criterion holds, but it explains only a small part.

## Amendment (2026-09-28, before the W1c count): W1c operating rules

From this date, the sequel is handled solely by one designated analysis stream. (The work of the former an earlier exploratory stream analysis stream has been fast-forward merged on main, dd63ccd.) Details not specified in §4 are completed below, before any count. W1c only counts sample structure and does not look at any method results.

1. **Trait**: only one main trait is counted per dataset, and a forward year is counted only once. G2F grain yield (`pheno.parquet` of `data_v2`); NUST `YieldBuA`; URSN `DIS` (Fusarium head blight score, the only trait in this dataset measured in most environments); SoyNAM `yield`; DROPS `yield`; VEF `Yd_BLUE`. The CUBIC phenotype table has no year field (each hybrid has one column for each of 5 locations), so forward years cannot be constructed; only the structure is reported.
2. **Environment and year**: the environment definitions of the loaders in `<lab-data>/analysis/atlas/atlas.py` are followed. NUST: location + the year at the end of the trial name; multiple trials within the same location-year are averaged by genotype. URSN: `env_code`, with the year taken from `year`. VEF: `Trial`, with the year taken from the two digits in the trial name. SoyNAM, DROPS: environment and year from the processed tables. G2F: `env`, `year`.
3. **Genotypes**: only genotypes with marker data are counted. IDs are normalised in the same way as in the atlas loaders (forward GP needs markers).
4. **Environment size threshold (main definition)**: ≥25 genotypes; ≥10 for URSN (already stated in the original text of §4). A relaxed count with ≥10 for all datasets is reported separately, as sensitivity only, and does not enter the decision.
5. **"At least 2 years of data before"**: before that year, at least 2 years each have ≥1 environment that reaches the main threshold.
6. **New-genotype proportion**: among the genotypes that appear in that year's environments reaching the threshold, the proportion that had no record for that trait in any earlier year (any environment, no threshold).
7. **Decision**: K = the sum, over datasets, of forward years that qualify under the main definition. B3/E5 are decided by the K ≥ 12 rule of §4.
8. Local data not in the §4 list (CAIGE wheat, IRRI rice) are not counted in K.

Script `scripts/w1c_forward_years.py`, run on the 4090 in the project environment. Non-G2F data are read-only copies of `<lab-data>/data/`; SHA256 is recorded on upload.

## W1c results (2026-09-28; 4090 CPU, code cbee7e3; input SHA256 in 4090 `<workstation>/dart-gxe/external_ro/gp_data_2026-09-28/SHA256SUMS`)

Output: `results/w1c/{per_year.csv, verdict.json, run_meta.json}`.

| Dataset | Qualifying forward years K | Years | Cells per year (median, range) | 3/√N (median) | Genotypes per year (median) |
|---|---|---|---|---|---|
| G2F | 5 | 2016, 2018, 2020, 2022, 2024 | 11,279 (7,243–13,850) | 0.028 | 1,039 |
| NUST soybean | 15 | 2005–2017, 2019, 2020 | 1,940 (1,056–3,493) | 0.068 | 195 |
| URSN spring wheat (≥10 definition) | 11 | 1999–2004, 2008, 2009, 2014, 2015, 2019 | 78 (56–148) | 0.340 | 14 |
| SoyNAM, DROPS, VEF | 0 | — | — | — | — |
| CUBIC | Undecidable | Phenotype table has no year field | — | — | — |

- **K = 31 ≥ 12. By the rule fixed in advance in §4: write pre-registrations for B3 (forward decision stacking) and E5 (CV ranking transfer), and start the runs no later than 2026-12-15.** Under the relaxed definition (all ≥10), K = 32, which adds only NUST 2004.
- Self-check of the counting logic: by the same rule, G2F gives exactly the 5 known new-hybrid years (2016, 2018, 2020, 2022, 2024). The repeated-hybrid years have new-genotype proportions of 0.000–0.194, and all fail.
- Reasons for failing: SoyNAM 2013 has a new-genotype proportion of 0 (all had already appeared in 2012), and 2012 has only 1 earlier year. DROPS has only 2 years with the same set of 246 hybrids. VEF plants the same panel repeatedly (new-genotype proportion 0.011–0.311, and only 1–4 trials per year).
- **Limitations that must be carried into the B3/E5 pre-registration (facts, not judgements)**: the amount of information per year differs by an order of magnitude. URSN has only 12–28 marker-typed lines and 56–148 cells per year. Its single-year resolution is 0.25–0.40, which gives almost no power for effects of order 0.03. NUST single-year resolution is 0.05–0.09, about 2–3 times that of G2F. The MDE table in the paths document assumes every forward year is like G2F (about 25 environments × 450 genotypes) and **cannot be applied directly**. The B3/E5 pre-registration must recompute the MDE using the actual number of cells per year and the heterogeneity between datasets. It must also state in advance how years are weighted and whether URSN is used only as sensitivity.

## W1d interim record (2026-09-28, no test-year results viewed)

- **SINN removed from A1**: the public repository WUR-AI/sinn, at its latest commit 00248e3 (2026-07-29), still contains only a README. The README states "This repository is currently a placeholder" and that the code "will be made available upon publication". This is handled under §5, "a model that does not run is removed from A1". In the sequel, SINN is only cited by its published numbers (MixINN Table 1 agrees with SINN preprint Table 2: pooled r 0.63 vs 0.43, within-environment 0.38 vs 0.38). There is no paired rerun and no reimplementation (a reimplementation would no longer be "the pipeline in the public code"). **Correction**: path A1 in `docs/sequel_paths_2026-09-28.md` states "SINN (WUR-AI/sinn is public ...)". This is wrong; the code is not public. The claim "its tuning is compliant" comes from the paper text, not from the code.
- **GE-BiFormer**: Zhoushuchang-lab/GE-BiFormer @ 6563b12 (2026-06-19, the same commit as in the literature-scan audit), MIT licence, PyTorch. The inputs are a marker × hybrid matrix, an environment × covariate table, and a long phenotype table. data_v2 can provide these directly: 2,425 markers; official APSIM environmental covariates, 644 columns. Environments missing covariates: 3/28 in 2022 and 1/22 in 2024 (1–4 in other years). Environments missing covariates are handled as for GEFormer in gate 2 (that model's cells are dropped, and the comparison uses the common set). This is written into the A1 pre-registration.
- Direct cloning from GitHub on the servers failed (90-second timeout). The code was transferred to the 4090 and amax as a git bundle from the local machine.
- Next step (time box until 10-05): an adapter (F2024m/F2022m splits → GE-BiFormer input format). The "own pipeline" calls, unchanged, its logic that tunes the learning rate, early-stops and selects the best epoch by `test_loss`. The "fair pipeline" is changed to monitor the validation year. Only one short training run and timing within the training years.

## W1d results (2026-09-28; 4090 GPU1 RTX 4090, code 8d6a523; `results/w1d/gebiformer_bench.json`)

**Correction**: the "official APSIM environmental covariates, 644 columns" in the interim record above is wrong. The data build report `build_report.json` records 654 columns (262 of 294 environments have covariates). The number 644 was taken from the header comment of `src/dartgxe/data/g2f.py`; that comment has been corrected at the same time.

**Verdict: GE-BiFormer is included in A1** (§5 rule: it runs, and the estimated GPU time under the budget is ≤ 72 hours).
- Budget: 2 scenarios × 3 seeds × 3 training runs per seed (own: 1; fair: 1 tuning + 1 retraining) × up to 100 epochs. Based on the measured time per epoch on the training years, scaled to the final fitting set by number of cells, the total estimate is about **7.3 GPU hours** (F2024m 4.1, F2022m 3.2). 9.0–12.8 seconds per epoch, peak memory 4.4–4.7 GB, 17.59 million parameters.
- Inputs: 2,425 markers, 654 environmental covariates. **Environments missing covariates cause 11–13% of cells to be dropped** (F2024m final fitting set 12,137 / 106,721; F2022m 10,122 / 83,419). The A1 pre-registration must state that comparisons use only the common cells for which both sides have predictions (as in gate 2), and that dropped environments are reported.
- Validation-year check (validation year only, which is permitted; 30-epoch short training, seed 147): for F2024m the validation year is 2022, with mean within-environment Spearman 0.236 (24 environments) and the lowest validation loss at epoch 21. For F2022m the validation year is 2020, with 0.042 (21 environments), and the validation loss rises from epoch 1. The latter is consistent with stage 1, where all methods were weak on new-hybrid validation years (0.05–0.16). Description only; no conclusion.

**Audit facts from reading the code (6563b12; only the behaviour of the public code is stated)**:
1. The generalisation experiment script `code/run_generalization_experiments.py` adjusts the learning rate by the test-set loss (line 157) and early-stops on it (line 168). What it reports is the minimum test loss across epochs (lines 164–165, 176–177);
2. The RobustScaler for markers and the StandardScaler for environmental covariates are fitted on all hybrids and all environments, including the test set (`load_genotype_data`, `load_environment_data` in `algorithm/dataset.py`). Missing markers are imputed at random without a seed;
3. The code reads `config.py` (100 epochs). The `config.json` in the README specifies 300 epochs, but no script reads it;
4. EarlyStopping restores the best weights only when it is triggered (`algorithm/utils.py` lines 58–61).

Wave W1 is fully complete: W1a, W1b, W1c, W1d. The next step is the wave-2 pre-registrations (A1, the E1 trial-level test, B3/E5), planned for commit before 10-12. The pooled criterion of A1 must be rewritten: after SINN is removed there are only 2 models × 2 scenarios = 4 groups. For "all with the same sign", the smallest possible p of a one-sided sign test is 0.0625, which cannot reach 0.05.

Addendum (2026-09-28, same day): the "11–13%" above refers only to the final fitting set. By role, the dropped proportions are: F2024m training 12.1%, validation 12.7%, final fitting 11.4%; F2022m training 10.8%, validation 16.5%, final fitting 12.1% (`results/w1d/gebiformer_bench.json`).

## Sensitivity (2026-09-30, WP4): W1b recomputed with the exact decomposition

The t and k of W1b also used the approximate decomposition (see the same-day correction in wave 2a). Recomputed in the exact form (`scripts/wp4_w1b_exact.py`, `results/wp4/w1b_exact*.{csv,json}`): in both scenarios, "the own pipeline has a smaller scale" and "is closer to t\*" still hold. The W1b conclusion is unchanged.
