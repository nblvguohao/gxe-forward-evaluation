English translation of the pre-registration document prereg_sequel_wave2a_2026-09-28.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Sequel pre-registration, wave 2 A: A1 multi-model paired reproduction + E1 epoch-level test (2026-09-28)

Written before any run in this wave. Not changed after commit; amendments are only appended as dated sections. Wave 2 B (B3 forward decision combination, E5 CV ranking transfer) is written separately and committed before 2026-10-12. Basis: `docs/prereg_sequel_wave1_2026-09-28.md` (W1 results), `docs/gate2_geformer_2026-09-26.md`, [internal planning document].

## 0. User decisions and known information

- User decision (2026-09-28, "go with your suggestion"): A1 no longer uses a pooled significance test across models. Instead, confidence intervals are reported for each (model, scenario) separately. While doing the A3 code audit, also look for models that can be plugged into the G2F forward splits; if one is found, write an additional appended pre-registration.
- Known information, declared as is:
  - The gate-2 GEFormer results are known (O = own − fair: within-environment Spearman F2024m −0.084, F2022m −0.058; pooled Pearson +0.202, +0.200). The W1b decomposition is also known.
  - GE-BiFormer has been viewed only once, on the validation year, in a 30-epoch short training (within-environment Spearman F2024m 0.236, F2022m 0.042). **The test year has never been predicted or scored**.

## 1. Models, scenarios, seeds

- **GE-BiFormer** (Zhoushuchang-lab/GE-BiFormer @ 6563b12, adapter `src/dartgxe/external/gebiformer.py`). Scenarios F2024m, F2022m (1 fold each for the forward scenarios). Seeds {42, 1, 2}, where 42 is the `random_state` in its `config.py`; its code itself does not set a torch seed.
- **GEFormer**: O and C_fair are taken from the gate-2 results and not recomputed. E1 needs test-year predictions for every epoch, and gate 2 did not save them. So only the gate-2 final retraining is rerun: own trains for 100 epochs on all years before the test year, with the best hyperparameters of the gate-2 study; fair trains for E\* epochs. Seeds {147, 1, 2}; test-year predictions are saved for every epoch. The rerun results **do not replace** the gate-2 numbers. If the predictions at the reported epoch differ from gate 2 (GPU non-determinism), the difference between the two is reported.
  - Hardware exception: the project protocol document requires that official numbers are run only on the 4090. The GEFormer rerun is done on the amax H200, the same hardware as the gate-2 GEFormer runs, so that it can be compared with gate 2. E1 is only a mechanistic description and produces no method ranking.

## 2. The three GE-BiFormer pipelines

All pipelines drop the environments without APSIM covariates (W1d: 10.8–16.5% of cells dropped across roles). Covariate availability is the same for the three pipelines, so the cells used in the comparisons are naturally the same. Dropped test environments are listed and not scored.

| Pipeline | Fitting set | Scalers fitted on | Learning-rate schedule and early stopping monitor | Reported prediction |
|---|---|---|---|---|
| **own** (public code moved to the forward setting) | All years before the test year (train ∪ val ∪ refit_extra) | All hybrids and environments of the scenario, **including the test year** (as its loader does) | **Test-year** Huber loss | The epoch with the lowest test loss (its script reports best_test_loss; the decision rule is the same as in its code: `loss < best − min_delta`) |
| **own_clean** (secondary, for decomposition) | Same as own | Only the hybrids and environments of the fitting set | Test-year Huber loss | Same as own |
| **fair** (fair pipeline) | Tuning: train; retraining: all years before the test year | Train during tuning; the retraining set during retraining | During tuning, the **validation year**; E\* is obtained by the same rule, and the learning rate is recorded for every epoch | Retrain for E\* epochs, replaying the tuning learning rate epoch by epoch, without early stopping; the test year is predicted only once, after epoch E\* |

- All three pipelines save test-year predictions for every epoch, for use in E1 only. The reported prediction of fair is fixed by E\* before any scoring.
- Everything else follows its public code: HuberLoss, AdamW, ReduceLROnPlateau, EarlyStopping, batch 64, gradient clipping 1.0, auxiliary loss weight, 100 epochs.

## 3. A1 main quantities and wording rules

- O = own − fair. Decomposition: O_sel = own_clean − fair (due only to monitoring on the test set); O_scale = own − own_clean (due only to transductive standardisation).
- C_fair = GE-BiFormer_fair − REML-GBLUP (gate-2 `B1r_gblup_reml` v2 predictions), computed on the common cells.
- Metrics: within-environment Spearman (primary); pooled Pearson (reported metric, diagnostic); within-environment 10% selection differential (decision metric). Only environments with ≥25 cells are counted. 95% CIs from a two-level bootstrap (environment × seed, B = 2000, `dartgxe.eval.compare.paired_delta`).
- **Wording per group** (each (model, scenario) is judged separately; no pooled test across models):
  - Pooled Pearson: only if the lower CI bound of O > 0 is it written that "the pipeline in the public code raised the reported metric";
  - Within-environment Spearman and selection differential: use `verdict()`, threshold 3/√N. resolved is written as "resolvably changed the ranking"; detectable as "statistically detectable, but below the selection resolution"; tied as "no effect on ranking detected";
  - The summary sentence gives only counts: "in k of the 4 (model, scenario) combinations pooled Pearson was raised (CI > 0); within-environment ranking ...". No p-values.
  - To write for a model that "the pipeline in its public code raises the reported metric", the following is required: the CI of O > 0 in at least one scenario, and point estimates > 0 in both scenarios.
- The wording restriction of gate 2 continues: write only "the pipeline in the public code ...", not "the original results are overestimated".

## 4. Directional predictions (written before any test-year result)

1. The O (pooled Pearson) of GE-BiFormer is > 0 in both scenarios. Basis: own selects epochs on the test year by the pooled loss, which rewards fitting the test-year environment means (the main route found by W1b for GEFormer).
2. On pooled Pearson, |O_scale| < |O_sel|: the inflation comes mainly from monitoring on the test set, not from transductive standardisation.
3. **No directional prediction** is made for O on within-environment Spearman.
- MDE: the design is the same as gate 2 (3 seeds, 21–27 scored environments per scenario). For within-environment Spearman it is about 0.06 per (model, scenario) (gate 2 amendment 1: environment SD 0.072, seed SD 0.027). For O on pooled Pearson, the CI width in gate 2 was about 0.4, i.e., a single group can only resolve an inflation of order 0.2. Stated here as is: power per group is low, so no pooled test is done; only intervals are reported.

## 5. E1 epoch-level test (descriptive, direction fixed in advance)

For every epoch of every run, using the test-year predictions and the W1b definitions, compute: t, t\* = k·v_b/a, a, v_b, k, S, pooled r, r_max, within-environment Spearman, and the environment-mean share φ = (a²/v_b) / (a²/v_b + k²), i.e., the proportion of r_max² that comes from the environment means. For epochs with a ≤ 0, t\* and φ are recorded as not applicable.

Set of runs:
- GE-BiFormer own and own_clean (6 runs each);
- GEFormer rerun own (6 runs);
- The fair retraining runs (GE-BiFormer 6, GEFormer 6) have fewer epochs; they are only described and do not enter the test.

For each own-type run (18 in total), let e_P = the epoch with the highest test-year pooled Pearson, and e_S = the epoch with the highest test-year within-environment Spearman:
- **E1-1**: φ(e_P) > φ(e_S), i.e., the epoch picked by pooled Pearson relies more on the environment means;
- **E1-2**: |log(t/t\*)|(e_P) < |log(t/t\*)|(e_S), i.e., at the epoch picked by pooled Pearson, the within-environment scale is closer to t\*.

Both use a one-sided sign test. Only runs with e_P ≠ e_S are counted, and the number of excluded runs is reported. With all 18 runs usable, ≥13 in the same direction are needed to reach p < 0.05. For GE-BiFormer, its own selection rule is also reported, i.e., the epoch e_L with the lowest test-year Huber loss; its position and φ are reported.

These calculations use the per-epoch test-year predictions only to describe the mechanism. No reported method or epoch is selected on this basis.

## 6. Hard-rule check and computing arrangements

- Leakage prevention: own and own_clean use test-year phenotypes by design; this is the object of study. fair does not touch test-year phenotypes before the single final scoring. All scores are computed by the analysis scripts together, after all runs have finished.
- Scripts:
  - `scripts/a1_gebiformer.py`: one run = (scenario, pipeline, seed); writes predictions and per-epoch predictions;
  - `scripts/a1_geformer_epochs.py`: GEFormer final retraining, saved per epoch;
  - `scripts/a1_analyze.py`: O, decomposition, C_fair, wording verdicts;
  - `scripts/e1_epochs.py`: E1.
- Compute: GE-BiFormer on the 4090 (W1d estimate 7.3 GPU hours; about 10 hours with own_clean); GEFormer rerun on amax GPU1/2.
- Time box: runs completed before 10-12, analysis completed before 10-15. Parts not completed by the deadline are reported as is; there is no extension and no later look.

## A1 results (2026-09-28; 4090, run code 9ea9108, analysis 2b8534e + description fd99655; the GEFormer E1 reruns are still running on amax, E1 reported separately)

Output: `results/a1/{summary.csv, verdict.json, levels.csv, epochs.csv, runs/}`. All 18 runs completed, with no failures. The REML-GBLUP predictions were copied from amax to the 4090; SHA256 checks agree (`results_v2/predictions/g2f/B1r_gblup_reml_from_amax.SHA256SUMS`). Scored environments: F2024m 21 (N = 9,064, 3/√N = 0.032), F2022m 24 (N = 10,283, 0.030). Comparisons on the common cells; test environments missing covariates are not scored.

**Levels** (mean ± SD over 3 seeds; `levels.csv`)

| Scenario | Method | Within-environment Spearman | 10% selection differential | Pooled Pearson |
|---|---|---|---|---|
| F2024m | GE-BiFormer own | 0.167 ± 0.049 | 0.205 | 0.225 |
| F2024m | GE-BiFormer own_clean | 0.222 ± 0.056 | 0.247 | 0.192 |
| F2024m | GE-BiFormer fair | −0.003 ± 0.101 | −0.080 | 0.150 |
| F2024m | REML-GBLUP | 0.228 | 0.191 | 0.150 |
| F2022m | GE-BiFormer own | 0.060 ± 0.131 | −0.045 | 0.533 |
| F2022m | GE-BiFormer own_clean | 0.051 ± 0.081 | −0.119 | 0.530 |
| F2022m | GE-BiFormer fair | 0.040 ± 0.055 | −0.078 | 0.459 |
| F2022m | REML-GBLUP | 0.217 | 0.241 | 0.160 |

**Reported epochs** (`epochs.csv`): the E\* selected by fair on the validation year is 1–6. The epochs selected by own on the test year are mostly epochs 2–5; F2024m seed 1 is epoch 50. own_clean is epochs 2–62. That is, with early stopping on the pooled loss, both kinds of monitoring stop at very early epochs.

**Contrasts** (two-level bootstrap 95% CI; `summary.csv`)

| Scenario | Contrast | Within-environment Spearman | Verdict | Pooled Pearson | Verdict |
|---|---|---|---|---|---|
| F2024m | O = own − fair | +0.170 [+0.068, +0.263] | resolved | +0.074 [−0.099, +0.231] | No detectable change |
| F2024m | O_sel | +0.226 [+0.091, +0.380] | resolved | +0.042 [−0.197, +0.276] | No detectable change |
| F2024m | O_scale | −0.056 [−0.153, +0.057] | tied | +0.033 [−0.140, +0.199] | No detectable change |
| F2024m | C_fair = fair − REML | −0.231 [−0.352, −0.120] | resolved (REML wins) | +0.000 [−0.339, +0.339] | No detectable change |
| F2022m | O | +0.021 [−0.135, +0.144] | tied | +0.075 [−0.081, +0.238] | No detectable change |
| F2022m | O_sel | +0.011 [−0.098, +0.094] | tied | +0.072 [−0.080, +0.222] | No detectable change |
| F2022m | O_scale | +0.010 [−0.049, +0.060] | tied | +0.003 [−0.103, +0.097] | No detectable change |
| F2022m | C_fair | −0.177 [−0.236, −0.125] | resolved (REML wins) | +0.299 [−0.017, +0.543] | No detectable change |

The verdicts for the 10% selection differential agree with those for within-environment Spearman (F2024m O +0.285 [+0.144, +0.414] resolved; C_fair resolved in both scenarios, REML wins).

**By the rules fixed in advance**
- Prediction 1 (O on pooled Pearson > 0 in both scenarios): point estimates +0.074 and +0.075, in the predicted direction, but both CIs contain 0. By the wording rules of §3, **it cannot be written that "the pipeline in GE-BiFormer's public code raised the reported metric"**.
- Prediction 2 (on pooled Pearson, |O_scale| < |O_sel|): point estimates 0.033 < 0.042 and 0.003 < 0.072; it holds, but all intervals are very wide. It is clearer on within-environment Spearman: on F2024m, O_sel +0.226 is resolved and O_scale is tied.
- No directional prediction was made in advance for O on within-environment Spearman. Result: on F2024m, own is resolvably **higher** than fair (+0.170); on F2022m they are tied.
- Summary over the 4 (model, scenario) combinations (the two GEFormer rows are taken from gate 2, not recomputed):
  - Pooled Pearson raised (CI > 0): 1/4 (GEFormer F2024m).
  - own − fair on within-environment Spearman with a CI excluding 0: 2/4, in opposite directions. GEFormer F2024m is −0.084 (own worse); GE-BiFormer F2024m is +0.170 (own better).
  - C_fair (within-environment ranking compared with REML-GBLUP under the fair pipeline): REML wins resolvably in 3/4 (GEFormer F2024m, GE-BiFormer in both scenarios); GEFormer wins in 1/4 (F2022m).

**One-sentence conclusion**: monitoring or selecting on the test year makes the reported numbers deviate from the level a deployable pipeline can reach. The deviation can be 0.08–0.17 (within-environment Spearman), above the selection resolution (about 0.03). The direction of the deviation depends on the selection criterion and the model: GEFormer selects by pooled Pearson, and its within-environment ranking becomes worse; GE-BiFormer selects by the pooled loss, and its within-environment ranking looks better. Under the deployable fair pipeline, the two published deep models have a resolvably worse within-environment ranking than REML-GBLUP in 3 of the 4 combinations. Pooled Pearson, on GE-BiFormer F2022m, ranks it above REML (0.46 vs 0.16; the CI of the difference contains 0), the opposite of the within-environment ranking.

**Mechanistic reading (description, not tested)**: the pooled loss in a new year consists mainly of environment-mean error. Early stopping on it stops at epochs where the genotype signal has not yet been learned; this is exactly what happens on the validation year of the fair pipeline (within-environment Spearman of F2024m fair ≈ 0). Monitoring on the test year amounts to borrowing information from the test year itself to pick the epoch. The per-epoch decomposition of E1 will test this reading.

## Amendment (2026-09-28, before any LSTM-GNN run): LSTM-GNN feasibility pre-study (following the W1d rules)

On 2026-09-28 the user agreed to proceed as suggested ("go with your suggestion"; "OK, continue"): first, a feasibility pre-study of the LSTM-GNN found in A3; if it passes, a separate A1b pre-registration is written.

- **Model**: Morshedian & Domaratzki 2026 PLoS Comput Biol (doi:10.1371/journal.pcbi.1013729), code amma/maize-yield-gxe-gnn @ 3af3c88. **Architecture C** is used: the paper abstract reports that C is better than A and B (pooled PCC 0.6945 vs 0.41, 0.66), and it is the paper's main architecture. This was fixed before any test-year prediction. The repository has no LICENSE file, so its unmodified code is only imported at run time and not redistributed (the same handling as for GEFormer).
- **Inputs**:
  - Markers: the 2,425 markers of data_v2, PCA as in its code, up to 548 dimensions. Deviation: the paper uses 437,214 SNPs, while our G2F v2 genotype matrix has only 2,425 markers. This is written into the deviation statement of A1b.
  - Weather: 5 NASA POWER daily variables from `env_daily`, with column names the same as its `WEATHER_FEATURES`.
  - Trait: yield.
- **Pre-study content**: only training years and the validation year are used. Test-year hybrids and environments do not enter any fitted transformation. The test year is not predicted and not scored.
  1. The environment LSTM encoder is trained only on the training-year environments;
  2. One short training of architecture C on the training years of F2024m and F2022m, monitored on the validation year;
  3. Record the time per epoch and GPU memory, and estimate the total GPU time under the A1 budget: 2 scenarios × 3 seeds × 3 training runs per seed × up to 100 epochs;
  4. Validation-year within-environment Spearman is only described.
- **Criterion**: it runs, and the estimated total GPU time ≤ 72 hours → write the A1b pre-registration; otherwise, it is removed from A1 and only the code-audit facts in A3 are kept. Time box until 2026-10-05.
- **Environment**: `torch_geometric` must be added (a pure Python package; GATv2Conv and global_mean_pool need no compiled extensions). It is written into `requirements.in` and `requirements.lock`.

## LSTM-GNN feasibility results (2026-09-28; 4090 GPU0 RTX 4090 24 GB, torch 2.11.0+cu128, torch_geometric 2.8.0.post1, code c2d3cd7; `results/w1d/lstmgnn_bench*.json`)

**Verdict: LSTM-GNN is removed from A1** (criterion of the section "LSTM-GNN feasibility pre-study": with the configuration of the paper and the code, it cannot run on our GPUs, and even with a smaller batch, the estimated GPU time is far above 72 hours).

- Configuration: architecture C; Table 1 of the paper agrees with the code defaults (batch 32, 30 rounds of message passing, hidden 128, 8 heads, top-k 10), 548 PCs, 21-dimensional environment vector. The graph of each sample has 569 nodes and about 29,000 edges.
- Measurements (training-year data; the test year was not touched):
  - Batch 1: peak memory 5.83 GB, 0.142 s per step; batch 2: 11.48 GB, 0.270 s per step; batch 4 exceeds 24 GB.
  - Memory and time both grow linearly with batch size: about 5.65 GB and about 0.135 s per sample.
  - Linear extrapolation to batch 32 requires about 181 GB, more than the RTX 4090 (24 GB) and the H200 NVL (141 GB).
  - At batch 2, one epoch on the final fitting set (F2024m, 106,721 cells) takes about 4.0 hours, and the full A1 budget (18 runs × up to 100 epochs) is about 6,400 GPU hours.
- The first measurement (`lstmgnn_bench_run1_leaked_oom.json`) is void: after batch 32 ran out of memory, the tensors of the failed step were not released, which affected the subsequent batch 8 and 4 measurements. The measurement was redone from small to large batch sizes, with memory freed between sizes; the conclusion was the same (batch 4 still ran out of memory).
- Facts that can be written into the A3 audit (only the measurements are stated, with the software environment noted): in the environment above, the configuration of Table 1 of the paper needs about 5.65 GB of GPU memory per sample, and about 181 GB when extrapolated to batch 32. The paper does not state which GPU was used. **No inference about the paper's results is drawn from this**.
- One further item to be checked (no conclusion drawn): Table 3 of the paper lists the PCC of architecture C (0.6945, computed by its test script pooled over all test rows) alongside the PCCs of competition teams, while the paper says the competition leaderboard was based on the mean RMSE and PCC over "different environments". Whether the two PCCs have the same definition must be verified against the competition paper (Washburn et al. 2025 Genetics) before it can be written.

A1 keeps the two models GEFormer + GE-BiFormer (4 combinations); the wording rules are unchanged.

## E1 results (2026-09-28; amax CPU; per-epoch predictions: GE-BiFormer 18 runs (4090, transferred to amax via the local machine, SHA256 checks agree) + GEFormer 12 reruns (amax H200, all completed, no failures); `results/e1/{epochs.parquet, runs.csv, verdict.json, geformer_rerun_vs_gate2.csv}`)

**By the direction fixed in advance** (18 own-type runs: GE-BiFormer own and own_clean, 6 each; GEFormer own, 6; runs with e_P ≠ e_S: 18/18)

| Test | Runs in the same direction | One-sided sign test | Conclusion |
|---|---|---|---|
| E1-1: φ(e_P) > φ(e_S), i.e., the epoch picked by pooled Pearson relies more on the environment means | **18/18** | p = 3.8 × 10⁻⁶ | Supported |
| E1-2: \|log(t/t\*)\|(e_P) < \|log(t/t\*)\|(e_S), i.e., the within-environment scale is closer to t\* | **13/16** (2 with a ≤ 0, not applicable) | p = 0.011 | Supported |

- Cost (description): at the epochs picked by pooled Pearson, within-environment Spearman is lower than at the epochs picked by within-environment Spearman; for example, GEFormer F2024m own: 0.045–0.114 vs 0.146–0.213.
- GE-BiFormer's own selection rule (the epoch e_L with the lowest test-year Huber loss) mostly falls in the first few epochs; φ(e_L) is 0.46–1.00.
- Reading: both routes are supported at the per-epoch level:
  - pooled metrics reward fitting the environment means (main route, 18/18);
  - pooled metrics reward pushing the within-environment scale towards t\* (secondary route, 13/16).
  - This agrees with the W1b conclusion at the level of final predictions; there is now per-epoch evidence across two models.

**Agreement of the GEFormer reruns with gate 2 (reporting required in advance)**
- Own pipeline, 6 runs: predictions **identical** to gate 2 (correlation 1.0, maximum difference 0.0), and the reported epochs are also the same.
- Fair pipeline, 6 runs: **not identical** (correlation 0.63–0.88); the reported epochs are the same (F2024m epoch 6, F2022m epoch 15).
- The cause has been identified: in the GEFormer public code (Deep-Breeding/GEFormer @ c99448a), `tools/TimeFeature_Block.py` uses `ProbAttention` from `tools/attn.py`, whose `_prob_QK` (line 38) calls `torch.randint` in **every forward pass** to sample keys at random, in both training and inference. In the rerun, an extra test-year prediction is made at every epoch. This advances the torch random-number state, so the training trajectory changes. The gate-2 fair retraining did not predict at every epoch, so the two do not match. The own pipeline in gate 2 already predicted at every epoch, so it matches.
- Implications:
  - The gate-2 numbers remain authoritative for the fair pipeline. E1 tests use only own-type runs and are not affected;
  - **A code fact that can be written into the A3 audit**: GEFormer inference is random; repeated predictions by the same model differ unless the random seed is reset. Any extra forward computation changes the subsequent training.
  - An untested inference: the own pipeline takes the maximum test-set pooled Pearson over 100 epochs, and the evaluation at each epoch itself carries sampling noise; taking the maximum would further amplify the optimistic bias. **Not tested; not written as a conclusion.**

## Correction (2026-09-30, WP4): the t and k of E1 used an approximate decomposition

E1 (and W1b) took t as the weighted mean of the within-environment prediction SDs t_j, and k as the weighted mean of the covariances of standardised scores. This agrees exactly with the proposition only when t_j is the same in all environments (the observed median coefficient of variation of t_j is 0.28). Recomputed in the exact form (`scripts/wp4_e1_exact.py`, `results/wp4/`; derivation in `docs/wp4_tstar_proposition_2026-09-30.md`): E1-1 is unchanged (18/18, p = 3.8e-6); **E1-2 changes from 13/16 (p = 0.011) to 12/16 (p = 0.038)**. The direction and the verdict "supported" are unchanged; the evidence is weaker. The original verdict is kept in the record; the paper uses the exact-form numbers.
