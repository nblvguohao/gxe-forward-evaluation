English translation of the pre-registration document prereg_headroom_sparse2_2026-09-29.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Headroom-check pre-registration SP2: three methods added on top of M×E GBLUP in sparse testing (2026-09-29)

Written before any implementation and before any fitting. Not changed after commit; changes are only appended as dated sections. Basis: the user's request of 2026-09-29 to write into a pre-registration, and run, "a decision ensemble with M×E as its backbone, a deep model that learns the M×E residuals, and M×E with environmental covariates".

**Nature**: a descriptive headroom check, not a confirmatory test. GO only means "worth writing a confirmatory pre-registration, and preparing data that did not take part in this check".

## 0. Known information, stated as is

- The SP headroom check (`docs/prereg_headroom_sparse_2026-09-29.md`, result commits 36545a2, d8fa274) has been seen: at 25% observed, M1 (M×E) − M0 (main effect) = +0.0187 [+0.0136, +0.0238]; M2 (learned environment correlation) − M1 = +0.0001, NO-GO. The M×E gain is largest on G2F and NUST; most URSN units do not select the G×E term.
- [reference to a separate unpublished study omitted]
- In the forward scenario, decision ensembles, deep models and covariate-based G×E learners did not exceed main-effect REML (already seen).
- **The three methods of this wave have never been run in sparse testing**. Masking, target years and scored environments are exactly the same as in SP (the same `sparse.sparse_mask`, the same f and seeds), so the scored cells are the same as in SP, and **the M0, M1 and M2 results on these cells have already been seen**.

## 1. Scenario, data, masking

Exactly the same as SP: 6 datasets, 48 target years, f ∈ {0.25 (main), 0.5 (secondary)}, seeds {0, 1}; observed cells = training, the rest = scoring; the scored-environment threshold is the same as in SP.

## 2. Internal cross-validation (shared by the three methods, training data only)

- The observed cells of the target year are stratified by environment and randomly split into 5 folds (random numbers determined by (dataset, Y, f, seed, "cv")).
- For each fold: fit with "history + observed cells of the other 4 folds", and predict the cells of that fold. This gives **out-of-fold (OOF) predictions** for all observed cells.
- The REML selection (ω, c²) is redone in each fold, consistent with the full fit. Lines that have no other observed cells in a fold are predicted from markers (the kernel).
- Full fit: with "history + all observed cells", predict the scored cells (M0 and M1 are the same as in SP; agreement is checked in §5).

## 3. Three methods (all with M1 = M×E GBLUP as the control)

- **C1 decision ensemble `sp_stk`**:
  - Library: M0, M1, `gbm_env`. `gbm_env` = LightGBM; features are the first 20 marker PCs on training lines and a one-hot encoding of target-year environments (for G2F and URSN, 5 environmental-covariate PCs are added); the target is the within-environment-centred y; it is **trained only on target-year observed cells**; fixed parameters: num_leaves 31, 300 rounds, learning rate 0.05, min_child_samples 20, seed 0.
  - Weights: non-negative weights are fitted on the OOF predictions following the definition of atlas `stack_weights` (within each environment, standardise predictions and y; c = mean over environments of Z′y/n; S = mean over environments of Z′Z/n; minimise w′(S + 0.05I)w − 2c′w, w ≥ 0; equal weights if all are 0).
  - Prediction: on scored cells, the full-fit prediction of each method is standardised within each environment, and the predictions are summed with the weights.
- **C2 neural network that learns the M×E residuals, `sp_dlres`**:
  - Target: the M1 OOF residuals on observed cells, r = y_c − (ŷ − within-environment mean), where y_c is the within-environment-centred y and ŷ is the M1 OOF prediction.
  - Features: the first 80 marker PCs on training lines and a one-hot encoding of target-year environments (for G2F and URSN, 5 environmental-covariate PCs are added).
  - Model: sklearn `MLPRegressor`, hidden layers (64, 32), alpha = 1e-3, early_stopping (validation_fraction 0.2, n_iter_no_change 20), max_iter 300; average over seeds 0, 1 and 2.
  - Prediction: on scored cells = M1 full-fit prediction + network prediction.
- **C3 M×E with environmental covariates, `sp_ecmxe` (G2F and URSN only)**:
  - G×E covariance K_A^(k) ⊗ E_ρ, E_ρ = (1 − ρ)I + ρΩ; Ω is the linear kernel of the environmental covariates of the target-year environments (5 PCs from `cellgxe.ec_pcs`, mode 'real'; standardisation and PCA are fitted on training environments only; environments without covariates get a zero vector), scaled to a mean diagonal of 1.
  - c² is fixed at the M1 REML value; ρ ∈ {0.25, 0.5, 0.75, 1.0} is selected by profile REML (the same mechanism as M2 in SP; only R̂ is replaced by Ω).
  - In sparse testing, the target-year environments already have partial harvests, and the measured weather of the season is known, so measured covariates are used.

## 4. Verdict (f = 0.25, within-environment Spearman, each method versus M1)

- The three comparisons are Holm-corrected (p values are obtained from the normal approximation of the pooled estimate; in order of increasing p, 98.33%, 97.5% and 95% intervals are used).
- **GO** (per method): lower CI bound at the corresponding Holm level > 0, pooled Δ ≥ 3/√N, and the pooled point estimates after leaving out each dataset all > 0.
- Scope: C1 and C2, all datasets (46 target years with scored environments); C3, G2F + URSN, with N and the resolution computed only from the scored cells of these two datasets; leave-one-dataset-out then has only 2 cases.
- Otherwise NO-GO. f = 0.5, within-environment Pearson and the top-10% selection differential are descriptive only. After NO-GO, no new variants are run, and the grids and parameters are not changed.

## 5. Consistency checks and unit tests (before any comparison)

- `tests/test_sparse2.py`:
  - folds of the internal cross-validation: reproducible, stratified by environment, containing only observed cells;
  - leakage prevention: when the phenotypes of scored cells are replaced by garbage values, the C1–C3 predictions, ensemble weights and REML choices are bitwise identical (MLP under fixed seeds);
  - C3 equals M1 as ρ → 0; on toy data, the ensemble weights agree with a direct NNLS solution.
- Real data: the full-fit M0 and M1 predictions of each unit agree cell by cell with the SP panels (relative difference < 1e-8). If they do not agree, check the implementation first, without looking at any comparison.

## 6. Directional predictions (written in advance)

- C1: a small positive value or a tie, 0 to +0.01; subjective probability of GO 0.2. Basis: the weights are learned on same-year data, which is more favourable than in the forward scenario; but M1 in the library is already strong, and `gbm_env` uses only observed cells, so it has little data.
- C2: tie, −0.005 to +0.005; GO probability 0.1.
- C3: a small positive value on G2F; GO probability 0.15 (only about 14 target years, low power).

## 7. Implementation, compute and time

- Only added: `src/dartgxe/forward/sparse2.py` (reuses functions of `sparse` and `cellspec`, **without modifying them**), `scripts/headroom_sparse2.py`, `tests/test_sparse2.py`.
- Compute: 4090; 6 fits per unit (5 folds + full fit); estimated total 3–5 hours (two GPUs in parallel).
- Output: `results/headroom_sparse2/`; record the git commit, config, GPU, versions, start and end times.

## 8. Use of results

- Any GO: write a confirmatory pre-registration for that method; confirmation must use data that did not take part in this check (candidates as in the SP pre-registration, §8).
- All NO-GO: recorded as "in sparse testing, decision ensembles, residual neural networks and covariate enhancement on top of M×E GBLUP brought no distinguishable gain", and written into the evaluation paper together with the SP results.

## Results (2026-09-30; 4090; code f906757; `results/headroom_sparse2/`; panel files remain on the 4090 in `scratch_ideas_2026-09-29/headroom_sparse2/panels/`)

**Process**: all 5 unit tests passed (reproducibility and stratification of folds, cell-by-cell agreement with the M0/M1 of SP, leakage prevention, C3 equals M1 under the identity kernel, ensemble weights agree with the direct solution). All 192 panels were generated, with no failures, from 2026-09-29 20:42 to 2026-09-30 00:11 (about 3.5 hours, two RTX 4090s). **Consistency check passed**: the full-fit M0 and M1 of each unit agree cell by cell with the SP panels; maximum relative difference 8.5×10⁻¹⁴. Scoring was run once. Scored environments and resolution are the same as in SP: at f = 0.25, 3/√N = 0.0103 (C3, G2F + URSN only: 0.0150).

**Verdict (f = 0.25, within-environment Spearman, versus M×E GBLUP, Holm-corrected): all three are NO-GO.**

| Method | Pooled Δ | CI at Holm level | p | Resolution | Leave-one-dataset-out | Verdict |
|---|---|---|---|---|---|---|
| C1 decision ensemble `stk` | **−0.0028** | [−0.0061, +0.0006] (98.33%) | 0.049 | 0.0103 | All < 0 (−0.0013 to −0.0043) | NO-GO |
| C2 residual-learning neural network `dlres` | **−0.0014** | [−0.0047, +0.0018] (97.5%) | 0.32 | 0.0103 | −0.0051 to +0.0004 | NO-GO |
| C3 M×E with environmental covariates `ecmxe` (G2F + URSN) | **−0.0012** | [−0.0040, +0.0017] (95%) | 0.41 | 0.0150 | −0.0035, −0.0001 | NO-GO |

**Descriptive**
- **C1 ensemble**: the 46-year point estimate is negative, and the uncorrected 95% CI is [−0.0055, −0.0000]; compared with M×E it is slightly worse, not better. By dataset: G2F +0.0037, NUST −0.0059 [−0.0109, −0.0010], ESWYT −0.0046, GEM_IA −0.0019, MU_SOY −0.0258, URSN −0.0005. f = 0.5: +0.0019 [−0.0013, +0.0052]. Mean weights: M×E 0.215, main effect 0.099, `gbm_env` 0.078 (on within-environment standardised predictions; the weights do not sum to 1); for URSN the main-effect weight is 0.264, while in the other datasets the weight is concentrated on M×E. Top-10% selection differential +0.0011 [−0.0086, +0.0108]; within-environment Pearson −0.0003.
- **C2 residual network**: a tie overall, but inconsistent across datasets: **ESWYT −0.0465 [−0.0680, −0.0249] (significantly worse)**, GEM_IA +0.0034 [+0.0003, +0.0065], the others close to 0. f = 0.5: −0.0006 [−0.0028, +0.0015].
- **C3 environmental covariates**: G2F −0.0001 [−0.0026, +0.0025], URSN −0.0035 [−0.0096, +0.0026]. Of the 64 G2F + URSN units, 30 selected the G×E term (26 with ρ = 0.25, 4 with ρ = 1.0); the other 34 (mostly URSN) had no G×E term, in which case C3 equals M1. f = 0.5: +0.0001 [−0.0019, +0.0022] (7 years).
- **`gbm_env` used alone** is much worse than M×E: −0.0655 [−0.0812, −0.0499] (selection differential −0.236, within-environment Pearson −0.160), NUST −0.133. This shows that a tree model trained only on target-year observed cells has far less information than M×E with history and a marker kernel, as expected.

**Comparison with the predictions in §6**: C1 was predicted at 0 to +0.01: the point estimate −0.0028 is below the prediction; C2 was predicted to tie: this held (but ESWYT being significantly worse was not anticipated); C3 was predicted to be a small positive value on G2F: −0.0001, did not hold. None of the three GO probabilities (0.2, 0.1, 0.15) materialised.

**Reading (per §8)**
1. **In sparse testing, decision ensembles, residual neural networks and environmental-covariate enhancement on top of M×E GBLUP brought no distinguishable gain**; the ensemble was even slightly worse. Together with the NO-GO for learned environment correlations in SP (+0.0001): so far, in this scenario, none of the improvements added on top of M×E GBLUP has exceeded it.
2. These conclusions are limited to the implementations and parameters used here. The parameters were all fixed in advance and not tuned. They cannot be generalised to "any ensemble or neural network".
3. Limitations: the out-of-fold predictions of the internal 5-fold cross-validation use only target-year observed cells. At 25% observed, each environment has very few observed cells, so the training data for the ensemble weights and for the residual network are very limited. This may be part of the reason for their poor performance, but this wave did not test it, and the settings are not changed on that basis.
