English translation of the pre-registration document prereg_sequel_wave2c_2026-09-28.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Sequel pre-registration, wave 2 C (B3b): after adding cell-level models and G×E learners, can decision combination beat a strong baseline? (2026-09-28)

Written before any new panel or rule in this wave was run. Not changed after commit; amendments are only appended as dated sections.

## 0. Basis and user decisions

- Pre-study 2 (`docs/prestudy_cell_level_2026-09-28.md`) Q2: cell-level REML is +0.020 [+0.006, +0.035] above two-stage REML. By the advance rule, the B3 method library must be rerun with cell-level models added; the existing B3 conclusions hold only within the two-stage library.
- Both "strong algorithm" pre-studies were judged NO-GO (year-by-year calibrated shrinkage: two-stage +0.0022, cell-level +0.0039). What remains is angle 2: adding learners that can handle G×E to the decision combination.
- On 2026-09-28 the user agreed to combine the two items into one pre-registration ("OK, continue").
- Known information, declared as is:
  - B3 (two-stage library): R_STK_FW − R_FW = +0.0155 (better), but −0.0015 on new genotypes (tied); R_STK_FW − two-stage REML = +0.009 (tied).
  - Pre-study 2: the per-environment values of cell-level REML on the 31 target years have already been computed. The G×E learners have never been fitted.

## 1. Data, target years, environmental covariates

- Target years, scored environments and historical years are the same as in wave 2b (31 target years).
- Environmental covariates (EC):
  - G2F: official APSIM EC of data_v2 (`env_ec.parquet`, 654 columns; environments flagged `ec_missing` are treated as having no EC);
  - URSN: `enviromics/envt_covariates_stages_NASA_URSN.tsv` (matched by `env_code`; soil plus weather by growth stage, 165 columns), uploaded to the 4090 as a read-only copy with SHA256 recorded;
  - **NUST: no EC in this wave**. Locally there are only location names, no coordinates. Getting NASA POWER weather would first require converting place names to coordinates, which is not done in this wave. The NUST method library adds only the cell-level model.
- **Timing of EC (the project protocol document §2.1)**:
  - At deployment, the observed weather of the target year is not available. So **the main analysis uses "historical-mean EC"**: the EC of a target environment = the mean of that location's EC over the years before Y; if the location has never appeared, the mean over all environments before Y is used. The location is the part of `env` before "_" in G2F, and `loc_code` in URSN.
  - **The observed-EC version is used only as an upper-bound sensitivity analysis**.
  - Training environments (before Y) use their observed EC. These are known after the fact, so this is not leakage.
  - EC standardisation and PCA are fitted only on training environments. Environments without EC are imputed with the training mean, i.e., the G×E term is constant and the prediction reduces to the genotype main effect.

## 2. Method library

In addition to the 10 two-stage methods of wave 2b, the following are added:
- `cell_reml`: cell-level marker ridge regression, environment fixed effects, REML λ (the same as in pre-study 2). **In all three datasets.**
- `rn_ridge` (G2F and URSN only): cell-level reaction-norm ridge regression. Features = [80 genotype PCs, 100 interaction terms of the first 20 genotype PCs ⊗ the first 5 EC PCs], all standardised on the training cells; environment fixed effects; a single REML λ.
- `gxe_gbm` (G2F and URSN only): LightGBM, features = [first 20 genotype PCs, first 5 EC PCs], target = yield or resistance centred within environment. Fixed parameters: num_leaves 31, 300 iterations, learning rate 0.05, min_data_in_leaf 50, seed 0.

Genotype PCs and EC PCs are fitted only on the training set. Therefore:
- The main method library for G2F and URSN has 13 methods (10 + `cell_reml` + `rn_ridge` + `gxe_gbm`; the G×E learners use historical-mean EC). The sensitivity library replaces the two G×E learners with their observed-EC versions.
- NUST has 11 methods.

In the CV history, the environments within the training years are all observed environments, and the G×E learners use their observed EC (equivalent to CV1). This is exactly one source of "random CV differs from deployment", and it is kept as is.

## 3. Rules

As in wave 2b (R_CV, R_FW, R_EQ, R_STK_CV, R_STK_FW, Oracle), but computed on each dataset's own method library. **The default baseline is changed to R_CELL**, i.e., always use `cell_reml`.

## 4. Confirmatory comparisons (2; Holm; two-sided; primary metric within-environment Spearman)

- **H3**: R_STK_FW − R_CELL, all scored genotypes, random-effects pooling over the 31 target years. Answers "does the decision combination fitted on the forward results of past years beat the strong baseline?"
- **H4**: the same contrast, scoring only new genotypes (W1c definition, at least 10 per environment). Answers "does this advantage extend to genotypes that never appeared before?" (B3 was tied on new genotypes).
- Verdict as in wave 2b: "better" is written only if the CI at the corresponding Holm level excludes 0 and |Δ| ≥ 3/√N. Zero-variance years are handled by the 2026-09-28 correction (SE floor = median of the non-zero SEs within the same dataset, contrast and metric).
- **Directional predictions (fixed in advance)**:
  - H3 **tied**. Basis: in B3, R_STK_FW of the two-stage library was only 0.009 above two-stage REML, while cell-level REML is 0.020 above two-stage REML. Once `cell_reml` is in the library, the combination can use it; G×E learners with historical-mean EC carry limited information. The net difference is expected to be within ±0.01.
  - H4 **tied**.
- MDE as in wave 2b (about 0.030 pooled over 31 years; for the 16 years of G2F + URSN only, about 0.04–0.05, computed separately).

## 5. Descriptive results

- R_FW − R_CELL, R_EQ − R_CELL, R_CV − R_CELL;
- Value of the G×E learners: R_STK_FW (this library) − R_STK_FW (B3 two-stage library), and the same contrast on G2F + URSN only;
- H3 with the observed-EC sensitivity library, i.e., the upper bound if the weather of the year were known;
- Levels of each method in the target years, including `rn_ridge` and `gxe_gbm` vs `cell_reml`: are G×E learners useful on their own?
- Selection-differential metric; results by dataset; the share of stacking weight given to the G×E learners.

## 6. Hard-rule check and implementation

- Leakage prevention: as in wave 2b, plus: the historical-mean EC of a target environment uses only years before Y; EC standardisation and PCA are fitted only on training environments. New unit tests assert these two points.
- Code:
  - `src/dartgxe/forward/cellgxe.py`: cell-level models and G×E learners;
  - `src/dartgxe/forward/data.py`: EC reading added;
  - `scripts/b3b_panels.py`: generates only the forward and CV-history predictions of the new methods, for both EC definitions;
  - `scripts/b3b_rules.py`: merges the new and old panels, computes rules and pooling; refuses to run when panels are incomplete.
- Compute: 4090 (cell-level Ridge runs on the GPU in float64); NUST and URSN data from read-only copies.
- Time box: completed before 2026-10-12.

## Results (2026-09-28; 4090; panel code 5a series to 6167413; `results/b3b/{pooled.json, year_effects.csv, choices_weights.csv, verdict.json, rules_per_env.parquet, meta/}`)

Panels: 56 forward years, 31 CV histories, 1.7 million rows of predictions for the new methods, no missing values. Cell-level REML never fell on a grid endpoint. Scored cells in the target years: all N = 87,294; new genotypes N = 68,781.

**Confirmatory comparisons (Holm; within-environment Spearman)**

| Contrast | Estimate | CI | Verdict | Advance prediction |
|---|---|---|---|---|
| H4: R_STK_FW − R_CELL (new genotypes; smaller p, tested first) | +0.0072 | [−0.0102, +0.0246] (97.5%) | Tied | Tied ✔ |
| H3: R_STK_FW − R_CELL (all genotypes) | +0.0028 | [−0.0079, +0.0135] (95%) | Holm stopped, not tested; the interval itself also contains 0 | Tied ✔ |

**By dataset and sensitivity (descriptive)**
- H3 by dataset: G2F +0.037 [+0.006, +0.068] (5 years), NUST −0.001, URSN −0.017. H4 in G2F: +0.040 [+0.001, +0.080]. The advantage of the combination appears only in G2F, and G2F has only 5 years.
- **Selection-differential metric**: H3 +0.031 [+0.010, +0.052], H4 +0.040 [+0.001, +0.080]; the CIs exclude 0. This is a descriptive metric in the pre-registration, and no confirmatory conclusion is drawn. It differs from the conclusion on the primary metric Spearman; the sequel must report the two side by side, as is.
- Each rule vs R_CELL: R_FW +0.002, R_CV −0.010, R_EQ −0.006, R_STK_CV −0.003; all tied. Oracle vs R_CELL is +0.020 [+0.007, +0.032], i.e., in this method library, even the post-hoc best single method is only 0.02 above cell-level REML.
- The top method picked by the forward history is `cell_reml` in 24 of the 31 years. In the stacking weights, `cell_reml` has 35% on average, and the G×E learners have 22% on G2F + URSN.
- Value of the new learners: R_STK_FW of this library − R_STK_FW of the B3 two-stage library = +0.016 [+0.008, +0.023], mainly from adding `cell_reml` (looking only at G2F + URSN, which have G×E learners: +0.008, tied).
- **G×E learners used on their own are worse than `cell_reml`** (G2F + URSN, 16 years):
  - `rn_ridge` −0.020 [−0.044, +0.005]; `gxe_gbm` −0.037 [−0.073, −0.001];
  - Using the observed EC of the year is not better: `rn_ridge_real` −0.028 [−0.056, −0.001], `gxe_gbm_real` −0.044 [−0.082, −0.006];
  - H3 with the observed-EC library is +0.002, tied.
- Cell-level REML − two-stage REML = +0.020 [+0.006, +0.035], consistent with pre-study 2 (the same data, as a re-check).

**One-sentence conclusion**: across 3 crops and 31 forward years, cell-level REML-GBLUP is a strong baseline that is hard to beat. Picking a single model by the forward results of past years or by CV, decision combination, and adding G×E learners that use environmental covariates (even when given the observed weather of the year) all give no resolvable improvement on the primary metric. The combination shows an advantage only in G2F and on the selection-differential metric; this must be reported as a descriptive result.

Correction (2026-09-28, same day): "panel code 5a series to ..." at the start of the results section is imprecise. It should read: panel code a6bf3e4, rule analysis code 6167413.
