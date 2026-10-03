English translation of the pre-registration document prereg_headroom_oldlines_2026-09-30.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

# Headroom check pre-registration OL: a relaxed forward scenario (old lines × new year) (2026-09-30)

Written before any implementation and before any fitting. Not changed after commit; changes are only appended as dated sections. Basis: the user's request on 2026-09-30, "try relaxing the forward scenario, write a pre-registration".

**Nature**: a descriptive headroom check, not a confirmatory test. GO only means "worth writing a confirmatory pre-registration, and preparing data that did not take part in this check".

## 0. Motivation and known information, stated as is

- In the strictest forward scenario (new lines × new year), the target year has no phenotypes at all. The only thing that can be borrowed is the main effect inferred from markers. Cell-level REML-GBLUP is hard to beat.
- In sparse testing (some phenotypes already exist within the target year), M×E GBLUP is about +0.019 to +0.035 above the main effect. But no improvement on top of M×E had any effect (SP, SP2).
- **Intermediate scenario: old lines × new year.** The target year has no phenotypes at all. But the target lines were tested in earlier years, so their own historical phenotypes are available. In breeding this corresponds to "should a line already in trials be advanced/retested next year". The new information is: **the line's own history (especially its history at the same location)**.
- **Results already seen that overlap with this scenario**:
  - The yield panel of I-B (`results/headroom_ia_ib`) was scored over "all genotypes", which include old lines. Results for old lines alone have not been computed and have not been seen. All genotypes +0.0120, new genotypes +0.0101. The two are close, which suggests the I-B effect on old lines will not be large either, but there is no direct evidence.
  - `ia_g` of I-A (genotype residual effect, ω chosen by REML) is +0.0015 on all genotypes; results on old lines have not been seen.
- **The models of this check have never been run**: OL0–OL4 have never been fitted or scored on the old-line subset.

## 1. Scenario and data

- Datasets and target years: the 6 datasets of the benchmark package, 48 forward years, and the scored environments of each target year (`results/benchmark/splits.json`).
- Training data for target year Y: all years before Y, all cells with markers (same as `cell_reml`). Phenotypes of Y do not enter training.
- **Old lines**: lines that have markers and have at least 1 cell with a GY (or that trait) record before Y.
- **Scored cells**: cells of **old lines** in the scored environments of Y. New lines are not scored.
- **Eligibility of a target year** (counted from the phenotype structure before any fitting; see §6): at least 5 scored environments, each with ≥ the dataset threshold (25; URSN 10) of old lines. Ineligible target years do not enter this check. The number of eligible years is reported as is.

## 2. Models (cell level, environment fixed effects, solved at genotype level, training cells only)

Notation as in `docs/prereg_headroom_ia_ib_2026-09-29.md` §2: K_A = U Λ U′, P = U_k Λ_k^(1/2) (k = min(100, n_g − 1)).

- **OL0 = `cell_reml`**: as before (`cellspec.fit_ia` with ω = 0, s = 1).
- **OL1 main effect + the line's own history (baseline)**: `cellspec.fit_ia`, with ω chosen on `W_GRID` by profile REML (i.e. `ia_g` of I-A). The prediction for an old line = the û of the training line (including the residual effect, i.e. using its own history).
- **OL2 = OL1 + marker G×location (old-line version of I-B)**: the main-effect block is the basis for the ω chosen by OL1 (`sparse.main_block`). The G×L block is the same as in I-B: k = 100, active-location rule (≥ 2 years, ≥ 50 cells), location mapping the same as in the I-B pre-registration §2.2. c² is chosen on `C2_GRID` by profile REML (c² = 0 is OL1). Old lines are training lines, so the G×L part uses their P_k rows directly. Only G2F, NUST, URSN and ESWYT have locations.
- **OL3 = OL1 + the line × location own history (new)**:
  - Fit OL1 and obtain the within-environment centred residuals of the training cells, r_ij = y_c,ij − (within-environment centred fitted value);
  - For each (line i, location l) pair, take the mean residual r̄_il of its training cells and the number of cells n_il (n_il is the number of years the line was tested at that location);
  - Variance components: a one-way random-effects model r_ij = v_{i,l} + e_ij, v ~ N(0, σ²_v), e ~ N(0, σ²_e), fitted by REML on all pairs with n_il ≥ 1 (unbalanced data, EM iterations, starting values σ²_v = σ²_e = var(r)/2, at most 200 iterations);
  - Shrinkage: v̂_il = n_il σ²_v /(n_il σ²_v + σ²_e) · r̄_il;
  - Prediction: OL1 + v̂_{i,l(j)} (0 when the line has no history at that location).
  - This is a two-step approximation (the residuals use in-sample fitted values, which slightly underestimates the residual variance). It is not estimated jointly with the main effect. It is limited to datasets with locations.
- **OL4 = OL2 + the line × location own history**: apply the OL3 steps to the residuals of OL2.

## 3. Metrics and pooling

- Primary metric: the within-environment Spearman of each scored environment on the **old-line scored cells**. Descriptive: within-environment Pearson, top-10% selection differential.
- Environment-level Δ; year effect = the mean Δ across scored environments; SE: environment bootstrap (B = 2000, seed 20260930); pooling: DerSimonian–Laird, with the same zero-SE floor as before (`dartgxe.forward.pool`); scopes: eligible target years, the original 31 years, the independent 17 years, per dataset; leave-one-dataset-out.
- Resolution: 3/√N, N = number of old-line scored cells in the eligible target years.

## 4. Decision

- **Two hypotheses tested before any confirmation (Holm correction, two-sided)**, both against OL1, Spearman, only on datasets with locations (eligible years in G2F, NUST, URSN, ESWYT):
  - **H1**: OL2 − OL1 (effect of marker G×location on old lines);
  - **H2**: OL3 − OL1 (effect of the line × location own history).
- **GO** (per hypothesis): the lower CI bound at the corresponding Holm level > 0 (97.5% at the first step, 95% at the second step), pooled Δ ≥ 3/√N, and all leave-one-dataset-out pooled point estimates > 0 (if only 2 datasets are eligible, both single-dataset point estimates must be > 0). Otherwise NO-GO.
- **Descriptive comparisons that must be reported**: OL1 − OL0 (value of the line's own history, expected to be clearly positive); OL2 − OL0; OL4 − OL1; OL4 − OL0; within-environment Pearson and selection-differential versions; per-dataset results. They do not change the decision.
- After a NO-GO, no new variants are run, and grids and parameters are not changed.

## 5. Consistency checks and unit tests (before any comparison)

- `tests/test_oldlines.py`:
  - Old-line definition: includes only lines with phenotypes before Y and with markers; scored cells contain no new lines; all training years < Y;
  - Leakage: after replacing the phenotypes of Y with junk values, the predictions and REML choices of OL0–OL4 are bitwise identical;
  - Nesting: OL2 equals OL1 when c² = 0; OL3 equals OL1 when σ²_v → 0; OL1 equals OL0 when ω = 0;
  - The EM variance components of OL3 agree with the closed-form ANOVA estimates on simulated balanced data (relative error < 1e-4);
  - Consistency of OL2 with the explicit-feature dense solution (following the approach of `test_cellspec.py`).
- Real data: the Spearman correlation between OL0 and the `cell_reml` predictions in the benchmark package, within each scored environment, has median ≥ 0.999 and minimum ≥ 0.99.
- Run `pytest tests/test_splits.py` as required by the project protocol document §2.1.

## 6. Execution order (fixed)

1. Commit this pre-registration.
2. Write `scripts/count_oldlines.py` (counts the sample structure only, fits no model): for each dataset and target year, the proportion of old lines, the number of old lines per environment, eligibility, the number of scored cells and the resolution. The results are committed as an appended section of this file; only then is the fitting code written.
3. **Stopping condition**: if fewer than 15 "dataset × target year" units are eligible, or fewer than 10 years are eligible in the datasets with locations, write "insufficient power" and do not continue fitting.
4. Implement `src/dartgxe/forward/oldlines.py` (reusing functions in `cellspec` and `sparse`, **without modifying them**), `scripts/headroom_oldlines.py` and `tests/test_oldlines.py`. After the unit tests pass, run on the 4090. Score once, after all panels are complete.
5. Append the results as a "Results" section of this file.

## 7. Direction predictions and power (fixed in advance)

- **Structural expectation (a judgement before counting; it may be overturned by the counts)**: datasets with a high proportion of new lines (ESWYT about 0.9, URSN about 0.7, G2F about 0.9) have few old lines per environment and may not reach the threshold. NUST (proportion of new lines about 0.6) is the most likely to be eligible. So the eligible datasets will probably be only NUST and G2F, and leave-one-dataset-out robustness will be weak. This is stated in advance; the counts take precedence.
- OL1 − OL0: clearly positive, expected +0.02 to +0.08.
- H1 (OL2 − OL1): small positive value, 0 to +0.02; GO probability 0.2. H2 (OL3 − OL1): 0 to +0.03, most likely positive on NUST; GO probability 0.25.
- Power: by reference to I-B and SP, the pooled SE is of the order of 0.003–0.005. The resolution depends on the number of old-line scored cells and may be above 0.01 (if N < 9,000). If 3/√N > 0.02, GO is almost impossible, and this must be stated explicitly in the counting appendix.

## 8. Use of the results (fixed in advance)

- **GO**: write a confirmatory pre-registration; confirmation must use data not used in this check (candidates: see SP pre-registration §8; whether the proportion of old lines in the `ARS_MRASeq` years of SRPN/NRPN is sufficient must first be counted, and permission obtained).
- **NO-GO**: record as "in old lines × new year, marker G×location and the line × location own history brought no resolvable gain". OL1 − OL0 is reported as a descriptive finding.
- **Use for the evaluation paper**: whatever the result, OL is reported as the third scenario of the scenario spectrum (new lines × new year → old lines × new year → sparse testing), together with the two existing scenarios, to show "how the gain of G×E models changes as more information becomes available".

## Appendix (2026-09-30, structural counts completed, before any model fitting)

Code: `scripts/count_oldlines.py` (run locally; uses only the cell table of the benchmark package to decide whether a cell exists; fits no model). Output `results/count_oldlines/{per_year.csv, summary.json}`.

**Eligible units (each target year has ≥ 5 scored environments, each with ≥ threshold old lines): 19**
- G2F 5 years (2016, 2018, 2020, 2022, 2024 all eligible; old lines are 4–17% of the lines in that year; the median number of old lines per eligible environment is 31–63);
- NUST 13 years (2009 and 2010 ineligible; old lines 28–49%);
- MU_SOY 1 year (2020; old lines 44%);
- URSN 0 years (at most 8 old lines per environment, below the threshold of 10), ESWYT 0 years (almost all lines are new in each cycle; 3–17 old lines per environment), GEM_IA 0 years (no old lines in the target years).

**Under the stopping rule of §6.3: continue** (eligible units 19 ≥ 15; eligible units in datasets with locations 18 ≥ 10).

**Power issue that must be stated explicitly (§7)**:
- Old-line scored cells N = 13,633 (all), 12,740 (G2F + NUST, which have locations). **3/√N = 0.0257 and 0.0266, both above 0.02**. As written in §7: **GO is almost impossible in this scenario**, unless the true gain exceeds about 0.027 (I-B on all genotypes is +0.012).
- Only G2F and NUST remain for the leave-one-dataset-out analysis of H1 and H2. Under §4, both single-dataset point estimates must be > 0.
- The material for the line × location own history (OL3) exists: in eligible environments, the proportion of old-line cells with a historical record at the same location is 65–89% for G2F and 59–94% for NUST.
- Conclusion: the main value of continuing the fitting is **descriptive**: OL1 − OL0 (value of the line's own history) and the completeness of the scenario spectrum. H1 and H2 are reported as usual under the original decision rules.

## Results (2026-09-30; 4090; code 40624af; `results/headroom_oldlines/`; panel files kept on the 4090 in `scratch_ideas_2026-09-29/headroom_oldlines/panels/`)

**Process**: 6 unit tests (`tests/test_oldlines.py`) together with `test_cellspec.py`, `test_sparse.py` and `test_splits.py`, 30 in total, all passed. Panels were generated for all 19 eligible units (G2F 5, NUST 13, MU_SOY 1), consistent with the structural counts; the other 29 units were ineligible and have skip records. **Consistency check passed**: across 305 scored environments, the within-environment Spearman between OL0 and the benchmark `cell_reml` has median 1.0 and minimum 1.0 (to machine precision), with no missing values. Scoring was executed after the SX pre-registration (c809bc2) was committed, and was executed only once. Resolution (G2F + NUST old-line cells N = 12,740) 3/√N = 0.0266.

**Decision (within-environment Spearman, against OL1, Holm correction): both H1 and H2 are NO-GO.**

| Hypothesis | Pooled Δ | Holm-level CI | p | Resolution | Single datasets | Decision |
|---|---|---|---|---|---|---|
| H2 OL3 − OL1 (line × location own history) | **+0.0115** | [+0.0035, +0.0195] (97.5%) | 0.0012 | 0.0266 | G2F +0.0076, NUST +0.0149 | NO-GO: statistically detectable, but below selection resolution |
| H1 OL2 − OL1 (marker G × location) | **+0.0127** | [−0.0040, +0.0293] (95%) | 0.14 | 0.0266 | G2F +0.0136, NUST +0.0130 | NO-GO |

**Descriptive (uncorrected)**
- **OL1 − OL0 (the line's own history, ω chosen by REML) is almost zero**: all 19 units +0.0010 [−0.0031, +0.0051]; G2F +0.0107 [−0.0007, +0.0222], NUST −0.0007, MU_SOY +0.0018. This does not match the prediction in §7 (+0.02 to +0.08): the `cell_reml` prediction for training lines already contains all of their historical phenotypes (the marker kernel is close to full rank), so adding a separate residual term brings almost no new information.
- Relative to OL0: OL2 +0.0180 [+0.0002, +0.0358], OL3 +0.0167 [+0.0081, +0.0252], OL4 +0.0196 [+0.0020, +0.0373]; OL4 − OL1 +0.0141 [−0.0027, +0.0309]. All are below 0.0266.
- H2 within-environment Pearson +0.0093 [+0.0026, +0.0161], top-10% selection differential +0.031 [−0.004, +0.067]; H1 Pearson +0.0098 [−0.0044, +0.0239], selection differential +0.033 [−0.026, +0.092].
- The year effects of H1 range from −0.044 to +0.079, with large differences between years (G2F 2018 −0.041, 2020 +0.079); 15 of the 18 year effects of H2 are positive.
- REML choices: c² is 0.5 for all G2F units and 0.75–1.5 for NUST; the G × location term was selected in every unit; ω is mostly 0.2.
- **An implementation limitation, recorded as is**: the variance-component EM of OL3/OL4 **did not converge in any case** within the pre-registered limit of 200 iterations (convergence threshold: relative change of 1e-10). The reason is that the vast majority of "line × location" pairs have only 1 cell, so σ²_v and σ²_e are hard to separate and EM is very slow. So the shrinkage coefficients of OL3/OL4 are the values at 200 iterations, not the REML optimum. Under §4, after a NO-GO the number of iterations is not changed and nothing is re-run; the direction of its effect on the H2 point estimate is unknown (not verified).

**Comparison with the predictions in §7**: H1 predicted 0 to +0.02: +0.0127, holds; H2 predicted 0 to +0.03: +0.0115, holds; OL1 − OL0 predicted clearly positive: does not hold. Neither GO probability (0.2, 0.25) materialised, mainly because 3/√N = 0.0266 is too high (as stated in the counting appendix).

**Reading (under §8)**
1. In old lines × new year, the line's history at the same location (H2) brings a gain that is **statistically detectable but below selection resolution** (about +0.012). Marker G × location (H1) has a similar point estimate but is uncertain. Both are far below 0.0266; "better than" is not written.
2. Seen together with the scenario spectrum: forward prediction (new lines) has almost no headroom; old lines have a small detectable G × location / line × location signal; in sparse testing M×E gives +0.019 to +0.035. The more information is available, the larger the gain of G × E terms. This description is consistent with the OL results.
3. SX (`docs/prereg_headroom_secondary_2026-09-30.md`) and OL use different scored cells; no joint correction is made for the two.
