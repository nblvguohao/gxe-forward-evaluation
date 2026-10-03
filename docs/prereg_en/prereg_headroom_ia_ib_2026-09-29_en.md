English translation of the pre-registration document prereg_headroom_ia_ib_2026-09-29.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Headroom-check pre-registration: I-A (cell model specification) and I-B (G×location marker kernel) (2026-09-29)

Written before any code for this wave was implemented and before any model was fitted. Not changed after commit; changes are only appended as dated sections. Basis: `docs/ideas_gblup_improvement_2026-09-29.md` (commit 92d4785), sections 3–4; the user's instruction of 2026-09-29: "first write the headroom-check pre-registration for I-A and I-B".

**Nature**: a descriptive headroom check, **not a confirmatory test**. All 48 forward years have already been seen, and the I-B hypothesis was derived from the phenotypic structure of these very data (§0). GO only means "worth writing a confirmatory pre-registration for, and preparing new independent data". The results do not say "better" or "confirmed"; they only say "crossed / did not cross the headroom threshold written in advance".

## 0. Known information, stated as is

- On the 48 forward years (G2F 5, NUST 15, URSN 11, ESWYT 12, GEM_IA 4, MU_SOY 1), the per-environment results of the existing method library, the rules and `cell_reml` have all been seen (summary48).
- The descriptive diagnostics in `docs/ideas_gblup_improvement_2026-09-29.md` have been seen. Those directly relevant to this wave:
  - Motivation for I-A: `cell_reml` − two-stage REML was +0.020 in the original 31 years and +0.002 in the independent 17 years; the proportion of environments whose agreement with other environments of the same year is < 0.1 was 30% for ESWYT and 27% for MU_SOY.
  - **The motivation for I-B comes from the same data**: correlations between environment pairs at the same location in different years are higher than between pairs at different locations in different years: NUST +0.157 (486 pairs vs 7,262 pairs), G2F +0.051, ESWYT +0.226 (only 16 pairs). Therefore, if I-B is GO, this only shows that "this structure can be used by a marker model"; it cannot be confirmed on these 43 years.
- Neither model of this wave has **ever been fitted**. No predictions with G×location structure, genotype residual effects or environment heteroscedasticity have ever been produced on these data.
- For location codes, only environment names have been looked at (the mapping rules in §2.2 of this file were written from them). No results have been looked at by location.

## 1. Target years, scoring and control (shared by both ideas)

- Target years, scored environments and scored cells: exactly the same as the benchmark package (`results/benchmark/splits.json`; N = 118,438 scored cells).
- Training data: all years before target year Y, all cells with markers (same as `cell_reml`, `dartgxe.forward.data.LOADERS`).
- **Control**: the model obtained in the same code path with the new parameters set to 0 (in I-A, ω = 0 and no weighting; in I-B, c = 0). Both are mathematically equal to `cell_reml`. They count only after the consistency check in §4 passes.
- Metrics: within-environment Spearman for each scored environment (primary) and top-10% selection differential (reported alongside, not part of the verdict), computed on the lines that have predictions from both models.
- Year effect d_Y = mean over scored environments of (new model − control); SE by environment bootstrap (B = 2000, seed 20260929); pooling with DerSimonian–Laird; zero SEs floored per the 2026-09-28 correction at the median of the non-zero SEs within the same dataset (`dartgxe.forward.pool`).
- Scope: all genotypes (primary); new genotypes (never seen in any earlier year, ≥10 per environment; descriptive).

## 2. Models

Notation: training lines i = 1..n_g, training cells (i, j), j the environment, l(j) the location. Markers are standardised on training lines (mean imputation; the GEM_IA GRM embedding is centred only). K_A = Z Z′ / p, then divided by the mean of its diagonal; K_A = U Λ U′ (all eigenvalues, negative values truncated to 0). The relationship between new lines and training lines, K_A(t, f), is computed with the same standardisation. Environment fixed effects are absorbed by within-environment centring (same as `cell_reml`).

### 2.1 I-A: genotype residual effect + environment heteroscedasticity

- Model: y_ij = μ_j + u_i + e_ij, u ~ N(0, σ²_u K(ω)), K(ω) = (1 − ω) K_A + ω I, e_ij ~ N(0, σ²_e s_j²).
  - ω is the share of "genotype effects not captured by markers": ω = 0 is `cell_reml`. As ω increases, the weight of lines that recur in many environments moves from "proportional to the number of cells" toward "equal weight per line".
  - s_j is the relative residual standard deviation of environment j.
- **Estimation of s_j (one step)**: on the training data, fit the control model (ω = 0, s ≡ 1); the residual variance of environment j is σ̂²_j = Σ r²_ij /(n_j − 1); s_j = σ̂_j / median(σ̂), truncated to the 5th to 95th percentiles of the s values of the training environments; environments with n_j < 3 take s_j = 1.
- **Solving**: at the genotype level. M = Σ_j s_j⁻² [diag(d_j) − d_j d_j′ / n_j] (d_j is the line-occurrence vector of environment j), b = Σ_j s_j⁻² D_j′ y_c,j, and ‖y‖² is weighted in the same way; features F = D_c U S(ω), S(ω) = diag(√((1 − ω) λ_k + ω)). For a given ω, the overall shrinkage λ is found with the same profile REML formula as `Ridge.reml` (same grid and golden-section refinement), and the value of the profile −2ℓ is returned.
- **Choice of ω (training data only)**: ω ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}; the value with the smallest profile REML −2ℓ is taken.
- **Prediction**: training lines û_f = U S(ω) â; new lines û_t = (1 − ω) K_A(t, f) U S(ω)⁻¹ â (when ω = 0, only directions with positive eigenvalues are used). Lines in the target year that appeared earlier use their û_f directly, which includes their residual effect.
- **Main variant (the only one used for the verdict)**: `ia_main` = heteroscedasticity (s_j as above) + REML-selected ω.
- **Descriptive decomposition (not part of the verdict; all reported as is)**:
  - `ia_g`: genotype residual effect only (s ≡ 1, REML ω);
  - `ia_het`: heteroscedasticity only (ω = 0);
  - `ia_std` (the sensitivity variant in the ideas document): training cells are divided, within environment, by the standard deviation of y in that environment; otherwise the same as the control;
  - cross-fitted upper bound for the ω grid: for each target year, the scored environments are randomly split in half; ω is chosen on one half (s fixed at the main-variant estimates, only ω varies), and the difference relative to ω = 0 (with the same weighting) is computed on the other half; both directions, 200 times; pooled in the same way as in pre-studies 1 and 2.

### 2.2 I-B: location-specific marker effects across years

- Model: y_ij = μ_j + a_i + b_{i,l(j)} + e_ij, a ~ N(0, σ²_G K_A), b_{·,l} ~ N(0, σ²_GL K_A^(k)), with b independent across locations; homoscedastic residuals, without the two I-A terms (so that attribution is separate from I-A).
  - K_A^(k) = U_k Λ_k U_k′ are the first k eigen-directions of K_A, k = min(100, n_g − 1) (the ideas document says 200 PCs; this was changed to 100 to control computation, and the change was fixed before any computation);
  - the coordinates of new lines in the G×L part are P_t = K_A(t, f) U_k Λ_k^(−1/2).
- **Active locations**: locations that have records in the training data in ≥ 2 different years and in ≥ 50 cells in total. Other locations have no b (their cells enter only the main effect).
- **Solving**: features [D_c U Λ^(1/2), c · (location-blocked D_c,l U_k Λ_k^(1/2))]; F′F and F′y are accumulated location by location (environments are nested within locations, so the G×L blocks are mutually orthogonal); for a given c, the overall λ is found by profile REML. c² = σ²_GL / σ²_G ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3}; the value with the smallest profile REML −2ℓ is taken (training data only).
- **Prediction**: when the target environment is at active location l, the â part + P_t γ̂_l; otherwise only the â part.
- **Location mapping (fixed before any computation, based only on environment names)**:
  - G2F: after removing the trailing `_year` from env, take the match of the regular expression `^[A-Z]{2}[HS]\d` (`IAH1a`, `IAH1b`, `IAH1c` → `IAH1`; `TXH1-Dry`, `TXH1-Early`, `TXH1-Late` → `TXH1`; `MOH1_1`, `MOH1_2` → `MOH1`; `NYS1` forms its own location);
  - NUST: remove the trailing `_year`; the part after the last `_` is the state/province; from the part before it, non-letter characters are removed (`Saginaw_County_MI` and `SaginawCounty_MI` are merged); locations with different names are not merged (e.g. `Boone_IA` and `BooneCounty_IA`, and `Portageville*_MO`, are each kept separate);
  - URSN: `loc_code`;
  - ESWYT: the full site name after removing the trailing `_year`;
  - GEM_IA, MU_SOY: no locations can be matched across years; they **do not take part in I-B**.
- **Main variant (the only one used for the verdict)**: `ib_main` = the locations above, k = 100, REML-selected c.
- **Descriptive variants (not part of the verdict)**:
  - `ib_region`: locations are replaced by regions: for G2F the first two letters (state/province), for NUST the state/province abbreviation, for ESWYT the country (first word; `South Africa` as one country); not applicable to URSN;
  - cross-fitted upper bound for the c grid (same method as for I-A);
  - reported separately by whether the target environment is at an active location (predictions for environments not at active locations are almost identical to the control, a natural negative control).

## 3. GO / NO-GO (written in advance)

Resolution 3/√N (N = number of scored cells in the relevant scope): 48 years 0.0087; original 31 years 0.0102; independent 17 years 0.0170; 43 years of the 4 I-B datasets 0.0091; NUST 0.0166.

- **I-A GO** requires all of the following:
  1. `ia_main` − control, 48-year pooled within-environment Spearman Δ ≥ 0.0087, and lower 95% CI bound > 0;
  2. after removing any one of the 6 datasets in turn, the pooled point estimate is always > 0;
  3. the situation "original 31 years Δ ≥ 0.0102 while the independent 17 years have a point estimate < 0" does not occur.

  All other cases are NO-GO.
- **I-B GO** (general) requires both of the following:
  1. `ib_main` − control, 43-year pooled Δ ≥ 0.0091, and lower 95% CI bound > 0;
  2. after removing any one of the 4 datasets in turn, the pooled point estimate is always > 0.
- **I-B GO** (NUST-specific): NUST 15-year pooled Δ ≥ 0.0166, and lower 95% CI bound > 0. Confirmation can then only use new data of the NUST type, and this must be noted in the conclusions.
- If neither I-B GO is met: NO-GO.
- Descriptive variants, cross-fitted upper bounds, selection differentials and new-genotype results **do not change the verdict**. Even if a descriptive variant crosses the threshold, it is only recorded and is not a basis for GO (to avoid picking among variants).
- After NO-GO, no new variants are run, the grids are not changed, and the location mappings are not changed.

## 4. Consistency checks (before any comparison; if they fail, stop and check the implementation without looking at comparison results)

- **Unit tests** (`tests/test_cellspec.py`, with small simulated data):
  - with ω = 0 and s ≡ 1, the predictions of `cellspec` agree with `Ridge` marker ridge regression (the implementation of `cell_reml`), relative error < 1e-6 (at the same λ);
  - with c = 0, I-B is the same as the I-A control;
  - no target-year cells appear in training (assertion, same as in the existing forward functions); the estimation of s_j, ω and c calls training data only (assert that input years < Y);
  - the location mapping gives the fixed result for each example listed in §2.2.
- **Real data**: for each target year, the within-environment Spearman correlation between the control-model predictions and the `cell_reml` predictions in the benchmark package, within each scored environment: median ≥ 0.999, minimum ≥ 0.99. If this is not met, first check the implementation (REML refinement or numerical error); no comparison is computed before it is fixed and an explanation is appended.
- Run `pytest tests/test_splits.py` per the project protocol document §2.1 (this wave does not change splits; it is still run as a regression check).

## 5. Directional predictions (written in advance)

- I-A: `ia_main` − control, 48-year pooled, between −0.005 and +0.010; subjective probability of GO 0.25. The REML-selected ω is > 0 in most years; `ia_het` is more likely to have positive point estimates on ESWYT and MU_SOY than on the other datasets.
- I-B: NUST point estimate between 0 and +0.02, G2F close to 0, URSN dominated by noise; subjective probability of general GO 0.2, of NUST-specific GO 0.35. The REML-selected c on NUST is > 0 in most years.
- Power note: the new models are highly correlated with the control. Taking as reference the methods in summary48 that are close to `cell_reml` (`dl_g`, two-stage `reml`, `reml_x10`, `reml_x0.1`), their 48-year (or 31-year) pooled SE is 0.0024–0.0045; for this wave it is assumed to be 0.0025–0.0045. Lower CI bound > 0 requires a point estimate ≥ about 1.96 × SE (≤ 0.0088), which is comparable to the resolution threshold 0.0087, so GO is in practice decided by the resolution threshold: when the true Δ is exactly 0.0087, the probability of GO is about 50%; when the true Δ = 0.012, about 75–85%. The power for the NUST-specific threshold 0.0166 is determined in the same way by its SE, and will be reported as is after the fact.

## 6. Implementation, compute and time

- New files (only added; no existing file is changed):
  - `src/dartgxe/forward/cellspec.py`: genotype-level solving and profile REML for I-A and I-B;
  - `scripts/headroom_ia_ib.py`: panels (predictions for each target year, including the control and all variants) and scoring; the analysis part refuses to run before all panels are complete;
  - `tests/test_cellspec.py`.
- Compute: 4090. Genotype-level matrices are at most about 5,000 dimensions (G2F); the I-B feature dimension is about n_g + 100 × number of active locations (NUST about 10,000), with one eigendecomposition per c; estimated total < 4 hours. Development and smoke checks run 1 year of 1 dataset on the 4090 CPU or the 5070 Ti.
- Output: `results/headroom_ia_ib/{panels/, year_effects.csv, pooled.csv, lodo.csv, crossfit.csv, choices.csv (ω, c and number of active locations for each year), consistency.json, verdict.json, meta/}`; each panel records the git commit, the full config, start and end times, hardware and library versions (the project protocol document §2.4).
- Time box: implementation and testing 2026-09-30 to 10-02; runs and report before 10-03. If time runs out, report the completed part as is; do not extend the grids.
- Hard-rule check (the project protocol document §2):
  - leakage prevention: standardisation, K_A, s_j, ω, c and active locations use only data before Y; target-year phenotypes are used once, only at scoring;
  - evaluation: primary metric within-environment Spearman; selection differential alongside; pooled PCC and RMSE are not reported;
  - fair comparison: control and new model share the same code path, the same training data and the same scored cells; each idea has only one verdict variant.
- Results are written in the "Results" section at the end of this file, and appended to `results/ablation_log.md` per the project protocol document §6 (appended by the authors when merging).

## 7. Use of results (written in advance)

- **Any GO**: write a confirmatory pre-registration for that idea; it can be tested only on new independent data (or on a prospective holdout fixed in advance). For any data that need to be downloaded, first list name, source, size and licence, and wait for the user's consent. In the confirmation stage, the two ideas are Holm-corrected.
- **Both NO-GO**: the evaluation paper's statement "cell-level REML-GBLUP is the most robust deployment choice" is extended to: neither the weighting scheme (genotype residual effects, environment heteroscedasticity) nor G×location marker structure gave headroom beyond the resolution. Whether I-C (non-additive kernels) is still done is decided by the user.
- Whatever the result, the I-A decomposition (`ia_g`, `ia_het`, `ia_std`) and the I-B comparison of active and inactive locations are reported as is.

## Amendment (2026-09-29, after implementation was complete and unit tests passed, before any real-data panel was run)

- The last item of §2.2, "predictions for environments not at active locations are almost identical to the control", is not accurate. When c2 > 0, the main effect â and the G×L terms are fitted jointly, so target environments at inactive locations get **the main-effect part of the joint model**, which is close to, but not the same as, the control (c2 = 0). The unit test checks this correct property: predictions at inactive locations are exactly equal to the main-effect part of the same fit (`test_ib_c2_zero_is_the_ia_control_and_gxl_only_in_active_locations`). The active/inactive split is still reported as planned, but only descriptively, and it is no longer called a negative control.
- The unit test for "heteroscedasticity identification" was also moved to separate toy data (no G×L, 7 years). The earlier version mixed G×L into the residuals. This was a problem of test design and does not concern the model.
- Implementation: the test fixes made after code commit 9f7e8ce are committed together. None of the above changes the models, grids, variants or verdict rules.

## Results (2026-09-29; 4090, code cf7bdff, panels 48/48 with no failures; `results/headroom_ia_ib/`; panel files (17 MB) remain on the 4090 in `scratch_ideas_2026-09-29/headroom_ia_ib/panels/`)

**Consistency check (§4) passed**: over the 48 panels, the within-environment Spearman between the control and benchmark `cell_reml`: in the worst year, median 1.0000 and minimum 0.9999997; no missing cells. Panel runs took 1,121 seconds in total (two 4090s in parallel, about 15 minutes); the analysis took about 5 minutes. `pytest tests/test_cellspec.py tests/test_splits.py tests/test_forward.py`: 25 passed, 6 skipped (re-run and verified after the results were written; the first draft wrongly said 28).

**Verdict (§3; all are descriptive headroom checks)**

| Idea | Scope | Δ (within-environment Spearman) | 95% CI | 3/√N | Verdict |
|---|---|---|---|---|---|
| I-A `ia_main` | 48 years | +0.0055 | [+0.0009, +0.0100] | 0.0087 | **NO-GO** (below resolution; CI includes 0 after removing NUST) |
| I-B `ib_main` general | 43 years (G2F, NUST, URSN, ESWYT) | **+0.0120** | [+0.0057, +0.0183] | 0.0091 | **GO** (still positive after removing any one dataset: +0.0108 to +0.0136) |
| I-B `ib_main` NUST-specific | 15 years | +0.0121 | [+0.0038, +0.0203] | 0.0166 | NO-GO (below the NUST resolution) |

**I-A details**
- By dataset: NUST +0.0079 [+0.0034, +0.0124], MU_SOY +0.034 (1 year), GEM_IA +0.010, URSN +0.007, ESWYT +0.0005, G2F −0.0007 (all include 0, except NUST). Original 31 years +0.0061, independent 17 years +0.0043; the situation in condition 3, "original 31 years cross the threshold while the independent years are negative", did not occur.
- Decomposition: `ia_g` (genotype residual effect only) +0.0015 [+0.0001, +0.0029], `ia_het` (heteroscedasticity only) +0.0010 [−0.0029, +0.0048], `ia_std` (within-environment standardisation) +0.0019 [−0.0010, +0.0048]. That is, the gain of `ia_main` is not a simple sum of the gains of the two terms, and it cannot be attributed to either one of them.
- Choice of ω: the REML-selected ω increases as the training set grows (NUST 0.02→0.2, G2F 0.2 in all years, ESWYT 0→0.2); for URSN it is 0 in all years. This shows that the "equal weight per line" component is adopted by REML in data-rich datasets.
- Cross-fitted upper bound for the ω grid: +0.0036 [+0.0004, +0.0068] (G2F +0.016), below resolution.
- Selection differential +0.0026 [−0.0149, +0.0201]; new genotypes +0.0051 [−0.0007, +0.0108].

**I-B details**
- By dataset: G2F +0.0158 [+0.0058, +0.0258] (5 years, resolution 0.0130, crossed), ESWYT +0.0145 [−0.0043, +0.0333], NUST +0.0121 [+0.0038, +0.0203], URSN −0.0006 [−0.0183, +0.0172]. Number of positive years: G2F 4/5, NUST 11/15, ESWYT 8/12, URSN 5/11.
- Target environments at active and at inactive locations: active +0.0149 [+0.0078, +0.0219] (969 environments), inactive −0.0025 [−0.0103, +0.0052] (168 environments). The gain appears only in environments where the G×L term is enabled, which is consistent with the mechanism.
- New genotypes: +0.0101 [+0.0024, +0.0177] (38 years). Selection differential: +0.0148 [+0.0010, +0.0286] (G2F +0.036).
- Descriptive variants: `ib_region` (region level) +0.0145 [+0.0072, +0.0219] (G2F, NUST, ESWYT, 32 years in total); its point estimate is higher than at location level, but the verdict was not changed on this basis. Cross-fitted upper bound for the c² grid: +0.0084 [+0.0027, +0.0141] (lower than the +0.0120 of the REML choice; the two use different evaluation environments and cannot be subtracted).
- Selected c²: G2F 0.2–0.3, NUST 0.75–1.5, URSN 0–0.5; for ESWYT, from 2009 onward, 9/12 years are at the grid upper limit 3.0 (ESWYT 2006 is 0). Per §3 the grid is fixed and is not extended. This means that the G×L strength on ESWYT may be truncated by the grid and the ESWYT point estimate may be conservative; but it may also be overfitting. This was not checked. The REML profile δ did not land on a search endpoint (`any_at_edge` all False).

**Check of the directional predictions (§5)**
- I-A: predicted between −0.005 and +0.010, GO probability 0.25: the point estimate +0.0055 is within the predicted interval, and NO-GO agrees with the majority expectation. "ω > 0 in most years" holds (34/48 years; the rest are mainly URSN and a few small years). The prediction that `ia_het` is positive on ESWYT and MU_SOY holds (+0.001, +0.034), but the magnitude is very small.
- I-B: predicted NUST between 0 and +0.02, G2F close to 0, URSN dominated by noise, general GO probability 0.2: **NUST holds, G2F does not hold** (G2F is +0.016, higher than predicted), URSN holds, and the general GO occurred (prior probability 0.2). The REML-selected c² on NUST is > 0 in 15/15 years; the prediction holds.

**Reading and limitations (stated as is)**
1. This is a headroom check, not a confirmation: the I-B hypothesis came from the phenotypic structure of the same 43 years (§0), and GO only means "worth writing a confirmatory pre-registration for, and preparing new independent data". Two ideas were checked on the same data, and the I-B general GO is not corrected for multiple comparisons. That it remains positive after leaving out one dataset (+0.011 to +0.014) shows that it is not driven by a single dataset.
2. The gain is about 1.2 percentage points (Spearman), just crossing the 43-year resolution of 0.0091. On new data of a size similar to the independent 17 years, the resolution is 0.017, and **a true effect of +0.012 would fall below the "resolution" threshold there**. The confirmatory pre-registration needs to make this explicit when choosing the threshold (for example, recompute it with the N of the new data, or use "CI excludes 0" together with "point estimate ≥ an effect size written in advance").
3. Parts not examined: the effect of the c² grid truncation on ESWYT, the robustness of the location definition (`ib_region` being higher shows that the location level is not the optimal granularity), and the sensitivity to k = 100. None of these was checked in this wave, and none should be tuned in this wave.
4. This wave did not modify any existing documents, results or scripts. `results/ablation_log.md` needs to be appended by the authors when merging. Suggested entry: "Headroom check I-A NO-GO (+0.0055 [+0.0009, +0.0100], below 0.0087); I-B general GO (+0.0120 [+0.0057, +0.0183], above 0.0091; active-location environments +0.0149, inactive −0.0025), NUST-specific NO-GO (+0.0121, below 0.0166); descriptive, needs new independent data for confirmation."
