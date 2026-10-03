English translation of the pre-registration document prereg_headroom_sparse_2026-09-29.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Headroom-check pre-registration: sparse testing within the target year (SP) — can learned environment correlations exceed M×E GBLUP (2026-09-29)

Written before any implementation and before any fitting. Not changed after commit; changes are only appended as dated sections. Basis: the user's request of 2026-09-29 ("when the target year has phenotypes from related environments in the same year … I hope you can also try this"), and the user's choices on three design questions: primary metric within-environment Spearman; done in this branch of dart-gxe; headroom check first.

**Nature**: a descriptive headroom check, not a confirmatory test. GO only means "worth writing a confirmatory pre-registration, and preparing data that did not take part in this check".

## 0. Known information, stated as is

- [reference to a separate unpublished study omitted]. **Those results have already been seen in this analysis**; for G2F and ESWYT, the data overlap with the data of this check (masking, splits and year definitions differ).
- **I-B in this branch** (`docs/prereg_headroom_ia_ib_2026-09-29.md`): location-specific marker effects across years gave +0.012 over 43 years in the forward scenario, and a tie in the holdout on other traits.
- **The three models in this check have never been fitted in the SP scenario**; the model that learns environment correlations (M2) has never been run on any data.
- Target years and scored environments are the same as in the benchmark package, and the yield outcomes of these years have all been scored in the forward scenario. The SP scenario uses part of the cells of these years for training and another part for scoring; **the cell-level split is new**.

## 1. Scenario and data

- Datasets and target years: the 6 datasets of the benchmark package, 48 target years, and the scored environments of each target year (`results/benchmark/splits.json`).
- For target year Y:
  - **History**: all years before Y, all cells with markers (same as `cell_reml`);
  - **Target-year cells**: cells of lines with markers in the scored environments of Y. Cells of Y that are not in scored environments are not used (neither for training nor for scoring).
- **Masking (observed / scored)**:
  - Line g appears n_g times in the scored environments of Y. When n_g ≥ 2, the observed quota is q_g = min(max(round(f·n_g), 1), n_g − 1); when n_g = 1, q_g = 1, i.e. that line is used only for training in Y and is not scored.
  - Allocation: lines are processed in random order; for each line, among the environments in which it appears, q_g environments with "the lowest observed proportion" are chosen, with ties broken by random numbers. The random numbers are determined by (dataset, Y, f, seed). Masking does not read any phenotype value.
  - **Observed cells** enter training; **the remaining cells are scored cells**.
- Proportion f ∈ {0.25 (main), 0.5 (secondary)}; seeds {0, 1}.
- Scored environments: target-year scored environments in which the number of scored cells ≥ the dataset threshold (25; URSN 10). For URSN at f = 0.5, most environments are expected to fall below the threshold; this will be reported as is.

## 2. Models (cell level, environment fixed effects, solved at the genotype level, all using training cells only)

Notation as in `docs/prereg_headroom_ia_ib_2026-09-29.md` §2: K_A = U Λ U′; P = U_k Λ_k^(1/2) for the first k directions, k = min(100, n_g − 1).

- **M0 main-effect GBLUP**: u ~ N(0, σ² K(ω)), K(ω) = (1 − ω)K_A + ωI, with ω selected by profile REML on `cellspec.W_GRID` (same as `ia_g` of I-A, without environment heteroscedasticity). ω is added so that the main-effect baseline can use the line's own same-year observations; ω = 0 is `cell_reml`.
- **M1 M×E GBLUP**: the main effect of M0 (with the ω selected by M0 fixed) + an environment-specific marker effect b_·j ~ N(0, σ²c² K_A^(k)) for each scored environment j of the target year, independent across environments; c² selected by profile REML on `cellspec.C2_GRID` (c² = 0 is M0). Historical environments have no environment-specific term and enter only the main effect.
- **M2 learned environment correlation (FA-like)**: the covariance of the G×E term is changed to K_A^(k) ⊗ E_ρ, E_ρ = (1 − ρ)I + ρR̂.
  - R̂: from the M1 fit (with its REML c²), obtain the vector of environment-specific genetic values for each target-year environment (P γ_j on training lines); R̂ is the correlation matrix between these column vectors;
  - c² is fixed at the value selected by M1; ρ ∈ {0.25, 0.5, 0.75, 1.0} is selected by profile REML (ρ = 0 is M1);
  - everything uses training cells only (history + observed cells); the phenotypes of scored cells are not read.
- Prediction: the prediction for scored cell (i, j) = main effect + the G×E term of that environment. Scored lines are all among the training lines (each scored line has at least 1 observed cell).

## 3. Metrics and pooling

- Primary metric: within-environment Spearman on **scored cells** in each scored environment. Descriptive: within-environment Pearson, top-10% selection differential (on scored cells).
- Environment-level Δ: mean over the two seeds (if the environment does not reach the threshold under one seed, only the seeds that reach it are used); year effect = mean of Δ over scored environments; SE: environment bootstrap (B = 2000, seed 20260930); pooling: DerSimonian–Laird, with the same zero-SE floor as before (`dartgxe.forward.pool`); scopes: 48 years, original 31 years, independent 17 years, by dataset; leave-one-dataset-out.
- Resolution: 3/√N, N = total number of scored cells at f = 0.25, seed 0.

## 4. Verdict (f = 0.25)

- **Main comparison (decides GO)**: M2 − M1.
  - **GO**: 48-year pooled Δ ≥ 3/√N, lower 95% CI bound > 0, and the pooled point estimates after leaving out each dataset all > 0.
  - Otherwise **NO-GO**.
- **Mandatory descriptive comparisons**: M1 − M0 (whether a G×E signal exists; expected to be positive; used to re-check the findings of [reference to a separate unpublished study omitted]); M2 − M0; all comparisons at f = 0.5.
- If M1 − M0 itself is not positive (upper CI bound < 0 or point estimate ≤ 0), the interpretation of M2 − M1 must also state "on these data the G×E term does not help"; the verdict rule is unchanged.
- Descriptive comparisons and f = 0.5 do not change the verdict. After NO-GO, no new variants are run and the grids are not changed.

## 5. Consistency checks and unit tests (before any comparison)

- `tests/test_sparse.py` (new file):
  - masking: per-line quota, balance across environments, reproducible with the same seed, independent of phenotype values, lines with n_g = 1 have no scored cells;
  - leakage prevention: when the phenotypes of scored cells are replaced by garbage values, the predictions and REML choices of M0, M1 and M2 are **bitwise identical**; the history contains only years before Y;
  - nesting: M1 at c² = 0 equals M0; M2 at ρ = 0 equals M1 (relative error < 1e-6);
  - M1 and M2 agree with dense ridge-regression solutions with explicit features on small simulated data (relative error < 1e-6).
- Real data: at f = 0.25, seed 0, the training of M0 at ω = 0 agrees with the control of `cellspec.fit_ia` (same data, same λ).
- Run `pytest tests/test_splits.py` per the project protocol document §2.1 (this wave adds a new split but does not change existing splits).

## 6. Directional predictions (written in advance)

- M1 − M0: clearly positive, expected +0.02 to +0.06, most evident on G2F, ESWYT and NUST.
- M2 − M1: a small positive value, expected 0 to +0.02; datasets with many environments and clear grouping (ESWYT, G2F) are more likely to be positive. Subjective probability of GO 0.3.
- REML choice of ρ: > 0.25 in most target years.

## 7. Implementation, compute and time

- New files (only added): `src/dartgxe/forward/sparse.py` (masking and M0–M2; imports `Prep`, `reml_solve` and `fit_ia` from `cellspec`, and **does not modify the frozen `cellspec.py`**), `scripts/headroom_sparse.py`, `tests/test_sparse.py`.
- Compute: 4090; 48 years × 2 proportions × 2 seeds = 192 units; per unit, 15 eigendecompositions for M0, 11 for M1 and 4 for M2; estimated total 2–4 hours (two GPUs in parallel).
- Output: `results/headroom_sparse/`; each panel records the git commit, config, GPU, CUDA/PyTorch version, start and end times.
- Time box: implementation and tests 1–2 days; runs and report 1 day.

## 8. Use of results (written in advance)

- **GO**: write a confirmatory pre-registration. Confirmation must use data not used in this check: candidates are the `ARS_MRASeq` years of SRPN and NRPN in T3 (2020–2026, about 45 lines × 15–30 locations per year, nearly fully crossed, suitable for sparse testing; ARS permission to use the data must be obtained first), or other new data. Structure counts and MDE will be written at that time.
- **NO-GO**: recorded as "in sparse testing, learned environment correlations did not exceed M×E GBLUP"; the M1 − M0 result is still reported as a descriptive finding (whether a G×E signal exists in sparse testing).
- Whatever the result, all comparisons, all datasets and both proportions are reported as is.

## Results (2026-09-29; 4090, code 9e609a9; `results/headroom_sparse/`; panel files remain on the 4090 in `scratch_ideas_2026-09-29/headroom_sparse/panels/`)

**Process**: all 5 unit tests passed (masking, leakage prevention, agreement with dense solutions with explicit features, nesting); the frozen `cellspec.py` blob is still 4ca8090; regression tests: 17 passed, 6 skipped. All 192 panels (6 datasets × 48 target years × 2 proportions × 2 seeds) were generated, with no failures, in 2,588 seconds in total (two RTX 4090s, 19:31–19:58). Scoring was run once. At f = 0.25, 46 target years have scored environments (for 2 URSN years, every environment has fewer than 10 scored cells), N = 84,508, 3/√N = 0.0103; at f = 0.5, 36 years, N = 45,642, 3/√N = 0.0140.

**Verdict (f = 0.25, main comparison M2 − M1, within-environment Spearman): NO-GO.**

| Comparison | 46-year pooled Δ | 95% CI | Years positive | Leave-one-dataset-out |
|---|---|---|---|---|
| **M2 − M1** (learned environment correlation − M×E) | **+0.0001** | [−0.0029, +0.0030] | 19/46 | −0.0008 to +0.0024 |
| M1 − M0 (M×E − main-effect GBLUP) | **+0.0187** | [+0.0136, +0.0238] | 36/46 | — |
| M2 − M0 | +0.0186 | [+0.0119, +0.0253] | — | — |

- **M2 did not exceed M×E**: point estimates by dataset: G2F −0.0083 [−0.0137, −0.0030] (M2 significantly worse), NUST +0.0026, ESWYT +0.0046, GEM_IA +0.0032, MU_SOY +0.0097, URSN +0.0001; independent 17 years +0.0048 [+0.0004, +0.0092], original 31 years −0.0017. On the top-10% selection differential, M2 − M1 is −0.0117 [−0.0230, −0.0005], so M2 is, if anything, slightly worse. At f = 0.5, M2 − M1 is −0.0017 [−0.0045, +0.0010].
- **REML choices**: the G×E term was selected (c² > 0) in 74.5% of units; in these units ρ was almost always selected at the upper end of the grid, 1.0 (ρ = 1 in 72.9% of all units). That is, the likelihood strongly prefers the learned environment correlation, but this did not translate into better ranking on scored cells. In most URSN units c² = 0 (no G×E term). The REML profile did not land on a δ search endpoint.

**Descriptive: the G×E signal in sparse testing (M1 − M0)**
- f = 0.25: +0.0187 [+0.0136, +0.0238], **crossing the resolution of 0.0103, with a CI that excludes 0**; by dataset G2F +0.0389, NUST +0.0279, MU_SOY +0.0266, GEM_IA +0.0224, ESWYT +0.0092, URSN +0.0001. Original 31 years +0.0234, independent 17 years +0.0118. Within-environment Pearson +0.0146; top-10% selection differential +0.0266 [+0.0141, +0.0392].
- f = 0.5: +0.0349 [+0.0253, +0.0445] (resolution 0.0140); selection differential +0.0668.
- Compared with the predictions in §6: M1 − M0 was predicted at +0.02 to +0.06; +0.019 at f = 0.25 is slightly below the lower end, and +0.035 at f = 0.5 is within the range; the direction agrees with the findings of [reference to a separate unpublished study omitted]. M2 − M1 was predicted at 0 to +0.02 with GO probability 0.3: the point estimate +0.0001 is at the lower end of the range, NO-GO; "ESWYT and G2F more likely to be positive" did not hold (G2F significantly negative). "ρ > 0.25 in most years" held.

**Reading**
1. In sparse testing where "the target year has phenotypes from related environments in the same year", **a G×E signal does exist, and it can be realised with the published M×E GBLUP**: relative to main-effect GBLUP, within-environment Spearman increases by about 0.02 (25% observed) to 0.035 (50% observed), crossing the resolution. This is a clear contrast with the forward scenario (where no method exceeded main-effect REML-GBLUP).
2. **On top of M×E, learning the genetic correlations between environments brought no additional gain**: M2 − M1 is +0.0001, and slightly worse on the selection differential.
3. Nature: this is a descriptive headroom check; M1 − M0 is a mandatory descriptive comparison written in advance, not a confirmatory test; M×E is a known method (Lopez-Cruz et al. 2015), so M1 − M0 by itself is not a methodological novelty.
4. Per §8: NO-GO, recorded as "in sparse testing, learned environment correlations did not exceed M×E GBLUP"; M1 − M0 is reported as a descriptive finding.
