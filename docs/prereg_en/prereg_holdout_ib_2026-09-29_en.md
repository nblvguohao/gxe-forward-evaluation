English translation of the pre-registration document prereg_holdout_ib_2026-09-29.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Pre-registration of a holdout confirmation for I-B: outcomes never scored (other traits) (2026-09-29)

Written before any structure count, panel fitting or scoring for any other trait. Not changed after commit; changes are only appended as dated sections. Basis: `docs/prereg_headroom_ia_ib_2026-09-29.md` (headroom-check results, commit a367831); `docs/count_plan_inia_2026-09-29.md` (INIA rice not usable, commit beff6f0); the user's choice on 2026-09-29: "adjust the confirmation design, and mark out a holdout in the existing data in advance".

## 0. First, what this design can and cannot show

- **The yield outcomes of all 43 forward years have been scored by I-B**, and the GO verdict was made on them. Therefore, splitting these 43 years again into "training years" and "holdout years" cannot be an independent confirmation: the yield result of every year has already entered the decision "is I-B worth pursuing". I do not design such a holdout.
- What in the existing data **has truly not been scored by any method, and whose outcomes I have not seen, is the other traits of the same datasets**. This design uses them as the holdout: with completely frozen I-B code and grids, predict NUST plant height, maturity and other traits, and URSN VSK.
- **What it tests**: whether the mechanism "location-specific marker effects repeated across years (G×location kernel) can improve the within-environment ranking of new lines in new years" also applies to traits other than yield.
- **What it cannot show**: "I-B improves yield prediction". The yield conclusion can still only be confirmed on new independent yield data (SRPN, or new years released in future).
- **Mechanistic link, stated in advance**: traits such as maturity and plant height are themselves among the causes of yield G×location (responses of lines to latitude and day length). So a positive effect on the other traits is **expected**. It is a mechanism test, not an independent replication of the yield conclusion.

## 1. Known information, stated as is

- Headroom check (commit a367831): `ib_main` − control, pooled over the 43 yield forward years, Δ = +0.0120 [+0.0057, +0.0183]; NUST 15 years +0.0121 [+0.0038, +0.0203]; URSN 11 years −0.0006 [−0.0183, +0.0172]. Environments with the G×L term enabled +0.0149; environments at inactive locations −0.0025.
- **Other traits: only which traits the datasets contain, and the number of records, number of locations, number of lines and year range for each trait, have been seen** (from statistics of the `Phenotype` column of the NUST phenotype file and of the column names of the URSN table). No mean, variance, correlation or method prediction of any trait has been seen.
- The processed G2F data contain only yield (`trait = yield_Mg_ha`); ESWYT contains only yield BLUEs; GEM_IA and MU_SOY have no locations across years. These three types do not take part in this design.
- Frozen code: `src/dartgxe/forward/cellspec.py`, git blob `4ca8090345cc73aca5fde9b9ba31483031fdf8a3`, last modified in commit cf7bdff (no later commit changed this file). The grids, K_GL, the active-location rule and the location mappings are all kept, **with no adjustment of any kind**.

## 2. Data and trait inclusion rules

- **Datasets**: NUST (traits other than YieldBuA), URSN (traits other than DIS). Markers, lines and environment definitions are the same as in the yield analysis (marker and line filters of `dartgxe.forward.data.load_nust` / `load_ursn`; environment = location × year; cell = mean of all plots/replicates of that line in that environment).
- **Candidate traits** (fixed by trait name before any results):
  - NUST: `Height`, `Maturity`, `SeedSize`, `Lodging`, `Protein`, `Oil`, `SeedQuality`;
  - URSN: `VSK` (DIS has already been used as the main trait).
- **Explicitly excluded**:
  - `YieldRank` (the within-trial rank of yield; a function of yield, not an independent outcome) and `DescriptiveCode` (a code for text descriptions);
  - the remaining NUST traits: candidates are limited to traits with ≥50,000 records that cover ≥100 locations (i.e. the 7 above). `Chlorosis` and the traits after it have ≤11,895 records and ≤40 locations (the fatty-acid composition traits cover only 2017–2019). With so few locations they cannot support G×location, so they are not included.
- **Counting rules for inclusion** (based only on sample structure; the same rule for each candidate trait):
  - Qualifying target year: at least 2 earlier years each have one environment meeting the threshold; that year has ≥5 environments meeting the threshold; proportion of new lines ≥0.5 (W1c rule). Thresholds: NUST ≥25 lines with markers per environment, URSN ≥10.
  - A trait is included only if it has at least **5 qualifying target years**; otherwise it is recorded and excluded.
  - The set of qualifying target years is computed separately for each trait (from the trait's own history and cells).
- The count results will be written as an appended section of this file and committed **before any fitting**.

## 3. Confirmatory comparison (1 comparison, two-sided 95%, no multiple-comparison correction needed)

- **H_hold**: `ib_main` − control (`ctrl`, i.e. `cell_reml`), within-environment Spearman, pooled over all included traits.
- **Unit and estimation**:
  - unit = (dataset, target year). Traits within the same year share environments and lines, and their errors are correlated, so traits are not treated as independent observations;
  - for each environment, first take the equally weighted mean of Δ over the traits in that environment; the year effect of a unit = the mean of this average over the scored environments of that year;
  - SE = environment-level bootstrap (B = 2000, seed 20260930; all traits share the same resampled environments);
  - pooling = DerSimonian–Laird random effects; zero SEs handled per the 2026-09-28 correction (`dartgxe.forward.pool`).
- **Resolution**: 3/√N, N = total number of **distinct (environment, line) cells** in the scored environments of all units (multiple traits on the same cell are counted once, because they are not independent scored cells).
- **Verdict** (the project protocol document §2.2):
  - Δ ≥ 3/√N **and** 95% CI excludes 0: write "on the holdout traits, I-B is better than the control";
  - only the CI condition met: write "statistically detectable, but below the selection resolution";
  - neither met: write "tie (no effect seen on the holdout traits)".
- **Sample-size condition**: if the included traits together give fewer than 5 units, **no verdict is made**; write only "the holdout design is underpowered".

## 4. Descriptive checks of robustness and mechanism (not part of the verdict)

- pooled estimate and CI for each trait separately; pooled point estimate after removing any one trait (leave-one-trait-out);
- separate estimates for environments at active locations and at inactive locations (for yield, +0.0149 and −0.0025; the same direction is expected);
- the region-level variant `ib_region` (state/province);
- the pooled Spearman level (control).
- The top-10% selection differential is not reported: for traits such as maturity and lodging, the "good" direction is not uniform, so the selection differential has no uniform meaning.

## 5. Directional prediction and power (written in advance)

- **Direction**: pooled Δ > 0. G×location is usually stronger for maturity and plant height (latitude and day-length responses), so Δ for these two traits is expected to exceed the +0.012 for yield; for rating-scale traits such as lodging and seed quality, Δ is expected to be close to 0. The gain in active-location environments is expected to be larger than in inactive-location environments.
- **Power**: taking as reference the NUST year SE in the yield analysis (median 0.014) and the between-year variance (τ² = 2.7×10⁻⁵), the NUST 15-year pooled SE ≈ 0.0038; averaging over several traits will make the SE smaller. Minimum detectable effect at 80% power ≈ 0.011 (conservative, using the yield SE).
- **The resolution is the stricter threshold**: when N is similar to that for yield (NUST about 33,000 cells), 3/√N ≈ 0.017, **higher than the +0.012 observed for yield**. So unless G×location is markedly stronger for some traits, the most likely result is "statistically detectable, but below the selection resolution". Subjective probabilities: CI excludes 0, about 0.7; Δ ≥ 3/√N and CI excludes 0, about 0.35.
- If the result is "detectable but below the resolution", this sentence is written as is, and the threshold is not redefined.

## 6. Order of execution and stopping conditions

1. Commit this pre-registration;
2. write `scripts/holdout_ib_traits.py` (trait cell construction, counting, panels, scoring); unit tests pass (trait cells contain only that trait, the exclusion list takes effect, counts agree with hand calculation);
3. run the **structure count** (no model fitting); determine the included traits and target years per §2; append to this file and commit;
4. if the included traits together give < 5 units: stop, write "the holdout design is underpowered", and do not continue;
5. otherwise, run panels on the 4090 with the frozen code (one panel per (trait, target year); first do the consistency check of the control versus `ib_main`: the control uses the same `cell_reml` implementation as for the same dataset in the yield analysis, and unit tests ensure equivalence at c² = 0);
6. once all panels are complete, score once, without adjusting any parameter after looking at intermediate results;
7. append the results as the "Results" section of this file.

## 7. Hard-rule check

- Leakage prevention: panels are generated by the frozen `panel_year` and trained on years before the target year; standardisation, kernels, the choice of c² and the active locations use training data only; the target-year outcomes of the test traits are used once, only at scoring; nothing is rerun because of any result.
- Evaluation: primary metric within-environment Spearman; RMSE and pooled PCC are not reported.
- Fair comparison: the control and `ib_main` share the same code path, the same training cells and the same scored cells.
- Reproducibility: record git commit, config, seed, GPU, CUDA/PyTorch version, start and end times (reusing the metadata recording of `headroom_ia_ib.py`); final numbers are run only on the 4090.
- Honesty: whatever the result, all traits (including excluded traits and the reasons) and all descriptive checks are reported together.

## 8. Use of results (written in advance)

- Verdict "better" or "detectable but below resolution": written in the evaluation paper as mechanism-level supporting evidence, clearly labelled "holdout test on other traits; confirmation on yield still needs new independent data". **Do not write "confirms that I-B improves yield prediction".**
- Verdict "tie": the I-B yield GO should be regarded as a selective finding on 43 years and written into the limitations of the evaluation paper; no further investment in confirmatory testing.
- **Frozen procedure for yield confirmation**: once new independent yield data are obtained (SRPN with an SNP table, or new years released in future), use the same frozen code (blob 4ca8090…), the same grids and the same verdict rules; write a separate confirmatory pre-registration with a structure count of those data; then score once.

## Amendment (2026-09-29, structure count complete, before any panel fitting)

Code: `src/dartgxe/forward/traitcells.py`, `scripts/holdout_ib_traits.py`, `tests/test_traitcells.py`; run on the 4090 (5 seconds); output `results/holdout_ib/count/{per_trait_year.csv, included.json, checks.json}`. No model was fitted.

**Consistency check (passed)**: the same code, recomputing target years with yield cells, gives NUST 15 years (2005–2017, 2019, 2020) and URSN 11 years (1999–2004, 2008, 2009, 2014, 2015, 2019), identical to the existing W1c results.

**By the rules of §2** (each trait has at least 5 qualifying target years): all 8 candidate traits **are included; no trait is excluded**.

| Trait | Qualifying target years (main rule) | Scored cells |
|---|---|---|
| NUST Height | 15 (2005–2017, 2019, 2020) | 27,379 |
| NUST Maturity | 15 (as above) | 29,409 |
| NUST SeedSize | 15 (as above) | 30,051 |
| NUST Lodging | 15 (as above) | 28,666 |
| NUST SeedQuality | 15 (as above) | 27,114 |
| NUST Protein | 14 (2005 missing) | 19,370 |
| NUST Oil | 14 (2005 missing) | 19,402 |
| URSN VSK | 11 (1999–2003, 2008, 2019–2022, 2024) | 1,118 |

- Relaxed definition (threshold 10): 15–16 years for each NUST trait, 11 years for URSN VSK; almost the same as the main rule; inclusion is not affected.
- **Units (dataset × target year) total 26**: NUST 15, URSN 11. This exceeds the condition "at least 5", so per §6.4 the work **continues**.
- Distinct (environment, line) cells within units total N = 33,284; **resolution 3/√N = 0.0164** (consistent with the estimate of 0.017 in §5). NUST units have about 1,000–3,400 cells each; URSN units have 55–196 cells each.
- The qualifying years of URSN VSK differ from those of yield (DIS): 2020, 2021, 2022 and 2024 are years in which DIS did not qualify but VSK did; 2004, 2009, 2014 and 2015 are years in which DIS qualified but VSK did not. This does not change any rule; it only shows that the URSN units do not fully coincide with the years of the yield analysis.
- Proportion of new lines: the minimum across NUST traits and target years is 0.51, and for URSN 0.58; both are above the 0.5 threshold.

Next step, per §6 step 5: generate a panel for each (trait, target year) with the frozen code, then score once.

## Results (2026-09-29; 4090; code commit 05c4176; the frozen `cellspec.py` blob 4ca8090 verified by a runtime assertion; `results/holdout_ib/`)

**Process**: the pre-check passed (recomputing the yield panels for NUST 2005 and 2012 and URSN 2003 through the holdout path, the predictions of the control and `ib_main` are bitwise identical to the existing yield panels; the difference between `ctrl` and c²=0 is ≤ 3.6×10⁻⁸). All 114 panels (NUST 5 traits × 15 years + 2 traits × 14 years, URSN VSK × 11 years) were generated, with no failures. Panel runs took 2,030 seconds in total, from 15:16 to 15:34, on two RTX 4090s. Scoring was run only once, and no parameter was adjusted in between. Panel files and the per-environment table (`per_env_trait.parquet`) remain on the 4090 in `scratch_ideas_2026-09-29/holdout_ib/`.

**Confirmatory comparison (H_hold, §3)**

| Comparison | Units | Δ (within-environment Spearman) | 95% CI | 3/√N | Verdict |
|---|---|---|---|---|---|
| `ib_main` − control, 8 traits pooled | 26 | **−0.0015** | [−0.0036, +0.0005] | 0.0164 (N = 33,284) | **Tie** |

- By dataset: NUST (15 units) −0.0016 [−0.0037, +0.0004]; URSN (11 units) +0.0019 [−0.0100, +0.0138].
- 11 of the 26 units are positive (NUST 5/15). The upper bound of the pooled CI is only +0.0005; the +0.012 observed for yield lies far outside the interval: **no gain comparable to that for yield was seen on the holdout traits, and an effect larger than +0.0005 is unlikely**.
- Compared with the prior predictions of §5: predicted pooled Δ > 0, maturity and plant height larger than the +0.012 for yield, and a larger gain in active-location environments. **The predictions did not hold**: maturity −0.0030, plant height +0.0032, active environments −0.0013.

**Descriptive checks (§4, not part of the verdict)**

| Check | Estimate | 95% CI |
|---|---|---|
| Height | +0.0032 | [−0.0016, +0.0079] |
| Maturity | −0.0030 | [−0.0072, +0.0013] |
| SeedSize | −0.0005 | [−0.0034, +0.0024] |
| Lodging | −0.0037 | [−0.0112, +0.0039] |
| SeedQuality | −0.0047 | [−0.0140, +0.0047] |
| Protein | −0.0008 | [−0.0040, +0.0024] |
| Oil | +0.0016 | [−0.0033, +0.0065] |
| URSN VSK | +0.0019 | [−0.0100, +0.0139] |
| Leave-one-trait-out (after removing any one trait) | −0.0007 to −0.0023 | All include 0 |
| Active-location environments (2,372) | −0.0013 | [−0.0037, +0.0011] |
| Inactive-location environments (459) | +0.0036 | [−0.0118, +0.0190] |
| Region-level variant `ib_region` (NUST) | −0.0006 | [−0.0031, +0.0019] |
| Mean within-environment Spearman of the control | 0.443 | — |

- No single trait has an interval that excludes 0, and no trait has a point estimate close to the +0.012 for yield.
- The difference between active-location and inactive-location environments (for yield, +0.0149 versus −0.0025) is not present on the holdout traits.
- REML-selected c²: median 0.1; only 1.8% of panels selected 0; none landed on the grid upper limit. Medians by trait: SeedQuality 1.5 (range 1.0–2.0), Lodging 0.3, URSN VSK 0.3, Height/Oil/Protein 0.1, Maturity/SeedSize 0.05. The median for NUST yield over 15 years was 1.0 (0.75–1.5). For SeedQuality, REML found a very strong G×location variance (c² comparable to yield), but there was no gain in prediction (−0.0047).

**Reading (per §8, written in advance)**
- Verdict "tie": **the I-B yield GO should be regarded as a selective finding on 43 years and written into the limitations of the evaluation paper; no further investment in confirmatory testing, unless new independent yield data are obtained in future.**
- It must be made clear that this result does not prove that there is no effect on yield. What it shows is this: on the other traits of NUST and URSN, the same frozen model did not improve the ranking of new lines in new years; maturity and plant height, whose G×location is usually the strongest, also showed no gain. Therefore, the mechanism "a G×location kernel can improve the ranking of new lines" is not supported, and the +0.012 on yield remains without independent confirmation.
- Limitations: URSN units have very few cells (55–196), so their weights are small and the URSN interval is wide; the NUST traits are highly correlated with each other, so 8 traits are not 8 independent tests; REML selected large c² while prediction did not improve, which may mean that the model-selection criterion is not aligned with the prediction-ranking objective. This wave does not investigate this further.
- Suggested entry to append to `results/ablation_log.md` (to be merged by the authors): "I-B holdout test (other traits, 7 NUST + URSN VSK, 26 units, frozen code): Δ −0.0015 [−0.0036, +0.0005], 3/√N = 0.0164, tie; the I-B gain on yield (+0.0120) was not confirmed and, per the pre-registration, is regarded as a selective finding on 43 years."
