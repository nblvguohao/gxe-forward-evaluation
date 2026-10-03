English translation of the pre-registration document prereg_sequel_wave2d_2026-09-28.md (original in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

---

# Sequel pre-registration, wave 2 D (positive sprint, direction 2): can a deep model trained with a within-environment objective beat cell-level REML? (2026-09-28)

Written before any deep model in this wave was run on the forward panels. Not changed after commit; amendments are only appended as dated sections. Basis: `docs/sequel_plan_v2_2026-09-28.md` §9. On 2026-09-28 the user agreed to carry out the "positive sprint" (direction 1 and direction 2 in parallel).

## 0. Known information, declared as is

- The per-environment performance of cell-level REML (`cell_reml`) on the 31 target years is known (pre-study 2, B3b).
- Deep models have **never been run** on these 31 target years. This project's network N1 has been run only on the two G2F scenarios F2024m and F2022m (gate 2: F2024m +0.059, F2022m tied).
- Literature precedent: in Eckhoff 2026, DNNs trained with a within-environment objective or the MSED loss gained only 0.2–3% over E-BLUP. In stage 1, all ranking losses were tied with twohead.

## 1. Models

Target years, scored environments and historical years are the same as in waves 2b/2c. Genotype PCs (80) and EC PCs (5) are fitted on the same scope as in wave 2c, i.e., the training set only.

- **`dl_g` (primary)**: multilayer perceptron; input = 80 genotype PCs; output = one genotype score.
  - Loss: mean squared error after within-environment centring, i.e., for each environment, the respective environment means are subtracted from both the targets and the predictions before computing the MSE. It differs from the mean squared error of within-environment pairwise differences (MSED, Piepho 1998; Eckhoff 2026) only by a constant factor, and learns only within-environment differences.
  - Each batch consists of several complete environments.
  - Used in all three datasets.
- **`dl_ge` (secondary)**: as above, with 5 EC PCs added to the input. Genotype and environment features are concatenated in the first layer, so G×E can be learned. Target environments use historical-mean EC (as in wave 2c). G2F and URSN only.

## 2. Tuning, early stopping and retraining (none touch target-year phenotypes)

- Validation year V = the most recent historical year (with scored environments) before target year Y.
- Train on the years before V. Using the **within-environment Spearman** of year V (an admissible metric) as the criterion, select (configuration, number of epochs E\*) among the following 8 configurations, each with up to 200 epochs:
  - hidden-layer width {64, 256};
  - dropout {0.1, 0.3};
  - weight decay {1e-4, 1e-2};
  - two hidden layers, Adam, learning rate 1e-3.
- With the selected configuration, retrain for E\* epochs on all years before Y, and predict Y.
- Seeds {0, 1, 2}; the reported prediction is the mean of the 3 seeds' predictions (each seed is tuned and retrained separately).
- Tuning budget stated explicitly: cell-level REML is not tuned (REML is its tuning).

## 3. Confirmatory comparisons (2; Holm; two-sided; primary metric within-environment Spearman)

- **H5**: `dl_g` − `cell_reml`, all scored genotypes, random-effects pooling over the 31 target years (the same year-level effects, environment-bootstrap SEs and zero-SE floor rule as in waves 2b/2c).
- **H6**: the same contrast, scoring only new genotypes (≥ 10 per environment).
- **GO criterion (positive sprint, §9.3)**: for H5 or H6, the CI at the corresponding Holm level excludes 0, and the point estimate is ≥ 3/√N (N = the corresponding number of scored cells: all 87,294; new genotypes 68,781).
- **Directional prediction (fixed in advance)**: both **tied** (basis in §0).

## 4. Descriptive results

- `dl_ge` − `cell_reml` (G2F + URSN);
- `dl_g` − two-stage `mlp` (atlas style, trained on genotype means), i.e., whether the within-environment objective and batching by environment have an effect;
- Selection-differential metric; results by dataset; the configuration and E\* selected for each target year.

## 5. Hard-rule check and implementation

- Leakage prevention: target-year phenotypes are used only once, at the final scoring. Tuning and early stopping use only validation year V. All transformations are fitted only on the training set. New unit tests assert that the tuning training set contains no data from V or later years, and that the retraining contains no data from Y or later years.
- Code: `src/dartgxe/forward/dl.py`, `scripts/b3c_panels.py`, `scripts/b3c_analyze.py` (refuses to run when panels are incomplete).
- Compute: 4090 GPU. Time box: completed before 2026-10-10.

## Results (2026-09-29; 4090 GPU; panel code 50e212a; analysis `scripts/b3c_analyze.py` run at fe47ffe; this script is identical to 50e212a, and fe47ffe only adds the reading code for wave 2e, which does not affect G2F/NUST/URSN)

Output: `results/b3c/{pooled.json, year_effects.csv, per_env.parquet, verdict.json, meta/}`. 141 panels (G2F 5 years × 2 models × 3 seeds, NUST 15 years × `dl_g` × 3, URSN 11 years × 2 × 3), no failures. Scored cells: all N = 87,294; new genotypes N = 68,781.

**Confirmatory comparisons (Holm; within-environment Spearman)**

| Contrast | Estimate | CI | Verdict | Advance prediction |
|---|---|---|---|---|
| H5: `dl_g` − `cell_reml` (all genotypes; smaller p, tested first) | −0.0083 | [−0.0192, +0.0025] (97.5%) | Tied | Tied ✔ |
| H6: as above, new genotypes | −0.0028 | Holm stopped, not tested (95% interval [−0.016, +0.010]) | — | Tied ✔ |

**The GO criterion (§3) is not met → direction 2 NO-GO.**

**Descriptive**
- H5 by dataset: G2F −0.006, NUST −0.004, URSN −0.020 [−0.042, +0.002]. 10 of the 31 years are positive (G2F 2/5, NUST 5/15, URSN 3/11). The point estimates are negative in all three datasets.
- Selection differential: H5 −0.012 [−0.033, +0.010], H6 +0.015 [−0.020, +0.049]; both tied.
- `dl_ge` − `cell_reml` (G2F + URSN, 16 years): −0.023 [−0.048, +0.003]; selection differential +0.009; tied. Adding historical-mean EC brought no improvement (consistent with the G×E learners of wave 2c).
- **The within-environment objective and batching by environment do have an effect**: `dl_g` − two-stage `mlp` = +0.089 [+0.063, +0.115] (31 years); selection differential +0.163 [+0.052, +0.275]. That is, the same network, switched to a within-environment loss, moves from "clearly worse" to tied with cell-level REML, but does not surpass it.
- The selected configurations are widely spread (all 8 configurations were selected at least once; the most frequent one 25 times). The median E\* is 26 epochs.

**One-sentence conclusion**: a deep model trained with a within-environment objective is tied with cell-level REML-GBLUP across 3 crops and 31 forward years (point estimates slightly lower). It does not reach the "beats the strong baseline" needed for a positive paper. The result is consistent with the literature (Eckhoff 2026: DNNs gain only 0.2–3% over E-BLUP) and with stage 1 of this project.
