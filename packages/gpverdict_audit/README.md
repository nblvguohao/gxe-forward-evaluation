# GPverdict Audit

**Audit how a genomic prediction model was tuned — from the predictions of the candidates it was chosen among.**

New genomic prediction and G×E models are usually reported after choosing hyperparameters, an early-stopping epoch
or a model among many candidates. GPverdict Audit asks four questions about that choice and one about the result:

| Check | Question | Validated in simulation |
|---|---|---|
| A1 | Was the choice made on the environments or years that are reported? | yes: detection 1.00, false alarms 0.00 |
| A2 | Is the criterion admissible for within-environment selection? Where does the pick of a pooled criterion (pooled Pearson, MSE or Huber loss) rank on within-environment Spearman? | yes: detection 0.85–0.88, false alarms ≤ 0.001 |
| A3 | Does the pooled score of the pick come from environment means? (φ, the share of the attainable pooled r² that environment means explain; the within-environment scale u against its optimum u\*) | descriptive |
| A4 | How much does choosing on the reported data inflate the reported value? (environment bootstrap of the choice, Efron 1983) | descriptive: removes most of the bias in simulation but does not reduce total error, so it is not a validated deployment estimate |
| A5 | Is the lead over a strong baseline resolvable? (two-level bootstrap over environments and genotypes; resolution 3/√N) | — |

The pooled-Pearson decomposition behind A3 is exact for environments of any size:
r(u) = (a + k·u) / √((v_b + u²)·S), maximised at u\* = k·v_b / a.

## Use

Web version (runs in your browser; nothing is uploaded): open `index.html` from a web server, choose the CSV, the
criterion used and, if known, the reported candidate.

```bash
pip install ./packages/gpverdict_audit        # until it is published
gpverdict-audit candidates.csv --manifest manifest.json --baseline baseline.csv --out audit.html
```

```python
import json, gpverdict_audit as ga
res = ga.audit("candidates.csv", json.load(open("manifest.json")), baseline="baseline.csv")
open("audit.html", "w").write(ga.render_html(res))
```

**candidates.csv**: one row per candidate, environment and genotype with columns `candidate`, `environment`,
`genotype`, `observed`, `predicted`; optional `set` (`selection` for the predictions the choice was made on, `report`
for the reported ones, the default).

**manifest.json**: `criterion` (`pooled_pearson`, `pooled_mse`, `pooled_huber`, `within_spearman`, `within_pearson`),
optional `chosen` (the reported candidate), optional `selected_on_report` (true when, e.g., early stopping watched the
test set), optional `selection` / `report` lists of `environments` or `years`.

**baseline.csv** (optional): `environment`, `genotype`, `predicted` of a strong baseline on the same cells, e.g. a
cell-level GBLUP.

Example: `examples/` holds a synthetic case (12 environments × 60 genotypes, 20 epochs whose environment-mean fit
and ranking quality trade off, chosen by pooled Pearson on the reported data): A1 FAIL, A2 WARN (the pick ranks 20 of
20 within environments), A3 WARN (φ = 1.00), worse than the baseline.

Dependencies: numpy and pandas. MIT licence. Lv G., Gu L.
