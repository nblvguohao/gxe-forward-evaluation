"""Synthetic example for the browser page and README: 12 environments x 60 genotypes, 20 candidates (e.g. training
epochs). Early candidates fit environment means well but rank genotypes poorly; later ones rank well but get the
environment means wrong. The 'author' picks by pooled Pearson on the reported environments (the pattern the audit is
built to catch). Writes example_candidates.csv, example_baseline.csv and example_manifest.json."""
import json

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
J, n, K = 12, 60, 20
mu = rng.normal(0, 2.5, J)
g = rng.normal(0, 1, n)
ge = rng.normal(0, 0.7, (J, n))
y = mu[:, None] + g[None, :] + ge + rng.normal(0, 1, (J, n))
z = (g[None, :] + ge) / np.sqrt(1.49)
rows = []
for c in range(K):
    q = c / (K - 1)
    s, rho, t = (0.1 + 1.2 * q) * 2.5, 0.05 + 0.5 * q, 0.3 + 0.9 * q
    p = mu[:, None] + rng.normal(0, s, J)[:, None] + t * (rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=(J, n)))
    for j in range(J):
        for i in range(n):
            rows.append((f"E{j:02d}", f"G{i:03d}", round(y[j, i], 4), round(p[j, i], 4), f"epoch{c + 1:02d}"))
df = pd.DataFrame(rows, columns=["environment", "genotype", "observed", "predicted", "candidate"])
df.to_csv("example_candidates.csv", index=False)
C = df.groupby("candidate").apply(lambda d: np.corrcoef(d.observed, d.predicted)[0, 1], include_groups=False)
chosen = C.idxmax()
base = pd.DataFrame({"environment": np.repeat([f"E{j:02d}" for j in range(J)], n),
                     "genotype": np.tile([f"G{i:03d}" for i in range(n)], J), "observed": y.ravel().round(4),
                     "predicted": (mu[:, None] * 0 + 0.45 * z + rng.normal(0, 0.9, (J, n))).ravel().round(4)})
base.to_csv("example_baseline.csv", index=False)
json.dump({"criterion": "pooled_pearson", "chosen": chosen, "selected_on_report": True}, open("example_manifest.json", "w"), indent=1)
print("chosen", chosen, "rows", len(df))
