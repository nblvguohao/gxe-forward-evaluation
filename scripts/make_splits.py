"""Write every split of spec 01 §1.4 to data/processed/g2f/splits/ and a summary table."""
import json

import pandas as pd

from dartgxe.paths import processed, splits_dir
from dartgxe.splits.core import make_all, novelty

cells = pd.read_parquet(processed("g2f") / "pheno.parquet")
out = splits_dir("g2f")
out.mkdir(parents=True, exist_ok=True)
summary = []
for name, sp in make_all(cells).items():
    sp.to_parquet(out / f"{name}.parquet", index=False)
    for f, s in sp.groupby("fold"):
        t = s[s["role"] == "test"]
        n_env = t.groupby("env").size()
        row = {"scenario": name, "fold": int(f), **{r: int((s["role"] == r).sum()) for r in ("train", "val", "refit_extra", "test", "drop")},
               "test_envs": int(n_env.size), "test_envs_ge25": int((n_env >= 25).sum()),
               "N_test_cells_ge25": int(n_env[n_env >= 25].sum())}
        if name.startswith("F"):
            row["val_new_hybrid_share"] = round(novelty(s, "val", ("train",)), 3)
            row["test_new_hybrid_share"] = round(novelty(s, "test", ("train", "val", "refit_extra")), 3)
        summary.append(row)
summ = pd.DataFrame(summary)
summ.to_csv(out / "summary.csv", index=False)
print(summ.to_string(index=False))
