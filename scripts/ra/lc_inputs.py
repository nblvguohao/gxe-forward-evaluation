"""RA §2.2 inputs for run_lc.R: benchmark G2F cells (env, year, genotype, y, scored), the 2,425 benchmark markers,
and the official APSIM ECs of training and 2024 testing environments. SCRAMBLE=<Y> replaces every yield of years
>= Y by junk (leakage check, prereg §5.2)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dartgxe.forward.data import LOADERS

RAW, BENCH, OUT = Path(os.environ["RAW"]), Path(os.environ["BENCH"]), Path(os.environ["OUT"])
OUT.mkdir(parents=True, exist_ok=True)
ds = LOADERS["G2F"]()
spl = json.load(open(BENCH / "splits.json"))["G2F"]["scorable_environments"]
c = ds.cells[["env", "year", "genotype", "y"]].copy()
c["scored"] = [str(y) in spl and e in set(spl[str(y)]) for e, y in zip(c["env"], c["year"])]
scr = os.environ.get("SCRAMBLE")
if scr:
    late = c["year"] >= int(scr)
    c.loc[late, "y"] = 1e6 * np.random.default_rng(int(scr)).normal(size=int(late.sum()))
c.to_parquet(OUT / "cells.parquet", index=False)
m = ds.markers.copy()
m.columns = [f"m{i}" for i in range(m.shape[1])]
m.reset_index().to_parquet(OUT / "markers.parquet", index=False)
ec = pd.concat([pd.read_csv(RAW / "6_Training_EC_Data_2014_2023.csv"), pd.read_csv(RAW / "6_Testing_EC_Data_2024.csv")])
ec.drop_duplicates("Env").set_index("Env").to_csv(OUT / "ec.csv")
print(len(c), int(c["scored"].sum()), m.shape, ec.shape)
