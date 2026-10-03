"""Gate-2 amendment 2: replace the 2024 test environments' daily weather in data_v2 with NASA POWER
(fetched by scripts/fetch_power_2024.py). The released file is kept as env_daily_released.parquet."""
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from dartgxe.paths import processed

g = processed("g2f")
src = g / "env_daily.parquet"
keep = g / "env_daily_released.parquet"
if not keep.exists():
    shutil.copy(src, keep)
d = pd.read_parquet(keep)
p = pd.read_csv(Path(os.environ["G2F_RAW"]) / "power_2024" / "power_2024_daily.csv")
p = p.rename(columns={"Env": "env", "Date": "date"})
p["date"] = pd.to_datetime(p["date"].astype(str), format="%Y%m%d")
p = p[p["env"].isin(set(d["env"]) | set(p["env"]))][d.columns]
is24 = d["env"].str.endswith("_2024")
out = pd.concat([d[~is24], p[p["env"].isin(set(d.loc[is24, "env"]))]], ignore_index=True)
out.to_parquet(src, index=False)
rep = {"envs_replaced": int(p["env"].isin(set(d.loc[is24, "env"])).groupby(p["env"]).any().sum()),
       "rows_before_2024": int(is24.sum()), "rows_after_2024": int(out["env"].str.endswith("_2024").sum()),
       "nan_share_2024_before": float(d.loc[is24].drop(columns=["env", "date"]).isna().mean().mean()),
       "nan_share_2024_after": float(out[out["env"].str.endswith("_2024")].drop(columns=["env", "date"]).isna().mean().mean())}
(g / "weather_patch_2024.json").write_text(json.dumps(rep, indent=1))
print(rep)
