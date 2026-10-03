"""Structure count for the INIA rice dataset (docs/count_plan_inia_2026-09-29.md). No model is fitted.
Usage: INIA_DIR=<dir with Phenotypes.txt, Trials.txt, Lines.txt, README.md> OUT=<dir> python count_inia_structure.py
Optional: INIA_GENOTYPED=<file with the LINE_ID of genotyped lines, one per line> restricts the lines to those."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

D, OUT = Path(os.environ["INIA_DIR"]), Path(os.environ["OUT"])
OUT.mkdir(parents=True, exist_ok=True)
ph = pd.read_csv(D / "Phenotypes.txt", sep=r"\s+", na_values=["NA"], usecols=["YEAR", "TRIAL", "LOCATION", "LINE_ID", "GY"])
ph = ph.dropna(subset=["GY"])
if os.environ.get("INIA_GENOTYPED"):
    keep = set(Path(os.environ["INIA_GENOTYPED"]).read_text().split())
    ph = ph[ph["LINE_ID"].isin(keep)]
tag = "genotyped" if os.environ.get("INIA_GENOTYPED") else "all_lines"
MIN_PRIOR = 2


def cells(env_cols):
    c = ph.assign(env=ph[env_cols].astype(str).agg("|".join, axis=1))
    return c.groupby(["env", "YEAR", "LOCATION", "LINE_ID"], as_index=False)["GY"].mean()


def count(cells_df, min_n):
    rows, envs = [], cells_df.groupby(["env", "YEAR", "LOCATION"]).agg(n=("LINE_ID", "nunique")).reset_index()
    first = cells_df.groupby("LINE_ID")["YEAR"].min()
    years = sorted(envs["YEAR"].unique())
    for Y in years:
        prior = [y for y in years if y < Y]
        e = envs[envs["YEAR"] == Y]
        ok = e[e["n"] >= min_n]
        prior_ok = sum(1 for y in prior if (envs[(envs["YEAR"] == y)]["n"] >= min_n).any())
        lines = cells_df[(cells_df["YEAR"] == Y) & cells_df["env"].isin(ok["env"])]["LINE_ID"].unique()
        new_share = float((first.reindex(lines) >= Y).mean()) if len(lines) else np.nan
        loc_prior = envs[envs["YEAR"] < Y].groupby("LOCATION")["YEAR"].nunique()
        seen1 = float(ok["LOCATION"].map(lambda l: loc_prior.get(l, 0) >= 1).mean()) if len(ok) else np.nan
        seen2 = float(ok["LOCATION"].map(lambda l: loc_prior.get(l, 0) >= 2).mean()) if len(ok) else np.nan
        n_cells = int(ok["n"].sum())
        rows.append({"year": Y, "min_n": min_n, "envs": len(e), "envs_ok": len(ok), "median_lines_env_ok": float(ok["n"].median()) if len(ok) else np.nan,
                     "lines_scored": len(lines), "new_share": new_share, "prior_years_ok": prior_ok,
                     "loc_seen_ge1": seen1, "loc_seen_ge2": seen2, "cells": n_cells,
                     "resolution": 3 / np.sqrt(n_cells) if n_cells else np.nan,
                     "qualifies": bool(prior_ok >= MIN_PRIOR and len(ok) >= 5 and new_share >= 0.5)})
    return pd.DataFrame(rows)


res = []
for label, cols in (("loc_year", ["LOCATION", "YEAR"]), ("trial", ["LOCATION", "YEAR", "TRIAL"])):
    c = cells(cols)
    for m in (25, 10):
        r = count(c, m)
        r.insert(0, "env_def", label)
        res.append(r)
PY = pd.concat(res, ignore_index=True)
PY.to_csv(OUT / f"per_year_{tag}.csv", index=False)

# same-location, different-year environment pairs: shared lines (environment = location x year)
c = cells(["LOCATION", "YEAR"])
c = c[c.groupby("env")["LINE_ID"].transform("nunique") >= 10]
S = {e: set(g["LINE_ID"]) for e, g in c.groupby("env")}
meta = c.drop_duplicates("env").set_index("env")[["LOCATION", "YEAR"]]
pairs = []
for loc, g in meta.groupby("LOCATION"):
    es = list(g.index)
    for i in range(len(es)):
        for j in range(i + 1, len(es)):
            if g.loc[es[i], "YEAR"] != g.loc[es[j], "YEAR"]:
                pairs.append({"loc": loc, "gap": abs(int(g.loc[es[i], "YEAR"]) - int(g.loc[es[j], "YEAR"])),
                              "shared": len(S[es[i]] & S[es[j]])})
PR = pd.DataFrame(pairs)
PR.to_csv(OUT / f"pairs_{tag}.csv", index=False)
q = PR["shared"].quantile([.25, .5, .75]).tolist() if len(PR) else []
tr = pd.read_csv(D / "Trials.txt", sep=r"\s+", na_values=["NA"])
summary = {"tag": tag, "plots_with_gy": int(len(ph)), "lines": int(ph["LINE_ID"].nunique()), "years": [int(ph["YEAR"].min()), int(ph["YEAR"].max())],
           "locations": int(ph["LOCATION"].nunique()), "trials_in_phenotypes": int(ph.groupby(["YEAR", "LOCATION", "TRIAL"]).ngroups),
           "trials_table_rows": int(len(tr)), "loc_year_environments": int(c["env"].nunique()),
           "same_location_pairs": int(len(PR)), "shared_lines_quartiles": q, "pairs_with_shared_ge10": int((PR["shared"] >= 10).sum()) if len(PR) else 0,
           "qualifying_years": {f"{a}_{b}": int(x["qualifies"].sum()) for (a, b), x in PY.groupby(["env_def", "min_n"])}}
json.dump(summary, open(OUT / f"summary_{tag}.json", "w"), indent=1)
with open(OUT / "SHA256SUMS", "w") as f:
    for n in ("README.md", "Trials.txt", "Lines.txt", "Phenotypes.txt"):
        f.write(f"{hashlib.sha256((D / n).read_bytes()).hexdigest()}  {n}\n")
pd.set_option("display.width", 250)
print(json.dumps(summary, indent=1))
print(PY[(PY.env_def == "loc_year")].round(3).to_string(index=False))
