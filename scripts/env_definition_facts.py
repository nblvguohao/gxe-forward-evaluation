"""Structure of NUST and URSN environments as described in the manuscript (Section 2.3), written to
results/env_definition_check/structure.json. Run from the repository root (GP_RAW = raw data root)."""
import json, os, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
RAW = os.environ.get("GP_RAW", "<local>/gp_project/data/raw")
S = json.load(open("results/benchmark/splits.json")); norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
ph = pd.read_csv(f"{RAW}/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz", low_memory=False)
ph = ph[ph.Phenotype == "YieldBuA"].copy(); ph["y"] = pd.to_numeric(ph.Value, errors="coerce")
ph["year"] = pd.to_numeric(ph.Experiment.str.extract(r"_(\d{4})$")[0], errors="coerce"); ph = ph.dropna(subset=["y", "year"]); ph["year"] = ph.year.astype(int)
ph["env"] = ph.Location.astype(str) + "_" + ph.year.astype(str); ph["genotype"] = ph.GermplasmId.map(norm); ph["trial"] = ph.Experiment.str.replace(r"_\d{4}$", "", regex=True)
cells = pd.read_parquet("results/benchmark/cells_NUST.parquet"); scored = set(e for v in S["NUST"]["scorable_environments"].values() for e in v)
ph = ph[ph.env.isin(scored) & ph.genotype.isin(set(cells.genotype))]
nt = ph.groupby("env").trial.nunique(); r = ph.groupby(["env", "trial", "genotype"], as_index=False).y.mean(); sh = []
for e, g in r.groupby("env"):
    if g.trial.nunique() < 2: continue
    tot = g.y.var(ddof=0); sh.append(g.groupby("trial").y.transform("mean").var(ddof=0) / tot if tot > 0 else np.nan)
u = pd.read_csv(f"{RAW}/ursn/data/URSN_cleaned_table_phenot_1995_2024.tsv", sep="\t").dropna(subset=["DIS"])
uc = pd.read_parquet("results/benchmark/cells_URSN.parquet"); us = set(e for v in S["URSN"]["scorable_environments"].values() for e in v)
uu = u[u.env_code.astype(str).isin(us) & u.line.astype(str).isin(set(uc.genotype))]
chk = uu.groupby("env_code").apply(lambda g: g.drop_duplicates("line").isCheck.mean())
worst = uu.groupby("env_code").apply(lambda g: bool(g.loc[g.DIS.idxmax(), "isCheck"])); n = uu.groupby("env_code").line.nunique()
out = {"nust_scored_envs": len(scored), "nust_envs_ge2_trials": int((nt >= 2).sum()), "nust_median_trials": float(nt.median()),
       "nust_trial_share_median": float(np.nanmedian(sh)), "ursn_scored_envs": len(us), "ursn_check_share_mean": float(chk.mean()),
       "ursn_envs_worst_is_check": int(worst.sum()), "ursn_envs_ge12": int((n >= 12).sum())}
json.dump(out, open("results/env_definition_check/structure.json", "w"), indent=1); print(out)
