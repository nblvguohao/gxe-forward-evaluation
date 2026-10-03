"""Run C2 (ceiling + structural overlap) and C3 (linear parental probe). See docs/prestudy_parental_2026-09-26.md."""
import json
import time

import numpy as np
import pandas as pd

from dartgxe.baselines.common import load_folds, pred_frame, write_predictions
from dartgxe.features.fit import load_ec, load_geno
from dartgxe.paths import RESULTS, processed, splits_dir
from dartgxe.prestudy.parental import env_repeatability, fit_probe, infer_parents

OUT = RESULTS / "prestudy"
OUT.mkdir(parents=True, exist_ok=True)
FYNEW = {"FYnew2018": 2018, "FYnew2020": 2020, "F2022m": 2022}

# ---- C2: repeatability ceiling in the FYnew test years
plot = pd.read_parquet(processed("g2f") / "pheno_plot.parquet")
rep = env_repeatability(plot)
rep["year"] = rep["env"].str.rsplit("_", n=1).str[1].astype(int)
rep.to_csv(OUT / "repeatability_by_env.csv", index=False)
c2 = {}
for yr in FYNEW.values():
    r = rep[rep.year == yr]
    c2[yr] = {"envs": int(r["H"].notna().sum()), "median_H": float(r["H"].median()), "median_ceiling": float(r["ceiling"].median()),
              "q25_ceiling": float(r["ceiling"].quantile(.25)), "q75_ceiling": float(r["ceiling"].quantile(.75)),
              "mean_reps": float(r["mean_reps"].mean())}
allfy = rep[rep.year.isin(FYNEW.values())]
c2["pooled_median_ceiling"] = float(allfy["ceiling"].median())

# ---- structural overlap: are test parents known from training?
cells = pd.read_parquet(processed("g2f") / "pheno.parquet")
struct = {}
for sc, yr in {**FYNEW, "CV1parent": None}.items():
    sp = pd.read_parquet(splits_dir("g2f") / f"{sc}.parquet")
    s = sp[sp.fold == 0]
    fit_roles = ["train", "val", "refit_extra"] if yr else ["train"]
    fit = set(s.loc[s.role.isin(fit_roles), "genotype"])
    te = s.loc[s.role == "test", "genotype"].drop_duplicates()
    p1_fit = {h.split("/")[0] for h in fit}
    t_fit = {h.split("/")[1] for h in fit if "/" in h}
    struct[sc] = {"test_hybrids": int(len(te)), "hybrid_seen": float(te.isin(fit).mean()),
                  "parent1_seen": float(te.str.split("/").str[0].isin(p1_fit).mean()),
                  "tester_seen": float(te.str.split("/").str[1].isin(t_fit).mean())}
json.dump({"C2": c2, "structure": struct}, open(OUT / "c2_structure.json", "w"), indent=1)
print(json.dumps({"C2": c2, "structure": struct}, indent=1), flush=True)

# ---- C3: linear probe
geno, ec = load_geno(), load_ec()
parents = infer_parents(geno)
p1, p2, p1g, tg = parents
print("parents: parent1", p1g.shape, "testers", tg.shape, "tester alleles undetermined", float(np.isnan(tg.to_numpy()).mean()), flush=True)
for sc in [*FYNEW, "CV1parent"]:
    frames, info, t0 = {}, {}, time.time()
    for fold in load_folds(sc):
        te, out, inf = fit_probe(fold, geno, ec, parents)
        for m, p in out.items():
            frames.setdefault(m, []).extend(pred_frame(te, p, m, sc, fold.fold, s) for s in range(5))
            info.setdefault(m, {})[f"fold{fold.fold}"] = inf[m]
        print(sc, fold.fold, round(time.time() - t0), "s", {k: round(v["val_spearman"], 4) for k, v in inf.items()}, flush=True)
    for m, fr in frames.items():
        write_predictions(fr, m, sc, range(5), {"deterministic": True, "tuning": info[m], "prestudy": True})
print("C3_DONE", flush=True)
