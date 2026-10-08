"""Post hoc sensitivity of the forward benchmark to the environment definitions of NUST and URSN (2026-10-04).

Prompted by a data audit: NUST environments (Location_year) pool several trials (median 4) whose means differ, and
URSN environments include check cultivars (mean share 0.286). The stored per-cell predictions of the benchmark are
rescored without refitting: NUST with yield centred within trial, or with trial x location x year as the scoring unit
(n >= 10); URSN without check cultivars (>= 3 or >= 8 non-check lines). Other datasets keep their year effects from
results/summary48. Not pre-registered.

Run from the repository root (GP_RAW = raw data root):  python3 scripts/env_definition_sensitivity.py"""
from pathlib import Path
import re, json, sys, warnings, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0,"src"); from dartgxe.forward.pool import dersimonian_laird, floor_se
warnings.filterwarnings("ignore")
import os
RAW=os.environ.get("GP_RAW","<local>/gp_project/data/raw"); S=json.load(open("results/benchmark/splits.json"))
norm=lambda s: re.sub(r"[^A-Z0-9]","",str(s).upper()); rng=np.random.default_rng(20261004); B=2000
LIB=["reml","reml_x0.1","reml_x10","reml_x100","ridge_pc20","rf","gbm","knn10","knn30","mlp","dl_g","rn_ridge","gxe_gbm"]
def sp(y,p): return np.nan if len(y)<3 or np.std(y)==0 or np.std(p)==0 else float(np.corrcoef(rankdata(y),rankdata(p))[0,1])
def score(ds, cells, minn):
    """cells: env, year, genotype, y (scoring units). predictions joined by (orig_env, genotype)."""
    P=pd.read_parquet(f"results/benchmark/predictions_{ds}.parquet").pivot_table(index=["env","genotype"],columns="method",values="pred")
    rows=[]
    for (e,Y),g in cells.groupby(["env","year"]):
        if g.genotype.nunique()<minn: continue
        pr=P.reindex(pd.MultiIndex.from_arrays([g.oenv,g.genotype])); y=g.y.to_numpy(float)
        base=sp(y,pr["cell_reml"].to_numpy(float))
        r={"dataset":ds,"target":Y,"env":e,"n":len(g),"cell_reml":base}
        for m in LIB:
            if m in pr: r[m]=sp(y,pr[m].to_numpy(float))
        rows.append(r)
    return pd.DataFrame(rows)
def yearfx(E):
    out=[]
    for m in LIB:
        if m not in E: continue
        for (d,Y),g in E.groupby(["dataset","target"]):
            v=(g[m]-g["cell_reml"]).dropna().to_numpy()
            if not len(v): continue
            out.append({"item":m,"dataset":d,"target":Y,"d":v.mean(),"se":v[rng.integers(0,len(v),(B,len(v)))].mean(1).std(ddof=1),"n_env":len(v)})
    return pd.DataFrame(out)
# ---------- NUST variants
ph=pd.read_csv(f"{RAW}/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz",low_memory=False)
ph=ph[ph.Phenotype=="YieldBuA"].copy(); ph["y"]=pd.to_numeric(ph.Value,errors="coerce")
ph["year"]=pd.to_numeric(ph.Experiment.str.extract(r"_(\d{4})$")[0],errors="coerce"); ph=ph.dropna(subset=["y","year"]); ph["year"]=ph.year.astype(int)
ph["env"]=ph.Location.astype(str)+"_"+ph.year.astype(str); ph["genotype"]=ph.GermplasmId.map(norm); ph["trial"]=ph.Experiment.str.replace(r"_\d{4}$","",regex=True)
tg={int(y):set(v) for y,v in S["NUST"]["scorable_environments"].items()}
ph=ph[ph.apply(lambda r: r.env in tg.get(r.year,()),axis=1)]
cN=pd.read_parquet("results/benchmark/cells_NUST.parquet"); cN=cN[cN.env.isin(set().union(*tg.values()))]
rec=ph.groupby(["env","year","trial","genotype"],as_index=False).y.mean(); rec=rec[rec.genotype.isin(set(cN.genotype))]
orig=cN.assign(oenv=cN.env)
rec["yd"]=rec.y-rec.groupby(["env","trial"]).y.transform("mean")
det=rec.groupby(["env","year","genotype"],as_index=False).yd.mean().rename(columns={"yd":"y"}); det["oenv"]=det.env
tle=rec.assign(oenv=rec.env, env=rec.env+"|"+rec.trial)[["env","oenv","year","genotype","y"]]
# ---------- URSN variants
u=pd.read_csv(f"{RAW}/ursn/data/URSN_cleaned_table_phenot_1995_2024.tsv",sep="\t").dropna(subset=["DIS"])
chk=set(u.loc[u.isCheck==True,"line"].astype(str))
ug={int(y):set(v) for y,v in S["URSN"]["scorable_environments"].items()}
cU=pd.read_parquet("results/benchmark/cells_URSN.parquet"); cU=cU[cU.apply(lambda r: r.env in ug.get(int(r.year),()),axis=1)].assign(oenv=lambda x:x.env)
noc=cU[~cU.genotype.isin(chk)]
V={"NUST":{"orig":(orig,25),"detrend":(det,25),"trial_env":(tle,10)},"URSN":{"orig":(cU,10),"nocheck_n3":(noc,3),"nocheck_n8":(noc,8)}}
res={}
for ds,vs in V.items():
    for k,(c,mn) in vs.items():
        E=score(ds,c,mn); res[(ds,k)]=E
        means=E[["cell_reml"]+[m for m in LIB if m in E]].mean().round(3)
        print(f"{ds:5s} {k:10s} units {len(E):5d} cells {int(E.n.sum()):6d} | mean within-unit Spearman: cell_reml {means['cell_reml']}, range of library {means.drop('cell_reml').min()}–{means.drop('cell_reml').max()}")
# ---------- pooled 48-year contrasts
Y48=pd.read_csv("results/summary48/year_effects.csv"); Y48=Y48[(Y48.metric=="spearman")&(Y48.kind=="method")]
chkd=yearfx(res[("NUST","orig")]).merge(Y48[Y48.dataset=="NUST"],on=["item","target"],suffixes=("","_s48"))
print("\ncheck vs summary48 (NUST orig): max |d diff| %.2e over %d rows"%((chkd.d-chkd.d_s48).abs().max(),len(chkd)))
chku=yearfx(res[("URSN","orig")]).merge(Y48[Y48.dataset=="URSN"],on=["item","target"],suffixes=("","_s48"))
print("check vs summary48 (URSN orig): max |d diff| %.2e over %d rows"%((chku.d-chku.d_s48).abs().max(),len(chku)))
res48=0.0087
def pooled(nk,uk):
    other=Y48[~Y48.dataset.isin(["NUST","URSN"])][["item","dataset","target","d","se"]]
    ye=pd.concat([other,yearfx(res[("NUST",nk)]),yearfx(res[("URSN",uk)])])
    out={}
    for m,g in ye.groupby("item"):
        r=dersimonian_laird(floor_se(g[["dataset","d","se"]].reset_index(drop=True)))
        v="resolved loss" if r["hi"]<0 and -r["est"]>=res48 else ("resolved gain" if r["lo"]>0 and r["est"]>=res48 else ("detectable" if (r["lo"]>0 or r["hi"]<0) else "tied"))
        out[m]=(r["est"],r["lo"],r["hi"],r["k"],v)
    return out
tab={}
for nk,uk in (("orig","orig"),("detrend","nocheck_n3"),("detrend","nocheck_n8"),("trial_env","nocheck_n3"),("trial_env","orig"),("detrend","orig")):
    tab[f"NUST:{nk} / URSN:{uk}"]=pooled(nk,uk)
T=pd.DataFrame(tab).T
pd.set_option("display.width",300); pd.set_option("display.max_colwidth",40)
for m in ["reml","rf","gbm","mlp","knn10","knn30","ridge_pc20","reml_x0.1","reml_x10","reml_x100","dl_g","rn_ridge","gxe_gbm"]:
    print("\n"+m); print(T[m].to_string())
OUT=Path("results/env_definition_check"); OUT.mkdir(parents=True,exist_ok=True)
rows=[{"variant":v,"item":m,"est":t[0],"lo":t[1],"hi":t[2],"k":t[3],"verdict":t[4]} for v,d in tab.items() for m,t in d.items()]
pd.DataFrame(rows).to_csv(OUT/"pooled_by_variant.csv",index=False)
pd.concat([E.assign(variant=k[1]) for k,E in res.items()]).to_csv(OUT/"per_unit.csv",index=False)
print("written",OUT)
