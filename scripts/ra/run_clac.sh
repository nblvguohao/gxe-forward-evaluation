#!/bin/bash
# RA §2.1: run the public CLAC script (third_party/CLAC/Script_CLAC_model.R, sha256 in SHA256SUMS_reanalysis_2026-09-30)
# for target year Y inside ra/Y<Y>/ (inputs from scripts/ra_build_inputs.py). The only change is recorded in
# clac_patch.txt: in Model B, a fixed-effect level of the target year unseen in training gets coefficient 0 instead
# of NA (constant within an environment, so within-environment ranks are unaffected).
set -euo pipefail
Y=$1
RA=<workstation>/dart-gxe/scratch_ideas_2026-09-29/ra
R=<workstation>/dart-gxe/envs/r-reanalysis/bin/Rscript
SRC=<workstation>/dart-gxe/third_party/CLAC/Script_CLAC_model.R
cd $RA/Y$Y
cp $SRC Script_CLAC_model.R
python3 - <<'PY'
p='Script_CLAC_model.R'
s=open(p).read()
old="""S42 = S42[,1:3]"""
new="""S42 = S42[,1:3]
# RA patch: unseen fixed-effect level -> 0 (constant within environment)
if(anyNA(S42$Yield_Mg_ha)){
  z0 = function(v, lev) { out = v[as.character(lev)]; out[is.na(out)] = 0; out }
  S42$Yield_Mg_ha = fit$Coefficients$Intercept +
    ifelse(PS_dta$Irr, ifelse(is.na(fit$Coefficients$Irr), 0, fit$Coefficients$Irr), 0) +
    z0(fit$Coefficients$Trt, PS_dta$Trt) + z0(fit$Coefficients$PC, PS_dta$PC) +
    z0(fit$Coefficients$State, PS_dta$State) + z0(fit$Coefficients$Station, PS_dta$Station) +
    c(X[as.character(PS_dta$Hybrid),] %*% fit$Structure$Hybrid)
  cat('RA patch applied to', sum(is.na(S42$Yield_Mg_ha)), 'rows\\n')
}"""
assert s.count(old)==1
s=s.replace(old,new)
# RA patch 2: the 2024-release VCF holds 2,425 markers; the public script samples 35,000 (a speed shortcut) -> take all
o2="sample_geno = sort(sample(nrow(gen2),35000))"
assert s.count(o2)==1
s=s.replace(o2,"sample_geno = sort(sample(nrow(gen2),min(35000,nrow(gen2))))  # RA patch 2")
# RA patch 4: bWGR::EigenGAU computes sqrt(xx + yy - 2xy) in single precision; identical EC rows (e.g. TXH1-Dry/Early
# share weather) give a tiny negative -> NaN, and the normalising sum turns the whole kernel NaN. Same formula
# (bWGR 2.2.18 src/RcppEigen20230423.cpp lines 29-38) in double precision with negatives clamped to 0.
# RA patch 5: bWGR 2.2.18 renamed mkr2X's variance output Vb2 -> VB2; alias it (the script's own checks unchanged)
o5="    fit = tryCatch(mkr2X(Y,G,Q),error = function(e) NULL)"
assert s.count(o5)==1
s=s.replace(o5, o5 + "\n    if(!is.null(fit) && is.null(fit$Vb2) && !is.null(fit$VB2)) fit$Vb2 = fit$VB2  # RA patch 5")
# RA patch 6: with bWGR 2.2.18 the spatial share can be NaN; route it to the script's own "odd results, use pheno"
o6="if(tmp<0|tmp>1){y = c(dta[,'YLD']);"
assert s.count(o6)==1
s=s.replace(o6, "if(is.na(tmp)|tmp<0|tmp>1){y = c(dta[,'YLD']);  # RA patch 6")
o4="EMD3X = cbind(EMD2,X_EC)"
assert s.count(o4)==1
s=s.replace(o4, """EigenGAU = function(X, phi = 1, cores = 1){  # RA patch 4 (double precision, same formula as bWGR)
  X = as.matrix(X); G = tcrossprod(X); dg = diag(G)
  D = sqrt(pmax(outer(dg, dg, '+') - 2 * G, 0)); diag(D) = 0
  n = nrow(X); exp(D * (phi * (-n * (n - 1)) / sum(D)))}
""" + o4)
open(p,'w').write(s)
open('clac_patch.txt','w').write('1. Model B: unseen fixed-effect levels of the target year -> coefficient 0\n2. marker sampling: min(35000, available); the 2024-release VCF has 2,425 markers\n3. (builder) environments without ECs dropped\n4. EigenGAU recomputed in double precision (same formula; single-precision NaN on identical EC rows)\n5. mkr2X output Vb2 aliased to VB2 (bWGR 2.2.18 rename)\n6. NaN spatial share -> the original use-pheno branch\n')
PY
sha256sum Script_CLAC_model.R $SRC > clac_script.sha256
date -Iseconds > clac_start.txt
$R Script_CLAC_model.R > clac_run.log 2>&1
date -Iseconds > clac_end.txt
echo CLAC_DONE $Y
