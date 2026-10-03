#!/bin/bash
# CLAC decomposition chain on the 4090 (prereg docs/prereg_clac_decomposition_2026-10-02.md): rehearsal 2023 first,
# then the five target years; one year at a time, 2 BLAS threads, 40 GB memory cap. Existing ra/Y<Y>/ are read only.
set -u
BASE=<workstation>/dart-gxe/scratch_ideas_2026-09-29
R=<workstation>/dart-gxe/envs/r-reanalysis/bin/Rscript
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for Y in 2023 2016 2018 2020 2022 2024; do
  D=$BASE/ra_decomp/Y$Y; mkdir -p $D; cd $D
  for f in ES_dta_QC.RData AdjYld.RData K2X.RData ES_MVGBLUP1.RData AccESPS.RData gen.RData Testing_Data; do ln -sfn $BASE/ra/Y$Y/$f $f; done
  systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 $R $BASE/ra_decomp/code/ablate.R > decomp.log 2>&1
  if grep -q DECOMP_DONE decomp.log; then echo "Y$Y done" >> $BASE/ra_decomp/chain.status
  else echo "Y$Y FAILED" >> $BASE/ra_decomp/chain.status; [ $Y = 2023 ] && exit 1; fi
done
echo CHAIN_DONE >> $BASE/ra_decomp/chain.status
