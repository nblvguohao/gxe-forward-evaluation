#!/bin/bash
# RA run chain on the 4090 (one heavy job at a time; prereg memory cap 48 GB for Lopez-Cruz; 2 BLAS threads).
cd <workstation>/dart-gxe/scratch_ideas_2026-09-29/ra
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
R=<workstation>/dart-gxe/envs/r-reanalysis/bin/Rscript
run_cap() { systemd-run --user --scope -q -p MemoryMax=$1 -p MemorySwapMax=0 /usr/bin/time -v "${@:2}"; }
what=$1
case "$what" in
  lc)
    for spec in "2022 real" "2024 real" "2024 hist"; do read -r y m <<< "$spec"
      run_cap 48G $R run_lc.R $y $m fit 0.9 > lc/fit_${y}_${m}.log 2>&1; done ;;
  clac)
    for Y in 2023 2016 2018 2020 2022 2024; do
      run_cap 40G bash ./run_clac.sh $Y > clac_$Y.log 2>&1
      grep -q CLAC_DONE clac_$Y.log || { echo "CLAC $Y failed" >> chain_clac.status; [ $Y = 2023 ] && exit 1; }
      echo "CLAC $Y done" >> chain_clac.status; done ;;
esac
echo CHAIN_DONE $what
