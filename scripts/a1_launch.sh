#!/bin/bash
# A1 GE-BiFormer runs (docs/prereg_sequel_wave2a_2026-09-28.md): 2 scenarios x 3 protocols x 3 seeds,
# SLOTS parallel runs per GPU. Usage on the 4090: DARTGXE_COMMIT=<sha> bash scripts/a1_launch.sh "0 1" 3
set -u
GPUS=${1:-"0 1"}; SLOTS=${2:-3}
cd "$(dirname "$0")/.."
LOG=${DARTGXE_ROOT:?}/logs/a1; mkdir -p "$LOG"
jobs=()
for seed in 42 1 2; do for sc in F2024m F2022m; do for pr in own fair own_clean; do jobs+=("$sc $pr $seed"); done; done; done
i=0
for g in $GPUS; do : > "$LOG/queue_gpu$g.txt"; done
ng=$(echo $GPUS | wc -w)
for j in "${jobs[@]}"; do g=$(echo $GPUS | cut -d' ' -f$(( i % ng + 1 ))); echo "$j" >> "$LOG/queue_gpu$g.txt"; i=$((i+1)); done
for g in $GPUS; do
  ( cat "$LOG/queue_gpu$g.txt" | xargs -P "$SLOTS" -L 1 bash -c \
      'out='"$LOG"'/$0_$1_seed$2.log; [ -f "${DARTGXE_RESULTS}/a1/runs/$0_$1_seed$2.json" ] && exit 0; CUDA_VISIBLE_DEVICES='"$g"' OMP_NUM_THREADS=4 ../.venv/bin/python scripts/a1_gebiformer.py --scenario $0 --protocol $1 --seed $2 > $out 2>&1 || echo FAILED $0 $1 $2 >> '"$LOG"'/failed.txt' ) &
done
wait
echo A1_ALL_DONE > "$LOG/status.txt"
