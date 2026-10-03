#!/bin/bash
# Stage 0 closed-form baselines on the 4090 (GPU given by $GPU, default 1). Idempotent per scenario.
cd "$(dirname "$0")/.."
export DARTGXE_ROOT=${DARTGXE_ROOT:-<workstation>/dart-gxe} OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-1}
PY=../.venv/bin/python
for sc in FYrep2023 FYrep2021 FYrep2019 FYnew2022 FYnew2020 FYnew2018 F2022 F2022m CV0 CV00 CV1 CV1parent; do
  if [ -f "$DARTGXE_ROOT/results/predictions/g2f/$sc/B2_mxe/meta.json" ]; then echo "skip $sc"; continue; fi
  $PY scripts/run_baselines.py --scenarios $sc || echo "FAILED $sc"
done
echo ALL_DONE
