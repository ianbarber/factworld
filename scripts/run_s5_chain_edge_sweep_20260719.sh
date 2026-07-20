#!/bin/bash
# s5_chain local calibration: walk k and depth down toward the settings where the
# components form locally (s5 forms at k=5; chain trains in-distribution at k=6),
# easiest cell first, to locate the operating point with dynamic range. Per-event
# checkpoint traces throughout; gdp_hybrid + fprm (the archs with formed state
# circuits); d320x4, seeds 0-2. Note the gated-item adversary floor is 1/(k-depth).
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

for cfg in "4 1" "5 1" "6 1" "4 2" "5 2" "6 2"; do
  set -- $cfg
  K=$1; D=$2
  echo "=== [$(date -u +%FT%TZ)] k=$K depth=$D ==="
  $PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm --seeds 0 1 2 \
      --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
      --k "$K" --chain_depth "$D" \
      --out_prefix "results/local_s5_chain_edge_k${K}d${D}_20260719"
  echo "=== [$(date -u +%FT%TZ)] k=$K depth=$D exit: $? ==="
done
echo "=== [$(date -u +%FT%TZ)] all done ==="
