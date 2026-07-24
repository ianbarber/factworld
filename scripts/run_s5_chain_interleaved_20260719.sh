#!/bin/bash
# Interleaved single-slot supervision (the protocol that formed s5): checkpoint token
# follows each event inside the TRAINING stream; evaluation is free-running answer-only.
# d1 arms across k test single-slot tracking formation; k4 d2 is the joint-map contrast.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

for cfg in "4 1" "6 1" "8 1" "4 2"; do
  set -- $cfg
  K=$1; D=$2
  echo "=== [$(date -u +%FT%TZ)] interleaved k=$K depth=$D ==="
  $PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm --seeds 0 1 2 \
      --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace --start_trace --interleaved \
      --k "$K" --chain_depth "$D" \
      --out_prefix "results/local_s5_chain_int_k${K}d${D}_20260719"
  echo "=== [$(date -u +%FT%TZ)] interleaved k=$K depth=$D exit: $? ==="
done
echo "=== [$(date -u +%FT%TZ)] all done ==="
