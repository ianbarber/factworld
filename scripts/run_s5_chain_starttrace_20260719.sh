#!/bin/bash
# s5-shaped supervision probe: single-slot checkpoints (a0(start) after each event),
# the exact supervision shape that formed s5, replacing the full-map dump that did not.
# d1 arms test single-slot tracking formation across k; the k4 d2 arm is the contrast —
# depth 2 needs a slot the trace never supervises, so d1-forms/d2-floors isolates
# joint-map maintenance as the unbought item.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

for cfg in "4 1" "6 1" "8 1" "4 2"; do
  set -- $cfg
  K=$1; D=$2
  echo "=== [$(date -u +%FT%TZ)] start_trace k=$K depth=$D ==="
  $PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm --seeds 0 1 2 \
      --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace --start_trace \
      --k "$K" --chain_depth "$D" \
      --out_prefix "results/local_s5_chain_stt_k${K}d${D}_20260719"
  echo "=== [$(date -u +%FT%TZ)] start_trace k=$K depth=$D exit: $? ==="
done
echo "=== [$(date -u +%FT%TZ)] all done ==="
