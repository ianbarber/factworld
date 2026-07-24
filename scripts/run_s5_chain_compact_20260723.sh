#!/bin/bash
# Issue #31 rendering ablation: s5-style compact event grammar vs the canonical wordy
# sentences, on the interleaved single-slot arms (the closest match to the protocol
# that formed s5). Streams and golds are byte-identical across renderings, so any
# formation difference is attributable to the grammar alone.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

for cfg in "4 1" "4 2" "8 1"; do
  set -- $cfg
  K=$1; D=$2
  echo "=== [$(date -u +%FT%TZ)] compact interleaved k=$K depth=$D ==="
  $PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm --seeds 0 1 2 \
      --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 \
      --worked_trace --start_trace --interleaved --compact_events \
      --k "$K" --chain_depth "$D" \
      --out_prefix "results/local_s5_chain_cmp_k${K}d${D}_20260723"
  echo "=== [$(date -u +%FT%TZ)] compact k=$K depth=$D exit: $? ==="
done
echo "=== [$(date -u +%FT%TZ)] all done ==="
