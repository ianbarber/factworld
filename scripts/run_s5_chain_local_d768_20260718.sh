#!/bin/bash
# s5_chain local capacity probe: d768x8 on the event-trace arms (the d320x4 battery
# reads chance on every arm post-scoring-fix). gdp_hybrid and fprm only — the two
# archs with any formed state circuit at smaller scale; depth 2 and depth 1.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] capacity: s5_chain_local_v2 event-trace d768x8 ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm --seeds 0 1 2 \
    --steps 8000 --d_model 768 --n_layers 8 --eval_n 200 --worked_trace \
    --out_prefix results/local_s5_chain_v2_d768_20260718
echo "=== [$(date -u +%FT%TZ)] capacity depth-2 exit: $? ==="

echo "=== [$(date -u +%FT%TZ)] capacity: depth-1 d768x8 ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm --seeds 0 1 2 \
    --steps 8000 --d_model 768 --n_layers 8 --eval_n 200 --worked_trace --chain_depth 1 \
    --out_prefix results/local_s5_chain_v2_d1_d768_20260718
echo "=== [$(date -u +%FT%TZ)] capacity depth-1 exit: $? ==="
echo "=== [$(date -u +%FT%TZ)] all done ==="
