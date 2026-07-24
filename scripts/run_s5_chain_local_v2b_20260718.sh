#!/bin/bash
# Rerun of the local trace-mode sweeps after the <eos> scoring fix (factworld/runner.py):
# pre-fix, trace-mode generation ran past the model's committed answer and last_n scored
# the budget-filling tail, reading every trace-mode sweep as chance. Same arms as
# run_s5_chain_local_v2_20260718.sh, plus the chain_v2 dense sweep and the commutative
# trace contingency whose published numbers came from the same path.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] arm B: s5_chain_local_v2 event-trace ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs fprm,transformer,gdp_hybrid --seeds 0 1 2 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
    --out_prefix results/local_s5_chain_v2_20260718
echo "=== [$(date -u +%FT%TZ)] arm B exit: $? ==="

echo "=== [$(date -u +%FT%TZ)] arm C: s5_chain_local_v2 depth-1 decomposition ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs fprm,transformer,gdp_hybrid --seeds 0 1 2 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace --chain_depth 1 \
    --out_prefix results/local_s5_chain_v2_d1_20260718
echo "=== [$(date -u +%FT%TZ)] arm C exit: $? ==="

echo "=== [$(date -u +%FT%TZ)] arm A: s5_chain_local_v2_path path-only contrast ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2_path --archs fprm,transformer,gdp_hybrid --seeds 0 1 2 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
    --out_prefix results/local_s5_chain_v2_path_20260718
echo "=== [$(date -u +%FT%TZ)] arm A exit: $? ==="

echo "=== [$(date -u +%FT%TZ)] arm D: chain_v2 dense rerun (published sweep was eos-artifact) ==="
$PY scripts/sweep.py --tasks chain_v2 --archs fprm,transformer,gdp_hybrid --seeds 0 1 2 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
    --out_prefix results/local_chain_v2_dense_20260718
echo "=== [$(date -u +%FT%TZ)] arm D exit: $? ==="

echo "=== [$(date -u +%FT%TZ)] arm E: commutative trace contingency rerun ==="
$PY scripts/experiment_commutative_local.py --use_trace --tag trace_rescored
echo "=== [$(date -u +%FT%TZ)] arm E exit: $? ==="
echo "=== [$(date -u +%FT%TZ)] all done ==="
