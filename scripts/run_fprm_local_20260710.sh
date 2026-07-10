#!/bin/bash
# fprm local runs, GPU-serialized (one process at a time on the 5090), 2026-07-10.
# Job A: fprm on the v2 breadth-mirror sweep (appends to results/local_breadth/sweep_runs.jsonl;
#        resume logic skips the 30 existing transformer/gdp_hybrid runs, summary re-aggregates all 3 archs).
# Job B: 3-arch chain_v1 local comparison at the canonical baseline recipe (d320x4, 8k steps),
#        seeds 0-2, train depths (2,3) / eval depths (4,5) from the registered spec.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] job A: fprm breadth sweep (B 6-24, 3 seeds, d256x4/8k) ==="
$PY scripts/experiment_local_breadth.py --archs fprm --tag sweep
echo "=== [$(date -u +%FT%TZ)] job A exit: $? ==="

echo "=== [$(date -u +%FT%TZ)] job B: chain_v1 x {fprm,transformer,gdp_hybrid} x seeds 0-2 (d320x4/8k) ==="
$PY scripts/sweep.py --tasks chain_v1 --archs fprm,transformer,gdp_hybrid --seeds 0 1 2 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 \
    --out_prefix results/local_chain_arch_20260710
echo "=== [$(date -u +%FT%TZ)] job B exit: $? ==="
echo "=== [$(date -u +%FT%TZ)] all done ==="
