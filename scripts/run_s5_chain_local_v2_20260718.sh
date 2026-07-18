#!/bin/bash
# s5_chain local-signal probes, GPU-serialized on the 5090, 2026-07-18.
# Arm B — dense per-EVENT supervision (the lever that formed s5 locally): the
#         s5_chain_local_v2 trace prefixes a full a0-map checkpoint after every
#         event, then the query path. The retired local_v1 pilot supervised only
#         the query path, so "dense traces do not help" was never actually tested.
# Arm C — depth-1 decomposition: same spec at chain_depth=1 (apply the final map
#         once). If depth-1 forms where depth-2 floors, the wall is the serial
#         dereference ON TOP of tracked state, not the state tracking itself.
# Arm A — supervision-density contrast: gated items with the path-only trace
#         (the local_v1 protocol on the clean v2 stream).
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] arm B: s5_chain_local_v2 event-trace x {fprm,transformer,gdp_hybrid} x seeds 0-2 ==="
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
echo "=== [$(date -u +%FT%TZ)] all done ==="
