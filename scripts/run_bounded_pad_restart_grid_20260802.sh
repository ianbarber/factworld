#!/usr/bin/env bash
# The from-scratch three-cell grid under the bounded pad at w=2, with the registered
# stage4_restart folded in, and the composed cell's pad accuracy TRACKED ACROSS both the last
# curriculum stage and the restart. GPU only; no endpoint is contacted and nothing is paid for.
#
# The number this run exists for is composed slot_acc: the components sit at 0.99-1.00 while the
# composed cell's own bounded pad has never exceeded 0.85 and degrades during the stage, and until
# that reaches component levels with the answer still at floor there is no composition claim to
# make. Tracking runs at n=128 (which reproduces n=512 slot_acc to ~0.03 on the checkpoints that
# have both); the DECISIVE end-of-run read is at n=512.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/20260802_s5bind_v3_bounded_pad_restart_grid
LOG=logs/20260802_bounded_pad_restart_grid.log
mkdir -p logs results

.venv-train/bin/python -u scripts/experiment_s5bind_v3_bounded_pad_20260802.py --grid \
    --full_guided_grid \
    --archs gdp_hybrid --seeds 0 1 2 \
    --steps 25000 --format moved2 --pad_answer_docs --answer_ratio 1 \
    --restart_steps 3000 --restart_lr 3e-4 --restart_warmup 200 \
    --track_cells state@34,bind@62,composed@48,composed@64,composed@96 \
    --track_every 500 --track_every_curriculum 1750 --track_n 128 \
    --guided_n 128 --final_guided_n 512 --guided_batch 128 \
    --eval_n 1000 --loss_log_interval 500 \
    --out_prefix "$OUT" >> "$LOG" 2>&1
