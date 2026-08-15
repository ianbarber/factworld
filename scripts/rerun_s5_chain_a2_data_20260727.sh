#!/bin/bash
# (a2) THE SAME CELL WITH TEN TIMES THE DATA AND THE SAME COMPUTE.
#
# QUESTION
#   Stage (a) ran k=6/depth=2 at 8 seeds and every gdp_hybrid seed read 0.160-0.235 against a
#   0.200 floor. It is not readable: all 8 fail the pre-registered undertraining check, held-out
#   minus train loss +0.527 to +0.994 nats/token against a 0.5 bar
#   (results/rerun_s5_chain_a_k6d2_20260727.jsonl). The loss curves are flat over their final
#   20% of steps (-0.0004 to -0.004 per 100 steps), so the deficit is DATA, not steps: 8,000
#   documents seen 32 times.
#
#   train_n changes the document POOL, not the number of gradient steps: 8,000 steps x batch 32
#   draws 256,000 samples either way. Ten times the pool is therefore 3.2 epochs instead of 32
#   at the same GPU cost, and it isolates memorisation from capacity — the one confound that
#   makes stage (a) unreadable — before spending the 8-17 GPU-hours stage (d) needs.
#
# DECISION RULE (pre-registered)
#   Floor is 0.200 at both lengths (max over the registered shallow adversaries at eval_n=200:
#   uniform-over-non-start under the distinct_path gate; the initial-map chase is 0.195 at L4).
#   Read the per-seed column, never the mean, and apply the undertraining check first.
#     held-out minus train loss now <= 0.5 on most seeds
#       -> the cell is READABLE. Then:
#          >= 3 of 8 seeds above 0.60 at L4  -> formation is real; report p(converge) at 0.60
#                                               with the per-seed values and take the k/depth
#                                               grid to 8 seeds under this recipe.
#          1-2 of 8 above 0.60               -> a rare mode; report as such.
#          0 of 8 and every seed within +-0.05 of the floor
#                                            -> the cell sits at the floor with the memorisation
#                                               confound removed. That is the first readable
#                                               null in this battery, and (b) guided eval and
#                                               (d) curriculum are what it licenses next.
#     held-out minus train loss still > 0.5 on most seeds
#       -> data volume is not the binding constraint at this capacity; go to (d), which changes
#          batch, steps and curriculum together, and do not read (a) or (a2) as a null.
#
# COST
#   16 runs (2 archs x 8 seeds), d320x4, 8,000 steps, batch 32 — identical GPU cost to stage (a),
#   ~1.1 GPU-hours on the 5090, plus a few minutes of CPU to generate the larger pool.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] (a2) k=6 depth=2 event_trace, 8 seeds, train_n=80000 ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm \
    --seeds 0 1 2 3 4 5 6 7 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
    --train_n 80000 \
    --k 6 --chain_depth 2 \
    --out_prefix "results/rerun_s5_chain_a2_k6d2_data_20260727"
rc=$?
echo "=== [$(date -u +%FT%TZ)] (a2) exit: $rc ==="
