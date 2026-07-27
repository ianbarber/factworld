#!/bin/bash
# (a) THE CELL WHERE A SEED ALREADY FORMED — k=6, depth 2, per-event map checkpoints.
#
# QUESTION
#   Is s5_chain formation at k=6/depth=2 a real bimodal mode or a single lucky seed?
#   results/local_s5_chain_edge_k6d2_20260719.jsonl has gdp_hybrid at 0.155 / 0.815 / 0.170
#   over seeds 0/1/2 at L4. Three seeds cannot distinguish "p(converge) ~ 1/3" from noise, and
#   the cell was published as "0.38+-0.31 (0%)" because p_converge tests >= 0.9 — a summary
#   under which a converged seed is invisible.
#
# DECISION RULE (pre-registered)
#   The floor for this cell is 0.200 at BOTH lengths: the max over the registered shallow
#   adversaries, recomputed from the exact scored items at eval_n=200. Here that max is
#   uniform-over-non-start (1/(k-1) = 0.200 under the distinct_path gate), not the initial-map
#   chase, which is worth only 0.195 at L4 and 0.160 at L8 — below even 1/k = 0.167. Read the
#   per-seed column, never the mean.
#     >= 3 of 8 seeds above 0.60 at L4  -> bimodal formation is real; report p(converge) at a
#                                          0.60 threshold with the per-seed values, and take
#                                          the k/depth grid to 8 seeds under this recipe.
#     1-2 of 8 seeds above 0.60         -> a rare mode. Report it as such; no null is claimable
#                                          at this budget and (d) is the next spend.
#     0 of 8 above 0.60 and every seed within +-0.05 of the floor
#                                       -> the cell sits at the shallow-adversary floor at this
#                                          budget. Still not a null for the TASK: (b) and (d)
#                                          change protocol and budget, not the task.
#   UNDERTRAINING CHECK (applies to every reading above). Each run records its training
#   loss curve, its held-out loss per eval length, and its epochs. A seed whose loss curve
#   is still falling over its final 20% of steps, or whose held-out loss exceeds its final
#   training loss by more than 0.5 nats/token, is undertrained: exclude it from the
#   reading rather than counting it as a floor.
#
# COST
#   16 runs (2 archs x 8 seeds), d320x4, 8000 steps. Measured 3.6 min/run for this cell
#   (logs/s5_chain_edge_sweep_20260719.log, 21.8 min for 6 runs) plus ~15% for the widened
#   trace-generation budget -> ~1.1 GPU-hours on the 5090.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] (a) k=6 depth=2 event_trace, 8 seeds ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm \
    --seeds 0 1 2 3 4 5 6 7 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
    --k 6 --chain_depth 2 \
    --out_prefix "results/rerun_s5_chain_a_k6d2_20260727"
rc=$?                        # capture first: a command substitution in the echo clobbers $?
echo "=== [$(date -u +%FT%TZ)] (a) exit: $rc ==="
