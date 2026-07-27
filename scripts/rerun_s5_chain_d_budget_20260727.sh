#!/bin/bash
# (d) BUDGET-MATCHED ARM — give s5_chain the recipe that made the composite converge.
#
# QUESTION
#   Is "s5_chain does not form locally" a statement about the task or about its budget?
#   Every s5_chain run to date: 8k documents, 8k steps, batch 32, d320x4 (d768x8 for one
#   capacity probe) — 32 epochs over 8k documents, 0.26M documents seen. The composite that
#   DID converge locally: 80k documents, 25k steps, batch 128, d768x8 — 3.2M documents seen,
#   12x the tokens — PLUS a staged curriculum (scripts/experiment_curriculum_staged.py).
#   s5_chain appears in no curriculum script at all. Two variables have never been moved.
#
#   Arm 1 moves the budget alone (same cold-start protocol, matched recipe).
#   Arm 2 adds the staged curriculum (scripts/experiment_s5_chain_curriculum.py): dereference
#   over a stated map, then tracking with a depth-1 readout, then the depth-2 composition,
#   each stage continuing from the previous model.
#
# DECISION RULE (pre-registered)
#   Per-seed values against the cell's OPERATIVE floor: the max over the registered shallow
#   adversaries, recomputed from the exact scored items. At k=6/depth=2 that is 0.200 at both
#   L4 and L8 — uniform-over-non-start, 1/(k-1) under the distinct_path gate — not the
#   initial-map chase, which is worth 0.195 at L4 and 0.160 at L8 at eval_n=200.
#     Arm 1 clears floor+0.25 on >= 2 of 3 seeds
#         -> the local null was a budget artifact. Retract it, re-run the k/depth grid at this
#            budget, and report the grid, not the null.
#     Arm 1 floors and Arm 2 clears
#         -> s5_chain needs its components trained first. That is a real and reportable
#            composition result — and the same result shape the composite already has.
#     Both floor, with held-out loss converged and the stage-1 chain arm at >= 0.9
#         -> the strongest local evidence available: the components train, the budget matches
#            the one recipe that produced local convergence, and the composition still does not
#            form. Only then is a local formation null claimable, and only with the curriculum
#            table alongside it.
#     Stage-1 chain arm below 0.9
#         -> the harness is not training at this width; nothing in this family is readable and
#            no null may be stated.
#
#   UNDERTRAINING CHECK (applies to every reading above). Each run records its training
#   loss curve, its held-out loss per eval length, and its epochs. A seed whose loss curve
#   is still falling over its final 20% of steps, or whose held-out loss exceeds its final
#   training loss by more than 0.5 nats/token, is undertrained: exclude it from the
#   reading rather than counting it as a floor.
#
# COST
#   The d768x8 / 25k-step / batch-128 / 80k-doc recipe measures 1.3-2.8 GPU-h per run
#   (scripts/remeasure_v2_issue11.sh:60 — 9 runs in 12-25 h). Pure compute scaling off the
#   measured 6.6 min/run at d768x8 / 8k steps / batch 32
#   (logs/s5_chain_local_d768_20260718.log) lands on the low end of that range; the high end is
#   what the recipe has actually cost in practice.
#   Arm 1: 6 runs (2 archs x 3 seeds)                                       ->  8-17 GPU-hours.
#   Arm 2: 6 runs at the same total step budget split across three stages,
#          plus a 3-arm evaluation at every stage (18 arm-evals per run
#          against 2 for a plain sweep run), which is on top of the range    ->  8-17 GPU-hours.
#   ~16-34 GPU-hours total. This is the expensive half of the plan: run (a), (b) and (c) first
#   and only start (d) if none of them has already settled the question.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] (d) arm 1: budget-matched cold start, k=6 depth=2 ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm --seeds 0 1 2 \
    --steps 25000 --batch 128 --d_model 768 --n_layers 8 --train_n 80000 --eval_n 200 \
    --worked_trace --k 6 --chain_depth 2 \
    --out_prefix "results/rerun_s5_chain_d_budget_k6d2_20260727"
rc=$?                        # capture first: a command substitution in the echo clobbers $?
echo "=== [$(date -u +%FT%TZ)] (d) arm 1 exit: $rc ==="

echo "=== [$(date -u +%FT%TZ)] (d) arm 2: staged curriculum at the same budget ==="
$PY scripts/experiment_s5_chain_curriculum.py --archs gdp_hybrid,fprm --seeds 0 1 2 \
    --steps 25000 --batch 128 --d_model 768 --n_layers 8 --train_n 80000 --eval_n 200 \
    --k 6 \
    --out_prefix "results/rerun_s5_chain_d_curriculum_20260727"
rc=$?
echo "=== [$(date -u +%FT%TZ)] (d) arm 2 exit: $rc ==="
echo "=== [$(date -u +%FT%TZ)] (d) all done ==="
