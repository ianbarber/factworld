#!/bin/bash
# s5_chain local-formation rerun plan (issue #31). Cheapest and most discriminating first.
#
# WHAT THE PLAN IS FOR
#   The local "s5_chain never forms" result is confounded, not established. The eval path is
#   sound (a stub backend emitting the gold continuation scores 1.000 on every arm), but the
#   published cells were produced under a tokenizer that mapped 17-23% of every s5_chain
#   training token to <unk> — including "takes", "old", "values", "simultaneously:", "a0,"
#   and "a0.", i.e. the tokens that carry the pointer-update semantics. Those runs measured a
#   corrupted input.
#
#   The s5_chain event grammar was the only corrupted one. The compact-grammar s5_chain cells
#   lost 5.8% of their tokens and chain_v2 5.7%, in both cases confined to the "(N" / "hops)"
#   query annotation — which restates a hop count the repeated "a0 of" already carries — while
#   binding_v2, composite_copy_v2 and commutative_v1 were at 0%. (Rates measured by encoding
#   each cell's training documents under the pre-extension tokenizer, n=2000 documents.)
#
#   The tokenizer covers every canonical spec losslessly (tests/test_tokenizer.py), so the
#   family has to be re-measured before any local claim.
#
# ORDER AND GATING
#   (a) ~1.1     GPU-h  the one cell where a seed already formed, at 8 seeds
#   (b) ~4.0-6.0 GPU-h  the s5 formation protocol adapted to this task (guided free-run eval)
#   (c) ~1.9     GPU-h  the typed-value ablation at depth 1
#   (d) ~16-34   GPU-h  budget-matched cold start + staged curriculum
#   Total ~23-43 GPU-hours on one 5090. Stage (d) dominates the spread: the d768x8 / 25k-step /
#   batch-128 recipe measures 1.3-2.8 GPU-h per run (scripts/remeasure_v2_issue11.sh:60).
#
#   Stop early on a positive: if (a) shows bimodal formation on >= 3 of 8 seeds, the local null
#   is already retracted and (d) is not worth its 16-34 hours — spend that budget on the
#   k/depth grid at the recipe that worked instead. Each script's header states its own
#   decision rule, and every one of them reads scores against the cell's OPERATIVE floor: the
#   max over the registered shallow adversaries, not a single named row (factworld.validity).
#
# Run the whole plan:      bash scripts/rerun_s5_chain_plan_20260727.sh
# Run one stage:           bash scripts/rerun_s5_chain_a_seeds_20260727.sh
set -u
cd /home/ianbarber/Projects/factworld

for stage in a_seeds b_guided c_typed d_budget; do
  script="scripts/rerun_s5_chain_${stage}_20260727.sh"
  echo "=== [$(date -u +%FT%TZ)] PLAN: $script ==="
  bash "$script"
  rc=$?                      # capture first: a command substitution in the echo clobbers $?
  echo "=== [$(date -u +%FT%TZ)] PLAN: $script exit: $rc ==="
done
echo "=== [$(date -u +%FT%TZ)] plan complete ==="
