#!/bin/bash
# (c) THE TYPED-VALUE ABLATION — does key/value type ambiguity carry the s5_chain null?
#
# QUESTION
#   s5 forms locally; s5_chain does not. One structural difference between them is not
#   composition depth at all: in s5 the tracked values (roles r0..r4) are a DIFFERENT token
#   type from the slots holding them (agents g0..g4), while in s5_chain the pointer values ARE
#   agents. Every agent token in an s5_chain stream is therefore ambiguous between "slot being
#   written" and "value being moved", and a from-scratch model has to resolve that from syntax
#   alone — a pretrained model gets it from English. CANONICAL["s5_chain_typed_v1"] is
#   s5_chain_local_v2 at depth 1 with the a0 map sending agents to ROLES: same event grammar,
#   same event distribution, same supervision, same k, same lengths.
#
#   Both arms are depth 1, so this is a readout contrast with composition held out of it.
#
# WHAT ELSE DIFFERS (the contrast is three changes, not one)
#   1. VALUE TYPE — the intended change. Slots stay agents; values become roles.
#   2. INITIAL-MAP STRUCTURE. The untyped map is a single k-cycle over the k sampled agents,
#      so it has no fixed points and every agent is reachable from every other. Typed values
#      are drawn from a disjoint pool, so the map is a uniform random agent->role bijection and
#      "cycle" is not defined for it. This is forced by the type split, not a free choice, and
#      it moves the initial-map-chase floor.
#   3. distinct_path. The untyped arm carries it (tasks.CANONICAL["s5_chain_local_v2"]); the
#      typed arm does not. At depth 1 the gate restricts the queried start to a non-fixed-point
#      of the FINAL map and, in the rare case where no such start exists, resamples the whole
#      event stream. The typed arm needs no gate — a role can never equal an agent, so the echo
#      adversary scores 0 by construction — but the consequence is that the two arms have
#      different chance levels: 1/(k-1) = 0.143 untyped against 1/k = 0.125 typed.
#   Read the arms against their OWN floors, never against each other's.
#
# DECISION RULE (pre-registered)
#   Per-seed values against each cell's OPERATIVE floor: the max over the registered shallow
#   adversaries, recomputed from the exact eval_n=200 items the cell scores.
#     cell              chase   uniform(_non_start)   operative
#     typed    @L4      0.320   0.125                 0.320  (chase)
#     typed    @L8      0.100   0.125                 0.125  (uniform — the chase is BELOW it)
#     untyped  @L4      0.335   0.143                 0.335  (chase)
#     untyped  @L8      0.210   0.143                 0.210  (chase)
#   Those rows are the adversary's score on the SAME items the model is scored on, so the
#   within-cell comparison is paired and the sampling noise largely cancels. They are NOT
#   stable estimates of a population quantity: the typed L8 chase is 0.100 over these 200
#   items and 0.144 over 5000 (block-of-200 spread 0.10-0.19). Any comparison ACROSS cells —
#   "the typed floor is lower than the untyped floor" — has to use the large-sample values
#   (typed 0.273 @L4 / 0.144 @L8, untyped 0.298 @L4 / 0.186 @L8 at n=5000), not the rows.
#     typed clears floor+0.25 on >= 3 of 8 seeds, untyped does not
#         -> key/value type ambiguity is the binding constraint at depth 1. The s5_chain null
#            is about representation, and the informative follow-up is a typed depth-2
#            construct, not more scale.
#     both clear
#         -> depth 1 is not where s5_chain fails; the null belongs to composition depth and
#            (b)'s depth-2 cell is the experiment.
#     neither clears
#         -> the depth-1 readout does not form at this budget at all, so no depth-2 result in
#            this family is interpretable until (d) lands.
#     untyped clears and typed does not
#         -> the ablation is confounded; inspect the saved predictions
#            (results/*.preds.jsonl) before reading anything else in the family.
#     either arm lands within 0.05 of its floor at L8 only
#         -> re-read that cell against the large-sample floor before calling it a floor; the
#            L8 rows are the noisiest in this pair.
#
# COST
#   2 cells x 16 runs (2 archs x 8 seeds) = 32 runs, d320x4, 8000 steps, ~3.6 min/run
#   -> ~1.9 GPU-hours on the 5090.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] (c) TYPED values, k=8 depth=1 ==="
$PY scripts/sweep.py --tasks s5_chain_typed_v1 --archs gdp_hybrid,fprm \
    --seeds 0 1 2 3 4 5 6 7 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
    --out_prefix "results/rerun_s5_chain_c_typed_k8d1_20260727"
rc=$?                        # capture first: a command substitution in the echo clobbers $?
echo "=== [$(date -u +%FT%TZ)] (c) typed exit: $rc ==="

echo "=== [$(date -u +%FT%TZ)] (c) UNTYPED control, k=8 depth=1 ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm \
    --seeds 0 1 2 3 4 5 6 7 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
    --k 8 --chain_depth 1 \
    --out_prefix "results/rerun_s5_chain_c_untyped_k8d1_20260727"
rc=$?
echo "=== [$(date -u +%FT%TZ)] (c) untyped exit: $rc ==="
echo "=== [$(date -u +%FT%TZ)] (c) all done ==="
