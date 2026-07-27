#!/bin/bash
# (b) THE s5 FORMATION PROTOCOL, ADAPTED TO SINGLE-TOKEN CHECKPOINTS — interleaved
#     checkpoints in TRAINING and GUIDED FREE-RUN eval with the events teacher-forced.
#
# QUESTION
#   Does s5_chain form under the protocol that formed s5 locally?
#   scripts/experiment_dense_supervision.py (docstring lines 10-13, e2e_eval) trains on streams
#   that interleave the oracle state after every event AND evaluates with the events forced and
#   the state slots GENERATED. scripts/sweep.py's --interleaved arm did the first half only: it
#   trained interleaved and then evaluated free-running on a checkpoint-free prompt. That is a
#   format shift, not a harder test — the model has never seen an event that is not followed by
#   a checkpoint.
#
#   TWO DIFFERENCES FROM THE REFERENCE, both from the slot shape. In s5 a checkpoint slot is a
#   span inside a sentence, so e2e_eval generates up to 4 tokens per slot, credits the first
#   one whose TYPE is valid (agent for a holder slot, value for a value slot), and resyncs at
#   the next ".". In s5_chain the interleaved document puts one bare id between an event and
#   the next step label, so sweep.guided_free_run_eval reads a single unconstrained argmax per
#   slot: no type filter, no resync.
#
#   At depth 1 the mismatch is total. The gold answer IS the last checkpoint, which sits
#   immediately before "what" in every training document with P=1.000, and the free-running
#   eval prompt deletes exactly that token. Five published cells were run that way
#   (run_s5_chain_interleaved_20260719.sh k4/k6/k8 depth 1; run_s5_chain_compact_20260723.sh
#   k4/k8 depth 1) and measure nothing. scripts/sweep.py refuses that combination without
#   --guided_eval; these are their replacements.
#
# DECISION RULE (pre-registered)
#   Two numbers per cell: `overall` (generated answer) and `checkpoint acc` (the model's own
#   per-event state tokens against the oracle trace). Read `overall` against the cell's
#   OPERATIVE floor: the max over the registered shallow adversaries, recomputed from the exact
#   eval_n=200 items. Which adversary supplies it changes cell by cell, so name it per cell —
#   the initial-map chase is the floor on only half of these:
#     cell        L4 floor                      L8 floor
#     k4 d1       0.335 chase                   0.333 uniform_non_start (chase 0.325)
#     k6 d1       0.275 chase                   0.200 uniform_non_start (chase 0.175)
#     k8 d1       0.335 chase                   0.210 chase
#     k6 d2       0.200 uniform_non_start       0.200 uniform_non_start (chase 0.195 / 0.160)
#   The compact-grammar cell shares the k8 d1 item stream exactly — compact_events changes only
#   the rendering, after every sampling draw — so it carries the same floors.
#
#   `checkpoint acc` is a LOWER BOUND on tracking. On-format it measures what the reference's
#   constrained scan measures; off-format — the model emits punctuation or a step label where
#   training put a bare id — it charges an error the reference would have skipped. It is
#   therefore thresholded at 0.95 only in the POSITIVE direction: a value at or above it is
#   unambiguous, a value below it is ambiguous between tracking and format. Every generated
#   checkpoint is written to <prefix>.preds.jsonl, so a cell between its floor and 0.95 gets
#   re-scored under the reference's type-constrained rule before it is called anything.
#     checkpoint acc >= 0.95 AND depth-1 overall >= 0.90 on >= 3 of 8 seeds
#         -> single-slot tracking forms under this protocol; the depth-2 cell in this same
#            script is then the composition read, with depth 1 as its positive control.
#     checkpoint acc >= 0.95 but overall at floor
#         -> the model tracks and cannot answer from what it tracked: a readout failure, and
#            the next probe is the query format, not scale.
#     checkpoint acc at chance (1/k) after the constrained re-score
#         -> tracking does not form at this width under this supervision; the depth-2 cell is
#            uninterpretable and (d) — budget and curriculum — is the only remaining move.
#     compact-grammar cell differs from the canonical-grammar cell at the same (k, depth) by
#     more than 0.15 -> the wordy rendering is load-bearing and that is the reportable result.
#
# COST
#   5 cells x 16 runs (2 archs x 8 seeds) = 80 runs, d320x4, 8000 steps. Measured 3.0 min/run
#   for interleaved cells (logs/s5_chain_interleaved_20260719.log, 17-19 min per 6-run cell).
#   Guided eval adds one uncached batch-1 forward per event per item on top of the answer
#   generation the free-running eval already did — 2,400 extra forwards per run at eval_n=200
#   over the two lengths — so budget 3.0-4.5 min/run, ~4.0-6.0 GPU-hours on the 5090.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

# canonical grammar: depth-1 tracking across k, then the depth-2 composition contrast at k=6.
for cfg in "4 1" "6 1" "8 1" "6 2"; do
  set -- $cfg
  K=$1; D=$2
  echo "=== [$(date -u +%FT%TZ)] (b) guided interleaved k=$K depth=$D ==="
  $PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm \
      --seeds 0 1 2 3 4 5 6 7 \
      --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 \
      --worked_trace --start_trace --interleaved --guided_eval \
      --k "$K" --chain_depth "$D" \
      --out_prefix "results/rerun_s5_chain_b_k${K}d${D}_20260727"
  rc=$?                      # capture first: a command substitution in the echo clobbers $?
  echo "=== [$(date -u +%FT%TZ)] (b) k=$K depth=$D exit: $rc ==="
done

# rendering contrast at the widest depth-1 cell: same stream and golds, s5-style compact events.
echo "=== [$(date -u +%FT%TZ)] (b) guided interleaved COMPACT k=8 depth=1 ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm \
    --seeds 0 1 2 3 4 5 6 7 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 \
    --worked_trace --start_trace --interleaved --guided_eval --compact_events \
    --k 8 --chain_depth 1 \
    --out_prefix "results/rerun_s5_chain_b_cmp_k8d1_20260727"
rc=$?
echo "=== [$(date -u +%FT%TZ)] (b) compact exit: $rc ==="
echo "=== [$(date -u +%FT%TZ)] (b) all done ==="
