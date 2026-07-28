#!/bin/bash
# (c2) THE TYPED-VALUE ABLATION, OUT OF THE MEMORISATION REGIME.
#
# QUESTION
#   In s5 the roles r0..r4 are a different token type from the agents g0..g4, so a checkpoint
#   token is unambiguously a VALUE. In s5_chain the pointer values ARE agents, so every id in an
#   event is ambiguous between the slot being written and the value being moved, and a
#   from-scratch model must recover the distinction from syntax alone. s5 forms under dense
#   supervision; s5_chain does not (stage (b): checkpoint accuracy at 1/k for k in {4,6,8} at
#   depth 1, where there is no composition to fail at). This pair changes the value TYPE and
#   nothing else.
#
# WHY THIS RERUN EXISTS
#   Stage (c) ran both arms at the sweep default train_n=8000 and is therefore unreadable: all
#   32 runs fail the pre-registered undertraining check, held-out minus train loss +0.54 to
#   +1.42 nats/token against a 0.5 bar (results/rerun_s5_chain_c_{typed,untyped}_k8d1_*.jsonl).
#   That is the same confound stage (a) hit and stage (a2) removed — the (c) script was written
#   before (a2) measured it. train_n changes the document POOL, not the gradient-step count
#   (8,000 steps x batch 32 draws 256,000 samples either way), so ten times the data is 3.2
#   epochs instead of 32 at identical GPU cost.
#
# DECISION RULE (pre-registered)
#   Apply the undertraining check first; a run whose held-out minus train loss exceeds 0.5 is
#   excluded from the reading. Floors are recomputed per cell from the exact scored items and
#   differ between the arms (the typed arm's map is a bijection to a distinct type, so the
#   initial-map chase means something different there) — read each arm against its own.
#     typed clears its floor and untyped does not
#       -> key/value token ambiguity is the blocker, not composition. The local rung needs a
#          typed construct, and the price table's open row is answered.
#     both at their floors
#       -> ambiguity is eliminated alongside rendering (stage (b)'s compact arm). The remaining
#          candidate is budget/curriculum, i.e. stage (d), and after (d) the honest statement is
#          that this construct does not form at this scale under any protocol tried.
#     untyped clears and typed does not
#       -> the typed spec is not the control it is meant to be; inspect before concluding.
#
# COST
#   2 cells x 16 runs, d320x4, 8,000 steps, batch 32 — identical GPU cost to stage (c),
#   ~1.9 GPU-hours on the 5090, plus CPU to generate the larger pools.
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] (c2) TYPED values, k=8 depth=1, train_n=80000 ==="
$PY scripts/sweep.py --tasks s5_chain_typed_v1 --archs gdp_hybrid,fprm \
    --seeds 0 1 2 3 4 5 6 7 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
    --train_n 80000 \
    --out_prefix "results/rerun_s5_chain_c2_typed_k8d1_20260728"
rc=$?
echo "=== [$(date -u +%FT%TZ)] (c2) typed exit: $rc ==="

echo "=== [$(date -u +%FT%TZ)] (c2) UNTYPED control, k=8 depth=1, train_n=80000 ==="
$PY scripts/sweep.py --tasks s5_chain_local_v2 --archs gdp_hybrid,fprm \
    --seeds 0 1 2 3 4 5 6 7 \
    --steps 8000 --d_model 320 --n_layers 4 --eval_n 200 --worked_trace \
    --train_n 80000 \
    --k 8 --chain_depth 1 \
    --out_prefix "results/rerun_s5_chain_c2_untyped_k8d1_20260728"
rc=$?
echo "=== [$(date -u +%FT%TZ)] (c2) untyped exit: $rc ==="
