#!/usr/bin/env bash
# GPU queue for the issue-#11 v2 re-measures (branch remeasure/v2-numbers, 2026-07-10).
# One job at a time on the RTX 5090, story order, quick items first, flagship last.
# Each job is crash-isolated: a failure logs and the chain continues.
# Logs: results/remeasure_v2/<job>.log ; status ledger: results/remeasure_v2/queue_status.log
set -u
cd "$(dirname "$0")/.."
PY=.venv-train/bin/python
LOG=results/remeasure_v2
mkdir -p "$LOG" results/commutative_local
STATUS="$LOG/queue_status.log"

run_job () {
    local name="$1"; shift
    echo "[$(date -u +%FT%TZ)] START $name" >> "$STATUS"
    "$PY" "$@" > "$LOG/$name.log" 2>&1
    local rc=$?
    echo "[$(date -u +%FT%TZ)] DONE  $name rc=$rc" >> "$STATUS"
}

# 1. Commutative rung local calibration: 3 archs x 3 seeds d256x4, ~1.5-3 h.
run_job commutative_local scripts/experiment_commutative_local.py

# 2. Reference baselines -> docs/results.md on v2, ~4-8 h.
run_job collect_baselines scripts/collect_baselines.py

# 3. CI dissociation -> docs/results-ci.md, ~4-8 h.
run_job ci_dissociation scripts/ci_dissociation.py

# 4. FLAGSHIP (overnight): section-5 staged-curriculum re-measure on v2 specs,
#    3 archs x 3 seeds d768x8, ~12-25 h. Read p(converge) + per-leg decomposition.
run_job curriculum_staged_v2 scripts/experiment_curriculum_staged.py \
    --archs gdp_hybrid,fprm,transformer --seeds 0 1 2 \
    --d_model 768 --n_layers 8 --batch 128 --train_n 80000 --eval_n 500 \
    --schedule 'binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.25,composite_p16:0.35:7500' \
    --use_trace --out_prefix results/curriculum_staged_v2_d768

# 5. Nights 2-3: compute-matched scale sweep on v2 (inherits staged_specs port), ~1-2 GPU-days.
run_job composite_scale_v2 scripts/experiment_composite_scale.py

echo "[$(date -u +%FT%TZ)] QUEUE COMPLETE" >> "$STATUS"
