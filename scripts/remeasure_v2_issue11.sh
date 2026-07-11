#!/usr/bin/env bash
# Issue #11 v2 re-measure runbook (branch remeasure/v2-numbers, prepared 2026-07-10).
#
# DOCUMENTED INVOCATIONS ONLY — nothing here runs automatically. Execute stages by hand
# in the order below (API first, then the GPU queue, one detached job at a time).
# Budget: ~$8-13 API of the $35 ceiling; ~2.5-4 GPU-days serial on the RTX 5090.
#
# Superseded items (do NOT re-run; see the issue #11 triage comment):
#   - d256 binding sweep (fprm 0.94@L64): superseded by the 45-run v2 breadth sweep
#     (results/local_breadth/sweep_summary.md).
#   - glm §7-API long-context: superseded by results/composite_frontier_20260709.jsonl
#     (0.94-0.98 to L1024). Only kimi needs the v2 confirm (stage A3).
#   - §7 LOCAL long-context: swap/cycle streams (experiment_dense_supervision.py), not the
#     give-stream — the recency defect never touched it; prose fix only.
#   - archived OpenRouter natural grids / scale.md: historical snapshots by design.
#   - MOPD binding domain: deferred with a confound note (experiments/mopd/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "This is a runbook: read the stages and run them by hand." && exit 0

# ============================ DAY 1 — API (story order) ==============================
# Stage A1 [story-critical]: E1b scaffolded leg on composite_copy_v2 — the missing third
# leg of the gap definition's decomposition (binding_only@L16 + composed@L16/L64 already
# exist for all 9 roster models in bench_v2_zb2_20260709). Implemented as the zero_budget
# (16, "scaffolded") facet cell (factworld/benchmark.py + run_frontier_benchmark.py):
# resume-keyed, renders into docs/benchmark/results.md. All other zero_budget cells
# already have history records, so resume skips them and ONLY the scaffolded leg runs.
# ~900 calls, ~$1.5-3, <30 min.
set -a; source .env; set +a
.venv-api/bin/python scripts/run_frontier_benchmark.py --dry-run          # verify plan: 1 new cell/model
.venv-api/bin/python scripts/run_frontier_benchmark.py --facets zero_budget
# Fallback (zero benchmark plumbing, not resume-keyed):
# .venv-api/bin/python scripts/experiment_autoregressive.py \
#     --tasks composite_copy_v2 --conditions scaffolded --composite_format --n 100 \
#     --models anthropic/claude-opus-4.8 anthropic/claude-sonnet-5 openai/gpt-5.5 \
#              google/gemini-3.5-flash moonshotai/kimi-k2.6 qwen/qwen3.7-max \
#              z-ai/glm-5.2 deepseek/deepseek-v4-pro nvidia/nemotron-3-ultra-550b-a55b

# Stage A2 [flagship]: reasoning dose-response on v2 (kimi 0.22->0.96 / glm 0.14->0.81
# were v1 cells; endpoints already re-established on v2, this buys the monotone curve).
# Defaults already v2 (REASONING_MODELS = kimi + glm, EFFORTS none/low/medium/high,
# 8192-token budget); 2 models x 4 efforts x 50 = 400 calls at L16, ~$2-4.
.venv-api/bin/python scripts/experiment_reasoning.py --tasks composite_copy_v2 --n 50

# Stage A3 [flagship]: §7 API long-context on v2 — KIMI ONLY (glm half superseded).
# Mirror the glm v2 protocol (results/composite_frontier_20260709.jsonl): thinking arm
# only, per-length budgets (run L256 then L512 with the doubled budget), n=25/cell,
# publish empty rates per AGENTS. Kimi is token-hungry: ~$2-6 for the two cells.
.venv-api/bin/python scripts/experiment_composite_frontier.py \
    --model moonshotai/kimi-k2.6 --lengths 256 --arm thinking --budget 16384 --n 25
.venv-api/bin/python scripts/experiment_composite_frontier.py \
    --model moonshotai/kimi-k2.6 --lengths 512 --arm thinking --budget 32768 --n 25

# Stage A4 [commutative calibration, cap $2]: frontier floors for the new rung.
# 6 cells (instant L16/L64 + thinking L64; thinking@L16 NOT bought), glm + deepseek, n=25.
.venv-api/bin/python scripts/experiment_commutative_frontier.py

# ====================== GPU QUEUE (one detached job at a time) =======================
# Night 1 [story-critical]: §5 staged-curriculum flagship on v2 (specs ported in
# experiment_curriculum_staged.py: binding_v2 / composite_copy_v2, 2026-07-10).
# Read p(converge) over >=3 seeds + per-leg (holder/value) decomposition, not the mean.
# 9 runs d768x8, 25k steps: ~12-25 h.
nohup .venv-train/bin/python scripts/experiment_curriculum_staged.py \
    --archs gdp_hybrid,fprm,transformer --seeds 0 1 2 \
    --d_model 768 --n_layers 8 --batch 128 --train_n 80000 --eval_n 500 \
    --schedule 'binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.25,composite_p16:0.35:7500' \
    --use_trace --out_prefix results/curriculum_staged_v2_d768 \
    > results/curriculum_staged_v2_d768.log 2>&1 &

# Day 2 [flagship]: reference baselines -> docs/results.md (script already CANONICAL/v2;
# the committed doc is stale). ~4-8 h. Then ci_dissociation -> docs/results-ci.md, ~4-8 h.
nohup .venv-train/bin/python scripts/collect_baselines.py > results/collect_baselines_v2.log 2>&1 &
nohup .venv-train/bin/python scripts/ci_dissociation.py > results/ci_dissociation_v2.log 2>&1 &

# Evening slot [commutative calibration]: local 3-arch x 3-seed sweep, d256x4, ~1.5-3 h.
nohup .venv-train/bin/python scripts/experiment_commutative_local.py \
    > results/commutative_local/sweep.log 2>&1 &

# Nights 2-3 [flagship]: compute-matched scale sweep on v2 (inherits the staged_specs
# port via experiment_composite_scale.py). Medium first if the envelope tightens;
# consider 3 seeds at large (documented bimodality). ~1-2 GPU-days.
nohup .venv-train/bin/python scripts/experiment_composite_scale.py \
    > results/composite_scale_v2.log 2>&1 &
