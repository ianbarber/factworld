#!/bin/bash
# DOES THE COMPOSITION EXTRAPOLATE IN LENGTH?
#
# QUESTION
#   composite_copy_v2 trains on lengths (4, 8, 16) and defines held-out cells at (16, 32, 64).
#   The staged-curriculum flagship evaluates composite_p16_L16 and nothing longer, so the one
#   architecture that composes locally — gdp_hybrid at 0.758 / 0.782 / 0.958 over three seeds —
#   has never been asked whether it still composes out of distribution.
#
#   This is the sharpest form of the composition question available, for a reason that the
#   frontier arc established the hard way. In the thinking regime a model may emit L intermediate
#   tokens, which buys sequential depth and lifts the single-pass circuit bound, so no fixed-size
#   construct stays hard: L, depth and breadth all read flat, and coupling two structures only
#   buys extra passes over a prompt that never leaves context. Length EXTRAPOLATION is immune to
#   that, because a length-specific shortcut is length-specific by construction. A from-scratch
#   model has no scratchpad, so its composition is non-serialisable to begin with.
#
#   The component answer is already known and is selective: under dense supervision gdp_hybrid
#   holds S5 at 0.75 at 4x the trained length while fprm reads 0.17 and the transformer 0.22
#   (report Table 1) — Liu et al.'s shortcut result reproduced. The composition has no such row.
#
# WHAT IS MEASURED
#   The flagship recipe unchanged (staged curriculum, d768x8, batch 128, 25k steps, 80k docs),
#   evaluated at L16 (in distribution), L32 (2x) and L64 (4x), with the binding and recall legs
#   at the same lengths so the composed cell is read against its own components rather than
#   against the L16 components. Three seeds; per-seed values reported, never a mean, because this
#   family is bimodal at the emergence threshold.
#
# READING IT
#   Read every cell against its own floor. The object-filter floor decays roughly as 1/L
#   (~0.41 at L16, ~0.15 at L64 for composite_copy_v2 per the TaskSpec note), so a composed score
#   that falls with L is only informative relative to a floor that falls faster.
#     composition holds at 2x and 4x        -> the local composition is a circuit, not a
#                                              length-specific fit, and the price table's open
#                                              row can be closed.
#     composition falls to its floor while the BINDING leg holds
#                                           -> the composition is a length-specific shortcut even
#                                              though the component is not. That is a result about
#                                              composition that the thinking regime cannot produce.
#     both fall together                    -> the deficit is the component, not the composition;
#                                              report it as component extrapolation.
#
# COST
#   3 seeds x 1 arch at the flagship recipe. scripts/remeasure_v2_issue11.sh records 1.3-2.8
#   GPU-h per run for d768x8 / 25k steps / batch 128, so ~4-8 GPU-hours. gdp_hybrid only: fprm
#   (0.109) and the transformer (0.001) do not compose in distribution, so their extrapolation
#   cells would read floor-to-floor and buy nothing. Add them only if gdp_hybrid holds.
#
#   REQUIRES THE GPU: stop the vLLM server first (it holds ~29.5 GiB).
set -u
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python

echo "=== [$(date -u +%FT%TZ)] composition extrapolation: gdp_hybrid, L16/32/64 ==="
$PY scripts/experiment_curriculum_staged.py \
    --archs gdp_hybrid --seeds 0 1 2 \
    --d_model 768 --n_layers 8 --batch 128 --steps 25000 --train_n 80000 \
    --eval_n 500 --extrap_lengths 32 64 \
    --out_prefix "results/composition_extrapolation_20260729"
rc=$?
echo "=== [$(date -u +%FT%TZ)] exit: $rc ==="
