#!/usr/bin/env bash
# ATTACK THE COMPOSED PAD CAP, on the checkpoints the registered grid leaves behind. GPU only, $0.
#
# The grid's composed cell sits at floor with a pad accuracy the components reach trivially, and
# that pair of numbers is not yet a composition result: "the composition fails" and "the model
# cannot write the scratchpad the protocol requires" predict it equally. These four probes are the
# levers, each measured rather than argued.
#
#   counts      is the composed pad simply LONGER? (no model; tokenizer arithmetic)
#   forced      is the per-event update MISSING, or only the closed loop? (gold pad, one forward)
#   decompose   WHERE the free-running pad breaks: by event ordinal, by (event kind, block
#               position) — i.e. by HOP COUNT — and the ordinal of each item's first wrong slot
#   overwrite   is the pad being traded away by ANSWER supervision, and does composed-only
#               training close it? Equal-step continuations under three mixes.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:-results/20260802_s5bind_v3_bounded_pad_restart_grid_ckpt}
SEEDS=${2:-0 1 2}
PY=.venv-train/bin/python
LOG=logs/20260802_composed_pad_attack.log
mkdir -p logs results

LADDER=composed@16,composed@32,composed@48,composed@64,composed@96
CTRL=state@34,bind@62

$PY -u scripts/probe_s5bind_v3_composed_pad_20260802.py --counts \
    --n_count 200 --out results/20260802_composed_pad_counts.json >> "$LOG" 2>&1

$PY -u scripts/probe_s5bind_v3_composed_pad_20260802.py --forced \
    --ckpt_dir "$CKPT" --seeds $SEEDS --cells "$CTRL,$LADDER" --n 512 \
    --out results/20260802_composed_pad_forced.json >> "$LOG" 2>&1

$PY -u scripts/probe_s5bind_v3_composed_pad_20260802.py --decompose \
    --ckpt_dir "$CKPT" --seeds $SEEDS --cells "$CTRL,$LADDER" --n 512 \
    --out results/20260802_composed_pad_decompose.json >> "$LOG" 2>&1

# The mix comparison reads composed@48 only: it is an EQUAL-BUDGET contrast between three arms,
# so one cell resolves it and the n=512 read at L=96 costs three times what it buys here.
$PY -u scripts/probe_s5bind_v3_composed_pad_20260802.py --overwrite \
    --ckpt_dir "$CKPT" --seeds $SEEDS --cells composed@48 --n 512 \
    --steps 600 --lr 3e-4 --warmup 200 \
    --out results/20260802_composed_pad_overwrite.json >> "$LOG" 2>&1

# The composed-only arm at the RESTART's own budget: if 3,000 steps of nothing but the composed
# cell does not move the pad, more curriculum weight on that cell is not the lever either.
$PY -u scripts/probe_s5bind_v3_composed_pad_20260802.py --overwrite --mixes composed_only \
    --ckpt_dir "$CKPT" --seeds $SEEDS --cells composed@48,composed@96 --n 512 \
    --steps 3000 --lr 3e-4 --warmup 200 \
    --out results/20260802_composed_pad_composed_only_3000.json >> "$LOG" 2>&1
