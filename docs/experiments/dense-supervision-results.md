# Dense-vs-sparse state supervision — s5 / non-abelian composite (reproduction)

**Question:** is the `s5_v1` non-abelian state-tracking wall movable for our locally trained
models, and if so, what loosens it — supervision density, or inference-time compute?

This reproduces the load-bearing Phase-2 claim ([`phases/02-non-abelian-state/`](../../phases/02-non-abelian-state/))
on the current natural-language benchmark, then sweeps supervision sparsity and tests
inference-time scaling. Run: `scripts/experiment_dense_supervision.py`.

## Setup

`composite_copy_scale_v1` scaled to k=5 (the non-abelian composite: track a role through
S₅ swap/cycle events, then recall the holder's value). `gdp_hybrid` d256×4, 4000 steps, 3 seeds.
Training streams interleave the oracle's **holder-of-the-queried-role every K events**
(K=1 dense → large K = answer-only). Eval is **guided free-run**: events are teacher-forced,
the holder slots and final value are *generated* by the model, scored on the generated output.

## Sparsity cliff (reproduced on natural format)

| K (stride) | labels / 16-ep | value @L16 | value @L64 | conv(>0.5) |
| --- | --- | --- | --- | --- |
| 1 (dense) | 16 | **1.00**±0.00 | **0.90**±0.06 | 3/3 |
| 2 | 8 | 0.96±0.04 | 0.50±0.33 | 3/3 |
| 4 | 4 | 0.20±0.01 | 0.19±0.02 | 0/3 |
| 8 | 2 | 0.20±0.03 | 0.19±0.04 | 0/3 |

**The wall moves.** Dense K=1 supervision solves s5 (1.00 in-distribution, 0.90 @L64) where
answer-only and our main sweep floored at the 0.20 chance. The circuit forms only down to a
checkpoint every ~2 events; at K≥4 it does not form at all. This reproduces Phase 2's
near-dense cliff on the natural format — the mechanism is format-independent.

**Length extrapolation.** K=1 trained at lengths ≤16 reaches **L128 ≈ 0.90** (2 seeds) — roughly
8× the trained horizon, with no target-length labels. (Phase 2 reported the same via post-training
deep-state coverage; here it shows up directly from the dense-trained base.)

## Inference-time scaling probe — does test-time compute loosen the cliff?

Self-consistency / majority vote over N generated holder+value traces (temperature 0.7–0.8):

| regime | greedy | majority (5) | majority (15) | majority (30) |
| --- | --- | --- | --- | --- |
| K=2 seed0 @L64 (circuit formed) | 0.96 | 0.96 | 0.96 | 0.96 |
| K=2 seed1 @L64 (no circuit) | 0.29 | 0.30 | 0.29 | 0.29 |
| K=2 seed2 @L64 (no circuit) | 0.25 | 0.17 | 0.23 | 0.20 |
| K=1 @L128 (circuit formed, 8× horizon) | 0.88–0.90 | — | 0.88–0.90 | — |

**Inference-time compute does not move the wall in either direction.** Seeds that formed the
circuit are already greedy-optimal (voting is a no-op, even at 30 votes); seeds that didn't stay at
the floor no matter how many samples. The wall is in **learnability (supervision density)**, not in
test-time compute.

## Synthesis with the API autoregressive result

| lever | composition (s5/composite) |
| --- | --- |
| answer-only / sparse supervision | floor (every model, every architecture) |
| self-generated CoT / trained scratchpad | floor (compounds errors — see autoregressive-api-results.md) |
| self-consistency / majority vote (test-time compute) | floor — this doc |
| **dense per-step supervision** | **0.90–1.00, extrapolates 8×** — the only thing that moves it |
| oracle-provided intermediate (API scaffold) | unlocks recall leg (0.80–1.00) — but that's giving the answer |

**Headline:** both walls (composition routing, and non-abelian state-tracking) resist test-time
compute. The composition wall yields only to an *oracle-provided* intermediate; the s5 wall yields
only to *dense per-step supervision*. Neither is movable by asking the model to think harder at
inference — they are learnability/routing limits, not capacity limits.

## Files

- `scripts/experiment_dense_supervision.py` — K-sweep + guided-free-run eval + majority-vote probe.
- `results/dense_sweep_*.md` / `.jsonl` — the sparsity table + per-seed data.
- `phases/02-non-abelian-state/dense_capstone.py` — the original atomic-token reference (still runs).
