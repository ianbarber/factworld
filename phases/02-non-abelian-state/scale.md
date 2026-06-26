# Scale ladder — does the internalisation/horizon wall soften with capacity? (5.7M -> 357M)

`followups/non-abelian-state/scale.py`. gdp_hybrid, mixed-density training, parametric. Free-running e2e composite. **Answer-only L64** is the headline (the dissociation/horizon wall: 0/5 floor at 6M in curriculum.py). Floor = 0.20. Three phases: (1) base ladder 5.7M/18.5M/44.8M lr 1e-3, 6000 steps, 3 seeds; (2) LR control 44.8M/70M x lr {1e-3, 5e-4}, 8000 steps, 2 seeds; (3) push 140M/268M/357M lr 1e-3, bs32, 8000 steps, 2 seeds.

## Ladder (headline answer-only L64)

| scale | answer-only L16 | answer-only L64 | dense (scratchpad) L64 |
|---|---|---|---|
| 5.7M | 0.67±0.24(2/3) | 0.23±0.04(0/3) | 0.45±0.38(1/3) |
| 18.5M | 0.99±0.01(3/3) | 0.20±0.02(0/3) | 0.71±0.36(2/3) |
| 44.8M | 0.95±0.07(3/3) | 0.22±0.05(0/3) | 0.68±0.39(2/3) |
| 140M | 1.00±0.00 | 0.42±0.02 | — |
| 268M | 0.81±0.18 | 0.21±0.03 | — |
| 357M | 1.00±0.00 | 0.20±0.03 | — |

The wall does not soften: answer-only L64 floors (~0.20) across 5.7M->357M, with only a weak, non-monotonic bump at 70M/140M (0.41/0.42) that vanishes again by 268M/357M. Capacity gives a bounded, non-solving bump; extrapolation needs length, not width.

## LR control (phase 2 — tuned LR + 70M ceiling, answer-only)

| scale | lr | answer-only L16 | answer-only L64 |
|---|---|---|---|
| 44.8M | 0.001 | 0.98±0.02 | 0.24±0.04 |
| 44.8M | 0.0005 | 0.79±0.21 | 0.19±0.05 |
| 70M | 0.001 | 1.00±0.00 | 0.41±0.05 |
| 70M | 0.0005 | 0.75±0.24 | 0.24±0.09 |

The L64 wall is not relieved by LR-tuning either: best tuned point (70M, lr 1e-3) reaches only 0.41 — not solving — firming "learnability, not capacity" within 3090 reach.
