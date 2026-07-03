# Autoregressive / test-time-compute experiment (API, E1)

n=30 per cell, max_new_tokens=256

## composite_copy_v1 @ L16
| model | binding (exact/holder/value) | none (exact/holder/value) | scaffolded (exact/holder/value) | structured (exact/holder/value) |
|---|---|---|---|---|
| z-ai/glm-5.2 | 0.97 / 0.97 / 0.00 | 0.00 / 0.00 / 0.00 | 0.97 / 1.00 / 0.97 | 0.00 / 1.00 / 0.00 |

_scaffolded = correct holder injected; the recall-leg ceiling for composition._

## s5_v1 @ L32
| model | binding (exact/holder/value) | none (exact/holder/value) | scaffolded (exact/holder/value) | structured (exact/holder/value) |
|---|---|---|---|---|
| z-ai/glm-5.2 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

_scaffolded = correct holder injected; the recall-leg ceiling for composition._
