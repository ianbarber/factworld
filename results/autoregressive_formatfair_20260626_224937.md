# Autoregressive / test-time-compute experiment (API, E1)

n=100 per cell, max_new_tokens=256

## composite_copy_v1 @ L16
| model | binding (exact/holder/value) | none (exact/holder/value) | scaffolded (exact/holder/value) | structured (exact/holder/value) |
|---|---|---|---|---|
| deepseek/deepseek-chat | 0.60 / 0.60 / 0.00 | 0.12 / 0.15 / 0.13 | 0.99 / 1.00 / 0.99 | 0.00 / 0.18 / 0.00 |
| meta-llama/llama-3.3-70b-instruct | 0.34 / 0.34 / 0.00 | 0.63 / 0.64 / 0.63 | 0.93 / 1.00 / 0.93 | 0.00 / 0.14 / 0.00 |
| moonshotai/kimi-k2.6 | 0.99 / 0.99 / 0.00 | 0.00 / 0.97 / 0.97 | 1.00 / 1.00 / 1.00 | 0.00 / 0.99 / 0.00 |
| openai/gpt-4o-mini | 0.28 / 0.28 / 0.00 | 0.14 / 0.14 / 0.14 | 1.00 / 1.00 / 1.00 | 0.00 / 0.13 / 0.00 |
| z-ai/glm-5.2 | 0.98 / 0.98 / 0.00 | 0.00 / 0.75 / 0.75 | 1.00 / 1.00 / 1.00 | 0.00 / 0.95 / 0.02 |

_scaffolded = correct holder injected; the recall-leg ceiling for composition._
