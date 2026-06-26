# Autoregressive / test-time-compute experiment (API, E1)

n=30 per cell, max_new_tokens=96

## composite_copy_v1 @ L16
| model | none (exact/holder/value) | scaffolded (exact/holder/value) | structured (exact/holder/value) |
|---|---|---|---|
| deepseek/deepseek-chat | 0.00 / 0.13 / 0.00 | 0.80 / 1.00 / 0.80 | 0.00 / 0.27 / 0.00 |
| google/gemini-2.5-flash-lite | 0.00 / 0.10 / 0.00 | 1.00 / 1.00 / 1.00 | 0.00 / 0.17 / 0.00 |
| meta-llama/llama-3.3-70b-instruct | 0.00 / 0.00 / 0.00 | 0.93 / 1.00 / 0.93 | 0.00 / 0.23 / 0.00 |
| openai/gpt-4o-mini | 0.00 / 0.00 / 0.00 | 0.97 / 1.00 / 0.97 | 0.00 / 0.10 / 0.00 |

_scaffolded = correct holder injected; the recall-leg ceiling for composition._

## s5_v1 @ L32
| model | none (exact/holder/value) | scaffolded (exact/holder/value) | structured (exact/holder/value) |
|---|---|---|---|
| deepseek/deepseek-chat | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| google/gemini-2.5-flash-lite | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| meta-llama/llama-3.3-70b-instruct | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| openai/gpt-4o-mini | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

_scaffolded = correct holder injected; the recall-leg ceiling for composition._
