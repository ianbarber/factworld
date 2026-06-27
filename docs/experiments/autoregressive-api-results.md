# Autoregressive / test-time-compute experiment (API, E1)

48 cells. Scoring: last_n + holder/value decomposition; scaffolded = recall-leg ceiling (correct holder injected); binding = binding-leg isolation (ask only for the holder).

## composite_copy_v1 @ L16
| model | binding (last_n / holder / value) | none (last_n / holder / value) | scaffolded (last_n / holder / value) | structured (last_n / holder / value) |
|---|---|---|---|---|
| deepseek-chat | 0.47 / 0.47 / 0.00 | 0.00 / 0.10 / 0.00 | 0.80 / 1.00 / 0.80 | 0.37 / 0.37 / 0.00 |
| gemini-2.5-flash-lite | 0.37 / 0.37 / 0.00 | 0.00 / 0.10 / 0.00 | 1.00 / 1.00 / 1.00 | 0.10 / 0.17 / 0.00 |
| llama-3.3-70b-instruct | 0.40 / 0.40 / 0.00 | 0.00 / 0.07 / 0.00 | 0.97 / 1.00 / 0.97 | 0.13 / 0.20 / 0.00 |
| kimi-k2.6 | 0.73 / 0.73 / 0.00 | 0.00 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 | 0.60 / 0.60 / 0.00 |
| gpt-4o-mini | 0.37 / 0.37 / 0.00 | 0.00 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 | 0.10 / 0.10 / 0.00 |
| glm-5.2 | 0.97 / 0.97 / 0.00 | 0.00 / 0.00 / 0.00 | 0.97 / 1.00 / 0.97 | 1.00 / 1.00 / 0.00 |

## s5_v1 @ L32
| model | binding (last_n / holder / value) | none (last_n / holder / value) | scaffolded (last_n / holder / value) | structured (last_n / holder / value) |
|---|---|---|---|---|
| deepseek-chat | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.13 / 0.00 / 0.00 |
| gemini-2.5-flash-lite | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| llama-3.3-70b-instruct | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.03 / 0.03 / 0.00 |
| kimi-k2.6 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| gpt-4o-mini | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| glm-5.2 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
