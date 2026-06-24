# Composite baseline (naive prompt)

Evaluated at 2026-06-24.
n = 30 per cell; L16; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

## Results

| model | composite_copy_v1 |
| --- | --- |
| llama-3.3-70b-instruct | 0.000 |
| gpt-4o-mini | 0.000 |
| qwen-2.5-7b-instruct | 0.000 |
| deepseek-chat | 0.000 |

All four models emit only the value token (or occasionally only the holder), so exact match on the two-token gold answer is 0%.


## Raw data

See `docs/openrouter-composite-baseline.json` for example-level predictions.
