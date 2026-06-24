# Composite format-prompt ablation

Evaluated at 2026-06-24.
n = 30 per cell; L16; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question. For questions that ask 'what is a0 of the holder of ...', answer with the holder's name followed by the requested value, like 'g3 v9'.

## Results

| model | composite_copy_v1 |
| --- | --- |
| llama-3.3-70b-instruct | 0.400 |
| gpt-4o-mini | 0.167 |
| qwen-2.5-7b-instruct | 0.000 |
| deepseek-chat | 0.500 |

Appending an explicit instruction to output `<holder> <value>` unlocks non-zero performance, confirming that the naive-prompt 0% is largely a format-instruction artifact.


## Raw data

See `docs/openrouter-composite-format.json` for example-level predictions.
