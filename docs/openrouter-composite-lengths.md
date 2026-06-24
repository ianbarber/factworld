# Composite length sweep (format prompt)

Evaluated at 2026-06-24.
n = 30 per cell; lengths 16, 32, 64; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question. For questions that ask 'what is a0 of the holder of ...', answer with the holder's name followed by the requested value, like 'g3 v9'.

## Results

| model | composite_copy_v1 |
| --- | --- |
| llama-3.3-70b-instruct | 0.211 |
| gpt-4o-mini | 0.156 |
| qwen-2.5-7b-instruct | 0.033 |
| deepseek-chat | 0.389 |

With the format instruction, accuracy degrades as the binding history lengthens, consistent with the instrument's OOD-length behavior.


## Raw data

See `docs/openrouter-composite-lengths.json` for example-level predictions.
