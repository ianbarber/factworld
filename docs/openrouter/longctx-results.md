# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-28T06:51:02.179237+00:00.
n = 30 examples per task/length; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question. For questions that ask 'what is a0 of the holder of ...', answer with the holder's name followed by the requested value, like 'g3 v9'.

## Exact-match results

| model | composite_copy_v1 | s5_v1 |
| --- | --- | --- |
| llama-3.3-70b-instruct | 0.033 | 0.167 |
| kimi-k2.6 | 0.111 | 0.044 |
| glm-5.2 | 0.044 | 0.044 |

## Semantic containment results (tokenizer-robust)

Every non-punctuation token in the gold answer appears somewhere in the prediction. For `composite_copy_v1` this means both the holder and value are present; for single-token tasks it is equivalent to 'the correct token appears anywhere'.

| model | composite_copy_v1 | s5_v1 |
| --- | --- | --- |
| llama-3.3-70b-instruct | 0.033 | 0.167 |
| kimi-k2.6 | 0.111 | 0.044 |
| glm-5.2 | 0.044 | 0.044 |

## Notes

- Exact match is the canonical metric; semantic containment is reported to separate formatting/tokenizer artifacts from whether the model knows the answer.
- `APIBackend` normalizes common answer prefixes ('The answer is...') and a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`).


## Raw data

Per-task aggregates are in the tables above. Example-level predictions are in the accompanying JSON output (see `--json_out`).
