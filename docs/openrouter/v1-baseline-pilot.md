# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-24T16:53:36.879472+00:00.
n = 10 examples per task/length; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

## Exact-match results

| model | composite_copy_v1 | s5_v1 |
| --- | --- | --- |
| kimi-k2 | 0.000 | 0.000 |
| deepseek-chat | 0.000 | 0.000 |
| gpt-4o-mini | 0.000 | 0.000 |

## Semantic containment results (tokenizer-robust)

Every non-punctuation token in the gold answer appears somewhere in the prediction. For `composite_copy_v1` this means both the holder and value are present; for single-token tasks it is equivalent to 'the correct token appears anywhere'.

| model | composite_copy_v1 | s5_v1 |
| --- | --- | --- |
| kimi-k2 | 0.000 | 0.000 |
| deepseek-chat | 0.000 | 0.000 |
| gpt-4o-mini | 0.000 | 0.000 |

## Notes

- Exact match is the canonical metric; semantic containment is reported to separate formatting/tokenizer artifacts from whether the model knows the answer.
- `APIBackend` normalizes common answer prefixes ('The answer is...') and a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`).


## Raw data

Per-task aggregates are in the tables above. Example-level predictions are in the accompanying JSON output (see `--json_out`).
