# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-24T17:06:08.118817+00:00.
n = 10 examples per task/length; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

## Exact-match results

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| kimi-k2 | 0.100 | 0.300 | 0.300 | 1.000 | 1.000 |
| deepseek-chat | 0.000 | 0.100 | 0.100 | 1.000 | 1.000 |
| gpt-4o-mini | 0.100 | 0.100 | 0.000 | 1.000 | 1.000 |
| llama-3.3-70b-instruct | 0.100 | 0.200 | 0.000 | 1.000 | 1.000 |
| qwen-2.5-7b-instruct | 0.300 | 0.400 | 0.000 | 1.000 | 1.000 |

## Semantic containment results (tokenizer-robust)

Every non-punctuation token in the gold answer appears somewhere in the prediction. For `composite_copy_v1` this means both the holder and value are present; for single-token tasks it is equivalent to 'the correct token appears anywhere'.

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| kimi-k2 | 0.700 | 0.300 | 0.300 | 1.000 | 1.000 |
| deepseek-chat | 1.000 | 0.100 | 0.400 | 1.000 | 1.000 |
| gpt-4o-mini | 1.000 | 0.100 | 0.000 | 1.000 | 1.000 |
| llama-3.3-70b-instruct | 1.000 | 0.200 | 0.600 | 1.000 | 1.000 |
| qwen-2.5-7b-instruct | 0.700 | 0.400 | 0.000 | 1.000 | 1.000 |

## Notes

- Exact match is the canonical metric; semantic containment is reported to separate formatting/tokenizer artifacts from whether the model knows the answer.
- `APIBackend` normalizes common answer prefixes ('The answer is...') and a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`).


## Raw data

Per-task aggregates are in the tables above. Example-level predictions are in the accompanying JSON output (see `--json_out`).
