# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-24T19:28:14.156735+00:00.
n = 20 examples per task/length; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

## Exact-match results

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 | s5_v1 |
| --- | --- | --- | --- | --- | --- | --- |
| llama-3.2-3b-instruct | 0.350 | 0.000 | 0.100 | 0.800 | 0.700 | 0.350 |
| llama-3.1-8b-instruct | 0.450 | 0.000 | 0.050 | 0.950 | 0.650 | 0.100 |
| qwen-2.5-7b-instruct | 0.300 | 0.250 | 0.050 | 1.000 | 0.950 | 0.100 |
| qwen3-32b | 0.200 | 0.050 | 0.100 | 0.550 | 0.400 | 0.200 |
| llama-3.3-70b-instruct | 0.500 | 0.100 | 0.800 | 1.000 | 1.000 | 0.150 |
| deepseek-chat | 0.300 | 0.200 | 0.000 | 1.000 | 1.000 | 0.300 |
| gpt-4o-mini | 0.350 | 0.200 | 0.000 | 1.000 | 1.000 | 0.150 |
| gemini-2.5-flash-lite | 0.350 | 0.200 | 0.050 | 1.000 | 1.000 | 0.300 |
| claude-3-haiku | 0.400 | 0.050 | 0.000 | 0.800 | 0.500 | 0.400 |

## Semantic containment results (tokenizer-robust)

Every non-punctuation token in the gold answer appears somewhere in the prediction. For `composite_copy_v1` this means both the holder and value are present; for single-token tasks it is equivalent to 'the correct token appears anywhere'.

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 | s5_v1 |
| --- | --- | --- | --- | --- | --- | --- |
| llama-3.2-3b-instruct | 0.350 | 0.000 | 0.100 | 0.800 | 0.700 | 0.350 |
| llama-3.1-8b-instruct | 0.450 | 0.000 | 0.050 | 0.950 | 0.650 | 0.100 |
| qwen-2.5-7b-instruct | 0.300 | 0.250 | 0.050 | 1.000 | 0.950 | 0.100 |
| qwen3-32b | 0.200 | 0.050 | 0.200 | 0.550 | 0.400 | 0.200 |
| llama-3.3-70b-instruct | 0.500 | 0.100 | 0.800 | 1.000 | 1.000 | 0.150 |
| deepseek-chat | 0.550 | 0.200 | 0.250 | 1.000 | 1.000 | 0.300 |
| gpt-4o-mini | 0.350 | 0.200 | 0.000 | 1.000 | 1.000 | 0.150 |
| gemini-2.5-flash-lite | 0.350 | 0.200 | 0.150 | 1.000 | 1.000 | 0.300 |
| claude-3-haiku | 0.400 | 0.050 | 0.300 | 0.800 | 0.500 | 0.400 |

## Notes

- Exact match is the canonical metric; semantic containment is reported to separate formatting/tokenizer artifacts from whether the model knows the answer.
- `APIBackend` normalizes common answer prefixes ('The answer is...') and a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`).


## Raw data

Per-task aggregates are in the tables above. Example-level predictions are in the accompanying JSON output (see `--json_out`).
