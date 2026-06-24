# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-24T04:28:28.882047+00:00.
n = 30 examples per task/length; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

## Exact-match results

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| kimi-k2 | 0.900 | 0.300 | 0.733 | 1.000 | 1.000 |
| kimi-k2.5 | 0.800 | 0.300 | 0.633 | 1.000 | 1.000 |
| kimi-k2.6 | 0.867 | 0.133 | 0.567 | 1.000 | 1.000 |

## Relaxed results (whitespace / period invariant)

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| kimi-k2 | 0.900 | 0.300 | 0.733 | 1.000 | 1.000 |
| kimi-k2.5 | 0.800 | 0.300 | 0.633 | 1.000 | 1.000 |
| kimi-k2.6 | 0.867 | 0.133 | 0.567 | 1.000 | 1.000 |

## Notes

- `APIBackend` normalizes a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`).
- Relaxed scoring strips spaces and trailing periods; exact match remains the canonical metric.


## Raw data

Per-task aggregates are in the tables above. Example-level predictions are in the accompanying JSON output (see `--json_out`).
