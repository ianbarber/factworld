# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-24T04:17:02.784779+00:00.
n = 30 examples per task/length; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

## Exact-match results

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| nemotron-3-nano-30b-a3b | 0.700 | 0.033 | 0.233 | 0.933 | 0.867 |
| nemotron-3-super-120b-a12b | 0.700 | 0.000 | 0.200 | 0.967 | 0.767 |
| nemotron-3-ultra-550b-a55b | 0.733 | 0.000 | 0.767 | 1.000 | 1.000 |
| llama-3.3-nemotron-super-49b-v1.5 | 0.600 | 0.067 | 0.267 | 1.000 | 0.867 |

## Relaxed results (whitespace / period invariant)

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| nemotron-3-nano-30b-a3b | 0.700 | 0.033 | 0.233 | 0.933 | 0.867 |
| nemotron-3-super-120b-a12b | 0.700 | 0.000 | 0.200 | 0.967 | 0.767 |
| nemotron-3-ultra-550b-a55b | 0.733 | 0.000 | 0.767 | 1.000 | 1.000 |
| llama-3.3-nemotron-super-49b-v1.5 | 0.600 | 0.067 | 0.267 | 1.000 | 0.867 |

## Notes

- `APIBackend` normalizes a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`).
- Relaxed scoring strips spaces and trailing periods; exact match remains the canonical metric.


## Raw data

Per-task aggregates are in the tables above. Example-level predictions are in the accompanying JSON output (see `--json_out`).
