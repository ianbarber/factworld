# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-26T21:50:59.939828+00:00.
n = 30 examples per task/length; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

## Exact-match results

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 | s5_v1 |
| --- | --- | --- | --- | --- | --- | --- |
| llama-3.2-3b-instruct | 0.267 | 0.000 | 0.100 | 0.900 | 0.833 | 0.233 |
| llama-3.1-8b-instruct | 0.433 | 0.033 | 0.000 | 0.800 | 0.700 | 0.100 |
| qwen-2.5-7b-instruct | 0.300 | 0.167 | 0.067 | 0.967 | 1.000 | 0.067 |
| qwen3-32b | 0.167 | 0.033 | 0.100 | 0.367 | 0.400 | 0.100 |
| llama-3.3-70b-instruct | 0.633 | 0.000 | 0.800 | 1.000 | 1.000 | 0.167 |
| deepseek-chat | 0.333 | 0.033 | 0.167 | 1.000 | 1.000 | 0.133 |
| gpt-4o-mini | 0.367 | 0.067 | 0.133 | 1.000 | 1.000 | 0.133 |
| gemini-2.5-flash-lite | 0.300 | 0.100 | 0.233 | 1.000 | 1.000 | 0.133 |
| claude-3-haiku | 0.433 | 0.100 | 0.100 | 0.767 | 0.600 | 0.300 |
| kimi-k2.6 | 0.633 | 0.033 | 0.400 | 1.000 | 1.000 | 0.200 |
| glm-5.2 | 0.767 | 0.133 | 0.800 | 1.000 | 1.000 | 0.167 |

## Semantic containment results (tokenizer-robust)

Every non-punctuation token in the gold answer appears somewhere in the prediction. For `composite_copy_v1` this means both the holder and value are present; for single-token tasks it is equivalent to 'the correct token appears anywhere'.

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 | s5_v1 |
| --- | --- | --- | --- | --- | --- | --- |
| llama-3.2-3b-instruct | 0.267 | 0.000 | 0.100 | 0.900 | 0.833 | 0.233 |
| llama-3.1-8b-instruct | 0.433 | 0.033 | 0.000 | 0.800 | 0.700 | 0.100 |
| qwen-2.5-7b-instruct | 0.300 | 0.167 | 0.067 | 0.967 | 1.000 | 0.067 |
| qwen3-32b | 0.167 | 0.033 | 0.233 | 0.367 | 0.400 | 0.100 |
| llama-3.3-70b-instruct | 0.633 | 0.000 | 0.800 | 1.000 | 1.000 | 0.167 |
| deepseek-chat | 0.533 | 0.067 | 0.367 | 1.000 | 1.000 | 0.133 |
| gpt-4o-mini | 0.367 | 0.067 | 0.133 | 1.000 | 1.000 | 0.133 |
| gemini-2.5-flash-lite | 0.300 | 0.100 | 0.300 | 1.000 | 1.000 | 0.133 |
| claude-3-haiku | 0.433 | 0.100 | 0.367 | 0.767 | 0.600 | 0.300 |
| kimi-k2.6 | 0.633 | 0.033 | 0.633 | 1.000 | 1.000 | 0.200 |
| glm-5.2 | 0.767 | 0.133 | 0.800 | 1.000 | 1.000 | 0.167 |

## Notes

- Exact match is the canonical metric; semantic containment is reported to separate formatting/tokenizer artifacts from whether the model knows the answer.
- `APIBackend` normalizes common answer prefixes ('The answer is...') and a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`).
- `chain_v1` and `s5_v1` cells here are measured without reasoning under a short completion budget. Reasoning-model cells require a large budget (`max_new_tokens=8192`, no early stop): with reasoning, glm-5.2 solves `chain_v1` at its designed depths (1.00 at depth 4, n=30; `results/chain_reasoning_pilot_20260705.json`) and s5 under a concrete rendering holds 0.90 at L128 (`results/s5_horizon_recheck_20260705.jsonl`).
- The `chain_v1` column above runs at depth 16. `chain_v1` builds a single k=6 pointer cycle and measures depth only at depths < k (`factworld/tasks.py`: "Depths stay < k so the cycle never wraps"); at depth 16 the chain wraps (effective difficulty depth mod 6), so the column is a wrapped-task floor check, not a depth score. Depth >= k belongs to the scaled no-wrap variant (`chain_nowrap`); the deeper cells in the pilot files (depths 6–32) likewise ran the wrapped task.


## Raw data

Per-task aggregates are in the tables above. Example-level predictions are in the accompanying JSON output (see `--json_out`).
