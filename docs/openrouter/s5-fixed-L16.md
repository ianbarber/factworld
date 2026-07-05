# FactWorld OpenRouter Model Grid

Evaluated at 2026-07-05T01:28:41.036645+00:00.
n = 30 examples per task/length; relaxed match is the canonical metric (exact / semantic-containment / last-N are reported below as diagnostics).

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question. For questions that ask 'what is a0 of the holder of ...', answer with the holder's name followed by the requested value, like 'g3 v9'. For 'what role does ... have?' questions, answer with only a role token (r0, r1, r2, r3, or r4) followed by a period. Example: 'r2 .'

## Relaxed-match results (canonical)

Strip a trailing period and score the first `len(gold)` tokens. This is the fair cross-regime metric: it handles reasoning scratchpads (`APIBackend` drops `<think>` blocks and common prefixes) and models that omit or glue the trailing period.

| model | s5_v1 |
| --- | --- |
| glm-5.2 | 0.200 |
| kimi-k2.6 | 0.200 |
| llama-3.3-70b-instruct | 0.200 |
| gemini-2.5-flash-lite | 0.167 |
| deepseek-chat | 0.133 |
| gpt-4o-mini | 0.200 |
| claude-3-haiku | 0.167 |

## Exact-match results (diagnostic)

Position-strict token-for-token match. Under-counts models that omit or glue the trailing period, so read this as a formatting artifact check, not the score.

| model | s5_v1 |
| --- | --- |
| glm-5.2 | 0.200 |
| kimi-k2.6 | 0.200 |
| llama-3.3-70b-instruct | 0.200 |
| gemini-2.5-flash-lite | 0.167 |
| deepseek-chat | 0.133 |
| gpt-4o-mini | 0.200 |
| claude-3-haiku | 0.167 |

## Semantic containment results (tokenizer-robust)

Every non-punctuation token in the gold answer appears somewhere in the prediction. For `composite_copy_v1` this means both the holder and value are present; for single-token tasks it is equivalent to 'the correct token appears anywhere'.

| model | s5_v1 |
| --- | --- |
| glm-5.2 | 0.200 |
| kimi-k2.6 | 0.200 |
| llama-3.3-70b-instruct | 0.200 |
| gemini-2.5-flash-lite | 0.167 |
| deepseek-chat | 0.133 |
| gpt-4o-mini | 0.200 |
| claude-3-haiku | 0.167 |

## Last-N results (answer-extraction robust)

Match the last len(gold) tokens of the prediction to the gold answer. This is the fair metric for reasoning/chat models that may emit a scratchpad or list intermediate holders before the final answer, and for runs without `stop_at='.'` where the model does not emit a trailing period.

| model | s5_v1 |
| --- | --- |
| glm-5.2 | 0.200 |
| kimi-k2.6 | 0.200 |
| llama-3.3-70b-instruct | 0.200 |
| gemini-2.5-flash-lite | 0.167 |
| deepseek-chat | 0.133 |
| gpt-4o-mini | 0.200 |
| claude-3-haiku | 0.167 |

## Notes

- Relaxed match is the canonical metric (see `factworld.tasks.CANONICAL_METRIC`). It strips a trailing period and scores the first `len(gold)` tokens, so it is fair across local models (which may append extra tokens after the answer) and API models (which often omit or glue the trailing period).
- Exact match is a diagnostic: it under-counts any model that does not emit the answer span verbatim, so it can read 0 even when the relaxed score is high.
- For API reasoning models run without `stop_at='.'`, `last_n` is a useful cross-check: it matches the last `len(gold)` tokens, tolerating a scratchpad before the final answer.
- Semantic containment reports whether the correct holder and value appear anywhere in the output.
- `APIBackend` normalizes common answer prefixes ('The answer is...') and a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`).


## Raw data

Per-task aggregates are in the tables above. Example-level predictions are in the accompanying JSON output (see `--json_out`).
