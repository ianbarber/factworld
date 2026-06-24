# FactWorld OpenRouter Model Grid — `s5_v1`

Evaluated at 2026-06-24T04:55:55.229749+00:00.
n = 30 examples per task/length; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question. For 'what role does ... have?' questions, answer with only a role token (r0, r1, r2, r3, or r4) followed by a period. Example: 'r2 .'

Task: `s5_v1` — S₅ role-permutation state-tracking (experimental). A sequence of `swap`/
`cycle_roles` events permutes which agent holds each of five roles. The query asks for the role of
a single agent at the end. The answer is one of five role tokens, so the random-guess floor is
**0.20**.

## Per-length exact-match results

| model | L32 | L64 | L128 | mean |
| --- | --- | --- | --- | --- |
| nemotron-3-ultra-550b-a55b | 0.233 | 0.167 | 0.267 | **0.222** |
| kimi-k2 | 0.067 | 0.200 | 0.233 | **0.167** |
| kimi-k2.5 | 0.200 | 0.233 | 0.267 | **0.233** |
| kimi-k2.6 | 0.133 | 0.233 | 0.200 | **0.189** |
| deepseek-chat | 0.200 | 0.067 | 0.100 | **0.122** |
| llama-3.3-70b-instruct | 0.300 | 0.167 | 0.067 | **0.178** |
| gpt-4o-mini | 0.100 | 0.167 | 0.167 | **0.144** |

## Relaxed results (whitespace / period invariant)

Relaxed scoring strips spaces and trailing periods. For this task it is identical to exact match
because the canonical answer is already a single token plus a period.

| model | L32 | L64 | L128 | mean |
| --- | --- | --- | --- | --- |
| nemotron-3-ultra-550b-a55b | 0.233 | 0.167 | 0.267 | **0.222** |
| kimi-k2 | 0.067 | 0.200 | 0.233 | **0.167** |
| kimi-k2.5 | 0.200 | 0.233 | 0.267 | **0.233** |
| kimi-k2.6 | 0.133 | 0.233 | 0.200 | **0.189** |
| deepseek-chat | 0.200 | 0.067 | 0.100 | **0.122** |
| llama-3.3-70b-instruct | 0.300 | 0.167 | 0.067 | **0.178** |
| gpt-4o-mini | 0.100 | 0.167 | 0.167 | **0.144** |

## What the numbers show

- **Every strong pretrained model is at the chance floor.** `s5_v1` has five possible role tokens,
  so random guessing scores 0.20. No model clears that floor reliably across lengths; the best
  per-cell score is Llama 3.3 70B at L32 (0.300), and it decays to 0.067 by L128.
- **Format instructions are necessary but not sufficient.** Without the role-token instruction,
  chat models echo the agent name (`g0`) instead of the role. With it they emit the right token
  *shape*, but they cannot track the running S₅ permutation.
- **This matches the custom-trained learnability map.** The same `s5_v1` task is solvable by a
  small recurrent model when trained with dense per-step process supervision (see
  `docs/state-tracking-results.md` and `followups/non-abelian-state/non-abelian-state.md`). The
  pretrained models are not failing because the task is unexpressible; they are failing because
  they were trained on sparse outcome-level text and have no reliable latent state-update circuit
  for this computation.

## Notes

- `APIBackend` normalizes a trailing period glued to the preceding token (e.g. `r2.` → `r2 .`).
- Relaxed scoring strips spaces and trailing periods; exact match remains the canonical metric.
- The strongest composite model in the grid, Nemotron 3 Ultra (0.767 on `composite_copy_v1`), is
  not stronger than the others here — `s5_v1` state-tracking is a distinct bottleneck.

## Raw data

Per-task aggregates are in the tables above. Example-level predictions are in
`docs/openrouter-s5-results.json`.
