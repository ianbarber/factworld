# Reasoning on/off/levels sweep — does background test-time compute help?

Direct dose-response: reasoning effort swept {none,low,medium,high} on the reasoning models.
n=100 (composite) / n=40-100 (s5). last_n = committed-answer accuracy.

## composite_copy_v1 @ L16 (value accuracy)

| model | none | low | medium | high |
|---|---|---|---|---|
| kimi-k2.6 | 0.22 | 0.94 | 0.98 | 0.96 |
| glm-5.2 | 0.14 | 0.74 | 0.78 | 0.81 |

## s5_v1 @ L32 (value accuracy)

| model | none | low | medium | high |
|---|---|---|---|---|
| kimi-k2.6 | 0.00 | 0.00 | 0.00 | 0.00 |
| glm-5.2 | 0.00 | 0.00 | 0.00 | - |

**Finding — background reasoning (test-time compute) helps composition but NOT s5.**
- **composite**: clear monotonic dose-response — kimi 0.22→0.96→0.98→0.96, glm 0.14→0.74→0.78→0.81.
  Background reasoning is a real test-time-compute lever for composition (the earlier 'test-time doesn't
  help' claim was WRONG — it applied only to explicit CoT prompting and to local non-reasoning models).
- **s5**: stays at/near floor regardless of effort (kimi 0.00→0.00→0.00→0.00 value; glm 0.00→0.00→0.00).
  Non-abelian state-tracking is NOT movable by reasoning — it needs dense supervision (see dense-supervision-results.md).

This sharpens the dissociation: **composition is movable by test-time compute (reasoning) for strong
models; s5 is movable only by training-time supervision density.** What does NOT help either wall:
explicit structured CoT prompting (hurts), and sampling/self-correction on non-reasoning local models.
