# Autoregressive / test-time-compute — API results (E1 + E1b)

Two API experiments: **E1** (no-format leg-isolation, the decomposition) and **E1b** (format-fair
re-run, the honest comparison). Both n=30–100, composite_copy_v1@L16 + s5_v1@L32. Scoring:
`last_n` + holder/value decomposition. `scripts/experiment_autoregressive.py`.

## E1b — Format-fair (the headline, n=100, output-format instruction given)

All conditions carry the composite format instruction, so the only variable is the reasoning regime.

`composite_copy_v1 @ L16` — value accuracy (binding column = holder accuracy):

| model | none | structured | binding (holder) | scaffolded (value) |
| --- | --- | --- | --- | --- |
| **kimi-k2.6** | **0.97** | 0.00 | 0.99 | 1.00 |
| **glm-5.2** | **0.75** | 0.02 | 0.98 | 1.00 |
| llama-3.3-70b | 0.63 | 0.00 | 0.34 | 0.93 |
| deepseek-chat | 0.13 | 0.00 | 0.60 | 0.99 |
| gpt-4o-mini | 0.14 | 0.00 | 0.28 | 1.00 |

**This is the corrected picture.**

1. **Reasoning models (kimi/glm) solve composition directly** with the format instruction (kimi 0.97,
   glm 0.75) — no scaffold needed. Their built-in test-time reasoning resolves the binding leg
   (binding-only: kimi 0.99, glm 0.98) and routes it into recall. The "composition wall" is really a
   **reasoning-model advantage**: non-reasoners (deepseek 0.13, gpt-4o 0.14) stay at the wall.
2. **Structured CoT actively HURTS** — value drops to ~0.00 for every model under the explicit
   `holder:` prompt, including the reasoners that scored 0.75–0.97 under plain `none`. Forcing an
   explicit intermediate disrupts models that reason better implicitly.
3. **The recall ceiling is universal** (scaffolded 0.93–1.00): given the holder, everyone recalls.

## E1 — No-format leg-isolation (the decomposition, n=30)

`none`/`structured` here carry **no** output-format instruction (a no-format lower bound). Useful for
the *relative* decomposition (binding/scaffolded isolate the legs); the absolute `none`/`structured`
scores are not comparable to the format-fair grid.

`composite_copy_v1 @ L16` (last_n / holder / value):

| model | binding | none | scaffolded | structured |
| --- | --- | --- | --- | --- |
| llama-3.3-70b | 0.40 / 0.40 / 0.00 | 0.00 / 0.07 / 0.00 | 0.97 / 1.00 / 0.97 | 0.13 / 0.20 / 0.00 |
| deepseek-chat | 0.47 / 0.47 / 0.00 | 0.00 / 0.10 / 0.00 | 0.80 / 1.00 / 0.80 | 0.37 / 0.37 / 0.00 |
| gpt-4o-mini | 0.37 / 0.37 / 0.00 | 0.00 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 | 0.10 / 0.10 / 0.00 |
| gemini-2.5-flash | 0.37 / 0.37 / 0.00 | 0.00 / 0.10 / 0.00 | 1.00 / 1.00 / 1.00 | 0.10 / 0.17 / 0.00 |
| kimi-k2.6 | 0.73 / 0.73 / 0.00 | 0.00 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 | 0.60 / 0.60 / 0.00 |
| glm-5.2 | 0.97 / 0.97 / 0.00 | 0.00 / 0.00 / 0.00 | 0.97 / 1.00 / 0.97 | 1.00 / 1.00 / 0.00 |

`s5_v1 @ L32`: every model floors (0.00) under every condition, including scaffolded — s5 is a
genuine non-abelian state-tracking wall with no decoupled single leg (see dense-supervision-results.md
for what does move it).

## Synthesis

- **Composition is gated by implicit reasoning + format.** Strong reasoners (kimi/glm) clear it given
  the output format; non-reasoners don't; explicit structured CoT hurts everyone. The recall leg is
  never the bottleneck (scaffolded 0.93–1.00).
- **Explicit self-produced intermediates don't help** — they hurt. This matches the local E2 result
  (a trained scratchpad collapses the holder leg) and the local self-correction probe (iterative
  refinement gives zero lift). Implicit reasoning works; explicit step-by-step does not.
- **s5 is a different, deeper wall** — floored even for reasoners, movable only by dense supervision.

## Files

- `results/autoregressive_formatfair_*.jsonl` — E1b (format-fair, n=100).
- `results/autoregressive_api_*.jsonl`, `results/autoregressive_glm_*.jsonl` — E1 (no-format, n=30).
- `scripts/experiment_autoregressive.py` (`--composite_format` for E1b).
