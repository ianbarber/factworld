# Reasoning on/off/levels sweep — does background test-time compute help?

The confound test. Reasoning models solved composite with reasoning *on* — that IS test-time
compute. This sweeps reasoning effort {none, low, medium, high} to measure the dose-response
directly. `scripts/experiment_reasoning.py`. n=100 (composite) / n=40 (s5, glm medium/high).
last_n = committed-answer accuracy after the reasoning preamble.

## composite_copy_v1 @ L16 — value accuracy

| model | none | low | medium | high |
|---|---|---|---|---|
| kimi-k2.6 | 0.22 | 0.94 | 0.98 | 0.96 |
| glm-5.2 | 0.14 | 0.74 | 0.78 | 0.81 |

## s5_v1 @ L32 — value accuracy

| model | none | low | medium | high |
|---|---|---|---|---|
| kimi-k2.6 | 0.00 | 0.00 | 0.00 | 0.00 |
| glm-5.2 | 0.00 | 0.00 | 0.00 | 0.00 |

(glm gets the *holder/role* partially right at higher effort — 0.50–0.53 — but never routes it to
the value; value stays 0.00 across all efforts.)

## Finding — background reasoning helps composition; s5 needs a concrete rendering too

- **composition**: clear monotonic dose-response (kimi 0.22→0.98, glm 0.14→0.81). Background
  reasoning **is** a test-time-compute lever for composition. The earlier "test-time doesn't help"
  claim was wrong — it held only for *explicit CoT prompting* and for *non-reasoning local models*.
- **s5** (token rendering): floor at every effort, for both models, on the value leg. Reasoning
  alone does not move non-abelian state-tracking under this rendering; combined with a concrete
  (people/jobs) rendering and an 8192-token completion budget it does — glm 1.00 @L32, 0.97 @L64,
  0.90 @L128 (`results/s5_horizon_recheck_20260705.jsonl`; consolidated report Appendix A). The
  local training-time lever is dense supervision (see `dense-supervision-results.md`).

This sharpens the central dissociation: **composition is movable by test-time compute (reasoning)
for strong models; s5 under the token rendering is not — its frontier lever is reasoning plus a
concrete rendering, and its local lever is training-time supervision density.** What does NOT help
either wall: explicit structured CoT prompting (hurts — see `autoregressive-api-results.md` E1b),
and sampling/self-correction on non-reasoning local models.

## Files

- `results/reasoning_sweep_*.jsonl` (kimi composite), `results/reasoning_kimi_s5_*.jsonl`,
  `results/reasoning_glm_*.jsonl`, `results/reasoning_glm_s5_n40_*.jsonl`.
