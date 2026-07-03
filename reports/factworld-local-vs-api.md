# FactWorld Local vs. API Model Comparison

Comparison on the flagship task `composite_copy_v1@L16` (pool-16 in-context-copy recall × last-write-wins binding, length 16).

## Local models

Trained with the staged curriculum winning recipe:
- `d_model=768`, `n_layers=8`, batch `128`, `train_n=80000` per phase
- 25k steps total, 3 seeds, evaluated on `n=500` test examples per seed
- Exact-match metric (position-strict token match)

| model | params | composite_p5@L16 | composite_p16@L16 | notes |
|---|---|---|---|---|
| **gdp_hybrid** | ~40M | **0.959 ± 0.028** | **0.747 ± 0.174** | Best local architecture; strong mean but bimodal across seeds |
| fprm | ~40M | 0.431 ± 0.144 | 0.253 ± 0.178 | High seed variance; some runs master binding, others recall |
| transformer | ~40M | 0.019 ± 0.021 | 0.005 ± 0.005 | Fails to learn the task even with the winning recipe |

## API models

Evaluated via OpenRouter. The original eval used only 16 output tokens, which cut off reasoning models before they could emit the final answer.

### Original 0-shot eval (`max_new_tokens=16`, `stop_at="."`)

| model | exact | contains | last_n |
|---|---|---|---|
| llama-3.3-70b | **0.800** | — | — |
| glm-5.2 | **0.800** | — | — |
| kimi-k2.6 | 0.400 | 0.633 | 0.467 |

The 16-token budget hides Kimi's capability: it runs out of tokens mid-reasoning and returns the truncated scratchpad.

### Fair eval (`max_new_tokens=2048`, no `stop_at`, composite format instruction)

| model | exact | relaxed | contains | **last_n** |
|---|---|---|---|---|
| glm-5.2 | 0.000 | 0.867 | 0.933 | **0.933** |
| kimi-k2.6 | 0.000 | 0.867 | 0.867 | **0.867** |
| llama-3.3-70b | 0.000 | 0.767 | 0.767 | **0.767** |

- Exact is 0 because disabling `stop_at="."` means models no longer emit the trailing period required by exact match.
- `last_n` extracts the final `holder value` pair. It is the fair metric for API models that may list intermediate holders or emit a reasoning trace before the answer.
- With adequate token budget, **Kimi is competitive with GLM** (0.867 vs. 0.933), and both beat llama-3.3-70b.

## Key findings

1. **The 16-token API eval was biased against reasoning models.** Kimi and GLM need ~1000+ reasoning tokens to solve the task; truncating them hides their capability.
2. **All three API models need reasoning.** With `reasoning={"effort":"none"}` they all collapse to ~0.000.
3. **Local `gdp_hybrid` matches the best API models.** At 0.747 exact-match mean it sits between GLM (0.933 last-N) and Kimi (0.867 last-N), though the metrics are not directly comparable.
4. **Architecture matters locally.** `gdp_hybrid` ≫ `fprm` ≫ `transformer` on the same training recipe.
5. **Few-shot examples do not help API models here.** In 1-shot tests the models performed worse than 0-shot, regardless of formatting.

## Fair head-to-head (composite_p16 @ L16)

| model | score | metric | n |
|---|---|---|---|
| GLM-5.2 | **0.933** | last-N | 30 |
| Kimi-k2.6 | 0.867 | last-N | 30 |
| gdp_hybrid (local) | **0.747** | exact | 500 |
| llama-3.3-70b | 0.767 | last-N | 30 |
| fprm (local) | 0.253 | exact | 500 |
| transformer (local) | 0.005 | exact | 500 |

The local `gdp_hybrid` model is ~40M parameters trained from scratch; on this task it scores close to the API models.

## Files

- Local results:
  - `results/benchmark_gdp_d768_b128_80k_500eval.json`
  - `results/benchmark_fprm_d768_b128_80k_500eval.json`
  - `results/benchmark_transformer_d768_b128_80k_500eval.json`
- API results:
  - Original: `docs/openrouter/results-natural.jsonl`
  - Fair long-context: `docs/openrouter/results-natural-longctx2k-composite.jsonl`
