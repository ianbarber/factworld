# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-24T04:28:28Z.

**Setup:** n = 30 examples per task/length, greedy decoding (temperature 0), held-out `test` split. The base system prompt is:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

For `composite_copy_v1` and `composite_v1` an additional format instruction is appended:

> For questions that ask 'what is a0 of the holder of ...', answer with the holder's name followed by the requested value, like 'g3 v9'.

`APIBackend` normalizes a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`) so the canonical exact-match metric is meaningful for chat-model output. In this run exact and relaxed scores coincide.

Nemotron 3 models were evaluated with built-in chain-of-thought disabled (`reasoning={"effort": "none"}`), because they otherwise emit a long reasoning trace that exceeds the short generation budget.

## Results

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| llama-3.2-3b-instruct | 0.267 | 0.033 | 0.067 | 0.800 | 0.933 |
| llama-3.1-8b-instruct | 0.233 | 0.000 | 0.000 | 0.867 | 0.800 |
| qwen-2.5-7b-instruct | 0.467 | 0.133 | 0.067 | 0.933 | 1.000 |
| qwen3-32b | 0.367 | 0.233 | 0.267 | 0.300 | 0.367 |
| llama-3.3-70b-instruct | 0.700 | 0.167 | 0.200 | 1.000 | 1.000 |
| deepseek-chat | 0.467 | 0.200 | 0.600 | 1.000 | 1.000 |
| gpt-4o-mini | 0.567 | 0.067 | 0.167 | 1.000 | 1.000 |
| gemini-2.5-flash-lite | 0.067 | 0.267 | 0.133 | 0.933 | 1.000 |
| claude-3-haiku | 0.433 | 0.033 | 0.033 | 0.767 | 0.900 |
| nemotron-3-nano-30b-a3b | 0.700 | 0.033 | 0.233 | 0.933 | 0.867 |
| nemotron-3-super-120b-a12b | 0.700 | 0.000 | 0.200 | 0.967 | 0.767 |
| nemotron-3-ultra-550b-a55b | 0.733 | 0.000 | **0.767** | 1.000 | 1.000 |
| llama-3.3-nemotron-super-49b-v1.5 | 0.600 | 0.067 | 0.267 | 1.000 | 0.867 |
| kimi-k2 | **0.900** | 0.300 | 0.733 | 1.000 | 1.000 |
| kimi-k2.5 | 0.800 | 0.300 | 0.633 | 1.000 | 1.000 |
| kimi-k2.6 | 0.867 | 0.133 | 0.567 | 1.000 | 1.000 |

## What the numbers show

- **Single-hop tasks are easy.** `recall_copy_v1` and `conflict_v1` are at or near ceiling for strong models.
- **Binding is scale-sensitive.** Kimi K2 (0.900) and Kimi K2.6 (0.867) lead; Nemotron 3 Ultra (0.733) and Llama 3.3 70B (0.700) follow.
- **Chain remains hard.** Even the best models are ≤ 0.300 on `chain_v1@L4`, consistent with the paper's depth-extrapolation claim.
- **Composition is the bottleneck — but hybrid models break through.** With an explicit output-format instruction, `composite_copy_v1` rises well above zero. Nemotron 3 Ultra scores **0.767** and Kimi K2 scores **0.733**, both well above the earlier best (DeepSeek 0.600). Without the format instruction every model scored 0%.

## Hybrid / state-space / MoE models

We added a second batch of models that are architecturally closer to the local recurrent baselines:

- **Nemotron 3** (Nano / Super / Ultra): hybrid Mamba-2 + sparse attention + MoE. The Ultra variant is the strongest composite model in the grid at **0.767**, with perfect recall/conflict and solid binding (0.733). Nano and Super are comparable to Llama 3.3 70B on composite.
- **Llama 3.3 Nemotron Super 49B**: a Nemotron-post-trained Llama variant. It matches the Nano/Super composite level (0.267) but with stronger single-hop recall.
- **Kimi K2 / K2.5 / K2.6**: Moonshot's MoE models (MLA-based; K2 is the strongest here at **0.733** composite and **0.900** binding). Note: the directly GatedDeltaNet-related `moonshotai/kimi-linear-48b-a3b-instruct` is not currently routable through OpenRouter, so these are the closest available proxies.

All Nemotron runs used `--no_reasoning` because the OpenRouter endpoints default to emitting a long internal reasoning trace; without disabling it the model returns no answer within the short generation budget.

## Format sensitivity on `composite_copy_v1`

The composite task requires a two-token answer (`<holder> <value> .`). A naive system prompt gives every model 0% because they emit only the value. Appending the explicit format instruction unlocks measurable performance:

| model | naive prompt | + format instruction |
| --- | --- | --- |
| llama-3.3-70b-instruct | 0.000 | 0.400 (pilot) / 0.200 (full grid) |
| deepseek/deepseek-chat | 0.000 | 0.500 (pilot) / 0.600 (full grid) |
| gpt-4o-mini | 0.000 | 0.167 |
| qwen/qwen-2.5-7b-instruct | 0.000 | 0.000 (pilot) / 0.067 (full grid) |

The pilot and full-grid numbers differ because of sampling variance at n=30; the qualitative point is stable.

## Length extrapolation on `composite_copy_v1`

With the format instruction, accuracy degrades as the binding history grows longer (n=30, 4 models):

| model | L16 | L32 | L64 |
| --- | --- | --- | --- |
| llama-3.3-70b-instruct | 0.367 | 0.200 | 0.067 |
| deepseek/deepseek-chat | 0.500 | 0.367 | 0.300 |
| gpt-4o-mini | 0.167 | 0.200 | 0.100 |
| qwen/qwen-2.5-7b-instruct | 0.067 | 0.000 | 0.033 |

This matches the instrument's OOD-length behavior: composition is learnable in-distribution but decays as the chain lengthens.

## Caveats

- **n=30 is a pilot sample.** Confidence intervals are wide; use these numbers for coarse floor/ceiling claims, not fine model rankings.
- **Single-length evaluation for the main table.** Each task is evaluated at `eval_lengths[0]`; binding and composite are therefore in-distribution. The length-sweep table above addresses this for composite.
- **Qwen3-32b anomaly.** It underperforms Qwen2.5-7B on recall/conflict, suggesting an OpenRouter routing or prompt-format issue rather than a true capability gap.
- **Gemini Flash Lite binding anomaly.** Near-floor binding (0.067) while chain is best-in-class (0.267) is unusual and should be replicated.

## Raw data

- `docs/openrouter-results.json` — example-level predictions for the main grid.
- `docs/openrouter-nemotron-results.json` — Nemotron 3 + Llama-Nemotron 49B outputs.
- `docs/openrouter-kimi-results.json` — Kimi K2 / K2.5 / K2.6 outputs.
- `docs/openrouter-composite-baseline.json` — naive-prompt composite outputs.
- `docs/openrouter-composite-format.json` — format-prompt ablation outputs.
- `docs/openrouter-composite-lengths.json` — L16/L32/L64 composite outputs.
