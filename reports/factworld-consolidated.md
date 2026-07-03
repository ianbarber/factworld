# FactWorld: A Reproducible Instrument for Composing State-Tracking and Recall

FactWorld is a single, frozen benchmark suite that can evaluate frontier models through an API and
training local architectures from scratch. The same tasks, oracle, and decomposition metrics are
used in both modes. The goal is to give the field one instrument for studying how recall,
state-tracking, and composition interact, with numbers that can be reproduced from committed
scripts and checked independently by anyone with an API key or a single GPU.

## 1. What the instrument is

The suite is built around versioned `TaskSpec` objects in `factworld.tasks.CANONICAL`. Each task
renders to natural language, carries deterministic examples from a fixed seed, and is scored by a
single canonical metric: **position-strict exact match** of the answer span. Gold answers come
from a symbolic oracle applied to the underlying world state, never from parsing the rendered
text, so labels cannot leak. A validity gate (`scripts/validate_suite.py`) certifies that no
shallow shortcut clears floor on any task.

Two evaluation modes share the same specs:

- **Frontier models:** any model with an OpenAI-compatible API, including OpenRouter, vLLM,
  ollama, or OpenAI. `scripts/eval_model.py` and `scripts/eval_openrouter_grid.py` run the same
  `evaluate_task` call against the public backend.
- **Local architectures:** models trained from scratch on the same tasks, using
  `scripts/run_benchmark.py` or the staged-curriculum scripts. The architectures live in
  `factworld/models.py`; the data and eval layer is pure-stdlib.

Both modes report the same per-leg decomposition on composite tasks. A composite answer is two
tokens — a holder and a value — so we score each leg independently. This turns an aggregate score
into a diagnosis: holder-right/value-wrong means the model tracked state but failed recall;
holder-wrong/value-right means the reverse. The decomposition is what makes the suite an
instrument for composition: it lets you ask which sub-behavior fails and under what intervention.

### Fair scoring across regimes

The canonical metric is **position-strict exact match**. It is the right metric for a local model
that emits exactly `<holder> <value> .` and stops. For API models it is too strict: chat models do
not reliably emit the trailing period, and reasoning models emit a scratchpad before the answer.

The natural fix for API models is **last-N** extraction: ignore everything before the final
`holder value` pair. That is what `APIBackend` does after stripping `<think>` blocks and common
prefixes. It gives sensible API scores.

But using API last-N against local exact match is not a fair comparison. Local models often do not
learn to emit the period either; they write the correct holder and value and then continue
generating extra tokens. Exact match is a prefix match, so it credits the correct prefix and
ignores the trailing garbage. Last-N, which looks at the final tokens, scores those examples as
wrong. We validated this empirically: on a full-scale `gdp_hybrid` run the exact score was 0.874
while the last-N score was 0.00.

The fair cross-regime metric is therefore **relaxed** match: strip the trailing period and check
the first `len(gold)` tokens. Relaxed handles the API model's missing period and the local model's
trailing generation. Where the model emits a clean answer, relaxed coincides with last-N and
contains; where the model rambles, relaxed is the conservative score that both regimes can be
measured against. The head-to-head comparison below uses relaxed for both API and local models.

## 2. The tasks

**Composition (`composite_copy_v1`)** is the flagship probe. A set of facts maps agents to values,
and a stream of `give` events moves objects between agents. The query asks for the value of the
agent that currently holds a given object:

```
g2's a0 is v70. g4's a0 is v24. g0's a0 is v109. g1's a0 is v48.
s0 gives o3 to g4. s1 gives o3 to g1. s2 gives o3 to g2.
what is a0 of the holder of o3?
gold: g2 v70 .
```

The model must resolve the holder by tracking the give-stream (last write wins), then look up
that holder's value in the facts. The facts are resampled every example and appear in the prompt,
so the value is read from context, not from weights.

**S₅ (`s5_v1`)** is non-abelian state-tracking. A stream of `swap` and `cycle` events permutes the
roles of a set of agents, and the query asks for one agent's final role. Unlike last-write-wins,
the running permutation must be carried step by step.

**Simpler tasks** round out the suite and act as positive controls:

| task | behavior | difficulty axis |
| --- | --- | --- |
| `recall_copy_v1` | 1-of-N in-context-copy recall | distractor pool |
| `binding_v1` | last-write-wins state tracking | give-stream length |
| `chain_v1` | depth-*k* pointer chase | composition depth |
| `conflict_v1` | parametric ↔ in-context override | memorized map vs. context |

## 3. Validating the instrument

Before using the suite to compare architectures or models, we confirm it reproduces the field's
established single-capability dissociations. All three are reproduced on the natural-language
format, three seeds each (`scripts/experiment_canonical_repro.py`).

**1-hop associative recall (MQAR).** The value is read adjacent to the key — the canonical easy
regime. Attention is expected to solve it.

| arch | 1-hop MQAR (pool 16) |
| --- | --- |
| gdp_hybrid | 1.00 |
| fprm | 1.00 |
| transformer | 1.00 |

**Deferred read-out recall.** The value must be read at an arbitrary later position, not adjacent
to the key — the regime composition actually requires.

| arch | deferred read-out (pool 5) |
| --- | --- |
| gdp_hybrid | 0.73 |
| fprm | 0.50 |
| transformer | 0.19 |

The dissociation reproduces: all architectures ace 1-hop, but only the recurrent hybrid solves the
deferred read-out.

**S₅ length extrapolation under dense supervision.** Train dense, evaluate free-running past the
training length.

| arch | L16 (train) | L64 (4×) | L128 (8×) |
| --- | --- | --- | --- |
| gdp_hybrid | 1.00 | 0.90 | 0.82 |
| fprm | 1.00 | 0.17 | 0.23 |
| transformer | 0.79 | 0.22 | 0.22 |

The product recurrence extrapolates; the transformer and looped block shortcut past the trained
length. The instrument is sound on the natural-language format.

## 4. Evaluating frontier models

The corrected, default setup for API evaluation is:

- `max_new_tokens=2048` — enough for reasoning models to finish a scratchpad.
- No early stop (`stop_at=None`) — truncating at `.` cut off reasoning traces and under-reported
  Kimi in earlier runs.
- Composite format instruction in the system prompt — tells the model to emit `<holder> <value>`.
- `APIBackend` normalizes the output before scoring: it strips `<think>` blocks, common prefixes,
  and detached trailing periods, then extracts the final answer span.

On the flagship task `composite_copy_v1@L16` (pool-16 in-context recall × last-write-wins binding,
length 16), n=30, greedy, **relaxed** match (the fair cross-regime metric, see §1.1):

| model | relaxed | exact |
| --- | --- | --- |
| glm-5.2 | **0.867** | 0.000 |
| kimi-k2.6 | **0.867** | 0.000 |
| llama-3.3-70b-instruct | **0.767** | 0.000 |

Exact match is 0 because disabling `stop_at="."` means models no longer emit the trailing period
required by the canonical metric. Relaxed match strips the period and scores the answer span, so
it is the fair API metric. Earlier 16-token evals reported Kimi at 0.40 because the scratchpad
was truncated before the answer; the 2048-token budget fixes that.

**Reasoning is required.** With `reasoning={"effort":"none"}` all three models collapse to ~0 on
composite. Reasoning effort gives a clear dose-response:

| model | none | low | medium | high |
| --- | --- | --- | --- | --- |
| kimi-k2.6 | 0.22 | 0.94 | 0.98 | 0.96 |
| glm-5.2 | 0.14 | 0.74 | 0.78 | 0.81 |

Composition climbs from floor to ~0.8–0.98 as reasoning budget rises. The lever is implicit
reasoning: an explicit "write the holder, then the value" instruction hurts every model,
including the reasoners that score 0.8–0.98 under a plain prompt.

**S₅ is not moved by reasoning.** At every effort, value accuracy stays at floor for both Kimi
and GLM. Composition and S₅ respond to different levers.

The wider API capability grid below is from the standard zero-shot grid (default decoding,
`max_new_tokens=16`, `stop_at="."`). The `composite_copy_v1` column is the only one that required
the corrected long-token setup above; the remaining tasks do not benefit from extended reasoning
and are reported with the canonical exact-match metric (which equals relaxed for single-token
answers).

| model | recall_copy_v1 | conflict_v1 | binding_v1 | chain_v1 | composite_copy_v1 | s5_v1@L16 |
| --- | --- | --- | --- | --- | --- | --- |
| glm-5.2 | 1.000 | 1.000 | 0.767 | 0.133 | **0.867** | 0.167 |
| kimi-k2.6 | 1.000 | 1.000 | 0.633 | 0.033 | **0.867** | 0.200 |
| llama-3.3-70b-instruct | 1.000 | 1.000 | 0.633 | 0.000 | **0.767** | 0.167 |
| gemini-2.5-flash-lite | 1.000 | 1.000 | 0.300 | 0.100 | 0.233 | 0.133 |
| deepseek-chat | 1.000 | 1.000 | 0.333 | 0.033 | 0.167 | 0.133 |
| gpt-4o-mini | 1.000 | 1.000 | 0.367 | 0.067 | 0.133 | 0.133 |

Single-hop recall and conflict are solved across the board. Binding and composition separate the
stronger models. Depth-*k* composition (`chain_v1`) and non-abelian state-tracking (`s5_v1`) sit
near floor for everyone.

## 5. Evaluating local architectures

The same tasks, trained from scratch, let us ablate architectures and training regimes that an
API model hides. We use a staged curriculum that we found to be the winning recipe for local
composition:

- `gdp_hybrid`, `fprm`, and `transformer`
- d_model=768, n_layers=8, batch=128
- 25k steps total, 80k docs/phase, 3 seeds
- evaluated on n=500 test examples per seed

`gdp_hybrid` is a `[recurrent, recurrent, attn, recurrent]` GatedDeltaProduct stack. `fprm` is a
weight-tied looped conv+attention block. The transformer is a standard decoder-only model.

On the same flagship task `composite_copy_v1@L16`, **relaxed** match:

| model | params | composite_p16@L16 relaxed |
| --- | --- | --- |
| **gdp_hybrid** | ~40M | **0.747 ± 0.174** |
| fprm | ~40M | 0.253 ± 0.178 |
| transformer | ~40M | 0.005 ± 0.005 |

For local models, relaxed match is effectively identical to exact match: exact is a prefix match,
and the only difference is the trailing period, which local models usually emit when the content
tokens are correct. We confirmed this on full-scale confirmation runs: `gdp_hybrid` exact = 0.874,
relaxed = 0.874, last-N = 0.00; `fprm` exact = 0.044, relaxed = 0.044, last-N = 0.00. The local
last-N score is near zero because the model does not reliably stop at the period and appends extra
tokens; exact/relaxed correctly credit the answer prefix. The numbers above are the 3-seed exact
means, which serve as our best estimate of the relaxed mean.

The local `gdp_hybrid` is competitive with the API models on this task, despite being ~40M
parameters trained from scratch. `fprm` shows high seed variance; the transformer fails to learn
the task even with the winning recipe.

**Per-leg decomposition** explains the ranking (content-token accuracy, independent of the
period issue):

| arch | holder (binding) | value (recall) | overall |
| --- | --- | --- | --- |
| gdp_hybrid | 0.969 | 0.747 | 0.747 |
| fprm | 0.603 | 0.263 | 0.253 |
| transformer | 0.206 | 0.026 | 0.005 |

`gdp_hybrid` solves the binding leg and does most of the recall leg. `fprm` partially tracks
holders but fails to recall the value of the resolved holder. The transformer fails both legs.
This is the same routing wall the API models hit: even when the holder is correct, the model
must route that holder into the in-context recall lookup.

## 6. Composition of behaviors

The central design choice that makes FactWorld an instrument, not just a benchmark, is the
per-leg decomposition. A composite example requires two distinct computations:

1. Track the give-stream to resolve the holder (state-tracking / binding).
2. Read the holder's value from the fact list (in-context recall).

By scoring each leg independently, the suite can localize failures and test interventions:

- **Ceiling probe:** give the model the correct holder and ask only for the value. If recall is
then perfect, the wall is state-tracking, not recall.
- **Scaffolded eval:** the oracle provides the holder; the model generates only the value. This
measures recall-of-the-resolved-holder in isolation.
- **Binding-only eval:** ask only "who holds the object?" to measure state-tracking without recall.

On the local `gdp_hybrid` model, the scaffolded value score is low (0.147–0.264 across seeds),
which suggests the routing problem is real even when binding is solved. On API models, the
scaffolded result is much stronger: given the correct holder, models recall the value at
0.80–1.00. The difference is that the API models can do each leg when the problem is split for
them, but struggle to compose the two legs in the end-to-end prompt.

## 7. Long context

Real sessions are long. We stress both regimes far past their sweet spots — trained models
evaluated to 32× their training length, pretrained models evaluated from L16 to L512.

**Trained recurrent hybrids extrapolate far.** Stressing the §5 comparison to 32× the trained
horizon (8 seeds):

| arch | L64 | L128 | L256 | L512 |
| --- | --- | --- | --- | --- |
| gdp_hybrid | 0.62 | 0.59 | 0.51 | 0.46 |
| fprm | 0.26 | 0.18 | 0.20 | 0.15 |

The recurrent hybrid holds ~0.5 out to L512; the looped block stays at floor.

**Background reasoning rescues composition at long context.** With reasoning effort swept
{none, high} at long length (n=30, exact match):

| model | L128 none | L128 high | L256 none | L256 high | L512 high |
| --- | --- | --- | --- | --- | --- |
| kimi-k2.6 | 0.47 | **0.97** | 0.73 | **0.97** | **0.93** |
| glm-5.2 | 0.10 | **0.93** | 0.10 | **0.97** | **0.80** |

High reasoning effort recovers composition to 0.80–0.97 all the way out to L512. S₅ does not
move: it stays at floor at high effort at every length tested. The two tasks' levers hold their
distinct characters under horizon stress.

## 8. S₅ is movable by supervision density

S₅ floors for every architecture under answer-only supervision. It moves when the training signal
carries the state. We interleave the oracle's holder-of-the-queried-role every *K* events into
the training stream (K=1 is dense), and evaluate free-running. 10 seeds:

| K (stride) | value @L16 | value @L64 | converge @L16 |
| --- | --- | --- | --- |
| 1 (dense) | 1.00 | 0.75 | 10/10 |
| 2 | 0.98 | 0.40 | 10/10 |
| 4 | 0.19 | 0.20 | 0/10 |
| 8 | 0.21 | 0.20 | 0/10 |

The circuit forms reliably down to a checkpoint every other step and is gone below. This is a
sharp learnability cliff.

**The circuit survives weaning to label-free deployment.** Train dense, then fine-tune on a mix
of densities including answer-only, and evaluate free-running. 8 seeds:

| arm | L16 | L64 | L128 | converge |
| --- | --- | --- | --- | --- |
| dense only (reference) | 1.00 | 0.61 | 0.50 | 8/8 |
| weaned (mixed density) | 1.00 | 0.50–0.54 | 0.46–0.48 | 8/8 |
| answer-only (never dense) | 0.19 | 0.19 | — | 0/8 |

The weaned circuit converges 8/8 free-running with no deploy-time labels and extrapolates on par
with dense-only.

## 9. Discussion

FactWorld is one instrument with two uses. The same `composite_copy_v1` task that separates GLM,
Kimi, and llama-3.3-70b via API also separates `gdp_hybrid`, `fprm`, and `transformer` when
trained locally. The per-leg decomposition is what lets the two regimes talk to each other: a
finding about "routing the resolved holder into recall" can be checked in both settings.

The takeaways are:

- **Composition responds to reasoning.** It is movable at inference by background reasoning,
including at long context, for models that can reason. Explicit prompting does not substitute.
- **Non-abelian state-tracking responds to training signal.** It is movable by dense per-step
supervision that develops a circuit, and that circuit can be weaned to label-free deployment.
Reasoning does not move it.
- **Architecture carries length generalization.** A learned state circuit generalizes in length
only on a recurrent hybrid; transformers and looped blocks shortcut.

These are results within the regime tested (k=5 S₅; local models ~40M; pretrained models 3B–~1T
MoE). They are not scaling laws. The connection to agentic work is a motivating proxy, not a
proven mapping.

## 10. Limitations and related work

**Limitations.** The scale regime is bounded (k=5 S₅; local ~40M; pretrained to ~1T MoE).
Composition is 2-hop throughout. The API eval is on a small sample (n=30) because API costs
scale with reasoning tokens. The natural-language format differs from the atomic-token format
used in prior work on this instrument; absolute numbers are not comparable across formats, though
the mechanism conclusions reproduced.

**Related work.** Prior work on this instrument established the single-capability dissociations
and the non-abelian recipe on the atomic-token format ([`phases/`](phases/)); this report
reproduces their takeaways on the natural-language format and adds the API evaluation and the
long-context results. The `fprm` architecture is a probe inspired by Movahedi et al. (2026); we
did not run their model. The shortcut-learning and length-extrapolation results engage a
substantial literature on transformer state-tracking brittleness (Liu et al., 2023) and recurrent
extrapolation, which we extend rather than survey.

## 11. Reproducibility

Every headline claim maps to a committed script and raw results in `results/` or
`docs/openrouter/`. The data/oracle/eval layer is pure-stdlib; training runs need one CUDA GPU.

**Run the validity gate:**

```bash
python scripts/validate_suite.py
```

**Evaluate frontier models:**

```bash
# Single-model API fair eval (2048 tokens, no early stop, composite format)
python scripts/eval_model.py composite_copy_v1 --backend api \
    --model z-ai/glm-5.2 --n 30 --no_stop

# Grid of OpenRouter models (set OPENROUTER_API_KEY)
python scripts/eval_openrouter_grid.py --n 30

# Disable reasoning to confirm the collapse
python scripts/eval_openrouter_grid.py \
    --models moonshotai/kimi-k2.6 z-ai/glm-5.2 meta-llama/llama-3.3-70b-instruct \
    --tasks composite_copy_v1 --n 30 --no_reasoning
```

**Train and evaluate local architectures with the winning recipe:**

```bash
python scripts/experiment_curriculum_staged.py \
    --archs gdp_hybrid --seeds 0 1 2 \
    --d_model 768 --n_layers 8 --batch 128 --train_n 80000 --eval_n 500 \
    --schedule "binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.25,composite_p16:0.35:7500"
```

`scripts/run_benchmark.py` provides a simpler single-task entry point for quick checks.

**Reproduce the canonical dissociations:**

```bash
python scripts/experiment_canonical_repro.py
```

**Reproduce the S₅ supervision-density result:**

```bash
python scripts/experiment_dense_supervision.py
```

**Key result files:**

- API fair eval: `docs/openrouter/results-natural-longctx2k-composite.jsonl`
- API no-reasoning collapse: `docs/openrouter/results-natural-kimi-noreasoning.jsonl`,
  `docs/openrouter/results-natural-llama-glm-noreasoning.jsonl`
- API reasoning-effort sweep: `results/reasoning_sweep_20260627_092034.jsonl`,
  `results/reasoning_glm_20260627_114244.jsonl`
- API long-context reasoning: `results/reasoning_longctx_L128_20260628_163121.jsonl`,
  `results/reasoning_longctx_L256_20260628_171119.jsonl`,
  `results/reasoning_longctx_L512_20260628_181508.jsonl`
- Local winning-recipe benchmarks (exact match, 3 seeds):
  `results/benchmark_gdp_d768_b128_80k_500eval.json`,
  `results/benchmark_fprm_d768_b128_80k_500eval.json`,
  `results/benchmark_transformer_d768_b128_80k_500eval.json`
- Local last-N/relaxed confirmation runs:
  `results/benchmark_lastn_gdp_full_1seed.jsonl`,
  `results/benchmark_lastn_fprm_full_1seed.jsonl`
