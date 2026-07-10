# FactWorld

**A composition instrument: recall and state tracking, measured independently and composed —
identically for frontier API models and local from-scratch models.**

Recall is well tested by MQAR; state tracking by the S₅ word-problem literature. FactWorld tests
both, independently *and* in composition, under one protocol for frontier models over an API and
for small models trained from scratch — so it can place a new frontier model *and* attribute
capabilities to architectural and training choices. Tasks render as natural language (so
pretrained models can take them) over a constrained vocabulary (so small-scale experiments can
too). Every task is a frozen, versioned ``TaskSpec`` with deterministic examples; gold answers
come from a symbolic **oracle**, never from parsing rendered text, so labels cannot leak; a
validity gate certifies that no shallow baseline clears floor. The canonical metric is **relaxed
match** of the answer span (exact / containment / last-*n* are diagnostics). The canonical story
is anchored in [`AGENTS.md`](AGENTS.md).

> **What this is — and isn't.** FactWorld is a **mechanism probe for the component capabilities that
> agent workloads depend on** — working-memory recall, state tracking, and multi-step composition —
> measured rigorously (oracle labels, validity gate, per-leg decomposition) and identically across
> frontier API models and local from-scratch architectures. It is **not** an end-to-end agent
> benchmark: every task is single-turn, single-answer-span, with no tool use, planning, or
> multi-turn action. The component→agent mapping is a motivating analogy, not a proven one. Closing
> the explicit agentic-behavior gap (tool use, multi-turn carry-over) is tracked separately in
> [#6](https://github.com/ianbarber/factworld/issues/6).

Evaluate any model that can continue a prompt — OpenAI-compatible APIs (vLLM, ollama, OpenAI),
HuggingFace ``transformers``, a tiny model trained from scratch locally, or your own Python
callable — in three commands:

```bash
# API fair eval for reasoning models (2048 tokens, no early stop)
python scripts/eval_model.py composite_copy_v2 --backend api --model gpt-4o-mini --n 50 --no_stop

# HuggingFace
python scripts/eval_model.py composite_copy_v2 --backend hf --model meta-llama/Llama-2-7b-hf --n 50

# Train a local model from scratch
python scripts/run_benchmark.py composite_copy_v2 --arch gdp_hybrid --d_model 320 --steps 8000
```

> **Composite-format note:** the API and HuggingFace backends automatically append
> an output-format instruction for composite-family tasks (e.g. ``composite_copy_v2``)
> so chat models emit the required ``<holder> <value> .`` answer span. Use
> ``--no-composite-format`` to disable it (e.g. for ablations).

## The task suite

A frozen, versioned registry (``factworld.tasks.CANONICAL``). The taxonomy — components first,
then their compositions:

| | task | notes |
|---|---|---|
| **Component: recall** | `recall_copy_v1` | single-query, deferred-readout MQAR variant; pool breadth = load axis |
| — parametric variant | `recall_v1` / `conflict_v1` | retrieval from weights (local models); `conflict_v1` scores the in-context override |
| **Component: state tracking** | `binding_v2` | last-write-wins (absorbing updates — not group ops) |
| — non-abelian variant | `s5_v1` | order-sensitive permutation streams; length = sequence stress |
| **Composition: state × recall** | `composite_copy_v2` | the two-hop; headline statistic = **gap** (binding − composed) |
| **Composition: recall ∘ recall** | `chain_v1` | pointer chase; depth axis at fixed breadth (the no-wrap staircase builds k=2d+1) |

Each axis tests a different thing: solve rate; pool/breadth (working-set load); depth/length
(iteration count); regime (**instant** = reasoning off + answer contract = in-weights, vs
**thinking** = generous budget); reasoning tokens needed to solve. Difficulty knobs are
**calibration parameters** — used to place each model class mid-scale, never published as axes.

**Retired** — the recency-defective v1 give-stream family (`binding_v1`,
`binding_load_v1`, `composite_v1`, `composite_copy_v1`, `composite_copy_scale_v1`)
lives in `tasks.RETIRED`: generable for historical reproduction, never scored
([#11](https://github.com/ianbarber/factworld/issues/11)). Only `benchmark`-kind
tasks are scored; the registry never presents confounded tasks as peers.

Scale any task to stress larger models via explicit difficulty knobs:

```python
hard = CANONICAL["composite_copy_v2"].scaled(k=64, eval_lengths=(32, 64, 128))
```

## The thesis: no element is free

Each element of the composition is paid for by an architectural or training choice. Current price
table (local evidence; two rows remain open):

| element | price | evidence |
|---|---|---|
| adjacent (1-hop) recall | attention — every architecture aces adjacent readout | [consolidated §3](reports/factworld-consolidated.md) |
| deferred recall | product recurrence — the transformer aces adjacent, fails deferred (0.19 vs gdp_hybrid 0.73) | consolidated §3; archived provenance [phases/01 §3.2](phases/01-instrument/factworld.md) |
| last-write state | recurrence, ordered by form — fprm (product recurrence) 1.00 @B6 on the binding leg, over gdp_hybrid (0.56) over transformer (0.23); fprm is first to break under breadth (B24), where only the gated hybrid holds | [frontier benchmark, local regime](reports/frontier-benchmark.md) |
| non-abelian state (formation) | dense per-step supervision — a state checkpoint every ≤2 events; architecture-independent | [consolidated §8](reports/factworld-consolidated.md); [experiments §1](docs/experiments/README.md); archived provenance [phases/02 §4](phases/02-non-abelian-state/report.md) |
| non-abelian state (length extrapolation) | recurrent hybrid — gdp_hybrid 0.75 @L64; fprm and transformer collapse past train length | [experiments §1](docs/experiments/README.md) |
| depth extrapolation | **open** — no measured choice buys it: trained at chain depths 2–3, all three architectures read at or below the 1/6 guess at depths 4–5 | [frontier benchmark, local regime](reports/frontier-benchmark.md); [experiments §15](docs/experiments/README.md) |
| local composition (value leg) | **open** — value ≤0.17 in all 45 breadth-sweep runs (at/below the 1/pool guess), even on binding-solved seeds of all three architectures | [frontier benchmark, local regime](reports/frontier-benchmark.md) |

## The findings

### 1. Frontier profiles (9 models, two regimes)

Instant regime (reasoning off, one-line answer contract): the binding leg against the composed
two-hop cell, floors as first-class rows. Relaxed match; marks in plain language — `*` cannot
disable reasoning (off-arm ran effort=minimal), `†` visible working on the canonical attempt,
`(x.xx @512)` escalated-budget diagnostic; full glossary in the
[report](reports/frontier-benchmark.md).

| Model | binding @L16 | composed @L16 | composed @L64 | gap @L16 |
|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 0.78 | 0.72 | 0.43 | +0.06 |
| anthropic/claude-sonnet-5 | 0.77 | 0.62 (0.76 @512)† | 0.32 (0.66 @512)† | +0.15† |
| deepseek/deepseek-v4-pro | 0.51 | 0.44 | 0.19 | +0.07 |
| google/gemini-3.5-flash | 0.66* | 0.64* | 0.28* | +0.02* |
| moonshotai/kimi-k2.6 | 0.94† | 0.77† | 0.93† | +0.17† |
| nvidia/nemotron-3-ultra-550b-a55b | 0.49 | 0.33 | 0.12 | +0.16 |
| openai/gpt-5.5 | 0.80 | 0.46 | 0.33 | +0.34 |
| qwen/qwen3.7-max | 0.51 | 0.24 | 0.08 | +0.27 |
| z-ai/glm-5.2 | 0.70† | 0.35† | 0.16 | +0.35† |
| *recency heuristic (floor)* | 0.04 | 0.04 | 0.06 | — |
| *object-filter floor* | 0.41 | 0.41 | 0.15 | — |

Recall is not the constraint: the scaffolded leg reads 0.98–1.00 for every model, and recall
under load is at ceiling — pool-64 `recall_copy_v1` @L64 reads 1.00 for all nine (chance 0.016).
The gap is the composition deficit.

Thinking regime (effort=high, 16,384 tokens): two state-stress rows. `⊘` = majority
finish=length, not measurable at this budget; `‡` = provider ignored the token cap.

| Model | chain d128 (k=257) | s5 @L256 |
|---|---|---|
| anthropic/claude-opus-4.8 | 0.08 | ⊘ |
| anthropic/claude-sonnet-5 | 0.04 | ⊘ |
| deepseek/deepseek-v4-pro | ⊘ ‡ | ⊘ |
| google/gemini-3.5-flash | 0.88 | 0.52 |
| moonshotai/kimi-k2.6 | 0.64‡ | 0.88 |
| nvidia/nemotron-3-ultra-550b-a55b | ⊘ | ⊘ |
| openai/gpt-5.5 | 0.36 | 0.96 |
| qwen/qwen3.7-max | 0.96 | 0.80 |
| z-ai/glm-5.2 | 0.36 | 0.88 |

The instant and thinking rankings are near-orthogonal — opus and sonnet, the strongest clean
instant composers, post the weakest measurable chain scores — so profiles are per-axis, never a
single scalar. Depth also dissociates by regime within a single cell: chain d16 (k=33) floors
instant for every model that answers cleanly (≤0.08 vs chance 0.03) and reads 0.44–1.00 with
thinking on the same items. Full narrative: [`reports/frontier-benchmark.md`](reports/frontier-benchmark.md);
rendered tables and per-cell Wilson intervals: [`docs/benchmark/results.md`](docs/benchmark/results.md).

### 2. Local attributions

Which architecture or training choice buys which component, on the same tasks trained from
scratch: product recurrence buys deferred recall and s5 length extrapolation, attention suffices
only for adjacent readout ([consolidated §3](reports/factworld-consolidated.md)); the composed
cell's failure localizes to the value leg at every scale
([consolidated §5](reports/factworld-consolidated.md)); dense per-step supervision (every ≤2
events) forms the non-abelian circuit on any architecture, and weaning keeps it label-free
([consolidated §8](reports/factworld-consolidated.md)). The running experiment log is
[`docs/experiments/README.md`](docs/experiments/README.md).

## Reports and prior work

- 📊 [`reports/frontier-benchmark.md`](reports/frontier-benchmark.md) — the recurring frontier
  benchmark (finding set 1): components, composed cells, floors, two regimes, protocol.
- 📄 [`reports/factworld-consolidated.md`](reports/factworld-consolidated.md) — the consolidated
  natural-language-format report (finding set 2): local multi-seed results, per-leg
  decomposition, supervision-density experiments, and the API reasoning analyses.
- 🗂 **Prior tech reports (archived in [`phases/`](phases/); atomic-token format — mechanism
  conclusions carry, absolute numbers do not):**
  - [`phases/01-instrument/factworld.md`](phases/01-instrument/factworld.md) — cited for the
    instrument's validity machinery: the oracle/no-leak render↔parse contract and the validity
    gate that `scripts/validate_suite.py` continues.
  - [`phases/02-non-abelian-state/report.md`](phases/02-non-abelian-state/report.md) — cited for
    the s5 supervision-density and training-length-distribution levers, measured on the
    atomic-token format (+ reproduction kit,
    [`REPRODUCE.md`](phases/02-non-abelian-state/REPRODUCE.md); scoped to that hybrid and scale
    regime, k=5 S₅, ≤357M).
- 🧪 **Experiments using FactWorld as an RL/distillation testbed** live under
  [`experiments/`](experiments/):
  [`experiments/mopd/`](experiments/mopd/README.md) — *Multi-teacher On-Policy Distillation*
  (MOPD) on FactWorld: RL-specialise Qwen3-1.7B on two abilities (binding, recall) with a
  verifiable reward, then distil both into one model that holds both (LoRA adapters on a shared
  backbone). See [`REPRODUCE.md`](experiments/mopd/REPRODUCE.md).

## Install

The data / oracle / eval layer is pure-stdlib (no GPU needed). Backends are
installed via extras:

```bash
# Core only — enough to generate tasks and score predictions
pip install -e .

# Add the backends you need
pip install -e ".[train]"     # local from-scratch training (torch + flash-linear-attention)
pip install -e ".[hf]"        # HuggingFace transformers
pip install -e ".[api]"       # OpenAI-compatible APIs
pip install -e ".[dev]"       # pytest + hf/api/train backend deps
```

## Quickstart

```python
from factworld.backends import FunctionBackend
from factworld.runner import evaluate_task
from factworld.tasks import CANONICAL

spec    = CANONICAL["composite_copy_v2"]    # binding × in-context-copy recall, in one query
backend = FunctionBackend(
    lambda prompts, n, stop: ["g0 ."] * len(prompts),
    name="always-g0",
)
result  = evaluate_task(backend, spec, n=50)  # deterministic; gold from the oracle
print(result["overall"])
```

```bash
# API fair eval for reasoning models (2048 tokens, no early stop)
python scripts/eval_model.py composite_copy_v2 --backend api --model gpt-4o-mini --n 50 --no_stop

# HuggingFace
python scripts/eval_model.py composite_copy_v2 --backend hf --model meta-llama/Llama-2-7b-hf --n 50

# Local from-scratch
python scripts/run_benchmark.py composite_copy_v2 --arch gdp_hybrid --d_model 320 --steps 8000

# Run a grid of OpenRouter models (set OPENROUTER_API_KEY)
python scripts/eval_openrouter_grid.py --n 30

# Hybrid / state-space models on OpenRouter (disable built-in chain-of-thought)
python scripts/eval_openrouter_grid.py \\
    --models nvidia/nemotron-3-ultra-550b-a55b moonshotai/kimi-k2.6 \\
    --n 30 --composite_format --no_reasoning

# Evaluate a local model and merge it into the OpenRouter table
python scripts/eval_model.py composite_copy_v2 --backend local --arch gdn_hybrid \\
    --d_model 320 --steps 8000 --n 50 --json_out results/local-gdn.json
python scripts/merge_grid_results.py docs/openrouter-results.json results/local-gdn.json \\
    --out docs/combined-results.md

python -m factworld.tasks             # suite self-test (determinism + oracle round-trip)
python scripts/validate_suite.py      # validity gate: no shallow shortcut clears floor
```

## Evaluate your own model

Implement the ``ModelBackend`` interface and pass it to
``factworld.runner.evaluate_task``:

```python
from factworld.runner import evaluate_task
from factworld.backends import ModelBackend
from factworld.tasks import CANONICAL

class MyBackend(ModelBackend):
    @property
    def name(self):
        return "my-backend"

    def generate(self, prompts: list[str], max_new_tokens: int,
                 stop_at: str | None = None) -> list[str]:
        # Return one continuation per prompt, not including the prompt.
        return [my_model.complete(p, max_tokens=max_new_tokens) for p in prompts]

spec = CANONICAL["composite_copy_v2"]
result = evaluate_task(MyBackend(), spec, n=50)
print(result["overall"])
```

Or use a one-liner with ``FunctionBackend``:

```python
from factworld.backends import FunctionBackend
from factworld.runner import evaluate_task
from factworld.tasks import CANONICAL

backend = FunctionBackend(
    lambda prompts, n, stop: ["g0 ."] * len(prompts),
    name="always-g0",
)
result = evaluate_task(backend, CANONICAL["composite_copy_v2"], n=50)
print(result["overall"])
```

See [`docs/USAGE.md`](docs/USAGE.md) for the full backend API reference, API
cost tips, and a custom-backend example. Concrete prompts, gold answers, and real model
mistakes for every task are in [`docs/tasks.md`](docs/tasks.md).

## Repository layout

```
phases/                  prior work, archived (ran on the atomic-token format)
  01-instrument/           original FactWorld paper (.md + .pdf)
  02-non-abelian-state/    non-abelian state-tracking report + reproduction kit
docs/
  tasks.md                concrete prompts, gold answers, and real model mistakes for every task
  USAGE.md                backend API reference and custom-backend examples
  related-work.md         related work with verified citations
  results.md              4-arch reference baselines (relaxed match)
  results-ci.md           3-seed CIs on the dissociating cells + attention-free recall ablation
  benchmark/              rendered frontier-benchmark tables, figures, results.csv
  openrouter/             external LLM API grid results
    results.md              benchmark tasks
    s5-results.md           experimental `s5_v1` task
  recall/                 recall-capability results
    readout.md              attention-free recall readout
  composition/            composition-capability results
    results.md              small-scale composite diagnostic + decomposition
  state-tracking/         state-tracking-capability results
    dense-supervised.md     dense-supervised S5/A5 word problem (§3.1 probe)
    scale.md                archived k=5 compute-matched scale + LR sweeps (retired composite_copy_scale_v1; distinct from the report's §5 composite scale sweep)
factworld/                the instrument (torch-free data/oracle/eval + the model zoo)
  world.py, oracle.py     deterministic KB + symbolic ground-truth solver
  render.py               template renderer + its exact inverse parser (no-leak contract)
  tasks.py                the frozen, scalable task registry + canonical metric
  benchmark.py            the frontier-benchmark registry (models, facets, budgets)
  backends.py             ModelBackend interface + local/hf/api/function backends
  runner.py               task-agnostic evaluate_task() entry point
  models.py, train.py     transformer / mamba2 / gdp_hybrid / gdn_hybrid / gru on one skeleton
scripts/                  the runnable suite (run_benchmark, eval_model, validate_suite, …)
tests/                    oracle, renderer, tokenizer, model-parity, and validity tests
phases/02-non-abelian-state/  archived reproduction scripts + per-claim tables (non-abelian report)
```

The hybrid configuration (`[recurrent, recurrent, attn, recurrent]`, n_h=4, neg-eig) lives in
`factworld/models.py`.

## Tests

```bash
python tests/test_world_oracle.py     # zero-dependency runner
python tests/test_backends.py         # backend / runner smoke tests
uv run --with pytest pytest -q        # full suite
```

<details>
<summary><b>Reproducing the reports</b></summary>

## Reproducing the reports

Every headline number maps to one script. The data/oracle/eval layer is
pure-stdlib; the training runs below need a CUDA GPU (validated on an RTX 3090).
Scripts that write a `docs/**/*.md` rebuild it after every cell (crash-safe); the
rest print their tables to stdout (transcribed into the cited doc).

```bash
# Instrument-level guarantee
python scripts/validate_suite.py          # validity gate: no majority/recency/first-pos shortcut clears floor (prints PASS)

# 1-command benchmark entry point (any task / scaled variant)
python scripts/run_benchmark.py composite_copy_v2 --arch gdp_hybrid --d_model 320 --steps 8000

# Frontier benchmark (registry-driven; resume-skips existing cells)
python scripts/run_frontier_benchmark.py --dry-run   # plan + cost preview
python scripts/render_benchmark.py                   # re-render docs/benchmark/

# Section 3.1 — state-tracking dissociation                 -> docs/state-tracking/dense-supervised.md
python scripts/dense_s5.py --group s5     # S5 matrix: gdp_pure / n_h=1 null / gdn / transformer / gru, 3 seeds
python scripts/dense_s5.py --group a5     # A5 not-S5-specific control panel

# Section 3.2 — recall                                       -> docs/results-ci.md, docs/recall/readout.md
python scripts/ci_dissociation.py         # recall_copy_v1 + binding_v2, 4 archs x 3 seeds (pool-2 dissociation CIs)
python scripts/recall_attention_test.py   # attention-free attribution: gdp_pure / gdn_pure / gdp_hybrid across pools 2..8
python scripts/recall_fair.py             # the 1-hop-vs-deferred differential (onehop/defsep/defpad; n_heads 4 vs 8)

# Section 4 — composition gap                                -> docs/results.md, docs/results-ci.md, docs/composition/results.md
python scripts/collect_baselines.py       # 4-arch from-scratch reference baselines, all scored tasks (seed 0)
python scripts/sk_composite.py            # memorization diagnostic + the n_h in {1,2,4} fixed-param mechanism control
python scripts/iso.py                      # the n_h ∈ {1,2,4} product-structure ablation at fixed params (neg-eig on/off)
python scripts/decompose.py               # the gap decomposed: state leg vs recall leg + routing on holder-wrong examples
python scripts/experiment_composite_scale.py  # compute-matched scale sweep: small/medium/large × gdp/fprm/transformer

# Section 5 — scale + the matched LR sweeps                  -> docs/state-tracking/scale.md
python scripts/scale_confirm.py           # 45M multi-seed confirmation: gdp 5 / transformer 5 / gdn 3 seeds (default recipe)
python scripts/transformer_lr_sweep.py    # transformer 45M, 5 LRs x 2 seeds  (negative-arm control: 0/10)
python scripts/gdn_lr_sweep.py            # gdn_hybrid 45M, 5 LRs x 2 seeds    (Layer 2: capable-but-LR-fragile, 1/10)
python scripts/gdp_lr_sweep.py            # gdp_hybrid 45M, 5 LRs x 2 seeds    (Layer 2: broad band, 7/10)
python scripts/gdp_confirm_5e4.py         # gdp 45M @ tuned lr 5e-4, 5 seeds   (pins the L16 5/5, L64 3/5 point estimate)
python scripts/gdn_confirm_3e4.py         # gdn 45M @ lr 3e-4, 5 seeds         (W2: 4/5 converge, 1/5 extrapolate)
python scripts/fair_config.py             # W3: transformer n_heads=8+resid (floor survives 0/10) + recurrent short-conv

# Non-abelian report (phases/02-non-abelian-state/report.md) — see phases/02-non-abelian-state/REPRODUCE.md
```

</details>
