# FactWorld

**Evaluate your own model on a recall × state-tracking × composition benchmark in three commands.**

> **Refactored to a single natural-language format.** The benchmark renders clean natural
> language (attached punctuation, one fixed phrasing per statement type — see
> `factworld/render.py`); the earlier atomic-token format and its papers are archived under
> [`phases/`](phases/). All headline numbers below are on the natural format (OpenRouter grid +
> local multi-seed sweeps in `results/`). The autoregressive / test-time-compute experiment and
> its decomposition (binding vs recall vs routing) is written up in
> [`docs/experiments/autoregressive-api-results.md`](docs/experiments/autoregressive-api-results.md).

FactWorld is a synthetic, oracle-validated evaluation instrument. Every task is a
frozen, versioned ``TaskSpec`` with deterministic examples and one canonical
metric — **relaxed match** of the answer span (strip trailing punctuation and score the first
`len(gold)` tokens). Gold answers come from a symbolic **oracle**, never from parsing rendered
text, so labels cannot leak. A validity gate certifies that no shallow baseline clears floor.

You can evaluate any model that can continue a prompt: OpenAI-compatible APIs
(vLLM, ollama, OpenAI), HuggingFace ``transformers``, a tiny model trained
from-scratch locally, or your own Python callable.

```bash
# API fair eval for reasoning models (2048 tokens, no early stop)
python scripts/eval_model.py composite_copy_v1 --backend api --model gpt-4o-mini --n 50 --no_stop

# HuggingFace
python scripts/eval_model.py composite_copy_v1 --backend hf --model meta-llama/Llama-2-7b-hf --n 50

# Train a local model from scratch
python scripts/run_benchmark.py composite_copy_v1 --arch gdp_hybrid --d_model 320 --steps 8000
```

> **Composite-format note:** the API and HuggingFace backends automatically append
> an output-format instruction for ``composite_copy_v1`` and ``composite_v1`` so
> chat models emit the required ``<holder> <value> .`` answer span. Use
> ``--no-composite-format`` to disable it (e.g. for ablations).

📄 **Prior tech reports (archived in [`phases/`](phases/)):** these ran on the earlier
atomic-token format; the benchmark now renders clean natural language.
- [`phases/01-instrument/factworld.md`](phases/01-instrument/factworld.md) — *FactWorld: An
  Oracle-Validated Instrument for Composing Recall, State-Tracking, and Knowledge.*
- [`phases/02-non-abelian-state/report.md`](phases/02-non-abelian-state/report.md) — *FactWorld:
  A Recipe for Length-Generalizing Non-Abelian State-Tracking* (+ reproduction kit,
  [`REPRODUCE.md`](phases/02-non-abelian-state/REPRODUCE.md)).

The primary report on the natural-language format is
[`reports/factworld-consolidated.md`](reports/factworld-consolidated.md). It treats FactWorld as
one reproducible instrument for both frontier-model evaluation and local-architecture
experimentation, with the corrected API numbers and the latest local multi-seed results.

🔬 **Reproduction code for the non-abelian report:**
[`phases/02-non-abelian-state/`](phases/02-non-abelian-state/) — every claim maps to one script;
see [`REPRODUCE.md`](phases/02-non-abelian-state/REPRODUCE.md). Scoped to this hybrid and scale
regime (k=5 S₅, ≤357M), not a claim about model scaling in general.

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

spec    = CANONICAL["composite_copy_v1"]    # binding × in-context-copy recall, in one query
backend = FunctionBackend(
    lambda prompts, n, stop: ["g0 ."] * len(prompts),
    name="always-g0",
)
result  = evaluate_task(backend, spec, n=50)  # deterministic; gold from the oracle
print(result["overall"])
```

```bash
# API fair eval for reasoning models (2048 tokens, no early stop)
python scripts/eval_model.py composite_copy_v1 --backend api --model gpt-4o-mini --n 50 --no_stop

# HuggingFace
python scripts/eval_model.py composite_copy_v1 --backend hf --model meta-llama/Llama-2-7b-hf --n 50

# Local from-scratch
python scripts/run_benchmark.py composite_copy_v1 --arch gdp_hybrid --d_model 320 --steps 8000

# Run a grid of OpenRouter models (set OPENROUTER_API_KEY)
python scripts/eval_openrouter_grid.py --n 30

# Hybrid / state-space models on OpenRouter (disable built-in chain-of-thought)
python scripts/eval_openrouter_grid.py \\
    --models nvidia/nemotron-3-ultra-550b-a55b moonshotai/kimi-k2.6 \\
    --n 30 --composite_format --no_reasoning

# Evaluate a local model and merge it into the OpenRouter table
python scripts/eval_model.py composite_copy_v1 --backend local --arch gdn_hybrid \\
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

spec = CANONICAL["composite_copy_v1"]
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
result = evaluate_task(backend, CANONICAL["composite_copy_v1"], n=50)
print(result["overall"])
```

See [`docs/USAGE.md`](docs/USAGE.md) for the full backend API reference, API
cost tips, and a custom-backend example.

## The task suite

A frozen, versioned registry (``factworld.tasks.CANONICAL``) with one canonical
metric — **relaxed match** of the answer span. Each task carries a ``kind``:

**Scored** (``REPORTED``):

| task | what it measures | difficulty axis |
|---|---|---|
| `recall_copy_v1` | genuine 1-of-N in-context-copy recall | distractor pool (binding load) |
| `binding_v1` | last-write-wins state (the delta-rule axis) | give-stream length |
| `composite_copy_v1` | binding × in-context-copy recall — the 2-hop composition probe | binding horizon |
| `conflict_v1` | parametric ↔ in-context override (memorized map vs context) | pool size |
| `chain_v1` | depth-*k* pointer chase | composition depth |

**Controls** — `recall_v1`, `composite_v1` (memorized fixed-map versions; positive
controls / binding isolation, degenerate as recall scores). **Experimental** —
`s5_v1` (non-abelian S₅), `binding_load_v1` (large working set),
`composite_copy_scale_v1` (the exact k=5 configuration used for the paper's §5
scale result). Only `benchmark`-kind tasks are scored; the registry never
presents confounded tasks as peers.

Scale any task to stress larger models via explicit difficulty knobs:

```python
hard = CANONICAL["composite_copy_v1"].scaled(k=64, eval_lengths=(32, 64, 128))
```

## How the tasks get harder — and who solves them

The suite is ordered by the *kind* of computation each task needs. The further down the table, the
more the model must maintain latent state over a long stream. Concrete examples and real model
mistakes are in [`docs/tasks.md`](docs/tasks.md).

| task | what it measures | answer space (floor) | difficulty axis |
| --- | --- | --- | --- |
| `recall_copy_v1` | 1-of-N in-context-copy recall | k facts (~0.016) | distractor pool |
| `conflict_v1` | parametric ↔ in-context override | k facts (~0.016) | memorized map vs. context |
| `binding_v1` | last-write-wins state tracking | k roles (0.200) | give-stream length |
| `composite_copy_v1` | binding × in-context recall | k² pairs (~0.031) | binding horizon |
| `chain_v1` | depth-*k* pointer chase | k agents (~0.167) | composition depth |
| `s5_v1` | S₅ role permutation | 5 roles (0.200) | permutation horizon |

The numbers below are **relaxed match**, the canonical metric. The evaluation pipeline also
reports exact, semantic-containment, and last-*n* scores to separate formatting artifacts from
whether the model actually knows the answer.

### Pretrained open models

OpenRouter grid (n = 30, greedy decoding, **natural-language format**, output-format
instructions appended for composite/s5). Full table in [`docs/openrouter/results-natural.md`](docs/openrouter/results-natural.md).

> **Update:** the original 16-token eval truncated reasoning models and under-reported Kimi.
> The corrected `composite_copy_v1@L16` numbers below use `max_new_tokens=2048`, no early
> stop, and **relaxed** match (strip trailing period, score the answer span) so reasoning
> models can finish their scratchpad and local trailing-generation artifacts are treated
> consistently. The primary write-up is
> [`reports/factworld-consolidated.md`](reports/factworld-consolidated.md).

| model | recall_copy_v1 | conflict_v1 | binding_v1 | chain_v1 | composite_copy_v1 | s5_v1@L16 |
| --- | --- | --- | --- | --- | --- | --- |
| glm-5.2 | 1.000 | 1.000 | **0.767** | 0.133 | **0.867** | 0.167 |
| kimi-k2.6 | 1.000 | 1.000 | 0.633 | 0.033 | 0.867 | 0.200 |
| llama-3.3-70b-instruct | 1.000 | 1.000 | 0.633 | 0.000 | 0.767 | 0.167 |
| gemini-2.5-flash-lite | 1.000 | 1.000 | 0.300 | 0.100 | 0.233 | 0.133 |
| deepseek-chat | 1.000 | 1.000 | 0.333 | 0.033 | 0.167 | 0.133 |
| gpt-4o-mini | 1.000 | 1.000 | 0.367 | 0.067 | 0.133 | 0.133 |
| claude-3-haiku | 0.600 | 0.767 | 0.433 | 0.100 | 0.100 | 0.300 |

Reading the ladder:

- **Single-hop recall and conflict are easy.** Strong models sit at ceiling.
- **Composition is format-gated and reasoning-bounded.** With enough tokens for reasoning,
  glm-5.2, kimi-k2.6, and llama-3.3-70b all clear `composite_copy_v1`. The autoregressive
  experiment below decomposes the residual failure: strong reasoners can do the binding leg
  and the recall leg *separately* but fail to **route** the resolved holder into the recall
  lookup without scaffolding.
- **Depth and non-abelian state stay at floor.** `chain_v1` and `s5_v1` are unsolved by every
  pretrained model regardless of reasoning — the genuine state-tracking/composition wall.

**Autoregressive / test-time-compute experiment (E1b).** Format-fair leg-isolation on
`composite_copy_v1@L16` (n=100) — full write-up in
[`docs/experiments/autoregressive-api-results.md`](docs/experiments/autoregressive-api-results.md):

| model | binding-only (holder leg) | scaffolded (recall given holder) |
| --- | --- | --- |
| kimi-k2.6 | **0.99** | 1.00 |
| glm-5.2 | **0.98** | 1.00 |
| deepseek-chat | 0.60 | 0.99 |
| llama-3.3-70b | 0.34 | 0.93 |
| gpt-4o-mini | 0.28 | 1.00 |

Given the correct holder, every model recalls the value (0.93–1.00) — **recall is not the
bottleneck**. Reasoning models (kimi/glm) also solve the **binding** leg in isolation (0.98–0.99).
The residual failure in the end-to-end prompt is **routing** the resolved holder into the recall
lookup: without an oracle-provided holder, the same models that recall perfectly when the holder is
given still fail to produce the correct value. Explicit structured CoT does not fix this; in the
format-fair ablation it drops the strong reasoners to ~0 (see
[`docs/experiments/autoregressive-api-results.md`](docs/experiments/autoregressive-api-results.md)).
Background reasoning effort, not explicit step-by-step prompting, is the inference-time lever that
lifts end-to-end composition (kimi 0.22→0.98, glm 0.14→0.81).

### Custom-trained recurrent models

Local multi-seed sweep (natural-language format, d=256, 4 layers, 8k steps, 5 seeds, holder/value
decomposition). Full table in [`results/sweep_main_*.md`](results/). `fprm` is a weight-tied looped
conv+attention block (the architecture the external FPRM work motivates); `gdp_hybrid` is the
`[recurrent, recurrent, attn, recurrent]` stack from `factworld/models.py`.

**`binding_v1` (length extrapolation) — relaxed match mean±std / p(converge)**

| arch | L16 | L32 | L64 |
| --- | --- | --- | --- |
| fprm | 1.00 / 100% | 0.97 / 100% | **0.94 / 100%** |
| gdp_hybrid | 0.99 / 100% | 0.58 / 0% | 0.55 / 0% |
| transformer | 0.45 / 0% | 0.35 / 0% | 0.33 / 0% |

**`composite_copy_v1` (k=32) @L16 — relaxed match mean±std**

Staged-curriculum recipe (d=768, 8 layers, 25k steps, 3 seeds); full write-up in
[`reports/factworld-consolidated.md`](reports/factworld-consolidated.md) §5.

| arch | composite_p16@L16 relaxed |
| --- | --- |
| gdp_hybrid | **0.747 ± 0.174** |
| fprm | 0.253 ± 0.178 |
| transformer | 0.005 ± 0.005 |

Relaxed match is the fair cross-regime metric: it handles API models that omit the trailing
period and local models that append extra tokens after the answer. The decomposition localizes the
composition gap: every recurrent architecture solves the **holder** (binding) leg but fails the
**value** (recall-of-the-resolved-holder) leg — the same routing wall the API models hit.

**Trained scratchpad (E2, local) makes it worse.** Supervising the per-step oracle trace
(`prompt→trace→answer`) on the staged curriculum collapses holder accuracy 0.96→0.14 and value
accuracy 0.79→0.02 — the model emits a structured-but-wrong trace that compounds errors, the
opposite of the API scaffolded result where the *oracle-provided* holder unlocks recall
(0.93–1.00). See `results/curriculum_staged_d768_b64_80k_trace.md` and
[`docs/experiments/autoregressive-api-results.md`](docs/experiments/autoregressive-api-results.md).

**But the s5 / non-abelian wall is movable with the right supervision** (reproduced on the natural
format, `scripts/experiment_dense_supervision.py`). Per-step state supervision every K events (guided
free-run eval — events forced, holder/value slots generated):

| K (stride) | value @L16 | value @L64 |
| --- | --- | --- |
| 1 (dense) | **1.00** | **0.75** |
| 2 | 0.98 | 0.40 (bimodal) |
| 4, 8 | ~0.20 | ~0.20 (floor) |

Dense K=1 solves s5 in-distribution (1.00) and extrapolates to 4× the trained length (L64 = 0.75);
K=2 is bimodal at L64 (0.40). The circuit fails to form below a checkpoint every ~2 events.
**Inference-time compute (self-consistency, up to 30 votes) does not move the wall in either
direction** — seeds with the circuit are already greedy-optimal; seeds without stay at the floor.
The wall is in *learnability/supervision density*, not test-time compute — consistent with the API
result that only an *oracle-provided* intermediate helps.

The prior (atomic-token format) non-abelian recipe and its dense-supervision `s5_v1` results
(0.94–0.99) are archived in [`phases/02-non-abelian-state/`](phases/02-non-abelian-state/).

**External context.** Movahedi et al. (2026) report strong `s5_v1` length generalization with a
looped-transformer architecture (FPRM) that uses a causal 1-D convolution / unroll-to-convergence
mechanism. Our `fprm` implements a weight-tied variant of that block; on `binding_v1` it is the
strongest length-extrapolator (0.94 @L64), but like every architecture it floors `s5_v1` under
answer-only supervision.

In short: the suite climbs from easy single-hop recall, through binding and composition, to the
`s5_v1` state-tracking wall. Two clean dissociations (full write-up: [`docs/experiments/`](docs/experiments/)):

1. **Composition is movable by test-time compute** for strong reasoning models. A reasoning-effort
   dose-response sweep shows composite value climbing with effort (kimi 0.22→0.98, glm 0.14→0.81);
   given the output format, reasoners solve it. Non-reasoners floor. (The earlier "test-time doesn't
   help" claim was wrong — it held only for explicit CoT *prompting* and for non-reasoning local models.)
2. **Non-abelian state-tracking (s5) is movable by training-time supervision density, NOT by reasoning.**
   It floors at every reasoning effort, but dense per-step supervision solves it (10/10 seeds at L16)
   and the circuit **survives weaning to answer-only** (wean_mixed 8/8) — so it deploys label-free.
   Only the recurrent hybrid (`gdp_hybrid`) **extrapolates** the learned circuit in length (L64 0.75,
   L128 0.50–0.59; `fprm` and `transformer` collapse past the train length).

The levers are **reasoning strength** (composition), **supervision density + weaning** (s5), and
**recurrent architecture** (s5 length extrapolation). Explicit structured CoT prompting never helps
(and hurts); recall is trivial once the holder is known.

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
  openrouter/             external LLM API grid results
    results.md              benchmark tasks
    s5-results.md           experimental `s5_v1` task
  recall/                 recall-capability results
    readout.md              attention-free recall readout
  composition/            composition-capability results
    results.md              small-scale composite diagnostic + decomposition
  state-tracking/         state-tracking-capability results
    dense-supervised.md     dense-supervised S5/A5 word problem (§3.1 probe)
    scale.md                §5 ~45M scale + matched LR sweeps
factworld/                the instrument (torch-free data/oracle/eval + the model zoo)
  world.py, oracle.py     deterministic KB + symbolic ground-truth solver
  render.py               template renderer + its exact inverse parser (no-leak contract)
  tasks.py                the frozen, scalable task registry + canonical metric
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
<summary><b>Reproducing the paper</b></summary>

## Reproducing the reports

Every headline number maps to one script. The data/oracle/eval layer is
pure-stdlib; the training runs below need a CUDA GPU (validated on an RTX 3090).
Scripts that write a `docs/**/*.md` rebuild it after every cell (crash-safe); the
rest print their tables to stdout (transcribed into the cited doc).

```bash
# Instrument-level guarantee
python scripts/validate_suite.py          # validity gate: no majority/recency/first-pos shortcut clears floor (prints PASS)

# 1-command benchmark entry point (any task / scaled variant)
python scripts/run_benchmark.py composite_copy_v1 --arch gdp_hybrid --d_model 320 --steps 8000

# Section 3.1 — state-tracking dissociation                 -> docs/state-tracking/dense-supervised.md
python scripts/dense_s5.py --group s5     # S5 matrix: gdp_pure / n_h=1 null / gdn / transformer / gru, 3 seeds
python scripts/dense_s5.py --group a5     # A5 not-S5-specific control panel

# Section 3.2 — recall                                       -> docs/results-ci.md, docs/recall/readout.md
python scripts/ci_dissociation.py         # recall_copy_v1 + binding_v1, 4 archs x 3 seeds (pool-2 dissociation CIs)
python scripts/recall_attention_test.py   # attention-free attribution: gdp_pure / gdn_pure / gdp_hybrid across pools 2..8
python scripts/recall_fair.py             # the 1-hop-vs-deferred differential (onehop/defsep/defpad; n_heads 4 vs 8)

# Section 4 — composition gap                                -> docs/results.md, docs/results-ci.md, docs/composition/results.md
python scripts/collect_baselines.py       # 4-arch from-scratch reference baselines, all scored tasks (seed 0)
python scripts/sk_composite.py            # memorization diagnostic + the n_h in {1,2,4} fixed-param mechanism control
python scripts/iso.py                      # the n_h ∈ {1,2,4} product-structure ablation at fixed params (neg-eig on/off)
python scripts/decompose.py               # the gap decomposed: state leg vs recall leg + routing on holder-wrong examples

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
