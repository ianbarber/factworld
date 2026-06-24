# FactWorld

**Evaluate your own model on a recall × state-tracking × composition benchmark in three commands.**

FactWorld is a synthetic, oracle-validated evaluation instrument. Every task is a
frozen, versioned ``TaskSpec`` with deterministic examples and one canonical
metric — **position-strict exact match** of the answer span. Gold answers come
from a symbolic **oracle**, never from parsing rendered text, so labels cannot
leak. A validity gate certifies that no shallow baseline clears floor.

You can evaluate any model that can continue a prompt: OpenAI-compatible APIs
(vLLM, ollama, OpenAI), HuggingFace ``transformers``, a tiny model trained
from-scratch locally, or your own Python callable.

```bash
# API (composite format instruction is appended automatically for composite tasks)
python scripts/eval_model.py composite_copy_v1 --backend api --model gpt-4o-mini --n 50

# HuggingFace
python scripts/eval_model.py composite_copy_v1 --backend hf --model meta-llama/Llama-2-7b-hf --n 50

# Train a local model from scratch
python scripts/run_benchmark.py composite_copy_v1 --arch gdp_hybrid --d_model 320 --steps 8000
```

> **Composite-format note:** the API and HuggingFace backends automatically append
> an output-format instruction for ``composite_copy_v1`` and ``composite_v1`` so
> chat models emit the required ``<holder> <value> .`` answer span. Use
> ``--no-composite-format`` to disable it (e.g. for ablations).

📄 **The paper:** [`paper.pdf`](paper.pdf) · [`paper.md`](paper.md) — *FactWorld:
An Oracle-Validated Instrument for Composing Recall, State-Tracking, and
Knowledge.* Reference numbers live in [`docs/results.md`](docs/results.md) and
related docs. The latest external-LLM grid (including Nemotron 3 / Kimi results)
is in [`docs/openrouter-results.md`](docs/openrouter-results.md); the new S₅ non-abelian
grid is in [`docs/openrouter-s5-results.md`](docs/openrouter-s5-results.md).

🔬 **Follow-up — validating non-abelian state-tracking with FactWorld:**
[`followups/non-abelian-state/`](followups/non-abelian-state/) — *FactWorld: A
Recipe for Length-Generalizing Non-Abelian State-Tracking.* Uses the instrument
to localize the non-abelian (S₅) state-tracking wall and derive a training
recipe — near-dense process supervision to form the circuit, mixed-density
internalization, a target-length training distribution, base-selection on
free-running accuracy, and post-training deep-state coverage (length-general to
~8× the trained horizon on a selected clean base, label-free) — scoped to this
hybrid and scale regime (k=5 S₅, ≤357M), not a claim about model scaling in
general. See
[`non-abelian-state.pdf`](followups/non-abelian-state/non-abelian-state.pdf)
and the reproduction guide
[`REPRODUCE.md`](followups/non-abelian-state/REPRODUCE.md).

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
# API (auto-appends composite format instruction for composite tasks)
python scripts/eval_model.py composite_copy_v1 --backend api --model gpt-4o-mini --n 50

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
metric — **position-strict exact match** of the answer span. Each task carries a
``kind``:

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

The REPORTED suite forms a difficulty ladder. The same strong pretrained open models
that are near ceiling on single-hop recall fall back to the floor once the task
requires maintaining non-abelian latent state.

### Pretrained open models on the benchmark tasks

OpenRouter grid (n = 30, greedy decoding, composite-format instruction appended;
see [`docs/openrouter-results.md`](docs/openrouter-results.md)):

| model | recall_copy_v1 | conflict_v1 | binding_v1 | chain_v1 | composite_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| nemotron-3-ultra-550b-a55b | 1.000 | 1.000 | 0.733 | 0.000 | **0.767** |
| kimi-k2 | 1.000 | 1.000 | **0.900** | 0.300 | 0.733 |
| kimi-k2.5 | 1.000 | 1.000 | 0.800 | 0.300 | 0.633 |
| kimi-k2.6 | 1.000 | 1.000 | 0.867 | 0.133 | 0.567 |
| deepseek-chat | 1.000 | 1.000 | 0.467 | 0.200 | 0.600 |
| llama-3.3-70b-instruct | 1.000 | 1.000 | 0.700 | 0.167 | 0.200 |
| gpt-4o-mini | 1.000 | 1.000 | 0.567 | 0.067 | 0.167 |

Reading the ladder:

- **Single-hop recall and conflict are easy.** Every strong model is at or near ceiling.
- **Binding is scale-sensitive.** Kimi K2 leads at 0.900; Nemotron 3 Ultra and Llama 3.3 70B follow.
- **Composition is the bottleneck — with a caveat.** Nemotron 3 Ultra scores 0.767 and Kimi K2 0.733
  on `composite_copy_v1`, but only after an explicit output-format instruction tells the model to
  emit `<holder> <value> .`. Without that instruction every model scores 0% because it emits only
  the value.
- **Depth extrapolation stays hard.** `chain_v1` peaks at 0.300, consistent with the paper's claim
  that pointer-chase depth generalization is poor for pretrained chat models.

### The non-abelian wall: S₅ role permutations

The experimental `s5_v1` task pushes one step further. A stream of `swap`/`cycle_roles` events
permutes which agent holds each of five roles; the query asks for the role of a single agent at the
end. There are five possible answers, so random guessing is 0.20.

OpenRouter grid with a role-token format instruction
([`docs/openrouter-s5-results.md`](docs/openrouter-s5-results.md)):

| model | L32 | L64 | L128 | mean |
| --- | --- | --- | --- | --- |
| nemotron-3-ultra-550b-a55b | 0.233 | 0.167 | 0.267 | 0.222 |
| kimi-k2 | 0.067 | 0.200 | 0.233 | 0.167 |
| kimi-k2.5 | 0.200 | 0.233 | 0.267 | 0.233 |
| kimi-k2.6 | 0.133 | 0.233 | 0.200 | 0.189 |
| deepseek-chat | 0.200 | 0.067 | 0.100 | 0.122 |
| llama-3.3-70b-instruct | 0.300 | 0.167 | 0.067 | 0.178 |
| gpt-4o-mini | 0.100 | 0.167 | 0.167 | 0.144 |

Every model is at the chance floor. Even Nemotron 3 Ultra, the strongest composite model in the
grid, does not lift above it. The format instruction gets the right token *shape*, but none of these
pretrained models tracks the running non-abelian permutation.

### Custom-trained architectures: the same wall, moved by supervision

The follow-up study in [`followups/non-abelian-state/`](followups/non-abelian-state/) holds the
architecture class fixed (a GatedDeltaProduct recurrent hybrid) and varies the supervision. It shows
that the wall is **learnability**, not capacity, and that the right training signal moves it.

| regime | recipe | headline result |
| --- | --- | --- |
| Small-scale baseline | `gdp_hybrid`, d_model=320, 4 layers, 8k steps, seed 0 | `composite_copy_v1` floors at ≤0.020 across L4–L64 (see [`docs/results.md`](docs/results.md)) |
| Dense-supervised S₅ | `gdp_pure`, per-step state trace, L32 train | L32/L64/L128 token-acc = 1.00 / 1.00 / **0.99** (see [`docs/state-tracking-results.md`](docs/state-tracking-results.md)) |
| Non-abelian recipe (follow-up) | dense process supervision → mixed-density internalization → target-length distribution → post-training deep-state coverage | dense K=1: L16=1.00, L64=0.95; internalized horizon curriculum: L64=**0.94**; post-train on clean base: L64=**0.99**, L128=**0.86** |

The comparison tells a clear story:

- **Architecture is not the immediate bottleneck.** The product-recurrence backbone that fails the
  small-scale composite under sparse supervision *solves* the S₅ word problem when given dense
  per-step state supervision.
- **Pretrained open models are in the sparse-supervision regime.** They were not trained with the
  oracle's intermediate role-permutation trace, so they behave like the answer-only baseline of the
  follow-up: at floor. This is consistent with the follow-up's claim that sparse outcome-level
  signal — including vanilla RL — cannot climb the non-abelian cliff.
- **Length generalization is a data-distribution problem, not a width problem.** Scaling the custom
  architecture to 357M does not move the internalized horizon wall (answer-only L64 stays at 0.20),
  but a sufficient density of target-length examples does, and post-training deep-state coverage
  extends the clean circuit to ≈8× the trained horizon with no labels at the target length.

In short: the suite climbs from easy single-hop recall, through binding and composition, to a
non-abelian state-tracking wall that pretrained chat models hit at chance. The wall is movable, but
the lever is the **supervision and training distribution**, not the model name or parameter count.

## Repository layout

```
paper.md                  the paper (Markdown source)
paper.pdf                 the typeset PDF        (rebuild: python scripts/build_pdf.py)
docs/results.md           4-arch reference baselines (position-strict exact match)
docs/results-ci.md        3-seed CIs on the dissociating cells + attention-free recall ablation
docs/state-tracking-results.md  dense-supervised S5/A5 word problem (the §3.1 state-tracking probe)
docs/scale-results.md     the §5 ~45M scale + matched LR sweeps
docs/composite-results.md the §4 small-scale composite (memorization diagnostic + decomposition)
docs/related-work.md      related work with verified citations
docs/USAGE.md             backend API reference and custom-backend examples
docs/openrouter-results.md       external LLM API grid (OpenRouter) on the benchmark tasks
docs/openrouter-s5-results.md    external LLM API grid on the experimental S₅ non-abelian task
factworld/                the instrument (torch-free data/oracle/eval + the model zoo)
  world.py, oracle.py     deterministic KB + symbolic ground-truth solver
  render.py               template renderer + its exact inverse parser (no-leak contract)
  tasks.py                the frozen, scalable task registry + canonical metric
  backends.py             ModelBackend interface + local/hf/api/function backends
  runner.py               task-agnostic evaluate_task() entry point
  models.py, train.py     transformer / mamba2 / gdp_hybrid / gdn_hybrid / gru on one skeleton
scripts/                  the runnable suite (run_benchmark, eval_model, validate_suite, …)
tests/                    oracle, renderer, tokenizer, model-parity, and validity tests
followups/non-abelian-state/  follow-up study: the non-abelian state-tracking recipe + learnability map (segregated)
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

## Reproducing the paper

Every headline number maps to one script. The data/oracle/eval layer is
pure-stdlib; the training runs below need a CUDA GPU (validated on an RTX 3090).
Scripts that write a `docs/*.md` rebuild it after every cell (crash-safe); the
rest print their tables to stdout (transcribed into the cited doc).

```bash
# Instrument-level guarantee
python scripts/validate_suite.py          # validity gate: no majority/recency/first-pos shortcut clears floor (prints PASS)

# 1-command benchmark entry point (any task / scaled variant)
python scripts/run_benchmark.py composite_copy_v1 --arch gdp_hybrid --d_model 320 --steps 8000

# Section 3.1 — state-tracking dissociation                 -> docs/state-tracking-results.md
python scripts/dense_s5.py --group s5     # S5 matrix: gdp_pure / n_h=1 null / gdn / transformer / gru, 3 seeds
python scripts/dense_s5.py --group a5     # A5 not-S5-specific control panel

# Section 3.2 — recall                                       -> docs/results-ci.md, docs/recall-readout-results.md
python scripts/ci_dissociation.py         # recall_copy_v1 + binding_v1, 4 archs x 3 seeds (pool-2 dissociation CIs)
python scripts/recall_attention_test.py   # attention-free attribution: gdp_pure / gdn_pure / gdp_hybrid across pools 2..8
python scripts/recall_fair.py             # the 1-hop-vs-deferred differential (onehop/defsep/defpad; n_heads 4 vs 8)

# Section 4 — composition gap                                -> docs/results.md, docs/results-ci.md, docs/composite-results.md
python scripts/collect_baselines.py       # 4-arch from-scratch reference baselines, all scored tasks (seed 0)
python scripts/sk_composite.py            # memorization diagnostic + the n_h in {1,2,4} fixed-param mechanism control
python scripts/iso.py                      # the n_h ∈ {1,2,4} product-structure ablation at fixed params (neg-eig on/off)
python scripts/decompose.py               # the gap decomposed: state leg vs recall leg + routing on holder-wrong examples

# Section 5 — scale + the matched LR sweeps                  -> docs/scale-results.md
python scripts/scale_confirm.py           # 45M multi-seed confirmation: gdp 5 / transformer 5 / gdn 3 seeds (default recipe)
python scripts/transformer_lr_sweep.py    # transformer 45M, 5 LRs x 2 seeds  (negative-arm control: 0/10)
python scripts/gdn_lr_sweep.py            # gdn_hybrid 45M, 5 LRs x 2 seeds    (Layer 2: capable-but-LR-fragile, 1/10)
python scripts/gdp_lr_sweep.py            # gdp_hybrid 45M, 5 LRs x 2 seeds    (Layer 2: broad band, 7/10)
python scripts/gdp_confirm_5e4.py         # gdp 45M @ tuned lr 5e-4, 5 seeds   (pins the L16 5/5, L64 3/5 point estimate)
python scripts/gdn_confirm_3e4.py         # gdn 45M @ lr 3e-4, 5 seeds         (W2: 4/5 converge, 1/5 extrapolate)
python scripts/fair_config.py             # W3: transformer n_heads=8+resid (floor survives 0/10) + recurrent short-conv
```

</details>
