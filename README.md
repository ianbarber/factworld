# FactWorld

**A composition instrument: recall and state tracking, measured independently and in composition —
identically for frontier API models and local from-scratch models.**

Recall is well tested by multi-query associative recall (MQAR); state tracking by the word-problem
literature on S₅ — the symmetric group on five elements, i.e. order-sensitive permutation
composition. FactWorld tests both, independently *and* in composition, under one protocol for
frontier models over an API and for small models trained from scratch. Every task is a frozen,
versioned ``TaskSpec`` rendered as natural language over a constrained vocabulary, with
deterministic examples; gold answers come from a symbolic **oracle**, never from parsing rendered
text, and a validity gate certifies that no shallow baseline clears floor. The canonical metric is
**match**: strip a trailing period from both sides and compare the model's first len(gold)
whitespace tokens to the gold answer — binary per item, no partial credit
(`factworld.tasks.score_relaxed`); containment is the one published diagnostic. The full report
is [`reports/frontier-benchmark.md`](reports/frontier-benchmark.md).

Three parts, in this order here and in the [full report](reports/frontier-benchmark.md):

1. **[The instrument](#1-the-instrument)** — recall and state tracking certified independently and in composition (report Part 1).
2. **[Benchmarking the frontier](#2-benchmarking-the-frontier)** — the benchmark built on it: 11 models, two regimes (report Part 2).
3. **[Exploring the architectures](#3-exploring-the-architectures)** — which components buy each capability at small scale (report Part 3).

> **What this is — and isn't.** A **mechanism probe for the component capabilities that agent
> workloads depend on** — working-memory recall, state tracking, and multi-step composition. It
> is not an end-to-end agent benchmark: every task is single-turn, single-answer-span, with
> no tool use, planning, or multi-turn action; the component→agent mapping is a motivating
> analogy, not a proven one.

## 1. The instrument

A frozen, versioned registry (``factworld.tasks.CANONICAL``). The taxonomy — components first,
then their compositions:

| | task | notes |
|---|---|---|
| **Component: recall** | `recall_copy_v1` | single-query, deferred-readout MQAR variant; pool breadth = load axis |
| — parametric variant | `recall_v1` / `conflict_v1` | retrieval from weights (local models); `conflict_v1` scores the in-context override |
| **Component: state tracking** | `binding_v2` | last-write-wins (absorbing updates — not group ops) |
| — commutative variant | `commutative_v1` | each event turns a named entity's dial a few clicks; every event matters, order does not — experimental, thinking-regime only (only gpt-5.5 CI-separates @L64; instant and d256-local at chance) |
| — non-abelian variant | `s5_v1` | order-sensitive permutation streams; length = sequence stress |
| **Composition: state × recall** | `composite_copy_v2` | the two-hop; headline statistic = **gap** (binding − composed) |
| **Composition: recall ∘ recall** | `chain_v1` | follow a chain of "ask X" pointers hop by hop to the fact at the end — recall applied to its own output; depth = number of hops, at fixed breadth (the no-wrap staircase builds k=2d+1) |

Each axis tests a different thing: solve rate; pool/breadth (working-set load); depth/length
(iteration count); regime (**instant** = reasoning off + answer contract — in-weights ability —
vs **thinking** = generous budget); reasoning tokens needed to solve. Difficulty knobs are
**calibration parameters** — used to place each model class mid-scale, never published as axes.

Tasks are versioned; a defective version is retired outright, never kept scored. The v1
give-stream family (`binding_v1`, `binding_load_v1`, `composite_v1`, `composite_copy_v1`,
`composite_copy_scale_v1`) lives in `tasks.RETIRED`: generable for historical reproduction,
never scored; only `benchmark`-kind tasks are scored. Why the v1 sampler was defective is the
recency methodological note in [Part 1 of the report](reports/frontier-benchmark.md).

Scale any task to stress larger models via explicit difficulty knobs:

```python
hard = CANONICAL["composite_copy_v2"].scaled(k=64, eval_lengths=(32, 64, 128))
```

Floors are first-class rows and marks are plain-language; the full machinery — floors, marks,
regimes, the validity gate — is Part 1 of [the report](reports/frontier-benchmark.md).

## Using it

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

> **Composite-format note:** the API and HuggingFace backends automatically append
> an output-format instruction for composite-family tasks (e.g. ``composite_copy_v2``)
> so chat models emit the required ``<holder> <value> .`` answer span. Use
> ``--no-composite-format`` to disable it (e.g. for ablations).

To evaluate your own model, implement the ``ModelBackend`` interface and pass it to
``factworld.runner.evaluate_task`` (or wrap any callable in ``FunctionBackend`` as above):

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

See [`docs/USAGE.md`](docs/USAGE.md) for the full backend API reference, API
cost tips, and a custom-backend example. Concrete prompts, gold answers, and real model
mistakes for every task are in [`docs/tasks.md`](docs/tasks.md).

## 2. Benchmarking the frontier

Twelve frontier models through the instrument, two regimes: **instant** (reasoning off, one-line
answer contract — what the weights compute) and **thinking** (effort=high, generous budgets —
what reasoning buys); two models are thinking-only — x-ai/grok-4.5 and muse-spark-1.1 — because
their endpoints cannot disable reasoning, so they carry no instant numbers by design. The two
rankings are near-orthogonal, so profiles are per-axis, never a single scalar. Full narrative,
marks glossary, and the add-a-model path: [`reports/frontier-benchmark.md`](reports/frontier-benchmark.md);
rendered tables and per-cell Wilson intervals: [`docs/benchmark/results.md`](docs/benchmark/results.md).

<!-- FRONTIER_TABLE_START -->
**Instant composition (reasoning off, answer contract)**

| Model | binding @L16 | composed @L16 | composed @L64 | gap |
|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 0.78 | 0.72 | 0.43 | +0.06 |
| anthropic/claude-sonnet-5 | 0.77 | 0.62† | 0.32† | +0.15† |
| deepseek/deepseek-v4-pro | 0.51 | 0.44 | 0.19 | —ᶠ |
| google/gemini-3.5-flash | 0.66* | 0.64* | 0.28* | +0.02* |
| moonshotai/kimi-k2.6 | ≤0.94† | ≤0.77† | ≤0.93† | +0.17† |
| muse-spark-1.1 | n/a | n/a | n/a | n/a |
| nvidia/nemotron-3-ultra-550b-a55b | 0.49 | 0.33 | 0.12 | —ᶠ |
| openai/gpt-5.5 | 0.80 | 0.46 | 0.33 | +0.34 |
| openai/gpt-5.6-sol | 0.82 | 0.65 | 0.33 | +0.17 |
| qwen/qwen3.7-max | 0.51 | 0.24 | 0.08 | —ᶠ |
| x-ai/grok-4.5 | n/a | n/a | n/a | n/a |
| z-ai/glm-5.2 | 0.71 | 0.38† | 0.13 | +0.33† |
| *recency heuristic (floor)* | 0.04 | 0.04 | 0.06 | — |
| *object-filter floor* | 0.41 | 0.41 | 0.15 | — |

**Thinking state-stress (reasoning on)**

| Model | chain d128 | s5 @L256 | s5@128 ctok |
|---|---|---|---|
| anthropic/claude-opus-4.8 | 0.08 | 1.00ʳ | 12683 |
| anthropic/claude-sonnet-5 | 0.04 | 1.00ʳ | 11866 |
| deepseek/deepseek-v4-pro | ⊘ʳ | ⊘ | 10043 |
| google/gemini-3.5-flash | 0.88 | 0.52 | 11022 |
| moonshotai/kimi-k2.6 | 0.64‡ | 0.88 | 17418 |
| muse-spark-1.1 | 0.88ʳ | 1.00ʳ | 9704 |
| nvidia/nemotron-3-ultra-550b-a55b | ⊘ʳ | ⊘ | 12250 |
| openai/gpt-5.5 | 0.36 | 0.96 | 6989 |
| openai/gpt-5.6-sol | 1.00 | n/a | 2657 |
| qwen/qwen3.7-max | 0.96 | 0.80 | 7904 |
| x-ai/grok-4.5 | n/a | 1.00‡ | 8069 |
| z-ai/glm-5.2 | 0.36 | 0.88 | 6282 |
<!-- FRONTIER_TABLE_END -->

The instant columns are `composite_copy_v2` cells (match, n=100): the binding leg (state
tracking), the composed two-hop, and **gap** = binding − composed @L16, the composition
deficit; the thinking columns are the state-stress cells — chain d128 (`chain_nowrap`, k=257: a
128-hop pointer chase) and s5 @L256 (`s5_concrete`: 256 permutation events) — at effort=high,
16,384 tokens, n=25. Marks: `†` visible working or covert reasoning on the canonical attempt,
`≤x†` an explicit upper bound (covert reasoning on most calls), `*` off-arm ran effort=minimal
(cannot disable reasoning), `ʳ` single rerun at a raised 32,768-token budget, `‡` provider
ignored the token cap, `⊘` not measurable at this budget (majority finish=length), `—ᶠ` gap
not interpretable (binding at the object-filter floor), `n/a` cell not run, `—` not applicable
to a floor row. Neither ⊘ nor ≤x† participates in orderings. Escalated instant cells show the
canonical first attempt (their @512tok diagnostics are in the report); the two floor rows are
the shallow baselines every instant cell is read against — the recency heuristic and the
object-filter floor E[1/w] (w = writes to the queried object).

Recall is not the constraint: the recall sanity cells read 0.97–1.00 for every model that runs
them, recall under load (`recall_copy_v1` pool-64 @L64, chance 0.016) reads 1.00 for all ten
instant-measured models, and the scaffolded leg reads 0.98–1.00 (qwen's ⊘ there is a
contract-phrasing artifact, not a recall failure; report Part 2). The gap is a composition
deficit, not a recall one.

Noise bars: the instant test-retest bar is ±0.06 (the replicate leg — prompts identical to the
composed @L16 cell). Thinking cells are n=25 — Wilson intervals ≈ ±0.15–0.19, and the one
thinking test-retest pair moved 0.16 — so thinking differences under ~0.2 are not an ordering.

## 3. Exploring the architectures

The same tasks trained from scratch — transformer, recurrent hybrids (gdp = GatedDeltaProduct,
gdn = GatedDeltaNet), and fprm (Fast Parallel Recurrent Model, a weight-tied looped
conv+attention block) — attribute each capability to an architectural or training component.
Comparisons are matched on compute, not parameters (fprm is weight-tied at ~5–11× fewer params),
at budgets sufficient for the capable configuration to converge.

- **Recall.** Every architecture aces adjacent 1-hop readout; deferred recall needs product
  recurrence — attention-free `gdp_pure` supplies it, `gdn_pure` fails
  ([consolidated §3](reports/factworld-consolidated.md); archived provenance
  [phases/01 §3.2](phases/01-instrument/factworld.md)).
- **Binding under breadth.** fprm leads the binding leg through B16 and breaks at B24, where
  gdp_hybrid holds 0.67; the transformer reads 0.08–0.23 throughout (45 runs, d256).
- **Composition.** The staged-curriculum flagship converges only for gdp_hybrid: composed
  0.833±0.089 @L16 (3 seeds 0.758/0.782/0.958, eval_n=500; holder 0.999 / value 0.833); fprm
  0.109±0.089 — perfect binding (0.998), dead value leg; transformer 0.001, a real floor.
  Scale is non-monotone: convergence peaks at medium d768 (0.732±0.013 corroborates), small
  fails the value leg, large is seed-bimodal
  ([consolidated §5](reports/factworld-consolidated.md)).
- **Chain.** No architecture extrapolates depth (3 archs × 3 seeds): gdp_hybrid fits training
  best yet scores below the guess at held-out depths.
- **s5.** Dense per-step supervision forms the non-abelian circuit in every architecture; only
  the recurrent hybrid extrapolates length
  ([consolidated §8](reports/factworld-consolidated.md)).
- **Commutative.** All three architectures read chance at d256, even with trace supervision —
  order-free aggregation is harder than last-write locally.

The price table: no element is free; each is paid for by an architectural
or training choice, and two rows remain open:

| element | price | evidence |
|---|---|---|
| adjacent (1-hop) recall | attention — every architecture aces adjacent readout | [consolidated §3](reports/factworld-consolidated.md) |
| deferred recall | product recurrence — the transformer aces adjacent, fails deferred (0.19 vs gdp_hybrid 0.73) | [consolidated §3](reports/factworld-consolidated.md); archived provenance [phases/01 §3.2](phases/01-instrument/factworld.md) |
| last-write state | recurrence, ordered by form — fprm (weight-tied looped block) 1.00 @B6, seed-consistent 0.97–0.98 @B16 on the binding leg, over gdp_hybrid over transformer; breaks @B24 where only the gated hybrid holds 0.67 | [report Part 3](reports/frontier-benchmark.md) |
| non-abelian state (formation) | dense per-step supervision — a state checkpoint every ≤2 events; architecture-independent | [consolidated §8](reports/factworld-consolidated.md); [experiments §1](docs/experiments/README.md); archived provenance [phases/02 §4](phases/02-non-abelian-state/report.md) |
| non-abelian state (length extrapolation) | recurrent hybrid — gdp_hybrid 0.75 @L64; fprm and transformer collapse past train length | [consolidated §8](reports/factworld-consolidated.md); [experiments §1](docs/experiments/README.md) |
| depth extrapolation | **open** — no measured choice buys it: trained at chain depths 2–3, no architecture clears the 1/6 guess at depths 4–5 | [report Part 3](reports/frontier-benchmark.md); [experiments §15](docs/experiments/README.md) |
| local composition (value leg) | **open** at the default recipe — value ≤0.17 in all 45 breadth-sweep runs (at/below the 1/pool guess), even on binding-solved seeds of all three architectures; only the staged curriculum at d768 converges it | [report Part 3](reports/frontier-benchmark.md); [consolidated §5](reports/factworld-consolidated.md) |

Depth: [Part 3 of the report](reports/frontier-benchmark.md). Local multi-seed detail and
per-leg decomposition: [the consolidated report](reports/factworld-consolidated.md). Running
log: [`docs/experiments/README.md`](docs/experiments/README.md).

## Reports and prior work

- 📊 [`reports/frontier-benchmark.md`](reports/frontier-benchmark.md) — the full report:
  the instrument (Part 1), the frontier benchmark (Part 2), the architecture exploration
  (Part 3).
- 📄 [`reports/factworld-consolidated.md`](reports/factworld-consolidated.md) — local
  multi-seed detail: per-leg decomposition, supervision-density experiments, and the API
  reasoning analyses.
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

## Repository layout

```
phases/                  prior work, archived (ran on the atomic-token format)
  01-instrument/           original FactWorld paper (.md + .pdf)
  02-non-abelian-state/    non-abelian state-tracking report + reproduction kit
docs/
  tasks.md                concrete prompts, gold answers, and real model mistakes for every task
  USAGE.md                backend API reference and custom-backend examples
  related-work.md         related work with verified citations
  results.md              4-arch reference baselines (match metric)
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
    scale.md                archived k=5 compute-matched scale + LR sweeps (retired composite_copy_scale_v1; distinct from the consolidated report's §5 compute-matched scale sweep)
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
`factworld/models.py`. Here n_h is the number of Householder reflections per DeltaProduct step,
and neg-eig is the negative-eigenvalue parameterization ([−1, 1] state-transition spectrum).

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

# State-tracking dissociation                                -> docs/state-tracking/dense-supervised.md
python scripts/dense_s5.py --group s5     # S5 matrix: gdp_pure / n_h=1 null / gdn / transformer / gru, 3 seeds
python scripts/dense_s5.py --group a5     # A5 not-S5-specific control panel

# Recall                                                     -> docs/results-ci.md, docs/recall/readout.md
python scripts/ci_dissociation.py         # recall_copy_v1 + binding_v2, 4 archs x 3 seeds (pool-2 dissociation CIs)
python scripts/recall_attention_test.py   # attention-free attribution: gdp_pure / gdn_pure / gdp_hybrid across pools 2..8
python scripts/recall_fair.py             # the 1-hop-vs-deferred differential (onehop/defsep/defpad; n_heads 4 vs 8)

# Composition gap                                            -> docs/results.md, docs/results-ci.md, docs/composition/results.md
python scripts/collect_baselines.py       # 4-arch from-scratch reference baselines, all scored tasks (seed 0)
python scripts/sk_composite.py            # memorization diagnostic + the n_h in {1,2,4} fixed-param mechanism control
python scripts/iso.py                      # the n_h ∈ {1,2,4} product-structure ablation at fixed params (neg-eig on/off)
python scripts/decompose.py               # the gap decomposed: state leg vs recall leg + routing on holder-wrong examples
python scripts/experiment_composite_scale.py  # compute-matched scale sweep: small/medium/large × gdp/fprm/transformer

# Scale + the matched LR sweeps                              -> docs/state-tracking/scale.md
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
