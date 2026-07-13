# FactWorld: A Reproducible Instrument for Composing State Tracking and Recall

FactWorld is a single, frozen benchmark suite that evaluates frontier models through an API and
local architectures trained from scratch. The same tasks, oracle, and decomposition metrics are
used in both modes. It is one instrument for studying how recall, state tracking, and
composition interact; every number reproduces from committed scripts and can be checked with an
API key or a single GPU.

**Scope.** FactWorld is a *mechanism probe for the component capabilities that agent workloads
depend on* — working-memory recall, state tracking, and multi-step composition — not an end-to-end
agent benchmark. Every task is single-turn and single-answer-span, with no tool use, planning, or
multi-turn action. The component→agent mapping is a motivating analogy, not a proven one (§9).

## 1. What the instrument is

The suite is built around versioned `TaskSpec` objects in `factworld.tasks.CANONICAL`. Each task
renders to natural language, carries deterministic examples from a fixed seed, and is scored by a
single canonical metric: **match** of the answer span (strip trailing punctuation and score the first `len(gold)` tokens). Gold answers come
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
holder-wrong/value-right means the reverse. The decomposition localizes which sub-behavior fails
and under what intervention.

Match is the cross-regime metric: it strips a trailing period and checks the first
`len(gold)` tokens, so it handles API models that omit the period and local models that emit the
correct answer and then continue generating. Where the output is clean, match coincides with
an exact-string scorer and with last-N extraction. `APIBackend` strips `<think>` blocks and
common prefixes before scoring; local training runs use the same scorer. The head-to-head
comparisons below use match for both API and local models.

## 2. The tasks

**Which task version each number is on.** The worked example below and the §4 grid are on the
retired v1 give-stream family — historical snapshots, kept as diagnostics (the recency
methodological note in [frontier-benchmark.md](frontier-benchmark.md) covers why the v1 sampler
was defective). The §5 local tables are on v2
(the trace-free 3-seed staged-curriculum measurement, corroborated by the compute-matched scale
sweep — see §5). The §4 dose-response, the §6 scaffolded numbers, and the §7 API
sweep are on v2 (glm to L1024; kimi at L256 and L512). The §7 local table is not
recency-exposed: it comes from `experiment_dense_supervision.py`, whose streams are swap/cycle permutations
(`world.sample_hard_chain`), not the give-stream, so the v1 sampler defect never touched it.
Cross-model claims defer to the frontier benchmark
([`frontier-benchmark.md`](frontier-benchmark.md)), which is on v2.

**Composition (`composite_copy_v2`)** is the flagship probe — the state × recall composition.
The worked example below is the retired v1 spec's rendering (same knobs and format; the samplers
differ — see the version note above). A set of facts maps agents to values,
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

**S₅ (`s5_v1`)** is non-abelian state tracking. A stream of `swap` and `cycle` events permutes the
roles of a set of agents, and the query asks for one agent's final role. Unlike last-write-wins,
the running permutation must be carried step by step.

**The remaining tasks**, grouped as components and compositions per the taxonomy
([`AGENTS.md`](../AGENTS.md)):

| task | role | behavior | difficulty axis |
| --- | --- | --- | --- |
| `recall_copy_v1` | component: recall | single-query, deferred-readout MQAR variant | pool breadth |
| `conflict_v1` | component: recall (parametric variant) | parametric ↔ in-context override | memorized map vs. context |
| `binding_v2` | component: state tracking | last-write-wins (absorbing updates — not group ops) | give-stream length |
| `chain_v1` | composition: recall ∘ recall | pointer chase at fixed breadth | depth |

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
length.

## 4. Evaluating frontier models

For frontier-model comparisons, the canonical source is the recurring benchmark report,
[`frontier-benchmark.md`](frontier-benchmark.md): a composition instrument run in an instant
(no-thinking) and a thinking regime on the de-skewed v2 tasks, with censoring and cleanliness
marks and per-cell provenance. The report reads the component legs (recall, binding) against the
composed two-hop cell at L16/L64, with a per-model composition gap (binding@L16 − composed@L16),
shallow-heuristic floors as first-class rows, and two state-stress rows under reasoning (chain
d128 at fixed k=257, s5 @L256). It supersedes the OpenRouter grid in this section for
cross-model claims; the grid below remains the pretrained-model snapshot on the v1 tasks. The
dose-response below is on the de-skewed v2 sampler.

The default setup for API evaluation is:

- `max_new_tokens=2048` — enough for reasoning models to finish a scratchpad.
- No early stop (`stop_at=None`) — truncating at `.` would cut off reasoning traces before the
  answer.
- Composite format instruction in the system prompt — tells the model to emit `<holder> <value>`.
- `APIBackend` normalizes the output before scoring: it strips `<think>` blocks, common prefixes,
  and detached trailing periods, then extracts the final answer span.

On the retired v1 flagship `composite_copy_v1@L16` (pool-16 in-context recall × last-write-wins
binding, length 16), n=30, greedy, match:

| model | match |
| --- | --- |
| glm-5.2 | 0.867 |
| kimi-k2.6 | 0.867 |
| llama-3.3-70b-instruct | 0.767 |

These runs use `max_new_tokens=2048` and no early stop so reasoning models can finish their
scratchpad; `APIBackend` extracts the final answer span. A short generation budget would truncate
the scratchpad before the answer.

**Reasoning moves composition, monotonically.** On `composite_copy_v2` @L16 (n=50 per cell,
answer-span extraction with a holder/value decomposition;
`results/reasoning_sweep_20260710_125924.jsonl`), reasoning effort gives a clear dose-response
for both models measured:

| model | none | low | medium | high |
| --- | --- | --- | --- | --- |
| kimi-k2.6 | 0.72 | 1.00 | 1.00 | 1.00 |
| glm-5.2 | 0.38 | 0.92 | 0.96 | 0.98 |

At effort=none the holder leg reads kimi 0.42 / glm 0.22 against the v2 object-filter floor of
0.41 — object filtering, not established composition (kimi's effort=none arm carries the covert
reasoning caveat from the frontier benchmark). Low effort already recovers most of the
composed cell (0.92–1.00), and the curve is monotone through high. The lever is implicit
reasoning: an explicit "write the holder, then the value" instruction hurts every model,
including the reasoners that solve the composed cell under a plain prompt (format-fair ablation
on `composite_copy_v1`@L16, n=100;
[`docs/experiments/autoregressive-api-results.md`](../docs/experiments/autoregressive-api-results.md)
E1b, `results/autoregressive_formatfair_20260626_224937.jsonl`).

**S₅ is movable by reasoning — under a concrete rendering.** In the standard grid below (token
rendering, no reasoning) S₅ sits at floor (~0.18). Allow reasoning *and* render the problem
concretely (people and jobs, initial assignment stated), and a strong reasoner solves it: GLM-5.2
holds 1.00 at L32, 0.97 at L64, and 0.90 at L128, degrading gradually with length
(`results/s5_horizon_recheck_20260705.jsonl`, 8192-token budget). The lever is the
combination — reasoning under the token rendering leaves GLM at ~0.33, and a concrete rendering
without reasoning leaves it at chance. Appendix A gives the full curve, the reasoning × rendering
interaction, and the per-example failure mode.

The wider API capability grid below uses the standard zero-shot setup (default decoding,
`max_new_tokens=16`, `stop_at="."`) — no reasoning — for all tasks, except that the
`composite_copy_v1` cells for glm-5.2, kimi-k2.6, and llama-3.3-70b-instruct use the long-token
setup above. S₅ floors in this no-reasoning grid but is movable by reasoning under a concrete
rendering (paragraph above; Appendix A); `chain_v1` floors here but is reasoning-solvable at its
designed depths under an 8192-token budget (paragraph after the table). The `chain_v1` column
runs at depth 16 — past the task's design gate — so it is a wrapped-task floor check, not a
depth score (wrap analysis after the table).

| model | recall_copy_v1 | conflict_v1 | binding_v1 | chain_v1 d16 (wrapped) | composite_copy_v1 | s5_v1@L16 |
| --- | --- | --- | --- | --- | --- | --- |
| glm-5.2 | 1.000 | 1.000 | 0.767 | 0.133 | 0.867 | 0.200 |
| kimi-k2.6 | 1.000 | 1.000 | 0.633 | 0.033 | 0.867 | 0.200 |
| llama-3.3-70b-instruct | 1.000 | 1.000 | 0.633 | 0.000 | 0.767 | 0.200 |
| gemini-2.5-flash-lite | 1.000 | 1.000 | 0.300 | 0.100 | 0.233 | 0.167 |
| deepseek-chat | 1.000 | 1.000 | 0.367 | 0.033 | 0.167 | 0.133 |
| gpt-4o-mini | 1.000 | 1.000 | 0.367 | 0.067 | 0.133 | 0.200 |

Single-hop recall and conflict are solved across the board. Binding and composition separate the
stronger models. Depth-*k* composition (`chain_v1`) sits near floor in this no-reasoning grid,
but with reasoning and `max_new_tokens=8192` glm-5.2 solves it at its designed depths — 1.00 at
depth 4, n=30 (`results/chain_reasoning_pilot_20260705.json`). `chain_v1` builds a single k=6
pointer cycle and measures depth only at depths < k (`factworld/tasks.py`: "Depths stay < k so
the cycle never wraps"); at depth ≥ 6 the chain wraps: gold returns to the start agent at
multiples of 6, and effective difficulty is depth mod 6. The deeper cells (the grid column
above at depth 16, and the depth 6–32 cells in the pilot files) therefore exercise the wrapped
task and are not depth results. Measuring depth beyond k is the axis of the scaled no-wrap
variant (`chain_nowrap`). Non-abelian state tracking (`s5_v1`) floors here but is movable by reasoning
under a concrete rendering (Appendix A). Reasoning-model cells require a large completion
budget: a short budget truncates the reasoning trace before the answer and reads as an empty
prediction.

## 5. Evaluating local architectures

The same tasks, trained from scratch, let us ablate architectures and training regimes that an
API model hides. We use a staged curriculum:

- `gdp_hybrid`, `fprm`, and `transformer`
- d_model=768, n_layers=8, batch=128
- 25k steps total, 80k docs total
- 3 seeds, evaluated on n=500 test examples per seed
  (`results/curriculum_staged_v2_d768_notrace.jsonl`; the specs are the v2 port —
  `composite_copy_v2`, uniform last-write sampler)

`gdp_hybrid` is a `[recurrent, recurrent, attn, recurrent]` GatedDeltaProduct stack. `fprm` is a
weight-tied looped conv+attention block. The transformer is a standard decoder-only model.

On the flagship task `composite_copy_v2` pool-16 @L16, match:

| model | params | per-tok FLOPs | composite_p16@L16 match |
| --- | --- | --- | --- |
| **gdp_hybrid** | 101M | 204 GFLOP | **0.833 ± 0.089** |
| fprm | 10M | 159 GFLOP | 0.109 ± 0.089 |
| transformer | 76M | 159 GFLOP | 0.001 ± 0.001 |

The v1 flagship (0.747 ± 0.174, 3 seeds, eval_n=500, retired sampler) reproduces on v2 for
`gdp_hybrid` — 0.758 / 0.782 / 0.958 across seeds, contains ≈ match, holder leg ≥ 0.998 on all
three. p(converge) at the ≥0.9 bar is 1/3 for `gdp_hybrid` — the composed cell trains on all
three seeds, with two landing at 0.758 / 0.782 — and 0/3 for the rest. The v1
number does not reproduce for `fprm`: its v1 0.253 ± 0.178 collapses to 0.109 on the de-skewed
sampler (the holder leg still solves at ≥ 0.99; the value leg dies). The transformer stays at
floor. The medium cell of the compute-matched scale sweep — the same recipe on 2 independent
seeds at eval_n=200 — reads 0.732 ± 0.013 / 0.033 ± 0.012 / 0.005 ± 0.005, corroborating both
the ordering and the per-leg decomposition.

*Protocol note.* The dedicated 3-seed/eval_n=500 curriculum re-measure
(`results/curriculum_staged_v2_d768.jsonl`) was launched with `--use_trace`, and under trace
training the model emits a self-trace before the answer — the prefix-committed match metric is
structurally 0 for any trace-emitting model while `contains` stays high (gdp p5 0.981) purely
from containment leniency over the longer emission. This reproduces the known v1 trace-mode
control (`results/curriculum_staged_d768_b64_80k_trace.md`: composite 0.00) rather than
measuring the flagship; composite capability is unmeasurable under that protocol, so those runs
are excluded (see `results/remeasure_v2/PENDING.md`). The table above is the identical command
re-run without `--use_trace` — the trace-free v2 measurement.

All three share `(d_model=768, depth=8)`; the match is on compute, not parameters. `fprm` is a
weight-tied looped block (one `FPRMBlock` applied `n_loops` times — see `factworld/models.py`), so
at matched `(d_model, depth)` its per-token FLOPs equal the transformer's (~159 GFLOP) while its
parameter count is ~8× lower at this medium scale (10M vs 76M). `gdp_hybrid`'s Householder-product recurrence costs
~1.25× the transformer's FLOPs (204 vs 159 GFLOP). Params are measured with the tied head/embed
counted once; per-token FLOPs are forward-pass, measured with torch's flop counter (it captures the
`fla` layers). A compute-matched
scale sweep that scales `(d_model, depth)` together across small/medium/large for all three
architectures is in `results/composite_scale_*.md`.

For local models, an exact-string scorer and last-N extraction differ from match only on
formatting (a missing trailing period or extra trailing generation). On the v2 runs: `gdp_hybrid`
match = 0.758 / 0.782 / 0.958 per seed, last-N extraction = 0.00 on all three, and contains ≈
match (0.768 / 0.788 / 0.958). The zero last-N-extraction scores show that local models often
append extra tokens after the answer, so the canonical metric is prefix-based; contains ≈ match
confirms the score is content, not formatting.

The local `gdp_hybrid` is competitive with the API models on this task, despite being trained from
scratch at ~100M params / 204 GFLOP/token. `fprm` and the transformer fail to learn the task even
with the staged curriculum at this scale.

**Per-leg decomposition** explains the ranking (content-token accuracy, independent of the
period issue):

| arch | holder (binding) | value (recall) | overall |
| --- | --- | --- | --- |
| gdp_hybrid | 0.999 | 0.833 | 0.833 |
| fprm | 0.998 | 0.109 | 0.109 |
| transformer | 0.065 | 0.041 | 0.001 |

`gdp_hybrid` solves the binding leg and does most of the recall leg. `fprm` solves the binding
leg but fails to recall the value of the resolved holder. The transformer fails both legs.
This is the same routing deficit the API models hit: even when the holder is correct, the model
must route that holder into the in-context recall lookup.

### Scale robustness (compute-matched sweep)

A natural objection to §5 is that the transformer was never given a fair size. We test that
directly: the same staged curriculum and eval at three sizes, with the comparison matched on
compute, not parameters (all architectures share `(d_model, depth)`; `fprm` is weight-tied so its
FLOPs match the transformer's at ~5–11× fewer params across scales — see the size table in
`results/composite_scale_*.md`). `composite_copy_v2` pool-16 @L16, match, 2 seeds,
`train_n=80000` (the medium column is the §5 flagship recipe on 2 seeds/eval_n=200; it
corroborates the 3-seed flagship table above):

| arch | small (384×6) | medium (768×8) | large (1024×12) |
| --- | --- | --- | --- |
| **gdp_hybrid** | 0.12 ± 0.08 | **0.73 ± 0.01** | 0.21 ± 0.21 |
| fprm | 0.12 ± 0.05 | 0.03 ± 0.01 | 0.03 ± 0.02 |
| transformer | 0.01 ± 0.00 | 0.01 ± 0.01 | 0.00 ± 0.00 |

(Per-scale params/FLOPs: small ~3–19M / ~32–38 GFLOP/token; medium ~10–101M / ~159–204;
large ~18–269M / ~417–540. Raw runs + the holder/value decomposition are in
`results/composite_scale_*.md`.)

- **Convergence is arch-specific and scale-dependent, peaking at medium.** `gdp_hybrid` at
  medium is the only cell that converges the composed task (0.720 / 0.745 per seed, holder 1.00
  on both). At small, `gdp_hybrid` solves binding (holder 1.00 both seeds) but the value leg
  fails (0.045 / 0.200) — the v1 small cell's 0.98 ± 0.01 was flattered by the retired
  recency-defective sampler, consistent with the d256 breadth sweep (value ≤ 0.17 in all 45
  runs). `fprm`'s composed cell is weak at small (0.120 ± 0.045) and dies at medium and large
  (0.033 / 0.028) even though its binding leg solves through medium (0.99).
- **The transformer floors at every scale, including 202M params / 417 GFLOP/token** — contains ~0
  as well, so this is a real floor, not a formatting miss; §5's "transformer fails" is not a
  small-model artifact, and compute-matching does not rescue it.
- **Large is seed-bimodal for `gdp_hybrid`, and the failure is genuine.** Seed 1 scores 0.000
  with holder 1.000 and contains 0.000 — the gold value token appears nowhere in the emission, a
  pure value-leg failure. Seed 0's 0.42 comes with the holder leg degraded to 0.68 (the batch-64
  large recipe trains unstably), so it is not partial convergence at scale. With 2 seeds this is
  a flag, not a measurement: characterizing the large regime (more seeds, an LR study at 269M)
  is the open follow-up. It is *not* the transformer catching up — the transformer remains at
  floor (0.00).
- **The routing deficit is scale-invariant where binding trains.** Wherever binding is solved
  and the composed cell fails (`gdp_hybrid` at small and at large seed 1; `fprm` at small,
  medium, and large seed 0), the value (recall-of-resolved-holder) leg is what collapses — the
  same deficit §6 localizes.

## 6. Composition of behaviors

The central design choice is the per-leg decomposition. A composite example requires two
distinct computations:

1. Track the give-stream to resolve the holder (state tracking / binding).
2. Read the holder's value from the fact list (in-context recall).

By scoring each leg independently, the suite can localize failures and test interventions:

- **Ceiling probe:** give the model the correct holder and ask only for the value. If recall is
then perfect, the deficit is state tracking, not recall.
- **Scaffolded eval:** the oracle provides the holder; the model generates only the value. This
measures recall-of-the-resolved-holder in isolation.
- **Binding-only eval:** ask only "who holds the object?" to measure state tracking without recall.

On the local `gdp_hybrid` model (the §5 flagship run;
`results/curriculum_staged_v2_d768_notrace.jsonl`), the scaffolded value score is low (mean
0.096, range 0.074–0.108 across seeds), which suggests the routing problem is real even when
binding is solved. On API models, the scaffolded result is much stronger: given the correct
holder, the frontier roster recalls the value at 0.98–1.00 on the `composite_copy_v2` items
(n=100 per model, instant protocol; 1.00 for seven of the ten that run it, nemotron 0.99, kimi
0.98). qwen3.7-max is ⊘ on this leg — empty on 98 of 100 calls, a contract-phrasing interaction
localized by a four-arm probe (`results/qwen_scaffold_probe/`), not a recall failure. The
difference is that the API models can do each leg when the problem is split for them, but
struggle to compose the two legs in the end-to-end prompt.

## 7. Long context

Both regimes are stressed well past their calibration points — trained models evaluated to 32×
their training length, pretrained models evaluated from L16 to L1024.

**Trained recurrent hybrids extrapolate far.** This table is the dense-supervised (K=1)
non-abelian composite at d256×4 (`experiment_dense_supervision.py`;
`results/longctx_gdp_20260627_223033.md`, `results/longctx_fprm_20260628_000834.md`) —
swap/cycle permutation streams via `world.sample_hard_chain`, a different task and scale from
the §5 comparison, and untouched by the v1 give-stream recency defect. Stressed to 32× the
trained length (8 seeds):

| arch | L64 | L128 | L256 | L512 |
| --- | --- | --- | --- | --- |
| gdp_hybrid | 0.62 | 0.59 | 0.51 | 0.46 |
| fprm | 0.26 | 0.18 | 0.20 | 0.15 |

The recurrent hybrid holds ~0.5 out to L512; the looped block stays at floor.

**The thinking regime holds composition at long context.** On the de-skewed v2 sampler at
k=32/pool16 (match, n=25 per cell; thinking = effort=high with a 16,384-token budget
through L256 and 32,768 at L512+; instant = effort=none with the answer contract;
`results/composite_frontier_20260709.jsonl`, `results/composite_frontier_20260710.jsonl`):

| model / arm | L64 | L128 | L256 | L512 | L1024 |
| --- | --- | --- | --- | --- | --- |
| glm-5.2 thinking | 0.98 | 0.98 | 0.94 | 0.96 | 0.94 |
| glm-5.2 instant | 0.24 | 0.02 | 0.00 | 0.06 | 0.06 |
| kimi-k2.6 thinking | n/a | n/a | 1.00 | 0.96 | n/a |
| *object-filter floor* | 0.14 | 0.08 | 0.05 | 0.02 | 0.01 |

Thinking holds 0.94–1.00 out to L1024 while the instant arm sits at or below the object-filter
floor from L128 on — at this breadth, length is not the binding constraint for the thinking
regime (the doubled-breadth rung k=64/pool64 @L1024 reads 0.64 as a budget-censored lower
bound). Kimi's measured cells are L256 (1.00, Wilson 95% [0.87, 1.00]) and L512 (0.96,
[0.80, 0.99]), both with empty rate 0.00 and no budget censoring; its unmeasured lengths are
predicted-ceiling cells and stay unbought (the spend rule in [AGENTS.md](../AGENTS.md): never
buy cells predicted to sit at ceiling or floor). S₅ under a
concrete rendering with reasoning holds 0.90 at L128 (Appendix A), degrading gradually rather
than abruptly; its concrete-rendering sweep extends to L128, short of composition's
L1024. Both tasks are reasoning-recoverable under length stress, with task- and model-dependent
limits.

## 8. S₅ is movable by supervision density (the local, from-scratch regime)

This section is about *training a small model from scratch* on S₅. Frontier inference is a
different regime — there S₅ is movable by reasoning under a concrete rendering (§4, Appendix A).
Here, S₅ floors for every architecture under answer-only supervision. It moves when the training
signal carries the state. We interleave the oracle's holder-of-the-queried-role every *K* events
into the training stream (K=1 is dense), and evaluate free-running. 10 seeds:

| K (stride) | value @L16 | value @L64 | converge @L16 |
| --- | --- | --- | --- |
| 1 (dense) | 1.00 | 0.75 | 10/10 |
| 2 | 0.98 | 0.40 | 10/10 |
| 4 | 0.19 | 0.20 | 0/10 |
| 8 | 0.21 | 0.20 | 0/10 |

The circuit forms reliably down to a checkpoint every other step and is gone below. This is a
sharp learnability threshold. Dense-arm @L64 values differ across independent trainings because
the L64 extrapolation is seed-bimodal (0.75 ± 0.22 here, 10 seeds): the 3-seed repro in §3 lands
at 0.90, the 8-seed weaning reference below at 0.61, and §7's 0.62 is a different task and scale
(noted there). The per-architecture comparison (fprm 0.19, transformer 0.22 @L64; 5 seeds, K=1)
is in [experiments §1](../docs/experiments/README.md).

**The circuit survives weaning to label-free deployment.** Train dense, then fine-tune on a mix
of densities including answer-only, and evaluate free-running. 8 seeds:

| arm | L16 | L64 | L128 | converge |
| --- | --- | --- | --- | --- |
| dense only (reference) | 1.00 | 0.61 | 0.50 | 8/8 |
| weaned (mixed density) | 1.00 | 0.50–0.54 | 0.46–0.48 | 8/8 |
| answer-only (never dense) | 0.19 | 0.19 | n/a | 0/8 |

The weaned circuit converges 8/8 free-running with no deploy-time labels and extrapolates on par
with dense-only.

## 9. Discussion

FactWorld is one instrument with two uses. The same composition probe that separates GLM,
Kimi, and llama-3.3-70b via API (the `composite_copy_v1` snapshot, §4) also separates
`gdp_hybrid`, `fprm`, and `transformer` trained locally (`composite_copy_v2`, §5). The per-leg
decomposition links the two regimes: a finding about "routing the resolved holder into recall"
can be checked in both settings.

The main findings:

- **Composition responds to reasoning.** It is movable at inference in the thinking regime,
including at long context (0.94–1.00 out to L1024), for models that can reason. Depth-*k*
composition (`chain_v1`) follows at its designed depths (< k=6, before the pointer cycle wraps):
with reasoning and an 8192-token budget glm-5.2 solves it where the no-reasoning grid floors;
depth behaviour beyond k is measured by the scaled no-wrap variant (`chain_nowrap`), not by
`chain_v1`. Explicit prompting does not substitute.
- **Non-abelian state tracking also responds to reasoning — but only under a concrete rendering,
and with a model-dependent length limit.** GLM-5.2 solves `s5_v1` with reasoning plus a concrete
(people/jobs) rendering — 1.00 at L32, 0.97 at L64, 0.90 at L128 — while kimi-k2.6 degrades
sooner (1.00 at L16, 0.83 at L32). Neither reasoning under the token rendering (~0.33) nor a
concrete rendering without reasoning (~chance) suffices — the combination does. Composition
holds 0.94–1.00 out to L1024.
- **For local from-scratch models, S₅'s lever is supervision density.** Dense per-step state
supervision develops a length-extrapolating circuit that weans to label-free deployment (§8) — a
distinct regime from frontier inference.
- **Architecture carries length generalization.** A learned state circuit generalizes in length
only on a recurrent hybrid; transformers and looped blocks shortcut.

These are results within the regime tested (§10); they are not scaling laws. FactWorld is a
mechanism probe for the component capabilities agents depend on, not an agent benchmark: the
component→agent connection is a motivating proxy, not a proven mapping, and no task here
exercises tool use, planning, or multi-turn action.

## 10. Limitations and related work

**Limitations.** The scale regime is bounded (k=5 S₅; local models ~3–269M params, matched on
compute at ~32–540 GFLOP/token rather than on parameters — §5; pretrained models from a few B
to ~1T params, MoE and dense).
Composition is 2-hop throughout. The API grids and sweeps in this report run small samples
(n=25–50 per cell) because API costs scale with reasoning tokens; cross-model claims defer to
the frontier benchmark ([frontier-benchmark.md](frontier-benchmark.md)), which runs n=100 on the
instant battery and n=25–50 elsewhere, with Wilson 95% intervals per cell.
The natural-language format differs from the atomic-token format
used in prior work on this instrument; absolute numbers are not comparable across formats, though
the mechanism conclusions reproduced.

**Related work.** The `fprm` architecture is a probe inspired by Movahedi et al. (2026), who
report strong S₅ word-problem length generalization with a looped-transformer architecture (FPRM) that
uses a causal 1-D convolution / unroll-to-convergence mechanism; our `fprm` is a weight-tied
variant of that block, and we did not run their model. Its v2 measurement is the breadth sweep
in [frontier-benchmark.md](frontier-benchmark.md): fprm leads the binding leg through B16 and
breaks at B24. The
shortcut-learning and length-extrapolation results engage a substantial literature on
transformer state-tracking brittleness (Liu et al., 2023) and recurrent extrapolation, which we
extend rather than survey.

### Provenance: what the archived phases established

Prior work on this instrument ran on an earlier atomic-token format
([`phases/`](../phases/)); mechanism conclusions carry across formats, absolute numbers do not.

**Phase 1** ([`phases/01-instrument/factworld.md`](../phases/01-instrument/factworld.md)) is the
instrument's provenance. Its oracle/no-leak render↔parse contract and validity gate (§1
contribution list; §2 "validity gate") still certify the current suite via
`scripts/validate_suite.py`, including the design choices the gate forced (identity initial role
assignment, touched-object queries, shared opaque value vocabulary). Its single-capability
dissociations — §3.1 (S₅/A₅ extrapolation under dense per-token supervision: product recurrence
extrapolates, transformer collapses) and §3.2 (attention aces adjacent/1-hop recall while
deferred readout needs product recurrence) — are reproduced on the natural-language format in §3
above, which supersedes the phase numbers. Its §4 resolve-then-recall pipeline finding is the
mechanism ancestor of this report's per-leg decomposition. Its §5 scale numbers are on a retired
task and metric and are superseded outright (the compute-matched comparison is §5 above:
10/76/101M params, matched on FLOPs, not the "~45M matched" framing of earlier drafts).

**Phase 2** ([`phases/02-non-abelian-state/report.md`](../phases/02-non-abelian-state/report.md))
is the provenance of the s5 supervision-density row. Its §4 density sweep (a state checkpoint
every ≤2 events, floor below) reproduces on the natural format (§8 above;
[`docs/experiments/README.md`](../docs/experiments/README.md) §1, where it is also shown
architecture-independent) — that reproduction is what makes the row format-independent. Its §3
"recall is free given a resolved pointer" (P(value|holder✓)=1.0) is the archived ancestor of the
frontier benchmark's scaffolded-leg argument. Levers not yet re-measured on the natural format
stay citable with atomic-token scoping: the scale ladder to 357M (capacity is not the
extrapolation lever; §5), the training-length-distribution threshold and
concentration-beats-coverage (§6), post-training deep-state coverage to ≈8× label-free with
clean-base selection (§6.1), the outcome-reward GRPO null (§4 — evidence the density requirement
spans training paradigms), and the CWM code-rendering bridge (§4.1 — the difficulty is the
computation, not the surface).

**One phase-2 headline does not carry.** Phase 2 reported that weaning internalizes the circuit
but never extrapolates. This report's §8 weaning result contradicts that on the natural format:
8/8 seeds converge and extrapolate on par with dense-only (L64 0.50–0.54). The reconciliation is
scope: the phase claim was measured on its atomic-token composite with a different supervision
target, and stays scoped there; §8 is the current claim.

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
python scripts/eval_model.py composite_copy_v2 --backend api \
    --model z-ai/glm-5.2 --n 30 --no_stop

# Grid of OpenRouter models (set OPENROUTER_API_KEY)
python scripts/eval_openrouter_grid.py --n 30

# Disable reasoning to confirm the collapse
python scripts/eval_openrouter_grid.py \
    --models moonshotai/kimi-k2.6 z-ai/glm-5.2 meta-llama/llama-3.3-70b-instruct \
    --tasks composite_copy_v2 --n 30 --no_reasoning
```

The §4 numbers on the retired v1 specs reproduce by substituting the retired task name — the
eval CLIs accept `tasks.RETIRED` names for historical reproduction only, never for scored runs.

**Train and evaluate local architectures with the staged curriculum:**

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

- API reasoning-effort dose-response (`composite_copy_v2`, de-skewed sampler):
  `results/reasoning_sweep_20260710_125924.jsonl`
- API long-context composition (thinking vs instant):
  `results/composite_frontier_20260709.jsonl`, `results/composite_frontier_20260710.jsonl`
- API chain reasoning pilot: `results/chain_reasoning_pilot_20260705.json`
- API S₅ concrete-rendering runs: `results/s5_horizon_recheck_20260705.jsonl`,
  `docs/openrouter/s5-{length-sweep,framing,framing-reasoning,horizon}.jsonl`
- API v1-task grid: `docs/openrouter/results-natural-longctx2k-composite.jsonl`
- API no-reasoning collapse: `docs/openrouter/results-natural-kimi-noreasoning.jsonl`,
  `docs/openrouter/results-natural-llama-glm-noreasoning.jsonl`
- Local flagship (match, 3 seeds, trace-free):
  `results/curriculum_staged_v2_d768_notrace.jsonl`
- Local compute-matched scale sweep: `results/composite_scale_*.md`
- Local long-context stress: `results/longctx_gdp_20260627_223033.md`,
  `results/longctx_fprm_20260628_000834.md`
- V1-task runs (retired sampler; source of the §5 v1 comparison numbers):
  `results/reasoning_sweep_20260627_092034.jsonl`,
  `results/reasoning_glm_20260627_114244.jsonl`,
  `results/reasoning_longctx_L128_20260628_163121.jsonl`,
  `results/reasoning_longctx_L256_20260628_171119.jsonl`,
  `results/reasoning_longctx_L512_20260628_181508.jsonl`,
  `results/benchmark_gdp_d768_b128_80k_500eval.json`,
  `results/benchmark_fprm_d768_b128_80k_500eval.json`,
  `results/benchmark_transformer_d768_b128_80k_500eval.json`,
  `results/benchmark_lastn_gdp_full_1seed.jsonl`,
  `results/benchmark_lastn_fprm_full_1seed.jsonl`

## Appendix A. S₅ — what frontier models can track, and where it breaks

S₅ (`s5_v1`) is non-abelian state tracking (§2). Without reasoning it floors at every length; with
reasoning under a concrete rendering a strong model solves it, degrading gradually with length
(GLM-5.2 holds 0.90 at L128). This appendix gives the no-reasoning failure mode, the reasoning ×
rendering interaction that unlocks the task, and how far it holds with length. Data:
`docs/openrouter/s5-{length-sweep,framing,framing-reasoning,horizon}.jsonl`,
`results/s5_horizon_recheck_20260705.jsonl`; scripts: `scripts/experiment_s5_framing.py`,
`scripts/analyze_s5_tracking.py`. All cells n=30, match, chance floor 0.20.

### A.1 The computation

State is an assignment `{agent → role}`. Each event is a permutation on roles: `swap gX and gY`
transposes two agents' roles; `cycle gA -> gB -> …` rotates roles one step along the arrow. The
query asks for one agent's final role after *L* events. Swaps and cycles do not commute, so the
running permutation must be carried step by step — there is no algebraic shortcut. Two renderings
are used below: the **token** rendering (`g`/`r` IDs, "swaps"/"cycles roles", initial assignment
stated) and a **concrete** rendering (people and jobs: "Eva and Bob swap jobs", "Cara takes Eva's
job, …", "what job does Cara have?"). Both encode the *same* permutation sequences and the same
oracle gold.

### A.2 Without reasoning: track-then-stall, then chance

Without reasoning, models do real step-by-step tracking on individual examples, then **stall** —
they report a role the queried agent held at a recent step rather than the final one. Worked
example (GLM-5.2, L16; the queried agent g2's correct role changes 7 times):

```
init r2 · s1 r1 · s3 r3 · s7 r4 · s8 r0 · s9 r4 · s12 r1 · s14 r2 (final)
```

GLM answered **r0** — exactly g2's role at **s8** — missing the three later updates. Three failures,
same shape (GLM's answer = a recently-held role, not the final):

| queried | GLM said | that role was the agent's state at… | gold |
| --- | --- | --- | --- |
| g2 | r0 | s8 (then g2 moved r0→r4→r1→r2) | r2 |
| g2 | r3 | s8 / s12 (last held s12; missed s13) | r4 |
| g4 | r0 | s14 (missed only the final s15) | r1 |

In aggregate this is chance at every length — there is no degradation gradient because accuracy
never rises above it. No reasoning, token rendering:

| model | L4 | L8 | L16 | L32 | L64 | L128 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glm-5.2 | 0.07 | 0.10 | 0.30 | 0.20 | 0.30 | 0.30 | 0.21 |
| kimi-k2.6 | 0.13 | 0.07 | 0.20 | 0.33 | 0.10 | 0.10 | 0.16 |
| llama-3.3-70b | 0.10 | 0.10 | 0.23 | 0.13 | 0.10 | 0.03 | 0.12 |
| deepseek-chat | 0.17 | 0.20 | 0.10 | 0.37 | 0.20 | 0.20 | 0.21 |
| gpt-4o-mini | 0.10 | 0.20 | 0.17 | 0.17 | 0.13 | 0.23 | 0.17 |
| gemini-2.5-flash-lite | 0.23 | 0.13 | 0.03 | 0.20 | 0.13 | 0.17 | 0.15 |
| **mean** | **0.13** | **0.13** | **0.17** | **0.23** | **0.16** | **0.17** | **0.17** |

The no-reasoning floor is genuine at every length; A.3 shows it is movable by reasoning under a
concrete rendering.

### A.3 The reasoning × rendering interaction

With reasoning on, the two renderings separate. Match accuracy, reasoning on:

| model | rendering | L4 | L8 | L16 |
| --- | --- | --- | --- | --- |
| glm-5.2 | token | 0.60 | 0.33 | 0.33 |
| glm-5.2 | concrete | **0.97** | **1.00** | **0.93** |
| kimi-k2.6 | token | 0.37 | 0.27 | 0.20 |
| kimi-k2.6 | concrete | 0.67 | 0.67 | 0.33 |

(Without reasoning, both renderings are at chance — A.2 / §4 grid.) Neither lever alone works:
reasoning under the token rendering leaves GLM at ~0.33, and a concrete rendering without reasoning
leaves it at chance. The combination — reasoning plus a concrete rendering — solves S₅ at
short lengths. The token rendering is the bottleneck on the reasoning arm: it is hard for the model
to track abstract IDs through a scratchpad, where named people and jobs are not. One caveat on
the reasoning-on cells: they use the framing script's 16-token completion budget, which
undercounts models that emit long traces — kimi's concrete L16 cell reads 0.33 here (20/30
empty predictions) but is 1.00 under an 8192-token budget, with 0.83 at L32
(`results/s5_horizon_recheck_20260705.jsonl`). A.4 gives the budget-controlled length profile
for the concrete arm.

### A.4 The length profile

The length profile of the reasoning-plus-concrete arm depends on the completion budget: reasoning traces
grow with sequence length, and cells measured under the framing script's 16-token completion
budget undercount reasoning models — the concrete L64 cell below is 27/30 empty predictions
(`docs/openrouter/s5-horizon.jsonl`). Sweeping GLM-5.2 (reasoning on) under that 16-token budget:

| rendering | L4 | L8 | L16 | L32 | L64 | L128 |
| --- | --- | --- | --- | --- | --- | --- |
| concrete | 0.97 | 1.00 | 0.93 | 0.97 | 0.10 | 0.00 |
| token | 0.60 | 0.33 | 0.33 | 0.13 | 0.00 | 0.00 |

Under an 8192-token budget (`max_new_tokens=8192`, no early stop, n=30;
`results/s5_horizon_recheck_20260705.jsonl`) the concrete+reasoning curve degrades gradually:

| model | L16 | L32 | L64 | L128 |
| --- | --- | --- | --- | --- |
| glm-5.2 | n/a | 1.00 | 0.97 | 0.90 |
| kimi-k2.6 | 1.00 | 0.83 | n/a | n/a |

GLM's empty-prediction rate under the large budget is 0/30 at L32 and L64 and 2/30 at L128 — the
residual errors are genuine wrong answers, not truncation. Kimi's 0.83 at L32 carries 5/30 empty
predictions (it emits very long traces, so 0.83 is an underestimate). The degradation point is
model-dependent (glm ≫ kimi), and the degradation is gradual — GLM holds 0.90 at L128, with no abrupt break. The capability is only visible under the concrete+reasoning setting; under the token
rendering or without reasoning the task looks unsolvable at every length (A.2).

### A.5 Two regimes, two levers

For **frontier inference**, S₅ is movable by reasoning under a concrete rendering, degrading
gradually in length with a model-dependent limit (A.3–A.4). For **local from-scratch training**,
the lever is supervision density: dense per-step state supervision develops a length-extrapolating
circuit that weans to label-free deployment (§8). These are different questions — what a frontier
reasoner can do at inference vs. what a small model can learn from scratch — and both hold.
Composition, chain (at its designed depths < k=6), and S₅ are all reasoning-movable at inference
(§9); what differs is how far each holds — composition holds 0.94–1.00 out to L1024 and S₅ holds 0.90 at
L128 for GLM, while chain's depth behaviour is measured by the scaled no-wrap variant
(`chain_nowrap`), since `chain_v1`'s pointer cycle wraps past depth 5.
