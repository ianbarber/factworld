# FactWorld: A Reproducible Instrument for Composing State Tracking and Recall

FactWorld is a benchmark suite that measures recall, state tracking, and their composition,
identically for frontier models over an API and for small models trained from scratch. Every task must both differentiate frontier models and support architecture exploration in small models trained from scratch. Every number reproduces from committed scripts and can be checked with an API
key or a single GPU.

We find that the component abilities are largely solved at the frontier, but their composition is not. With reasoning off, most models hold little composition in their weights
and architecture. With reasoning on, composition is bought by the token, and the price
differs significantly by model. In task-specific training, each element of the composition
requires a specific architectural or training capability.

**Scope.** FactWorld is a mechanism probe for component capabilities, not an end-to-end agent
benchmark. Every task is single-turn and single-answer-span, with no tool use, planning, or
multi-turn action.

## 1. The instrument

The suite is a frozen, versioned registry of `TaskSpec` objects (`factworld.tasks.CANONICAL`).
Each task renders to natural language over a constrained vocabulary, with deterministic
examples from fixed seeds. Gold answers come from a symbolic oracle applied to the underlying
world state, never from parsing rendered text, so labels cannot leak. A validity gate (`scripts/validate_suite.py`) certifies that no shallow shortcut succeeds on any task: majority class, recency, first position, and entity-blind aggregates are all checked.

The canonical metric is **match**: strip a trailing period from both sides and compare the
model's first len(gold) whitespace tokens to the gold answer, binary per item, no partial
credit. Containment is the one published diagnostic, and any cell (one model on one task at one setting) where containment exceeds match by 0.08 or more has its raw predictions read before the number is believed. When reasoning is on, match scores the answer a multi-line emission commits to (a single-token final line, an
emphasized answer closing a statement, or a lone token in a trailing code fence), never the
working's first tokens. The commitment is located structurally, never by demanding a rigid
output format.

Frontier models run over any OpenAI-compatible API. Local architectures train from scratch on
the same tasks; the data, oracle, and eval layer is pure stdlib.

Every frontier model runs in two regimes. **Instant** disables reasoning under a hard one-line answer contract and a 96-token cap; it measures what the weights alone compute. **Thinking** grants a generous reasoning budget at the shared top effort setting; it measures what reasoning adds. The two rankings are
near-orthogonal, so profiles are per-axis, never a single scalar.

Instant cells are read against two floors, recomputed at render time from the exact task
items. The *recency heuristic* answers the last event's recipient and scores 0.04 (chance,
because the sampler places the queried object's last write uniformly over the stream). The
*object-filter floor* filters events to the queried object but guesses among its writes and
scores E[1/w]: 0.41 at L16, decaying roughly as 1/L. A score near the floor shows object
filtering, not state tracking.

Difficulty knobs (length L, pool breadth, chain depth) are calibration parameters, used to
place each model class mid-scale. Any task scales to stress larger models:

```python
hard = CANONICAL["composite_copy_v2"].scaled(k=64, eval_lengths=(32, 64, 128))
```

> **Methodological note: the recency shortcut.** A give-stream sampler that draws events
> uniformly leaves the queried object's resolving write close to the end of the stream, so a
> one-line recency heuristic scores 0.33, indistinguishable from genuine mid-pack state tracking.
> The current sampler places the queried object's last write uniformly over the stream, which
> drives that heuristic to chance and exposes the object-filter floor as the real bar:
> cheap-tier models sit at or below 0.41 where they previously looked mid-pack. This is why
> the recency heuristic is a permanent floor row and the validity gate checks it on every
> give-stream task.

## 2. The tasks

| | task | role |
|---|---|---|
| Component: recall | `recall_copy_v1` | single-query, deferred-readout MQAR variant; pool breadth is the load axis |
| Component: recall (parametric) | `conflict_v1` | a memorized map contradicted in context; the answer is the in-context value |
| Component: state tracking | `binding_v2` | last-write-wins over a give-stream (absorbing updates, not group ops) |
| Component: state tracking (commutative) | `commutative_v1` | per-entity accumulation mod k; every event matters, order does not |
| Component: state tracking (non-abelian) | `s5_v1` | order-sensitive role permutations; length is sequence stress |
| Composition: state × recall | `composite_copy_v2` | two-hop composition; §4.2 derives the gap statistic from it |
| Composition: recall ∘ recall | `chain_v2` | pointer chase at fixed breadth with an explicit hop count |
| Composition: non-abelian state × serial dereference | `s5_chain_v3` | the ranking task for the frontier benchmark (§4.1) |

**Composition (`composite_copy_v2`).** A set of facts maps agents to values, and a stream of
give events moves objects between agents. The query asks for the value of the agent that
currently holds a given object:

```
g2's a0 is v70. g4's a0 is v24. g0's a0 is v109. g1's a0 is v48.
s0 gives o3 to g4. s1 gives o3 to g1. s2 gives o3 to g2.
what is a0 of the holder of o3?
gold: g2 v70 .
```

The model must resolve the holder by tracking the give-stream (last write wins), then look up
that holder's value in the facts. The facts are resampled every example, so the value is read
from context, not from weights. The answer is two tokens, a holder and a value, and each leg
is scored independently: holder-right/value-wrong means the model tracked state but failed the
recall hop, and the reverse means the opposite. This decomposition localizes failures
throughout the report.

**S5 (`s5_v1`).** A stream of swap and cycle events permutes the roles of a set of agents, and
the query asks for one agent's final role. Swaps and cycles do not commute, so the running
permutation must be carried step by step.

**s5_chain (`s5_chain_v3`).** The ranking task composes the two stress components in one task.
k=16 agents hold an `a0` pointer map, initially one 16-cycle. L order-sensitive swap/cycle
events permute the pointer values: the S5-style state-tracking load. The query then
dereferences the final map 8 hops deep (`what is a0 of a0 of ... of gX? (8 hops)`): the
pointer-chase load. Neither component alone suffices. Every item is gated so that the query start sits on a final-map cycle longer than the query depth. The nine path nodes are therefore distinct, echoing the queried agent or any fixed hop scores exactly 0, difficulty is uniform across items, and chance is 1/16. Cycle events state all their assignments simultaneously (`g5's a0 takes g9's old a0, ...`), so no sequential misreading is available. Unit tests pin both the gate and the rendering.

## 3. Validating the instrument

Before the suite compares architectures or models, it must reproduce the field's established single-capability dissociations. Three of them reproduce on the natural-language format, three seeds each (`scripts/experiment_canonical_repro.py`).

**1-hop associative recall (MQAR).** The value is read adjacent to the key. Attention solves
it: gdp_hybrid 1.00, fprm 1.00, transformer 1.00 (pool 16).

**Deferred readout recall.** The value must be read out at an arbitrary later position; this is the regime that composition actually requires. Scores: gdp_hybrid 0.73, fprm 0.50, transformer 0.19 (pool 5). Every architecture aces 1-hop; only the recurrent hybrid solves deferred readout.

**S5 length extrapolation under dense supervision.** Train dense, evaluate free-running past
the training length:

| arch | L16 (train) | L64 (4×) | L128 (8×) |
| --- | --- | --- | --- |
| gdp_hybrid | 1.00 | 0.90 | 0.82 |
| fprm | 1.00 | 0.17 | 0.23 |
| transformer | 0.79 | 0.22 | 0.22 |

The product recurrence extrapolates; the transformer and the looped block shortcut past the
trained length.

## 4. Benchmarking the frontier

We run thirteen frontier models through the instrument. Instant cells
run n=100 under the answer contract. Thinking cells run n=25 with per-length completion
budgets sized so truncation, scored as wrong, stays a rounding error; Wilson 95% intervals
accompany every cell, and thinking differences under about 0.2 are not an ordering. Models run
at the shared top effort setting, `xhigh`, mapped down where the endpoint's ceiling is `high`.
Cells that carry a contamination mark are excluded from orderings: ⊘ means not measurable at
this budget, ≤x† means an upper bound from covert reasoning, and ‡ means the provider ignored
the token cap.
Three models (grok-4.5, muse-spark-1.1, claude-fable-5) are thinking-only: their endpoints
cannot disable reasoning, so they carry no instant numbers.

### 4.1 FactWorldBench

`s5_chain_v3` is the ranking task: one number per model for how well it holds a mutating
pointer map and then acts through it. The L96 cell ranks the full roster, the L128 cell
separates the top cluster, and the matched L64 cell prices completion tokens per call
(nearly every model solves L64, so token spend compares like for like):

| Model | s5_chain @L96 | @L128 | ctok/call @L64 |
|---|---|---|---|
| anthropic/claude-fable-5 | 1.00 | 1.00 | 5014 |
| openai/gpt-5.5 | 1.00 | 1.00 | 9343 |
| x-ai/grok-4.5 | 1.00 | 0.96 | 7711 |
| anthropic/claude-opus-4.8 | 0.96 | 0.96 | 9702 |
| moonshotai/kimi-k3 | 0.96 | 0.96 | 10941 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.96ʳ | 0.96 | 17071 |
| muse-spark-1.1 | 0.96 | 0.92 | 12484 |
| google/gemini-3.6-flash | 0.92 | 0.96 | 8166 |
| anthropic/claude-sonnet-5 | 0.92 | 0.96 | 12729 |
| deepseek/deepseek-v4-pro | 0.92 | 0.96 | 17052 |
| z-ai/glm-5.2 | 0.92 | 0.80 | 17982 |
| qwen/qwen3.7-max | 0.72 | 0.44 | 12588 |
| openai/gpt-5.6-sol | 0.60 | 0.80 | 2444 |

L96 separates a 0.92 to 1.00 band from a tail. L128 spreads the roster from 0.44 to 1.00: fable and gpt-5.5 hold 1.00, qwen halves to 0.44, glm drops to
0.80. The ʳ on nemotron's L96 marks a single rerun at a raised 98,304-token budget. At the planned
budget it scored 0.84 with 12% truncation, a budget artifact rather than a capability limit;
at L128 it holds 0.96 with no truncation. Token spend also separates the top cluster: the L64
spend spans 5.0k (fable) to 17.1k (nemotron) per call, a 3.4× range for the same scores.

The serving stack constrains the measurement. gpt-5.6-sol's Chat Completions shim caps the
effort ladder at `xhigh`, while its native Responses API exposes a further `max` level that
lifts L96 from 0.60 to 0.88 at 2.3× the reasoning tokens; the scored rows stay on the shared
`xhigh` setting for cross-model fairness, and the `max` probe is reported in
`results/probes/sol_responses_20260724.json`. Sol is also the only model whose scores rise
with prompt length: it allocates reasoning in proportion to input size, gives short prompts
too little, and its failures are genuine wrong answers (match equals containment).

### 4.2 The composed cell and the gap (instant regime)

With reasoning off, does the composition exist in weights at all? The composed cell is the
two-hop query under the instant protocol at L16 and L64. The **gap**, binding_only@L16
minus composed@L16, is interpretable only where the binding component is established. The recall half is free: every measurable model scores 0.97 to 1.00 across all recall
variants, from the sanity check to the scaffolded probe. The gap is therefore a composition
deficit, not a recall one:

| Model | recall | binding @L16 | composed @L16 | composed @L64 | gap @L16 |
|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 0.78 | 0.72 | 0.43 | +0.06 |
| anthropic/claude-sonnet-5 | 0.97 | 0.77 | 0.62† | 0.32† | +0.15† |
| deepseek/deepseek-v4-pro | 1.00 | 0.51 | 0.44 | 0.19 | —ᶠ |
| google/gemini-3.6-flash | 1.00 | 0.69* | 0.67* | 0.26* | +0.02* |
| moonshotai/kimi-k3 | 1.00 | 0.65 | 0.33 | 0.29 | +0.32 |
| nvidia/nemotron-3-ultra-550b-a55b | 1.00 | 0.49 | 0.33 | 0.12 | —ᶠ |
| openai/gpt-5.5 | 1.00 | 0.80 | 0.46 | 0.33 | +0.34 |
| openai/gpt-5.6-sol | 1.00 | 0.82 | 0.65 | 0.33 | +0.17 |
| qwen/qwen3.7-max | 1.00 | 0.51 | 0.24 | 0.08 | —ᶠ |
| z-ai/glm-5.2 | 1.00 | 0.71 | 0.38† | 0.13 | +0.33† |
| *recency heuristic (floor)* | — | 0.04 | 0.04 | 0.06 | — |
| *object-filter floor* | — | 0.41 | 0.41 | 0.15 | — |

Where binding is solid the roster separates. Opus composes essentially for free: its gap of 0.06 is within instant test-retest
variability. gpt-5.5 pays the largest clean deficit (binding
0.80, composed 0.46, gap +0.34): the model that shares the top of the thinking ranking has
among the least in-weights composition. kimi-k3 is
the same shape (gap +0.32 against a top-cluster ranking). For deepseek, qwen, and nemotron
the binding leg's interval overlaps the 0.41 object-filter floor, so the gap renders —ᶠ:
floor minus floor is zero by construction, not a measurement.

### 4.3 Reasoning buys composition

On the composed cell at L16 (n=50 per cell), reasoning effort gives a clean dose-response:

| model | none | low | medium | high |
| --- | --- | --- | --- | --- |
| kimi-k2.6 | 0.72 | 1.00 | 1.00 | 1.00 |
| glm-5.2 | 0.38 | 0.92 | 0.96 | 0.98 |

At effort none the holder leg reads against the 0.41 object-filter floor: object filtering,
not established composition. Low effort already recovers most of the composed cell, and the
curve is monotone through high. The lever is implicit reasoning: an explicit "write the
holder, then the value" instruction hurts every model, including the reasoners that solve the
composed cell under a plain prompt (format-fair ablation, n=100).

### 4.4 The components under stress

The two state-stress components are read separately in the thinking regime, far past the
composed cell's settings: chain d128 (a 128-hop pointer chase at fixed breadth k=257, chance
below 0.01) and s5 @L256 (256 permutation events, concrete rendering, chance 0.20). The
s5@128 ctok column is completion spend on the matched L128 cell that every model runs:

| Model | chain d128 | s5 @L256 | s5@128 ctok |
|---|---|---|---|
| anthropic/claude-fable-5 | 1.00 | 1.00 | 6405 |
| openai/gpt-5.5 | 1.00 | 1.00 | 6989 |
| google/gemini-3.6-flash | 0.96 | 1.00 | 8234 |
| muse-spark-1.1 | 1.00 | 1.00 | 9704 |
| deepseek/deepseek-v4-pro | 1.00 | 1.00 | 10043 |
| anthropic/claude-sonnet-5 | 1.00 | 1.00 | 11866 |
| anthropic/claude-opus-4.8 | 1.00 | 1.00 | 12683 |
| openai/gpt-5.6-sol | 0.88 | 0.92 | 2657 |
| x-ai/grok-4.5 | 1.00 | 0.88 | 8069 |
| qwen/qwen3.7-max | 0.96 | 0.80 | 7904 |
| moonshotai/kimi-k3 | 1.00 | 0.80 | 11355 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.60 | 0.80 | 12250 |
| z-ai/glm-5.2 | 0.92 | 0.76 | 6282 |

The top half of the roster holds both components at or near ceiling, which is why the
composite, not the components, carries the ranking. Every cell runs at a budget sized so
truncation stays a rounding error; the one exception is nemotron, which truncates on 20% of
its s5 @L256 calls even at 32,768 tokens. The chain query includes an explicit hop
count because the bare nested phrase is a hop-counting confound at depth 128: models miscount
the repetitions and stop a few hops short, a prompt-format artifact rather than a state
failure.

The components also dissociate the regimes within one item set. The chain d16 cell runs in
both regimes on identical items. In the thinking regime, twelve of thirteen models score 0.96 to 1.00. In the instant regime,
every model that answers cleanly is at floor (best 0.08). A 16-hop pointer chase is
serial work no roster model holds in weights, and every strong model solves it given room to
work.

### 4.5 Long context

Both regimes are stressed well past their calibration points. The thinking regime holds
composition at long context: glm-5.2 holds 0.94 to 0.98 from L64 out to L1024 on the composed
task (k=32/pool16), while its instant arm sits at or below the object-filter floor from L128
on. kimi's measured cells confirm the shape (1.00 at L256, 0.96 at L512, no truncation). At
this breadth, length is not the binding constraint for the thinking regime. S5 under a
concrete rendering with reasoning holds 0.90 at L128, degrading gradually (Appendix A).

## 5. Exploring the architectures

The same tasks supply training data for models trained from scratch with next-token
prediction. Supervision is answer-only by default, with a staged curriculum for the composite flagship; dense per-step supervision
appears only where it is the measured lever (the s5 and commutative formation results).
The architectures:

* **transformer**: a standard decoder-only stack, the attention baseline.
* **gdn_pure / gdn_hybrid**: GatedDeltaNet (arXiv:2412.06464), gated delta-rule linear
  attention; pure is attention-free, hybrid interleaves one full-attention layer per four.
* **gdp_pure / gdp_hybrid**: GatedDeltaProduct (arXiv:2502.10297), the delta rule generalized
  to a product of Householder transformations per token (n_h=4). This is the non-commutative
  recurrence the state-tracking results turn on; same pure/hybrid split.
* **fprm**: a weight-tied looped conv+attention block, a variant of the fixed-point looped
  transformer of Movahedi et al. (2026, arXiv:2606.18206); one block
  applied repeatedly, so per-token FLOPs match the transformer at 5 to 11× fewer parameters.

Comparisons are matched on compute, not parameters, at budgets sufficient for the capable
configuration to converge. At the flagship scale (d_model=768, 8 layers), fprm's per-token
FLOPs equal the transformer's (~159 GFLOP) at 10M vs 76M params, and gdp_hybrid's
Householder-product recurrence costs 1.25× the transformer's (204 GFLOP at 101M params).

### 5.1 The composite flagship

On `composite_copy_v2` pool-16 @L16 with the staged curriculum (25k steps, 80k docs, 3 seeds,
eval n=500):

| arch | params | per-tok FLOPs | composite @L16 | holder leg | value leg |
| --- | --- | --- | --- | --- | --- |
| gdp_hybrid | 101M | 204 GFLOP | 0.833 ± 0.089 | 0.999 | 0.833 |
| fprm | 10M | 159 GFLOP | 0.109 ± 0.089 | 0.998 | 0.109 |
| transformer | 76M | 159 GFLOP | 0.001 ± 0.001 | 0.065 | 0.041 |

gdp_hybrid solves the binding leg and most of the recall leg, and is competitive with API
models on this task despite training from scratch at 100M params. fprm solves binding
perfectly and fails to recall the value of the holder it just resolved. The transformer fails
both legs. The per-leg decomposition names the deficit: the failure is routing the resolved holder into the in-context
recall lookup, the same deficit the frontier's instant gap measures.

The scaffolded probe sharpens the contrast. Given the correct holder, frontier models recall
the value at 0.98 to 1.00; the local flagship scores 0.096. API models can do each leg when the
problem is split for them and struggle to compose the legs in one prompt; the local models
that fail cannot do the second hop even when handed the first.

### 5.2 Scale does not fix the composite

The same curriculum at three compute-matched sizes (2 seeds each):

| arch | small (384×6) | medium (768×8) | large (1024×12) |
| --- | --- | --- | --- |
| gdp_hybrid | 0.12 ± 0.08 | 0.73 ± 0.01 | 0.21 ± 0.21 |
| fprm | 0.12 ± 0.05 | 0.03 ± 0.01 | 0.03 ± 0.02 |
| transformer | 0.01 ± 0.00 | 0.01 ± 0.01 | 0.00 ± 0.00 |

Convergence is architecture-specific and scale-dependent, peaking at medium: gdp_hybrid at
768×8 is the only converging cell. The transformer floors at every scale including 202M
params and 417 GFLOP/token, with containment near zero as well: a real floor, not a
formatting miss. Wherever binding is solved and the composed cell fails, the value leg is
what collapses; the routing deficit is scale-invariant.

### 5.3 Binding under breadth

fprm's product recurrence leads the binding leg through B16 (1.00 at B6, 0.97 to 0.98
seed-consistent at B16) and stops fitting at B24, where only the gated hybrid holds (0.67);
the transformer reads 0.08 to 0.23 throughout (45 runs at d256). Last-write state requires recurrence, and the form of the recurrence sets a breadth ceiling:
fprm holds through breadth 16, only the gated hybrid holds at breadth 24, and the transformer
never fits.

### 5.4 Chain depth does not extrapolate

Trained at chain depths 2 and 3, no architecture scores above the 1/6 guess at depths 4 and 5
(fprm 0.21/0.15, transformer 0.16/0.09, gdp_hybrid 0.01/0.00; 3 seeds each). gdp_hybrid fits
the training distribution best and scores worst held-out: a depth-specific circuit,
systematically wrong one hop out, not a guesser. Dense intermediate-hop supervision does not
help (0.00 to 0.10 held-out). Over the API the same composition solves at d16, but only in the thinking regime.

### 5.5 S5 forms under dense supervision and survives weaning

Under answer-only supervision S5 floors for every architecture. It forms when the training
signal carries the state: interleave the oracle's state checkpoint every K events and evaluate
free-running (10 seeds):

| K (stride) | value @L16 | value @L64 | converge @L16 |
| --- | --- | --- | --- |
| 1 (dense) | 1.00 | 0.75 | 10/10 |
| 2 | 0.98 | 0.40 | 10/10 |
| 4 | 0.19 | 0.20 | 0/10 |
| 8 | 0.21 | 0.20 | 0/10 |

The circuit forms reliably down to a checkpoint every other event and is gone below: a sharp
learnability threshold. Formation is architecture-independent; length extrapolation is not
(§3). The circuit also survives weaning to label-free deployment: train dense, fine-tune on
mixed densities including answer-only, and 8/8 seeds converge free-running, extrapolating on
par with dense-only. Stressed to 32× the trained length on the dense-supervised non-abelian
composite, the recurrent hybrid holds about 0.5 out to L512 while the looped block stays at
floor.

### 5.6 Commutative state

The commutative rung (per-entity accumulation, order-free) does not form under answer-only
supervision at any measured setting: chance for every architecture at d256. Dense per-step
traces form it in-distribution for the recurrent architectures (gdp_hybrid 0.82 ± 0.15, fprm
0.65 ± 0.26 at L16; transformer at chance), and no run carries it past the training lengths.
The pattern matches S5: supervision density enables formation, and extrapolation is a
separate, unmet requirement.

## 6. What each element of the composition requires

The table below assigns each element of the composition the architectural or training
capability that produced it in local training:

| element | requirement |
|---|---|
| adjacent (1-hop) recall | attention: every architecture aces adjacent readout |
| deferred recall | product recurrence: the transformer aces adjacent, fails deferred (0.19 vs 0.73) |
| last-write state | recurrence, ordered by form: fprm through B16, only the gated hybrid at B24, transformer floors |
| non-abelian state (formation) | dense per-step supervision, a checkpoint every ≤2 events, architecture-independent |
| non-abelian state (length extrapolation) | recurrent hybrid: gdp_hybrid 0.75 @L64; fprm and transformer collapse past train length |
| commutative state (formation) | dense per-step supervision plus recurrence; extrapolation does not follow |
| depth extrapolation | **open**: unsolved by every measured choice, with or without intermediate-hop traces |
| local composition (value leg) | **open** at the default recipe: only the staged curriculum at 768×8 converges it |

Depth extrapolation and the value leg of the composed cell remain unsolved at the scales
tested: no measured architectural or training choice provides the first, and only one
curriculum at one scale converges the second.

## 7. Discussion

The same composition probes that rank the frontier roster also separate architectures trained
locally, and the per-leg decomposition links the regimes: a finding about routing the resolved holder into recall can be checked in both
settings.

The main findings:

- **Composition is where frontier models still separate.** The components are largely solved
  in the thinking regime, but their composition is not: the ranking composite differentiates
  by score at both ends and by tokens-to-solve within the top cluster. With reasoning off, the
  composed cell shows most of the roster holds little composition in weights, with an ordering
  the thinking ranking does not predict: gpt-5.5 shares the top of one and pays the largest
  gap on the other.
- **Composition responds to reasoning, monotonically.** Effort moves the composed cell from
  near floor to 0.92 and above, and the thinking regime holds it out to L1024. Explicit
  prompting does not substitute.
- **Non-abelian state tracking responds to reasoning only under a concrete rendering.**
  Neither reasoning over abstract tokens nor a concrete rendering without reasoning suffices;
  the combination solves it, with a model-dependent length limit (Appendix A).
- **For local models, S5 forms under dense supervision.** A state checkpoint at least every
  two events develops a length-extrapolating circuit that weans to label-free deployment.
- **Architecture carries length generalization.** A learned state circuit generalizes in
  length only on a recurrent hybrid; transformers and looped blocks shortcut.

These are results within the regime tested, not scaling laws.

## 8. Limitations and related work

**Limitations.** The scale regime is bounded: k=5 S5, local models of 3M to 269M params
matched on compute at 32 to 540 GFLOP/token, pretrained models from a few billion to about a
trillion parameters. Composition is 2-hop throughout except the ranking task's 8-hop dereference.
Instant cells run n=100; thinking cells run n=25 to 50 because API costs scale with reasoning
tokens, and thinking differences under about 0.2 are not an ordering. The component-to-agent
mapping is a motivating analogy, not a proven one.

**Related work.** Recall is grounded in multi-query associative recall (Arora et al., 2023).
State tracking is grounded in the S5 word-problem literature: Barrington (1989) makes
S5-recognition NC¹-complete; Merrill and Sabharwal (2023) and Merrill, Petty, and Sabharwal
(2024) bound transformers and SSMs within TC⁰; Liu et al. (2023) show transformers learn
depth-limited shortcuts; Grazzi et al. (2024) unlock state tracking in linear RNNs via
negative eigenvalues; Siems et al. (2025) introduce the Householder-product recurrence
(DeltaProduct) that our gdp models use; Yang, Kautz, and Hatamizadeh (2024) introduce
GatedDeltaNet. The fprm architecture is a weight-tied variant of the looped-transformer
approach of Movahedi et al. (2026, arXiv:2606.18206); we did not run their model. FactWorld's contribution is
measuring the components independently and composed, under one protocol, for API models and
from-scratch models alike.

## 9. Reproducibility

Every claim maps to a committed script and raw results in `results/`. The data,
oracle, and eval layer is pure stdlib; training runs need one CUDA GPU.

```bash
python scripts/validate_suite.py                       # validity gate
python scripts/run_frontier_benchmark.py --dry-run     # full plan + cost estimate, no calls
python scripts/run_frontier_benchmark.py               # appends per-cell records to history
python scripts/render_benchmark.py                     # regenerates the benchmark feed
python scripts/experiment_canonical_repro.py           # the §3 dissociations
python scripts/experiment_dense_supervision.py         # the §5.5 density sweep
```

Raw per-cell records (all attempts, usage, diagnostics) are one JSON object per cell in
`results/benchmark/history.jsonl`; the rendered feed (`docs/benchmark/`) carries per-cell
Wilson intervals, marks, figures, and provenance. The staged-curriculum flagship is
`results/curriculum_staged_v2_d768_notrace.jsonl`; the compute-matched scale sweep is
`results/composite_scale_*.md`.

## Appendix A. S5 at the frontier

Without reasoning, S5 floors at every length. Models do real step-by-step tracking on
individual examples, then stall: they report a role the queried agent held at a recent step
rather than the final one. In aggregate this is chance at every length from L4 to L128, for
every model measured.

With reasoning on, the outcome depends on the rendering. Reasoning over the abstract token
rendering leaves a strong model near 0.33; a concrete rendering (people and jobs) without reasoning is chance;
the combination solves the task. Under an 8,192-token budget the concrete-plus-reasoning curve
degrades gradually: glm-5.2 holds 1.00 at L32, 0.97 at L64, and 0.90 at L128, with residual
errors that are genuine wrong answers, not truncation. The degradation point is
model-dependent (kimi-k2.6 degrades sooner), and there is no abrupt break.

Frontier models solve S5 by reasoning over a concrete rendering. Small models trained from
scratch solve it when the training stream carries a state checkpoint at least every two
events (§5.5).
