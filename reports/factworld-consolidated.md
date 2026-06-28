# FactWorld: A Decomposition-Centric Instrument for State-Tracking and Recall

*Draft — consolidated technical report. Supersedes and corrects the atomic-token-format reports
in [`phases/`](../phases/). All numbers are on the natural-language benchmark; every claim maps to
one script in [`docs/experiments/`](experiments/).*

## Abstract

Large models must compose two capabilities to act over long horizons: **state-tracking** (maintain
an evolving world) and **recall** (look up facts given a resolved pointer). We introduce FactWorld,
an oracle-validated instrument whose central contribution is a **per-leg decomposition** of the
flagship composition task into a state leg (resolve the holder) and a recall leg (look up its value).
The decomposition exposes a simple, load-bearing fact: **recall is free** — every model we test
recalls the value given the correct holder (0.93–1.00); the wall is *generating* the holder.

The instrument is, fundamentally, an **evaluation**: it maps a capability ladder and isolates
*which* capability fails. Training small models from scratch is how we explore the architecture
space under that evaluation, and pretrained models up to 70B provide an external validity check
that the mechanisms we find are not small-model artifacts. The picture converges. As a capability
map, the composition wall is movable by **test-time compute**: a reasoning-effort dose-response
lifts composition from 0.14 to 0.81 (GLM-5.2) and 0.22 to 0.98 (Kimi-K2.6), while explicit
chain-of-thought prompting actively hurts. As an architecture exploration, the deeper
non-abelian state-tracking wall (S₅) is *not* movable by reasoning at any effort, but is movable by
**training-time supervision density**: dense per-step state supervision solves it (10/10 seeds), the
circuit survives weaning to label-free deployment, and only the recurrent hybrid extrapolates it
(0.77 at 32× the trained horizon). Two walls, two levers, recall free.

---

## 1. A decomposition, and the instrument it needs

### 1.1 The finding that motivates the instrument

Consider a composition task: a stream of *"s0 gives o0 to g30"* events establishes the current
holder of each object, a set of *"g30's a0 is v92"* facts maps each holder to a value, and the query
*"what is a0 of the holder of o0?"* requires resolving the holder of `o0` (state-tracking) and then
looking up its value (recall). On this task, a strong pretrained model produces:

```
... s15 gives o0 to g30. what is a0 of the holder of o0?
gold : g30 v92 .          # holder g30, value v92
pred : g30 v9  .          # holder ✓   value ✗
```

It resolves the holder correctly but emits the wrong value. This **"holder right, value wrong"**
pattern is the dominant failure mode on composition across every model we tested — and it is
ambiguous on a single scalar metric. Did the model fail to track state, fail to recall, or fail to
*route* the resolved holder into the recall lookup? An aggregate accuracy of 0.0 cannot say.

The contribution that resolves this ambiguity is a **per-leg score**: score the holder (the
state-tracking leg) and the value (the recall leg) independently. Applied to the example above, it
reads "state leg ✓, recall leg ✗." Applied at scale, it separates *which* capability failed — and,
crucially, it admits a **ceiling probe**: inject the correct holder into the prompt (an oracle
scaffold) and ask only for the value. If recall is then perfect, the wall is state-tracking, not
recall. We find it is: across all models, **value accuracy given the correct holder is 0.93–1.00**
(§2.2). Recall is free.

### 1.2 FactWorld

This decomposition requires a controlled, oracle-validated instrument — one where the two legs are
cleanly separable and labels cannot leak. FactWorld provides it. The suite is a frozen, versioned
registry of tasks (`factworld.tasks.CANONICAL`) ordered by the kind of computation each demands:

| task | what it measures | difficulty axis |
| --- | --- | --- |
| `recall_copy_v1` | 1-of-N in-context-copy recall | distractor pool |
| `binding_v1` | last-write-wins state (the delta-rule axis) | give-stream length |
| `composite_copy_v1` | binding × in-context recall — the 2-hop composition probe | binding horizon |
| `chain_v1` | depth-*k* pointer chase | composition depth |
| `s5_v1` | S₅ role permutation (non-abelian state-tracking) | permutation horizon |

Three properties make it a usable instrument:

- **Symbolic oracle, no label leakage.** Every example's gold answer comes from a symbolic solver
  (`factworld.oracle.Oracle`) operating on the KB, never from parsing rendered text. The renderer
  and its parser are an exact inverse, and a validity gate (`scripts/validate_suite.py`) certifies
  that no shallow shortcut — majority, recency, or first-position — clears floor on any task.
- **Clean natural-language format.** Statements are clean English with attached punctuation
  (`s0 gives o0 to g30.`, `g30's a0 is v92.`), one fixed phrasing per statement type. This is the
  single canonical format; the earlier atomic-token format is archived in `phases/`. (A controlled
  comparison showed the natural format roughly doubles composition convergence at small scale
  versus the atomic-token format; we adopt it throughout.)
- **The leg-decomposition is a first-class metric.** `factworld.tasks.decompose_composite` scores
  the holder and value legs; `trace_accuracy` scores a self-generated state trajectory against the
  oracle's. The runner reports exact match plus the decomposition.

Any model that can continue a prompt evaluates against it — OpenAI-compatible APIs, HuggingFace
transformers, or a model trained from scratch in the included harness. **The rest of the paper
uses the latter as a controlled probe of the architecture space, and pretrained models as the
external check that the findings generalize beyond small models.**

### 1.3 Scope

A single instrument cannot be a scaling law. We hold the world fixed at k=5 S₅ and evaluate two
regimes: locally trained models up to ~45M parameters (gdp_hybrid, fprm, transformer) and pretrained
API models up to 70B. The claims are about the *mechanisms* this instrument isolates — recall,
state-tracking, composition, and their levers — within this regime, not about model scaling in
general. Where a claim depends on architecture (§3.3) or scale, we state the dependence.

---

## 2. The capability map

Applied across pretrained models (the external check) and small trained models (the controlled
probe), the instrument yields a consistent ladder. We report **position-strict exact match** of the
answer span as the canonical metric, plus the holder/value decomposition where the answer is
two-token. Floor is the chance rate of each task's answer space.

### 2.1 The ladder

**Pretrained models** (OpenRouter, n=30, natural format, output-format instructions appended;
`docs/openrouter/results-natural.md`):

| model | recall_copy | conflict | binding | chain | composite_copy | s5 |
| --- | --- | --- | --- | --- | --- | --- |
| glm-5.2 | 1.00 | 1.00 | **0.77** | 0.13 | **0.80** | 0.17 |
| llama-3.3-70b | 1.00 | 1.00 | 0.63 | 0.00 | **0.80** | 0.17 |
| kimi-k2.6 | 1.00 | 1.00 | 0.63 | 0.03 | 0.40 | 0.20 |
| gemini-2.5-flash | 1.00 | 1.00 | 0.30 | 0.10 | 0.23 | 0.13 |
| deepseek-chat | 1.00 | 1.00 | 0.33 | 0.03 | 0.17 | 0.13 |
| gpt-4o-mini | 1.00 | 1.00 | 0.37 | 0.07 | 0.14 | 0.13 |

**Locally trained models** (5 seeds, d=256, 4 layers, 8k steps; holder/value decomposition @L16;
`results/sweep_main_*`):

| arch | binding @L64 | composite_scale holder/value @L16 | s5 @L32 |
| --- | --- | --- | --- |
| gdp_hybrid | 0.55 | 0.95 / **0.51** | 0.20 |
| fprm | **0.94** | 0.99 / 0.20 | 0.19 |
| transformer | 0.33 | 0.38 / 0.13 | 0.20 |

The ladder is consistent across both regimes. Single-hop recall and conflict are at ceiling.
Binding is scale- and architecture-sensitive. Composition and S₅ are the walls, and on both the
locally trained models land at the same floor as the pretrained ones under answer-only
supervision. **The pretrained grid confirms the small-model picture is not an artifact: these are
genuine capability walls, not training-budget limitations.**

### 2.2 Recall is free — the wall is the state leg

The decomposition resolves where composition breaks. On `composite_copy_v1@L16`, decomposing every
prediction into its holder leg and value leg yields a uniform pattern: the holder is often right,
the value is wrong, and the aggregate score hides both. The decisive probe is the **scaffolded
ceiling**: inject the correct holder into the prompt and ask only for the value.

| model | composite (none) | scaffolded (value given holder) |
| --- | --- | --- |
| kimi-k2.6 | 0.40 | **1.00** |
| glm-5.2 | 0.80 | **1.00** |
| gpt-4o-mini | 0.14 | **1.00** |
| llama-3.3-70b | 0.80 | 0.93 |

(`results/autoregressive_formatfair_*`; n=100.) Given the correct holder, every model recalls the
value at 0.93–1.00. **Recall is not the bottleneck.** The wall is *generating* the holder — i.e.,
state-tracking — and routing it into the recall lookup. The same pattern holds locally: free-run
holder accuracy ≈ free-run value accuracy (they track), while value-given-correct-holder is 1.00.

This single result reframes the rest of the paper: the question is not "can models compose?" but
"what moves the state-tracking leg?" — and the answer differs by task (§3).

---

## 3. Two walls, two levers — an architecture exploration

With recall isolated, we ask what moves the state leg on each wall. Training from scratch is the
controlled probe: it lets us vary one thing at a time (supervision density, architecture,
intermediate generation) that pretrained models cannot be cleanly varied on. We use `composite_copy`
(the 2-hop wall) and `s5_v1` (the non-abelian, permutation-tracking wall) as the two test cases, and
check every conclusion against the pretrained models.

### 3.1 Composition is movable by test-time compute

On `composite_copy`, the lever is **background reasoning**. Sweeping reasoning effort
{none, low, medium, high} on the reasoning models gives a clear dose-response
(`results/reasoning_*`, n=100/40):

| model | none | low | medium | high |
| --- | --- | --- | --- | --- |
| kimi-k2.6 | 0.22 | 0.94 | 0.98 | 0.96 |
| glm-5.2 | 0.14 | 0.74 | 0.78 | 0.81 |

(value accuracy.) Composition climbs from floor to ~0.8–0.98 as reasoning budget increases. The
lever is *implicit* reasoning ability: **explicit prompting backfires**. Under a structured CoT
instruction ("first write `holder: <g>`, then the value"), value accuracy drops to ~0.00 for every
model, including the reasoners that score 0.8–0.98 under a plain prompt. Forcing an explicit
intermediate disrupts models that reason better implicitly. Non-reasoning models (deepseek 0.13,
gpt-4o 0.14) stay at the wall regardless. (Composition is also not movable from the training side:
dense supervision of the holder-of-object does *not* fix the composite's holder leg — last-write-wins
over four interleaved objects is harder to unroll than the single-role tracking of S₅; see §3.2.)
**So for composition the lever lives in the pretrained model's reasoning strength — not in training,
not in prompting.**

### 3.2 S₅ is movable by supervision density, not reasoning

The S₅ wall behaves oppositely. First, **reasoning cannot move it**: at every effort, value
accuracy stays at 0.00 for both kimi and glm (holder sometimes partially right, never routed to the
value). This is the clean counterpoint to §3.1.

Second, **training-time supervision density moves it decisively**. We interleave the oracle's
holder-of-the-queried-role every K events into the training stream (K=1 is dense), and evaluate
free-running (events forced, holder slots and value *generated*, no oracle at eval). 10 seeds
(`results/dense_power_gdp_*`):

| K (stride) | value @L16 | value @L64 | converge @L16 |
| --- | --- | --- | --- |
| 1 (dense) | **1.00**±0.00 | **0.75**±0.22 | 10/10 |
| 2 | 0.98±0.03 | 0.40±0.26 | 10/10 |
| 4 | 0.19±0.02 | 0.20 | 0/10 |
| 8 | 0.21±0.02 | 0.20 | 0/10 |

The circuit forms reliably in-distribution down to K=2 (10/10) and is gone at K≥4 (0/10) — a sharp
**learnability cliff**, not a capacity limit. The same dense signal that floors under sparse
supervision solves the task; sparse/answer-only signal (the agentic regime, and what pretrained
models effectively have) gives no traction. This is the load-bearing result for the training side.

### 3.3 Architecture gates extrapolation, not formation

The dense result holds across architectures, but **length extrapolation does not**. All three
architectures form the S₅ circuit in-distribution at K=1 (gdp_hybrid/fprm 1.00, transformer 0.83).
Evaluated far past the training lengths (≤16):

| arch | L16 | L64 | L128 |
| --- | --- | --- | --- |
| **gdp_hybrid** | 1.00 | 0.74 | **0.64** |
| fprm | 1.00 | 0.20 | 0.23 |
| transformer | 0.83 | 0.19 | — |

(`results/dense_extrap_*`, 10 seeds.) Only the recurrent hybrid (`gdp_hybrid`, a
`[recurrent, recurrent, attn, recurrent]` GatedDeltaProduct stack) extrapolates the learned circuit;
fprm (a weight-tied looped conv+attention block) solves in-distribution but collapses past L32; the
transformer floors by L64. **Circuit formation is architecture-independent (a supervision-density
phenomenon); circuit extrapolation is architecture-dependent.** This is the one place architecture
materially matters in the whole program — and it is about *length-generalizing* a learned state
circuit, not about solving the task.

### 3.4 The bridge to label-free deployment

Dense supervision moves the S₅ wall but cannot be deployed with per-step labels. The question is
whether the dense-learned circuit survives **weaning** to answer-only. We train dense (K=1), then
fine-tune on a mix of densities including answer-only, and evaluate free-running with no labels.
8 seeds (`results/weaning_deep_*`):

| arm | L16 | L64 | L128 | converge @L16 |
| --- | --- | --- | --- | --- |
| dense_only (reference) | 1.00 | 0.61 | 0.50 | 8/8 |
| wean (mixed density) | 1.00 | 0.50–0.54 | 0.46–0.48 | **8/8** |
| answer_only (never dense) | 0.19 | 0.19 | — | 0/8 |

The circuit survives weaning (8/8 converge free-running, no deploy-time labels) and extrapolates on
par with dense-only. **Honest negative:** weaning does *not* improve extrapolation over dense — all
mixes land within noise of dense (L128 0.46–0.53 vs 0.50); a prior hint that mixed density helps
extrapolation did not reproduce at power. The win is purely label-free *deployment*. The specific
density mix barely matters; the key is just some answer-only exposure alongside dense. Deployment
recipe: train dense → wean with any answer-only-inclusive mix → deploy answer-only.

### 3.5 What does not move the walls

For completeness, the nulls (each in its task context): explicit structured CoT prompting *hurts*
composition (§3.1); a trained self-generated scratchpad collapses the holder leg by error compounding
(`results/sweep_e2_trace_*`); sampling-based self-consistency to 30 votes and iterative
self-correction over 3 rounds give exactly zero lift on local models (`results/self_correct_*`). The
qualified statement: these nulls hold for *explicit* elicitation and *non-reasoning* models;
implicit background reasoning is the one exception, and only for composition (§3.1), not S₅.

---

*§4 (extended sessions / horizon scaling) to follow once the long-context runs complete.*
