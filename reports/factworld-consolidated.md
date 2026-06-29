# FactWorld: Evaluating State-Tracking and Recall

FactWorld is an evaluation for a pattern of behavior we think is connected to agentic
work: maintaining an evolving world state across a long context and recalling facts about
it on demand. We can't prove this pattern is exactly what agents do — but it is a
controllable, interesting proxy for some of the things that happen in agentic sessions,
and it is sharp enough to separate models and architectures.

The instrument fills a gap. Frontier API models can be evaluated end-to-end, but you
cannot ablate an architecture inside one — you cannot ask "does the attention pattern
or the recurrence carry the state?" of a model you only call over HTTP. Small models
trained from scratch let you ablate architectures freely, but it is hard to know whether
their behaviors reflect general mechanisms or the idiosyncrasies of a tiny run.
FactWorld runs the same controlled tasks in both regimes, so a finding can be checked
both ways. This is not a recipe to train a frontier model; it is a way to close the eval
gap between the two.

Three findings carry the report:

- **A. Composition is movable by background reasoning.** On the 2-hop composition task,
  accuracy climbs with reasoning effort (GLM-5.2: 0.14 → 0.81; Kimi-K2.6: 0.22 → 0.98),
  and the benefit survives long context (0.93–0.97 at L256 with high effort).
- **B. Non-abelian state-tracking (S₅) is movable by training-time supervision that
  develops a circuit.** Dense per-step state supervision solves S₅ (10/10 seeds); the
  circuit forms down to a checkpoint every other step and is gone below. It survives
  weaning to label-free deployment.
- **C. Architecture determines whether a learned circuit generalizes in length.**
  Transformers and weight-tied looped blocks shortcut — they solve the trained length
  and collapse past it. A recurrent hybrid carries the learned state computation to
  ~32× the trained horizon.

## 1. The tasks

Two task families do the work of this report. They look similar — both ask the model to
maintain state across a stream of events — but they test different mechanisms and, as
the results show, respond to different levers.

**Composition** combines last-write-wins state-tracking with in-context recall. A set of
facts maps each agent to a value, and a stream of `give` events moves objects between
agents. The query asks for the value of whichever agent currently holds a given object:

```
g2's a0 is v70. g4's a0 is v24. g0's a0 is v109. g1's a0 is v48.
s0 gives o3 to g4. s1 gives o3 to g1. s2 gives o3 to g2.
what is a0 of the holder of o3?
gold: g2 v70 .
```

The model must resolve the holder of `o3` by tracking the give-stream (last write wins),
then look up that holder's value in the facts. The facts are resampled every example and
appear in the prompt, so this is **in-context recall** — the value is read from the
context, not from weights. (A parametric variant fixes the map across training so the
model can memorize it; we use the in-context form here unless stated.)

**S₅** is non-abelian state-tracking. A stream of `swap` and `cycle` events permutes the
roles of a set of agents, and the query asks for one agent's final role:

```
s0 swaps g3 and g0. s1 swaps g3 and g1. s2 swaps g1 and g2.
s3 cycles roles: g3 -> g2 -> g1 -> g0 -> g4.
what role does g3 have?
gold: r4 .
```

Unlike last-write-wins, the running permutation cannot be recovered by scanning the
history for the last write — it must be carried step by step. This is the harder,
non-abelian case.

**How eval works.** Every example's gold answer comes from a symbolic solver applied to
the underlying world state — never from parsing the rendered text — so labels cannot
leak. The renderer and its parser are exact inverses, and a validity gate certifies that
no shallow shortcut (majority, recency, or first-position) clears floor on any task.
Answers are scored by position-strict exact match of the answer span.

**The per-leg decomposition.** A composition answer is two tokens — a holder and a value
— corresponding to the two steps. We score each leg independently. This turns an
ambiguous aggregate score into a diagnosis: a model that gets the holder right and the
value wrong failed recall; the reverse failed state-tracking. The decomposition also
admits a ceiling probe: tell the model the correct holder and ask only for the value. If
recall is then perfect, the wall is state-tracking, not recall.

The rest of the report uses the decomposition throughout. Three simpler tasks round out
the suite — single-hop in-context recall, single-hop binding, and a depth-*k* pointer
chase — and are reported in the capability table.

## 2. Evaluating API models

We evaluate pretrained models from 3B to ~1T parameters (MoE) via OpenRouter, on the
natural-language format with output-format instructions appended where the answer shape
is ambiguous (n=30, greedy).

| model | recall | binding | chain | composite | s5 |
| --- | --- | --- | --- | --- | --- |
| glm-5.2 | 1.00 | 0.77 | 0.13 | 0.80 | 0.17 |
| llama-3.3-70b | 1.00 | 0.63 | 0.00 | 0.80 | 0.17 |
| kimi-k2.6 | 1.00 | 0.63 | 0.03 | 0.40 | 0.20 |
| gemini-2.5-flash | 1.00 | 0.30 | 0.10 | 0.23 | 0.13 |
| deepseek-chat | 1.00 | 0.33 | 0.03 | 0.17 | 0.13 |
| gpt-4o-mini | 1.00 | 0.37 | 0.07 | 0.14 | 0.13 |

(position-strict exact match at L16; full grid in `docs/openrouter/results-natural.md`.)

Single-hop recall is solved across the board. Binding and composition separate the
stronger models. The two hardest tasks — depth-*k* composition (`chain`) and non-abelian
state-tracking (`s5`) — sit near floor for everyone.

**Composition is movable by background reasoning.** The grid above uses default
decoding. Sweeping reasoning effort {none, low, medium, high} on the reasoning models
gives a clear dose-response on composition (n=100/40, value accuracy):

| model | none | low | medium | high |
| --- | --- | --- | --- | --- |
| kimi-k2.6 | 0.22 | 0.94 | 0.98 | 0.96 |
| glm-5.2 | 0.14 | 0.74 | 0.78 | 0.81 |

Composition climbs from floor to ~0.8–0.98 as reasoning budget rises. The lever is
*implicit* reasoning: an explicit "write the holder, then the value" instruction hurts
every model (value drops to ~0.00), including the reasoners that score 0.8–0.98 under a
plain prompt. Forcing an explicit intermediate disrupts models that reason better
implicitly. (This was a correction: an earlier draft read the implicit-reasoning result
as "test-time compute doesn't help," which was wrong — it does, for composition.)

**S₅ is not moved by reasoning.** At every effort, value accuracy stays at 0.00 for both
kimi and glm. The two tasks respond to different levers, and reasoning is not the one
for S₅.

## 3. Evaluating architectures by training small models

The same tasks, trained from scratch, let us ablate architectures and training regimes
that an API model hides. We train three architectures at matched compute (d=256, 4
layers, 8k steps, 5 seeds): a transformer, a weight-tied looped conv+attention block
(`fprm`), and a recurrent hybrid (`gdp_hybrid` — a `[recurrent, recurrent, attn,
recurrent]` GatedDeltaProduct stack).

**Capability at L16** (position-strict exact match, mean over seeds):

| arch | recall | binding | composite | s5 |
| --- | --- | --- | --- | --- |
| gdp_hybrid | 0.62 | 0.99 | 0.50 | 0.20 |
| fprm | 0.48 | 1.00 | 0.20 | 0.19 |
| transformer | 0.14 | 0.45 | 0.05 | 0.20 |

(recall at pool-6; composite is the learnable k=5 variant; full table in
`results/sweep_main_*` and `results/recall_arch_*`.)

Two things stand out, and both reproduce takeaways established in the instrument's
earlier phases.

**Recall is architecture-dependent at this scale; the recurrent hybrid solves it, the
transformer does not.** On single-hop in-context recall, the recurrent hybrid (gdp_hybrid)
scores 0.62–1.00 while the transformer scores 0.11–0.48. This is not a new finding: it
reproduces the Phase 1 data exactly (gdp_hybrid 1.00 in-distribution vs transformer 0.48 at
the same scale and step budget), and it is a genuine dissociation, not an artifact of
undertraining. We checked the obvious alternative explanations for the transformer's weakness:
training on the eval pool sizes (not just smaller ones) lifted pool-6 only 0.11 → ~0.20;
quadrupling the data added nothing; and scaling width 16× (d=256 / 4M → d=1024 / 68M) gave no
improvement and on wide pools made it worse. So at this scale the transformer does not solve
this recall task, and more compute of the kinds we varied does not change that.

This does not contradict the wider literature. The well-known result that attention solves
associative recall (MQAR; Arora et al. 2023) is for 1-hop lookup over a *large* key set
in-distribution; our `recall_copy_v1` is a small pool (2–8) evaluated out-of-distribution on
pool size, a binding-load extrapolation the transformer specifically fails. (The pretrained
grid shows recall at ceiling because those models are far larger and trained on vastly more
data — a different regime.) We do not claim the transformer cannot do in-context recall in
principle; we report that, at the matched-compute scale where we can ablate architectures, the
recurrent hybrid is the one that generalizes recall across pool sizes. Parametric recall —
facts stored in weights — is the complementary case; we do not validate it for API models here,
though fine-tuning a fixed fact set into a model and testing recall of it would be a clean way.

**Architecture determines whether a learned circuit generalizes in length.** On S₅ under
dense supervision (below), all three architectures form the circuit in-distribution. But
evaluated past the training lengths:

| arch | L16 | L64 | L128 |
| --- | --- | --- | --- |
| gdp_hybrid | 1.00 | 0.74 | 0.64 |
| fprm | 1.00 | 0.20 | 0.23 |
| transformer | 0.83 | 0.19 | — |

The transformer and the looped block solve the trained length and collapse past it —
**shortcutting**, the well-documented transformer failure mode on state-tracking (Liu
et al., 2023). The recurrent hybrid carries the learned state computation far past its
training horizon. This reproduces the Phase 1 finding: product recurrences extrapolate
non-abelian state where shortcut-learning mixers collapse.

## 4. S₅ is movable by supervision density

S₅ floors for every architecture under answer-only supervision (the table above). It
moves when the training signal carries the state. We interleave the oracle's
holder-of-the-queried-role every *K* events into the training stream (K=1 is dense), and
evaluate free-running — events forced, the holder slots and final value generated by the
model with no oracle at eval. 10 seeds:

| K (stride) | value @L16 | value @L64 | converge @L16 |
| --- | --- | --- | --- |
| 1 (dense) | 1.00 | 0.75 | 10/10 |
| 2 | 0.98 | 0.40 | 10/10 |
| 4 | 0.19 | 0.20 | 0/10 |
| 8 | 0.21 | 0.20 | 0/10 |

The circuit forms reliably in-distribution down to a checkpoint every other step (10/10
at K=2) and is gone below (0/10 at K≥4). This is a sharp **learnability cliff**, and it
reproduces the Phase 2 finding: S₅ needs near-dense process supervision to form the
circuit; the sparse, answer-only signal an agent effectively has gives no traction.

**The circuit survives weaning to label-free deployment.** Dense supervision moves the
wall but cannot ship with per-step labels. We train dense, then fine-tune on a mix of
densities including answer-only, and evaluate free-running. 8 seeds:

| arm | L16 | L64 | L128 | converge |
| --- | --- | --- | --- | --- |
| dense only (reference) | 1.00 | 0.61 | 0.50 | 8/8 |
| weaned (mixed density) | 1.00 | 0.50–0.54 | 0.46–0.48 | 8/8 |
| answer-only (never dense) | 0.19 | 0.19 | — | 0/8 |

The weaned circuit converges 8/8 free-running with no deploy-time labels and extrapolates
on par with dense-only. (Honest negative: weaning does not *improve* extrapolation over
dense — a prior hint that it would did not reproduce at power. The win is label-free
deployment.) The specific density mix barely matters; the key is some answer-only
exposure alongside dense.

## 5. Long context

Real sessions are long. We stress both regimes far past their sweet spots — trained
models (trained at ≤16) evaluated to L512 (32×), pretrained models evaluated from L16 to
L512.

**Trained recurrent hybrids extrapolate far.** Stressing the §3 comparison to 32× the
trained horizon (8 seeds):

| arch | L64 | L128 | L256 | L512 |
| --- | --- | --- | --- | --- |
| gdp_hybrid | 0.62 | 0.59 | 0.51 | 0.46 |
| fprm | 0.26 | 0.18 | 0.20 | 0.15 |

The recurrent hybrid holds ~0.5 out to L512 — a graceful tail, not a cliff. The looped
block stays at floor throughout. The architecture's length-generalization role is
durable under stress.

**Background reasoning rescues composition at long context.** At default decoding,
composition collapses for pretrained models past L128 (llama-3.3-70b: 0.80 @L16 → 0.07
@L128). But that collapse is a default-effort artifact. With reasoning effort swept
{none, high} at long length (n=30, value accuracy):

| model | L128 none | L128 high | L256 none | L256 high | L512 high |
| --- | --- | --- | --- | --- | --- |
| kimi-k2.6 | 0.03 | **0.97** | 0.00 | **0.97** | **0.93** |
| glm-5.2 | 0.10 | **0.93** | 0.10 | **0.97** | **0.80** |

High reasoning effort recovers composition to 0.80–0.97 all the way out to L512 — roughly
32× the L16 sweet spot and ~3.5k-token prompts. The composition lever (§2) is remarkably
horizon-robust when the reasoning budget is there. S₅ does not move: it stays at 0.00 at
high effort at every length we tested. The two tasks' levers hold their distinct characters
under horizon stress — reasoning carries composition, supervision density carries S₅.

## 6. Discussion

The instrument lets the same controlled behavior be measured two ways, and the two ways
agree where they overlap. Transformers — whether a 45M local model or a 70B+ pretrained
one — solve composition's recall leg and shortcut on state-tracking. Recurrent hybrids
solve in-context recall less well but carry a learned state circuit far past their
training horizon. Pretrained reasoning models add a third lever: background reasoning
moves composition at inference, including at long context, in a way no training
intervention or prompting trick reproduced.

The takeaways we want a reader to carry:

- **Composition responds to reasoning.** It is movable at inference by background
  reasoning, including at long context, for models that can reason. Explicit prompting
  does not substitute.
- **Non-abelian state-tracking responds to training signal.** It is movable by dense
  per-step supervision that develops a circuit, and that circuit can be weaned to
  label-free deployment. Reasoning does not move it.
- **Architecture carries length generalization.** A learned state circuit generalizes in
  length only on a recurrent hybrid; transformers and looped blocks shortcut.

We state these as results within the regime tested (k=5 S₅; local models ≤45M;
pretrained models 3B–~1T). They are not scaling laws. The connection to agentic work is
a motivating proxy, not a proven mapping.

## 7. Limitations and related work

**Limitations.** The scale regime is bounded (k=5 S₅; local ≤45M; pretrained to ~1T
MoE). Composition is 2-hop throughout. In-context recall is validated per-architecture;
parametric recall is not validated for API models (fine-tuning a fixed fact set into a
model would close this). The natural-language format differs from the atomic-token
format of earlier phases; absolute numbers are not comparable across formats, though the
mechanism conclusions reproduced. Weaning does not improve extrapolation over dense-only.

**Related work.** Earlier phases of this instrument established the single-capability
dissociations and the non-abelian recipe on the atomic-token format
([`phases/`](phases/)); this report reproduces their takeaways on the natural-language
format and adds the API evaluation and the long-context results. The `fprm` architecture
is a probe inspired by Movahedi et al. (2026); we did not run their model. The
shortcut-learning and length-extrapolation results engage a substantial literature on
transformer state-tracking brittleness (Liu et al., 2023) and recurrent extrapolation,
which we extend rather than survey.

## 8. Reproducibility

Every claim maps to one script in `docs/experiments/`; raw results in `results/`; the
validity gate (`scripts/validate_suite.py`) certifies the suite. The data and eval layer
is pure-stdlib (no GPU); training runs need a single CUDA GPU.
