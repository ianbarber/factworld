# Experiment design — autoregressive reasoning / test-time compute

**Status:** design (not yet run). The "subagent" workstream: does letting a model
*generate more* (a self-produced scratchpad / chain-of-thought) before answering
unlock the composition and state-tracking tasks — for both API and from-scratch
models — and is the gain real computation or just more tokens?

This builds directly on machinery that already exists in the harness:

- `TaskSpec.worked_trace` produces a gold **oracle intermediate trajectory** in
  `example.meta["trace"]` (`composite_*` = the per-step holder; `s5_v1` = the
  per-step role). This is the *supervised* form of the scratchpad and the
  reference we score self-generated traces against.
- `build_docs(examples, use_trace=True)` already trains "prompt + trace + answer".
- `score_last_n` already scores the final answer after an arbitrary prefix.
- `evaluate_task(max_new_tokens=...)` + `stop_at="."` already control budget.

So most of what follows is a configuration of things that exist, plus one new
scoring helper (trace accuracy).

## Why this should help here (and where it should not)

Our decomposition shows the dominant failure on `composite_copy_*` is **"holder
right, value wrong"** — the model resolves the binding leg but fails to *route*
that result into the recall leg. A scratchpad that **emits the resolved holder
explicitly** (`the holder of o3 is g0`) collapses the 2-hop into two 1-hops. That
is the single most direct prediction:

- **H1 (composition):** a "state the holder, then recall" scratchpad recovers the
  *value* leg for models that already get the *holder* leg. Predicted win on
  `composite_copy_*` and `chain_v1` (multi-hop); near-zero on `recall_copy_v1` /
  `conflict_v1` (single-hop, already at ceiling).
- **H2 (state-tracking):** step-by-step state **unrolling** (`g0 g1 g2 …`, the
  `worked_trace`) lets a model *compute* `s5_v1` / `binding_v1` token-by-token
  instead of holding it latently — the inference-time analogue of Phase 2's dense
  supervision. Open question: does a *self-generated* unroll match a
  *teacher-forced* one, and does it extrapolate OOD?
- **H3 (trained vs API):** API models CoT natively; from-scratch models must be
  **trained** to emit a useful trace. Sub-question: does a trained scratchpad
  extrapolate to longer lengths better than answer-only (compute-vs-memorize)?

## Experiments

### E1 — Inference-time CoT on API models (cheapest, run first)

Four conditions on `composite_copy_v1` and `chain_v1` (and `s5_v1`), n≥30:

1. **answer-only** (current default).
2. **free CoT** — system prompt: "think step by step, then answer".
3. **structured CoT** — "first write `holder: <g>`, then `<g> <value>.`" so the
   intermediate is machine-parseable.
4. **scaffolded upper bound** — inject the *correct* holder into the prompt and
   ask only for the value. This isolates the recall leg given perfect binding: it
   is the ceiling for any CoT that merely gets the holder right.

Read with the holder/value decomposition: the CoT win is real only if condition 3
lifts *value_acc* (not just *holder_acc*), and condition 4 tells you how much
headroom remains. **Knob:** `max_new_tokens` must scale with the eval length (≥ length +
answer); the runner already takes it as an argument.

Deliverable: per-condition `(overall, holder_acc, value_acc)`; conclusion = does
self-generated CoT close the composition gap, and how far from the scaffolded
ceiling.

### E2 — Trained scratchpad vs answer-only (local, the key result)

Train `gdp_hybrid` / `fprm` / `transformer` on `composite_copy_scale_v1` and
`s5_v1` under two regimes, matched seeds:

- **A. answer-only** (current).
- **B. trace-supervised** (`use_trace=True`): target = prompt + oracle trace +
  answer. At eval, generate freely (no teacher forcing) and score the **final
  answer** with `score_last_n`.

Hypothesis (H3): B extrapolates further OOD (L64/L128) because the state is
*regenerated* rather than retrieved from a fixed-length latent. Compare at
**matched token budget**: B generates ~`length` extra tokens, so also run A with
`max_new_tokens` padded to match, to separate "more compute" from "structured
unrolling".

Deliverable: p(converge) over ≥5 seeds per (arch × regime × length); the
decomposition at each length.

### E3 — Trace accuracy (does the model actually unroll correctly?)

New helper (one function in the runner): align the self-generated trace to the
oracle `meta["trace"]` token-by-token and report per-step agreement + the first
divergence step. This separates "can't track state" from "tracks state but fails
to terminate/format". For `s5_v1` this is the dense-supervision probe from Phase
2, now measured on **self-generated** traces.

Deliverable: `trace_token_acc` and `first_diverge` distributions per condition.

### E4 — Explicit vs implicit test-time compute (architecture angle)

`fprm`'s looped block is *implicit* recurrent depth; a scratchpad is *explicit*
depth. Run a 2×2: `{answer-only, trace-supervised}` × `{transformer, fprm (vary
n_loops)}`. Question: does explicit scratchpad substitute for implicit depth, or
stack with it? This is the clean architecture × test-time-compute interaction.

## Controls & validity

- **No leakage:** the oracle intermediates are never in the prompt; the model
  must compute them. The validity gate (`scripts/validate_suite.py`) must be
  re-run on any trace-scoring variant to confirm no new shortcut (e.g. a trace
  that lets the model echo the last-seen token).
- **Compute matching:** always pair CoT with a token-budget-matched answer-only
  baseline so "more tokens" is not confounded with "structured reasoning".
- **Format discipline:** raise `max_new_tokens` with the eval length; keep `stop_at="."`
  so traces terminate. Document the per-task budget formula.
- **Scoring robustness:** use `score_last_n` (final answer after trace) plus the
  holder/value decomposition; never score the trace *as* the answer.

## Implementation order (smallest diff first)

1. **E1** — API only, no training: a `--cot {none,free,structured}` flag on
   `scripts/eval_openrouter_grid.py` + per-task `max_new_tokens` scaling. Reuses
   existing scoring + decomposition.
2. **E3 helper** — one `trace_accuracy(pred_trace, gold_trace)` fn added to
   `factworld/tasks.py`, surfaced in the sweep/runner.
3. **E2** — extend `scripts/sweep.py` with `--use_trace` (already plumbed through
   `build_docs`) and a matched-budget answer-only arm.
4. **E4** — add `n_loops` to the sweep's arch matrix for `fprm`.

## Prediction summary

| task | expected CoT/scratchpad effect | why |
| --- | --- | --- |
| `recall_copy_v1`, `conflict_v1` | ~none | single-hop, at ceiling |
| `binding_v1` | small (unroll helps OOD only) | last-write-wins is cheap latently |
| `composite_copy_*`, `chain_v1` | **large** | externalizes the hop that currently breaks (holder→value) |
| `s5_v1` | **large if unrolled** | turns latent non-abelian tracking into step-wise computation |

The headline question this answers for the consolidated report: **is the
composition/state-tracking deficit a capacity limit or a routing limit?** If a
self-generated scratchpad recovers the value leg (E1/E2) while the scaffolded
ceiling (E1.4) is high, the deficit is routing — movable at inference time — not
capacity.
