# Autoregressive / test-time-compute — results (E1 API + E2 local)

**Question:** is the composition/state-tracking wall a capacity limit or a routing
limit — and does generating more (CoT, a scratchpad) before answering move it?

Two complementary experiments: **E1** (API models, inference-time CoT + leg-isolation
scaffolds) and **E2** (local from-scratch models, *trained* scratchpad). Run:
`scripts/experiment_autoregressive.py` (E1) and `scripts/sweep.py --worked_trace` (E2).

## E1 — API leg-isolation (the decomposition)

Four conditions on `composite_copy_v1@L16` and `s5_v1@L32`, n=30. Scored with
`last_n` + the holder/value decomposition. Reasoning models (Kimi, GLM) get an 8192-token
budget and no `.` stop so they can finish thinking.

- **none** — answer directly. *No output-format instruction* (so this is a no-format lower
  bound, not comparable to the format-instructed grid numbers — see the caveat below).
- **binding** — rewrite the query to ask *only* for the holder (binding leg in isolation).
- **scaffolded** — inject the *correct* holder; the model only recalls (recall-leg ceiling).
- **structured** — "write `holder: <g>` then `<g> <value>.`" (a parseable self-produced
  intermediate that externalizes the binding leg).

### `composite_copy_v1 @ L16` — last_n / holder / value

| model | none | binding | scaffolded | structured |
| --- | --- | --- | --- | --- |
| llama-3.3-70b | 0.00 / 0.07 / 0.00 | 0.40 / 0.40 / 0.00 | 0.97 / 1.00 / 0.97 | 0.13 / 0.20 / 0.00 |
| deepseek-chat | 0.00 / 0.10 / 0.00 | 0.47 / 0.47 / 0.00 | 0.80 / 1.00 / 0.80 | 0.37 / 0.37 / 0.00 |
| gpt-4o-mini | 0.00 / 0.00 / 0.00 | 0.37 / 0.37 / 0.00 | 1.00 / 1.00 / 1.00 | 0.10 / 0.10 / 0.00 |
| gemini-2.5-flash | 0.00 / 0.10 / 0.00 | 0.37 / 0.37 / 0.00 | 1.00 / 1.00 / 1.00 | 0.10 / 0.17 / 0.00 |
| **kimi-k2.6** | 0.00 / 0.00 / 0.00 | **0.73 / 0.73** / 0.00 | 1.00 / 1.00 / 1.00 | 0.60 / 0.60 / 0.00 |
| **glm-5.2** | 0.00 / 0.00 / 0.00 | **0.97 / 0.97** / 0.00 | 0.97 / 1.00 / 0.97 | **1.00 / 1.00** / 0.00 |

(value = 0.00 for binding by construction — gold is holder-only; N/A.)

### `s5_v1 @ L32`

Every model scores 0.00 under every condition, including scaffolded. S5 is a genuine
non-abelian state-tracking wall with no decoupled single leg — consistent with Phase 2.

### Reading it — a capability ladder

1. **The recall leg is never the bottleneck.** Given the holder (scaffolded), every model
   recalls its value 0.80–1.00. Recall is trivially in-capacity.
2. **Binding separates reasoners from non-reasoners.** Asked only for the holder, the
   reasoning models (kimi 0.73, glm 0.97) far outperform the rest (0.37–0.47). Test-time
   reasoning *does* buy state-tracking.
3. **Routing is the wall that survives even strong reasoners.** Under `structured`, glm
   writes the correct holder **1.00** of the time — yet its *value* stays 0.00. It resolves
   the binding leg correctly but does not feed that result into the recall lookup. kimi
   shows the same pattern (holder 0.60, value 0.00). The 2-hop composition is not solved by
   making the model externalize the first hop.

So: for weaker models the wall is **binding**; for strong reasoners it is **routing**.
Scaffolded (oracle-provided intermediate) is the only condition that recovers the value leg.

> **Format caveat.** The `none` and `structured` conditions here carry *no* output-format
> instruction, so their low scores partly reflect format, not just reasoning. The
> format-instructed grid (`docs/openrouter/results-natural.md`) is the fair capability
> ceiling: there llama-3.3-70b and glm-5.2 reach **0.80** on `composite_copy_v1`. The value
> of E1 is the *relative* decomposition (binding/scaffolded isolate the legs), which holds
> regardless of format.

## E2 — trained scratchpad (local, the complement)

Train `gdp_hybrid`/`fprm` on `composite_copy_scale_v1` and `s5_v1` with the oracle
**worked-trace** as a per-step scratchpad (`prompt → trace → answer`), vs the answer-only
baseline. Scored on the committed answer tail (`prefix_decomp(trace_mode=True)`).

### `composite_copy_scale_v1 @ L16` — holder / value (mean over seeds)

| regime | holder | value | converge |
| --- | --- | --- | --- |
| answer-only (gdp_hybrid) | 0.95 | 0.51 | 40% |
| **trained scratchpad** (gdp_hybrid) | **0.08** | **0.00** | **0%** |
| answer-only (fprm) | 0.99 | 0.20 | 0% |
| **trained scratchpad** (fprm) | **0.06** | **0.00** | **0%** |

**A trained self-generated scratchpad makes the model worse, not better** — holder
collapses 0.95→0.08. The model emits a structured-looking but *wrong* per-step trace
(error compounding), and that wrongness cascades into the answer. This is the exact opposite
of the API scaffolded result, where the *oracle-provided* intermediate gives 0.97–1.00.

This matches Phase 2's finding that `s5_v1` worked-traces "learn train length but compound
at generation," which is why `s5_v1` is flagged `experimental`.

## Synthesis — when does test-time compute help?

| intermediate | who provides it | composite result |
| --- | --- | --- |
| none (latent) | — | strong models 0.80 *with format help*; the latent circuit works for the final answer |
| **oracle** scaffold | provided (correct) | **0.80–1.00** — recall leg trivial once the holder is given |
| **self-generated** CoT/trace | the model | does not recover the value leg (API) and **collapses** the holder leg (local) |

**Headline:** test-time compute helps only when the intermediate is **correct**. An
oracle-provided holder unlocks the recall leg for every model; a self-generated holder/trace
does not — because state-tracking and routing are the hard parts, and autoregressively
unrolling them compounds errors rather than verifying them. The wall is movable at inference
time only by *giving* the model the binding answer, not by asking it to produce it.

## Files

- `scripts/experiment_autoregressive.py` — E1 runner (binding/none/structured/scaffolded).
- `scripts/sweep.py --worked_trace` — E2 trained-scratchpad sweep (trace-mode scoring).
- `results/autoregressive_api_*.jsonl`, `results/autoregressive_glm_*.jsonl` — per-example E1 data.
- `results/sweep_e2_trace_fixed_*.md` — E2 table.
- `factworld/tasks.py` — `decompose_composite`, `trace_accuracy`, `content_tokens`.
