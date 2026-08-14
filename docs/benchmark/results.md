# FactWorld frontier benchmark — results

Generated 2026-08-14 05:36 UTC from `results/benchmark/history.jsonl` (807 latest cells).

## Settings

Canonical metric: **match** — strip a trailing period from both sides and compare the model's first len(gold) whitespace tokens to the gold answer; binary per item, no partial credit (`factworld.tasks.score_relaxed`). Containment is the one published diagnostic.
Figures draw a dotted reference line at match 0.8.
Error bars / intervals: Wilson 95% CI.

Observed generation settings (effort -> max_new_tokens, stop_at; annotated with the facets that ran under each combo):

- effort=default: max_new_tokens=2048, stop_at=None — facets: chain_depth, composite_length, decomposition, dose_response, floor, s5_concrete, sanity
- effort=high: max_new_tokens=16384, stop_at=None — facets: chain_nowrap, s5_chain, s5_concrete
- effort=high: max_new_tokens=24576, stop_at=None — facets: s5_chain
- effort=high: max_new_tokens=32768, stop_at=None — facets: chain_nowrap, s5_chain, s5_concrete
- effort=high: max_new_tokens=49152, stop_at=None — facets: s5_chain
- effort=high: max_new_tokens=65536, stop_at=None — facets: s5_chain
- effort=high: max_new_tokens=8192, stop_at=None — facets: chain_depth, commutative, composite_length, dose_response, s5_concrete
- effort=low: max_new_tokens=8192, stop_at=None — facets: dose_response
- effort=medium: max_new_tokens=8192, stop_at=None — facets: dose_response
- effort=minimal: max_new_tokens=2048, stop_at=None — facets: composite_length, decomposition, dose_response, floor, sanity
- effort=minimal: max_new_tokens=96, stop_at=None — facets: chain_instant, gap_stability, recall_load, zero_budget
- effort=none: max_new_tokens=2048, stop_at=None — facets: composite_length, decomposition, dose_response, floor, sanity
- effort=none: max_new_tokens=96, stop_at=None — facets: chain_instant, gap_stability, recall_load, zero_budget
- effort=xhigh: max_new_tokens=16384, stop_at=None — facets: s5_chain
- effort=xhigh: max_new_tokens=24576, stop_at=None — facets: s5_chain
- effort=xhigh: max_new_tokens=32768, stop_at=None — facets: s5_chain
- effort=xhigh: max_new_tokens=49152, stop_at=None — facets: s5_chain
- effort=xhigh: max_new_tokens=65536, stop_at=None — facets: s5_chain
- effort=xhigh: max_new_tokens=98304, stop_at=None — facets: s5_chain

## Instant headline (current roster)

Current roster only (factworld.benchmark.MODELS); models dropped from the roster render in the archived-models section below.

The benchmark is a composition instrument: recall and state tracking are the component abilities, and 'instant' cells (reasoning off, hard one-line answer contract) measure whether the model composes them in-weights — the composition gap column is the deficit. 'thinking' cells measure composition with reasoning: ~ceiling at canonical settings for this roster, so the state-stress columns (chain d128 at k=257, s5 @L256) carry the thinking discrimination.

Instant cells: task **composite_copy_v2** with reasoning off (effort=none) under a one-line answer contract (settings.contract=true); match. Escalated cells show the CANONICAL first attempt at the shared base budget, with the escalated rerun as a parenthesised diagnostic.

Notation: `@Ln` = stream length (events, or hops for chain depth d); `@Ntok` = a completion-token budget. Instant escalations render `(diag x.xx @512tok)`; thinking cells rerun at a raised budget render it with the number, e.g. `1.00 @32,768tok (raised budget)`.

History also contains zero-budget cells on composite_copy_v1; the zero-budget columns below use the latest task's records (composite_copy_v2) only — the archived task's cells remain in the per-cell tables.

| Model | instant: recall (sanity, recall_copy_v1) | instant: state tracking (binding_only @L16, v2) | instant: composed @L16 (match, v2) | instant: composed @L64 (v2) | instant: composition gap (binding_only - composed @L16) | instant: replicate noise (|composed - replicate| @L16) |
|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 0.78 | 0.72 | 0.43 | +0.06 | ±0.05 |
| google/gemini-3.6-flash | 1.00 | 0.69* | 0.67* | 0.26* | +0.02* | ±0.02 |
| openai/gpt-5.6-sol | 1.00 | 0.82 | 0.65 | 0.33 | +0.17 | ±0.05 |
| anthropic/claude-sonnet-5 | 0.97 | 0.77 | 0.62 (diag 0.76 @512tok)† | 0.32 (diag 0.66 @512tok)† | +0.15† | ±0.03 |
| openai/gpt-5.5 | 1.00 | 0.80 | 0.46 | 0.33 | +0.34 | ±0.00 |
| deepseek/deepseek-v4-pro | 1.00 | 0.51 | 0.44 | 0.19 | —ᶠ | ±0.00 |
| z-ai/glm-5.2 | 1.00 | 0.71 | 0.38† | 0.13 | +0.33† | ±0.01 |
| moonshotai/kimi-k3 | 1.00 | 0.65 | 0.33 | 0.29 | +0.32 | ±0.01 |
| nvidia/nemotron-3-ultra-550b-a55b | 1.00 | 0.49 | 0.33 | 0.12 | —ᶠ | ±0.03 |
| qwen/qwen3.7-max | 1.00 | 0.51 | 0.24 | 0.08 | —ᶠ | ±0.01 |
| recency heuristic (floor, composite_copy_v2) | — | 0.04 | 0.04 | 0.06 | — | — |
| object-filter floor (composite_copy_v2) | — | 0.41 | 0.41 | 0.15 | — | — |

Read small-L zero-budget cells against the object-filter floor, not chance: the floor is inherent to last-write-wins (filter the stream to the queried object, guess among its w writes) and decays only ~1/L, so it sits well above chance at L16 — a score near the floor row shows object filtering, not state tracking; genuine last-write resolution has to clear it.

(*) off-arm ran effort=minimal (model cannot disable reasoning).

(†, trigger 1 — visible working) the canonical attempt's completion carries short visible working instead of a bare answer: median (per-example) or mean ctok (completion tokens) per call > 32 (~3x the 8-11 token answers), or the cell needed a budget escalation.

(†, trigger 2 — covert reasoning) the model reasoned despite effort=none: mean rtok (reasoning tokens) per call > 2 on the published attempt. Where MORE THAN 50% of the canonical attempt's calls carry reasoning tokens the covert reasoning is pervasive and the cell renders as the explicit upper bound ≤x†.

(‡) cap-escape: per-example ctok exceeded settings.max_new_tokens on >10% of calls (the provider did not enforce the cap); token counts and budget comparisons for those cells are not cap-comparable.

(diag x.xx @512tok) escalated diagnostic: the cell was rerun once at an escalated token budget after majority finish=length; the CANONICAL number is the first attempt at the shared base budget — the escalated value is a marked diagnostic, not the headline.

(—ᶠ) gap not interpretable where the state-tracking component sits at the floor: the binding cell's Wilson CI overlaps the object-filter floor's, so the composed cell is floor-shaped too and binding − composed reads floor − floor ≈ 0 by construction.

recency heuristic (floor, <task>): one-line floor recomputed at render time on the exact deterministic items of the task named in the row label (the same task as the zero-budget columns) — answer the LAST event's recipient plus that holder's fact (binding leg: the last recipient).

object-filter floor (<task>): E[1/w] recomputed at render time on the same exact items — for each item, 1/(number of writes to the queried object): a reader that filters events by the queried object but picks a RANDOM write (no last-write-wins resolution) scores this with no state tracking at all; the binding leg derives from the same items, so its floor is the same 1/w.

n/a = facet/cell not run for this model; — = run, but no qualifying value.

⊘ = not measurable at this budget, or the cell's calls failed; ≤x† = upper bound, covert reasoning on most calls; neither participates in orderings.

composition gap = state tracking (binding_only @L16) - composed @L16, marks from either input cell propagated. recall|holder is ~1.0 for every roster model (the scaffolded leg), so if composition were free the composed cell would match the binding leg; the gap is the composition deficit.

replicate noise: the zero_budget replicate leg (recorded as end_to_end in earlier runs) builds prompts IDENTICAL to the composed @L16 cell (same runner path), so |composed - replicate| is a test-retest delta; max across models = 0.06 — read that as the run-to-run noise bar on the headline numbers (including the gap column). Future runs keep this arm intentionally as leg='replicate'.

## Thinking headline (current roster)

Thinking-regime state-stress cells (effort=high): chain d128 is a pointer chase 128 hops deep at fixed breadth k=257; s5 @L256 is non-abelian state tracking over 256 events. The s5@128 mean ctok/call column measures efficiency on the matched L128 cell that every current-roster model runs.

Notation: `@Ln` = stream length (events, or hops for chain depth d); `@Ntok` = a completion-token budget. Instant escalations render `(diag x.xx @512tok)`; thinking cells rerun at a raised budget render it with the number, e.g. `1.00 @32,768tok (raised budget)`.

| Model | thinking: chain d128 (chain_nowrap, k=257, match) | thinking: s5 @L256 (s5_concrete, match) | thinking: s5@128 mean ctok/call |
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
| moonshotai/kimi-k3 | 1.00 | 0.80 (trunc 0.16) | 11355 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.60 (trunc 0.32) | 0.80 (trunc 0.20) | 12250 |
| z-ai/glm-5.2 | 0.92 (calls failed 0.04) (trunc 0.04) | 0.76 (trunc 0.04) | 6282 |

Thinking columns: n=25 per cell; Wilson intervals ≈ ±0.15–0.19. 2 current-roster thinking cells that take part in orderings were run more than once at identical settings (Repeat runs below) and their run-to-run spreads reach 0.16 — read a difference smaller than that as noise, not an ordering. Marked cells set no part of this bar: their repeat spread is variance in what the mark names, not measurement noise.

s5@128 mean ctok/call: the cell's total completion tokens divided by n on the matched s5_concrete L128 cell (run by every current-roster model). This replaces ctok/solve, which averaged only over cells a model SOLVED and therefore rewarded models that failed early (selection bias: the published 2.7x opus-vs-kimi ctok/solve gap is ~1.4x on the matched cell).

Two completion-token statistics appear, and they are named apart everywhere. **mean ctok/call** is the cell's usage total divided by n — the efficiency columns and the report quote this one, because it is the token spend a practitioner pays. **median ctok/call** is the per-example median; it is used only as the instant regime's visible-working trigger (†), where a handful of long completions must not drag a cell of bare answers over the line.

The work rate is a property of the model AND of the system prompt the benchmark sends, not of the model alone. Three arms over the identical deterministic items (openai/gpt-5.6-sol, s5_chain_v3 @L64, n=25, top effort, 49,152-token budget) differ only in that prompt. Under the scored protocol prompt ("You are taking a short test... no explanation") the model matches 0.68, works 0.68 of its calls and answers event-blind on 0.33; with the two clauses that read as instructions to spend less effort removed and the identical answer-format contract kept, it matches 0.96, works 0.96 and answers event-blind on 0.04 — 7 of the 8 calls the scored prompt left unworked are worked under the neutral one. The third arm sends no system prompt at all: it works 0.96 and matches 0.84, and that 0.84 is a format reading rather than a composition one — all 24 of its worked calls carry the gold value, three of them committed in LaTeX (`**Answer: \(g15\)**`, `\boxed{g0}`) that the committed-answer rule does not read. The event-blind rates run through the published column's eligibility rule, so they read against it. Every scored thinking cell carries the scored prompt, so the completion-token columns are token spend under an instruction to be brief. One model, one length, n=25: `results/probes/sol_system_prompt_20260727.json`.

(ᵘ) unworked answers on a large fraction of calls; the cell measures engagement, not capability. Set when more than 10% of a thinking cell's calls fall below the 512-token working line AND those calls score materially below the cell's worked calls (disjoint Wilson 95% intervals) — the rate alone is not enough, since answering a shallow cell correctly without visible working is a measurement. Like ⊘ >budget and ≤x†, a marked cell takes no part in orderings. Engagement moves between runs: the marked cells repeated at identical settings spread up to 0.32 run to run, against 0.16 across the cells that set the thinking noise bar.

(trunc 0.NN) the cell's truncation rate, where some calls ended finish=length: those calls score 0 whatever the model knew, so the published score is a lower bound. Above 0.50 the cell renders ⊘ >budget instead of a number.

(calls failed 0.NN) the fraction of the cell's calls the API rejected — a billing, rate-limit or provider failure, so the model never saw those prompts. They score 0 like any other empty answer, so the published score is a lower bound. Above 10% the cell renders ⊘ calls failed instead of a number: the calls did not happen, so the cell is not a measurement of the model, and like every ⊘ cell it takes no part in orderings. Failed calls are also out of the per-call diagnostics — the worked-calls split reads the calls that completed, so a billing failure cannot render as the unworked-answers pathology.

## S5 efficiency ranking

S5 efficiency ranking: models sorted by s5 @L256 score, then by s5@128 mean completion tokens per call (lower is better) on the matched s5_concrete L128 cell (the cell every current-roster model runs). At s5 @L256 several models hit 1.00, so token efficiency is the practical discriminator.

| Model | s5 @L256 | s5@128 mean ctok/call |
|---|---|---|
| anthropic/claude-fable-5 | 1.00 | 6405 |
| openai/gpt-5.5 | 1.00 | 6989 |
| google/gemini-3.6-flash | 1.00 | 8234 |
| muse-spark-1.1 | 1.00 | 9704 |
| deepseek/deepseek-v4-pro | 1.00 | 10043 |
| anthropic/claude-sonnet-5 | 1.00 | 11866 |
| anthropic/claude-opus-4.8 | 1.00 | 12683 |
| openai/gpt-5.6-sol | 0.92 | 2657 |
| x-ai/grok-4.5 | 0.88 | 8069 |
| qwen/qwen3.7-max | 0.80 | 7904 |
| moonshotai/kimi-k3 | 0.80 (trunc 0.16) | 11355 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.80 (trunc 0.20) | 12250 |
| z-ai/glm-5.2 | 0.76 (trunc 0.04) | 6282 |

## s5_chain ranking (headline)

s5_chain is the headline composite stressor: k=16 agents with an a0 pointer map, L order-sensitive swap/cycle events on the pointer targets, then an 8-hop serial dereference query (`what is a0 of ... of gX? (8 hops)`). Every item is gated so the query path visits 9 distinct agents: answering the queried agent, or any fixed hop, scores exactly 0, and chance is 1/16. Protocol: the shared xhigh arm for every model (cross-model fairness over per-vendor ceilings), budgets sized so truncation stays a rounding error, n=25 per cell. Sorted by the @L96 score (the full-roster cell), then by the @L128 top-cluster separator, then by mean completion tokens per call on the matched @L64 cell. A cell marked `ᵘ` or `⊘` takes no part in the ordering: its row sorts last on its name whatever its score, here and in the figure.

History also contains s5_chain cells on s5_chain_v4. The table publishes s5_chain_v3: a replacement version publishes once it covers the ranked cell (@L96, effort=xhigh) for every model the published version covers, so a battery in flight neither empties nor shrinks the table — s5_chain_v4 is missing it for every model in the table. The unpublished version's cells are in the per-cell tables meanwhile.

| Model | s5_chain @L96 | @L128 | worked calls @L96 | event-blind @L96 | s5_chain@64 mean ctok/call |
|---|---|---|---|---|---|
| anthropic/claude-fable-5 | 1.00 | 1.00 | 1.00 | 0.00 | 5014 |
| openai/gpt-5.5 | 1.00 | 1.00 | 1.00 | 0.00 | 9343 |
| x-ai/grok-4.5 | 1.00 | 0.96 | 1.00 | 0.00 | 7711 |
| anthropic/claude-opus-4.8 | 0.96 | 0.96 | 1.00 | 0.00 | 9702 |
| moonshotai/kimi-k3 | 0.96 | 0.96 | 1.00 | 0.00 | 10941 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.96 @98,304tok (raised budget) | 0.96 | 1.00 | 0.00 | 17071 |
| muse-spark-1.1 | 0.96 | 0.92 | 1.00 | 0.00 | 12484 |
| google/gemini-3.6-flash | 0.92 | 0.96 | 1.00 | 0.00 | 8166 |
| anthropic/claude-sonnet-5 | 0.92 | 0.96 | 1.00 | 0.00 | 12729 |
| deepseek/deepseek-v4-pro | 0.92 | 0.96 | 1.00 | 0.00 | 17052 |
| z-ai/glm-5.2 | 0.92 | 0.80 (trunc 0.04) | 1.00 | 0.00 | 17982 |
| qwen/qwen3.7-max | 0.72 | 0.44 | 1.00 | 0.00 | 12588 |
| openai/gpt-5.6-sol | 0.60ᵘ (3 runs, spread 0.24) | 0.80ᵘ (3 runs, spread 0.12) | 0.56 (1.00/0.09) | 0.42 | 2444 |

worked calls: the fraction of a cell's calls whose completion exceeds 512 completion tokens, with (match | worked / match | unworked) beside it where the cell has calls of both kinds. The signal is completion tokens, not reasoning tokens: reasoning-token accounting is not comparable across providers (on this history the median ctok-minus-rtok is 5184 for sonnet-5, 4184 for opus and 1675 for fable, but 2-17 for every other model, and glm reports rtok=0 on correct 8k-token answers). Cells run before per-example token logging report n/a.

The work rate is a property of the model AND of the system prompt the benchmark sends, not of the model alone. Three arms over the identical deterministic items (openai/gpt-5.6-sol, s5_chain_v3 @L64, n=25, top effort, 49,152-token budget) differ only in that prompt. Under the scored protocol prompt ("You are taking a short test... no explanation") the model matches 0.68, works 0.68 of its calls and answers event-blind on 0.33; with the two clauses that read as instructions to spend less effort removed and the identical answer-format contract kept, it matches 0.96, works 0.96 and answers event-blind on 0.04 — 7 of the 8 calls the scored prompt left unworked are worked under the neutral one. The third arm sends no system prompt at all: it works 0.96 and matches 0.84, and that 0.84 is a format reading rather than a composition one — all 24 of its worked calls carry the gold value, three of them committed in LaTeX (`**Answer: \(g15\)**`, `\boxed{g0}`) that the committed-answer rule does not read. The event-blind rates run through the published column's eligibility rule, so they read against it. Every scored thinking cell carries the scored prompt, so the completion-token columns are token spend under an instruction to be brief. One model, one length, n=25: `results/probes/sol_system_prompt_20260727.json`.

event-blind: the fraction of a cell's predictions equal to the 8-hop dereference of the INITIAL pointer map — the answer a model gives if it reads the fact block and skips the whole event stream. Items where that answer coincides with the gold answer are dropped (it coincides at chance, 0.063-0.078 across lengths against 1/16), so the rate names which cheaper task the model substituted rather than crediting luck. Across the roster's 52 scored s5_chain cells the blind answer is given on 46 of 1,261 eligible items, 42 of them by openai/gpt-5.6-sol; over the other 12 models the rate is 4 of 1,164 (0.003), against 0.063 for a uniform guess.

(ᵘ) unworked answers on a large fraction of calls; the cell measures engagement, not capability. Set when more than 10% of a thinking cell's calls fall below the 512-token working line AND those calls score materially below the cell's worked calls (disjoint Wilson 95% intervals) — the rate alone is not enough, since answering a shallow cell correctly without visible working is a measurement. Like ⊘ >budget and ≤x†, a marked cell takes no part in orderings. Engagement moves between runs: the marked cells repeated at identical settings spread up to 0.32 run to run, against 0.16 across the cells that set the thinking noise bar.

Repeat runs at identical settings, the table publishing the last run: openai/gpt-5.6-sol @L96 0.72/0.84/0.60; openai/gpt-5.6-sol @L128 0.68/0.80/0.80; openai/gpt-5.6-sol @L64 0.64/0.84/0.60.

The chain column reads the `chain_nowrap` facet only (staircase k=2d+1, so the d128 cell is k=257). `chain_v2` builds a single k=6 pointer cycle and measures depth only for depths < k (`factworld/tasks.py`: "Depths stay < k so the cycle never wraps"); `chain_depth` cells at depth >= 6 wrapped the cycle (gold == start agent at depths 12/24/48; effective difficulty depth mod 6), measure the wrapped task rather than depth, and are marked `INVALID (k=6 cycle wrap — task redesigned as chain_nowrap)` in the tables below and excluded from the chain figure.

## Instant stress rows (recall under load; chain d16)

Two instant cells beyond the composite headline, same protocol (reasoning off, one-line answer contract, 96-token cap; marks and escalated diagnostics as in the headline). recall_load scales the recall pool with the length (recall_copy_v1 @L64, pool 64, n=50): single-query deferred recall under working-set load. chain_instant runs chain_v2 d16 on the same k=33 staircase items as the thinking d16 cell (n=25): the within-item regime contrast for depth. The floor row is the uniform guess over the answer pool; escalated cells show the CANONICAL first attempt with the escalated rerun as a parenthesised diagnostic.

| Model | instant: recall under load (recall_load, recall_copy_v1 pool-64 @L64) | instant: chain d16 (chain_instant, chain_v2, k=33) |
|---|---|---|
| anthropic/claude-fable-5 | n/a | n/a |
| anthropic/claude-opus-4.8 | 1.00 | 0.00 (diag 0.96 @512tok)† |
| anthropic/claude-sonnet-5 | 1.00 | 0.28 (diag 0.96 @512tok)† |
| deepseek/deepseek-v4-pro | 1.00 | 0.00 |
| google/gemini-3.6-flash | 1.00* | 0.00* |
| moonshotai/kimi-k3 | 1.00 | 0.04 |
| muse-spark-1.1 | n/a | n/a |
| nvidia/nemotron-3-ultra-550b-a55b | 1.00 | 0.00 |
| openai/gpt-5.5 | 1.00 | 0.08 |
| openai/gpt-5.6-sol | 1.00 | 0.00 |
| qwen/qwen3.7-max | 1.00 | 0.00 |
| x-ai/grok-4.5 | n/a | n/a |
| z-ai/glm-5.2 | 1.00 | 0.00† |
| uniform-guess floor (chance) | 0.016 (1/64) | 0.030 (1/33) |

## Archived models (dropped from the roster)

Models present in history but no longer in factworld.benchmark.MODELS, with their v1-facet columns (historical facet names). Numbers in this table are on retired v1 tasks/facets (pre-redesign samplers and settings) and are NOT comparable to the current headline. Their per-cell rows — any facet — remain in the tables below.

| Model | dose_response (match) | composite_length (match @ L512, high) | decomposition (bind / e2e / scaffold) |
|---|---|---|---|
| google/gemini-3.1-pro-preview | 0.98 @ high | 0.90 | 1.00 / 0.98 / 1.00 |
| google/gemini-3.5-flash | 1.00 @ high | 0.97 | 0.82 / 0.44 / 1.00 |
| meta-llama/llama-4-maverick | 0.16 @ default | 0.10 | 0.96 / 0.18 / 1.00 |
| moonshotai/kimi-k2.6 | 0.98 @ high | 0.97 | 0.78 / 0.26 / 1.00 |
| openai/gpt-5.4 | 0.96 @ high | 0.93 | 0.86 / 0.30 / 1.00 |
| x-ai/grok-4.3 | 0.22 @ high | 0.73 | 0.16 / 0.18 / 1.00 |
| x-ai/grok-build-0.1 | — | — | — / — / — |

## v1 archived facets (pre-redesign)

Legacy headline columns for the pre-redesign v1-only facets (dose_response, composite_length, decomposition), current-roster models only; superseded by the ladder headline above. Numbers in this table are on retired v1 tasks/facets (pre-redesign samplers and settings) and are NOT comparable to the current headline. Per-cell rows remain in the tables below.

| Model | dose_response (match) | composite_length (match @ L512, high) | decomposition (bind / e2e / scaffold) |
|---|---|---|---|
| anthropic/claude-opus-4.8 | 0.96 @ high | 1.00 | 0.88 / 0.08 / 1.00 |
| anthropic/claude-sonnet-5 | 0.94 @ high | 1.00 | 0.86 / 0.00 / 1.00 |
| deepseek/deepseek-v4-pro | 0.92 @ high | 0.87 | 0.24 / 0.24 / 0.98 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.84 @ high | 0.23 | 0.60 / 0.22 / 1.00 |
| openai/gpt-5.5 | 1.00 @ high | 1.00 | 0.96 / 0.74 / 1.00 |
| qwen/qwen3.7-max | 0.92 @ high | 1.00 | 0.60 / 0.18 / 1.00 |
| z-ai/glm-5.2 | 0.94 @ high | 0.93 | 0.56 / 0.20 / 1.00 |

## Full per-cell results

match is the CANONICAL value (first attempt for escalated cells; the escalated diagnostic is in the note column). ‡ = cap-escape (see headline footnotes). ⊘ calls failed = the API rejected the cell's calls, so it has no score and no interval. INVALID chain_depth cells are quarantined in the provenance section at the end.

| Model | Facet | Task | Length | Arm | n | match [95% CI] | containment (diagnostic) | note |
|---|---|---|---|---|---|---|---|---|
| anthropic/claude-fable-5 | chain_nowrap | chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-fable-5 | chain_nowrap | chain_v2 | 32 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| anthropic/claude-fable-5 | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-fable-5 | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-fable-5 | commutative | commutative_v1 | 64 | effort=high | 25 | 0.84 [0.65, 0.94] | 0.84 | — |
| anthropic/claude-fable-5 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-fable-5 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-fable-5 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-fable-5 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-fable-5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-fable-5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.96 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| anthropic/claude-opus-4.8 | chain_instant | chain_v2 | 16 | contract, effort=none | 25 | 0.04 [0.01, 0.20] | 0.96 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.08 [0.02, 0.25] | 0.08 | truncation 0.04 |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v2 | 32 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | commutative | commutative_v1 | 64 | effort=high | 25 | 0.80 [0.61, 0.91] | 0.80 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.40 [0.25, 0.58] | 0.67 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.30 [0.17, 0.48] | 0.70 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.10 [0.03, 0.26] | 0.57 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.23 [0.12, 0.41] | 0.37 | — |
| anthropic/claude-opus-4.8 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.88 [0.76, 0.94] | — | — |
| anthropic/claude-opus-4.8 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.08 [0.03, 0.19] | 0.96 | — |
| anthropic/claude-opus-4.8 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| anthropic/claude-opus-4.8 | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.96 [0.87, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.36 [0.24, 0.50] | 0.62 | — |
| anthropic/claude-opus-4.8 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.00 [0.00, 0.11] | 1.00 | — |
| anthropic/claude-opus-4.8 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 50 | 0.50 [0.37, 0.63] | — | — |
| anthropic/claude-opus-4.8 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 50 | 0.54 [0.40, 0.67] | 0.54 | — |
| anthropic/claude-opus-4.8 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v1 | 16 | effort=high | 25 | 0.08 [0.02, 0.25] | 0.08 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v1 | 32 | effort=high | 25 | 0.20 [0.09, 0.39] | 0.20 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v1 | 64 | effort=high | 25 | 0.20 [0.09, 0.39] | 0.20 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 16 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | truncation 0.04 |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.82 [0.73, 0.88] | — | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.84 [0.76, 0.90] | 0.86 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.84 [0.76, 0.90] | 0.88 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.57 [0.47, 0.66] | 0.57 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.78 [0.69, 0.85] | — | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.77 [0.68, 0.84] | 0.85 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 1.00 [0.96, 1.00] | — | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.72 [0.63, 0.80] | 0.79 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.43 [0.34, 0.53] | 0.43 | — |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.28 [0.14, 0.48] | 0.96 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | chain_instant | chain_v2 | 16 | contract, effort=none | 25 | 0.20 [0.09, 0.39] | 1.00 | escalated @512tok diagnostic 1.00; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.72 [0.52, 0.86] | 0.72 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.24 [0.11, 0.43] | 0.24 | truncation 0.04 |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.04 [0.01, 0.20] | 0.04 | truncation 0.28 |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v2 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | commutative | commutative_v1 | 64 | effort=high | 50 | 0.64 [0.50, 0.76] | 0.70 | 2 runs, spread 0.04 |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.03 [0.01, 0.17] | 0.83 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.00 [0.00, 0.11] | 0.87 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.00 [0.00, 0.11] | 0.70 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.00 [0.00, 0.11] | 0.80 | — |
| anthropic/claude-sonnet-5 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.86 [0.74, 0.93] | — | — |
| anthropic/claude-sonnet-5 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.00 [0.00, 0.07] | 0.74 | — |
| anthropic/claude-sonnet-5 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| anthropic/claude-sonnet-5 | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.94 [0.84, 0.98] | 0.94 | — |
| anthropic/claude-sonnet-5 | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.00 [0.00, 0.07] | 0.92 | — |
| anthropic/claude-sonnet-5 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.00 [0.00, 0.11] | 0.87 | truncation 0.03 |
| anthropic/claude-sonnet-5 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 50 | 0.64 [0.50, 0.76] | — | escalated @512tok diagnostic 0.72; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 50 | 0.50 [0.37, 0.63] | 0.64 | escalated @512tok diagnostic 0.64; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 25 | 0.84 [0.65, 0.94] | 0.88 | truncation 0.04 |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.88 | truncation 0.04 |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.96 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 32 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 64 | effort=high | 25 | 0.76 [0.57, 0.89] | 0.76 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 96 | effort=high | 25 | 0.84 [0.65, 0.94] | 0.84 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 0.97 [0.83, 0.99] | 1.00 | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.71 [0.61, 0.79] | — | truncation 0.03 |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.57 [0.47, 0.66] | 0.75 | escalated @512tok diagnostic 0.75; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.59 [0.49, 0.68] | 0.77 | escalated @512tok diagnostic 0.77; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.28 [0.20, 0.37] | 0.67 | escalated @512tok diagnostic 0.67; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.77 [0.68, 0.84] | — | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.65 [0.55, 0.74] | 0.82 | escalated @512tok diagnostic 0.82; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 1.00 [0.96, 1.00] | — | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.62 [0.52, 0.71] | 0.76 | escalated @512tok diagnostic 0.76; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.32 [0.24, 0.42] | 0.66 | escalated @512tok diagnostic 0.66; canonical = first attempt @96tok |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| deepseek/deepseek-v4-pro | chain_instant | chain_v2 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.04 | escalated @512tok diagnostic 0.04; canonical = first attempt @96tok; truncation 0.04 |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | truncation 0.04 |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.08 [0.02, 0.25] | 0.08 | truncation 0.76 |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.83 [0.66, 0.93] | 0.83 | truncation 0.03 |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.27 [0.14, 0.44] | 0.27 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.10 [0.03, 0.26] | 0.10 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | truncation 0.03 |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.03 [0.01, 0.17] | 0.03 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.87 [0.70, 0.95] | 0.87 | truncation 0.10 |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.07 [0.02, 0.21] | 0.07 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.24 [0.14, 0.37] | — | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.24 [0.14, 0.37] | 0.24 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 0.98 [0.90, 1.00] | — | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.92 [0.81, 0.97] | 0.92 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 0.92 [0.81, 0.97] | 0.92 | truncation 0.02 |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 0.84 [0.71, 0.92] | 0.84 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.28 [0.17, 0.42] | 0.28 | — |
| deepseek/deepseek-v4-pro | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.17 [0.07, 0.34] | 0.17 | — |
| deepseek/deepseek-v4-pro | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 25 | 0.84 [0.65, 0.94] | 0.84 | truncation 0.16 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | truncation 0.04 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 25 | 0.68 [0.48, 0.83] | 0.68 | truncation 0.28 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 25 | 0.84 [0.65, 0.94] | 0.84 | truncation 0.16 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v4 | 32 | sysprompt=04153d7439, effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | truncation 0.08 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v4 | 128 | sysprompt=04153d7439, effort=xhigh | 25 | ⊘ calls failed | — | calls failed 0.36 |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | truncation 0.04 |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.41 [0.32, 0.51] | — | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.33 [0.25, 0.43] | 0.33 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.32 [0.24, 0.42] | 0.32 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.15 [0.09, 0.23] | 0.15 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.51 [0.41, 0.61] | — | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.44 [0.35, 0.54] | 0.44 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 1.00 [0.96, 1.00] | — | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.44 [0.35, 0.54] | 0.44 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.19 [0.13, 0.28] | 0.19 | — |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 16 | effort=minimal | 30 | 0.93 [0.79, 0.98] | 0.93 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.67 [0.49, 0.81] | 0.67 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 64 | effort=minimal | 30 | 0.57 [0.39, 0.73] | 0.57 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.70 [0.52, 0.83] | 0.70 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 128 | effort=minimal | 30 | 0.77 [0.59, 0.88] | 0.77 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.90 [0.74, 0.97] | 0.90 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 512 | effort=minimal | 30 | 0.90 [0.74, 0.97] | 0.90 | — |
| google/gemini-3.1-pro-preview | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=minimal | 50 | 1.00 [0.93, 1.00] | — | — |
| google/gemini-3.1-pro-preview | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=minimal | 50 | 0.98 [0.90, 1.00] | 0.98 | — |
| google/gemini-3.1-pro-preview | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=minimal | 50 | 1.00 [0.93, 1.00] | — | — |
| google/gemini-3.1-pro-preview | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.98 [0.90, 1.00] | 0.98 | — |
| google/gemini-3.1-pro-preview | dose_response | composite_copy_v1 | 16 | effort=minimal | 50 | 0.96 [0.87, 0.99] | 0.96 | — |
| google/gemini-3.1-pro-preview | floor | s5 | 16 | rendering=abstract_stated, effort=minimal | 30 | 0.70 [0.52, 0.83] | 0.70 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | truncation 0.04 |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| google/gemini-3.1-pro-preview | sanity | conflict_v1 | 4 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.1-pro-preview | sanity | recall_copy_v1 | 6 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_instant | chain_v1 | 16 | contract, effort=minimal | 25 | 0.00 [0.00, 0.13] | 1.00 | escalated @512tok diagnostic 1.00; canonical = first attempt @96tok |
| google/gemini-3.5-flash | chain_instant | chain_v2 | 16 | contract, effort=minimal | 25 | 0.04 [0.01, 0.20] | 0.96 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | commutative | commutative_v1 | 64 | effort=high | 25 | 0.80 [0.61, 0.91] | 0.80 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 16 | effort=minimal | 30 | 0.60 [0.42, 0.75] | 0.70 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 64 | effort=minimal | 30 | 0.40 [0.25, 0.58] | 0.50 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | truncation 0.03 |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 128 | effort=minimal | 30 | 0.50 [0.33, 0.67] | 0.50 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | truncation 0.03 |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 512 | effort=minimal | 30 | 0.23 [0.12, 0.41] | 0.23 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=minimal | 50 | 0.82 [0.69, 0.90] | — | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=minimal | 50 | 0.44 [0.31, 0.58] | 0.54 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=minimal | 50 | 1.00 [0.93, 1.00] | — | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=minimal | 50 | 0.62 [0.48, 0.74] | 0.74 | — |
| google/gemini-3.5-flash | floor | s5 | 16 | rendering=abstract_stated, effort=minimal | 30 | 0.23 [0.12, 0.41] | 0.63 | truncation 0.03 |
| google/gemini-3.5-flash | recall_load | recall_copy_v1 | 64 | contract, effort=minimal | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v1 | 16 | effort=high | 25 | 0.16 [0.06, 0.35] | 0.16 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v1 | 32 | effort=high | 25 | 0.24 [0.11, 0.43] | 0.28 | truncation 0.12 |
| google/gemini-3.5-flash | s5_chain | s5_chain_v1 | 64 | effort=high | 25 | 0.12 [0.04, 0.30] | 0.28 | truncation 0.24 |
| google/gemini-3.5-flash | s5_chain | s5_chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 0.76 [0.57, 0.89] | 0.76 | truncation 0.12 |
| google/gemini-3.5-flash | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.56 [0.37, 0.73] | 0.56 | truncation 0.04 |
| google/gemini-3.5-flash | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.68 [0.48, 0.83] | 0.68 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.60 [0.41, 0.77] | 0.60 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | truncation 0.08 |
| google/gemini-3.5-flash | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | truncation 0.08 |
| google/gemini-3.5-flash | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.52 [0.33, 0.70] | 0.56 | truncation 0.40 |
| google/gemini-3.5-flash | sanity | conflict_v1 | 4 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | sanity | recall_copy_v1 | 6 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=minimal | 100 | 0.56 [0.46, 0.65] | — | truncation 0.06 |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=minimal | 100 | 0.47 [0.38, 0.57] | 0.48 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | contract, effort=minimal | 100 | 0.45 [0.36, 0.55] | 0.46 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 64 | contract, effort=minimal | 100 | 0.34 [0.25, 0.44] | 0.34 | truncation 0.03 |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=minimal | 100 | 0.66 [0.56, 0.75] | — | truncation 0.03 |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=minimal | 100 | 0.65 [0.55, 0.74] | 0.66 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=minimal | 100 | 1.00 [0.96, 1.00] | — | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | contract, effort=minimal | 100 | 0.64 [0.54, 0.73] | 0.64 | truncation 0.01 |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 64 | contract, effort=minimal | 100 | 0.28 [0.20, 0.37] | 0.28 | — |
| google/gemini-3.6-flash | chain_instant | chain_v2 | 16 | contract, effort=minimal | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| google/gemini-3.6-flash | chain_nowrap | chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.6-flash | chain_nowrap | chain_v2 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.6-flash | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.6-flash | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| google/gemini-3.6-flash | commutative | commutative_v1 | 64 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| google/gemini-3.6-flash | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=minimal | 50 | 0.64 [0.50, 0.76] | — | — |
| google/gemini-3.6-flash | gap_stability | composite_copy_v2 | 32 | contract, effort=minimal | 50 | 0.34 [0.22, 0.48] | 0.44 | — |
| google/gemini-3.6-flash | recall_load | recall_copy_v1 | 64 | contract, effort=minimal | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| google/gemini-3.6-flash | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| google/gemini-3.6-flash | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| google/gemini-3.6-flash | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| google/gemini-3.6-flash | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| google/gemini-3.6-flash | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.6-flash | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.6-flash | sanity | conflict_v1 | 4 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.6-flash | sanity | recall_copy_v1 | 6 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=minimal | 100 | 0.69 [0.59, 0.77] | — | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=minimal | 100 | 0.65 [0.55, 0.74] | 0.68 | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=minimal | 100 | 1.00 [0.96, 1.00] | — | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 16 | contract, effort=minimal | 100 | 0.67 [0.57, 0.75] | 0.69 | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 64 | contract, effort=minimal | 100 | 0.26 [0.18, 0.35] | 0.29 | — |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 4 | effort=default | 30 | 0.20 [0.10, 0.37] | 0.23 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 16 | effort=default | 30 | 0.23 [0.12, 0.41] | 0.73 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 64 | effort=default | 30 | 0.13 [0.05, 0.30] | 0.27 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 128 | effort=default | 30 | 0.00 [0.00, 0.11] | 0.57 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 512 | effort=default | 30 | 0.10 [0.03, 0.26] | 0.20 | — |
| meta-llama/llama-4-maverick | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=default | 50 | 0.96 [0.87, 0.99] | — | — |
| meta-llama/llama-4-maverick | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=default | 50 | 0.18 [0.10, 0.31] | 0.68 | — |
| meta-llama/llama-4-maverick | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=default | 50 | 1.00 [0.93, 1.00] | — | — |
| meta-llama/llama-4-maverick | dose_response | composite_copy_v1 | 16 | effort=default | 50 | 0.16 [0.08, 0.29] | 0.70 | — |
| meta-llama/llama-4-maverick | floor | s5 | 16 | rendering=abstract_stated, effort=default | 30 | 0.20 [0.10, 0.37] | 0.20 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 16 | rendering=concrete, effort=default | 25 | 0.24 [0.11, 0.43] | 0.24 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 32 | rendering=concrete, effort=default | 25 | 0.28 [0.14, 0.48] | 0.28 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 64 | rendering=concrete, effort=default | 25 | 0.08 [0.02, 0.25] | 0.08 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 128 | rendering=concrete, effort=default | 25 | 0.20 [0.09, 0.39] | 0.20 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 256 | rendering=concrete, effort=default | 25 | 0.24 [0.11, 0.43] | 0.24 | — |
| meta-llama/llama-4-maverick | sanity | conflict_v1 | 4 | effort=default | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| meta-llama/llama-4-maverick | sanity | recall_copy_v1 | 6 | effort=default | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.32 [0.17, 0.52] | 0.96 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | truncation 0.04 |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | ‡ cap-escape; truncation 0.08 |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | ‡ cap-escape; truncation 0.16 |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | commutative | commutative_v1 | 64 | effort=high | 50 | 0.66 [0.52, 0.78] | 0.66 | ‡ cap-escape; 2 runs, spread 0.02 |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.40 [0.25, 0.58] | 0.53 | truncation 0.23 |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.97 [0.83, 0.99] | 1.00 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.17 [0.07, 0.34] | 0.27 | truncation 0.13 |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.07 [0.02, 0.21] | 0.10 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.10 [0.03, 0.26] | 0.40 | — |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.78 [0.65, 0.87] | — | truncation 0.24 |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.26 [0.16, 0.40] | 0.96 | truncation 0.14 |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.98 [0.90, 1.00] | 0.98 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.48 [0.35, 0.61] | 0.56 | truncation 0.28 |
| moonshotai/kimi-k2.6 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.27 [0.14, 0.44] | 0.27 | — |
| moonshotai/kimi-k2.6 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 50 | 0.36 [0.24, 0.50] | — | ‡ cap-escape; truncation 0.26 |
| moonshotai/kimi-k2.6 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 50 | 0.56 [0.42, 0.69] | 0.86 | escalated @512tok diagnostic 0.86; canonical = first attempt @96tok; truncation 0.10 |
| moonshotai/kimi-k2.6 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | ‡ cap-escape; truncation 0.04 |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | truncation 0.04 |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 25 | 0.88 [0.70, 0.96] | 0.88 | truncation 0.08 |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.68 [0.48, 0.83] | 0.68 | truncation 0.16 |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | truncation 0.08 |
| moonshotai/kimi-k2.6 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.52 [0.42, 0.62] | — | escalated @512tok diagnostic 0.92; canonical = first attempt @96tok; truncation 0.03 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.48 [0.38, 0.58] | 0.85 | escalated @512tok diagnostic 0.82; canonical = first attempt @96tok; truncation 0.06 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.48 [0.38, 0.58] | 0.86 | escalated @512tok diagnostic 0.83; canonical = first attempt @96tok; truncation 0.09 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.38 [0.29, 0.48] | 0.97 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.94 [0.88, 0.97] | — | ‡ cap-escape; calls failed 0.02 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.83 [0.74, 0.89] | 0.87 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 0.98 [0.93, 0.99] | — | — |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.77 [0.68, 0.84] | 0.84 | ‡ cap-escape; calls failed 0.03; truncation 0.01 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.93 [0.86, 0.97] | 0.95 | ‡ cap-escape; calls failed 0.02 |
| moonshotai/kimi-k3 | chain_instant | chain_v2 | 16 | contract, effort=none | 25 | 0.04 [0.01, 0.20] | 0.04 | — |
| moonshotai/kimi-k3 | chain_nowrap | chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k3 | chain_nowrap | chain_v2 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k3 | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k3 | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k3 | commutative | commutative_v1 | 64 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| moonshotai/kimi-k3 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 50 | 0.42 [0.29, 0.56] | — | — |
| moonshotai/kimi-k3 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 50 | 0.34 [0.22, 0.48] | 0.34 | — |
| moonshotai/kimi-k3 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| moonshotai/kimi-k3 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k3 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k3 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| moonshotai/kimi-k3 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| moonshotai/kimi-k3 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | truncation 0.04 |
| moonshotai/kimi-k3 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.80 [0.61, 0.91] | 0.80 | truncation 0.16 |
| moonshotai/kimi-k3 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k3 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.65 [0.55, 0.74] | — | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.32 [0.24, 0.42] | 0.32 | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 1.00 [0.96, 1.00] | — | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.33 [0.25, 0.43] | 0.33 | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.29 [0.21, 0.39] | 0.29 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.92 | — |
| muse-spark-1.1 | chain_nowrap | chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | chain_nowrap | chain_v2 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | commutative | commutative_v1 | 64 | effort=high | 25 | 0.28 [0.14, 0.48] | 0.40 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.96 [0.80, 0.99] | 1.00 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v4 | 32 | sysprompt=04153d7439, effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v4 | 128 | sysprompt=04153d7439, effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| muse-spark-1.1 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_instant | chain_v2 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 0.44 [0.27, 0.63] | 0.44 | truncation 0.12 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.04 [0.01, 0.20] | 0.04 | truncation 0.12 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.00 [0.00, 0.13] | 0.00 | truncation 0.80 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.00 [0.00, 0.13] | 0.00 | truncation 0.64 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 0.48 [0.30, 0.67] | 0.48 | truncation 0.52 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 0.60 [0.41, 0.77] | 0.60 | truncation 0.32 |
| nvidia/nemotron-3-ultra-550b-a55b | commutative | commutative_v1 | 64 | effort=high | 25 | 0.44 [0.27, 0.63] | 0.44 | truncation 0.16 |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.70 [0.52, 0.83] | 0.70 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.20 [0.10, 0.37] | 0.20 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.57 [0.39, 0.73] | 0.57 | truncation 0.07 |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.20 [0.10, 0.37] | 0.20 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.17 [0.07, 0.34] | 0.17 | truncation 0.03 |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.13 [0.05, 0.30] | 0.13 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.23 [0.12, 0.41] | 0.23 | truncation 0.20 |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.03 [0.01, 0.17] | 0.03 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.60 [0.46, 0.72] | — | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.22 [0.13, 0.35] | 0.22 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.84 [0.71, 0.92] | 0.86 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 0.50 [0.37, 0.63] | 0.54 | truncation 0.02 |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 0.62 [0.48, 0.74] | 0.64 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.16 [0.08, 0.29] | 0.16 | — |
| nvidia/nemotron-3-ultra-550b-a55b | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.30 [0.17, 0.48] | 0.30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | truncation 0.08 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | truncation 0.24 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.80 [0.61, 0.91] | 0.80 | truncation 0.16 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.84 [0.65, 0.94] | 0.84 | truncation 0.08 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 0.44 [0.27, 0.63] | 0.44 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 0.56 [0.37, 0.73] | 0.56 | truncation 0.04 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 0.36 [0.20, 0.55] | 0.36 | truncation 0.40 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | truncation 0.20 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.80 [0.61, 0.91] | 0.80 | truncation 0.20 |
| nvidia/nemotron-3-ultra-550b-a55b | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.55 [0.45, 0.64] | — | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.35 [0.26, 0.45] | 0.35 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.36 [0.27, 0.46] | 0.36 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.20 [0.13, 0.29] | 0.20 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.49 [0.39, 0.59] | — | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.30 [0.22, 0.40] | 0.30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 0.99 [0.95, 1.00] | — | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.33 [0.25, 0.43] | 0.33 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.12 [0.07, 0.20] | 0.12 | — |
| openai/gpt-5.4 | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.70 [0.52, 0.83] | 0.70 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.33 [0.19, 0.51] | 0.33 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.27 [0.14, 0.44] | 0.27 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.97 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.30 [0.17, 0.48] | 0.30 | — |
| openai/gpt-5.4 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.86 [0.74, 0.93] | — | — |
| openai/gpt-5.4 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.30 [0.19, 0.44] | 0.30 | — |
| openai/gpt-5.4 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| openai/gpt-5.4 | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.96 [0.87, 0.99] | 0.98 | — |
| openai/gpt-5.4 | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.64 [0.50, 0.76] | 0.66 | — |
| openai/gpt-5.4 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.13 [0.05, 0.30] | 0.13 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| openai/gpt-5.4 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.4 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.5 | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.5 | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.08 [0.02, 0.25] | 0.08 | — |
| openai/gpt-5.5 | chain_instant | chain_v2 | 16 | contract, effort=none | 25 | 0.08 [0.02, 0.25] | 0.08 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.84 [0.65, 0.94] | 0.84 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.36 [0.20, 0.55] | 0.36 | truncation 0.04 |
| openai/gpt-5.5 | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | commutative | commutative_v1 | 64 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.67 [0.49, 0.81] | 0.67 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.33 [0.19, 0.51] | 0.33 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.43 [0.27, 0.61] | 0.43 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.37 [0.22, 0.54] | 0.37 | — |
| openai/gpt-5.5 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.96 [0.87, 0.99] | — | — |
| openai/gpt-5.5 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.74 [0.60, 0.84] | 0.74 | — |
| openai/gpt-5.5 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| openai/gpt-5.5 | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| openai/gpt-5.5 | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.70 [0.56, 0.81] | 0.70 | — |
| openai/gpt-5.5 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.27 [0.14, 0.44] | 0.27 | — |
| openai/gpt-5.5 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 50 | 0.68 [0.54, 0.79] | — | — |
| openai/gpt-5.5 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 50 | 0.32 [0.21, 0.46] | 0.32 | — |
| openai/gpt-5.5 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v1 | 16 | effort=high | 25 | 0.12 [0.04, 0.30] | 0.12 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v1 | 32 | effort=high | 25 | 0.16 [0.06, 0.35] | 0.16 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v1 | 64 | effort=high | 25 | 0.12 [0.04, 0.30] | 0.12 | truncation 0.04 |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.5 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.92 [0.85, 0.96] | — | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.63 [0.53, 0.72] | 0.63 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.58 [0.48, 0.67] | 0.58 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.59 [0.49, 0.68] | 0.59 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.80 [0.71, 0.87] | — | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.46 [0.37, 0.56] | 0.46 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 1.00 [0.96, 1.00] | — | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.46 [0.37, 0.56] | 0.46 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.33 [0.25, 0.43] | 0.33 | — |
| openai/gpt-5.6-sol | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| openai/gpt-5.6-sol | chain_instant | chain_v2 | 16 | contract, effort=none | 25 | 0.76 [0.57, 0.89] | 1.00 | escalated @512tok diagnostic 1.00; canonical = first attempt @96tok |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.76 [0.57, 0.89] | 0.76 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v2 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| openai/gpt-5.6-sol | commutative | commutative_v1 | 64 | effort=high | 25 | 0.76 [0.57, 0.89] | 0.76 | — |
| openai/gpt-5.6-sol | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 50 | 0.58 [0.44, 0.71] | — | — |
| openai/gpt-5.6-sol | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 50 | 0.26 [0.16, 0.40] | 0.26 | — |
| openai/gpt-5.6-sol | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 0.16 [0.06, 0.35] | 0.16 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 25 | 0.20 [0.09, 0.39] | 0.20 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.24 [0.11, 0.43] | 0.24 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 25 | 0.44 [0.27, 0.63] | 0.44 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.56 [0.37, 0.73] | 0.56 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 25 | 0.52 [0.33, 0.70] | 0.52 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.32 [0.17, 0.52] | 0.32 | ᵘ unworked answers on a large fraction of calls; 3 runs, spread 0.32 |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.60 [0.41, 0.77] | 0.60 | ᵘ unworked answers on a large fraction of calls; 3 runs, spread 0.24 |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.60 [0.41, 0.77] | 0.60 | ᵘ unworked answers on a large fraction of calls; 3 runs, spread 0.24 |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.80 [0.61, 0.91] | 0.80 | ᵘ unworked answers on a large fraction of calls; 3 runs, spread 0.12 |
| openai/gpt-5.6-sol | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| openai/gpt-5.6-sol | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.82 [0.73, 0.88] | — | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.60 [0.50, 0.69] | 0.60 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 1.00 [0.96, 1.00] | — | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.65 [0.55, 0.74] | 0.65 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.33 [0.25, 0.43] | 0.33 | — |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| qwen/qwen3.7-max | chain_instant | chain_v2 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| qwen/qwen3.7-max | commutative | commutative_v1 | 64 | effort=high | 25 | 0.80 [0.61, 0.91] | 0.80 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.93 [0.79, 0.98] | 1.00 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.63 [0.46, 0.78] | 0.63 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.97 [0.83, 0.99] | 1.00 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.13 [0.05, 0.30] | 0.13 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.87 [0.70, 0.95] | 0.97 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.13 [0.05, 0.30] | 0.13 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.07 [0.02, 0.21] | 0.07 | — |
| qwen/qwen3.7-max | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.60 [0.46, 0.72] | — | — |
| qwen/qwen3.7-max | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.18 [0.10, 0.31] | 0.18 | — |
| qwen/qwen3.7-max | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.92 [0.81, 0.97] | 1.00 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 0.94 [0.84, 0.98] | 0.96 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 0.96 [0.87, 0.99] | 0.96 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.64 [0.50, 0.76] | 0.64 | — |
| qwen/qwen3.7-max | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.27 [0.14, 0.44] | 0.27 | — |
| qwen/qwen3.7-max | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v1 | 16 | effort=high | 25 | 0.12 [0.04, 0.30] | 0.12 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v1 | 32 | effort=high | 25 | 0.24 [0.11, 0.43] | 0.24 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v1 | 64 | effort=high | 25 | 0.16 [0.06, 0.35] | 0.16 | ‡ cap-escape |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 0.84 [0.65, 0.94] | 0.84 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 25 | 0.72 [0.52, 0.86] | 0.72 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 25 | 0.64 [0.45, 0.80] | 0.64 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.64 [0.45, 0.80] | 0.64 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.72 [0.52, 0.86] | 0.72 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.44 [0.27, 0.63] | 0.44 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.80 [0.61, 0.91] | 0.80 | — |
| qwen/qwen3.7-max | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.53 [0.43, 0.62] | — | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.24 [0.17, 0.33] | 0.24 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.25 [0.18, 0.34] | 0.25 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.12 [0.07, 0.20] | 0.12 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.51 [0.41, 0.61] | — | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.25 [0.18, 0.34] | 0.25 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 0.02 [0.01, 0.07] | — | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.24 [0.17, 0.33] | 0.24 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.08 [0.04, 0.15] | 0.08 | — |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.20 [0.10, 0.37] | 0.83 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.10 [0.03, 0.26] | 0.23 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.37 [0.22, 0.54] | 0.80 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.10 [0.03, 0.26] | 0.10 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.60 [0.42, 0.75] | 0.67 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.03 [0.01, 0.17] | 0.07 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.73 [0.56, 0.86] | 0.73 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.03 [0.01, 0.17] | 0.13 | — |
| x-ai/grok-4.3 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.16 [0.08, 0.29] | — | — |
| x-ai/grok-4.3 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.18 [0.10, 0.31] | 0.20 | — |
| x-ai/grok-4.3 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.22 [0.13, 0.35] | 0.86 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 0.22 [0.13, 0.35] | 0.42 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 0.30 [0.19, 0.44] | 0.66 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.14 [0.07, 0.26] | 0.24 | — |
| x-ai/grok-4.3 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.27 [0.14, 0.44] | 0.27 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | — |
| x-ai/grok-4.3 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| x-ai/grok-4.3 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | ‡ cap-escape |
| x-ai/grok-4.5 | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.5 | commutative | commutative_v1 | 64 | effort=high | 25 | 0.72 [0.52, 0.86] | 0.72 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 0.84 [0.65, 0.94] | 0.84 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| x-ai/grok-4.5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.72 [0.52, 0.86] | 0.72 | ‡ cap-escape; truncation 0.04 |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.60 [0.41, 0.77] | 0.60 | ‡ cap-escape; truncation 0.04 |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.00 [0.00, 0.13] | 0.00 | ‡ cap-escape; truncation 0.68 |
| x-ai/grok-build-0.1 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| x-ai/grok-build-0.1 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | ‡ cap-escape |
| x-ai/grok-build-0.1 | sanity | conflict_v1 | 4 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| x-ai/grok-build-0.1 | sanity | recall_copy_v1 | 6 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | truncation 0.04 |
| z-ai/glm-5.2 | chain_instant | chain_v2 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | truncation 0.04 |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.28 [0.14, 0.48] | 0.28 | truncation 0.04 |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.48 [0.30, 0.67] | 0.48 | truncation 0.08 |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.36 [0.20, 0.55] | 0.36 | truncation 0.20 |
| z-ai/glm-5.2 | chain_nowrap | chain_v2 | 64 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | calls failed 0.04 |
| z-ai/glm-5.2 | chain_nowrap | chain_v2 | 128 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | calls failed 0.04; truncation 0.04 |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.97 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.67 [0.49, 0.81] | 0.67 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.37 [0.22, 0.54] | 0.37 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.13 [0.05, 0.30] | 0.13 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | truncation 0.03 |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.17 [0.07, 0.34] | 0.17 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.56 [0.42, 0.69] | — | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.20 [0.11, 0.33] | 0.28 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.94 [0.84, 0.98] | 0.96 | truncation 0.02 |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 0.94 [0.84, 0.98] | 0.96 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.68 [0.54, 0.79] | 0.68 | — |
| z-ai/glm-5.2 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.23 [0.12, 0.41] | 0.23 | — |
| z-ai/glm-5.2 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 32 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | truncation 0.04 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 25 | 0.84 [0.65, 0.94] | 0.84 | truncation 0.12 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 64 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | truncation 0.04 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 96 | effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | truncation 0.04 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 25 | 0.72 [0.52, 0.86] | 0.72 | truncation 0.20 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| z-ai/glm-5.2 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | truncation 0.08 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| z-ai/glm-5.2 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 25 | 0.80 [0.61, 0.91] | 0.80 | truncation 0.04 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v4 | 32 | sysprompt=04153d7439, effort=xhigh | 25 | ⊘ calls failed | — | calls failed 1.00 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v4 | 128 | sysprompt=04153d7439, effort=xhigh | 25 | ⊘ calls failed | — | calls failed 1.00 |
| z-ai/glm-5.2 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.84 [0.65, 0.94] | 0.84 | truncation 0.04; 2 runs, spread 0.16 |
| z-ai/glm-5.2 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.76 [0.57, 0.89] | 0.76 | truncation 0.04 |
| z-ai/glm-5.2 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.64 [0.54, 0.73] | — | truncation 0.01 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.35 [0.26, 0.45] | 0.36 | truncation 0.03 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.31 [0.23, 0.41] | 0.31 | truncation 0.04 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.15 [0.09, 0.23] | 0.17 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.71 [0.61, 0.79] | — | 3 runs, spread 0.03 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.37 [0.28, 0.47] | 0.39 | 3 runs, spread 0.03 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 1.00 [0.96, 1.00] | — | 3 runs, spread 0.00 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.38 [0.29, 0.48] | 0.42 | truncation 0.03; 3 runs, spread 0.04 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.13 [0.08, 0.21] | 0.14 | truncation 0.02; 3 runs, spread 0.03 |

## Diagnostics per cell

finish_errors counts per-example finish=='error' calls (surfaced even where diagnostics.api_errors is 0). ctok = completion tokens; rtok = reasoning tokens. The worked-calls, event-blind and truncation columns are the per-cell diagnostics described under the headline tables; instant cells render — for worked calls, where a short completion is the answer contract rather than a model declining to work.

worked calls: the fraction of a cell's calls whose completion exceeds 512 completion tokens, with (match | worked / match | unworked) beside it where the cell has calls of both kinds. The signal is completion tokens, not reasoning tokens: reasoning-token accounting is not comparable across providers (on this history the median ctok-minus-rtok is 5184 for sonnet-5, 4184 for opus and 1675 for fable, but 2-17 for every other model, and glm reports rtok=0 on correct 8k-token answers). Cells run before per-example token logging report n/a.

The work rate is a property of the model AND of the system prompt the benchmark sends, not of the model alone. Three arms over the identical deterministic items (openai/gpt-5.6-sol, s5_chain_v3 @L64, n=25, top effort, 49,152-token budget) differ only in that prompt. Under the scored protocol prompt ("You are taking a short test... no explanation") the model matches 0.68, works 0.68 of its calls and answers event-blind on 0.33; with the two clauses that read as instructions to spend less effort removed and the identical answer-format contract kept, it matches 0.96, works 0.96 and answers event-blind on 0.04 — 7 of the 8 calls the scored prompt left unworked are worked under the neutral one. The third arm sends no system prompt at all: it works 0.96 and matches 0.84, and that 0.84 is a format reading rather than a composition one — all 24 of its worked calls carry the gold value, three of them committed in LaTeX (`**Answer: \(g15\)**`, `\boxed{g0}`) that the committed-answer rule does not read. The event-blind rates run through the published column's eligibility rule, so they read against it. Every scored thinking cell carries the scored prompt, so the completion-token columns are token spend under an instruction to be brief. One model, one length, n=25: `results/probes/sol_system_prompt_20260727.json`.

event-blind: the fraction of a cell's predictions equal to the 8-hop dereference of the INITIAL pointer map — the answer a model gives if it reads the fact block and skips the whole event stream. Items where that answer coincides with the gold answer are dropped (it coincides at chance, 0.063-0.078 across lengths against 1/16), so the rate names which cheaper task the model substituted rather than crediting luck. Across the roster's 52 scored s5_chain cells the blind answer is given on 46 of 1,261 eligible items, 42 of them by openai/gpt-5.6-sol; over the other 12 models the rate is 4 of 1,164 (0.003), against 0.063 for a uniform guess.

| Model | Facet | Task | Length | Arm | empty_rate | api_errors | finish_errors | reasoning_tokens | worked calls | event-blind | truncation | finish_reasons | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-fable-5 | chain_nowrap | chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 1083 | 0.00 (—/1.00) | — | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | chain_nowrap | chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 1064 | 0.00 (—/0.92) | — | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 2068 | 0.64 (1.00/1.00) | — | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 2583 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 1584 | 0.04 (0.00/0.88) | — | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 7260 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 13561 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 20353 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 27879 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 40257 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-fable-5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 89512 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 1382 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| anthropic/claude-opus-4.8 | chain_instant | chain_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 1244 | 0.00 (—/1.00) | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 1703 | 0.36 (1.00/0.94) | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 2146 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 128 | effort=high | 0.040 | 0 | 0 | 14665 | 1.00 | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 1101 | 0.00 (—/1.00) | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 1541 | 0.64 (0.94/1.00) | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 1962 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 3496 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 2395 | 0.08 (0.50/0.83) | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 1319 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 2557 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 3352 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 2761 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-opus-4.8 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-opus-4.8 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-opus-4.8 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 2448 | n/a | — | 0.00 | stop:50 | — |
| anthropic/claude-opus-4.8 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-opus-4.8 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-opus-4.8 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-opus-4.8 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 8440 | 0.80 (0.10/0.00) | 0.21 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 14316 | 0.92 (0.17/0.50) | 0.04 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 23482 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 9145 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 12390 | 1.00 | 0.04 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 0.000 | 0 | 0 | 17299 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 21063 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 0.000 | 0 | 0 | 33314 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 96 | effort=high | 0.000 | 0 | 0 | 31595 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 0.000 | 0 | 0 | 44957 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 15025 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 25693 | 1.00 | 0.04 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 43323 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 58382 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 7578 | n/a | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 13440 | n/a | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 28491 | n/a | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 48411 | n/a | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 141239 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-opus-4.8 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 1303 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | chain_instant | chain_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | escalated @512tok diagnostic 1.00; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 1351 | 0.00 (—/1.00) | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 1564 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 64 | effort=high | 0.040 | 0 | 0 | 4323 | 1.00 | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 128 | effort=high | 0.280 | 0 | 0 | 9958 | 1.00 | — | 0.28 | length:7, stop:18 | truncation 0.28 |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 1169 | 0.00 (—/1.00) | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 1590 | 0.92 (1.00/1.00) | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 2616 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 4345 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 3370 | 0.06 (0.33/0.66) | — | 0.00 | stop:50 | 2 runs, spread 0.04 |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 1534 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 2595 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 2809 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 5228 | n/a | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-sonnet-5 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-sonnet-5 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-sonnet-5 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 2502 | n/a | — | 0.00 | stop:50 | — |
| anthropic/claude-sonnet-5 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-sonnet-5 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.03 | length:1, stop:29 | truncation 0.03 |
| anthropic/claude-sonnet-5 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | escalated @512tok diagnostic 0.72; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | escalated @512tok diagnostic 0.64; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 15910 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 0.040 | 0 | 0 | 19276 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | truncation 0.04 |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 28911 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 0.000 | 0 | 0 | 36651 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 96 | effort=high | 0.040 | 0 | 0 | 45415 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | truncation 0.04 |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 0.000 | 0 | 0 | 50871 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 32 | effort=high | 0.000 | 0 | 0 | 12688 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 17365 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 64 | effort=high | 0.000 | 0 | 0 | 29502 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 33324 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 96 | effort=high | 0.000 | 0 | 0 | 51483 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 45586 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 68590 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 6510 | n/a | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 13050 | n/a | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.040 | 0 | 1 | 28952 | n/a | — | 0.00 | error:1, stop:24 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 53155 | n/a | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 164247 | 1.00 | — | 0.00 | stop:25 | — |
| anthropic/claude-sonnet-5 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.020 | 0 | 0 | 0 | — | — | 0.03 | length:3, stop:97 | truncation 0.03 |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | escalated @512tok diagnostic 0.75; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | escalated @512tok diagnostic 0.77; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | escalated @512tok diagnostic 0.67; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | escalated @512tok diagnostic 0.82; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | escalated @512tok diagnostic 0.76; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | escalated @512tok diagnostic 0.66; canonical = first attempt @96tok |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 7441 | n/a | — | 0.00 | stop:30 | — |
| deepseek/deepseek-v4-pro | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | chain_instant | chain_v2 | 16 | contract, effort=none | 0.040 | 0 | 0 | 1362 | — | — | 0.04 | length:1, stop:24 | escalated @512tok diagnostic 0.04; canonical = first attempt @96tok; truncation 0.04 |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 17023 | 0.72 (1.00/1.00) | — | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 62570 | 1.00 | — | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 64 | effort=high | 0.040 | 0 | 0 | 151615 | 1.00 | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 128 | effort=high | 0.760 | 0 | 0 | 698721 | 1.00 | — | 0.76 | length:19, stop:5 | truncation 0.76 |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 109235 | 0.92 (1.00/0.00) | — | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 224013 | 1.00 | — | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 16 | effort=high | 0.033 | 0 | 0 | 26943 | n/a | — | 0.03 | length:1, stop:29 | truncation 0.03 |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 44114 | n/a | — | 0.00 | stop:30 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 128 | effort=high | 0.033 | 0 | 0 | 53840 | n/a | — | 0.03 | length:1, stop:29 | truncation 0.03 |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 512 | effort=high | 0.100 | 0 | 0 | 83206 | n/a | — | 0.10 | length:3, stop:27 | truncation 0.10 |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 28490 | n/a | — | 0.00 | stop:50 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=low | 0.020 | 0 | 0 | 58200 | n/a | — | 0.02 | length:1, stop:49 | truncation 0.02 |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 33409 | n/a | — | 0.00 | stop:50 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| deepseek/deepseek-v4-pro | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| deepseek/deepseek-v4-pro | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 215630 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 0.160 | 0 | 0 | 306282 | 1.00 | 0.00 | 0.16 | length:4, stop:21 | truncation 0.16 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 64 | effort=high | 0.040 | 0 | 0 | 306283 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | truncation 0.04 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 0.280 | 0 | 0 | 500335 | 1.00 | 0.00 | 0.28 | length:7, stop:18 | truncation 0.28 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 96 | effort=high | 0.000 | 0 | 0 | 377966 | 1.00 | 0.04 | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 0.160 | 0 | 0 | 638655 | 1.00 | 0.00 | 0.16 | length:4, stop:21 | truncation 0.16 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 281020 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 426214 | 0.96 (0.96/0.00) | 0.00 | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 671848 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 793155 | 0.96 (1.00/0.00) | 0.00 | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v4 | 32 | sysprompt=04153d7439, effort=xhigh | 0.080 | 0 | 0 | 573592 | 1.00 | 0.00 | 0.08 | length:2, stop:23 | truncation 0.08 |
| deepseek/deepseek-v4-pro | s5_chain | s5_chain_v4 | 128 | sysprompt=04153d7439, effort=xhigh | 0.360 | 9 | 0 | 807383 | 1.00 | 0.00 | 0.00 | stop:16 | calls failed 0.36 |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 35520 | n/a | — | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 65442 | n/a | — | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 127456 | n/a | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 244509 | n/a | — | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 498748 | 1.00 | — | 0.00 | stop:25 | — |
| deepseek/deepseek-v4-pro | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| deepseek/deepseek-v4-pro | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 9852 | n/a | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 41148 | n/a | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 16 | effort=minimal | 0.000 | 0 | 0 | 13820 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 100821 | n/a | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 64 | effort=minimal | 0.000 | 0 | 0 | 23649 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 122323 | n/a | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 128 | effort=minimal | 0.000 | 0 | 0 | 19528 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 111633 | n/a | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 512 | effort=minimal | 0.000 | 0 | 0 | 27161 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=minimal | 0.000 | 0 | 0 | 7894 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.1-pro-preview | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=minimal | 0.020 | 0 | 1 | 12102 | — | — | 0.00 | error:1, stop:49 | — |
| google/gemini-3.1-pro-preview | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=minimal | 0.000 | 0 | 0 | 6138 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.1-pro-preview | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 61752 | n/a | — | 0.00 | stop:50 | — |
| google/gemini-3.1-pro-preview | dose_response | composite_copy_v1 | 16 | effort=minimal | 0.000 | 0 | 0 | 22069 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.1-pro-preview | floor | s5 | 16 | rendering=abstract_stated, effort=minimal | 0.000 | 0 | 0 | 29524 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 57602 | n/a | — | 0.00 | stop:25 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 103871 | n/a | — | 0.00 | stop:25 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 164515 | n/a | — | 0.00 | stop:25 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 344753 | n/a | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 384750 | n/a | — | 0.00 | stop:25 | — |
| google/gemini-3.1-pro-preview | sanity | conflict_v1 | 4 | effort=minimal | 0.000 | 0 | 0 | 6169 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.1-pro-preview | sanity | recall_copy_v1 | 6 | effort=minimal | 0.000 | 0 | 0 | 6405 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 15205 | n/a | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | chain_instant | chain_v1 | 16 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | escalated @512tok diagnostic 1.00; canonical = first attempt @96tok |
| google/gemini-3.5-flash | chain_instant | chain_v2 | 16 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 39265 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 68234 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 140042 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 128 | effort=high | 0.000 | 0 | 0 | 272521 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 103093 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 264592 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 89076 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 30421 | n/a | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 16 | effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 52078 | n/a | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 64 | effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 74015 | n/a | — | 0.03 | length:1, stop:29 | truncation 0.03 |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 128 | effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 100637 | n/a | — | 0.03 | length:1, stop:29 | truncation 0.03 |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 512 | effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 47756 | n/a | — | 0.00 | stop:50 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 17044 | n/a | — | 0.00 | stop:50 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 35154 | n/a | — | 0.00 | stop:50 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.5-flash | floor | s5 | 16 | rendering=abstract_stated, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.03 | length:1, stop:29 | truncation 0.03 |
| google/gemini-3.5-flash | recall_load | recall_copy_v1 | 64 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 205901 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 302670 | 1.00 | 0.00 | 0.12 | length:3, stop:22 | truncation 0.12 |
| google/gemini-3.5-flash | s5_chain | s5_chain_v1 | 64 | effort=high | 0.040 | 0 | 0 | 367384 | 1.00 | 0.05 | 0.24 | length:6, stop:19 | truncation 0.24 |
| google/gemini-3.5-flash | s5_chain | s5_chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 177192 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 283115 | 1.00 | 0.04 | 0.12 | length:3, stop:22 | truncation 0.12 |
| google/gemini-3.5-flash | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 396885 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v2 | 96 | effort=high | 0.120 | 0 | 1 | 536851 | 0.96 (0.58/0.00) | 0.00 | 0.04 | error:1, length:1, stop:23 | truncation 0.04 |
| google/gemini-3.5-flash | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 261298 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 484090 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 671830 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 710496 | 1.00 | 0.04 | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 56548 | n/a | — | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 102124 | n/a | — | 0.00 | stop:25 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 141222 | n/a | — | 0.08 | length:2, stop:23 | truncation 0.08 |
| google/gemini-3.5-flash | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 274191 | n/a | — | 0.08 | length:2, stop:23 | truncation 0.08 |
| google/gemini-3.5-flash | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 388039 | n/a | — | 0.40 | length:10, stop:15 | truncation 0.40 |
| google/gemini-3.5-flash | sanity | conflict_v1 | 4 | effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | sanity | recall_copy_v1 | 6 | effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=minimal | 0.040 | 0 | 0 | 0 | — | — | 0.06 | length:6, stop:94 | truncation 0.06 |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 64 | contract, effort=minimal | 0.030 | 0 | 0 | 0 | — | — | 0.03 | length:3, stop:97 | truncation 0.03 |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=minimal | 0.030 | 0 | 0 | 0 | — | — | 0.03 | length:3, stop:97 | truncation 0.03 |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.01 | length:1, stop:99 | truncation 0.01 |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 64 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.6-flash | chain_instant | chain_v2 | 16 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | chain_nowrap | chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 32141 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | chain_nowrap | chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 57469 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 108723 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 146582 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 70393 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.6-flash | gap_stability | composite_copy_v2 | 32 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.6-flash | recall_load | recall_copy_v1 | 64 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| google/gemini-3.6-flash | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 166755 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 204101 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 340141 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 353569 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 205791 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 377144 | 1.00 | — | 0.00 | stop:25 | — |
| google/gemini-3.6-flash | sanity | conflict_v1 | 4 | effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.6-flash | sanity | recall_copy_v1 | 6 | effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 16 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| google/gemini-3.6-flash | zero_budget | composite_copy_v2 | 64 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 4 | effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:30 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 16 | effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:30 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 64 | effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:30 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 128 | effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:30 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 512 | effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:30 | — |
| meta-llama/llama-4-maverick | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:50 | — |
| meta-llama/llama-4-maverick | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:50 | — |
| meta-llama/llama-4-maverick | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:50 | — |
| meta-llama/llama-4-maverick | dose_response | composite_copy_v1 | 16 | effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:50 | — |
| meta-llama/llama-4-maverick | floor | s5 | 16 | rendering=abstract_stated, effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:30 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 16 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:25 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 32 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:25 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 64 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:25 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 128 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:25 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 256 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | n/a | — | 0.00 | stop:25 | — |
| meta-llama/llama-4-maverick | sanity | conflict_v1 | 4 | effort=default | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| meta-llama/llama-4-maverick | sanity | recall_copy_v1 | 6 | effort=default | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 9535 | n/a | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 15 | — | — | 0.00 | stop:25 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 54422 | 1.00 | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 32 | effort=high | 0.040 | 0 | 0 | 213892 | 1.00 | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 64 | effort=high | 0.080 | 0 | 0 | 334007 | 1.00 | — | 0.08 | length:2, stop:23 | ‡ cap-escape; truncation 0.08 |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 128 | effort=high | 0.160 | 0 | 1 | 924878 | 1.00 | — | 0.16 | error:1, length:4, stop:20 | ‡ cap-escape; truncation 0.16 |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 171828 | 1.00 | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 336646 | 1.00 | — | 0.00 | stop:25 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 258261 | 1.00 | — | 0.00 | stop:50 | ‡ cap-escape; 2 runs, spread 0.02 |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 69598 | n/a | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 1 | — | — | 0.23 | length:7, stop:23 | truncation 0.23 |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 88013 | n/a | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 2 | — | — | 0.13 | length:4, stop:26 | truncation 0.13 |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 102138 | n/a | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 2 | — | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 141034 | n/a | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 4 | — | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 15 | — | — | 0.24 | length:12, stop:38 | truncation 0.24 |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 21 | — | — | 0.14 | length:7, stop:43 | truncation 0.14 |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 23 | — | — | 0.00 | stop:50 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 114566 | n/a | — | 0.00 | stop:50 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 69452 | n/a | — | 0.00 | stop:50 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 81842 | n/a | — | 0.00 | stop:50 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 5 | — | — | 0.28 | length:14, stop:36 | truncation 0.28 |
| moonshotai/kimi-k2.6 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 13 | — | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.620 | 0 | 0 | 15 | — | — | 0.26 | length:13, stop:19 | ‡ cap-escape; truncation 0.26 |
| moonshotai/kimi-k2.6 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.100 | 0 | 0 | 35 | — | — | 0.10 | length:5, stop:45 | escalated @512tok diagnostic 0.86; canonical = first attempt @96tok; truncation 0.10 |
| moonshotai/kimi-k2.6 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 19 | — | — | 0.00 | stop:50 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 304981 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 0.000 | 0 | 0 | 301855 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 521123 | 1.00 | 0.00 | 0.00 | stop:25 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 0.040 | 0 | 0 | 507832 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | ‡ cap-escape; truncation 0.04 |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 96 | effort=high | 0.040 | 0 | 0 | 612645 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | truncation 0.04 |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 0.080 | 0 | 0 | 626782 | 1.00 | 0.00 | 0.08 | length:2, stop:23 | truncation 0.08 |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 284971 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 467631 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 600097 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.280 | 0 | 1 | 647838 | 1.00 | 0.00 | 0.16 | error:1, length:4, stop:18 | truncation 0.16 |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 97965 | n/a | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 171577 | n/a | — | 0.00 | stop:24 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 247166 | n/a | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 421554 | n/a | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.080 | 0 | 0 | 636870 | n/a | — | 0.08 | length:2, stop:23 | truncation 0.08 |
| moonshotai/kimi-k2.6 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 12 | — | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 13 | — | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.030 | 0 | 0 | 78 | — | — | 0.03 | length:3, stop:97 | escalated @512tok diagnostic 0.92; canonical = first attempt @96tok; truncation 0.03 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.040 | 0 | 0 | 52 | — | — | 0.06 | length:6, stop:94 | escalated @512tok diagnostic 0.82; canonical = first attempt @96tok; truncation 0.06 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.080 | 0 | 0 | 50 | — | — | 0.09 | length:9, stop:91 | escalated @512tok diagnostic 0.83; canonical = first attempt @96tok; truncation 0.09 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 58 | — | — | 0.00 | stop:100 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.020 | 0 | 0 | 89 | — | — | 0.00 | stop:98 | ‡ cap-escape; calls failed 0.02 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.010 | 0 | 0 | 80 | — | — | 0.00 | stop:100 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.020 | 0 | 0 | 40 | — | — | 0.00 | stop:100 | — |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.080 | 0 | 0 | 65 | — | — | 0.01 | length:1, stop:92 | ‡ cap-escape; calls failed 0.03; truncation 0.01 |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.030 | 0 | 0 | 81 | — | — | 0.00 | stop:98 | ‡ cap-escape; calls failed 0.02 |
| moonshotai/kimi-k3 | chain_instant | chain_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | chain_nowrap | chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 12020 | 0.28 (1.00/1.00) | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | chain_nowrap | chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 29367 | 1.00 | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 58383 | 1.00 | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 116412 | 1.00 | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 19411 | 0.72 (0.89/0.86) | — | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| moonshotai/kimi-k3 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| moonshotai/kimi-k3 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| moonshotai/kimi-k3 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 181476 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 273112 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 403973 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 474233 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| moonshotai/kimi-k3 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 283452 | 1.00 | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| moonshotai/kimi-k3 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.160 | 0 | 0 | 510303 | 1.00 | — | 0.16 | length:4, stop:21 | truncation 0.16 |
| moonshotai/kimi-k3 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k3 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| moonshotai/kimi-k3 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 36177 | 1.00 | — | 0.00 | stop:25 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 74957 | 1.00 | — | 0.00 | stop:25 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 191568 | 1.00 | — | 0.00 | stop:25 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 128 | effort=high | 0.080 | 0 | 0 | 449036 | 1.00 | — | 0.00 | incomplete:2, stop:23 | — |
| muse-spark-1.1 | chain_nowrap | chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 33394 | 1.00 | — | 0.00 | stop:25 | — |
| muse-spark-1.1 | chain_nowrap | chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 58622 | 1.00 | — | 0.00 | stop:25 | — |
| muse-spark-1.1 | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 98660 | 1.00 | — | 0.00 | stop:25 | — |
| muse-spark-1.1 | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 188054 | 1.00 | — | 0.00 | stop:25 | — |
| muse-spark-1.1 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 114743 | 1.00 | — | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 224007 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 0.000 | 0 | 0 | 243349 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 326873 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 0.000 | 0 | 0 | 364311 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 96 | effort=high | 0.000 | 0 | 0 | 387842 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 0.000 | 0 | 0 | 497558 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 196646 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 310963 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 448706 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 560469 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v4 | 32 | sysprompt=04153d7439, effort=xhigh | 0.000 | 0 | 0 | 258234 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_chain | s5_chain_v4 | 128 | sysprompt=04153d7439, effort=xhigh | 0.000 | 0 | 0 | 732690 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 242298 | 1.00 | — | 0.00 | stop:25 | — |
| muse-spark-1.1 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 467720 | 1.00 | — | 0.00 | stop:25 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 5435 | n/a | — | 0.00 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_instant | chain_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 16 | effort=high | 0.160 | 0 | 0 | 83841 | 1.00 | — | 0.12 | length:3, stop:22 | truncation 0.12 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 32 | effort=high | 0.120 | 0 | 0 | 162217 | 1.00 | — | 0.12 | length:3, stop:22 | truncation 0.12 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 64 | effort=high | 0.840 | 0 | 0 | 327413 | 1.00 | — | 0.80 | length:20, stop:5 | truncation 0.80 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 128 | effort=high | 0.720 | 0 | 2 | 504794 | 1.00 | — | 0.64 | error:2, length:16, stop:7 | truncation 0.64 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v2 | 64 | effort=high | 0.520 | 0 | 0 | 268946 | 1.00 | — | 0.52 | length:13, stop:12 | truncation 0.52 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v2 | 128 | effort=high | 0.320 | 0 | 0 | 268689 | 0.92 (0.65/0.00) | — | 0.32 | length:8, stop:17 | truncation 0.32 |
| nvidia/nemotron-3-ultra-550b-a55b | commutative | commutative_v1 | 64 | effort=high | 0.160 | 0 | 0 | 99780 | 1.00 | — | 0.16 | length:4, stop:21 | truncation 0.16 |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 16 | effort=high | 0.300 | 0 | 0 | 5645 | n/a | — | 0.00 | stop:21 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 64 | effort=high | 0.400 | 0 | 0 | 36358 | n/a | — | 0.07 | length:2, stop:18 | truncation 0.07 |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 128 | effort=high | 0.833 | 0 | 1 | 16761 | n/a | — | 0.03 | error:1, length:1, stop:5 | truncation 0.03 |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 512 | effort=high | 0.767 | 0 | 0 | 79903 | n/a | — | 0.20 | length:6, stop:7 | truncation 0.20 |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=high | 0.140 | 0 | 0 | 12977 | n/a | — | 0.00 | stop:43 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=low | 0.460 | 0 | 0 | 21082 | n/a | — | 0.02 | length:1, stop:27 | truncation 0.02 |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=medium | 0.300 | 0 | 0 | 12936 | n/a | — | 0.00 | stop:35 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v2 | 32 | effort=high | 0.080 | 0 | 0 | 251228 | 1.00 | 0.04 | 0.08 | length:2, stop:23 | truncation 0.08 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v2 | 64 | effort=high | 0.280 | 0 | 0 | 314362 | 1.00 | 0.00 | 0.24 | length:6, stop:19 | truncation 0.24 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v2 | 96 | effort=high | 0.160 | 0 | 0 | 501304 | 1.00 | 0.00 | 0.16 | length:4, stop:21 | truncation 0.16 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 194394 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.160 | 0 | 0 | 339009 | 1.00 | 0.00 | 0.08 | length:2, stop:23 | truncation 0.08 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 523532 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 642450 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.560 | 0 | 0 | 27679 | n/a | — | 0.00 | stop:11 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.440 | 0 | 0 | 77065 | n/a | — | 0.04 | length:1, stop:14 | truncation 0.04 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.600 | 0 | 0 | 136506 | n/a | — | 0.40 | length:10, stop:9 | truncation 0.40 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.320 | 0 | 0 | 297749 | n/a | — | 0.20 | length:5, stop:17 | truncation 0.20 |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.200 | 0 | 0 | 667824 | 1.00 | — | 0.20 | length:5, stop:20 | truncation 0.20 |
| nvidia/nemotron-3-ultra-550b-a55b | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.050 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.090 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.010 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.040 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.140 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.4 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 3229 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 18908 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 21230 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 25692 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 30627 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.4 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.4 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.4 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 28169 | n/a | — | 0.00 | stop:50 | — |
| openai/gpt-5.4 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.4 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 24474 | n/a | — | 0.00 | stop:25 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 42463 | n/a | — | 0.00 | stop:25 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 75328 | n/a | — | 0.00 | stop:25 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 126972 | n/a | — | 0.00 | stop:25 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.040 | 0 | 1 | 203651 | n/a | — | 0.00 | error:1, stop:24 | — |
| openai/gpt-5.4 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.4 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 2686 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | chain_instant | chain_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 9114 | 0.00 (—/1.00) | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 21206 | 1.00 | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 69752 | 1.00 | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 128 | effort=high | 0.040 | 0 | 0 | 226834 | 1.00 | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| openai/gpt-5.5 | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 22826 | 0.96 (1.00/1.00) | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 37415 | 1.00 | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 10673 | 0.32 (1.00/0.94) | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 8213 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 9903 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 13011 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 10373 | n/a | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.5 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.5 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.5 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 12798 | n/a | — | 0.00 | stop:50 | — |
| openai/gpt-5.5 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.5 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.5 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.5 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 116920 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 174090 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v1 | 64 | effort=high | 0.040 | 0 | 0 | 294009 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | truncation 0.04 |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 54515 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 93937 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 0.000 | 0 | 0 | 136909 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 181860 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 0.000 | 0 | 0 | 239545 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 96 | effort=high | 0.040 | 0 | 1 | 242595 | 0.96 (1.00/0.00) | 0.00 | 0.00 | error:1, stop:24 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 0.000 | 0 | 0 | 301324 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 136417 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 230665 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 332651 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 408163 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 23664 | n/a | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 44799 | n/a | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 107196 | n/a | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 174517 | n/a | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 255842 | 1.00 | — | 0.00 | stop:25 | — |
| openai/gpt-5.5 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.6-sol | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | chain_instant | chain_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 4029 | — | — | 0.00 | stop:25 | escalated @512tok diagnostic 1.00; canonical = first attempt @96tok |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 2744 | 0.00 (—/0.96) | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 5621 | 0.00 (—/1.00) | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 25501 | 0.88 (0.73/1.00) | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 128 | effort=high | 0.000 | 0 | 0 | 42235 | 1.00 | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 2374 | 0.00 (—/1.00) | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 4598 | 0.00 (—/1.00) | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 8629 | 0.12 (1.00/1.00) | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 17322 | 1.00 | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 5038 | 0.00 (—/0.76) | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.6-sol | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.6-sol | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 6482 | 0.12 (1.00/0.05) | 0.88 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 0.000 | 0 | 0 | 10629 | 0.16 (1.00/0.05) | 0.83 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 19798 | 0.20 (1.00/0.05) | 0.79 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 0.000 | 0 | 0 | 41976 | 0.40 (1.00/0.07) | 0.58 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 96 | effort=high | 0.000 | 0 | 0 | 57098 | 0.52 (1.00/0.08) | 0.48 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 0.000 | 0 | 0 | 71313 | 0.44 (1.00/0.14) | 0.52 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 16083 | 0.32 (1.00/0.00) | 0.68 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls; 3 runs, spread 0.32 |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 60888 | 0.60 (1.00/0.00) | 0.42 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls; 3 runs, spread 0.24 |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 77188 | 0.56 (1.00/0.09) | 0.42 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls; 3 runs, spread 0.24 |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 175420 | 0.76 (1.00/0.17) | 0.21 | 0.00 | stop:25 | ᵘ unworked answers on a large fraction of calls; 3 runs, spread 0.12 |
| openai/gpt-5.6-sol | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 66218 | 1.00 | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 157060 | 1.00 | — | 0.00 | stop:25 | — |
| openai/gpt-5.6-sol | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.6-sol | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 11331 | n/a | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | chain_instant | chain_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 31051 | 1.00 | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 68118 | 1.00 | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 135259 | 1.00 | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 128 | effort=high | 0.000 | 0 | 0 | 229053 | 1.00 | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 128370 | 1.00 | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 239096 | 1.00 | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 49912 | 1.00 | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 27705 | n/a | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 58421 | n/a | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 65182 | n/a | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 80118 | n/a | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| qwen/qwen3.7-max | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| qwen/qwen3.7-max | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 47533 | n/a | — | 0.00 | stop:50 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 45037 | n/a | — | 0.00 | stop:50 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 44170 | n/a | — | 0.00 | stop:50 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| qwen/qwen3.7-max | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 150878 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 239482 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 352106 | 1.00 | 0.05 | 0.00 | stop:25 | ‡ cap-escape |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 16 | effort=high | 0.000 | 0 | 0 | 127174 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 231327 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 0.000 | 0 | 0 | 187988 | 1.00 | 0.04 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 314347 | 1.00 | 0.04 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 0.000 | 0 | 0 | 320650 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 96 | effort=high | 0.000 | 0 | 0 | 393514 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 0.000 | 0 | 0 | 414306 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 215999 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 314522 | 1.00 | 0.04 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 434606 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 439884 | 1.00 | 0.08 | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 42853 | n/a | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 80913 | n/a | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 129591 | n/a | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 197412 | n/a | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 391517 | 1.00 | — | 0.00 | stop:25 | — |
| qwen/qwen3.7-max | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.980 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | — |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 12612 | n/a | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 82868 | n/a | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 136113 | n/a | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 132340 | n/a | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 153800 | n/a | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| x-ai/grok-4.3 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| x-ai/grok-4.3 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 148305 | n/a | — | 0.00 | stop:50 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 47245 | n/a | — | 0.00 | stop:50 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 102537 | n/a | — | 0.00 | stop:50 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| x-ai/grok-4.3 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 43136 | n/a | — | 0.00 | stop:25 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 70491 | n/a | — | 0.00 | stop:25 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 128954 | n/a | — | 0.00 | stop:25 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 222011 | n/a | — | 0.00 | stop:25 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 368483 | n/a | — | 0.00 | stop:25 | — |
| x-ai/grok-4.3 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| x-ai/grok-4.3 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:30 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 28854 | 1.00 | — | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 56099 | 1.00 | — | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 290597 | 1.00 | — | 0.00 | stop:25 | ‡ cap-escape |
| x-ai/grok-4.5 | chain_nowrap | chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 69411 | 1.00 | — | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v2 | 128 | effort=high | 0.000 | 0 | 0 | 157673 | 1.00 | — | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 41534 | 0.96 (0.71/1.00) | — | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v2 | 32 | effort=high | 0.000 | 0 | 0 | 142231 | 1.00 | 0.12 | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 1 | 195223 | 1.00 | 0.04 | 0.00 | error:1, stop:24 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v2 | 96 | effort=high | 0.000 | 0 | 0 | 288651 | 1.00 | 0.04 | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 113232 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.000 | 0 | 0 | 192716 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 253855 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.000 | 0 | 0 | 313992 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 201685 | 1.00 | — | 0.00 | stop:25 | — |
| x-ai/grok-4.5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 343024 | 0.92 (0.96/0.00) | — | 0.00 | stop:25 | — |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 45652 | 1.00 | — | 0.00 | stop:25 | — |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 32 | effort=high | 0.080 | 0 | 1 | 369760 | 1.00 | — | 0.04 | error:1, length:1, stop:23 | ‡ cap-escape; truncation 0.04 |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 64 | effort=high | 0.160 | 0 | 3 | 535933 | 1.00 | — | 0.04 | error:3, length:1, stop:21 | ‡ cap-escape; truncation 0.04 |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 128 | effort=high | 1.000 | 0 | 7 | 4443830 | 1.00 | — | 0.68 | error:7, length:17, stop:1 | ‡ cap-escape; truncation 0.68 |
| x-ai/grok-build-0.1 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 248308 | 1.00 | — | 0.00 | stop:25 | — |
| x-ai/grok-build-0.1 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 506163 | 1.00 | — | 0.00 | stop:25 | ‡ cap-escape |
| x-ai/grok-build-0.1 | sanity | conflict_v1 | 4 | effort=minimal | 0.000 | 0 | 0 | 15624 | — | — | 0.00 | stop:30 | — |
| x-ai/grok-build-0.1 | sanity | recall_copy_v1 | 6 | effort=minimal | 0.000 | 0 | 0 | 17408 | — | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 3554 | n/a | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.040 | 0 | 0 | 96 | — | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| z-ai/glm-5.2 | chain_instant | chain_v2 | 16 | contract, effort=none | 0.040 | 0 | 0 | 96 | — | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 10245 | 0.28 (1.00/0.94) | — | 0.00 | stop:25 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 32 | effort=high | 0.040 | 0 | 0 | 36687 | 0.72 (0.17/0.57) | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 64 | effort=high | 0.080 | 0 | 0 | 65558 | 1.00 | — | 0.08 | length:2, stop:23 | truncation 0.08 |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 128 | effort=high | 0.200 | 0 | 0 | 130971 | 1.00 | — | 0.20 | length:5, stop:20 | truncation 0.20 |
| z-ai/glm-5.2 | chain_nowrap | chain_v2 | 64 | effort=high | 0.040 | 0 | 0 | 31148 | 0.96 (1.00/1.00) | — | 0.00 | stop:24 | calls failed 0.04 |
| z-ai/glm-5.2 | chain_nowrap | chain_v2 | 128 | effort=high | 0.080 | 0 | 0 | 92814 | 1.00 | — | 0.04 | length:1, stop:23 | calls failed 0.04; truncation 0.04 |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 9631 | n/a | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 791 | — | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 12782 | n/a | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 533 | — | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 17258 | n/a | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 510 | — | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 512 | effort=high | 0.033 | 0 | 0 | 49122 | n/a | — | 0.03 | length:1, stop:29 | truncation 0.03 |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 383 | — | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 1083 | — | — | 0.00 | stop:50 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 2195 | — | — | 0.00 | stop:50 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:50 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=high | 0.020 | 0 | 0 | 28410 | n/a | — | 0.02 | length:1, stop:49 | truncation 0.02 |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 17587 | n/a | — | 0.00 | stop:50 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 23551 | n/a | — | 0.00 | stop:50 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 1432 | — | — | 0.00 | stop:50 | — |
| z-ai/glm-5.2 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 2621 | — | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 36 | — | — | 0.00 | stop:50 | — |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 32 | effort=high | 0.040 | 0 | 0 | 146112 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | truncation 0.04 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 32 | effort=xhigh | 0.120 | 0 | 0 | 180673 | 1.00 | 0.00 | 0.12 | length:3, stop:22 | truncation 0.12 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 64 | effort=high | 0.000 | 0 | 0 | 211838 | 1.00 | 0.04 | 0.00 | stop:25 | — |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 64 | effort=xhigh | 0.040 | 0 | 0 | 275998 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | truncation 0.04 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 96 | effort=high | 0.040 | 0 | 0 | 264298 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | truncation 0.04 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v2 | 96 | effort=xhigh | 0.200 | 0 | 0 | 424708 | 1.00 | 0.00 | 0.20 | length:5, stop:20 | truncation 0.20 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 0.000 | 0 | 0 | 151646 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| z-ai/glm-5.2 | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 0.080 | 0 | 0 | 378767 | 1.00 | 0.00 | 0.08 | length:2, stop:23 | truncation 0.08 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 0.000 | 0 | 0 | 306101 | 1.00 | 0.00 | 0.00 | stop:25 | — |
| z-ai/glm-5.2 | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 0.040 | 0 | 0 | 512645 | 1.00 | 0.00 | 0.04 | length:1, stop:24 | truncation 0.04 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v4 | 32 | sysprompt=04153d7439, effort=xhigh | 1.000 | 25 | 0 | 0 | n/a | — | — | — | calls failed 1.00 |
| z-ai/glm-5.2 | s5_chain | s5_chain_v4 | 128 | sysprompt=04153d7439, effort=xhigh | 1.000 | 25 | 0 | 0 | n/a | — | — | — | calls failed 1.00 |
| z-ai/glm-5.2 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 19329 | n/a | — | 0.00 | stop:25 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 33990 | n/a | — | 0.00 | stop:25 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 62202 | n/a | — | 0.00 | stop:25 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 139869 | 1.00 | — | 0.04 | length:1, stop:24 | truncation 0.04; 2 runs, spread 0.16 |
| z-ai/glm-5.2 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 317303 | 0.96 (0.75/1.00) | — | 0.04 | length:1, stop:24 | truncation 0.04 |
| z-ai/glm-5.2 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 336 | — | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 57 | — | — | 0.00 | stop:30 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.010 | 0 | 0 | 96 | — | — | 0.01 | length:1, stop:99 | truncation 0.01 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.030 | 0 | 0 | 288 | — | — | 0.03 | length:3, stop:97 | truncation 0.03 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.040 | 0 | 0 | 384 | — | — | 0.04 | length:4, stop:96 | truncation 0.04 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 510 | — | — | 0.00 | stop:100 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | 3 runs, spread 0.03 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | — | — | 0.00 | stop:100 | 3 runs, spread 0.03 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 85 | — | — | 0.00 | stop:100 | 3 runs, spread 0.00 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.030 | 0 | 0 | 288 | — | — | 0.03 | length:3, stop:97 | truncation 0.03; 3 runs, spread 0.04 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.020 | 0 | 0 | 192 | — | — | 0.02 | length:2, stop:98 | truncation 0.02; 3 runs, spread 0.03 |

## Repeat runs

Cells run more than once at identical settings (same model, facet, task, length, arm and token budget). The tables above publish the LAST run; the spread column is the run-to-run range of the same cell. A rerun at a different token budget is a budget rerun, not a repeat, and is not listed here.

| Model | Facet | Task | Length | Arm | runs | match per run | spread | published (last run) |
|---|---|---|---|---|---|---|---|---|
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 32 | effort=xhigh | 3 | 0.48 / 0.64 / 0.32 | 0.32 | 0.32 |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 64 | effort=xhigh | 3 | 0.64 / 0.84 / 0.60 | 0.24 | 0.60 |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 96 | effort=xhigh | 3 | 0.72 / 0.84 / 0.60 | 0.24 | 0.60 |
| z-ai/glm-5.2 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 2 | 1.00 / 0.84 | 0.16 | 0.84 |
| openai/gpt-5.6-sol | s5_chain | s5_chain_v3 | 128 | effort=xhigh | 3 | 0.68 / 0.80 / 0.80 | 0.12 | 0.80 |
| anthropic/claude-sonnet-5 | commutative | commutative_v1 | 64 | effort=high | 2 | 0.60 / 0.64 | 0.04 | 0.64 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 3 | 0.35 / 0.34 / 0.38 | 0.04 | 0.38 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 3 | 0.70 / 0.68 / 0.71 | 0.03 | 0.71 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 3 | 0.36 / 0.34 / 0.37 | 0.03 | 0.37 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 3 | 0.16 / 0.15 / 0.13 | 0.03 | 0.13 |
| moonshotai/kimi-k2.6 | commutative | commutative_v1 | 64 | effort=high | 2 | 0.68 / 0.66 | 0.02 | 0.66 |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 3 | 1.00 / 1.00 / 1.00 | 0.00 | 1.00 |

## Provenance: INVALID chain_depth cells (wrapped k=6 cycle)

These cells ran chain_v1 past its design gate (depth >= k=6, so the pointer cycle wrapped): they measure the wrapped task, not depth, and are excluded from every figure and headline column above. They are kept here as provenance only; the redesigned facet is chain_nowrap.

| Model | Facet | Task | Length | Arm | n | match [95% CI] | containment (diagnostic) | note |
|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 16 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 32 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.73 [0.56, 0.86] | 0.73 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.70 [0.52, 0.83] | 0.70 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.67 [0.49, 0.81] | 0.67 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.03 |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.03 [0.01, 0.17] | 0.03 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 24 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.50 [0.33, 0.67] | 0.50 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.03 |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.83 [0.66, 0.93] | 0.83 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.17 |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.50 [0.33, 0.67] | 0.50 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.50 |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 32 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 48 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 32 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 48 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 8 | effort=default | 30 | 0.17 [0.07, 0.34] | 0.27 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 12 | effort=default | 30 | 0.00 [0.00, 0.11] | 0.23 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 16 | effort=default | 30 | 0.10 [0.03, 0.26] | 0.20 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 24 | effort=default | 30 | 0.00 [0.00, 0.11] | 0.67 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 32 | effort=default | 30 | 0.10 [0.03, 0.26] | 0.23 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.03 |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 48 | effort=default | 30 | 0.03 [0.01, 0.17] | 0.27 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 64 | effort=default | 30 | 0.03 [0.01, 0.17] | 0.13 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.03 |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.90 [0.74, 0.97] | 0.90 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.10 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 16 | effort=high | 30 | 0.77 [0.59, 0.88] | 0.77 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 24 | effort=high | 30 | 0.00 [0.00, 0.11] | 0.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.00 [0.00, 0.11] | 0.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.10 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.00 [0.00, 0.11] | 0.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 1.00 |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.00 [0.00, 0.11] | 0.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 1.00 |
| openai/gpt-5.4 | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.4 | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.4 | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.4 | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.4 | chain_depth | chain_v1 | 32 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.4 | chain_depth | chain_v1 | 48 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.4 | chain_depth | chain_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.5 | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.5 | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.5 | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.5 | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.5 | chain_depth | chain_v1 | 32 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.5 | chain_depth | chain_v1 | 48 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| openai/gpt-5.5 | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.83 [0.66, 0.93] | 0.83 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.77 [0.59, 0.88] | 0.77 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.43 [0.27, 0.61] | 0.43 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 16 | effort=high | 30 | 0.70 [0.52, 0.83] | 0.70 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.47 [0.30, 0.64] | 0.47 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.83 [0.66, 0.93] | 0.83 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 16 | effort=high | 30 | 0.87 [0.70, 0.95] | 0.87 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 24 | effort=high | 30 | 0.67 [0.49, 0.81] | 0.67 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.10 [0.03, 0.26] | 0.10 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.03 |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.20 [0.10, 0.37] | 0.20 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.03 |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.27 [0.14, 0.44] | 0.27 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap); truncation 0.17 |
