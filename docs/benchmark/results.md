# FactWorld frontier benchmark — results

Generated 2026-07-14 11:46 UTC from `results/benchmark/history.jsonl` (574 latest cells).

## Settings

Canonical metric: **match** — strip a trailing period from both sides and compare the model's first len(gold) whitespace tokens to the gold answer; binary per item, no partial credit (`factworld.tasks.score_relaxed`). Containment is the one published diagnostic.
Figures draw a dotted reference line at match 0.8.
Error bars / intervals: Wilson 95% CI.

Observed generation settings (effort -> max_new_tokens, stop_at; annotated with the facets that ran under each combo):

- effort=default: max_new_tokens=2048, stop_at=None — facets: chain_depth, composite_length, decomposition, dose_response, floor, s5_concrete, sanity
- effort=high: max_new_tokens=16384, stop_at=None — facets: chain_nowrap, s5_concrete
- effort=high: max_new_tokens=32768, stop_at=None — facets: chain_nowrap, s5_concrete
- effort=high: max_new_tokens=8192, stop_at=None — facets: chain_depth, commutative, composite_length, dose_response, s5_concrete
- effort=low: max_new_tokens=8192, stop_at=None — facets: dose_response
- effort=medium: max_new_tokens=8192, stop_at=None — facets: dose_response
- effort=minimal: max_new_tokens=2048, stop_at=None — facets: composite_length, decomposition, dose_response, floor, sanity
- effort=minimal: max_new_tokens=96, stop_at=None — facets: chain_instant, recall_load, zero_budget
- effort=none: max_new_tokens=2048, stop_at=None — facets: composite_length, decomposition, dose_response, floor, sanity
- effort=none: max_new_tokens=96, stop_at=None — facets: chain_instant, gap_stability, recall_load, zero_budget

## Instant headline (current roster)

Current roster only (factworld.benchmark.MODELS); models dropped from the roster render in the archived-models section below.

The benchmark is a composition instrument: recall and state tracking are the component abilities, and 'instant' cells (reasoning off, hard one-line answer contract) measure whether the model composes them in-weights — the composition gap column is the deficit. 'thinking' cells measure composition with reasoning: ~ceiling at canonical settings for this roster, so the state-stress columns (chain d128 at k=257, s5 @L256) carry the thinking discrimination.

Instant cells: task **composite_copy_v2** with reasoning off (effort=none) under a one-line answer contract (settings.contract=true); match. Escalated cells show the CANONICAL first attempt at the shared base budget, with the escalated rerun as a parenthesised diagnostic.

Notation: `@Ln` = stream length (events, or hops for chain depth d); `@Ntok` = a completion-token budget. Instant escalations render `(diag x.xx @512tok)`; thinking cells rerun at a raised budget render it with the number, e.g. `1.00 @32,768tok (raised budget)`.

History also contains zero-budget cells on composite_copy_v1; the zero-budget columns below use the latest task's records (composite_copy_v2) only — the archived task's cells remain in the per-cell tables.

| Model | instant: recall (sanity, recall_copy_v1) | instant: state tracking (binding_only @L16, v2) | instant: composed @L16 (match, v2) | instant: composed @L64 (v2) | instant: composition gap (binding_only - composed @L16) | instant: replicate noise (|composed - replicate| @L16) |
|---|---|---|---|---|---|---|
| moonshotai/kimi-k2.6 | 1.00 | ≤0.94† | ≤0.77† | ≤0.93† | +0.17† | ±0.06 |
| anthropic/claude-opus-4.8 | 1.00 | 0.78 | 0.72 | 0.43 | +0.06 | ±0.05 |
| openai/gpt-5.6-sol | 1.00 | 0.82 | 0.65 | 0.33 | +0.17 | ±0.05 |
| google/gemini-3.5-flash | 1.00 | 0.66* | 0.64* | 0.28* | +0.02* | ±0.01 |
| anthropic/claude-sonnet-5 | 0.97 | 0.77 | 0.62 (diag 0.76 @512tok)† | 0.32 (diag 0.66 @512tok)† | +0.15† | ±0.03 |
| openai/gpt-5.5 | 1.00 | 0.80 | 0.46 | 0.33 | +0.34 | ±0.00 |
| deepseek/deepseek-v4-pro | 1.00 | 0.51 | 0.44 | 0.19 | —ᶠ | ±0.00 |
| z-ai/glm-5.2 | 1.00 | 0.71 | 0.38† | 0.13 | +0.33† | ±0.01 |
| nvidia/nemotron-3-ultra-550b-a55b | 1.00 | 0.49 | 0.33 | 0.12 | —ᶠ | ±0.03 |
| qwen/qwen3.7-max | 1.00 | 0.51 | 0.24 | 0.08 | —ᶠ | ±0.01 |
| muse-spark-1.1 | n/a | n/a | n/a | n/a | n/a | n/a |
| x-ai/grok-4.5 | n/a | n/a | n/a | n/a | n/a | n/a |
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

⊘ = not measurable at this budget; ≤x† = upper bound, covert reasoning on most calls; neither participates in orderings.

composition gap = state tracking (binding_only @L16) - composed @L16, marks from either input cell propagated. recall|holder is ~1.0 for every roster model (the scaffolded leg), so if composition were free the composed cell would match the binding leg; the gap is the composition deficit.

replicate noise: the zero_budget replicate leg (recorded as end_to_end in earlier runs) builds prompts IDENTICAL to the composed @L16 cell (same runner path), so |composed - replicate| is a test-retest delta; max across models = 0.06 — read that as the run-to-run noise bar on the headline numbers (including the gap column). Future runs keep this arm intentionally as leg='replicate'.

## Thinking headline (current roster)

Thinking-regime state-stress cells (effort=high): chain d128 is a pointer chase 128 hops deep at fixed breadth k=257; s5 @L256 is non-abelian state tracking over 256 events. The s5@128 ctok column measures efficiency on the matched L128 cell that every current-roster model runs.

Notation: `@Ln` = stream length (events, or hops for chain depth d); `@Ntok` = a completion-token budget. Instant escalations render `(diag x.xx @512tok)`; thinking cells rerun at a raised budget render it with the number, e.g. `1.00 @32,768tok (raised budget)`.

| Model | thinking: chain d128 (chain_nowrap, k=257, match) | thinking: s5 @L256 (s5_concrete, match) | thinking: s5@128 ctok |
|---|---|---|---|
| x-ai/grok-4.5 | n/a | 1.00‡ | 8069 |
| muse-spark-1.1 | 0.88 @32,768tok (raised budget) | 1.00 @32,768tok (raised budget) | 9704 |
| anthropic/claude-sonnet-5 | 0.04 | 1.00 @32,768tok (raised budget) | 11866 |
| anthropic/claude-opus-4.8 | 0.08 | 1.00 @32,768tok (raised budget) | 12683 |
| openai/gpt-5.5 | 0.36 | 0.96 | 6989 |
| z-ai/glm-5.2 | 0.36 | 0.88 | 6282 |
| moonshotai/kimi-k2.6 | 0.64‡ | 0.88 | 17418 |
| qwen/qwen3.7-max | 0.96 | 0.80 | 7904 |
| google/gemini-3.5-flash | 0.88 | 0.52 | 11022 |
| openai/gpt-5.6-sol | 1.00 | n/a | 2657 |
| deepseek/deepseek-v4-pro | ⊘ >budget @32,768tok (raised budget) | ⊘ >budget | 10043 |
| nvidia/nemotron-3-ultra-550b-a55b | ⊘ >budget @32,768tok (raised budget) | ⊘ >budget | 12250 |

Thinking columns: n=25 per cell; Wilson intervals ≈ ±0.15–0.19, and the one thinking test-retest pair moved 0.16 — differences under ~0.2 are not an ordering.

s5@128 ctok: completion tokens per call on the matched s5_concrete L128 cell (run by every current-roster model). This replaces ctok/solve, which averaged only over cells a model SOLVED and therefore rewarded models that failed early (selection bias: the published 2.7x opus-vs-kimi ctok/solve gap is ~1.4x on the matched cell).

## S5 efficiency ranking

S5 efficiency ranking: models sorted by s5 @L256 score, then by s5@128 completion tokens per call (lower is better) on the matched s5_concrete L128 cell (the cell every current-roster model runs). At s5 @L256 several models hit 1.00, so token efficiency is the practical discriminator.

| Model | s5 @L256 | s5@128 ctok/call |
|---|---|---|
| x-ai/grok-4.5 | 1.00‡ | 8069 |
| muse-spark-1.1 | 1.00 @32,768tok (raised budget) | 9704 |
| anthropic/claude-sonnet-5 | 1.00 @32,768tok (raised budget) | 11866 |
| anthropic/claude-opus-4.8 | 1.00 @32,768tok (raised budget) | 12683 |
| openai/gpt-5.5 | 0.96 | 6989 |
| z-ai/glm-5.2 | 0.88 | 6282 |
| moonshotai/kimi-k2.6 | 0.88 | 17418 |
| qwen/qwen3.7-max | 0.80 | 7904 |
| google/gemini-3.5-flash | 0.52 | 11022 |
| openai/gpt-5.6-sol | n/a | 2657 |
| deepseek/deepseek-v4-pro | ⊘ >budget | 10043 |
| nvidia/nemotron-3-ultra-550b-a55b | ⊘ >budget | 12250 |

The chain column reads the `chain_nowrap` facet only (staircase k=2d+1, so the d128 cell is k=257). `chain_v1` builds a single k=6 pointer cycle and measures depth only for depths < k (`factworld/tasks.py`: "Depths stay < k so the cycle never wraps"); `chain_depth` cells at depth >= 6 wrapped the cycle (gold == start agent at depths 12/24/48; effective difficulty depth mod 6), measure the wrapped task rather than depth, and are marked `INVALID (k=6 cycle wrap — task redesigned as chain_nowrap)` in the tables below and excluded from the chain figure.

## Instant stress rows (recall under load; chain d16)

Two instant cells beyond the composite headline, same protocol (reasoning off, one-line answer contract, 96-token cap; marks and escalated diagnostics as in the headline). recall_load scales the recall pool with the length (recall_copy_v1 @L64, pool 64, n=50): single-query deferred recall under working-set load. chain_instant runs chain_v1 d16 on the same k=33 staircase items as the thinking d16 cell (n=25): the within-item regime contrast for depth. The floor row is the uniform guess over the answer pool; escalated cells show the CANONICAL first attempt with the escalated rerun as a parenthesised diagnostic.

| Model | instant: recall under load (recall_load, recall_copy_v1 pool-64 @L64) | instant: chain d16 (chain_instant, chain_v1, k=33) |
|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 0.00 (diag 0.96 @512tok)† |
| anthropic/claude-sonnet-5 | 1.00 | 0.28 (diag 0.96 @512tok)† |
| deepseek/deepseek-v4-pro | 1.00 | 0.00 |
| google/gemini-3.5-flash | 1.00* | 0.00 (diag 1.00 @512tok)*† |
| moonshotai/kimi-k2.6 | 1.00 | 0.32 (diag 0.96 @512tok)† |
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
| meta-llama/llama-4-maverick | 0.16 @ default | 0.10 | 0.96 / 0.18 / 1.00 |
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
| google/gemini-3.5-flash | 1.00 @ high | 0.97 | 0.82 / 0.44 / 1.00 |
| moonshotai/kimi-k2.6 | 0.98 @ high | 0.97 | 0.78 / 0.26 / 1.00 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.84 @ high | 0.23 | 0.60 / 0.22 / 1.00 |
| openai/gpt-5.5 | 1.00 @ high | 1.00 | 0.96 / 0.74 / 1.00 |
| qwen/qwen3.7-max | 0.92 @ high | 1.00 | 0.60 / 0.18 / 1.00 |
| z-ai/glm-5.2 | 0.94 @ high | 0.93 | 0.56 / 0.20 / 1.00 |

## Full per-cell results

match is the CANONICAL value (first attempt for escalated cells; the escalated diagnostic is in the note column). ‡ = cap-escape (see headline footnotes). INVALID chain_depth cells are quarantined in the provenance section at the end.

| Model | Facet | Task | Length | Arm | n | match [95% CI] | containment (diagnostic) | note |
|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.96 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.08 [0.02, 0.25] | 0.08 | — |
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
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
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
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.72 [0.52, 0.86] | 0.72 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.24 [0.11, 0.43] | 0.24 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.04 [0.01, 0.20] | 0.04 | — |
| anthropic/claude-sonnet-5 | commutative | commutative_v1 | 64 | effort=high | 50 | 0.64 [0.50, 0.76] | 0.70 | — |
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
| anthropic/claude-sonnet-5 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.00 [0.00, 0.11] | 0.87 | — |
| anthropic/claude-sonnet-5 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 50 | 0.64 [0.50, 0.76] | — | escalated @512tok diagnostic 0.72; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 50 | 0.50 [0.37, 0.63] | 0.64 | escalated @512tok diagnostic 0.64; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| anthropic/claude-sonnet-5 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 0.97 [0.83, 0.99] | 1.00 | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.71 [0.61, 0.79] | — | — |
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
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.08 [0.02, 0.25] | 0.08 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.83 [0.66, 0.93] | 0.83 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.27 [0.14, 0.44] | 0.27 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.10 [0.03, 0.26] | 0.10 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.03 [0.01, 0.17] | 0.03 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.87 [0.70, 0.95] | 0.87 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.07 [0.02, 0.21] | 0.07 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.24 [0.14, 0.37] | — | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.24 [0.14, 0.37] | 0.24 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 0.98 [0.90, 1.00] | — | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.92 [0.81, 0.97] | 0.92 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 0.92 [0.81, 0.97] | 0.92 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 0.84 [0.71, 0.92] | 0.84 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.28 [0.17, 0.42] | 0.28 | — |
| deepseek/deepseek-v4-pro | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.17 [0.07, 0.34] | 0.17 | — |
| deepseek/deepseek-v4-pro | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.28 [0.14, 0.48] | 0.28 | — |
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
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| google/gemini-3.1-pro-preview | sanity | conflict_v1 | 4 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.1-pro-preview | sanity | recall_copy_v1 | 6 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_instant | chain_v1 | 16 | contract, effort=minimal | 25 | 0.00 [0.00, 0.13] | 1.00 | escalated @512tok diagnostic 1.00; canonical = first attempt @96tok |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| google/gemini-3.5-flash | commutative | commutative_v1 | 64 | effort=high | 25 | 0.80 [0.61, 0.91] | 0.80 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 16 | effort=minimal | 30 | 0.60 [0.42, 0.75] | 0.70 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 64 | effort=minimal | 30 | 0.40 [0.25, 0.58] | 0.50 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 128 | effort=minimal | 30 | 0.50 [0.33, 0.67] | 0.50 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 512 | effort=minimal | 30 | 0.23 [0.12, 0.41] | 0.23 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=minimal | 50 | 0.82 [0.69, 0.90] | — | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=minimal | 50 | 0.44 [0.31, 0.58] | 0.54 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=minimal | 50 | 1.00 [0.93, 1.00] | — | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=minimal | 50 | 0.62 [0.48, 0.74] | 0.74 | — |
| google/gemini-3.5-flash | floor | s5 | 16 | rendering=abstract_stated, effort=minimal | 30 | 0.23 [0.12, 0.41] | 0.63 | — |
| google/gemini-3.5-flash | recall_load | recall_copy_v1 | 64 | contract, effort=minimal | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.52 [0.33, 0.70] | 0.56 | — |
| google/gemini-3.5-flash | sanity | conflict_v1 | 4 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | sanity | recall_copy_v1 | 6 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=minimal | 100 | 0.56 [0.46, 0.65] | — | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=minimal | 100 | 0.47 [0.38, 0.57] | 0.48 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | contract, effort=minimal | 100 | 0.45 [0.36, 0.55] | 0.46 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 64 | contract, effort=minimal | 100 | 0.34 [0.25, 0.44] | 0.34 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=minimal | 100 | 0.66 [0.56, 0.75] | — | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=minimal | 100 | 0.65 [0.55, 0.74] | 0.66 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=minimal | 100 | 1.00 [0.96, 1.00] | — | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | contract, effort=minimal | 100 | 0.64 [0.54, 0.73] | 0.64 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 64 | contract, effort=minimal | 100 | 0.28 [0.20, 0.37] | 0.28 | — |
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
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.64 [0.45, 0.80] | 0.68 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | commutative | commutative_v1 | 64 | effort=high | 50 | 0.66 [0.52, 0.78] | 0.66 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.40 [0.25, 0.58] | 0.53 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.97 [0.83, 0.99] | 1.00 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.17 [0.07, 0.34] | 0.27 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.07 [0.02, 0.21] | 0.10 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.10 [0.03, 0.26] | 0.40 | — |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.78 [0.65, 0.87] | — | — |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.26 [0.16, 0.40] | 0.96 | — |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.98 [0.90, 1.00] | 0.98 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.48 [0.35, 0.61] | 0.56 | — |
| moonshotai/kimi-k2.6 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.27 [0.14, 0.44] | 0.27 | — |
| moonshotai/kimi-k2.6 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 50 | 0.36 [0.24, 0.50] | — | ‡ cap-escape |
| moonshotai/kimi-k2.6 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 50 | 0.56 [0.42, 0.69] | 0.86 | escalated @512tok diagnostic 0.86; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| moonshotai/kimi-k2.6 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.52 [0.42, 0.62] | — | escalated @512tok diagnostic 0.92; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.48 [0.38, 0.58] | 0.85 | escalated @512tok diagnostic 0.82; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.48 [0.38, 0.58] | 0.86 | escalated @512tok diagnostic 0.83; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.38 [0.29, 0.48] | 0.97 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.94 [0.88, 0.97] | — | ‡ cap-escape |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.83 [0.74, 0.89] | 0.87 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 0.98 [0.93, 0.99] | — | — |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.77 [0.68, 0.84] | 0.84 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.93 [0.86, 0.97] | 0.95 | ‡ cap-escape |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.92 | — |
| muse-spark-1.1 | commutative | commutative_v1 | 64 | effort=high | 25 | 0.16 [0.06, 0.35] | 0.40 | — |
| muse-spark-1.1 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| muse-spark-1.1 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 0.44 [0.27, 0.63] | 0.44 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.04 [0.01, 0.20] | 0.04 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | commutative | commutative_v1 | 64 | effort=high | 25 | 0.44 [0.27, 0.63] | 0.44 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.70 [0.52, 0.83] | 0.70 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.20 [0.10, 0.37] | 0.20 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 0.57 [0.39, 0.73] | 0.57 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.20 [0.10, 0.37] | 0.20 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 0.17 [0.07, 0.34] | 0.17 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.13 [0.05, 0.30] | 0.13 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.23 [0.12, 0.41] | 0.23 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.03 [0.01, 0.17] | 0.03 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.60 [0.46, 0.72] | — | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.22 [0.13, 0.35] | 0.22 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.84 [0.71, 0.92] | 0.86 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 0.50 [0.37, 0.63] | 0.54 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 0.62 [0.48, 0.74] | 0.64 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.16 [0.08, 0.29] | 0.16 | — |
| nvidia/nemotron-3-ultra-550b-a55b | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.30 [0.17, 0.48] | 0.30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 0.44 [0.27, 0.63] | 0.44 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 0.56 [0.37, 0.73] | 0.56 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 0.36 [0.20, 0.55] | 0.36 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.68 [0.48, 0.83] | 0.68 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
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
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.84 [0.65, 0.94] | 0.84 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.36 [0.20, 0.55] | 0.36 | — |
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
| openai/gpt-5.5 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
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
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.76 [0.57, 0.89] | 0.76 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | commutative | commutative_v1 | 64 | effort=high | 25 | 0.76 [0.57, 0.89] | 0.76 | — |
| openai/gpt-5.6-sol | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 50 | 0.58 [0.44, 0.71] | — | — |
| openai/gpt-5.6-sol | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 50 | 0.26 [0.16, 0.40] | 0.26 | — |
| openai/gpt-5.6-sol | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.82 [0.73, 0.88] | — | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.60 [0.50, 0.69] | 0.60 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 1.00 [0.96, 1.00] | — | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.65 [0.55, 0.74] | 0.65 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.33 [0.25, 0.43] | 0.33 | — |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
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
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.23 [0.12, 0.41] | 0.83 | — |
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
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 0.32 [0.21, 0.46] | 0.66 | — |
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
| x-ai/grok-4.5 | commutative | commutative_v1 | 64 | effort=high | 25 | 0.72 [0.52, 0.86] | 0.72 | — |
| x-ai/grok-4.5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| x-ai/grok-4.5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | ‡ cap-escape |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.72 [0.52, 0.86] | 0.72 | ‡ cap-escape |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.60 [0.41, 0.77] | 0.60 | ‡ cap-escape |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.00 [0.00, 0.13] | 0.00 | ‡ cap-escape |
| x-ai/grok-build-0.1 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | — |
| x-ai/grok-build-0.1 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.92 [0.75, 0.98] | 0.92 | ‡ cap-escape |
| x-ai/grok-build-0.1 | sanity | conflict_v1 | 4 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| x-ai/grok-build-0.1 | sanity | recall_copy_v1 | 6 | effort=minimal | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 4 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | chain_instant | chain_v1 | 16 | contract, effort=none | 25 | 0.00 [0.00, 0.13] | 0.00 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 16 | effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 32 | effort=high | 25 | 0.28 [0.14, 0.48] | 0.28 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 64 | effort=high | 25 | 0.48 [0.30, 0.67] | 0.48 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 128 | effort=high | 25 | 0.36 [0.20, 0.55] | 0.36 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 16 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.97 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 16 | effort=none | 30 | 0.67 [0.49, 0.81] | 0.67 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 64 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 64 | effort=none | 30 | 0.37 [0.22, 0.54] | 0.37 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 128 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 128 | effort=none | 30 | 0.13 [0.05, 0.30] | 0.13 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 512 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 512 | effort=none | 30 | 0.17 [0.07, 0.34] | 0.17 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 50 | 0.56 [0.42, 0.69] | — | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 50 | 0.20 [0.11, 0.33] | 0.28 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 50 | 1.00 [0.93, 1.00] | — | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=high | 50 | 0.94 [0.84, 0.98] | 0.96 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=low | 50 | 0.94 [0.84, 0.98] | 0.96 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=medium | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=none | 50 | 0.68 [0.54, 0.79] | 0.68 | — |
| z-ai/glm-5.2 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 30 | 0.23 [0.12, 0.41] | 0.23 | — |
| z-ai/glm-5.2 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 50 | 1.00 [0.93, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 25 | 0.96 [0.80, 0.99] | 0.96 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 25 | 1.00 [0.87, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 25 | 0.84 [0.65, 0.94] | 0.84 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 25 | 0.88 [0.70, 0.96] | 0.88 | — |
| z-ai/glm-5.2 | sanity | conflict_v1 | 4 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | sanity | recall_copy_v1 | 6 | effort=none | 30 | 1.00 [0.89, 1.00] | 1.00 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 100 | 0.64 [0.54, 0.73] | — | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 100 | 0.35 [0.26, 0.45] | 0.36 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 100 | 0.31 [0.23, 0.41] | 0.31 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 100 | 0.15 [0.09, 0.23] | 0.17 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 100 | 0.71 [0.61, 0.79] | — | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 100 | 0.37 [0.28, 0.47] | 0.39 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 100 | 1.00 [0.96, 1.00] | — | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 100 | 0.38 [0.29, 0.48] | 0.42 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 100 | 0.13 [0.08, 0.21] | 0.14 | — |

## Diagnostics per cell

finish_errors counts per-example finish=='error' calls (surfaced even where diagnostics.api_errors is 0). ctok = completion tokens; rtok = reasoning tokens.

| Model | Facet | Task | Length | Arm | empty_rate | api_errors | finish_errors | reasoning_tokens | finish_reasons | note |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 1382 | stop:30 | — |
| anthropic/claude-opus-4.8 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:25 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 1244 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 1703 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 2146 | stop:25 | — |
| anthropic/claude-opus-4.8 | chain_nowrap | chain_v1 | 128 | effort=high | 0.040 | 0 | 0 | 14665 | length:1, stop:24 | — |
| anthropic/claude-opus-4.8 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 2395 | stop:25 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 1319 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 2557 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 3352 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 2761 | stop:30 | — |
| anthropic/claude-opus-4.8 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-opus-4.8 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-opus-4.8 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-opus-4.8 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-opus-4.8 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 2448 | stop:50 | — |
| anthropic/claude-opus-4.8 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-opus-4.8 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-opus-4.8 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-opus-4.8 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-opus-4.8 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 7578 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 13440 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 28491 | stop:25 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 48411 | length:1, stop:24 | — |
| anthropic/claude-opus-4.8 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 141239 | stop:25 | — |
| anthropic/claude-opus-4.8 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-opus-4.8 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-opus-4.8 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 1303 | stop:30 | — |
| anthropic/claude-sonnet-5 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:25 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 1351 | stop:25 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 1564 | stop:25 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 64 | effort=high | 0.040 | 0 | 0 | 4323 | length:1, stop:24 | — |
| anthropic/claude-sonnet-5 | chain_nowrap | chain_v1 | 128 | effort=high | 0.280 | 0 | 0 | 9958 | length:7, stop:18 | — |
| anthropic/claude-sonnet-5 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 3370 | stop:50 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 1534 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 2595 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 2809 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 5228 | stop:30 | — |
| anthropic/claude-sonnet-5 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-sonnet-5 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-sonnet-5 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-sonnet-5 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-sonnet-5 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 2502 | stop:50 | — |
| anthropic/claude-sonnet-5 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-sonnet-5 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | length:1, stop:29 | — |
| anthropic/claude-sonnet-5 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | escalated @512tok diagnostic 0.72; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | escalated @512tok diagnostic 0.64; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 6510 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 13050 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.040 | 0 | 1 | 28952 | error:1, stop:24 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 53155 | stop:25 | — |
| anthropic/claude-sonnet-5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 164247 | stop:25 | — |
| anthropic/claude-sonnet-5 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-sonnet-5 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.020 | 0 | 0 | 0 | length:3, stop:97 | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | escalated @512tok diagnostic 0.75; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | escalated @512tok diagnostic 0.77; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | escalated @512tok diagnostic 0.67; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | escalated @512tok diagnostic 0.82; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | escalated @512tok diagnostic 0.76; canonical = first attempt @96tok |
| anthropic/claude-sonnet-5 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | escalated @512tok diagnostic 0.66; canonical = first attempt @96tok |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 7441 | stop:30 | — |
| deepseek/deepseek-v4-pro | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:25 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 17023 | stop:25 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 62570 | stop:25 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 64 | effort=high | 0.040 | 0 | 0 | 151615 | length:1, stop:24 | — |
| deepseek/deepseek-v4-pro | chain_nowrap | chain_v1 | 128 | effort=high | 0.760 | 0 | 0 | 698721 | length:19, stop:5 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 16 | effort=high | 0.033 | 0 | 0 | 26943 | length:1, stop:29 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 44114 | stop:30 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 128 | effort=high | 0.033 | 0 | 0 | 53840 | length:1, stop:29 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 512 | effort=high | 0.100 | 0 | 0 | 83206 | length:3, stop:27 | — |
| deepseek/deepseek-v4-pro | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| deepseek/deepseek-v4-pro | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 28490 | stop:50 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=low | 0.020 | 0 | 0 | 58200 | length:1, stop:49 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 33409 | stop:50 | — |
| deepseek/deepseek-v4-pro | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| deepseek/deepseek-v4-pro | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| deepseek/deepseek-v4-pro | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 35520 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 65442 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 127456 | length:1, stop:24 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 244509 | stop:25 | — |
| deepseek/deepseek-v4-pro | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.720 | 0 | 0 | 396133 | length:18, stop:7 | — |
| deepseek/deepseek-v4-pro | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| deepseek/deepseek-v4-pro | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| deepseek/deepseek-v4-pro | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| google/gemini-3.1-pro-preview | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 9852 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 41148 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 16 | effort=minimal | 0.000 | 0 | 0 | 13820 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 100821 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 64 | effort=minimal | 0.000 | 0 | 0 | 23649 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 122323 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 128 | effort=minimal | 0.000 | 0 | 0 | 19528 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 111633 | stop:30 | — |
| google/gemini-3.1-pro-preview | composite_length | composite_copy_v1 | 512 | effort=minimal | 0.000 | 0 | 0 | 27161 | stop:30 | — |
| google/gemini-3.1-pro-preview | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=minimal | 0.000 | 0 | 0 | 7894 | stop:50 | — |
| google/gemini-3.1-pro-preview | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=minimal | 0.020 | 0 | 1 | 12102 | error:1, stop:49 | — |
| google/gemini-3.1-pro-preview | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=minimal | 0.000 | 0 | 0 | 6138 | stop:50 | — |
| google/gemini-3.1-pro-preview | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 61752 | stop:50 | — |
| google/gemini-3.1-pro-preview | dose_response | composite_copy_v1 | 16 | effort=minimal | 0.000 | 0 | 0 | 22069 | stop:50 | — |
| google/gemini-3.1-pro-preview | floor | s5 | 16 | rendering=abstract_stated, effort=minimal | 0.000 | 0 | 0 | 29524 | stop:30 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 57602 | stop:25 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 103871 | stop:25 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 164515 | stop:25 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 344753 | length:1, stop:24 | — |
| google/gemini-3.1-pro-preview | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 384750 | stop:25 | — |
| google/gemini-3.1-pro-preview | sanity | conflict_v1 | 4 | effort=minimal | 0.000 | 0 | 0 | 6169 | stop:30 | — |
| google/gemini-3.1-pro-preview | sanity | recall_copy_v1 | 6 | effort=minimal | 0.000 | 0 | 0 | 6405 | stop:30 | — |
| google/gemini-3.5-flash | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 15205 | stop:30 | — |
| google/gemini-3.5-flash | chain_instant | chain_v1 | 16 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | stop:25 | escalated @512tok diagnostic 1.00; canonical = first attempt @96tok |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 39265 | stop:25 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 68234 | stop:25 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 140042 | stop:25 | — |
| google/gemini-3.5-flash | chain_nowrap | chain_v1 | 128 | effort=high | 0.000 | 0 | 0 | 272521 | stop:25 | — |
| google/gemini-3.5-flash | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 89076 | stop:25 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 30421 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 16 | effort=minimal | 0.000 | 0 | 0 | 0 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 52078 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 64 | effort=minimal | 0.000 | 0 | 0 | 0 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 74015 | length:1, stop:29 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 128 | effort=minimal | 0.000 | 0 | 0 | 0 | stop:30 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 100637 | length:1, stop:29 | — |
| google/gemini-3.5-flash | composite_length | composite_copy_v1 | 512 | effort=minimal | 0.000 | 0 | 0 | 0 | stop:30 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=minimal | 0.000 | 0 | 0 | 0 | stop:50 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=minimal | 0.000 | 0 | 0 | 0 | stop:50 | — |
| google/gemini-3.5-flash | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=minimal | 0.000 | 0 | 0 | 0 | stop:50 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 47756 | stop:50 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 17044 | stop:50 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 35154 | stop:50 | — |
| google/gemini-3.5-flash | dose_response | composite_copy_v1 | 16 | effort=minimal | 0.000 | 0 | 0 | 0 | stop:50 | — |
| google/gemini-3.5-flash | floor | s5 | 16 | rendering=abstract_stated, effort=minimal | 0.000 | 0 | 0 | 0 | length:1, stop:29 | — |
| google/gemini-3.5-flash | recall_load | recall_copy_v1 | 64 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | stop:50 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 56548 | stop:25 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 102124 | stop:25 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 141222 | length:2, stop:23 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 274191 | length:2, stop:23 | — |
| google/gemini-3.5-flash | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 388039 | length:10, stop:15 | — |
| google/gemini-3.5-flash | sanity | conflict_v1 | 4 | effort=minimal | 0.000 | 0 | 0 | 0 | stop:30 | — |
| google/gemini-3.5-flash | sanity | recall_copy_v1 | 6 | effort=minimal | 0.000 | 0 | 0 | 0 | stop:30 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=minimal | 0.040 | 0 | 0 | 0 | length:6, stop:94 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=minimal | 0.000 | 0 | 0 | 0 | stop:100 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 16 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | stop:100 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v1 | 64 | contract, effort=minimal | 0.030 | 0 | 0 | 0 | length:3, stop:97 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=minimal | 0.030 | 0 | 0 | 0 | length:3, stop:97 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=minimal | 0.000 | 0 | 0 | 0 | stop:100 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=minimal | 0.000 | 0 | 0 | 0 | stop:100 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 16 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | length:1, stop:99 | — |
| google/gemini-3.5-flash | zero_budget | composite_copy_v2 | 64 | contract, effort=minimal | 0.000 | 0 | 0 | 0 | stop:100 | — |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 4 | effort=default | 0.000 | 0 | 0 | 0 | stop:30 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 16 | effort=default | 0.000 | 0 | 0 | 0 | stop:30 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 64 | effort=default | 0.000 | 0 | 0 | 0 | stop:30 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 128 | effort=default | 0.000 | 0 | 0 | 0 | stop:30 | — |
| meta-llama/llama-4-maverick | composite_length | composite_copy_v1 | 512 | effort=default | 0.000 | 0 | 0 | 0 | stop:30 | — |
| meta-llama/llama-4-maverick | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=default | 0.000 | 0 | 0 | 0 | stop:50 | — |
| meta-llama/llama-4-maverick | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=default | 0.000 | 0 | 0 | 0 | stop:50 | — |
| meta-llama/llama-4-maverick | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=default | 0.000 | 0 | 0 | 0 | stop:50 | — |
| meta-llama/llama-4-maverick | dose_response | composite_copy_v1 | 16 | effort=default | 0.000 | 0 | 0 | 0 | stop:50 | — |
| meta-llama/llama-4-maverick | floor | s5 | 16 | rendering=abstract_stated, effort=default | 0.000 | 0 | 0 | 0 | stop:30 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 16 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | stop:25 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 32 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | stop:25 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 64 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | stop:25 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 128 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | stop:25 | — |
| meta-llama/llama-4-maverick | s5_concrete | s5 | 256 | rendering=concrete, effort=default | 0.000 | 0 | 0 | 0 | stop:25 | — |
| meta-llama/llama-4-maverick | sanity | conflict_v1 | 4 | effort=default | 0.000 | 0 | 0 | 0 | stop:30 | — |
| meta-llama/llama-4-maverick | sanity | recall_copy_v1 | 6 | effort=default | 0.000 | 0 | 0 | 0 | stop:30 | — |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 9535 | stop:30 | — |
| moonshotai/kimi-k2.6 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 15 | stop:25 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 54422 | stop:25 | — |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 32 | effort=high | 0.040 | 0 | 0 | 213892 | length:1, stop:24 | — |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 64 | effort=high | 0.080 | 0 | 0 | 334007 | length:2, stop:23 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | chain_nowrap | chain_v1 | 128 | effort=high | 0.160 | 0 | 1 | 924878 | error:1, length:4, stop:20 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 258261 | stop:50 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 69598 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 1 | length:7, stop:23 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 88013 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 2 | length:4, stop:26 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 102138 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 2 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 141034 | stop:30 | — |
| moonshotai/kimi-k2.6 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 4 | stop:30 | — |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 15 | length:12, stop:38 | — |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 21 | length:7, stop:43 | — |
| moonshotai/kimi-k2.6 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 23 | stop:50 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 114566 | stop:50 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 69452 | stop:50 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 81842 | stop:50 | — |
| moonshotai/kimi-k2.6 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 5 | length:14, stop:36 | — |
| moonshotai/kimi-k2.6 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 13 | stop:30 | — |
| moonshotai/kimi-k2.6 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.620 | 0 | 0 | 15 | length:13, stop:19 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.100 | 0 | 0 | 35 | length:5, stop:45 | escalated @512tok diagnostic 0.86; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 19 | stop:50 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 97965 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 171577 | stop:24 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 247166 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 421554 | stop:25 | — |
| moonshotai/kimi-k2.6 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.080 | 0 | 0 | 636870 | length:2, stop:23 | — |
| moonshotai/kimi-k2.6 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 12 | stop:30 | — |
| moonshotai/kimi-k2.6 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 13 | stop:30 | — |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.030 | 0 | 0 | 78 | length:3, stop:97 | escalated @512tok diagnostic 0.92; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.040 | 0 | 0 | 52 | length:6, stop:94 | escalated @512tok diagnostic 0.82; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.080 | 0 | 0 | 50 | length:9, stop:91 | escalated @512tok diagnostic 0.83; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 58 | stop:100 | escalated @512tok diagnostic 0.96; canonical = first attempt @96tok |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.020 | 0 | 0 | 89 | stop:98 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.010 | 0 | 0 | 80 | stop:100 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.020 | 0 | 0 | 40 | stop:100 | — |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.080 | 0 | 0 | 65 | length:1, stop:92 | ‡ cap-escape |
| moonshotai/kimi-k2.6 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.030 | 0 | 0 | 81 | stop:98 | ‡ cap-escape |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 36177 | stop:25 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 74957 | stop:25 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 191568 | stop:25 | — |
| muse-spark-1.1 | chain_nowrap | chain_v1 | 128 | effort=high | 0.080 | 0 | 0 | 449036 | incomplete:2, stop:23 | — |
| muse-spark-1.1 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 114743 | stop:25 | — |
| muse-spark-1.1 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 242298 | stop:25 | — |
| muse-spark-1.1 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 467720 | stop:25 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 5435 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:25 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 16 | effort=high | 0.160 | 0 | 0 | 83841 | length:3, stop:22 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 32 | effort=high | 0.120 | 0 | 0 | 162217 | length:3, stop:22 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 64 | effort=high | 0.840 | 0 | 0 | 327413 | length:20, stop:5 | — |
| nvidia/nemotron-3-ultra-550b-a55b | chain_nowrap | chain_v1 | 128 | effort=high | 0.720 | 0 | 2 | 504794 | error:2, length:16, stop:7 | — |
| nvidia/nemotron-3-ultra-550b-a55b | commutative | commutative_v1 | 64 | effort=high | 0.160 | 0 | 0 | 99780 | length:4, stop:21 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 16 | effort=high | 0.300 | 0 | 0 | 5645 | stop:21 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 64 | effort=high | 0.400 | 0 | 0 | 36358 | length:2, stop:18 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 128 | effort=high | 0.833 | 0 | 1 | 16761 | error:1, length:1, stop:5 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 512 | effort=high | 0.767 | 0 | 0 | 79903 | length:6, stop:7 | — |
| nvidia/nemotron-3-ultra-550b-a55b | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=high | 0.140 | 0 | 0 | 12977 | stop:43 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=low | 0.460 | 0 | 0 | 21082 | length:1, stop:27 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=medium | 0.300 | 0 | 0 | 12936 | stop:35 | — |
| nvidia/nemotron-3-ultra-550b-a55b | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.560 | 0 | 0 | 27679 | stop:11 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.440 | 0 | 0 | 77065 | length:1, stop:14 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.600 | 0 | 0 | 136506 | length:10, stop:9 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.320 | 0 | 0 | 297749 | length:5, stop:17 | — |
| nvidia/nemotron-3-ultra-550b-a55b | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.960 | 0 | 0 | 344064 | length:22 | — |
| nvidia/nemotron-3-ultra-550b-a55b | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.050 | 0 | 0 | 0 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.090 | 0 | 0 | 0 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.010 | 0 | 0 | 0 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.040 | 0 | 0 | 0 | stop:100 | — |
| nvidia/nemotron-3-ultra-550b-a55b | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.140 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.4 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 3229 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 18908 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 21230 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 25692 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 30627 | stop:30 | — |
| openai/gpt-5.4 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.4 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.4 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.4 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.4 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 28169 | stop:50 | — |
| openai/gpt-5.4 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.4 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 24474 | stop:25 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 42463 | stop:25 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 75328 | stop:25 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 126972 | stop:25 | — |
| openai/gpt-5.4 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.040 | 0 | 1 | 203651 | error:1, stop:24 | — |
| openai/gpt-5.4 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.4 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.5 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 2686 | stop:30 | — |
| openai/gpt-5.5 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:25 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 9114 | stop:25 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 21206 | stop:25 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 69752 | stop:25 | — |
| openai/gpt-5.5 | chain_nowrap | chain_v1 | 128 | effort=high | 0.040 | 0 | 0 | 226834 | length:1, stop:24 | — |
| openai/gpt-5.5 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 10673 | stop:25 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 8213 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 9903 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 13011 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 10373 | stop:30 | — |
| openai/gpt-5.5 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.5 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.5 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.5 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.5 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 12798 | stop:50 | — |
| openai/gpt-5.5 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.5 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.5 | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.5 | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.5 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 23664 | stop:25 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 44799 | stop:25 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 107196 | stop:25 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 174517 | stop:25 | — |
| openai/gpt-5.5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 320457 | length:1, stop:24 | — |
| openai/gpt-5.5 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.5 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.5 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.6-sol | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 2744 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 5621 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 25501 | stop:25 | — |
| openai/gpt-5.6-sol | chain_nowrap | chain_v1 | 128 | effort=high | 0.000 | 0 | 0 | 42235 | stop:25 | — |
| openai/gpt-5.6-sol | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 5038 | stop:25 | — |
| openai/gpt-5.6-sol | gap_stability | composite_copy_v2 | 32 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.6-sol | gap_stability | composite_copy_v2 | 32 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.6-sol | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| openai/gpt-5.6-sol | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 66218 | stop:25 | — |
| openai/gpt-5.6-sol | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.6-sol | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| openai/gpt-5.6-sol | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| qwen/qwen3.7-max | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 11331 | stop:30 | — |
| qwen/qwen3.7-max | chain_instant | chain_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 31051 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 68118 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 135259 | stop:25 | — |
| qwen/qwen3.7-max | chain_nowrap | chain_v1 | 128 | effort=high | 0.000 | 0 | 0 | 229053 | stop:25 | — |
| qwen/qwen3.7-max | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 49912 | stop:25 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 27705 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 58421 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 65182 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 80118 | stop:30 | — |
| qwen/qwen3.7-max | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| qwen/qwen3.7-max | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| qwen/qwen3.7-max | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| qwen/qwen3.7-max | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 47533 | stop:50 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 45037 | stop:50 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 44170 | stop:50 | — |
| qwen/qwen3.7-max | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| qwen/qwen3.7-max | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| qwen/qwen3.7-max | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 42853 | stop:25 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 80913 | stop:25 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 129591 | stop:25 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 197412 | stop:25 | — |
| qwen/qwen3.7-max | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 397612 | stop:25 | — |
| qwen/qwen3.7-max | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| qwen/qwen3.7-max | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.980 | 0 | 0 | 0 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| qwen/qwen3.7-max | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| x-ai/grok-4.3 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 12612 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 82868 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 136113 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 132340 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 512 | effort=high | 0.000 | 0 | 0 | 153800 | stop:30 | — |
| x-ai/grok-4.3 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| x-ai/grok-4.3 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| x-ai/grok-4.3 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| x-ai/grok-4.3 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 148305 | stop:50 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 47245 | stop:50 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 102537 | stop:50 | — |
| x-ai/grok-4.3 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| x-ai/grok-4.3 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 43136 | stop:25 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 70491 | stop:25 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 128954 | stop:25 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 222011 | stop:25 | — |
| x-ai/grok-4.3 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 368483 | stop:25 | — |
| x-ai/grok-4.3 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| x-ai/grok-4.3 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 0 | stop:30 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 28854 | stop:25 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v1 | 32 | effort=high | 0.000 | 0 | 0 | 56099 | stop:25 | — |
| x-ai/grok-4.5 | chain_nowrap | chain_v1 | 64 | effort=high | 0.000 | 0 | 0 | 290597 | stop:25 | ‡ cap-escape |
| x-ai/grok-4.5 | commutative | commutative_v1 | 64 | effort=high | 0.000 | 0 | 0 | 41534 | stop:25 | — |
| x-ai/grok-4.5 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 201685 | stop:25 | — |
| x-ai/grok-4.5 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 376142 | stop:25 | ‡ cap-escape |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 45652 | stop:25 | — |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 32 | effort=high | 0.080 | 0 | 1 | 369760 | error:1, length:1, stop:23 | ‡ cap-escape |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 64 | effort=high | 0.160 | 0 | 3 | 535933 | error:3, length:1, stop:21 | ‡ cap-escape |
| x-ai/grok-build-0.1 | chain_nowrap | chain_v1 | 128 | effort=high | 1.000 | 0 | 7 | 4443830 | error:7, length:17, stop:1 | ‡ cap-escape |
| x-ai/grok-build-0.1 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 248308 | stop:25 | — |
| x-ai/grok-build-0.1 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 506163 | stop:25 | ‡ cap-escape |
| x-ai/grok-build-0.1 | sanity | conflict_v1 | 4 | effort=minimal | 0.000 | 0 | 0 | 15624 | stop:30 | — |
| x-ai/grok-build-0.1 | sanity | recall_copy_v1 | 6 | effort=minimal | 0.000 | 0 | 0 | 17408 | stop:30 | — |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 4 | effort=high | 0.000 | 0 | 0 | 3554 | stop:30 | — |
| z-ai/glm-5.2 | chain_instant | chain_v1 | 16 | contract, effort=none | 0.040 | 0 | 0 | 96 | length:1, stop:24 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 16 | effort=high | 0.000 | 0 | 0 | 10245 | stop:25 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 32 | effort=high | 0.040 | 0 | 0 | 36687 | length:1, stop:24 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 64 | effort=high | 0.080 | 0 | 0 | 65558 | length:2, stop:23 | — |
| z-ai/glm-5.2 | chain_nowrap | chain_v1 | 128 | effort=high | 0.200 | 0 | 0 | 130971 | length:5, stop:20 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 16 | effort=high | 0.000 | 0 | 0 | 9631 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 791 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 64 | effort=high | 0.000 | 0 | 0 | 12782 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 64 | effort=none | 0.000 | 0 | 0 | 533 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 128 | effort=high | 0.000 | 0 | 0 | 17258 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 128 | effort=none | 0.000 | 0 | 0 | 510 | stop:30 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 512 | effort=high | 0.033 | 0 | 0 | 49122 | length:1, stop:29 | — |
| z-ai/glm-5.2 | composite_length | composite_copy_v1 | 512 | effort=none | 0.000 | 0 | 0 | 383 | stop:30 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=binding_only, effort=none | 0.000 | 0 | 0 | 1083 | stop:50 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=end_to_end, effort=none | 0.000 | 0 | 0 | 2195 | stop:50 | — |
| z-ai/glm-5.2 | decomposition | composite_copy_v1 | 16 | leg=scaffolded, effort=none | 0.000 | 0 | 0 | 0 | stop:50 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=high | 0.020 | 0 | 0 | 28410 | length:1, stop:49 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=low | 0.000 | 0 | 0 | 17587 | stop:50 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=medium | 0.000 | 0 | 0 | 23551 | stop:50 | — |
| z-ai/glm-5.2 | dose_response | composite_copy_v1 | 16 | effort=none | 0.000 | 0 | 0 | 1432 | stop:50 | — |
| z-ai/glm-5.2 | floor | s5 | 16 | rendering=abstract_stated, effort=none | 0.000 | 0 | 0 | 2621 | stop:30 | — |
| z-ai/glm-5.2 | recall_load | recall_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 36 | stop:50 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 16 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 19329 | stop:25 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 32 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 33990 | stop:25 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 64 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 62202 | stop:25 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 128 | rendering=concrete, effort=high | 0.040 | 0 | 0 | 139869 | length:1, stop:24 | — |
| z-ai/glm-5.2 | s5_concrete | s5 | 256 | rendering=concrete, effort=high | 0.000 | 0 | 0 | 263382 | stop:25 | — |
| z-ai/glm-5.2 | sanity | conflict_v1 | 4 | effort=none | 0.000 | 0 | 0 | 336 | stop:30 | — |
| z-ai/glm-5.2 | sanity | recall_copy_v1 | 6 | effort=none | 0.000 | 0 | 0 | 57 | stop:30 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | leg=binding_only, contract, effort=none | 0.010 | 0 | 0 | 96 | length:1, stop:99 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | leg=end_to_end, contract, effort=none | 0.030 | 0 | 0 | 288 | length:3, stop:97 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 16 | contract, effort=none | 0.040 | 0 | 0 | 384 | length:4, stop:96 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v1 | 64 | contract, effort=none | 0.000 | 0 | 0 | 510 | stop:100 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=binding_only, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=replicate, contract, effort=none | 0.000 | 0 | 0 | 0 | stop:100 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | leg=scaffolded, contract, effort=none | 0.000 | 0 | 0 | 85 | stop:100 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 16 | contract, effort=none | 0.030 | 0 | 0 | 288 | length:3, stop:97 | — |
| z-ai/glm-5.2 | zero_budget | composite_copy_v2 | 64 | contract, effort=none | 0.020 | 0 | 0 | 192 | length:2, stop:98 | — |

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
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 24 | effort=high | 30 | 0.97 [0.83, 0.99] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.67 [0.49, 0.81] | 0.67 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| anthropic/claude-sonnet-5 | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.03 [0.01, 0.17] | 0.03 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 24 | effort=high | 30 | 0.93 [0.79, 0.98] | 0.93 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.50 [0.33, 0.67] | 0.50 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.83 [0.66, 0.93] | 0.83 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| deepseek/deepseek-v4-pro | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.50 [0.33, 0.67] | 0.50 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
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
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 32 | effort=default | 30 | 0.10 [0.03, 0.26] | 0.23 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 48 | effort=default | 30 | 0.03 [0.01, 0.17] | 0.27 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| meta-llama/llama-4-maverick | chain_depth | chain_v1 | 64 | effort=default | 30 | 0.03 [0.01, 0.17] | 0.13 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 16 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 24 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.97 [0.83, 0.99] | 0.97 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| moonshotai/kimi-k2.6 | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.90 [0.74, 0.97] | 0.90 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 8 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 12 | effort=high | 30 | 1.00 [0.89, 1.00] | 1.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 16 | effort=high | 30 | 0.77 [0.59, 0.88] | 0.77 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 24 | effort=high | 30 | 0.00 [0.00, 0.11] | 0.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.00 [0.00, 0.11] | 0.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.00 [0.00, 0.11] | 0.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| nvidia/nemotron-3-ultra-550b-a55b | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.00 [0.00, 0.11] | 0.00 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
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
| z-ai/glm-5.2 | chain_depth | chain_v1 | 32 | effort=high | 30 | 0.10 [0.03, 0.26] | 0.10 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 48 | effort=high | 30 | 0.20 [0.10, 0.37] | 0.20 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
| z-ai/glm-5.2 | chain_depth | chain_v1 | 64 | effort=high | 30 | 0.27 [0.14, 0.44] | 0.27 | INVALID (k=6 cycle wrap — task redesigned as chain_nowrap) |
