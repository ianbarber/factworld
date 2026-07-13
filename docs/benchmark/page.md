# Frontier models can recall and track state. Combining the two is what costs them.

- **What it measures.** Two abilities every long task leans on — recalling a stated fact, and tracking state through a stream of updates — measured independently and then composed into a single two-hop question. Recall is cheap for every model; state tracking is established for six of the nine — and for those six, composition is where they diverge.
- **One metric.** Every cell is **match**: strip a trailing period from both sides and compare the model's first len(gold) whitespace tokens to the gold answer — binary per item, no partial credit (`factworld.tasks.score_relaxed`).
- **Two regimes, plain words.** *Instant* = reasoning off, one-line answer required: what the weights compute directly. *Thinking* = a generous reasoning budget: what the model can do given room to work. The two rank the roster in very different orders, so every score is regime-labelled and there is no single leaderboard number.
- **Floors and marks.** Scores are read against shallow-heuristic floors — cheats run as first-class rows — and marks flag models whose numbers need a caveat. A score only counts when the cheats can't earn it.
- **Where everything lives.** Code, tasks and the add-a-model path: [github.com/ianbarber/factworld](https://github.com/ianbarber/factworld); every number with per-cell confidence intervals: [full results](https://github.com/ianbarber/factworld/blob/main/docs/benchmark/results.md); a worked item — one full prompt, its gold answer, and real model mistakes: [docs/tasks.md](https://github.com/ianbarber/factworld/blob/main/docs/tasks.md).

## Results (current roster)

How to read the columns: the first three are *instant* cells (reasoning off) — hold state through a 16-event stream, answer the composed two-hop question on the same stream, and the gap between them; the last two are *thinking* cells (reasoning on) — a 128-hop pointer chase and a 256-event permutation stream. `@Ln` = stream length in events or hops; `@Ntok` = a completion-token budget (raised budgets are stated with the number).

| Model | State tracking (binding @L16) | Composed @L16 | Composition gap | Chain d128 (thinking) | S5 @L256 (thinking) |
|---|---|---|---|---|---|
| Claude Opus 4.8 | 0.78 | 0.72 | +0.06 | 0.08 | 1.00 @32,768tok (raised budget) |
| Claude Sonnet 5 | 0.77 | 0.62† | +0.15† | 0.04 | 1.00 @32,768tok (raised budget) |
| DeepSeek V4 Pro | 0.51 | 0.44 | —ᶠ | ⊘ >budget @32,768tok (raised budget) | ⊘ >budget |
| Gemini 3.5 Flash | 0.66* | 0.64* | +0.02* | 0.88 | 0.52 |
| Kimi K2.6 | ≤0.94† | ≤0.77† | +0.17† | 0.64‡ | 0.88 |
| Nemotron 3 Ultra | 0.49 | 0.33 | —ᶠ | ⊘ >budget | ⊘ >budget |
| GPT-5.5 | 0.80 | 0.46 | +0.34 | 0.36 | 0.96 |
| Qwen3.7 Max | 0.51 | 0.24 | —ᶠ | 0.96 | 0.80 |
| GLM-5.2 | 0.68† | 0.34 | +0.34† | 0.36 | 0.88 |
| *recency heuristic (floor)* | 0.04 | 0.04 | — | — | — |
| *object-filter floor* | 0.41 | 0.41 | — | — | — |

Marks: `*` the model cannot fully disable reasoning; `†` visible working or covert reasoning leaked onto a supposedly instant attempt — read it as a soft upper bound; `≤x†` covert reasoning on *most* calls — the number is an explicit upper bound; `‡` the provider did not enforce the token cap, so token comparisons are off; `⊘ >budget` the token budget ran out before an answer — not measurable at that budget, which is different from a zero; `—ᶠ` the gap is not interpretable because the model's state tracking sits at the object-filter floor (floor − floor ≈ 0 by construction). ⊘ and ≤x† are the same principle from both sides: neither participates in orderings.

Kimi's composed @L64 exceeding its @L16 is the covert-reasoning artifact (reasoning tokens on most calls despite reasoning off), not a length effect.

Thinking columns: n=25 per cell; Wilson intervals ≈ ±0.15–0.19, and the one thinking test-retest pair moved 0.16 — differences under ~0.2 are not an ordering.

*Recency heuristic (floor)*: answer with the last event's recipient — a one-line cheat with no state tracking at all.

*Object-filter floor*: filter the stream to the queried object but pick a random one of its writes — a score near this row shows filtering, not state tracking.

## Figures

Four figures carry the shape of the results:

1. `fig_zero_budget.png` — Components vs. composition with reasoning off: state tracking beside the composed two-hop cell — the annotated gap is what composing costs each model.
   *Alt text:* Grouped bar chart of the current model roster with reasoning off. For each model, bars show state tracking at length 16, the composed two-hop task at lengths 16 and 64, and a test-retest replicate, with Wilson 95% error bars and the composition gap annotated (a floor marker where the gap is not interpretable). Models are ordered by the composed length-16 score; models whose instant numbers are upper bounds from covert reasoning are hatched and placed last, outside the ordering. The legend sits outside the plot area.
2. `fig_profiles.png` — One profile per model: its normalized position on every axis, instant and thinking side by side — the reason this page has no single ranking.
   *Alt text:* Small-multiples panel, one chart per current-roster model. Horizontal bars give each model's normalized position on six axes — state tracking, composed score, composition gap (inverted), pointer-chase depth 128, S5 at length 256, and completion tokens on the matched S5 cell (inverted) — with raw values printed beside the bars; unmeasurable cells are gaps, not zeros.
3. `fig_chain_nowrap.png` — Pointer chases with reasoning on: score vs. chain depth — the instant-regime leaders are not the leaders here.
   *Alt text:* Line chart of match accuracy versus pointer-chase depth, 16 to 128 on a log scale, with reasoning enabled; one line per current-roster model (legend outside the plot area), hollow markers where the token budget ran out before an answer.
4. `fig_s5_horizon.png` — Permutation state with reasoning on: tracking five people across five jobs through up to 256 swap/cycle events.
   *Alt text:* Line chart of match accuracy versus permutation stream length, 16 to 256 on a log scale, with reasoning enabled; one line per current-roster model (legend outside the plot area), hollow markers where the token budget ran out before an answer.

---

Last updated: 2026-07-13 · 9 models · data, methodology, and the add-a-model path: [github.com/ianbarber/factworld](https://github.com/ianbarber/factworld).
