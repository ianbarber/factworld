# Frontier models can recall and track state. Combining the two is what costs them.

- **What it measures.** Two abilities every long task leans on — recalling a stated fact, and tracking state through a stream of updates — measured independently and then composed into a single two-hop question. The components are cheap for every model; the composition is where they diverge.
- **Two regimes, plain words.** *Instant* = reasoning off, one-line answer required: what the weights compute directly. *Thinking* = a generous reasoning budget: what the model can do given room to work. The two rank the roster in very different orders, so every score is regime-labelled and there is no single leaderboard number.
- **Floors and marks.** Scores are read against shallow-heuristic floors — cheats run as first-class rows — and marks flag models whose numbers need a caveat. A score only counts when the cheats can't earn it.
- **Where everything lives.** Code, tasks and the add-a-model path: [github.com/ianbarber/factworld](https://github.com/ianbarber/factworld); every number with per-cell confidence intervals: [full results](https://github.com/ianbarber/factworld/blob/main/docs/benchmark/results.md).

## Results (current roster)

| Model | State tracking (binding @L16) | Composed @L16 | Composition gap | Chain d128 (thinking) | S5 @L256 (thinking) |
|---|---|---|---|---|---|
| Claude Opus 4.8 | 0.78 | 0.72 | +0.06 | 0.08 | 1.00 |
| Claude Sonnet 5 | 0.77 | 0.62† | +0.15† | 0.04 | 1.00 |
| DeepSeek V4 Pro | 0.51 | 0.44 | +0.07 | ⊘ >budget | ⊘ >budget |
| Gemini 3.5 Flash | 0.66* | 0.64* | +0.02* | 0.88 | 0.52 |
| Kimi K2.6 | 0.94† | 0.77† | +0.17† | 0.64‡ | 0.88 |
| Nemotron 3 Ultra | 0.49 | 0.33 | +0.16 | ⊘ >budget | ⊘ >budget |
| GPT-5.5 | 0.80 | 0.46 | +0.34 | 0.36 | 0.96 |
| GPT-5.6 Sol | 0.82 | 0.65 | +0.17 | 1.00 | n/a |
| Qwen3.7 Max | 0.51 | 0.24 | +0.27 | 0.96 | 0.80 |
| Grok 4.5 | n/a | n/a | n/a | n/a | 1.00‡ |
| GLM-5.2 | 0.71 | 0.38† | +0.33† | 0.36 | 0.88 |
| *recency heuristic (floor)* | 0.04 | 0.04 | — | — | — |
| *object-filter floor* | 0.41 | 0.41 | — | — | — |

Marks: `*` the model cannot fully disable reasoning; `†` visible working leaked onto a supposedly instant attempt — read it as a soft upper bound; `‡` the provider did not enforce the token cap, so token comparisons are off; `⊘ >budget` the token budget ran out before an answer — not measurable at that budget, which is different from a zero.

*Recency heuristic (floor)*: answer with the last event's recipient — a one-line cheat with no state tracking at all.

*Object-filter floor*: filter the stream to the queried object but pick a random one of its writes — a score near this row shows filtering, not state tracking.

## Figures

Upload from `docs/benchmark/` (an SVG sits alongside each PNG):

1. `fig_zero_budget.png` — Components vs. composition with reasoning off: state tracking beside the composed two-hop cell — the annotated gap is what composing costs each model.
   *Alt text:* Grouped bar chart of the model roster with reasoning off. For each model, bars show state tracking at length 16, the composed two-hop task at lengths 16 and 64, and a test-retest replicate, with Wilson 95% error bars and the composition gap annotated.
2. `fig_profiles.png` — One profile per model: its normalized position on every axis, instant and thinking side by side — the reason this page has no single ranking.
   *Alt text:* Small-multiples panel, one chart per roster model. Horizontal bars give each model's normalized position on six axes — state tracking, composed score, composition gap (inverted), pointer-chase depth 128, S5 at length 256, and completion tokens on the matched S5 cell (inverted) — with raw values printed beside the bars; unmeasurable cells are gaps, not zeros.
3. `fig_chain_nowrap.png` — Pointer chases with reasoning on: score vs. chain depth — the instant-regime leaders are not the leaders here.
   *Alt text:* Line chart of relaxed accuracy versus pointer-chase depth, 16 to 128 on a log scale, with reasoning enabled; one line per roster model, hollow markers where the token budget ran out before an answer.
4. `fig_s5_horizon.png` — Permutation state with reasoning on: tracking five people across five jobs through up to 256 swap/cycle events.
   *Alt text:* Line chart of relaxed accuracy versus permutation stream length, 16 to 256 on a log scale, with reasoning enabled; one line per roster model, hollow markers where the token budget ran out before an answer.

---

Last updated: 2026-07-13 · 11 models · benchmark data and methodology: [github.com/ianbarber/factworld](https://github.com/ianbarber/factworld) · refresh: three commands (register the model's slug, run `scripts/run_frontier_benchmark.py`, re-render with `scripts/render_benchmark.py`).
