# FactWorld frontier benchmark

FactWorld's recurring benchmark is a composition instrument. Recall and last-write-wins state
tracking are the component abilities; the question each model answers is whether they compose —
a two-hop query (resolve the current holder of an object, then recall that holder's fact) scored
against the same components measured in isolation. Every model runs two regimes: **instant**
(reasoning off, hard one-line answer contract — in-weights ability, no visible working) and
**thinking** (generous reasoning budget — what the model can do when allowed to work). One
instrument serves both frontier API models and local from-scratch architectures: same tasks, same
legs, same floors; the difficulty settings (length L, pool breadth, chain depth) are calibration
parameters chosen to place each model class mid-scale, never axes of the headline.

Every score is relaxed match, the canonical metric (Wilson 95% intervals per cell are in the
rendered tables). Marks, in plain language: `*` the model cannot disable reasoning (its off-arm
ran effort=minimal); `†` visible working appeared on the canonical instant attempt (the number is
not purely in-weights); `‡` the provider ignored the token cap; `⊘` the cell's calls were
majority finish=length — not measurable at this budget; `(x.xx @512)` a single escalated-budget
rerun published as a diagnostic (the canonical number is always the first attempt at the shared
budget).

## Components

**Recall (sanity).** Copy one fact out of an in-context map (`recall_copy_v1` @L6) and override a
memorizable map with an in-context value (`conflict_v1` @L4); reasoning off, n=30 each. Positive
controls: every current-roster model scores 0.97–1.00 (sonnet-5's 0.97 recall is the only cell
off 1.00). Any model below ~1.0 here would flag a harness problem, not a capability difference.

**State tracking (binding leg).** A stream of give events reassigns objects to holders; the model
reports the *current* holder — last write wins. The cell is the `binding_only` leg of the
zero-budget battery (`composite_copy_v2` items, reasoning off, one-line contract, 96-token cap,
n=100): state tracking isolated from recall.

**Floors.** Two first-class rows. The *recency heuristic* (answer the last event's recipient)
scores 0.04 — chance, because the v2 sampler places the queried object's last write uniformly
over the stream. The *object-filter floor* is E[1/w]: a reader that filters events to the queried
object but picks a random write scores 0.41 @L16, decaying ~1/L to 0.15 @L64, with no last-write
resolution at all. Instant cells are read against the object-filter floor, not chance: a score
near it shows object filtering; genuine state tracking has to clear it.

| Model | binding_only @L16 |
|---|---|
| anthropic/claude-opus-4.8 | 0.78 |
| anthropic/claude-sonnet-5 | 0.77 |
| deepseek/deepseek-v4-pro | 0.51 |
| google/gemini-3.5-flash | 0.66* |
| moonshotai/kimi-k2.6 | 0.94† |
| nvidia/nemotron-3-ultra-550b-a55b | 0.49 |
| openai/gpt-5.5 | 0.80 |
| qwen/qwen3.7-max | 0.51 |
| z-ai/glm-5.2 | 0.70† |
| *recency heuristic (floor)* | 0.04 |
| *object-filter floor* | 0.41 |

Opus (0.78), sonnet (0.77), gpt-5.5 (0.80), kimi (0.94†), and glm (0.70†) clear the floor
decisively; deepseek and qwen (0.51) and nemotron (0.49) sit within noise of it — object
filtering, not established state tracking.

## Composition

The core statistic. The composed cell is the two-hop query in one shot ("what is a0 of the holder
of o0 ?", answered with the holder–value pair), under the same zero-budget protocol (reasoning
off, contract, 96-token cap, n=100) at L16 and L64. The **composition gap** is
binding_only@L16 − composed@L16.

The gap isolates composition because the recall half is free: on the scaffolded leg (E1b) — the
same query with the resolved holder provided — recall reads 0.98–1.00 for every model measured
(1.00 for eight of the nine roster models; deepseek 0.98). If composing were free, the composed
cell would match the binding leg; the gap is the composition deficit.

| Model | recall | binding @L16 | composed @L16 | composed @L64 | gap @L16 |
|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 0.78 | 0.72 | 0.43 | +0.06 |
| anthropic/claude-sonnet-5 | 0.97 | 0.77 | 0.62 (0.76 @512)† | 0.32 (0.66 @512)† | +0.15† |
| deepseek/deepseek-v4-pro | 1.00 | 0.51 | 0.44 | 0.19 | +0.07 |
| google/gemini-3.5-flash | 1.00 | 0.66* | 0.64* | 0.28* | +0.02* |
| moonshotai/kimi-k2.6 | 1.00 | 0.94† | 0.77† | 0.93† | +0.17† |
| nvidia/nemotron-3-ultra-550b-a55b | 1.00 | 0.49 | 0.33 | 0.12 | +0.16 |
| openai/gpt-5.5 | 1.00 | 0.80 | 0.46 | 0.33 | +0.34 |
| qwen/qwen3.7-max | 1.00 | 0.51 | 0.24 | 0.08 | +0.27 |
| z-ai/glm-5.2 | 1.00 | 0.70† | 0.35† | 0.16 | +0.35† |
| *recency heuristic (floor)* | — | 0.04 | 0.04 | 0.06 | — |
| *object-filter floor* | — | 0.41 | 0.41 | 0.15 | — |

The run-to-run noise bar on every instant number is 0.06: the battery carries a `replicate` leg
(prompts identical to the plain @L16 cell), and the maximum observed |plain − replicate| across
models is 0.06.

**Reading the gap.** It is interpretable only where the binding component is established. For
deepseek, qwen, and nemotron the binding leg itself sits at the 0.41 floor, so their composed
cells are floor-shaped and the gap is not a composition measurement. Where binding is solid, the
roster separates: opus composes essentially for free (gap 0.06, equal to the noise bar) and gemini
flash's 0.02 is the same shape under its `*` caveat; sonnet pays 0.15† and kimi 0.17†; gpt-5.5 pays
the largest clean deficit on the roster — binding 0.80, composed 0.46, gap 0.34 — and glm the
largest overall (0.35†). At L64 the floor decays to 0.15 and the same ordering holds: opus 0.43,
gpt-5.5 0.33, sonnet 0.32 clear it; kimi's 0.93† is the covert-working outlier.

**Marks on this table, in plain language.** Sonnet's cells were the battery's only budget
escalations (majority finish=length at 96 tokens); the canonical numbers are the first attempts,
the `@512` values single-rerun diagnostics. Kimi emitted reasoning tokens on 65–89% of zero-budget
calls despite effort=none, and its provider does not enforce the token cap, so its instant numbers
overstate in-weights ability by an unknown margin. Glm showed short visible working on its marked
cells. Gemini-flash cannot disable reasoning; its off-arm ran effort=minimal throughout.

## Composition under reasoning

With reasoning on, the composed cell reads at or near ceiling at canonical settings across this
roster — 0.84–1.00 on the effort=high arm of the v1 dose-response cell, and the calibration
probes hold glm at 0.92–1.00 on v2 out to L1024 at k=32 (`results/v3_probes/`,
`results/composite_frontier_20260709.jsonl`). That is a calibration fact about the settings, so
the thinking regime is read through two state-stress rows, reported as plain scores at named
settings, plus a practical efficiency column:

- **chain d128 @ k=257** — a pointer chase 128 hops deep over 257 agents (the `chain_nowrap`
  staircase builds k=2d+1, so every d128 cell is the same fixed-breadth k=257 cell across the
  roster); effort=high, 16,384 tokens, n=25. Chance is under 0.01.
- **s5 @L256** — non-abelian state: 256 swap/cycle events permute five people over five jobs,
  concrete rendering, effort=high, 16,384 tokens, n=25. Chance is 0.20; the roster's
  reasoning-off floor arm on the abstract rendering scores 0.00–0.30.
- **s5@128 ctok** — completion tokens per call on the matched s5 L128 cell, which every roster
  model ran (it replaces a per-solve average that rewarded early failure).

| Model | chain d128 (k=257) | s5 @L256 | s5@128 ctok |
|---|---|---|---|
| anthropic/claude-opus-4.8 | 0.08 | ⊘ | 12683 |
| anthropic/claude-sonnet-5 | 0.04 | ⊘ | 11866 |
| deepseek/deepseek-v4-pro | ⊘ ‡ | ⊘ | 10043 |
| google/gemini-3.5-flash | 0.88 | 0.52 | 11022 |
| moonshotai/kimi-k2.6 | 0.64‡ | 0.88 | 17418 |
| nvidia/nemotron-3-ultra-550b-a55b | ⊘ | ⊘ | 12250 |
| openai/gpt-5.5 | 0.36 | 0.96 | 6989 |
| qwen/qwen3.7-max | 0.96 | 0.80 | 7904 |
| z-ai/glm-5.2 | 0.36 | 0.88 | 5980 |

`⊘` cells are not measurable at this budget: majority finish=length, so the score reflects the
16,384-token budget rather than the ability (opus and sonnet emit no visible answer on all 25 s5
L256 calls; deepseek hits length on 22/25 chain and 18/25 s5 calls; nemotron on 13/25 and 22/25 —
nemotron also stays below every other model at every measured length, so budget is not its only
problem). `‡` cap-escape: deepseek (chain d128) and kimi (chain d64/d128) exceeded the token cap
on >10% of calls, so those token spends are not cap-comparable.

The scores discriminate where the composed cell cannot: qwen (0.96 chain, 0.80 s5) and
gemini-flash (0.88, 0.52) hold deep state under reasoning that they cannot hold in weights, while
opus and sonnet — the strongest clean instant composers — post the weakest measurable chain
scores (0.08, 0.04). The efficiency column is the practical note: token-hungry state tracking is
rented, not owned. Gpt-5.5 holds 0.96 at s5 L256 while spending 6,989 ctok on the matched L128
cell — 2.5x less than kimi's 17,418 for a similar score; glm is the cheapest on the roster at
5,980.

## The local regime

The same instrument evaluates architectural changes in from-scratch models: same tasks, same
legs, same floors, same relaxed metric. The calibration parameters move to place d256-class
models mid-scale; the statistic — components versus the composed cell — does not.

Local smoke evidence (RTX 5090, gdp_hybrid d256×4, 4,000 steps;
`results/local_smoke_20260709/`): the binding leg reads 0.82 @L16 / 0.21 @L32 / 0.23 @L64 on the
v2 sampler, against 0.99 / 0.77 / 0.70 on v1 — the v1 sampler grants recency credit locally just
as it does over the API, and the v2 floors apply unchanged (object-filter 0.41 @L16 / 0.15 @L64
at m=4).

The operating-point sweep mirrors the frontier breadth rungs on from-scratch models: transformer
vs gdp_hybrid, d256×4, 8k steps flat next-token training (train L ∈ {4, 8, 16}), rungs
B ∈ {6, 8, 12, 16, 24} via `composite_copy_v2.scaled(k=2B, recall_pool=B)`, m=4, 3 seeds per
cell, RTX 5090 (30 runs; `results/local_breadth/sweep_summary.md`). L16 is in-distribution, L64
is length extrapolation; pconv = fraction of seeds ≥0.9 (composite convergence is bimodal — read
pconv, not the mean).

| B | arch | composed @L16 (pconv) | composed @L64 | binding / recall leg @L16 | @L64 |
|---|---|---|---|---|---|
| 6 | gdp_hybrid | 0.04±0.02 (0%) | 0.00 | 0.56 / 0.06 | 0.09 / 0.01 |
| 6 | transformer | 0.02±0.00 (0%) | 0.01 | 0.23 / 0.07 | 0.18 / 0.03 |
| 8 | gdp_hybrid | 0.01±0.01 (0%) | 0.00 | 0.41 / 0.02 | 0.14 / 0.00 |
| 8 | transformer | 0.00±0.00 (0%) | 0.00 | 0.17 / 0.01 | 0.15 / 0.01 |
| 12 | gdp_hybrid | 0.00±0.00 (0%) | 0.00 | 0.43 / 0.01 | 0.19 / 0.01 |
| 12 | transformer | 0.00±0.00 (0%) | 0.00 | 0.15 / 0.01 | 0.10 / 0.00 |
| 16 | gdp_hybrid | 0.00±0.00 (0%) | 0.00 | 0.41 / 0.01 | 0.16 / 0.01 |
| 16 | transformer | 0.00±0.00 (0%) | 0.00 | 0.13 / 0.02 | 0.08 / 0.00 |
| 24 | gdp_hybrid | 0.01±0.01 (0%) | 0.00 | 0.67 / 0.01 | 0.20 / 0.01 |
| 24 | transformer | 0.00±0.00 (0%) | 0.00 | 0.08 / 0.01 | 0.03 / 0.00 |

- **The composed cell reads floor for this model class under flat training** — best single run
  0.06 @L16 (gdp_hybrid, B6), p(converge) 0/30. At d256 the instrument reads through the legs;
  converging the composed cell locally takes the staged-curriculum recipe
  ([consolidated §5](factworld-consolidated.md), d768).
- **Local operating point: B8.** The largest rung where the better architecture reads mid-scale
  seed-consistently on the binding leg: gdp_hybrid 0.41 @L16 (seeds 0.34–0.48) against a 1/k =
  0.06 agent-guess. From B12 up the leg is bimodal — single seeds solve binding outright (1.00
  @B12, 0.99 @B16, 0.98/1.00 @B24) while the rest sit near floor — so per-rung means stop being
  readable and seed counting takes over.
- **The architecture comparison reads on the binding leg.** gdp_hybrid over transformer at every
  rung: 0.56 vs 0.23 @B6, 0.41 vs 0.17 @B8. The transformer decays monotonically to 0.08 @B24
  and no longer fits the training distribution there (final loss 1.49–1.65, against gdp_hybrid's
  0.05–0.14 everywhere). Binding does not extrapolate reliably to L64 for either architecture
  (gdp_hybrid 0.56 → 0.09 @B6; the binding-solved seeds keep 0.42–0.52 except one at 0.08).
- **The local composition deficit sits on the recall leg.** The value leg is ≤0.11 in all 30
  runs — including the seeds that solve binding outright (binding 0.98–1.00 @L16, value ≤0.02):
  the resolved holder is not routed into the lookup. This is the same leg the d768
  staged-curriculum decomposition localizes (gdp_hybrid binding 0.97 / value 0.75; wherever an
  architecture fails, the binding leg holds and the value leg collapses).

For the v1-family re-measure ([#11](https://github.com/ianbarber/factworld/issues/11)):
`composite_copy_v2.scaled(k=16, recall_pool=8)` is the d256 calibration cell, and local
composed-cell numbers keep the staged-curriculum recipe with p(converge) over ≥3 seeds as the
statistic.

## Protocol appendix

**Roster and pinning.** Nine models via OpenRouter, registered in `factworld.benchmark.MODELS`
(slug, tier, per-million pricing, capability flags); models removed from the roster render in the
archived section of results.md, and their cells stay in history. x-ai is unrepresented: no
current xAI endpoint is cleanly measurable on this suite — mainline grok's endpoint safety filter
blocks a majority of the composite prompts as apparent gene/variant nomenclature, and grok-build
is served with reasoning pinned at ~256k tokens regardless of the requested cap (its one measured
cycle sits in the archived section). Task items are deterministic from fixed seeds; each spec
carries a pinned stream version, so existing cells resume byte-identically and only genuinely new
cells run.

**Facets and budgets.** Instant: `zero_budget` (composite_copy_v2; composed @L16 and @L64,
binding_only @L16, replicate @L16; n=100; effort=none; 96-token cap; hard "Answer:" contract line
with last-line extraction) and `sanity` (recall_copy_v1 @L6, conflict_v1 @L4, n=30). Thinking:
`chain_nowrap` (k=2d+1 staircase; the reported cell is d128, k=257; n=25; 16,384 tokens) and
`s5_concrete` (the reported cell is L256; n=25; 16,384 tokens; effort=high throughout).

**Contracts and escalation.** Per-cell diagnostics gate publication: contract adherence,
covert-CoT rate, reasoning-token rates, finish errors, and API errors. A zero-budget cell whose
first attempt is majority finish=length is rerun once at a 512-token budget; the canonical value
is always the first attempt, and the escalated value publishes only as the `(x.xx @512)`
diagnostic. In the current v2 battery: 3 escalations (all sonnet-5), 0 API errors, 0 finish
errors, contract adherence 0.86–1.00.

**Marks glossary.** `*` off-arm ran effort=minimal (reasoning cannot be disabled). `†` visible
working on the canonical instant attempt. `‡` cap-escape (provider ignored `max_new_tokens`).
`⊘` majority finish=length — not measurable at this budget. `(x.xx @512)` escalated-budget
diagnostic. `n/a` cell not run; `—` run, no qualifying value.

**Cost.** The full 502-cell history carries an estimated $235.74 of API spend; the 36-cell
zero-budget v2 battery (9 models x 4 legs, 3,600 calls) cost an estimated $5.08. Adding one model
runs from a few dollars for cheap models to a few tens of dollars for frontier pricing with long
reasoning traces.

**Adding a model.** Register the slug in `factworld.benchmark.MODELS` (tier, pricing, flags),
then:

```bash
python scripts/run_frontier_benchmark.py --models <slug> --dry-run   # plan + cost preview; everything else resume-skips
python scripts/run_frontier_benchmark.py --models <slug>             # run; appends per-cell records to history.jsonl
python scripts/render_benchmark.py                                   # re-render results.md/csv, figures, index.html
```

## Provenance

- Rendered tables, figures, and per-cell Wilson intervals:
  [`docs/benchmark/results.md`](../docs/benchmark/results.md) (with `results.csv` and
  `index.html` alongside).
- Raw per-cell records (one JSON object per cell, all attempts, usage, diagnostics):
  `results/benchmark/history.jsonl` (zero-budget battery: run `bench_v2_zb2_20260709`; chain/s5:
  `bench_v2_20260708`).
- Operating-point calibration probes: `results/v3_probes/` and
  `results/composite_frontier_20260709.jsonl` (log-only material;
  [`docs/experiments/README.md`](../docs/experiments/README.md) §12).
- Local smoke evidence: `results/local_smoke_20260709/`; breadth sweep:
  `results/local_breadth/`.
- v1-family retirement and the re-measure plan:
  [#11](https://github.com/ianbarber/factworld/issues/11).
- **Which task version each number is on.** Every instant column in this report is on
  `composite_copy_v2`, the de-skewed sampler (uniform last-write placement; the recency heuristic
  scores chance). The sanity, chain, and s5 cells are unchanged tasks with pinned streams. The
  scaffolded (E1b) leg is on the v1 composite items (the recall map is unchanged between
  samplers). The OpenRouter grid and reasoning analyses in
  [`factworld-consolidated.md`](factworld-consolidated.md) and the README, and the local
  from-scratch training numbers, are on the retired v1 family; their re-measure on v2 is tracked
  in [#11](https://github.com/ianbarber/factworld/issues/11).
