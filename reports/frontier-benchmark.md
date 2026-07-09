# FactWorld frontier benchmark

FactWorld's recurring frontier benchmark measures the component mechanics that agent workloads
depend on — working-memory recall, last-write-wins state tracking, two-hop composition, deep
pointer chasing, and long-horizon permutation tracking — as a capability ladder, in two regimes.
**Instant** cells run with reasoning off under a hard one-line answer contract: they measure
in-weights ability, with no visible working. **Thinking** cells run with a generous reasoning
budget (8,192–16,384 completion tokens): they measure what the model can do when allowed to work.
Reading the tables: every score is relaxed match, the canonical metric (Wilson 95% intervals per
cell are in the rendered tables). A *chance floor* is what random guessing scores. A
*shallow-heuristic floor* is what a trivial reader scores with no state tracking at all — instant
cells must be read against it, not against chance. Horizons marked `>=N` are censored lower
bounds (the model never failed cleanly below the mark — either every tested length passed, or the
first failure was the token budget running out, marked *budget-censored*), and *(borderline)*
horizons are threshold calls the confidence interval does not resolve. Cleanliness marks flag
cells that are not what the column claims: `*` the model cannot disable reasoning (off-arm ran
effort=minimal), `†` visible working appeared on the canonical attempt (the instant number is not
purely in-weights), `‡` the provider ignored the token cap (token counts not cap-comparable), and
`(x.xx @512)` is a marked diagnostic from a single escalated-budget rerun — the canonical number
is always the first attempt at the shared base budget.

## The ladder at a glance (current roster)

| Model | instant: recall | instant: binding @L16 | instant: composite @L16 | instant: composite @L64 | instant: replicate @L16 | thinking: chain horizon | thinking: s5 horizon | thinking: s5@128 ctok |
|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 0.78 | 0.72 | 0.43 | 0.77 | 32 (borderline) | >=128 (budget-censored) | 12683 |
| anthropic/claude-sonnet-5 | 0.97 | 0.77 | 0.62 (0.76 @512)† | 0.32 (0.66 @512)† | 0.65 (0.82 @512)† | 16 (borderline) | >=128 (budget-censored) | 11866 |
| deepseek/deepseek-v4-pro | 1.00 | 0.51 | 0.44 | 0.19 | 0.44 | >=64 (budget-censored) | >=128 (budget-censored) | 10043 |
| google/gemini-3.5-flash | 1.00 | 0.66* | 0.64* | 0.28* | 0.65* | >=128 | 128 | 11022 |
| moonshotai/kimi-k2.6 | 1.00 | 0.94† | 0.77† | 0.93† | 0.83† | 64 | >=256 | 17418 |
| nvidia/nemotron-3-ultra-550b-a55b | 1.00 | 0.49 | 0.33 | 0.12 | 0.30 | — | — | 12250 |
| openai/gpt-5.5 | 1.00 | 0.80 | 0.46 | 0.33 | 0.46 | 64 | >=256 | 6989 |
| qwen/qwen3.7-max | 1.00 | 0.51 | 0.24 | 0.08 | 0.25 | >=128 | >=256 | 7904 |
| z-ai/glm-5.2 | 1.00 | 0.70† | 0.35† | 0.16 | 0.36 | 16 | >=256 | 5980 |
| *recency heuristic (floor)* | — | 0.04 | 0.04 | 0.06 | 0.04 | — | — | — |
| *object-filter floor* | — | 0.41 | 0.41 | 0.15 | 0.41 | — | — | — |

All instant columns run on `composite_copy_v2` (last write placed uniformly over the stream, so
recency copying scores chance); `n/a` = cell not run for the model; `—` = run, but no qualifying
value. Archived models (dropped from the roster) render in a separate section of
[`docs/benchmark/results.md`](../docs/benchmark/results.md).

## Rung 1 — Recall (sanity)

**What it tests.** Copy one fact out of an in-context map (`recall_copy_v1`: "g7 has a0 v56 . g5
has a0 v18 . ... what is a0 of g7 ?") and override a memorizable map with an in-context value
(`conflict_v1`). These are positive controls: reasoning off, n=30 each.

**What good looks like.** Ceiling. Chance is a random pick from the fact pool; any current-roster
model below ~1.0 would flag a harness problem, not a capability difference.

| Model | recall_copy_v1 @L6 | conflict_v1 @L4 |
|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 1.00 |
| anthropic/claude-sonnet-5 | 0.97 | 1.00 |
| deepseek/deepseek-v4-pro | 1.00 | 1.00 |
| google/gemini-3.5-flash | 1.00 | 1.00 |
| moonshotai/kimi-k2.6 | 1.00 | 1.00 |
| nvidia/nemotron-3-ultra-550b-a55b | 1.00 | 1.00 |
| openai/gpt-5.5 | 1.00 | 1.00 |
| qwen/qwen3.7-max | 1.00 | 1.00 |
| z-ai/glm-5.2 | 1.00 | 1.00 |

**Caveats.** None; the rung is saturated by design.

## Rung 2 — State tracking (binding, instant)

**What it tests.** A stream of give events reassigns objects to holders ("s0 : o3 is given to
g0 . s1 : give o0 to g0 . ... what is the holder of o0 ?"); the model must report the *current*
holder — last write wins. The cell is the `binding_only` leg of the zero-budget battery
(`composite_copy_v2` items, reasoning off, one-line answer contract, 96-token cap, n=100), so it
isolates state tracking from recall.

**What good looks like.** The one-line recency heuristic (answer the last event's recipient)
scores 0.04 — chance level, because the v2 sampler places the queried object's last write
uniformly over the stream. The meaningful floor is the *object-filter floor* at 0.41: a reader
that filters events to the queried object but picks a random write scores E[1/w] with no
last-write resolution at all. A score near 0.41 shows object filtering; genuine state tracking
has to clear it.

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

The roster splits: opus (0.78), gpt-5.5 (0.80), kimi (0.94†), and glm (0.70†) clear the floor
decisively; deepseek, qwen (0.51), and nemotron (0.49) sit within noise of it — consistent with
shallow object filtering rather than state tracking.

**Caveats.** Kimi's and glm's daggers mean visible working appeared on the canonical attempt, so
their instant numbers are not purely in-weights (kimi emitted reasoning tokens on 65–89% of
zero-budget calls despite effort=none). Gemini-flash cannot disable reasoning; its off-arm ran
effort=minimal (`*`). Grok-build's endpoint makes reasoning mandatory and its "minimal" is not
minimal, so its instant cells are structurally skipped (`n/a`).

## Rung 3 — Composition (composite, instant)

**What it tests.** Two hops in one shot: resolve the current holder of an object (binding), then
recall that holder's fact from the in-context map ("g17 's a0 is v26 . ... s15 : give o0 to
g30 . what is a0 of the holder of o0 ?"), answering with the holder–value pair. Same zero-budget
protocol: reasoning off, contract, 96-token cap, n=100, at L16 and L64. The `replicate` column is
a test-retest duplicate of the plain L16 cell — the prompts are intentionally identical — and the
maximum observed |plain − replicate| across models is 0.06, the run-to-run noise bar on every
instant number.

**What good looks like.** Chance on the full pair is ~0.03 (k² pairs). The recency heuristic
scores 0.04 @L16 / 0.06 @L64 — chance. The object-filter floor is 0.41 @L16 and decays ~1/L to
0.15 @L64, so small-L cells must be read against it: last-write-wins leaves an irreducible
shallow floor that only genuine binding clears.

| Model | composite @L16 | composite @L64 | replicate @L16 |
|---|---|---|---|
| anthropic/claude-opus-4.8 | 0.72 | 0.43 | 0.77 |
| anthropic/claude-sonnet-5 | 0.62 (0.76 @512)† | 0.32 (0.66 @512)† | 0.65 (0.82 @512)† |
| deepseek/deepseek-v4-pro | 0.44 | 0.19 | 0.44 |
| google/gemini-3.5-flash | 0.64* | 0.28* | 0.65* |
| moonshotai/kimi-k2.6 | 0.77† | 0.93† | 0.83† |
| nvidia/nemotron-3-ultra-550b-a55b | 0.33 | 0.12 | 0.30 |
| openai/gpt-5.5 | 0.46 | 0.33 | 0.46 |
| qwen/qwen3.7-max | 0.24 | 0.08 | 0.25 |
| z-ai/glm-5.2 | 0.35† | 0.16 | 0.36 |
| *recency heuristic (floor)* | 0.04 | 0.06 | 0.04 |
| *object-filter floor* | 0.41 | 0.15 | 0.41 |

Two tiers, at both lengths. The frontier tier — opus (0.72/0.43), sonnet (0.62/0.32 canonical),
gpt-5.5 (0.46/0.33), kimi (0.77/0.93, with the covert-working caveat) — clears the object-filter
floor: genuine last-write resolution composed with recall, in weights. The cheap tier — qwen
(0.24/0.08), nemotron (0.33/0.12), glm (0.35/0.16), deepseek (0.44/0.19) — sits at or below the
floor at both lengths: object filtering, not composition.

**Caveats.** Sonnet's cells were the batch's only budget escalations (majority finish=length at
96 tokens); the canonical number is the first attempt, the `@512` value a single-rerun
diagnostic. Kimi's daggers cover both covert reasoning at effort=none and cap-escape (the
provider does not enforce `max_new_tokens`), so its instant numbers overstate in-weights ability
by an unknown margin. Gemini-flash carries the effort=minimal asterisk throughout.

## Rung 4 — Chain depth (thinking)

**What it tests.** A pointer chase: each agent's fact names another agent ("what is a0 of a0 of
a0 of a0 of g1 ?"), followed d hops deep. The `chain_nowrap` facet runs a staircase — depth d
over 2d+1 agents, so the chain never wraps and no direction of traversal is cheaper than the
measured depth. Reasoning on (effort=high), 16,384-token budget, n=25 per depth.

**What good looks like.** Chance is one agent in 2d+1 (~0.03 at d=16, under 0.01 at d=128); there
is no shallow floor above chance. The *horizon* is the deepest d with relaxed >= 0.8.

| Model | d=16 | d=32 | d=64 | d=128 | chain horizon |
|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 0.96 | 0.68 | 0.08 | 32 (borderline) |
| anthropic/claude-sonnet-5 | 1.00 | 0.72 | 0.24 | 0.04 | 16 (borderline) |
| deepseek/deepseek-v4-pro | 1.00 | 0.88 | 0.88 | 0.08‡ | >=64 (budget-censored) |
| google/gemini-3.5-flash | 1.00 | 1.00 | 1.00 | 0.88 | >=128 |
| moonshotai/kimi-k2.6 | 1.00 | 0.92 | 0.92‡ | 0.64‡ | 64 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.44 | 0.04 | 0.00 | 0.00 | — |
| openai/gpt-5.5 | 1.00 | 1.00 | 0.84 | 0.36 | 64 |
| qwen/qwen3.7-max | 1.00 | 1.00 | 0.88 | 0.96 | >=128 |
| z-ai/glm-5.2 | 0.96 | 0.28 | 0.48 | 0.36 | 16 |

Gemini-flash and qwen hold >= 0.88 through d=128; nemotron never reaches threshold at any tested
depth (the `—`).

**Caveats.** Borderline horizons (opus, sonnet) mean the first failing cell's Wilson
interval crosses the 0.8 line — the horizon versus the next depth is not statistically resolved.
Deepseek's d=128 failure is majority finish=length, hence the budget-censored `>=64`. Cap-escape
marks (deepseek d=128, kimi d=64/128) mean the provider exceeded the token
cap on >10% of calls, so those cells' token spends are not comparable to capped cells.

## Rung 5 — Long-horizon state (s5, thinking)

**What it tests.** Non-abelian state: a sequence of swap/cycle events permutes five people over
five jobs ("Five people — Alice, Bob, Cara, Dan, Eva — each hold one job. Initially: Alice is
Manager, ... Alice and Bob swap jobs. ... what job does Dan have?"), L events deep — order matters, and
no summary statistic of the stream shortcuts the permutation product. Concrete rendering,
reasoning on (effort=high), n=25 per length, 16,384-token budget at L128/L256.

**What good looks like.** Chance is one job in five (0.20). With reasoning off under the abstract
rendering, the roster's floor arm scores 0.00–0.30 — at or below chance — so everything above is
purchased by the thinking budget. The horizon is the longest L with relaxed >= 0.8.

| Model | L16 | L32 | L64 | L128 | L256 | s5 horizon | s5@128 ctok |
|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 1.00 | 1.00 | 0.96 | 0.00 | >=128 (budget-censored) | 12683 |
| anthropic/claude-sonnet-5 | 1.00 | 1.00 | 0.96 | 1.00 | 0.00 | >=128 (budget-censored) | 11866 |
| deepseek/deepseek-v4-pro | 1.00 | 1.00 | 0.96 | 1.00 | 0.28 | >=128 (budget-censored) | 10043 |
| google/gemini-3.5-flash | 1.00 | 1.00 | 0.92 | 0.88 | 0.52 | 128 | 11022 |
| moonshotai/kimi-k2.6 | 1.00 | 0.96 | 1.00 | 1.00 | 0.88 | >=256 | 17418 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.44 | 0.56 | 0.36 | 0.68 | 0.00 | — | 12250 |
| openai/gpt-5.5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | >=256 | 6989 |
| qwen/qwen3.7-max | 1.00 | 1.00 | 0.88 | 0.92 | 0.80 | >=256 | 7904 |
| z-ai/glm-5.2 | 1.00 | 0.96 | 1.00 | 1.00 | 0.88 | >=256 | 5980 |

Everything except nemotron solves the mid-band; the discrimination is entirely at L128–L256. The
facet's scheduled lengths are the discriminating pair (128, 256); shorter lengths are retained
where measured from the earlier run.

**Caveats.** The opus/sonnet/deepseek `>=128` horizons are budget-censored: their L256 failures
are majority finish=length (opus and sonnet emit no visible answer on all 25 calls), so L256
measures the token budget, not the tracking ability. `s5@128 ctok` is completion tokens per call
on the matched L128 cell, which every roster model ran; it deliberately replaces a per-solve
average that rewarded early failure. Grok-build's L256 carries the cap-escape mark.

## How the regimes differ

The two regimes dissociate, in both directions. Qwen is the sharpest case: at or below the
shallow floor instant (0.24 composite @L16, 0.51 binding), yet >=128 chain depth and >=256 s5
horizon when allowed to think — its state tracking lives entirely in the visible trace. Glm is
the same shape (0.35† instant composite; >=256 s5 at the roster's lowest matched-cell spend,
5,980 ctok). The Anthropic pair inverts it: opus and sonnet post the strongest clean instant
composition (0.72 and 0.62 canonical @L16 — kimi's 0.77† is covert-reasoned), but their thinking
horizons are the shortest measured cleanly (chain 32/16 borderline; s5 censored at >=128 by the
shared budget). Gpt-5.5 is the most balanced: clears every instant floor (0.80 binding, 0.46
composite) and holds s5 to >=256 at 6,989 ctok — roughly 2.5x fewer tokens than kimi's 17,418 on
the identical cell. The efficiency column matters because token-hungry horizons are rented, not
owned: on the matched L128 cell the opus-versus-kimi spend gap is ~1.4x, and glm/gpt-5.5/qwen
reach >=256 at less than half kimi's spend. For scaffolding, the practical reading: agent designs that give
the model room to think can treat the cheap tier's thinking horizons as real; designs that need
state tracked silently between tool calls — in weights, mid-trajectory — can rely only on the
instant tier, which today means the frontier pair, gpt-5.5, and (with its covert-CoT caveat)
kimi. Nemotron clears neither regime.

## Protocol appendix

**Roster and pinning.** Nine models via OpenRouter, registered in
`factworld.benchmark.MODELS` (slug, tier, per-million pricing, capability flags); models removed
from the roster render in the archived section of results.md, and their cells stay in history.
x-ai is unrepresented: no current xAI endpoint is cleanly measurable on this suite — mainline
grok's endpoint safety filter blocks a majority of the composite prompts as apparent gene/variant
nomenclature, and grok-build cannot disable reasoning and is served with reasoning pinned at
~256k tokens regardless of the requested cap (its one measured cycle sits in the archived
section).
Task items are deterministic from fixed seeds; each spec carries a pinned stream version, so
existing cells resume byte-identically and only genuinely new cells run.

**Facets and budgets.** Instant: `zero_budget` (composite_copy_v2; L16, L64, binding_only @L16,
replicate @L16; n=100; effort=none; 96-token cap; hard "Answer:" contract line with last-line
extraction) and `sanity` (recall_copy_v1 @L6, conflict_v1 @L4, n=30). Thinking: `chain_nowrap`
(depths 16–128, k=2d+1 staircase, n=25, 16,384 tokens) and `s5_concrete` (L128/L256, n=25, 16,384
tokens; effort=high throughout). Horizon threshold: relaxed >= 0.8.

**Contracts and escalation.** Per-cell diagnostics gate publication: contract adherence,
covert-CoT rate, reasoning-token rates, finish errors, and API errors. A zero-budget cell whose
first attempt is majority finish=length is rerun once at a 512-token budget; the canonical value
is always the first attempt, and the escalated value publishes only as the `(x.xx @512)`
diagnostic. In the current v2 battery: 3 escalations (all sonnet-5), 0 API errors, 0 finish
errors, contract adherence 0.86–1.00.

**Marks glossary.** `*` off-arm ran effort=minimal (reasoning cannot be disabled). `†` visible
working on the canonical attempt. `‡` cap-escape (provider ignored `max_new_tokens`).
`(x.xx @512)` escalated-budget diagnostic. `>=N` censored lower bound; *(budget-censored)* the
first failure above N was majority finish=length. *(borderline)* the first failing cell's Wilson
CI crosses 0.8. `n/a` cell not run; `—` run, no qualifying value.

**Cost.** The full 502-cell history carries an estimated $235.74 of API spend; the 36-cell
zero-budget v2 battery (9 models x 4 legs, 3,600 calls) cost an estimated $5.08. Adding one model
(12 cells under the current facet plan) runs from a few dollars for cheap models to a few tens of
dollars for frontier pricing with long reasoning traces.

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
  `results/benchmark/history.jsonl`.
- v1-family retirement and the re-measure plan:
  [#11](https://github.com/ianbarber/factworld/issues/11).
- **Which task version each number is on.** Every instant column in this report is on
  `composite_copy_v2`, the de-skewed sampler (uniform last-write placement; the recency heuristic
  scores chance). The sanity, chain, and s5 cells are unchanged tasks with pinned streams. The
  OpenRouter grid and reasoning analyses in
  [`factworld-consolidated.md`](factworld-consolidated.md) and the README, and the local
  from-scratch training numbers, are on the retired v1 family; their re-measure on v2 is tracked
  in [#11](https://github.com/ianbarber/factworld/issues/11).
