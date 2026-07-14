# FactWorld: recall and state tracking, certified independently and in composition

*An instrument for the two component capabilities and their composition; the frontier benchmark
built on it; and the architecture exploration it enables.*

FactWorld is a composition instrument. Recall and last-write-wins state tracking are the
component abilities; the question every model answers is whether they compose — a two-hop query
(resolve the current holder of an object, then recall that holder's fact) scored against the same
components measured in isolation. Part 1 describes the instrument: how each component is
certified independently, how the composed cell decomposes into legs, and the measurement rules. Part 2 is the recurring frontier benchmark built on it. Part 3 is the
architecture exploration it enables at small scale.

## 1. The instrument

Every model runs two regimes: **instant** (reasoning off, hard one-line answer contract —
in-weights ability, no visible working) and **thinking** (generous reasoning budget — what the
model can do when allowed to work). One protocol serves both frontier API models and local
from-scratch architectures: same tasks, same legs, same floors; the difficulty settings (length
L, pool breadth, chain depth) are calibration parameters chosen to place each model class
mid-scale, never axes of the headline.

Where this sits in the literature: `recall_copy` is a single-query, deferred-readout variant of
multi-query associative recall (MQAR), with pool breadth as the load axis. `binding` is
last-write-wins state — absorbing updates, a different object from the group word problems.
`commutative_v1` (experimental) fills the rung between them: per-entity accumulation mod k, where
every event matters but order does not — commutative-vs-s5 isolates order sensitivity,
commutative-vs-binding isolates aggregate-all vs last-write. `s5`
is the non-abelian variant from the S₅ word-problem literature. The instrument's contribution is
measuring these components independently *and* composed, under one protocol, for API models and
from-scratch local models alike.

### Validity machinery

Gold answers come from a symbolic oracle applied to the underlying world state, never from
parsing the rendered text — the renderer and its exact inverse parser form a no-leak contract, so
labels cannot leak. A validity gate (`scripts/validate_suite.py`) certifies that no shallow
shortcut — majority class, recency, first-position, entity-blind aggregates — clears floor on any
task. Every score is **match**, the canonical metric: strip a trailing period from both sides and
compare the model's first len(gold) whitespace tokens to the gold answer — binary per item, no
partial credit (`factworld.tasks.score_relaxed`); containment is the one published diagnostic.
Wilson 95% intervals per cell are in the rendered tables. The rendered `README.md` and
`docs/benchmark/results.md` headline blocks separate the instant-composition cells from the
thinking state-stress cells, and `docs/benchmark/results.md` includes an S5 efficiency ranking
that sorts ceiling-solvers by completion-token efficiency. Task items are deterministic from fixed
seeds, so any cell can be reproduced independently.

### Floors

Two first-class rows. The *recency heuristic* (answer the last event's recipient)
scores 0.04 — chance, because the v2 sampler places the queried object's last write uniformly
over the stream. The *object-filter floor* is E[1/w]: a reader that filters events to the queried
object but picks a random write scores 0.41 @L16, decaying ~1/L to 0.15 @L64, with no last-write
resolution at all. Instant cells are read against the object-filter floor, not chance: a score
near it shows object filtering; genuine state tracking has to clear it. Both floors are
recomputed at render time from the exact deterministic task items (pure stdlib), so they are
independently checkable without any API access.

### Marks and contracts

Per-cell diagnostics gate publication: contract adherence, covert-CoT rate, reasoning-token
rates, finish errors, and API errors. A zero-budget cell whose first attempt is majority
finish=length is rerun once at a 512-token budget; the canonical value is always the first
attempt, and the escalated value publishes only as the `(diag x.xx @512tok)` diagnostic. In the current
v2 battery: 3 escalations (all sonnet-5), 0 API errors, 0 finish errors, contract adherence
0.86–1.00.

| mark | meaning |
|---|---|
| `*` | the off-arm ran effort=minimal (the model cannot disable reasoning) |
| `†` | visible working (ctok — completion tokens — above the working line, or a budget escalation) or covert reasoning (rtok — reasoning tokens — despite effort=none) on the canonical instant attempt: not purely in-weights |
| `≤x†` | pervasive covert reasoning — rtok on more than half the canonical attempt's calls: the score is an explicit upper bound |
| `‡` | cap-escape — the provider ignored the token cap |
| `⊘` | majority finish=length — not measurable at this budget |
| `—ᶠ` | gap not interpretable — the state-tracking component sits at the object-filter floor (floor − floor ≈ 0 by construction) |
| `(diag x.xx @512tok)` | single escalated-budget rerun, published as a diagnostic |
| `@Ntok (raised budget)` | thinking cell rerun once at a stated raised completion budget |
| `n/a` / `—` | cell not run / run, no qualifying value |

Notation: `@Ln` = stream length (events, or hops for chain depth d); `@Ntok` = a
completion-token budget.

The contamination marks are symmetric: ⊘ = not measurable at this budget; ≤x† = upper bound,
covert reasoning on most calls; neither participates in orderings — not in figure sorts, not in
cross-model ordering prose.

`⊘` is budget-conditional, not a verdict: a cell showing completion evidence under it is
eligible for a single raised-budget rerun with the budget stated (two s5 cells cleared to 1.00
at 32,768 tokens this way; deepseek's chain d128 stayed ⊘ at the same raise — §2). Nemotron's
chain d128 showed completion evidence (12/25 clean stops, all wrong), so its raise was bought
too: at 32,768 tokens it scores 0.00 with 16/25 still truncated — the failure is capability,
not budget, and the rule has now been applied to every eligible cell.

Two methodological notes:

> **Methodological note: completion budgets.** A reasoning-model cell without an explicit large
> completion budget and a published empty-prediction rate measures truncation, not capability.
> The same s5 concrete-rendering cell reads 0.10 @L64 with 27/30 empty predictions at a 16-token
> completion budget, and 0.90–1.00 through L128 at an 8,192-token budget — the per-cell empty
> rate is the diagnostic that separates "wrong" from "cut off". This is why `⊘` (majority
> finish=length — not measurable at this budget) is a first-class mark and every thinking cell
> publishes its empty rate ([experiments §6](../docs/experiments/README.md)).

> **Methodological note: the recency shortcut behind the floors.** A give-stream sampler that
> draws events uniformly leaves the queried object's resolving write ~Geometric(1/4) from the end
> of the stream, so a one-line recency heuristic (answer the last event's recipient) scores
> 0.33 — indistinguishable from mid-roster state tracking. The current sampler places the queried
> object's last write uniformly over the stream, which drives that heuristic to chance (0.06) and
> exposes the object-filter floor as the real bar: cheap-tier models sit at or below the 0.41
> floor where they previously looked mid-pack, and a local gdp_hybrid's binding leg drops
> 0.99→0.82 @L16 and 0.70→0.23 @L64. Under the earlier recency-skewed sampler fprm's binding leg
> read 0.94 @L64; de-skewed, the architecture ordering survives but that magnitude was recency
> credit (§3.2). This is why the recency heuristic is a permanent floor row and why the validity
> gate checks it on every give-stream task ([experiments §10](../docs/experiments/README.md)).

Every number in this report is on the current versioned specs (`composite_copy_v2` for the
give-stream cells); pinned streams make cells resume byte-identically.

## 2. Benchmarking the frontier

The recurring benchmark reads twelve frontier models through the instrument: the component legs
first, then the composed cell in both regimes. Two models are thinking-only by endpoint design —
x-ai/grok-4.5 and muse-spark-1.1 — because their endpoints cannot disable reasoning. A third,
moonshotai/kimi-k2.6, is instant-excluded: its effort=none arm leaks reasoning on most calls and
its provider does not enforce the token cap, so its instant cells are explicit upper bounds rather
than in-weights measurements. All three carry no instant numbers (the roster mechanics are under
*Adding a model* below).

### Components

**Recall.** The load axis for recall is pool breadth — how many facts the in-context map holds.
Two cells are positive controls near ceiling: copy one fact out of a pool-6 map
(`recall_copy_v1` @L6) and override a memorizable map with an in-context value (`conflict_v1`
@L4); reasoning off, n=30 each — every model that runs the instant battery (nine of the twelve;
grok-4.5 and muse-spark-1.1 are thinking-only, and moonshotai/kimi-k2.6 is instant-excluded
because its effort=none arm leaks reasoning on most calls) scores 0.97–1.00 (sonnet-5's 0.97 recall is the only cell off 1.00).
Any model below ~1.0 there would flag a harness problem, not a capability difference.

**Recall under load.** The measured load row scales the pool with the length:
`recall_copy_v1` @L64 with 64 distinct agents and facts (chance 1/64 ≈ 0.016), instant protocol
(effort=none, answer contract, 96-token cap), n=50. **All nine instant-measured models score
1.00**, with clean diagnostics throughout (contract 1.00, covert working 0.00). Single-query
deferred recall is at
ceiling for this roster out to pool-64: whatever composition costs a frontier model, it is not
the recall component.

**State tracking (binding leg).** A stream of give events reassigns objects to holders; the model
reports the *current* holder — last write wins. The cell is the `binding_only` leg of the
zero-budget battery (`composite_copy_v2` items, reasoning off, one-line contract, 96-token cap,
n=100): state tracking isolated from recall, read against the floors of §1.

| Model | binding_only @L16 |
|---|---|
| anthropic/claude-opus-4.8 | 0.78 |
| anthropic/claude-sonnet-5 | 0.77 |
| deepseek/deepseek-v4-pro | 0.51 |
| google/gemini-3.5-flash | 0.66* |
| muse-spark-1.1 | n/a |
| nvidia/nemotron-3-ultra-550b-a55b | 0.49 |
| openai/gpt-5.5 | 0.80 |
| openai/gpt-5.6-sol | 0.82 |
| qwen/qwen3.7-max | 0.51 |
| x-ai/grok-4.5 | n/a |
| z-ai/glm-5.2 | 0.71 |
| *recency heuristic (floor)* | 0.04 |
| *object-filter floor* | 0.41 |

Opus (0.78), sonnet (0.77), gpt-5.5 (0.80), gpt-5.6-sol (0.82), and glm (0.71) clear the
floor decisively; deepseek and qwen (0.51) and nemotron (0.49) sit within noise of it —
object filtering, not established state tracking. moonshotai/kimi-k2.6 is omitted from the
instant table: its effort=none arm leaks reasoning on most calls and its provider does not
enforce the token cap, so its instant numbers are explicit upper bounds, not in-weights
measurements.

**Commutative state (experimental).** `commutative_v1` is the rung between last-write and
non-abelian state: per-entity accumulation mod k — each event turns a named entity's dial some
clicks, the query asks one dial's final position (closed answer set, k=5) — so every event is
load-bearing but order is not, and distractor entities force per-entity filtering. Floors:
chance 1/k = 0.20; the strongest of its four shallow adversaries (initial-only, last-turn,
entity-blind-sum, count-mod-k) reads 0.22 on the validity gate (n=500, all four gated ≤ 0.4;
`scripts/validate_suite.py`). Instant sits at the floors — glm
0.24 @L16 / 0.12 @L64, deepseek 0.20 / 0.12 (effort=none, contract, n=25) — so the cell reads
only in the thinking regime (effort=high, 8,192 tokens, @L64). The full roster ran there under
a pre-registered promotion bar (issue #18), except muse-spark-1.1, which was not added to this
experimental row: the row joins the headline only if it produces at
least three tiers whose Wilson 95% intervals separate. Result (n=25, with a pre-registered
top-up to n=50 at the one tier boundary): gpt-5.5 0.96 [0.80, 0.99]; opus, gemini-flash, qwen
0.80 [0.61, 0.91]; deepseek 0.80 (calibration cell, reused); kimi 0.66 (n=50) [0.52, 0.78]‡;
sonnet 0.64 (n=50) [0.50, 0.76]; glm 0.52 [0.34, 0.70]; nemotron 0.44 [0.27, 0.63]. Only
gpt-5.5 separates from the bottom four, and the 0.80 group overlaps both ends — two fuzzy
tiers, not three separated ones, so the row **stays experimental**; the two cells bought after
the adjudication (gpt-5.6-sol 0.76 [0.57, 0.89], grok-4.5 0.72 [0.52, 0.86]) land inside the
overlapping middle and do not change it (per-cell values in
`docs/benchmark/results.md`; calibration cells in `results/commutative_frontier/runs.jsonl`).
Two things it already earns its keep on: no model is at ceiling (the composed cell is,
at these lengths), and it carries a reversal — deepseek 0.80 over glm 0.52 — that no other
axis shows. Locally the rung does not form at the binding operating point (§3.6).

### Composition

The core statistic. The composed cell is the two-hop query in one shot ("what is a0 of the holder
of o0 ?", answered with the holder–value pair), under the same zero-budget protocol (reasoning
off, contract, 96-token cap, n=100) at L16 and L64. The **composition gap** is
binding_only@L16 − composed@L16.

The gap isolates composition because the recall half is free: on the scaffolded leg (E1b) — the
same `composite_copy_v2` items with the resolved holder provided (n=100, instant protocol) —
recall reads 0.98–1.00 for every measurable roster model: 1.00 for six of the eight that run
it, nemotron 0.99; qwen3.7-max is ⊘ on this leg, and a four-arm probe localizes
why: it answers
the scaffolded question correctly on every probed call (the gold value appears in the raw text
10/10 in all four arms) but never obeys the exact contract line "Reply with only one line:
Answer: <value>" (0/10 compliance, unchanged at a 512-token cap and with no contract at all),
while a reworded contract line gets 10/10 compliance and 10/10 scored — a contract-phrasing
interaction on this leg, not a recall failure (`results/qwen_scaffold_probe/`). Its
recall-given-holder is at ceiling like the rest of the roster; the ⊘ stands because the leg's
published protocol is the fixed contract. If composing were free, the composed cell would match
the binding leg; the gap is the composition deficit.

| Model | recall | binding @L16 | composed @L16 | composed @L64 | gap @L16 |
|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 0.78 | 0.72 | 0.43 | +0.06 |
| anthropic/claude-sonnet-5 | 0.97 | 0.77 | 0.62 (diag 0.76 @512tok)† | 0.32 (diag 0.66 @512tok)† | +0.15† |
| deepseek/deepseek-v4-pro | 1.00 | 0.51 | 0.44 | 0.19 | —ᶠ |
| google/gemini-3.5-flash | 1.00 | 0.66* | 0.64* | 0.28* | +0.02* |
| muse-spark-1.1 | n/a | n/a | n/a | n/a | n/a |
| nvidia/nemotron-3-ultra-550b-a55b | 1.00 | 0.49 | 0.33 | 0.12 | —ᶠ |
| openai/gpt-5.5 | 1.00 | 0.80 | 0.46 | 0.33 | +0.34 |
| openai/gpt-5.6-sol | 1.00 | 0.82 | 0.65 | 0.33 | +0.17 |
| qwen/qwen3.7-max | 1.00 | 0.51 | 0.24 | 0.08 | —ᶠ |
| x-ai/grok-4.5 | n/a | n/a | n/a | n/a | n/a |
| z-ai/glm-5.2 | 1.00 | 0.71 | 0.38† | 0.13 | +0.33† |
| *recency heuristic (floor)* | — | 0.04 | 0.04 | 0.06 | — |
| *object-filter floor* | — | 0.41 | 0.41 | 0.15 | — |

The run-to-run noise bar on every instant number is 0.06: the battery carries a `replicate` leg
(prompts identical to the plain @L16 cell; recorded as end_to_end in earlier runs), and the
maximum observed |plain − replicate| across models is 0.06.

**Reading the gap.** It is interpretable only where the binding component is established. For
deepseek, qwen, and nemotron the binding leg's Wilson CI overlaps the 0.41 object-filter floor's,
so their composed cells are floor-shaped and the gap renders `—ᶠ` — floor − floor ≈ 0 by
construction, not a composition measurement. Where binding is solid, the roster separates: opus
composes essentially for free (gap 0.06, equal to the noise bar) and gemini flash's 0.02 is the
same shape under its `*` caveat. Sonnet's +0.15† is budget-conditional: its cells were the
battery's only escalations, and the @512tok diagnostics close the gap to ~0.01 @L16 (+0.08 at the
L32 stability check) — read it as a truncation-inflected upper bound on its composition deficit.
Gpt-5.5 pays the largest clean deficit on the roster — binding 0.80, composed 0.46, gap 0.34 —
with glm just under it at +0.33† and gpt-5.6-sol at a clean +0.17. At L64 the floor decays to
0.15 and the same ordering holds among the clean cells: opus 0.43, gpt-5.5 and gpt-5.6-sol
0.33, sonnet 0.32 clear it.

**Marks on this table, in plain language.** moonshotai/kimi-k2.6 is omitted from the instant
table: its effort=none arm emits reasoning tokens on 65–89% of calls despite the contract, and
its provider does not enforce the token cap, so its instant cells are explicit upper bounds
rather than in-weights measurements. Sonnet's cells were the battery's only budget
escalations (majority finish=length at 96 tokens); the canonical numbers are the first attempts,
the `@512tok` values single-rerun diagnostics. Glm's marked composed cell carries a small
covert-reasoning leak (rtok on ~3% of calls, mean 2.9 per call — over the 2-rtok line, nowhere
near the pervasive bar). Gemini-flash cannot disable reasoning; its off-arm ran effort=minimal
throughout.

### Composition under reasoning

With reasoning on, the composed cell reads at or near ceiling at canonical settings across this
roster — 0.98–1.00 on the effort=high arm of the v2 dose-response cell (kimi 1.00, glm 0.98
@L16, n=50; `results/reasoning_sweep_20260710_125924.jsonl`), and the calibration
probes hold glm at 0.92–1.00 on v2 out to L1024 at k=32 (`results/v3_probes/`,
`results/composite_frontier_20260709.jsonl`). That is a calibration fact about the settings, so
the thinking regime is read through two state-stress rows, reported as plain scores at named
settings, plus a practical efficiency column:

- **chain d128 @ k=257** — a pointer chase 128 hops deep over 257 agents (the `chain_nowrap`
  staircase builds k=2d+1, so every d128 cell is the same fixed-breadth k=257 cell across the
  roster); effort=high, 16,384 tokens, n=25. Chance is under 0.01.
- **s5 @L256** — non-abelian state: 256 swap/cycle events permute five people over five jobs,
  concrete rendering, effort=high, 16,384 tokens, n=25. Chance is 0.20; the roster's
  reasoning-off floor arm on the abstract rendering scores 0.00–0.30. Cells whose diagnostics
  showed truncation with completion evidence were rerun once at 32,768 tokens (the raised
  budget is stated wherever it applies).
- **s5@128 ctok** — completion tokens per call on the matched s5 L128 cell, which every roster
  model ran (a per-solve average would reward early failure; the matched-cell total does not).

| Model | chain d128 (k=257) | s5 @L256 | s5@128 ctok |
|---|---|---|---|
| x-ai/grok-4.5 | n/a | 1.00‡ | 8069 |
| muse-spark-1.1 | 0.88ʳ | 1.00ʳ | 9704 |
| anthropic/claude-sonnet-5 | 0.04 | 1.00 @32,768tok (raised budget) | 11866 |
| anthropic/claude-opus-4.8 | 0.08 | 1.00 @32,768tok (raised budget) | 12683 |
| openai/gpt-5.5 | 0.36 | 0.96 | 6989 |
| z-ai/glm-5.2 | 0.36 | 0.88 | 6282 |
| moonshotai/kimi-k2.6 | 0.64‡ | 0.88 | 17418 |
| qwen/qwen3.7-max | 0.96 | 0.80 | 7904 |
| openai/gpt-5.6-sol | 1.00 | 0.72 | 2657 |
| google/gemini-3.5-flash | 0.88 | 0.52 | 11022 |
| deepseek/deepseek-v4-pro | ⊘ @32,768tok (raised budget) | ⊘ | 10043 |
| nvidia/nemotron-3-ultra-550b-a55b | ⊘ @32,768tok (raised budget) | ⊘ | 12250 |

n=25 per cell; Wilson intervals ≈ ±0.15–0.19, and the one thinking test-retest pair moved 0.16 —
differences under ~0.2 are not an ordering.

`⊘` cells are not measurable at the stated budget: majority finish=length, so the score
reflects the token budget rather than the ability. The s5 cells marked @32,768tok (raised budget) are the
demonstration that the mark means what it says: at 16,384 tokens opus, sonnet, and muse-spark-1.1 emitted no
visible answer on most or all of their 25 s5 L256 calls; at 32,768 all three solve all 25 with clean stops
(opus ctok mean 23,898, max 28,986; sonnet mean 24,071; muse-spark mean 24,434 — the cells simply need ~24k tokens).
Deepseek's chain d128 is the contrast case: rerun at the same 32,768 budget it still hits
length on 19/25 calls (median ctok = the cap) and scores 0.08 — still budget-bound, so its ⊘
stands with the budget stated. Nemotron's chain d128 raise (bought on its completion evidence:
12/25 clean-but-wrong stops) reads 0.00 at 32,768 with 16/25 still truncated — a capability
failure confirmed at double budget, not a budget artifact. `‡` cap-escape: kimi (chain
d64/d128) and grok-4.5 (chain d64, s5 L256 — its endpoint does not bound reasoning by the
requested cap) exceeded the token cap on >10% of calls, so those token spends are not
cap-comparable.
(Deepseek's superseded 16,384-token chain d128 attempt also escaped the cap; the published
32,768 rerun did not — max ctok is exactly the cap.) `n/a` cells never ran: grok-4.5's chain
d128 and every instant cell, and muse-spark-1.1's every instant cell.

The scores discriminate where the composed cell cannot: qwen (0.96 chain, 0.80 s5), gpt-5.6-sol
(1.00 chain, 0.72 s5), muse-spark-1.1 (0.88 chain, 1.00 s5), and gemini-flash (0.88, 0.52) hold
deep state under reasoning that they cannot hold in weights, while opus and sonnet — the
strongest clean instant composers — post the weakest measurable chain scores (0.08, 0.04) even
as they solve s5 L256 outright once the budget covers their ~24k-token traces.

**S5 efficiency.** The matched L128 cell completion-token spend (every model runs the same cell)
spans a wide range, and the score on the harder L256 cell separates ceiling from non-ceiling — so
token efficiency becomes the practical discriminator:

| Model | s5 @L256 | s5@128 ctok/call |
|---|---|---|
| x-ai/grok-4.5 | 1.00‡ | 8069 |
| muse-spark-1.1 | 1.00ʳ @32,768tok | 9704 |
| anthropic/claude-sonnet-5 | 1.00ʳ @32,768tok | 11866 |
| anthropic/claude-opus-4.8 | 1.00ʳ @32,768tok | 12683 |
| openai/gpt-5.5 | 0.96 | 6989 |
| z-ai/glm-5.2 | 0.88 | 6282 |
| moonshotai/kimi-k2.6 | 0.88 | 17418 |
| qwen/qwen3.7-max | 0.80 | 7904 |
| openai/gpt-5.6-sol | 0.72 | 2657 |
| google/gemini-3.5-flash | 0.52 | 11022 |

The full roster ranking (including budget-censored models) is rendered live in
`docs/benchmark/results.md#s5-efficiency-ranking`.

The efficiency column is the practical note: token-hungry state tracking is
rented, not owned. gpt-5.6-sol is by far the cheapest per call on the roster at 2,657 ctok, and it
reaches 0.72 on s5 @L256 without a raised budget — but that cheapness comes with a length
extrapolation cliff: it is perfect on the same task at L128 (1.00) and then drops to 0.72 when the
stream doubles to 256 events. Its wrong answers are not concentrated on a single event type; the
model simply does not spend enough tokens to track the full 256-step permutation history reliably.
Among the models that score ≥0.95 at s5 @L256, grok-4.5 uses the fewest tokens (8,069 ctok) and
opus the most (12,683 ctok) for the same perfect score. grok-4.5's `‡` means its provider did not
enforce the requested cap, so that token spend is not strictly cap-comparable.

Kimi's instant composed scores look high (≤0.94† / ≤0.77† / ≤0.93†) because its cells carry the
same `†` leak: the model emits reasoning tokens on 65–89% of zero-budget calls despite
effort=none, and its provider does not enforce the token cap. Those numbers are explicit upper
bounds, not in-weights ability; the composed @L64 score exceeding @L16 is a covert-reasoning
artifact, not a length effect.

Muse-spark-1.1 is served directly by the Meta Model API (`https://api.meta.ai/v1`) using the
OpenAI Responses API, not OpenRouter. Like grok-4.5, its endpoint cannot disable reasoning:
even `effort=minimal` emits thousands of reasoning tokens per call, so the instant contract
cells (96-token cap) produce no visible answer and are unplanned. In the thinking regime it
matches the strongest state-stress scores on the roster once the budget is raised to 32,768
tokens, but its reasoning traces are expensive.

**Depth dissociates by regime within one cell.** The chain d16 cell (k=33, deterministic items
shared between regimes, n=25, chance ≈ 0.03) runs in both regimes. Thinking (effort=high, 16,384
tokens): nine of twelve models score 1.00; gpt-5.6-sol and glm 0.96; nemotron 0.44. Instant
(effort=none, contract, 96-token cap; grok-4.5 and muse-spark-1.1 run no instant cell): every model that answers
cleanly floors — gpt-5.5 0.08; gpt-5.6-sol, qwen, deepseek, and nemotron 0.00, glm 0.00† — and
the other four spend the budget trying to emit working instead of an answer: opus and gemini-flash* hit the cap on 25/25 calls (canonical 0.00,
escalated diagnostics 0.96 and 1.00 @512tok), sonnet on 18/25 (0.28, 0.96 @512tok), kimi on 16/25
(0.32, 0.96 @512tok). The @512tok diagnostics are short visible working, not in-weights answers. A
16-hop pointer chase is serial work: no roster model holds it in weights, and every strong model
solves it given room to work. This is the within-depth regime contrast that the depth axis is
read against.

### Profiles

The two regimes never merge into one number; the per-model view is split into two profile
grids ([`fig_profiles_instant.png`](../docs/benchmark/fig_profiles_instant.png) and
[`fig_profiles_thinking.png`](../docs/benchmark/fig_profiles_thinking.png), regenerated every
render cycle): one panel per model, its normalized position on each axis — binding @L16,
composed @L16, and gap (inverted: smaller better) for the instant grid; chain d128, s5 @L256,
and s5@128 ctok (inverted) for the thinking grid — with raw values and marks alongside, and
⊘/never-run cells drawn as gaps rather than zeros. Read against a pinned intuitive ranking of
the roster
([`docs/benchmark/profiles-analysis.md`](../docs/benchmark/profiles-analysis.md) — a data note
pinned to the nine-model roster; gpt-5.6-sol, grok-4.5, and muse-spark-1.1 postdate it), the axes
split three ways: in-weights state tracking is what intuition tracks (binding @L16 Spearman
+0.81, +0.97 without kimi's daggered cell — the best single-axis match, and binding plus
inverted ctok reaches +0.95 over all nine); chain d128 *inverts* it (-0.72: qwen 0.96 and
gemini-flash 0.88 lead where opus 0.08 and sonnet 0.04 trail), the instant/thinking
near-orthogonality showing up against the prior; and the commutative row carries the
same reversal (thinking @L64: deepseek 0.80 over glm 0.52, against any prior that puts glm
well ahead). The thinking axes measure something intuition does not already contain.

### Adding a model

**Roster and pinning.** Twelve models via OpenRouter, registered in `factworld.benchmark.MODELS`
(slug, tier, per-million pricing, capability flags); models removed from the roster render in the
archived section of results.md, and their cells stay in history. x-ai is represented by
grok-4.5, thinking facets only: its endpoint rejects effort=none outright and its
effort=minimal arm still reasons covertly (547 rtok on an L16 probe), so the registry skips
every instant facet structurally (`skip_facets`) and grok-4.5 carries no instant numbers.
muse-spark-1.1 is served directly by the Meta Model API (`https://api.meta.ai/v1`) through a
Responses-backend path, not OpenRouter; like grok-4.5, it cannot disable reasoning, so the
registry also skips every instant facet and it carries no instant numbers.
moonshotai/kimi-k2.6 is instant-excluded for a different reason: its effort=none arm leaks
reasoning tokens on 65–89% of calls and its provider does not enforce the requested cap, so its
instant cells render as explicit upper bounds (`≤x†`) rather than in-weights measurements; the
registry skips every instant facet for it too.
openai/gpt-5.6-sol is routed directly to the OpenAI API (`https://api.openai.com/v1`) instead
of OpenRouter; its registry entry sets `model_name` to the vendor model id, `max_completion_tokens`
to true, and `supports_reasoning_effort` to false because the direct endpoint uses different
parameter names.
Earlier xAI endpoints were not cleanly measurable — mainline grok's safety filter blocked a
majority of the composite prompts as apparent gene/variant nomenclature, and grok-build was
served with reasoning pinned at ~256k tokens regardless of the requested cap (its one measured
cycle sits in the archived section). Task items are deterministic from fixed seeds; each spec
carries a pinned stream version, so existing cells resume byte-identically and only genuinely new
cells run.

Register the slug in `factworld.benchmark.MODELS` (tier, pricing, flags), then:

```bash
python scripts/run_frontier_benchmark.py --models <slug> --dry-run   # plan + cost preview; everything else resume-skips
python scripts/run_frontier_benchmark.py --models <slug>             # run; appends per-cell records to history.jsonl
python scripts/render_benchmark.py                                   # re-render results.md/csv, figures, index.html
```

**Facets and budgets.** Instant: `zero_budget` (composite_copy_v2; composed @L16 and @L64,
binding_only @L16, scaffolded @L16, replicate @L16; n=100; effort=none; 96-token cap; hard "Answer:" contract line
with last-line extraction), `sanity` (recall_copy_v1 @L6, conflict_v1 @L4, n=30), `recall_load`
(recall_copy_v1 @L64 with the agent pool scaled to the length — pool 64; n=50; contract;
96-token cap), and `chain_instant` (chain_v1 d16 on the same k=33 staircase items as the
thinking d16 cell; n=25; contract; 96-token cap). Thinking: `chain_nowrap` (k=2d+1 staircase;
the reported cell is d128, k=257; n=25; 16,384 tokens) and `s5_concrete` (the reported cell is
L256; n=25; 16,384 tokens; effort=high throughout). A truncated thinking cell with completion
evidence can be rerun once at a stated raised budget
(`run_frontier_benchmark.py --budget-override facet:length:budget`); the rerun replaces the
cell in the rendered tables and the raised budget publishes with the number.

## 3. Exploring the architectures

The same instrument evaluates architectural choices in from-scratch models: same tasks, same
legs, same floors, same match metric. The local roster is the transformer, the gated hybrids
(gdp_hybrid — a GatedDeltaProduct stack with one attention layer — and the gdn variants, gdn =
GatedDeltaNet), the pure-recurrence probes (gdp_pure, gdn_pure), and fprm (Fast Parallel
Recurrent Model — a weight-tied looped conv+attention block). Comparisons are compute-matched,
not parameter-matched: architectures share
`(d_model, depth)`, and fprm's weight-tying makes its per-token FLOPs equal the transformer's at
~5–11× fewer parameters. The calibration parameters move to place d256-class models mid-scale; the
statistic — components versus the composed cell — does not. The sections below are organized by
capability: which component each architectural choice buys, and where each breaks.

### 3.1 Recall: adjacent versus deferred

Every architecture aces adjacent 1-hop readout (1.00 across the board) — attention suffices for
the canonical easy MQAR regime. Deferred readout — the value read at an arbitrary later position,
the regime composition actually requires — needs product recurrence: the transformer aces
adjacent but fails deferred (0.19, against gdp_hybrid's 0.73), and the attention-free contrast
localizes the mechanism — gdp_pure supplies deferred recall with no attention at all, while the
single-delta gdn_pure fails. Recall is not one capability: attention buys the adjacent read;
product recurrence buys the deferred one ([consolidated §3](factworld-consolidated.md);
provenance [phases/01 §3.2](../phases/01-instrument/factworld.md)).

### 3.2 Last-write state under breadth

The operating-point sweep mirrors the frontier breadth rungs on from-scratch models: fprm vs
gdp_hybrid vs transformer, d256×4, 8k steps flat next-token training (train L ∈ {4, 8, 16}),
rungs B ∈ {6, 8, 12, 16, 24} via `composite_copy_v2.scaled(k=2B, recall_pool=B)`, m=4, 3 seeds
per cell, RTX 5090 (45 runs; `results/local_breadth/sweep_summary.md`). L16 is in-distribution,
L64 is length extrapolation; pconv = fraction of seeds ≥0.9 (composite convergence is bimodal —
read pconv, not the mean).

| B | arch | composed @L16 (pconv) | composed @L64 | binding / recall leg @L16 | @L64 |
|---|---|---|---|---|---|
| 6 | fprm | 0.15±0.01 (0%) | 0.01 | 1.00 / 0.15 | 0.41 / 0.01 |
| 6 | gdp_hybrid | 0.04±0.02 (0%) | 0.00 | 0.56 / 0.06 | 0.09 / 0.01 |
| 6 | transformer | 0.02±0.00 (0%) | 0.01 | 0.23 / 0.07 | 0.18 / 0.03 |
| 8 | fprm | 0.04±0.01 (0%) | 0.01 | 0.73 / 0.06 | 0.24 / 0.01 |
| 8 | gdp_hybrid | 0.01±0.01 (0%) | 0.00 | 0.41 / 0.02 | 0.14 / 0.00 |
| 8 | transformer | 0.00±0.00 (0%) | 0.00 | 0.17 / 0.01 | 0.15 / 0.01 |
| 12 | fprm | 0.01±0.01 (0%) | 0.00 | 0.64 / 0.02 | 0.28 / 0.01 |
| 12 | gdp_hybrid | 0.00±0.00 (0%) | 0.00 | 0.43 / 0.01 | 0.19 / 0.01 |
| 12 | transformer | 0.00±0.00 (0%) | 0.00 | 0.15 / 0.01 | 0.10 / 0.00 |
| 16 | fprm | 0.01±0.00 (0%) | 0.00 | 0.97 / 0.01 | 0.38 / 0.01 |
| 16 | gdp_hybrid | 0.00±0.00 (0%) | 0.00 | 0.41 / 0.01 | 0.16 / 0.01 |
| 16 | transformer | 0.00±0.00 (0%) | 0.00 | 0.13 / 0.02 | 0.08 / 0.00 |
| 24 | fprm | 0.00±0.00 (0%) | 0.00 | 0.20 / 0.00 | 0.07 / 0.01 |
| 24 | gdp_hybrid | 0.01±0.01 (0%) | 0.00 | 0.67 / 0.01 | 0.20 / 0.01 |
| 24 | transformer | 0.00±0.00 (0%) | 0.00 | 0.08 / 0.01 | 0.03 / 0.00 |

- **The composed cell reads floor for this model class under flat training** — p(converge)
  0/45. The best single run is fprm's 0.17 @L16 (B6, seed 0), and that number is its solved
  binding leg times a 1/pool value guess (1/6 ≈ 0.17), not composition. At d256 the instrument
  reads through the legs; converging the composed cell locally takes the staged-curriculum
  recipe, and only gdp_hybrid at d768×8 does it (§3.3; 0.833 match, scale-dependent).
- **Local operating point: B8** (set on the gdp_hybrid/transformer pair). The largest rung where
  gdp_hybrid reads mid-scale seed-consistently on the binding leg: 0.41 @L16 (seeds 0.34–0.48)
  against a 1/k = 0.06 agent-guess. From B12 up gdp_hybrid's leg is bimodal — single seeds solve
  binding outright (1.00 @B12, 0.99 @B16, 0.98/1.00 @B24) while the rest sit near floor — so
  per-rung means stop being readable and seed counting takes over. fprm sits near ceiling rather
  than mid-scale at most rungs (below), so B8 remains the calibration cell for the pair the
  operating point was set on.
- **The architecture comparison reads on the binding leg, and fprm leads it through B16.** fprm
  solves binding @L16 on 9/15 seeds — all three at B6 (1.00) and all three at B16 (0.97–0.98,
  the only seed-consistent solve in the sweep) — over gdp_hybrid (0.56 @B6, bimodal above B8)
  over transformer (≤0.23 everywhere). The ordering inverts at B24: fprm stops fitting the
  training distribution (final loss 1.02–1.10; binding 0.13–0.30) while gdp_hybrid still fits
  (loss 0.05–0.14 at every rung; binding 0.67) — product recurrence buys the strongest binding
  but is the first to break under breadth, where the gated hybrid degrades gracefully. The
  transformer decays monotonically to 0.08 @B24 and no longer fits training there (loss
  1.49–1.65). Binding does not extrapolate reliably to L64 for any architecture: fprm keeps the
  most (0.24–0.41 at B6–B16 against the 0.15 object-filter floor; 0.07 @B24), gdp_hybrid
  0.09–0.20, transformer 0.03–0.18. (The earlier recency-skewed sampler read fprm's binding at
  0.94 @L64 — the methodological note in §1 covers why that magnitude was recency credit.)
- **The local composition deficit sits on the recall leg — for all three architectures.** The
  value leg is ≤0.17 in all 45 runs and at or below the 1/pool guess wherever binding is solved
  (fprm @B6: binding 1.00, value 0.14–0.17 ≈ 1/6; fprm @B16: binding 0.97–0.98, value ≤0.01;
  gdp_hybrid binding-solved seeds: value ≤0.02): the resolved holder is not routed into the
  lookup. This is the same leg the d768 staged-curriculum decomposition localizes (§3.3;
  wherever an architecture fails with binding trained, the binding leg holds and the value leg
  collapses).

### 3.3 Composition: the flagship and the scale window

The composed cell converges locally only under the staged curriculum — binding and recall staged
in before the composite mix — and only for gdp_hybrid at d768×8. The statistic for local
composed cells is p(converge ≥ 0.9) over ≥3 seeds (convergence is bimodal, so seed counting, not
means, is the readable number), and `composite_copy_v2.scaled(k=16, recall_pool=8)` is the d256
calibration cell. The flagship measurement — 3 seeds, eval_n=500, match metric, `composite_copy_v2`
pool-16 @L16 (`results/curriculum_staged_v2_d768_notrace.jsonl`):

| arch | composite @L16 | per seed | holder / value leg | p(converge) |
|---|---|---|---|---|
| **gdp_hybrid** | **0.833 ± 0.089** | 0.758 / 0.782 / 0.958 | 0.999 / 0.833 | 1/3 |
| fprm | 0.109 ± 0.089 | 0.056 / 0.036 / 0.234 | 0.998 / 0.109 | 0/3 |
| transformer | 0.001 ± 0.001 | 0.000 / 0.002 / 0.000 | 0.065 / 0.041 | 0/3 |

gdp_hybrid trains the composed cell on all three seeds — one clears the ≥0.9 convergence bar,
the other two read 0.758/0.782 — with the holder leg ≥0.998 throughout, and the compute-matched
scale sweep's medium cell corroborates the number on 2 independent seeds (0.732 ± 0.013). fprm
dissociates the legs: binding 0.998 with a dead value leg. The transformer's 0.001 is a real floor, not formatting: contains reads ~0 as
well, at every scale up to 202M params / 417 GFLOP-tok.

Scale shape, from the compute-matched sweep — the same staged curriculum at three sizes sharing
`(d_model, depth)`, 2 seeds each (`results/composite_scale_*.md`):

| arch | small (384×6) | medium (768×8) | large (1024×12) |
|---|---|---|---|
| **gdp_hybrid** | 0.12 ± 0.08 | **0.73 ± 0.01** | 0.21 ± 0.21 |
| fprm | 0.12 ± 0.05 | 0.03 ± 0.01 | 0.03 ± 0.02 |
| transformer | 0.01 ± 0.00 | 0.01 ± 0.01 | 0.00 ± 0.00 |

Convergence peaks at medium. Small solves binding (gdp_hybrid holder 1.00 on both seeds) but
fails the value leg. Large is seed-bimodal for gdp_hybrid, with the holder leg itself degrading
on the weaker seed — the batch-64 large recipe trains unstably, and with 2 seeds this is a flag,
not a measurement (the raw runs and per-leg decomposition are in
[consolidated §5](factworld-consolidated.md)). The through-line is scale-invariant: wherever
binding trains and the composed cell fails — every architecture, every scale, and all 45 runs of
the d256 breadth sweep (§3.2) — the value leg is what collapses.

### 3.4 Chain: depth does not extrapolate

Chain (recall ∘ recall) at the canonical baseline recipe (d320×4, 8k steps, registered spec:
train depths 2–3, eval depths 4–5, n=200, 3 seeds per architecture;
`results/local_chain_arch_20260710.jsonl`), against a 1/6 agent-guess ≈ 0.17:

| arch | composed @d4 (pconv) | @d5 | final loss |
|---|---|---|---|
| fprm | 0.20±0.01 (0%) | 0.21±0.02 | 0.38–0.40 |
| transformer | 0.22±0.01 (0%) | 0.06±0.02 | 0.40–0.41 |
| gdp_hybrid | 0.02±0.01 (0%) | 0.00±0.00 | 0.23–0.25 |

No run converges (pconv 0/9). fprm and the transformer sit at the guess floor at d4 — fprm
stays there at d5, the transformer falls below it — and gdp_hybrid fits the training
distribution best (lowest final loss) yet scores 0.00–0.03 at both held-out depths: a
depth-specific circuit that is systematically wrong one hop out, not a guesser. The
depth-extrapolation row of the price table is open, with all three architectures measured at 3
seeds each. The contrast with the frontier is the point: the same composition solves at d16 over
the API, but only in the thinking regime (§2).

### 3.5 s5: the lever is supervision density

Non-abelian state floors for every architecture under answer-only supervision and forms in every
architecture under dense per-step supervision — a state checkpoint every ≤2 events; below that
density the circuit is gone. The formation lever is the training signal, not the architecture.
What the architecture buys is length extrapolation: only the recurrent hybrid carries the
circuit past the training length (gdp_hybrid 0.75 @L64; fprm 0.19 and transformer 0.22 solve
in-distribution and collapse past train length). Weaning keeps it label-free: train dense,
fine-tune on mixed densities including answer-only, and 8/8 seeds converge free-running with no
deploy-time labels, extrapolating on par with dense-only
([consolidated §8](factworld-consolidated.md); [experiments §1](../docs/experiments/README.md);
provenance [phases/02 §4](../phases/02-non-abelian-state/report.md)).

### 3.6 Commutative state locally

The commutative rung does not form at the binding operating point: d256×4, three architectures ×
three seeds, every run at chance (0.15–0.24) at L16/32/64 — including the worked-trace
contingency (`results/commutative_local/`). Aggregation across all events is locally harder than
last-write: the same operating point that reads mid-scale on the binding leg leaves the
accumulation task untouched. Over the API the rung reads only in the thinking regime (§2).

### 3.7 What buys each element — the price table

The local runs give the frontier profiles their thesis: **no element of the composition is
free — each is paid for by an architectural or training choice.** Two rows remain open.

| element | price | evidence |
|---|---|---|
| adjacent (1-hop) recall | attention — every architecture aces adjacent readout (1.00) | §3.1; [consolidated §3](factworld-consolidated.md); archived provenance phases/01 §3.2 |
| deferred recall | product recurrence — the transformer aces adjacent, fails deferred (0.19 vs gdp_hybrid 0.73) | §3.1; consolidated §3; phases/01 §3.2 (atomic format: gdp_pure 1.00 attention-free vs transformer 0.48) |
| last-write state | recurrence, ordered by form — fprm (product recurrence) 1.00 @B6 / 0.97–0.98 seed-consistent @B16 on the binding leg, over gdp_hybrid (0.56 @B6) over transformer (0.23 @B6); at B24 fprm stops fitting (0.20, loss ≥1.0) and only the gated hybrid holds (0.67) | §3.2, breadth sweep |
| non-abelian state (formation) | dense per-step supervision — a state checkpoint every ≤2 events, architecture-independent | §3.5; consolidated §8; [experiments §1](../docs/experiments/README.md); archived provenance phases/02 §4 |
| non-abelian state (length extrapolation) | recurrent hybrid — gdp_hybrid 0.75 @L64; fprm (0.19) and transformer (0.22) solve in-distribution but collapse past train length | §3.5; experiments §1 |
| depth extrapolation | **open** — no measured choice buys it: trained at chain depths 2–3, all three architectures read at or below the 1/6 guess at depths 4–5 (fprm 0.20/0.21, transformer 0.22/0.06, gdp_hybrid 0.02/0.00) | §3.4, local chain table |
| local composition (value leg) | **open** — value ≤0.17 in all 45 breadth-sweep runs (at/below the 1/pool guess), even on binding-solved seeds of all three architectures | §3.2, breadth sweep |

The two open rows are the instrument's active edge: nothing measured so far buys depth
extrapolation, and no local training choice yet converges the value leg of the composed cell
outside the staged-curriculum recipe — and that recipe converges it only for gdp_hybrid
at d768×8 (0.833; the small and large cells of the compute-matched sweep fail the value leg
too).

## Appendix: protocol stability

Three checks behind the part-2 numbers, each read against a pre-stated bar (raw records in
`results/benchmark/history.jsonl`, runs `bench_16_*_20260711`).

**The gap ordering holds off its anchor.** The composition gap is defined at L16; a second
operating point (`gap_stability` facet: composed and binding legs at L32, instant protocol,
n=50) checks that the ordering is not an artifact of the anchor. For the three cleanly
measurable gap-interpretable models it holds: gpt-5.5 +0.34 → +0.36 (binding 0.68, composed
0.32), sonnet +0.15† → +0.14† (canonical first attempts, binding 0.64 / composed 0.50; the
escalated @512tok diagnostic reads +0.08), opus +0.06 → −0.04 — at or below zero at both
operating points, the compose-for-free profile. Gpt-5.6-sol's L32 cells, bought after the
check, read binding 0.58 / composed 0.26 — +0.17 → +0.32, holding its slot between gpt-5.5
and sonnet at both operating points. Kimi's L32 cells are not interpretable
(covert working with an unenforced cap: empty 0.40 on the canonical composed attempt, 0.62 on
the binding leg, which was also cost-limited), so no L32 gap publishes for it.

**Thinking cells carry a wider noise bar than instant ones.** The instant test-retest bar is
0.06 (the replicate leg, §2). The first thinking-regime pair — glm s5 concrete @L128 rerun
under identical settings (effort=high, 16,384 tokens, n=25) — reads 0.84 against a stored
1.00: |Δ| = 0.16. Both records stay in history (the rendered value is the latest). One pair is
a datum, not a bar, but until more pairs exist, differences of ~0.15 between thinking cells of
one model should not be read as movement.

**Drift canary: clean.** One model's full zero-budget battery (glm, all five legs, n=100)
is re-bought each cycle and compared to the stored cells against the ±0.06 replicate bar. The
first pass read composed @L16 −0.01, composed @L64 −0.01, binding −0.02, replicate −0.02,
scaffolded 0.00 — max |Δ| 0.02; the second pass moved at most 0.04 against the first. No
provider or endpoint drift on either pass. The latest pass's cells are the rendered values
(latest-timestamp-wins), so glm's headline row (binding 0.71, composed 0.38†/0.13) is the
current canary, a few hundredths off the cycle before — inside the bar.

## Provenance

- Rendered tables, figures, and per-cell Wilson intervals:
  [`docs/benchmark/results.md`](../docs/benchmark/results.md) (with `results.csv` and
  `index.html` alongside).
- Raw per-cell records (one JSON object per cell, all attempts, usage, diagnostics):
  `results/benchmark/history.jsonl` (zero-budget battery: run `bench_v2_zb2_20260709`;
  scaffolded leg: `bench_20260710_124904`; chain/s5: `bench_v2_20260708`; recall-under-load and
  chain d16 instant: `bench_20260710_frontier_rows`; raised-budget s5/chain cells:
  `bench_17_budget32k_20260711`; commutative roster: `bench_18_commutative_20260711` and its
  top-up; stability checks: `bench_16_*_20260711`; gpt-5.6-sol / grok-4.5 cells:
  `bench_15_newmodels_20260712`; second canary pass (the rendered glm zero-budget cells):
  `bench_15_canary_20260712`; qwen contract probe:
  `results/qwen_scaffold_probe/`).
- Operating-point calibration probes: `results/v3_probes/` and
  `results/composite_frontier_20260709.jsonl` (log-only material;
  [`docs/experiments/README.md`](../docs/experiments/README.md) §12).
- Local smoke evidence: `results/local_smoke_20260709/`; breadth sweep:
  `results/local_breadth/`.
- Staged-curriculum flagship: `results/curriculum_staged_v2_d768_notrace.jsonl`; compute-matched
  scale sweep: `results/composite_scale_*.md`.
- Local chain comparison: `results/local_chain_arch_20260710.jsonl`; commutative calibration:
  `results/commutative_local/` and `results/commutative_frontier/`.
- History and the running experiment log: [`phases/`](../phases/) (provenance) and
  [`docs/experiments/README.md`](../docs/experiments/README.md) (archival log).
