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

Where this sits in the literature: `recall_copy` is a single-query, deferred-readout variant of
multi-query associative recall (MQAR), with pool breadth as the load axis. `binding` is
last-write-wins state — absorbing updates, a different object from the group word problems.
`commutative_v1` (experimental) fills the rung between them: per-entity accumulation mod k, where
every event matters but order does not — commutative-vs-s5 isolates order sensitivity,
commutative-vs-binding isolates aggregate-all vs last-write. `s5`
is the non-abelian variant from the S₅ word-problem literature. The instrument's contribution is
measuring these components independently *and* composed, under one protocol, for API models and
from-scratch local models alike.

Every score is relaxed match, the canonical metric (Wilson 95% intervals per cell are in the
rendered tables). Marks, in plain language: `*` the model cannot disable reasoning (its off-arm
ran effort=minimal); `†` visible working appeared on the canonical instant attempt (the number is
not purely in-weights); `‡` the provider ignored the token cap; `⊘` the cell's calls were
majority finish=length — not measurable at this budget; `(x.xx @512)` a single escalated-budget
rerun published as a diagnostic (the canonical number is always the first attempt at the shared
budget).

## Components

**Recall.** The load axis for recall is pool breadth — how many facts the in-context map holds.
Two cells are positive controls near ceiling: copy one fact out of a pool-6 map
(`recall_copy_v1` @L6) and override a memorizable map with an in-context value (`conflict_v1`
@L4); reasoning off, n=30 each — every current-roster model scores 0.97–1.00 (sonnet-5's 0.97
recall is the only cell off 1.00). Any model below ~1.0 there would flag a harness problem, not
a capability difference.

**Recall under load.** The measured load row scales the pool with the length:
`recall_copy_v1` @L64 with 64 distinct agents and facts (chance 1/64 ≈ 0.016), instant protocol
(effort=none, answer contract, 96-token cap), n=50. **All nine models score 1.00**, with clean
diagnostics throughout (contract 1.00, covert working 0.00). Single-query deferred recall is at
ceiling for this roster out to pool-64: whatever composition costs a frontier model, it is not
the recall component.

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

**Commutative state (experimental).** `commutative_v1` is the rung between last-write and
non-abelian state: per-entity accumulation mod k — each event turns a named entity's dial some
clicks, the query asks one dial's final position (closed answer set, k=5) — so every event is
load-bearing but order is not, and distractor entities force per-entity filtering. Floors:
chance 1/k = 0.20; the strongest of its four shallow adversaries (initial-only, last-turn,
entity-blind-sum, count-mod-k) reads 0.22 on the validity gate (n=500, all four gated ≤ 0.4;
`scripts/validate_suite.py`). Calibration, not a roster row: instant sits at the floors — glm
0.24 @L16 / 0.12 @L64, deepseek 0.20 / 0.12 (effort=none, contract, n=25) — while thinking @L64
discriminates: deepseek 0.80, glm 0.52 (effort=high, 8,192 tokens, n=25, neither at ceiling).
Locally the rung does not form at the binding operating point: d256×4, three architectures ×
three seeds, every run at chance (0.15–0.24) at L16/32/64, including the worked-trace
contingency (`results/commutative_local/`, `results/commutative_frontier/runs.jsonl`). The rung
reads only in the thinking regime at these settings; it stays experimental until a full roster
run.

## Composition

The core statistic. The composed cell is the two-hop query in one shot ("what is a0 of the holder
of o0 ?", answered with the holder–value pair), under the same zero-budget protocol (reasoning
off, contract, 96-token cap, n=100) at L16 and L64. The **composition gap** is
binding_only@L16 − composed@L16.

The gap isolates composition because the recall half is free: on the scaffolded leg (E1b) — the
same `composite_copy_v2` items with the resolved holder provided (n=100, instant protocol) —
recall reads 0.98–1.00 for every measurable roster model: 1.00 for six of the nine, nemotron
0.99, kimi 0.98; qwen3.7-max is ⊘ on this leg (empty completion on 98 of 100 scaffolded calls).
If composing were free, the composed cell would match the binding leg; the gap is the
composition deficit.

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

**Depth dissociates by regime within one cell.** The chain d16 cell (k=33, deterministic items
shared between regimes, n=25, chance ≈ 0.03) runs in both regimes. Thinking (effort=high, 16,384
tokens): seven of nine models score 1.00; glm 0.96; nemotron 0.44. Instant (effort=none,
contract, 96-token cap): every model that answers cleanly floors — gpt-5.5 0.08;
qwen, glm, deepseek, and nemotron 0.00 — and the other four spend the budget trying to emit
working instead of an answer: opus and gemini-flash* hit the cap on 25/25 calls (canonical 0.00,
escalated diagnostics 0.96 and 1.00 @512), sonnet on 18/25 (0.28, 0.96 @512), kimi on 16/25
(0.32, 0.96 @512). The @512 diagnostics are short visible working, not in-weights answers. A
16-hop pointer chase is serial work: no roster model holds it in weights, and every strong model
solves it given room to work. This is the within-depth regime contrast that the depth axis is
read against.

## The local regime

The same instrument evaluates architectural changes in from-scratch models: same tasks, same
legs, same floors, same relaxed metric. The calibration parameters move to place d256-class
models mid-scale; the statistic — components versus the composed cell — does not.

Local smoke evidence (RTX 5090, gdp_hybrid d256×4, 4,000 steps;
`results/local_smoke_20260709/`): the binding leg reads 0.82 @L16 / 0.21 @L32 / 0.23 @L64 on the
v2 sampler, against 0.99 / 0.77 / 0.70 on v1 — the v1 sampler grants recency credit locally just
as it does over the API, and the v2 floors apply unchanged (object-filter 0.41 @L16 / 0.15 @L64
at m=4).

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
  recipe, and on v2 only gdp_hybrid at d768×8 does it
  ([consolidated §5](factworld-consolidated.md); 0.73 relaxed, scale-dependent).
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
  (loss 0.05–0.14 at every rung; binding 0.67) — product recurrence buys the sharpest binding
  but is the first to break under breadth, where the gated hybrid degrades gracefully. The
  transformer decays monotonically to 0.08 @B24 and no longer fits training there (loss
  1.49–1.65). Binding does not extrapolate reliably to L64 for any architecture: fprm keeps the
  most (0.24–0.41 at B6–B16 against the 0.15 object-filter floor; 0.07 @B24), gdp_hybrid
  0.09–0.20, transformer 0.03–0.18. fprm's retired-v1 flagship (binding_v1 0.94 @L64) does not
  carry to the v2 sampler — the ordering survives, the magnitude was v1 recency credit.
- **The local composition deficit sits on the recall leg — for all three architectures.** The
  value leg is ≤0.17 in all 45 runs and at or below the 1/pool guess wherever binding is solved
  (fprm @B6: binding 1.00, value 0.14–0.17 ≈ 1/6; fprm @B16: binding 0.97–0.98, value ≤0.01;
  gdp_hybrid binding-solved seeds: value ≤0.02): the resolved holder is not routed into the
  lookup. This is the same leg the d768 staged-curriculum decomposition localizes (gdp_hybrid
  binding 1.00 / value 0.73 on v2; wherever an architecture fails with binding trained, the
  binding leg holds and the value leg collapses).

For the v1-family re-measure ([#11](https://github.com/ianbarber/factworld/issues/11)):
`composite_copy_v2.scaled(k=16, recall_pool=8)` is the d256 calibration cell, and local
composed-cell numbers keep the staged-curriculum recipe with p(converge) over ≥3 seeds as the
statistic. The current v2 flagship cell (consolidated §5) is on 2 seeds from the compute-matched
sweep; the 3-seed/eval_n=500 curriculum measurement is queued with a corrected trace-free
protocol.

**Chain (recall ∘ recall) locally: depth does not extrapolate for any architecture.** chain_v1
at the canonical baseline recipe (d320×4, 8k steps, registered spec: train depths 2–3, eval
depths 4–5, n=200, 3 seeds per architecture;
`results/local_chain_arch_20260710.jsonl`), against a 1/6 agent-guess ≈ 0.17:

| arch | composed @d4 (pconv) | @d5 | final loss |
|---|---|---|---|
| fprm | 0.20±0.01 (0%) | 0.21±0.02 | 0.38–0.40 |
| transformer | 0.22±0.01 (0%) | 0.06±0.02 | 0.40–0.41 |
| gdp_hybrid | 0.02±0.01 (0%) | 0.00±0.00 | 0.23–0.25 |

No run converges (pconv 0/9). fprm and the transformer sit at the guess floor at d4 — fprm
stays there at d5, the transformer falls below it — and gdp_hybrid fits the training
distribution best (lowest final loss) yet scores 0.00–0.03 at both held-out depths: a
depth-specific circuit that is systematically wrong one hop out, not a guesser. This is fprm's
first chain datum and puts the three-architecture chain comparison on 3 seeds; the
depth-extrapolation row of the price table stays open with all three architectures now
measured.

## What buys each element — local evidence

The local runs give the frontier profiles their thesis: **no element of the composition is
free — each is paid for by an architectural or training choice.** Two rows remain open.

| element | price | evidence |
|---|---|---|
| adjacent (1-hop) recall | attention — every architecture aces adjacent readout (1.00) | [consolidated §3](factworld-consolidated.md); archived provenance phases/01 §3.2 |
| deferred recall | product recurrence — the transformer aces adjacent, fails deferred (0.19 vs gdp_hybrid 0.73) | consolidated §3; phases/01 §3.2 (atomic format: gdp_pure 1.00 attention-free vs transformer 0.48) |
| last-write state | recurrence, ordered by form — fprm (product recurrence) 1.00 @B6 / 0.97–0.98 seed-consistent @B16 on the binding leg, over gdp_hybrid (0.56 @B6) over transformer (0.23 @B6); at B24 fprm stops fitting (0.20, loss ≥1.0) and only the gated hybrid holds (0.67) | this report, breadth sweep above |
| non-abelian state (formation) | dense per-step supervision — a state checkpoint every ≤2 events, architecture-independent | consolidated §8; [experiments §1](../docs/experiments/README.md); archived provenance phases/02 §4 |
| non-abelian state (length extrapolation) | recurrent hybrid — gdp_hybrid 0.75 @L64; fprm (0.19) and transformer (0.22) solve in-distribution but collapse past train length | experiments §1 |
| depth extrapolation | **open** — no measured choice buys it: trained at chain depths 2–3, all three architectures read at or below the 1/6 guess at depths 4–5 (fprm 0.20/0.21, transformer 0.22/0.06, gdp_hybrid 0.02/0.00) | this report, local chain table above |
| local composition (value leg) | **open** — value ≤0.17 in all 45 breadth-sweep runs (at/below the 1/pool guess), even on binding-solved seeds of all three architectures | this report, breadth sweep above |

The two open rows are the instrument's active edge: nothing measured so far buys depth
extrapolation, and no local training choice yet converges the value leg of the composed cell
outside the staged-curriculum recipe — and on v2 that recipe converges it only for gdp_hybrid
at d768×8 (0.73; the small and large cells of the compute-matched sweep fail the value leg too).

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
binding_only @L16, scaffolded @L16, replicate @L16; n=100; effort=none; 96-token cap; hard "Answer:" contract line
with last-line extraction), `sanity` (recall_copy_v1 @L6, conflict_v1 @L4, n=30), `recall_load`
(recall_copy_v1 @L64 with the agent pool scaled to the length — pool 64; n=50; contract;
96-token cap), and `chain_instant` (chain_v1 d16 on the same k=33 staircase items as the
thinking d16 cell; n=25; contract; 96-token cap). Thinking: `chain_nowrap` (k=2d+1 staircase;
the reported cell is d128, k=257; n=25; 16,384 tokens) and `s5_concrete` (the reported cell is
L256; n=25; 16,384 tokens; effort=high throughout).

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

**Cost.** The full 529-cell history carries an estimated $237.90 of API spend; the 45-cell
zero-budget v2 battery (9 models x 5 legs, 4,500 calls) cost an estimated $5.92. Adding one model
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
  `results/benchmark/history.jsonl` (zero-budget battery: run `bench_v2_zb2_20260709`;
  scaffolded leg: `bench_20260710_124904`; chain/s5: `bench_v2_20260708`; recall-under-load and
  chain d16 instant: `bench_20260710_frontier_rows`).
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
  scaffolded (E1b) leg is on the same `composite_copy_v2` items as the composed cells (run
  `bench_20260710_124904`). The OpenRouter grid and reasoning analyses in
  [`factworld-consolidated.md`](factworld-consolidated.md) and the README, and the local
  from-scratch training numbers, are on the retired v1 family; their re-measure on v2 is tracked
  in [#11](https://github.com/ianbarber/factworld/issues/11).
