# Experiments — index and synthesis

FactWorld's experiment program on the natural-language format. Each experiment
has a script, a results file, and a finding below. The four pieces here close the power,
format-fairness, architecture-independence, weaning, and test-time-compute questions raised
in review; sections 6–15 log the frontier-benchmark arc (2026-07-05 → 07-10) — completion
budgets, task validity (chain wrap, give-stream recency), answer contracts, thinking-budget
elicitation, breadth-vs-length composition probes, operating-point calibration, and the local
breadth mirror; sections 16–22 log the issue-#11 v2 re-measures and the commutative-rung
calibration (2026-07-10).

## 1. Dense-vs-sparse state supervision (the s5 deficit) — `experiment_dense_supervision.py`

10-seed sparsity sweep, gdp_hybrid. K = holder supervised every K events; guided free-run eval.

| K | value @L16 | value @L64 | conv @L16 |
| --- | --- | --- | --- |
| 1 (dense) | **1.00**±0.00 | **0.75**±0.22 | 10/10 |
| 2 | 0.98±0.03 | 0.40±0.26 | 10/10 |
| 4 | 0.19±0.02 | 0.20 | 0/10 |
| 8 | 0.21±0.02 | 0.20 | 0/10 |

**Finding:** the s5 deficit moves under dense supervision (1.00 in-distribution, 10/10 converge) and
floors at K≥4 — a sharp, architecture-agnostic learnability threshold. Bimodality is at *length
extrapolation* (L64), not in-distribution. Doc: `dense-supervision-results.md`.

### Architecture-independence (fprm, transformer at K∈{1,8}, 5 seeds)

| arch | K=1 @L16 | K=1 @L64 | K=8 |
| --- | --- | --- | --- |
| gdp_hybrid | 1.00 (10/10) | **0.75** | 0.20 |
| fprm | 1.00 (5/5) | 0.19 | 0.20 |
| transformer | 0.86 (4/5) | 0.22 | 0.20 |

**Finding:** the *threshold* is architecture-independent (all floor at K≥4, all form the circuit
in-distribution at K=1). **Length extrapolation is not** — only the recurrent hybrid (gdp_hybrid)
extrapolates the learned circuit (0.75@L64); fprm and transformer solve in-distribution but don't
generalize in length.

## 2. Composition — format-fair API (E1b) — `experiment_autoregressive.py --composite_format`

n=100, composite_copy_v1@L16, output-format instruction given. Value accuracy:

| model | none | structured | binding(holder) | scaffolded |
| --- | --- | --- | --- | --- |
| kimi-k2.6 | **0.97** | 0.00 | 0.99 | 1.00 |
| glm-5.2 | **0.75** | 0.02 | 0.98 | 1.00 |
| llama-3.3-70b | 0.63 | 0.00 | 0.34 | 0.93 |
| deepseek-chat | 0.13 | 0.00 | 0.60 | 0.99 |
| gpt-4o-mini | 0.14 | 0.00 | 0.28 | 1.00 |

**Finding:** reasoning models (kimi/glm) solve composition given the format; non-reasoners don't;
**structured CoT actively hurts** (0.00 for all). The recall ceiling is universal (0.93–1.00). The
"composition deficit" is a reasoning-model advantage, and explicit self-produced intermediates are
counterproductive. Doc: `autoregressive-api-results.md`.

## 3. Composition routing vs state-tracking (local) — `experiment_composite_dense.py`

5 seeds, composite_copy_scale_v1, gdp_hybrid. Three metrics per condition:

| condition | free-run holder | free-run value (e2e) | value (given correct trace) |
| --- | --- | --- | --- |
| answer-only | 0.38 | 0.38 | **1.00** |
| dense (holder supervised) | 0.40 | 0.40 | 1.00 |
| dense→wean | 0.37 | 0.37 | 1.00 |

**Finding:** routing is *not* the deficit — given the correct holder, every model recalls (1.00). The
deficit is **generating the holder** (free-run holder ≈ free-run value ≈ 0.38; they track perfectly).
Crucially, **dense holder supervision does NOT fix the composite's holder leg** (0.40 vs 0.38),
unlike s5 where it reaches 1.00 — composite's last-write-wins-over-4-objects binding is harder to
unroll than s5's single-role tracking. So composition and s5 are **distinct deficits** with distinct
fixes. Weaning to answer-only does not preserve a circuit (none forms to begin with here).

## 4. Test-time compute — strong null — `experiment_self_correct.py`

K=2 partial-circuit model, L64, iterative self-correction (3 rounds of "check and regenerate holder"):

| round | holder | value |
| --- | --- | --- |
| 0 | 0.80 | 0.80 |
| 1 | 0.80 | 0.80 |
| 2 | 0.80 | 0.80 |
| 3 | 0.80 | 0.80 |

**Finding:** iterative self-correction gives **exactly zero lift** (flat across rounds), on top of
the earlier majority-vote null — for *local* (non-reasoning) models. **Caveat:** this does NOT settle
whether test-time compute helps in general. Reasoning models (kimi/glm) solved composite in E1b
(0.97/0.75) with their background reasoning *on* — that IS test-time compute working. What these
local probes show is that *explicit* CoT prompting and *sampling-based* self-correction don't help a
model that lacks implicit reasoning ability. Whether background reasoning effort helps the API
reasoners is tested directly in the reasoning on/off sweep below.

## Synthesis

| deficit (what fails) | what it is | what moves it | what doesn't |
| --- | --- | --- | --- |
| **composition** | generating the holder (last-write-wins over objects) | **background reasoning + format** (kimi 0.22→0.98, glm 0.14→0.81 with effort) | explicit CoT prompting (hurts), dense holder supervision, local self-correction |
| **s5 / non-abelian** | tracking a single role through permutations | **dense per-step supervision → wean to answer-only** (wean_mixed 8/8), then gdp_hybrid extrapolates 4–8×; at the frontier, **reasoning + a concrete rendering** (consolidated report Appendix A) | reasoning effort under the token rendering (floor at all levels), sparse/answer-only |
| **recall (value)** | — | given the holder, trivially solved (0.93–1.00) | — |

**Two dissociations, both now clean:**
- **Composition is movable by test-time compute** (reasoning) for strong models; **s5 under the
  token rendering is not** — locally it needs training-time supervision density, and that circuit
  can be *weaned* to label-free deployment; at the frontier the lever is reasoning plus a concrete
  rendering (consolidated report Appendix A).
- Architecture (gdp_hybrid vs fprm vs transformer) gates **length extrapolation** of a learned s5
  circuit, not its formation.

What never helps: explicit structured CoT prompting, and sampling/self-correction on non-reasoning
models. The levers are **reasoning strength** (composition), **supervision density + weaning** (s5),
and **recurrent architecture** (s5 extrapolation).

**Open question (the confound) — RESOLVED by the reasoning sweep.** kimi/glm solving
composition *was* test-time compute working. The reasoning on/off/levels sweep
(`reasoning-results.md`) shows a clear dose-response: composite value climbs with effort
(kimi 0.22→0.96→0.98; glm 0.14→0.74→0.81) while **s5 under the token rendering stays at floor
regardless of effort** (under a concrete rendering with an 8192-token budget, reasoning solves it
— `results/s5_horizon_recheck_20260705.jsonl`). So: **background reasoning (test-time compute)
IS a lever for composition; for s5 it works only combined with a concrete rendering.** What does
not help either deficit: explicit structured CoT prompting (hurts), and sampling/self-correction on
non-reasoning local models.

## 5. Weaning bridge — deep-dive (8 seeds) — `experiment_weaning.py`

Can a dense-learned s5 circuit survive weaning to answer-only, and does weaning change extrapolation?

| arm | L16 | L32 | L64 | L128 | conv @L16 |
| --- | --- | --- | --- | --- | --- |
| dense_only | 1.00±0.00 | 0.68±0.28 | 0.61±0.27 | 0.50±0.26 | 8/8 |
| wean_mixed:{1,2,4,inf} | 1.00±0.00 | 0.61±0.33 | 0.50±0.34 | 0.46±0.33 | 8/8 |
| wean_mixed:{1,inf} | 0.99±0.01 | 0.69±0.30 | 0.54±0.33 | 0.48±0.29 | 8/8 |
| wean_mixed:{1,4} | 1.00±0.00 | 0.68±0.29 | 0.53±0.33 | 0.47±0.33 | 8/8 |
| answer_only | 0.19 | 0.21 | 0.19 | — | 0/8 |

**Findings:** (1) **the circuit survives weaning** — every mix converges 8/8 @L16 free-run, no
 deploy-time labels. (2) **weaning does NOT improve extrapolation over dense-only** — all mixes track
 dense within noise (L128 0.46–0.53 vs dense 0.50). The win is label-free *deployment*, not better
 length generalization. (3) **the specific mix barely matters** — the key is just *some* answer-only
 exposure alongside dense. Deployment recipe: train dense → fine-tune on any mix including answer-only
 → deploy answer-only. Doc: `weaning_deep_results.md`.

Two clean dissociations: one movable by supervision density (s5, local), one movable by
base-model reasoning strength (composition). Architecture matters only for length extrapolation
of a *learned* s5 circuit (gdp_hybrid wins).

## 6. s5 completion-budget recheck — `experiment_s5_framing.py`

max_new_tokens=8192 (the script's default is 16), V1_concrete rendering, reasoning on, n=30/cell:

| model | L | relaxed | empty preds |
| --- | --- | --- | --- |
| glm-5.2 | 32 | **1.00** | 0/30 |
| glm-5.2 | 64 | 0.97 | 0/30 |
| glm-5.2 | 128 | 0.90 | 2/30 |
| kimi-k2.6 | 16 | 1.00 | 0/30 |
| kimi-k2.6 | 32 | 0.83 | 5/30 |

The same V1_concrete cell at the 16-token default (`docs/openrouter/s5-horizon.jsonl`) reads
0.10@L64 with **27/30 empty predictions**.

**Finding:** reasoning-model cells demand an explicit large completion budget with a published
empty-prediction rate — the per-cell empty rate is the diagnostic that separates "wrong" from
"cut off". At 8192 tokens glm holds 0.90–1.00 through L128 under the concrete rendering; the
16-token cell's 0.10@L64 measures truncation, not capability. Data:
`results/s5_horizon_recheck_20260705.jsonl`.

## 7. chain_v1 cycle-wrap validity + no-wrap depth — `test_chain_validity.py`, `experiment_v2_pilot_chain.py`

chain_v1's pointer map is a single k-cycle (k=6): at depth ≥ 6 gold collapses to
`nxt^(depth mod 6)(start)`, so effective difficulty is depth mod 6, and depth ≡ 0 (mod 6) is
the identity — the wrapped pilot cells at depths 6/12/24 have gold == start on 30/30 items
(`results/chain_reasoning_pilot*_20260705.jsonl`). The generator raises at depth ≥ k unless
wrap is explicitly opted in (`chain_allow_wrap=True`); deep chains run scaled, benchmark
protocol k = 2·depth+1, which also prices the backward walk at depth+1 hops (k = depth+2
would leave gold two reverse lookups from start). No-wrap pilot (gold ≠ start asserted before
any spend), effort=high, n=15/cell:

| model | depth 16 | depth 32 |
| --- | --- | --- |
| glm-5.2 | 0.80 | 0.13 |
| kimi-k2.6 | 1.00 | — |
| gpt-5.4 | **1.00** | **1.00** |
| opus-4.8 | **1.00** | **1.00** |

**Finding:** true depth separates models where the wrapped task could not. glm's d32 failure
has a hop-miscount fingerprint: 9 of 13 wrong answers are exactly one hop past gold (chain
position 33 of 32), so the misses are step-counting errors, not lookup failures. Data:
`results/v2_pilots/pilot1_chain_nowrap.jsonl`; gate `tests/test_chain_validity.py`.

## 8. Zero-budget answer contract — `experiment_v2_pilot_contract.py`

composite_copy_v1@L16, effort=none, n=50, hard one-line contract ("Reply with only one line:
Answer: ...") with last-Answer-line extraction; base budget 96, one-shot escalation to 512 on
finish=length:

| model | relaxed | diagnostics |
| --- | --- | --- |
| sonnet-5 | 0.72 | contract_rate 1.00 |
| opus-4.8 | **0.90** | contract_rate 1.00 |
| kimi-k2.6 | 0.82 | residual in-content CoT: 16% of calls > 350 ctok at cap 512 |
| grok-4.20 | 0.04 | 28/50 finish=content_filter |

The same sonnet cell under raw munging (run bench_20260706, n=50): relaxed 0.00, contains
0.92 — the answers are present, wrapped in working.

**Finding:** instant-regime scores are meaningless without a format-fair extraction contract —
raw munging bounds sonnet anywhere in 0.00–0.92; the contract pins it at 0.72. Provider safety
filters are a measurable failure mode: xAI's bio filter (SAFETY_CHECK_TYPE_BIO — the g/v token
soup reads as gene nomenclature) deterministically blocks 28/50 composite prompts, verified on
grok-4.20 and grok-4.3 (roster note in `factworld/benchmark.py`). Data:
`results/v2_pilots/pilot2_contract.jsonl`.

## 9. Anthropic thinking-budget probe — `experiment_v2_pilot_anthropic_budget.py`

s5_concrete@L128, sonnet-5 (n=15) and opus-4.8 (n=8): effort=high vs explicit
`reasoning.max_tokens`:

| model | arm | relaxed | mean rtok |
| --- | --- | --- | --- |
| sonnet-5 | effort=high | **1.00** | 1958 |
| sonnet-5 | thinking 4096 | 1.00 | 1721 |
| sonnet-5 | thinking 16000 | 1.00 | 2342 |
| opus-4.8 | effort=high | **1.00** | 2110 |
| opus-4.8 | thinking 4096 | 1.00 | 1884 |

**Finding:** effort=high is a valid Claude elicitation; explicit thinking budgets buy nothing
(every arm 1.00, rtok 1.7–2.3k). The binding constraint is *visible* working — ~12k mean /
15.6k max ctok per call — so `max_tokens` must cover visible output on top of the thinking
budget: the arm with 2k visible headroom (6144 total, 4096 thinking) scores 0.00 with 15/15
finish=length (kept as `pilot3_anthropic_budget_rejected.jsonl`). Data:
`results/v2_pilots/pilot3_anthropic_budget.jsonl`.

## 10. Uniform last-write sampler (composite/binding v2) — `test_composite_v2.py`, `validate_suite.py`

The v1 give-stream sampler draws every event's object uniformly, so the queried object's
resolving write sits ~Geometric(1/4) from the stream *end* at every L; the v2 sampler
(`TaskSpec.last_write_uniform`) picks the queried object first and places its last write
uniformly over [0.1·L, L−2]. The strong-recency one-liner (last give's recipient + that
holder's fact):

| task | L16 | L64 |
| --- | --- | --- |
| composite_copy_v1 | 0.325 | 0.225 |
| composite_copy_v2 | 0.060 | 0.055 |

(chance = 1/16 = 0.063; `validate_suite.py` gates this baseline on every give-stream task.)
Re-measure on the de-skewed task (run bench_v2_zb2_20260709, zero-budget contract, n=100):

| model | L16 | L64 |
| --- | --- | --- |
| opus-4.8 | **0.72** | **0.43** |
| sonnet-5 | 0.62 | 0.32 |
| deepseek-v4-pro | 0.44 | 0.19 |
| glm-5.2 | 0.35 | 0.16 |
| nemotron-3-ultra | 0.33 | 0.12 |
| qwen3.7-max | 0.24 | 0.08 |
| object-filter floor E[1/w] | 0.41 | 0.15 |

(sonnet's escalated @512 diagnostics read 0.76/0.66; canonical = first attempt @96.)

**Finding:** the v1 family measured recency adoption at the low end; v2 separates object
filtering from genuine last-write resolution, and the floor row is part of the instrument —
the cheap tier sits at (deepseek 0.44) or below (qwen 0.24) the 0.41 floor while opus 0.72
and sonnet 0.62 clear it. Local mirror (RTX 5090, gdp_hybrid d256×4, 4,000 steps): binding v1
0.99/0.77/0.70 → v2 0.82/0.21/0.23 @L16/32/64 — the local headroom was also recency-inflated.
Data: `results/benchmark/history.jsonl` (run bench_v2_zb2_20260709); local smoke
`results/local_smoke_20260709/`; sampler pins `tests/test_composite_v2.py`.

## 11. Composition frontier: breadth, not length — `experiment_composite_frontier.py`

glm-5.2 on composite_copy_v2: thinking (effort=high; budgets scale with L — 16384 through
L256, 32768 at L512+) vs instant (effort=none, contract, 96 tokens); k=32/pool16, two
replicate cells of n=25 per L (per-L mean):

| arm | L64 | L128 | L256 | L512 | L1024 |
| --- | --- | --- | --- | --- | --- |
| thinking | **0.98** | **0.98** | 0.94 | 0.96 | 0.94 |
| instant | 0.24 | 0.02 | 0.00 | 0.06 | 0.06 |
| object-filter floor | 0.14 | 0.08 | 0.05 | 0.02 | 0.01 |

Breadth rung: k=64/pool64@L1024 thinking = 0.64 — a budget-censored lower bound (7/25 calls
at the 32768-token reasoning cap; the one cap-escaping call, 54k rtok, solved its item).

**Finding:** composition-under-thinking is breadth-bound, not length-bound, in the probed
range — accuracy is flat to L1024 at k=32 while reasoning spend grows only ~linearly
(≈5–10 rtok/event), and the doubled-breadth rung drops to 0.64. Failure anatomy: every
non-empty thinking wrong is an *earlier write of the correct object* with consistent value
lookup; instant falls below the object-filter floor by L128 (0.02 vs 0.08) with
primacy-dominated picks (mostly the object's first write). Working-set breadth is the
organizing hypothesis for the frontier benchmark (v3 probes in flight). Data:
`results/composite_frontier_20260709.jsonl`.

## 12. Operating-point calibration (v3 probes) — `experiment_composite_frontier.py`, `experiment_v3_probe_chain.py`

Five probes pin where each benchmark regime reads mid-scale. Calibration facts, not headline
rows; the benchmark's difficulty settings follow from them.

**P1 — breadth × length under thinking (glm-5.2, composite_copy_v2, effort=high).** The composed
cell stays near ceiling across the probed grid: 0.96 @k128/L64, 0.92 @k256/L64, 0.88 @k128/L256,
0.84 @k256/L256, 0.84 @k256/L512, 0.80 @k64/L1024 (n=25 each). What moves is spend, not
accuracy: ctok/call 756 → 3,347 → 4,880 → 21,059 across the grid, with 4/25 finish=length at
k64/L1024. No budget-feasible (k, L) places thinking composite mid-scale; the thinking regime
reads through the state-stress rows (chain d128 @k257, s5 @L256). Data:
`results/v3_probes/p1_interaction_bridge.jsonl`.

**P2 — instant downsweep (glm-5.2, qwen3.7-max; k∈{8,16,24}, L64, n=50).** Shrinking breadth
raises the shallow floor faster than the score: glm 0.22/0.18/0.06 and qwen 0.30/0.12/0.12
against E[max-share] floors 0.47/0.32/0.30 — at or below floor everywhere. The instant regime is
informative only near canonical settings (L16, k=32, where the object-filter floor is 0.41 and
the strong tier clears it). Data: `results/v3_probes/p2_instant_downsweep.jsonl`.

**P3 — depth at fixed breadth (chain_v1, k=257, d=16, effort=high, n=15).** glm 0.93, qwen 1.00,
gemini-flash 1.00 — ceiling at shallow depth at the full k=257 pool (chance 0.004). The
`chain_nowrap` d128 staircase cells build k=2d+1=257, so this pins them as depth measurements,
not breadth measurements: depth at fixed breadth stays informative where the composed cell
saturates. Data: `results/v3_probes/p3_chain_fixedk.jsonl`.

**P4 — interference knob inert (glm-5.2, k=32/L64, m∈{8,16}).** Raising writes-per-object m
moves the E[1/w] floor 0.30 → 0.51 and the instant score with it (0.32 → 0.54, floor-shaped in
both cells); thinking holds 0.96 at both. The knob re-prices the floor without separating models.
Data: `results/v3_probes/p4_interference.jsonl`.

**P5 — billed vs reported thinking tokens (k=128/L256, n=5).** On the same cell, Anthropic
reports reasoning tokens far below billed completion: sonnet-5 2,128 ctok/call vs 182 rtok,
opus-4.8 590 vs 107; gpt-5.5 reports rtok ≈ ctok (671 vs 681). Cross-provider token-efficiency
comparisons must use billed completion tokens (the benchmark's ctok columns), never reported
reasoning tokens. Data: `results/v3_probes/p5_frontier_rtok.jsonl`.

**Finding:** the instant regime measures in-weights composition only near canonical settings;
the thinking regime saturates the composed cell at every budget-feasible setting and is measured
on the state-stress rows; interference is a floor knob, not a difficulty knob; and ctok is the
only fair spend unit across providers. Data: `results/v3_probes/`.

## 13. Local breadth mirror — `experiment_local_breadth.py`

The frontier breadth rungs, trained from scratch: `composite_copy_v2.scaled(k=2B,
recall_pool=B)`, m=4, fprm vs gdp_hybrid vs transformer, d256×4, 8k steps flat next-token
training (train L ∈ {4, 8, 16}), eval L16 (in-distribution) / L64 (extrapolation), B ∈ {6, 8,
12, 16, 24} × 3 seeds = 45 runs (RTX 5090). Relaxed match; per-leg content-token decomposition
(holder = binding leg, value = recall leg); pconv = seeds ≥0.9.

| B | arch | L16 (pconv) | L64 | holder/value @L16 | @L64 |
| --- | --- | --- | --- | --- | --- |
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

fprm solves the binding leg @L16 on 9/15 seeds: 1.00/1.00/1.00 @B6, 0.20/0.99/0.99 @B8,
1.00/0.17/0.75 @B12, 0.97/0.98/0.98 @B16 (the only seed-consistent solve in the sweep). At B24
it stops fitting the training distribution (final loss 1.02–1.10; holder 0.13–0.30) while
gdp_hybrid still fits everywhere (loss 0.05–0.14; holder 0.67 @B24). The binding leg is bimodal
for gdp_hybrid from B12 up: per-seed @L16 it reads 1.00/0.10/0.21 @B12, 0.07/0.99/0.18 @B16,
0.98/0.04/1.00 @B24 — single seeds solve binding outright while the rest sit near floor, so
per-rung means stop being readable above B8. The transformer never exceeds 0.26 on any leg and
no longer fits the training distribution at B24 (final loss 1.49–1.65). At L64 fprm keeps the
most binding at B6–B16 (0.24–0.41 per-rung means vs the 0.15 object-filter floor; gdp_hybrid
0.09–0.20) — far from its retired-v1 flagship (binding_v1 0.94 @L64): the arch ordering
survives on the v2 sampler, the magnitude was v1 recency credit.

**Finding:** at d256×4 / 8k flat training the composed cell reads floor at every rung (pconv
0/45; best single run fprm 0.17 @L16 = solved binding × a 1/6 pool guess) and the instrument
reads through the legs: the binding leg orders fprm > gdp_hybrid > transformer through B16 with
an inversion at B24 where only the gated hybrid still fits; the local operating point stays B8,
the largest rung where gdp_hybrid — the architecture the calibration was set on — is mid-scale
seed-consistently (0.41 @L16, seeds 0.34–0.48, vs transformer 0.17; 1/k agent-guess 0.06); and
the composition deficit sits on the recall leg for all three architectures (value ≤0.17 in all
45 runs, at/below the 1/pool guess on binding-solved seeds), the same leg the d768
staged-curriculum decomposition localizes. Local composed-cell numbers require the
staged-curriculum recipe (consolidated §5); flat training does not converge composition at any
rung. Data: `results/local_breadth/sweep_runs.jsonl`, `sweep_summary.{md,json}`.

## 14. Frontier rows: recall under load + chain d16 instant — `run_frontier_benchmark.py` (facets `recall_load`, `chain_instant`)

Two new instant cells per roster model (run `bench_20260710_frontier_rows`), filling the
benchmark's recall-under-load and within-depth regime-contrast gaps.

**recall_load** — `recall_copy_v1` @L64 with the agent pool scaled to the length (pool 64,
verified 64 distinct agents/facts per item; chance 1/64 ≈ 0.016), effort=none + contract,
96-token cap, n=50: **all nine models 1.00**, contract 1.00 and covert working 0.00 everywhere
(kimi rtok_any 0.38 at ~0.4 tok/call). Single-query deferred recall is at ceiling out to
pool-64 for this roster — documented ceiling, kept as the recall-under-load row.

**chain_instant** — `chain_v1` d16 on the same k=33 staircase items as the thinking d16 cell
(chance ≈ 0.03), effort=none + contract, 96-token cap, n=25; canonical = first attempt @96,
escalated @512 published as a diagnostic:

| model | instant @96 (canonical) | finish=length @96 | @512 diagnostic | thinking (chain_nowrap d16) |
| --- | --- | --- | --- | --- |
| opus-4.8 | 0.00 | 25/25 | 0.96 | 1.00 |
| sonnet-5 | 0.28 | 18/25 | 0.96 | 1.00 |
| gpt-5.5 | 0.08 | 0/25 | — | 1.00 |
| gemini-3.5-flash* | 0.00 | 25/25 | 1.00 | 1.00 |
| kimi-k2.6 | 0.32 | 16/25 | 0.96 | 1.00 |
| qwen3.7-max | 0.00 | 0/25 | — | 1.00 |
| glm-5.2 | 0.00 | 1/25 | — | 0.96 |
| deepseek-v4-pro | 0.00 | 0/25 | — | 1.00 |
| nemotron-3-ultra | 0.00 | 0/25 | — | 0.44 |

(* = effort=minimal; reasoning cannot be disabled.)

**Finding:** the within-depth regime contrast is clean — every model that answers within the
instant budget floors at d16 (0.00–0.08 vs chance 0.03) while the same items read 0.96–1.00
under thinking (nemotron 0.44); the four models with majority finish=length at 96 tokens are
spending the budget on working, not answers, and their @512 diagnostics (0.96–1.00) are short
visible working, not in-weights answers. "Multi-hop needs the thinking regime" is now measured,
not assumed. Runner note: plain-contract cells pick the contract line by spec family
(composite / recall / chain), failing loudly on unsupported families. Data:
`results/benchmark/history.jsonl`; rendered `docs/benchmark/results.md`.

## 15. Local chain architecture comparison — `sweep.py --tasks chain_v1`

chain_v1 (recall ∘ recall; the k=6 pointer map) at the canonical baseline recipe: fprm vs
transformer vs gdp_hybrid, d320×4, 8k steps, registered spec train depths (2, 3) / eval depths
(4, 5), eval_n=200, seeds 0–2 (RTX 5090; 9 runs). Relaxed match; agent-guess floor 1/6 ≈ 0.17;
pconv = seeds ≥0.9. fprm's first chain datum, and the multi-architecture chain comparison on 3
seeds.

| arch | d4 mean±std (pconv) | d5 mean±std (pconv) | final loss |
| --- | --- | --- | --- |
| fprm | 0.20±0.01 (0%) | 0.21±0.02 (0%) | 0.38–0.40 |
| transformer | 0.22±0.01 (0%) | 0.06±0.02 (0%) | 0.40–0.41 |
| gdp_hybrid | 0.02±0.01 (0%) | 0.00±0.00 (0%) | 0.23–0.25 |

**Finding:** depth does not extrapolate for any architecture — no run converges (pconv 0/9) and
no cell clears the 1/6 guess. fprm and the transformer sit at the guess floor at d4; fprm stays
there at d5 while the transformer falls below it (0.03–0.09 per seed); gdp_hybrid fits the
training distribution best (lowest final loss, 0.23–0.25) yet scores 0.00–0.03 at both held-out
depths — a depth-specific circuit that is systematically wrong one hop past training, not a
guesser. The depth-extrapolation row of the price table stays open with all three architectures
now measured; contrast the frontier, where the same composition solves at d16 only in the
thinking regime (§14). Data: `results/local_chain_arch_20260710.{jsonl,md,json}`.

## 16. Scaffolded leg (recall-given-holder) on v2 — `run_frontier_benchmark.py` (facet `zero_budget`, leg `scaffolded`)

The gap definition's premise — the recall half of the composed cell is free — previously cited
v1 items. Re-measured on the same `composite_copy_v2` items as the composed cells: all nine
roster models, @L16, n=100, instant protocol (effort=none, contract, 96-token cap), run
`bench_20260710_124904`.

| model | relaxed | empty rate |
| --- | --- | --- |
| opus-4.8 | 1.00 | 0.00 |
| sonnet-5 | 1.00 | 0.00 |
| gpt-5.5 | 1.00 | 0.00 |
| gemini-3.5-flash | 1.00 | 0.00 |
| glm-5.2 | 1.00 | 0.00 |
| deepseek-v4-pro | 1.00 | 0.00 |
| nemotron-3-ultra | 0.99 | 0.00 |
| kimi-k2.6 | 0.98 | 0.02 |
| qwen3.7-max | ⊘ 0.02 | 0.98 |

Scorer note: the scaffolded prompt injects the resolved holder, and models legitimately echo it
before the value, which the strict prefix-commit extractor scored as wrong (opus's first
attempt read 0.05 on echoes of the injected holder). The contract extractor in
`scripts/run_frontier_benchmark.py` now tolerates a holder-prefixed answer span on the
scaffolded leg; the one mis-scored record was purged (backup
`results/remeasure_v2/history.pre_scaffold_fix.bak`) and the cell re-run.

**Finding:** recall-given-holder ≈ 1.0 re-founded on the v2 sampler — 0.98–1.00 for every
measurable roster model; qwen3.7-max returns an empty completion on 98/100 scaffolded calls
(finish=stop) and is ⊘ on this leg under the contract. The composition gap's foundation now
cites the same items in both legs. Data: `results/benchmark/history.jsonl`; rendered
`docs/benchmark/results.md`.

## 17. Reasoning dose-response on v2 — `experiment_reasoning.py`

The v1 dose-response (§ consolidated §4) re-measured on `composite_copy_v2` @L16 (k=32/pool16),
n=50 per cell, answer-span extraction with holder/value decomposition:

| model | none | low | medium | high | holder/value @none | @high |
| --- | --- | --- | --- | --- | --- | --- |
| kimi-k2.6 | 0.72 | 1.00 | 1.00 | 1.00 | 0.42 / 0.40 | 0.98 / 0.98 |
| glm-5.2 | 0.38 | 0.92 | 0.96 | 0.98 | 0.22 / 0.22 | 0.80 / 0.80 |

**Finding:** the dose-response survives the de-skewed sampler and is monotone for both models.
The effort=none holder legs (0.42 / 0.22) sit at or below the 0.41 object-filter floor —
object filtering, not composition (kimi's none-arm carries its covert-reasoning caveat) — and
low effort already recovers 0.92–1.00 on the answer span. Data:
`results/reasoning_sweep_20260710_125924.jsonl`.

## 18. Long-context composition on v2, kimi arm — `experiment_composite_frontier.py`

The §11 glm length profile (thinking flat 0.94–0.98 to L1024 at k=32) gets its second model:
kimi-k2.6, thinking, k=32/pool16, n=25 per cell — @L256 (effort=high, 16,384 tokens) relaxed
**1.00**, Wilson 95% [0.87, 1.00], ctok median 5,592 (~$0.56); @L512 (32,768 tokens) relaxed
**0.96**, [0.80, 0.99], ctok median 12,123 (~$1.21). Empty 0.00 and finish=stop 25/25 on both
cells; reasoning spend roughly doubles with length (rtok median 5,358 → 12,119, ≈ 21–24
rtok/event) while accuracy holds. Longer kimi cells are predicted-ceiling and stay unbought.

**Finding:** kimi matches glm's flat thinking length profile at both measured points — the
"reasoning holds composition at long context" claim (consolidated §7) now rests on v2 items
for both models. Data: `results/composite_frontier_20260710.jsonl`.

## 19. Commutative rung calibration — `experiment_commutative_local.py`, `experiment_commutative_frontier.py`

`commutative_v1` fills the taxonomy rung between last-write and non-abelian state: per-entity
dial accumulation mod k (k=5 positions; every event matters, order does not; distractor
entities force per-entity filtering). Validity gate (n=500): chance 1/k = 0.200; four dedicated
shallow adversaries (initial-only, last-turn, entity-blind-sum, count-mod-k) max at 0.224, all
gated ≤ 0.4 (`scripts/validate_suite.py`; determinism/oracle/gate tests in
`tests/test_commutative_v1.py`, 8/8).

Local (d256×4, 8k steps, train L ∈ {4, 8, 16}, 3 seeds, RTX 5090), relaxed match:

| arch | L16 | L32 | L64 |
| --- | --- | --- | --- |
| fprm | 0.17±0.00 | 0.18±0.02 | 0.20±0.01 |
| gdp_hybrid | 0.21±0.02 | 0.19±0.02 | 0.21±0.02 |
| transformer | 0.16±0.01 | 0.20±0.01 | 0.20±0.02 |

Every run at chance (pconv 0/9), and the documented trace contingency (gdp_hybrid seed 0,
worked trace) also reads chance (0.22 / 0.17 / 0.20). Frontier calibration (n=25 per cell,
$0.21): instant (effort=none, contract) floors both probes — glm 0.24 @L16 / 0.12 @L64,
deepseek 0.20 / 0.12 — while thinking @L64 (effort=high, 8,192 tokens) discriminates:
deepseek 0.80, glm 0.52, neither at ceiling.

**Finding:** the rung is shallow-proof by construction and reads only in the thinking regime
at these settings — commutative aggregation does not form locally at the binding operating
point (dense-supervision analog untested beyond the trace contingency), and instant frontier
cells sit at the floors. Placed in the taxonomy (AGENTS.md, README, frontier report
Components) as experimental until a full roster run. Data:
`results/commutative_local/{sweep,trace}_runs.jsonl`, `results/commutative_frontier/runs.jsonl`.

## 20. Reference baselines re-collected on v2 — `collect_baselines.py`

`docs/results.md` (the d320×4 / 8k-step / seed-0 reference table) rebuilt with `binding_v2` and
`composite_copy_v2` replacing the retired v1 specs; recall_copy/conflict/chain are unchanged
tasks re-trained under the same recipe. Headlines of the rebuilt table: binding_v2 orders
gdp_hybrid (0.99 @L16, 0.85 @L32) over gdn_hybrid (0.74 @L16) over transformer/gru (floor);
the composite row stays at floor for every architecture at every length — the flat baseline
recipe does not converge composition on v2 either, consistent with the §13 breadth sweep
(composed cells locally require the staged curriculum). Single-seed reference numbers: read
orderings against floors, not third-decimal differences (recall_copy's gdn_hybrid moved from
mid-scale to ceiling on re-train — seed variance at this scale). Data: `docs/results.md`
(rendered), 62-minute run, rc=0.

## 21. Staged-curriculum flagship re-measure — trace-mode protocol artifact — `experiment_curriculum_staged.py --use_trace`

The §5 flagship re-measure on v2 specs (3 archs × 3 seeds, d768×8, batch 128, 25k steps,
80k docs, eval_n=500) was launched with `--use_trace` — the v1 flagship it re-measures
(gdp 0.747, `use_trace=False`) was not. Under trace training, composite docs are
"prompt trace answer", so the model emits a ~16-token self-trace before the 2-token gold
answer, and the prefix-committed relaxed metric is structurally 0 for any trace-emitting model.
Composite relaxed read 0.000 on all nine runs while `contains` stayed high (gdp p5 0.981 /
p16 0.742 — containment leniency over the longer emission, tracking pool size). This is the
known artifact signature (relaxed 0 with contains ~1), and adjudication on the raw records
confirms artifact, not capability: the trace-aware tail decomposition is also low (gdp p16
holder 0.221 / value 0.023) and the tail scoring is itself corrupted by budget babble (binding
relaxed 0.999 on the same models vs tail-scored holder as low as 0.18). The v1-era trace-mode
control (`results/curriculum_staged_d768_b64_80k_trace.md`) already scored composite 0.00 /
holder 0.14 / value 0.02 — this run reproduced the known trace-mode failure, not the flagship.

**Finding:** composite capability is unmeasurable under the trace protocol — the runs are
excluded, not folded (per-example predictions and checkpoints were not stored, so rescoring is
impossible). The trace-free v2 flagship number comes from the scale sweep's medium cell (§22);
the corrected 3-seed/eval_n=500 curriculum measurement is queued: the identical command from
`scripts/gpu_queue_remeasure_v2.sh` without `--use_trace` (~12–25 GPU-h). Data:
`results/curriculum_staged_v2_d768.jsonl`.

## 22. Compute-matched scale sweep on v2 — `experiment_composite_scale.py`

The §5 flagship and its scale-robustness check, re-measured on the v2 staged specs
(`composite_copy_v2` pool-16 @L16 via `staged_specs()`, uniform last-write sampler; trace-free).
Same staged curriculum at three sizes, matched on compute, not parameters (shared
`(d_model, depth)`; fprm weight-tied at ~5–11× fewer params); 2 seeds, `train_n=80000`,
eval_n=200. Relaxed match, mean±std:

| arch | small (384×6) | medium (768×8) | large (1024×12) |
| --- | --- | --- | --- |
| **gdp_hybrid** | 0.12±0.08 | **0.73±0.01** | 0.21±0.21 |
| fprm | 0.12±0.05 | 0.03±0.01 | 0.03±0.02 |
| transformer | 0.01±0.00 | 0.01±0.01 | 0.00±0.00 |

The medium column is the exact §5 recipe (d768×8, batch 128, 25k steps, 80k docs) and stands as
the v2 flagship measurement: gdp_hybrid 0.720 / 0.745 per seed, holder 1.00 on both, contains ≈
relaxed (no artifact signature). Small gdp_hybrid solves binding (holder 1.0) but fails the
value leg (0.045 / 0.200) — v1's small 0.98±0.01 was flattered by the retired sampler. Large
gdp_hybrid is seed-bimodal with a genuine value-leg failure (seed 1: relaxed 0.000, holder
1.000, contains 0.000 — the gold value appears nowhere) and an unstable seed 0 (0.42 with holder
degraded to 0.68 on the batch-64 recipe). fprm's composed cell dies from medium up (v1's
0.253±0.178 does not carry); the transformer floors at every scale, contains ~0 too.

**Finding:** the v1 flagship claim survives on v2 but narrows — the staged curriculum converges
the composed cell for gdp_hybrid only, at d768×8 (0.732±0.013), and convergence is
non-monotone in scale; wherever binding trains and the composed cell fails, the value leg is
what collapses. Folded into consolidated §5 and the frontier report's local-regime lines. Data:
`results/composite_scale_20260710_221530.jsonl`.
