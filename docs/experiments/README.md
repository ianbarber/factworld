# Experiments — index and synthesis

FactWorld's experiment program, post-refactor (natural-language format). Each experiment
has a script, a results file, and a finding below. The four pieces here close the power,
format-fairness, architecture-independence, weaning, and test-time-compute questions raised
in review; sections 6–13 log the frontier-benchmark arc (2026-07-05 → 07-09) — completion
budgets, task validity (chain wrap, give-stream recency), answer contracts, thinking-budget
elicitation, breadth-vs-length composition probes, operating-point calibration, and the local
breadth mirror.

## 1. Dense-vs-sparse state supervision (the s5 wall) — `experiment_dense_supervision.py`

10-seed sparsity sweep, gdp_hybrid. K = holder supervised every K events; guided free-run eval.

| K | value @L16 | value @L64 | conv @L16 |
| --- | --- | --- | --- |
| 1 (dense) | **1.00**±0.00 | **0.75**±0.22 | 10/10 |
| 2 | 0.98±0.03 | 0.40±0.26 | 10/10 |
| 4 | 0.19±0.02 | 0.20 | 0/10 |
| 8 | 0.21±0.02 | 0.20 | 0/10 |

**Finding:** the s5 wall moves under dense supervision (1.00 in-distribution, 10/10 converge) and
floors at K≥4 — a sharp, architecture-agnostic learnability cliff. Bimodality is at *length
extrapolation* (L64), not in-distribution. Doc: `dense-supervision-results.md`.

### Architecture-independence (fprm, transformer at K∈{1,8}, 5 seeds)

| arch | K=1 @L16 | K=1 @L64 | K=8 |
| --- | --- | --- | --- |
| gdp_hybrid | 1.00 (10/10) | **0.75** | 0.20 |
| fprm | 1.00 (5/5) | 0.19 | 0.20 |
| transformer | 0.86 (4/5) | 0.22 | 0.20 |

**Finding:** the *cliff* is architecture-independent (all floor at K≥4, all form the circuit
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
"composition wall" is a reasoning-model advantage, and explicit self-produced intermediates are
counterproductive. Doc: `autoregressive-api-results.md`.

## 3. Composition routing vs state-tracking (local) — `experiment_composite_dense.py`

5 seeds, composite_copy_scale_v1, gdp_hybrid. Three metrics per condition:

| condition | free-run holder | free-run value (e2e) | value (given correct trace) |
| --- | --- | --- | --- |
| answer-only | 0.38 | 0.38 | **1.00** |
| dense (holder supervised) | 0.40 | 0.40 | 1.00 |
| dense→wean | 0.37 | 0.37 | 1.00 |

**Finding:** routing is *not* the wall — given the correct holder, every model recalls (1.00). The
wall is **generating the holder** (free-run holder ≈ free-run value ≈ 0.38; they track perfectly).
Crucially, **dense holder supervision does NOT fix the composite's holder leg** (0.40 vs 0.38),
unlike s5 where it reaches 1.00 — composite's last-write-wins-over-4-objects binding is harder to
unroll than s5's single-role tracking. So composition and s5 are **distinct walls** with distinct
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

| wall | what it is | what moves it | what doesn't |
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
not help either wall: explicit structured CoT prompting (hurts), and sampling/self-correction on
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
recall_pool=B)`, m=4, transformer vs gdp_hybrid, d256×4, 8k steps flat next-token training
(train L ∈ {4, 8, 16}), eval L16 (in-distribution) / L64 (extrapolation), B ∈ {6, 8, 12, 16,
24} × 3 seeds = 30 runs (RTX 5090). Relaxed match; per-leg content-token decomposition
(holder = binding leg, value = recall leg); pconv = seeds ≥0.9.

| B | arch | L16 (pconv) | L64 | holder/value @L16 | @L64 |
| --- | --- | --- | --- | --- | --- |
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

The binding leg is bimodal for gdp_hybrid from B12 up: per-seed @L16 it reads 1.00/0.10/0.21
@B12, 0.07/0.99/0.18 @B16, 0.98/0.04/1.00 @B24 — single seeds solve binding outright while the
rest sit near floor, so per-rung means stop being readable above B8. The transformer never
exceeds 0.26 on any leg and no longer fits the training distribution at B24 (final loss
1.49–1.65; gdp_hybrid 0.05–0.14 at every rung).

**Finding:** at d256×4 / 8k flat training the composed cell reads floor at every rung (best
single run 0.06 @L16; pconv 0/30) and the instrument reads through the legs: the local
operating point is B8 — the largest rung where the better architecture is mid-scale
seed-consistently on the binding leg (gdp_hybrid 0.41 @L16, seeds 0.34–0.48, vs transformer
0.17; 1/k agent-guess 0.06) — and the composition deficit sits on the recall leg (value ≤0.11
in all 30 runs, ≤0.02 even on binding-solved seeds), the same leg the d768 staged-curriculum
decomposition localizes. Local composed-cell numbers require the staged-curriculum recipe
(consolidated §5); flat training does not converge composition at any rung. Data:
`results/local_breadth/sweep_runs.jsonl`, `sweep_summary.{md,json}`.
