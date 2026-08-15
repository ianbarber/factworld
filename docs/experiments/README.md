# Experiments — index and synthesis

FactWorld's experiment program on the natural-language format. Each experiment
has a script, a results file, and a finding below. The four pieces here close the power,
format-fairness, architecture-independence, weaning, and test-time-compute questions raised
in review; sections 6–15 log the frontier-benchmark arc (2026-07-05 → 07-10) — completion
budgets, task validity (chain wrap, give-stream recency), answer contracts, thinking-budget
elicitation, breadth-vs-length composition probes, operating-point calibration, and the local
breadth mirror; sections 16–23 log the issue-#11 v2 re-measures and the commutative-rung
calibration (2026-07-10 → 07-11); sections 24–27 log the close-out cycle (2026-07-11 →
07-12) — raised completion budgets for ⊘ cells, the qwen contract-phrasing diagnosis, the
commutative roster adjudication, and the statistical-power checks; section 28 logs issue #11's
last item, the MOPD binding re-pin (2026-07-12); sections 29–32 log the s5_chain arc and the
validation of the benchmark built on it (2026-07-17 → 07-27) — the system-prompt probe behind
the engagement mark, the echo defect and distinct-path gate that produced the scored task, the
local battery and its audit, and what the frontier ranking does and does not resolve.

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
impossible). The trace-free v2 flagship number comes from the corrected rerun — the identical
command from `scripts/gpu_queue_remeasure_v2.sh` without `--use_trace` (§23); the scale sweep's
medium cell (§22) corroborates it. Data (excluded trace runs):
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

The medium column is the exact §5 recipe (d768×8, batch 128, 25k steps, 80k docs) and
corroborates the 3-seed flagship measurement (§23): gdp_hybrid 0.720 / 0.745 per seed, holder 1.00 on both, contains ≈
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

## 23. Staged-curriculum flagship on v2, trace-free — `experiment_curriculum_staged.py`

The corrected §21 rerun: the identical flagship command (3 archs × 3 seeds, d768×8, batch 128,
25k steps, 80k docs, eval_n=500, v2 staged specs) without `--use_trace`. Relaxed match,
mean±std over 3 seeds; pconv = seeds ≥0.9; holder/value at composite p16 @L16:

| arch | comp p16 @L16 (pconv) | per seed | holder / value | binding L16 | recall e/m/h | comp p5 |
|---|---|---|---|---|---|---|
| **gdp_hybrid** | **0.83±0.09 (1/3)** | 0.758 / 0.782 / 0.958 | 1.00 / 0.83 | 1.00±0.00 | 0.92 / 0.95 / 0.85 | 0.99±0.00 |
| fprm | 0.11±0.09 (0/3) | 0.056 / 0.036 / 0.234 | 1.00 / 0.11 | 1.00±0.00 | 0.63 / 0.39 / 0.20 | 0.38±0.29 |
| transformer | 0.00±0.00 (0/3) | 0.000 / 0.002 / 0.000 | 0.07 / 0.04 | 0.03±0.00 | 0.09 / 0.04 / 0.03 | 0.01±0.01 |

No artifact signature: contains ≈ relaxed on every gdp_hybrid seed (0.768 / 0.788 / 0.958) and
last-N 0.00 across the board. gdp_hybrid learns the composed cell on all three seeds — one
clears the ≥0.9 convergence bar, the other two read 0.758 / 0.782 — with the holder leg ≥ 0.998
everywhere. fprm solves binding on every seed (≥ 0.998) while the value leg stays collapsed
(0.109 mean), and its recall legs are seed-volatile (easy 0.954 / 0.164 / 0.772). The
transformer floors both legs on every task.

**Finding:** the §5 flagship on 3 seeds / eval_n=500 reads gdp_hybrid 0.833±0.089,
fprm 0.109±0.089, transformer 0.001±0.001 (composite p16 @L16, relaxed) — the same ordering and
the same value-leg localization as the sweep's 2-seed medium cell (0.732±0.013, §22), which
reads within one seed-std and stands as corroboration. This is the cell the consolidated §5
flagship table carries. Data: `results/curriculum_staged_v2_d768_notrace.jsonl`
(log `results/remeasure_v2/curriculum_notrace.log`).

## 24. Raised completion budgets for ⊘ cells — `run_frontier_benchmark.py --budget-override`

The three thinking cells that read ⊘ at 16,384 tokens with completion evidence (issue #17;
opus and sonnet s5_concrete @L256 emitted no visible answer on 25/25 calls; deepseek chain
d128 truncated at exactly the cap with high conditional accuracy on the calls that finished)
were rerun once at 32,768 tokens via a new tested `--budget-override facet:length:budget`
flag (the resume key includes max_new_tokens, so the rerun is a fresh cell; the render dedup
key does not, so it replaces the old cell in the tables automatically). Results (n=25 each):

| cell | @16,384 | @32,768 | diagnostics @32,768 |
| --- | --- | --- | --- |
| opus s5 @L256 | ⊘ (25/25 length) | **1.00** | 25/25 stop, empty 0.00, ctok mean 23,898 / max 28,986 ($15.82) |
| sonnet s5 @L256 | ⊘ (25/25 length) | **1.00** | 25/25 stop, empty 0.00, ctok mean 24,071 ($6.37) |
| deepseek chain d128 | ⊘ ‡ | ⊘ (0.08) | 19/25 length, empty 0.76, ctok median = cap ($0.69) |

**Finding:** the 16,384-token zeros on the two Claude s5 cells were pure truncation — both
models solve L256 outright given the ~24k tokens their traces need — while deepseek's chain
d128 is still budget-bound at 32,768, so its ⊘ stands with the budget stated. Nemotron's ⊘
cells were not raised: it trails every measured model at every s5/chain length where it does
answer, so the ⊘ is documented as a model trait, not a budget artifact (report §2 marks
note). Data: `results/benchmark/history.jsonl` run `bench_17_budget32k_20260711`.

## 25. Qwen scaffolded-leg ⊘: contract-phrasing interaction — `probe_qwen_scaffold.py`

Qwen3.7-max is ⊘ on the scaffolded (recall-given-holder) leg — empty extraction on 98/100
calls (§16) — while at ceiling on every other recall cell. A four-arm probe (n=10 per arm,
composite_copy_v2 @L16, $0.02, no history writes) isolates the mechanism: with the exact
published contract line ("Reply with only one line: Answer: <value>") compliance is 0/10 but
the gold value appears in the raw completion 10/10 (qwen answers in the system prompt's
'g13 v70' shape instead); a 512-token cap changes nothing; removing the contract line changes
nothing; a REWORDED contract line ("End your reply with exactly one line of the form
'Answer: <value>' ...") gets 10/10 compliance and 10/10 scored.

**Finding:** not refusal and not a recall failure — a contract-phrasing interaction on this
one leg: qwen's recall-given-holder is at ceiling like the rest of the roster, and the
published ⊘ stands because the leg's protocol is the fixed contract (report §2 notes the
diagnosis next to the scaffolded numbers). Data: `results/qwen_scaffold_probe/`.

## 26. Commutative rung across the roster — adjudication against the pre-registered bar — `run_frontier_benchmark.py` (facet `commutative`)

The §19 calibration scaled to the full roster (thinking @L64, effort=high, 8,192 tokens,
n=25; glm and deepseek reused from the calibration file, not re-bought) under issue #18's
pre-registered promotion bar: the row joins the headline only if ≥3 tiers separate at Wilson
95%. Scores: gpt-5.5 0.96 [0.80, 0.99]; opus / gemini-flash / qwen 0.80 [0.61, 0.91];
deepseek 0.80 (calibration); kimi 0.66 [0.52, 0.78]‡ (n=50); sonnet 0.64 [0.50, 0.76]
(n=50); glm 0.52 [0.34, 0.70]; nemotron 0.44 [0.27, 0.63] (empty 0.16). Exactly two tiers
with an overlapping boundary triggered the pre-registered kimi+sonnet top-up to n=50, which
did not resolve the kimi-vs-sonnet boundary.

**Finding:** fails the bar — only gpt-5.5 CI-separates (from kimi, sonnet, glm, nemotron;
not from the 0.80 group), so `commutative_v1` stays an experimental report row, not a
headline column (report §2 Components carries the roster numbers and the verdict). The row
still earns its keep as calibration: nothing at ceiling, and a reversal (deepseek 0.80 over
glm 0.52) that no other axis shows. Roster spend $3.94. Data: `results/benchmark/history.jsonl`
runs `bench_18_commutative_20260711` + top-up; `results/commutative_frontier/runs.jsonl`.

## 27. Statistical power: gap stability, thinking noise, drift canary — `run_frontier_benchmark.py` (facet `gap_stability`)

Three pre-stated checks (issue #16, $1.89 total; report part-2 appendix carries the summary).
(a) **Gap stability**: composed + binding legs at a second operating point (L32, instant,
n=50). The gap ordering holds off its L16 anchor for the three cleanly measurable
gap-interpretable models — gpt-5.5 +0.34 → +0.36, sonnet +0.15† → +0.14† (canonical; the
@512 escalation diagnostic reads +0.08), opus +0.06 → −0.04 (compose-for-free at both
points). Kimi's L32 cells are not interpretable (covert working: empty 0.40 composed / 0.62
binding, cost-limited). (b) **Thinking noise**: glm s5_concrete @L128 replicated at identical
settings — 0.84 vs stored 1.00, |Δ| 0.16 (both in history; first thinking-regime test-retest
datum, vs the 0.06 instant bar). (c) **Drift canary**: glm's full zero-budget battery
re-bought (5 legs, n=100): deltas composed@L16 −0.01, composed@L64 −0.01, binding −0.02,
replicate −0.02, scaffolded 0.00 — max |Δ| 0.02 against the ±0.06 bar, no drift.

**Finding:** the headline gap ordering is not an artifact of the L16 anchor; instant cells
are stable to re-purchase within 0.02; thinking cells carry a wider test-retest bar (~0.16 on
one pair) and should be read at coarser resolution. The canary rerun is now the rendered glm
row (latest-timestamp-wins), moving it by ≤0.02. Data: `results/benchmark/history.jsonl`
runs `bench_16_gap_L32_20260711`, `bench_16_glm_s5_replicate_20260711`,
`bench_16_canary_20260711`.

## 28. MOPD binding re-pin: outcome-RL from the object-filter floor — `experiments/mopd/` (issue #11, last item)

The MOPD testbed's binding domain re-pinned from the retired `binding_v1` (recency-defective
sampler) to CANONICAL `binding_v2` (uniform resolving-write position; same knobs — train
L∈{8,16}, eval L∈{16,24,32}), pins updated in `mopd_hf.py`, `mopd.py`, `bench_qwen.py`, and the
full Qwen3-1.7B GRPO + distillation pipeline re-run (RTX 5090, ~3.8 h wall). Base placement
first (greedy, no thinking, relaxed match, n=200; floors recomputed on the same deterministic
test items, n=300; chance 1/k = 0.200):

| L | base | object-filter floor E[1/w] | recency heuristic | teacher (GRPO 300 steps) |
| --- | --- | --- | --- | --- |
| 16 | 0.420 | 0.471 | 0.177 | **0.975** |
| 24 (held-out) | 0.325 | 0.348 | 0.220 | **0.980** |
| 32 (held-out) | 0.290 | 0.261 | 0.160 | **0.910** |

The base sits at the floor at every eval length and decays with the floor's ~1/L shape — the
published "partial capability" (0.33) was the v1 sampler's recency artifact; on the clean
sampler the base has no binding capability above object filtering. Outcome-RL lifts it to
near-solved including the held-out lengths, where the floor keeps falling and the teacher does
not — last-write resolution, not a longer-range shallow heuristic. Recall (unchanged domain)
reproduces within noise (base 0.39/0.25, teacher 0.99/1.00 @L16/L24). MOPD headline across 3
seeds (fresh 150-step teachers + 200-step students per seed, greedy n=300, normalised score
mean±std):

| model | binding | recall | avg |
| --- | --- | --- | --- |
| teacher_binding | 1.000±0.000 | 0.889±0.086 | 0.944±0.083 |
| teacher_recall | 0.094±0.021 | 1.000±0.000 | 0.547±0.453 |
| mopd_pg | 1.051±0.031 | 1.002±0.003 | 1.027±0.033 |
| mopd_kl | 1.083±0.043 | 1.000±0.004 | 1.042±0.052 |

**Finding:** the MOPD result is re-founded on the clean sampler and strengthens — the binding
lift is floor-to-solved (0.42→0.98 @L16 against the 0.47 floor; 0.29→0.91 at the held-out L32
against 0.26), not partial-to-better, and the normalised score's 0-anchor is an honest shallow
bound for the first time on binding. The single distilled student still matches or exceeds both
teachers on both domains on every seed, while neither teacher does both (`teacher_recall` stays
at the binding floor, 0.094; `teacher_binding` transfers to recall only partially, 0.889±0.086
— the output-format effect). Two calibration notes from the raw stage outputs: the measured
distillation reverse-KL on binding now starts materially nonzero (pg 6.4, kl 1.6 at the first
log point — the v2 binding teacher moves far from the base) yet still collapses to ≈0 within
~40 steps, so distillation stability survives a far-from-base teacher; and the student-exceeds-
teacher effect on binding (norm 1.01–1.14 every seed) holds against 150-step teachers but
flattens to parity (0.97/0.99) against 300-step teachers — teacher-strength-dependent. Data:
`results/mopd_v2/` (stage logs, tables, pipeline log);
`experiments/mopd/{hf_teachers,hf_mopd,hf_evaluate,hf_seeds,bench_qwen}.md`.

## 29. Harness system prompt vs reasoning engagement — `probe_sol_system_prompt_20260727.py`

gpt-5.6-sol's published `s5_chain` row is its engagement rate rather than its accuracy: per-call
completion spend is bimodal with a literal gap (no call between 294 and 1,136 tokens across the
300 archived `s5_chain_v3` calls), match conditional on working is 191/191, and 42/42 of its
wrong answers are the 8-hop dereference of the INITIAL pointer map — it answers the `chain` task
and skips the event stream. Whether that gate belongs to the model or to the harness is testable:
every scored thinking cell carries `run_frontier_benchmark.BASE_SYSTEM_PROMPT`, two of whose
clauses ("a short test", "no explanation") read as instructions to spend less effort. Three arms
on the identical deterministic items (`s5_chain_v3` @L64, n=25, effort xhigh, Responses endpoint,
49,152-token budget) differ only in that prompt; engaged = ctok ≥ 500, which sits in an empty
region (disengaged calls run 86–258 tokens, engaged calls start at 2,078).

| arm | system prompt | match | containment | engaged | match \| engaged | initial-map | ctok/call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| canonical | `BASE_SYSTEM_PROMPT` verbatim | 0.68 | 0.68 | 0.68 | 1.00 | 0.36 | 3128 |
| none | (no system prompt) | 0.84 | 0.88 | 0.96 | 0.88 | 0.08 | 5942 |
| neutral | same answer contract, two clauses dropped | **0.96** | 0.96 | 0.96 | 1.00 | 0.08 | 4086 |

The pre-registered rule was that engagement rising by ≥ ~0.2 on `none`/`neutral` makes the scored
thinking regime a measurement under an anti-thinking instruction, and flat engagement makes the
gate endogenous to the model. Engagement rose 0.68 → 0.96. Per item, 8 of 25 were disengaged
under the canonical prompt and 7 of those 8 were worked under the neutral prompt; disengaged
accuracy is 0 in every arm. The `none` arm engages as often as `neutral` and all 24 of its
engaged calls carry the gold value, but three commit it in LaTeX (`**Answer: \(g15\)**`,
`\boxed{g0}`) that `committed_answer` does not read — the answer contract is what makes the
emission parseable, and the two clauses are what suppress the work. The canonical arm's 0.68
sits inside the run-to-run spread of the three archived L64 batches (0.60/0.64/0.84), so the
arms are read against a reproduction of the published cell.

**Finding:** the engagement gate is elicited by the instrument, not only by the model — dropping
"a short test" and "no explanation" while holding the answer-format contract fixed moves the same
25 items from 0.68 to 0.96 and the initial-map rate from 0.36 to 0.08. Scope is one model, one
length, n=25, and the probe measures nothing about the other twelve roster models. One consequence
does reach the rest of the benchmark: every scored thinking cell carries this system prompt, so
the completion-token columns price spend under an instruction to answer without explanation. The
arms are off-protocol (a different system prompt is a different measurement) and are therefore not
in `history.jsonl`. Data: `results/probes/sol_system_prompt_20260727.json` ($9.87 completion spend).

## 30. s5_chain: the echo defect, the distinct-path gate, and the frontier battery — `run_frontier_benchmark.py` (facet `s5_chain`)

`s5_chain` composes non-abelian pointer-map state tracking with serial dereference: k agents
each hold an `a0` pointer to another agent, L swap/cycle events permute the map, and the query
dereferences the FINAL map `chain_depth` times. The battery below ran on `s5_chain_v3`
(k=16, depth 8), which supersedes `s5_chain_v1` and `s5_chain_v2`; all three are in
`tasks.RETIRED`, each with its defect annotated there. Floors are recomputed from the exact
deterministic test items (`factworld.validity.s5_chain_floors`, n=500 per length, ranges over
L32–L128):

| stream | cycle-event rendering | echo | initial-map chase | operative floor | degenerate query paths |
| --- | --- | --- | --- | --- | --- |
| v1 (retired) | arrow list, `swaps the a0 of X and the a0 of Y` | 0.214–0.234 | 0.036–0.064 | 0.214–0.234 | 0.470–0.504 |
| v2 (retired) | per-agent value updates | 0.228–0.244 | 0.038–0.058 | 0.228–0.244 | 0.472–0.502 |
| v3 (retired) | simultaneity-explicit, `takes X's old a0` | **0.000** | 0.056–0.086 | 0.067–0.086 | **0.000** |

Without the `distinct_path` gate the final permutation has cycles whose length divides
chain_depth=8, so the queried agent is its own answer on 0.21–0.24 of items and roughly half
the items have a query path visiting fewer than 9 distinct agents. The v3 gate requires the start
to sit on a final-map cycle of length ≥ 9: echo and every fixed-hop heuristic score exactly 0,
chance is 1/16, and item difficulty is uniform. The v1→v2 change is rendering only, at matched
budget (L64, effort high, 16,384 tokens, n=25, same four models):

| model | v1 | v2 |
| --- | --- | --- |
| openai/gpt-5.5 | 0.12 | 1.00 |
| anthropic/claude-opus-4.8 | 0.20 | 0.80 |
| google/gemini-3.5-flash | 0.12 | 0.60 |
| qwen/qwen3.7-max | 0.16 | 0.72 |

The v3 battery is 13 models × L∈{32, 64, 96, 128}, n=25, effort xhigh, per-length budgets
32,768 / 49,152 / 65,536 / 98,304 sized so truncation stays a rounding error.

**Finding:** no v1 cell scored above the echo floor recomputed from the 25 items it was scored
on: match 0.08–0.24 against echo floors of 0.280 at L16, 0.360 at L32 and 0.200 at L64, with the
highest cell (opus at L64, 0.20) exactly at its floor and the floor itself carrying sd 0.08–0.10
at n=25. The arrow rendering left the events unreadable and the ungated stream left a degenerate
strategy worth more than the measurements. The rendering fix alone moves the same four models to
0.60–1.00 at matched budget, and the gate is what makes a sub-0.4 score interpretable at all:
under v1/v2 a model scoring 0.24 is indistinguishable from one answering the queried agent. On
v3 the shallow adversaries are held at 0.000 (echo, any fixed hop) and 0.056–0.086 (chase the
initial map, ignore the events), so the measured range sits an order of magnitude above the
floor. What the gate does not reach is the direction of the computation: v3's events permute
the map's DOMAIN, so pushing one symbol backward through the event list and applying the stated
initial map answers the query exactly, at 4 bits of carried state per hop — available to an
attention model over the full context and not to a streaming recurrent one, which is why v3 is
retired in favour of `s5_chain_v4` (`factworld/tasks.py` carries the derivation). v4 blocks that
walk — the shallow adversary goes 1.000 to 0 — and the FAMILY is retired anyway, on a defect the
shortcut fix does not reach: on the published v3 battery the top eleven models have zero pairwise
separations at n=25, so the ranked cell orders by noise. v4's own evidence is thin by comparison
— three models at two lengths, three of six cells censored on call failure — and on the one model
with clean cells at both versions it reads the same on each, so nothing shows the fix made the
cell harder for the frontier.
Spend: v1 $38.02, v2 $174.16, v3 $259.30. Data: `results/benchmark/history.jsonl` (runs
`bench_20260717_071141`, `bench_20260717_113626`, `bench_20260717_131813`,
`bench_20260718_012918`, `bench_s5v3_scout`, `bench_roster_20260724`); retired specs and their
defect annotations in `factworld/tasks.py`.

## 31. Local s5_chain battery and its audit — `sweep.py`, `scripts/run_s5_chain_*.sh`

141 training runs across 22 sweep files (`results/local_s5_chain_*.jsonl`): gdp_hybrid, fprm and
transformer at k∈{4,5,6,8} and chain_depth∈{1,2}, d_model 320×4 and 768×8, 8,000 documents ×
8,000 steps × batch 32, eval_n 200, across appended worked traces, per-event map checkpoints,
s5-shaped single-slot checkpoints, interleaved supervision, and a compact-grammar rendering
ablation. Every arm read at or below its floor. The 2026-07-27 audit of that battery:

| what was measured | value |
| --- | --- |
| tokenizer coverage on s5_chain training documents (pre-extension vocabulary, n=2,000 docs per spec) | `<unk>` rate **0.171–0.261**; out-of-vocabulary types `takes`, `old`, `values`, `simultaneously:`, `a0,`, `a0.`, `(N`, `hops)` |
| the same measurement on every other scored family | chain_v2 0.057 (its redundant hop annotation); binding_v2 / composite_copy_v2 / commutative_v1 / recall_copy_v1 / conflict_v1 / s5_v1 all 0.000 |
| the same measurement after the vocabulary extension, on the specs this battery trained (`s5_chain_local_v2`, `s5_chain_local_v2_path`; plain, trace and path-trace surfaces) | `<unk>` rate **0.000** |
| interleaved cells at chain_depth 1 (k=4, 6, 8; 18 runs) | gold answer equals the token immediately preceding `what` in 2,000 of 2,000 training documents (P=1.000); the free-running eval prompt deletes that token |
| k=6 / depth=2 event_trace, gdp_hybrid @L4, per seed | 0.155 / **0.815** / 0.170 against the 0.200 operative floor; published as `0.38±0.31 (0%)` |
| floors in the run scripts | annotated 1/(k−depth), which is the accuracy of no policy; the operative floor is max(initial-map chase, uniform over non-start), 0.200–0.335 across the registered local cells |
| trace-mode generation budget | len(trace)+6 — five tokens of slack, so one spurious checkpoint row truncates a correct answer to 0 at k ≥ 6 |
| eval path | a stub backend emitting the gold continuation scores 1.000 on every arm |
| budget | 0.26M documents seen per arm (32 epochs); the composite that converges locally saw 3.2M plus a staged curriculum, and s5_chain appears in no curriculum script |

**Finding:** the battery measured a corrupted input. The one task family that never formed
locally is the only one whose training documents lost a sixth to a quarter of their tokens to a
single `<unk>`, and the lost types are the ones carrying the pointer-update semantics — a cycle
event reached the model as `s0 cycles a0 <unk> g0's a0 <unk> g4's <unk> <unk> ...`. Three further
defects are independent of that one: the depth-1 interleaved cells score a copy rule against a
prompt with the source token deleted, a formed seed (0.815 against a 0.200 floor) was averaged
into a null by a p_converge rule the repo applies to bimodal emergence elsewhere, and no cell was
ever read against the accuracy of a shallow policy. The frontier cells are untouched by all of
it: items come from `tasks.generate`, which never consults the tokenizer, so every frontier
score and every floor computed from those items stands. Local s5_chain and chain_v2 numbers
produced under the pre-extension vocabulary are not comparable to results from the extended one
and do not belong in one table with them. Data: `results/local_s5_chain_*.jsonl`; floors in
`factworld/validity.py`; the round-trip contract over every CANONICAL and RETIRED spec in
`tests/test_tokenizer.py`.

## 32. FactWorldBench validation: engagement, event-blindness, and what n=25 resolves — `render_benchmark.py` (facet `s5_chain`)

Three diagnostics recomputed from the stored per-call completion tokens, reasoning tokens, finish
reasons and predictions in `history.jsonl`, over the 52-cell `s5_chain_v3` battery of §30.

**Work rate** is the fraction of a cell's calls above 512 completion tokens, with accuracy read
conditional on working. Eleven of the thirteen roster models have 25/25 working calls on every
`s5_chain` cell and deepseek-v4-pro has 24/25 on two, so the column is inert for them.
gpt-5.6-sol is bimodal with a literal gap: across its 300 archived `s5_chain_v3` calls no call
spends between 294 and 1,136 completion tokens.

| gpt-5.6-sol cell | L32 | L64 | L96 | L128 |
| --- | --- | --- | --- | --- |
| working calls (of 25) | 8 | 15 | 14 | 19 |
| match | 0.32 | 0.60 | 0.60 | 0.80 |

Over all 300 archived calls it is correct on 191 of 191 working calls and 8 of 109 others,
against 1/16 chance.

**Event-blind rate** is the fraction of predictions equal to the 8-hop dereference of the INITIAL
pointer map — the answer a model gives if it reads the fact block and skips the whole event
stream. Items where that answer coincides with gold are dropped: 39 of the 1,300 scored items —
none at L32, one of the 25 items at each of L64, L96 and L128. Across the roster's 52 scored
`s5_chain` cells the blind answer is given on 46 of the 1,261 eligible items, 42 of them by
gpt-5.6-sol; over the other twelve models it is 4 of 1,164 (0.003). The same coincidence measured
over the n=500 stream rather than the scored items is §30's initial-map-chase floor, 0.056–0.086.

**Resolution at n=25.** Per-item outcomes at L96, two-sided Fisher exact on every pair, and
Cochran's Q over the eleven models above qwen and sol:

| statistic | value |
| --- | --- |
| separating pairs, all 13 models (78 pairs, p < 0.05) | 18 — every one of them involves qwen or sol |
| separating pairs inside the top eleven (55 pairs) | **0** (smallest p = 0.49) |
| Cochran's Q over those eleven, 25 shared items | Q = 6.44, df = 10, **p = 0.777** |
| roster mean match at L32 / L64 / L96 / L128 | 0.90 / 0.89 / 0.91 / 0.90 |
| completion tokens per call at L64, the nine models scoring ≥ 0.92 | 5,014 (fable-5) to 17,982 (glm-5.2), a 3.6× range |

**Finding:** at n=25 the score resolves the tail and nothing else — eleven of thirteen models are
one undifferentiated band at L96, and length moves individual models (qwen 0.72 → 0.44 from L96
to L128) rather than the roster, whose mean is flat across all four lengths. Token spend
separates what the score does not: a 3.6× range inside a single score band. gpt-5.6-sol's
separation is engagement, not capability: it is perfect when it works, its wrong answers are the
initial-map dereference — it answers the `chain` task and skips the events — and its score rises
with prompt length because it disengages less often. Its cells therefore carry the unworked mark
and take no part in any ordering; that also removes them from the thinking-noise bar, which reads
0.16 over unmarked cells against the 0.32 spread of the marked ones. Data:
`results/benchmark/history.jsonl`; rendered feed `docs/benchmark/results.md` (work-rate,
event-blind and truncation columns) and `docs/benchmark/results.csv`.
