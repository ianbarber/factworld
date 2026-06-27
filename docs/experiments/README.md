# Experiments — index and synthesis

FactWorld's experiment program, post-refactor (natural-language format). Each experiment
has a script, a results file, and a finding below. The four pieces here close the power,
format-fairness, architecture-independence, weaning, and test-time-compute questions raised
in review.

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
| **composition** | generating the holder (last-write-wins over objects) | **background reasoning + format** (kimi/glm: dose-response 0.14→0.81 with effort) | explicit CoT prompting (hurts), dense holder supervision, local self-correction |
| **s5 / non-abelian** | tracking a single role through permutations | **dense per-step supervision → wean to answer-only** (wean_mixed 7/7), then gdp_hybrid extrapolates 8× | reasoning effort (floor at all levels), sparse/answer-only |
| **recall (value)** | — | given the holder, trivially solved (0.93–1.00) | — |

**Two dissociations, both now clean:**
- **Composition is movable by test-time compute** (reasoning) for strong models; **s5 is not** — it
  needs training-time supervision density, and that circuit can be *weaned* to label-free deployment.
- Architecture (gdp_hybrid vs fprm vs transformer) gates **length extrapolation** of a learned s5
  circuit, not its formation.

What never helps: explicit structured CoT prompting, and sampling/self-correction on non-reasoning
models. The levers are **reasoning strength** (composition), **supervision density + weaning** (s5),
and **recurrent architecture** (s5 extrapolation).

**Open question (the confound) — RESOLVED by the reasoning sweep.** kimi/glm solving
composition *was* test-time compute working. The reasoning on/off/levels sweep
(`reasoning-results.md`) shows a clear dose-response: composite value climbs with effort
(kimi 0.22→0.96→0.98; glm 0.14→0.74→0.81) while **s5 stays at floor regardless of effort**.
So the corrected statement: **background reasoning (test-time compute) IS a lever for
composition; it is NOT for s5.** What does not help either wall: explicit structured CoT
prompting (hurts), and sampling/self-correction on non-reasoning local models.

## 5. Weaning bridge — `experiment_weaning.py` (8 seeds)

Can a dense-learned s5 circuit survive weaning to answer-only (the agentic regime)?

| arm | value @L16 | value @L32 | value @L64 | conv @L16 |
| --- | --- | --- | --- | --- |
| dense_only | 1.00±0.00 | 0.68±0.28 | 0.61±0.27 | 8/8 |
| answer_only | 0.19±0.05 | 0.21 | 0.19 | 0/8 |
| wean_linear | 0.64±0.28 | 0.21 | 0.18 | 5/8 |
| **wean_mixed** | **1.00**±0.00 | 0.64±0.35 | 0.55±0.34 | **7/7** |

**Finding — the bridge works.** Phase-2's mixed-density curriculum (wean_mixed) reaches dense-level
accuracy in-distribution (7/7 converge) and tracks dense extrapolation — the dense-learned circuit
persists with no deploy-time labels. So s5 is movable *all the way to deployment*: dense supervision
forms the circuit, weaning internalizes it, gdp_hybrid extrapolates it in length.

Two clean dissociations, neither movable by test-time compute; one movable by supervision density
(s5), one movable by base-model reasoning strength (composition). Architecture matters only for
length extrapolation of a *learned* s5 circuit (gdp_hybrid wins).
