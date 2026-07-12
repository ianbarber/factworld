# Profiles vs the pinned intuitive ranking — data note

A cross-check of the benchmark's per-model profiles (`fig_profiles.png`, rendered every cycle
by `scripts/render_benchmark.py`) against a prior: the owner's pinned intuitive ranking of the
roster, best first — gpt-5.5 > opus > glm > sonnet > kimi > gemini-flash > qwen > deepseek >
nemotron (grok is unrepresented on the roster and skipped). The ranking is a prior, not a
measurement; the question is which axes of the instrument that prior is tracking, and which
axes say something the prior does not.

Data: `results/benchmark/history.jsonl` snapshot of 2026-07-12 (instant cells on
`composite_copy_v2`; thinking state-stress at chain d128 / s5 @L256, 16,384-token budgets,
with the two raised-budget s5 cells — opus and sonnet 1.00 @32,768 — included as measured).
Numbers regenerate with `scripts/profile_intuition.py`; the axis definitions are shared with
the figure (`render_benchmark.PROFILE_AXES`). Censored (⊘) and never-run cells are dropped
from each correlation, never scored as zeros. Kimi's instant cells are daggered (covert
working; provider does not enforce the cap), so every number is given with and without kimi.

## Per-axis Spearman

| axis | rho (all measurable) | rho (without kimi †) |
|---|---|---|
| binding @L16 | +0.81 (n=9) | +0.97 (n=8) |
| composed @L16 | +0.45 (n=9) | +0.55 (n=8) |
| gap (inv) | -0.28 (n=9) | -0.24 (n=8) |
| chain d128 | -0.72 (n=7) | -0.55 (n=6) |
| s5 @L256 | +0.73 (n=7) | +0.61 (n=6) |
| s5@128 ctok (inv) | +0.18 (n=9) | +0.24 (n=8) |

Scale for reading the table: with n=9 pairs, |rho| ≥ 0.68 is the two-sided p<0.05 line
(0.74 at n=8, 0.79 at n=7). Only the binding row clears its line; everything else is a
direction, not a certified effect.

## Reading

**What tracks the prior.** In-weights state tracking is what the intuitive ranking is mostly
made of: binding @L16 correlates at +0.81, and removing kimi's daggered cell takes it to
+0.97 — near-perfect. s5 @L256 points the same way (+0.73 with the raised-budget opus and
sonnet cells folded in; deepseek and nemotron stay ⊘), on seven models against a 0.79
significance line at n=7 — a direction, not a certified effect.

**What inverts.** The chain d128 axis runs *against* intuition (-0.72): the roster's best
pointer-chasers are qwen (0.96) and gemini-flash (0.88), ranked 7th and 6th in the prior,
while opus and sonnet — 2nd and 4th — post the weakest measurable scores (0.08, 0.04).
Deep serial pointer-chasing under a reasoning budget is a real capability axis that the
intuitive ranking does not contain; it is not noise, it is the instant/thinking
near-orthogonality showing up against the prior. The composition gap also leans negative
(-0.28): the two largest gaps belong to gpt-5.5 (+0.34) and glm (+0.34†), ranked 1st
and 3rd — a large gap is compatible with a strong overall impression, because thinking
buys the composition back.

**Reversal flags.** The commutative row is a concrete inversion: thinking @L64,
deepseek 0.80 vs glm 0.52 (n=25 each, neither at ceiling;
`results/commutative_frontier/runs.jsonl`) — the prior puts glm five places above deepseek.
Chain d128 carries the same pattern (qwen and gemini-flash above every frontier-pair model).
The full roster ran under issue #18's pre-registered bar and failed promotion (only gpt-5.5
CI-separates), so the row stays experimental — but its disagreement with intuition is an
argument for it carrying information, not against.

**Two-axis combinations.** Averaging normalized positions pairwise (models measurable on
both axes), the best tracker of the prior is **binding @L16 + s5@128 ctok (inverted)**:
+0.95 over all nine models (+0.93 without kimi) — better than either axis alone. The
intuitive ranking looks like "holds state in weights, and doesn't burn tokens doing it."
This is a post-hoc selection over 15 pairs, so treat it as descriptive; the runner-up
(composed @L16 + ctok inverted, +0.82/+0.88) has the same shape.

| combination | rho (all measurable) | rho (without kimi †) |
|---|---|---|
| binding @L16 + s5@128 ctok (inv) | +0.95 (n=9) | +0.93 (n=8) |
| composed @L16 + s5@128 ctok (inv) | +0.82 (n=9) | +0.88 (n=8) |
| s5 @L256 + s5@128 ctok (inv) | +0.61 (n=7) | +0.54 (n=6) |

## Coarse tier view

The profile grid compresses to three tiers on the composition question (rules first-class,
marks carried; replicate noise bar 0.06, object-filter floor 0.41 @L16):

| tier | rule | models |
|---|---|---|
| compose-for-free | binding clears the object-filter floor decisively AND gap ≤ the 0.06 noise bar | opus (+0.06), gemini-flash (+0.02*) |
| compose-with-thinking | binding established, gap > noise instant; composed reads ~ceiling with reasoning on | gpt-5.5 (+0.34), sonnet (+0.15†), kimi (+0.17†), glm (+0.34†) |
| components-only (in weights) | recall at ceiling but the binding leg itself sits at the 0.41 floor — the gap is not a composition measurement | deepseek, qwen, nemotron |

Thinking-regime ceiling evidence as in the report: 0.98–1.00 on the effort=high composed
cell where measured (kimi 1.00, glm 0.98 @L16, n=50) and glm 0.92–1.00 out to L1024 on the
calibration probes. The tiers are an instant-regime ladder — every model in the bottom two
tiers still composes with reasoning; the tier says what it holds in weights.

## Caveats

- n=9 models; every rho here has a wide interval and the per-axis table should be re-read
  after each cycle's re-render, not treated as fixed.
- chain d128 and s5 @L256 drop ⊘ cells (deepseek, nemotron), so those correlations run on
  7 models and are budget-conditioned. The #17 raised-budget reruns did move s5: folding
  opus and sonnet in at 1.00 (@32,768) took it from +0.87 (n=5) to +0.73 (n=7).
- The pinned ranking is a single rater's prior with no uncertainty model of its own.
