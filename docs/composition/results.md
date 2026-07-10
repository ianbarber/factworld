# Small-scale composite results (the composition gap) — final

`scripts/sk_composite.py` (memorization diagnostic) and `scripts/iso.py` (the n_h mechanism control). Compute-matched archs (d256, 4 layers, ≈6–12M params): agent→property facts + a `give`-stream binding,
chained query *"what is a0 of the holder of o3?"* (resolve the holder, then recall its property). Floor =
1/k = 0.20. Backs §4. **Key diagnostic:** the composite's recall leg is solvable by *memorization* unless the
agent→value map is resampled per example — distinguishing the memorized shortcut from genuine in-context copy.

## The memorization diagnostic (k=5 composite, gdp_hybrid; L16 / L64)

| recall leg | result | reading |
|---|---|---|
| **memorized** (map fixed per world) | 0.79 / 0.35 | solvable — the map is learned in weights |
| **in-context copy** (map resampled per example) | 0.01 / 0.00 (floor) | the genuine composite floors |

So the small-scale composite holds up at length **only when the recall degenerates to a memorized lookup**.

## Binding × memorized recall — survivorship with length (k=5, 3 seeds; eval to 4×)

| arch | L4 | L8 | L16 | L32 | L64 |
|---|---|---|---|---|---|
| transformer | 0.34±.05 | 0.28±.01 | 0.21±.07 | 0.22±.02 | 0.21±.04 |
| gru | 0.33±.01 | 0.27±.01 | 0.22±.02 | 0.20±.02 | 0.21±.01 |
| gdn_hybrid | 0.45±.04 | 0.42±.04 | 0.35±.01 | 0.27±.04 | 0.23±.02 |
| **gdp_hybrid** | **0.92±.12** | **0.84±.19** | **0.74±.27** | **0.58±.26** | **0.56±.25** |

On the *memorized*-recall composite, gdp_hybrid is the only arch well above floor at long lengths (0.56 vs
≈0.21 at L64); the field collapses to floor and only the unified arch stays functional. The variance (±0.25)
is convergence variance, not a capacity ceiling (lr≤1e-3 reliably solves it; lr=2e-3 breaks).

## Mechanism is the product structure at fixed parameters (n_h ∈ {1,2,4}, ~5.68M params, strict holder+value, 5 seeds)

`d_ff` absorbs the Householder cost so all configs match params; n_h=1 is the single-delta (gdn-like) point
*inside* gdp, isolating the product structure from the capacity it normally adds.

| config | L16 | L64 | L128 | #converged |
|---|---|---|---|---|
| n_h=1 (d_ff=1408) | 0.52±.22 | 0.38±.16 | 0.32±.10 | 1/5 |
| n_h=2 (d_ff=1280) | 0.61±.24 | 0.51±.22 | 0.45±.18 | 2/5 |
| **n_h=4 (d_ff=1024)** | **0.95±.07** | **0.74±.22** | **0.68±.22** | **5/5** |
| n_h=4, neg-eig OFF | 0.66±.22 | 0.35±.12 | 0.35±.13 | 1/5 |

Convergence is **monotone in the Householder count at fixed parameters** (1/5 → 2/5 → 5/5), and turning
negative eigenvalues off collapses length-extrapolation — both the product structure (≥4 reflections) and
the [−1,1] spectrum are load-bearing, capacity controlled out.

## The limit: even the memorized win is in-context-recall-specific

With **parametric (in-weights)** facts, gdp collapses (L16 0.53 → L64 0.23 → L128 0.20 = floor), like every
other arch, and no knob (more steps, bigger model, decomposed CoT) cracks it. A "final-give-agent" shortcut
tops out at ≈0.30, so gdp's ≈0.95 on the memorized composite is genuine composition, not the shortcut.

## Genuine composition is unsolved at the benchmark baseline

At the frozen-benchmark baseline (`docs/results.md`), the genuine in-context-copy composite
(`composite_copy_v1`) floors for **all four** architectures (≤0.02). The scale step that lifts it is in
`docs/state-tracking/scale.md`.

## Decomposition — the composite is a resolve-then-recall pipeline, not two legs (`scripts/decompose.py`)

Train gdp_hybrid on the genuine composite (5 seeds, **free-running** eval — the model emits its own holder
and value, nothing is forced), then decompose the strict eval into the **state leg** (holder resolved?) and
the **recall leg** (value correct?). The discriminating control inspects, on holder-**wrong** examples, what
value the model emits: `route` = it emits the property of *the (wrong) agent it actually resolved*; `other` =
some other in-vocab value; `none` = no value token at all.

| seed | L16 holder | L16 value | P(v\|h✓) | P(v\|h✗) | route | other | none |
|---|---|---|---|---|---|---|---|
| **s1** (converged) | 0.405 | 0.405 | **1.000** | 0.000 | **0.996** | 0.004 | 0.000 |
| **s2** (converged) | 0.400 | 0.398 | **0.994** | 0.000 | **0.967** | 0.033 | 0.000 |
| **s4** (converged) | 0.320 | 0.318 | **0.992** | 0.000 | **1.000** | 0.000 | 0.000 |
| s0 (floored) | 0.390 | 0.013 | 0.026 | 0.004 | 0.020 | 0.980 | 0.000 |
| s3 (floored) | 0.468 | 0.015 | 0.011 | 0.019 | 0.014 | 0.986 | 0.000 |

(5-seed means — L16 holder 0.397±0.047, value 0.230±0.179, P(v|h✓) 0.604, P(v|h✗) 0.005, route 0.599,
converged 3/5; L64 holder 0.286±0.038, value 0.106±0.081, route 0.420. The means mix the two modes and are
not the headline — the per-seed split is.)

**Recall is routed through the binding, not an independent parallel leg.** On the 3/5 seeds that form the
circuit, the model essentially never recalls the correct value without first resolving the binding
(P(value | holder wrong) ≈ 0). The token-level reason is genuine routing, not breakdown: on holder-wrong
examples it emits the property of *the agent it actually resolved* 97–100% of the time at the train length
(`route` ≈ 1.0) and **never falls silent** (`none` = 0). So it executes resolve-then-recall and recalls the
*wrong* agent's value when it resolves the wrong agent — which is why a wrong binding yields a wrong value.
The 2/5 floored seeds never form the recall circuit (`value` ≈ chance, `route` ≈ chance) **even when the
binding works** (s3: holder 0.47, value 0.02), locating the bottleneck at recall-under-composition, not
binding. At 4× the pipeline degrades gracefully with the binding (converged-seed `route` 0.53 / 0.62 / 0.91;
P(v|h✓) 0.50 / 0.56 / 0.89) but keeps its structure (`none` ≈ 0). This is all-or-nothing **optimization** —
the circuit forms or it does not, matching the scale-run bimodality (`docs/state-tracking/scale.md`) — so we
characterize how the composite is *learned*, not claim the architecture forces it. (The earlier
`P(both)/P(state)·P(recall)` "coupling" ratio is dropped: within a converged seed both = holder = value, so
the ratio is mechanically 1/p and is not independent evidence.)
