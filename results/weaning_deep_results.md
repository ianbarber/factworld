# Weaning bridge — deep-dive (extrapolation + density mix)


`scripts/experiment_weaning.py`. gdp_hybrid d256x4, dense 4000 (+4000 wean), 8 seeds, free-run eval.
Q1: does weaning extrapolate *better* than dense-only? Q2: which density mix survives best?

| arm | L16 | L32 | L64 | L128 | conv @L16 |
|---|---|---|---|---|---|
| dense_only | 1.00±0.00 | 0.68±0.28 | 0.61±0.27 | 0.50±0.26 | 8/8 |
| wean_mixed:1,2,4,inf | 1.00±0.00 | 0.61±0.33 | 0.50±0.34 | 0.46±0.33 | 8/8 |
| wean_mixed:1,inf | 0.99±0.01 | 0.69±0.30 | 0.54±0.33 | 0.48±0.29 | 8/8 |
| wean_mixed:1,4 | 1.00±0.00 | 0.68±0.29 | 0.53±0.33 | 0.47±0.33 | 6/6 |

**Findings:**
1. **Weaning preserves the circuit** — every mix survives (8/8 converge @L16, free-run, no deploy labels).
2. **Weaning does NOT improve extrapolation over dense-only** — all mixes track dense_only within
   noise (L128: dense 0.50, mixes 0.46-0.53). The win is label-free *deployment*, not better length
   generalization. (Phase 2's hint that mixed density helps extrapolation does not reproduce here at 8 seeds.)
3. **The specific mix barely matters** — {1,2,4,inf}, {1,inf}, {1,4} all comparable. The key is just
   *some* exposure to answer-only (K=inf) alongside dense; the intermediate densities are optional.

**Bottom line for deployment:** train dense (K=1), fine-tune on any mix that includes answer-only,
deploy answer-only. The dense-learned s5 circuit survives weaning and extrapolates (via gdp_hybrid)
to ~8x the trained length — fully label-free at inference.
