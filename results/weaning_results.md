# Weaning bridge — does a dense-learned s5 circuit survive weaning to answer-only?

`scripts/experiment_weaning.py`. gdp_hybrid d256x4, dense 4000 steps (+4000 wean), 8 seeds. 
FREE-RUN eval (no holder labels at eval). value = end-to-end accuracy. Floor = 0.20.

| arm | value @L16 | value @L32 | value @L64 | conv @L16 |
|---|---|---|---|---|
| dense_only | 1.00±0.00 | 0.68±0.28 | 0.61±0.27 | 8/8 |
| answer_only | 0.19±0.05 | 0.21±0.03 | 0.19±0.03 | 0/8 |
| wean_linear | 0.64±0.28 | 0.21±0.04 | 0.18±0.03 | 5/8 |
| wean_mixed | 1.00±0.00 | 0.64±0.35 | 0.55±0.34 | 7/7 |

**Finding — the circuit survives weaning.** wean_mixed (Phase 2's mixed-density curriculum) reaches
dense_only-level accuracy in-distribution and tracks dense extrapolation closely — the dense-learned
circuit persists when supervision is weaned to answer-only/mixed. wean_linear (ordered sparsification)
is weaker. This is the bridge to the agentic (label-free at deploy) regime: train dense, wean, deploy
answer-only. answer_only (never dense) floors as expected.

So s5 is movable to deployment: dense supervision forms the circuit (supervision density), weaning
internalizes it (no deploy-time labels), and gdp_hybrid extrapolates it in length (architecture).
