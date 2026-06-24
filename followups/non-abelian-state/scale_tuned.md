# Scale-done-right — LR-tuned + 70M ceiling: is the horizon wall capacity or under-training?

`followups/non-abelian-state/scale_tuned.py`. gdp_hybrid, mixed-density, 8000 steps, 2 seeds. Largest scales with a per-scale LR sweep + a ~70M ceiling point. Headline = **answer-only L64** (the internalized horizon wall; floored 0/3 across 5.7M→44.8M at fixed lr in scale.py). Floor = 0.20. Still floor here = not relieved by capacity OR LR-tuning within 3090 reach.

| scale | lr | answer-only L16 | answer-only L64 |
|---|---|---|---|
| 44.8M | 0.001 | 0.98±0.02 | 0.24±0.04 |
| 44.8M | 0.0005 | 0.79±0.21 | 0.19±0.05 |
| 70M | 0.001 | 1.00±0.00 | 0.41±0.05 |
| 70M | 0.0005 | 0.75±0.24 | 0.24±0.09 |
