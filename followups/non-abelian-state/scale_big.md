# Scale push — does the horizon wall lift with capacity past 70M? (`scale_big.py`, lr 1e-3, bs32, 2 seeds)

gdp_hybrid, mixed-density, 8000 steps, batch 32 (consistent with scale_tuned's 70M point). Answer-only L64 is the internalized horizon wall. Context from scale_tuned (same recipe): 44.8M = 0.24, 70M = 0.41 (lr 1e-3). Floor = 0.20.

| parameters | answer-only L16 | answer-only L64 (the wall) |
|---|---|---|
| 44.8M (scale_tuned) | — | 0.24 |
| 70M (scale_tuned) | — | 0.41 |
| 140M | 1.00±0.00 | 0.42±0.02 |
| 268M | 0.81±0.18 | 0.21±0.03 |
| 357M | 1.00±0.00 | 0.20±0.03 |
