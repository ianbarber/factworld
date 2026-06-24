# Scale check — does the internalisation dissociation soften? (mixed-density recipe)

`followups/non-abelian-state/scale.py`. gdp_hybrid, mixed-density training, 6000 steps, 3 seeds, parametric. Free-running e2e composite. **Answer-only L64** is the headline (the dissociation wall: 0/5 floor at 6M in curriculum.py). Floor = 0.20.

| scale | answer-only L16 | answer-only L64 | dense (scratchpad) L64 |
|---|---|---|---|
| 5.7M | 0.67±0.24(2/3) | 0.23±0.04(0/3) | 0.45±0.38(1/3) |
| 18.5M | 0.99±0.01(3/3) | 0.20±0.02(0/3) | 0.71±0.36(2/3) |
| 44.8M | 0.95±0.07(3/3) | 0.22±0.05(0/3) | 0.68±0.39(2/3) |
