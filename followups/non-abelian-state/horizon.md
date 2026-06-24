# Horizon-extension curriculum — does training the recurrence at length unlock internalised extrapolation?

`followups/non-abelian-state/horizon.py`. gdp_hybrid d384x6 (18.5M), mixed density, training lengths grow 8->16->32->48->64 over 6000 steps, 3 seeds. Free-running answer-only e2e (no scratchpad). Baseline (train<=16, from scale.py 18.5M): answer-only L64 = 0.20 floor. Floor = 0.20.

| eval | answer-only e2e | note |
|---|---|---|
| L16 | 1.00±0.00(3/3) | trained edge |
| L64 | 0.94±0.01(3/3) | in extended train range |
| L128 | 0.48±0.12(2/3) | truly OOD (2x max train) |
