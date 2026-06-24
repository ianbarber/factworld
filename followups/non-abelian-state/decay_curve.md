# Decay curve — real circuit vs shortcut (`decay_curve.py`, 18.5M, 3 seeds)

gdp_hybrid d384x6, trained at the short envelope {4,8,16} (Lmax=16), evaluated internalized (answer-only, no scratchpad) at a fine length grid to 16x. Same e2e value metric for all arms. Floor = 0.20. Shape: flat = length-general circuit; cliff near ~2xLmax (32) = length-bounded shortcut.

| arm | L16 | L24 | L32 | L48 | L64 | L96 | L128 | L192 | L256 |
|---|---|---|---|---|---|---|---|---|---|
| abelian_native | 0.57±0.30 | 0.30±0.06 | 0.24±0.03 | 0.29±0.05 | 0.25±0.03 | 0.21±0.05 | 0.25±0.02 | 0.23±0.04 | 0.23±0.01 |
| abelian_mixed | 1.00±0.00 | 0.36±0.01 | 0.24±0.02 | 0.22±0.03 | 0.24±0.04 | 0.23±0.06 | 0.20±0.05 | 0.21±0.03 | 0.23±0.01 |
| nonabelian_mixed | 0.86±0.05 | 0.51±0.08 | 0.32±0.08 | 0.23±0.09 | 0.23±0.02 | 0.21±0.04 | 0.22±0.03 | 0.18±0.01 | 0.21±0.01 |
