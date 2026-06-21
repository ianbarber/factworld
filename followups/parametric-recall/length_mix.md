# Length-mix — which training-length distribution unlocks extrapolation? (`length_mix.py`, 18.5M, 3 seeds)

Fixed capacity (d384x6, 18.5M), mixed density. Vary the training-length distribution; answer-only (internalized) e2e at L32/L64/L128. `short16`/`mid`/`full` describe the lengths trained on; `longN` = {4,8,16} plus N% examples at L64. Floor = 0.20.

| training lengths | L32 | L64 | L128 |
|---|---|---|---|
| {4,8,16} only | 0.27±0.07 | 0.21±0.04 | 0.20±0.03 |
| {4,8,16} + 5% L64 | 0.32±0.07 | 0.20±0.04 | 0.20±0.01 |
| {4,8,16} + 20% L64 | 0.88±0.06 | 0.66±0.19 | 0.23±0.07 |
| {4,8,16} + 50% L64 | 0.85±0.18 | 0.67±0.22 | 0.32±0.07 |
| {16,32} only | 0.92±0.01 | 0.37±0.10 | 0.20±0.02 |
| uniform {4..64} | 0.78±0.18 | 0.49±0.15 | 0.23±0.05 |
