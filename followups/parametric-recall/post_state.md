# Post-training state coverage — does it extend the circuit when from-scratch didn't? (`post_state.py`, 18.5M, 3 seeds)

gdp_hybrid d384x6. `base` = trained short {4,8,16} mixed-K (the in-distribution circuit). `post` = base + 1500 steps of burn-in post-training (unlabeled deep-state coverage, B in {0..192}) at lr 0.0003. Eval internalized (answer-only) to L256. Floor = 0.20. Contrast: carried_state from-scratch burn-in floored AND hurt in-distribution.

| arm | L16 | L32 | L64 | L128 | L256 |
|---|---|---|---|---|---|
| base | 0.82±0.14 | 0.31±0.13 | 0.20±0.01 | 0.20±0.02 | 0.23±0.03 |
| post | 0.88±0.08 | 0.60±0.28 | 0.47±0.37 | 0.44±0.30 | 0.22±0.06 |
