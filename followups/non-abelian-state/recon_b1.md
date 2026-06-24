# B1 reconciliation — R1 (abelian + parametric) under lenient vs position-strict scoring

`followups/non-abelian-state/recon_b1.py`. gdp_hybrid d256x4, 4000 steps, 3 seeds. `scan` = first emitted value token (the ladder's 0.83 metric); `strict` = argmax at the answer position (companion paper's position-strict style). Floor = 0.20.

| eval | value-scan | position-strict |
|---|---|---|
| L16 | 0.83±0.14 | 0.83±0.14 |
| L64 | 0.57±0.17 | 0.57±0.17 |
