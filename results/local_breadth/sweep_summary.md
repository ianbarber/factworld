# Local breadth-mirror — composite_copy_v2.scaled(k=2B, recall_pool=B), m=4

d_model=256 n_layers=4 steps=8000 batch=32 train_n=8000 eval_n=200 seeds=[0, 1, 2] train_lengths=[4, 8, 16]

| B | arch | L16 mean±std (pconv) | L64 mean±std (pconv) | holder/value @L16 | holder/value @L64 |
|---|---|---|---|---|---|
| 6 | gdp_hybrid | 0.04±0.02 (0%) | 0.00±0.00 (0%) | 0.56 / 0.06 | 0.09 / 0.01 |
| 6 | transformer | 0.02±0.00 (0%) | 0.01±0.01 (0%) | 0.23 / 0.07 | 0.18 / 0.03 |
| 8 | gdp_hybrid | 0.01±0.01 (0%) | 0.00±0.00 (0%) | 0.41 / 0.02 | 0.14 / 0.00 |
| 8 | transformer | 0.00±0.00 (0%) | 0.00±0.00 (0%) | 0.17 / 0.01 | 0.15 / 0.01 |
| 12 | gdp_hybrid | 0.00±0.00 (0%) | 0.00±0.00 (0%) | 0.43 / 0.01 | 0.19 / 0.01 |
| 12 | transformer | 0.00±0.00 (0%) | 0.00±0.00 (0%) | 0.15 / 0.01 | 0.10 / 0.00 |
| 16 | gdp_hybrid | 0.00±0.00 (0%) | 0.00±0.00 (0%) | 0.41 / 0.01 | 0.16 / 0.01 |
| 16 | transformer | 0.00±0.00 (0%) | 0.00±0.00 (0%) | 0.13 / 0.02 | 0.08 / 0.00 |
| 24 | gdp_hybrid | 0.01±0.01 (0%) | 0.00±0.00 (0%) | 0.67 / 0.01 | 0.20 / 0.01 |
| 24 | transformer | 0.00±0.00 (0%) | 0.00±0.00 (0%) | 0.08 / 0.01 | 0.03 / 0.00 |

_Relaxed match (canonical). pconv = fraction of seeds >=0.9 (composite is bimodal — read pconv, not the mean). holder = binding leg, value = lookup leg. L16 is in-distribution, L64 is OOD extrapolation. Per-rung object-filter floor E[1/w] moves with (m, L) not pool: ~0.41@L16 / ~0.15@L64 at m=4._