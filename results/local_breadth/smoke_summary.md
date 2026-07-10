# Local breadth-mirror — composite_copy_v2.scaled(k=2B, recall_pool=B), m=4

d_model=256 n_layers=4 steps=1000 batch=32 train_n=8000 eval_n=50 seeds=[0] train_lengths=[4, 8, 16]

| B | arch | L16 mean±std (pconv) | L64 mean±std (pconv) | holder/value @L16 | holder/value @L64 |
|---|---|---|---|---|---|
| 6 | gdp_hybrid | 0.00±0.00 (0%) | 0.00±0.00 (0%) | 0.26 / 0.00 | 0.08 / 0.02 |

_Relaxed match (canonical). pconv = fraction of seeds >=0.9 (composite is bimodal — read pconv, not the mean). holder = binding leg, value = lookup leg. L16 is in-distribution, L64 is OOD extrapolation. Per-rung object-filter floor E[1/w] moves with (m, L) not pool: ~0.41@L16 / ~0.15@L64 at m=4._