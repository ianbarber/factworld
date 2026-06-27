# Dense-vs-sparse state-supervision sweep (s5 / non-abelian composite)

`scripts/experiment_dense_supervision.py`. transformer d256x4, 4000 steps, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. K = holder supervised every K events (K=1 dense). Guided free-run eval (events forced, holder/value slots generated). Floor = 0.20.

| K (stride) | value @L16 | value @L64 | conv(>0.5) @L16 |
|---|---|---|---|
| 1 | 0.83±0.25 | 0.19±0.04 | 8/10 |
| 2 | 0.22±0.03 | 0.20±0.03 | 0/10 |
| 4 | 0.18±0.05 | 0.20±0.04 | 0/10 |
| 8 | 0.18±0.06 | 0.19±0.03 | 0/10 |

_value = end-to-end free-running composite accuracy (generated holder + value). K=1 dense is the reproduction target (~0.95-1.00); the cliff as K grows is the sparsity result._