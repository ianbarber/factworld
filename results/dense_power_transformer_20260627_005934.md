# Dense-vs-sparse state-supervision sweep (s5 / non-abelian composite)

`scripts/experiment_dense_supervision.py`. transformer d256x4, 4000 steps, seeds [0, 1, 2, 3, 4]. K = holder supervised every K events (K=1 dense). Guided free-run eval (events forced, holder/value slots generated). Floor = 0.20.

| K (stride) | value @L16 | value @L64 | conv(>0.5) @L16 |
|---|---|---|---|
| 1 | 0.86±0.24 | 0.22±0.01 | 4/5 |
| 8 | 0.21±0.07 | 0.20±0.04 | 0/5 |

_value = end-to-end free-running composite accuracy (generated holder + value). K=1 dense is the reproduction target (~0.95-1.00); the cliff as K grows is the sparsity result._