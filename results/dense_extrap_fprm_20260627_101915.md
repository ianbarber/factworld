# Dense-vs-sparse state-supervision sweep (s5 / non-abelian composite)

`scripts/experiment_dense_supervision.py`. fprm d256x4, 4000 steps, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. K = holder supervised every K events (K=1 dense). Guided free-run eval (events forced, holder/value slots generated). Floor = 0.20.

| K (stride) | value @L16 | value @L32 | value @L64 | value @L128 | conv(>0.5) @L16 |
|---|---|---|---|---|---|
| 1 | 1.00±0.00 | 0.58±0.30 | 0.20±0.05 | 0.23±0.05 | 10/10 |

_value = end-to-end free-running composite accuracy (generated holder + value). K=1 dense is the reproduction target (~0.95-1.00); the cliff as K grows is the sparsity result._