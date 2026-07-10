# Local sweep — ['chain_v1'] 

d_model=320 n_layers=4 steps=8000 seeds=[0, 1, 2] train_n=8000 eval_n=200

## chain_v1  (eval lengths 4, 5)
| arch | L4 mean±std (pconv) | L5 mean±std (pconv) | holder/value @L4 |
|---|---|---|---|
| fprm | 0.20±0.01 (0%) | 0.21±0.02 (0%) | 0.20 / 0.00 |
| gdp_hybrid | 0.02±0.01 (0%) | 0.00±0.00 (0%) | 0.02 / 0.00 |
| transformer | 0.22±0.01 (0%) | 0.06±0.02 (0%) | 0.22 / 0.00 |

_pconv = fraction of seeds reaching >=0.9 at that length; holder/value = leg accuracies (2-token answers) at L4._
