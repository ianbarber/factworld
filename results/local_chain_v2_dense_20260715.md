# Local sweep — ['chain_v2'] 

d_model=320 n_layers=4 steps=8000 seeds=[0, 1, 2] train_n=8000 eval_n=200

## chain_v2  (eval lengths 4, 5)
| arch | L4 mean±std (pconv) | L5 mean±std (pconv) | holder/value @L4 |
|---|---|---|---|
| fprm | 0.12±0.04 (0%) | 0.11±0.03 (0%) | 0.12 / 0.00 |
| gdp_hybrid | 0.05±0.04 (0%) | 0.04±0.04 (0%) | 0.05 / 0.00 |
| transformer | 0.09±0.03 (0%) | 0.03±0.02 (0%) | 0.09 / 0.00 |

_pconv = fraction of seeds reaching >=0.9 at that length; holder/value = leg accuracies (2-token answers) at L4._
