# Local sweep — ['s5_chain_local_v2'] 

d_model=320 n_layers=4 steps=8000 seeds=[0, 1, 2] train_n=8000 eval_n=200

## s5_chain_local_v2  (eval lengths 4, 8)
| arch | L4 mean±std (pconv) | L8 mean±std (pconv) | holder/value @L4 |
|---|---|---|---|
| fprm | 0.37±0.03 (0%) | 0.28±0.01 (0%) | 0.37 / 0.00 |
| gdp_hybrid | 0.33±0.01 (0%) | 0.33±0.02 (0%) | 0.33 / 0.00 |

_pconv = fraction of seeds reaching >=0.9 at that length; holder/value = leg accuracies (2-token answers) at L4._
