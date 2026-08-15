# Local sweep — ['s5_chain_local_v2'] 

d_model=320 n_layers=4 steps=8000 seeds=[0, 1, 2, 3, 4, 5, 6, 7] train_n=80000 eval_n=200

## s5_chain_local_v2  (eval lengths 4, 8)
| arch | L4 per-seed | L8 per-seed | held-out loss @L4 |
|---|---|---|---|
| fprm | 0.24 0.18 0.23 0.23 0.20 0.23 0.20 0.23<br>_min 0.18 / med 0.23 / max 0.24_ | 0.22 0.18 0.14 0.17 0.21 0.15 0.19 0.21<br>_min 0.14 / med 0.19 / max 0.22_ | 0.404 |
| gdp_hybrid | 0.16 0.18 0.26 0.23 0.15 0.21 0.20 0.14<br>_min 0.14 / med 0.20 / max 0.26_ | 0.14 0.17 0.20 0.19 0.17 0.19 0.27 0.19<br>_min 0.14 / med 0.19 / max 0.27_ | 0.388 |
| _floor: uniform_non_start_ | **0.200** | **0.200** | — |
| _floor: initial_map_chase_ | 0.195 | 0.160 | — |
| _floor: uniform_ | 0.167 | 0.167 | — |
| _floor: echo_ | 0.000 | 0.000 | — |

_Per-seed values, one per seed in seed order: this family is bimodal at the emergence threshold, so a mean hides a converged seed. Floor rows are shallow-adversary accuracies recomputed from the exact scored items (factworld.validity): `initial_map_chase` ignores every event and dereferences the stated initial map; `uniform_non_start` is chance given that the gated stream never answers the queried agent. A cell's operative floor is the LARGEST value in its column (bold) — which adversary that is changes with k, depth and length. Held-out loss is nats/token on the same document construction the run trained on._
