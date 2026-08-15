# Local sweep — ['s5_chain_local_v2'] 

d_model=320 n_layers=4 steps=8000 seeds=[0, 1, 2, 3, 4, 5, 6, 7] train_n=80000 eval_n=200

## s5_chain_local_v2  (eval lengths 4, 8)
| arch | L4 per-seed | L8 per-seed | held-out loss @L4 |
|---|---|---|---|
| fprm | 0.16 0.17 0.12 0.15 0.12 0.15 0.12 0.16<br>_min 0.12 / med 0.15 / max 0.17_ | 0.12 0.13 0.17 0.19 0.14 0.15 0.12 0.15<br>_min 0.12 / med 0.14 / max 0.19_ | 0.485 |
| gdp_hybrid | 0.14 0.14 0.11 0.11 0.13 0.20 0.14 0.14<br>_min 0.11 / med 0.14 / max 0.20_ | 0.12 0.12 0.14 0.12 0.14 0.17 0.11 0.17<br>_min 0.11 / med 0.13 / max 0.17_ | 0.457 |
| _floor: initial_map_chase_ | **0.335** | **0.210** | — |
| _floor: uniform_non_start_ | 0.143 | 0.143 | — |
| _floor: uniform_ | 0.125 | 0.125 | — |
| _floor: echo_ | 0.000 | 0.000 | — |

_Per-seed values, one per seed in seed order: this family is bimodal at the emergence threshold, so a mean hides a converged seed. Floor rows are shallow-adversary accuracies recomputed from the exact scored items (factworld.validity): `initial_map_chase` ignores every event and dereferences the stated initial map; `uniform_non_start` is chance given that the gated stream never answers the queried agent. A cell's operative floor is the LARGEST value in its column (bold) — which adversary that is changes with k, depth and length. Held-out loss is nats/token on the same document construction the run trained on._
