# Local sweep — ['s5_chain_local_v2'] 

d_model=320 n_layers=4 steps=8000 seeds=[0, 1, 2, 3, 4, 5, 6, 7] train_n=8000 eval_n=200

## s5_chain_local_v2  (eval lengths 4, 8)
| arch | L4 per-seed | L8 per-seed | checkpoint acc @L4 | held-out loss @L4 |
|---|---|---|---|---|
| fprm | 0.28 0.28 0.24 0.27 0.27 0.29 0.26 0.28<br>_min 0.24 / med 0.28 / max 0.29_ | 0.27 0.20 0.23 0.23 0.19 0.22 0.27 0.23<br>_min 0.19 / med 0.23 / max 0.27_ | 0.27 | 0.263 |
| gdp_hybrid | 0.27 0.23 0.29 0.27 0.22 0.23 0.24 0.24<br>_min 0.22 / med 0.24 / max 0.29_ | 0.25 0.27 0.29 0.23 0.28 0.21 0.30 0.22<br>_min 0.21 / med 0.26 / max 0.30_ | 0.25 | 0.349 |
| _floor: initial_map_chase_ | **0.335** | 0.325 | — | — |
| _floor: uniform_non_start_ | 0.333 | **0.333** | — | — |
| _floor: initial_map_backhop_ | 0.320 | 0.330 | — | — |
| _floor: uniform_ | 0.250 | 0.250 | — | — |
| _floor: echo_ | 0.000 | 0.000 | — | — |

_Per-seed values, one per seed in seed order: this family is bimodal at the emergence threshold, so a mean hides a converged seed. Floor rows are shallow-adversary accuracies recomputed from the exact scored items (factworld.validity): `initial_map_chase` ignores every event and dereferences the stated initial map; `uniform_non_start` is chance given that the gated stream never answers the queried agent. A cell's operative floor is the LARGEST value in its column (bold) — which adversary that is changes with k, depth and length. Held-out loss is nats/token on the same document construction the run trained on._
