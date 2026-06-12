# FactWorld reference baselines

Scored suite (`REPORTED`), from-scratch, position-strict exact match (canonical metric). Recipe: d_model=320, n_layers=4, 8000 steps, seed=0, **matched across architectures**. This is the **baseline scale**; composite tasks are expected near the wall here (see §5 for the scale step that lifts the flagship composite). Columns are eval lengths tagged **(id)** in-distribution / **(ood)** held-out; `length` = depth for chain_v1, #facts for recall_copy_v1/conflict_v1, binding-chain length otherwise.


## recall_copy_v1  (floor ≈ 0.016)

| arch | L2 (id) | L3 (id) | L4 (id) | L5 (id) | L6 (ood) | L8 (ood) |
|---|---|---|---|---|---|---|
| gdp_hybrid | 1.000 | 1.000 | 0.965 | 0.850 | 0.610 | 0.340 |
| gdn_hybrid | 0.535 | 0.295 | 0.260 | 0.240 | 0.145 | 0.160 |
| transformer | 0.485 | 0.285 | 0.230 | 0.175 | 0.105 | 0.140 |
| gru | 0.465 | 0.300 | 0.220 | 0.165 | 0.130 | 0.080 |

## binding_v1  (floor ≈ 0.200)

| arch | L4 (id) | L8 (id) | L16 (id) | L32 (ood) | L64 (ood) |
|---|---|---|---|---|---|
| gdp_hybrid | 1.000 | 0.935 | 0.840 | 0.380 | 0.400 |
| gdn_hybrid | 1.000 | 1.000 | 0.995 | 0.895 | 0.780 |
| transformer | 0.475 | 0.420 | 0.410 | 0.360 | 0.350 |
| gru | 0.360 | 0.320 | 0.285 | 0.205 | 0.250 |

## composite_copy_v1  (floor ≈ 0.031)

| arch | L4 (id) | L8 (id) | L16 (id) | L32 (ood) | L64 (ood) |
|---|---|---|---|---|---|
| gdp_hybrid | 0.015 | 0.020 | 0.005 | 0.000 | 0.010 |
| gdn_hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| transformer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gru | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 |

## conflict_v1  (floor ≈ 0.016)

| arch | L2 (id) | L3 (id) | L4 (ood) | L5 (ood) |
|---|---|---|---|---|
| gdp_hybrid | 1.000 | 0.935 | 0.400 | 0.310 |
| gdn_hybrid | 0.515 | 0.365 | 0.240 | 0.220 |
| transformer | 0.525 | 0.315 | 0.255 | 0.185 |
| gru | 0.445 | 0.270 | 0.165 | 0.115 |

## chain_v1  (floor ≈ 0.167)

| arch | L2 (id) | L3 (id) | L4 (ood) | L5 (ood) |
|---|---|---|---|---|
| gdp_hybrid | 0.660 | 0.840 | 0.000 | 0.000 |
| gdn_hybrid | 0.805 | 0.725 | 0.120 | 0.005 |
| transformer | 0.210 | 0.205 | 0.210 | 0.180 |
| gru | 0.210 | 0.255 | 0.250 | 0.225 |
