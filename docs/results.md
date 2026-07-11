# FactWorld reference baselines

Scored suite (`REPORTED`), from-scratch, relaxed match (canonical metric; exact/contains/last_n are diagnostics). Recipe: d_model=320, n_layers=4, 8000 steps, seed=0, **matched across architectures**. This is the **baseline scale**; composite tasks are expected near floor here (see §5 for the scale step that lifts the flagship composite). Columns are eval lengths tagged **(id)** in-distribution / **(ood)** held-out; `length` = depth for chain_v1, #facts for recall_copy_v1/conflict_v1, binding-chain length otherwise.


## recall_copy_v1  (floor ≈ 0.016)

| arch | L2 (id) | L3 (id) | L4 (id) | L5 (id) | L6 (ood) | L8 (ood) |
|---|---|---|---|---|---|---|
| gdp_hybrid | 0.985 | 0.910 | 0.870 | 0.730 | 0.680 | 0.500 |
| gdn_hybrid | 1.000 | 1.000 | 1.000 | 0.995 | 1.000 | 1.000 |
| transformer | 0.490 | 0.325 | 0.240 | 0.235 | 0.160 | 0.125 |
| gru | 0.600 | 0.355 | 0.250 | 0.180 | 0.140 | 0.080 |

## binding_v2  (floor ≈ 0.200)

| arch | L4 (id) | L8 (id) | L16 (id) | L32 (ood) | L64 (ood) |
|---|---|---|---|---|---|
| gdp_hybrid | 1.000 | 1.000 | 0.990 | 0.845 | 0.650 |
| gdn_hybrid | 1.000 | 0.900 | 0.740 | 0.325 | 0.180 |
| transformer | 0.490 | 0.325 | 0.250 | 0.225 | 0.240 |
| gru | 0.185 | 0.165 | 0.205 | 0.150 | 0.265 |

## composite_copy_v2  (floor ≈ 0.031)

| arch | L4 (id) | L8 (id) | L16 (id) | L32 (ood) | L64 (ood) |
|---|---|---|---|---|---|
| gdp_hybrid | 0.005 | 0.020 | 0.000 | 0.000 | 0.005 |
| gdn_hybrid | 0.005 | 0.000 | 0.005 | 0.000 | 0.005 |
| transformer | 0.000 | 0.005 | 0.005 | 0.000 | 0.000 |
| gru | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## conflict_v1  (floor ≈ 0.016)

| arch | L2 (id) | L3 (id) | L4 (ood) | L5 (ood) |
|---|---|---|---|---|
| gdp_hybrid | 0.930 | 0.650 | 0.450 | 0.360 |
| gdn_hybrid | 0.495 | 0.325 | 0.205 | 0.190 |
| transformer | 0.510 | 0.345 | 0.215 | 0.220 |
| gru | 0.460 | 0.260 | 0.170 | 0.190 |

## chain_v1  (floor ≈ 0.167)

| arch | L2 (id) | L3 (id) | L4 (ood) | L5 (ood) |
|---|---|---|---|---|
| gdp_hybrid | 0.840 | 0.755 | 0.050 | 0.000 |
| gdn_hybrid | 0.695 | 0.570 | 0.090 | 0.005 |
| transformer | 0.220 | 0.210 | 0.225 | 0.125 |
| gru | 0.345 | 0.310 | 0.245 | 0.050 |
