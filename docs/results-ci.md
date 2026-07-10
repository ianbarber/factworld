# FactWorld dissociation cells — 3-seed CIs

The two architecture-dissociating tasks across 4 archs, seeds [0, 1, 2] (mean ± std), d_model=320×4, 8000 steps, matched compute. Firms up the single-seed `docs/results.md`: the hybrids dissociate from the transformer/gru floors on both cells; gdp reads with tight CIs, gdn is seed-bimodal (stds to ±0.39). Columns tagged (id)/(ood).


## recall_copy_v1

| arch | L2 (id) | L3 (id) | L4 (id) | L5 (id) | L6 (ood) | L8 (ood) |
|---|---|---|---|---|---|---|
| gdp_hybrid | 0.99±0.01 | 0.97±0.04 | 0.92±0.04 | 0.80±0.05 | 0.63±0.07 | 0.42±0.07 |
| gdn_hybrid | 0.90±0.15 | 0.82±0.25 | 0.79±0.29 | 0.77±0.33 | 0.73±0.37 | 0.72±0.39 |
| transformer | 0.49±0.01 | 0.31±0.02 | 0.25±0.02 | 0.19±0.03 | 0.15±0.01 | 0.12±0.01 |
| gru | 0.56±0.04 | 0.35±0.00 | 0.26±0.01 | 0.19±0.03 | 0.17±0.02 | 0.10±0.02 |

## binding_v2

| arch | L4 (id) | L8 (id) | L16 (id) | L32 (ood) | L64 (ood) |
|---|---|---|---|---|---|
| gdp_hybrid | 1.00±0.00 | 1.00±0.00 | 0.99±0.01 | 0.77±0.10 | 0.59±0.14 |
| gdn_hybrid | 1.00±0.00 | 0.97±0.05 | 0.91±0.12 | 0.74±0.30 | 0.69±0.36 |
| transformer | 0.49±0.01 | 0.34±0.03 | 0.27±0.01 | 0.23±0.01 | 0.25±0.01 |
| gru | 0.18±0.00 | 0.17±0.00 | 0.20±0.00 | 0.15±0.00 | 0.27±0.00 |
