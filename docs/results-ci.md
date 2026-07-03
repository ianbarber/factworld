# FactWorld dissociation cells — 3-seed CIs

The two architecture-dissociating tasks across 4 archs, seeds [0, 1, 2] (mean ± std), d_model=320×4, 8000 steps, matched compute. Firms up the single-seed `docs/results.md`: gdn is the binding specialist, gdp the recall/composition generalist. Columns tagged (id)/(ood).


## recall_copy_v1

| arch | L2 (id) | L3 (id) | L4 (id) | L5 (id) | L6 (ood) | L8 (ood) |
|---|---|---|---|---|---|---|
| gdp_hybrid | 0.98±0.02 | 0.96±0.03 | 0.88±0.10 | 0.78±0.07 | 0.63±0.08 | 0.35±0.05 |
| gdn_hybrid | … | … | … | … | … | … |
| transformer | … | … | … | … | … | … |
| gru | … | … | … | … | … | … |

## binding_v1

| arch | L4 (id) | L8 (id) | L16 (id) | L32 (ood) | L64 (ood) |
|---|---|---|---|---|---|
| gdp_hybrid | … | … | … | … | … |
| gdn_hybrid | … | … | … | … | … |
| transformer | … | … | … | … | … |
| gru | … | … | … | … | … |
