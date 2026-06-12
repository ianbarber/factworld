# FactWorld dissociation cells — 3-seed CIs

The two architecture-dissociating tasks across 4 archs, seeds [0, 1, 2] (mean ± std), d_model=320×4, 8000 steps, matched compute. Firms up the single-seed `docs/results.md`: gdn is the binding specialist, gdp the recall/composition generalist. Columns tagged (id)/(ood).


## recall_copy_v1

| arch | L2 (id) | L3 (id) | L4 (id) | L5 (id) | L6 (ood) | L8 (ood) |
|---|---|---|---|---|---|---|
| gdp_hybrid | 1.00±0.00 | 0.99±0.01 | 0.96±0.04 | 0.85±0.06 | 0.65±0.04 | 0.33±0.02 |
| gdn_hybrid | 0.51±0.02 | 0.29±0.02 | 0.27±0.01 | 0.21±0.02 | 0.16±0.03 | 0.12±0.03 |
| transformer | 0.48±0.02 | 0.28±0.00 | 0.24±0.01 | 0.19±0.01 | 0.13±0.02 | 0.13±0.00 |
| gru | 0.33±0.22 | 0.22±0.15 | 0.15±0.09 | 0.11±0.06 | 0.09±0.05 | 0.05±0.03 |

## binding_v1

| arch | L4 (id) | L8 (id) | L16 (id) | L32 (ood) | L64 (ood) |
|---|---|---|---|---|---|
| gdp_hybrid | 1.00±0.00 | 0.97±0.03 | 0.92±0.06 | 0.51±0.33 | 0.54±0.32 |
| gdn_hybrid | 0.91±0.12 | 0.81±0.17 | 0.79±0.19 | 0.56±0.25 | 0.48±0.22 |
| transformer | 0.50±0.02 | 0.42±0.03 | 0.42±0.02 | 0.32±0.06 | 0.30±0.07 |
| gru | 0.37±0.00 | 0.32±0.02 | 0.27±0.01 | 0.22±0.01 | 0.25±0.03 |

## Attention-free recall ablation — what supplies recall? (`scripts/recall_attention_test.py`, 3 seeds)

In-context-copy recall (`recall_copy_v1`) for the **attention-free** pure backbones vs the hybrid, by pool
size. The pure product backbone recalls like the hybrid; the diagonal pure backbone floors — so the
**product recurrence**, not the attention layer, supplies recall here.

| arch | pool 2 | pool 3 | pool 4 | pool 6 | pool 8 |
|---|---|---|---|---|---|
| **gdp_pure** (product, no attention) | **1.00** | 0.98 | 0.92 | 0.59 | 0.35 |
| gdn_pure (diagonal, no attention) | 0.61 | 0.36 | 0.29 | 0.17 | 0.12 |
| gdp_hybrid (product + attention) | 1.00 | 1.00 | 0.96 | 0.65 | 0.33 |

`gdp_pure` ≈ `gdp_hybrid` (attention adds ~nothing to recall) ≫ `gdn_pure` (0.61). Combined with the
attention-based transformer (0.48) and gdn-hybrid (0.51) from the table above, the discriminating variable
for recall is the product transition, not the attention pathway.
