# Dense-supervised state-tracking results (S5 / A5) — our own runs

`scripts/dense_s5.py` (reimplemented from the canonical group word problem, Liu et al. 2023; not imported
from any external repo). Read a sequence of group elements g_1…g_t; predict the running prefix product
g_1·…·g_t at **every** position (dense per-token supervision, loss on all but position 0). Eval = a single
forward pass + masked argmax (no autoregressive generation). Train length 32, eval 32/64/128 (1×/2×/4×),
3 seeds, 25k steps, d_model 288, 6 layers. Metric = seq-exact (whole sequence correct) / token-acc
(per-position argmax). The state probe uses the **attention-free pure GatedDeltaProduct** (`gdp_pure`), so it
isolates the product *recurrence* — the attention layer is for recall (§3.2), not state.

This is the regime an earlier answer-only run lacked: supervising only the single final role and evaluating
by autoregressive generation floored (~0.20). Non-abelian state-tracking needs the dense per-token signal.

## S5 (5 generators; |S5| = 120)

| arch | L32 | L64 (2×) | L128 (4×) | pattern |
|---|---|---|---|---|
| **gru** (NC¹ via unrolled recurrence) | 1.00 / 1.00 | 1.00 / 1.00 | **1.00 / 1.00** | flat |
| **gdp n_h=4, neg-eig** (product scan) | 1.00 / 1.00 | 1.00 / 1.00 | **0.75 / 0.99** | flat |
| gdp n_h=1 null, neg-eig | 1.00 / 1.00 | 0.88 / 0.99 | 0.44 / 0.86 | seed-fragile (2/3 extrapolate) |
| gdn (diagonal Δ, [0,1]) | 0.95 / 1.00 | **0.00** / 0.74 | **0.00** / 0.74 | shortcut (seq-exact collapses) |
| transformer (softmax) | 0.94 / 0.99 | **0.00** / 0.58 | **0.00** / 0.29 | shortcut |

(gdp n_h=4 per-seed token-acc @4×: 0.988 / 0.987 / 0.999. gdp n_h=1 null per-seed @4×: 0.983 / 0.990 / 0.619.)

## A5 (the smallest non-abelian simple group; not-S5-specific control)

| arch | L32 | L64 (2×) | L128 (4×) |
|---|---|---|---|
| **gdp n_h=4, neg-eig** | 1.00 / 1.00 | 1.00 / 1.00 | **0.86 / 1.00** |
| gdp n_h=1 null, neg-eig | 1.00 / 1.00 | 1.00 / 1.00 | 0.69 / 0.98 |

## Reading (honest)

1. **The product recurrence does non-abelian state-tracking, attention-free.** `gdp_pure` (n_h=4, neg-eig)
   and the GRU flat-extrapolate S5 and A5 (token-acc ≈0.99 / 1.00 @4×); the diagonal GatedDeltaNet and the
   softmax transformer **shortcut-learn** — they fit the train length then collapse to seq-exact 0 at 2×/4×.
   This reproduces the prior-art dissociation (Grazzi et al. 2024; Siems et al. 2025; Liu et al. 2023) at our
   scale, in our own repo, and attributes the **state leg to the product recurrence** (not the attention layer).
2. **We do NOT claim a clean n_h=4-necessity from the isolated probe.** The n_h=1 single-reflection null is
   **seed-fragile** here — it extrapolates on 2/3 seeds (token-acc 0.98) and decays on one. Over ≥4 tokens a
   product of single reflections can compose a 5-cycle (Cartan–Dieudonné), so this isolated probe cannot
   separate optimization from expressivity.
   The clean "product structure is load-bearing" evidence lives on the **composite** (§4 iso-param control
   1/5→2/5→5/5; §5 gdn 0/3), not on this isolated probe.
3. **Implication for FactWorld (§4).** Sₖ state-tracking *does* train under dense supervision; FactWorld's
   hard-state composite is **single-query / answer-only by design** (the harder, agentic regime), which is why
   direct Sₖ floors there. So the composite floor is the regime, not the backbone.

*Caveat:* gdp n_h=4 seq-exact @4× (0.75) is below the DeltaProduct-scale result (Siems et al. 2025; ≈0.98) because of fewer steps
(25k vs 50k) and the strict whole-sequence metric; token-acc (0.99) matches. The dissociation (gdp/gru
extrapolate; gdn/transformer collapse to seq-exact 0) is robust either way.
