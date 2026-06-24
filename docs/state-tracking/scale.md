# Scale results — the architecture lever (final)

`scripts/scale_confirm.py`. Task: the k=5 in-context-copy
composite (`composite_copy_scale_v1`; 1-of-5 recall composed with a `give`-stream binding, resampled per
example so the recall leg cannot be memorized). Metric: position-strict exact match. The k=5 pool keeps the
recall leg independently learnable, so a composite floor is attributable to composition, not recall capacity.

## Confirmation — matched ~45M params, matched compute, multi-seed

25 000 steps; mean ± std; **p(converge) = fraction of seeds > 0.5**. gdp 5 seeds, transformer 5, gdn 3.

| arm (~45M, matched compute) | L16 (in-distribution) | L64 (4×) |
|---|---|---|
| **gdp_hybrid** (GatedDeltaProduct, n_h=4 Householder) | **0.48 ± 0.21  (1/5)** | 0.17 ± 0.09  (0/5) |
| transformer (param-matched, d_ff=3072) | 0.01 ± 0.01  (0/5) | 0.00 ± 0.00  (0/5) |
| gdn_hybrid (GatedDeltaNet, no product structure) | 0.01 ± 0.00  (0/3) | 0.01 ± 0.00  (0/3) |

gdp per-seed @L16: 0.90 / 0.41 / 0.34 / 0.38 / 0.39 (mean 0.48; the 0.90 is the single seed that converges).

**Discovery point (scale-solubility):** a 6M gdp hybrid floors the same composite (0.08 ± 0.07 @L16,
0.02 ± 0.01 @L64); the ≈45M step lifts it off the floor. A pure-copy positive control passes (1.00) at 45M,
so the composite floor is composition, not retrieval.

## What it establishes

1. **Architecture is the deciding variable, multi-seed.** Only the gdp hybrid rises off the 0.01 floor
   in-distribution; the param-matched transformer floors 5/5 and the GatedDeltaNet hybrid 3/3.
2. **The lever is gated-delta recurrence vs. attention — not the product structure specifically.** At the
   matched recipe (lr 1e-3) only gdp lifts and gdn floors 0/3, which initially looked product-specific. But
   a matched LR sweep of the gdn arm (below) lifts it too: at lr 3e-4 gdn converges in-distribution and
   length-extrapolates (L16 1.00, L64 0.825), 1/10 cells over the grid, while reproducing the 0/3 floor at
   lr 1e-3. The gdn floor was therefore LR-specific, not a capability bound. Both gated-delta recurrent
   hybrids learn this composite (fragilely, at different optimal LRs); the transformer floors across its
   entire sweep (0/10). This composite is last-write-wins binding × in-context recall, neither of which
   requires the non-abelian product transition — that is the S5 *state* leg's need (§3.1), which this task
   does not exercise — so the dissociation here is recurrence vs. attention. WITHIN recurrence, the product
   hybrid is markedly more learning-rate-robust (gdp converges 7/10 cells across the LR grid vs gdn's 1/10;
   `gdp_lr_sweep.py`) and length-extrapolates when tuned, so it is the architecture we carry forward — on
   measured robustness, not only because it also supplies the state leg.
3. **Learnable, and length-extrapolable when tuned.** At the default recipe the lift is seed-fragile (1/5 at
   lr 1e-3); but a learning-rate sweep shows the product hybrid converges across a broad band (7/10 cells)
   and length-extrapolates at a tuned LR (0.875 @4× at lr 5e-4) — the earlier "0/5 at 4×" was a 1e-3-recipe
   artifact. The ≈45M step is still a two-point step (6M floor → 45M lift), not a curve or an emergence claim.

## The transformer floor is convergence, not tuning (`scripts/transformer_lr_sweep.py`)

To rule out "the transformer was just badly tuned," we sweep the learning rate for the param-matched 45M
transformer on the same k=5 composite (25k steps, 2 seeds per LR; strict-acc, L16 in-distribution):

| lr | 3e-4 | 5e-4 | 1e-3 | 2e-3 | 3e-3 |
|---|---|---|---|---|---|
| L16 mean | 0.045 | 0.003 | 0.077 | 0.000 | 0.005 |
| max / converge | 0.08 (0/2) | 0.005 (0/2) | 0.155 (0/2) | 0.0 (0/2) | 0.005 (0/2) |

The transformer floors in-distribution in **every** cell (best single run 0.155, 0/10 converge). The floor
is robust to the learning rate — a convergence failure at this matched budget, not a tuning artifact (cf.
the convergence-not-capacity account of recall, Okpekpe & Orvieto 2025). We do not claim transformers
cannot recall (Arora et al. 2023); at this budget the dense transformer does not find the composite circuit.

## The gdn floor is also learning-rate-specific (`scripts/gdn_lr_sweep.py`)

The confirmation above ran gdn at the default lr 1e-3 only (0/3). Because Okpekpe & Orvieto (2025) report
recurrent models are the LR-sensitive ones, we sweep the gdn 45M arm on the same composite (25k steps,
2 seeds per LR; strict-acc, L16 in-distribution):

| lr | 3e-4 | 5e-4 | 1e-3 | 2e-3 | 3e-3 |
|---|---|---|---|---|---|
| L16 mean | 0.675 | 0.028 | 0.007 | 0.005 | 0.045 |
| max / converge | 1.00 (1/2) | 0.035 (0/2) | 0.010 (0/2) | 0.010 (0/2) | 0.090 (0/2) |

At lr 3e-4 one of two seeds converges in-distribution (L16 1.00) and length-extrapolates (L64 0.825); the
default lr 1e-3 reproduces the 0/3 floor. The 1/2 at 3e-4 is a 2-seed undercount: the 5-seed confirmation
below (W2) gives gdn **4/5 converged in-distribution at 3e-4**, so gdn converges *robustly* at its tuned
rate — the "1/10 cells over the grid" reflects LR-fragility (it floors at every other rate), not weak
in-distribution capability. So gdn is **capable-but-LR-fragile** on this composite, not floored like the
transformer (0/10 across its own sweep). This overturns the earlier "product is the lever"
reading: the composite floor reflects **gated-delta recurrence vs. attention**, with the product structure's
distinctive necessity scoped to the S5 state leg (§3.1). The single gdn cell that converges also
length-extrapolates (0.825 @4×); a matched gdp sweep (next section) shows gdp converges far more broadly
across LRs and also length-extrapolates at a tuned LR.

## The product hybrid is far more learning-rate-robust (`scripts/gdp_lr_sweep.py`)

To complete the symmetry, we sweep the gdp 45M arm on the same grid (25k steps, 2 seeds per LR):

| lr | 3e-4 | 5e-4 | 1e-3 | 2e-3 | 3e-3 |
|---|---|---|---|---|---|
| L16 mean | 0.877 | 0.992 | 0.865 | 0.407 | 0.203 |
| max / converge | 0.995 (2/2) | 1.000 (2/2) | 0.940 (2/2) | 0.515 (1/2) | 0.405 (0/2) |

gdp converges in-distribution across the whole 3e-4–2e-3 band (**7/10 cells**), flooring only at the highest
LR (3e-3) — versus gdn's **1/10** and the transformer's **0/10**. It also length-extrapolates at a tuned LR
(5e-4 s1: L16 0.985, L64 **0.875** @4×), so the original "gdp does not length-extrapolate (0/5 at 4×)" was a
1e-3-recipe artifact. (2 seeds/cell is noisy — at 1e-3 the sweep gives 2/2 where the original 5-seed run
gave 1/5 — so the robustness *band* is the claim, not the exact per-LR rate. A 5-seed confirmation at the
best LR (`scripts/gdp_confirm_5e4.py`, lr 5e-4) pins the tuned-rate point estimate: **L16 0.87 ± 0.15
(5/5 converge), L64 0.76 ± 0.25 (3/5 extrapolate)** — robust in-distribution, majority length-extrapolation,
versus the default recipe's 0.48 / 0.17 (1/5, 0/5).)

**Net — all three arms, matched 5-LR × 2-seed sweeps: gdp 7/10, gdn 1/10, transformer 0/10** (grid cells;
the gdn cell-count understates its in-distribution capability — at its one converging rate the 5-seed confirm
is 4/5, see W2 below). The §5 reading is three-layer: (i) *recurrence vs. attention* for can/can't compose —
both recurrent hybrids have a converging LR (gdn robustly so at 3e-4, 4/5), the transformer none across its
sweep; (ii) within recurrence, the *product structure* makes composition far more learning-rate-robust (gdp's
broad band vs gdn's single LR) and more reliably length-extrapolable when tuned (gdp 3/5 vs gdn 1/5 at 4×);
(iii) the non-abelian product transition is required for the S5 *state* leg (§3.1), which this composite does
not exercise.

## Fair-config controls (critical-review W2/W3)

The 2-seed sweeps could under-power a cell, and the architecture axis co-varied with attention config.
Two follow-ups settle both.

**W2 — gdn at its converging rate, 5 seeds (`scripts/gdn_confirm_3e4.py`, lr 3e-4):** L16 0.78 ± 0.27
(**4/5 converge**), L64 0.26 ± 0.19 (**1/5 extrapolate**). So gdn converges in-distribution robustly at 3e-4
(the 2-seed "1/10" undersold it) but extrapolates only fragilely — the lone 0.825 in the sweep was bf16
variance, not robust. The product edge is therefore specifically *extrapolation* (gdp 3/5 vs gdn 1/5).

**W3 — fair config (`scripts/fair_config.py`):** giving each arm its best config does not change the story.
- Transformer at **n_heads=8 (head_dim 64) + residual-scaled init**, 5 LRs × 2 seeds: still **0/10** (best
  single run 0.20). The floor is earned, not a head-count/init artifact.
- Recurrent arms with the **published short-convolution enabled** (previously forced off): gdp (lr 5e-4)
  L16 0.95 (3/3), L64 **0.70 (3/3)**; gdn (lr 3e-4) L16 0.98 (3/3), L64 0.34 (1/3). Short-conv if anything
  firms up the recurrents (gdp extrapolation 3/5 → 3/3). The recurrence-vs-attention + product-extrapolation
  dissociation holds when every arm gets its best config.
