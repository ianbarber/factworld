# Recipe: training a recurrent model for non-abelian state-tracking + recall that generalizes in length

A consolidated, reproducible protocol distilled from this follow-on study. Target: a sequence model that
maintains **non-abelian latent state** (the S₅ role-permutation word problem — NC¹, not shortcuttable) and
**recalls** a fact keyed by that state, and that **runs past its trained length**. Architecture held fixed:
`gdp_hybrid` (a GatedDeltaProduct linear-attention recurrence with `num_householder=4`, negative eigenvalues,
interleaved with one RoPE attention layer per four), d384×6 ≈ 18.5M, single RTX 3090. Floor = 1/k = 0.20.

Each step is tagged **[validated]** (an experiment in this repo supports it) or **[open]** (the live frontier).
Script pointers are to `followups/non-abelian-state/`.

---

### Step 0 — Architecture. [validated]
Use the product-structured recurrence (`gdp_hybrid`, `num_householder=4`, `allow_neg_eigval=True`). The negative
eigenvalues / Householder products are load-bearing for S₅ expressivity; transformer / attention-only floor.
**Do not** scale width to fix extrapolation — the horizon wall is flat to 357M (`scale_big.py`). **Do not**
turn on `use_short_conv` as a drop-in — it *floors in-distribution learning* in this config (a clean negative,
`base_reliability.py`; see Step 5), despite fla's generic warning and FPRM's result in a different architecture.

### Step 1 — Form the state circuit with near-dense process supervision. [validated]
Interleave the oracle's intermediate holder after events at stride **K ≤ 2** (every event, or every other). This
is a **cliff, not a slope**: at K = 4 even teacher-forced per-step accuracy is at floor — the recurrence never
discovers the permutation computation (`supervision_sweep.py`). Answer-only supervision gives *zero* traction,
and outcome-reward RL (GRPO) cannot climb it (`rl_grpo.py`) — the exploration barrier is ~k⁻ⁿ. **K = 1 (dense)
is required for the formed circuit to later extrapolate**; K = 2 forms it at the trained length only.

### Step 2 — Let recall ride free. [validated]
Given a *resolved* holder, recalling its property is free: `P(value | holder✓) = 1.0`, identically for parametric
(in-weights) and in-context facts, and parametric extrapolates at least as well (`decompose_r3.py`,
`dense_capstone.py`). **Never supervise recall separately** — it is not the bottleneck; the entire difficulty is
the state leg.

### Step 3 — Internalize (remove the scratchpad) by mixed-density exposure. [validated]
Train on a *mix* of dense-supervised and answer-only examples (random K per example). `mixed ≥ anneal` on every
metric — exposure to *both* densities is the lever, the dense→sparse *order* is not (`curriculum.py`). This yields
a model that answers with no scratchpad at the trained length.

### Step 4 — Set the trained-length distribution to the target horizon. [validated]
Extrapolation is gated by *what lengths you train on*, not width. Include a **≥ 20 % fraction of examples at (near)
the target length**: 5 % does nothing, 20 % unlocks, 50 % adds nothing (threshold in (5 %, 20 %], plateau ≈ 0.67
at 18.5M); **concentration at the target beats uniform coverage** (`length_mix.py`). The reachable horizon is a
*soft cap* at ≈ max-trained-length, decaying to floor by ≈ 2×. Supervision density (Step 1) and training horizon
are **orthogonal** levers. *Labeled-long training alone plateaus below 1.0 — Steps 5–6 reach 8× further,
label-free.*

### Step 5 — Select a clean base by free-running L16. [validated; reliability is the open part]
The post-training lever (Step 6) is **reliable given a clean base** but base quality is a lottery
(`post_reliability.py`: between- vs within-base variance ≈ 300×). So:
- Train **K base seeds**; rank by **free-running answer-only L16 e2e**; keep **L16 ≥ 0.95**. The base whose
  in-distribution circuit is clean reliably posts to a length-general circuit; ≤ 0.86 floors (`base_select.py`:
  top L128 0.93/0.89, bottom floors).
- **Do not** rank by teacher-forced per-step accuracy (`dense_h16`) — it is **saturated at 1.00 for every base**,
  clean or not; the failure is *free-running* autoregressive stability, not learning the map. The L24/L32 base
  decay is *inverted* and misleading too.
- **Cost / the open frontier:** clean bases (L16 ≥ 0.95) are **rare — ~1/7 of seeds** — so budget **K ≈ 15–20**.
  Cheap levers that modestly help (`base_reliability.py`): **GDP forget-gate retention-init + EMA** over the
  cosine tail roughly *doubles* the clean rate (~1/8 → 2/8, small-n) by lifting the mid-distribution; EMA alone is
  marginal. **Scheduled sampling** (`schedsample.py`: train on the model's own holders) gives the best
  *distributional* lift but does not clear the clean bar more often; **short-conv** drop-in *floors* in-distribution
  (non-transfer). Post-training **repairs in-distribution accuracy but cannot install length-generality on a base
  that was not already clean** (the clean recurrence must pre-exist). **Making clean bases common (raising the ~1/7
  rate) is the principal open problem** — no lever tried makes them common. [open]

### Step 6 — Post-train with unlabeled deep-state coverage. [validated, given a clean base]
On the selected clean base, run **1500 steps at lr 3e-4** on *unlabeled* burn-in examples — prepend B ∈ {0…192}
random events before a 16-event labeled window so full BPTT drives the recurrent state to depths only seen at
test time, with **no labels at length** (`post_state.py`, `build_burnin`). This calibrates the state distribution
the online-carry path never visits in training (the "unexplored states" mechanism, Buitrago Ruiz & Gu 2025).
Result: **length-general non-abelian extrapolation to ≈ 8× the trained length, label-free** (clean base →
L128 ≈ 0.85–0.93). **From-scratch** burn-in does *not* work (it blocks circuit formation, `carried_state.py`) —
this must be a *post* step on an already-clean circuit.

---

## TL;DR
Form the state circuit with dense (K ≤ 2, ideally K = 1) process supervision; let recall ride free; internalize by
mixed-density exposure; train ≥ 20 % of examples at the target length; **select a base by free-running L16 ≥ 0.95**
(the rare, decisive ingredient — train K ≈ 15–20 and pick); then **post-train with unlabeled deep-state burn-in**
to extend the horizon 8× label-free. The non-abelian leg has no length-general *shortcut*, but its online-carry
solution **can** be calibrated to extrapolate — reliably, once you have a clean base.

## What is validated vs open
- **Validated end-to-end:** Steps 0–4, 6, and the *select-then-post* reliability of Step 5. Given enough base
  seeds, this reliably yields a length-general (8×) non-abelian state + recall circuit, with parametric recall
  free.
- **Open / the frontier:** (a) raising the clean-base rate (~1/7) so K is small — short-conv is a non-transfer,
  gate-init + EMA is only a modest ~2× help, so this is unsolved; (b) the threshold / capacity×length law (how
  much target-length coverage for horizon H); (c) transfer off the oracle instrument to natural workloads.
