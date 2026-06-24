# Reproducing the non-abelian state-tracking results

Reproduction guide for **[`non-abelian-state.md`](non-abelian-state.md)** (the report). This is a
post-publication, **segregated** follow-up — nothing here changes the shipped `paper.md` / `docs/`. Every claim
maps to one script below; each writes a `<name>.md` table next to it on completion (crash-safe). The
data/oracle layer is pure-stdlib; training needs a CUDA GPU (validated on a single RTX 3090, ≤ 357M params).

```bash
.venv/bin/python followups/non-abelian-state/<script>.py
```

Scripts reuse the repo's `factworld/` package (and `scripts/iso.py`). Infrastructure modules
(`dense_capstone.py`, `supervision_sweep.py`, `length_mix.py`, `curriculum.py`, `post_state.py`) are imported by
the experiment scripts; running them directly also works where they have a `main`.

## Claim → script → expected headline

| Claim (report §) | Script | Expected headline result |
|---|---|---|
| Bottleneck is the non-abelian **state** leg, not recall — parametric ≡ in-context given a resolved pointer (§3) | `ladder.py`, `decompose_r3.py` | abelian deref R1 = 0.83/0.57 (L16/L64); non-abelian R3a floors; `P(value\|holder✓)=1.0`, route=1.0 both arms |
| **Dense process supervision solves it**; parametric rides along (§4) | `dense_capstone.py` | e2e L16=1.00 both arms; L64 parametric **0.95** vs in-context 0.55 |
| Supervision density is a **cliff** (§4) | `supervision_sweep.py` | K=1 L64 0.78; K=2 forms L16 0.98 / L64 0.29; **K≥4 floors** (all seeds) |
| Reconcile R1 vs the companion paper (scoring) (§3) | `recon_b1.py` | position-strict == value-scan to the digit (R1 L16 0.83/0.83) |
| **Internalization / horizon dissociation** (order isn't the lever) (§5) | `curriculum.py` | mixed internalizes L16 (3/5) ≥ anneal (1/5); answer-only **L64 floor both** |
| **Scale is not the lever** — flat to 357M; the largest fully-converged model floors (§5) | `scale.py` | answer-only L64 ladder 0.23→0.20→0.22→0.41→0.42→0.21→**0.20** (5.7M→357M); + LR-control 44.8M/70M |
| **Training-length distribution is the lever** — ~20% target-length unlocks; soft cap; concentration > coverage (§6) | `length_mix.py` | L64: 5% floor, **20% = 0.66**, 50% = 0.67; uniform {4..64} = 0.49; `{16,32}` = 0.37 |
| **Real circuit vs shortcut** — abelian length-general via read-then-lookup; non-abelian only online-carry (§3.1) | `decay_curve.py`, `reeval_endquery.py` | front-query: all arms cliff ~1.5×; end-query: abelian R1/R2 hold ≈0.55 to **L256 (16×)**, non-abelian floors |
| **From-scratch deep-state coverage doesn't build the circuit** (null; hurts in-dist) (§6.1) | `carried_state.py` | burn-in from scratch floors at every length incl. L16 (0.38) |
| **Post-training deep-state coverage DOES extend it** — floor→0.99 at 4×, **0.86 at 8×**, label-free (§6.1) | `post_state.py` | base floors past L32; post (best seed) L64 0.99 / L128 0.86; seed-fragile |
| **Lever is reliable; fragility is base-side** — between/within variance ≈300×, predictor = free-running L16 (§6.1) | `post_reliability.py`, `base_select.py` | clean base posts L128 **0.93/0.89**; degraded base floors; clean bases ≈1/7 seeds |
| **Base-training reliability** (open) (§6.1) | `base_reliability.py`, `schedsample.py` | gate-init+EMA ~2× clean rate (1/8→2/8); scheduled sampling lifts distribution not clean rate; short-conv floors |
| **Horizon-extension curriculum** moves the wall; token re-anchoring doesn't (§6) | `horizon.py`, `coarse.py` | curriculum answer-only L64 **0.94** (3/3); digest re-anchoring floors all cadences |
| Internalized cap **tracks max training length** (§6) | `horizon_mech.py` | solvable within ~[0, Lmax], floors at ~2×Lmax |
| Supervision density × horizon are **orthogonal** (§6) | `sup_horizon.py` | K≥4 floors regardless of train length; K=2 extrapolation 0.29→0.98 at long-horizon |
| **RL vs static** — outcome-reward GRPO doesn't climb the cliff (§4) | `rl_grpo.py` (`--flicker` for the powered run) | GRPO reward stays at chance (~0.20) across 5–7.5k steps; holder-resolution 0.00 |
| **Construct-validity bridge** — the cliff reproduces in code-execution (CWM) clothing (§4.1) | `code_trace.py` | L64 0.97→0.89→0.50→0.29→floor as snapshots thin |

All numbers are small-scale (≤ 357M, single RTX 3090) and scoped accordingly in the report. The two consolidated
scripts run the full ladder in one file: `scale.py` covers 5.7M→357M (base ladder + LR control + push), and
`rl_grpo.py` runs the GRPO null (add `--flicker` for the powered 7.5k-step trajectory variant).
