# Follow-on: the composition gap is a state-supervision problem

A post-publication, **segregated** follow-on to the FactWorld paper (nothing here changes the shipped paper or
`docs/`). It started from "where does the parametric composite actually break?" and grew into a learnability
study of non-abelian state-tracking, with a 4-thread rigor program.

- **`non-abelian-state.md`** — the draft write-up (twice advisor-reviewed, writing-guide-edited). Start here.
- **`FINDINGS.md`** — the full evidence trail; every claim maps to a script + result table below.

Each script writes a `<name>.md` table next to it. All reuse the repo's `factworld/` package (and a couple of
`scripts/` helpers); the data/oracle layer is pure-stdlib, training needs a CUDA GPU. Run e.g.
`.venv/bin/python followups/parametric-recall/ladder.py`.

## Map of claims → scripts

| Claim | Script(s) |
|---|---|
| The bottleneck is the non-abelian **state** leg, not recall (parametric ≡ in-context given a resolved pointer) | `ladder.py`, `decompose_r3.py` |
| **Dense process supervision solves it**; parametric rides along | `dense_capstone.py` |
| Supervision density is a **cliff** (forms only ~every other step) | `supervision_sweep.py` |
| Reconcile R1 vs the companion paper (scoring axis) | `recon_b1.py` |
| **Internalization / horizon dissociation** (weaning; order isn't the lever) | `curriculum.py` |
| **Scale** largely doesn't relieve the horizon wall (flat to 44.8M even LR-tuned; weak lift at a 70M control point) | `scale.py`, `scale_tuned.py` |
| **Horizon-extension curriculum** moves the wall; token re-anchoring doesn't | `horizon.py`, `coarse.py` |
| The internalized cap **tracks max training length** | `horizon_mech.py` |
| Supervision density × horizon are **orthogonal** levers | `sup_horizon.py` |
| **RL vs static**: outcome-reward GRPO doesn't climb the cliff either | `rl_grpo.py`, `rl_flicker.py` |
| **Construct-validity bridge**: the cliff reproduces in code-execution-trace (CWM) clothing | `code_trace.py` |

All numbers are small-scale (≤70M, single RTX 3090) and scoped accordingly in the write-up.
