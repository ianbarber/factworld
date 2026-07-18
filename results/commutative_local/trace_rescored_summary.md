# Commutative rung — local calibration (commutative_v1, arch x seed, d256x4)

d_model=256 n_layers=4 steps=8000 batch=32 train_n=8000 eval_n=200 seeds=[0, 1, 2] train_lengths=[4, 8, 16] use_trace=True

| arch | L16 mean±std (pconv) | L32 mean±std (pconv) | L64 mean±std (pconv) |
|---|---|---|---|
| fprm | 0.65±0.26 (33%) | 0.21±0.03 (0%) | 0.21±0.02 (0%) |
| gdp_hybrid | 0.82±0.15 (33%) | 0.17±0.01 (0%) | 0.20±0.01 (0%) |
| transformer | 0.20±0.02 (0%) | 0.21±0.01 (0%) | 0.16±0.07 (0%) |

_Relaxed match (canonical). pconv = fraction of seeds >=0.9. Floors as rows: chance 1/k_positions = 0.200; the strongest of the four shallow adversaries sits ~0.20-0.22 (all four gated <= 0.4 by scripts/validate_suite.py) — a run only 'solves' if it clears that band. L16 is in-distribution, L32/L64 are OOD aggregation depth. fprm is weight-tied: compare on FLOPs, not params._