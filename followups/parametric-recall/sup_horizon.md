# Supervision × horizon — does training at length lower the density needed?

`followups/parametric-recall/sup_horizon.py`. gdp_hybrid d384 (18.5M), 3 seeds. Density sweep K with LONG-horizon training (lengths ≤64), parametric composite, single-role checkpoint, free-running e2e. Compare the cliff threshold to `supervision_sweep.md` (same build, train ≤16): if the circuit now forms at sparser K, length and density trade off. Floor = 0.20.

| K | L64 (in extended range) | L128 (2× OOD) |
|---|---|---|
| 1 | 0.98±0.01(3/3) | 0.97±0.01(3/3) |
| 2 | 0.99±0.01(3/3) | 0.98±0.00(3/3) |
| 4 | 0.19±0.00(0/3) | 0.22±0.03(0/3) |
| 8 | 0.18±0.03(0/3) | 0.18±0.03(0/3) |
| inf | 0.19±0.04(0/3) | 0.18±0.01(0/3) |
