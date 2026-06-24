# Horizon mechanism — does the internalized length cap track the max training length?

`followups/non-abelian-state/horizon_mech.py`. gdp_hybrid d384 (18.5M), mixed density, 3 seeds. Train internalized to max length Lmax; answer-only e2e at Lmax (in-range) and 2×Lmax (OOD). If in-range stays high and the cliff falls near ~2×Lmax for each Lmax, the cap = trained step-count. Floor = 0.20.

| Lmax (max train len) | in-range (L=Lmax) | OOD (L=2×Lmax) |
|---|---|---|
| 16 | 0.82±0.07 | 0.27±0.07 (L32) |
| 32 | 0.62±0.11 | 0.23±0.01 (L64) |
| 48 | 0.37±0.29 | 0.23±0.06 (L96) |
| 64 | 0.64±0.14 | 0.30±0.07 (L128) |
