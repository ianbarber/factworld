# fprm vs gdp_hybrid length extrapolation (K=1 dense)


`scripts/experiment_dense_supervision.py`. gdp_hybrid vs fprm, K=1 dense, ~10 seeds, trained at
lengths 4/8/16, evaluated at 16/32/64/128. value = free-run end-to-end accuracy.

| arch | L16 | L32 | L64 | L128 |
|---|---|---|---|---|
| gdp_hybrid | 1.00±0.00 | 0.90±0.12 | 0.74±0.24 | 0.64±0.30 |
| fprm | 1.00±0.00 | 0.58±0.30 | 0.20±0.05 | 0.23±0.05 |

**Finding:** both solve s5 in-distribution (L16=1.00), but only gdp_hybrid extrapolates the
learned circuit (L128=0.64 vs fprm 0.23). fprm's looped attention learns the task but doesn't
generalize in length; the recurrent hybrid's state structure does. Length extrapolation is the
architecture-dependent axis (circuit formation is not).
