# Length extrapolation of a learned s5 circuit — architecture comparison

`scripts/experiment_dense_supervision.py`, K=1 (dense) supervision, ~10 seeds each, trained at
lengths 4/8/16, evaluated free-run at 16/32/64/128. The question: once the s5 circuit is learned
in-distribution (it is, for every arch), which architectures **extrapolate** it in length?

| arch | L16 | L32 | L64 | L128 |
|---|---|---|---|---|
| **gdp_hybrid** | 1.00±0.00 | 0.90±0.12 | 0.74±0.24 | **0.64±0.30** |
| fprm | 1.00±0.00 | 0.58±0.30 | 0.20±0.05 | 0.23±0.05 |
| transformer | 0.83±0.25 | — | 0.19±0.04 | — |

(transformer: K=1 from the 10-seed full sweep, L16/L64 only; floors by L64. gdp_hybrid/fprm: 10-seed
extrapolation runs.)

## Finding

**All three architectures form the s5 circuit in-distribution under dense supervision** (L16:
gdp_hybrid/fprm 1.00, transformer 0.83). **Length extrapolation is the architecture-dependent axis,
and it is not uniform:**

- **gdp_hybrid extrapolates smoothly** to ~8× the trained horizon (L128 = 0.64). Its recurrent state
  structure generalizes the learned permutation computation.
- **fprm collapses past L32** (L64 = 0.20). The weight-tied looped conv+attention learns the task
  but does not generalize the circuit in length.
- **transformer floors by L64** even in-distribution-weak (0.83 @L16).

So: **circuit formation is architecture-independent (a supervision-density phenomenon); circuit
*extrapolation* is architecture-dependent (gdp_hybrid's recurrence is what carries it).** This is
the one place architecture materially matters in the whole program.

## Files

- `results/dense_extrap_gdp_*.jsonl`, `results/dense_extrap_fprm_*.jsonl` (10-seed K=1 curves).
- `results/dense_power_transformer_*.jsonl` (full K-sweep, 10 seeds; K=1 cell cited above).
