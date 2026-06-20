# Dense-supervision capstone — does lifting the state leg rescue the PARAMETRIC composite?

`followups/parametric-recall/dense_capstone.py`. gdp_hybrid d256x4, 4000 steps, 5 seeds. Interleaved dense per-step holder supervision (oracle `hard_trace`) on the non-abelian composite. Baseline for contrast: R3b (final-only CoT) floored at 0.20 for both arms. `dense_h` per-step holder acc; `val|trace` value acc given true trace; `e2e_*` guided free-run.

| arm | L | dense_h | val\|trace | e2e_holder | e2e_value |
|---|---|---|---|---|---|
| inctx | 16 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| inctx | 64 | 0.96±0.04 | 1.00±0.00 | 0.55±0.37 | 0.55±0.37 |
| param | 16 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| param | 64 | 1.00±0.00 | 1.00±0.00 | 0.95±0.06 | 0.95±0.06 |
