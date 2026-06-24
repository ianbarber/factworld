# Code-trace non-abelian cliff — does the wall appear in execution-trace clothing?

`followups/non-abelian-state/code_trace.py`. gdp_hybrid d384 (18.5M), 3 seeds. Our non-abelian dynamics rendered as a variable-swap execution trace (CWM surface grammar): vars hold values, `swap`/`cycle` ops, full-state snapshot every K ops, query a variable's final value (free-running). Same density sweep as `supervision_sweep.py` (the role-rendered cliff). Floor = 1/k = 0.20. A matching cliff = the wall is non-abelian composition, not the rendering.

| K (snapshot / op) | L16 | L64 |
|---|---|---|
| 1 (every op) | 1.00±0.00(3/3) | 0.97±0.01(3/3) |
| 2 (every 2nd) | 0.99±0.00(3/3) | 0.89±0.02(3/3) |
| 4 (every 4th) | 0.87±0.09(3/3) | 0.50±0.22(1/3) |
| 8 (every 8th) | 0.72±0.38(2/3) | 0.29±0.14(0/3) |
| inf (answer-only) | 0.46±0.37(1/3) | 0.25±0.05(0/3) |
