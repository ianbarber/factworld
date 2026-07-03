# Coarse re-anchoring — does a periodic compressed full-state summary unlock length-robust tracking?

`followups/non-abelian-state/coarse.py`. gdp_hybrid d384x6 (18.5M), 3 seeds, parametric. Model emits a bounded k-token full-state digest every C events and re-anchors. Train lengths {16,24,32}; free-running composite eval (model emits its own summaries) at L64/L128. `single_C8` control = only the queried role's holder every 8 (~= sweep K=8, floored). Beating floor (0.20) at L128 = re-anchoring works.

| condition | L64 | L128 |
|---|---|---|
| full_C8 | 0.15±0.01(0/3) | 0.22±0.03(0/3) |
| full_C16 | 0.20±0.02(0/3) | 0.19±0.04(0/3) |
| single_C8 | 0.23±0.03(0/3) | 0.21±0.03(0/3) |
