# Carried-state — does deep-state coverage form a length-general non-abelian circuit? (`carried_state.py`, 18.5M, 3 seeds)

gdp_hybrid d384x6, front-query online-carry non-abelian task, mixed-density labels. `short_only` = lengths {4,8,16}, no burn-in (the decay_curve baseline). `burnin` = a 16-event labelled window plus an UNLABELED random burn-in B in {0,16,32,64,96,128,192} (state coverage to depth ~208, no labels at length). Eval internalized (answer-only, no burn-in) to L256. Floor = 0.20.

| arm | L16 | L32 | L64 | L128 | L256 |
|---|---|---|---|---|---|
| short_only | 0.82±0.14 | 0.31±0.13 | 0.20±0.01 | 0.20±0.02 | 0.23±0.03 |
| burnin | 0.38±0.08 | 0.22±0.01 | 0.20±0.05 | 0.23±0.01 | 0.20±0.02 |
