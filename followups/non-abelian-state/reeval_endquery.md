# End-query re-eval — does the ladder recipe hold to high length? (`reeval_endquery.py`, d256x4, 3 seeds)

Exact ladder recipe (gdp_hybrid d256x4, 4000 steps, train {4,8,16}, parametric recall). Query stated at the END (read history, then ask) — the contrast to decay_curve's front-loaded query. R1 abelian no-CoT (value metric); R2 abelian CoT (strict holder+value); R3a non-abelian no-CoT (floors at L16, control). Floor = 0.20.

| rung | L16 | L24 | L32 | L48 | L64 | L96 | L128 | L192 | L256 |
|---|---|---|---|---|---|---|---|---|---|
| R1 (abelian, no CoT) | 0.83±0.14 | 0.59±0.18 | 0.58±0.18 | 0.54±0.19 | 0.57±0.17 | 0.55±0.16 | 0.52±0.13 | 0.50±0.17 | 0.52±0.12 |
| R2 (abelian, CoT) | 0.82±0.19 | 0.65±0.23 | 0.61±0.25 | 0.57±0.24 | 0.57±0.23 | 0.56±0.22 | 0.55±0.25 | 0.55±0.20 | 0.52±0.23 |
| R3a (non-abelian, no CoT) | 0.19±0.04 | 0.21±0.01 | 0.23±0.02 | 0.20±0.02 | 0.21±0.03 | 0.18±0.02 | 0.22±0.02 | 0.21±0.03 | 0.21±0.02 |
