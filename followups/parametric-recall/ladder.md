# Parametric-composite localization ladder (follow-on, segregated)

`followups/parametric-recall/ladder.py`. gdp_hybrid, d256x4 (~6M), 4000 steps, 3 seeds. Parametric recall = fixed agent->a0 map, facts NOT in prompt. Floor = 1/k = 0.20. CoT rungs (R2/R3b) use strict eval (holder AND value); no-CoT rungs (R0/R1/R3a) score the emitted value only.

| rung | what it adds | metric | L16 | L64 |
|---|---|---|---|---|
| R0 | literal key, no state | value | 1.00±0.00(3/3) | — |
| R1 | abelian binding, NO CoT (latent deref) | value | 0.83±0.14(3/3) | 0.57±0.17(2/3) |
| R2 | abelian binding, CoT (verbalized bridge) | holder+value | 0.82±0.19(3/3) | 0.57±0.23(1/3) |
| R3a | non-abelian state, NO CoT | value | 0.19±0.04(0/3) | 0.21±0.03(0/3) |
| R3b | non-abelian state, CoT | holder+value | 0.21±0.02(0/3) | 0.21±0.00(0/3) |
