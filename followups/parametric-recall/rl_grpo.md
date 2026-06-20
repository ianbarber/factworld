# RL (GRPO, outcome-only reward) vs static SFT — does reward climb the supervision cliff?

`followups/parametric-recall/rl_grpo.py`. gdp_hybrid d256x4, 3 seeds. Within-seed contrast: answer-only SFT (2500 steps) — the static regime that floors — then GRPO (group 8, outcome 0/1 reward on the composite answer, free scratchpad, 1500 steps) from that same model. `value` = composite accuracy (greedy); `holder` = state-leg resolution rate (never rewarded directly). Floor = 0.20. RL value >> SFT value would mean outcome reward climbs the cliff static SFT cannot.

| eval | SFT value | RL value | RL holder |
|---|---|---|---|
| L16 | 0.19±0.04 | 0.19±0.04 | 0.05±0.07 |
| L64 | 0.18±0.01 | 0.21±0.01 | 0.08±0.11 |
