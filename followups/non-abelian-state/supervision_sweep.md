# Supervision-sparsity sweep — how sparse can the state supervision be?

`followups/non-abelian-state/supervision_sweep.py`. gdp_hybrid d256x4, 4000 steps, 5 seeds, parametric recall. Holder supervised every K events (+ always the final). K=1 == dense capstone; K=inf == answer-only (only the final holder, the R3b floor). Train lengths (4,8,16) so a 16-event episode gets 16/8/4/2/1 checkpoints at K=1/2/4/8/inf. Metric: end-to-end free-running composite (`e2e_v`); `dense_h` = teacher-forced acc at labelled slots. Floor = 0.20.

| K (stride) | labels / 16-ep | L | dense_h | e2e_holder | e2e_value | conv(e2e_v>0.5) |
|---|---|---|---|---|---|---|
| 1 | 16 | 16 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 5/5 |
| 1 | 16 | 64 | 0.99±0.01 | 0.78±0.24 | 0.78±0.24 | 4/5 |
| 2 | 8 | 16 | 1.00±0.00 | 0.98±0.01 | 0.98±0.01 | 5/5 |
| 2 | 8 | 64 | 0.83±0.12 | 0.29±0.20 | 0.29±0.20 | 1/5 |
| 4 | 4 | 16 | 0.20±0.01 | 0.19±0.03 | 0.19±0.03 | 0/5 |
| 4 | 4 | 64 | 0.20±0.01 | 0.19±0.02 | 0.19±0.02 | 0/5 |
| 8 | 2 | 16 | 0.21±0.03 | 0.22±0.04 | 0.22±0.04 | 0/5 |
| 8 | 2 | 64 | 0.20±0.01 | 0.21±0.01 | 0.21±0.01 | 0/5 |
| inf | 1 | 16 | 0.19±0.02 | 0.19±0.02 | 0.19±0.02 | 0/5 |
| inf | 1 | 64 | 0.20±0.04 | 0.20±0.04 | 0.20±0.04 | 0/5 |
