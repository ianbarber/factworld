# Stage 2 — per-domain RL teachers (GRPO)

> v1-sampler-era table (binding rows used the retired `binding_v1`; `mopd.py` now pins
> `binding_v2`). Kept as the from-scratch control's provenance — its conclusion is qualitative.

`experiments/mopd/stage2_teachers.py`. GRPO from `base.pt`: 1500 steps, 16 prompts/step x group 8, lr 0.0003, outcome 0/1 reward. Greedy relaxed match, n=500. `delta` = teacher - base (the norm-score headroom).


**binding**

| L | base | teacher | delta |
|---|---|---|---|
| 16 | 0.358 | 0.364 | +0.006 |
| 32 | 0.312 | 0.322 | +0.010 |
| 64 | 0.276 | 0.374 | +0.098 |

**recall**

| L | base | teacher | delta |
|---|---|---|---|
| 3 | 0.330 | 0.312 | -0.018 |
| 4 | 0.264 | 0.270 | +0.006 |
| 5 | 0.164 | 0.186 | +0.022 |
