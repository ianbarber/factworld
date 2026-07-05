# Stage 1 — shared mixed base (per-domain accuracy)

`experiments/mopd/stage1_base.py`. transformer d256x4, 15000 steps, mix={'binding': 0.5, 'recall': 0.5}, seed 0. Greedy relaxed match, n=300. Bold = training lengths (must be > 0 for RL to have pass@k signal); the rest are held-out OOD lengths (the headroom RL/MOPD can chase).


**binding** (`*` = train length)

| L4* | L8* | L16* | L32 | L64 |
|---|---|---|---|---|
| 0.403 | 0.410 | 0.370 | 0.317 | 0.263 |

**recall** (`*` = train length)

| L2* | L3* | L4* | L5 |
|---|---|---|---|
| 0.490 | 0.337 | 0.247 | 0.160 |
