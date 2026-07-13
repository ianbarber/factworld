# Stage 2 (Qwen3-1.7B) — per-domain RL teachers (GRPO LoRA)

`experiments/mopd/hf_teachers.py`. Qwen/Qwen3-1.7B, LoRA GRPO from the frozen backbone: 300 steps, 8 prompts/step x group 8, lr 0.0001, temp 1.0, verifiable relaxed-match reward, thinking off. Greedy eval, n=200. `delta` = teacher - base.


**binding**

| L | base | teacher | delta |
|---|---|---|---|
| 16 | 0.420 | 0.975 | +0.555 |
| 24 | 0.325 | 0.980 | +0.655 |
| 32 | 0.290 | 0.910 | +0.620 |

**recall**

| L | base | teacher | delta |
|---|---|---|---|
| 16 | 0.390 | 0.990 | +0.600 |
| 24 | 0.245 | 0.995 | +0.750 |
