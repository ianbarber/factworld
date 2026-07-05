# Stage 2 (Qwen3-1.7B) — per-domain RL teachers (GRPO LoRA)

`experiments/mopd/hf_teachers.py`. Qwen/Qwen3-1.7B, LoRA GRPO from the frozen backbone: 150 steps, 8 prompts/step x group 8, lr 0.0001, temp 1.0, verifiable relaxed-match reward, thinking off. Greedy eval, n=120. `delta` = teacher - base.


**binding**

| L | base | teacher | delta |
|---|---|---|---|
| 16 | 0.392 | 0.983 | +0.592 |
| 24 | 0.292 | 0.933 | +0.642 |
| 32 | 0.317 | 0.917 | +0.600 |

**recall**

| L | base | teacher | delta |
|---|---|---|---|
| 16 | 0.425 | 1.000 | +0.575 |
| 24 | 0.283 | 1.000 | +0.717 |
