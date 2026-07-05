# MOPD on FactWorld (Qwen3-1.7B) — multi-seed robustness

`experiments/mopd/hf_seeds.py`. Qwen/Qwen3-1.7B, seeds [0, 1, 2]. Per seed: fresh GRPO teachers (150 steps) + MOPD students (200 steps, both loss forms), greedy eval n=300. Normalised score `(model-base)/(teacher-base)` (0=base, 1=domain teacher), mean over eval lengths; cells are mean±std across seeds.

| model | binding | recall | avg |
|---|---|---|---|
| teacher_binding | 1.000±0.000 | 0.846±0.154 | 0.923±0.133 |
| teacher_recall | 0.026±0.026 | 1.000±0.000 | 0.513±0.487 |
| mopd_pg | 1.135±0.140 | 1.002±0.015 | 1.069±0.120 |
| mopd_kl | 1.106±0.108 | 1.007±0.011 | 1.057±0.091 |
