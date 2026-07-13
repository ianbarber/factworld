# MOPD on FactWorld (Qwen3-1.7B) — multi-seed robustness

`experiments/mopd/hf_seeds.py`. Qwen/Qwen3-1.7B, seeds [0, 1, 2]. Per seed: fresh GRPO teachers (150 steps) + MOPD students (200 steps, both loss forms), greedy eval n=300. Normalised score `(model-base)/(teacher-base)` (0=base, 1=domain teacher), mean over eval lengths; cells are mean±std across seeds.

| model | binding | recall | avg |
|---|---|---|---|
| teacher_binding | 1.000±0.000 | 0.889±0.086 | 0.944±0.083 |
| teacher_recall | 0.094±0.021 | 1.000±0.000 | 0.547±0.453 |
| mopd_pg | 1.051±0.031 | 1.002±0.003 | 1.027±0.033 |
| mopd_kl | 1.083±0.043 | 1.000±0.004 | 1.042±0.052 |
