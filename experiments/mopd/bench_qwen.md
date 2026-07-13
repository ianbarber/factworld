# Bench — Qwen3-1.7B on FactWorld (base, before RL)

`experiments/mopd/bench_qwen.py`. Qwen/Qwen3-1.7B, chat template, think=0, max_new=24, greedy, n=40. `relaxed` = strict answer span; `contains` = gold token appears; `last_n` = gold is the tail. Pick RL-teacher domains where the base is PARTIAL (headroom), not at ceiling or floor.

| task | L | relaxed | contains | last_n |
|---|---|---|---|---|
| recall_copy_v1 | 6 | 0.675 | 0.925 | 0.925 |
| conflict_v1 | 4 | 0.475 | 0.900 | 0.900 |
| binding_v2 | 16 | 0.375 | 0.375 | 0.375 |
| composite_copy_v2 | 16 | 0.000 | 0.000 | 0.000 |
| chain_v1 | 4 | 0.075 | 0.075 | 0.075 |
| s5_v1 | 32 | 0.000 | 0.125 | 0.125 |
