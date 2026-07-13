# MOPD on FactWorld (Qwen3-1.7B) — headline results

`experiments/mopd/hf_evaluate.py`. Qwen/Qwen3-1.7B, greedy relaxed match, n=300. Normalised `s~ = (model - base)/(teacher - base)`: 0 at the base backbone, 1 at the per-domain teacher. Headline = uniform mean across domains; one student near 1 on BOTH domains = MOPD composing both abilities.


## Normalised score (per domain, mean over eval lengths)

| model | binding | recall | avg |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.000 |
| teacher_binding | 1.000 | 0.734 | 0.867 |
| teacher_recall | 0.103 | 1.000 | 0.551 |
| mopd_pg | 0.967 | 1.007 | 0.987 |
| mopd_kl | 0.995 | 0.998 | 0.997 |

## Raw accuracy (per domain x length)


**binding**

| model | L16 | L24 | L32 |
|---|---|---|---|
| base | 0.407 | 0.320 | 0.300 |
| teacher_binding | 0.980 | 0.977 | 0.917 |
| teacher_recall | 0.453 | 0.383 | 0.380 |
| mopd_pg | 0.977 | 0.947 | 0.887 |
| mopd_kl | 0.987 | 0.960 | 0.917 |

**recall**

| model | L16 | L24 |
|---|---|---|
| base | 0.327 | 0.257 |
| teacher_binding | 0.863 | 0.747 |
| teacher_recall | 0.993 | 0.997 |
| mopd_pg | 1.000 | 1.000 |
| mopd_kl | 1.000 | 0.987 |
