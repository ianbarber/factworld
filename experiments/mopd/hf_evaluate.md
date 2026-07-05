# MOPD on FactWorld (Qwen3-1.7B) — headline results

`experiments/mopd/hf_evaluate.py`. Qwen/Qwen3-1.7B, greedy relaxed match, n=500. Normalised `s~ = (model - base)/(teacher - base)`: 0 at the base backbone, 1 at the per-domain teacher. Headline = uniform mean across domains; one student near 1 on BOTH domains = MOPD composing both abilities.


## Normalised score (per domain, mean over eval lengths)

| model | binding | recall | avg |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.000 |
| teacher_binding | 1.000 | 0.245 | 0.622 |
| teacher_recall | 0.030 | 1.000 | 0.515 |
| mopd_pg | 1.087 | 1.007 | 1.047 |
| mopd_kl | 1.050 | 1.007 | 1.029 |

## Raw accuracy (per domain x length)


**binding**

| model | L16 | L24 | L32 |
|---|---|---|---|
| base | 0.340 | 0.322 | 0.278 |
| teacher_binding | 0.986 | 0.932 | 0.888 |
| teacher_recall | 0.384 | 0.332 | 0.282 |
| mopd_pg | 0.990 | 0.990 | 0.986 |
| mopd_kl | 0.996 | 0.952 | 0.950 |

**recall**

| model | L16 | L24 |
|---|---|---|
| base | 0.336 | 0.278 |
| teacher_binding | 0.502 | 0.448 |
| teacher_recall | 0.992 | 0.998 |
| mopd_pg | 1.000 | 1.000 |
| mopd_kl | 1.000 | 1.000 |
