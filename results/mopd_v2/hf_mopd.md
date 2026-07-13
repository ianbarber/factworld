# Stage 3 (Qwen3-1.7B) — MOPD distillation dynamics

`experiments/mopd/hf_mopd.py`. Student LoRA adapter on the shared backbone; both frozen teacher adapters distilled on the student's own rollouts. Per-token reverse KL should start LOW (same-origin) and stay stable; entropy should not collapse. Per domain, initial -> final.

## loss = `pg`  (200 steps, lr 0.0001)

| domain | KL init | KL final | H init | H final |
|---|---|---|---|---|
| binding | 6.3654 | 0.0196 | 0.171 | 0.010 |
| recall | 0.0001 | 0.0000 | 0.000 | 0.000 |

## loss = `kl`  (200 steps, lr 0.0001)

| domain | KL init | KL final | H init | H final |
|---|---|---|---|---|
| binding | 0.0000 | 0.0000 | 0.000 | 0.000 |
| recall | 0.0000 | 0.0000 | 0.000 | 0.000 |

## loss = `pg`  (300 steps, lr 0.0001)

| domain | KL init | KL final | H init | H final |
|---|---|---|---|---|
| binding | 6.3756 | 0.0000 | 0.264 | 0.000 |
| recall | 0.0002 | 0.0006 | 0.000 | 0.000 |

## loss = `kl`  (300 steps, lr 0.0001)

| domain | KL init | KL final | H init | H final |
|---|---|---|---|---|
| binding | 1.5978 | 0.0000 | 0.029 | 0.000 |
| recall | 0.0002 | 0.0006 | 0.000 | 0.000 |
