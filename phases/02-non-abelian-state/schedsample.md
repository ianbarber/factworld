# Scheduled sampling — free-running exposure in base training (`schedsample.py`, 18.5M)

At holder slots, with prob p(ramp 0->p_max) replace the gold holder in CONTEXT with the model's own argmax (targets stay gold). Metric: clean-base fraction (free-running L16 e2e >= 0.95); baseline ~1/8. EMA = weight-avg over the cosine tail. L128 flags native length-gen. Floor = 0.20.

| arm | clean L16 raw | clean L16 ema | max L128 (raw / ema) | L16 per seed (raw) |
|---|---|---|---|---|
| baseline | 1/6 | 1/6 | 0.26 / 0.24 | 0.97 0.86 0.77 0.62 0.37 0.30 |
| ss | 1/6 | 1/6 | 0.20 / 0.22 | 0.99 0.89 0.88 0.65 0.47 0.36 |
| ss+gate | 0/6 | 0/6 | 0.23 / 0.23 | 0.93 0.89 0.75 0.55 0.47 0.25 |
