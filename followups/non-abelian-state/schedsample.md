# Scheduled sampling — free-running exposure in base training (`schedsample.py`, 18.5M)

At holder slots, with prob p(ramp 0->p_max) replace the gold holder in CONTEXT with the model's own argmax (targets stay gold). Metric: clean-base fraction (free-running L16 e2e >= 0.95); baseline ~1/8. EMA = weight-avg over the cosine tail. L128 flags native length-gen. Floor = 0.20.

| arm | clean L16 raw | clean L16 ema | max L128 (raw / ema) | L16 per seed (raw) |
|---|---|---|---|---|
| baseline | 0/1 | 0/1 | 0.00 / 0.00 | 0.20 |
| ss | 0/1 | 0/1 | 0.00 / 0.00 | 0.20 |
