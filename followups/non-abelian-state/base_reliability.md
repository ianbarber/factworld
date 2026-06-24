# Base-reliability — short-conv x gate-init, clean-base fraction (`base_reliability.py`, 18.5M)

Free-running answer-only L16 e2e (clean = >= 0.95); baseline ~1/8. `shortconv` = use_short_conv (fla 'crucial'; FPRM's state-tracking lever); `gate_slow` = GDP forget-gate retention-init. EMA = weight-avg over the cosine tail. L128 raw/ema flags any NATIVE length-generalization (no post-training). Floor = 0.20.

| arm | clean L16 raw | clean L16 ema | max L128 (raw / ema) | L16 per seed (raw) |
|---|---|---|---|---|
| default | 1/8 | 1/8 | 0.26 / 0.24 | 0.97 0.86 0.77 0.62 0.37 0.30 0.27 0.26 |
| shortconv | 0/8 | 0/8 | 0.21 / 0.21 | 0.33 0.24 0.23 0.23 0.22 0.21 0.18 0.16 |
| gate_slow | 0/8 | 2/8 | 0.23 / 0.23 | 0.92 0.89 0.87 0.82 0.62 0.55 0.29 0.20 |
| conv+gate | 0/8 | 0/8 | 0.27 / 0.27 | 0.31 0.25 0.24 0.24 0.23 0.21 0.20 0.17 |
