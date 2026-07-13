# De-risk probe — RL headroom on the base (pass@k)

> v1-sampler-era table (binding rows used the retired `binding_v1`; `mopd.py` now pins
> `binding_v2`). Kept as the from-scratch control's provenance — its conclusion is qualitative.

`experiments/mopd/probe_headroom.py`. Base `base.pt`, n=150, k=8, temp=1.0. `greedy` = the norm-score 0-anchor; `pass@k` = P(>=1 of k samples correct) = exploration headroom; `var-frac` = fraction of prompts with a non-degenerate GRPO group (0<succ<k). RL-improvable band = greedy in [0.1, 0.7] and var-frac >= 0.2.

| domain | config | L | greedy | pass@k | var-frac | |
|---|---|---|---|---|---|---|
| binding | m4 (default) | 4 | 0.387 | 0.853 | 0.840 | <-- band |
| binding | m4 (default) | 8 | 0.413 | 0.840 | 0.840 | <-- band |
| binding | m4 (default) | 16 | 0.393 | 0.800 | 0.787 | <-- band |
| recall | default | 2 | 0.493 | 0.660 | 0.427 | <-- band |
| recall | default | 3 | 0.360 | 0.533 | 0.387 | <-- band |
| recall | default | 4 | 0.240 | 0.407 | 0.313 | <-- band |
