# R3 decomposition — in-context vs parametric (which leg breaks?)

`followups/non-abelian-state/decompose_r3.py`. gdp_hybrid d256x4, 4000 steps, 5 seeds, CoT. Same fixed map + same non-abelian chains in both arms; only difference is whether the agent->a0 facts are rendered (`inctx`) or omitted (`param`). Free-running eval (model emits its own holder then value). Floor = 0.20.

| arm | L | holder | value | both | P(v\|h_ok) | route | other | none | conv |
|---|---|---|---|---|---|---|---|---|---|
| inctx | 16 | 0.206 | 0.206 | 0.206 | 1.000 | 1.000 | 0.000 | 0.000 | 0/5 |
| inctx | 64 | 0.197 | 0.197 | 0.197 | 1.000 | 1.000 | 0.000 | 0.000 | 0/5 |
| param | 16 | 0.212 | 0.212 | 0.212 | 1.000 | 1.000 | 0.000 | 0.000 | 0/5 |
| param | 64 | 0.203 | 0.203 | 0.203 | 1.000 | 1.000 | 0.000 | 0.000 | 0/5 |
