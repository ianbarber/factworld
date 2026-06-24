# Curriculum — can dense->sparse weaning internalize the state circuit?

`followups/non-abelian-state/curriculum.py`. gdp_hybrid d256x4, 6000 steps, 5 seeds, parametric. `anneal` = K schedule 1->2->4->8->inf over training; `mixed` = random K per example. Eval is free-running e2e composite. **Answer-only** (K=inf, no scratchpad) is the agentic target; baselines from the sweep: from-scratch K=1 = 1.00/0.78, from-scratch K=inf = **0.20 floor**. Floor = 0.20.

| arm | answer-only L16 | answer-only L64 | dense L16 | dense L64 |
|---|---|---|---|---|
| anneal | 0.43±0.26(1/5) | 0.21±0.01(0/5) | 0.85±0.21(4/5) | 0.43±0.22(2/5) |
| mixed | 0.66±0.28(3/5) | 0.22±0.02(0/5) | 1.00±0.00(5/5) | 0.50±0.38(2/5) |
