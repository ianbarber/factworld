"""Matched gdp LR sweep — the missing symmetry for the §5 recurrence-vs-attention claim.

§5 now reads: both gated-delta recurrent hybrids learn the composite (gdn 1/10 over its LR sweep,
transformer 0/10 over its LR sweep), so the lever is gated-delta recurrence vs. attention. But gdp was only
run at the default lr 1e-3 (1/5 seeds), never LR-swept. This sweeps the gdp 45M arm on the same composite
with the same grid as gdn_lr_sweep.py / transformer_lr_sweep.py (25k steps, 2 seeds per LR), so all three
arms are evaluated under matched protocols. gdp 45M config = scale_confirm.py (d512x8, d_ff=2048).

Reading: (a) if gdp converges at more LR cells than gdn (1/10), it has a robustness edge — supports carrying
it forward beyond "it also does the state leg." (b) if gdp is also ~1/10, the two recurrent arms are equally
fragile and the recurrence-vs-attention dissociation is fully symmetric.

  .venv/bin/python scripts/gdp_lr_sweep.py
"""
import os, statistics, sys
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); sys.path.insert(0, REPO+"/scripts")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from recall_learn import _world, pure_copy, composite, composite_eval
from iso import strict_eval

STEPS, BATCH, EVAL_L = 25000, 32, [16, 64]
LRS = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]   # same grid as the gdn / transformer sweeps
SEEDS = 2

def main():
    import torch
    from factworld import train as T
    w, r = _world()
    tr = composite(w, r, 8000, 2, True) + pure_copy(w, r, 4000, 3, True)
    ev = {L: composite_eval(w, r, L, 200, 300 + L) for L in EVAL_L}
    tok, docs, _ = T.prepare(tr, [], [w])
    res = defaultdict(list)
    for lr in LRS:
        for s in range(SEEDS):
            run = T.run("gdp_hybrid", tok, docs, [], steps=STEPS, batch=BATCH, d_model=512, n_layers=8,
                        d_ff=2048, lr=lr, seed=s, return_model=True)   # 2048 = scale_confirm.py gdp 45M
            acc16 = strict_eval(run["model"], tok, w, ev[16])
            acc64 = strict_eval(run["model"], tok, w, ev[64])
            res[lr].append((acc16, acc64))
            print(f"  gdp-45M lr={lr:.0e} s{s} :: L16={acc16:.3f}  L64={acc64:.3f}", flush=True)
            del run["model"]; torch.cuda.empty_cache()
    print("\n=== GDP LR SWEEP (45M, k=5 composite, 25k steps; strict-acc, L16 in-distribution) ===", flush=True)
    for lr in LRS:
        xs16 = [a for a, _ in res[lr]]
        print(f"  lr={lr:.0e}  L16: {statistics.mean(xs16):.3f} (max {max(xs16):.3f}, {sum(x>0.5 for x in xs16)}/{len(xs16)} converge)", flush=True)
    print("gdp_lr_sweep done.", flush=True)

if __name__ == "__main__":
    main()
