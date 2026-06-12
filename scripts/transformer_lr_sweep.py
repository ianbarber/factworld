"""Is the §5 transformer floor an LR-tuning artifact? Sweep the learning rate for the param-matched
45M transformer on the exact §5 composite (k=5 in-context-copy), 2 seeds per LR. If it floors
in-distribution (L16) across every LR cell, the floor is convergence-not-tuning (cf. the pure-recurrent
LR sweep of Okpekpe & Orvieto 2025). 1e-3 reproduces the §5 cell."""
import os, statistics, sys
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); sys.path.insert(0, REPO+"/scripts")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from recall_learn import _world, pure_copy, composite, composite_eval
from iso import strict_eval

STEPS, BATCH, EVAL_L = 25000, 32, [16, 64]
LRS = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
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
            run = T.run("transformer", tok, docs, [], steps=STEPS, batch=BATCH, d_model=512, n_layers=8,
                        d_ff=3072, lr=lr, seed=s, return_model=True)
            acc16 = strict_eval(run["model"], tok, w, ev[16])
            acc64 = strict_eval(run["model"], tok, w, ev[64])
            res[lr].append((acc16, acc64))
            print(f"  transf-45M lr={lr:.0e} s{s} :: L16={acc16:.3f}  L64={acc64:.3f}", flush=True)
            del run["model"]; torch.cuda.empty_cache()
    print("\n=== TRANSFORMER LR SWEEP (45M, k=5 composite, 25k steps; strict-acc, L16 in-distribution) ===", flush=True)
    for lr in LRS:
        xs16 = [a for a, _ in res[lr]]
        print(f"  lr={lr:.0e}  L16: {statistics.mean(xs16):.3f} (max {max(xs16):.3f}, {sum(x>0.5 for x in xs16)}/{len(xs16)} converge)", flush=True)
    print("transformer_lr_sweep done.", flush=True)

if __name__ == "__main__":
    main()
