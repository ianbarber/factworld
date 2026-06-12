"""Is the §5 gdn floor an LR-tuning artifact? (RF-3 / the §5 negative-arm control.)

The §5 claim "product structure is THE lever" is a difference-of-floors argument: gdp lifts, gdn floors.
But the transformer arm was defended by a 5-LR sweep (transformer_lr_sweep.py: 0/10) while the gdn 0/3 arm
ran a single LR. Okpekpe & Orvieto (2025) report recurrent models are the LR-sensitive ones, so the gdn
floor is exactly the case that warrants a sweep before being read as incapacity. This mirrors the
transformer sweep cell-for-cell on the gdn 45M arm (d512x8, d_ff=2816 = the scale_confirm.py config).

If gdn floors in-distribution (L16) across every LR, the negative arm is as well-tuned as the positive
control and "product is the lever" stands. If any LR lifts it off 0.01, N2 weakens to "most sample-efficient
lever at this recipe." (n_heads is largely exonerated by recall_fair: 4 vs 8 was identical for the
transformer; the composite floor is composition, not recall.)

  .venv/bin/python scripts/gdn_lr_sweep.py
"""
import os, statistics, sys
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); sys.path.insert(0, REPO+"/scripts")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from recall_learn import _world, pure_copy, composite, composite_eval
from iso import strict_eval

STEPS, BATCH, EVAL_L = 25000, 32, [16, 64]
LRS = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]   # same grid as the transformer sweep
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
            run = T.run("gdn_hybrid", tok, docs, [], steps=STEPS, batch=BATCH, d_model=512, n_layers=8,
                        d_ff=2816, lr=lr, seed=s, return_model=True)   # 2816 = scale_confirm.py gdn 45M
            acc16 = strict_eval(run["model"], tok, w, ev[16])
            acc64 = strict_eval(run["model"], tok, w, ev[64])
            res[lr].append((acc16, acc64))
            print(f"  gdn-45M lr={lr:.0e} s{s} :: L16={acc16:.3f}  L64={acc64:.3f}", flush=True)
            del run["model"]; torch.cuda.empty_cache()
    print("\n=== GDN LR SWEEP (45M, k=5 composite, 25k steps; strict-acc, L16 in-distribution) ===", flush=True)
    for lr in LRS:
        xs16 = [a for a, _ in res[lr]]
        print(f"  lr={lr:.0e}  L16: {statistics.mean(xs16):.3f} (max {max(xs16):.3f}, {sum(x>0.5 for x in xs16)}/{len(xs16)} converge)", flush=True)
    print("gdn_lr_sweep done.", flush=True)

if __name__ == "__main__":
    main()
