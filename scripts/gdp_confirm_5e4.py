"""Defensible headline for §5: gdp 45M at its best LR (5e-4, from gdp_lr_sweep.py), 5 seeds.

The 5-LR × 2-seed sweep showed gdp converges in-distribution across the 3e-4–2e-3 band (7/10 cells) and
length-extrapolates at 5e-4 (one seed L64 0.875), but 2 seeds/cell is noisy (at 1e-3 the sweep got 2/2
while the original scale_confirm 5-seed run got 1/5). This pins the point estimate at the best LR with the
same seed budget as the original headline: gdp_hybrid 45M, lr 5e-4, 5 seeds, same k=5 composite.

Reports p(converge) at L16 (in-distribution) and L64 (4× extrapolation) — the defensible "at a tuned LR the
product hybrid converges X/5 and length-extrapolates Y/5" numbers for the §5 rewrite.

  .venv/bin/python scripts/gdp_confirm_5e4.py
"""
import os, statistics, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); sys.path.insert(0, REPO+"/scripts")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from recall_learn import _world, pure_copy, composite, composite_eval
from iso import strict_eval

STEPS, BATCH, EVAL_L, LR, SEEDS = 25000, 32, [16, 64], 5e-4, 5

def main():
    import torch
    from factworld import train as T
    w, r = _world()
    tr = composite(w, r, 8000, 2, True) + pure_copy(w, r, 4000, 3, True)
    ev = {L: composite_eval(w, r, L, 200, 300 + L) for L in EVAL_L}
    tok, docs, _ = T.prepare(tr, [], [w])
    rows = []
    for s in range(SEEDS):
        run = T.run("gdp_hybrid", tok, docs, [], steps=STEPS, batch=BATCH, d_model=512, n_layers=8,
                    d_ff=2048, lr=LR, seed=s, return_model=True)
        a16 = strict_eval(run["model"], tok, w, ev[16]); a64 = strict_eval(run["model"], tok, w, ev[64])
        rows.append((a16, a64))
        print(f"  gdp-45M lr=5e-04 s{s} :: L16={a16:.3f}  L64={a64:.3f}", flush=True)
        del run["model"]; torch.cuda.empty_cache()
    a16s = [a for a, _ in rows]; a64s = [b for _, b in rows]
    print(f"\n=== GDP 5e-4 CONFIRM (45M, k=5 composite, 25k steps, 5 seeds) ===", flush=True)
    print(f"  L16: {statistics.mean(a16s):.3f}±{statistics.pstdev(a16s):.3f}  ({sum(x>0.5 for x in a16s)}/5 converge)", flush=True)
    print(f"  L64: {statistics.mean(a64s):.3f}±{statistics.pstdev(a64s):.3f}  ({sum(x>0.5 for x in a64s)}/5 extrapolate)", flush=True)
    print("gdp_confirm_5e4 done.", flush=True)

if __name__ == "__main__":
    main()
