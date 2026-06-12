"""The load-bearing §5 control (critical-review W2): gdn 45M at lr 3e-4, 5 seeds.

The recurrence-vs-attention reframe rests on gdn converging at lr 3e-4 (the 2-seed sweep gave 1/2 there).
With only 2 seeds that single converging cell could be a fluke. This pins it with 5 seeds, same budget as
gdp_confirm_5e4.py. If gdn is 0/5 or 1/5 here, "recurrence vs attention" collapses back to "product is the
lever" and §5 must say so. gdn 45M config = scale_confirm.py (d512x8, d_ff=2816).

  .venv/bin/python scripts/gdn_confirm_3e4.py
"""
import os, statistics, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); sys.path.insert(0, REPO+"/scripts")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from recall_learn import _world, pure_copy, composite, composite_eval
from iso import strict_eval

STEPS, BATCH, EVAL_L, LR, SEEDS = 25000, 32, [16, 64], 3e-4, 5

def main():
    import torch
    from factworld import train as T
    w, r = _world()
    tr = composite(w, r, 8000, 2, True) + pure_copy(w, r, 4000, 3, True)
    ev = {L: composite_eval(w, r, L, 200, 300 + L) for L in EVAL_L}
    tok, docs, _ = T.prepare(tr, [], [w])
    rows = []
    for s in range(SEEDS):
        run = T.run("gdn_hybrid", tok, docs, [], steps=STEPS, batch=BATCH, d_model=512, n_layers=8,
                    d_ff=2816, lr=LR, seed=s, return_model=True)
        a16 = strict_eval(run["model"], tok, w, ev[16]); a64 = strict_eval(run["model"], tok, w, ev[64])
        rows.append((a16, a64))
        print(f"  gdn-45M lr=3e-04 s{s} :: L16={a16:.3f}  L64={a64:.3f}", flush=True)
        del run["model"]; torch.cuda.empty_cache()
    a16s = [a for a, _ in rows]; a64s = [b for _, b in rows]
    print(f"\n=== GDN 3e-4 CONFIRM (45M, k=5 composite, 25k steps, 5 seeds) ===", flush=True)
    print(f"  L16: {statistics.mean(a16s):.3f}±{statistics.pstdev(a16s):.3f}  ({sum(x>0.5 for x in a16s)}/5 converge)", flush=True)
    print(f"  L64: {statistics.mean(a64s):.3f}±{statistics.pstdev(a64s):.3f}  ({sum(x>0.5 for x in a64s)}/5 extrapolate)", flush=True)
    print("gdn_confirm_3e4 done.", flush=True)

if __name__ == "__main__":
    main()
