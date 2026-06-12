"""W3 (critical-review): does the transformer floor survive its BEST config, and does short-conv
change the recurrent arms? 45M composite (k=5 genuine in-context-copy), 25k steps, scan-lenient eval.

- transformer FAIR: n_heads=8 (head_dim 64, vs the confounded 128) + residual-scaled init, 5 LRs x 2 seeds.
  Confounded baseline floored 0/10 (transformer_lr_sweep.py, n_heads=4). Does the fair config escape it?
- gdp_hybrid / gdn_hybrid: use_short_conv=True (the published recipe component, previously forced off) at
  their best LR (5e-4 / 3e-4), 3 seeds. Compare to short_conv=OFF (gdp_confirm 0.87/0.76; gdn ~seed-fragile).

  .venv/bin/python scripts/fair_config.py
"""
import os, statistics, sys
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); sys.path.insert(0, REPO+"/scripts")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from recall_learn import _world, pure_copy, composite, composite_eval
from iso import strict_eval

STEPS, BATCH, EVAL_L = 25000, 32, [16, 64]
LRS = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]


def main():
    import torch
    from factworld import train as T
    w, r = _world()
    tr = composite(w, r, 8000, 2, True) + pure_copy(w, r, 4000, 3, True)
    ev = {L: composite_eval(w, r, L, 200, 300 + L) for L in EVAL_L}
    tok, docs, _ = T.prepare(tr, [], [w])

    def go(arch, lr, seed, **kw):
        run = T.run(arch, tok, docs, [], steps=STEPS, batch=BATCH, d_model=512, n_layers=8,
                    lr=lr, seed=seed, return_model=True, **kw)
        a16 = strict_eval(run["model"], tok, w, ev[16]); a64 = strict_eval(run["model"], tok, w, ev[64])
        del run["model"]; torch.cuda.empty_cache()
        return a16, a64

    print("=== TRANSFORMER FAIR (n_heads=8 / head_dim 64 + resid_init) — does the 0/10 floor survive? ===", flush=True)
    res = defaultdict(list)
    for lr in LRS:
        for s in range(2):
            a16, a64 = go("transformer", lr, s, n_heads=8, d_ff=3072, resid_init=True)
            res[lr].append((a16, a64))
            print(f"  transf-fair lr={lr:.0e} s{s} :: L16={a16:.3f}  L64={a64:.3f}", flush=True)
    allmax = 0.0
    for lr in LRS:
        xs = [a for a, _ in res[lr]]
        allmax = max(allmax, max(xs))
        print(f"  lr={lr:.0e}  L16 {statistics.mean(xs):.3f} (max {max(xs):.3f}, {sum(x>0.5 for x in xs)}/{len(xs)} conv)", flush=True)
    print(f"  -> transformer-fair best single run {allmax:.3f} (floor survives if <0.5)", flush=True)

    print("=== RECURRENT short_conv=ON at best LR (vs OFF) ===", flush=True)
    for arch, lr, dff, off in [("gdp_hybrid", 5e-4, 2048, "0.87/0.76"), ("gdn_hybrid", 3e-4, 2816, "seed-fragile")]:
        rows = [go(arch, lr, s, d_ff=dff, use_short_conv=True) for s in range(3)]
        a16s = [a for a, _ in rows]; a64s = [b for _, b in rows]
        print(f"  {arch} sc=ON lr={lr:.0e} :: L16 {statistics.mean(a16s):.3f} ({sum(x>0.5 for x in a16s)}/3)  "
              f"L64 {statistics.mean(a64s):.3f} ({sum(x>0.5 for x in a64s)}/3)   [sc=OFF was {off}]", flush=True)
    print("fair_config done.", flush=True)


if __name__ == "__main__":
    main()
