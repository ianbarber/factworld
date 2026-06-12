"""Confirm the scale result: composition is scale-soluble + architecture matters (p(converge), multi-seed).

A discovery run found (control valid) that gdp 45M solves the in-context-copy composite and extrapolates
where 6M and a same-scale transformer floor — but on 2 gdp / 1 transformer seed. This firms it:
  - gdp_hybrid 45M, 5 seeds          (does it converge reliably? p(converge))
  - transformer ~45M (PARAM-MATCHED), 5 seeds   (does it robustly floor? d_ff widened to match the params
                                                 the transformer lacks from the recurrent mixer)
  - gdn_hybrid 45M, 3 seeds          (is it GDP-specific, or does ANY recurrent+attention hybrid compose?)
Task: k=5 random-map composite + copy-curriculum. Reports mean±std and p(converge),
i.e. #seeds with strict-acc>0.5 (the cells are bimodal: a seed either finds the circuit or floors).
Run on the 3090:  .venv/bin/python scripts/scale_confirm.py
"""
import os
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from recall_learn import _world, pure_copy, composite, composite_eval  # noqa: E402
from iso import strict_eval  # noqa: E402

STEPS = 25000
BATCH = 32
EVAL_L = [16, 64]
# (label, arch, d_model, n_layers, d_ff, n_seeds) — d_ff matched so transformer ≈ gdp params (~45M)
ARMS = [
    ("gdp-45M",    "gdp_hybrid",  512, 8, 2048, 5),
    ("transf-45M", "transformer", 512, 8, 3072, 5),
    ("gdn-45M",    "gdn_hybrid",  512, 8, 2816, 3),
]


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    w, r = _world()
    tr = composite(w, r, 8000, 2, True) + pure_copy(w, r, 4000, 3, True)
    ev = {L: composite_eval(w, r, L, 200, 300 + L) for L in EVAL_L}
    tok, docs, _ = T.prepare(tr, [], [w])

    res = defaultdict(lambda: defaultdict(list))
    pm = {}
    for label, arch, dm, nl, dff, nseed in ARMS:
        for s in range(nseed):
            run = T.run(arch, tok, docs, [], steps=STEPS, batch=BATCH, d_model=dm, n_layers=nl, d_ff=dff,
                        seed=s, return_model=True)
            pm[label] = run["model"].num_params() / 1e6
            for L in EVAL_L:
                res[label][L].append(strict_eval(run["model"], tok, w, ev[L]))
            del run["model"]; torch.cuda.empty_cache()
            print(f"  {label} s{s} ({pm[label]:.0f}M) :: " + " ".join(f"L{L}={res[label][L][-1]:.2f}" for L in EVAL_L), flush=True)

    print(f"\n=== SCALE CONFIRM (k=5 in-context-copy composite, {STEPS} steps; mean±std, p(converge)=#>0.5) ===", flush=True)
    for label, *_ in ARMS:
        cells = []
        for L in EVAL_L:
            xs = res[label][L]
            cells.append(f"L{L}:{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f} ({sum(x > 0.5 for x in xs)}/{len(xs)})")
        print(f"  {label:<12} (~{pm.get(label, 0):.0f}M)  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
