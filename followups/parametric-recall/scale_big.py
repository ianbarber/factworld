"""Push capacity to the 3090 ceiling: does the weak 70M lift become a real capacity trend?

scale_tuned showed answer-only L64 floors to 44.8M and lifts weakly at 70M (0.41 at lr 1e-3). The 70M run used
only 4.6 GB of 24 GB, so we were nowhere near the hardware limit. Here we extend the ladder at the SAME recipe
(mixed-density, lr 1e-3, batch 32, 8000 steps — comparable to scale_tuned's 70M point) to 140M / 268M / 357M
(357M ≈ 21.9 GB peak, the batch-32 ceiling). Headline = answer-only L64 (the internalized horizon wall).
  - L64 climbs monotonically with params -> the wall IS capacity-limited, just past 44.8M (a scaling trend).
  - L64 plateaus weak -> capacity gives a bounded, non-solving bump; extrapolation needs length, not width.

Combine with scale_tuned's 44.8M/70M for the full 44.8M->357M ladder. Reuses scale_tuned.train (bs32, 8000 steps).

  .venv/bin/python followups/parametric-recall/scale_big.py
"""
import os
import random
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from scale_tuned import _world, e2e_eval, build, make_pools, train  # train: bs32, STEPS=8000, n_heads=4

SEEDS = [0, 1]
LR = 1e-3
SCALES = [("140M", 640, 16, 2560), ("268M", 1024, 12, 4096), ("357M", 1024, 16, 4096)]
EVAL_LEN = [16, 64]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scale_big.md")


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    pools = make_pools(w, r, origins, oracle)
    tok, _, _ = T.prepare(pools[1][:50], [], [w])
    evs = {L: [build(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(150)]
           for L in EVAL_LEN}
    agg = defaultdict(dict)
    print("=== SCALE PUSH: 140M / 268M / 357M @ lr 1e-3 bs32 — does answer-only L64 climb? ===", flush=True)
    print("    (combine with scale_tuned: 44.8M=0.24, 70M=0.41 at lr 1e-3)", flush=True)
    for name, d_model, n_layers, d_ff in SCALES:
        for s in SEEDS:
            try:
                model = train(tok, pools, d_model, n_layers, d_ff, LR, s)  # optimizer freed on return
                for L in EVAL_LEN:
                    _h, ev = e2e_eval(model, tok, w, evs[L])
                    agg[name][s] = agg[name].get(s, {}); agg[name][s][L] = ev
                print(f"  {name} s{s} :: " + "  ".join(f"L{L}={agg[name][s][L]:.3f}" for L in EVAL_LEN), flush=True)
                del model; torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  {name} s{s} :: OOM (skipped)", flush=True); torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("scale_big done.", flush=True)


def write_md(agg):
    lines = [
        "# Scale push — does the horizon wall lift with capacity past 70M? (`scale_big.py`, lr 1e-3, bs32, 2 seeds)\n",
        "gdp_hybrid, mixed-density, 8000 steps, batch 32 (consistent with scale_tuned's 70M point). Answer-only "
        "L64 is the internalized horizon wall. Context from scale_tuned (same recipe): 44.8M = 0.24, 70M = 0.41 "
        "(lr 1e-3). Floor = 0.20.\n",
        "| parameters | answer-only L16 | answer-only L64 (the wall) |",
        "|---|---|---|",
        "| 44.8M (scale_tuned) | — | 0.24 |",
        "| 70M (scale_tuned) | — | 0.41 |",
    ]
    for name, _d, _l, _ff in SCALES:
        d = agg.get(name, {})
        def col(L):
            xs = [d[s][L] for s in d if L in d[s]]
            return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "…"
        lines.append(f"| {name} | {col(16)} | {col(64)} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
