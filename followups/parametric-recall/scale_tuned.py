"""Scale-done-right: is the horizon wall capacity or under-training? LR-tuned + more steps + 70M ceiling.

The scale check (scale.py) held lr=1e-3 fixed across 5.7M→44.8M, so the bigger models may be under-tuned —
"capacity doesn't help" could be "capacity-at-fixed-recipe doesn't help" (the advisor's M1). Here we re-run the
LARGEST scales with a small per-scale LR sweep, a larger step budget, and a ~70M point (the 3090 ceiling). If
answer-only L64 (the internalized horizon wall) STILL floors at 70M with tuned LR, the wall is not relieved by
capacity or by under-training within reach — firming "learnability, not capacity."

  44.8M (d512 x8), 70M (d640 x8) × lr {1e-3, 5e-4} × 2 seeds × 8000 steps, mixed-density recipe.

  .venv/bin/python followups/parametric-recall/scale_tuned.py
"""
import math
import os
import random
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dense_capstone import _world, e2e_eval
from supervision_sweep import build
from curriculum import make_pools, KS

SEEDS = [0, 1]
STEPS = 8000
LRS = [1e-3, 5e-4]
SCALES = [("44.8M", 512, 8, 2048), ("70M", 640, 8, 2560)]
EVAL_LEN = [16, 64]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scale_tuned.md")


def train(tok, pools, d_model, n_layers, d_ff, lr, seed, device="cuda"):
    import torch
    import torch.nn.functional as F
    from factworld.models import build_model
    torch.manual_seed(seed)
    model = build_model("gdp_hybrid", tok.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=4,
                        d_ff=d_ff, num_householder=4, allow_neg_eigval=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup, pad = 1000, tok.pad_id
    rng = random.Random(7000 + seed)
    allmix = [d for K in KS for d in pools[K]]
    model.train()
    for step in range(STEPS):
        for pg in opt.param_groups:
            pg["lr"] = lr * (min(1.0, (step + 1) / warmup) if step < warmup else
                             0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, STEPS - warmup))))
        seqs = [tok.encode(d) for d in rng.sample(allmix, 32)]
        ml = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), ml), pad, dtype=torch.long, device=device)
        for ri, s in enumerate(seqs):
            inp[ri, : len(s)] = torch.tensor(s, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(inp[:, :-1]); tgt = inp[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, tok.vocab_size), tgt.reshape(-1), reduction="none")
            mask = (tgt != pad).float().reshape(-1)
            loss = (ce * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


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
    print("=== SCALE-DONE-RIGHT: LR-tuned + 70M ceiling; does answer-only L64 still floor? ===", flush=True)
    for name, d_model, n_layers, d_ff in SCALES:
        for lr in LRS:
            for s in SEEDS:
                model = train(tok, pools, d_model, n_layers, d_ff, lr, s)
                for L in EVAL_LEN:
                    _h, ev = e2e_eval(model, tok, w, evs[L])
                    agg[(name, lr)][(L, s)] = ev
                print(f"  {name} lr={lr} s{s} :: " +
                      "  ".join(f"L{L}={agg[(name, lr)][(L, s)]:.3f}" for L in EVAL_LEN), flush=True)
                del model; torch.cuda.empty_cache()
            write_md(agg)
    write_md(agg)
    print("scale_tuned done.", flush=True)


def write_md(agg):
    lines = [
        "# Scale-done-right — LR-tuned + 70M ceiling: is the horizon wall capacity or under-training?\n",
        "`followups/parametric-recall/scale_tuned.py`. gdp_hybrid, mixed-density, 8000 steps, 2 seeds. Largest "
        "scales with a per-scale LR sweep + a ~70M ceiling point. Headline = **answer-only L64** (the internalized "
        "horizon wall; floored 0/3 across 5.7M→44.8M at fixed lr in scale.py). Floor = 0.20. Still floor here = "
        "not relieved by capacity OR LR-tuning within 3090 reach.\n",
        "| scale | lr | answer-only L16 | answer-only L64 |",
        "|---|---|---|---|",
    ]
    for name, _d, _l, _ff in SCALES:
        for lr in LRS:
            d = agg.get((name, lr))
            if not d:
                continue
            def col(L):
                xs = [d[(L, s)] for s in SEEDS if (L, s) in d]
                return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "…"
            lines.append(f"| {name} | {lr} | {col(16)} | {col(64)} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
