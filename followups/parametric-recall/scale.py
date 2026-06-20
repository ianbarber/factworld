"""Does scale soften the internalisation dissociation? (length-generalisation OR no-scratchpad -> maybe both?)

Curriculum (curriculum.py) found, at ~6M: internalised (no-scratchpad) state-tracking works at the trained
length (answer-only L16, 3/5) but does NOT extrapolate (answer-only L64 = floor, 0/5), while the externalised
(scratchpad) tracker DOES extrapolate. The open question: is that dissociation a capacity limit that softens at
scale? Scale the mixed-density recipe (the best internaliser; order didn't matter) across:

    5.7M  (d256 x4)   -- the curriculum baseline
   18.5M  (d384 x6)
   44.8M  (d512 x8)   -- the paper's ~45M scale

Headline metric: ANSWER-ONLY L64 (was 0/5 floor at 6M). If it lifts with scale, the dissociation softens.
Also report answer-only L16 (internalisation at trained length) and dense L64 (externalised reference).

  .venv/bin/python followups/parametric-recall/scale.py
"""
import math
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import statistics

from dense_capstone import _world, e2e_eval
from supervision_sweep import build
from curriculum import make_pools, KS

SEEDS = [0, 1, 2]
TRAIN_LEN = (4, 8, 16)
STEPS = 6000
SCALES = [("5.7M", 256, 4, 1024), ("18.5M", 384, 6, 1536), ("44.8M", 512, 8, 2048)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scale.md")


def train_mixed(tok, pools, d_model, n_layers, d_ff, seed, device="cuda"):
    """Mixed-density training (random K per example) at a given model size. Recipe matches train.run."""
    import torch
    import torch.nn.functional as F
    from factworld.models import build_model
    torch.manual_seed(seed)
    model = build_model("gdp_hybrid", tok.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=4,
                        d_ff=d_ff, num_householder=4, allow_neg_eigval=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warmup, pad = 1000, tok.pad_id
    rng = random.Random(7000 + seed)
    allmix = [d for K in KS for d in pools[K]]
    model.train()
    for step in range(STEPS):
        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * (min(1.0, (step + 1) / warmup) if step < warmup else
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
    # eval: answer-only L16 + L64 (the dissociation), dense L64 (externalised reference). n=150.
    evs = {("ao", 16): [build(w, r, origins, oracle, 16, 10**9, random.Random(900 + j)) for j in range(150)],
           ("ao", 64): [build(w, r, origins, oracle, 64, 10**9, random.Random(964 + j)) for j in range(150)],
           ("dn", 64): [build(w, r, origins, oracle, 64, 1, random.Random(801 + j)) for j in range(150)]}
    agg = defaultdict(dict)
    print("=== SCALE CHECK: mixed-density internalisation across 5.7M / 18.5M / 44.8M ===", flush=True)
    for name, d_model, n_layers, d_ff in SCALES:
        for s in SEEDS:
            model = train_mixed(tok, pools, d_model, n_layers, d_ff, s)
            for key, ex in evs.items():
                _h, ev = e2e_eval(model, tok, w, ex)
                agg[(name, key)][s] = ev
            print(f"  {name:<6} s{s} :: answer-only L16={agg[(name,('ao',16))][s]:.3f} "
                  f"L64={agg[(name,('ao',64))][s]:.3f} | dense L64={agg[(name,('dn',64))][s]:.3f}", flush=True)
            del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("scale done.", flush=True)


def write_md(agg):
    lines = [
        "# Scale check — does the internalisation dissociation soften? (mixed-density recipe)\n",
        "`followups/parametric-recall/scale.py`. gdp_hybrid, mixed-density training, 6000 steps, 3 seeds, "
        "parametric. Free-running e2e composite. **Answer-only L64** is the headline (the dissociation wall: "
        "0/5 floor at 6M in curriculum.py). Floor = 0.20.\n",
        "| scale | answer-only L16 | answer-only L64 | dense (scratchpad) L64 |",
        "|---|---|---|---|",
    ]
    for name, _d, _l, _ff in SCALES:
        def cell(key):
            d = agg.get((name, key))
            if not d:
                return "…"
            xs = list(d.values())
            return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}({sum(x>0.5 for x in xs)}/{len(xs)})"
        lines.append(f"| {name} | {cell(('ao',16))} | {cell(('ao',64))} | {cell(('dn',64))} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
