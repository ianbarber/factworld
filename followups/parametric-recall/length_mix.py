"""What nature of training examples triggers extrapolation? Vary the training-LENGTH DISTRIBUTION at fixed capacity.

The horizon curriculum showed the solvable cap tracks max training length, but not WHAT distribution of lengths
unlocks extrapolation. Fix capacity (18.5M) and vary the training-length mix; measure answer-only (internalized,
no-scratchpad) accuracy at L32 / L64 / L128:

  short16   lengths {4,8,16} only                     (baseline — floors past L16)
  long5     {4,8,16} + 5%  at L64                      (does a SMALL fraction of long examples unlock L64?)
  long20    {4,8,16} + 20% at L64
  long50    {4,8,16} + 50% at L64
  mid       {16,32} only                               (longer-than-16 but NOT 64 — is the cap max-len or coverage?)
  full      uniform {4,8,16,32,48,64}                  (full coverage)

Answers: is extrapolation a sharp threshold (any long example unlocks it) or graded in the long-fraction; and
is the reachable horizon set by the MAX trained length or by needing examples AT the target length.

  .venv/bin/python followups/parametric-recall/length_mix.py
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
from curriculum import KS

SEEDS = [0, 1, 2]
STEPS = 6000
EVAL_LEN = [32, 64, 128]
D_MODEL, N_LAYERS, D_FF = 384, 6, 1536
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "length_mix.md")

SHORT = [4, 8, 16]
CONDS = {                                          # name -> sampler(rng) -> length
    "short16": lambda g: g.choice(SHORT),
    "long5":   lambda g: 64 if g.random() < 0.05 else g.choice(SHORT),
    "long20":  lambda g: 64 if g.random() < 0.20 else g.choice(SHORT),
    "long50":  lambda g: 64 if g.random() < 0.50 else g.choice(SHORT),
    "mid":     lambda g: g.choice([16, 32]),
    "full":    lambda g: g.choice([4, 8, 16, 32, 48, 64]),
}


def train(tok, pool, seed, device="cuda"):
    import torch
    import torch.nn.functional as F
    from factworld.models import build_model
    torch.manual_seed(seed)
    model = build_model("gdp_hybrid", tok.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=4,
                        d_ff=D_FF, num_householder=4, allow_neg_eigval=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warmup, pad = 1000, tok.pad_id
    rng = random.Random(7000 + seed)
    model.train()
    for step in range(STEPS):
        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * (min(1.0, (step + 1) / warmup) if step < warmup else
                               0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, STEPS - warmup))))
        seqs = [tok.encode(d) for d in rng.sample(pool, 32)]
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
    tok, _, _ = T.prepare([" ".join(build(w, r, origins, oracle, 16, 1, random.Random(1))[0])], [], [w])
    evs = {L: [build(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(150)]
           for L in EVAL_LEN}
    agg = defaultdict(lambda: defaultdict(dict))
    print("=== LENGTH-MIX: which training-length distribution unlocks extrapolation? (18.5M) ===", flush=True)
    for name, sampler in CONDS.items():
        prng = random.Random(2)
        pool = [" ".join(build(w, r, origins, oracle, sampler(prng), prng.choice(KS), prng)[0]) for _ in range(8000)]
        for s in SEEDS:
            model = train(tok, pool, s)
            for L in EVAL_LEN:
                _h, ev = e2e_eval(model, tok, w, evs[L])
                agg[name][L][s] = ev
            print(f"  {name:<8} s{s} :: " + "  ".join(f"L{L}={agg[name][L][s]:.3f}" for L in EVAL_LEN), flush=True)
            del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("length_mix done.", flush=True)


def write_md(agg):
    lines = [
        "# Length-mix — which training-length distribution unlocks extrapolation? (`length_mix.py`, 18.5M, 3 seeds)\n",
        "Fixed capacity (d384x6, 18.5M), mixed density. Vary the training-length distribution; answer-only "
        "(internalized) e2e at L32/L64/L128. `short16`/`mid`/`full` describe the lengths trained on; `longN` = "
        "{4,8,16} plus N% examples at L64. Floor = 0.20.\n",
        "| training lengths | L32 | L64 | L128 |",
        "|---|---|---|---|",
    ]
    desc = {"short16": "{4,8,16} only", "long5": "{4,8,16} + 5% L64", "long20": "{4,8,16} + 20% L64",
            "long50": "{4,8,16} + 50% L64", "mid": "{16,32} only", "full": "uniform {4..64}"}
    for name in CONDS:
        if name not in agg:
            continue
        cells = []
        for L in EVAL_LEN:
            xs = list(agg[name][L].values())
            cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "…")
        lines.append(f"| {desc[name]} | " + " | ".join(cells) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
