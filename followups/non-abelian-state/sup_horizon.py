"""Rigor thread: do supervision DENSITY and training HORIZON interact? (the unmeasured 2D cell)

The cliff (supervision_sweep) was measured at train length ≤16; the horizon result (horizon.py) varied length
at mixed density. We never crossed them. Question: does training the recurrence at LENGTH lower the supervision
density needed to form the circuit? Run the density sweep K∈{1,2,4,8,inf} but with training lengths grown to 64
(vs the ≤16 baseline in supervision_sweep.md), single-role checkpoint, parametric composite, eval at L64/L128.
  - cliff threshold moves to SPARSER K at long-horizon training -> length and density trade off (good news).
  - threshold unchanged -> density requirement is horizon-independent.

gdp_hybrid d384 (18.5M), 3 seeds. Compare directly to supervision_sweep.md (same build, train ≤16).

  .venv/bin/python followups/non-abelian-state/sup_horizon.py
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

SEEDS = [0, 1, 2]
TRAIN_LEN = (4, 8, 16, 32, 48, 64)     # LONG-horizon training (vs ≤16 baseline)
EVAL_LEN = [64, 128]
KS = [1, 2, 4, 8, 10**9]
STEPS = 6000
D_MODEL, N_LAYERS, D_FF = 384, 6, 1536
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sup_horizon.md")


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
    agg = defaultdict(lambda: defaultdict(dict))
    print("=== SUPERVISION × HORIZON: density sweep at long-horizon training (≤64) ===", flush=True)
    for K in KS:
        prng = random.Random(2)
        pool = [" ".join(build(w, r, origins, oracle, prng.choice(TRAIN_LEN), K, prng)[0]) for _ in range(8000)]
        evs = {L: [build(w, r, origins, oracle, L, K, random.Random(900 + L + j)) for j in range(150)] for L in EVAL_LEN}
        tag = "inf" if K >= 10**9 else str(K)
        for s in SEEDS:
            model = train(tok, pool, s)
            for L in EVAL_LEN:
                _h, ev = e2e_eval(model, tok, w, evs[L])
                agg[tag][L][s] = ev
            print(f"  K={tag:<4} s{s} :: " + "  ".join(f"L{L}={agg[tag][L][s]:.3f}" for L in EVAL_LEN), flush=True)
            del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("sup_horizon done.", flush=True)


def write_md(agg):
    lines = [
        "# Supervision × horizon — does training at length lower the density needed?\n",
        "`followups/non-abelian-state/sup_horizon.py`. gdp_hybrid d384 (18.5M), 3 seeds. Density sweep K with "
        "LONG-horizon training (lengths ≤64), parametric composite, single-role checkpoint, free-running e2e. "
        "Compare the cliff threshold to `supervision_sweep.md` (same build, train ≤16): if the circuit now forms "
        "at sparser K, length and density trade off. Floor = 0.20.\n",
        "| K | L64 (in extended range) | L128 (2× OOD) |",
        "|---|---|---|",
    ]
    for tag in ["1", "2", "4", "8", "inf"]:
        if tag not in agg:
            continue
        cells = []
        for L in EVAL_LEN:
            xs = list(agg[tag][L].values())
            cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}({sum(x>0.5 for x in xs)}/{len(xs)})")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
