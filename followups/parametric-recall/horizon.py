"""Candidate fix #1: horizon-extension curriculum. Can internalised tracking run PAST its trained length?

The scale check ruled out capacity: answer-only L64 stays at floor across 8x params. The dissociation
(internalised works at trained length, brittle beyond) is learnability/structural. One lever left untested is
the TRAINING HORIZON itself: the prior runs trained on lengths (4,8,16). Here we grow the max training length
over training (8 -> 16 -> 32 -> 48 -> 64), mixed-density throughout, and ask:

  - Does internalised (answer-only) tracking work at L64 when the recurrence is TRAINED at that step-count?
    (in-range generalisation)  -> distinguishes "never trained the recurrence that deep" from a deeper wall.
  - Does it then EXTRAPOLATE to L128, truly past training?  (the agentic target)

18.5M (d384x6), mixed density, 3 seeds. Eval answer-only at L16 / L64 / L128.

  .venv/bin/python followups/parametric-recall/horizon.py
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
LBUCKETS = [8, 16, 32, 48, 64]
D_MODEL, N_LAYERS, D_FF = 384, 6, 1536
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "horizon.md")


def unlocked(step):
    """Length curriculum: progressively admit longer sequences."""
    f = step / STEPS
    if f < 0.25: return [8, 16]
    if f < 0.50: return [8, 16, 32]
    if f < 0.75: return [8, 16, 32, 48]
    return LBUCKETS


def make_len_pools(w, r, origins, oracle, per=4000):
    """Per-length pools, each mixed over supervision density K."""
    pools = {}
    for L in LBUCKETS:
        rng = random.Random(5000 + L)
        pools[L] = [" ".join(build(w, r, origins, oracle, L, rng.choice(KS), rng)[0]) for _ in range(per)]
    return pools


def train(tok, pools, seed, device="cuda"):
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
        buckets = unlocked(step)
        docs = [rng.choice(pools[rng.choice(buckets)]) for _ in range(32)]
        seqs = [tok.encode(d) for d in docs]
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
    pools = make_len_pools(w, r, origins, oracle)
    tok, _, _ = T.prepare(pools[16][:50], [], [w])
    # answer-only eval at trained-edge (16), in-extended-range (64), and truly-OOD (128)
    evs = {16: [build(w, r, origins, oracle, 16, 10**9, random.Random(900 + j)) for j in range(150)],
           64: [build(w, r, origins, oracle, 64, 10**9, random.Random(964 + j)) for j in range(150)],
           128: [build(w, r, origins, oracle, 128, 10**9, random.Random(1028 + j)) for j in range(100)]}
    agg = defaultdict(dict)
    print("=== HORIZON CURRICULUM: train lengths grow to 64; answer-only eval at 16/64/128 ===", flush=True)
    for s in SEEDS:
        model = train(tok, pools, s)
        for L, ex in evs.items():
            _h, ev = e2e_eval(model, tok, w, ex)
            agg[L][s] = ev
        print(f"  s{s} :: answer-only L16={agg[16][s]:.3f} L64={agg[64][s]:.3f} L128={agg[128][s]:.3f}", flush=True)
        del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("horizon done.", flush=True)


def write_md(agg):
    lines = [
        "# Horizon-extension curriculum — does training the recurrence at length unlock internalised extrapolation?\n",
        "`followups/parametric-recall/horizon.py`. gdp_hybrid d384x6 (18.5M), mixed density, training lengths "
        "grow 8->16->32->48->64 over 6000 steps, 3 seeds. Free-running answer-only e2e (no scratchpad). Baseline "
        "(train<=16, from scale.py 18.5M): answer-only L64 = 0.20 floor. Floor = 0.20.\n",
        "| eval | answer-only e2e | note |",
        "|---|---|---|",
    ]
    note = {16: "trained edge", 64: "in extended train range", 128: "truly OOD (2x max train)"}
    for L in (16, 64, 128):
        d = agg.get(L)
        if not d:
            continue
        xs = list(d.values())
        cell = f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}({sum(x>0.5 for x in xs)}/{len(xs)})"
        lines.append(f"| L{L} | {cell} | {note[L]} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
