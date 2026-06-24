"""Horizon mechanism: what sets the internalized length cap? Does it track the max training length?

The horizon curriculum (horizon.py) showed internalized tracking works in-range and partially extrapolates.
Question: what determines the cap — is it simply the maximum length the recurrence was trained to run? Train
internalized (mixed-density) to several MAX training lengths Lmax ∈ {16,32,48,64}, and eval answer-only at Lmax
(in-range) and 2×Lmax (OOD). If the solvable horizon tracks Lmax (in-range high, ~2× cliff each), the cap is the
trained step-count, and you extend reach by training longer — confirming the §6 lesson mechanistically.

  gdp_hybrid d384 (18.5M), mixed density, 3 seeds, answer-only e2e.

  .venv/bin/python followups/non-abelian-state/horizon_mech.py
"""
from __future__ import annotations

import math
import os
import random
import statistics
import sys
from collections import defaultdict
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dense_capstone import _world, e2e_eval
from supervision_sweep import build
from curriculum import KS

SEEDS = [0, 1, 2]
LMAXES = [16, 32, 48, 64]
STEPS = 6000
D_MODEL, N_LAYERS, D_FF = 384, 6, 1536
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "horizon_mech.md")


def lengths_up_to(Lmax: int) -> tuple:
    """Return the training lengths from the fixed grid that do not exceed ``Lmax``."""
    return tuple(L for L in (4, 8, 16, 24, 32, 48, 64) if L <= Lmax)


def train(tok: Any, pool: list[str], seed: int, device: str = "cuda") -> Any:
    """Train a gdp_hybrid model on a mixed-density pool (warmup + cosine; matches train.run).

    Args:
        tok: the atomic tokenizer.
        pool: training strings (the mixed-density pool for one ``Lmax``).
        seed: RNG seed for init and minibatch sampling.
        device: torch device.

    Returns:
        The trained ``HybridLM``.
    """
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


def main() -> None:
    """Train internalized to each Lmax across seeds and tabulate in-range vs 2x-Lmax OOD accuracy."""
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    tok, _, _ = T.prepare([" ".join(build(w, r, origins, oracle, 16, 1, random.Random(1))[0])], [], [w])
    agg = defaultdict(dict)
    print("=== HORIZON MECHANISM: internalized answer-only vs max-training-length ===", flush=True)
    for Lmax in LMAXES:
        lens = lengths_up_to(Lmax)
        prng = random.Random(2)
        pool = [" ".join(build(w, r, origins, oracle, prng.choice(lens), prng.choice(KS), prng)[0]) for _ in range(8000)]
        eval_at = [Lmax, min(2 * Lmax, 128)]
        evs = {L: [build(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(150)] for L in eval_at}
        for s in SEEDS:
            model = train(tok, pool, s)
            for L in eval_at:
                _h, ev = e2e_eval(model, tok, w, evs[L])
                agg[Lmax][(L, s)] = ev
            print(f"  Lmax={Lmax} s{s} :: in-range L{Lmax}={agg[Lmax][(Lmax, s)]:.3f}  "
                  f"OOD L{eval_at[1]}={agg[Lmax][(eval_at[1], s)]:.3f}", flush=True)
            del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("horizon_mech done.", flush=True)


def write_md(agg: dict) -> None:
    """Write the per-Lmax in-range vs OOD accuracy table (mean ± pstdev over seeds) to ``OUT``."""
    lines = [
        "# Horizon mechanism — does the internalized length cap track the max training length?\n",
        "`followups/non-abelian-state/horizon_mech.py`. gdp_hybrid d384 (18.5M), mixed density, 3 seeds. Train "
        "internalized to max length Lmax; answer-only e2e at Lmax (in-range) and 2×Lmax (OOD). If in-range stays "
        "high and the cliff falls near ~2×Lmax for each Lmax, the cap = trained step-count. Floor = 0.20.\n",
        "| Lmax (max train len) | in-range (L=Lmax) | OOD (L=2×Lmax) |",
        "|---|---|---|",
    ]
    for Lmax in LMAXES:
        d = agg.get(Lmax)
        if not d:
            continue
        ood = min(2 * Lmax, 128)
        def col(L):
            xs = [d[(L, s)] for s in SEEDS if (L, s) in d]
            return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "…"
        lines.append(f"| {Lmax} | {col(Lmax)} | {col(ood)} (L{ood}) |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
