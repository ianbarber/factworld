"""Does scale soften the internalisation dissociation, and is the horizon wall capacity or under-training?

Curriculum (curriculum.py) found, at ~6M: internalised (no-scratchpad) state-tracking works at the trained
length (answer-only L16, 3/5) but does NOT extrapolate (answer-only L64 = floor, 0/5), while the externalised
(scratchpad) tracker DOES extrapolate. The open question: is that dissociation a capacity limit that softens at
scale? This script answers it across the full 5.7M -> 357M ladder in THREE phases (mixed-density recipe, the
best internaliser; order didn't matter). Headline metric throughout: ANSWER-ONLY L64 (was 0/5 floor at 6M).

  Phase 1 — base ladder: 5.7M (d256x4) / 18.5M (d384x6) / 44.8M (d512x8), lr 1e-3, 6000 steps, 3 seeds.
            Also reports answer-only L16 (internalisation at trained length) and dense L64 (externalised ref).
  Phase 2 — LR control: 44.8M / 70M (d640x8) x lr {1e-3, 5e-4} x 2 seeds x 8000 steps. Is "capacity doesn't
            help" really "capacity-at-fixed-recipe doesn't help"? If answer-only L64 still floors at 70M with
            tuned LR, the wall is not relieved by capacity or by under-training within reach.
  Phase 3 — push: 140M (d640x16) / 268M (d1024x12) / 357M (d1024x16), lr 1e-3, bs32, 8000 steps, 2 seeds
            (357M ~ 21.9 GB peak, the batch-32 ceiling). Does answer-only L64 climb monotonically (capacity-
            limited wall, a scaling trend) or plateau weak (bounded bump; extrapolation needs length not width)?

  .venv/bin/python followups/non-abelian-state/scale.py
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
from curriculum import make_pools, KS

# --- Phase 1: base ladder (lr 1e-3, 6000 steps, 3 seeds) ---
BASE_SEEDS = [0, 1, 2]
TRAIN_LEN = (4, 8, 16)
BASE_STEPS = 6000
BASE_SCALES = [("5.7M", 256, 4, 1024), ("18.5M", 384, 6, 1536), ("44.8M", 512, 8, 2048)]

# --- Phase 2: LR control (8000 steps, 2 seeds, lr sweep) ---
TUNED_SEEDS = [0, 1]
TUNED_STEPS = 8000
TUNED_LRS = [1e-3, 5e-4]
TUNED_SCALES = [("44.8M", 512, 8, 2048), ("70M", 640, 8, 2560)]
EVAL_LEN = [16, 64]

# --- Phase 3: push (lr 1e-3, bs32, 8000 steps, 2 seeds) ---
BIG_SEEDS = [0, 1]
BIG_LR = 1e-3
BIG_STEPS = 8000
BIG_SCALES = [("140M", 640, 16, 2560), ("268M", 1024, 12, 4096), ("357M", 1024, 16, 4096)]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scale.md")


def train(tok: Any, pools: dict, d_model: int, n_layers: int, d_ff: int, lr: float, seed: int,
          steps: int, device: str = "cuda") -> Any:
    """Train a gdp_hybrid at the given scale and LR on the mixed-density pools (random K per example).

    The canonical lr-parameterized trainer reused by all three phases. Recipe matches train.run: batch 32,
    AdamW(wd=0.01), 1000-step warmup then cosine decay, bf16 autocast, grad-clip 1.0.

    Args:
        tok: the atomic tokenizer.
        pools: maps chain-length K -> a list of training strings.
        d_model: model width.
        n_layers: number of layers.
        d_ff: feed-forward width.
        lr: peak learning rate.
        seed: RNG seed for init and minibatch sampling.
        steps: number of optimizer steps.
        device: torch device.

    Returns:
        The trained ``gdp_hybrid`` model.
    """
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
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr * (min(1.0, (step + 1) / warmup) if step < warmup else
                             0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
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


def main() -> None:
    """Run the three scale phases (base ladder, LR control, push) and tabulate the 5.7M->357M ladder."""
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
    base_evs = {("ao", 16): [build(w, r, origins, oracle, 16, 10**9, random.Random(900 + j)) for j in range(150)],
                ("ao", 64): [build(w, r, origins, oracle, 64, 10**9, random.Random(964 + j)) for j in range(150)],
                ("dn", 64): [build(w, r, origins, oracle, 64, 1, random.Random(801 + j)) for j in range(150)]}
    # phases 2 & 3 share answer-only L16/L64 eval prompts.
    evs = {L: [build(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(150)]
           for L in EVAL_LEN}

    base_agg: dict = defaultdict(dict)
    tuned_agg: dict = defaultdict(dict)
    big_agg: dict = defaultdict(dict)

    # ============================================================================================
    # PHASE 1 — base ladder: 5.7M / 18.5M / 44.8M, lr 1e-3, 6000 steps, 3 seeds.
    # ============================================================================================
    print("=== PHASE 1 — base ladder: mixed-density internalisation across 5.7M / 18.5M / 44.8M ===", flush=True)
    for name, d_model, n_layers, d_ff in BASE_SCALES:
        for s in BASE_SEEDS:
            model = train(tok, pools, d_model, n_layers, d_ff, 1e-3, s, BASE_STEPS)
            for key, ex in base_evs.items():
                _h, ev = e2e_eval(model, tok, w, ex)
                base_agg[(name, key)][s] = ev
            print(f"  {name:<6} s{s} :: answer-only L16={base_agg[(name,('ao',16))][s]:.3f} "
                  f"L64={base_agg[(name,('ao',64))][s]:.3f} | dense L64={base_agg[(name,('dn',64))][s]:.3f}",
                  flush=True)
            del model; torch.cuda.empty_cache()
        write_md(base_agg, tuned_agg, big_agg)

    # ============================================================================================
    # PHASE 2 — LR control: 44.8M / 70M x lr {1e-3, 5e-4} x 2 seeds x 8000 steps.
    # ============================================================================================
    print("=== PHASE 2 — LR control: tuned LR + 70M ceiling; does answer-only L64 still floor? ===", flush=True)
    for name, d_model, n_layers, d_ff in TUNED_SCALES:
        for lr in TUNED_LRS:
            for s in TUNED_SEEDS:
                model = train(tok, pools, d_model, n_layers, d_ff, lr, s, TUNED_STEPS)
                for L in EVAL_LEN:
                    _h, ev = e2e_eval(model, tok, w, evs[L])
                    tuned_agg[(name, lr)][(L, s)] = ev
                print(f"  {name} lr={lr} s{s} :: " +
                      "  ".join(f"L{L}={tuned_agg[(name, lr)][(L, s)]:.3f}" for L in EVAL_LEN), flush=True)
                del model; torch.cuda.empty_cache()
            write_md(base_agg, tuned_agg, big_agg)

    # ============================================================================================
    # PHASE 3 — push: 140M / 268M / 357M, lr 1e-3, bs32, 8000 steps, 2 seeds (OOM-guarded).
    # ============================================================================================
    print("=== PHASE 3 — push: 140M / 268M / 357M @ lr 1e-3 bs32 — does answer-only L64 climb? ===", flush=True)
    for name, d_model, n_layers, d_ff in BIG_SCALES:
        for s in BIG_SEEDS:
            try:
                model = train(tok, pools, d_model, n_layers, d_ff, BIG_LR, s, BIG_STEPS)  # opt freed on return
                for L in EVAL_LEN:
                    _h, ev = e2e_eval(model, tok, w, evs[L])
                    big_agg[name][s] = big_agg[name].get(s, {}); big_agg[name][s][L] = ev
                print(f"  {name} s{s} :: " + "  ".join(f"L{L}={big_agg[name][s][L]:.3f}" for L in EVAL_LEN),
                      flush=True)
                del model; torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  {name} s{s} :: OOM (skipped)", flush=True); torch.cuda.empty_cache()
        write_md(base_agg, tuned_agg, big_agg)

    write_md(base_agg, tuned_agg, big_agg)
    print("scale done.", flush=True)


def write_md(base_agg: dict, tuned_agg: dict, big_agg: dict) -> None:
    """Write the full 5.7M->357M ladder (headline answer-only L64) plus the LR-control table to ``OUT``.

    Args:
        base_agg: phase-1 results; maps (scale name, eval key) -> seed -> accuracy.
        tuned_agg: phase-2 results; maps (scale name, lr) -> (length, seed) -> accuracy.
        big_agg: phase-3 results; maps scale name -> seed -> {length: accuracy}.
    """
    def base_cell(name: str, key: tuple) -> str:
        """Format mean ± pstdev (hit-count) for one phase-1 (scale, eval-key) cell (``…`` if empty)."""
        d = base_agg.get((name, key))
        if not d:
            return "…"
        xs = list(d.values())
        return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}({sum(x>0.5 for x in xs)}/{len(xs)})"

    def big_cell(name: str, L: int) -> str:
        """Format mean ± pstdev accuracy at length ``L`` across seeds for a phase-3 scale (``…`` if empty)."""
        d = big_agg.get(name, {})
        xs = [d[s][L] for s in d if L in d[s]]
        return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "…"

    lines = [
        "# Scale ladder — does the internalisation/horizon wall soften with capacity? (5.7M -> 357M)\n",
        "`followups/non-abelian-state/scale.py`. gdp_hybrid, mixed-density training, parametric. Free-running "
        "e2e composite. **Answer-only L64** is the headline (the dissociation/horizon wall: 0/5 floor at 6M in "
        "curriculum.py). Floor = 0.20. Three phases: (1) base ladder 5.7M/18.5M/44.8M lr 1e-3, 6000 steps, 3 "
        "seeds; (2) LR control 44.8M/70M x lr {1e-3, 5e-4}, 8000 steps, 2 seeds; (3) push 140M/268M/357M lr "
        "1e-3, bs32, 8000 steps, 2 seeds.\n",
        "## Ladder (headline answer-only L64)\n",
        "| scale | answer-only L16 | answer-only L64 | dense (scratchpad) L64 |",
        "|---|---|---|---|",
    ]
    # base ladder (phase 1) rows: full L16/L64/dense.
    for name, _d, _l, _ff in BASE_SCALES:
        lines.append(f"| {name} | {base_cell(name, ('ao', 16))} | {base_cell(name, ('ao', 64))} "
                     f"| {base_cell(name, ('dn', 64))} |")
    # push (phase 3) rows: answer-only L16/L64 only.
    for name, _d, _l, _ff in BIG_SCALES:
        lines.append(f"| {name} | {big_cell(name, 16)} | {big_cell(name, 64)} | — |")

    # LR-control (phase 2) table.
    lines += [
        "\n## LR control (phase 2 — tuned LR + 70M ceiling, answer-only)\n",
        "| scale | lr | answer-only L16 | answer-only L64 |",
        "|---|---|---|---|",
    ]
    for name, _d, _l, _ff in TUNED_SCALES:
        for lr in TUNED_LRS:
            d = tuned_agg.get((name, lr))
            if not d:
                continue
            def col(L: int) -> str:
                xs = [d[(L, s)] for s in TUNED_SEEDS if (L, s) in d]
                return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "…"
            lines.append(f"| {name} | {lr} | {col(16)} | {col(64)} |")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
