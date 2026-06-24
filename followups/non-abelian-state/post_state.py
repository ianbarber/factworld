"""Post-training state coverage: does it extend the non-abelian circuit, when from-scratch burn-in didn't?

carried_state.py showed from-scratch deep-state coverage floors AND hurts in-distribution — but that conflates
two things: maybe burn-in just prevents the circuit from *forming*. Buitrago Ruiz & Gu (2025, arXiv:2507.02782)
apply state coverage as a cheap POST-training intervention on an ALREADY-trained model (~500 steps). This tests
that faithfully: train a clean in-distribution circuit first (short {4,8,16}, mixed-K, the model that solves
L16), THEN briefly post-train it with deep-state exposure (unlabeled burn-in B in {0..192}) at low LR, and see
whether L64/128/256 lift off the floor while L16 is preserved.

  base : trained short {4,8,16} only (the in-distribution circuit; = carried_state short_only, floors past L32)
  post : base + POST_STEPS of burn-in post-training at low LR (deep-state coverage on the formed circuit)

Outcomes: post lifts L64+ off floor (keeping L16) -> coverage works once the circuit exists (softens §3.1's
"no length-general path"); post stays at floor / degrades L16 -> coverage does not help non-abelian (strengthens
§3.1). Input-level burn-in is used as the state-coverage proxy (the fla Cache API makes true noise-state-init
surgery awkward, and the gdp_hybrid attention layer complicates state-passing); this is the tractable faithful
test of the post-training hypothesis.

  .venv/bin/python followups/non-abelian-state/post_state.py
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
from supervision_sweep import build as build_eval
from carried_state import build_burnin, KS, LWIN, BURN, SHORT_LEN
from length_mix import train                              # base model: gdp_hybrid d384x6, lr1e-3, 6000 steps

SEEDS = [0, 1, 2]
POST_STEPS = 1500
POST_LR = 3e-4
EVAL_LEN = [16, 32, 64, 128, 256]
N_POOL, N_EVAL = 8000, 150
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "post_state.md")


def make_pool(arm: str, w: Any, r: Any, origins: dict, oracle: Any, seed: int) -> list[str]:
    """Build a training pool of rendered documents for one arm.

    Args:
        arm: ``"short"`` (no burn-in; the in-distribution circuit) or anything else
            (``"burnin"``: an unlabeled deep-state-coverage prefix before a short labeled window).
        w: the FactWorld ``World``.
        r: the ``Renderer``.
        origins: the fixed agent -> a0-value map (parametric recall).
        oracle: the symbolic ``Oracle``.
        seed: RNG seed for sampling lengths, densities, and chains.

    Returns:
        A list of ``N_POOL`` whitespace-joined training strings.
    """
    rng = random.Random(seed)
    pool = []
    for _ in range(N_POOL):
        K = rng.choice(KS)
        if arm == "short":
            pool.append(" ".join(build_burnin(w, r, origins, oracle, 0, rng.choice(SHORT_LEN), K, rng)))
        else:                                              # burnin (deep-state coverage)
            pool.append(" ".join(build_burnin(w, r, origins, oracle, rng.choice(BURN), LWIN, K, rng)))
    return pool


def post_train(model: Any, tok: Any, pool: list[str], steps: int, lr: float, seed: int,
               device: str = "cuda") -> Any:
    """Continue training an existing model (low-LR, own cosine) — the post-training coverage phase.

    Args:
        model: a built, already-trained ``HybridLM`` (mutated in place).
        tok: the atomic tokenizer.
        pool: training strings (the burn-in coverage pool).
        steps: number of post-training steps.
        lr: peak learning rate for the post phase.
        seed: RNG seed for minibatch sampling.
        device: torch device.

    Returns:
        The same ``model``, post-trained in place.
    """
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup, pad = max(1, steps // 8), tok.pad_id
    rng = random.Random(9000 + seed)
    model.train()
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr * (min(1.0, (step + 1) / warmup) if step < warmup else
                             0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
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


def _eval(model: Any, tok: Any, w: Any, evs: dict) -> dict:
    """Internalized answer-only end-to-end value accuracy at each length in ``EVAL_LEN``.

    Args:
        model: the model to evaluate.
        tok: the atomic tokenizer.
        w: the FactWorld ``World``.
        evs: maps length -> a list of pre-built eval examples.

    Returns:
        A dict ``{length: value_accuracy}`` (NaN for any length whose eval raises).
    """
    out = {}
    for L in EVAL_LEN:
        try:
            _h, ev = e2e_eval(model, tok, w, evs[L])
        except Exception as exc:                           # pragma: no cover
            print(f"    eval error L{L}: {exc}", flush=True); ev = float("nan")
        out[L] = ev
    return out


def main() -> None:
    """Train a clean base per seed, post-train it with deep-state coverage, and tabulate base vs post to L256."""
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    short_pool = make_pool("short", w, r, origins, oracle, seed=2)
    burn_pool = make_pool("burnin", w, r, origins, oracle, seed=3)
    tok, _, _ = T.prepare(short_pool + burn_pool[:2000], [], [w])     # shared vocab (burnin superset)
    evs = {L: [build_eval(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(N_EVAL)]
           for L in EVAL_LEN}
    agg = defaultdict(lambda: defaultdict(dict))
    print("=== POST-TRAINING STATE COVERAGE: does it extend the circuit (when from-scratch didn't)? ===",
          flush=True)
    for s in SEEDS:
        model = train(tok, short_pool, s)                  # base: in-distribution circuit
        base = _eval(model, tok, w, evs)
        for L in EVAL_LEN:
            agg["base"][L][s] = base[L]
        print(f"  base s{s} :: " + "  ".join(f"L{L}={base[L]:.2f}" for L in EVAL_LEN), flush=True)
        post_train(model, tok, burn_pool, POST_STEPS, POST_LR, s)     # post-training coverage
        post = _eval(model, tok, w, evs)
        for L in EVAL_LEN:
            agg["post"][L][s] = post[L]
        print(f"  post s{s} :: " + "  ".join(f"L{L}={post[L]:.2f}" for L in EVAL_LEN), flush=True)
        del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("post_state done.", flush=True)


def write_md(agg: dict) -> None:
    """Write the base-vs-post accuracy table (mean ± pstdev over seeds) to ``OUT``."""
    lines = [
        "# Post-training state coverage — does it extend the circuit when from-scratch didn't? "
        "(`post_state.py`, 18.5M, 3 seeds)\n",
        "gdp_hybrid d384x6. `base` = trained short {4,8,16} mixed-K (the in-distribution circuit). `post` = base "
        f"+ {POST_STEPS} steps of burn-in post-training (unlabeled deep-state coverage, B in {{0..192}}) at lr "
        f"{POST_LR}. Eval internalized (answer-only) to L256. Floor = 0.20. Contrast: carried_state from-scratch "
        "burn-in floored AND hurt in-distribution.\n",
        "| arm | " + " | ".join(f"L{L}" for L in EVAL_LEN) + " |",
        "|---|" + "---|" * len(EVAL_LEN),
    ]
    for arm in ("base", "post"):
        if arm not in agg:
            continue
        cells = []
        for L in EVAL_LEN:
            xs = [v for v in agg[arm][L].values() if v == v]
            cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "…")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
