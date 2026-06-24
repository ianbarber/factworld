"""Base-selection makes the post-training lever RELIABLE: select the base by free-running L16, then post.

post_reliability.py located the post-training-coverage fragility entirely on the BASE side (H1): between-base
L128 variance 0.087 vs within-base 0.0002 (~430x). A base whose free-running in-distribution accuracy is clean
(L16 e2e ~1.0) reliably posts to a length-general circuit (L128 0.85-0.91 across restarts); a base at L16<=0.86
reliably fails. The teacher-forced dense_h16 probe is saturated (1.0 everywhere, useless); FREE-RUNNING L16 e2e
is the predictor. Recipe: train K bases, SELECT max-L16, post-train it.

This validates the rule prospectively: train K=8 base seeds, rank by L16 e2e, post-train (1500 steps, lr 3e-4)
the TOP and BOTTOM bases (2 post-restarts each), eval to L256. Reports whether select-by-L16 reliably yields a
length-general circuit while the bottom base floors, and the base-L16 distribution -> how many seeds K to get an
L16>=0.95 base with high probability.

  .venv/bin/python followups/non-abelian-state/base_select.py
"""
from __future__ import annotations

import copy
import math
import os
import random
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dense_capstone import _world, e2e_eval
from supervision_sweep import build as build_ck
from post_state import post_train, make_pool
from length_mix import train

BASE_SEEDS = list(range(8))
POST_SEEDS = [0, 1]
EVAL_LEN = [16, 32, 64, 128, 256]
CLEAN_THR = 0.95          # base counts as "clean" if free-running L16 e2e >= this
SUCCESS_L, SUCCESS_THR = 128, 0.5
N_EVAL = 150
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_select.md")


def main() -> None:
    """Train K=8 bases, rank by free-running L16, post-train the top & bottom, and tabulate to L256."""
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    short_pool = make_pool("short", w, r, origins, oracle, seed=2)
    burn_pool = make_pool("burnin", w, r, origins, oracle, seed=3)
    tok, _, _ = T.prepare(short_pool + burn_pool[:2000], [], [w])
    evs = {L: [build_ck(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(N_EVAL)]
           for L in EVAL_LEN}

    print("=== BASE-SELECT: train K=8 bases, rank by free-running L16, post top & bottom ===", flush=True)
    bases = []
    for s in BASE_SEEDS:
        m = train(tok, short_pool, s)
        l16 = e2e_eval(m, tok, w, evs[16])[1]
        bases.append([s, l16, m])
        print(f"  base s{s} :: L16={l16:.2f}", flush=True)
        write_md(bases, {})
    bases.sort(key=lambda t: -t[1])                       # rank by free-running L16

    posts = {}
    for label, idx in (("top", 0), ("bottom", len(bases) - 1)):
        s, l16, m = bases[idx]
        for ps in POST_SEEDS:
            mm = copy.deepcopy(m)
            post_train(mm, tok, burn_pool, 1500, 3e-4, ps)
            acc = {L: e2e_eval(mm, tok, w, evs[L])[1] for L in EVAL_LEN}
            posts[(label, s, ps)] = (l16, acc)
            print(f"  {label} (base s{s}, L16={l16:.2f}) post s{ps} :: "
                  + "  ".join(f"L{L}={acc[L]:.2f}" for L in EVAL_LEN)
                  + f"   (success={acc[SUCCESS_L] > SUCCESS_THR})", flush=True)
            del mm; torch.cuda.empty_cache()
        write_md(bases, posts)
    write_md(bases, posts)
    print("base_select done.", flush=True)


def write_md(bases: list, posts: dict) -> None:
    """Write the base-L16 distribution, recommended K, and top/bottom post-training table to ``OUT``."""
    ranked = sorted(([s, l16] for s, l16, *_ in bases), key=lambda t: -t[1])
    n_clean = sum(l16 >= CLEAN_THR for _, l16 in ranked)
    p_clean = n_clean / len(ranked) if ranked else 0.0
    k_rec = (math.ceil(math.log(0.05) / math.log(1 - p_clean)) if 0 < p_clean < 1
             else (1 if p_clean >= 1 else None))
    lines = [
        "# Base-select — reliability via free-running-L16 base selection (`base_select.py`, 18.5M)\n",
        "Train K=8 base seeds (short {4,8,16}); rank by FREE-RUNNING L16 e2e; post-train (1500 steps, lr 3e-4) "
        "the top & bottom, 2 restarts each; eval to L256. post_reliability established H1 (base-quality gates; "
        f"between/within L128 var 0.087/0.0002). `clean` = L16 >= {CLEAN_THR}. Floor = 0.20, success = "
        f"L{SUCCESS_L} > {SUCCESS_THR}.\n",
        f"**Base L16 distribution:** " + ", ".join(f"{l16:.2f}" for _, l16 in ranked)
        + f"  → {n_clean}/{len(ranked)} clean (L16≥{CLEAN_THR}), p_clean≈{p_clean:.2f}"
        + (f"; train K≈{k_rec} bases for a clean one at 95% confidence." if k_rec else "."),
        "",
    ]
    if posts:
        lines += ["| selected | base L16 | post | " + " | ".join(f"L{L}" for L in EVAL_LEN) + " | success |",
                  "|---|---|---|" + "---|" * (len(EVAL_LEN) + 1)]
        for (label, s, ps), (l16, acc) in sorted(posts.items()):
            ok = acc[SUCCESS_L] > SUCCESS_THR
            lines.append(f"| {label} (s{s}) | {l16:.2f} | s{ps} | "
                         + " | ".join(f"{acc[L]:.2f}" for L in EVAL_LEN) + f" | {'YES' if ok else 'no'} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
