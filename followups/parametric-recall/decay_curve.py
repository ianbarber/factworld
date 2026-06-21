"""Real circuit vs shortcut: the LENGTH-DECAY CURVE, abelian vs non-abelian.

Settles the mechanistic question — is the internalized circuit a true length-general automaton, or a
length-bounded shortcut (Liu et al. 2023, "transformers learn shortcuts to automata")? Train at a fixed short
envelope (lengths {4,8,16}, Lmax=16) and evaluate internalized (answer-only, no-scratchpad) accuracy at a fine
length grid out to 16x the trained max (256). The SHAPE decides:
  - flat to high multiples  -> a real, length-general circuit
  - graceful decay          -> a partial circuit
  - cliff near ~2*Lmax      -> a length-bounded shortcut

Three arms, identical arch (gdp_hybrid 18.5M) / recipe / eval, differing ONLY in the binding + supervision:
  abelian_native     give-events (last-write-wins), NO process supervision (the R1 condition)
  abelian_mixed      give-events,  mixed-density holder supervision (same recipe as non-abelian)
  nonabelian_mixed   swap/cycle (S5), mixed-density holder supervision (the internalized non-abelian circuit)

Prediction: abelian holds far above floor to high multiples (a real, if simple, register circuit); non-abelian
cliffs near 2*Lmax (a shortcut). Same e2e_eval / metric for all three -> apples-to-apples decay shapes.

  .venv/bin/python followups/parametric-recall/decay_curve.py
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

from factworld.world import Event
from dense_capstone import _world, e2e_eval
from supervision_sweep import build as build_nonabelian   # build(w,r,origins,oracle,L,K,rng)
from length_mix import train                              # gdp_hybrid d384x6, lr1e-3, bs32, 6000 steps

SEEDS = [0, 1, 2]
TRAIN_LEN = (4, 8, 16)
KS = [1, 2, 4, 8, 10**9]                                  # mixed-density supervision set
EVAL_LEN = [16, 24, 32, 48, 64, 96, 128, 192, 256]        # 1x .. 16x the trained max
N_POOL, N_EVAL = 8000, 150
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decay_curve.md")


def build_abelian(w, r, origins, oracle, L, K, rng):
    """Abelian analogue of supervision_sweep.build: give-events, holder = last writer of a fixed object,
    holder checkpoints every K (+ always final). Same (words, hidx, vidx) shape so e2e_eval is reused."""
    ev = [Event("give", (rng.choice(w.objects), rng.choice(w.agents))) for _ in range(L)]
    obj = ev[0].args[0]                                   # given at step 0 -> holder always defined
    hist = r.render_history(tuple(ev), with_steps=True)
    words, hidx, cur = ["object", str(obj), "."], [], None
    for i, hstr in enumerate(hist):
        words += hstr.split()
        if ev[i].args[0] == obj:                          # last-write-wins update
            cur = ev[i].args[1]
        if (i + 1) % K == 0 or i == L - 1:                # labelled checkpoint
            words.append("holder"); hidx.append(len(words)); words.append(cur); words.append(".")
    words += ["what", "is", "a0", "?", ":"]
    vidx = len(words); words.append(origins[cur]); words.append(".")
    return words, hidx, vidx


def make_pool(buildfn, w, r, origins, oracle, mixedK, seed):
    rng = random.Random(seed)
    pool = []
    for _ in range(N_POOL):
        K = rng.choice(KS) if mixedK else 10**9
        pool.append(" ".join(buildfn(w, r, origins, oracle, rng.choice(TRAIN_LEN), K, rng)[0]))
    return pool


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    arms = [
        ("abelian_native",   build_abelian,    False),
        ("abelian_mixed",    build_abelian,    True),
        ("nonabelian_mixed", build_nonabelian, True),
    ]
    agg = defaultdict(lambda: defaultdict(dict))
    print("=== DECAY CURVE: real circuit vs shortcut (train Lmax=16, eval to 16x = L256) ===", flush=True)
    for name, buildfn, mixedK in arms:
        pool = make_pool(buildfn, w, r, origins, oracle, mixedK, seed=2)
        tok, _, _ = T.prepare(pool, [], [w])
        evs = {L: [buildfn(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(N_EVAL)]
               for L in EVAL_LEN}                          # answer-only eval (no scratchpad)
        for s in SEEDS:
            model = train(tok, pool, s)
            for L in EVAL_LEN:
                try:
                    _h, ev = e2e_eval(model, tok, w, evs[L])
                except Exception as exc:                   # pragma: no cover - robustness at extreme length
                    print(f"    {name} s{s} L{L} eval error: {exc}", flush=True); ev = float("nan")
                agg[name][L][s] = ev
            print(f"  {name:<17} s{s} :: " + "  ".join(f"L{L}={agg[name][L][s]:.2f}" for L in EVAL_LEN),
                  flush=True)
            del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("decay_curve done.", flush=True)


def write_md(agg):
    lines = [
        "# Decay curve — real circuit vs shortcut (`decay_curve.py`, 18.5M, 3 seeds)\n",
        "gdp_hybrid d384x6, trained at the short envelope {4,8,16} (Lmax=16), evaluated internalized "
        "(answer-only, no scratchpad) at a fine length grid to 16x. Same e2e value metric for all arms. "
        "Floor = 0.20. Shape: flat = length-general circuit; cliff near ~2xLmax (32) = length-bounded "
        "shortcut.\n",
        "| arm | " + " | ".join(f"L{L}" for L in EVAL_LEN) + " |",
        "|---|" + "---|" * len(EVAL_LEN),
    ]
    for name in ("abelian_native", "abelian_mixed", "nonabelian_mixed"):
        if name not in agg:
            continue
        cells = []
        for L in EVAL_LEN:
            xs = [v for v in agg[name][L].values() if v == v]   # drop NaN
            cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "…")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
