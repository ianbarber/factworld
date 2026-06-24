"""Carried-state / state-distribution coverage: can we form a LENGTH-GENERAL non-abelian circuit?

decay_curve + reeval_endquery isolated the failure: non-abelian has no read-then-lookup escape, so it must
be solved by ONLINE state-carry, and that path cliffs past ~1.5xLmax because the recurrent state-distribution
beyond training is never visited (the diagnosis of arXiv:2507.02782, "Understanding and Improving Length
Generalization in Recurrent Models").

True chunked state-passing is awkward here (gdp_hybrid has an attention layer that can't carry state across
chunks). So we test the SAME mechanism a different way — STATE-DISTRIBUTION COVERAGE via an unlabeled burn-in:
prepend B random (UNLABELED) events before a short labeled window, so during training the recurrent state is
driven to depths [0 .. B+Lwin] while loss sits only on the window. Full BPTT carries gradient back through the
burn-in, so the per-step update gets signal at deep states WITHOUT any labels at length. This is a stronger
claim than length_mix (which labelled long examples): does mere state-exposure-at-depth suffice?

Front-query online-carry task (the one that cliffed in decay_curve). Two arms, eval internalized (answer-only,
no burn-in) to L256 — directly comparable to decay_curve's nonabelian_mixed:
  short_only  mixed-K labels, lengths {4,8,16}, no burn-in           (baseline -> expect cliff past ~24)
  burnin      mixed-K labels on a 16-event window + unlabeled burn-in B in {0,16,32,64,96,128,192}

If burnin lifts L64/128/256 off the floor where short_only cliffs, state-coverage is the lever and the
non-abelian circuit can be made length-general without labels at the target length.

  .venv/bin/python followups/non-abelian-state/carried_state.py
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

from dense_capstone import _world, e2e_eval
from supervision_sweep import build as build_eval        # build(w,r,origins,oracle,L,K,rng) -> answer-only eval
from length_mix import train                             # gdp_hybrid d384x6, lr1e-3, bs32, 6000 steps

SEEDS = [0, 1, 2]
KS = [1, 2, 4, 8, 10**9]                                 # mixed-density labels on the window
LWIN = 16
BURN = [0, 16, 32, 64, 96, 128, 192]                     # unlabeled burn-in depths (state coverage to ~208)
SHORT_LEN = (4, 8, 16)
EVAL_LEN = [16, 32, 64, 128, 256]
N_POOL, N_EVAL = 8000, 150
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carried_state.md")


def build_burnin(w, r, origins, oracle, B, Lwin, K, rng):
    """role r . [B unlabeled events] [Lwin events, holder labelled every K] what is a0 ? : value .
    Burn-in drives the recurrent state to depth B (no labels); the window supplies the state supervision."""
    ev = w.sample_hard_chain(B + Lwin, episode_seed=f"b{B}|{rng.random()}")
    trace = oracle.hard_trace(ev)                         # trace[i] = assignment after i events
    role = rng.choice(w.roles)
    hist = r.render_history(tuple(ev), with_steps=True)
    words = ["role", role, "."]
    for i, hstr in enumerate(hist):
        words += hstr.split()
        if i >= B and ((i - B + 1) % K == 0 or i == B + Lwin - 1):   # label only inside the window
            words.append("holder")
            inv = {ro: ag for ag, ro in trace[i + 1].items()}
            words.append(inv[role]); words.append(".")
    final_holder = {ro: ag for ag, ro in trace[B + Lwin].items()}[role]
    words += ["what", "is", "a0", "?", ":"]
    words.append(origins[final_holder]); words.append(".")
    return words


def make_pool(arm, w, r, origins, oracle, seed):
    rng = random.Random(seed)
    pool = []
    for _ in range(N_POOL):
        K = rng.choice(KS)
        if arm == "short_only":
            pool.append(" ".join(build_burnin(w, r, origins, oracle, 0, rng.choice(SHORT_LEN), K, rng)))
        else:                                              # burnin
            pool.append(" ".join(build_burnin(w, r, origins, oracle, rng.choice(BURN), LWIN, K, rng)))
    return pool


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    evs = {L: [build_eval(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(N_EVAL)]
           for L in EVAL_LEN}                              # clean answer-only eval, no burn-in
    agg = defaultdict(lambda: defaultdict(dict))
    print("=== CARRIED-STATE: does deep-state coverage form a length-general non-abelian circuit? ===",
          flush=True)
    for arm in ("short_only", "burnin"):
        pool = make_pool(arm, w, r, origins, oracle, seed=2)
        tok, _, _ = T.prepare(pool, [], [w])
        for s in SEEDS:
            model = train(tok, pool, s)
            for L in EVAL_LEN:
                try:
                    _h, ev = e2e_eval(model, tok, w, evs[L])
                except Exception as exc:                   # pragma: no cover
                    print(f"    {arm} s{s} L{L} eval error: {exc}", flush=True); ev = float("nan")
                agg[arm][L][s] = ev
            print(f"  {arm:<11} s{s} :: " + "  ".join(f"L{L}={agg[arm][L][s]:.2f}" for L in EVAL_LEN),
                  flush=True)
            del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("carried_state done.", flush=True)


def write_md(agg):
    lines = [
        "# Carried-state — does deep-state coverage form a length-general non-abelian circuit? "
        "(`carried_state.py`, 18.5M, 3 seeds)\n",
        "gdp_hybrid d384x6, front-query online-carry non-abelian task, mixed-density labels. `short_only` = "
        "lengths {4,8,16}, no burn-in (the decay_curve baseline). `burnin` = a 16-event labelled window plus an "
        "UNLABELED random burn-in B in {0,16,32,64,96,128,192} (state coverage to depth ~208, no labels at "
        "length). Eval internalized (answer-only, no burn-in) to L256. Floor = 0.20.\n",
        "| arm | " + " | ".join(f"L{L}" for L in EVAL_LEN) + " |",
        "|---|" + "---|" * len(EVAL_LEN),
    ]
    for arm in ("short_only", "burnin"):
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
