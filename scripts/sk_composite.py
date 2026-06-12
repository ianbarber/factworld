"""FIX #2: a REAL Sₖ composite — non-abelian state (swap/cycle) then recall, gold via the ORACLE.

The give-only composite is abelian (k = table size, not Sₖ group complexity) and its
gold bypasses the oracle. Here the binding stream is `swap_role`/`cycle_roles` over k=5 agents
holding k roles (identity start = the canonical S5 word problem), each agent carries an a0 fact, and
the query resolves a ROLE to its current agent then recalls that agent's a0:
    "what is a0 of the holder of role r2 ?"
The role→agent resolution is computed by `oracle.hard_assignment` (the hard-state solver), not an
inline loop, so the headline task is now under the instrument's correctness guarantee and exercises
genuine non-abelian state tracking. Length = #permutations (state-axis extrapolation, train ≤16 → 64).

STRICT eval (reuses iso.strict_eval): generated CoT must get BOTH the resolved agent AND its value.
Run on the 3090:  .venv/bin/python scripts/sk_composite.py
"""
import os
import random
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from iso import strict_eval  # noqa: E402  (generated agent-token AND value-token must match)

ARCHS = ["transformer", "gru", "gdn_hybrid", "gdp_hybrid"]
SEEDS = [0, 1, 2, 3, 4]
TRAIN_LEN = (4, 8, 16)
EVAL_LEN = [16, 32, 64]


def _world():
    from factworld.config import WorldConfig
    from factworld.render import Renderer
    from factworld.world import World
    wc = WorldConfig(seed=0, n_entities=8, n_attributes=2, value_vocab_size=64,
                     n_objects=8, n_locations=6, k=5)
    w = World(wc)
    r = Renderer()
    origins = dict(zip(w.agents, random.Random(0).sample(list(w.value_vocab), wc.k)))
    return w, r, origins


def examples(w, r, origins, oracle, lengths, n, seed, with_cot):
    """Return (doc) strings if with_cot else (prompt, gold_agent, gold_value, L) eval tuples."""
    rng = random.Random(seed)
    facts = " ".join(r.render_fact(a, "a0", origins[a], key=str(a)) for a in w.agents)
    out = []
    for i in range(n):
        L = rng.choice(lengths)
        events = w.sample_hard_chain(L, episode_seed=f"{seed}|{i}")
        assignment = oracle.hard_assignment(events)            # agent -> role, via the oracle
        inv = {role: ag for ag, role in assignment.items()}    # role -> agent (a permutation, bijective)
        role = rng.choice(w.roles)
        gold_agent = inv[role]
        gold_value = origins[gold_agent]
        hist = " ".join(r.render_history(tuple(events), with_steps=True))
        q = f"what is a0 of the holder of role {role} ?"
        prompt = f"{facts} {hist} {q} : "
        if with_cot:
            out.append(f"{prompt}{gold_agent} {gold_value} .")
        else:
            out.append((prompt, gold_agent, gold_value, L))
    return out


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    # sanity: oracle round-trips identity at L=0 and a known swap
    train_docs = examples(w, r, origins, oracle, TRAIN_LEN, 8000, 2, with_cot=True)
    ev = {L: examples(w, r, origins, oracle, (L,), 200, 200 + L, with_cot=False) for L in EVAL_LEN}
    tok, docs, _ = T.prepare(train_docs, [], [w])

    res = defaultdict(lambda: defaultdict(list))
    for a in ARCHS:
        for s in SEEDS:
            run = T.run(a, tok, docs, [], steps=4000, batch=32, d_model=256, n_layers=4, d_ff=1024,
                        seed=s, return_model=True)
            for L in EVAL_LEN:
                res[a][L].append(strict_eval(run["model"], tok, w, ev[L]))
            del run["model"]
            torch.cuda.empty_cache()
            print(f"  {a} s{s}", flush=True)

    print(f"\nReal Sₖ composite (S5, swap/cycle, oracle gold, STRICT eval, floor {1/5:.2f})", flush=True)
    print("  mean±std across 5 seeds  (#seeds converged, strict-acc>0.5)", flush=True)
    for a in ARCHS:
        cells = []
        for L in EVAL_LEN:
            xs = res[a][L]
            cells.append(f"L{L}:{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}({sum(x > 0.5 for x in xs)}/5)")
        print(f"  {a:<12} " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
