"""FOLLOW-ON (post-publication, segregated): where does the PARAMETRIC composite break?

The shipped paper reports one dismissive line: with parametric (in-weights) facts the binding x
recall composite floors for every architecture and no knob cracks it. This script localizes the
break with a 4-rung ladder, on the one architecture that *can* compose in-context (gdp_hybrid).

"Parametric" = the agent->a0 map (`origins`) is FIXED across all examples (so it is memorizable into
the weights) AND the facts are NOT rendered into the prompt. The only path to the value is the weight
memory. Floor = 1/k = 0.20 (guess among the k memorized values).

  R0  literal key, no state      "what is a0 of {agent} ? : {value} ."
  R1  abelian binding, NO CoT    "{gives} ... holder of {obj} ? : {value} ."     (latent dereference)
  R2  abelian binding, CoT       "{gives} ... holder of {obj} ? : {holder} {value} ."
  R3a non-abelian state, NO CoT  "{swap/cycle} ... holder of role {r} ? : {value} ."
  R3b non-abelian state, CoT     "{swap/cycle} ... holder of role {r} ? : {holder} {value} ."

Contrasts:  R0->R1 dereference of a computed pointer;  R1->R2 does verbalizing the bridge fix it
(latency vs capability);  R2->R3 cost of non-abelian state depth;  R3a->R3b CoT under full state.

Run on the 3090:  .venv/bin/python followups/non-abelian-state/ladder.py
"""
import os
import random
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld.world import Event  # noqa: E402

ARCH = "gdp_hybrid"
SEEDS = [0, 1, 2]
TRAIN_LEN = (4, 8, 16)
EVAL_LEN = [16, 64]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ladder.md")


def _world():
    from factworld.config import WorldConfig
    from factworld.render import Renderer
    from factworld.world import World
    wc = WorldConfig(seed=0, n_entities=8, n_attributes=2, value_vocab_size=64,
                     n_objects=8, n_locations=6, k=5)
    w = World(wc)
    r = Renderer()
    origins = dict(zip(w.agents, random.Random(0).sample(list(w.value_vocab), wc.k)))  # FIXED map
    return w, r, origins


# --------------------------------------------------------------------------- data per rung
def _gives(w, rng, L):
    ev = [Event("give", (rng.choice(w.objects), rng.choice(w.agents))) for _ in range(L)]
    obj = rng.choice(sorted({e.args[0] for e in ev}))
    holder = None
    for e in ev:
        if e.args[0] == obj:
            holder = e.args[1]
    return ev, obj, holder


def make_docs(rung, w, r, origins, oracle, n, seed):
    """Training documents (no facts rendered -> recall must come from weights)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        L = rng.choice(TRAIN_LEN)
        if rung == "R0":
            ag = rng.choice(w.agents)
            out.append(f"what is a0 of {ag} ? : {origins[ag]} .")
        elif rung in ("R1", "R2"):
            ev, obj, holder = _gives(w, rng, L)
            hist = " ".join(r.render_history(tuple(ev), with_steps=True))
            cot = f"{holder} " if rung == "R2" else ""
            out.append(f"{hist} what is a0 of the holder of {obj} ? : {cot}{origins[holder]} .")
        else:  # R3a / R3b -- non-abelian state, oracle gold
            ev = w.sample_hard_chain(L, episode_seed=f"{seed}|{i}")
            assignment = oracle.hard_assignment(ev)
            inv = {role: ag for ag, role in assignment.items()}
            role = rng.choice(w.roles)
            holder = inv[role]
            hist = " ".join(r.render_history(tuple(ev), with_steps=True))
            cot = f"{holder} " if rung == "R3b" else ""
            out.append(f"{hist} what is a0 of the holder of role {role} ? : {cot}{origins[holder]} .")
    return out


def make_eval(rung, w, r, origins, oracle, n, seed, L):
    """(prompt, holder, value, L) eval tuples; prompt ends ' : ' (model generates the answer)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if rung == "R0":
            ag = rng.choice(w.agents)
            out.append((f"what is a0 of {ag} ? : ", ag, origins[ag], 0))
        elif rung in ("R1", "R2"):
            ev, obj, holder = _gives(w, rng, L)
            hist = " ".join(r.render_history(tuple(ev), with_steps=True))
            out.append((f"{hist} what is a0 of the holder of {obj} ? : ", holder, origins[holder], L))
        else:
            ev = w.sample_hard_chain(L, episode_seed=f"e{seed}|{i}")
            inv = {role: ag for ag, role in oracle.hard_assignment(ev).items()}
            role = rng.choice(w.roles)
            holder = inv[role]
            hist = " ".join(r.render_history(tuple(ev), with_steps=True))
            out.append((f"{hist} what is a0 of the holder of role {role} ? : ", holder, origins[holder], L))
    return out


# --------------------------------------------------------------------------- eval
def value_eval(model, tok, w, exs, device="cuda", max_new=8):
    """No-CoT: first emitted value token must equal the (parametric) gold value."""
    import torch
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    dot = tok.token_to_id["."]
    model.eval()
    correct = 0
    with torch.no_grad():
        for prompt, _holder, value, _L in exs:
            ids = tok.encode(prompt)
            got = None
            for _ in range(max_new):
                with torch.autocast(device, dtype=torch.bfloat16):
                    nxt = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                if nxt in val_ids and got is None:
                    got = nxt
                ids.append(nxt)
                if nxt == dot:
                    break
            correct += int(got == tok.token_to_id[value])
    return correct / len(exs)


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from iso import strict_eval  # CoT rungs: holder AND value must match

    w, r, origins = _world()
    oracle = Oracle(w)
    RUNGS = ["R0", "R1", "R2", "R3a", "R3b"]
    is_cot = {"R0": False, "R1": False, "R2": True, "R3a": False, "R3b": True}

    res = defaultdict(lambda: defaultdict(list))
    for rung in RUNGS:
        lens = [16] if rung == "R0" else EVAL_LEN
        train_docs = make_docs(rung, w, r, origins, oracle, 8000, 2)
        evs = {L: make_eval(rung, w, r, origins, oracle, 200, 200 + L, L) for L in lens}
        tok, docs, _ = T.prepare(train_docs, [], [w])
        for s in SEEDS:
            run = T.run(ARCH, tok, docs, [], steps=4000, batch=32, d_model=256, n_layers=4,
                        d_ff=1024, seed=s, return_model=True)
            efn = strict_eval if is_cot[rung] else value_eval
            for L in lens:
                res[rung][L].append(efn(run["model"], tok, w, evs[L]))
            del run["model"]; torch.cuda.empty_cache()
            print(f"  {rung} s{s} :: " + "  ".join(f"L{L}={res[rung][L][-1]:.2f}" for L in lens), flush=True)
        write_md(res)
    write_md(res)
    print("ladder done.", flush=True)


def write_md(res):
    lines = [
        "# Parametric-composite localization ladder (follow-on, segregated)\n",
        "`followups/non-abelian-state/ladder.py`. gdp_hybrid, d256x4 (~6M), 4000 steps, 3 seeds. "
        "Parametric recall = fixed agent->a0 map, facts NOT in prompt. Floor = 1/k = 0.20. "
        "CoT rungs (R2/R3b) use strict eval (holder AND value); no-CoT rungs (R0/R1/R3a) score the "
        "emitted value only.\n",
        "| rung | what it adds | metric | "
        + " | ".join(f"L{L}" for L in EVAL_LEN) + " |",
        "|---|---|---|" + "---|" * len(EVAL_LEN),
    ]
    desc = {
        "R0": ("literal key, no state", "value"),
        "R1": ("abelian binding, NO CoT (latent deref)", "value"),
        "R2": ("abelian binding, CoT (verbalized bridge)", "holder+value"),
        "R3a": ("non-abelian state, NO CoT", "value"),
        "R3b": ("non-abelian state, CoT", "holder+value"),
    }
    for rung in ["R0", "R1", "R2", "R3a", "R3b"]:
        if rung not in res:
            continue
        what, metric = desc[rung]
        cells = []
        for L in EVAL_LEN:
            xs = res[rung].get(16) if rung == "R0" and L == EVAL_LEN[0] else res[rung].get(L) if rung != "R0" else None
            if not xs:
                cells.append("—" if rung == "R0" else "…")
            else:
                cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}"
                             f"({sum(x > 0.5 for x in xs)}/{len(xs)})")
        lines.append(f"| {rung} | {what} | {metric} | " + " | ".join(cells) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
