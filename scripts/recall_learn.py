"""Can genuine IN-CONTEXT COPY recall (random per-example map) be learned — and then composed?

diag_recall showed random-map recall floors at 4k steps with answer-only training. User call: keep
recall central, try to make copy learnable before reframing. Ladder:
  1. pure-copy (NO binding): "g3 's a0 is v17 ... what is a0 of g3 ? : v17 ." — can the arch form the
     induction/copy circuit at all? (transformer = all-attention is the easy case; gdp has 1 attn layer)
  2. composite + copy-curriculum: mix pure-copy docs into composite training so the copy circuit is
     bootstrapped, then require binding+copy. More steps (12k).
Random maps throughout (no memorization possible). If a setting lifts composite copy-recall off floor,
the in-context recall × state win is recoverable; if even pure-copy floors, copy is the bottleneck.

Run:  .venv/bin/python scripts/recall_learn.py
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

from iso import strict_eval  # noqa: E402  (composite: holder AND value)

SEEDS = [0, 1, 2]
NFACTS = 5
GIVE = (4, 8, 16)


def _world():
    from factworld.config import WorldConfig
    from factworld.render import Renderer
    from factworld.world import World
    wc = WorldConfig(seed=0, n_entities=8, n_attributes=2, value_vocab_size=64, n_objects=8, n_locations=6, k=5)
    return World(wc), Renderer()


def _rand_origins(w, rng):
    return dict(zip(w.agents, rng.sample(list(w.value_vocab), len(w.agents))))


def pure_copy(w, r, n, seed, with_cot):
    """Random map, NO binding: query names the agent directly; answer is the copied value."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        og = _rand_origins(w, rng)
        g = rng.choice(w.agents)
        facts = " ".join(r.render_fact(a, "a0", og[a], key=f"{a}|{rng.random()}") for a in w.agents)
        prompt = f"{facts} what is a0 of {g} ? : "
        out.append(f"{prompt}{og[g]} ." if with_cot else (prompt, og[g]))
    return out


def composite(w, r, n, seed, with_cot):
    """Random map, binding + copy: resolve holder then copy its value.

    Give-streams come from the uniform-last-write (v2) sampler in factworld.tasks — the old
    script-local copy drew every event's object uniformly, reproducing the retired v1 recency
    defect (the resolving write clustered near the stream end; see tasks.RETIRED / issue #11)."""
    from factworld.tasks import _uniform_last_write_stream
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        og = _rand_origins(w, rng)
        L = rng.choice(GIVE) if with_cot else n  # n overloaded as L for eval builder below
        obj, p, events = _uniform_last_write_stream(rng, L, list(w.objects), list(w.agents))
        holder = events[p].args[1]
        facts = " ".join(r.render_fact(a, "a0", og[a], key=f"{a}|{rng.random()}") for a in w.agents)
        hist = " ".join(r.render_history(tuple(events), with_steps=True))
        prompt = f"{facts} {hist} what is a0 of the holder of {obj} ? : "
        out.append(f"{prompt}{holder} {og[holder]} ." if with_cot else (prompt, holder, og[holder], L))
    return out


def composite_eval(w, r, L, n, seed):
    """Eval items on the same uniform-last-write (v2) give-stream sampler as composite()."""
    from factworld.tasks import _uniform_last_write_stream
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        og = _rand_origins(w, rng)
        obj, p, events = _uniform_last_write_stream(rng, L, list(w.objects), list(w.agents))
        holder = events[p].args[1]
        facts = " ".join(r.render_fact(a, "a0", og[a], key=f"{a}|{rng.random()}") for a in w.agents)
        hist = " ".join(r.render_history(tuple(events), with_steps=True))
        out.append((f"{facts} {hist} what is a0 of the holder of {obj} ? : ", holder, og[holder], L))
    return out


def value_eval(model, tok, w, exs, device="cuda", max_new=4):
    import torch
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    dot = tok.token_to_id["."]
    model.eval(); c = 0
    with torch.no_grad():
        for prompt, gold in exs:
            ids = tok.encode(prompt); pred = None
            for _ in range(max_new):
                with torch.autocast(device, dtype=torch.bfloat16):
                    nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                if nx in val_ids:
                    pred = nx
                ids.append(nx)
                if nx == dot:
                    break
            c += int(pred == tok.token_to_id[gold])
    return c / len(exs)


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    w, r = _world()
    out = {}

    # 1. pure-copy learnability (value-only eval)
    tr = pure_copy(w, r, 8000, 2, True)
    ev = pure_copy(w, r, 300, 99, False)
    tok, docs, _ = T.prepare(tr, [], [w])
    for arch in ("transformer", "gdp_hybrid"):
        accs = []
        for s in SEEDS:
            run = T.run(arch, tok, docs, [], steps=8000, batch=32, d_model=256, n_layers=4, d_ff=1024,
                        seed=s, return_model=True)
            accs.append(value_eval(run["model"], tok, w, ev)); del run["model"]; torch.cuda.empty_cache()
            print(f"  pure-copy {arch} s{s}", flush=True)
        out[f"pure-copy {arch}"] = {"copy": accs}

    # 2. composite + copy-curriculum (mix pure-copy docs in), strict eval
    cur = composite(w, r, 8000, 2, True) + pure_copy(w, r, 4000, 3, True)
    tok2, docs2, _ = T.prepare(cur, [], [w])
    evc = {L: composite_eval(w, r, L, 200, 300 + L) for L in (16, 64)}
    for arch in ("transformer", "gdp_hybrid"):
        per = defaultdict(list)
        for s in SEEDS:
            run = T.run(arch, tok2, docs2, [], steps=12000, batch=32, d_model=256, n_layers=4, d_ff=1024,
                        seed=s, return_model=True)
            for L in (16, 64):
                per[L].append(strict_eval(run["model"], tok2, w, evc[L]))
            del run["model"]; torch.cuda.empty_cache()
            print(f"  comp+curric {arch} s{s}", flush=True)
        out[f"comp+curric {arch}"] = per

    print("\nRECALL-LEARNABILITY (random map = genuine in-context copy; 3 seeds)", flush=True)
    for arch in ("transformer", "gdp_hybrid"):
        c = out[f"pure-copy {arch}"]["copy"]
        print(f"  pure-copy   {arch:<12} acc {statistics.mean(c):.2f}±{statistics.pstdev(c):.2f}  (floor 1/64≈.016)", flush=True)
    for arch in ("transformer", "gdp_hybrid"):
        p = out[f"comp+curric {arch}"]
        print(f"  comp+curric {arch:<12} " + "  ".join(
            f"L{L}:{statistics.mean(p[L]):.2f}±{statistics.pstdev(p[L]):.2f}" for L in (16, 64)), flush=True)


if __name__ == "__main__":
    main()
