"""The decisive control: is the GatedDeltaProduct composite win MECHANISM or CAPACITY?

Vary num_householder ∈ {1,2,4} at a FIXED parameter budget (d_model=256 held constant; d_ff traded
against the Householder cost so all configs ≈ 5.68M params) + a negative-eigenvalue ablation. Same
gdp code path throughout (n_h=1 is single-delta / no product = the matched-capacity 'gdn-like' point).
5 seeds. STRICT eval: the generated CoT must get BOTH the resolved holder AND the recalled value
(addresses the lenient last-value-token metric). Reports mean±std and p(success)=#seeds with
strict-acc>0.5.

If n_h=4 > n_h=2 > n_h=1 at matched params -> the Householder-product STRUCTURE helps per-param
(mechanism). If flat -> the earlier win was capacity. floor = 1/k = 0.20.

Run on the 3090:  .venv/bin/python scripts/iso.py
"""
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld.world import Event  # noqa: E402

# d_ff chosen so all four are ~5.68M params at d_model=256 (see param-budget search)
CONFIGS = [
    ("n_h=1", dict(d_ff=1408, num_householder=1, allow_neg_eigval=True)),
    ("n_h=2", dict(d_ff=1280, num_householder=2, allow_neg_eigval=True)),
    ("n_h=4", dict(d_ff=1024, num_householder=4, allow_neg_eigval=True)),
    ("n_h=4 noNeg", dict(d_ff=1024, num_householder=4, allow_neg_eigval=False)),
]
SEEDS = [0, 1, 2, 3, 4]
LENS = [16, 64, 128]
TRAIN_LEN = (4, 8, 16)


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


def examples(w, r, origins, lengths, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        L = rng.choice(lengths)
        ev = [Event("give", (rng.choice(w.objects), rng.choice(w.agents))) for _ in range(L)]
        obj = rng.choice(sorted({e.args[0] for e in ev}))
        holder = None
        for e in ev:
            if e.args[0] == obj:
                holder = e.args[1]
        g = origins[holder]
        facts = " ".join(r.render_fact(a, "a0", origins[a], key=str(a)) for a in w.agents)
        hist = " ".join(r.render_history(tuple(ev), with_steps=True))
        q = f"what is a0 of the holder of {obj} ?"
        out.append((f"{facts} {hist} {q} : ", holder, g, L))
    return out


def strict_eval(model, tok, w, exs, device="cuda", max_new=8):
    import torch
    agent_ids = {tok.token_to_id[a] for a in w.agents}
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    dot = tok.token_to_id["."]
    model.eval()
    correct = 0
    with torch.no_grad():
        for prompt, holder, value, _L in exs:
            ids = tok.encode(prompt)
            first_agent = the_val = None
            for _ in range(max_new):
                with torch.autocast(device, dtype=torch.bfloat16):
                    nxt = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                if nxt in agent_ids and first_agent is None:
                    first_agent = nxt
                if nxt in val_ids:
                    the_val = nxt
                ids.append(nxt)
                if nxt == dot:
                    break
            correct += int(first_agent == tok.token_to_id[holder] and the_val == tok.token_to_id[value])
    return correct / len(exs)


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    w, r, origins = _world()
    ev = {L: examples(w, r, origins, (L,), 200, 200 + L) for L in LENS}
    # build training docs (history + query + CoT 'holder value .')
    rng = random.Random(2)
    train_docs = []
    for _ in range(8000):
        L = rng.choice(TRAIN_LEN)
        events = [Event("give", (rng.choice(w.objects), rng.choice(w.agents))) for _ in range(L)]
        obj = rng.choice(sorted({e.args[0] for e in events}))
        holder = None
        for e in events:
            if e.args[0] == obj:
                holder = e.args[1]
        facts = " ".join(r.render_fact(a, "a0", origins[a], key=str(a)) for a in w.agents)
        hist = " ".join(r.render_history(tuple(events), with_steps=True))
        train_docs.append(f"{facts} {hist} what is a0 of the holder of {obj} ? : {holder} {origins[holder]} .")
    tok, docs, _ = T.prepare(train_docs, [], [w])

    res = defaultdict(lambda: defaultdict(list))
    for name, kw in CONFIGS:
        for s in SEEDS:
            run = T.run("gdp_hybrid", tok, docs, [], steps=4000, batch=32, d_model=256, n_layers=4,
                        seed=s, return_model=True, **kw)
            for L in LENS:
                res[name][L].append(strict_eval(run["model"], tok, w, ev[L]))
            del run["model"]
            torch.cuda.empty_cache()
            print(f"  {name} s{s}", flush=True)

    print(f"\niso-param Householder ablation (gdp, d=256, ~5.68M, k=5, STRICT eval, floor {1/5:.2f})", flush=True)
    print("  mean±std across 5 seeds  (p_success = #seeds with strict-acc>0.5)", flush=True)
    for name, _kw in CONFIGS:
        cells = []
        for L in LENS:
            xs = res[name][L]
            cells.append(f"L{L}:{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}({sum(x > 0.5 for x in xs)}/5)")
        print(f"  {name:<12} " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
