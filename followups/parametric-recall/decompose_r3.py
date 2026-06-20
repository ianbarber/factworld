"""R3 decomposition: when the non-abelian composite floors under PARAMETRIC recall, which leg breaks?

The ladder (ladder.py) localized the parametric break to R3 (non-abelian state + in-weights recall),
and showed CoT does not rescue it. But the strict score is a holder-AND-value conjunction, so floor is
ambiguous: (a) the STATE leg resolves the holder but parametric recall keyed by a state-computed pointer
fails, or (b) the state leg itself collapses once the in-context facts are removed.

Decisive control: identical world / chains / CoT, decomposed into holder-acc, value-acc, P(v|holder_ok)
and the routing breakdown on holder-wrong examples — run for two arms that differ ONLY in where the
agent->a0 fact lives:
    inctx  facts rendered in the prompt (in-context copy of a fixed, memorizable map)
    param  facts omitted (recall must come from the weights)
Same FIXED origins map in both, so the only varying axis is in-context vs parametric.

If holder-acc matches across arms and only value diverges -> genuine recall-leg wall (parametric recall
keyed by a non-abelian pointer fails). If holder also collapses in `param` -> state-leg/optimization story.

  .venv/bin/python followups/parametric-recall/decompose_r3.py
"""
import os
import random
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SEEDS = [0, 1, 2, 3, 4]
TRAIN_LEN = (4, 8, 16)
EVAL_LEN = [16, 64]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decompose_r3.md")


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


def _facts(w, r, origins):
    return " ".join(r.render_fact(a, "a0", origins[a], key=str(a)) for a in w.agents)


def make(arm, w, r, origins, oracle, n, seed, with_cot, lengths):
    """with_cot True -> training docs; False -> (prompt, holder, value, L) eval tuples. CoT = 'holder value'."""
    rng = random.Random(seed)
    facts = _facts(w, r, origins) + " " if arm == "inctx" else ""
    out = []
    for i in range(n):
        L = rng.choice(lengths)
        ev = w.sample_hard_chain(L, episode_seed=f"{arm}|{seed}|{i}")
        inv = {role: ag for ag, role in oracle.hard_assignment(ev).items()}
        role = rng.choice(w.roles)
        holder = inv[role]
        hist = " ".join(r.render_history(tuple(ev), with_steps=True))
        prompt = f"{facts}{hist} what is a0 of the holder of role {role} ? : "
        if with_cot:
            out.append(f"{prompt}{holder} {origins[holder]} .")
        else:
            out.append((prompt, holder, origins[holder], L))
    return out


def decompose_eval(model, tok, w, origins, exs, device="cuda", max_new=6):
    """Free-running; emit holder then value. Routing on holder-wrong uses the FIXED parametric map."""
    import torch
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    ag_ids = {tok.token_to_id[g] for g in w.agents}
    id2ag = {tok.token_to_id[g]: g for g in w.agents}
    dot = tok.token_to_id["."]
    model.eval()
    h_ok = v_ok = both = nh = 0
    route = other = none = 0
    with torch.no_grad():
        for prompt, holder, value, _L in exs:
            ids = tok.encode(prompt); ph = pv = None
            for _ in range(max_new):
                with torch.autocast(device, dtype=torch.bfloat16):
                    nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                if ph is None and nx in ag_ids: ph = nx
                if pv is None and nx in val_ids: pv = nx
                ids.append(nx)
                if nx == dot: break
            hc = (ph == tok.token_to_id[holder]); vc = (pv == tok.token_to_id[value])
            h_ok += hc; v_ok += vc; both += (hc and vc)
            if not hc:
                nh += 1
                resolved = tok.token_to_id[origins[id2ag[ph]]] if (ph in id2ag) else None
                if pv is None: none += 1
                elif resolved is not None and pv == resolved: route += 1
                else: other += 1
    n = len(exs); d = nh or 1
    return dict(h=h_ok / n, v=v_ok / n, both=both / n, vgh=both / (h_ok or 1),
                route=route / d, other=other / d, none=none / d)


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    agg = defaultdict(lambda: defaultdict(list))
    print("=== R3 decompose: in-context vs parametric, non-abelian composite (gdp_hybrid, CoT) ===", flush=True)
    for arm in ("inctx", "param"):
        train = make(arm, w, r, origins, oracle, 8000, 2, True, TRAIN_LEN)
        evs = {L: make(arm, w, r, origins, oracle, 300, 300 + L, False, (L,)) for L in EVAL_LEN}
        tok, docs, _ = T.prepare(train, [], [w])
        for s in SEEDS:
            run = T.run("gdp_hybrid", tok, docs, [], steps=4000, batch=32, d_model=256, n_layers=4,
                        d_ff=1024, seed=s, return_model=True)
            for L in EVAL_LEN:
                m = decompose_eval(run["model"], tok, w, origins, evs[L]); agg[arm][L].append(m)
                print(f"  {arm:<5} s{s} L{L:<3} :: holder={m['h']:.3f} value={m['v']:.3f} both={m['both']:.3f} "
                      f"| P(v|h_ok)={m['vgh']:.3f} | [h-wrong] route={m['route']:.3f} other={m['other']:.3f} "
                      f"none={m['none']:.3f}", flush=True)
            del run["model"]; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("decompose_r3 done.", flush=True)


def write_md(agg):
    lines = [
        "# R3 decomposition — in-context vs parametric (which leg breaks?)\n",
        "`followups/parametric-recall/decompose_r3.py`. gdp_hybrid d256x4, 4000 steps, 5 seeds, CoT. "
        "Same fixed map + same non-abelian chains in both arms; only difference is whether the agent->a0 "
        "facts are rendered (`inctx`) or omitted (`param`). Free-running eval (model emits its own holder "
        "then value). Floor = 0.20.\n",
        "| arm | L | holder | value | both | P(v\\|h_ok) | route | other | none | conv |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("inctx", "param"):
        for L in EVAL_LEN:
            xs = agg.get(arm, {}).get(L)
            if not xs:
                continue
            def ms(k):
                return statistics.mean(x[k] for x in xs)
            conv = sum(x["h"] > 0.3 and x["v"] > 0.3 for x in xs)
            lines.append(f"| {arm} | {L} | {ms('h'):.3f} | {ms('v'):.3f} | {ms('both'):.3f} | "
                         f"{ms('vgh'):.3f} | {ms('route'):.3f} | {ms('other'):.3f} | {ms('none'):.3f} | "
                         f"{conv}/{len(xs)} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
