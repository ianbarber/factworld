"""W5: how is the genuine composite learned — two independent legs, or a conditional resolve-then-recall
circuit? Train gdp_hybrid on the genuine composite (copy-curriculum), then decompose the strict eval into
the state leg (holder correct) and the recall leg (value correct), and — the discriminating control —
inspect what the model emits on holder-WRONG examples:

  route  = P(emitted value == the property of the *resolved* (wrong) holder | holder wrong)
  other  = P(emitted some other in-vocab value | holder wrong)
  none   = P(emitted no value token at all | holder wrong)

A high `route` means the model genuinely recalls the property of whatever agent it resolves — recall is
mechanistically routed through the binding (resolve-then-recall), NOT off-distribution garbage. That is the
token-level evidence that P(value|holder wrong)≈0 is composition (the model implements the task's causal
graph), not a model that simply breaks when the binding is lost. Reports per-seed; eval is free-running
(the model emits its own holder and value), so nothing is forced.

  .venv/bin/python scripts/decompose.py
"""
import os, random, statistics, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); sys.path.insert(0, REPO+"/scripts")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from recall_learn import _world, _rand_origins, pure_copy, composite

SEEDS = [0, 1, 2, 3, 4]


def composite_eval_og(w, r, L, n, seed):
    """Same distribution as recall_learn.composite_eval, but also returns the per-example origin map `og`."""
    from factworld.world import Event
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        og = _rand_origins(w, rng)
        events = [Event("give", (rng.choice(w.objects), rng.choice(w.agents))) for _ in range(L)]
        obj = rng.choice(sorted({e.args[0] for e in events}))
        holder = [e.args[1] for e in events if e.args[0] == obj][-1]
        facts = " ".join(r.render_fact(a, "a0", og[a], key=f"{a}|{rng.random()}") for a in w.agents)
        hist = " ".join(r.render_history(tuple(events), with_steps=True))
        out.append((f"{facts} {hist} what is a0 of the holder of {obj} ? : ", holder, og[holder], L, og))
    return out


def decompose_eval(model, tok, w, exs, device="cuda", max_new=6):
    import torch
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    ag_ids = {tok.token_to_id[g] for g in w.agents}
    id2ag = {tok.token_to_id[g]: g for g in w.agents}
    dot = tok.token_to_id["."]
    model.eval()
    h_ok = v_ok = both = vh = vnh = nh = 0
    route = other = none = 0   # breakdown on holder-WRONG examples
    with torch.no_grad():
        for prompt, holder, value, L, og in exs:
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
            if hc:
                vh += vc
            else:
                nh += 1; vnh += vc
                resolved = tok.token_to_id[og[id2ag[ph]]] if (ph is not None and ph in id2ag) else None
                if pv is None: none += 1
                elif resolved is not None and pv == resolved: route += 1
                else: other += 1
    n = len(exs); d = nh or 1
    return dict(h=h_ok / n, v=v_ok / n, both=both / n, vgh=vh / (h_ok or 1), vgw=vnh / d,
                route=route / d, other=other / d, none=none / d)


def main():
    import torch
    from factworld import train as T
    w, r = _world()
    cur = composite(w, r, 8000, 2, True) + pure_copy(w, r, 4000, 3, True)
    tok, docs, _ = T.prepare(cur, [], [w])
    evc = {L: composite_eval_og(w, r, L, 400, 300 + L) for L in (16, 64)}
    agg = {L: [] for L in (16, 64)}
    print("=== W5 DECOMPOSE (gdp_hybrid genuine composite; routing breakdown on holder-wrong) ===", flush=True)
    for s in SEEDS:
        run = T.run("gdp_hybrid", tok, docs, [], steps=12000, batch=32, d_model=256, n_layers=4,
                    d_ff=1024, seed=s, return_model=True)
        for L in (16, 64):
            m = decompose_eval(run["model"], tok, w, evc[L]); agg[L].append(m)
            print(f"  s{s} L{L:<3} :: holder={m['h']:.3f} value={m['v']:.3f} both={m['both']:.3f} | "
                  f"P(v|h_ok)={m['vgh']:.3f} P(v|h_bad)={m['vgw']:.3f} | "
                  f"[h-wrong] route={m['route']:.3f} other={m['other']:.3f} none={m['none']:.3f}", flush=True)
        del run["model"]; torch.cuda.empty_cache()
    print(f"\n=== MEANS over {len(SEEDS)} seeds ===", flush=True)
    for L in (16, 64):
        xs = agg[L]
        def ms(k): return statistics.mean(x[k] for x in xs), statistics.pstdev(x[k] for x in xs)
        conv = [x for x in xs if x["h"] > 0.3 and x["v"] > 0.3]
        print(f"  L{L:<3} holder={ms('h')[0]:.3f}±{ms('h')[1]:.3f} value={ms('v')[0]:.3f}±{ms('v')[1]:.3f} "
              f"| P(v|h_ok)={ms('vgh')[0]:.3f} P(v|h_bad)={ms('vgw')[0]:.3f} "
              f"| [h-wrong means] route={ms('route')[0]:.3f} other={ms('other')[0]:.3f} none={ms('none')[0]:.3f} "
              f"| converged {len(conv)}/{len(xs)}", flush=True)
    print("decompose done.", flush=True)


if __name__ == "__main__":
    main()
