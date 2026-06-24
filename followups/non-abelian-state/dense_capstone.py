"""CAPSTONE: dense per-step state supervision lifts the non-abelian composite — for PARAMETRIC recall too.

The ladder + decompose showed the composition gap is entirely the non-abelian STATE leg (holder floors at
chance under single-query / answer-only supervision); the recall leg is free (P(value|holder)=1.0, route=1.0)
and parametric-vs-in-context made zero difference. R3b also showed *final-only* CoT does not help — because it
supervises output order, not the intermediate permutation.

This is the affirmative test: interleave DENSE per-step holder supervision into the stream (port the dense_s5
lesson to the composite), and check the prediction:
  (1) the non-abelian state leg rises off the floor (vs R3b = 0.20), and
  (2) the PARAMETRIC composite succeeds in lockstep with the in-context one.

Doc:  [facts?]  role r .  <ev0> holder <h1> .  <ev1> holder <h2> . ...  what is a0 ? : <value> .
where h_{i+1} = holder of role r after applying events[:i+1] (from oracle.hard_trace). `holder` slots are the
dense state signal; the final value is the recall leg, keyed by the last emitted holder. Two arms differ ONLY
in whether the agent->a0 facts are rendered (inctx) or must come from the weights (param).

Metrics (5 seeds, gdp_hybrid):
  dense_h   per-step holder argmax accuracy (teacher-forced)         -- did the state leg learn?
  val|trace value argmax given the true trace teacher-forced         -- recall leg, correct-state-conditioned
  e2e_h/e2e_v  guided free-run: model generates holders + value      -- end-to-end composite

  .venv/bin/python followups/non-abelian-state/dense_capstone.py
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dense_capstone.md")


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


def build(arm, w, r, origins, oracle, L, rng):
    """Return (words, holder_word_idx[list], value_word_idx) for one dense example."""
    ev = w.sample_hard_chain(L, episode_seed=f"{arm}|{rng.random()}")
    trace = oracle.hard_trace(ev)                      # len L+1; trace[i] = assignment after i events
    role = rng.choice(w.roles)
    hist = r.render_history(tuple(ev), with_steps=True)  # list of per-event strings "sK : ... ."
    words = []
    if arm == "inctx":
        words += _facts(w, r, origins).split()
    words += ["role", role, "."]
    hidx = []
    for i, hstr in enumerate(hist):
        words += hstr.split()
        words.append("holder")
        inv = {ro: ag for ag, ro in trace[i + 1].items()}
        hidx.append(len(words)); words.append(inv[role])   # dense holder target after prefix i+1
        words.append(".")
    final_holder = {ro: ag for ag, ro in trace[L].items()}[role]
    words += ["what", "is", "a0", "?", ":"]
    vidx = len(words); words.append(origins[final_holder]); words.append(".")
    return words, hidx, vidx


def make_docs(arm, w, r, origins, oracle, n, seed):
    rng = random.Random(seed)
    return [" ".join(build(arm, w, r, origins, oracle, rng.choice(TRAIN_LEN), rng)[0]) for _ in range(n)]


def make_eval(arm, w, r, origins, oracle, n, seed, L):
    rng = random.Random(seed)
    return [build(arm, w, r, origins, oracle, L, rng) for _ in range(n)]


def dense_eval(model, tok, w, exs, device="cuda"):
    """Teacher-forced single forward: per-step holder accuracy + value-accuracy given the true trace."""
    import torch
    model.eval()
    h_hit = h_tot = v_hit = v_tot = 0
    off = None
    with torch.no_grad():
        for words, hidx, vidx in exs:
            ids = tok.encode(" ".join(words))
            if off is None:
                off = len(ids) - len(words)          # specials prefix (0 if none), constant
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(torch.tensor([ids], device=device))[0].float()
            for wi in hidx:
                p = wi + off
                h_hit += int(logits[p - 1].argmax().item() == ids[p]); h_tot += 1
            pv = vidx + off
            v_hit += int(logits[pv - 1].argmax().item() == ids[pv]); v_tot += 1
    return h_hit / max(1, h_tot), v_hit / max(1, v_tot)


def e2e_eval(model, tok, w, exs, device="cuda", max_slot=4):
    """Guided free-run: events forced, holders + final value GENERATED (model's own state). Score final
    holder and value."""
    import torch
    ag_ids = {tok.token_to_id[g] for g in w.agents}
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    dot = tok.token_to_id["."]
    model.eval()
    fh = fv = 0
    with torch.no_grad():
        for words, hidx, vidx in exs:
            # segment the word stream into forced runs and generated slots (holders + value)
            slots = set(hidx) | {vidx}
            ids, i, last_holder = [], 0, None
            true_final_holder = words[hidx[-1]]
            true_value = words[vidx]
            while i < len(words):
                if i in slots:
                    gen = None
                    for _ in range(max_slot):
                        with torch.autocast(device, dtype=torch.bfloat16):
                            nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                        ids.append(nx)
                        if gen is None and (nx in ag_ids if i in hidx else nx in val_ids):
                            gen = nx
                        if nx == dot:
                            break
                    if i in hidx:
                        last_holder = gen
                    else:
                        fv += int(gen == tok.token_to_id[true_value])
                    # skip the true slot word(s) up to and including its trailing '.'
                    i += 1
                    while i < len(words) and words[i] != ".":
                        i += 1
                    i += 1  # past the '.'
                else:
                    ids += tok.encode(words[i]); i += 1
            fh += int(last_holder == tok.token_to_id[true_final_holder])
    n = len(exs)
    return fh / n, fv / n


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    agg = defaultdict(lambda: defaultdict(dict))
    print("=== DENSE CAPSTONE: interleaved state supervision, in-context vs parametric ===", flush=True)
    for arm in ("inctx", "param"):
        train = make_docs(arm, w, r, origins, oracle, 8000, 2)
        evs = {L: make_eval(arm, w, r, origins, oracle, 200, 300 + L, L) for L in EVAL_LEN}
        tok, docs, _ = T.prepare(train, [], [w])
        for s in SEEDS:
            run = T.run("gdp_hybrid", tok, docs, [], steps=4000, batch=32, d_model=256, n_layers=4,
                        d_ff=1024, seed=s, return_model=True)
            for L in EVAL_LEN:
                dh, vt = dense_eval(run["model"], tok, w, evs[L])
                eh, ev = e2e_eval(run["model"], tok, w, evs[L])
                agg[arm][L][s] = dict(dense_h=dh, val_trace=vt, e2e_h=eh, e2e_v=ev)
                print(f"  {arm:<5} s{s} L{L:<3} :: dense_h={dh:.3f} val|trace={vt:.3f} | "
                      f"e2e_h={eh:.3f} e2e_v={ev:.3f}", flush=True)
            del run["model"]; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("dense_capstone done.", flush=True)


def write_md(agg):
    lines = [
        "# Dense-supervision capstone — does lifting the state leg rescue the PARAMETRIC composite?\n",
        "`followups/non-abelian-state/dense_capstone.py`. gdp_hybrid d256x4, 4000 steps, 5 seeds. "
        "Interleaved dense per-step holder supervision (oracle `hard_trace`) on the non-abelian composite. "
        "Baseline for contrast: R3b (final-only CoT) floored at 0.20 for both arms. "
        "`dense_h` per-step holder acc; `val|trace` value acc given true trace; `e2e_*` guided free-run.\n",
        "| arm | L | dense_h | val\\|trace | e2e_holder | e2e_value |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ("inctx", "param"):
        for L in EVAL_LEN:
            d = agg.get(arm, {}).get(L)
            if not d:
                continue
            xs = list(d.values())
            def ms(k):
                return statistics.mean(x[k] for x in xs), statistics.pstdev(x[k] for x in xs)
            cells = " | ".join(f"{ms(k)[0]:.2f}±{ms(k)[1]:.2f}"
                               for k in ("dense_h", "val_trace", "e2e_h", "e2e_v"))
            lines.append(f"| {arm} | {L} | {cells} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
