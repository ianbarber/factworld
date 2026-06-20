"""Candidate fix #2: coarse-grained re-anchoring with a compressed FULL-STATE summary.

The agentic "long session = composed short sub-sessions" idea, realized in tokens: every C events the model
emits the full k-role assignment (a bounded, length-INDEPENDENT state digest: k tokens) and re-anchors. Each
inter-summary segment is then a fixed C-step tracking sub-problem seeded by the previous summary — so if the
model learns the C-step sub-problem, re-anchoring should bound drift and let it run to ARBITRARY length. This
is the paper's "coarse-grained recurrence" (Mozer et al. 2604.17121, sec 5.3) / our compressed-state-summary.

Conditions (18.5M, 3 seeds):
  full_C8    full k-state summary every 8 events
  full_C16   full k-state summary every 16 events
  single_C8  CONTROL: only the queried role's holder every 8 (~= sweep K=8, which floored) — isolates whether
             FULL-state re-anchoring is what helps vs a single periodic checkpoint.
Train lengths {16,24,32}; eval the composite (parametric recall of the resolved holder), free-running with the
model emitting its OWN summaries, at L64 and L128 (2x-4x train). Floor = 0.20. Beating it at L128 = the fix works.

  .venv/bin/python followups/parametric-recall/coarse.py
"""
import math
import os
import random
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dense_capstone import _world

SEEDS = [0, 1, 2]
STEPS = 6000
TRAIN_LEN = (16, 24, 32)
EVAL_LEN = [64, 128]
D_MODEL, N_LAYERS, D_FF = 384, 6, 1536
CONDS = [("full_C8", 8, True), ("full_C16", 16, True), ("single_C8", 8, False)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coarse.md")


def make_builder(oracle):
    def build(w, r, origins, L, C, full, rng):
        ev = w.sample_hard_chain(L, episode_seed=f"{C}{full}|{rng.random()}")
        trace = oracle.hard_trace(ev)
        role = rng.choice(w.roles)
        hist = r.render_history(tuple(ev), with_steps=True)
        words, spans = ["role", role, "."], []
        for i, h in enumerate(hist):
            words += h.split()
            if (i + 1) % C == 0 and i < L - 1:                 # periodic state summary (not the final step)
                inv = {ro: ag for ag, ro in trace[i + 1].items()}
                words.append("roles")
                start = len(words)
                if full:
                    for ro in w.roles:
                        words.append(inv[ro])                  # full k-state digest, role order r0..r_{k-1}
                else:
                    words.append(inv[role])                    # control: only the queried role
                spans.append((start, len(words)))
                words.append(".")
        final_holder = {ro: ag for ag, ro in trace[L].items()}[role]
        words += ["what", "is", "a0", "?", ":"]
        vidx = len(words)
        words.append(origins[final_holder]); words.append(".")
        return words, spans, vidx, origins[final_holder]
    return build


def coarse_eval(model, tok, w, exs, device="cuda", max_tok=8):
    """Free-running: events forced; the model GENERATES its own summaries + final value. Score final value."""
    import torch
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    dot = tok.token_to_id["."]
    model.eval()
    correct = 0
    with torch.no_grad():
        for words, spans, vidx, gold in exs:
            starts = {s: e for s, e in spans}
            ids, i, got = [], 0, None
            while i < len(words):
                if i == vidx:
                    for _ in range(max_tok):
                        with torch.autocast(device, dtype=torch.bfloat16):
                            nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                        ids.append(nx)
                        if got is None and nx in val_ids: got = nx
                        if nx == dot: break
                    break
                elif i in starts:
                    for _ in range(max_tok):
                        with torch.autocast(device, dtype=torch.bfloat16):
                            nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                        ids.append(nx)
                        if nx == dot: break
                    i = starts[i]                              # jump past the true summary tokens ...
                    while i < len(words) and words[i] != ".": i += 1
                    i += 1                                     # ... and its '.'
                else:
                    ids += tok.encode(words[i]); i += 1
            correct += int(got == tok.token_to_id[gold])
    return correct / len(exs)


def train(tok, pool, seed, device="cuda"):
    import torch
    import torch.nn.functional as F
    from factworld.models import build_model
    torch.manual_seed(seed)
    model = build_model("gdp_hybrid", tok.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=4,
                        d_ff=D_FF, num_householder=4, allow_neg_eigval=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warmup, pad = 1000, tok.pad_id
    rng = random.Random(7000 + seed)
    model.train()
    for step in range(STEPS):
        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * (min(1.0, (step + 1) / warmup) if step < warmup else
                               0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, STEPS - warmup))))
        seqs = [tok.encode(d) for d in rng.sample(pool, 32)]
        ml = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), ml), pad, dtype=torch.long, device=device)
        for ri, s in enumerate(seqs):
            inp[ri, : len(s)] = torch.tensor(s, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(inp[:, :-1]); tgt = inp[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, tok.vocab_size), tgt.reshape(-1), reduction="none")
            mask = (tgt != pad).float().reshape(-1)
            loss = (ce * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    build = make_builder(oracle)
    tok, _, _ = T.prepare([" ".join(build(w, r, origins, 16, 8, True, random.Random(1))[0])], [], [w])
    agg = defaultdict(lambda: defaultdict(dict))
    print("=== COARSE RE-ANCHORING: full-state digest every C; composite eval at L64/L128 ===", flush=True)
    for name, C, full in CONDS:
        prng = random.Random(2)
        pool = [" ".join(build(w, r, origins, prng.choice(TRAIN_LEN), C, full, prng)[0]) for _ in range(8000)]
        evs = {L: [build(w, r, origins, L, C, full, random.Random(900 + L + j)) for j in range(150)]
               for L in EVAL_LEN}
        for s in SEEDS:
            model = train(tok, pool, s)
            for L in EVAL_LEN:
                agg[name][L][s] = coarse_eval(model, tok, w, evs[L])
            print(f"  {name:<10} s{s} :: L64={agg[name][64][s]:.3f} L128={agg[name][128][s]:.3f}", flush=True)
            del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("coarse done.", flush=True)


def write_md(agg):
    lines = [
        "# Coarse re-anchoring — does a periodic compressed full-state summary unlock length-robust tracking?\n",
        "`followups/parametric-recall/coarse.py`. gdp_hybrid d384x6 (18.5M), 3 seeds, parametric. Model emits a "
        "bounded k-token full-state digest every C events and re-anchors. Train lengths {16,24,32}; free-running "
        "composite eval (model emits its own summaries) at L64/L128. `single_C8` control = only the queried "
        "role's holder every 8 (~= sweep K=8, floored). Beating floor (0.20) at L128 = re-anchoring works.\n",
        "| condition | L64 | L128 |",
        "|---|---|---|",
    ]
    for name, _C, _f in CONDS:
        cells = []
        for L in EVAL_LEN:
            d = agg.get(name, {}).get(L)
            if not d:
                cells.append("…"); continue
            xs = list(d.values())
            cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}({sum(x>0.5 for x in xs)}/{len(xs)})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
