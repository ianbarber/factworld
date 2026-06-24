"""Construct-validity bridge: does the non-abelian cliff appear in CODE-EXECUTION-TRACE clothing?

The worry (and a reviewer's): the S5 word problem is abstract — does the non-abelian wall matter for real
agentic/code work? We re-render our SAME oracle-validated non-abelian dynamics as a variable-swap execution
trace in the surface grammar of Code World Model (CWM, Meta FAIR 2025, arXiv 2510.02387): variables hold
values, `swap`/`cycle` ops permute them, and the program's full state is interleaved after each op (CWM-style
observation). The query asks a variable's final value — which requires composing the swap/cycle sequence (it
cannot be last-write-wins shortcut, so it is genuinely non-abelian). We then re-run the supervision-density
sweep (snapshot every K ops): if the SAME cliff appears, the wall is a property of non-abelian composition, not
of the abstract role-permutation rendering — bridging "S5 word problem" to "tracking program state."

  variables = agents g0..g4 ; values = roles r0..r4 ; init identity (g_i = r_i).
  K=1 dense snapshots .. K=inf answer-only (the cliff variable). gdp_hybrid d384 (18.5M), 3 seeds.

  .venv/bin/python followups/non-abelian-state/code_trace.py
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
TRAIN_LEN = (4, 8, 16)
EVAL_LEN = [16, 64]
STEPS = 6000
KS = [1, 2, 4, 8, 10**9]
D_MODEL, N_LAYERS, D_FF = 384, 6, 1536
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_trace.md")


def build(w, oracle, L, K, rng):
    """Variable-swap execution trace. Returns (words, snapshot_spans[(start,end)], ans_idx, gold_value)."""
    ev = w.sample_hard_chain(L, episode_seed=f"ct|{rng.random()}")
    trace = oracle.hard_trace(ev)                       # trace[i][agent] = value (role) after i ops; trace[0]=identity
    words, spans = [], []
    for g in w.agents:                                  # initial full state (forced prompt context)
        words += [g, trace[0][g]]
    words.append(".")
    for i, e in enumerate(ev):
        if e.kind == "swap_role":
            words += ["swap", e.args[0], e.args[1], "."]
        else:                                           # cycle_roles
            words += ["cycle"] + list(e.args) + ["."]
        if (i + 1) % K == 0 or i == L - 1:              # interleaved full-state snapshot every K ops (+ final)
            words.append("roles")
            start = len(words)
            for g in w.agents:
                words.append(trace[i + 1][g])
            spans.append((start, len(words)))
            words.append(".")
    q = rng.choice(w.agents)
    words += [q, "?", ":"]
    ans = len(words)
    words += [trace[L][q], "."]
    return words, spans, ans, trace[L][q]


def code_eval(model, tok, w, exs, device="cuda", max_tok=8):
    """Free-running: ops forced; the model GENERATES the state snapshots + the final value. Score final value."""
    import torch
    role_ids = {tok.token_to_id[r] for r in w.roles}
    dot = tok.token_to_id["."]
    model.eval()
    correct = 0
    with torch.no_grad():
        for words, spans, ans, gold in exs:
            starts = {s: e for s, e in spans}
            ids, i, got = [], 0, None
            while i < len(words):
                if i == ans:
                    for _ in range(max_tok):
                        with torch.autocast(device, dtype=torch.bfloat16):
                            nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                        ids.append(nx)
                        if got is None and nx in role_ids: got = nx
                        if nx == dot: break
                    break
                elif i in starts:
                    for _ in range(max_tok):
                        with torch.autocast(device, dtype=torch.bfloat16):
                            nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                        ids.append(nx)
                        if nx == dot: break
                    i = starts[i]
                    while i < len(words) and words[i] != ".": i += 1
                    i += 1
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
    tok, _, _ = T.prepare([" ".join(build(w, oracle, 16, 1, random.Random(1))[0])], [], [w])
    agg = defaultdict(lambda: defaultdict(dict))
    print("=== CODE-TRACE non-abelian cliff (variable swaps, CWM-style state snapshots) ===", flush=True)
    for K in KS:
        prng = random.Random(2)
        pool = [" ".join(build(w, oracle, prng.choice(TRAIN_LEN), K, prng)[0]) for _ in range(8000)]
        evs = {L: [build(w, oracle, L, K, random.Random(900 + L + j)) for j in range(150)] for L in EVAL_LEN}
        tag = "inf" if K >= 10**9 else str(K)
        for s in SEEDS:
            model = train(tok, pool, s)
            for L in EVAL_LEN:
                agg[tag][L][s] = code_eval(model, tok, w, evs[L])
            print(f"  K={tag:<4} s{s} :: " + "  ".join(f"L{L}={agg[tag][L][s]:.3f}" for L in EVAL_LEN), flush=True)
            del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("code_trace done.", flush=True)


def write_md(agg):
    lines = [
        "# Code-trace non-abelian cliff — does the wall appear in execution-trace clothing?\n",
        "`followups/non-abelian-state/code_trace.py`. gdp_hybrid d384 (18.5M), 3 seeds. Our non-abelian dynamics "
        "rendered as a variable-swap execution trace (CWM surface grammar): vars hold values, `swap`/`cycle` ops, "
        "full-state snapshot every K ops, query a variable's final value (free-running). Same density sweep as "
        "`supervision_sweep.py` (the role-rendered cliff). Floor = 1/k = 0.20. A matching cliff = the wall is "
        "non-abelian composition, not the rendering.\n",
        "| K (snapshot / op) | L16 | L64 |",
        "|---|---|---|",
    ]
    nlab = {"1": "every op", "2": "every 2nd", "4": "every 4th", "8": "every 8th", "inf": "answer-only"}
    for tag in ["1", "2", "4", "8", "inf"]:
        if tag not in agg:
            continue
        cells = []
        for L in EVAL_LEN:
            xs = list(agg[tag][L].values())
            cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}({sum(x>0.5 for x in xs)}/{len(xs)})")
        lines.append(f"| {tag} ({nlab[tag]}) | " + " | ".join(cells) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
