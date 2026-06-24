"""Can a curriculum move the cliff? Dense early to FORM the state circuit, then wean to sparse to INTERNALIZE it.

The sparsity sweep showed a cliff: the non-abelian state circuit forms only with near-dense process
supervision (checkpoint ~every other step), and sparse/answer-only gives zero traction from scratch. The
agentic hypothesis (Ian): a long session is a composition of short, well-supervised sub-sessions — so dense
supervision on short windows might bootstrap a circuit that then runs under sparse/no supervision. This is
gradual scratchpad removal / internalization (cf. Deng et al. 2024, explicit->implicit CoT).

Arms (parametric non-abelian composite, gdp_hybrid, custom loop so density can vary over TRAINING steps):
  anneal  K schedule 1 -> 2 -> 4 -> 8 -> inf over training (dense forms, sparse internalizes)
  mixed   each example a random K in {1,2,4,8,inf} (order-agnostic control: is it just 'see dense examples'?)
Baselines (from supervision_sweep.py): from-scratch K=1 solves (L16 1.0); from-scratch K=inf floors (0.20).

Headline: ANSWER-ONLY eval (K=inf, no intermediate scratchpad) — did the circuit internalize? Beat 0.20?

  .venv/bin/python followups/non-abelian-state/curriculum.py
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

from dense_capstone import _world, e2e_eval          # reuse world + free-running eval
from supervision_sweep import build                  # doc builder, holder checkpoint every K events

SEEDS = [0, 1, 2, 3, 4]
TRAIN_LEN = (4, 8, 16)
EVAL_LEN = [16, 64]
KS = [1, 2, 4, 8, 10**9]                              # 10**9 == answer-only
STEPS = 6000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curriculum.md")


def anneal_K(step):
    """1 -> 2 -> 4 -> 8 -> inf; most budget on the dense (form) and the answer-only (internalize) ends."""
    f = step / STEPS
    if f < 0.40: return 1
    if f < 0.55: return 2
    if f < 0.70: return 4
    if f < 0.85: return 8
    return 10**9


def make_pools(w, r, origins, oracle, per=8000):
    """A doc pool per density K (built once; sampled during training)."""
    pools = {}
    for K in KS:
        rng = random.Random(1000 + (0 if K >= 10**9 else K))
        pools[K] = [" ".join(build(w, r, origins, oracle, rng.choice(TRAIN_LEN), K, rng)[0]) for _ in range(per)]
    return pools


def train_curriculum(arm, tok, pools, seed, device="cuda"):
    import torch
    import torch.nn.functional as F
    from factworld.models import build_model
    torch.manual_seed(seed)
    model = build_model("gdp_hybrid", tok.vocab_size, d_model=256, n_layers=4, d_ff=1024,
                        num_householder=4, allow_neg_eigval=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warmup, pad = 1000, tok.pad_id
    rng = random.Random(7000 + seed)
    allmix = [d for K in KS for d in pools[K]]                    # union, uniform over K, for the mixed arm
    model.train()
    for step in range(STEPS):
        for pg in opt.param_groups:                              # warmup + cosine (matches train.run)
            pg["lr"] = 1e-3 * (min(1.0, (step + 1) / warmup) if step < warmup else
                               0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, STEPS - warmup))))
        if arm == "anneal":
            batch_docs = rng.sample(pools[anneal_K(step)], 32)
        else:
            batch_docs = rng.sample(allmix, 32)
        seqs = [tok.encode(d) for d in batch_docs]
        ml = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), ml), pad, dtype=torch.long, device=device)
        for ri, s in enumerate(seqs):
            inp[ri, : len(s)] = torch.tensor(s, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(inp[:, :-1])
            tgt = inp[:, 1:]
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
    pools = make_pools(w, r, origins, oracle)
    tok, _, _ = T.prepare(pools[1][:50], [], [w])                # closed vocab from world; texts irrelevant
    # eval sets: answer-only (K=inf, internalized) and dense (K=1, retained scratchpad), at L16/L64
    evs = {(Ke, L): [build(w, r, origins, oracle, L, Ke, random.Random(900 + L + j)) for j in range(200)]
           for Ke in (10**9, 1) for L in EVAL_LEN}
    agg = defaultdict(dict)
    print("=== CURRICULUM: dense->sparse anneal vs mixed; ANSWER-ONLY eval is the agentic target ===", flush=True)
    for arm in ("anneal", "mixed"):
        for s in SEEDS:
            model = train_curriculum(arm, tok, pools, s)
            for (Ke, L), ex in evs.items():
                eh, ev = e2e_eval(model, tok, w, ex)
                agg[(arm, Ke, L)][s] = ev
            ao16 = agg[(arm, 10**9, 16)][s]; ao64 = agg[(arm, 10**9, 64)][s]
            print(f"  {arm:<7} s{s} :: answer-only L16={ao16:.3f} L64={ao64:.3f} | "
                  f"dense L16={agg[(arm,1,16)][s]:.3f} L64={agg[(arm,1,64)][s]:.3f}", flush=True)
            del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("curriculum done.", flush=True)


def write_md(agg):
    lines = [
        "# Curriculum — can dense->sparse weaning internalize the state circuit?\n",
        "`followups/non-abelian-state/curriculum.py`. gdp_hybrid d256x4, 6000 steps, 5 seeds, parametric. "
        "`anneal` = K schedule 1->2->4->8->inf over training; `mixed` = random K per example. Eval is "
        "free-running e2e composite. **Answer-only** (K=inf, no scratchpad) is the agentic target; baselines "
        "from the sweep: from-scratch K=1 = 1.00/0.78, from-scratch K=inf = **0.20 floor**. Floor = 0.20.\n",
        "| arm | answer-only L16 | answer-only L64 | dense L16 | dense L64 |",
        "|---|---|---|---|---|",
    ]
    for arm in ("anneal", "mixed"):
        def cell(Ke, L):
            d = agg.get((arm, Ke, L))
            if not d:
                return "…"
            xs = list(d.values())
            return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}({sum(x>0.5 for x in xs)}/{len(xs)})"
        lines.append(f"| {arm} | {cell(10**9,16)} | {cell(10**9,64)} | {cell(1,16)} | {cell(1,64)} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
