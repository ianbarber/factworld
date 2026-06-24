"""The agentic question the capstone skipped: how SPARSE can the state supervision be?

ladder/decompose/capstone established: the non-abelian composite floors under answer-only supervision (R3b),
is SOLVED under dense per-step oracle supervision (capstone), and recall is free either way. But dense
per-step state labels are privileged info an agent won't have. The transferable result is the curve BETWEEN
the two extremes: supervise the holder only every K steps (always keep the final one so recall stays keyed),
and find the threshold K* where the state circuit stops forming.

  K=1     dense, every step                     == capstone (solves)
  K=2,4,8 sparse checkpoints
  K=inf   only the final holder                 == R3b (answer-only, floors)

Parametric recall throughout (facts in weights, not in prompt) — the agentic "fact lives in the model" case.
Headline metric: end-to-end free-running composite accuracy vs supervision stride.

  .venv/bin/python followups/non-abelian-state/supervision_sweep.py
"""
import os
import random
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dense_capstone import _world, dense_eval, e2e_eval  # reuse world + eval (hidx/vidx-generic)

SEEDS = [0, 1, 2, 3, 4]
TRAIN_LEN = (4, 8, 16)
EVAL_LEN = [16, 64]
STRIDES = [1, 2, 4, 8, 10**9]   # 10**9 == answer-only (only final holder labelled)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supervision_sweep.md")


def build(w, r, origins, oracle, L, K, rng):
    """Parametric non-abelian composite with holder checkpoints every K steps (+ always the final step)."""
    ev = w.sample_hard_chain(L, episode_seed=f"K{K}|{rng.random()}")
    trace = oracle.hard_trace(ev)                         # trace[i] = assignment after i events
    role = rng.choice(w.roles)
    hist = r.render_history(tuple(ev), with_steps=True)
    words, hidx = ["role", role, "."], []
    for i, hstr in enumerate(hist):
        words += hstr.split()
        if (i + 1) % K == 0 or i == L - 1:               # labelled checkpoint
            words.append("holder")
            inv = {ro: ag for ag, ro in trace[i + 1].items()}
            hidx.append(len(words)); words.append(inv[role])
            words.append(".")
    final_holder = {ro: ag for ag, ro in trace[L].items()}[role]
    words += ["what", "is", "a0", "?", ":"]
    vidx = len(words); words.append(origins[final_holder]); words.append(".")
    return words, hidx, vidx


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    agg = defaultdict(lambda: defaultdict(dict))
    print("=== SUPERVISION-SPARSITY SWEEP (parametric non-abelian composite, gdp_hybrid) ===", flush=True)
    for K in STRIDES:
        rng = random.Random(2)
        train = [" ".join(build(w, r, origins, oracle, rng.choice(TRAIN_LEN), K, rng)[0]) for _ in range(8000)]
        evs = {L: [build(w, r, origins, oracle, L, K, random.Random(300 + L + j)) for j in range(200)]
               for L in EVAL_LEN}
        tok, docs, _ = T.prepare(train, [], [w])
        tag = "inf" if K >= 10**9 else str(K)
        for s in SEEDS:
            run = T.run("gdp_hybrid", tok, docs, [], steps=4000, batch=32, d_model=256, n_layers=4,
                        d_ff=1024, seed=s, return_model=True)
            for L in EVAL_LEN:
                dh, _vt = dense_eval(run["model"], tok, w, evs[L])
                eh, ev = e2e_eval(run["model"], tok, w, evs[L])
                agg[tag][L][s] = dict(dense_h=dh, e2e_h=eh, e2e_v=ev)
                print(f"  K={tag:<4} s{s} L{L:<3} :: dense_h={dh:.3f} e2e_h={eh:.3f} e2e_v={ev:.3f}", flush=True)
            del run["model"]; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("supervision_sweep done.", flush=True)


def write_md(agg):
    lines = [
        "# Supervision-sparsity sweep — how sparse can the state supervision be?\n",
        "`followups/non-abelian-state/supervision_sweep.py`. gdp_hybrid d256x4, 4000 steps, 5 seeds, "
        "parametric recall. Holder supervised every K events (+ always the final). K=1 == dense capstone; "
        "K=inf == answer-only (only the final holder, the R3b floor). Train lengths (4,8,16) so a 16-event "
        "episode gets 16/8/4/2/1 checkpoints at K=1/2/4/8/inf. Metric: end-to-end free-running composite "
        "(`e2e_v`); `dense_h` = teacher-forced acc at labelled slots. Floor = 0.20.\n",
        "| K (stride) | labels / 16-ep | L | dense_h | e2e_holder | e2e_value | conv(e2e_v>0.5) |",
        "|---|---|---|---|---|---|---|",
    ]
    nlab = {"1": 16, "2": 8, "4": 4, "8": 2, "inf": 1}
    for tag in ["1", "2", "4", "8", "inf"]:
        for L in EVAL_LEN:
            d = agg.get(tag, {}).get(L)
            if not d:
                continue
            xs = list(d.values())
            def ms(k):
                return statistics.mean(x[k] for x in xs), statistics.pstdev(x[k] for x in xs)
            conv = sum(x["e2e_v"] > 0.5 for x in xs)
            lines.append(f"| {tag} | {nlab[tag]} | {L} | {ms('dense_h')[0]:.2f}±{ms('dense_h')[1]:.2f} | "
                         f"{ms('e2e_h')[0]:.2f}±{ms('e2e_h')[1]:.2f} | {ms('e2e_v')[0]:.2f}±{ms('e2e_v')[1]:.2f} | "
                         f"{conv}/{len(xs)} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
