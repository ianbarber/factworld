"""Reconciliation: does the END-QUERY recipe (ladder R1/R2/R3a) hold to high length, or also cliff?

decay_curve.py showed the FRONT-loaded-query (pure online-recurrence) setup floors ALL arms past ~1.5xLmax,
contradicting ladder R1 abelian (0.57@L64). Ladder states the query at the END (read the whole history, THEN
ask), which sidesteps carrying the query identity through the recurrence. This re-runs the EXACT ladder recipe
(gdp_hybrid d256x4, 4000 steps, train {4,8,16}, parametric, facts not in prompt) but evaluates on the fine
length grid L16..256, to settle the contradiction:
  - abelian R1 (no CoT) / R2 (emit holder then value): graceful decay (real-ish register) or a cliff?
  - non-abelian R3a (no CoT): floors at L16 already (no per-step supervision) -> stays at floor (control).

If R1 decays gracefully and R3a sits at floor, the end-query setup gives the clean circuit-vs-shortcut contrast
and the decay_curve cliff was the front-query online-recurrence carry artifact.

  .venv/bin/python followups/parametric-recall/reeval_endquery.py
"""
import os
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import ladder as Ldr   # reuse EXACT recipe: _world, make_docs, make_eval, value_eval, ARCH

SEEDS = [0, 1, 2]
RUNGS = ["R1", "R2", "R3a"]
EVAL_LEN = [16, 24, 32, 48, 64, 96, 128, 192, 256]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reeval_endquery.md")


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from iso import strict_eval
    w, r, origins = Ldr._world()
    oracle = Oracle(w)
    is_cot = {"R1": False, "R2": True, "R3a": False}
    res = defaultdict(lambda: defaultdict(list))
    print("=== END-QUERY RE-EVAL: ladder recipe on fine grid L16..256 ===", flush=True)
    for rung in RUNGS:
        train_docs = Ldr.make_docs(rung, w, r, origins, oracle, 8000, 2)
        evs = {L: Ldr.make_eval(rung, w, r, origins, oracle, 200, 200 + L, L) for L in EVAL_LEN}
        tok, docs, _ = T.prepare(train_docs, [], [w])
        efn = strict_eval if is_cot[rung] else Ldr.value_eval
        for s in SEEDS:
            run = T.run(Ldr.ARCH, tok, docs, [], steps=4000, batch=32, d_model=256, n_layers=4,
                        d_ff=1024, seed=s, return_model=True)
            for L in EVAL_LEN:
                res[rung][L].append(efn(run["model"], tok, w, evs[L]))
            print(f"  {rung:<4} s{s} :: " + "  ".join(f"L{L}={res[rung][L][-1]:.2f}" for L in EVAL_LEN),
                  flush=True)
            del run["model"]; torch.cuda.empty_cache()
        write_md(res)
    write_md(res)
    print("reeval_endquery done.", flush=True)


def write_md(res):
    lines = [
        "# End-query re-eval — does the ladder recipe hold to high length? (`reeval_endquery.py`, d256x4, 3 seeds)\n",
        "Exact ladder recipe (gdp_hybrid d256x4, 4000 steps, train {4,8,16}, parametric recall). Query stated "
        "at the END (read history, then ask) — the contrast to decay_curve's front-loaded query. R1 abelian "
        "no-CoT (value metric); R2 abelian CoT (strict holder+value); R3a non-abelian no-CoT (floors at L16, "
        "control). Floor = 0.20.\n",
        "| rung | " + " | ".join(f"L{L}" for L in EVAL_LEN) + " |",
        "|---|" + "---|" * len(EVAL_LEN),
    ]
    desc = {"R1": "abelian, no CoT", "R2": "abelian, CoT", "R3a": "non-abelian, no CoT"}
    for rung in RUNGS:
        if rung not in res:
            continue
        cells = []
        for L in EVAL_LEN:
            xs = res[rung].get(L)
            cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "…")
        lines.append(f"| {rung} ({desc[rung]}) | " + " | ".join(cells) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
