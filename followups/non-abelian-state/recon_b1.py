"""B1 reconciliation: does the abelian+parametric "dereferences fine (0.83)" claim survive STRICT scoring?

The review flagged that R1's 0.83 uses a lenient value-scan metric (first value token anywhere in the
generated span), while the companion paper scores binding x parametric recall position-strict. The follow-on
must not publicly "correct" the companion paper on a non-comparable metric. Here we re-run R1 (abelian
last-write-wins binding + parametric recall, no CoT) and score the SAME models two ways:

  value_scan : first emitted value token == gold        (lenient; the ladder's 0.83 metric)
  strict_pos : argmax at the answer position == gold     (position-strict; single forward, no generation)

If strict_pos ~ value_scan, the dereference claim holds and we can state the correction with confidence.
If strict_pos collapses, the 0.83 was a scoring artifact and the §3 rung must be requalified.

  .venv/bin/python followups/non-abelian-state/recon_b1.py
"""
import os
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ladder import _world, make_docs, make_eval, value_eval

SEEDS = [0, 1, 2]
EVAL_LEN = [16, 64]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon_b1.md")


def strict_pos_eval(model, tok, w, exs, device="cuda"):
    """Position-strict: the argmax at the answer position (prompt's last token) must equal gold value.
    Single forward per example, no autoregressive scan."""
    import torch
    model.eval()
    correct = 0
    with torch.no_grad():
        for prompt, _holder, value, _L in exs:
            ids = tok.encode(prompt)
            with torch.autocast(device, dtype=torch.bfloat16):
                nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
            correct += int(nx == tok.token_to_id[value])
    return correct / len(exs)


def main():
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    train_docs = make_docs("R1", w, r, origins, oracle, 8000, 2)
    evs = {L: make_eval("R1", w, r, origins, oracle, 200, 200 + L, L) for L in EVAL_LEN}
    tok, docs, _ = T.prepare(train_docs, [], [w])
    res = defaultdict(lambda: defaultdict(list))
    print("=== B1 RECON: R1 abelian+parametric, value-scan vs position-strict ===", flush=True)
    for s in SEEDS:
        run = T.run("gdp_hybrid", tok, docs, [], steps=4000, batch=32, d_model=256, n_layers=4,
                    d_ff=1024, seed=s, return_model=True)
        for L in EVAL_LEN:
            res[L]["scan"].append(value_eval(run["model"], tok, w, evs[L]))
            res[L]["strict"].append(strict_pos_eval(run["model"], tok, w, evs[L]))
        print(f"  s{s} :: " + "  ".join(
            f"L{L} scan={res[L]['scan'][-1]:.3f} strict={res[L]['strict'][-1]:.3f}" for L in EVAL_LEN),
            flush=True)
        del run["model"]; torch.cuda.empty_cache()
        write_md(res)
    write_md(res)
    print("recon_b1 done.", flush=True)


def write_md(res):
    lines = [
        "# B1 reconciliation — R1 (abelian + parametric) under lenient vs position-strict scoring\n",
        "`followups/non-abelian-state/recon_b1.py`. gdp_hybrid d256x4, 4000 steps, 3 seeds. `scan` = first "
        "emitted value token (the ladder's 0.83 metric); `strict` = argmax at the answer position (companion "
        "paper's position-strict style). Floor = 0.20.\n",
        "| eval | value-scan | position-strict |",
        "|---|---|---|",
    ]
    for L in EVAL_LEN:
        if not res[L]["scan"]:
            continue
        def ms(k):
            xs = res[L][k]
            return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}"
        lines.append(f"| L{L} | {ms('scan')} | {ms('strict')} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
