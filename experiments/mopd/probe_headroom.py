"""De-risk gate — is there RL headroom on the weak transformer base, and at what difficulty?

MOPD's whole premise is that per-domain RL lifts the base; the normalised score is
degenerate if the teacher cannot beat the base. RL (GRPO) only gets signal where the
base already samples correct answers with intermediate probability — too easy (base at
ceiling) leaves no headroom, too hard (base at floor) leaves no pass@k for exploration.
So before spending on teachers/MOPD we measure, per candidate difficulty config:

  * greedy accuracy (the norm-score 0-anchor), and
  * pass@k (k=8, temp=1.0) — the fraction of prompts where >=1 of 8 samples is correct.
    Groups with 0 < successes < k have reward variance -> GRPO learning signal.

The RL-improvable band is roughly greedy in [0.1, 0.7] AND pass@8 clearly above greedy
(exploration headroom). This script prints a table per domain over a few `.scaled()`
configs; we freeze the chosen config into the Stage-2/3 scripts. If NO config lands in
the band for the transformer, that is the signal to invoke the gdp_hybrid fallback.

Metrics (per config, held-out test split):
  greedy@L   greedy relaxed-match accuracy at length L (the 0-anchor)
  pass@8     P(>=1 of 8 temp-1.0 samples correct) — RL exploration headroom
  var-frac   fraction of prompts with a non-degenerate group (0<succ<8) — direct GRPO-signal read

  .venv/bin/python experiments/mopd/probe_headroom.py --ckpt base.pt
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import mopd as M
from factworld import tasks as TK

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_headroom.md")

# Candidate difficulty configs per domain (label -> spec). Centered on the learnable edge.
def candidates() -> dict[str, dict[str, TK.TaskSpec]]:
    return {
        "binding": {
            "m4 (default)": M.binding_spec(),
        },
        "recall": {
            "default": M.recall_spec(),
        },
    }


def pass_at_k(model, tok, spec, length, n, k, device) -> tuple[float, float, float]:
    """Return (greedy_acc, pass@k, var_frac) at one length.

    var_frac = fraction of prompts whose k-sample group has 0 < successes < k
    (i.e. a non-degenerate GRPO advantage).
    """
    exs = TK.generate(spec, "test", n=n, length=length)
    greedy_ok = passk = varfrac = 0
    for e in exs:
        pids = tok.encode(e.prompt)
        g = M.sample_completions(model, tok, [pids], greedy=True, device=device)[0]
        greedy_ok += M.reward(g, tok, e.answer)
        samples = M.sample_completions(model, tok, [pids] * k, greedy=False, device=device)
        succ = sum(M.reward(c, tok, e.answer) for c in samples)
        passk += int(succ >= 1)
        varfrac += int(0 < succ < k)
    N = len(exs)
    return greedy_ok / N, passk / N, varfrac / N


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="base.pt")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    model, tok, ck = M.load_ckpt(os.path.join(M.CKPT_DIR, a.ckpt))
    print(f"loaded {a.ckpt}  dims={ck['dims']}  meta={ck.get('meta', {})}", flush=True)

    rows: list[tuple] = []
    for dom, cfgs in candidates().items():
        for label, spec in cfgs.items():
            for L in spec.train_lengths:                    # probe at the training lengths RL will use
                g, pk, vf = pass_at_k(model, tok, spec, L, a.n, a.k, "cuda")
                band = "<-- band" if (0.1 <= g <= 0.7 and vf >= 0.2) else ""
                rows.append((dom, label, L, g, pk, vf, band))
                print(f"  {dom:8} {label:14} L{L}: greedy={g:.3f} pass@{a.k}={pk:.3f} "
                      f"var={vf:.3f} {band}", flush=True)
    write_md(a, rows)
    print("probe_headroom done.", flush=True)


def write_md(a, rows: list[tuple]) -> None:
    """Write the headroom table to OUT."""
    lines = [
        "# De-risk probe — RL headroom on the base (pass@k)\n",
        f"`experiments/mopd/probe_headroom.py`. Base `{a.ckpt}`, n={a.n}, k={a.k}, temp=1.0. "
        "`greedy` = the norm-score 0-anchor; `pass@k` = P(>=1 of k samples correct) = exploration "
        "headroom; `var-frac` = fraction of prompts with a non-degenerate GRPO group (0<succ<k). "
        "RL-improvable band = greedy in [0.1, 0.7] and var-frac >= 0.2.\n",
        "| domain | config | L | greedy | pass@k | var-frac | |",
        "|---|---|---|---|---|---|---|",
    ]
    for dom, label, L, g, pk, vf, band in rows:
        lines.append(f"| {dom} | {label} | {L} | {g:.3f} | {pk:.3f} | {vf:.3f} | {band} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
