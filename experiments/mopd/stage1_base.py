"""Stage 1 — the shared SFT base: one weak transformer over a mixed binding+chain(+recall) corpus.

MOPD's first stage is a broad SFT checkpoint that (a) covers every target domain and
(b) is the SHARED origin for both RL teachers and the student — the same-origin
condition the paper shows is load-bearing for stable distillation. We train a
deliberately weak transformer (d=256, 4 layers) so RL and distillation have real
headroom. The mix covers both teacher domains (binding, chain); recall is included
as a shared primitive (the fact/agent format) even though it is not a teacher domain.

We adapt the repo's 80k-step "mixed" idea (scripts/experiment_curriculum_staged.py):
one model, next-token LM over a proportional concatenation of the domain corpora. The
prior mix omitted chain, so a chain teacher would start from floor with no pass@k for
RL to bootstrap — hence chain is added here (a documented deviation).

Metrics (greedy relaxed match on held-out test splits):
  train-len acc   is there pass@k for RL to exploit? (must be > 0 for GRPO to have signal)
  eval-len  acc   the extrapolation gap the weak base leaves open (RL/MOPD headroom)

  .venv/bin/python experiments/mopd/stage1_base.py --steps 40000
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
from factworld import train as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1_base.md")

# Stage-1 corpus mix (fraction of total docs) — the two teacher domains, 50/50.
# Deliberately UNDER-trained (see --steps) so the base sits at partial competence with
# real RL headroom on both domains. (chain is omitted — it is an RL wall, README §3.)
MIX = {"binding": 0.5, "recall": 0.5}
SPEC_OF = {"binding": M.binding_spec, "chain": M.chain_spec, "recall": M.recall_spec}
DOMAINS = ("binding", "recall")


def build_base_docs(train_n: int) -> list[str]:
    """Proportional concatenation of per-domain LM docs (prompt + answer)."""
    docs: list[str] = []
    for dom, frac in MIX.items():
        spec = SPEC_OF[dom]()
        n = int(train_n * frac)
        docs += [M.doc_of(e) for e in TK.generate(spec, "train", n=n)]
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--train_n", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arch", default=M.DIMS["arch"], help="base arch (transformer; gdp_hybrid fallback)")
    ap.add_argument("--d_model", type=int, default=M.DIMS["d_model"])
    ap.add_argument("--n_layers", type=int, default=M.DIMS["n_layers"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--out_name", default="base.pt")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return

    dims = {"arch": a.arch, "d_model": a.d_model, "n_layers": a.n_layers,
            "n_heads": M.DIMS["n_heads"], "d_ff": 4 * a.d_model}
    tok, w, r = M.shared_tokenizer()
    print(f"vocab_size={tok.vocab_size}  dims={dims}", flush=True)

    docs = build_base_docs(a.train_n)
    print(f"stage-1 corpus: {len(docs)} docs  mix={MIX}", flush=True)
    _tok, packed, _ = T.prepare(docs, [], [w], renderer=r)
    assert _tok.vocab_size == tok.vocab_size

    run = T.run(dims["arch"], tok, packed, [], steps=a.steps, batch=a.batch, lr=a.lr,
                d_model=dims["d_model"], n_layers=dims["n_layers"], n_heads=dims["n_heads"],
                d_ff=dims["d_ff"], seed=a.seed, return_model=True)
    model = run["model"]
    print(f"final LM loss {run['final_loss']:.4f}", flush=True)

    path = os.path.join(M.CKPT_DIR, a.out_name)
    M.save_ckpt(path, model, tok, dims, {"stage": 1, "mix": MIX, "steps": a.steps, "seed": a.seed,
                                         "final_loss": run["final_loss"]})
    print(f"saved base -> {path}", flush=True)

    # per-domain accuracy at train and eval lengths
    report: dict[str, dict[int, float]] = {}
    for dom in DOMAINS:
        spec = SPEC_OF[dom]()
        Ls = sorted(set(spec.train_lengths) | set(spec.eval_lengths))
        report[dom] = {L: M.accuracy(model, tok, spec, L, n=a.eval_n) for L in Ls}
        print(f"  {dom}: " + "  ".join(f"L{L}={v:.3f}" for L, v in report[dom].items()), flush=True)

    write_md(dims, a, report)
    print("stage1_base done.", flush=True)


def write_md(dims: dict, a, report: dict) -> None:
    """Write the per-domain base-accuracy table (train vs eval lengths) to OUT."""
    lines = [
        "# Stage 1 — shared mixed base (per-domain accuracy)\n",
        f"`experiments/mopd/stage1_base.py`. {dims['arch']} d{dims['d_model']}x{dims['n_layers']}, "
        f"{a.steps} steps, mix={MIX}, seed {a.seed}. Greedy relaxed match, n={a.eval_n}. "
        "Bold = training lengths (must be > 0 for RL to have pass@k signal); the rest are held-out "
        "OOD lengths (the headroom RL/MOPD can chase).\n",
    ]
    for dom in DOMAINS:
        spec = SPEC_OF[dom]()
        train_ls = set(spec.train_lengths)
        Ls = sorted(report[dom])
        head = " | ".join(f"L{L}" + ("*" if L in train_ls else "") for L in Ls)
        row = " | ".join(f"{report[dom][L]:.3f}" for L in Ls)
        lines += [f"\n**{dom}** (`*` = train length)\n", f"| {head} |", "|" + "---|" * len(Ls),
                  f"| {row} |"]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
