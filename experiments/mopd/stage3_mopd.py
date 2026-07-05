"""Stage 3 — MOPD: distil both frozen RL teachers into one student on the student's own rollouts.

The student is forked from the SAME Stage-1 base as the teachers (same-origin -> low
initial KL -> stable distillation, the paper's load-bearing condition). Each step draws a
balanced per-domain batch, the student rolls out (on-policy, N=1), each rollout is routed
to its domain teacher, and the student is updated on the per-token reverse KL toward that
teacher. The result is ONE model that should hold BOTH teachers' abilities.

We run both distillation loss forms:
  pg   policy-gradient with the clipped teacher-minus-student log-diff advantage (paper eq. 4)
  kl   exact full-vocabulary per-token reverse KL (our low-variance analogue of the paper's
       top-k form; exact here because the atomic vocab is small)

Metrics: written by evaluate.py. This script logs the training dynamics the paper reports
(Fig. 3): per-token reverse KL (should start LOW under same-origin teachers and stay stable)
and student entropy, per domain.

  .venv/bin/python experiments/mopd/stage3_mopd.py --loss pg
  .venv/bin/python experiments/mopd/stage3_mopd.py --loss kl
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage3_mopd.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base.pt")
    ap.add_argument("--loss", choices=["pg", "kl"], default="pg")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--prompts_per_domain", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domains", default="binding,recall")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return

    base, tok, ck = M.load_ckpt(os.path.join(M.CKPT_DIR, a.base))
    dims = ck["dims"]
    domains = a.domains.split(",")
    specs = {d: M.TEACHER_DOMAINS[d]() for d in domains}

    teachers = {}
    for d in domains:
        tpath = os.path.join(M.CKPT_DIR, f"teacher_{d}.pt")
        teachers[d], _tok, _ck = M.load_ckpt(tpath)
        teachers[d].eval()
    print(f"loaded base {a.base} + teachers {domains}  dims={dims}", flush=True)

    student = M.clone_model(base, tok, dims)                     # same-origin student init
    info = M.mopd_train(student, teachers, tok, specs, steps=a.steps,
                        prompts_per_domain=a.prompts_per_domain, loss_form=a.loss,
                        lr=a.lr, seed=a.seed)

    path = os.path.join(M.CKPT_DIR, f"student_{a.loss}.pt")
    M.save_ckpt(path, student, tok, dims,
                {"stage": 3, "loss": a.loss, "steps": a.steps, "dynamics": info["dynamics"]})
    print(f"saved student -> {path}", flush=True)

    write_md(a, info["dynamics"])
    print("stage3_mopd done.", flush=True)


def write_md(a, dynamics: dict) -> None:
    """Append the training-dynamics summary (initial/final reverse KL + entropy) for this loss form."""
    header = [
        "# Stage 3 — MOPD distillation dynamics\n",
        "`experiments/mopd/stage3_mopd.py`. Student forked from the shared base; both frozen "
        "teachers distilled on the student's own rollouts. Per-token reverse KL should start LOW "
        "(same-origin teachers) and stay stable; entropy should not collapse. Per domain, "
        "initial -> final over training.\n",
    ]
    # Merge with any existing file so pg and kl rows coexist.
    prior = ""
    if os.path.exists(OUT):
        prior = open(OUT).read()
        if prior.startswith("# Stage 3"):
            prior = prior.split("\n\n", 1)[1] if "\n\n" in prior else ""
    block = [f"\n## loss = `{a.loss}`  ({a.steps} steps, lr {a.lr})\n",
             "| domain | KL init | KL final | H init | H final |", "|---|---|---|---|---|"]
    for d, series in dynamics.items():
        if not series:
            continue
        (_s0, kl0, h0), (_s1, kl1, h1) = series[0], series[-1]
        block.append(f"| {d} | {kl0:.4f} | {kl1:.4f} | {h0:.3f} | {h1:.3f} |")
    with open(OUT, "w") as f:
        f.write("\n".join(header) + "\n" + prior.strip() + "\n" + "\n".join(block) + "\n")


if __name__ == "__main__":
    main()
