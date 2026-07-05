"""Stage 2 — per-domain RL teachers: GRPO-specialise the shared base on each domain independently.

Each teacher is forked from the SAME Stage-1 base and trained with GRPO (outcome-only
verifiable reward = relaxed match of the sampled answer) on ONE domain. This is the
paper's Stage 2: teachers are same-origin (so the student they later distil into starts
at low KL from them) and independent (trainable fully in parallel). Each teacher defines
the normalised-score 1-anchor for its domain; the base defines the 0-anchor.

This is also the direct test of the user's precondition — "can we get good results from
RL?" — on a weak from-scratch transformer. A teacher that clearly beats the base on its
domain is the green light for MOPD.

Metrics (greedy relaxed match, held-out test):
  base@L / teacher@L    per-length accuracy before/after GRPO
  delta                 teacher - base (must be clearly > 0 for a usable norm-score)

  .venv/bin/python experiments/mopd/stage2_teachers.py --steps 1500
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_teachers.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--prompts_per_step", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domains", default="binding,recall")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return

    base, tok, ck = M.load_ckpt(os.path.join(M.CKPT_DIR, a.base))
    dims = ck["dims"]
    print(f"loaded base {a.base}  dims={dims}", flush=True)

    domains = a.domains.split(",")
    specs = {d: M.TEACHER_DOMAINS[d]() for d in domains}

    # base accuracy (the 0-anchor) at every eval length
    base_acc = M.eval_domains(base, tok, specs, n=a.eval_n)
    for d in domains:
        print(f"  base {d}: " + "  ".join(f"L{L}={v:.3f}" for L, v in base_acc[d].items()), flush=True)

    report: dict[str, dict] = {}
    for d in domains:
        print(f"=== GRPO teacher: {d} ===", flush=True)
        teacher = M.clone_model(base, tok, dims)                 # same-origin fork
        info = M.grpo_train(teacher, tok, specs[d], steps=a.steps,
                            prompts_per_step=a.prompts_per_step, group=a.group, lr=a.lr, seed=a.seed)
        t_acc = M.eval_domains(teacher, tok, {d: specs[d]}, n=a.eval_n)[d]
        path = os.path.join(M.CKPT_DIR, f"teacher_{d}.pt")
        M.save_ckpt(path, teacher, tok, dims,
                    {"stage": 2, "domain": d, "steps": a.steps, "lr": a.lr, "reward_traj": info["traj"]})
        report[d] = {"base": base_acc[d], "teacher": t_acc}
        print(f"  teacher {d}: " + "  ".join(f"L{L}={v:.3f}" for L, v in t_acc.items()), flush=True)
        print(f"  saved -> {path}", flush=True)
        del teacher
        torch.cuda.empty_cache()
        write_md(a, report)

    write_md(a, report)
    print("stage2_teachers done.", flush=True)


def write_md(a, report: dict) -> None:
    """Write the per-domain base-vs-teacher accuracy table to OUT."""
    lines = [
        "# Stage 2 — per-domain RL teachers (GRPO)\n",
        f"`experiments/mopd/stage2_teachers.py`. GRPO from `{a.base}`: {a.steps} steps, "
        f"{a.prompts_per_step} prompts/step x group {a.group}, lr {a.lr}, outcome 0/1 reward. "
        f"Greedy relaxed match, n={a.eval_n}. `delta` = teacher - base (the norm-score headroom).\n",
    ]
    for d, r in report.items():
        Ls = sorted(r["teacher"])
        lines += [f"\n**{d}**\n", "| L | base | teacher | delta |", "|---|---|---|---|"]
        for L in Ls:
            b, t = r["base"].get(L, float("nan")), r["teacher"][L]
            lines.append(f"| {L} | {b:.3f} | {t:.3f} | {t - b:+.3f} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
