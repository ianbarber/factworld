"""Stage 2 (Qwen3) — per-domain RL teachers: GRPO-train one LoRA adapter per domain.

Each teacher is a fresh LoRA adapter on the shared frozen Qwen3 backbone (same-origin by
construction), GRPO-trained on its domain with the verifiable relaxed-match reward. The base
(adapters disabled) is the norm-score 0-anchor; each teacher is its domain's 1-anchor.

This is the pivot's key test: does outcome-RL genuinely specialise a *pretrained* model where
it barely moved the from-scratch model? A teacher clearly beating base is the green light for MOPD.

  .venv/bin/python experiments/mopd/hf_teachers.py --steps 300
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import mopd_hf as H

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_teachers.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--domains", default="binding,recall")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return

    domains = a.domains.split(",")
    specs = {d: H.DOMAINS[d]() for d in domains}
    model, tok = H.load_backbone()
    pmodel = H.to_peft(model, f"teacher_{domains[0]}")
    for d in domains[1:]:
        H.add_adapter(pmodel, f"teacher_{d}")
    print(f"loaded {H.MODEL} + {len(domains)} teacher adapters", flush=True)

    base_acc = H.eval_all(pmodel, tok, None, specs, n=a.eval_n)     # adapters disabled
    for d in domains:
        print(f"  base {d}: " + "  ".join(f"L{L}={v:.3f}" for L, v in base_acc[d].items()), flush=True)

    report = {}
    for d in domains:
        print(f"=== GRPO teacher: {d} ===", flush=True)
        info = H.grpo_train(pmodel, tok, f"teacher_{d}", specs[d], steps=a.steps,
                            prompts_per_step=a.prompts_per_step, group=a.group, lr=a.lr, temp=a.temp)
        t_acc = H.eval_all(pmodel, tok, f"teacher_{d}", {d: specs[d]}, n=a.eval_n)[d]
        H.save_adapter(pmodel, f"teacher_{d}")
        report[d] = {"base": base_acc[d], "teacher": t_acc, "traj": info["traj"]}
        print(f"  teacher {d}: " + "  ".join(f"L{L}={v:.3f}" for L, v in t_acc.items()), flush=True)
        print(f"  saved -> {os.path.join(H.CKPT_DIR, f'teacher_{d}')}", flush=True)
        write_md(a, report)
    write_md(a, report)
    print("hf_teachers done.", flush=True)


def write_md(a, report: dict) -> None:
    lines = [
        "# Stage 2 (Qwen3-1.7B) — per-domain RL teachers (GRPO LoRA)\n",
        f"`experiments/mopd/hf_teachers.py`. {H.MODEL}, LoRA GRPO from the frozen backbone: "
        f"{a.steps} steps, {a.prompts_per_step} prompts/step x group {a.group}, lr {a.lr}, "
        f"temp {a.temp}, verifiable relaxed-match reward, thinking off. Greedy eval, n={a.eval_n}. "
        "`delta` = teacher - base.\n",
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
