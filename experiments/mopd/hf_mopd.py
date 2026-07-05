"""Stage 3 (Qwen3) — MOPD: distil both RL-teacher LoRA adapters into one student adapter.

The student is a fresh LoRA adapter on the same frozen Qwen3 backbone (same-origin as the
teachers). Each step draws a balanced per-domain batch, the student rolls out on-policy, each
rollout is routed to its domain teacher adapter (swapped in for a no-grad scoring pass), and the
student adapter is updated on the per-token reverse KL toward that teacher. One student adapter
should end up holding BOTH teachers' abilities.

  .venv/bin/python experiments/mopd/hf_mopd.py --loss pg
  .venv/bin/python experiments/mopd/hf_mopd.py --loss kl
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_mopd.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["pg", "kl"], default="pg")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts_per_domain", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--domains", default="binding,recall")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return

    domains = a.domains.split(",")
    specs = {d: H.DOMAINS[d]() for d in domains}
    student = f"student_{a.loss}"
    model, tok = H.load_backbone()
    pmodel = H.to_peft(model, student)
    teachers = {}
    for d in domains:
        H.load_adapter(pmodel, f"teacher_{d}")
        teachers[d] = f"teacher_{d}"
    print(f"loaded backbone + student '{student}' + teachers {list(teachers)}", flush=True)

    info = H.mopd_train(pmodel, tok, student, teachers, specs, steps=a.steps,
                        prompts_per_domain=a.prompts_per_domain, loss_form=a.loss,
                        lr=a.lr, temp=a.temp)
    H.save_adapter(pmodel, student)
    print(f"saved student -> {os.path.join(H.CKPT_DIR, student)}", flush=True)
    write_md(a, info["dynamics"])
    print("hf_mopd done.", flush=True)


def write_md(a, dynamics: dict) -> None:
    header = [
        "# Stage 3 (Qwen3-1.7B) — MOPD distillation dynamics\n",
        "`experiments/mopd/hf_mopd.py`. Student LoRA adapter on the shared backbone; both frozen "
        "teacher adapters distilled on the student's own rollouts. Per-token reverse KL should start "
        "LOW (same-origin) and stay stable; entropy should not collapse. Per domain, initial -> final.\n",
    ]
    prior = ""
    if os.path.exists(OUT):
        body = open(OUT).read()
        if "## loss" in body:
            prior = body[body.index("## loss"):].strip()
    block = [f"\n## loss = `{a.loss}`  ({a.steps} steps, lr {a.lr})\n",
             "| domain | KL init | KL final | H init | H final |", "|---|---|---|---|---|"]
    for d, series in dynamics.items():
        if not series:
            continue
        (_s0, kl0, h0), (_s1, kl1, h1) = series[0], series[-1]
        block.append(f"| {d} | {kl0:.4f} | {kl1:.4f} | {h0:.3f} | {h1:.3f} |")
    with open(OUT, "w") as f:
        f.write("\n".join(header) + ("\n" + prior + "\n" if prior else "\n") + "\n".join(block) + "\n")


if __name__ == "__main__":
    main()
