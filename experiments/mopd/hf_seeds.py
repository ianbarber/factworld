"""Multi-seed robustness — run the whole Qwen3 MOPD pipeline over several seeds, aggregate mean±std.

For each seed: fork fresh LoRA adapters, GRPO-train a teacher per domain, MOPD-distil both into a
student (both loss forms), evaluate, and record normalised scores. The base backbone is fixed
(seed-independent), so per-seed variance comes from the RL rollouts and prompt order (seeded).
Reports the mean±std of the normalised score per model per domain across seeds — the robustness
check on the single-seed headline.

  .venv/bin/python experiments/mopd/hf_seeds.py --seeds 0 1 2 --teacher_steps 150 --mopd_steps 200
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import mopd_hf as H

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_seeds.md")
MODELS = ["teacher_binding", "teacher_recall", "mopd_pg", "mopd_kl"]


def _fresh(pmodel, name: str) -> None:
    """Re-initialise adapter ``name`` to a fresh (zero-init) fork of the base."""
    if name in pmodel.peft_config:
        pmodel.delete_adapter(name)
    H.add_adapter(pmodel, name)


def run_seed(pmodel, tok, specs, seed, a) -> dict:
    """Train teachers + students for one seed; return accuracy {model: {domain: {L: acc}}}."""
    import torch
    domains = list(specs)
    # teachers
    for d in domains:
        _fresh(pmodel, f"teacher_{d}")
        H.grpo_train(pmodel, tok, f"teacher_{d}", specs[d], steps=a.teacher_steps,
                     prompts_per_step=8, group=8, lr=a.lr, temp=a.temp, seed=seed)
    teachers = {d: f"teacher_{d}" for d in domains}
    # students (both loss forms)
    for loss in ("pg", "kl"):
        _fresh(pmodel, f"student_{loss}")
        H.mopd_train(pmodel, tok, f"student_{loss}", teachers, specs, steps=a.mopd_steps,
                     prompts_per_domain=8, loss_form=loss, lr=a.lr, temp=a.temp, seed=seed)
    torch.cuda.empty_cache()
    # eval every model on ALL domains (cross-domain contrast)
    acc = {}
    for d in domains:
        acc[f"teacher_{d}"] = H.eval_all(pmodel, tok, f"teacher_{d}", specs, n=a.eval_n)
    acc["mopd_pg"] = H.eval_all(pmodel, tok, "student_pg", specs, n=a.eval_n)
    acc["mopd_kl"] = H.eval_all(pmodel, tok, "student_kl", specs, n=a.eval_n)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--teacher_steps", type=int, default=150)
    ap.add_argument("--mopd_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--eval_n", type=int, default=300)
    ap.add_argument("--domains", default="binding,recall")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return

    domains = a.domains.split(",")
    specs = {d: H.DOMAINS[d]() for d in domains}
    model, tok = H.load_backbone()
    pmodel = H.to_peft(model, "teacher_" + domains[0])
    base_acc = H.eval_all(pmodel, tok, None, specs, n=a.eval_n)     # adapters disabled; seed-independent
    print("base: " + "  ".join(f"{d} " + " ".join(f"L{L}={v:.3f}" for L, v in base_acc[d].items())
                               for d in domains), flush=True)

    # per-seed normalised score: {model: {domain: [norm across seeds]}}
    norms = {m: {d: [] for d in domains} for m in MODELS}
    for seed in a.seeds:
        print(f"\n===== SEED {seed} =====", flush=True)
        acc = run_seed(pmodel, tok, specs, seed, a)
        for m in MODELS:
            for d in domains:
                tacc = acc[f"teacher_{d}"][d]
                vals = [H.normalized_score(acc[m][d][L], base_acc[d][L], tacc[L]) for L in acc[m][d]]
                vals = [v for v in vals if v == v]
                if vals:
                    norms[m][d].append(statistics.mean(vals))
        for m in MODELS:
            print(f"  {m:16}: " + "  ".join(f"{d}={norms[m][d][-1]:.3f}" for d in domains), flush=True)
        write_md(a, domains, norms)

    write_md(a, domains, norms)
    print("\n=== AGGREGATE (mean±std normalised score across seeds) ===", flush=True)
    for m in MODELS:
        cells = " ".join(f"{d}={_ms(norms[m][d])}" for d in domains)
        allv = [v for d in domains for v in norms[m][d]]
        print(f"  {m:16}: {cells}   avg={_ms(allv)}", flush=True)
    print("hf_seeds done.", flush=True)


def _ms(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f}±{statistics.pstdev(vals):.3f}"


def write_md(a, domains, norms) -> None:
    lines = [
        "# MOPD on FactWorld (Qwen3-1.7B) — multi-seed robustness\n",
        f"`experiments/mopd/hf_seeds.py`. {H.MODEL}, seeds {a.seeds}. Per seed: fresh GRPO teachers "
        f"({a.teacher_steps} steps) + MOPD students ({a.mopd_steps} steps, both loss forms), greedy "
        f"eval n={a.eval_n}. Normalised score `(model-base)/(teacher-base)` (0=base, 1=domain "
        "teacher), mean over eval lengths; cells are mean±std across seeds.\n",
        "| model | " + " | ".join(domains) + " | avg |", "|---|" + "---|" * (len(domains) + 1),
    ]
    for m in MODELS:
        allv = [v for d in domains for v in norms[m][d]]
        lines.append(f"| {m} | " + " | ".join(_ms(norms[m][d]) for d in domains) + f" | {_ms(allv)} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
