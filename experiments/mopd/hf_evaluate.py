"""Evaluate (Qwen3) — the headline MOPD table: one student adapter, both teachers' abilities?

Loads the shared backbone and every adapter (teachers + MOPD students) and reports, per domain
and length: greedy accuracy + the normalised score s~ = (model - base)/(teacher - base), 0 at the
base backbone and 1 at the per-domain teacher. Headline = uniform average across domains. The
cross-domain contrast is the point: teacher_binding is trained only on binding, teacher_recall only
on recall; the MOPD student should match each on its own domain from ONE adapter.

  .venv/bin/python experiments/mopd/hf_evaluate.py
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_evaluate.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_n", type=int, default=300)
    ap.add_argument("--domains", default="binding,recall")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return

    domains = a.domains.split(",")
    specs = {d: H.DOMAINS[d]() for d in domains}
    model, tok = H.load_backbone()

    # adapters to load if present
    adapters = [f"teacher_{d}" for d in domains] + ["student_pg", "student_kl"]
    present = [n for n in adapters if H.adapter_exists(n)]
    if not present:
        print("no adapters found in", H.CKPT_DIR); return
    pmodel = H.to_peft(model, "_tmp")               # throwaway init adapter; disable_adapter() = base
    for name in present:
        H.load_adapter(pmodel, name)
    pmodel.delete_adapter("_tmp")
    display = {n: ("mopd_pg" if n == "student_pg" else "mopd_kl" if n == "student_kl" else n)
               for n in present}
    print(f"loaded adapters: {present}", flush=True)

    # accuracy: base (adapters disabled) + each adapter
    accs = {"base": H.eval_all(pmodel, tok, None, specs, n=a.eval_n)}
    for name in present:
        accs[display[name]] = H.eval_all(pmodel, tok, name, specs, n=a.eval_n)
        print(f"  evaluated {display[name]}", flush=True)

    # normalised scores
    base = accs["base"]
    norms = {}
    for name, per in accs.items():
        norms[name] = {}
        for d in domains:
            tk = f"teacher_{d}"
            if tk not in accs:
                continue
            vals = [H.normalized_score(per[d][L], base[d][L], accs[tk][d][L]) for L in per[d]]
            vals = [v for v in vals if v == v]
            norms[name][d] = statistics.mean(vals) if vals else float("nan")

    write_md(a, domains, accs, norms)
    print("\n=== headline (avg normalised score) ===", flush=True)
    for name in accs:
        ds = [norms[name][d] for d in domains if norms.get(name, {}).get(d) == norms.get(name, {}).get(d)]
        avg = statistics.mean(ds) if ds else float("nan")
        print(f"  {name:12}: " + "  ".join(f"{d}={norms[name].get(d, float('nan')):.3f}"
                                           for d in domains) + f"   avg={avg:.3f}", flush=True)
    print("hf_evaluate done.", flush=True)


def write_md(a, domains, accs, norms) -> None:
    lines = [
        "# MOPD on FactWorld (Qwen3-1.7B) — headline results\n",
        f"`experiments/mopd/hf_evaluate.py`. {H.MODEL}, greedy relaxed match, n={a.eval_n}. "
        "Normalised `s~ = (model - base)/(teacher - base)`: 0 at the base backbone, 1 at the "
        "per-domain teacher. Headline = uniform mean across domains; one student near 1 on BOTH "
        "domains = MOPD composing both abilities.\n",
        "\n## Normalised score (per domain, mean over eval lengths)\n",
        "| model | " + " | ".join(domains) + " | avg |", "|---|" + "---|" * (len(domains) + 1),
    ]
    for name in accs:
        cells, vals = [], []
        for d in domains:
            v = norms.get(name, {}).get(d, float("nan"))
            cells.append(f"{v:.3f}" if v == v else "—")
            if v == v:
                vals.append(v)
        avg = statistics.mean(vals) if vals else float("nan")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {avg:.3f} |")
    lines.append("\n## Raw accuracy (per domain x length)\n")
    for d in domains:
        Ls = sorted(accs["base"][d])
        lines += [f"\n**{d}**\n", "| model | " + " | ".join(f"L{L}" for L in Ls) + " |",
                  "|---|" + "---|" * len(Ls)]
        for name in accs:
            lines.append(f"| {name} | " + " | ".join(f"{accs[name][d][L]:.3f}" for L in Ls) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
