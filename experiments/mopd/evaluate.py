"""Evaluate — the headline MOPD table: does one distilled student hold BOTH teachers' abilities?

Loads every checkpoint (base, per-domain teachers, MOPD students in both loss forms) and
reports, per domain and length: raw greedy accuracy and the paper's normalised score
  s~_d = (student - base) / (teacher - base)
which is 0 at the shared base and 1 at the per-domain teacher. The headline is the uniform
average of s~_d across domains — a single model scoring ~1 on BOTH domains is MOPD working.

The cross-domain contrast is the point: teacher_binding is (by construction) strong on
binding but is NOT trained on recall, and vice-versa; the MOPD student should match each
teacher ON ITS OWN domain from one set of weights.

Metrics (greedy relaxed match, held-out test):
  acc        raw accuracy
  norm       normalised score (0=base, 1=domain teacher)
  avg-norm   uniform mean of norm across domains (the headline)

  .venv/bin/python experiments/mopd/evaluate.py
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

import mopd as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_n", type=int, default=500)
    ap.add_argument("--domains", default="binding,recall")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return

    domains = a.domains.split(",")
    specs = {d: M.TEACHER_DOMAINS[d]() for d in domains}

    # models: name -> checkpoint file (skip any that are missing)
    wanted = {"base": "base.pt", **{f"teacher_{d}": f"teacher_{d}.pt" for d in domains},
              "mopd_pg": "student_pg.pt", "mopd_kl": "student_kl.pt"}
    accs: dict[str, dict[str, dict[int, float]]] = {}
    tok = None
    for name, fname in wanted.items():
        path = os.path.join(M.CKPT_DIR, fname)
        if not os.path.exists(path):
            print(f"  (skip {name}: {fname} not found)", flush=True)
            continue
        model, tok, _ck = M.load_ckpt(path)
        accs[name] = M.eval_domains(model, tok, specs, n=a.eval_n)
        del model
        torch.cuda.empty_cache()
        print(f"  evaluated {name}", flush=True)

    # normalised scores per model, using base (0) and the matching domain teacher (1) as anchors
    base = accs.get("base", {})
    norms: dict[str, dict[str, float]] = {}
    for name, per in accs.items():
        norms[name] = {}
        for d in domains:
            tkey = f"teacher_{d}"
            if tkey not in accs:
                continue
            Ls = sorted(per[d])
            vals = [M.normalized_score(per[d][L], base.get(d, {}).get(L, 0.0), accs[tkey][d][L])
                    for L in Ls]
            vals = [v for v in vals if v == v]                  # drop NaN (no headroom)
            norms[name][d] = statistics.mean(vals) if vals else float("nan")

    write_md(a, domains, accs, norms)
    print("\n=== headline (avg normalised score across domains) ===", flush=True)
    for name in accs:
        ds = [norms[name][d] for d in domains if d in norms[name] and norms[name][d] == norms[name][d]]
        avg = statistics.mean(ds) if ds else float("nan")
        print(f"  {name:12}: " + "  ".join(
            f"{d}={norms[name].get(d, float('nan')):.3f}" for d in domains) + f"   avg={avg:.3f}",
            flush=True)
    print("evaluate done.", flush=True)


def write_md(a, domains, accs: dict, norms: dict) -> None:
    """Write the raw-accuracy tables + the normalised-score headline to OUT."""
    lines = [
        "# MOPD on FactWorld — headline results\n",
        f"`experiments/mopd/evaluate.py`. Greedy relaxed match, n={a.eval_n}. Normalised score "
        "`s~ = (model - base) / (teacher - base)`: 0 at the shared base, 1 at the per-domain RL "
        "teacher. The headline is the uniform average across domains — one distilled student near "
        "1 on BOTH domains is MOPD composing both abilities.\n",
        "\n## Normalised score (per domain, mean over eval lengths)\n",
        "| model | " + " | ".join(domains) + " | avg |",
        "|---|" + "---|" * (len(domains) + 1),
    ]
    for name in accs:
        cells = []
        vals = []
        for d in domains:
            v = norms.get(name, {}).get(d, float("nan"))
            cells.append(f"{v:.3f}" if v == v else "—")
            if v == v:
                vals.append(v)
        avg = statistics.mean(vals) if vals else float("nan")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {avg:.3f} |")

    lines.append("\n## Raw accuracy (per domain x length)\n")
    for d in domains:
        Ls = sorted(next(iter(accs.values()))[d])
        lines += [f"\n**{d}**\n", "| model | " + " | ".join(f"L{L}" for L in Ls) + " |",
                  "|---|" + "---|" * len(Ls)]
        for name in accs:
            lines.append(f"| {name} | " + " | ".join(f"{accs[name][d][L]:.3f}" for L in Ls) + " |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
