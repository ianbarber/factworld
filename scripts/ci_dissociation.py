"""3-seed confidence intervals for the dissociation cells.

The single-seed baseline (docs/results.md) shows the load-bearing dissociation — gdn best at isolated
binding (0.78 @4× vs gdp 0.40), gdp best at recall (1.00 vs ≈0.5 for the rest) — but those rest on seed 0.
This reruns just the two dissociation tasks (recall_copy_v1, binding_v2 — the uniform-last-write
binding spec; the recency-defective binding_v1 is retired, see tasks.RETIRED / issue #11) across the 4 archs at seeds
{0,1,2} and reports mean±std per length, writing docs/results-ci.md incrementally so partial progress
survives. (composite_copy floors for all → no CI needed; conflict/chain CIs are a cheap follow-up.)

  .venv/bin/python scripts/ci_dissociation.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK            # noqa: E402
from run_benchmark import run_task           # noqa: E402

ARCHS = ["gdp_hybrid", "gdn_hybrid", "transformer", "gru"]
TASKS = ["recall_copy_v1", "binding_v2"]
SEEDS = [0, 1, 2]
D_MODEL, N_LAYERS, STEPS = 320, 4, 8000
OUT = os.path.join(REPO, "docs", "results-ci.md")


def eval_points(spec):
    train = set(spec.train_lengths)
    return [(L, "id" if L in train else "ood") for L in sorted(train | set(spec.eval_lengths))]


def mean_std(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return m, sd


def write_md(rows):
    lines = [
        "# FactWorld dissociation cells — 3-seed CIs\n",
        f"The two architecture-dissociating tasks across {len(ARCHS)} archs, seeds {SEEDS} "
        f"(mean ± std), d_model={D_MODEL}×{N_LAYERS}, {STEPS} steps, matched compute. Firms up the "
        "single-seed `docs/results.md`: gdn is the binding specialist, gdp the recall/composition "
        "generalist. Columns tagged (id)/(ood).\n",
    ]
    for task in TASKS:
        spec = TK.spec_for(task)
        pts = eval_points(spec)
        lines.append(f"\n## {task}\n")
        lines.append("| arch | " + " | ".join(f"L{L} ({t})" for L, t in pts) + " |")
        lines.append("|" + "---|" * (len(pts) + 1))
        for arch in ARCHS:
            cell = rows.get((task, arch))
            if cell is None:
                body = " | ".join("…" for _ in pts)
            else:
                body = " | ".join(f"{cell[L][0]:.2f}±{cell[L][1]:.2f}" for L, _ in pts)
            lines.append(f"| {arch} | {body} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = {}
    write_md(rows)
    for task in TASKS:
        spec = TK.spec_for(task)
        union = tuple(L for L, _ in eval_points(spec))
        espec = spec.scaled(eval_lengths=union)
        for arch in ARCHS:
            t0 = time.time()
            per_len = {L: [] for L in union}
            for s in SEEDS:
                print(f"[{task}/{arch} seed {s}] training…", flush=True)
                try:
                    acc = run_task(task, spec=espec, arch=arch, d_model=D_MODEL, n_layers=N_LAYERS,
                                   steps=STEPS, seed=s)
                except Exception as e:
                    print(f"  !! {task}/{arch} seed {s} failed: {e}", flush=True)
                    continue
                for L in union:
                    per_len[L].append(acc.get(L, float("nan")))
            rows[(task, arch)] = {L: mean_std(v) if v else (float("nan"), 0.0) for L, v in per_len.items()}
            write_md(rows)
            print(f"  {task}/{arch} done ({time.time() - t0:.0f}s): "
                  + " ".join(f"L{L}={rows[(task, arch)][L][0]:.2f}±{rows[(task, arch)][L][1]:.2f}" for L in union),
                  flush=True)
    print("ci_dissociation done.", flush=True)


if __name__ == "__main__":
    main()
