"""Reference baselines for the scored FactWorld suite → docs/results.md (incremental).

A benchmark without baselines is just a task generator. This trains each scored (REPORTED)
task from scratch under a fixed, modest, compute-matched recipe for two principal architectures
(product-structured recurrent hybrid vs transformer) and records the canonical relaxed-match accuracy
at every eval length. Results are written to docs/results.md after EACH cell, so a crash or interrupt keeps
all completed numbers. This is the baseline scale (d320x4, ~8k steps) — composite tasks are expected to
sit near floor here (cf. §5 for the scale step that lifts the flagship composite).

  .venv/bin/python scripts/collect_baselines.py
"""
import os
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK            # noqa: E402
from run_benchmark import run_task           # noqa: E402

ARCHS = ["gdp_hybrid", "gdn_hybrid", "transformer", "gru"]  # the 4 reference archs (mamba2 dropped: naive kernels OOM at L64)
D_MODEL, N_LAYERS, STEPS, SEED = 320, 4, 8000, 0
OUT = os.path.join(REPO, "docs", "results.md")

# scored tasks, in the suite's reported order
TASKS = [n for n in TK.CANONICAL if TK.CANONICAL[n].kind == "benchmark"]


def eval_points(spec):
    """All lengths to report = train ∪ eval, sorted; each tagged id (in-distribution) or ood."""
    train = set(spec.train_lengths)
    pts = sorted(train | set(spec.eval_lengths))
    return [(L, "id" if L in train else "ood") for L in pts]


def empirical_floor(spec):
    """1 / (#distinct gold answers in a sample) at the deepest eval length — the random-guess baseline."""
    test = TK.generate(spec, "test", n=500, length=spec.eval_lengths[-1])
    return 1.0 / len(Counter(e.answer.split()[0] for e in test))


def write_md(rows, floors):
    """rows: {(task, arch): {L: acc}}. Rebuild the whole file from accumulated rows each time."""
    lines = [
        "# FactWorld reference baselines\n",
        f"Scored suite (`REPORTED`), from-scratch, relaxed match (canonical metric; exact/contains/last_n "
        f"are diagnostics). Recipe: "
        f"d_model={D_MODEL}, n_layers={N_LAYERS}, {STEPS} steps, seed={SEED}, **matched across "
        "architectures**. This is the **baseline scale**; composite tasks are expected near floor here "
        "(see §5 for the scale step that lifts the flagship composite). Columns are eval lengths tagged "
        "**(id)** in-distribution / **(ood)** held-out; `length` = depth for chain_v1, #facts for "
        "recall_copy_v1/conflict_v1, binding-chain length otherwise.\n",
    ]
    for task in TASKS:
        spec = TK.CANONICAL[task]
        pts = eval_points(spec)
        fl = floors.get(task)
        lines.append(f"\n## {task}  (floor ≈ {('%.3f' % fl) if fl else '?'})\n")
        header = "| arch | " + " | ".join(f"L{L} ({tag})" for L, tag in pts) + " |"
        sep = "|" + "---|" * (len(pts) + 1)
        lines += [header, sep]
        for arch in ARCHS:
            acc = rows.get((task, arch))
            cells = (" | ".join("…" for _ in pts) if acc is None
                     else " | ".join(f"{acc.get(L, float('nan')):.3f}" for L, _ in pts))
            lines.append(f"| {arch} | {cells} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    floors = {t: empirical_floor(TK.CANONICAL[t]) for t in TASKS}
    rows = {}
    write_md(rows, floors)  # skeleton up front
    for task in TASKS:
        spec = TK.CANONICAL[task]
        union = tuple(L for L, _ in eval_points(spec))
        eval_spec = spec.scaled(eval_lengths=union)        # train once, eval at id ∪ ood lengths
        for arch in ARCHS:
            t0 = time.time()
            print(f"[{task} / {arch}] training…", flush=True)
            try:
                acc = run_task(task, spec=eval_spec, arch=arch, d_model=D_MODEL, n_layers=N_LAYERS,
                               steps=STEPS, seed=SEED)
            except Exception as e:  # keep going; record the failure cell as empty
                print(f"  !! {task}/{arch} failed: {e}", flush=True)
                continue
            rows[(task, arch)] = acc
            write_md(rows, floors)  # persist after every cell
            print(f"  {task}/{arch} {acc}  ({time.time() - t0:.0f}s)", flush=True)
    print("baselines done.", flush=True)


if __name__ == "__main__":
    main()
