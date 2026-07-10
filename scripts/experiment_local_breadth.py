"""Local breadth-mirror sweep (v3 stage 0): the frontier breadth staircase, mirrored on tiny local models.

The frontier probes walk composite_copy_v2 breadth rungs via spec.scaled(k=2*B, recall_pool=B).
This mirrors the SAME rung protocol on from-scratch d256x4 local models (transformer vs gdp_hybrid)
so the working-set-breadth constructs (order-breadth / lookup-breadth) get a local capacity anchor:

  B in {6, 8, 12, 16, 24}, m = n_objects_active = 4 (spec default), memorized_recall=False,
  train lengths (4, 8, 16), eval at L16 (in-distribution) and L64 (OOD extrapolation),
  3 seeds per (arch, rung) -- composite is documented BIMODAL, so p(converge) is the headline,
  per-run relaxed match (canonical) + holder/value leg decomposition recorded per length.

Crash-safe + resumable: every completed run is appended to results/local_breadth/runs.jsonl as it
finishes, and on restart any (B, arch, seed, steps) already logged there is skipped. Runs execute
strictly one at a time (GPU-serialized). Summary markdown/JSON are rewritten after every run.

Smoke (prove the pipeline end-to-end, ~1-2 min):
    .venv-train/bin/python scripts/experiment_local_breadth.py \
        --rungs 6 --archs gdp_hybrid --seeds 0 --steps 1000 --eval_n 50 --tag smoke

Full overnight sweep (2 archs x 5 rungs x 3 seeds = 30 runs, ~5-10h):
    nohup .venv-train/bin/python scripts/experiment_local_breadth.py \
        > results/local_breadth/sweep.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from sweep import run_one  # house recipe: train + eval + prefix (holder/value) decomposition

OUT_DIR = Path(REPO) / "results" / "local_breadth"
BASE_TASK = "composite_copy_v2"


def rung_spec(B: int, eval_lengths: tuple):
    """The frontier breadth-rung protocol: k=2*B agents, 1-of-B lookup pool, m=4 interference."""
    return TK.spec_for(BASE_TASK).scaled(
        k=2 * B, recall_pool=B, memorized_recall=False, eval_lengths=eval_lengths,
    )


def run_key(rec) -> tuple:
    return (rec["B"], rec["arch"], rec["seed"], rec["steps"])


def load_done(log_path: Path) -> dict:
    """(B, arch, seed, steps) -> record, for completed (non-error) runs already in the log."""
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "lengths" in rec:
                done[run_key(rec)] = rec
    return done


def aggregate(runs):
    """Per (B, arch, length): mean+-std of relaxed overall, p(converge), mean holder/value legs."""
    by = defaultdict(lambda: defaultdict(list))     # (B, arch)[L] -> [overall,...]
    dec = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        key = (r["B"], r["arch"])
        for L, v in r["lengths"].items():
            by[key][L].append(v["overall"])
            for dk in ("holder_acc", "value_acc"):
                dec[key][L][dk].append(v[dk])
    summary = {}
    for (B, arch), lens in sorted(by.items()):
        summary.setdefault(str(B), {})[arch] = {}
        for L, ov in sorted(lens.items(), key=lambda kv: int(kv[0])):
            summary[str(B)][arch][L] = {
                "mean": statistics.mean(ov),
                "std": statistics.pstdev(ov) if len(ov) > 1 else 0.0,
                "n": len(ov),
                "p_converge": sum(1 for x in ov if x >= 0.9) / len(ov),
                "holder_acc": statistics.mean(dec[(B, arch)][L]["holder_acc"]),
                "value_acc": statistics.mean(dec[(B, arch)][L]["value_acc"]),
            }
    return summary


def write_markdown(summary, cfg, path: Path):
    lens = [str(L) for L in cfg["eval_lengths"]]
    lines = [
        "# Local breadth-mirror — composite_copy_v2.scaled(k=2B, recall_pool=B), m=4",
        "",
        f"d_model={cfg['d_model']} n_layers={cfg['n_layers']} steps={cfg['steps']} "
        f"batch={cfg['batch']} train_n={cfg['train_n']} eval_n={cfg['eval_n']} "
        f"seeds={cfg['seeds']} train_lengths={cfg['train_lengths']}",
        "",
        "| B | arch | " + " | ".join(f"L{L} mean±std (pconv)" for L in lens)
        + " | " + " | ".join(f"holder/value @L{L}" for L in lens) + " |",
        "|" + "---|" * (2 + 2 * len(lens)),
    ]
    for B, archs in sorted(summary.items(), key=lambda kv: int(kv[0])):
        for arch, ld in sorted(archs.items()):
            cells = [f"{ld[L]['mean']:.2f}±{ld[L]['std']:.2f} ({ld[L]['p_converge']:.0%})" for L in lens if L in ld]
            cells += [f"{ld[L]['holder_acc']:.2f} / {ld[L]['value_acc']:.2f}" for L in lens if L in ld]
            lines.append(f"| {B} | {arch} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "_Relaxed match (canonical). pconv = fraction of seeds >=0.9 (composite is bimodal — read pconv, "
        "not the mean). holder = binding leg, value = lookup leg. L16 is in-distribution, L64 is OOD "
        "extrapolation. Per-rung object-filter floor E[1/w] moves with (m, L) not pool: ~0.41@L16 / "
        "~0.15@L64 at m=4._",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Local breadth-mirror sweep on composite_copy_v2 rungs.")
    ap.add_argument("--rungs", type=int, nargs="+", default=[6, 8, 12, 16, 24])
    ap.add_argument("--archs", default="transformer,gdp_hybrid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)      # documented d256 recipe (scripts/sweep.py)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--eval_lengths", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="sweep", help="Log-file basename tag (use 'smoke' for the pipeline check).")
    a = ap.parse_args()

    archs = [x.strip() for x in a.archs.split(",")]
    eval_lengths = tuple(a.eval_lengths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"{a.tag}_runs.jsonl"
    md_path = OUT_DIR / f"{a.tag}_summary.md"
    json_path = OUT_DIR / f"{a.tag}_summary.json"

    cfg = {"rungs": a.rungs, "archs": archs, "seeds": a.seeds, "steps": a.steps,
           "d_model": a.d_model, "n_layers": a.n_layers, "batch": a.batch,
           "train_n": a.train_n, "eval_n": a.eval_n, "eval_lengths": list(eval_lengths),
           "train_lengths": list(TK.spec_for(BASE_TASK).train_lengths)}

    done = load_done(log_path)
    runs = list(done.values())
    grid = [(B, arch, seed) for B in a.rungs for arch in archs for seed in a.seeds]
    total = len(grid)
    print(f"=== local breadth-mirror: {total} runs ({len(done)} already done) -> {log_path} ===", flush=True)
    print(f"    cfg: {cfg}", flush=True)

    for i, (B, arch, seed) in enumerate(grid):
        if (B, arch, seed, a.steps) in done:
            print(f"--- [{i+1}/{total}] B{B} | {arch} | seed {seed} : already done, skipping ---", flush=True)
            continue
        spec = rung_spec(B, eval_lengths)
        t0 = datetime.now(timezone.utc)
        print(f"\n--- [{i+1}/{total}] B{B} (k={spec.k}, pool={spec.recall_pool}) | {arch} | seed {seed} "
              f"| start {t0.isoformat(timespec='seconds')} ---", flush=True)
        try:
            r = run_one(spec, arch, seed, d_model=a.d_model, n_layers=a.n_layers, steps=a.steps,
                        batch=a.batch, train_n=a.train_n, eval_n=a.eval_n,
                        use_short_conv=False, use_trace=False, device=a.device)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            r = {"error": str(e)}
        mins = (datetime.now(timezone.utc) - t0).total_seconds() / 60
        rec = {"task": BASE_TASK, "B": B, "k": spec.k, "recall_pool": spec.recall_pool,
               "m": spec.n_objects_active, "arch": arch, "seed": seed, **cfg,
               "wall_min": round(mins, 2), **r}
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if "lengths" in r:
            runs.append(rec)
            ov = {L: round(v["overall"], 3) for L, v in r["lengths"].items()}
            print(f"    -> {ov}  loss={r['final_loss']:.3f}  ({mins:.1f} min)", flush=True)
        else:
            print(f"    -> ERROR: {r['error']}  ({mins:.1f} min)", flush=True)
        summary = aggregate(runs)
        write_markdown(summary, cfg, md_path)
        json_path.write_text(json.dumps({"cfg": cfg, "summary": summary, "runs": runs}, indent=2))

    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()
