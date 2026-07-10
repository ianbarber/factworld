"""Local calibration for the commutative rung (commutative_v1) — arch x seed at binding's
operating point.

The v3.1 taxonomy ladder is retrieval < last-write < commutative < non-abelian; commutative_v1
(per-entity dial accumulation mod 5, every event load-bearing, order irrelevant) fills the
abelian rung. This sweep asks: does any small architecture buy the commutative fold, and which
one? 3 archs (transformer, gdp_hybrid, fprm — fprm is weight-tied: compare on FLOPs, not
params) x 3 seeds at the d256x4 house recipe, on the CANONICAL spec as-is (k_positions=5, m=4
— floors matched to binding_v2: chance 0.2, working set 4). Eval at L16 (in-distribution
edge), L32 and L64 (OOD aggregation-depth extrapolation).

READ AGAINST: binding_v2 local numbers at the same d256x4 recipe, and the four documented
shallow floors (factworld.validity.comm_shallow_accuracy — the strongest sits ~0.20-0.22, so a
run only "solves" if it clears that band). Composite-style bimodality is possible: read
p(converge), not just the mean. SUCCESS CRITERION for the rung: it discriminates SOMEWHERE —
an arch split at L16, or an extrapolation split at L32/64.

CONTINGENCY (--use_trace, run only if all 9 answer-only runs floor in-distribution like s5
did): dense per-step supervision via spec.scaled(worked_trace=True) — the s5 lesson says the
fold may need per-step supervision to form; one gdp_hybrid seed is the diagnostic.

Crash-safe + resumable (experiment_local_breadth idiom): every run appends to
results/commutative_local/<tag>_runs.jsonl; on restart any (arch, seed, steps, trace) already
logged is skipped. GPU-serialized. Summary markdown/JSON rewritten after every run.

Smoke (prove the pipeline, ~1-2 min):
    .venv-train/bin/python scripts/experiment_commutative_local.py \
        --archs gdp_hybrid --seeds 0 --steps 1000 --eval_n 50 --tag smoke

Full sweep (3 archs x 3 seeds = 9 runs, ~1.5-3h on the RTX 5090 — evening slot, launch
DETACHED after thread-1's story-critical GPU work):
    nohup .venv-train/bin/python scripts/experiment_commutative_local.py \
        > results/commutative_local/sweep.log 2>&1 &
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
from sweep import run_one  # house recipe: train + eval + prefix decomposition

OUT_DIR = Path(REPO) / "results" / "commutative_local"
TASK = "commutative_v1"


def run_key(rec) -> tuple:
    return (rec["arch"], rec["seed"], rec["steps"], rec.get("use_trace", False))


def load_done(log_path: Path) -> dict:
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
    """Per (arch, length): mean+-std of relaxed overall + p(converge) (bimodality-aware)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for L, v in r["lengths"].items():
            by[r["arch"]][L].append(v["overall"])
    summary = {}
    for arch, lens in sorted(by.items()):
        summary[arch] = {}
        for L, ov in sorted(lens.items(), key=lambda kv: int(kv[0])):
            summary[arch][L] = {
                "mean": statistics.mean(ov),
                "std": statistics.pstdev(ov) if len(ov) > 1 else 0.0,
                "n": len(ov),
                "p_converge": sum(1 for x in ov if x >= 0.9) / len(ov),
            }
    return summary


def write_markdown(summary, cfg, path: Path):
    lens = [str(L) for L in cfg["eval_lengths"]]
    lines = [
        "# Commutative rung — local calibration (commutative_v1, arch x seed, d256x4)",
        "",
        f"d_model={cfg['d_model']} n_layers={cfg['n_layers']} steps={cfg['steps']} "
        f"batch={cfg['batch']} train_n={cfg['train_n']} eval_n={cfg['eval_n']} "
        f"seeds={cfg['seeds']} train_lengths={cfg['train_lengths']} use_trace={cfg['use_trace']}",
        "",
        "| arch | " + " | ".join(f"L{L} mean±std (pconv)" for L in lens) + " |",
        "|" + "---|" * (1 + len(lens)),
    ]
    for arch, ld in sorted(summary.items()):
        cells = [f"{ld[L]['mean']:.2f}±{ld[L]['std']:.2f} ({ld[L]['p_converge']:.0%})"
                 for L in lens if L in ld]
        lines.append(f"| {arch} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "_Relaxed match (canonical). pconv = fraction of seeds >=0.9. Floors as rows: chance "
        "1/k_positions = 0.200; the strongest of the four shallow adversaries sits ~0.20-0.22 "
        "(all four gated <= 0.4 by scripts/validate_suite.py) — a run only 'solves' if it clears "
        "that band. L16 is in-distribution, L32/L64 are OOD aggregation depth. fprm is "
        "weight-tied: compare on FLOPs, not params._",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Local commutative_v1 calibration sweep (arch x seed).")
    ap.add_argument("--archs", default="transformer,gdp_hybrid,fprm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)      # documented d256 recipe (scripts/sweep.py)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--eval_lengths", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--use_trace", action="store_true",
                    help="Dense-supervision contingency: train with the oracle position trace "
                         "(spec.scaled(worked_trace=True)); the s5 lesson — run one gdp_hybrid "
                         "seed only if every answer-only run floors in-distribution.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="sweep")
    a = ap.parse_args()

    archs = [x.strip() for x in a.archs.split(",")]
    eval_lengths = tuple(a.eval_lengths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"{a.tag}_runs.jsonl"
    md_path = OUT_DIR / f"{a.tag}_summary.md"
    json_path = OUT_DIR / f"{a.tag}_summary.json"

    spec = TK.spec_for(TASK).scaled(eval_lengths=eval_lengths, worked_trace=a.use_trace)
    cfg = {"task": TASK, "archs": archs, "seeds": a.seeds, "steps": a.steps,
           "d_model": a.d_model, "n_layers": a.n_layers, "batch": a.batch,
           "train_n": a.train_n, "eval_n": a.eval_n, "eval_lengths": list(eval_lengths),
           "train_lengths": list(spec.train_lengths), "use_trace": a.use_trace,
           "k_positions": spec.k_positions, "m": spec.n_objects_active}

    done = load_done(log_path)
    runs = list(done.values())
    grid = [(arch, seed) for arch in archs for seed in a.seeds]
    total = len(grid)
    print(f"=== commutative local calibration: {total} runs ({len(done)} already done) -> {log_path} ===", flush=True)
    print(f"    cfg: {cfg}", flush=True)

    for i, (arch, seed) in enumerate(grid):
        if (arch, seed, a.steps, a.use_trace) in done:
            print(f"--- [{i+1}/{total}] {arch} | seed {seed} : already done, skipping ---", flush=True)
            continue
        t0 = datetime.now(timezone.utc)
        print(f"\n--- [{i+1}/{total}] {arch} | seed {seed} | start {t0.isoformat(timespec='seconds')} ---", flush=True)
        try:
            r = run_one(spec, arch, seed, d_model=a.d_model, n_layers=a.n_layers, steps=a.steps,
                        batch=a.batch, train_n=a.train_n, eval_n=a.eval_n,
                        use_short_conv=False, use_trace=a.use_trace, device=a.device)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            r = {"error": str(e)}
        mins = (datetime.now(timezone.utc) - t0).total_seconds() / 60
        rec = {"arch": arch, "seed": seed, **cfg, "wall_min": round(mins, 2), **r}
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
