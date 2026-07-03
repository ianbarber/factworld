"""Grid sweep over architecture and training hyperparameters for staged curriculum.

Useful for quickly filtering promising configs with short runs before committing to
full 25k-step multi-seed training.

Example width × batch sweep at 10k steps:
    .venv-train/bin/python scripts/experiment_curriculum_grid.py \
        --grid "d_model:512,768,1024;n_layers:8;batch:32,64;steps:10000;seeds:0" \
        --archs gdp_hybrid

The grid is the cartesian product of every comma-separated value list. One seed per
config by default; add more seeds to the list if you want variance estimates.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from experiment_curriculum_staged import (
    default_schedule,
    eval_grid,
    flatten_eval,
    staged_specs,
    train_stages,
)
from factworld import tasks as TK, train as T


def parse_grid(s: str) -> list[dict]:
    """Parse a grid spec into a list of config dicts.

    Format: key:v1,v2;key2:v1;...
    Values that look like ints or floats are converted. ``seeds`` becomes a list of ints.
    """
    axes = {}
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        key, vals = part.split(":", 1)
        key = key.strip()
        parsed = []
        for v in vals.split(","):
            v = v.strip()
            try:
                parsed.append(int(v))
            except ValueError:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
        axes[key] = parsed

    # seeds is special: list of ints
    if "seeds" in axes and not isinstance(axes["seeds"][0], int):
        axes["seeds"] = [int(x) for x in axes["seeds"]]
    if "seeds" not in axes:
        axes["seeds"] = [0]

    keys = [k for k in axes if k != "seeds"]
    seed_list = axes["seeds"]
    configs = []
    for vals in product(*(axes[k] for k in keys)):
        base = dict(zip(keys, vals))
        for seed in seed_list:
            cfg = dict(base)
            cfg["seed"] = seed
            configs.append(cfg)
    return configs


def cfg_to_str(cfg: dict) -> str:
    """Short deterministic label for a config."""
    return "_".join(f"{k}{v}" for k, v in sorted(cfg.items()) if k != "seed")


def run_grid(arch: str, configs: list[dict], *, tok, specs, eval_n: int, device: str,
             loss_log_interval: int, schedule, use_trace: bool, dense_format: str):
    """Run every config and return per-config results."""
    results = []
    for i, cfg in enumerate(configs):
        seed = cfg["seed"]
        d_model = cfg["d_model"]
        n_layers = cfg["n_layers"]
        batch = cfg["batch"]
        steps = cfg["steps"]
        train_n = cfg.get("train_n", 8000)
        label = cfg_to_str(cfg)
        print(f"\n=== [{i+1}/{len(configs)}] {arch} {label} seed{seed} ===", flush=True)
        try:
            stage_records, model = train_stages(
                arch, seed, schedule, tok, specs, d_model=d_model, n_layers=n_layers,
                batch=batch, train_n=train_n, eval_n=eval_n, device=device,
                loss_log_interval=loss_log_interval,
                use_trace=use_trace, dense_format=dense_format,
            )
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({"cfg": cfg, "label": label, "error": str(e)})
            continue
        import torch
        final_eval = stage_records[-1]["eval"]
        flat = flatten_eval(final_eval)
        results.append({
            "cfg": cfg,
            "label": label,
            "stage_records": stage_records,
            "final_loss": stage_records[-1]["final_loss"],
            "final_eval": final_eval,
            "flat": flat,
        })
        print(f"    -> {flat}  loss={stage_records[-1]['final_loss']}", flush=True)
        del model
        torch.cuda.empty_cache()
    return results


def aggregate_grid(results):
    """Group by non-seed config and average over seeds."""
    by_label = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "flat" not in r:
            continue
        label = r["label"]
        for k, v in r["flat"].items():
            by_label[label][k].append(v)
    summary = {}
    for label, metrics in by_label.items():
        summary[label] = {
            m: {"mean": statistics.mean(v), "std": statistics.pstdev(v) if len(v) > 1 else 0.0, "n": len(v)}
            for m, v in metrics.items()
        }
    return summary


def write_grid_markdown(results, summary, grid_spec: str, path: Path):
    lines = [
        "# Staged curriculum — architecture/batch grid sweep",
        "",
        f"grid={grid_spec}",
        "",
        "## Per-config final eval (mean over seeds)",
        "",
        "| config | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 holder | p16 value | p16 scaffold | final loss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in sorted(summary):
        m = summary[label]
        def g(key):
            return m.get(key, {}).get("mean", float("nan"))
        loss_vals = [r["final_loss"] for r in results if r.get("label") == label and "final_loss" in r]
        loss_mean = statistics.mean(loss_vals) if loss_vals else float("nan")
        lines.append(
            f"| {label} | {g('binding_L16_overall'):.2f} | {g('recall_easy_L4_overall'):.2f} | "
            f"{g('recall_med_L8_overall'):.2f} | {g('recall_hard_L16_overall'):.2f} | "
            f"{g('composite_p5_L16_overall'):.2f} | {g('composite_p16_L16_overall'):.2f} | "
            f"{g('composite_p16_L16_holder'):.2f} | {g('composite_p16_L16_value'):.2f} | "
            f"{g('composite_p16_scaffolded'):.2f} | {loss_mean:.3f} |"
        )
    lines += ["", "## Raw results", "", "```json", json.dumps(results, default=float, indent=2), "```"]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Grid sweep over staged-curriculum configs.")
    ap.add_argument("--grid", required=True,
                    help="Semicolon-separated axes. Example: d_model:512,768;n_layers:8;batch:32,64;steps:10000")
    ap.add_argument("--archs", default="gdp_hybrid")
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loss_log_interval", type=int, default=200)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--use_trace", action="store_true")
    ap.add_argument("--dense_format", default="trace", choices=["trace", "interleaved", "marker"])
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    configs = parse_grid(a.grid)
    # Inject shared defaults if not in grid.
    for cfg in configs:
        cfg.setdefault("train_n", a.train_n)

    schedule = default_schedule(max(cfg["steps"] for cfg in configs))
    archs = [x.strip() for x in a.archs.split(",")]

    specs = staged_specs()
    base = specs["composite_p16"]
    w, r = TK.build_world(base)
    tok, _, _ = T.prepare([], [], [w], renderer=r)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = a.out_prefix or f"results/grid_{ts}"
    log_path = Path(f"{prefix}.jsonl")
    md_path = Path(f"{prefix}.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for arch in archs:
        results = run_grid(
            arch, configs, tok=tok, specs=specs, eval_n=a.eval_n, device=a.device,
            loss_log_interval=a.loss_log_interval, schedule=schedule,
            use_trace=a.use_trace, dense_format=a.dense_format,
        )
        for r in results:
            r["arch"] = arch
            with log_path.open("a") as f:
                f.write(json.dumps(r, default=float) + "\n")
        all_results.extend(results)

    summary = aggregate_grid(all_results)
    write_grid_markdown(all_results, summary, a.grid, md_path)
    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()
