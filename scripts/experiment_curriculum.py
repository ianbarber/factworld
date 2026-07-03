"""Multi-task curriculum: one model, mixed training, per-task eval.

Hypothesis: the full composite (pool=16, binding+deferred recall) may be too hard to
learn from scratch in isolation; blending leg pressures lets sub-circuits form before
routing is required.

Training mix (default weights, configurable via --mix):
  binding      — last-write-wins tracking (no recall map)
  recall       — deferred read-out, pools staged 2→16 in training
  composite_p5 — binding × recall @ pool 5 (known learnable regime)
  composite_p16 — flagship composite @ pool 16

One model per (arch, seed); eval independently on each leg. Compare to single-task
specialists in results/sweep_* and cliff_diag_*.

Example:
    .venv-train/bin/python scripts/experiment_curriculum.py \\
        --seeds 0 1 2 --steps 25000 --d_model 512 --n_layers 8

Mix sweep (once there is signal):
    .venv-train/bin/python scripts/experiment_curriculum.py \\
        --mix binding:0.15,recall:0.35,composite_p5:0.30,composite_p16:0.20 \\
        --out_prefix results/curriculum_mix_sweep_a
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

from factworld import tasks as TK, train as T
from factworld.backends import LocalBackend
from factworld.runner import evaluate_task
from sweep import build_docs, prefix_decomp  # noqa: E402

# Shared world so every arm uses the same agent/value vocabulary.
_SHARED = dict(k=32, value_vocab_size=128, seed=0)


def curriculum_specs():
    """Task specs for training arms and eval (eval uses canonical difficulty)."""
    binding = TK.CANONICAL["binding_v1"].scaled(**_SHARED, name="curriculum_binding")
    recall = TK.CANONICAL["recall_copy_v1"].scaled(
        **_SHARED,
        memorized_recall=False,
        train_lengths=(2, 3, 4, 5, 8, 12, 16),
        eval_lengths=(5, 16),
        name="curriculum_recall",
    )
    composite_p5 = TK.CANONICAL["composite_copy_v1"].scaled(
        **_SHARED, recall_pool=5, name="curriculum_composite_p5",
    )
    composite_p16 = TK.CANONICAL["composite_copy_v1"].scaled(
        **_SHARED, recall_pool=16, name="curriculum_composite_p16",
    )
    return {
        "binding": binding,
        "recall": recall,
        "composite_p5": composite_p5,
        "composite_p16": composite_p16,
    }


def parse_mix(s: str) -> dict[str, float]:
    weights = {}
    for part in s.split(","):
        key, val = part.split(":")
        weights[key.strip()] = float(val.strip())
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"mix weights must sum to 1.0, got {total}")
    unknown = set(weights) - set(curriculum_specs())
    if unknown:
        raise ValueError(f"unknown mix arms: {unknown}")
    return weights


def build_training_docs(specs: dict[str, TK.TaskSpec], weights: dict[str, float], train_n: int):
    """Concatenate per-arm docs with counts proportional to mix weights."""
    docs = []
    arm_counts = {}
    for arm, w in weights.items():
        n = max(1, int(round(train_n * w)))
        arm_counts[arm] = n
        examples = TK.generate(specs[arm], "train", n=n)
        docs.extend(build_docs(examples))
    return docs, arm_counts


def scaffold_prompt(prompt: str, holder: str | None) -> str:
    if holder is None:
        return prompt
    return f"{prompt} (the holder is {holder})"


def eval_task(backend, spec, *, eval_n: int, length: int, scaffolded: bool = False):
    if not scaffolded:
        res = evaluate_task(backend, spec, split="test", n=eval_n, length=length)
        dec = prefix_decomp(res["examples"])
        return {"overall": res["overall"], **dec}
    examples = TK.generate(spec, "test", n=eval_n, length=length)
    prompts = [scaffold_prompt(e.prompt, e.meta.get("holder")) for e in examples]
    preds = backend.generate(prompts, max_new_tokens=8, stop_at=".")
    value_hits = n_two = 0
    for e, pred in zip(examples, preds):
        gold_ct = TK.content_tokens(e.answer)
        if len(gold_ct) < 2:
            continue
        n_two += 1
        if gold_ct[1] in TK.content_tokens(pred):
            value_hits += 1
    return {"scaffolded_value": value_hits / max(1, n_two)}


def eval_grid(backend, specs: dict[str, TK.TaskSpec], *, eval_n: int):
    """Independent per-task eval — the headline readout."""
    g = {}
    g["binding_L16"] = eval_task(backend, specs["binding"], eval_n=eval_n, length=16)
    g["recall_pool5"] = eval_task(backend, specs["recall"], eval_n=eval_n, length=5)
    g["recall_pool16"] = eval_task(backend, specs["recall"], eval_n=eval_n, length=16)
    g["composite_p5_L16"] = eval_task(backend, specs["composite_p5"], eval_n=eval_n, length=16)
    g["composite_p16_L16"] = eval_task(backend, specs["composite_p16"], eval_n=eval_n, length=16)
    g["composite_p16_scaffolded"] = eval_task(
        backend, specs["composite_p16"], eval_n=eval_n, length=16, scaffolded=True,
    )
    return g


def run_one(arch: str, seed: int, *, weights, d_model, n_layers, steps, batch, train_n,
            eval_n, device):
    import torch

    specs = curriculum_specs()
    base = specs["composite_p16"]
    w, r = TK.build_world(base)
    texts, arm_counts = build_training_docs(specs, weights, train_n)
    tok, docs, _ = T.prepare(texts, [], [w], renderer=r)
    d_ff = 4 * d_model
    run = T.run(
        arch, tok, docs, [], steps=steps, batch=batch, d_model=d_model, n_layers=n_layers,
        d_ff=d_ff, seed=seed, return_model=True, device=device,
    )
    backend = LocalBackend([w], arch=arch, model=run["model"], tokenizer=tok, device=device)
    evals = eval_grid(backend, specs, eval_n=eval_n)
    del run["model"]
    torch.cuda.empty_cache()
    return {
        "final_loss": run["final_loss"],
        "arm_counts": arm_counts,
        "n_train_docs": len(texts),
        "eval": evals,
    }


def flatten_eval(evals: dict) -> dict[str, float]:
    flat = {}
    for key, val in evals.items():
        if "scaffolded" in key:
            flat[key] = val["scaffolded_value"]
        else:
            flat[f"{key}_overall"] = val["overall"]
            flat[f"{key}_holder"] = val.get("holder_acc", 0.0)
            flat[f"{key}_value"] = val.get("value_acc", 0.0)
    return flat


def aggregate(runs):
    by_arch = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for k, v in r["flat"].items():
            by_arch[r["arch"]][k].append(v)
    summary = {}
    for arch, metrics in by_arch.items():
        summary[arch] = {
            m: {"mean": statistics.mean(v), "std": statistics.pstdev(v) if len(v) > 1 else 0.0, "n": len(v)}
            for m, v in metrics.items()
        }
    return summary


def write_markdown(summary, cfg, path: Path):
    lines = [
        "# Multi-task curriculum — one model, per-task eval",
        "",
        f"mix={cfg['mix']} d_model={cfg['d_model']} n_layers={cfg['n_layers']} "
        f"steps={cfg['steps']} seeds={cfg['seeds']} train_n={cfg['train_n']}",
        "",
        "## Per-task eval (mean over seeds)",
        "",
        "| arch | binding L16 | recall p5 | recall p16 | comp p5 L16 | comp p16 L16 | p16 holder | p16 value | p16 scaffold |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arch, m in sorted(summary.items()):
        def g(key):
            return m.get(key, {}).get("mean", float("nan"))
        lines.append(
            f"| {arch} | {g('binding_L16_overall'):.2f} | {g('recall_pool5_overall'):.2f} | "
            f"{g('recall_pool16_overall'):.2f} | {g('composite_p5_L16_overall'):.2f} | "
            f"{g('composite_p16_L16_overall'):.2f} | {g('composite_p16_L16_holder'):.2f} | "
            f"{g('composite_p16_L16_value'):.2f} | {g('composite_p16_scaffolded'):.2f} |"
        )
    lines += [
        "",
        "_Compare to single-task specialists: sweep_main_* (per-task training), "
        "cliff_diag_* (pool-16 probes)._",
        "",
        "## Mix sweep",
        "",
        "Re-run with `--mix binding:W,recall:W,composite_p5:W,composite_p16:W` once baseline shows signal.",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Multi-task curriculum training + per-task eval.")
    ap.add_argument("--mix", default="binding:0.25,recall:0.30,composite_p5:0.25,composite_p16:0.20",
                    help="Comma-separated arm:weight (must sum to 1).")
    ap.add_argument("--archs", default="gdp_hybrid,fprm,transformer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_n", type=int, default=8000,
                    help="Total training examples split across arms by mix weights.")
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    weights = parse_mix(a.mix)
    archs = [x.strip() for x in a.archs.split(",")]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = a.out_prefix or f"results/curriculum_{ts}"
    log_path = Path(f"{prefix}.jsonl")
    md_path = Path(f"{prefix}.md")
    json_path = Path(f"{prefix}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "mix": a.mix, "weights": weights, "archs": archs, "seeds": a.seeds,
        "steps": a.steps, "d_model": a.d_model, "n_layers": a.n_layers,
        "train_n": a.train_n, "eval_n": a.eval_n,
    }
    runs = []
    total = len(archs) * len(a.seeds)
    print(f"=== curriculum: {total} runs, mix={a.mix} -> {log_path} ===", flush=True)

    for i, arch in enumerate(archs):
        for seed in a.seeds:
            tag = f"{arch} seed {seed}"
            print(f"\n--- [{len(runs)+1}/{total}] {tag} ---", flush=True)
            try:
                result = run_one(
                    arch, seed, weights=weights, d_model=a.d_model, n_layers=a.n_layers,
                    steps=a.steps, batch=a.batch, train_n=a.train_n, eval_n=a.eval_n,
                    device=a.device,
                )
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                result = {"error": str(e)}

            flat = flatten_eval(result.get("eval", {})) if "eval" in result else {}
            rec = {"arch": arch, "seed": seed, **cfg, **result, "flat": flat}
            with log_path.open("a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            if flat:
                runs.append(rec)
                print(f"    -> {flat}  loss={result.get('final_loss', 'n/a')}", flush=True)

            summary = aggregate(runs)
            write_markdown(summary, cfg, md_path)
            json_path.write_text(json.dumps({"cfg": cfg, "summary": summary, "runs": runs}, indent=2))

    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()