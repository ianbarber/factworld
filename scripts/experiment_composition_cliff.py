"""Composition cliff diagnostics — localize where the recall leg breaks.

Three probes to separate binding, standalone recall, and coupled composition:

  (P1) ceiling_k32 — train composite_copy_v1 (k=32, pool=16); eval end-to-end AND
       scaffolded (correct holder injected, value-only score). If scaffolded >> e2e,
       the wall is the coupled pipeline, not raw pool-16 recall capacity.

  (P2) recall_pool16 — deferred read-out standalone at pool=16 (no binding stream).
       Trains with pool sizes up to 16 so recall exposure matches the composite task.
       If this is learnable but composite is not, composition coupling is the extra cost.

  (P3) pool_sweep — composite_copy_v1 with recall_pool in {8, 12, 16} at matched
       ~45M scale. Maps where architecture separation re-emerges on the composition
       axis (pool=5 separates at this scale; pool=16 floors for all).

Use the pool where legs separate for architecture recipes; use leg probes to explain
why the k=32 flagship is past the local training frontier.

Example:
    .venv-train/bin/python scripts/experiment_composition_cliff.py \\
        --probes ceiling_k32,recall_pool16,pool_sweep \\
        --seeds 0 1 2 --d_model 512 --n_layers 8 --steps 25000
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK, train as T
from factworld.backends import LocalBackend
from factworld.render import Renderer
from factworld.runner import evaluate_task

# Reuse sweep helpers
sys.path.insert(0, os.path.join(REPO, "scripts"))
from sweep import build_docs, prefix_decomp  # noqa: E402


def scaffold_prompt(prompt: str, holder: str | None) -> str:
    if holder is None:
        return prompt
    return f"{prompt} (the holder is {holder})"


def eval_e2e(backend, spec, *, eval_n, length):
    res = evaluate_task(backend, spec, split="test", n=eval_n, length=length)
    dec = prefix_decomp(res["examples"])
    return {"overall": res["overall"], **dec}


def eval_scaffolded(backend, spec, *, eval_n, length):
    """Recall-leg ceiling: holder given in the prompt; score value token presence."""
    examples = TK.generate(spec, "test", n=eval_n, length=length)
    prompts = [scaffold_prompt(e.prompt, e.meta.get("holder")) for e in examples]
    preds = backend.generate(prompts, max_new_tokens=8, stop_at=".")
    value_hits = 0
    n_two = 0
    for e, pred in zip(examples, preds):
        gold_ct = TK.content_tokens(e.answer)
        if len(gold_ct) < 2:
            continue
        n_two += 1
        value = gold_ct[1]
        if value in TK.content_tokens(pred):
            value_hits += 1
    return {
        "value_acc": value_hits / max(1, n_two),
        "n": n_two,
    }


def train_and_eval(spec, arch, seed, *, d_model, n_layers, steps, batch, train_n, eval_n,
                   device, probe: str):
    import torch

    d_ff = 4 * d_model
    w, r = TK.build_world(spec)
    train = TK.generate(spec, "train", n=train_n)
    tok, docs, _ = T.prepare(build_docs(train), [], [w], renderer=r)
    run = T.run(
        arch, tok, docs, [], steps=steps, batch=batch, d_model=d_model, n_layers=n_layers,
        d_ff=d_ff, seed=seed, return_model=True, device=device,
    )
    backend = LocalBackend([w], arch=arch, model=run["model"], tokenizer=tok, device=device)
    out: dict = {"final_loss": run["final_loss"]}
    eval_L = spec.eval_lengths[0]

    if probe in ("ceiling_k32", "pool_sweep"):
        out["e2e"] = eval_e2e(backend, spec, eval_n=eval_n, length=eval_L)
    if probe == "ceiling_k32":
        out["scaffolded"] = eval_scaffolded(backend, spec, eval_n=eval_n, length=eval_L)
    if probe == "recall_pool16":
        out["recall"] = eval_e2e(backend, spec, eval_n=eval_n, length=eval_L)

    del run["model"]
    torch.cuda.empty_cache()
    return out


def specs_for_probe(probe: str):
    if probe == "ceiling_k32":
        return [("ceiling_k32", TK.RETIRED["composite_copy_v1"], None)]
    if probe == "recall_pool16":
        spec = TK.CANONICAL["recall_copy_v1"].scaled(
            k=32,
            value_vocab_size=128,
            train_lengths=(4, 8, 12, 16),
            eval_lengths=(16,),
        )
        return [("recall_pool16", spec, 16)]
    if probe == "pool_sweep":
        return [
            ("pool_sweep", TK.RETIRED["composite_copy_v1"].scaled(recall_pool=p), p)
            for p in (8, 12, 16)
        ]
    raise ValueError(f"unknown probe: {probe}")


def aggregate(runs):
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        key = (r["probe"], r.get("pool"), r["arch"])
        for metric, val in r["metrics"].items():
            by[key][metric].append(val)
    summary = {}
    for key, metrics in by.items():
        probe, pool, arch = key
        label = f"{probe}|pool{pool}" if pool is not None else probe
        summary.setdefault(label, {})[arch] = {
            m: {"mean": statistics.mean(v), "std": statistics.pstdev(v) if len(v) > 1 else 0.0, "n": len(v)}
            for m, v in metrics.items()
        }
    return summary


def write_markdown(summary, cfg, path: Path):
    lines = [
        "# Composition cliff diagnostics",
        "",
        f"d_model={cfg['d_model']} n_layers={cfg['n_layers']} steps={cfg['steps']} "
        f"seeds={cfg['seeds']} probes={cfg['probes']}",
        "",
    ]
    for label, archs in summary.items():
        lines.append(f"## {label}")
        if "ceiling" in label:
            lines.append("| arch | e2e overall | holder | value | scaffolded value |")
            lines.append("| --- | --- | --- | --- | --- |")
            for arch, m in sorted(archs.items()):
                e2e = m.get("e2e_overall", {}).get("mean", float("nan"))
                h = m.get("holder_acc", {}).get("mean", float("nan"))
                v = m.get("value_acc", {}).get("mean", float("nan"))
                sv = m.get("scaffolded_value", {}).get("mean", float("nan"))
                lines.append(f"| {arch} | {e2e:.2f} | {h:.2f} | {v:.2f} | {sv:.2f} |")
        elif "recall" in label:
            lines.append("| arch | recall @pool16 |")
            lines.append("| --- | --- |")
            for arch, m in sorted(archs.items()):
                acc = m.get("recall_overall", {}).get("mean", float("nan"))
                lines.append(f"| {arch} | {acc:.2f} |")
        else:
            lines.append("| arch | e2e overall | holder | value | p(conv) |")
            lines.append("| --- | --- | --- | --- | --- |")
            for arch, m in sorted(archs.items()):
                e2e = m.get("e2e_overall", {}).get("mean", float("nan"))
                h = m.get("holder_acc", {}).get("mean", float("nan"))
                v = m.get("value_acc", {}).get("mean", float("nan"))
                conv = m.get("e2e_converge", {}).get("mean", float("nan"))
                lines.append(f"| {arch} | {e2e:.2f} | {h:.2f} | {v:.2f} | {conv:.0%} |")
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Composition cliff diagnostics (P1/P2/P3).")
    ap.add_argument("--probes", default="ceiling_k32,recall_pool16,pool_sweep",
                    help="Comma-separated: ceiling_k32, recall_pool16, pool_sweep")
    ap.add_argument("--archs", default="gdp_hybrid,fprm,transformer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    probes = [p.strip() for p in a.probes.split(",")]
    archs = [x.strip() for x in a.archs.split(",")]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = a.out_prefix or f"results/cliff_diag_{ts}"
    log_path = Path(f"{prefix}.jsonl")
    md_path = Path(f"{prefix}.md")
    json_path = Path(f"{prefix}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for probe in probes:
        for probe_name, spec, pool in specs_for_probe(probe):
            for arch in archs:
                for seed in a.seeds:
                    jobs.append((probe_name, spec, pool, arch, seed))

    cfg = {
        "probes": probes, "archs": archs, "seeds": a.seeds, "steps": a.steps,
        "d_model": a.d_model, "n_layers": a.n_layers, "train_n": a.train_n, "eval_n": a.eval_n,
    }
    runs = []
    total = len(jobs)
    print(f"=== cliff diagnostics: {total} runs -> {log_path} ===", flush=True)

    for i, (probe_name, spec, pool, arch, seed) in enumerate(jobs):
        tag = f"{probe_name} pool={pool} | {arch} | seed {seed}"
        print(f"\n--- [{i+1}/{total}] {tag} ---", flush=True)
        try:
            metrics = train_and_eval(
                spec, arch, seed, d_model=a.d_model, n_layers=a.n_layers, steps=a.steps,
                batch=a.batch, train_n=a.train_n, eval_n=a.eval_n, device=a.device,
                probe=probe_name,
            )
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            metrics = {"error": str(e)}

        flat = {}
        if "e2e" in metrics:
            flat["e2e_overall"] = metrics["e2e"]["overall"]
            flat["holder_acc"] = metrics["e2e"]["holder_acc"]
            flat["value_acc"] = metrics["e2e"]["value_acc"]
            flat["e2e_converge"] = float(metrics["e2e"]["overall"] >= 0.9)
        if "scaffolded" in metrics:
            flat["scaffolded_value"] = metrics["scaffolded"]["value_acc"]
        if "recall" in metrics:
            flat["recall_overall"] = metrics["recall"]["overall"]

        rec = {
            "probe": probe_name, "pool": pool, "arch": arch, "seed": seed,
            "task": spec.name, **cfg, "metrics": flat, **metrics,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        if flat:
            runs.append(rec)
            print(f"    -> {flat}  loss={metrics.get('final_loss', 'n/a')}", flush=True)

        summary = aggregate(runs)
        write_markdown(summary, cfg, md_path)
        json_path.write_text(json.dumps({"cfg": cfg, "summary": summary, "runs": runs}, indent=2))

    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()