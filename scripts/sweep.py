"""Multi-seed local architecture sweep for FactWorld (natural-language format).

For each (task x architecture x seed): train from scratch and score position-strict
exact match on every held-out eval length, plus a **prefix-match decomposition**
that separates the legs of multi-token answers (e.g. composite = holder leg then
value leg). Composition is bimodal, so the summary reports both mean+-std and
p(converge) over seeds.

Crash-safe: each completed run is appended to a JSONL log as it finishes, so
partial results survive an interrupt. A markdown + JSON summary is (re)written at
the end and after every aggregation pass.

Example:
    .venv-train/bin/python scripts/sweep.py \\
        --tasks binding_v2,composite_copy_v2 \\
        --archs gdp_hybrid,fprm,transformer \\
        --seeds 0 1 2 3 4 --steps 8000 --d_model 256
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld import train as T
from factworld.backends import LocalBackend
from factworld.render import Renderer
from factworld.runner import evaluate_task


def build_docs(examples, use_trace=False):
    """prompt + (optional oracle worked-trace) + answer, single-space separated."""
    docs = []
    for e in examples:
        if use_trace and "trace" in e.meta:
            docs.append(f"{e.prompt} {e.meta['trace']} {e.answer}")
        else:
            docs.append(f"{e.prompt} {e.answer}")
    return docs


def _content_tokens(s):
    """Normalized tokens with punctuation stripped: the semantic answer span."""
    return [t for t in Renderer.normalize(s).split() if t != "."]


def prefix_decomp(inspected, trace_mode=False):
    """Prefix-match decomposition over (prompt, gold, pred, correct) tuples, on CONTENT tokens.

    For a 2-token composite answer (holder, value) this yields prefix {0: neither, 1: holder-only,
    2: both} -- a direct read of where composition breaks. `holder_acc` is first-content-token
    accuracy (the binding/state leg); `value_acc` is second-content-token accuracy over the
    2-content-token answers (the recall leg of composition).

    In ``trace_mode`` the prediction is a self-generated scratchpad (trace) FOLLOWED BY the
    answer, so we score the LAST len(gold) content tokens (the committed answer), not the
    prefix -- otherwise the trace tokens are misread as the answer.
    """
    n = len(inspected)
    buckets = {0: 0, 1: 0, 2: 0}
    leg1 = 0           # first content token correct (holder / single-token answer)
    two_token = 0      # answers with >=2 content tokens (composite)
    leg2 = 0           # second content token correct (value), over two_token answers
    for _prompt, gold, pred, _ok in inspected:
        g = _content_tokens(gold)
        p = _content_tokens(pred)
        if trace_mode and len(p) >= len(g):
            p = p[-len(g):]                          # score the committed answer (tail), not the trace
        k = 0
        while k < len(g) and k < len(p) and p[k] == g[k]:
            k += 1
        buckets[min(k, 2)] = buckets.get(min(k, 2), 0) + 1
        if len(p) >= 1 and len(g) >= 1 and p[0] == g[0]:
            leg1 += 1
        if len(g) >= 2:
            two_token += 1
            if len(p) >= 2 and p[1] == g[1]:
                leg2 += 1
    return {
        "prefix": {str(k): buckets.get(k, 0) / n for k in (0, 1, 2)},
        "holder_acc": leg1 / n,
        "value_acc": leg2 / max(1, two_token),
    }


def run_one(spec, arch, seed, *, d_model, n_layers, steps, batch, train_n, eval_n,
            use_short_conv, use_trace, device):
    """Train one config; return {length: {overall, decomp}} and final loss."""
    import torch

    d_ff = 4 * d_model
    w, r = TK.build_world(spec)
    train = TK.generate(spec, "train", n=train_n)
    tok, docs, _ = T.prepare(build_docs(train, use_trace), [], [w], renderer=r)
    run = T.run(
        arch, tok, docs, [], steps=steps, batch=batch, d_model=d_model, n_layers=n_layers,
        d_ff=d_ff, seed=seed, return_model=True, device=device, use_short_conv=use_short_conv,
    )
    backend = LocalBackend([w], arch=arch, model=run["model"], tokenizer=tok, device=device)
    out = {}
    for L in spec.eval_lengths:
        # In trace mode the model emits the full scratchpad (length tokens) THEN the answer,
        # so size the generation budget for trace + answer, and score the committed tail.
        max_new = (L + 6) if use_trace else None
        res = evaluate_task(backend, spec, split="test", n=eval_n, length=L, max_new_tokens=max_new)
        out[str(L)] = {"overall": res["overall"], **prefix_decomp(res["examples"], trace_mode=use_trace)}
    del run["model"]
    torch.cuda.empty_cache()
    return {"lengths": out, "final_loss": run["final_loss"]}


def aggregate(runs):
    """Per (task, arch, length): mean+-std of overall, p(converge), mean decomp."""
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(list))  # (task,arch)[length] -> [overall,...]
    dec = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        key = (r["task"], r["arch"])
        for L, v in r["lengths"].items():
            by[key][L].append(v["overall"])
            for dk in ("holder_acc", "value_acc"):
                dec[key][L][dk].append(v[dk])
    summary = {}
    for (task, arch), lens in by.items():
        summary.setdefault(task, {})[arch] = {}
        for L, ov in lens.items():
            conv = sum(1 for x in ov if x >= 0.9)
            summary[task][arch][L] = {
                "mean": statistics.mean(ov),
                "std": statistics.pstdev(ov) if len(ov) > 1 else 0.0,
                "n": len(ov),
                "p_converge": conv / len(ov),
                "holder_acc": statistics.mean(dec[(task, arch)][L]["holder_acc"]),
                "value_acc": statistics.mean(dec[(task, arch)][L]["value_acc"]),
            }
    return summary


def write_markdown(summary, cfg, path):
    lines = [f"# Local sweep — {cfg['tasks']} ", ""]
    lines.append(f"d_model={cfg['d_model']} n_layers={cfg['n_layers']} steps={cfg['steps']} "
                 f"seeds={cfg['seeds']} train_n={cfg['train_n']} eval_n={cfg['eval_n']}")
    lines.append("")
    for task, archs in summary.items():
        spec = TK.spec_for(task)
        lens = [str(L) for L in spec.eval_lengths]
        lines.append(f"## {task}  (eval lengths {', '.join(lens)})")
        header = "| arch | " + " | ".join(f"L{L} mean±std (pconv)" for L in lens) + " | holder/value @L" + lens[0] + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(lens) + 2))
        for arch, ld in sorted(archs.items()):
            cells = []
            for L in lens:
                d = ld[L]
                cells.append(f"{d['mean']:.2f}±{d['std']:.2f} ({d['p_converge']:.0%})")
            d0 = ld[lens[0]]
            cells.append(f"{d0['holder_acc']:.2f} / {d0['value_acc']:.2f}")
            lines.append(f"| {arch} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"_pconv = fraction of seeds reaching >=0.9 at that length; "
                     f"holder/value = leg accuracies (2-token answers) at L{lens[0]}._")
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Multi-seed local architecture sweep.")
    ap.add_argument("--tasks", default="binding_v2,composite_copy_v2",
                    help="Comma-separated canonical task names (RETIRED names accepted "
                         "for historical reproduction only; see tasks.RETIRED).")
    ap.add_argument("--archs", default="gdp_hybrid,fprm,transformer",
                    help="Comma-separated architectures (gdp_hybrid_shortconv => gdp_hybrid+shortconv).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--eval_n", type=int, default=100, help="Test examples per length.")
    ap.add_argument("--use_trace", action="store_true", help="Append oracle worked-trace (s5/composite).")
    ap.add_argument("--worked_trace", action="store_true",
                    help="Force worked_trace=True on the spec (needed for the composite "
                         "tasks, whose default is False). Implies --use_trace.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default=None,
                    help="Output prefix (default: results/sweep_<task0>_<timestamp>).")
    a = ap.parse_args()

    tasks = [t.strip() for t in a.tasks.split(",")]
    archs = [x.strip() for x in a.archs.split(",")]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = a.out_prefix or f"results/sweep_{tasks[0]}_{ts}"
    from pathlib import Path
    log_path = Path(f"{prefix}.jsonl")
    md_path = Path(f"{prefix}.md")
    json_path = Path(f"{prefix}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {"tasks": tasks, "archs": archs, "seeds": a.seeds, "steps": a.steps,
           "d_model": a.d_model, "n_layers": a.n_layers, "train_n": a.train_n, "eval_n": a.eval_n}

    runs = []
    total = len(tasks) * len(archs) * len(a.seeds)
    done = 0
    print(f"=== sweep: {total} runs -> {log_path} ===", flush=True)
    for task in tasks:
        spec = TK.spec_for(task)
        if a.worked_trace:
            spec = spec.scaled(worked_trace=True)
        for arch in archs:
            use_short, resolved = False, arch
            if arch == "gdp_hybrid_shortconv":
                resolved, use_short = "gdp_hybrid", True
            for seed in a.seeds:
                tag = f"{task} | {arch} | seed {seed}"
                print(f"\n--- [{done+1}/{total}] {tag} ---", flush=True)
                try:
                    r = run_one(spec, resolved, seed, d_model=a.d_model, n_layers=a.n_layers,
                                steps=a.steps, batch=a.batch, train_n=a.train_n, eval_n=a.eval_n,
                                use_short_conv=use_short,
                                use_trace=(a.use_trace or a.worked_trace), device=a.device)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    r = {"error": str(e)}
                rec = {"task": task, "arch": arch, "seed": seed, **cfg, **r}
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if "lengths" in r:
                    runs.append(rec)
                    ov = {L: v["overall"] for L, v in r["lengths"].items()}
                    print(f"    -> {ov}  loss={r.get('final_loss'):.3f}", flush=True)
                done += 1
                # incremental summary
                summary = aggregate(runs)
                write_markdown(summary, cfg, md_path)
                json_path.write_text(json.dumps({"cfg": cfg, "summary": summary, "runs": runs}, indent=2))
    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()
