"""Evaluate a grid of OpenRouter models on the FactWorld benchmark tasks.

Examples:
    export OPENROUTER_API_KEY=...

    # Default grid at the first eval length of each task
    python scripts/eval_openrouter_grid.py --n 30 --max_workers 4

    # Length sweep for composite_copy_v1
    python scripts/eval_openrouter_grid.py \
        --models meta-llama/llama-3.3-70b-instruct openai/gpt-4o-mini \
        --tasks composite_copy_v1 --lengths 16 32 64 --n 30

    # Composite format-prompt ablation
    python scripts/eval_openrouter_grid.py \
        --tasks composite_copy_v1 \
        --task_prompts '{"composite_copy_v1": "Answer with the holder name followed by the value."}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factworld import tasks as TK
from factworld.backends import APIBackend
from factworld.runner import evaluate_task


DEFAULT_MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen3-32b",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-3-haiku",
]

DEFAULT_TASKS = list(TK.REPORTED)

DEFAULT_SYSTEM_PROMPT = (
    "You are taking a short test. Answer each question with only the requested "
    "value or values, no explanation. Use the same spelling as in the question."
)

COMPOSITE_FORMAT_PROMPT = (
    "For questions that ask 'what is a0 of the holder of ...', "
    "answer with the holder's name followed by the requested value, "
    "like 'g3 v9'."
)

S5_FORMAT_PROMPT = (
    "For 'what role does ... have?' questions, answer with only a role token "
    "(r0, r1, r2, r3, or r4) followed by a period. Example: 'r2 .'"
)


def _build_system_prompt(base_prompt: str, task_name: str | None,
                         task_prompts: dict[str, str]) -> str:
    parts = [base_prompt]
    key = task_name if task_name in task_prompts else None
    if key is not None:
        parts.append(task_prompts[key])
    return " ".join(parts)


def run_grid(models, tasks, n, lengths, max_workers, base_url,
             system_prompt, task_prompts, max_new_tokens, no_reasoning, jsonl_path=None,
             md_path=None):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    results: list[dict] = []
    for model in models:
        print(f"\n>>> {model}")
        for task_name in tasks:
            spec = TK.CANONICAL[task_name]
            task_lengths = lengths if lengths else [spec.eval_lengths[0]]
            prompt = _build_system_prompt(system_prompt, task_name, task_prompts)
            extra_body = {"reasoning": {"effort": "none"}} if no_reasoning else None
            backend = APIBackend(
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_workers=max_workers,
                system_prompt=prompt,
                extra_body=extra_body,
            )
            for L in task_lengths:
                t0 = time.time()
                result = evaluate_task(
                    backend,
                    spec,
                    split="test",
                    n=n,
                    length=L,
                    max_new_tokens=max_new_tokens,
                )
                elapsed = time.time() - t0
                examples = []
                for (p, g, pred, ok), metrics in zip(
                    result["examples"], result.get("example_metrics", [])
                ):
                    ex = {"prompt": p, "gold": g, "pred": pred, "exact": bool(ok)}
                    ex.update(metrics)
                    examples.append(ex)
                # Fallback if runner does not yet return example_metrics.
                if not examples:
                    examples = [
                        {"prompt": p, "gold": g, "pred": pred, "exact": bool(ok)}
                        for p, g, pred, ok in result["examples"]
                    ]
                metrics_agg = {
                    name: {"correct": sum(e[name] for e in examples), "acc": sum(e[name] for e in examples) / n}
                    for name in ("relaxed", "contains", "last_n")
                    if name in examples[0]
                }
                correct_exact = sum(e["exact"] for e in examples)
                row = {
                    "model": model,
                    "task": task_name,
                    "length": L,
                    "n": n,
                    "system_prompt": prompt,
                    "accuracy_exact": result["overall"],
                    "correct_exact": correct_exact,
                    "elapsed": elapsed,
                    "examples": examples,
                }
                row.update({f"accuracy_{name}": m["acc"] for name, m in metrics_agg.items()})
                row.update({f"correct_{name}": m["correct"] for name, m in metrics_agg.items()})
                results.append(row)
                # Crash-safe: append each completed cell to JSONL and re-emit markdown
                # so a transient upstream error (or interrupt) does not lose the run.
                if jsonl_path is not None:
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                if md_path is not None:
                    write_markdown(results, md_path)
                parts = [f"{task_name}@L{L}: exact={row['accuracy_exact']:.3f}"]
                for name in ("relaxed", "contains", "last_n"):
                    if f"accuracy_{name}" in row:
                        parts.append(f"{name}={row[f'accuracy_{name}']:.3f}")
                parts.append(f"({correct_exact}/{n}) [{elapsed:.1f}s]")
                print("  " + " ".join(parts))
    return results


def _model_label(model: str) -> str:
    return model.split("/")[-1]


def write_markdown(results: list[dict], path: str):
    tasks = sorted({r["task"] for r in results})
    models = [r["model"] for r in results]
    seen = set()
    model_order = []
    for m in models:
        if m not in seen:
            seen.add(m)
            model_order.append(m)

    # Pivot: model x task accuracy (averaged over lengths if multiple)
    metric_names = ["exact", "relaxed", "contains", "last_n"]
    pivot: dict[str, dict[str, dict[str, list[float]]]] = {
        m: {t: {name: [] for name in metric_names} for t in tasks} for m in model_order
    }
    for r in results:
        for name in metric_names:
            if f"accuracy_{name}" in r:
                pivot[r["model"]][r["task"]][name].append(r[f"accuracy_{name}"])

    prompt = results[0]["system_prompt"] if results else ""

    lines = [
        "# FactWorld OpenRouter Model Grid",
        "",
        f"Evaluated at {datetime.now(timezone.utc).isoformat()}.",
        f"n = {results[0]['n'] if results else 0} examples per task/length; position-strict exact match.",
        "",
        "System prompt:",
        "",
        f"> {prompt}",
        "",
        "## Exact-match results",
        "",
        "| model | " + " | ".join(tasks) + " |",
        "| " + " | ".join(["---"] * (len(tasks) + 1)) + " |",
    ]
    for model in model_order:
        accs = []
        for t in tasks:
            vals = pivot[model][t]["exact"]
            accs.append(f"{sum(vals) / len(vals):.3f}" if vals else "-")
        lines.append(f"| {_model_label(model)} | " + " | ".join(accs) + " |")

    if any(pivot[m][t]["contains"] for m in model_order for t in tasks):
        lines += [
            "",
            "## Semantic containment results (tokenizer-robust)",
            "",
            "Every non-punctuation token in the gold answer appears somewhere in the prediction. "
            "For `composite_copy_v1` this means both the holder and value are present; for "
            "single-token tasks it is equivalent to 'the correct token appears anywhere'.",
            "",
            "| model | " + " | ".join(tasks) + " |",
            "| " + " | ".join(["---"] * (len(tasks) + 1)) + " |",
        ]
        for model in model_order:
            accs = []
            for t in tasks:
                vals = pivot[model][t]["contains"]
                accs.append(f"{sum(vals) / len(vals):.3f}" if vals else "—")
            lines.append(f"| {_model_label(model)} | " + " | ".join(accs) + " |")

    lines += [
        "",
        "## Notes",
        "",
        "- Exact match is the canonical metric; semantic containment is reported to separate "
        "formatting/tokenizer artifacts from whether the model knows the answer.",
        "- `APIBackend` normalizes common answer prefixes ('The answer is...') and a trailing "
        "period glued to the preceding token (e.g. `v56.` → `v56 .`).",
        "",
    ]

    lines += [
        "",
        "## Raw data",
        "",
        "Per-task aggregates are in the tables above. Example-level predictions "
        "are in the accompanying JSON output (see `--json_out`).",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote markdown to {path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate OpenRouter models on FactWorld tasks.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="OpenRouter model IDs to evaluate.")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                    choices=list(TK.CANONICAL),
                    help="FactWorld tasks to evaluate.")
    ap.add_argument("--n", type=int, default=30,
                    help="Number of examples per task/length.")
    ap.add_argument("--length", type=int, default=None,
                    help="Deprecated: use --lengths.")
    ap.add_argument("--lengths", nargs="+", type=int, default=None,
                    help="Override eval lengths (default: task's first eval length).")
    ap.add_argument("--max_workers", type=int, default=4,
                    help="Concurrent API calls per model.")
    ap.add_argument("--base_url", default="https://openrouter.ai/api/v1",
                    help="OpenRouter-compatible API base URL.")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT,
                    help="Base system prompt sent to chat models.")
    ap.add_argument("--task_prompts", default=None,
                    help='JSON mapping task name -> extra instruction, e.g. \'{"composite_copy_v1": "..."}\'.')
    ap.add_argument("--composite_format", action="store_true",
                    help="Shorthand: append the composite two-token format instruction.")
    ap.add_argument("--s5_format", action="store_true",
                    help="Shorthand: append the S5 role-output format instruction.")
    ap.add_argument("--max_new_tokens", type=int, default=16,
                    help="Generation budget per example (default: 16).")
    ap.add_argument("--no_reasoning", action="store_true",
                    help="Pass reasoning={\"effort\":\"none\"} to disable chain-of-thought (OpenRouter).")
    ap.add_argument("--out", default="docs/openrouter/results.md",
                    help="Markdown output path.")
    ap.add_argument("--json_out", default=None,
                    help="Optional separate JSON output path.")
    a = ap.parse_args()

    lengths = a.lengths
    if a.length is not None and lengths is None:
        lengths = [a.length]

    task_prompts: dict[str, str] = {}
    if a.task_prompts:
        task_prompts = json.loads(a.task_prompts)
    if a.composite_format:
        task_prompts.setdefault("composite_copy_v1", COMPOSITE_FORMAT_PROMPT)
        task_prompts.setdefault("composite_v1", COMPOSITE_FORMAT_PROMPT)
    if a.s5_format:
        task_prompts.setdefault("s5_v1", S5_FORMAT_PROMPT)

    # Crash-safe JSONL alongside the markdown (rewritten after every cell).
    from pathlib import Path
    jsonl_path = Path(a.out).with_suffix(".jsonl") if a.out else None
    results = run_grid(
        a.models, a.tasks, a.n, lengths, a.max_workers,
        a.base_url, a.system_prompt, task_prompts, a.max_new_tokens,
        a.no_reasoning, jsonl_path=str(jsonl_path) if jsonl_path else None,
        md_path=a.out,
    )

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote JSON to {a.json_out}")

    write_markdown(results, a.out)


if __name__ == "__main__":
    main()
