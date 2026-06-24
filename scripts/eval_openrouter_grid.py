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


def _relaxed_score(pred: str, gold: str) -> int:
    """Tokenization-agnostic score: ignore whitespace and trailing periods."""
    pred_norm = pred.replace(".", "").replace(" ", "").strip()
    gold_norm = gold.replace(".", "").replace(" ", "").strip()
    return int(pred_norm == gold_norm)


def _build_system_prompt(base_prompt: str, task_name: str | None,
                         task_prompts: dict[str, str]) -> str:
    parts = [base_prompt]
    key = task_name if task_name in task_prompts else None
    if key is not None:
        parts.append(task_prompts[key])
    return " ".join(parts)


def run_grid(models, tasks, n, lengths, max_workers, base_url,
             system_prompt, task_prompts, max_new_tokens, no_reasoning):
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
                examples = [
                    {
                        "prompt": p,
                        "gold": g,
                        "pred": pred,
                        "exact": bool(ok),
                        "relaxed": bool(_relaxed_score(pred, g)),
                    }
                    for p, g, pred, ok in result["examples"]
                ]
                correct_exact = sum(e["exact"] for e in examples)
                correct_relaxed = sum(e["relaxed"] for e in examples)
                row = {
                    "model": model,
                    "task": task_name,
                    "length": L,
                    "n": n,
                    "system_prompt": prompt,
                    "accuracy_exact": result["overall"],
                    "accuracy_relaxed": correct_relaxed / n,
                    "correct_exact": correct_exact,
                    "correct_relaxed": correct_relaxed,
                    "elapsed": elapsed,
                    "examples": examples,
                }
                results.append(row)
                print(
                    f"  {task_name}@L{L}: exact={row['accuracy_exact']:.3f} "
                    f"relaxed={row['accuracy_relaxed']:.3f} "
                    f"({correct_exact}/{n} | {correct_relaxed}/{n}) [{elapsed:.1f}s]"
                )
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
    pivot: dict[str, dict[str, dict[str, list[float]]]] = {
        m: {t: {"exact": [], "relaxed": []} for t in tasks} for m in model_order
    }
    for r in results:
        pivot[r["model"]][r["task"]]["exact"].append(r["accuracy_exact"])
        pivot[r["model"]][r["task"]]["relaxed"].append(r["accuracy_relaxed"])

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
        accs = [f"{sum(pivot[model][t]['exact']) / len(pivot[model][t]['exact']):.3f}" for t in tasks]
        lines.append(f"| {_model_label(model)} | " + " | ".join(accs) + " |")

    lines += [
        "",
        "## Relaxed results (whitespace / period invariant)",
        "",
        "| model | " + " | ".join(tasks) + " |",
        "| " + " | ".join(["---"] * (len(tasks) + 1)) + " |",
    ]
    for model in model_order:
        accs = [f"{sum(pivot[model][t]['relaxed']) / len(pivot[model][t]['relaxed']):.3f}" for t in tasks]
        lines.append(f"| {_model_label(model)} | " + " | ".join(accs) + " |")

    lines += [
        "",
        "## Notes",
        "",
        "- `APIBackend` normalizes a trailing period glued to the preceding token (e.g. `v56.` → `v56 .`).",
        "- Relaxed scoring strips spaces and trailing periods; exact match remains the canonical metric.",
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
    ap.add_argument("--out", default="docs/openrouter-results.md",
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

    results = run_grid(
        a.models, a.tasks, a.n, lengths, a.max_workers,
        a.base_url, a.system_prompt, task_prompts, a.max_new_tokens,
        a.no_reasoning,
    )

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote JSON to {a.json_out}")

    write_markdown(results, a.out)


if __name__ == "__main__":
    main()
