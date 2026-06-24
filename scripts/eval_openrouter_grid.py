"""Evaluate a grid of OpenRouter models on the FactWorld benchmark tasks.

Example:
    export OPENROUTER_API_KEY=...
    python scripts/eval_openrouter_grid.py \
        --models meta-llama/llama-3.1-8b-instruct qwen/qwen-2.5-7b-instruct \
        --tasks recall_copy_v1 binding_v1 composite_copy_v1 conflict_v1 chain_v1 \
        --n 30 --max_workers 4 --out docs/openrouter-results.md
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


def _relaxed_score(pred: str, gold: str) -> int:
    """Tokenization-agnostic score: ignore whitespace and trailing periods.

    External chat models often omit the space before the period or merge
    punctuation, so the canonical exact match can be artificially low. This
    relaxed metric strips those formatting differences while still requiring
    the correct tokens.
    """
    pred_norm = pred.replace(".", "").replace(" ", "").strip()
    gold_norm = gold.replace(".", "").replace(" ", "").strip()
    return int(pred_norm == gold_norm)


def run_grid(models, tasks, n, length, max_workers, base_url, system_prompt):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    results: list[dict] = []
    for model in models:
        print(f"\n>>> {model}")
        backend = APIBackend(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_workers=max_workers,
            system_prompt=system_prompt,
        )
        for task_name in tasks:
            spec = TK.CANONICAL[task_name]
            lengths = [length] if length is not None else [spec.eval_lengths[0]]
            for L in lengths:
                t0 = time.time()
                result = evaluate_task(
                    backend,
                    spec,
                    split="test",
                    n=n,
                    length=L,
                    max_new_tokens=16,
                )
                elapsed = time.time() - t0
                correct_exact = sum(1 for _, _, _, ok in result["examples"] if ok)
                correct_relaxed = sum(
                    _relaxed_score(pred, gold)
                    for _, gold, pred, _ in result["examples"]
                )
                row = {
                    "model": model,
                    "task": task_name,
                    "length": L,
                    "n": n,
                    "accuracy_exact": result["overall"],
                    "accuracy_relaxed": correct_relaxed / n,
                    "correct_exact": correct_exact,
                    "correct_relaxed": correct_relaxed,
                    "elapsed": elapsed,
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


def write_markdown(results: list[dict], path: str, system_prompt: str):
    tasks = sorted({r["task"] for r in results})
    models = [r["model"] for r in results]
    # preserve input order of models
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

    lines = [
        "# FactWorld OpenRouter Model Grid",
        "",
        f"Evaluated at {datetime.now(timezone.utc).isoformat()}.",
        f"n = {results[0]['n']} examples per task; position-strict exact match.",
        "",
        "System prompt:",
        "",
        f"> {system_prompt}",
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

    lines += ["", "## Notes", "", "- Relaxed scoring strips spaces and trailing periods. It is provided because external chat models often emit `v56.` instead of the canonical `v56 .`.", "- Exact match remains the canonical FactWorld metric.", ""]

    lines += ["", "## Raw data", ""]
    lines.append("```json")
    lines.append(json.dumps(results, indent=2))
    lines.append("```")
    lines.append("")

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
                    help="Override eval length (default: task's first eval length).")
    ap.add_argument("--max_workers", type=int, default=4,
                    help="Concurrent API calls per model.")
    ap.add_argument("--base_url", default="https://openrouter.ai/api/v1",
                    help="OpenRouter-compatible API base URL.")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT,
                    help="System prompt sent to chat models.")
    ap.add_argument("--out", default="docs/openrouter-results.md",
                    help="Markdown output path.")
    ap.add_argument("--json_out", default=None,
                    help="Optional separate JSON output path.")
    a = ap.parse_args()

    results = run_grid(a.models, a.tasks, a.n, a.length, a.max_workers, a.base_url, a.system_prompt)

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote JSON to {a.json_out}")

    write_markdown(results, a.out, a.system_prompt)


if __name__ == "__main__":
    main()
