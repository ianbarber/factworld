"""Reasoning on/off/levels sweep — does background test-time compute help?

The confound: reasoning models (kimi/glm) solved composition with reasoning ON, which IS test-time
compute. This sweeps reasoning effort {none, low, medium, high} to measure the dose-response directly.
If accuracy rises with effort, background reasoning (test-time compute) is a real lever and our
earlier 'test-time doesn't help' claim was wrong (it applied only to explicit CoT prompting and to
local non-reasoning models).

Reasoning models get a generous token budget and no '.' stop (reasoning consumes tokens before the
answer); the backend strips <think> blocks and scores the committed answer.

Example:
    set -a; source .env; set +a
    .venv-api/bin/python scripts/experiment_reasoning.py --n 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import tasks as TK
from factworld.backends import APIBackend
from factworld.render import Renderer
from factworld.runner import evaluate_task

REASONING_MODELS = ["moonshotai/kimi-k2.6", "z-ai/glm-5.2"]
EFFORTS = ["none", "low", "medium", "high"]
COMPOSITE_FORMAT = (
    "For questions that ask 'what is a0 of the holder of ...', "
    "answer with the holder's name followed by the requested value, like 'g3 v9'.")
S5_FORMAT = ("For 'what role does ... have?' questions, answer with only a role token "
             "(r0, r1, r2, r3, or r4) followed by a period.")


def run_cell(model, spec, effort, n, length, api_key, base_url, max_workers, max_new_tokens):
    """Run one (model, task, effort) cell; return per-example rows with holder/value decomp.
    Calls the backend directly with stop_at=None so reasoning models aren't truncated mid-answer
    (the runner's default '.' stop can cut a reasoning-model answer; last_n handles any preamble)."""
    base_sys = ("You are taking a short test about facts and state. "
                "Answer using only tokens that appear in the question.")
    if spec.family == "composite":
        base_sys += " " + COMPOSITE_FORMAT
    elif spec.family == "s5":
        base_sys += " " + S5_FORMAT
    extra_body = {"reasoning": {"effort": effort}} if effort != "default" else None
    backend = APIBackend(model=model, api_key=api_key, base_url=base_url,
                         max_workers=max_workers, system_prompt=base_sys, extra_body=extra_body)
    examples = TK.generate(spec, split="test", n=n, length=length)
    preds = backend.generate([e.prompt for e in examples], max_new_tokens=max_new_tokens, stop_at=None)
    rows = []
    for e, pred in zip(examples, preds):
        dec = TK.decompose_composite(pred, e.answer)
        rows.append({"gold": e.answer, "pred": pred, "last_n": TK.score_last_n(pred, e.answer), **dec})
    return rows


def summarize(rows):
    n = len(rows) or 1
    two = [r for r in rows if len(TK.content_tokens(r["gold"])) >= 2]
    return {
        "exact": sum(r["last_n"] for r in rows) / n,  # last_n as headline (handles reasoning preamble)
        "holder_acc": sum(r["holder_ok"] for r in rows) / n,
        "value_acc": (sum(r["value_ok"] for r in two) / len(two)) if two else 0.0,
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser(description="Reasoning on/off/levels sweep.")
    ap.add_argument("--models", nargs="+", default=REASONING_MODELS)
    ap.add_argument("--tasks", nargs="+", default=["composite_copy_v1", "s5_v1"])
    ap.add_argument("--efforts", nargs="+", default=EFFORTS,
                    help="Reasoning effort levels (OpenRouter). Add 'default' for the model default.")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--length", type=int, default=None,
                    help="Eval length (default: task's first eval length). Use for long-context runs.")
    ap.add_argument("--max_new_tokens", type=int, default=8192,
                    help="Generous budget so reasoning can finish before the answer.")
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    from pathlib import Path
    prefix = Path(a.out_prefix or f"results/reasoning_sweep_{ts}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl = Path(f"{prefix}.jsonl"); md = Path(f"{prefix}.md")

    print(f"=== reasoning sweep -> {jsonl} ===", flush=True)
    all_rows = []
    for model in a.models:
        for task in a.tasks:
            spec = TK.CANONICAL[task]
            length = a.length or spec.eval_lengths[0]
            for effort in a.efforts:
                tag = f"{model.split('/')[-1]} | {task}@L{length} | effort={effort}"
                print(f"\n--- {tag} ---", flush=True)
                try:
                    rows = run_cell(model, spec, effort, a.n, length, api_key,
                                    a.base_url, a.max_workers, a.max_new_tokens)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    rows = []
                summ = summarize(rows)
                print(f"    last_n={summ['exact']:.2f} holder={summ['holder_acc']:.2f} value={summ['value_acc']:.2f}", flush=True)
                rec = {"model": model, "task": task, "length": length, "effort": effort,
                       "summary": summ, "examples": rows}
                with jsonl.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                all_rows.append(rec)

    # markdown: model x task table, columns = effort
    by = defaultdict(dict)
    for r in all_rows:
        by[(r["model"], r["task"])][r["effort"]] = r["summary"]
    lines = ["# Reasoning on/off/levels sweep — does background test-time compute help?", "",
             f"n={a.n} per cell. Reasoning models with effort swept {{none, low, medium, high}}. "
             f"last_n scoring (committed answer after any reasoning preamble). "
             f"If accuracy rises with effort, background reasoning IS a test-time-compute lever.", ""]
    for (model, task), cells in by.items():
        lines.append(f"## {model.split('/')[-1]} — {task}")
        efforts = [e for e in a.efforts if e in cells]
        lines.append("| effort | last_n | holder | value |")
        lines.append("|---|---|---|---|")
        for e in efforts:
            s = cells[e]
            lines.append(f"| {e} | {s['exact']:.2f} | {s['holder_acc']:.2f} | {s['value_acc']:.2f} |")
        lines.append("")
    md.write_text("\n".join(lines))
    print(f"\n=== wrote {md} ===", flush=True)


if __name__ == "__main__":
    main()
