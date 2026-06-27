"""Autoregressive / test-time-compute experiment (E1 + E2).

Question: does letting a model generate MORE before answering (a self-produced
scratchpad / chain-of-thought) unlock composition and state-tracking — and is the
gain real computation or just more tokens?

This script runs the API side (E1: answer-only vs free-CoT vs structured-CoT vs
scaffolded upper bound) and is structured so the local trained-scratchpad side
(E2) can be run via scripts/sweep.py --use_trace.

Run (API):
    set -a; source .env; set +a
    .venv-api/bin/python scripts/experiment_autoregressive.py \\
        --tasks composite_copy_v1 s5_v1 --n 30

Conditions (E1):
    none        — answer directly (baseline).
    free        — "think step by step, then answer" (free chain-of-thought).
    structured  — "first write `holder: <g>`, then `<g> <value>.`" (parseable intermediate).
    scaffolded  — inject the CORRECT holder into the prompt; model only recalls.
                  This is the ceiling for any CoT that merely gets the holder right.

Scoring: final answer via score_last_n, plus the holder/value decomposition and
(for trace-bearing tasks) self-trace accuracy against the oracle trajectory.
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


# Per-task scaffolds: the system-prompt instruction + (optionally) a prompt rewriter.
FREE_COT = (
    "Think step by step about who holds the object and what their value is, "
    "then end your answer with the final holder and value."
)
STRUCTURED_COT = (
    "First write the resolved holder on its own line as `holder: <g>`, "
    "then on the next line write the final answer as `<holder> <value>.` "
    "Use only tokens that appear in the question."
)
STRUCTURED_S5 = (
    "Track the role step by step. First write `role: <r>` then on the next line "
    "write the final role token followed by a period, like `r2.`"
)

# Reasoning models: they emit a <think> block (inline, e.g. GLM) or reason in a separate
# `reasoning` field and leave `content` clean (e.g. Kimi). They consume the token budget on
# thinking, so they need (a) a much larger budget and (b) NO `.` stop token, which would cut
# reasoning mid-stream. The backend's <think> strip handles inline reasoners; field reasoners
# already return clean content. last_n + decomposition scoring find the answer after any preamble.
REASONING_MODELS = {"moonshotai/kimi-k2.6", "z-ai/glm-5.2", "moonshotai/kimi-k2.7-code",
                    "moonshotai/kimi-k2-thinking"}


def _model_budget(model: str, max_new_tokens: int) -> tuple[int, str | None]:
    """Return (token_budget, stop_at) for a model."""
    if model in REASONING_MODELS:
        # Reasoning models consume tokens on thinking before committing; GLM-5.2 in particular
        # can reason past 4096 on the 16-event composite and return empty. 8192 is enough for
        # both Kimi and GLM to finish reasoning then emit the answer (~5s/call).
        return 8192, None          # let reasoning finish; no '.' stop (would truncate thinking)
    return max_new_tokens, "."


def system_prompt(base: str, cond: str, task_name: str) -> str:
    if cond == "none":
        return base
    if cond == "free":
        return base + " " + FREE_COT
    if cond == "structured":
        extra = STRUCTURED_S5 if task_name == "s5_v1" else STRUCTURED_COT
        return base + " " + extra
    return base  # scaffolded uses a plain prompt + a rewritten user prompt


def scaffold_prompt(example, task_name: str) -> str:
    """Inject the correct holder into the prompt so the model only does the recall leg.

    This is the recall-leg upper bound: given the binding answer, can the model recall?
    """
    if task_name in ("composite_copy_v1", "composite_copy_scale_v1", "composite_v1"):
        holder = example.meta.get("holder")
        if holder is None:
            return example.prompt
        return f"{example.prompt} (the holder is {holder})"
    return example.prompt  # s5 has no single decoupled leg; scaffold is a no-op


def binding_prompt(example, task_name: str) -> str:
    """Rewrite the composite query to ask ONLY for the holder (binding leg in isolation).

    Symmetric to the recall-scaffold: instead of 'what is a0 of the holder of o3?' (binding +
    recall + routing), ask 'who is the final holder of o3?' so only last-write-wins state
    tracking is needed. Gold becomes the holder alone. This localizes whether the API
    composition failure is binding itself (state-tracking) or routing the holder into recall.
    """
    if task_name not in ("composite_copy_v1", "composite_copy_scale_v1", "composite_v1"):
        return example.prompt, example.answer
    obj = example.meta.get("obj")
    holder = example.meta.get("holder")
    if obj is None or holder is None:
        return example.prompt, example.answer
    # strip the composite query and replace with a plain binding query
    import re
    base = re.sub(r"\s*what is a0 of the holder of \S+\?$", "", example.prompt)
    prompt = f"{base} who is the final holder of {obj}?"
    gold = f"{holder}."
    return prompt, gold


def run_condition(model, spec, cond, n, length, api_key, base_url, max_workers,
                  base_system, max_new_tokens):
    """Run one (model, task, condition) cell; return per-example dicts.

    For the scaffolded condition (correct holder/role injected) the model only does
    the recall leg, so the gold is the VALUE alone and we score value-only match.
    For CoT conditions the final answer may be preceded by a scratchpad, so the
    headline metric is ``last_n`` (last len(gold) tokens), not position-strict exact.
    """
    sysp = system_prompt(base_system, cond, spec.name)
    examples = TK.generate(spec, "test", n=n, length=length)
    if cond == "scaffolded":
        prompts = [scaffold_prompt(e, spec.name) for e in examples]
    elif cond == "binding":
        # binding-leg isolation: rewrite to ask only for the holder; gold = holder alone
        rewritten = [binding_prompt(e, spec.name) for e in examples]
        prompts = [p for p, _g in rewritten]
        binding_gold = [g for _p, g in rewritten]
    else:
        prompts = [e.prompt for e in examples]
    backend = APIBackend(model=model, api_key=api_key, base_url=base_url,
                         max_workers=max_workers, system_prompt=sysp)
    budget, stop_at = _model_budget(model, max_new_tokens)
    preds = backend.generate(prompts, max_new_tokens=budget, stop_at=stop_at)
    rows = []
    for e, pred in zip(examples, preds):
        if cond == "scaffolded" and spec.family == "composite":
            # recall leg only: gold is the value (2nd content token); does pred contain it?
            gold_ct = TK.content_tokens(e.answer)
            value = gold_ct[1] if len(gold_ct) >= 2 else None
            pred_ct = TK.content_tokens(pred)
            value_ok = int(value is not None and value in pred_ct)
            row = {"gold": e.answer, "pred": pred, "exact": value_ok,
                   "last_n": value_ok, "holder_ok": 1, "value_ok": value_ok, "prefix": 1 + value_ok}
        elif cond == "binding" and spec.family == "composite":
            # binding leg only: gold is the holder; does pred contain it?
            holder = e.meta.get("holder")
            pred_ct = TK.content_tokens(pred)
            holder_ok = int(holder is not None and holder in pred_ct)
            row = {"gold": f"{holder}.", "pred": pred, "exact": holder_ok,
                   "last_n": holder_ok, "holder_ok": holder_ok, "value_ok": 1, "prefix": 1 + holder_ok}
        else:
            dec = TK.decompose_composite(pred, e.answer)
            row = {"gold": e.answer, "pred": pred,
                   "exact": TK.score_exact(Renderer.normalize(pred), Renderer.normalize(e.answer)),
                   "last_n": TK.score_last_n(pred, e.answer), **dec}
        if "trace" in e.meta:
            row["trace"] = TK.trace_accuracy(pred, e.meta["trace"])
        rows.append(row)
    return rows


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    two_gold = [r for r in rows if len(TK.content_tokens(r["gold"])) >= 2]
    return {
        "exact": sum(r["exact"] for r in rows) / n,
        "last_n": sum(r["last_n"] for r in rows) / n,
        "holder_acc": sum(r["holder_ok"] for r in rows) / n,
        # value_acc is only meaningful for 2-token (holder,value) answers; report 0 when N/A
        "value_acc": (sum(r["value_ok"] for r in two_gold) / len(two_gold)) if two_gold else 0.0,
        "trace_acc": (sum(r["trace"]["token_acc"] for r in rows if "trace" in r)
                      / max(1, sum(1 for r in rows if "trace" in r))),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser(description="Autoregressive / test-time-compute experiment (API).")
    ap.add_argument("--models", nargs="+", default=[
        "meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-chat",
        "openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"])
    ap.add_argument("--tasks", nargs="+", default=["composite_copy_v1", "s5_v1"])
    ap.add_argument("--conditions", nargs="+",
                    default=["none", "free", "structured", "scaffolded"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--length", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="Generous budget for CoT (default 128).")
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--base_system", default=(
        "You are taking a short test about facts and state. "
        "Answer using only tokens that appear in the question."))
    ap.add_argument("--composite_format", action="store_true",
                    help="Append the composite two-token format instruction (format-fair comparison vs the grid).")
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    from pathlib import Path
    prefix = Path(a.out_prefix or f"results/autoregressive_api_{ts}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl = Path(f"{prefix}.jsonl")
    md = Path(f"{prefix}.md")

    COMPOSITE_FORMAT = (
        "For questions that ask 'what is a0 of the holder of ...', "
        "answer with the holder's name followed by the requested value, like 'g3 v9'.")
    all_rows = []
    print(f"=== autoregressive API experiment -> {jsonl} ===", flush=True)
    for model in a.models:
        for task in a.tasks:
            spec = TK.CANONICAL[task]
            length = a.length or spec.eval_lengths[0]
            # Format-fair: append the composite format instruction so the only variable
            # across conditions is the reasoning regime (not whether the model knows the output shape).
            base_sys = a.base_system
            if a.composite_format and spec.family == "composite":
                base_sys = base_sys + " " + COMPOSITE_FORMAT
            for cond in a.conditions:
                tag = f"{model} | {task}@L{length} | {cond}"
                print(f"\n--- {tag} ---", flush=True)
                try:
                    rows = run_condition(model, spec, cond, a.n, length, api_key,
                                         a.base_url, a.max_workers, base_sys,
                                         a.max_new_tokens)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    rows = []
                summ = summarize(rows)
                print(f"    exact={summ.get('exact',0):.2f} last_n={summ.get('last_n',0):.2f} "
                      f"holder={summ.get('holder_acc',0):.2f} value={summ.get('value_acc',0):.2f}", flush=True)
                rec = {"model": model, "task": task, "length": length, "condition": cond,
                       "summary": summ, "examples": rows}
                with jsonl.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                all_rows.append(rec)

    # markdown summary
    by = defaultdict(dict)
    for r in all_rows:
        by[(r["task"], r["length"])][(r["model"], r["condition"])] = r["summary"]
    lines = ["# Autoregressive / test-time-compute experiment (API, E1)", ""]
    lines.append(f"n={a.n} per cell, max_new_tokens={a.max_new_tokens}")
    lines.append("")
    for (task, length), cells in by.items():
        lines.append(f"## {task} @ L{length}")
        conds = sorted({c for _m, c in cells})
        models = sorted({m for m, _c in cells})
        lines.append("| model | " + " | ".join(
            f"{c} (exact/holder/value)" for c in conds) + " |")
        lines.append("|" + "---|" * (len(conds) + 1))
        for m in models:
            row = [m]
            for c in conds:
                s = cells.get((m, c), {})
                if s:
                    row.append(f"{s.get('exact',0):.2f} / {s.get('holder_acc',0):.2f} / {s.get('value_acc',0):.2f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("_scaffolded = correct holder injected; the recall-leg ceiling for composition._")
        lines.append("")
    md.write_text("\n".join(lines))
    print(f"\n=== wrote {md} ===", flush=True)


if __name__ == "__main__":
    main()
