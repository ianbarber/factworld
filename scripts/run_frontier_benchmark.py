"""Run the recurring frontier-model benchmark and append C3 records to history.

Executes the cell plan from ``factworld.benchmark.arms_for`` (contract C4) against
OpenRouter, one model at a time, one cell at a time (examples fan out concurrently
inside ``APIBackend``). Each completed cell appends ONE crash-safe JSONL record to
the history file (contract C3) — metrics, diagnostics (empty-pred rate, api errors,
finish reasons), token usage and an estimated cost. Resume is automatic: any cell
whose (model, facet, task, length, n, settings_hash, suite_version) key already has
a history record is skipped (latest-wins dedup lives in scripts/render_benchmark.py).
n is part of the key so a low-n scouting pass (--n-scale) never satisfies resume for
the full-n cell.

Protocol rule: reasoning-on cells run with max_new_tokens=8192 and stop_at=None
(smaller budgets manufactured the old "s5 L64 cliff" / "chain floor" as truncation
artifacts — see results/s5_horizon_recheck_20260705.jsonl). A cell whose empty-pred
rate exceeds 0.5 gets a loud truncation-suspect warning but the record is kept.

Examples:
    set -a; source .env; set +a

    # Print the full plan + cost estimate, no API calls
    .venv-api/bin/python scripts/run_frontier_benchmark.py --dry-run

    # Scouting pass at 1/5 the per-facet n
    .venv-api/bin/python scripts/run_frontier_benchmark.py --n-scale 0.2

    # Re-run the drift canary even where history already has its cells
    .venv-api/bin/python scripts/run_frontier_benchmark.py \\
        --models z-ai/glm-5.2 --canary
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factworld import s5_concrete as S5
from factworld import tasks as TK
from factworld.backends import APIBackend
from factworld.benchmark import (
    CANARY_MODEL,
    FACETS,
    MODELS,
    REASONING_EFFORTS,
    arms_for,
    cost_estimate,
    settings_hash,
)
from factworld.render import Renderer
from factworld.runner import evaluate_task

# Reuse the grid script's system-prompt plumbing (composite format instruction) and
# the autoregressive experiment's decomposition-leg prompt builders, per the design.
from eval_openrouter_grid import (  # noqa: E402
    COMPOSITE_FORMAT_PROMPT,
    _build_system_prompt,
)
from experiment_autoregressive import binding_prompt, scaffold_prompt  # noqa: E402

# The canonical published grid's base system prompt (docs/openrouter/results-natural.jsonl
# records this verbatim for every non-composite cell). NOTE: this is deliberately NOT
# eval_openrouter_grid.DEFAULT_SYSTEM_PROMPT — that constant later had the composite
# holder-name instruction folded into it, which leaks the two-token answer format into
# single-answer tasks (models then answer 'g5 v37' on recall/conflict and score 0 on the
# canonical relaxed prefix match).
BASE_SYSTEM_PROMPT = (
    "You are taking a short test. Answer each question with only the requested "
    "value or values, no explanation. Use the same spelling as in the question."
)

# Grid mechanics: --composite_format appends the two-token format instruction for
# composite tasks (format-fair across models that don't guess the output shape).
TASK_PROMPTS = {
    "composite_copy_v1": COMPOSITE_FORMAT_PROMPT,
    "composite_v1": COMPOSITE_FORMAT_PROMPT,
}

# Same base system prompt as scripts/experiment_autoregressive.py (its --base_system
# default) with the composite format instruction appended (its --composite_format
# flag), so the decomposition legs match that experiment's format-fair protocol.
DECOMP_BASE_SYSTEM = (
    "You are taking a short test about facts and state. "
    "Answer using only tokens that appear in the question. " + COMPOSITE_FORMAT_PROMPT
)

S5_FACETS = ("s5_concrete", "floor")  # cells rendered via factworld.s5_concrete

EMPTY_META = {
    "calls": 0, "errors": 0,
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
    "served_models": [], "providers": [], "finish_reasons": {},
}


# --- plan / resume ----------------------------------------------------------------

def cell_key(model: str, cell: dict) -> tuple:
    """Resume key: skip a cell if ANY history record already carries this key.

    Includes the cell's n so a low-n scouting run (--n-scale) does not mark the
    full-n cell as done.
    """
    return (model, cell["facet"], cell["task"], cell["length"], cell["n"],
            settings_hash(cell), TK.SUITE_VERSION)


def history_keys(history_path: str) -> set[tuple]:
    """Read the history file and return the set of already-run cell keys."""
    keys: set[tuple] = set()
    if not os.path.exists(history_path):
        return keys
    with open(history_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn tail line from a crashed run
            keys.add((rec.get("model"), rec.get("facet"), rec.get("task"),
                      rec.get("length"), rec.get("n"),
                      settings_hash({"settings": rec.get("settings") or {}}),
                      rec.get("suite_version")))
    return keys


def should_skip(model: str, cell: dict, done: set[tuple], force: bool, canary: bool) -> bool:
    if force:
        return False
    if canary and model == CANARY_MODEL:
        return False  # drift canary: always re-run glm cells
    return cell_key(model, cell) in done


def build_plan(models: list[str], facets: list[str] | None, n_scale: float) -> dict[str, list[dict]]:
    """Per-model cell lists with the scouting multiplier applied (n floor of 5)."""
    plan = {}
    for model in models:
        cells = [c for c in arms_for(model) if facets is None or c["facet"] in facets]
        for c in cells:
            c["n"] = max(5, round(c["n"] * n_scale))
        plan[model] = cells
    return plan


# --- backend construction -----------------------------------------------------------

def system_prompt_for(cell: dict) -> str:
    """The per-cell system prompt (constant across a cell's examples)."""
    if cell["facet"] in S5_FACETS:
        # The framing-specific system prompt from the single source of truth
        # (identical for every example/length of a framing).
        return S5.gen_examples(4, 1, framing=cell["settings"]["rendering"])[0][0]
    if cell["facet"] == "decomposition":
        return DECOMP_BASE_SYSTEM
    return _build_system_prompt(BASE_SYSTEM_PROMPT, cell["task"], TASK_PROMPTS)


def build_backend(model: str, cell: dict, api_key: str, base_url: str, max_workers: int) -> APIBackend:
    settings = cell["settings"]
    extra_body: dict = {}
    if settings["effort"] is not None:  # None = default arm: omit the param entirely
        extra_body["reasoning"] = {"effort": settings["effort"]}
    if MODELS[model]["open_weights"] and MODELS[model].get("quantization_filter", True):
        # Quantization filter is only meaningful for open-weight models (C4);
        # models whose endpoints don't declare a quantization opt out via the
        # registry's quantization_filter flag (the filter 404s them otherwise).
        extra_body["provider"] = {"require_parameters": False,
                                  "quantizations": ["fp8", "bf16", "fp16"]}
    return APIBackend(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_workers=max_workers,
        system_prompt=system_prompt_for(cell),
        extra_body=extra_body or None,
        # words mode only for the concrete/natural-word s5 facet (contract C1).
        answer_mode="words" if cell["facet"] == "s5_concrete" else "tokens",
    )


# --- cell execution ------------------------------------------------------------------

def _run_s5_cell(backend, cell, n) -> tuple[dict, list[dict], list[str]]:
    settings = cell["settings"]
    triples = S5.gen_examples(cell["length"], n, framing=settings["rendering"])
    prompts = [user for _sys, user, _gold in triples]
    golds = [gold for _sys, _user, gold in triples]
    preds = backend.generate(prompts, max_new_tokens=settings["max_new_tokens"],
                             stop_at=settings["stop_at"])
    scores = [S5.score(p, g) for p, g in zip(preds, golds)]
    metrics = {
        "relaxed": sum(s["relaxed"] for s in scores) / n,
        "exact": None,
        "contains": sum(s["contains"] for s in scores) / n,
        "last_n": None,
    }
    examples = [{"gold": g, "pred": p, "relaxed": s["relaxed"]}
                for g, p, s in zip(golds, preds, scores)]
    return metrics, examples, preds


def _run_decomposition_cell(backend, cell, n) -> tuple[dict, list[dict], list[str]]:
    """One routing leg, adapted from scripts/experiment_autoregressive.py.

    Leg prompts and scorers mirror that script's ``run_condition``:
      binding_only — query rewritten to ask only for the holder; score = holder
                     appears in the content tokens (the state-tracking leg alone).
      scaffolded   — correct holder injected into the prompt; score = value appears
                     in the content tokens (the recall-leg ceiling).
      end_to_end   — the unmodified composite cell, scored with the canonical
                     relaxed metric (plus the full diagnostic set).
    """
    settings = cell["settings"]
    spec = TK.CANONICAL[cell["task"]]
    examples_in = TK.generate(spec, "test", n=n, length=cell["length"])
    leg = settings["leg"]
    if leg == "binding_only":
        rewritten = [binding_prompt(e, spec.name) for e in examples_in]
        prompts = [p for p, _g in rewritten]
        golds = [g for _p, g in rewritten]
    elif leg == "scaffolded":
        prompts = [scaffold_prompt(e, spec.name) for e in examples_in]
        golds = [e.answer for e in examples_in]
    elif leg == "end_to_end":
        prompts = [e.prompt for e in examples_in]
        golds = [e.answer for e in examples_in]
    else:
        raise ValueError(f"unknown decomposition leg {leg!r}")
    preds = backend.generate(prompts, max_new_tokens=settings["max_new_tokens"],
                             stop_at=settings["stop_at"])

    rel: list[int] = []
    diag = {"exact": [], "contains": [], "last_n": []}
    for e, gold, pred in zip(examples_in, golds, preds):
        if leg == "binding_only":
            holder = e.meta.get("holder")
            rel.append(int(holder is not None and holder in TK.content_tokens(pred)))
        elif leg == "scaffolded":
            gold_ct = TK.content_tokens(e.answer)
            value = gold_ct[1] if len(gold_ct) >= 2 else None
            rel.append(int(value is not None and value in TK.content_tokens(pred)))
        else:
            pred_n, gold_n = Renderer.normalize(pred), Renderer.normalize(gold)
            rel.append(TK.score_relaxed(pred_n, gold_n))
            diag["exact"].append(TK.score_exact(pred_n, gold_n))
            diag["contains"].append(TK.score_contains(pred_n, gold_n))
            diag["last_n"].append(TK.score_last_n(pred_n, gold_n))
    metrics = {"relaxed": sum(rel) / n}
    for name in ("exact", "contains", "last_n"):
        metrics[name] = (sum(diag[name]) / n) if diag[name] else None
    examples = [{"gold": g, "pred": p, "relaxed": r}
                for g, p, r in zip(golds, preds, rel)]
    return metrics, examples, preds


def _run_task_cell(backend, cell, n) -> tuple[dict, list[dict], list[str]]:
    """Canonical-task cells (dose_response / composite_length / chain_depth / sanity)
    via the same ``evaluate_task`` path as scripts/eval_openrouter_grid.py."""
    settings = cell["settings"]
    spec = TK.CANONICAL[cell["task"]]
    result = evaluate_task(
        backend, spec, split="test", n=n, length=cell["length"],
        max_new_tokens=settings["max_new_tokens"], n_shot=settings["n_shot"],
        stop_at=settings["stop_at"],
    )
    metrics = {name: result["metrics"][name]["overall"]
               for name in ("relaxed", "exact", "contains", "last_n")}
    preds = [pred for _p, _g, pred, _ok in result["examples"]]
    examples = [{"gold": gold, "pred": pred, "relaxed": ms["relaxed"]}
                for (_prompt, gold, pred, _ok), ms in zip(result["examples"],
                                                          result["example_metrics"])]
    return metrics, examples, preds


def execute_cell(backend, model: str, cell: dict, *, n: int, run_id: str,
                 git_commit: str) -> dict:
    """Run one cell and return the C3-conformant history record (not yet written).

    Works with any ``ModelBackend``; per-call diagnostics/usage come from
    ``pop_call_meta`` when the backend provides it (APIBackend), else zeros.
    """
    t0 = time.time()
    if cell["facet"] in S5_FACETS:
        metrics, examples, preds = _run_s5_cell(backend, cell, n)
    elif cell["facet"] == "decomposition":
        metrics, examples, preds = _run_decomposition_cell(backend, cell, n)
    else:
        metrics, examples, preds = _run_task_cell(backend, cell, n)
    elapsed = time.time() - t0

    meta = backend.pop_call_meta() if hasattr(backend, "pop_call_meta") else dict(EMPTY_META)
    reg = MODELS.get(model, {})
    usage = meta["usage"]
    cost = (usage["prompt_tokens"] / 1e6 * reg.get("prompt_price_per_M", 0.0)
            + usage["completion_tokens"] / 1e6 * reg.get("completion_price_per_M", 0.0))
    empty_rate = sum(1 for p in preds if not p.strip()) / max(1, len(preds))
    return {
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "suite_version": TK.SUITE_VERSION,
        "model": model,
        "served_models": meta["served_models"],
        "providers": meta["providers"],
        "facet": cell["facet"],
        "task": cell["task"],
        "length": cell["length"],
        "n": n,
        "settings": dict(cell["settings"]),
        "metrics": metrics,
        "diagnostics": {
            "empty_rate": round(empty_rate, 4),
            "api_errors": meta["errors"],
            "finish_reasons": meta["finish_reasons"],
        },
        "usage": {**usage, "cost_usd_est": round(cost, 4)},
        "elapsed_s": round(elapsed, 2),
        # gold/pred/relaxed only — NO prompt text (prompts are deterministic and
        # regenerable; keeping them out of history reduces the contamination surface).
        "examples": examples,
    }


def append_record(history_path: str, record: dict) -> None:
    """Crash-safe single-line JSONL append (mkdir -p the parent first)."""
    parent = os.path.dirname(os.path.abspath(history_path))
    os.makedirs(parent, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# --- reporting ------------------------------------------------------------------------

def _arm_label(cell: dict) -> str:
    s = cell["settings"]
    parts = [f"effort={s['effort'] or 'default'}"]
    if s["leg"]:
        parts.append(f"leg={s['leg']}")
    if s["rendering"]:
        parts.append(f"rendering={s['rendering']}")
    return " ".join(parts)


def print_plan(plan, done, assumed_output_tokens, force, canary):
    """Dry-run: per-model cell tables with per-cell/total cost estimates."""
    grand_cost = grand_calls = grand_cells = grand_skipped = 0
    for model, cells in plan.items():
        reg = MODELS[model]
        print(f"\n>>> {model} ({reg['tier']})")
        print(f"  {'facet':<17} {'task':<19} {'L':>4} {'n':>4} {'arm':<42} {'est_$':>8}")
        model_cost = model_calls = model_skipped = 0
        for cell in cells:
            est = cost_estimate(model, [cell], assumed_output_tokens)
            skip = should_skip(model, cell, done, force, canary)
            marker = "  SKIP (in history)" if skip else ""
            print(f"  {cell['facet']:<17} {cell['task']:<19} {cell['length']:>4} "
                  f"{cell['n']:>4} {_arm_label(cell):<42} {est['cost_usd']:>8.2f}{marker}")
            if skip:
                model_skipped += 1
            else:
                model_cost += est["cost_usd"]
                model_calls += est["calls"]
        n_run = len(cells) - model_skipped
        print(f"  -- {n_run} cells to run ({model_skipped} skipped by resume), "
              f"{model_calls} calls, est ${model_cost:.2f}")
        grand_cost += model_cost
        grand_calls += model_calls
        grand_cells += n_run
        grand_skipped += model_skipped
    print(f"\nTOTAL: {grand_cells} cells to run ({grand_skipped} skipped by resume), "
          f"{grand_calls} calls, est ${grand_cost:.2f} "
          f"(assumed {assumed_output_tokens} output tokens per reasoning call, "
          f"{','.join(REASONING_EFFORTS)} arms)")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --- entry point ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run the recurring frontier benchmark (contracts C3/C4).")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS),
                    help="Registry model slugs to run (default: all).")
    ap.add_argument("--facets", nargs="+", default=None, choices=list(FACETS),
                    help="Facets to run (default: all, including sanity and floor rows).")
    ap.add_argument("--n-scale", type=float, default=1.0, dest="n_scale",
                    help="Scouting multiplier applied to each facet's n (floor 5).")
    ap.add_argument("--run-id", default=None, dest="run_id",
                    help="Run identifier (default: bench_<UTC stamp>).")
    ap.add_argument("--history", default=os.path.join(REPO, "results", "benchmark", "history.jsonl"),
                    help="History JSONL path (contract C3).")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Print the full cell plan + cost estimates, no API calls.")
    ap.add_argument("--max-workers", type=int, default=8, dest="max_workers",
                    help="Concurrent API calls per cell (default: 8).")
    ap.add_argument("--canary", action="store_true",
                    help=f"Force-rerun {CANARY_MODEL} cells even if present in history.")
    ap.add_argument("--force", action="store_true",
                    help="Force-rerun every selected cell (ignore resume).")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1", dest="base_url",
                    help="OpenRouter-compatible API base URL.")
    ap.add_argument("--assumed-output-tokens", type=int, default=2000, dest="assumed_output_tokens",
                    help="Per-call completion-token assumption for reasoning cells (cost estimate).")
    a = ap.parse_args()

    run_id = a.run_id or f"bench_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    plan = build_plan(a.models, a.facets, a.n_scale)
    done = history_keys(a.history)
    n_done = sum(1 for m, cells in plan.items() for c in cells
                 if should_skip(m, c, done, a.force, a.canary))
    print(f"run_id={run_id} history={a.history} "
          f"({len(done)} keys in history; {n_done} planned cells already present)")

    if a.dry_run:
        print_plan(plan, done, a.assumed_output_tokens, a.force, a.canary)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    git_commit = _git_commit()

    total_cost = 0.0
    for model, cells in plan.items():
        print(f"\n>>> {model} ({MODELS[model]['tier']})", flush=True)
        skipped = 0
        for cell in cells:
            if should_skip(model, cell, done, a.force, a.canary):
                skipped += 1
                continue
            tag = f"{cell['facet']}/{cell['task']}@L{cell['length']} [{_arm_label(cell)}]"
            try:
                backend = build_backend(model, cell, api_key, a.base_url, a.max_workers)
                rec = execute_cell(backend, model, cell, n=cell["n"],
                                   run_id=run_id, git_commit=git_commit)
            except Exception:  # noqa: BLE001 — one bad cell must not kill the run
                print(f"  {tag}: FAILED (no record written)", flush=True)
                traceback.print_exc()
                continue
            append_record(a.history, rec)
            done.add(cell_key(model, cell))
            d, u = rec["diagnostics"], rec["usage"]
            total_cost += u["cost_usd_est"]
            print(f"  {tag}: relaxed={rec['metrics']['relaxed']:.3f} "
                  f"empty={d['empty_rate']:.2f} err={d['api_errors']} "
                  f"rtok={u['reasoning_tokens']} ${u['cost_usd_est']:.2f} "
                  f"[{rec['elapsed_s']:.1f}s]", flush=True)
            if d["empty_rate"] > 0.5:
                print(f"  !!! WARNING {tag}: empty_rate={d['empty_rate']:.2f} > 0.5 — "
                      f"truncation suspect (check finish_reasons={d['finish_reasons']}); "
                      f"record kept.", flush=True)
        print(f"  -- done ({skipped} cells skipped by resume)", flush=True)
    print(f"\nrun {run_id} complete: total est cost ${total_cost:.2f}; history -> {a.history}")


if __name__ == "__main__":
    main()
