"""Composition-under-thinking frontier probe on composite_copy_v2 (design probe, not a benchmark row).

The de-skewed v2 sampler places the queried object's resolving write ~Uniform[0.1L, L-2],
so deep L is genuinely deep binding (the retired v1 recency shortcut kept everything near
the stream end). On the OLD skewed task glm-5.2 held 0.93-1.00 under thinking to L512;
this probe staircases L = 64..1024 on v2 to find where composition-under-thinking breaks.

Arms (house probe protocol):
  - thinking: effort=high via extra_body reasoning param, answer_mode=tokens, stop_at=None,
    max_new_tokens 16384 through L256 / 32768 at L512+ (budgets scale with L so a budget
    knee is never mistaken for a capability knee).
  - instant: effort=none + the Answer:-contract line + max_new_tokens 96, answer_mode=raw,
    last-Answer:-line extraction (no escalation — finish=length rate is recorded and read).

Per-call {ctok, rtok, finish} recorded per example; crash-safe JSONL appends (resume by
(model, arm, k, length, budget_mult)); deterministic items; relaxed match canonical with
the composite two-token format instruction appended (the frontier runner's TASK_PROMPTS
text, i.e. eval_openrouter_grid.COMPOSITE_FORMAT_PROMPT). Output goes to
results/composite_frontier_*.jsonl — NEVER results/benchmark/history.jsonl (these are
design probes, same convention as results/v2_pilots/).

Examples:
    set -a; source .env; set +a
    # sanity only (no spend): render + gold-recompute + token estimate at L1024
    .venv-api/bin/python scripts/experiment_composite_frontier.py --sanity-only

    # glm staircase, both arms
    .venv-api/bin/python scripts/experiment_composite_frontier.py \\
        --model z-ai/glm-5.2 --lengths 64 128 256 512 1024 --arm both

    # breadth probe at the deepest held L
    .venv-api/bin/python scripts/experiment_composite_frontier.py \\
        --model z-ai/glm-5.2 --k 64 --lengths 512 --arm thinking
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from factworld import tasks as TK
from factworld.backends import APIBackend
from factworld.render import Renderer

# The composite two-token format instruction — the exact text the frontier runner's
# TASK_PROMPTS maps composite_copy_v2 to (run_frontier_benchmark.py imports the same
# constant). Imported from the grid script, NOT from the runner, so this probe stays
# decoupled from the files the benchmark workflow is concurrently editing.
from eval_openrouter_grid import COMPOSITE_FORMAT_PROMPT

# Copied verbatim from scripts/run_frontier_benchmark.py (not imported: that file and
# factworld/benchmark.py are being edited concurrently by the benchmark workflow).
BASE_SYSTEM_PROMPT = (
    "You are taking a short test. Answer each question with only the requested "
    "value or values, no explanation. Use the same spelling as in the question."
)
CONTRACT_LINE_COMPOSITE = "Reply with only one line: Answer: <holder> <value>"
_ANSWER_LINE_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)

TASK = "composite_copy_v2"
DEFAULT_LENGTHS = (64, 128, 256, 512, 1024)
INSTANT_MAX_NEW_TOKENS = 96
STAIRCASE_STOP_RELAXED = 0.5   # thinking staircase stops early below this ...
BUDGET_SUSPECT_EMPTY = 0.2     # ... only when empty_rate is below this (a genuine knee)

# OpenRouter pricing snapshot 2026-07-09 ($/token in, $/token out) — for the running
# spend estimate printed per cell (ground truth is the credits endpoint).
PRICES = {
    "z-ai/glm-5.2": (0.93e-6, 3.0e-6),
    "qwen/qwen3.7-max": (1.25e-6, 3.75e-6),
}
# Open-weight models get the benchmark's quantization filter (fp8/bf16/fp16).
OPEN_WEIGHTS = {"z-ai/glm-5.2"}


def budget_for(length: int) -> int:
    """Thinking-arm budget scales with L: 16384 through L256, 32768 at L512+."""
    return 16384 if length <= 256 else 32768


def extract_contract_answer(text: str) -> str | None:
    """Answer span after the LAST ``Answer:`` line (copied from the frontier runner)."""
    spans = [m.strip().strip("*_` ") for m in _ANSWER_LINE_RE.findall(text)]
    spans = [s for s in spans if s]
    return spans[-1] if spans else None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def build_spec(k: int | None) -> TK.TaskSpec:
    """CANONICAL composite_copy_v2, optionally scaled to breadth k (pool scales with it)."""
    spec = TK.CANONICAL[TASK]
    if k is not None:
        # breadth probe: scale agents AND the 1-of-N recall pool together (the pool is
        # both the fact set and the give-recipient set, so this is the breadth axis).
        spec = spec.scaled(k=k, recall_pool=k)
    return spec


# --- pre-spend sanity gate ------------------------------------------------------------
_GIVE_RE = re.compile(r"gives (o\d+) to (g\d+)\.")
_FACT_RE = re.compile(r"(g\d+)'s a0 is (v\d+)\.")
_QUERY_RE = re.compile(r"the holder of (o\d+)\?")


def sanity_check(spec: TK.TaskSpec, length: int, n: int = 5) -> None:
    """Before ANY spend: render n items at the deepest planned L, log the resolving-write
    position distribution, recompute gold by last-write-wins from the RENDERED prompt,
    and report a prompt-token estimate. Raises on any mismatch."""
    items = TK.generate(spec, "test", n=n, length=length)
    print(f"--- sanity gate: {spec.name} k={spec.k} pool={spec.recall_pool} L={length} n={n} ---")
    for i, ex in enumerate(items):
        obj = _QUERY_RE.search(ex.prompt).group(1)
        gives = _GIVE_RE.findall(ex.prompt)
        holder = None
        for o, h in gives:                       # last-write-wins from the rendered text
            if o == obj:
                holder = h
        facts = dict(_FACT_RE.findall(ex.prompt))
        recomputed = f"{holder} {facts[holder]}."
        assert recomputed == ex.answer, (
            f"gold mismatch item {i}: recomputed {recomputed!r} vs oracle {ex.answer!r}")
        pos = ex.meta.get("last_write_pos")
        assert pos is not None and length // 10 <= pos <= length - 2, f"bad pos {pos}"
        n_words = len(ex.prompt.split())
        est_tok = int(len(ex.prompt) / 3.5)      # rough chat-tokenizer estimate
        print(f"  item {i}: obj={obj} last_write_pos={pos} "
              f"(frac {pos/length:.2f}) gold={ex.answer!r} words={n_words} est_tok~{est_tok}")
    print(f"  gold recomputes from rendered prompt on all {n} items; "
          f"resolving-write positions span the stream (Uniform[{length//10}, {length-2}]).")


# --- cell execution -------------------------------------------------------------------
def build_backend(model: str, arm: str, api_key: str, max_workers: int) -> APIBackend:
    extra_body: dict = {"reasoning": {"effort": "high" if arm == "thinking" else "none"}}
    if model in OPEN_WEIGHTS:
        extra_body["provider"] = {"require_parameters": False,
                                  "quantizations": ["fp8", "bf16", "fp16"]}
    return APIBackend(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_workers=max_workers,
        system_prompt=f"{BASE_SYSTEM_PROMPT} {COMPOSITE_FORMAT_PROMPT}",
        extra_body=extra_body,
        answer_mode="raw" if arm == "instant" else "tokens",
        timeout=1800.0,
    )


def run_cell(model: str, spec: TK.TaskSpec, arm: str, length: int, n: int,
             api_key: str, max_workers: int, budget_mult: int = 1) -> dict:
    items = TK.generate(spec, "test", n=n, length=length)
    if arm == "thinking":
        prompts = [ex.prompt for ex in items]
        max_new = budget_for(length) * budget_mult
    else:
        prompts = [f"{ex.prompt}\n{CONTRACT_LINE_COMPOSITE}" for ex in items]
        max_new = INSTANT_MAX_NEW_TOKENS * budget_mult

    backend = build_backend(model, arm, api_key, max_workers)
    t0 = time.time()
    raw_preds = backend.generate(prompts, max_new_tokens=max_new, stop_at=None)
    elapsed = time.time() - t0
    ex_meta = backend.pop_example_meta()
    call_meta = backend.pop_call_meta()

    examples, hits, empties, cap_outs = [], 0, 0, 0
    for ex, raw, m in zip(items, raw_preds, ex_meta):
        pred = (extract_contract_answer(raw) or "") if arm == "instant" else raw
        pred_n = Renderer.normalize(pred)
        gold_n = Renderer.normalize(ex.answer)
        ok = TK.score_relaxed(pred_n, gold_n)
        hits += ok
        empty = not pred_n.strip()
        empties += empty
        finish = m.get("finish_reason")
        cap_outs += finish == "length"
        examples.append({
            "gold": ex.answer, "pred": pred[:200], "relaxed": ok,
            "empty": int(empty), "last_write_pos": ex.meta.get("last_write_pos"),
            "ctok": m.get("completion_tokens"), "rtok": m.get("reasoning_tokens"),
            "finish": finish,
        })

    relaxed = hits / n
    lo, hi = wilson_ci(hits, n)
    ctoks = [e["ctok"] or 0 for e in examples]
    rtoks = [e["rtok"] or 0 for e in examples]
    usage = call_meta["usage"]
    price = PRICES.get(model, (0.0, 0.0))
    cost = usage["prompt_tokens"] * price[0] + \
        (usage["completion_tokens"] + usage["reasoning_tokens"]) * price[1]
    empty_rate = empties / n
    rec = {
        "run_kind": "composite_frontier_probe",
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model": model, "task": TASK, "arm": arm,
        "k": spec.k, "recall_pool": spec.recall_pool, "length": length, "n": n,
        "settings": {"effort": "high" if arm == "thinking" else "none",
                     "max_new_tokens": max_new, "budget_mult": budget_mult,
                     "stop_at": None, "contract": arm == "instant",
                     "format_prompt": COMPOSITE_FORMAT_PROMPT},
        "metrics": {"relaxed": round(relaxed, 4), "wilson95": [round(lo, 4), round(hi, 4)]},
        "diagnostics": {
            "empty_rate": round(empty_rate, 4),
            "finish_length_rate": round(cap_outs / n, 4),
            "budget_suspect": empty_rate > BUDGET_SUSPECT_EMPTY,
            "api_errors": call_meta["errors"],
            "finish_reasons": call_meta["finish_reasons"],
            "ctok_mean": round(statistics.mean(ctoks), 1),
            "ctok_median": statistics.median(ctoks),
            "rtok_mean": round(statistics.mean(rtoks), 1),
            "rtok_median": statistics.median(rtoks),
            "elapsed_s": round(elapsed, 1),
            "providers": call_meta["providers"],
            "served_models": call_meta["served_models"],
        },
        "usage": usage, "cost_est_usd": round(cost, 4),
        "examples": examples,
    }
    return rec


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def cell_key(rec_or: dict) -> tuple:
    return (rec_or["model"], rec_or["arm"], rec_or["k"], rec_or["length"],
            rec_or["settings"]["budget_mult"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="z-ai/glm-5.2")
    ap.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    ap.add_argument("--k", type=int, default=None,
                    help="breadth override: spec.scaled(k=K, recall_pool=K)")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--arm", choices=("thinking", "instant", "both"), default="both")
    ap.add_argument("--out", default=os.path.join(
        REPO, "results", f"composite_frontier_{datetime.now().strftime('%Y%m%d')}.jsonl"))
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--budget-mult", type=int, default=1,
                    help="2 = the one-shot 2x re-run of a budget-suspect cell")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="disable the thinking-staircase capability-knee stop")
    ap.add_argument("--sanity-only", action="store_true")
    a = ap.parse_args()

    spec = build_spec(a.k)
    lengths = sorted(a.lengths)
    sanity_check(spec, lengths[-1], n=5)          # ALWAYS before any spend
    if a.sanity_only:
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    done: set[tuple] = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(cell_key(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue

    arms = ("thinking", "instant") if a.arm == "both" else (a.arm,)
    thinking_stopped = False
    total_cost = 0.0
    for L in lengths:                             # ascending staircase
        for arm in arms:
            if arm == "thinking" and thinking_stopped:
                print(f"SKIP thinking L{L}: staircase stopped at a capability knee")
                continue
            key = (a.model, arm, spec.k, L, a.budget_mult)
            if key in done:
                print(f"SKIP (resume): {a.model} {arm} k={spec.k} L{L}")
                continue
            rec = run_cell(a.model, spec, arm, L, a.n, api_key, a.max_workers,
                           budget_mult=a.budget_mult)
            with open(a.out, "a", encoding="utf-8") as fh:  # crash-safe append per cell
                fh.write(json.dumps(rec) + "\n")
            total_cost += rec["cost_est_usd"]
            d = rec["diagnostics"]
            print(f"CELL {a.model} {arm} k={spec.k} L{L} n={a.n}: "
                  f"relaxed={rec['metrics']['relaxed']:.2f} "
                  f"CI[{rec['metrics']['wilson95'][0]:.2f},{rec['metrics']['wilson95'][1]:.2f}] "
                  f"empty={d['empty_rate']:.2f} len_rate={d['finish_length_rate']:.2f} "
                  f"ctok_med={d['ctok_median']} rtok_med={d['rtok_median']} "
                  f"cost~${rec['cost_est_usd']:.2f} cum~${total_cost:.2f} "
                  f"{'BUDGET-SUSPECT' if d['budget_suspect'] else ''}")
            if (arm == "thinking" and not a.no_early_stop
                    and rec["metrics"]["relaxed"] < STAIRCASE_STOP_RELAXED
                    and d["empty_rate"] < BUDGET_SUSPECT_EMPTY):
                thinking_stopped = True
                print(f"STAIRCASE STOP: thinking relaxed "
                      f"{rec['metrics']['relaxed']:.2f} < {STAIRCASE_STOP_RELAXED} with "
                      f"empty_rate {d['empty_rate']:.2f} < {BUDGET_SUSPECT_EMPTY} at L{L} "
                      f"(capability knee, not budget)")
    print(f"done. total estimated cell cost ~${total_cost:.2f} -> {a.out}")


if __name__ == "__main__":
    main()
