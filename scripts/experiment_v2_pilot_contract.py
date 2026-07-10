"""V2 pilot 2 — capped-contract effort=none check (PR #10 amendment pilots).

Runs the zero_budget answer-contract cell composite_copy_v1@L16 (leg None),
effort=none, n=50, max_new_tokens=96 with the runner's one-shot finish=length
escalation (to 512), on four models. Questions:
  1. sonnet's TRUE native score under extraction (v1 bounded it 0.00-0.92).
  2. does the contract kill kimi's covert in-content CoT (~2.7k ctok at none)?
  3. does grok's composite format failure vanish under last-Answer-line extraction?
  4. does anyone hit finish=length at 96 (validates the escalation path)?

Appends C3-style records to results/v2_pilots/pilot2_contract.jsonl (resume by
model). Records carry per-call {ctok, rtok, finish} plus the contract diagnostics
(contract_rate / covert_cot_rate / rtok_leak_rate) and, when escalated, the first
attempt's numbers under record["escalation"]["first_attempt"].
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from factworld.benchmark import ZERO_BUDGET_MAX_NEW_TOKENS, _settings
from run_frontier_benchmark import (
    _git_commit,
    append_record,
    build_backend,
    execute_cell,
)

PILOT_MODELS = (
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4.8",
    "moonshotai/kimi-k2.6",
    "x-ai/grok-4.20",  # slug verified live on OpenRouter 2026-07-08
)
LENGTH = 16
N = 50
OUT = os.path.join(REPO, "results", "v2_pilots", "pilot2_contract.jsonl")


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    done: set[str] = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["model"])
                except (json.JSONDecodeError, KeyError):
                    continue

    run_id = f"v2pilot2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    git_commit = _git_commit()
    for model in PILOT_MODELS:
        if model in done:
            print(f"SKIP (resume): {model}")
            continue
        cell = {
            "facet": "zero_budget", "task": "composite_copy_v1",
            "length": LENGTH, "n": N,
            "settings": _settings(
                "none", format_prompt="composite", leg=None,
                max_new_tokens=ZERO_BUDGET_MAX_NEW_TOKENS, contract=True),
        }
        backend = build_backend(model, cell, api_key,
                                "https://openrouter.ai/api/v1", 8)
        try:
            rec = execute_cell(backend, model, cell, n=N, run_id=run_id,
                               git_commit=git_commit)
        except Exception as exc:  # noqa: BLE001 — keep the other cells running
            print(f"FAILED {model}: {exc}")
            continue
        append_record(OUT, rec)
        d, u = rec["diagnostics"], rec["usage"]
        ctoks = [e["ctok"] for e in rec["examples"] if e["ctok"]]
        esc = ""
        if rec["escalated"]:
            first = rec["escalation"]["first_attempt"]
            esc = (f" ESCALATED (96: relaxed={first['relaxed']:.2f} "
                   f"length_rate={first['length_rate']:.2f})")
        print(f"{model}: relaxed={rec['metrics']['relaxed']:.3f} "
              f"contains={rec['metrics']['contains']:.3f} "
              f"contract={d['contract_rate']:.2f} covert={d['covert_cot_rate']:.2f} "
              f"leak={d['rtok_leak_rate']:.2f} empty={d['empty_rate']:.2f} "
              f"finish={d['finish_reasons']} "
              f"mean_ctok={sum(ctoks)/max(1,len(ctoks)):.0f} "
              f"${u['cost_usd_est']:.2f}{esc}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
