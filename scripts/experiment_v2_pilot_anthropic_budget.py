"""V2 pilot 3 — Anthropic thinking-budget probe on s5_concrete@L128 (PR #10 pilots).

Question: is Claude's flat ~43-101 rtok at effort=high an elicitation artifact?
OpenRouter maps ``reasoning: {max_tokens: N}`` to Anthropic's
``thinking.budget_tokens`` (effort is translated to a budget fraction); this pilot
contrasts, for opus-4.8 and sonnet-5 on s5_concrete@L128 (n=15):

  effort_high  — {"reasoning": {"effort": "high"}},      max_new_tokens=16384 (v1 arm)
  budget_4096  — {"reasoning": {"max_tokens": 4096}},    max_new_tokens=6144
  budget_16000 — {"reasoning": {"max_tokens": 16000}},   max_new_tokens=20000

A 1-call probe (--probe, also run automatically first) asserts the param is
accepted and billed reasoning tokens respect a small explicit budget before the
full spend. Appends C3-style records (plus a top-level "condition") to
results/v2_pilots/pilot3_anthropic_budget.jsonl; resume by (model, condition).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from factworld import s5_concrete as S5
from factworld.backends import APIBackend
from factworld.benchmark import _settings
from run_frontier_benchmark import _git_commit, append_record, execute_cell

PILOT_MODELS = ("anthropic/claude-opus-4.8", "anthropic/claude-sonnet-5")
LENGTH = 128
N = 15
# Cost control (observed 2026-07-08): sonnet@effort_high emitted ~12.3k ctok/call
# (~10k visible working + ~2k thinking) = $1.96/cell; at opus prices (2.5x) n=15
# would blow the pilot budget, so opus runs n=8 per condition.
N_BY_MODEL = {"anthropic/claude-sonnet-5": 15, "anthropic/claude-opus-4.8": 8}
CONDITIONS = {
    # name -> (reasoning param, max_new_tokens). max_new_tokens must exceed the
    # thinking budget (Anthropic requires max_tokens > budget_tokens) AND leave
    # ~14k of visible headroom: at s5 L128 sonnet emits ~10-12k tokens of visible
    # in-content working even while also thinking, so the first cut of this pilot
    # (budget+2048 headroom) zeroed budget_4096 outright — all 15 calls
    # finish=length, empty preds (kept in pilot3_anthropic_budget_rejected.jsonl).
    "effort_high": ({"effort": "high"}, 16384),
    "budget_4096": ({"max_tokens": 4096}, 20480),
    "budget_16000": ({"max_tokens": 16000}, 32768),
}
# Cost control: the 16000-budget arm runs on sonnet only — opus at 2.5x sonnet's
# completion price answers the budget-vs-effort question with the 4096 arm.
SKIP_CONDITIONS = {"anthropic/claude-opus-4.8": ("budget_16000",)}
OUT = os.path.join(REPO, "results", "v2_pilots", "pilot3_anthropic_budget.jsonl")


def build_backend(model: str, reasoning: dict, api_key: str) -> APIBackend:
    sysp = S5.gen_examples(4, 1, framing="concrete")[0][0]
    return APIBackend(model=model, api_key=api_key,
                      base_url="https://openrouter.ai/api/v1", max_workers=8,
                      system_prompt=sysp, extra_body={"reasoning": reasoning},
                      answer_mode="words", timeout=1800.0)


def probe(api_key: str) -> None:
    """1-call sanity: reasoning.max_tokens is accepted and caps billed rtok."""
    budget = 1024
    backend = build_backend("anthropic/claude-sonnet-5",
                            {"max_tokens": budget}, api_key)
    _sys, user, gold = S5.gen_examples(8, 1, framing="concrete")[0]
    preds = backend.generate([user], max_new_tokens=budget + 1024, stop_at=None)
    meta = backend.pop_example_meta()[0]
    agg = backend.pop_call_meta()
    print(f"probe: pred={preds[0]!r} gold={gold!r} rtok={meta['reasoning_tokens']} "
          f"ctok={meta['completion_tokens']} finish={meta['finish_reason']} "
          f"errors={agg['errors']}")
    assert agg["errors"] == 0, "probe call errored — param likely rejected"
    rtok = meta["reasoning_tokens"]
    assert rtok and rtok <= budget * 1.25, (
        f"billed rtok={rtok} does not respect budget {budget} "
        f"(0 = param ignored; >>budget = wrong param name)")
    print(f"probe OK: reasoning.max_tokens honored (rtok={rtok} <= {budget}*1.25)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run only the 1-call probe")
    ap.add_argument("--skip-probe", action="store_true")
    a = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    if a.probe or not a.skip_probe:
        probe(api_key)
        if a.probe:
            return

    done: set[tuple] = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done.add((rec["model"], rec["condition"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    run_id = f"v2pilot3_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    git_commit = _git_commit()
    for model in PILOT_MODELS:  # sonnet (cheaper) first — cost canary for opus
        n = N_BY_MODEL.get(model, N)
        for cond, (reasoning, max_new) in CONDITIONS.items():
            if cond in SKIP_CONDITIONS.get(model, ()):
                print(f"SKIP (cost policy): {model} {cond}")
                continue
            if (model, cond) in done:
                print(f"SKIP (resume): {model} {cond}")
                continue
            settings = _settings("high" if cond == "effort_high" else None,
                                 rendering="concrete", max_new_tokens=max_new)
            settings["reasoning"] = reasoning  # recorded truthfully
            cell = {"facet": "s5_concrete", "task": "s5", "length": LENGTH,
                    "n": n, "settings": settings}
            backend = build_backend(model, reasoning, api_key)
            try:
                rec = execute_cell(backend, model, cell, n=n, run_id=run_id,
                                   git_commit=git_commit)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {model} {cond}: {exc}")
                continue
            rec["condition"] = cond
            append_record(OUT, rec)
            d, u = rec["diagnostics"], rec["usage"]
            rtoks = [e["rtok"] or 0 for e in rec["examples"]]
            print(f"{model} {cond}: relaxed={rec['metrics']['relaxed']:.3f} "
                  f"contains={rec['metrics']['contains']:.3f} "
                  f"empty={d['empty_rate']:.2f} finish={d['finish_reasons']} "
                  f"mean_rtok={sum(rtoks)/max(1,len(rtoks)):.0f} "
                  f"max_rtok={max(rtoks) if rtoks else 0} "
                  f"${u['cost_usd_est']:.2f} [{rec['elapsed_s']:.0f}s]", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
