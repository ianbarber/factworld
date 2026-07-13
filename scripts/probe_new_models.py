"""Pre-registry probes for the issue #15 roster candidates (2026-07-12).

Probes x-ai/grok-4.5 and openai/gpt-5.6-sol BEFORE any registry entry, per the
registry's x-ai caveats (mainline grok's endpoint bio-safety filter blocked ~56%
of the g/v-token composite prompts with finish_reason=content_filter /
SAFETY_CHECK_TYPE_BIO; grok-build's provider pinned reasoning at ~256k tokens
ignoring the requested cap). Probe battery:

  gpt-5.6-sol:
    1 call, reasoning effort=none on a composite contract prompt (support check;
    pricing separately verified against /api/v1/models: $5/$30 per M).

  grok-4.5 (cost-ordered — the CAP check gates the filter fan-out, so a pinned
  ~256k generator can burn at most one call):
    1. effort=none support on a composite contract prompt (doubles as filter
       call #1).
    2. capped reasoning call (effort=high, max_tokens=1024): billed
       usage.completion_tokens must respect the cap (<= 1.2x tolerance).
       If billed ctok exceeds 2x the cap, the remaining probes are ABORTED.
    3. filter fan-out: 2 more composite contract prompts + 1 chain d16
       (staircase k=33) prompt — any finish_reason=content_filter reproduces
       the mainline-grok blocker.

Prompts/system prompts are byte-identical to what run_frontier_benchmark.py
sends for the corresponding cells (zero_budget composite contract cells;
chain_nowrap d16). Raw responses go to results/probes/new_models_20260712.jsonl.

Usage:  set -a; source .env; set +a
        .venv-api/bin/python scripts/probe_new_models.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from factworld import tasks as TK
from factworld.benchmark import spec_for_cell
from run_frontier_benchmark import (
    BASE_SYSTEM_PROMPT,
    CONTRACT_LINE_COMPOSITE,
    TASK_PROMPTS,
)
from eval_openrouter_grid import _build_system_prompt

OUT = os.path.join(REPO, "results", "probes", "new_models_20260712.jsonl")
GROK = "x-ai/grok-4.5"
SOL = "openai/gpt-5.6-sol"
CAP_TOLERANCE = 1.2   # billed ctok may exceed max_tokens by a small margin
CAP_ABORT = 2.0       # past this, treat the provider as cap-ignoring and stop


def call(client, *, probe, model, system_prompt, user_prompt, max_tokens,
         effort=None):
    req = {"model": model, "max_tokens": max_tokens, "effort": effort}
    extra_body = {"reasoning": {"effort": effort}} if effort is not None else None
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "probe": probe,
           "request": req, "prompt_chars": len(user_prompt)}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=0, top_p=1, max_tokens=max_tokens,
            extra_body=extra_body,
        )
        rec["response"] = resp.model_dump()
        rec["error"] = None
    except Exception as exc:  # noqa: BLE001 — a 400 IS a probe result
        rec["response"] = None
        rec["error"] = f"{type(exc).__name__}: {exc}"
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")

    if rec["error"]:
        print(f"[{probe}] ERROR: {rec['error']}")
        return rec
    r = rec["response"]
    ch = (r.get("choices") or [{}])[0]
    usage = r.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    rtok = details.get("reasoning_tokens", usage.get("reasoning_tokens"))
    content = (ch.get("message") or {}).get("content") or ""
    print(f"[{probe}] finish={ch.get('finish_reason')} "
          f"native={ch.get('native_finish_reason')} "
          f"ptok={usage.get('prompt_tokens')} ctok={usage.get('completion_tokens')} "
          f"rtok={rtok} provider={r.get('provider')} served={r.get('model')}")
    print(f"  content[:160]: {content[:160]!r}")
    return rec


def ctok(rec):
    if not rec.get("response"):
        return None
    return (rec["response"].get("usage") or {}).get("completion_tokens")


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
                    timeout=600.0)

    # The exact zero_budget composite cell prompts (composite_copy_v2 @L16,
    # contract line) and the chain_nowrap d16 staircase prompt (k=33).
    comp_sys = _build_system_prompt(BASE_SYSTEM_PROMPT, "composite_copy_v2",
                                    TASK_PROMPTS)
    comp = TK.generate(TK.CANONICAL["composite_copy_v2"], "test", n=3, length=16)
    comp_prompts = [f"{e.prompt}\n{CONTRACT_LINE_COMPOSITE}" for e in comp]
    chain_spec = spec_for_cell("chain_v1", 16)
    chain = TK.generate(chain_spec, "test", n=1, length=16)[0]

    # --- gpt-5.6-sol: effort=none support ------------------------------------
    call(client, probe="sol_effort_none", model=SOL, system_prompt=comp_sys,
         user_prompt=comp_prompts[0], max_tokens=96, effort="none")

    # --- grok-4.5, cost-ordered ----------------------------------------------
    call(client, probe="grok_effort_none", model=GROK, system_prompt=comp_sys,
         user_prompt=comp_prompts[0], max_tokens=96, effort="none")

    cap = call(client, probe="grok_cap_1024_high", model=GROK,
               system_prompt=comp_sys, user_prompt=comp_prompts[0],
               max_tokens=1024, effort="high")
    c = ctok(cap)
    if c is not None and c > CAP_ABORT * 1024:
        print(f"!!! grok billed ctok {c} > {CAP_ABORT}x the 1024 cap — "
              f"cap-ignoring provider (grok-build pathology). ABORTING the "
              f"filter fan-out.")
        return
    if c is not None and c > CAP_TOLERANCE * 1024:
        print(f"!!! grok billed ctok {c} exceeds the cap tolerance "
              f"({CAP_TOLERANCE}x1024) — inspect before adding.")

    call(client, probe="grok_filter_composite_2", model=GROK,
         system_prompt=comp_sys, user_prompt=comp_prompts[1],
         max_tokens=1024, effort="low")
    call(client, probe="grok_filter_composite_3", model=GROK,
         system_prompt=comp_sys, user_prompt=comp_prompts[2],
         max_tokens=1024, effort="low")
    call(client, probe="grok_filter_chain_d16", model=GROK,
         system_prompt=BASE_SYSTEM_PROMPT, user_prompt=chain.prompt,
         max_tokens=1024, effort="low")

    print(f"\nprobe records -> {OUT}")


if __name__ == "__main__":
    main()
