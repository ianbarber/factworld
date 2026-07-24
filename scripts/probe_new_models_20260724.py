"""Pre-registry probes for the 2026-07-24 roster candidates (roster refresh).

Candidates: anthropic/claude-fable-5, moonshotai/kimi-k3, google/gemini-3.6-flash.
Per candidate, cost-ordered:
  1. effort=none support on a composite contract prompt — does the endpoint accept
     a reasoning-off arm, does it emit reasoning tokens anyway (covert-CoT risk),
     and does it obey the answer contract (instant-cell eligibility).
  2. capped reasoning call (effort=high, max_tokens=1024) — billed
     usage.completion_tokens must respect the cap (<=1.2x tolerance; past 2x the
     provider is cap-ignoring: the ‡ risk).

Prompts are byte-identical to the zero_budget composite contract cells. Raw
responses -> results/probes/new_models_20260724.jsonl.

Usage:  set -a; source .env; set +a
        .venv-api/bin/python scripts/probe_new_models_20260724.py
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
from run_frontier_benchmark import (
    BASE_SYSTEM_PROMPT,
    CONTRACT_LINE_COMPOSITE,
    TASK_PROMPTS,
)
from eval_openrouter_grid import _build_system_prompt

OUT = os.path.join(REPO, "results", "probes", "new_models_20260724.jsonl")
CANDIDATES = ("anthropic/claude-fable-5", "moonshotai/kimi-k3", "google/gemini-3.6-flash")
CAP = 1024
CAP_TOLERANCE = 1.2


def contract_prompt():
    spec = TK.CANONICAL["composite_copy_v2"]
    ex = TK.generate(spec, "test", n=1, length=16)[0]
    system = _build_system_prompt(BASE_SYSTEM_PROMPT, "composite_copy_v2", TASK_PROMPTS)
    return system, f"{ex.prompt}\n{CONTRACT_LINE_COMPOSITE}"


def call(client, model, system, user, max_tokens, effort):
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "extra_body": {"reasoning": {"effort": effort}}}
    try:
        r = client.chat.completions.create(**body)
        ch = r.choices[0]
        u = r.usage
        det = getattr(u, "completion_tokens_details", None)
        rtok = getattr(det, "reasoning_tokens", None) if det else None
        return {"ok": True, "finish": ch.finish_reason, "text": (ch.message.content or "")[:400],
                "ctok": u.completion_tokens, "rtok": rtok}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def main():
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    system, user = contract_prompt()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as fh:
        for model in CANDIDATES:
            for probe, effort, cap in (("effort_none_contract", "none", 96),
                                        ("cap_respect_high", "high", CAP)):
                res = call(client, model, system, user, cap, effort)
                rec = {"ts": datetime.now(timezone.utc).isoformat(), "model": model,
                       "probe": probe, "effort": effort, "max_tokens": cap, **res}
                fh.write(json.dumps(rec) + "\n")
                verdict = ""
                if probe == "cap_respect_high" and res.get("ok"):
                    ratio = (res["ctok"] or 0) / cap
                    verdict = f" cap_ratio={ratio:.2f}" + (" CAP-ESCAPE" if ratio > CAP_TOLERANCE else " ok")
                print(f"{model:<32} {probe:<22} -> " +
                      (f"finish={res['finish']} ctok={res['ctok']} rtok={res['rtok']}{verdict} text={res['text'][:60]!r}"
                       if res.get("ok") else f"ERROR {res['error']}"), flush=True)
    print(f"raw -> {OUT}")


if __name__ == "__main__":
    main()
