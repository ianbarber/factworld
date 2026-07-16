"""Sweep reasoning_effort for openai/gpt-5.6-sol on s5 @L256 via direct OpenAI.

Outputs a JSONL of per-effort records; not committed to the benchmark history.
Review the results before deciding whether to add a line to the report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import s5_concrete as S5
from factworld.backends import APIBackend

MODEL = "openai/gpt-5.6-sol"
API_NAME = "gpt-5.6-sol"
BASE_URL = "https://api.openai.com/v1"
KEY_ENV = "OPENAI_API_KEY"
SYSTEM_PROMPT = (
    "You are taking a short test. Answer each question with only the requested value or values, "
    "no explanation. Use the same spelling as in the question."
)


def run_effort(effort: str | None, n: int, max_new_tokens: int, max_workers: int) -> dict:
    api_key = os.environ.get(KEY_ENV)
    if not api_key:
        raise SystemExit(f"{KEY_ENV} not set")

    triples = S5.gen_examples(256, n, framing="concrete")
    prompts = [user for _sys, user, _gold in triples]
    golds = [gold for _sys, _user, gold in triples]

    backend = APIBackend(
        model=MODEL,
        api_key=api_key,
        base_url=BASE_URL,
        model_name=API_NAME,
        max_workers=max_workers,
        system_prompt=SYSTEM_PROMPT,
        answer_mode="words",
        timeout=1800.0,
        max_completion_tokens=True,
        reasoning_model=True,
        reasoning_effort=effort,
    )
    preds = backend.generate(prompts, max_new_tokens=max_new_tokens, stop_at=None)

    examples = []
    for gold, pred in zip(golds, preds):
        s = S5.score(pred, gold)
        examples.append({"gold": gold, "pred": pred, **s})

    meta = backend.pop_example_meta()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    for m in meta:
        for k in usage:
            usage[k] += m.get(k, 0)
    n_calls = len(meta) or 1
    for k in usage:
        usage[k] /= n_calls

    relaxed = sum(e["relaxed"] for e in examples) / len(examples)
    contains = sum(e["contains"] for e in examples) / len(examples)
    return {
        "effort": effort,
        "n": len(examples),
        "max_new_tokens": max_new_tokens,
        "relaxed": relaxed,
        "contains": contains,
        "usage_per_call": usage,
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--efforts", nargs="+", default=["default", "low", "medium", "high"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--out", default="results/sol_s5_reasoning_sweep.jsonl")
    a = ap.parse_args()

    out_path = os.path.join(REPO, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Sweeping {MODEL} on s5 @L256; writing to {out_path}", flush=True)
    for effort in a.efforts:
        effort_val = None if effort == "default" else effort
        tag = effort if effort_val else "default"
        print(f"\n--- effort={tag} ---", flush=True)
        rec = run_effort(effort_val, a.n, a.max_new_tokens, a.max_workers)
        rec["model"] = MODEL
        rec["length"] = 256
        rec["ts"] = datetime.now(timezone.utc).isoformat()
        print(f"    relaxed={rec['relaxed']:.2f} contains={rec['contains']:.2f} "
              f"ctok/call={rec['usage_per_call']['completion_tokens']:.0f} "
              f"rtok/call={rec['usage_per_call']['reasoning_tokens']:.0f}", flush=True)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
