"""Construct-validity test for S₅: is the frontier-model floor a capability wall or a presentation wall?

Same S₅ problems (same world state, same permutation sequences, same oracle gold) rendered under
three framings, evaluated on the same models. The only thing that varies is the surface text, so a
change in accuracy attributes the gap to presentation, not computation.

  V0  abstract baseline       — g/r tokens, "swaps"/"cycles roles", initial assignment UNSTATED
  V0' abstract + stated init   — same, but "Initially g0=r0, g1=r1, ... " is given
  V1  concrete English         — people + jobs ("Eva and Bob swap jobs", "Cara takes Eva's job, ...",
                                 "what job does Cara have?"), initial stated

The events come from the task's deterministic sampler (``sample_hard_chain`` + ``_rng`` for the
queried agent), and the gold from the symbolic oracle — so every framing is the same problem, just
re-worded. A cycle is rendered explicitly ("X takes Y's job") matching the oracle's semantics, so
there is no arrow-direction ambiguity.

Crash-safe: each (model, framing, length) cell is appended to a JSONL as it completes.

Smoke test (prints sample prompts, no API):
    python scripts/experiment_s5_framing.py --print-samples
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld.tasks import CANONICAL, _world  # noqa: E402
from factworld.s5_concrete import (  # noqa: E402,F401 — single source of truth for renderings
    NAMES, JOBS, FRAMINGS, INIT_ABSTRACT, INIT_CONCRETE,
    render_event_v0, render_event_v1, render_prompt, gen_problems, score,
)


def gen_examples(spec, w, oracle, length, n):
    """Deterministic (events, agent, gold) list — identical problems across framings."""
    return gen_problems(spec, w, oracle, length, n)


def run_cell(backend, framing, examples, max_new_tokens, stop_at):
    sys_prompts, users, golds = [], [], []
    for events, agent, gold in examples:
        s, u, g = render_prompt(framing, events, agent, gold)
        sys_prompts.append(s); users.append(u); golds.append(g)
    backend.system_prompt = sys_prompts[0]   # all examples in a cell share the framing's system prompt
    preds = backend.generate(users, max_new_tokens=max_new_tokens, stop_at=stop_at)
    rows = []
    for u, g, pred in zip(users, golds, preds):
        sc = score(pred, g)
        rows.append({"gold": g, "pred": pred, "relaxed": sc["relaxed"], "contains": sc["contains"]})
    acc_relaxed = sum(r["relaxed"] for r in rows) / len(rows)
    acc_contains = sum(r["contains"] for r in rows) / len(rows)
    return acc_relaxed, acc_contains, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["z-ai/glm-5.2", "moonshotai/kimi-k2.6",
                             "meta-llama/llama-3.3-70b-instruct", "openai/gpt-4o-mini"])
    ap.add_argument("--framings", nargs="+", default=list(FRAMINGS))
    ap.add_argument("--lengths", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--stop_at", default=".",
                    help="Stop string; 'none' or '' disables it (needed for reasoning models so the "
                         "<think> scratchpad is not truncated before the answer).")
    ap.add_argument("--no_reasoning", action="store_true", default=True)
    ap.add_argument("--reasoning", dest="no_reasoning", action="store_false")
    ap.add_argument("--out", default="docs/openrouter/s5-framing.jsonl")
    ap.add_argument("--print-samples", action="store_true",
                    help="Print one rendered prompt per framing at L8 and exit (no API calls).")
    a = ap.parse_args()

    spec = CANONICAL["s5_v1"]
    w, _r, oracle = _world(spec)

    if a.print_samples:
        ex = gen_examples(spec, w, oracle, 8, 1)[0]
        for fr in a.framings:
            s, u, g = render_prompt(fr, *ex)
            print(f"\n========== {fr}  (gold={g}) ==========")
            print("SYSTEM:", s)
            print("USER:", u)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    from factworld.backends import APIBackend

    stop_at = None if a.stop_at.lower() in ("", "none") else a.stop_at
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"=== s5 framing sweep: {len(a.models)} models x {len(a.framings)} framings x "
          f"{len(a.lengths)} lengths x n={a.n} -> {a.out} ===", flush=True)

    for model in a.models:
        extra_body = {"reasoning": {"effort": "none"}} if a.no_reasoning else None
        backend = APIBackend(model=model, api_key=api_key,
                             base_url="https://openrouter.ai/api/v1", max_workers=4,
                             extra_body=extra_body)
        for length in a.lengths:
            examples = gen_examples(spec, w, oracle, length, a.n)
            for framing in a.framings:
                t0 = time.time()
                acc_r, acc_c, rows = run_cell(backend, framing, examples,
                                              a.max_new_tokens, stop_at=stop_at)
                rec = {"model": model, "framing": framing, "length": length, "n": a.n,
                       "acc_relaxed": acc_r, "acc_contains": acc_c,
                       "no_reasoning": a.no_reasoning, "elapsed": time.time() - t0,
                       "examples": rows}
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"  {model:<32} {framing:<26} L{length:<3} "
                      f"relaxed={acc_r:.2f} contains={acc_c:.2f} ({a.n}) [{rec['elapsed']:.0f}s]",
                      flush=True)


if __name__ == "__main__":
    main()
