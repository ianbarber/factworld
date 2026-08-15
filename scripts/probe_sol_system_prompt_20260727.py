"""Does the harness system prompt suppress reasoning engagement? (gpt-5.6-sol, s5_chain_v3 @L64)

gpt-5.6-sol's published s5_chain row is its ENGAGEMENT RATE, not its accuracy: per-call
completion tokens are bimodal with a literal gap (rtok jumps 286 -> 1128 across the 300
archived calls), accuracy conditional on engaging is 191/191, and 42/42 of its wrong
answers are the 8-hop dereference of the INITIAL map — i.e. it answers the `chain` task and
skips the event stream. What the archive cannot say is WHY.

One candidate is under the experimenter's control. Every thinking cell carries
``run_frontier_benchmark.BASE_SYSTEM_PROMPT`` — "You are taking a short test. Answer each
question with only the requested value or values, no explanation." Two clauses there
("a short test", "no explanation") are plausible non-engagement triggers, and if they
suppress deliberation the effect is not confined to one model: it applies to every scored
thinking cell in the benchmark.

Three arms on the IDENTICAL deterministic items (s5_chain_v3, L64, n=25 — the matched cell
the report prices token spend on), same endpoint, same effort, same budget as the scored
facet:

  canonical  — BASE_SYSTEM_PROMPT verbatim (the scored protocol)
  none       — no system prompt at all
  neutral    — the same answer-format contract with the two suppressive clauses removed

Decision rule, pre-registered:
  * engagement rises materially (>= ~0.2) on `none`/`neutral` -> the scored thinking regime
    is measured under an anti-thinking instruction; the facet needs re-running under a
    neutral prompt and no cross-model ordering survives as published.
  * engagement flat across arms -> the gate is endogenous to the model/endpoint, and the
    protocol fix is a per-cell work-rate diagnostic, not a prompt change.

Spend guard: hard USD cap (default $15), checked between chunks; the run stops submitting
and records what completed rather than overrunning.

Not written to results/benchmark/history.jsonl — this is an off-protocol probe (a different
system prompt is a different measurement), so it lands in results/probes/.
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

from factworld import tasks as TK
from factworld.backends import ResponsesBackend
from factworld.render import Renderer

MODEL = "openai/gpt-5.6-sol"
API_NAME = "gpt-5.6-sol"
BASE_URL = "https://api.openai.com/v1"
KEY_ENV = "OPENAI_API_KEY"
PROMPT_PRICE_PER_M = 5.0
COMPLETION_PRICE_PER_M = 30.0

# The scored protocol's system prompt (scripts/run_frontier_benchmark.py BASE_SYSTEM_PROMPT).
CANONICAL = (
    "You are taking a short test. Answer each question with only the requested "
    "value or values, no explanation. Use the same spelling as in the question."
)
# Same output contract, minus "a short test" and "no explanation" — the two clauses that
# could read as instructions to spend less effort. Nothing else changes.
NEUTRAL = (
    "Answer the question with only the requested value or values. "
    "Use the same spelling as in the question."
)
ARMS = {"canonical": CANONICAL, "none": None, "neutral": NEUTRAL}

# Engagement split. The archived gap is 286 -> 1128 reasoning tokens, so any threshold in
# between yields the same partition; 500 sits in it and is the repo's working-line scale.
ENGAGE_CTOK = 500


def score(pred: str, gold: str) -> int:
    """The scored path for a thinking cell: committed-answer extraction, then match."""
    return TK.score_relaxed(
        Renderer.normalize(TK.committed_answer(pred)), Renderer.normalize(gold)
    )


def initial_map_answer(ex) -> str:
    """The 8-hop dereference of the INITIAL map — the answer an event-blind reader gives.

    Parsed out of the rendered fact block, so it needs nothing but the prompt text.
    """
    import re

    nxt = dict(re.findall(r"(g\d+)'s a0 is (g\d+)\.", ex.prompt))
    x = ex.meta["start"]
    for _ in range(ex.meta["depth"]):
        x = nxt[x]
    return x


def run_arm(name: str, system_prompt: str | None, examples, budget: int,
            max_workers: int, usd_left: float) -> dict:
    api_key = os.environ.get(KEY_ENV)
    if not api_key:
        raise SystemExit(f"{KEY_ENV} not set")
    backend = ResponsesBackend(
        model=MODEL, api_key=api_key, base_url=BASE_URL, model_name=API_NAME,
        max_workers=max_workers, system_prompt=system_prompt,
        answer_mode="tokens", timeout=1800.0,
        reasoning_model=True, reasoning_effort="xhigh",
    )
    preds, metas, i = [], [], 0
    aborted = False
    while i < len(examples):
        chunk = examples[i:i + 5]
        preds.extend(backend.generate([e.prompt for e in chunk], max_new_tokens=budget))
        metas.extend(backend.pop_example_meta())
        i += len(chunk)
        spent = sum(m["completion_tokens"] or 0 for m in metas) / 1e6 * COMPLETION_PRICE_PER_M
        if spent > usd_left:
            aborted = True
            print(f"  [guard] arm {name} stopped after {i} calls (${spent:.2f} of ${usd_left:.2f})")
            break

    rows = []
    for ex, pred, meta in zip(examples, preds, metas):
        ctok = meta["completion_tokens"] or 0
        blind = initial_map_answer(ex)
        rows.append({
            "gold": ex.answer, "pred": pred, "match": score(pred, ex.answer),
            "ctok": ctok, "rtok": meta["reasoning_tokens"] or 0,
            "finish": meta["finish_reason"], "engaged": int(ctok >= ENGAGE_CTOK),
            "event_blind": int(TK.content_tokens(TK.committed_answer(pred))[:1] == [blind]),
            "blind_is_gold": int(TK.score_relaxed(blind, ex.answer)),
        })
    n = len(rows) or 1
    eng = [r for r in rows if r["engaged"]]
    dis = [r for r in rows if not r["engaged"]]
    # The published event-blind column (render_benchmark.event_blind_counts) drops items
    # where the event-blind answer happens to BE the gold answer — otherwise a model is
    # credited with the shortcut for a correct answer. One of these 25 items qualifies, so
    # the eligible rate is over 24 and is the one to quote; the raw rate is kept for
    # comparison against the raw per-row flags.
    elig = [r for r in rows if not r["blind_is_gold"]]
    return {
        "arm": name, "system_prompt": system_prompt, "n": len(rows), "aborted": aborted,
        "match": sum(r["match"] for r in rows) / n,
        "engagement_rate": len(eng) / n,
        "acc_given_engaged": (sum(r["match"] for r in eng) / len(eng)) if eng else None,
        "acc_given_disengaged": (sum(r["match"] for r in dis) / len(dis)) if dis else None,
        "event_blind_rate": (sum(r["event_blind"] for r in elig) / len(elig)) if elig else None,
        "event_blind_rate_raw": sum(r["event_blind"] for r in rows) / n,
        "event_blind_eligible": len(elig),
        "mean_ctok": sum(r["ctok"] for r in rows) / n,
        "mean_ctok_engaged": (sum(r["ctok"] for r in eng) / len(eng)) if eng else None,
        "mean_ctok_disengaged": (sum(r["ctok"] for r in dis) / len(dis)) if dis else None,
        "usd_completion": sum(r["ctok"] for r in rows) / 1e6 * COMPLETION_PRICE_PER_M,
        "rows": rows,
        "_calls": ptok,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--budget", type=int, default=49152, help="max_output_tokens (the L64 facet budget)")
    ap.add_argument("--max-workers", type=int, default=5)
    ap.add_argument("--usd-cap", type=float, default=15.0, help="hard cap on completion spend")
    ap.add_argument("--arms", default="canonical,none,neutral")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spec = TK.CANONICAL["s5_chain_v3"]
    examples = TK.generate(spec, "test", n=a.n, length=a.length)
    print(f"{MODEL} s5_chain_v3 @L{a.length}, n={a.n}, effort=xhigh, budget={a.budget}")
    print(f"hard cap ${a.usd_cap:.2f} completion spend\n")

    out, spent = [], 0.0
    for name in a.arms.split(","):
        print(f"arm {name!r}: {ARMS[name]!r}")
        rec = run_arm(name, ARMS[name], examples, a.budget, a.max_workers,
                      a.usd_cap - spent)
        spent += rec["usd_completion"]
        out.append(rec)
        print(f"  match {rec['match']:.2f}  engagement {rec['engagement_rate']:.2f}  "
              f"acc|eng {rec['acc_given_engaged']}  acc|dis {rec['acc_given_disengaged']}  "
              f"event-blind {rec['event_blind_rate']:.2f}  "
              f"ctok {rec['mean_ctok']:.0f}  ${rec['usd_completion']:.2f}\n")
        if spent >= a.usd_cap:
            print("[guard] cap reached; remaining arms skipped")
            break

    path = a.out or os.path.join(
        REPO, "results", "probes",
        f"sol_system_prompt_{datetime.now(timezone.utc):%Y%m%d}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": MODEL, "task": "s5_chain_v3", "length": a.length, "n": a.n,
                   "effort": "xhigh", "budget": a.budget,
                   "engage_ctok_threshold": ENGAGE_CTOK,
                   "ts": datetime.now(timezone.utc).isoformat(),
                   "usd_completion_total": spent, "arms": out}, fh, indent=1)
    print(f"total completion spend ${spent:.2f} -> {path}")


if __name__ == "__main__":
    main()
