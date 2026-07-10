"""Frontier calibration for the commutative rung (commutative_v1) — place the floors,
confirm discrimination, buy NO predicted-ceiling cells.

Models: the two cheapest roster reasoners — z-ai/glm-5.2 (the canary, $0.93/$3.00 per M) and
deepseek/deepseek-v4-pro ($0.435/$0.87). Cells (6, not 8): per model, instant@L16,
instant@L64 (effort="none", hard answer contract, 96-token cap) and thinking@L64
(effort="high", 8192 tokens, stop_at=None per protocol). thinking@L16 is deliberately NOT
bought — CoT trivially sums ~5 amounts and glm already solves the harder s5-concrete to L32
(discriminate before spending: never buy predicted-ceiling cells). thinking@L64 is kept as
the ONE ceiling reference proving the construct is solvable-with-work.

Predicted informative ranges: instant@L16 ~0.5-0.9 (the anchor); instant@L64 ~0.2-0.6 (the
discriminating cell — above s5-instant, below binding-instant is the ladder-confirming
outcome); thinking@L64 ~0.9+. DECISION RULE: if instant@L64 lands in (2x floor, 0.9) for
either model the rung discriminates (promotion toward kind=benchmark in a later PR); if both
instant cells are >= 0.95, re-run once at L128 before concluding regime-degenerate.

Protocol: relaxed match canonical; empty-pred rate, contract rate, covert-CoT rate, finish
reasons and est. cost published per cell. Crash-safe JSONL append to
results/commutative_frontier/runs.jsonl; resume by (model, length, effort). HARD-CAPPED at
$2 total estimated spend (of the shared $35 ceiling — thread-1's E1b leg outranks this).

Run:
    set -a; source .env; set +a
    .venv-api/bin/python scripts/experiment_commutative_frontier.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import tasks as TK
from factworld.backends import APIBackend
from factworld.render import Renderer

TASK = "commutative_v1"
N = 25
BUDGET_CAP_USD = 2.00
OUT_DIR = os.path.join(REPO, "results", "commutative_frontier")
OUT = os.path.join(OUT_DIR, "runs.jsonl")

# (slug, prompt $/M, completion $/M) — the two cheapest roster reasoners.
MODELS = (
    ("z-ai/glm-5.2", 0.93, 3.00),
    ("deepseek/deepseek-v4-pro", 0.435, 0.87),
)

# (length, effort, max_new_tokens). thinking@L16 deliberately absent (predicted ceiling).
CELLS = ((16, "none", 96), (64, "none", 96), (64, "high", 8192))

# zero-budget answer contract (run_frontier_benchmark idiom), position-flavored.
CONTRACT_LINE = "Reply with only one line: Answer: <position>"
BASE_SYSTEM_PROMPT = (
    "You are taking a short test. Answer each question with only the requested "
    "value or values, no explanation. Use the same spelling as in the question."
)
_ANSWER_RE = re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$")
COVERT_COT_CTOK_THRESHOLD = 350   # factworld.benchmark: clean contract answers are tens of tokens


def extract_contract_answer(text: str) -> str | None:
    """Span after the LAST 'Answer:' line of the visible output (None = contract miss)."""
    hits = _ANSWER_RE.findall(text or "")
    return hits[-1] if hits else None


def load_done() -> set[tuple]:
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((rec["model"], rec["length"], rec["effort"]))
    return done


def main():
    ap = argparse.ArgumentParser(description="Frontier calibration probes for commutative_v1.")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    a = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (set -a; source .env; set +a)")
    os.makedirs(OUT_DIR, exist_ok=True)

    spec = TK.spec_for(TASK)
    done = load_done()
    run_id = f"comm_frontier_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    spent = 0.0

    for model, p_price, c_price in MODELS:
        for length, effort, max_new in CELLS:
            if (model, length, effort) in done:
                print(f"SKIP (resume): {model} L{length} effort={effort}")
                continue
            if spent >= BUDGET_CAP_USD:
                print(f"BUDGET CAP ${BUDGET_CAP_USD} reached (est ${spent:.2f}) — stopping.")
                return
            examples = TK.generate(spec, "test", n=a.n, length=length)
            instant = effort not in ("low", "medium", "high")
            prompts = [f"{e.prompt}\n{CONTRACT_LINE}" if instant else e.prompt
                       for e in examples]
            backend = APIBackend(
                model=model, api_key=api_key, base_url=a.base_url,
                max_workers=a.max_workers, system_prompt=BASE_SYSTEM_PROMPT,
                answer_mode="raw" if instant else "tokens",
                extra_body={"reasoning": {"effort": effort}},
                timeout=1800.0,
            )
            t0 = datetime.now(timezone.utc)
            raw = backend.generate(prompts, max_new_tokens=max_new, stop_at=None)
            ex_meta = backend.pop_example_meta()
            call_meta = backend.pop_call_meta()
            if instant:
                spans = [extract_contract_answer(t) for t in raw]
                preds = [s if s is not None else "" for s in spans]
                contract_rate = sum(1 for s in spans if s is not None) / a.n
            else:
                preds = raw
                contract_rate = None
            rel = [TK.score_relaxed(Renderer.normalize(p), Renderer.normalize(e.answer))
                   for p, e in zip(preds, examples)]
            ctoks = [m.get("completion_tokens") or 0 for m in ex_meta]
            usage = call_meta.get("usage", {})
            cost = (usage.get("prompt_tokens", 0) / 1e6 * p_price
                    + usage.get("completion_tokens", 0) / 1e6 * c_price)
            spent += cost
            rec = {
                "run_id": run_id, "task": TASK, "model": model, "length": length,
                "effort": effort, "max_new_tokens": max_new, "n": a.n,
                "contract": instant, "metrics": {"relaxed": sum(rel) / a.n},
                "diagnostics": {
                    "empty_rate": sum(1 for p in preds if not p.strip()) / a.n,
                    "contract_rate": contract_rate,
                    "covert_cot_rate": (sum(1 for c in ctoks if c > COVERT_COT_CTOK_THRESHOLD) / a.n
                                        if instant else None),
                    "finish_reasons": sorted({m.get("finish_reason") for m in ex_meta if m}),
                    "mean_ctok": sum(ctoks) / max(1, len(ctoks)),
                },
                "usage": {**usage, "cost_usd_est": round(cost, 4)},
                "examples": [{"gold": e.answer, "pred": p, "relaxed": r,
                              "ctok": m.get("completion_tokens"),
                              "rtok": m.get("reasoning_tokens"),
                              "finish": m.get("finish_reason")}
                             for e, p, r, m in zip(examples, preds, rel, ex_meta)],
                "elapsed_s": (datetime.now(timezone.utc) - t0).total_seconds(),
            }
            with open(OUT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            d = rec["diagnostics"]
            print(f"{model} L{length} effort={effort}: relaxed={rec['metrics']['relaxed']:.3f} "
                  f"empty={d['empty_rate']:.2f} contract={d['contract_rate']} "
                  f"mean_ctok={d['mean_ctok']:.0f} ${cost:.3f} "
                  f"[{rec['elapsed_s']:.0f}s]  (cum est ${spent:.2f})", flush=True)
    print(f"done -> {OUT}  (total est ${spent:.2f})")


if __name__ == "__main__":
    main()
