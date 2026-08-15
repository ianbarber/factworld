"""Where does s5_chain actually break? A breadth/depth staircase on one cheap model.

WHY THIS EXISTS
    Every s5_chain operating point so far was chosen and then found to be at ceiling: v3 @L96
    separates 0 of 55 pairs inside the top eleven, and v4 — which doubles k and depth and blocks
    the backward walk — left the two measurable scout models unchanged (deepseek 23/23 and 16/16,
    muse 25/25 and 23/25). Nobody has measured where the edge is.

    The reason to expect the edge to be far away is structural. Composed@L96 minus
    min(chain d128, s5 @L256) is +0.029 on average over 15 models, and 9 of 15 score at or above
    their weakest component: in the thinking regime the composition is SERIALISABLE — track the
    whole map, then dereference it — so it costs max(component), not more. Every knob raises one
    component, and both components are solved. This staircase tests that reading: the composite
    should break where its weaker component breaks, and not before.

THE MIXING CONSTRAINT (measured here, and it governs the breadth axis)
    An event writes ~2.5 slots, so a stream of L events makes 2.5L/k writes per slot. Below ~2.5
    the final map is still mostly the stated initial map and the shallow floors explode: at
    k=128, L=64 (1.2 writes/slot) the operative floor is 0.1200 against 1/128 = 0.0078, i.e. 15x
    chance, supplied by resolving references against the initial map. At >= 2.5 writes/slot every
    rung sits at 1.0-1.7x chance. The rungs below therefore hold writes/slot fixed at 5.0, so
    breadth varies and mixing does not.

RUNGS (depth 16 unless stated; L = 2k throughout)
    k= 32 L= 64   the v4 scored point, known ceiling for this model
    k= 64 L=128   2x the map
    k=128 L=256   4x the map
    k= 64 L=128 depth 32   the depth arm at fixed breadth

DECISION RULE (pre-registered)
    Read each rung against its own floor, printed with the score.
      a rung lands mid-range (~0.3-0.7)
        -> an operating point with resolution exists. Buy a roster battery THERE, not at the
           current point, and expect the spread to track the weaker component.
      every rung at ceiling
        -> the composite is solved at the frontier across the reachable range of these knobs.
           Stop buying composed thinking cells; the discriminating axes are the instant regime,
           token price, and engagement. A harder task needs a non-serialisable construct, not a
           bigger one.
      a rung collapses straight to its floor with no mid-range
        -> the knob is a cliff rather than an axis; report the bracketing rungs and narrow.

    Off-protocol by construction (non-canonical k/depth/L), so this writes to results/probes/ and
    never to the benchmark history.
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
from factworld.backends import APIBackend
from factworld.benchmark import MODELS, endpoint_for
from factworld.render import Renderer
from factworld.validity import operative_floor, s5_chain_floors

# The thinking-regime prompt (run_frontier_benchmark.NEUTRAL_SYSTEM_PROMPT): the answer-format
# contract without the two clauses that suppress engagement.
NEUTRAL = ("Answer the question with only the requested value or values. "
           "Use the same spelling as in the question.")

RUNGS = [  # (k, chain_depth, L, max_new_tokens)
    (32, 16, 64, 49152),
    (64, 16, 128, 98304),
    (128, 16, 256, 131072),
    (64, 32, 128, 98304),
]


def score(pred: str, gold: str) -> int:
    return TK.score_relaxed(Renderer.normalize(TK.committed_answer(pred)),
                            Renderer.normalize(gold))


def run_rung(model: str, k: int, depth: int, L: int, budget: int, n: int,
             max_workers: int, usd_left: float) -> dict:
    reg = MODELS[model]
    base_url, key_env = endpoint_for(model)
    api_key = os.environ.get(key_env)
    if not api_key:
        raise SystemExit(f"{key_env} not set")

    spec = TK.CANONICAL["s5_chain_v4"].scaled(k=k, chain_depth=depth)
    exs = TK.generate(spec, "test", n=n, length=L)
    floors = s5_chain_floors(exs, k)

    extra: dict = {"reasoning": {"effort": "xhigh"}}
    if reg["open_weights"] and reg.get("quantization_filter", True) and not reg.get("base_url"):
        extra["provider"] = {"require_parameters": False,
                             "quantizations": ["fp8", "bf16", "fp16"]}
    backend = APIBackend(model=model, api_key=api_key, base_url=base_url,
                         model_name=reg.get("model_name"), max_workers=max_workers,
                         system_prompt=NEUTRAL, extra_body=extra, answer_mode="tokens",
                         timeout=1800.0, reasoning_model=reg.get("reasoning_model", False))

    preds, metas, i = [], [], 0
    aborted = False
    while i < len(exs):
        chunk = exs[i:i + 5]
        preds.extend(backend.generate([e.prompt for e in chunk], max_new_tokens=budget))
        metas.extend(backend.pop_example_meta())
        i += len(chunk)
        spent = sum(m["completion_tokens"] or 0 for m in metas) / 1e6 * reg["completion_price_per_M"]
        if spent > usd_left:
            aborted = True
            print(f"  [guard] stopped after {i} calls (${spent:.2f} of ${usd_left:.2f})")
            break

    rows = []
    for e, p, m in zip(exs, preds, metas):
        rows.append({"gold": e.answer, "pred": p, "match": score(p, e.answer),
                     "ctok": m["completion_tokens"] or 0, "finish": m["finish_reason"]})
    clean = [r for r in rows if r["finish"] == "stop" and r["pred"].strip()]
    usd = sum(r["ctok"] for r in rows) / 1e6 * reg["completion_price_per_M"]
    return {
        "k": k, "depth": depth, "length": L, "budget": budget, "n": len(rows),
        "writes_per_slot": round(2.5 * L / k, 2),
        "match": (sum(r["match"] for r in rows) / len(rows)) if rows else None,
        "match_clean": (sum(r["match"] for r in clean) / len(clean)) if clean else None,
        "n_clean": len(clean),
        "truncated": sum(1 for r in rows if r["finish"] == "length"),
        # A provider-side failure surfaces as finish="error" on the response; only an
        # exhausted retry leaves finish None. Counting only the latter reported err=0 on a
        # rung that lost 4 of 20 calls — the same defect the renderer carried for
        # calls-failed cells, in the scout that found it.
        "errors": sum(1 for r in rows if r["finish"] in (None, "error")),
        "mean_ctok": (sum(r["ctok"] for r in rows) / len(rows)) if rows else 0,
        "floor": operative_floor(floors), "chance": 1.0 / k, "floors": floors,
        "usd": usd, "aborted": aborted, "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro",
                    help="cheapest capable roster model by completion price")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-workers", type=int, default=5)
    ap.add_argument("--usd-cap", type=float, default=25.0)
    ap.add_argument("--rungs", default=None,
                    help="override RUNGS: comma-separated k:depth:L:budget (e.g. 128:16:256:262144)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"{a.model} — s5_chain_v4 staircase, n={a.n}, effort=xhigh, neutral prompt")
    print(f"hard cap ${a.usd_cap:.2f}\n")
    rungs = ([tuple(int(x) for x in r.split(":")) for r in a.rungs.split(",")]
             if a.rungs else RUNGS)
    out, spent = [], 0.0
    for k, depth, L, budget in rungs:
        print(f"k={k} depth={depth} L={L} (writes/slot {2.5*L/k:.1f}, chance {1/k:.4f})")
        rec = run_rung(a.model, k, depth, L, budget, a.n, a.max_workers, a.usd_cap - spent)
        spent += rec["usd"]
        out.append(rec)
        mc = rec["match_clean"]
        print(f"  match {rec['match']:.2f}  clean {mc if mc is None else round(mc,2)} "
              f"({rec['n_clean']}/{rec['n']})  floor {rec['floor']:.4f}  "
              f"ctok {rec['mean_ctok']:.0f}  trunc {rec['truncated']}  err {rec['errors']}  "
              f"${rec['usd']:.2f}\n")
        if spent >= a.usd_cap:
            print("[guard] cap reached; remaining rungs skipped")
            break

    path = a.out or os.path.join(REPO, "results", "probes",
                                 f"s5_chain_staircase_{datetime.now(timezone.utc):%Y%m%d}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model, "task": "s5_chain_v4", "effort": "xhigh",
                   "system_prompt": NEUTRAL, "n": a.n,
                   "ts": datetime.now(timezone.utc).isoformat(),
                   "usd_total": spent, "rungs": out}, fh, indent=1)
    print(f"total ${spent:.2f} -> {path}")


if __name__ == "__main__":
    main()
