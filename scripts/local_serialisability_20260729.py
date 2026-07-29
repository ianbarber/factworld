"""Is the composition serialisable? A within-harness decomposition on a locally served model.

THE CLAIM UNDER TEST
    Across 15 frontier models, composed@L96 minus min(chain d128, s5 @L256) is +0.029, and 9 of
    15 score at or above their weakest component. That says the thinking-regime composition is
    SERIALISABLE — track the whole map, then dereference it — so it costs max(component) rather
    than more, which would explain why no operating point tried so far discriminates: every knob
    raises one component, and both components are solved at the frontier.

    That reading rests on 15 cells at n=25, cross-model, with two components measured on
    different task families at different breadths. It is suggestive and nothing more. A locally
    served model makes n free, so the same comparison becomes a measurement.

THE DECOMPOSITION (one variable, one harness)
    Rather than compare against s5_concrete and chain_v2 — different renderings, different k —
    both components are cut from the SAME spec as the composed cell:

      A  state tracking   s5_chain(k, depth=1, L)   same stream, same map, one readout
      B  dereference      chain_v2(k=k, depth=d)    the stated map, d hops, no event stream
      C  composed         s5_chain(k, depth=d, L)   both

    A and C share their event stream and rendering exactly, so C - A isolates the dereference
    on tracked state; B gives the dereference on stated state. If the composition is
    serialisable, C ~= min(A, B). If composition costs something, C < min(A, B) materially.

    Every cell is read against its own shallow-adversary floor, recomputed from its exact items
    (factworld.validity), because the floors differ per k and per depth.

    Note the mixing constraint the frontier staircase measured: an event writes ~2.5 slots, so
    below ~2.5 writes per slot (L < k) the final map is still mostly the stated one and the
    floors explode. Every cell here keeps L >= 2k.

WHY IT IS NOT A RANKING
    A 35B-A3B model is far weaker than the frontier roster on these tasks, so its absolute
    numbers say nothing about the roster. What transfers is the SHAPE — whether composition is
    free once both components are held — measured where this model is mid-range, which is the
    instrument's own stated discipline for difficulty knobs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import tasks as TK
from factworld.backends import APIBackend
from factworld.render import Renderer
from factworld.validity import operative_floor, s5_chain_floors

NEUTRAL = ("Answer the question with only the requested value or values. "
           "Use the same spelling as in the question.")


def cells(k: int, depth: int, L: int):
    """(label, spec, length) for the three arms at one operating point."""
    v4 = TK.CANONICAL["s5_chain_v4"]
    return [
        ("A_state", v4.scaled(k=k, chain_depth=1), L),
        ("B_deref", TK.CANONICAL["chain_v2"].scaled(k=k), depth),
        ("C_composed", v4.scaled(k=k, chain_depth=depth), L),
    ]


def run_cell(b: APIBackend, label: str, spec, length: int, n: int, budget: int) -> dict:
    exs = TK.generate(spec, "test", n=n, length=length)
    t = time.time()
    preds = b.generate([e.prompt for e in exs], max_new_tokens=budget)
    meta = b.pop_example_meta()
    hits = [TK.score_relaxed(Renderer.normalize(TK.committed_answer(p)),
                             Renderer.normalize(e.answer)) for p, e in zip(preds, exs)]
    trunc = sum(1 for m in meta if m["finish_reason"] == "length")
    err = sum(1 for m in meta if m["finish_reason"] in (None, "error"))
    clean = [h for h, m, p in zip(hits, meta, preds)
             if m["finish_reason"] == "stop" and p.strip()]
    floors = (s5_chain_floors(exs, spec.k) if spec.family in ("s5_chain", "chain")
              else {})
    return {
        "label": label, "task": spec.name, "k": spec.k, "depth": spec.chain_depth,
        "length": length, "n": n, "budget": budget,
        "match": sum(hits) / n,
        "match_clean": (sum(clean) / len(clean)) if clean else None,
        "n_clean": len(clean), "truncated": trunc, "errors": err,
        "floor": operative_floor(floors) if floors else None, "chance": 1.0 / spec.k,
        "mean_ctok": sum(m["completion_tokens"] or 0 for m in meta) / n,
        "elapsed_s": round(time.time() - t, 1),
        "rows": [{"gold": e.answer, "pred": p, "match": h,
                  "ctok": m["completion_tokens"] or 0, "finish": m["finish_reason"]}
                 for e, p, h, m in zip(exs, preds, hits, meta)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3.6-35b-a3b-local")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--points", default="8:2:16,16:4:32,16:8:32",
                    help="operating points as k:depth:L, comma separated")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--max-workers", type=int, default=12,
                    help="KV cache bounds real concurrency for long generations, not max_num_seqs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    b = APIBackend(model=a.model, api_key="EMPTY", base_url=a.base_url,
                   max_workers=a.max_workers, system_prompt=NEUTRAL,
                   answer_mode="tokens", timeout=1800.0)
    out = []
    for pt in a.points.split(","):
        k, depth, L = (int(x) for x in pt.split(":"))
        print(f"\n=== k={k} depth={depth} L={L} (writes/slot {2.5*L/k:.1f}, chance {1/k:.3f}) ===",
              flush=True)
        arms = {}
        for label, spec, length in cells(k, depth, L):
            r = run_cell(b, label, spec, length, a.n, a.budget)
            arms[label] = r
            fl = "n/a" if r["floor"] is None else f"{r['floor']:.3f}"
            mc = "n/a" if r["match_clean"] is None else f"{r['match_clean']:.2f}"
            print(f"  {label:12} {r['task']:16} match {r['match']:.2f}  clean {mc} "
                  f"({r['n_clean']}/{r['n']})  floor {fl}  ctok {r['mean_ctok']:.0f}  "
                  f"trunc {r['truncated']}  err {r['errors']}  [{r['elapsed_s']:.0f}s]", flush=True)
        A, B, C = arms["A_state"]["match"], arms["B_deref"]["match"], arms["C_composed"]["match"]
        gap = C - min(A, B)
        print(f"  --> composed {C:.2f} - min(state {A:.2f}, deref {B:.2f}) = {gap:+.3f}", flush=True)
        out.append({"k": k, "depth": depth, "length": L, "arms": arms,
                    "composed_minus_min_component": gap})

    path = a.out or os.path.join(REPO, "results", "probes",
                                 f"local_serialisability_{datetime.now(timezone.utc):%Y%m%d}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model, "n": a.n, "budget": a.budget,
                   "system_prompt": NEUTRAL,
                   "ts": datetime.now(timezone.utc).isoformat(), "points": out}, fh, indent=1)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
