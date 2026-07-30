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
    (factworld.validity), because the floors differ per k and per depth — and they differ a lot
    more than 1/k suggests. READ ARM A AGAINST ITS OWN FLOOR, NOT AGAINST CHANCE: the
    initial-ref-resolution policy (resolve every conditional reference against the stated map
    instead of the running one) is strongly depth-sensitive, because at depth 1 only the queried
    slot has to be right while at higher depth an error anywhere on the path propagates. Measured
    at n=2000, writes/slot 5.0: 0.378 at k=8/depth 1 against 0.125 chance, 0.363 at k=16/depth 1
    against 0.062, falling to 0.221 at k=8/depth 2 and 0.071 at k=16/depth 4. So the state arm —
    depth 1 by construction — carries the highest floor of the three, and min(A, B) is inflated
    by exactly that much if the model is near it. A model well clear of 0.378 (this one reads
    0.96) makes the comparison sound; a model near it does not.

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


def cells(k: int, depth: int, L: int, state_depth: int = 2):
    """(label, spec, length) for the three arms at one operating point.

    state_depth is 2, NOT 1, and that is a correction rather than a preference. At depth 1 this
    model answered with a bare index — "3" where gold is "g3" — on about a quarter of its calls,
    which score_relaxed correctly scores 0 because the system prompt asks for the question's own
    spelling. A single-hop query invites the index; from depth 2 the model emits a path and
    formats the commitment properly, and no depth>=2 arm shows the artifact. Containment does NOT
    detect it (delta 0.00 on every arm) because the prediction genuinely lacks the gold token, so
    the fix is the arm's depth and not the metric. Frontier models essentially never do this: 5
    bare indices in 6,670 published chain/s5_chain predictions, all qwen.

    Keeping state_depth well below the composed depth preserves what the arm is for — tracking
    with minimal dereference load — while B_deref prices the dereference leg separately.
    """
    v4 = TK.CANONICAL["s5_chain_v4"]
    return [
        ("A_state", v4.scaled(k=k, chain_depth=state_depth), L),
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
    # Containment as the published diagnostic, per the instrument's own rule: a cell where
    # containment exceeds match by >= 0.08 has its raw predictions read before the number is
    # believed. Without it this harness reported a depth-1 control at 0.61 that was really the
    # model answering "3" where gold was "g3" — right agent, dropped prefix — on a quarter of
    # its calls. A single-hop query invites a bare index in a way a multi-hop one does not, so
    # the arm built as the CONTROL was the one the artifact hit.
    cont = [TK.score_contains(Renderer.normalize(TK.committed_answer(p)),
                              Renderer.normalize(e.answer)) for p, e in zip(preds, exs)]
    trunc = sum(1 for m in meta if m["finish_reason"] == "length")
    err = sum(1 for m in meta if m["finish_reason"] in (None, "error"))
    clean = [h for h, m, p in zip(hits, meta, preds)
             if m["finish_reason"] == "stop" and p.strip()]
    # has_events matters: the chain family has no event stream, so "chase the stated initial
    # map" IS its oracle and scores 1.000. validity.py drops that row under has_events=False;
    # omitting the flag printed a floor of 1.000 for the dereference arm.
    floors = (s5_chain_floors(exs, spec.k, has_events=(spec.family == "s5_chain"))
              if spec.family in ("s5_chain", "chain") else {})
    return {
        "label": label, "task": spec.name, "k": spec.k, "depth": spec.chain_depth,
        "length": length, "n": n, "budget": budget,
        "match": sum(hits) / n,
        "contains": sum(cont) / n,
        "contains_minus_match": (sum(cont) - sum(hits)) / n,
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
    ap.add_argument("--budget", type=int, default=49152,
                    help="16384 truncated 5 of 24 composed calls at k=16/depth 4")
    ap.add_argument("--max-workers", type=int, default=12,
                    help="KV cache bounds real concurrency for long generations, not max_num_seqs")
    ap.add_argument("--state-depth", type=int, default=2,
                    help="depth of the tracking control; 1 is format-fragile (see cells())")
    ap.add_argument("--reps", type=int, default=2,
                    help="repeats per s5_chain arm. vLLM batch non-determinism gave a 0.12 "
                         "test-retest spread on identical items at n=24 even at temperature 0, "
                         "so the gap is read against a MEASURED noise bar, not an assumed one.")
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
        arms, reps = {}, {}
        for label, spec, length in cells(k, depth, L, a.state_depth):
            n_rep = 1 if label == "B_deref" else a.reps      # B is a 1.00 positive control
            runs = [run_cell(b, label, spec, length, a.n, a.budget) for _ in range(n_rep)]
            reps[label] = [x["match_clean"] for x in runs]
            r = runs[0]
            r["replicate_match_clean"] = reps[label]
            r["replicate_spread"] = (max(v for v in reps[label] if v is not None)
                                     - min(v for v in reps[label] if v is not None)
                                     if len([v for v in reps[label] if v is not None]) > 1 else None)
            arms[label] = r
            fl = "n/a" if r["floor"] is None else f"{r['floor']:.3f}"
            mc = "n/a" if r["match_clean"] is None else f"{r['match_clean']:.2f}"
            print(f"  {label:12} {r['task']:16} match {r['match']:.2f}  clean {mc} "
                  f"({r['n_clean']}/{r['n']})  cont {r['contains']:.2f}  floor {fl}  "
                  f"ctok {r['mean_ctok']:.0f}  trunc {r['truncated']}  err {r['errors']}  "
                  f"[{r['elapsed_s']:.0f}s]", flush=True)
            if r["contains_minus_match"] >= 0.08:
                print(f"      !! containment exceeds match by {r['contains_minus_match']:+.2f} "
                      f"— read the raw predictions before believing this arm", flush=True)
        # The gap MUST be computed on completed calls. A truncated call scores wrong, and the
        # composed arm generates more than either component, so a raw-match gap reports the
        # budget as a composition cost: at k=16/depth 4 the raw gap read -0.083 while the
        # completed-call gap was +0.010, the difference being 5 truncations against 2.
        def g(label, key):
            return arms[label][key]
        A, B, C = (g("A_state", "match_clean"), g("B_deref", "match_clean"),
                   g("C_composed", "match_clean"))
        Ar, Br, Cr = (g("A_state", "match"), g("B_deref", "match"), g("C_composed", "match"))
        gap = None if None in (A, B, C) else C - min(A, B)
        gap_raw = Cr - min(Ar, Br)
        trunc = {lbl: arms[lbl]["truncated"] for lbl in ("A_state", "B_deref", "C_composed")}
        spreads = {k2: arms[k2].get("replicate_spread") for k2 in arms
                   if arms[k2].get("replicate_spread") is not None}
        print(f"  --> clean: composed {C:.2f} - min(state {A:.2f}, deref {B:.2f}) = "
              f"{gap:+.3f}   [raw {gap_raw:+.3f}, truncations {trunc}]", flush=True)
        print(f"      replicates: " + "  ".join(
            f"{k2}={['%.2f' % v if v is not None else 'n/a' for v in reps[k2]]}" for k2 in reps)
            + (f"   max spread {max(spreads.values()):.3f}" if spreads else ""), flush=True)
        if spreads and abs(gap) <= max(spreads.values()):
            print(f"      NOTE: |gap| {abs(gap):.3f} is within the measured replicate spread "
                  f"{max(spreads.values()):.3f} — not an effect.", flush=True)
        if any(v > 0.05 * a.n for v in trunc.values()):
            print("      NOTE: >5% truncation in an arm — raise --budget before reading this "
                  "point; the raw gap is a budget artifact, not a composition cost.", flush=True)
        out.append({"k": k, "depth": depth, "length": L, "arms": arms,
                    "composed_minus_min_component": gap,
                    "composed_minus_min_component_raw": gap_raw,
                    "truncations": trunc})

    path = a.out or os.path.join(REPO, "results", "probes",
                                 f"local_serialisability_{datetime.now(timezone.utc):%Y%m%d}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model, "n": a.n, "budget": a.budget,
                   "system_prompt": NEUTRAL,
                   "ts": datetime.now(timezone.utc).isoformat(), "points": out}, fh, indent=1)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
