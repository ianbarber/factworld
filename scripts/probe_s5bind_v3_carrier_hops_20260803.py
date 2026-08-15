"""THE CARRIER CHAIN OF s5_bind_v3, MEASURED off the rendered prompts instead of predicted.

WHY THIS EXISTS. ``validity.s5_bind_v3_carrier_hops`` returns ``2 n_swap / k`` — the expected
number of swaps that touch ONE FIXED agent when the operands are uniform over k. The composed
spec's query gates (``q_no_surface``, ``q_tail``) choose which agent is queried, so the stream a
report reads is conditioned on that agent and the uniform expectation is not what it carries. A
table that prints the formula therefore prints a number no cell has.

TWO QUANTITIES ARE MEASURED, because the formula and the algorithm are not counting the same thing.

  touch   THE FORMULA'S OWN QUANTITY, measured: swaps that NAME the queried agent, either as the
          written target or as the agent its reference resolves to. This is what ``2 n_swap / k``
          estimates, so it is the like-for-like replacement for that column.
  carry   THE ANSWER'S OWN DEPENDENCY CHAIN: the backward carrier walk. The carrier MOVES at every
          hop, so this counts swaps that touch a moving slot rather than a fixed one, and it is
          the number of events whose contents the answer depends on. It is checked rather than
          asserted — the walk ends on a stated initial-map entry, and that entry must be the gold
          answer on every item, which ``--check`` verifies.

Both are read off the PROMPT by code that shares nothing with the sampler (``composition.read``),
and a reference's identity is recovered by replaying the stream forward, which is the only way it
is available: that is what the coupled rendering is for.

    .venv-api/bin/python scripts/probe_s5bind_v3_carrier_hops_20260803.py --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sweep_s5bind_v3_local_kL_20260802 as S            # noqa: E402
from factworld import tasks as TK                        # noqa: E402
from factworld import validity as V                      # noqa: E402
from factworld.composition import SWAP, read             # noqa: E402

OUT = os.path.join(REPO, "results", "probes", "s5bind_v3_carrier_hops_20260803.json")

# The full (k, L) surface the one-structure grid is measured on, so every cell a report cites
# has a measured chain rather than a formula standing in for one.
KS = (6, 12, 24, 32, 48, 64)
LS = (32, 64, 96, 128, 192, 256, 384, 512, 768)
EXTRA = ()


def item_hops(prompt: str) -> tuple[int, int, str, str] | None:
    """``(touch, carry, walk_origin, gold_slot)`` for one rendered item.

    The forward pass fixes every reference's identity; the backward walk then follows the value
    the query reads out. ``walk_origin`` is the stated-map entry the walk lands on, which is the
    answer iff the walk is right — returned so the caller can check it rather than trust it.
    """
    rec = read(prompt)
    if rec is None or rec["query"][0] != "state":
        return None
    P, B = dict(rec["P0"]), dict(rec["B0"])
    resolved: list[str | None] = []
    for kind, tgt, ref, src in rec["events"]:
        x = ref if src == "N" else (P.get(ref) if src == "P" else B.get(ref))
        resolved.append(x)
        if x is None:
            return None
        if kind == SWAP:
            if tgt not in P or x not in P:
                return None
            P[tgt], P[x] = P[x], P[tgt]
        else:
            B[tgt] = x
    q = rec["query"][1]
    touch = sum(1 for i, (kind, tgt, _r, _s) in enumerate(rec["events"])
                if kind == SWAP and (tgt == q or resolved[i] == q))
    slot, carry = q, 0
    for i in range(len(rec["events"]) - 1, -1, -1):
        kind, tgt, _ref, _src = rec["events"][i]
        if kind != SWAP:
            continue
        x = resolved[i]
        if slot == tgt:
            slot, carry = x, carry + 1
        elif slot == x:
            slot, carry = tgt, carry + 1
    return touch, carry, rec["P0"].get(slot, ""), P.get(q, "")


def cell(k: int, L: int, n: int, check: bool = True) -> dict | None:
    spec = S.scaled_spec("composed", k)
    try:
        items = TK.generate(spec, "test", n=n, length=L)
    except RuntimeError as exc:
        return {"k": k, "L": L, "ungeneratable": str(exc)[:160]}
    rows, bad = [], 0
    for e in items:
        r = item_hops(e.prompt)
        if r is None:
            continue
        rows.append(r)
        if check and r[2] != e.answer.strip().rstrip("."):
            bad += 1
    if not rows:
        return None
    ns, ng = V.s5_bind_v3_shape(items)
    touch = sorted(r[0] for r in rows)
    carry = sorted(r[1] for r in rows)
    nn = len(rows)
    return {"k": k, "L": L, "n": nn, "n_swap": ns, "n_give": ng,
            "formula": V.s5_bind_v3_carrier_hops(k, ns),
            "touch": sum(touch) / nn, "carry": sum(carry) / nn,
            "touch_min": touch[0], "touch_max": touch[-1],
            "carry_min": carry[0], "carry_max": carry[-1],
            "touch_zero_frac": sum(1 for h in touch if h == 0) / nn,
            "carry_zero_frac": sum(1 for h in carry if h == 0) / nn,
            "walk_mismatch": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ks", nargs="+", type=int, default=list(KS))
    ap.add_argument("--lengths", nargs="+", type=int, default=list(LS))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    want = [(k, L) for L in a.lengths for k in a.ks] + [c for c in EXTRA
                                                        if c not in
                                                        {(k, L) for L in a.lengths for k in a.ks}]
    cells = []
    for k, L in want:
        c = cell(k, L, a.n)
        if c is None:
            continue
        cells.append(c)
        if c.get("ungeneratable"):
            print(f"k={k:<3} L={L:<4} UNGENERATABLE", flush=True)
            continue
        print(f"k={k:<3} L={L:<4} n_swap={c['n_swap']:<4} formula {c['formula']:6.2f}  "
              f"touch {c['touch']:6.2f}  carry {c['carry']:6.2f}  "
              f"zero(touch/carry) {c['touch_zero_frac']:.3f}/{c['carry_zero_frac']:.3f}  "
              f"walk mismatches {c['walk_mismatch']}", flush=True)

    live = [c for c in cells if not c.get("ungeneratable")]
    bad = sum(c["walk_mismatch"] for c in live)
    print(f"\nbackward walk lands on the gold answer on {sum(c['n'] for c in live) - bad} of "
          f"{sum(c['n'] for c in live)} items")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"ts": datetime.now(timezone.utc).isoformat(), "n": a.n,
                   "suite_version": TK.SUITE_VERSION, "walk_mismatches": bad, "cells": live,
                   "ungeneratable": [{"k": c["k"], "L": c["L"]}
                                     for c in cells if c.get("ungeneratable")]}, fh, indent=1)
    print(f"wrote {a.out} ({len(live)} cells)")


if __name__ == "__main__":
    main()
