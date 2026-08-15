"""THE BOUNDED-PAD FLOOR: what a checkpoint format of width w leaves the composed cell floored by.

WHY THIS EXISTS. The GUIDED protocol's format writes the whole of P then the whole of B after
every event, so the k + m live slots the one-structure bound prices are handed to every policy and
the composed cell has no floor under it (``validity``, "THE GUIDED PROTOCOL", T2). The route back
to a floor is to make the format NARROW: emit w slots per event instead of k + m. This module
derives, before any model is trained, exactly which w restore a floor and what that floor is.

THE DERIVATION, and it is one inequality.
    A pad of w slots is w free live slots (W3: the pad substitutes for REGISTERS, and a policy may
    allocate them as it likes — a pad is a pad). A row whose true cost is W therefore costs
    W - w of the policy's OWN slots, and the class rule admits it iff

        W - w <= max(k, m) + 1.

    The composed cell's own algorithm costs W = k + m + 1, so the class EXCLUDES the task iff

        k + m + 1 - w > max(k, m) + 1   <=>   w < min(k, m)   <=>   w <= min(k, m) - 1,

    which at the k = m = 6 local operating point is w <= 5. At w = 6 the task ties the bound and
    is admitted, so the cell is unfloorable there; at w = k + m = 12, the shipped format, every
    row including the whole block-drop continuum is admitted, which is the retraction at HEAD.

    THE SAME INEQUALITY SETS THE FLOOR'S VALUE, and this is the half that decides whether a
    floored format is worth training. ``partial_carry_j`` (carry P in full and j of the m holder
    cells) costs W = k + j + 1, so a pad of w admits exactly j <= w. The floor at width w is the
    max over the plain protocol's admitted rows and ``partial_carry_j0 .. j{w}``. That family is
    NOT flat in j — at HEAD's k = 12 cell it runs 1.08x chance at j = 0 to 6.47x at j = 11 — so
    the widest floorable format is not necessarily a USEFUL one, and which w are useful is a
    measurement rather than an argument. This script is that measurement.

WHAT IS PRINTED, per composed cell and per width:
    the floor, the row that sets it, the ratio to informed chance, and the CLEARS bar a score
    would have to beat (floor + MARGIN). A width whose bar exceeds 1.0 is unbuyable and is
    reported as such rather than left to be discovered after a training run.

Both item sets are measured, exactly as ``protocol.cell_floor`` does it: the DISJOINT pool (the
max over rows carries an upward selection bias at small n) and the EXACT items the guided read
scores, with the larger operative.
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

from factworld import tasks as TK          # noqa: E402
from factworld import validity as V        # noqa: E402
import protocol_s5bind_v3_three_cell_20260731 as P   # noqa: E402


def cell_pad_floors(spec, L, widths, n_scored=P.N_GUIDED, n_big=P.N_SCORE):
    """Every registered row, the partial-carry profile, and the floor at each pad width.

    The rule is ``validity.s5_bind_v3_pad_operative_floor`` and is not reimplemented here: the
    retraction at HEAD was put in code so the two channels could not diverge again, and the
    restoration is in code for the same reason. Both item sets are measured — the DISJOINT pool
    because a max over rows carries an upward selection bias at small n, and the EXACT items a
    score is read against because a floor must be recomputed from those — with the larger
    operative.
    """
    k, m = spec.k, spec.n_objects_active
    pool = TK.generate(spec, "test", n=n_scored + n_big, length=L)
    scored, big = pool[:n_scored], pool[n_scored:]
    named = V.s5_bind_v3_is_named(big)
    query = V.s5_bind_v3_query_kind(big)
    fs, fb = (V.s5_bind_v3_pad_floors(scored, k, m, named, query),
              V.s5_bind_v3_pad_floors(big, k, m, named, query))
    nss, ngs = V.s5_bind_v3_shape(scored)
    nsb, ngb = V.s5_bind_v3_shape(big)
    out = {"cell": spec.name, "L": L, "k": k, "m": m, "named": named, "query": query,
           "chance": 1.0 / (k - 1),
           "n_scored": len(scored), "n_disjoint": len(big),
           "pad_max_width": V.s5_bind_v3_pad_max_width(k, m),
           "plain_floor": max(x for x in
                              (V.s5_bind_v3_operative_floor(fs, k, m, nss, ngs, named, query),
                               V.s5_bind_v3_operative_floor(fb, k, m, nsb, ngb, named, query))
                              if x is not None),
           "pad_reach": None if named else V.s5_bind_v3_pad_reach(scored),
           "partial_carry_scored": [fs.get(f"partial_carry_j{j}") for j in range(m + 1)],
           "partial_carry_disjoint": [fb.get(f"partial_carry_j{j}") for j in range(m + 1)],
           "widths": {}}
    for w in widths:
        both = [V.s5_bind_v3_pad_operative_floor(fs, k, m, nss, ngs, named, query, pad=w),
                V.s5_bind_v3_pad_operative_floor(fb, k, m, nsb, ngb, named, query, pad=w)]
        if not V.s5_bind_v3_pad_floorable(k, m, w, named):
            out["widths"][w] = {"floor": None, "basis": "unfloorable",
                                "task_admitted": True, "row": None, "bar": None}
            continue
        floor = max(x for x in both if x is not None)
        rows = {}
        for fl, ns, ng, tag in ((fs, nss, ngs, "scored"), (fb, nsb, ngb, "disjoint")):
            for r, v in fl.items():
                if v is not None and V.s5_bind_v3_pad_admits(r, k, m, ns, ng, named, query, w):
                    rows[f"{r}[{tag}]"] = v
        best = max(rows.items(), key=lambda x: x[1])
        out["widths"][w] = {"floor": round(floor, 4), "row": best[0],
                            "basis": "measured", "task_admitted": False,
                            "ratio_to_chance": round(floor * (k - 1), 3),
                            "bar": round(min(1.0, floor + P.MARGIN), 4),
                            "buyable": bool(floor + P.MARGIN <= 1.0),
                            "top5": {r: round(v, 4) for r, v in
                                     sorted(rows.items(), key=lambda x: -x[1])[:5]}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,2,3,4,5,6,12")
    ap.add_argument("--lengths", default="48,64,96")
    ap.add_argument("--n_big", type=int, default=P.N_SCORE)
    ap.add_argument("--out", default="results/s5bind_v3_bounded_pad_floor_20260802.json")
    a = ap.parse_args()
    widths = [int(x) for x in a.widths.split(",")]
    lengths = [int(x) for x in a.lengths.split(",")]

    rec = {"generated": datetime.now(timezone.utc).isoformat(), "widths": widths,
           "one_structure_bound_k6": V.one_structure_bound(6, 6),
           "margin": P.MARGIN, "cells": {}}
    for L in lengths:
        r = cell_pad_floors(TK.CANONICAL[P.LOCAL_CELLS["composed"]], L, widths, n_big=a.n_big)
        rec["cells"][f"composed@{L}"] = r
        print(f"\ncomposed@{L}  chance {r['chance']:.3f}  plain floor {r['plain_floor']:.4f}  "
              f"pad_reach {r['pad_reach']:.4f}", flush=True)
        print("  partial_carry j=0..m  scored   "
              + " ".join(f"{v:.3f}" for v in r["partial_carry_scored"]))
        print("                        disjoint "
              + " ".join(f"{v:.3f}" for v in r["partial_carry_disjoint"]))
        for w in widths:
            d = r["widths"][w]
            if d["floor"] is None:
                print(f"    w={w:2d}: UNFLOORABLE (the task ties or clears the bound)")
            else:
                print(f"    w={w:2d}: floor {d['floor']:.4f} ({d['ratio_to_chance']:.2f}x) "
                      f"[{d['row']}]  bar {d['bar']:.3f}"
                      + ("" if d["buyable"] else "  UNBUYABLE"))
    # the component cells: the pad changes nothing there, but it is measured rather than asserted
    for key, L in (("state", 17), ("state", 80), ("bind", 31), ("bind", 132)):
        r = cell_pad_floors(TK.CANONICAL[P.LOCAL_CELLS[key]], L, widths, n_big=a.n_big)
        rec["cells"][f"{key}@{L}"] = r
        fl = {w: r["widths"][w]["floor"] for w in widths}
        print(f"\n{key}@{L}  plain floor {r['plain_floor']:.4f}  pad floors " +
              " ".join(f"w{w}={v}" for w, v in fl.items()), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
