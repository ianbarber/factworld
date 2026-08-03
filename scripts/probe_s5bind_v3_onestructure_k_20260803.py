"""What raising k at fixed L does to the ONE-STRUCTURE read of the composed cell.

The composed cell exists to make one structure insufficient: P is rewritten by swaps, B by
gives, and a reference is resolved through whichever structure the clause names, so a solver
that maintains only P (treating B as stated) or only B must be wrong wherever the other
structure has moved under it. How often it is nonetheless RIGHT is a property of the sampled
stream, and this script measures it across the k x L grid.

It needs no model and no endpoint: the policies are replayed by ``factworld.validity`` against
the same generator the scored cells draw from, so the number is the task's, not a reading of
anything. Two rows are reported per cell:

  one_structure_P   track P through the swaps, resolve every reference against the STATED B0.
  one_structure_B   the mirror: track B through the gives, resolve against the STATED P0.

and beside them ``initial_only`` — the stream-blind read — which the sampler's ``q_no_surface``
gate gives no items to and which reads 0.0000 everywhere, so a rising one-structure number is
not the stream-blind shortcut coming back under a different name.

``cheap_max`` and ``cheap_max_row`` also record the strongest of EVERY registered non-chance
row. That number is reported separately and is NOT the admissibility measure: at high k the
winner is ``window_90``, which replays the task's own algorithm on a truncated stream, holds
both structures, and is rejected by ``validity.s5_bind_v3_admits`` for exactly that reason. It
says how much of the stream carries no information, which is a different fact.

THIS IS NOT A FLOOR AND MAY NOT BE READ AS ONE. In the scratchpad regime the composed cell has
no floor: its floor argument bounds LIVE SLOTS and a visible trace supplies them. What is
measured here is how often an algorithm that does strictly LESS WORK lands on the answer —
1+k live slots against the task's 1+k+m, and 0.66x its steps at the shipped point — i.e. how
much of the composed cell's separation from its components a (k, L) choice leaves intact. No
model score is compared against it as a bound.

    .venv-api/bin/python scripts/probe_s5bind_v3_onestructure_k_20260803.py --n 2000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sweep_s5bind_v3_local_kL_20260802 as S            # noqa: E402
from factworld import tasks as TK                        # noqa: E402
from factworld import validity as V                      # noqa: E402

OUT = os.path.join(REPO, "results", "probes", "s5bind_v3_onestructure_k_20260803.json")
KS = (6, 12, 24, 32, 48)
LS = (32, 64, 96, 128, 192, 256)
ROWS = ("one_structure_P", "one_structure_B", "initial_only", "last_swap_ref")


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return ((c - h) / d, (c + h) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ks", nargs="+", type=int, default=list(KS))
    ap.add_argument("--lengths", nargs="+", type=int, default=list(LS))
    a = ap.parse_args()

    recs = []
    for L in a.lengths:
        for k in a.ks:
            spec = S.scaled_spec("composed", k)
            try:
                items = TK.generate(spec, "test", n=a.n, length=L)
            except RuntimeError as exc:
                # A (k, L) the sampler cannot fill: the composed spec's q_no_surface gate runs
                # out of admissible items at narrow k and long L. That is a real boundary of the
                # instrument, so it is RECORDED rather than skipped silently or allowed to kill
                # the sweep and lose every cell computed before it.
                recs.append({"k": k, "L": L, "n": 0, "chance": 1.0 / (k - 1),
                             "ungeneratable": str(exc)[:160]})
                print(f"k={k:<3} L={L:<4} UNGENERATABLE: {exc}", flush=True)
                continue
            fl = V.s5_bind_v3_floors(items, k, spec.n_objects_active)
            ns, ng = V.s5_bind_v3_shape(items)
            cheap = {r: v for r, v in fl.items()
                     if r not in V.S5_BIND_V3_CHANCE_ROWS and v is not None}
            best_row = max(cheap, key=lambda r: cheap[r])
            rec = {"k": k, "L": L, "n": a.n,
                   "chance": 1.0 / (k - 1),
                   "hops": V.s5_bind_v3_carrier_hops(k, ns),
                   "n_swap": ns, "n_give": ng,
                   "rows": {r: round(v, 4) for r, v in sorted(cheap.items())},
                   **{r: fl.get(r) for r in ROWS}}
            best = max((fl.get(r) or 0.0) for r in ("one_structure_P", "one_structure_B"))
            rec["one_structure_max"] = best
            # Recorded, but NOT the admissibility measure: at high k the strongest row is
            # window_90, which holds both structures and only skips the stream's first tenth,
            # so it is not a cheaper algorithm in the sense the cell is defined by.
            rec["cheap_max"] = cheap[best_row]
            rec["cheap_max_row"] = best_row
            rec["ci"] = wilson(cheap[best_row], a.n)
            recs.append(rec)
            print(f"k={k:<3} L={L:<4} hops={rec['hops']:5.2f} chance={rec['chance']:.4f} "
                  f"1struct={best:.4f} ({best / rec['chance']:5.2f}x)  "
                  f"best={best_row:<16} {cheap[best_row]:.4f} "
                  f"({cheap[best_row] / rec['chance']:6.2f}x) "
                  f"init_only={rec['initial_only']:.4f}", flush=True)

    # MERGE, never overwrite: the grid is built up over several invocations (the long lengths
    # cost more to generate and are drawn at a smaller pool), and a later run of one corner must
    # not silently delete the rest of the surface. A cell is keyed by (k, L); the larger pool
    # wins, and ties go to the newer record.
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    merged: dict[tuple[int, int], dict] = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for c in json.load(fh).get("cells", []):
                merged[(c["k"], c["L"])] = c
    def rank(c):
        # A record that carries the full row breakdown wins over one that does not, whatever
        # the pool: the breakdown is what the admissibility read and the window read are built
        # from, and a richer record at a smaller n is more useful than a bare one at a larger.
        return (1 if c.get("rows") else 0, c["n"])

    for c in recs:
        prev = merged.get((c["k"], c["L"]))
        if prev is None or rank(c) >= rank(prev):
            merged[(c["k"], c["L"])] = c
    cells = [merged[key] for key in sorted(merged, key=lambda t: (t[1], t[0]))]
    live = [c for c in cells if not c.get("ungeneratable")]
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"ts": datetime.now(timezone.utc).isoformat(),
                   "n": min((c["n"] for c in live), default=0), "n_by_cell": True,
                   "suite_version": TK.SUITE_VERSION, "cells": live,
                   "ungeneratable": [{"k": c["k"], "L": c["L"], "why": c["ungeneratable"]}
                                     for c in cells if c.get("ungeneratable")]}, fh, indent=1)
    print(f"\nwrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
