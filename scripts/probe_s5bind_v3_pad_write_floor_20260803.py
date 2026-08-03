"""THE PAD-WRITE FLOOR: what a cheap policy scores on the composed pad, per slot and per token.

WHY THIS EXISTS. Under the bounded pad the composed ANSWER is generated from the model's own pad,
and a per-item perfect pad makes that context byte-identical to the gold-pad one — so the answer
axis registers the pad write and cannot register a composition gap. The only quantity left that
could is the PAD WRITE itself, and a per-slot score means nothing until the cheap policies
available to it have been priced and measured. This module is that measurement.

THE CLASS IS THE REGISTERED ONE (``validity.s5_bind_v3_pad_write_admits``), transferred to the new
quantity rather than invented for it: live slots ``W - pad <= max(k, m) + 1``, steps no more than
the cell's own algorithm pays to PRODUCE THE PAD, and — reported beside it, never as the operative
number — the sub-class that may not itself compose two hops.

WHAT IS PRINTED, per cell and per length:
    the chance baseline, DERIVED for a per-slot read (every pad token is an agent name, so it is
    1 / k and not the answer read's 1 / (k - 1));
    the floor for the pooled per-slot read and for each of the four (event kind, block position)
    cells, with the row that sets it;
    the same under the one-hop sub-class, which is what the two-hop token would be read against if
    the component rule's depth conjunct were carried over;
    the EXCLUDED backward-scan row's score, so the exclusion is a judgement about cost rather than
    about the number it would have produced;
    and, where ``--decompose`` is given, every measured seed against that floor under the
    registered ``clears`` rule.

Both item sets are measured exactly as ``protocol.cell_floor`` does it — the EXACT items the read
scores and a DISJOINT pool, with the larger operative — because a max over rows carries an upward
selection bias at small n.

Usage:
    .venv/bin/python scripts/probe_s5bind_v3_pad_write_floor_20260803.py \\
        --decompose results/20260802_composed_pad_decompose.json \\
        --forced results/20260802_composed_pad_forced.json \\
        --out results/20260803_pad_write_floor.json
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
sys.path.insert(0, os.path.join(REPO, "scripts"))

from factworld import tasks as TK                                          # noqa: E402
from factworld import validity as V                                        # noqa: E402
import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402
import experiment_s5bind_v3_three_cell_local_20260731 as E                 # noqa: E402

CELLS = V.S5_BIND_V3_PAD_CELLS


def f4(v):
    return "—" if v is None else f"{v:.4f}"


def row_table(sc, k, m, ns, ng, pad):
    """Each admitted row's own score: its best emission per cell, and its pooled per-slot value.

    The floor is a max over the class, so the class has to be readable as more than its winner —
    a row that comes close says something a row at chance does not.
    """
    tot = sum(sc["counts"].values())
    out = {}
    for row, cells in sc["rows"].items():
        rec, w_sum = {}, 0.0
        for c in CELLS:
            if not sc["counts"].get(c):
                continue
            cand = [(v, code) for code, v in cells[c].items() if v is not None
                    and V.s5_bind_v3_pad_write_admits(row, code, c, k, m, ns, ng, pad)]
            best = max(cand)
            rec[c] = [best[0], best[1]]
            w_sum += sc["counts"][c] * best[0]
        out[row] = {"per_slot": w_sum / tot if tot else None, "cells": rec,
                    "W": V.s5_bind_v3_pad_write_cost(row, k, m, ns, ng)[0]}
    return out


def cell_pad_write_floor(spec, L, n, n_big, pad, forced=True):
    """Every pad-write row at one cell, on the scored items and on a disjoint pool."""
    k, m = spec.k, spec.n_objects_active
    pool = TK.generate(spec, "test", n=n + n_big, length=L)
    scored, big = pool[:n], pool[n:]
    ns, ng = V.s5_bind_v3_shape(scored)
    out = {"cell": spec.name, "L": L, "k": k, "m": m, "pad": pad, "n_scored": len(scored),
           "n_disjoint": len(big), "n_swap": ns, "n_give": ng,
           "chance": V.s5_bind_v3_pad_write_chance(scored, k),
           "task_pad_cost": V.s5_bind_v3_pad_write_task_cost(k, m, ns, ng),
           "n_rows": len(V.s5_bind_v3_pad_carry_rows(k, m, pad)),
           "scan_row": {"cost": V.s5_bind_v3_pad_write_cost("pad_scan_last_write", k, m, ns, ng),
                        "admitted": V.s5_bind_v3_pad_write_admits(
                            "pad_scan_last_write", "own_gold", "swap_p0", k, m, ns, ng, pad),
                        "scores": V.s5_bind_v3_pad_scan_last_write(scored, k, m)}}
    out["chance"].pop("marginal", None)
    for tag, forced_flag in (("free_run", False), ("teacher_forced", True)):
        if forced_flag and not forced:
            continue
        legs = {}
        for name, items in (("scored", scored), ("disjoint", big)):
            if not items:
                continue
            sc = V.s5_bind_v3_pad_write_scores(items, k, m, pad=pad, forced=forced_flag)
            nsi, ngi = V.s5_bind_v3_shape(items)
            legs[name] = {
                "all": V.s5_bind_v3_pad_write_floor(sc, k, m, nsi, ngi, pad=pad),
                "one_hop": V.s5_bind_v3_pad_write_floor(sc, k, m, nsi, ngi, pad=pad, max_hops=1),
                # EVERY ROW'S SCORE, not just the max: a floor is a max over a class and the
                # class has to be readable, so each admitted policy's own number is kept.
                "rows": row_table(sc, k, m, nsi, ngi, pad)}
        op = {}
        for cls in ("all", "one_hop"):
            best = {"per_slot": None, "per_slot_row": None, "cells": {}, "cell_rows": {}}
            for c in CELLS:
                vals = [(legs[nm][cls]["cells"][c], legs[nm][cls]["cell_rows"][c], nm)
                        for nm in legs if legs[nm][cls]["cells"][c] is not None]
                if not vals:
                    best["cells"][c] = best["cell_rows"][c] = None
                    continue
                v, row, nm = max(vals)
                best["cells"][c] = v
                best["cell_rows"][c] = f"{row}[{nm}]"
            vals = [(legs[nm][cls]["per_slot"], legs[nm][cls]["per_slot_row"], nm)
                    for nm in legs if legs[nm][cls]["per_slot"] is not None]
            v, row, nm = max(vals)
            best["per_slot"], best["per_slot_row"] = v, f"{row}[{nm}]"
            # the CLEARS bar the registered rule sets: a floor whose bar is above 1.0 cannot be
            # cleared by any score, and the cell is unbuyable on this read at this length
            best["bar"] = v + P.MARGIN
            best["buyable"] = bool(v + P.MARGIN <= 1.0)
            op[cls] = best
        out[tag] = {"legs": legs, "operative": op}
    return out


def print_cell(r):
    ch = r["chance"]
    print(f"\n{r['cell']}@{r['L']}  k={r['k']} m={r['m']} pad={r['pad']}  "
          f"n_scored={r['n_scored']} n_disjoint={r['n_disjoint']}  rows={r['n_rows']}", flush=True)
    print(f"  chance (per slot, uniform over k agents) {ch['uniform']:.4f}   "
          f"best fixed agent {f4(ch['best_const'])}")
    for tag in ("free_run", "teacher_forced"):
        if tag not in r:
            continue
        for cls in ("all", "one_hop"):
            b = r[tag]["operative"][cls]
            lab = "registered class" if cls == "all" else "one-hop sub-class"
            print(f"  {tag:14s} {lab:18s} per_slot {f4(b['per_slot'])} "
                  f"({b['per_slot'] / ch['uniform']:.2f}x)  bar {b['bar']:.3f}"
                  + ("" if b["buyable"] else " UNBUYABLE") + f"  [{b['per_slot_row']}]")
            print("      " + "  ".join(
                f"{c} {f4(b['cells'][c])}"
                + ("" if b["cells"][c] is None else f" ({b['cells'][c] / ch['uniform']:.2f}x)")
                for c in CELLS))
    tab = r["free_run"]["legs"]["scored"]["rows"]
    top = sorted(tab.items(), key=lambda z: -(z[1]["per_slot"] or 0))[:6]
    print("  every row, best 6 by per_slot (of "
          + f"{len(tab)}; the worst reads {min(v['per_slot'] for v in tab.values()):.4f}): "
          + ", ".join(f"{nm}(W={v['W']}) {v['per_slot']:.4f}" for nm, v in top))
    print("      their swap_p0: "
          + ", ".join(f"{nm} {f4((v['cells'].get('swap_p0') or [None])[0])}" for nm, v in top))
    sr = r["scan_row"]
    print(f"  EXCLUDED pad_scan_last_write (W,S)={sr['cost']} vs task {r['task_pad_cost']} "
          f"-> admitted={sr['admitted']}; scores "
          + " ".join(f"{c}={f4(sr['scores'][c])}" for c in CELLS))


def confront(rows, floors, n, tag, label):
    """Every measured seed against the floor, under the registered ``clears`` rule."""
    print(f"\n== {label} ==")
    print(f"{'seed':>4} {'cell':>12} {'per_slot':>9} {'floor':>8} {'bar':>7} {'clears':>7}   "
          f"{'swap_p0':>8} {'floor':>8} {'bar':>7} {'clears':>7}")
    out = []
    for r in rows:
        key = f"{r['cell']}@{r['L']}"
        f = floors.get(key)
        if f is None or tag not in f:
            continue
        b = f[tag]["operative"]["all"]
        sp, sp0 = r["per_slot"], (r.get("by_kind_position") or {}).get("swap_p0")
        cl, _z = P.clears(sp, b["per_slot"], n)
        cl0, _z0 = (P.clears(sp0, b["cells"]["swap_p0"], n)
                    if sp0 is not None and b["cells"]["swap_p0"] is not None else (False, None))
        bar = None if b["per_slot"] is None else b["per_slot"] + P.MARGIN
        bar0 = (None if b["cells"]["swap_p0"] is None else b["cells"]["swap_p0"] + P.MARGIN)
        print(f"{r['seed']:>4} {key:>12} {sp:>9.3f} {f4(b['per_slot']):>8} {f4(bar):>7} "
              f"{str(cl):>7}   {(f'{sp0:.3f}' if sp0 is not None else '—'):>8} "
              f"{f4(b['cells']['swap_p0']):>8} {f4(bar0):>7} {str(cl0):>7}")
        out.append({"seed": r["seed"], "cell": key, "read": tag, "per_slot": sp,
                    "per_slot_floor": b["per_slot"], "per_slot_clears": cl,
                    "swap_p0": sp0, "swap_p0_floor": b["cells"]["swap_p0"],
                    "swap_p0_clears": cl0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="16,32,48,64,96")
    ap.add_argument("--cells", default="composed,state,bind")
    ap.add_argument("--component_lengths", default="34,62",
                    help="the state@L and bind@L the pad read was measured at")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n_big", type=int, default=1024)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--no_forced", action="store_true")
    ap.add_argument("--decompose", default=None)
    ap.add_argument("--forced", default=None)
    ap.add_argument("--out", default="results/20260803_pad_write_floor.json")
    a = ap.parse_args()

    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    want = [c for c in a.cells.split(",") if c]
    comp_L = [int(x) for x in a.component_lengths.split(",") if x]
    grid = []
    for c in want:
        if c == "composed":
            grid += [(c, int(x)) for x in a.lengths.split(",") if x]
        else:
            grid += [(c, L) for L in comp_L
                     if (c == "state" and L in (34,)) or (c == "bind" and L in (62,))]
    rec = {"generated": datetime.now(timezone.utc).isoformat(), "cfg": vars(a),
           "registered_composed_lengths": list(P.registered_lengths("composed")),
           "margin": P.MARGIN, "z_clear": P.Z_CLEAR, "cells": {}}
    for cell, L in grid:
        t0 = time.time()
        r = cell_pad_write_floor(specs[cell], L, a.n, a.n_big, a.pad, forced=not a.no_forced)
        r["key"] = f"{cell}@{L}"
        rec["cells"][f"{cell}@{L}"] = r
        print_cell(r)
        print(f"  [{time.time() - t0:.0f}s]", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1, default=float)
    if a.decompose:
        d = json.load(open(a.decompose))
        rec["confront_free_run"] = confront(
            d["rows"], rec["cells"], d["cfg"]["n"], "free_run",
            "THE SCORED READ: the model's own free-running pad against the free-running class")
    if a.forced:
        d = json.load(open(a.forced))
        rec["confront_teacher_forced"] = confront(
            d["rows"], rec["cells"], d["cfg"]["n"], "teacher_forced",
            "THE DIAGNOSTIC READ: teacher-forced, against a class handed the same gold history")
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
