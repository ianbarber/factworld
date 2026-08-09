"""THE PAD-WRITE FLOOR: what a cheap policy scores on the composed pad, per slot and per token.

WHY THIS EXISTS. Under the bounded pad the composed ANSWER is generated from the model's own pad,
and a per-item perfect pad makes that context byte-identical to the gold-pad one — so the answer
axis registers the pad write and cannot register a composition gap. The only quantity left that
could is the PAD WRITE itself, and a per-slot score means nothing until the cheap policies
available to it have been priced and measured. This module is that measurement.

TWO CLASSES, because the pad has two kinds of token in it and they are floored by different
conjuncts of one rule (``validity.floor_eligible``):
    THE ONE-HOP TOKENS take the composed cell's own conjunct — live slots
    ``W - pad <= max(k, m) + 1``, steps no more than the algorithm pays to PRODUCE THE PAD.
    THE TWO-HOP TOKEN takes the COMPONENT cells' conjunct, ``depth <= S5_BIND_V3_MAX_DEPTH``,
    applied per EMITTED TOKEN: an admitted row may hold 8 of the 12 map cells and still not
    perform the resolve-then-read the token is (``validity.s5_bind_v3_pad_two_hop_floor``).

AND THE TWO-HOP TOKEN IS SPLIT BY SOURCE, with only the CROSS half scored. A same-source swap's
two reads are both of P, which a row holding P alone performs exactly and which is the state
component's own carrier depth; only the cross write needs a value out of the holder map and then
the pointer map. Pooling them lets a model that does the first and floors on the second read as
though it composed.

WHAT IS PRINTED, per cell and per length:
    the chance baseline, DERIVED for a per-slot read (every pad token is an agent name, so it is
    1 / k and not the answer read's 1 / (k - 1));
    the floor for the pooled per-slot read and for each of the four (event kind, block position)
    cells, with the row that sets it, under both classes;
    the registered two-hop floor on the cross partition and the same-source one beside it;
    the EXCLUDED backward-scan row's score, so the exclusion is a judgement about cost rather than
    about the number it would have produced;
    the SATURATION control, which is a policy and not a model, and what that costs;
    and, where the decodes are given, every measured seed against those floors under
    ``clears_headroom`` — teacher-forced as the SCORED read, free-running as the tracking
    diagnostic — with ``--grid`` conjoining the gates PER SEED so no claim spans two models.

Both item sets are measured exactly as ``protocol.cell_floor`` does it — the EXACT items the read
scores and a DISJOINT pool, with the larger operative — because a max over rows carries an upward
selection bias at small n.

Usage:
    .venv/bin/python scripts/probe_s5bind_v3_pad_write_floor_20260803.py \\
        --decompose results/20260802_composed_pad_decompose.json \\
        --forced results/20260802_composed_pad_forced.json \\
        --grid results/20260802_s5bind_v3_bounded_pad_restart_grid.json \\
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


def gold_family(sc, part, top=32):
    """THE SWEPT GOLD-PAD SURFACE, as the report has to print it.

    The class contains every FIXED POSITIONAL READ of the gold pad — any (block class, index,
    token position) — because under teacher forcing the pad is in the context and copying one
    token out of it costs 0 hops, 0 slots and 0 steps. It is closed by that rule and not by a
    list, so what is kept here is the whole surface for the SCORED partition (every address at
    every offset, both positions) and the best ``top`` addresses elsewhere: a reader can then see
    where the max sits instead of taking the max on trust.
    """
    gr = sc.get("gold_reads")
    if not gr:
        return None
    return {"best": gr["best"], "cells": gr["cells"], "n_addresses": gr["n_addresses"],
            "counts": gr["counts"],
            "surface": {p: (v if p == part else dict(list(v.items())[:top]))
                        for p, v in gr["surface"].items()}}


def gold_profile(surface, part):
    """The scored partition's surface as a (class, position) x backward-offset table.

    The family's power is ADJACENCY — the max sits at offset 1 in whichever class — and printing
    the profile is what shows that rather than asserting it.
    """
    prof = {}
    for addr, v in (surface.get(part) or {}).items():
        if "[" not in addr:
            continue
        cls, rest = addr.split("[", 1)
        idx, q = rest.split("]p")
        if not idx.startswith("-"):
            continue
        prof.setdefault(f"{cls}|p{q}", {})[int(idx[1:])] = v
    return prof


def saturation_control(items, k, m, ns, ng, pad, fmt="moved2"):
    """THE SATURATION CONTROL THAT CAN BE BUILT, and it is a POLICY and not a model.

    A saturation control asks whether the measurement can register a clear at all. The cell that
    would answer it with a MODEL does not exist on this read: a component cell's operands are all
    NAMED, so the scored token — ``swap_p0`` on an event whose operand is resolved through the
    other structure — has zero events there, and its floor being 1.000 is the smaller half of the
    reason. The quantity is defined on the composed cell and nowhere else.

    What can be built is the other end of the same scale. The composed cell's OWN ALGORITHM is a
    row in this family — carry both maps in full — and it is excluded by the live-slot conjunct at
    W = 1 + k + m = 13 against the bound 8. Scored on the exact items it writes the two-hop token
    perfectly, so the read separates the admitted class from a policy that does compose, on the
    tokens the model is scored on. It exercises the SCORER, not the decode, and the difference is
    stated rather than papered over.
    """
    row = f"pad_carry_P{k}B{m}_first"
    sc = V.s5_bind_v3_pad_write_scores(items, k, m, pad=pad, rows=(row,), fmt=fmt)
    part = P.PAD_WRITE_TOKEN
    return {"row": row, "W": V.s5_bind_v3_pad_write_cost(row, k, m, ns, ng)[0],
            "bound": V.one_structure_bound(k, m) + pad,
            "admitted": V.s5_bind_v3_pad_write_admits(row, "own_gold", "swap_p0", k, m, ns, ng,
                                                      pad),
            "own_gold": {c: sc["rows"][row][c]["own_gold"] for c in CELLS},
            "own_gold_cross": (sc["rows_src"][row].get(part) or {}).get("own_gold"),
            "n_cross": sc["counts_src"].get(part)}


def cell_pad_write_floor(spec, L, n, n_big, pad, forced=True, fmt="moved2"):
    """Every pad-write row at one cell, on the scored items and on a disjoint pool."""
    k, m = spec.k, spec.n_objects_active
    pool = TK.generate(spec, "test", n=n + n_big, length=L)
    scored, big = pool[:n], pool[n:]
    ns, ng = V.s5_bind_v3_shape(scored)
    out = {"cell": spec.name, "L": L, "k": k, "m": m, "pad": pad, "n_scored": len(scored),
           "n_disjoint": len(big), "n_swap": ns, "n_give": ng,
           "fmt": fmt,
           "chance": V.s5_bind_v3_pad_write_chance(scored, k, fmt),
           "task_pad_cost": V.s5_bind_v3_pad_write_task_cost(k, m, ns, ng),
           "n_rows": len(V.s5_bind_v3_pad_carry_rows(k, m, pad)),
           # THE EXCLUDED ROW, priced on the read the model is scored on. The registered score is
           # the two-hop token teacher-forced on the CROSS partition, so that is the number this
           # row has to be printed at; the free-running source-pooled figure is a different
           # quantity and is kept beside it rather than standing in for it.
           "scan_row": {"cost": V.s5_bind_v3_pad_write_cost("pad_scan_last_write", k, m, ns, ng),
                        "admitted": V.s5_bind_v3_pad_write_admits(
                            "pad_scan_last_write", "own_gold", "swap_p0", k, m, ns, ng, pad),
                        "scores": V.s5_bind_v3_pad_scan_last_write(scored, k, m, forced=True,
                                                                  fmt=fmt),
                        "free_run": V.s5_bind_v3_pad_scan_last_write(scored, k, m, fmt=fmt)},
           "saturation": saturation_control(scored, k, m, ns, ng, pad, fmt)}
    out["chance"].pop("marginal", None)
    for tag, forced_flag in (("free_run", False), ("teacher_forced", True)):
        if forced_flag and not forced:
            continue
        legs = {}
        # THE DEPTH-<=1 CLOSURE IS SELECTED ON THE OTHER LEG. A max over a five-figure family
        # carries an upward selection bias a few-hundred-item read does not average out — measured
        # free-running, the in-sample max is attained by a HEADER address sitting at chance — so
        # each leg scores the member the OTHER leg's items chose. Real members survive it (the
        # same address wins on both) and artifacts do not. The in-sample max is kept beside it.
        picked = {}
        for name, items in (("scored", scored), ("disjoint", big)):
            if items:
                sc0 = V.s5_bind_v3_pad_write_scores(items, k, m, pad=pad, forced=forced_flag,
                                                    fmt=fmt, rows=())
                picked[name] = (sc0.get("fixed_reads") or {}).get("keys") or {}
        for name, items in (("scored", scored), ("disjoint", big)):
            if not items:
                continue
            other = picked.get("disjoint" if name == "scored" else "scored") or {}
            sc = V.s5_bind_v3_pad_write_scores(items, k, m, pad=pad, forced=forced_flag,
                                               fmt=fmt, fixed_members=other)
            nsi, ngi = V.s5_bind_v3_shape(items)
            legs[name] = {
                "all": V.s5_bind_v3_pad_write_floor(sc, k, m, nsi, ngi, pad=pad),
                "one_hop": V.s5_bind_v3_pad_write_floor(sc, k, m, nsi, ngi, pad=pad, max_hops=1),
                # EVERY ROW'S SCORE, not just the max: a floor is a max over a class and the
                # class has to be readable, so each admitted policy's own number is kept.
                "rows": row_table(sc, k, m, nsi, ngi, pad),
                # AND THE WHOLE GOLD-PAD SURFACE, for the same reason one rung up: the family is
                # closed by a rule, so what the rule admits has to be printable.
                "gold_family": gold_family(sc, P.PAD_WRITE_TOKEN),
                # AND THE DEPTH-<=1 CLOSURE'S OWN MAX, with the member that attains it, so the
                # family that now sets this floor is readable and not only its number
                "fixed_family": (sc.get("fixed_reads") or {}).get("best", {})}
        op = {}
        for cls in ("all", "one_hop"):
            best = {"per_slot": None, "per_slot_row": None, "cells": {}, "cell_rows": {},
                    "parts": {}, "part_rows": {}}
            for c in CELLS:
                vals = [(legs[nm][cls]["cells"][c], legs[nm][cls]["cell_rows"][c], nm)
                        for nm in legs if legs[nm][cls]["cells"][c] is not None]
                if not vals:
                    best["cells"][c] = best["cell_rows"][c] = None
                    continue
                v, row, nm = max(vals)
                best["cells"][c] = v
                best["cell_rows"][c] = f"{row}[{nm}]"
            for part in {p for nm in legs for p in legs[nm][cls]["parts"]}:
                vals = [(legs[nm][cls]["parts"][part], legs[nm][cls]["part_rows"][part], nm)
                        for nm in legs if legs[nm][cls]["parts"].get(part) is not None]
                v, row, nm = max(vals)
                best["parts"][part] = v
                best["part_rows"][part] = f"{row}[{nm}]"
            vals = [(legs[nm][cls]["per_slot"], legs[nm][cls]["per_slot_row"], nm)
                    for nm in legs if legs[nm][cls]["per_slot"] is not None]
            v, row, nm = max(vals)
            best["per_slot"], best["per_slot_row"] = v, f"{row}[{nm}]"
            # THE BAR THE REGISTERED RULE SETS, under both margins. The additive one is undefined
            # where floor + MARGIN > 1 — no score whatever clears it — and the pad read is
            # registered on the fractional one for that reason (protocol.MARGIN_FRAC).
            best["bar_additive"] = v + P.MARGIN
            best["buyable_additive"] = bool(v + P.MARGIN <= 1.0)
            best["bar"] = P.bar_for(v)
            op[cls] = best
        # THE REGISTERED TWO-HOP READ: the one-hop sub-class on the CROSS partition of swap_p0,
        # operative over the scored items and the disjoint pool.
        two = {}
        for key in (P.PAD_WRITE_TOKEN, f"{V.S5_BIND_V3_TWO_HOP_CELL}|same"):
            two[key] = {"floor": op["one_hop"]["parts"].get(key),
                        "row": op["one_hop"]["part_rows"].get(key),
                        "unrestricted": op["all"]["parts"].get(key),
                        "unrestricted_row": op["all"]["part_rows"].get(key),
                        "bar": P.bar_for(op["one_hop"]["parts"].get(key))}
        # THE GOLD-PAD FAMILY, operative over the two legs on the scored partition, with the
        # address that attains it and the backward-offset profile it sits on.
        fam = {}
        for nm in legs:
            gf = legs[nm].get("gold_family")
            if gf and (gf["best"].get(P.PAD_WRITE_TOKEN) or {}).get("acc") is not None:
                fam[nm] = gf["best"][P.PAD_WRITE_TOKEN]
        gold = None
        if fam:
            leg = max(fam, key=lambda z: fam[z]["acc"])
            gold = {"acc": fam[leg]["acc"], "address": fam[leg]["address"], "leg": leg,
                    "per_leg": fam,
                    "n_addresses": legs[leg]["gold_family"]["n_addresses"],
                    "profile": gold_profile(legs[leg]["gold_family"]["surface"],
                                            P.PAD_WRITE_TOKEN)}
        out[tag] = {"legs": legs, "operative": op, "two_hop": two, "gold_family": gold}
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
                  + ("" if b["buyable_additive"] else
                     f" (additive bar {b['bar_additive']:.3f} UNBUYABLE)")
                  + f"  [{b['per_slot_row']}]")
            print("      " + "  ".join(
                f"{c} {f4(b['cells'][c])}"
                + ("" if b["cells"][c] is None else f" ({b['cells'][c] / ch['uniform']:.2f}x)")
                for c in CELLS))
        for key, t in r[tag]["two_hop"].items():
            if t["floor"] is None:
                continue
            print(f"      {key:16s} one-hop floor {f4(t['floor'])} "
                  f"({t['floor'] / ch['uniform']:.2f}x) bar {t['bar']:.4f}  [{t['row']}]   "
                  f"unrestricted {f4(t['unrestricted'])} [{t['unrestricted_row']}]")
        gold = r[tag].get("gold_family")
        if gold:
            print(f"      GOLD-PAD FAMILY, all {gold['n_addresses']} fixed addresses swept: best "
                  f"{gold['acc']:.4f} at {gold['address']} [{gold['leg']}]"
                  + "  per leg " + " ".join(f"{nm}={v['acc']:.4f}@{v['address']}"
                                            for nm, v in sorted(gold["per_leg"].items())))
            prof = gold["profile"]
            for lab in sorted(prof, key=lambda z: -max(prof[z].values()))[:4]:
                tail = [v for d, v in prof[lab].items() if d >= 9]
                print(f"        {lab:18s} back-offset 1..8 "
                      + " ".join(f4(prof[lab].get(d)) for d in range(1, 9))
                      + f"   max over d>=9 {f4(max(tail) if tail else None)}")
    tab = r["free_run"]["legs"]["scored"]["rows"]
    top = sorted(tab.items(), key=lambda z: -(z[1]["per_slot"] or 0))[:6]
    print("  every row, best 6 by per_slot (of "
          + f"{len(tab)}; the worst reads {min(v['per_slot'] for v in tab.values()):.4f}): "
          + ", ".join(f"{nm}(W={v['W']}) {v['per_slot']:.4f}" for nm, v in top))
    print("      their swap_p0: "
          + ", ".join(f"{nm} {f4((v['cells'].get('swap_p0') or [None])[0])}" for nm, v in top))
    sr = r["scan_row"]
    print(f"  EXCLUDED pad_scan_last_write (W,S)={sr['cost']} vs task {r['task_pad_cost']} "
          f"-> admitted={sr['admitted']}; SCORED READ (teacher-forced) "
          f"{P.PAD_WRITE_TOKEN}={f4((sr['scores'].get('parts') or {}).get(P.PAD_WRITE_TOKEN))}"
          f"  per cell " + " ".join(f"{c}={f4(sr['scores'][c])}" for c in CELLS)
          + "; free-running pooled "
          + " ".join(f"{c}={f4(sr['free_run'][c])}" for c in CELLS))
    sat = r["saturation"]
    print(f"  SATURATION (policy) {sat['row']} W={sat['W']} vs bound {sat['bound']} -> "
          f"admitted={sat['admitted']}; own_gold "
          + " ".join(f"{c}={f4(sat['own_gold'][c])}" for c in CELLS)
          + f"   {P.PAD_WRITE_TOKEN}={f4(sat['own_gold_cross'])} on n={sat['n_cross']}")


def confront(rows, floors, n, tag, label):
    """Every measured seed against the floors, under the registered rules.

    Three columns and they are three different questions:
      per_slot          the whole pad against the REGISTERED class (no depth conjunct) — a
                        state-CAPACITY read, since a row holding 8 of the 12 map cells writes most
                        of this pad from cells it holds;
      swap_p0|cross     THE SCORED TWO-HOP QUANTITY against the one-hop sub-class, which is the
                        only column on which a clear is a composition reading;
      swap_p0|same      the same token on the events whose two reads are both of P, which a row
                        holding P alone performs exactly. It is printed because it is what the
                        pooled swap_p0 was mixing the scored column with.
    ``clears_headroom`` throughout: the per-slot floors run to 0.93, where an additive margin is
    not a bar (protocol.MARGIN_FRAC).
    """
    print(f"\n== {label} ==")
    print(f"{'seed':>4} {'cell':>12} {'per_slot':>9} {'floor':>8} {'bar':>7} {'clr':>5}   "
          f"{'sp0|cross':>9} {'floor':>8} {'bar':>7} {'clr':>5}   "
          f"{'sp0|same':>9} {'floor':>8} {'clr':>5}")
    out = []
    for r in rows:
        key = f"{r['cell']}@{r['L']}"
        f = floors.get(key)
        if f is None or tag not in f:
            continue
        b = f[tag]["operative"]["all"]
        src = r.get("by_kind_position_source") or {}
        sp = r["per_slot"]
        cl, _z = P.clears_headroom(sp, b["per_slot"], n)
        row = {"seed": r["seed"], "cell": key, "read": tag, "per_slot": sp,
               "per_slot_floor": b["per_slot"], "per_slot_clears": cl,
               "swap_p0": (r.get("by_kind_position") or {}).get("swap_p0")}
        cols = []
        for part in (P.PAD_WRITE_TOKEN, f"{V.S5_BIND_V3_TWO_HOP_CELL}|same"):
            got = src.get(part)
            fl = f[tag]["two_hop"].get(part, {}).get("floor")
            ok, _z2 = (P.clears_headroom(got, fl, n) if got is not None and fl is not None
                       else (False, None))
            row[part] = got
            row[f"{part}_floor"] = fl
            row[f"{part}_clears"] = ok
            row[f"{part}_n"] = src.get(f"n_{part}")
            cols.append((got, fl, ok))
        bar = P.bar_for(b["per_slot"])
        print(f"{r['seed']:>4} {key:>12} {sp:>9.4f} {f4(b['per_slot']):>8} {f4(bar):>7} "
              f"{str(cols and cl):>5}   "
              f"{f4(cols[0][0]):>9} {f4(cols[0][1]):>8} {f4(P.bar_for(cols[0][1])):>7} "
              f"{str(cols[0][2]):>5}   "
              f"{f4(cols[1][0]):>9} {f4(cols[1][1]):>8} {str(cols[1][2]):>5}")
        out.append(row)
    return out


def component_gate(grid_path, n):
    """{seed: do BOTH components form on this seed at every registered length}, from the grid.

    IT IS THE READOUT GATE TOO, on this read and by construction. A component's pad is
    byte-perfect on every seed of this grid (slot_acc 1.000), so its ANSWER read differs from its
    pad read only in whether the model can read back a scratchpad it demonstrably wrote: a seed
    whose components are at floor on the answer with a perfect pad has a dead readout, measured on
    eight cells, and cannot carry a claim about the composed cell's answer or about anything read
    out of its pad. There is no separate gold-pad column to take here and none is needed.
    """
    res = json.load(open(grid_path))
    nn = res["cfg"].get("final_guided_n") or res["cfg"]["guided_n"]
    gg = res["cfg"]["guided_grid"]
    ans: dict = {}
    for r in res["runs"]:
        g = r["stages"][-1]["guided"]
        for c in ("state", "bind"):
            for L in gg.get(c, ()):
                blk = g.get(c, {}).get(str(L)) or {}
                ans.setdefault(c, {}).setdefault(r["seed"], {})[L] = blk.get("match")
    out = {}
    per = {}
    for c in ("state", "bind"):
        floors = {L: (res["floors"].get(f"{c}@{L}") or {}).get("floor") for L in gg.get(c, ())}
        lengths = tuple(L for L in P.registered_lengths(c) if L in gg.get(c, ()))
        _ok, counts, seeds = P.forms(ans[c], floors, lengths, n=nn)
        per[c] = seeds
        out[c] = {"lengths": list(lengths), "counts": counts, "per_seed": seeds}
    seeds = sorted(set(per["state"]) | set(per["bind"]))
    return {s: bool(per["state"].get(s) and per["bind"].get(s)) for s in seeds}, out


def two_hop_gate(confronted, lengths):
    """{seed: does the SCORED two-hop token clear at every registered composed length}."""
    by_seed: dict = {}
    for r in confronted:
        if not r["cell"].startswith("composed@"):
            continue
        by_seed.setdefault(r["seed"], {})[int(r["cell"].split("@")[1])] = r
    return {s: bool(lengths) and all(by_seed[s].get(L, {}).get(f"{P.PAD_WRITE_TOKEN}_clears")
                                     for L in lengths)
            for s in by_seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="16,32,48,64,96")
    ap.add_argument("--cells", default="composed,state,bind")
    ap.add_argument("--component_lengths", default="34,62",
                    help="the state@L and bind@L the pad read was measured at")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n_big", type=int, default=1024)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--format", default="moved2",
                    help="the pad format the run being scored was trained on")
    ap.add_argument("--no_forced", action="store_true")
    ap.add_argument("--decompose", default=None)
    ap.add_argument("--forced", default=None)
    ap.add_argument("--grid", default=None,
                    help="the bounded-pad grid JSON: the component ANSWER read, which is this "
                         "read's components-form AND readout gate")
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
           "margin": P.MARGIN, "margin_frac": P.MARGIN_FRAC, "z_clear": P.Z_CLEAR,
           "scored_read": P.PAD_WRITE_SCORED_READ, "scored_token": P.PAD_WRITE_TOKEN,
           "diagnostic_read": P.PAD_WRITE_DIAGNOSTIC_READ, "cells": {}}
    for cell, L in grid:
        t0 = time.time()
        r = cell_pad_write_floor(specs[cell], L, a.n, a.n_big, a.pad, forced=not a.no_forced,
                                 fmt=a.format)
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
            "THE SCORED READ (protocol.PAD_WRITE_SCORED_READ): the model's own free-running pad, "
            "against a free-running class whose held-out floor is INFORMED CHANCE — the gold pad "
            "is not an address space here, so no admitted policy is above 1/k and a clear cannot "
            "be anything but the two-hop write. A FLOOR here is read with the teacher-forced "
            "number beside it, because it compounds a per-event residual over the stream.")
    if a.forced:
        d = json.load(open(a.forced))
        rec["confront_teacher_forced"] = confront(
            d["rows"], rec["cells"], d["cfg"]["n"], "teacher_forced",
            "THE FLOOR-INTERPRETING READ: teacher-forced per event, against a class handed the "
            "same gold history and reading 2.1x chance off it. It answers the narrower 'given "
            "the true state, is the write performed' and claims nothing about the model's own "
            "writes.")
    if a.grid and (a.forced or a.decompose):
        lengths = P.registered_lengths("composed")
        comp, detail = component_gate(a.grid, a.n)
        scored = rec.get(f"confront_{P.PAD_WRITE_SCORED_READ}") or []
        two = two_hop_gate(scored, lengths)
        gates = {"components_form_and_read_out": comp, "two_hop_clears": two}
        ok, per, n_ok, by_gate = P.seeds_carrying(gates)
        rec["claim"] = {"gates": gates, "per_seed": per, "n_seeds": n_ok, "by_gate": by_gate,
                        "seeds_clear": P.SEEDS_CLEAR, "holds": ok, "components": detail,
                        "lengths": list(lengths)}
        print(f"\n== THE CLAIM, CONJOINED PER SEED (registered lengths {list(lengths)}) ==")
        for g, v in gates.items():
            print(f"  {g:32s} {v}")
        print(f"  seeds carrying ALL of it: {sorted(s for s in per if per[s])} "
              f"({n_ok}, needs {P.SEEDS_CLEAR}) -> {'HOLDS' if ok else 'DOES NOT HOLD'}")
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
