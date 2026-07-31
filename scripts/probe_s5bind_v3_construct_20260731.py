"""s5_bind_v3 calibration: the floor class, the block-drop scan, the cost convention, the grid.

Everything is recomputed from the exact rendered items a cell scores, through
``factworld.composition``'s own parser (which shares no code with the sampler).

  FLOORS   every registered row, split by the PARETO CLASS RULE (strictly cheaper than the
           cell's cheapest correct algorithm in live slots W, no more expensive in steps S).
           The operative floor is the max over the admitted rows, as a ratio to informed chance.
  SCAN     the block-drop family over every position and every width from w to L/2 — the
           continuum the class rule excludes. Reported as a maximum with what it reads, so the
           exclusion can be judged.
  DEMAND   the memo-free demand-driven resolver under a step budget: the policy a W-only rule
           would wrongly admit. Reported at every budget.
  COST     composed / component steps and live slots under the stated convention, and under the
           opposite (free content-addressed access to the event stream), so the size of the
           convention choice is visible.
  GRID     the (k, L) sweep the scored lengths are cut from.

Run:  .venv-train/bin/python scripts/probe_s5bind_v3_construct_20260731.py [--n 1500]
"""
import argparse
import os
import random
import sys

sys.setrecursionlimit(100000)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import composition as C                              # noqa: E402
from factworld import validity as V                                 # noqa: E402
from factworld.tasks import CANONICAL, generate                     # noqa: E402

POSITIONS = [round(0.05 * i, 2) for i in range(20)]                  # 0.00 .. 0.95
WIDTHS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def cell_spec(k, L, local=False):
    base = CANONICAL["s5_bind_local_v3" if local else "s5_bind_v3"]
    if k != base.k:
        base = base.scaled(k=k, n_objects=max(k, base.n_objects), n_objects_active=k)
    return base


def floors_block(spec, ex, label):
    k, m = spec.k, spec.n_objects_active
    ns, ng = V.s5_bind_v3_shape(ex)
    fl = V.s5_bind_v3_floors(ex, k, m)
    cls = V.s5_bind_v3_classify(k, m, ns, ng)
    op = V.s5_bind_v3_operative_floor(fl, k, m, ns, ng)
    ch = 1.0 / (k - 1)
    wt, st = V.s5_bind_v3_task_cost(k, m, ns, ng)
    print(f"\n== {label}  k={k} m={m} L={ns + ng}  swaps={ns} gives={ng}  "
          f"chance={ch:.4f}  operative floor={op:.4f} ({op / ch:.2f}x)")
    print(f"   task cheapest correct algorithm: W={wt} live slots, S={st} steps")
    print(f"   {'row':<22}{'acc':>8}{'x chance':>10}   {'W':>5}{'S':>8}   class")
    for row, v in fl.items():
        w, s = V.s5_bind_v3_row_cost(row, k, m, ns, ng)
        print(f"   {row:<22}{v:>8.4f}{v / ch:>10.2f}   {w:>5}{s:>8}   "
              f"{'FLOOR' if cls[row] else 'diagnostic'}")
    return op, ch


def scan_block(spec, ex, ch, op):
    best = (0.0, None)
    rows = []
    for wf in WIDTHS:
        row = []
        for p in POSITIONS:
            a = V.s5_bind_v3_block_drop(ex, wf, p)
            row.append(a)
            if a > best[0]:
                best = (a, (wf, p))
        rows.append((wf, row))
    print("   block-drop scan (accuracy / chance); every member carries BOTH maps, so W = k + m "
          "ties the task and the whole continuum is class-excluded:")
    print("     width\\pos " + "".join(f"{p:>6.2f}" for p in POSITIONS[::2]))
    for wf, row in rows:
        print(f"     {wf:>5.2f}L   " + "".join(f"{v / ch:>6.2f}" for v in row[::2])
              + f"   | max {max(row) / ch:>5.2f}x")
    print(f"   BEST member over the whole scan: width {best[1][0]:.2f}L @ pos {best[1][1]:.2f} "
          f"= {best[0]:.4f} = {best[0] / ch:.2f}x chance, {best[0] / op:.2f}x the operative floor")
    return best


def demand_resolver(rec, budget):
    """The memo-free demand-driven resolver: resolve only what the query needs, recursively,
    re-walking the event list for each dependency; fall back to the stated map when the step
    budget is spent. Carries NO map (a bounded stack of frames), which is exactly why a
    live-slots-only rule admits it and the Pareto rule does not."""
    evs = rec["events"]
    P0, B0 = rec["P0"], rec["B0"]
    state = {"steps": 0, "max_depth": 0}

    def value(struct, cell, t, depth):
        """The value of ``struct[cell]`` just before event ``t``."""
        state["max_depth"] = max(state["max_depth"], depth)
        for i in range(t - 1, -1, -1):
            state["steps"] += 1
            if state["steps"] > budget:
                return (P0 if struct == "P" else B0).get(cell)
            kind, tgt, ref, src = evs[i]
            if kind == C.SWAP and struct == "P":
                x = ref if src == "N" else value(src, ref, i, depth + 1)
                if x is None:
                    continue
                if cell == tgt:
                    return value("P", x, i, depth + 1)
                if cell == x:
                    return value("P", tgt, i, depth + 1)
            elif kind == C.GIVE and struct == "B" and tgt == cell:
                return ref if src == "N" else value(src, ref, i, depth + 1)
        return (P0 if struct == "P" else B0).get(cell)

    kind, target = rec["query"]
    ans = value("P" if kind == "state" else "B", target, len(evs), 1)
    return ans, state["steps"], state["max_depth"]


def demand_block(ex, ch, budgets, task_steps=1.0):
    print("   memo-free demand-driven resolver (carries no map, only a stack of frames) under a "
          "step budget, falling back to the stated map when spent. This is the policy a "
          "live-slots-ONLY rule admits and the Pareto rule does not:")
    for b in budgets:
        hits = n = mx = 0
        used = 0
        for e in ex:
            rec = C.read(e.prompt)
            a, st, d = demand_resolver(rec, b)
            n += 1
            used += min(st, b)
            mx = max(mx, d)
            hits += int(f"{a}." == e.answer)
        print(f"     budget {b:>7} = {b / task_steps:6.1f}x the task's own steps: "
              f"{hits / n:.4f} = {hits / n / ch:.2f}x chance   "
              f"(mean steps used {used / n:.0f}, max live frames {mx})")


def cost_block(spec, ex, comp_ex_state, comp_ex_bind):
    k, m = spec.k, spec.n_objects_active
    r = C.cost_report(ex, k, m)
    print(f"   COST under the stated convention (a backward walk IS charged for the events it "
          f"scans and rejects):")
    print(f"     composed forward pass          S={r['composed_S']:8.1f}  W={r['composed_W']}")
    print(f"     state leg, other structure free S={r['state_S']:8.1f}  W={r['state_W']}"
          f"   -> step multiplier {r['composed_S'] / r['state_S']:.2f}x, "
          f"slot multiplier {r['composed_W'] / r['state_W']:.1f}x")
    if comp_ex_state:
        rs = C.cost_report(comp_ex_state, k, m)
        print(f"     registered state component      S={rs['state_S']:8.1f}  W={rs['state_W']}"
              f"   -> step multiplier {r['composed_S'] / rs['state_S']:.2f}x")
    if comp_ex_bind:
        rb = C.cost_report(comp_ex_bind, k, m)
        print(f"     registered retrieval component  S={rb['bind_S']:8.1f}  W={rb['bind_W']}"
              f"   -> step multiplier {r['composed_S'] / rb['bind_S']:.2f}x")
    print(f"   the OPPOSITE convention (free content-addressed access to the event stream): "
          f"composed S={r['composed_S_free']:.1f}, state leg S={r['state_S_free']:.1f} "
          f"-> multiplier {r['composed_S_free'] / r['state_S_free']:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--stage", default="all",
                    choices=("all", "cells", "grid", "demand", "scan"))
    a = ap.parse_args()

    cells = [("frontier", CANONICAL["s5_bind_v3"], 128),
             ("frontier", CANONICAL["s5_bind_v3"], 192),
             ("local", CANONICAL["s5_bind_local_v3"], 48),
             ("local", CANONICAL["s5_bind_local_v3"], 64)]

    if a.stage in ("all", "cells", "scan", "demand"):
        for tag, spec, L in cells:
            ex = generate(spec, "test", n=a.n, length=L)
            op, ch = floors_block(spec, ex, f"{spec.name} @ L={L}")
            if a.stage in ("all", "scan"):
                scan_block(spec, ex, ch, op)
            if a.stage in ("all", "demand"):
                sub = ex[:min(len(ex), 300)]
                demand_block(sub, ch, (0, 100, 1000, 5000, 20000, 100000),
                         task_steps=C.cost_report(sub, spec.k,
                                                  spec.n_objects_active)['composed_S'])
            st = generate(CANONICAL[f"s5_bind{'_local' if tag == 'local' else ''}_v3_state"],
                          "test", n=200, length=L)
            bd = generate(CANONICAL[f"s5_bind{'_local' if tag == 'local' else ''}_v3_bind"],
                          "test", n=200, length=L)
            cost_block(spec, ex, st, bd)

    if a.stage in ("all", "grid"):
        print("\n\n== the (k, L) grid: operative floor as a ratio to informed chance, and the "
              "step multiplier ==")
        print(f"   {'k':>3} {'L':>5} {'chance':>8} {'floor':>8} {'x ch':>6} "
              f"{'S_comp':>8} {'S_state':>8} {'mult':>6}  {'top row':>18}")
        for k in (6, 8, 12, 16):
            for L in (32, 48, 64, 96, 128, 192, 256):
                spec = cell_spec(k, L, local=(k == 6))
                try:
                    ex = generate(spec, "test", n=400, length=L)
                except RuntimeError as exc:
                    print(f"   {k:>3} {L:>5}   sampler exhausted: {exc}")
                    continue
                m = spec.n_objects_active
                ns, ng = V.s5_bind_v3_shape(ex)
                fl = V.s5_bind_v3_floors(ex, k, m)
                cls = V.s5_bind_v3_classify(k, m, ns, ng)
                op = V.s5_bind_v3_operative_floor(fl, k, m, ns, ng)
                top = max((v, r) for r, v in fl.items() if cls[r])[1]
                ch = 1.0 / (k - 1)
                r = C.cost_report(ex, k, m)
                print(f"   {k:>3} {L:>5} {ch:>8.4f} {op:>8.4f} {op / ch:>6.2f} "
                      f"{r['composed_S']:>8.0f} {r['state_S']:>8.0f} "
                      f"{r['composed_S'] / r['state_S']:>6.2f}  {top:>18}")


if __name__ == "__main__":
    random.seed(0)
    main()
