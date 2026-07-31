"""The block-drop family on s5_bind, scanned over (position, width), and the cost class.

Three things, all recomputed from the exact rendered items a cell scores:

  SCAN     drop a block of ``width`` events at ``position`` and play everything else exactly.
           window_f and prefix_f are the two endpoints (position 0 and position 1); the
           registered rows were six samples of this surface. Parsed back off the rendered
           prompt by a simulator that shares no code with factworld.validity, so the scan is an
           independent attack on the floor rather than a re-run of it.
  FLOORS   the registered rows, split by resource class: the operative floor is the max over
           the rows that carry no map, and the map-carrying rows are printed as diagnostics.
  COST     the cheapest correct algorithm's steps and live slots for the composed cell and its
           two components, on a step-counted register machine.

``--before`` reports the same grid with the chain gate off (chain_max_gap=0), which reproduces
the pre-gate stream byte for byte and is what the before/after columns are read from.

Run:  .venv/bin/python scripts/probe_s5bind_block_drop_20260730.py
        [--n 1500] [--before] [--gap F] [--cells 12@128,6@48]
"""
import argparse
import collections
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld.tasks import CANONICAL, generate                      # noqa: E402
from factworld.validity import (                                     # noqa: E402
    S5_BIND_MAP_CARRYING_ROWS,
    s5_bind_floors,
    s5_bind_operative_floor,
)

# --- an independent parser + simulator ----------------------------------------------------
RE_P0 = re.compile(r"(g\d+) has role (r\d+) at the start\.")
RE_B0 = re.compile(r"(g\d+) holds (o\d+) at the start\.")
RE_SWAP = re.compile(r"^swaps the roles of (g\d+) and the agent who holds (o\d+) "
                     r"(at this point|at the start)\.$")
RE_GIVE = re.compile(r"^gives (o\d+) to the agent whose role (at this point|at the start) "
                     r"is (r\d+)\.$")
RE_QR = re.compile(r"what role does (g\d+) have at the end\?")
RE_QH = re.compile(r"who is the holder of (o\d+) at the end\?")


def parse(prompt):
    P0 = dict(RE_P0.findall(prompt))
    B0 = {o: g for g, o in RE_B0.findall(prompt)}
    body = RE_QR.sub("", RE_QH.sub("", prompt)).strip()
    parts = re.split(r"\bs(\d+) ", body)
    evs = []
    for i in range(1, len(parts) - 1, 2):
        txt = parts[i + 1].strip()
        m = RE_SWAP.match(txt)
        if m:
            evs.append(("swap", m.group(1), m.group(2), m.group(3) == "at this point"))
            continue
        m = RE_GIVE.match(txt)
        if m:
            evs.append(("give", m.group(1), m.group(3), m.group(2) == "at this point"))
            continue
        return None, None, None, None
    q = RE_QR.search(prompt)
    return P0, B0, evs, ("role", q.group(1)) if q else ("hold", RE_QH.search(prompt).group(1))


def simulate(P0, B0, evs, lo=-1, hi=-1):
    """Play every event except the block [lo, hi). Returns (P, B) and the queried role's
    chain — the event indices at which each role moved."""
    P, B = dict(P0), dict(B0)
    inv = {r: g for g, r in P0.items()}
    chain = collections.defaultdict(list)
    for i, (kind, a, b, dyn) in enumerate(evs):
        if lo <= i < hi:
            continue
        if kind == "swap":
            other = (B if dyn else B0).get(b)
            if other is None or other == a:
                continue
            chain[P[a]].append(i)
            chain[P[other]].append(i)
            P[a], P[other] = P[other], P[a]
            inv = {r: g for g, r in P.items()}
        else:
            tgt = (inv if dyn else {r: g for g, r in P0.items()}).get(b)
            if tgt is not None:
                B[a] = tgt
    return P, B, chain


# --- the cost model -----------------------------------------------------------------------
# A step-counted register machine. Content-addressed retrieval of a STATED fact or of an event
# record is one step; resolving an operand against a CARRIED map is one step; each entry
# written to a carried map is one step. Live slots W = symbol registers held simultaneously.
#
#   composed (coupled, state query) — forward pass carrying P, its inverse and B.
#       swap: resolve B[o] (1) + write two P entries (2) + write two Pinv entries (2) = 5
#       give: resolve Pinv[r] (1) + write one B entry (1)                              = 2
#       W = 2k + m
#   state component (decoupled, state query) — sparse backward carrier walk over ONE register.
#       per chain event: retrieve the last swap naming the carrier (1), retrieve the last swap
#       whose object the carrier holds at the start (1), compare (1), read the other operand
#       (1), resolve it through the stated holder map (1)                              = 5
#       plus the closing stated-role read and the query                                = 2
#       W = 2
#   retrieval component (decoupled, bind query) — last-write-wins lookup.
#       locate the last give to the queried object (1), read the role it names (1), resolve
#       that role through the stated map (1)                                           = 3
#       W = 3
COST_SWAP, COST_GIVE, COST_CHAIN_STEP, COST_RETRIEVAL = 5, 2, 5, 3


def cost_composed(evs, k, m):
    n_swap = sum(1 for e in evs if e[0] == "swap")
    return COST_SWAP * n_swap + COST_GIVE * (len(evs) - n_swap), 2 * k + m


def cost_state_component(P0, B0, evs, q):
    """The decoupled arm's cheapest correct algorithm: walk the answer's role backward."""
    static = [(kind, a, b, False) for kind, a, b, _dyn in evs]
    P, _B, chain = simulate(P0, B0, static)
    return COST_CHAIN_STEP * len(chain[P[q]]) + 2, 2


# --- the report ---------------------------------------------------------------------------
POSITIONS = [round(0.05 * i, 2) for i in range(19)]           # 0.00 .. 0.90
WIDTHS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def cell(name, L, k, m, n, gap):
    spec = CANONICAL[name].scaled(chain_max_gap=gap) if gap is not None else CANONICAL[name]
    if k != spec.k:
        spec = spec.scaled(k=k, n_objects=m, n_objects_active=m)
    ex = generate(spec, "test", n=n, length=L)
    hits = collections.Counter()
    missed, missed_hits = collections.Counter(), collections.Counter()
    oracle = 0
    comp_steps = comp_slots = st_steps = st_slots = 0
    maxrun = []
    n_swap = 0
    for e in ex:
        P0, B0, evs, q = parse(e.prompt)
        assert q is not None and evs, f"{name}@{L}: parse failure"
        gold = e.answer.strip().rstrip(".")
        Pf, Bf, chain = simulate(P0, B0, evs)
        got = (Pf if q[0] == "role" else Bf).get(q[1])
        oracle += int(got == gold)
        n_swap += sum(1 for x in evs if x[0] == "swap")
        Lc = len(evs)
        idx = chain[Pf[q[1]]] if q[0] == "role" else []
        rest = [Lc - 1 - idx[-1]] + [idx[j + 1] - idx[j] - 1 for j in range(len(idx) - 1)]
        maxrun.append((idx[0], max(rest)))       # (leading run, longest run after it)
        cs, cw = cost_composed(evs, k, m)
        comp_steps += cs
        comp_slots = cw
        ss, sw = cost_state_component(P0, B0, evs, q[1])
        st_steps += ss
        st_slots = sw
        for wf in WIDTHS:
            w = max(1, int(round(wf * Lc)))
            for p in POSITIONS:
                s = int(round(p * (Lc - w)))
                P, B, _c = simulate(P0, B0, evs, s, s + w)
                ok = int((P if q[0] == "role" else B).get(q[1]) == gold)
                hits[(wf, p)] += ok
                if not any(s <= i < s + w for i in idx):     # the block MISSED the chain
                    missed[wf] += 1
                    missed_hits[wf] += ok
    assert oracle >= 0.99 * len(ex), f"{name}@{L}: simulator reproduces gold on {oracle}/{len(ex)}"
    fl = s5_bind_floors(ex, k=k)
    return {
        "n": len(ex), "L": L, "k": k, "chance": 1.0 / (k - 1),
        "floor": s5_bind_operative_floor(fl),
        "rows": fl,
        "scan": {kk: v / len(ex) for kk, v in hits.items()},
        "lead": max(x[0] for x in maxrun),
        "lead_mean": sum(x[0] for x in maxrun) / len(maxrun),
        "rest": max(x[1] for x in maxrun),
        "missed": {w: (missed[w] / (len(ex) * len(POSITIONS)),
                       missed_hits[w] / missed[w] if missed[w] else 0.0) for w in WIDTHS},
        "n_swap": n_swap / len(ex),
        "composed_steps": comp_steps / len(ex), "composed_slots": comp_slots,
        "state_steps": st_steps / len(ex), "state_slots": st_slots,
        "retr_steps": COST_RETRIEVAL, "retr_slots": 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--before", action="store_true",
                    help="report the un-gated stream (chain_max_gap=0)")
    ap.add_argument("--gap", type=float, default=None,
                    help="override chain_max_gap (default: whatever the spec registers)")
    ap.add_argument("--cells", default="")
    args = ap.parse_args()
    gap = 0.0 if args.before else args.gap

    grid = [("s5_bind_v2", L, 12, 12) for L in (128, 192, 256)]
    grid += [("s5_bind_local_v2", L, 6, 6) for L in (48, 64)]
    grid += [("s5_bind_v2", L, 8, 8) for L in (128, 192)]
    grid += [("s5_bind_v2", L, 16, 16) for L in (192, 256)]
    if args.cells:
        want = set(args.cells.split(","))
        grid = [g for g in grid if f"{g[2]}@{g[1]}" in want]

    print(f"s5_bind block-drop scan  (n={args.n}, "
          f"{'chain_max_gap=0 — the UN-GATED stream' if args.before else 'gated stream'})\n")
    for name, L, k, m in grid:
        r = cell(name, L, k, m, args.n, gap)
        ch, fl = r["chance"], r["floor"]
        best = max(r["scan"].items(), key=lambda kv: kv[1])
        print(f"== {name} k={k} L={L}  chance={ch:.4f}  operative floor={fl:.4f} "
              f"({fl / ch:.2f}x)  swaps={r['n_swap']:.1f}/{L}  "
              f"off-chain runs: leading max={r['lead']} (mean {r['lead_mean']:.1f}), "
              f"after it max={r['rest']}")
        print(f"   cost: composed {r['composed_steps']:.1f} steps / {r['composed_slots']} live"
              f" | state {r['state_steps']:.1f} / {r['state_slots']}"
              f" | retrieval {r['retr_steps']} / {r['retr_slots']}"
              f"  -> multiplier {r['composed_steps'] / r['state_steps']:.2f}x")
        head = "   width\\pos " + "".join(f"{p:>6.2f}" for p in POSITIONS[::2])
        print(head)
        for wf in WIDTHS:
            row = "".join(f"{r['scan'][(wf, p)] / ch:>6.2f}" for p in POSITIONS[::2])
            wbest = max(r["scan"][(wf, p)] for p in POSITIONS)
            print(f"   {wf:>5.2f}L    " + row + f"   | max {wbest / ch:>5.2f}x chance")
        print("   blocks that MISSED the chain (share of all (item, position) pairs, and their "
              "accuracy over chance):")
        print("     " + " ".join(f"{w:.2f}L {r['missed'][w][0]:.2f}/{r['missed'][w][1] / ch:.2f}x"
                                 for w in WIDTHS))
        print(f"   BEST over the whole scan: width {best[0][0]:.2f}L @ pos {best[0][1]:.2f} = "
              f"{best[1]:.4f} = {best[1] / ch:.2f}x chance, {best[1] / fl:.2f}x floor")
        diag = {n: v for n, v in r["rows"].items() if n in S5_BIND_MAP_CARRYING_ROWS}
        print("   map-carrying rows (class-excluded, printed as diagnostics): "
              + " ".join(f"{n}={v:.3f}" for n, v in diag.items()))
        print()


if __name__ == "__main__":
    main()
