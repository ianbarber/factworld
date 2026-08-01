"""The source-structure STRUCTURE-SWITCH diagnostic, and the cost convention it is priced against.

Two things live here, both for the ``s5_bind_v3`` family (TaskSpec.source_ablation):

  THE STRUCTURE-SWITCH DIAGNOSTIC  a within-item, within-cell contrast between the two op classes
                          the construct ablates on (``contrast``). It is NOT a composition
                          measure on this rendering and cannot be made into one — the
                          identification argument is below and it is an impossibility, not a
                          calibration gap. It is reported alongside match as what it identifies:
                          whether a solver is worse when the reference SWITCHES structure.
  THE COST CONVENTION     a stated, testable rule for what one step is, and a counter that
                          implements it (``cost_report``). The convention is what decides the
                          step multiplier and the Pareto floor class, so it is written down
                          rather than left as a bookkeeping choice.

Everything reads the RENDERED prompt through this module's own parser: nothing here consults
``Example.meta`` or the sampler, so every number is recomputed from exactly what a model saw.
Pure stdlib, like the rest of the package.

-------------------------------------------------------------------------------------------
THE STATISTIC — WHAT IT IDENTIFIES, AND WHAT IT CANNOT
-------------------------------------------------------------------------------------------
IT IS NOT A COMPOSITION MEASURE ON THIS RENDERING, and no reweighting, stratification or sample
size makes it one. Within an event kind the class label IS the printed clause. The only four
forms that occur are (swap, reads B, CROSS), (swap, reads P, SAME), (give, reads P, CROSS),
(give, reads B, SAME), and ``src`` is set by the clause regex in ``read`` — so "which structure
this reference reads" and "which clause is printed" are ONE VARIABLE, not two. A solver that
cannot hold B fails on exactly {swap CROSS, give SAME}: sign-flipped across the kinds, which is
precisely the ANTI-symmetric direction the kind-balancing below is built to annihilate. The
repair for the clause confound removes the deficit of interest with it.

MEASURED, at matched accuracy, k=6/L=64, n=800, 40 replicates
(scripts/probe_s5bind_v3_statistic_20260731.py): against a ``one_structure_P`` carrier the
contrast is -0.0190 with 0/40 rejections; against ``one_structure_B``, -0.0344 with 0/40; against
a reads-B-only slip, -0.0116 with 0/40. Zero power against every single-structure policy, and
the contrast points the WRONG WAY. A single-structure solver is the exact thing a composition
measure has to detect, so this is an identification impossibility on this rendering, not a
threshold that a larger n moves.

WHAT IT IS. A STRUCTURE-SWITCH diagnostic: it is sensitive to solvers that degrade when the
reference clause switches structure — a stated-map fallback (power 0.867) and an outright
garbled cross reference (0.899) — and it must be reported under that name. ``contrast`` returns
``"identifies": "structure_switch"`` so a caller cannot read it as composition by accident, and
``tests/test_s5_bind_v3.py`` asserts the zero power above so the composition claim cannot be
quietly re-made. The composition evidence has to come from the three-cell comparison — state
component, retrieval component, composed cell, each against its own floor at the stated step
multiplier — and not from within one cell.

The design below is kept, and it is what the numbers above were measured on.

-------------------------------------------------------------------------------------------
THE DESIGN THE DIAGNOSTIC RUNS ON
-------------------------------------------------------------------------------------------
The composed cell's events each name their second operand LIVE through one of the two
structures. An op is CROSS when it reads the structure it does not write (a swap reading the
holder map; a give reading the pointer map) and SAME when it reads the one it does write. The
sampler draws the two candidate reference cells MATCHED on read history and decides the class
with an independent coin (TaskSpec.match_reads), so a failure that depends on how stale or how
overwritten the read cell is falls on the two classes alike, while a solver that cannot hold the
other structure fails only on CROSS.

THE SURFACE CONFOUND. Within an event kind the class is carried by the reference clause: on a
swap CROSS is "belongs to" and SAME is "points to", on a give it is the other way round. So a
solver that is simply WORSE AT ONE CLAUSE — a surface failure with no composition deficit in it
— slips on swap-CROSS and give-SAME. The clause-to-class map flips between the kinds, so the
effect lives entirely in the ANTI-symmetric combination of the two kinds' class differences.

THE DESIGN. Over the items of one cell,

    P(item correct) = exp(-(theta_w W + sum_s theta_s T_s + theta_c C))

where ``W`` counts the writes and the readout, ``T_s`` is the item's slice mass in stratum s
over BOTH classes, and ``C`` is a fixed combination of the per-stratum class differences (cross
mass minus same mass). Every op is weighted by its ANSWER SENSITIVITY — the probability that
garbling it changes the answer, measured by one-at-a-time perturbation of the clean trajectory.
The statistic is ``theta_c``, tested one-sided by dropping ``C``.

TWO PROPERTIES DO THE WORK, and both are structural rather than asymptotic:

  THE MASS COLUMNS ARE RAW. Any hazard that is a function of the stratum alone is then exactly
  in the model's span with ``theta_c = 0``. A reweighting of the class columns instead — one
  divisor per kind, say — only holds where the within-kind class MASSES are equal, and they are
  not: a CROSS give's object cannot be referenced again until its pin dies, so its mean answer
  sensitivity is about half a SAME give's, and a reweighting reports that as a coefficient.
  THE CONTRAST IS PRECISION-WEIGHTED. With ``w = Sigma^-1 1`` the contrast column is exactly
  uncorrelated with every anti-symmetric combination of the strata, which is where a clause
  failure lives, while a real cross-only deficit is symmetric and survives. That is a statement
  about the COLUMNS, and it does not carry all the way through the fit: measured, a pure clause
  slip still rejects at 0.118 (below).

``T_kind``, the default, strata on the event kind; ``T_strat`` adds retrieval-distance
quartiles, which extends the same argument to any hazard that is a function of the distance —
including a hard forgetting horizon whose cutoff falls inside a bin. ``T_cross``, the two raw
class columns tested against each other with no stratification, stays measured as the
diagnostic that shows what the design buys.

This replaces the temporal contrast the family carried before, which could not identify
composition at all: the temporal ablation's composition class was, by construction, the
overwritten-cell class (see TaskSpec.source_ablation).

WHAT IT IDENTIFIES, AND AT WHAT SAMPLE SIZE. Measured on the generated cells with an independent
parser and replay, R = 1000 resamples, one-sided alpha = 0.05, and a composition-free executor
family dialled to the SAME accuracy cost as the deficits — uniform per-op slip; slip linear in
the write count, the retrieval distance, the stream depth, the derivation depth or the
distinct-value count; stale-value intrusion; surface-clause slip; kind slip; FIFO and LRU at
several capacities; and hard forgetting horizons
(scripts/probe_s5bind_v3_statistic_20260731.py). Size and power are reported there as numbers at
every cell, executor, cost and n.

WHAT IS AND IS NOT NULL, at n = 2000, k=6/L=64, a 0.20 accuracy cost, alpha = 0.05 (Monte Carlo
error +/- 0.007). ``T_kind`` reads 0.055 on a uniform slip, 0.047 on a write-count slip, 0.064 on
a retrieval-distance slip, 0.075 on a stream-depth slip, 0.052 on a distinct-value slip — and
0.060 on a stale-value intrusion — and 0.088 on a DERIVATION-DEPTH slip and 0.118 on a
SURFACE-CLAUSE slip. Power against a stated-map fallback at the same cost is 0.867, and 0.899
against an outright garbled cross reference. ``T_strat`` takes the distance slip to 0.041 and
leaves the derivation-depth one at 0.110; the raw ``T_cross`` reads 0.29 on the clause slip.

A BOUNDED WORKING SET IS THE LARGEST LEAK. LRU at 11 of the 12 live cells reads 0.102 / 0.153 /
0.275 at n = 250 / 500 / 2000 with a contrast of +0.04, and FIFO at the same capacity 0.163 /
0.079. Their accuracies are 0.633 and 0.734, not the 0.70 the continuous executors are dialled
to — a capacity is integer-valued and cannot be dialled — so they are reported at the cost they
land on rather than matched to one.

BOTH SLIP LEAKS ARE THE LOCAL CELL'S. At k=12/L=192 and n = 2000 the clause slip reads 0.053 on
``T_kind`` (against 0.118 at k=6/L=64) and the derivation-depth slip 0.046 (against 0.088), and
every other executor measured there sits at 0.046-0.065. The raw ``T_cross`` reads 0.132 on the
clause slip at k=12 and 0.29 at k=6. Six cells and twelve are not the same instrument: the k=6
pools are six cells wide, which is what leaves the residual the matched draw cannot close.

THE LIVE LEAKS ARE NAMED RATHER THAN ABSORBED, and they bound what the diagnostic reading can
say even about a structure SWITCH. The
DERIVATION DEPTH of the value a reference reads is not matched between the classes and cannot be
matched by choosing cells — a P cell's value is deep because swaps chain, a B cell's because
gives chain (-22.8% on swaps and +29.4% on gives at k=12), so a solver that degrades with
derivation depth is a composition-free executor the design does not control. The SURFACE CLAUSE
is anti-symmetric across the kinds by construction, and the precision weighting removes its
linear projection but not its effect through the fit. A CAPACITY BOUND is matched in the read
cell's recency and write count, which is what the sampler controls, but not in how many other
cells sit between two reads of the same one.

THE THRESHOLD IS NOT CALIBRATED BELOW n = 2000, and that is a property of the LRT here rather
than of the design. Under a plain uniform slip at k=6/L=64 the measured size is 0.05 at n = 2000
and 0.07-0.13 at n = 250, and it moves that far for the RAW two-class contrast as well
(0.05 -> 0.07-0.10), so it is not the stratification or the contrast column that does it. Raising
the trajectories behind each item's accuracy from 100 to 1600 does not move it either. A cell
scored at a few hundred items therefore needs a resampled threshold rather than the chi-square
one; the size at n = 2000 is the number this statistic is registered on.

-------------------------------------------------------------------------------------------
THE COST CONVENTION  (one step = ..., stated so the multiplier is testable)
-------------------------------------------------------------------------------------------
A policy runs on a register machine over the rendered prompt. ``W`` is the number of symbol
registers it holds simultaneously; ``S`` is its steps. ONE STEP IS EXACTLY ONE OF:

  H  read one line of the STATED FACT BLOCK by its key.       The header is content-addressable:
                                                              it is a table the prompt states in
                                                              full, keyed by the content ids.
  E  read the NEXT event statement, forward or backward.      The event stream is NOT
                                                              content-addressable.
  R  resolve one operand against a carried map.
  M  write one entry of a carried map.
  C  compare two symbols.

THE RULE THAT WAS AMBIGUOUS, MADE EXPLICIT: **a backward walk IS charged for the events it
scans and rejects.** "The last event that writes cell c" is not one step; it is one E step per
event scanned back from the current position, plus one C per event. Under the opposite
convention — free content-addressed access to the event stream — the same walks cost only their
hits; ``cost_report`` reports that variant too, as ``S_free``, so the size of the choice is
visible rather than implicit.

Under the stated rule:

  composed cell, forward pass carrying P and B (W = k + m)
      per swap  E read the event, R resolve the reference, R read P[a], R read P[x],
                M write P[a], M write P[x]                                        = 6
      per give  E read the event, R resolve the reference, M write B[o]           = 3
      plus H (k + m) to load the stated maps and 1 to emit the answer.
  the state leg in isolation (the other structure free) — sparse backward carrier walk (W = 2)
      one E + one C per event scanned, and one R per chain hit, scanning back from the end to
      the carrier's first move; then one H for the stated pointer, and 1 to emit.
  the retrieval leg in isolation — the same walk over the give stream, scanning back to the
      queried object's resolving write (W = 2).

P is read FORWARD in this family ("the agent g7 points to"), so no inverse map is carried; that
is why W is k + m here and not 2k + m.
"""
from __future__ import annotations

import math
import random
import re

# --- the cost convention, as constants a test can read ------------------------------------
STEP_HEADER = 1        # H: one keyed read of the stated fact block
STEP_EVENT = 1         # E: one event statement read, forward or backward (scans are charged)
STEP_RESOLVE = 1       # R: one lookup in a carried map
STEP_WRITE = 1         # M: one entry written to a carried map
STEP_COMPARE = 1       # C: one symbol comparison
CHARGE_SCANNED_EVENTS = True   # THE rule: a backward walk pays for what it scans and rejects


# --- an independent surface parser --------------------------------------------------------
_RE_P0 = re.compile(r"\b(g\d+) points to (g\d+) at the start\.")
_RE_B0 = re.compile(r"\b(o\d+) belongs to (g\d+) at the start\.")
_RE_SWAP_B = re.compile(r"\bs(\d+) swaps the pointers of (g\d+) and "
                        r"the agent (o\d+) belongs to at this point\.")
_RE_SWAP_P = re.compile(r"\bs(\d+) swaps the pointers of (g\d+) and "
                        r"the agent (g\d+) points to at this point\.")
_RE_SWAP_N = re.compile(r"\bs(\d+) swaps the pointers of (g\d+) and (g\d+)\.")
_RE_GIVE_P = re.compile(r"\bs(\d+) gives (o\d+) to the agent (g\d+) points to at this point\.")
_RE_GIVE_B = re.compile(r"\bs(\d+) gives (o\d+) to the agent (o\d+) belongs to at this point\.")
_RE_GIVE_N = re.compile(r"\bs(\d+) gives (o\d+) to (g\d+)\.")
_RE_Q_STATE = re.compile(r"which agent does (g\d+) point to at the end\?")
_RE_Q_BIND = re.compile(r"which agent does (o\d+) belong to at the end\?")
_RE_Q_ALL = re.compile(r"which agent does each of ((?:g\d+, )+g\d+) point to at the end\?")

# An event is ``(kind, target, ref, src)`` where kind is 'swap'|'give', ``target`` is the named
# first operand (an agent for a swap, the written object for a give), ``ref`` is the referenced
# slot, and ``src`` is which structure resolves it: 'P', 'B' or 'N' (named, no reference).
SWAP, GIVE = "swap", "give"


def read(prompt: str) -> dict | None:
    """The stated maps, the event stream in rendered order, and the query — off one prompt.

    Returns None when the prompt is not a source-structure item. Shares no code with the
    sampler: every field is recovered from the sentences a model reads.
    """
    P0 = dict(_RE_P0.findall(prompt))
    B0 = dict(_RE_B0.findall(prompt))
    found = []
    for rex, kind, src in ((_RE_SWAP_B, SWAP, "B"), (_RE_SWAP_P, SWAP, "P"),
                           (_RE_SWAP_N, SWAP, "N"), (_RE_GIVE_P, GIVE, "P"),
                           (_RE_GIVE_B, GIVE, "B"), (_RE_GIVE_N, GIVE, "N")):
        for m in rex.finditer(prompt):
            found.append((int(m.group(1)), (kind, m.group(2), m.group(3), src)))
    if not found:
        return None
    found.sort()
    events = [e for _i, e in found]
    m_all = _RE_Q_ALL.search(prompt)
    if m_all is not None:
        query = ("state_all", tuple(m_all.group(1).split(", ")))
    else:
        m_s, m_b = _RE_Q_STATE.search(prompt), _RE_Q_BIND.search(prompt)
        if m_s is not None:
            query = ("state", m_s.group(1))
        elif m_b is not None:
            query = ("bind", m_b.group(1))
        else:
            return None
    if not P0 and not B0:
        return None
    return {"P0": P0, "B0": B0, "events": events, "query": query}


def _resolve(ev, P, B, P0, B0, mode="live"):
    """The agent one event's second operand names, under a policy's reading.

    mode 'live'   the exact semantics: resolve against the running map.
         'stated' resolve against the STATED maps (a policy that reads the header instead).
         'P_live' the running pointer map, the STATED holder map — a solver carrying P alone.
         'B_live' the mirror — a solver carrying B alone.
    """
    kind, _tgt, ref, src = ev
    if src == "N":
        return ref
    if src == "P":
        live = mode in ("live", "P_live")
        return (P if live else P0).get(ref)
    live = mode in ("live", "B_live")
    return (B if live else B0).get(ref)


def replay(rec, mode="live", drop=None):
    """Play the stream and return ``(P, B)``. ``drop=(lo, hi)`` skips a block of events."""
    P, B = dict(rec["P0"]), dict(rec["B0"])
    for i, ev in enumerate(rec["events"]):
        if drop is not None and drop[0] <= i < drop[1]:
            continue
        x = _resolve(ev, P, B, rec["P0"], rec["B0"], mode)
        if x is None:
            return None
        kind, tgt, _ref, _src = ev
        if kind == SWAP:
            if tgt not in P or x not in P:
                return None
            P[tgt], P[x] = P[x], P[tgt]
        else:
            B[tgt] = x
    return P, B


def answer_of(rec, maps) -> str | None:
    """The rendered answer a policy's final maps imply for this item's query."""
    if maps is None:
        return None
    P, B = maps
    kind, target = rec["query"]
    if kind == "state":
        return None if target not in P else f"{P[target]}."
    if kind == "bind":
        return None if target not in B else f"{B[target]}."
    if any(a not in P for a in target):
        return None
    return " ".join(P[a] for a in target) + "."


def is_cross(ev) -> bool:
    """Does this op read the structure it does NOT write? A swap writes P, a give writes B."""
    kind, _t, _r, src = ev
    if src == "N":
        return False
    return (src == "B") if kind == SWAP else (src == "P")


# --- the cost counter, against the convention above ---------------------------------------
def cost_composed(rec, k: int, m: int) -> tuple[int, int]:
    """``(S, W)`` for the composed cell's cheapest correct algorithm: one forward pass
    carrying P and B."""
    n_swap = sum(1 for e in rec["events"] if e[0] == SWAP)
    n_give = len(rec["events"]) - n_swap
    s = (k + m) * STEP_HEADER
    s += n_swap * (STEP_EVENT + 3 * STEP_RESOLVE + 2 * STEP_WRITE)
    s += n_give * (STEP_EVENT + STEP_RESOLVE + STEP_WRITE)
    return s + 1, k + m


def cost_isolated_state(rec, k: int, m: int) -> tuple[int, int]:
    """``(S, W)`` for the STATE leg of this same stream with the other structure free — the
    sparse backward carrier walk that is available exactly when every event's operands are
    fixed on the surface.

    THE CONVENTION BITES HERE. The walk scans back from the end and must READ each event to see
    whether it moves the carrier, so under the stated rule it pays one E and one C for EVERY
    event it passes — gives included — and one R for each hit. That is the whole difference
    between a step multiplier of ~2 and one of ~9, which is why the rule is written down."""
    resolved = _resolved_operands(rec)
    carrier = rec["query"][1]
    scanned = hits = 0
    for i in range(len(rec["events"]) - 1, -1, -1):
        ev = rec["events"][i]
        scanned += 1
        if ev[0] != SWAP:
            continue
        a, x = ev[1], resolved[i]
        if carrier == a:
            carrier, hits = x, hits + 1
        elif carrier == x:
            carrier, hits = a, hits + 1
    walk = scanned * (STEP_EVENT + STEP_COMPARE) + hits * STEP_RESOLVE + STEP_HEADER + 1
    # a forward pass carrying P is the alternative; the leg costs whichever is cheaper
    n_swap = sum(1 for e in rec["events"] if e[0] == SWAP)
    n_give = len(rec["events"]) - n_swap
    fwd = k * STEP_HEADER + n_swap * (STEP_EVENT + 3 * STEP_RESOLVE + 2 * STEP_WRITE) \
        + n_give * STEP_EVENT + 1
    return (walk, 2) if walk <= fwd else (fwd, k)


def cost_isolated_bind(rec, k: int, m: int) -> tuple[int, int]:
    """``(S, W)`` for the RETRIEVAL leg in isolation: scan back to the queried object's
    resolving write, then read the value it names off the surface."""
    evs = rec["events"]
    target = rec["query"][1]
    scanned = 0
    for i in range(len(evs) - 1, -1, -1):
        scanned += 1                      # every event passed is read and compared
        if evs[i][0] == GIVE and evs[i][1] == target:
            break
    return scanned * (STEP_EVENT + STEP_COMPARE) + STEP_RESOLVE + STEP_HEADER + 1, 2


def cost_free_variant(rec, k: int, m: int) -> tuple[int, int, int]:
    """The SAME three algorithms under the opposite convention — free content-addressed access
    to the event stream, so a backward walk pays only for its hits. Reported next to the stated
    rule's numbers so the size of the convention choice is visible."""
    n_swap = sum(1 for e in rec["events"] if e[0] == SWAP)
    n_give = len(rec["events"]) - n_swap
    composed = n_swap * (3 * STEP_RESOLVE + 2 * STEP_WRITE) + n_give * (STEP_RESOLVE + STEP_WRITE)
    resolved = _resolved_operands(rec)
    carrier, hits = rec["query"][1], 0
    for i in range(len(rec["events"]) - 1, -1, -1):
        ev = rec["events"][i]
        if ev[0] != SWAP:
            continue
        a, x = ev[1], resolved[i]
        if carrier == a:
            carrier, hits = x, hits + 1
        elif carrier == x:
            carrier, hits = a, hits + 1
    return composed, 2 * hits + 2, 2


def _resolved_operands(rec) -> list:
    """Each event's resolved second operand under the exact semantics."""
    P, B = dict(rec["P0"]), dict(rec["B0"])
    out = []
    for ev in rec["events"]:
        x = _resolve(ev, P, B, rec["P0"], rec["B0"])
        out.append(x)
        kind, tgt, _ref, _src = ev
        if kind == SWAP:
            P[tgt], P[x] = P[x], P[tgt]
        else:
            B[tgt] = x
    return out


def cost_report(examples, k: int, m: int) -> dict:
    """Mean steps and live slots for the composed pass and for each leg in isolation, over the
    exact rendered items of one cell, under the stated convention (and under the free-retrieval
    variant)."""
    n = 0
    tot = {"composed_S": 0, "state_S": 0, "bind_S": 0,
           "composed_S_free": 0, "state_S_free": 0, "events": 0, "swaps": 0}
    W = {}
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        n += 1
        cs, cw = cost_composed(rec, k, m)
        tot["composed_S"] += cs
        W["composed_W"] = cw
        if rec["query"][0] == "state":
            ss, sw = cost_isolated_state(rec, k, m)
            tot["state_S"] += ss
            W["state_W"] = sw
        if rec["query"][0] == "bind":
            bs, bw = cost_isolated_bind(rec, k, m)
            tot["bind_S"] += bs
            W["bind_W"] = bw
        fc, fs, fw = cost_free_variant(rec, k, m)
        tot["composed_S_free"] += fc
        tot["state_S_free"] += fs
        W["free_W"] = fw
        tot["events"] += len(rec["events"])
        tot["swaps"] += sum(1 for x in rec["events"] if x[0] == SWAP)
    if not n:
        return {}
    out = {key: v / n for key, v in tot.items()}
    out.update(W)
    out["n"] = n
    return out


# --- the answer-sensitivity slice ---------------------------------------------------------
def op_slice(rec, draws: int = 2, rng: random.Random | None = None) -> list[dict]:
    """The answer's dependency slice, one record per op.

    Each record carries the op's class (``cross`` / ``same`` / ``write``), the WRITE COUNT and
    the RETRIEVAL DISTANCE of the cell it read, and ``sens`` — the measured probability that
    garbling this op changes the answer, by one-at-a-time perturbation of the clean trajectory
    followed by an exact replay of the rest. Ops with ``sens == 0`` are off the slice.
    """
    rng = rng or random.Random(0)
    evs = rec["events"]
    agents = sorted(rec["P0"], key=lambda s: int(s[1:])) or sorted(
        {v for v in rec["B0"].values()}, key=lambda s: int(s[1:]))
    kq, target = rec["query"]
    P, B = dict(rec["P0"]), dict(rec["B0"])
    snaps, xs, wcol, dcol = [], [], [], []
    wcnt, last = {}, {}
    for j, ev in enumerate(evs):
        snaps.append((dict(P), dict(B)))
        kind, tgt, ref, src = ev
        cell = (src, ref)
        wcol.append(wcnt.get(cell, 0))
        dcol.append(j - last.get(cell, -1))
        x = _resolve(ev, P, B, rec["P0"], rec["B0"])
        xs.append(x)
        if kind == SWAP:
            P[tgt], P[x] = P[x], P[tgt]
            for g in (tgt, x):
                wcnt[("P", g)] = wcnt.get(("P", g), 0) + 1
                last[("P", g)] = j
        else:
            B[tgt] = x
            wcnt[("B", tgt)] = wcnt.get(("B", tgt), 0) + 1
            last[("B", tgt)] = j
    gold = (P if kq == "state" else B).get(target)

    out = []
    for j, ev in enumerate(evs):
        kind = ev[0]
        Ps, Bs = snaps[j]
        hit = 0
        for _ in range(draws):
            alt = rng.choice([a for a in agents if a != xs[j]])
            Pc, Bc = dict(Ps), dict(Bs)
            _apply(ev, Pc, Bc, alt)
            _replay_from(rec, Pc, Bc, j + 1)
            hit += int((Pc if kq == "state" else Bc).get(target) != gold)
        cls = "write" if ev[3] == "N" else ("cross" if is_cross(ev) else "same")
        out.append({"i": j, "kind": kind, "cls": cls, "w": wcol[j], "d": dcol[j],
                    "sens": hit / draws})
        # the WRITE the event performs is a second op: its sensitivity is whether LOSING it
        # changes the answer, measured the same way.
        Pc, Bc = dict(Ps), dict(Bs)
        _replay_from(rec, Pc, Bc, j + 1)
        out.append({"i": j, "kind": kind, "cls": "write", "w": 0, "d": 0,
                    "sens": float((Pc if kq == "state" else Bc).get(target) != gold)})
    return out


def _apply(ev, P, B, x):
    kind, tgt, _ref, _src = ev
    if kind == SWAP:
        P[tgt], P[x] = P[x], P[tgt]
    else:
        B[tgt] = x


def _replay_from(rec, P, B, j):
    for ev in rec["events"][j:]:
        x = _resolve(ev, P, B, rec["P0"], rec["B0"])
        if x is None:
            return
        _apply(ev, P, B, x)


def _cov_from_ops(ops) -> dict:
    """Per-item covariates of the fit.

    ``nw`` writes and the readout; ``nz`` SAME resolutions; ``nx`` CROSS resolutions; each
    weighted by answer sensitivity. ``W``/``W2``/``D`` are the item's READ-HISTORY LOAD, summed
    over EVERY resolution the item performs and entered as free nuisances by the repairs.

    THE NUISANCE COLUMNS RUN OVER BOTH CLASSES. A read-history effect is a property of the cell
    read, not of the class reading it, so the load it puts on an item is the sum over all of that
    item's reads; a column summed over the CROSS reads alone absorbs the cross half of the effect
    and leaves the same half loading on the SAME class column, which is a repair that moves the
    contrast in the direction of the defect it claims to remove rather than a no-op.
    """
    cov = {"nw": 0.0, "nz": 0.0, "nx": 0.0, "W": 0.0, "W2": 0.0, "D": 0.0}
    for op in ops:
        s = op["sens"]
        if op["cls"] == "write":
            cov["nw"] += s
            continue
        cov["nz" if op["cls"] == "same" else "nx"] += s
        cov["W"] += s * op["w"]
        cov["W2"] += s * op["w"] * op["w"]
        cov["D"] += s * op["d"]
    cov["nw"] += 1.0                                   # the readout
    return cov


def item_covariates(rec, draws: int = 2, rng: random.Random | None = None) -> dict:
    """Per-item covariates of the fit — see ``_cov_from_ops``."""
    return _cov_from_ops(op_slice(rec, draws=draws, rng=rng))


# --- the stratified design: one mass column per stratum, one contrast column ---------------
def strata(all_ops, n_bins: int = 1):
    """``(stratum_of(op), n_strata)``. A stratum is the EVENT KIND, crossed with a
    retrieval-distance bin when ``n_bins > 1``; the bin edges are the pooled distance quantiles
    of this cell's own ops, so the stratification is a property of the cell and not a constant.
    """
    kinds = (SWAP, GIVE)
    edges: list[int] = []
    if n_bins > 1:
        ds = sorted(op["d"] for ops in all_ops for op in ops if op["cls"] != "write")
        if ds:
            edges = sorted({ds[len(ds) * j // n_bins] for j in range(1, n_bins)})
    width = len(edges) + 1

    def of(op):
        b = sum(1 for e in edges if op["d"] >= e)
        return kinds.index(op["kind"]) * width + b

    return of, len(kinds) * width


def _stratum_columns(all_ops, n_bins: int):
    """``(mass columns, contrast column)`` — the design the composition coefficient sits in.

    For every stratum s the MASS column T_s carries the item's whole slice mass in s, over both
    classes, unweighted; the CONTRAST column is a fixed combination of the per-stratum class
    differences (cross mass minus same mass).

    THE MASS COLUMNS ARE RAW, AND THAT IS THE WHOLE POINT. Any hazard that is a function of the
    stratum alone — a per-kind slip, a slip linear in the retrieval distance, a hard forgetting
    horizon whose cutoff falls inside a bin — is then EXACTLY in the model's span with a zero
    coefficient on the contrast column, so the fit returns zero by construction rather than by
    cancellation. Divide the class columns by anything and the truth leaves the span the moment
    the divisor varies with the class mix: the within-kind class MASSES are not equal here (a
    CROSS give's object cannot be referenced again until its pin dies, so its mean answer
    sensitivity is about half a SAME give's), and a reweighting reports that imbalance as a
    coefficient.

    THE CONTRAST COLUMN IS PRECISION-WEIGHTED, and that is what kills the surface confound. A
    clause failure is CROSS on a swap and SAME on a give, so it loads on the ANTI-symmetric
    combination of the per-stratum differences. Weighting them by ``w = Sigma^-1 1``, with Sigma
    the covariance of the difference columns, makes the contrast exactly uncorrelated with every
    anti-symmetric combination (``w' Sigma v = 1' v = 0`` whenever ``v`` sums to zero), while a
    real cross-only deficit loads on the symmetric one and survives. Equal weights do that only
    where the strata carry equal variance, and they do not — a swap resolution's mean sensitivity
    is ~0.72 against a give's ~0.13.
    """
    of, ns = strata(all_ops, n_bins)
    T = [[0.0] * ns for _ in all_ops]
    Dm = [[0.0] * ns for _ in all_ops]
    for i, ops in enumerate(all_ops):
        for op in ops:
            if op["cls"] == "write":
                continue
            s = of(op)
            T[i][s] += op["sens"]
            Dm[i][s] += op["sens"] if op["cls"] == "cross" else -op["sens"]
    n = max(1, len(all_ops))
    keep = [s for s in range(ns) if any(row[s] for row in T)]
    live = [s for s in keep if _var([row[s] for row in Dm]) > 1e-12]
    if not live:
        return [[row[s] for s in keep] for row in T], [0.0] * n
    w = _precision_weights([[row[s] for s in live] for row in Dm])
    scale = sum(w[j] * sum(row[live[j]] for row in T) / n for j in range(len(live)))
    if scale > 0:                       # one unit of contrast = one unit of mean slice mass
        w = [x / scale for x in w]
    else:
        # a negative total weight would flip what "the cross class is worse" means, so a
        # covariance that produces one is not used: fall back to equal weights, which keep the
        # direction by construction and give up only the exact orthogonality.
        w = [1.0 / len(live)] * len(live)
    dif = [sum(w[j] * row[live[j]] for j in range(len(live))) for row in Dm]
    return [[row[s] for s in keep] for row in T], dif


def _var(xs):
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)


def _precision_weights(D):
    """``Sigma^-1 1`` for the columns of ``D``, with a ridge; equal weights if that is singular."""
    p = len(D[0])
    mu = [sum(row[j] for row in D) / len(D) for j in range(p)]
    S = [[sum((row[a] - mu[a]) * (row[b] - mu[b]) for row in D) / max(1, len(D) - 1)
          for b in range(p)] for a in range(p)]
    tr = sum(S[a][a] for a in range(p)) or 1.0
    for a in range(p):
        S[a][a] += 1e-6 * tr / p
    w = _solve(S, [1.0] * p)
    return w if w is not None else [1.0] * p


# --- the fit: P(correct) = exp(-X theta); one-sided LRT on a contrast of columns -----------
def _solve(A, b):
    """Dense linear solve by Gaussian elimination with partial pivoting (pure stdlib; the
    design matrices here have three to five columns)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-14:
            return None
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def _loglik(X, th, y, ridge):
    ll = 0.0
    ps = []
    for row, yi in zip(X, y):
        eta = sum(a * b for a, b in zip(row, th))
        p = math.exp(-min(max(eta, -30.0), 30.0))
        p = min(max(p, 1e-9), 1 - 1e-9)
        ps.append(p)
        ll += math.log(p) if yi else math.log1p(-p)
    return ll - ridge * sum(t * t for t in th), ps


def fit(X, y, ridge=1e-6, iters=60):
    """ML fit of P(correct) = exp(-X theta) by damped Newton. theta = 0 is a likelihood
    boundary, so the fit starts by matching the mean accuracy along the column means."""
    d = len(X[0])
    means = [sum(r[j] for r in X) / len(X) for j in range(d)]
    ybar = min(max(sum(y) / len(y), 1e-4), 1 - 1e-4)
    denom = sum(v * v for v in means)
    eta0 = -math.log(ybar)
    th = [eta0 / denom * v for v in means] if denom > 0 else [0.0] * d
    ll, ps = _loglik(X, th, y, ridge)
    for _ in range(iters):
        grad = [0.0] * d
        H = [[0.0] * d for _ in range(d)]
        for row, yi, p in zip(X, y, ps):
            u = (yi / p) - ((1 - yi) / (1 - p))
            dp = -p
            g = u * dp
            h = -(yi / p ** 2 + (1 - yi) / (1 - p) ** 2) * dp * dp - u * dp
            for a in range(d):
                grad[a] += row[a] * g
                for b in range(d):
                    H[a][b] += row[a] * row[b] * h
        for a in range(d):
            grad[a] -= 2 * ridge * th[a]
            H[a][a] -= 2 * ridge + 1e-7
        step = _solve(H, grad)
        if step is None:
            break
        t, improved = 1.0, False
        for _bt in range(30):
            cand = [th[a] - t * step[a] for a in range(d)]
            llc, psc = _loglik(X, cand, y, ridge)
            if llc > ll + 1e-12:
                th, ll, ps, improved = cand, llc, psc, True
                break
            t *= 0.5
        if not improved or max(abs(t * s) for s in step) < 1e-10:
            break
    return th, ll


CHI2_1_90 = 2.70554          # one-sided alpha = 0.05 on a single contrast

# The registered forms of the diagnostic, each ``(free nuisance columns, retrieval-distance
# bins)``. The design is always those columns, then one MASS column per stratum, then the
# CONTRAST column; the statistic is the coefficient on that last column and the test drops it.
# NONE of them is a primary: no form of this contrast identifies composition on this rendering,
# because within a kind the class label IS the printed clause (module docstring). What varies
# between them is only which composition-free hazards they are blind to.
#   T_kind     THE DEFAULT FORM: strata are the two event kinds. Any hazard that is a function of
#              the event kind is in the span with a zero contrast, and the precision weighting
#              makes the contrast orthogonal to the surface-clause direction (_stratum_columns).
#   T_strat    strata are kind x retrieval-distance quartile — the same argument extended to any
#              hazard that is a function of the distance, at the resolution of the bins.
#   T_kindWD   T_kind with the item's read-history load entered as free nuisances as well.
#   T_cross    the shape the family carried before: the two raw class columns tested against each
#              other, with no stratification. Kept measured so what the stratified design buys is
#              visible rather than asserted.
STATS = {
    "T_kind": ((), 1),
    "T_strat": ((), 4),
    "T_kindWD": (("W", "D"), 1),
}
TWO_CLASS_STATS = {
    "T_cross": ("nw", "nz", "nx"),
}
DIAGNOSTIC_STAT = "T_kind"       # the default form of the structure-switch diagnostic


def lrt(cols, y, i_hi, i_lo):
    """One-sided LRT of theta[i_hi] > theta[i_lo]. Returns ``(contrast, reject)``."""
    X = [list(row) for row in zip(*cols)]
    th, ll1 = fit(X, y)
    Xn = []
    for row in X:
        r = list(row)
        r[i_lo] += r[i_hi]
        del r[i_hi]
        Xn.append(r)
    _th0, ll0 = fit(Xn, y)
    c = th[i_hi] - th[i_lo]
    return c, bool(c > 0 and 2 * (ll1 - ll0) > CHI2_1_90)


def lrt_drop(cols, y, i):
    """One-sided LRT of theta[i] > 0, against the model with that column removed."""
    X = [list(row) for row in zip(*cols)]
    th, ll1 = fit(X, y)
    _th0, ll0 = fit([row[:i] + row[i + 1:] for row in X], y)
    return th[i], bool(th[i] > 0 and 2 * (ll1 - ll0) > CHI2_1_90)


def contrast(examples, correct, draws: int = 2, seed: int = 0,
             stat: str = DIAGNOSTIC_STAT) -> dict:
    """THE STRUCTURE-SWITCH DIAGNOSTIC for one cell: ``theta_cross - theta_same``, within item.

    NOT a composition measure. Within a kind the class label is the printed clause, so a solver
    that holds one structure and not the other is invisible to this: measured against
    ``one_structure_P`` / ``one_structure_B`` / a reads-B-only slip at k=6/L=64, n=800, it rejects
    0/40 times with the contrast pointing the wrong way (module docstring). It has power against
    solvers that degrade when the reference clause SWITCHES structure, which is what it is named
    for and all a caller may claim from it.

    ``examples`` are the cell's exact items and ``correct`` the per-item 0/1 match outcomes, in
    the same order. Returns the contrast, the one-sided LRT decision at alpha = 0.05, what the
    statistic identifies, and the class balance and read-history matching it rests on — so a
    caller reporting it reports both what makes it valid and what it does not show.
    """
    rng = random.Random(seed)
    all_ops, y = [], []
    bal = {"wx": 0.0, "wz": 0.0, "cx": 0, "cz": 0, "dx": 0.0, "dz": 0.0}
    per_kind: dict = {}
    for e, c in zip(examples, correct):
        rec = read(e.prompt)
        if rec is None:
            continue
        ops = op_slice(rec, draws=draws, rng=rng)
        for op in ops:
            if op["cls"] == "write":
                continue
            k = "x" if op["cls"] == "cross" else "z"
            bal["c" + k] += 1
            bal["w" + k] += op["w"]
            bal["d" + k] += op["d"]
            row = per_kind.setdefault(op["kind"], {"cx": 0, "cz": 0, "dx": 0.0, "dz": 0.0,
                                                   "wx": 0.0, "wz": 0.0, "sx": 0.0, "sz": 0.0})
            row["c" + k] += 1
            row["d" + k] += op["d"]
            row["w" + k] += op["w"]
            row["s" + k] += op["sens"]
        all_ops.append(ops)
        y.append(float(c))
    if not all_ops:
        return {}
    cov = [_cov_from_ops(ops) for ops in all_ops]
    if stat in TWO_CLASS_STATS:
        names = TWO_CLASS_STATS[stat]
        c, rej = lrt([[r[n] for r in cov] for n in names], y, len(names) - 1, len(names) - 2)
    else:
        free, bins = STATS[stat]
        T, dif = _stratum_columns(all_ops, bins)
        cols = [[r[n] for r in cov] for n in ("nw",) + free]
        cols += [[row[j] for row in T] for j in range(len(T[0]))] + [dif]
        c, rej = lrt_drop(cols, y, len(cols) - 1)
    return {
        "stat": stat, "contrast": c, "reject": rej, "n": len(y), "acc": sum(y) / len(y),
        # what a rejection licenses. Within a kind the class IS the clause, so a single-structure
        # carrier is not identified here at any n; this key travels with the number.
        "identifies": "structure_switch",
        "slice_cross": sum(r["nx"] for r in cov) / len(cov),
        "slice_same": sum(r["nz"] for r in cov) / len(cov),
        "slice_write": sum(r["nw"] for r in cov) / len(cov),
        "class_balance_cross": bal["cx"] / max(1, bal["cx"] + bal["cz"]),
        "mean_write_count_cross": bal["wx"] / max(1, bal["cx"]),
        "mean_write_count_same": bal["wz"] / max(1, bal["cz"]),
        "mean_distance_cross": bal["dx"] / max(1, bal["cx"]),
        "mean_distance_same": bal["dz"] / max(1, bal["cz"]),
        # WITHIN KIND is what the default strata are, and pooling hides it: the two kinds carry
        # the two clauses in opposite directions, so a class imbalance inside one kind can cancel
        # in the pooled column and still load on the contrast at full size.
        "within_kind": {kd: within_kind_matching(row) for kd, row in sorted(per_kind.items())},
    }


def within_kind_matching(row: dict) -> dict:
    """One kind's class matching: the share of ops that are CROSS, and the relative gap between
    the classes in retrieval distance, write count and slice mass. Every gap is reported as a
    signed fraction of the larger side, so a reader sees the size of the imbalance the contrast
    is being asked to be blind to."""
    def gap(x, z):
        return 0.0 if max(abs(x), abs(z)) == 0 else (x - z) / max(abs(x), abs(z))
    cx, cz = max(1, row["cx"]), max(1, row["cz"])
    return {"n_cross": row["cx"], "n_same": row["cz"],
            "cross_share": row["cx"] / (row["cx"] + row["cz"]) if row["cx"] + row["cz"] else 0.0,
            "distance_gap": gap(row["dx"] / cx, row["dz"] / cz),
            "write_count_gap": gap(row["wx"] / cx, row["wz"] / cz),
            "slice_mass_gap": gap(row["sx"] / cx, row["sz"] / cz)}
