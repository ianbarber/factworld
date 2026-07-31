"""The source-structure composition statistic, and the cost convention it is priced against.

Two things live here, both for the ``s5_bind_v3`` family (TaskSpec.source_ablation):

  THE PRIMARY STATISTIC   ``theta_cross - theta_same``, a within-item, within-cell contrast
                          between the two op classes the construct ablates on. A cell reports
                          it alongside match (``contrast``).
  THE COST CONVENTION     a stated, testable rule for what one step is, and a counter that
                          implements it (``cost_report``). The convention is what decides the
                          step multiplier and the Pareto floor class, so it is written down
                          rather than left as a bookkeeping choice.

Everything reads the RENDERED prompt through this module's own parser: nothing here consults
``Example.meta`` or the sampler, so every number is recomputed from exactly what a model saw.
Pure stdlib, like the rest of the package.

-------------------------------------------------------------------------------------------
THE STATISTIC
-------------------------------------------------------------------------------------------
The composed cell's events each name their second operand LIVE through one of the two
structures. An op is CROSS when it reads the structure it does not write (a swap reading the
holder map; a give reading the pointer map) and SAME when it reads the one it does write. Both
classes are live reads of overwritten cells at matched write counts and matched retrieval
distances — the sampler's ``p_swap = 1/3`` at ``m = k`` equalises the two structures' write
rates exactly — so a failure that depends on read history is common to the two classes and
cancels in the contrast, while a solver that cannot hold the other structure fails only on
CROSS.

THE SURFACE CONFOUND, AND WHAT CANCELS IT. Within an event kind the class is carried by the
reference clause: on a swap CROSS is "belongs to" and SAME is "points to", on a give it is the
other way round. So a solver that is simply WORSE AT ONE CLAUSE — a surface failure with no
composition deficit in it — slips on swap-CROSS and give-SAME. The clause-to-class map flips
between the kinds, so the effect would cancel if the two kinds carried equal weight; they do
not (a swap resolution sits directly on the answer's path, a give resolution only reaches it
through a later swap, so their mean answer sensitivities are ~0.72 and ~0.15). Measured, the
raw contrast rejects a pure clause slip at 0.117-0.211 at n=500, which is the same order as its
power against a real deficit.

THE KIND-BALANCED CONTRAST fixes that by construction: each op's weight is divided by its event
kind's mean slice mass in this cell, so swaps and gives contribute equally to both class columns
and a clause effect enters the two columns at equal size. It is the registered primary
(``T_kind``); the raw ``T_cross`` stays measured, as the diagnostic that shows what the
balancing buys.

Fit, over items of one cell:

    P(item correct) = q,     q = exp(-(theta_w w + theta_z z + theta_x x))

where, for the answer's dependency slice, ``w`` counts the writes and the readout, ``z`` the
SAME resolutions and ``x`` the CROSS ones, each weighted by that op's ANSWER SENSITIVITY (the
probability that garbling it changes the answer, measured by one-at-a-time perturbation of the
clean trajectory). The reported statistic is ``theta_x - theta_z`` with a one-sided likelihood
ratio test. z and x are the same operation, in the same position of the same algorithm, at the
same cost; they differ only in which structure supplied the value.

This replaces the temporal contrast the family carried before, which could not identify
composition at all: the temporal ablation's composition class was, by construction, the
overwritten-cell class (see TaskSpec.source_ablation).

WHAT IT IDENTIFIES, AND AT WHAT SAMPLE SIZE. Measured on the generated cells with an
independent parser and replay, R = 1000 resamples, seven composition-free executors and two
composition deficits all dialled to the SAME accuracy cost (a drop of 0.10 / 0.20 / 0.30 from a
0.90 base), one-sided alpha = 0.05 (scripts/probe_s5bind_v3_statistic_20260731.py):

  TYPE-I, over uniform per-op slip, read slip linear in the write count / in the retrieval
  distance / in the op's depth in the stream, stale-value interference on overwritten cells, a
  bounded working set with LRU eviction, and a surface-clause slip:
      k=6,  L=64 :  0.041 - 0.084 at every executor, cost and n
      k=12, L=192:  0.044 - 0.085 at every executor, cost and n
  The raw contrast is at alpha on six of the seven and reads 0.072-0.211 on the surface-clause
  one, which is what the kind balancing buys.

  POWER against a stated-map fallback / an outright garbled cross reference, at n = 100/200/500:
      k=6,  L=64 :  0.11/0.13/0.22, 0.17/0.20/0.36, 0.20/0.31/0.52 at the three costs
      k=12, L=192:  0.08/0.10/0.13, 0.12/0.15/0.22, 0.13/0.19/0.31
  So the statistic is an instrument for the FROM-SCRATCH regime, where thousands of items are
  free, and not for a few-hundred-item frontier budget: at n = 500 it detects a deficit costing
  0.30 of accuracy about half the time at the local cell and a third of the time at the frontier
  one. Entering the read-history load as a free nuisance (``T_kindWD``) holds every null at or
  below 0.068 and costs about half the power.

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


def _cov_from_ops(ops, balance=None) -> dict:
    """Per-item covariates of the fit.

    ``nw`` writes and the readout; ``nz`` SAME resolutions; ``nx`` CROSS resolutions; each
    weighted by answer sensitivity. ``W``/``W2``/``D`` are the read-history load the CROSS class
    carries, entered as nuisances by the repairs. ``bz``/``bx`` are the KIND-BALANCED class
    columns: each op divided by its event kind's mean slice mass (``balance``), so swaps and
    gives contribute equally and a surface-clause effect cancels in ``bx - bz``.
    """
    cov = {"nw": 0.0, "nz": 0.0, "nx": 0.0, "bz": 0.0, "bx": 0.0,
           "W": 0.0, "W2": 0.0, "D": 0.0}
    for op in ops:
        s = op["sens"]
        if op["cls"] == "write":
            cov["nw"] += s
            continue
        g = 1.0 if balance is None else balance.get(op["kind"], 1.0)
        if op["cls"] == "same":
            cov["nz"] += s
            cov["bz"] += s * g
        else:
            cov["nx"] += s
            cov["bx"] += s * g
            cov["W"] += s * op["w"]
            cov["W2"] += s * op["w"] * op["w"]
            cov["D"] += s * op["d"]
    cov["nw"] += 1.0                                   # the readout
    return cov


def kind_balance(all_ops) -> dict:
    """The per-kind weights that make swaps and gives contribute equally to the class columns:
    one over each kind's mean CROSS slice mass. A swap resolution sits on the answer's path and
    a give resolution reaches it only through a later swap, so the raw masses differ by ~5x and
    the surface-clause confound does not cancel without this."""
    mass: dict[str, float] = {}
    for ops in all_ops:
        for op in ops:
            if op["cls"] == "cross":
                mass[op["kind"]] = mass.get(op["kind"], 0.0) + op["sens"]
    n = max(1, len(all_ops))
    return {k: (n / v if v > 0 else 0.0) for k, v in mass.items()}


def item_covariates(rec, draws: int = 2, rng: random.Random | None = None,
                    balance=None) -> dict:
    """Per-item covariates of the fit — see ``_cov_from_ops``."""
    return _cov_from_ops(op_slice(rec, draws=draws, rng=rng), balance)


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

# The registered statistics. Each is (columns, the composition column, the reference column).
#   T_cross    the published contrast: CROSS resolutions against SAME ones, at matched slice
#              depth and matched op count.
#   T_crossW   the write-count repair — the CROSS class's read-history load entered as a free
#              nuisance. Valid against a read-history effect linear in the write count.
#   T_crossW2  adds a quadratic term: any smooth read-history effect vanishing at zero writes.
#   T_crossWD  adds retrieval distance as well.
#   T_kind     THE PRIMARY: the same contrast on the KIND-BALANCED columns, so a surface-clause
#              failure — worse at "belongs to" than at "points to", with no composition deficit
#              in it — enters the two class columns at equal size and cancels.
STATS = {
    "T_kind": (("nw", "bz", "bx"), 2, 1),
    "T_kindW": (("nw", "bz", "bx", "W"), 2, 1),
    "T_kindWD": (("nw", "bz", "bx", "W", "D"), 2, 1),
    "T_cross": (("nw", "nz", "nx"), 2, 1),
    "T_crossW": (("nw", "nz", "nx", "W"), 2, 1),
    "T_crossW2": (("nw", "nz", "nx", "W", "W2"), 2, 1),
    "T_crossWD": (("nw", "nz", "nx", "W", "D"), 2, 1),
}
PRIMARY_STAT = "T_kind"


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


def contrast(examples, correct, draws: int = 2, seed: int = 0, stat: str = PRIMARY_STAT) -> dict:
    """THE PRIMARY STATISTIC for one cell: ``theta_cross - theta_same``, per item, within item.

    ``examples`` are the cell's exact items and ``correct`` the per-item 0/1 match outcomes, in
    the same order. Returns the contrast, the one-sided LRT decision at alpha = 0.05, and the
    class balance and read-history matching the contrast rests on — so a caller reporting the
    statistic reports what makes it valid alongside it.
    """
    rng = random.Random(seed)
    all_ops, y = [], []
    bal = {"wx": 0.0, "wz": 0.0, "cx": 0, "cz": 0, "dx": 0.0, "dz": 0.0}
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
        all_ops.append(ops)
        y.append(float(c))
    if not all_ops:
        return {}
    weights = kind_balance(all_ops)
    cov = [_cov_from_ops(ops, weights) for ops in all_ops]
    names, hi, lo = STATS[stat]
    cols = [[r[n] for r in cov] for n in names]
    c, rej = lrt(cols, y, hi, lo)
    return {
        "stat": stat, "contrast": c, "reject": rej, "n": len(y), "acc": sum(y) / len(y),
        "slice_cross": sum(r["nx"] for r in cov) / len(cov),
        "slice_same": sum(r["nz"] for r in cov) / len(cov),
        "slice_write": sum(r["nw"] for r in cov) / len(cov),
        "kind_balance": weights,
        "class_balance_cross": bal["cx"] / max(1, bal["cx"] + bal["cz"]),
        "mean_write_count_cross": bal["wx"] / max(1, bal["cx"]),
        "mean_write_count_same": bal["wz"] / max(1, bal["cz"]),
        "mean_distance_cross": bal["dx"] / max(1, bal["cx"]),
        "mean_distance_same": bal["dz"] / max(1, bal["cz"]),
    }
