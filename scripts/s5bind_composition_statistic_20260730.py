"""Whether any statistic on s5_bind separates a composition deficit from read history. It does not.

WHAT WAS ON THE TABLE. The composed cell's claim is that the coupled rendering forces a forward
pass carrying both structures. A score cannot support it: an executor with no composition-specific
failure at all — one per-op slip rate — already reads far below the component arms, because the
composed arm's dependency slice is longer. The candidate answer was an OP-TYPE contrast inside one
cell: fit ``P(correct) = exp(-(theta_w w + theta_z z + theta_x x))`` over items, splitting the
answer's dependency slice into writes, resolutions that did NOT need the other structure's running
state, and resolutions that DID, and read ``theta_x - theta_z``. The two classes are the same
operation in the same position of the same algorithm at the same cost, so a slip rate that does
not care where the value came from cancels.

WHY IT CANNOT BE REPAIRED. A reference witnesses composition exactly when the coupled and
uncoupled readings return different operands, and that happens exactly when the referenced cell
has been written since the start — otherwise the two renderings of the item produce the same
trajectory and no observable distinguishes them. The x class is therefore not correlated with the
read cell's write history: it IS its indicator. Measured on the generator,
P(needs running state | w=0) = 0.000 and P(. | w=1) = 1.000 at both scored cells.

The consequence is an identity, not a correlation. The deficit the construct was built to detect —
resolve a cross-structure reference against the STATED map with probability gamma — and an
interference null — an overwritten cell returns its STALE value with probability delta — fire on
the same op set and write the same wrong value. They are one executor. Matching on the write count
has no support (there is no cross-structure reference at w=0), a within-item contrast controls
item-level nuisances and this one is per-op, and a discontinuity at the first write is exactly
where a distractor first exists. So the type-I rate of any x-versus-z test under the interference
null equals its power against the deficit, at every n.

WHAT THIS SCRIPT MEASURES, on generated items with an independent parser and simulator:
  1. SUPPORT      the joint distribution of (needs-running-state, write count) over the slice.
  2. THE IDENTITY the two executors' per-item P(correct), algebraically and by direct replay, and
                  the knob values they calibrate to at matched accuracy cost.
  3. THE TESTS    the op-type contrast and three write-history-controlled repairs, under four
                  composition-free executors and two deficits, all scaled to the SAME accuracy
                  cost; type-I and power at R=1000, n = 100/200/500, at both scored cells and one
                  rung of the coupling-dose ladder.
  4. THE TIE-BREAK the decoupled bind component is the only arm the two executors differ on, and
                  the sample size that costs.
  5. THE DESIGN   the minimal construct change that restores identification — move the ablation
                  from the TIME INDEX to the SOURCE STRUCTURE — run through the same tests.

Accuracies come from a direct vectorised noisy replay of the forward pass, not from a linearised
model; only the covariates are one-at-a-time sensitivities on the clean trajectory.

Nothing here trains or calls an API.

Usage:
    .venv-train/bin/python scripts/s5bind_composition_statistic_20260730.py --stage features
    .venv-train/bin/python scripts/s5bind_composition_statistic_20260730.py --stage report
    .venv-train/bin/python scripts/s5bind_composition_statistic_20260730.py --stage design
"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, "/home/ianbarber/Projects/factworld")
import numpy as np

from factworld.tasks import generate, spec_for

CACHE = os.environ.get(
    "FW_SCRATCH",
    "/tmp/claude-1000/-home-ianbarber-Projects-factworld/"
    "4cbf0f9f-690a-4617-9f5a-bb6ae025f925/scratchpad")

# (spec, length, k, m) — the two scored operating points of the composed arm, plus one rung of
# the coupling-dose ladder, which is the only cell of the family that renders BOTH a static and
# a dynamic reference and so is the only place a rendering contrast has support at all.
CELLS = {
    "frontier": ("s5_bind_v2", 192, 12, 12),
    "local":    ("s5_bind_local_v2", 64, 6, 6),
    "ladder50": ("s5_bind_v2_lad50", 192, 12, 12),
}

# op kinds
RES_ST, RES_DYN, WRITE, READOUT = 0, 1, 2, 3

RE_P0 = re.compile(r"(g\d+) has role (r\d+) at the start\.")
RE_B0 = re.compile(r"(g\d+) holds (o\d+) at the start\.")
RE_SWAP = re.compile(r"^swaps the roles of (g\d+) and the agent who holds (o\d+) "
                     r"(at this point|at the start)\.$")
RE_GIVE = re.compile(r"^gives (o\d+) to the agent whose role (at this point|at the start) "
                     r"is (r\d+)\.$")
RE_QR = re.compile(r"what role does (g\d+) have at the end\?")


def parse(prompt):
    """Rendered surface -> (P0, B0, events, queried agent), all as small ints.

    Shares no code with the sampler: the events, their operands and their temporal phrase are
    read back off the prompt a model sees.
    """
    p0 = RE_P0.findall(prompt)
    b0 = RE_B0.findall(prompt)
    agents = sorted({g for g, _ in p0}, key=lambda s: int(s[1:]))
    roles = sorted({r for _, r in p0}, key=lambda s: int(s[1:]))
    objs = sorted({o for _, o in b0}, key=lambda s: int(s[1:]))
    ai = {a: i for i, a in enumerate(agents)}
    ri = {r: i for i, r in enumerate(roles)}
    oi = {o: i for i, o in enumerate(objs)}
    P0 = [0] * len(agents)
    for g, r in p0:
        P0[ai[g]] = ri[r]
    B0 = [0] * len(objs)
    for g, o in b0:
        B0[oi[o]] = ai[g]
    body = prompt[prompt.index(" s0 ") + 1:]
    body = RE_QR.sub("", body).strip()
    parts = re.split(r"\bs(\d+) ", body)
    evs = []
    for i in range(1, len(parts) - 1, 2):
        txt = parts[i + 1].strip()
        m = RE_SWAP.match(txt)
        if m:
            evs.append((0, ai[m.group(1)], oi[m.group(2)], m.group(3) == "at this point"))
            continue
        m = RE_GIVE.match(txt)
        if m:
            evs.append((1, oi[m.group(1)], ri[m.group(3)], m.group(2) == "at this point"))
            continue
        raise ValueError(f"unparsed event: {txt!r}")
    q = RE_QR.search(prompt)
    return P0, B0, evs, ai[q.group(1)], roles, len(agents), len(objs)


def _resolve(ev, P0inv, Pi, B0, B):
    """The event's second operand under the rendering's own semantics."""
    kind, u, v, dyn = ev
    if kind == 0:                      # swap: names an object, reads the holder map
        return B[v] if dyn else B0[v]
    return Pi[v] if dyn else P0inv[v]  # give: names a role, reads the role map's inverse


def _apply(ev, P, Pi, B, x):
    kind, u, v, _ = ev
    if kind == 0:
        ra, rx = P[u], P[x]
        if ra != rx:
            P[u], P[x] = rx, ra
            Pi[rx], Pi[ra] = u, x
    else:
        B[u] = x


def _replay(P, Pi, B, evs, j, P0inv, B0):
    for t in range(j, len(evs)):
        e = evs[t]
        _apply(e, P, Pi, B, _resolve(e, P0inv, Pi, B0, B))


def item_features(P0, B0, evs, qa, k, m, rng, draws):
    """Per-op covariates and answer sensitivities for one composed item.

    Ops, in the cheapest correct algorithm (one forward pass carrying P, its inverse and B):
    per event one RESOLUTION of the second operand and one WRITE, plus the final READOUT of the
    queried agent's slot. Each op carries

      kind  RES_ST / RES_DYN / WRITE / READOUT
      w     the write count of the cell the op READS, at the moment it reads it (0 for a static
            resolution, which reads the stated map in the prompt header)
      moved 1 if the running reading differs from the stated one — i.e. the resolution actually
            needed the other structure's running state
      qgar  P(the final answer changes | this op returns a uniformly random wrong agent)
      qsta  1 if returning the STATED value of the read cell changes the final answer

    ``qgar`` is the op's weight in the answer's dependency slice: it is 0 for an op the answer
    does not depend on and 1 for one that always propagates. ``qsta`` is what a stated-map
    fallback costs at that op, and is identically 0 wherever the cell has not moved.
    """
    P0inv = [0] * k
    for a, r in enumerate(P0):
        P0inv[r] = a
    P, Pi, B = list(P0), list(P0inv), list(B0)
    snaps, xs, ws, ds = [], [], [], []
    ogive = [0] * m          # writes to each object so far
    orole = [0] * k          # changes to each role's holder so far
    lgive = [-1] * m         # index of the last write to each object
    lrole = [-1] * k         # index of the last event that moved each role's holder
    ptouch = 0               # changes to the queried agent's slot
    for j, e in enumerate(evs):
        snaps.append((tuple(P), tuple(Pi), tuple(B)))
        x = _resolve(e, P0inv, Pi, B0, B)
        xs.append(x)
        if e[0] == 0:
            ws.append(ogive[e[2]])
            ds.append(j - lgive[e[2]])          # events back to the value the read returns
            ra, rx = P[e[1]], P[x]
            if ra != rx:
                orole[ra] += 1
                orole[rx] += 1
                lrole[ra] = lrole[rx] = j
                if e[1] == qa or x == qa:
                    ptouch += 1
        else:
            ws.append(orole[e[2]])
            ds.append(j - lrole[e[2]])
            ogive[e[1]] += 1
            lgive[e[1]] = j
        _apply(e, P, Pi, B, x)
    gold = P[qa]
    L = len(evs)

    kinds, wcol, dcol, moved, qgar, qsta = [], [], [], [], [], []
    for j, e in enumerate(evs):
        Ps, Pis, Bs = snaps[j]
        xt = xs[j]
        xstale = (B0[e[2]] if e[0] == 0 else P0inv[e[2]])
        dyn = e[3]
        # ---- resolution op
        hit = 0
        for _ in range(draws):
            xp = rng.randrange(k - 1)
            if xp >= xt:
                xp += 1
            Pc, Pic, Bc = list(Ps), list(Pis), list(Bs)
            _apply(e, Pc, Pic, Bc, xp)
            _replay(Pc, Pic, Bc, evs, j + 1, P0inv, B0)
            hit += (Pc[qa] != gold)
        qg = hit / draws
        if dyn and xstale != xt:
            Pc, Pic, Bc = list(Ps), list(Pis), list(Bs)
            _apply(e, Pc, Pic, Bc, xstale)
            _replay(Pc, Pic, Bc, evs, j + 1, P0inv, B0)
            qs = float(Pc[qa] != gold)
        else:
            qs = 0.0
        kinds.append(RES_DYN if dyn else RES_ST)
        wcol.append(ws[j] if dyn else 0)
        dcol.append(ds[j] if dyn else 0)
        moved.append(1 if (dyn and xstale != xt) else 0)
        qgar.append(qg)
        qsta.append(qs)
        # ---- write op: the update is lost
        Pc, Pic, Bc = list(Ps), list(Pis), list(Bs)
        _replay(Pc, Pic, Bc, evs, j + 1, P0inv, B0)
        kinds.append(WRITE)
        wcol.append(0)
        dcol.append(0)
        moved.append(0)
        qgar.append(float(Pc[qa] != gold))
        qsta.append(0.0)
    # ---- readout of the queried slot
    kinds.append(READOUT)
    wcol.append(ptouch)
    dcol.append(0)
    moved.append(1)
    qgar.append(1.0)
    qsta.append(float(P0[qa] != gold))
    return (np.array(kinds, np.int8), np.array(wcol, np.int16),
            np.array(dcol, np.int16), np.array(moved, np.int8),
            np.array(qgar, np.float32), np.array(qsta, np.float32), gold, L)


def build(cell, n, draws, seed=20260730):
    name, L, k, m = CELLS[cell]
    spec = spec_for(name)
    ex = generate(spec, "test", n=n, length=L)
    rng = __import__("random").Random(seed)
    K, W, D, M, QG, QS, IDX, BW = [], [], [], [], [], [], [], []
    t0 = time.time()
    for i, e in enumerate(ex):
        P0, B0, evs, qa, roles, kk, mm = parse(e.prompt)
        assert kk == k and mm == m and len(evs) == L
        ki, wi, di, mi, qg, qs, gold, _ = item_features(P0, B0, evs, qa, k, m, rng, draws)
        assert roles[gold] == e.answer.strip().rstrip("."), "simulator disagrees with gold"
        K.append(ki); W.append(wi); D.append(di); M.append(mi); QG.append(qg); QS.append(qs)
        IDX.append(np.full(len(ki), i, np.int32))
        BW.append(e.meta["writes"])          # writes to the bind arm's queried object
        if (i + 1) % 400 == 0:
            print(f"  {cell}: {i+1}/{n} items  {time.time()-t0:.0f}s", flush=True)
    out = dict(kind=np.concatenate(K), w=np.concatenate(W), d=np.concatenate(D),
               moved=np.concatenate(M),
               qgar=np.concatenate(QG), qsta=np.concatenate(QS),
               item=np.concatenate(IDX), bind_w=np.array(BW, np.int32),
               n=np.int32(n), k=np.int32(k), L=np.int32(L))
    np.savez_compressed(os.path.join(CACHE, f"s5bind_ops_{cell}.npz"), **out)
    print(f"  {cell}: {n} items, {len(out['kind'])} ops, {time.time()-t0:.0f}s", flush=True)
    return out


def load(cell):
    return dict(np.load(os.path.join(CACHE, f"s5bind_ops_{cell}.npz")))


# ---------------------------------------------------------------------------
# executors: a per-op garble rate and a per-op stale rate; P(item correct) is the product over
# ops of surviving both channels.  Nothing here is composition-aware except A_comp / A_garble.
# ---------------------------------------------------------------------------
def rates(F, eps, wlin=0.0, wjump=0.0, dlin=0.0, gamma=0.0, garble=0.0,
          readout_stale=False):
    """Per-op garble and stale rates for one executor.

    Composition-free (the test must NOT fire):
      wlin   read slip linear in the read cell's write count
      dlin   read slip linear in the retrieval distance back to the value the read returns
      wjump  a cell that has been overwritten returns its STALE value — proactive interference,
             flat in the write count, and the shape the write-linear control cannot absorb
    Composition deficits (the test MUST fire):
      gamma  a reference rendered "at this point" is resolved against the STATED map
      garble the same reference is resolved to a wrong agent outright
    """
    kind, w = F["kind"], F["w"].astype(np.float64)
    d = F["d"].astype(np.float64)
    dyn = kind == RES_DYN
    e_gar = np.full(kind.shape, float(eps))
    e_sta = np.zeros(kind.shape)
    if wlin:
        e_gar = e_gar + wlin * w * dyn
    if dlin:
        e_gar = e_gar + dlin * d * dyn
    if wjump:
        e_sta = e_sta + wjump * ((w >= 1) & dyn)
        if readout_stale:
            e_sta = e_sta + wjump * (kind == READOUT)
    if gamma:
        e_sta = e_sta + gamma * dyn
    if garble:
        e_gar = e_gar + garble * dyn
    return e_gar, e_sta


def p_correct(F, **kw):
    """P(the item is answered correctly) under one executor.

    An op that slips is wrong only if the answer depends on it, which is what ``qgar``/``qsta``
    carry, so the guessing floor is already inside those weights and none is added here.
    """
    e_gar, e_sta = rates(F, **kw)
    surv = np.log1p(-np.clip(e_gar * F["qgar"], 0, 0.999)) \
         + np.log1p(-np.clip(e_sta * F["qsta"], 0, 0.999))
    n = int(F["n"])
    return np.exp(np.bincount(F["item"], weights=surv, minlength=n))


# ---------------------------------------------------------------------------
# The executors, run for real: a vectorised noisy replay of the forward pass, ``reps``
# trajectories per item at once.  The linearised product-of-survivals model reproduces these
# accuracies in aggregate but not item by item (slips interact), so every accuracy below comes
# from this replay and only the COVARIATES — one-at-a-time sensitivities on the clean
# trajectory — come from ``item_features``.
# ---------------------------------------------------------------------------
def noisy_acc(items, k, m, reps, rng, eps=0.0, wlin=0.0, dlin=0.0, wjump=0.0,
              gamma=0.0, garble=0.0):
    """P(correct) per item under one executor, by direct replay.

    An item is ``(P0, B0, evs, qa)`` with ``evs`` a list of
    ``(kind, first operand, reference, dynamic?, reads_B?, cross?)``; the last two default to
    the s5_bind rendering, where a swap reads the holder map, a give reads the role map's
    inverse, and every dynamic reference is a cross-structure one.

    Every op can slip: a resolution returns a wrong agent, a write is lost, the readout is
    garbled. ``wlin``/``dlin``/``wjump`` are read-history failures, ``gamma``/``garble``
    composition ones; nothing else differs between them.
    """
    out = np.empty(len(items))
    ar = np.arange(reps)
    for it, (P0, B0, evs, qa) in enumerate(items):
        P0a = np.array(P0)
        P0inv = np.empty(k, np.int64)
        P0inv[P0a] = np.arange(k)
        B0a = np.array(B0)
        P = np.tile(P0a, (reps, 1))
        Pi = np.tile(P0inv, (reps, 1))
        B = np.tile(B0a, (reps, 1))
        ogive = np.zeros((reps, m), np.int64)
        orole = np.zeros((reps, k), np.int64)
        lgive = np.full((reps, m), -1, np.int64)
        lrole = np.full((reps, k), -1, np.int64)
        for j, ev in enumerate(evs):
            kind, u, v, dyn = ev[:4]
            reads_B = ev[4] if len(ev) > 4 else (kind == 0)
            cross = ev[5] if len(ev) > 5 else dyn
            if reads_B:
                x = B[:, v].copy() if dyn else np.full(reps, B0a[v])
                w, last, stale = ogive[:, v], lgive[:, v], np.full(reps, B0a[v])
            else:
                x = Pi[:, v].copy() if dyn else np.full(reps, P0inv[v])
                w, last, stale = orole[:, v], lrole[:, v], np.full(reps, P0inv[v])
            if dyn:
                e = eps + wlin * w + dlin * (j - last) + garble * cross
                r = wjump * (w >= 1) + gamma * cross
                if np.any(r):
                    x = np.where(rng.random(reps) < r, stale, x)
            else:
                e = np.full(reps, float(eps))
            bad = rng.random(reps) < e
            if bad.any():
                alt = rng.integers(0, k - 1, reps)
                x = np.where(bad, alt + (alt >= x), x)
            drop = rng.random(reps) < eps
            if kind == 0:
                ra, rx = P[ar, u], P[ar, x]
                mv = ra != rx
                orole[mv, ra[mv]] += 1
                orole[mv, rx[mv]] += 1
                lrole[mv, ra[mv]] = j
                lrole[mv, rx[mv]] = j
                go = mv & ~drop
                P[go, u] = rx[go]
                P[go, x[go]] = ra[go]
                Pi[go, rx[go]] = u
                Pi[go, ra[go]] = x[go]
            else:
                ogive[:, u] += 1
                lgive[:, u] = j
                B[~drop, u] = x[~drop]
        Pc, Pic, Bc = list(P0), P0inv.tolist(), list(B0)
        _replay_gen(Pc, Pic, Bc, evs, P0inv.tolist(), B0)
        gold = Pc[qa]
        ans = P[:, qa].copy()
        bad = rng.random(reps) < eps
        if bad.any():
            alt = rng.integers(0, k - 1, reps)
            ans = np.where(bad, alt + (alt >= ans), ans)
        out[it] = float(np.mean(ans == gold))
    return out


def _replay_gen(P, Pi, B, evs, P0inv, B0):
    """The exact forward pass, honouring the per-event read-structure flag."""
    for ev in evs:
        kind, u, v, dyn = ev[:4]
        reads_B = ev[4] if len(ev) > 4 else (kind == 0)
        if reads_B:
            x = B[v] if dyn else B0[v]
        else:
            x = Pi[v] if dyn else P0inv[v]
        if kind == 0:
            ra, rx = P[u], P[x]
            if ra != rx:
                P[u], P[x] = rx, ra
                Pi[rx], Pi[ra] = u, x
        else:
            B[u] = x


def cell_items(cell):
    name, L, k, m = CELLS[cell]
    ex = generate(spec_for(name), "test", n=int(load(cell)["n"]), length=L)
    return [parse(e.prompt)[:4] for e in ex]


def solve_noisy(items, k, m, knob, eps, target, rng, reps=192, sub=600):
    """The knob value at which the composed cell reads ``target``, by direct replay."""
    lo, hi = 0.0, 0.5
    it = items[:sub]
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        kw = {"eps": mid} if knob == "eps" else {"eps": eps, knob: mid}
        if noisy_acc(it, k, m, reps, rng, **kw).mean() > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)



def covariates(F):
    """Per-item dependency-slice covariates, weighted by each op's answer sensitivity.

    The resolution ops partition three ways, not two:
      nz  rendered "at the start" — reads the prompt header, no running state, no write history
      nu  rendered "at this point" but the referenced cell has NOT moved, so the running and
          stated readings coincide
      nx  rendered "at this point" and the cell HAS moved — the only class on which the two
          renderings of the item produce different operands
    The published contrast is theta_x - theta_{z+u}. ``W`` and ``D`` are the write-history and
    retrieval-distance load the x class carries; they are what a repair enters as a nuisance.
    """
    n, kind, mv, qg = int(F["n"]), F["kind"], F["moved"], F["qgar"]
    w, d = F["w"].astype(np.float64), F["d"].astype(np.float64)
    dyn = kind == RES_DYN

    def agg(mask, weight=None):
        v = qg * mask if weight is None else qg * mask * weight
        return np.bincount(F["item"], weights=v, minlength=n)

    return dict(
        nw=agg((kind == WRITE) | (kind == READOUT)),
        nz=agg(kind == RES_ST),
        nu=agg(dyn & (mv == 0)),
        nzu=agg((kind == RES_ST) | (dyn & (mv == 0))),
        nx=agg(dyn & (mv == 1)),
        W=agg(dyn & (mv == 1), w),
        W2=agg(dyn & (mv == 1), w * w),
        D=agg(dyn & (mv == 1), d),
    )


# ---------------------------------------------------------------------------
# the fit: P(correct) = q + (1-q)/k, q = exp(-X theta); one-sided LRT on a contrast of columns
# ---------------------------------------------------------------------------
def _ll(X, th, y, a, ridge):
    q = np.exp(-np.clip(X @ th, -30, 30))
    p = np.clip(a + (1 - a) * q, 1e-9, 1 - 1e-9)
    return (float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))) - ridge * float(th @ th)), p, q


def _fit(X, y, a, ridge=1e-6, iters=80):
    """ML fit of P(correct) = q + (1-q)/k, q = exp(-X theta), by damped Newton.

    theta = 0 is a likelihood boundary (q = 1, p = 1), so the fit is started by matching the
    mean accuracy along the column means rather than at the origin.
    """
    m = X.mean(0)
    ybar = min(max(y.mean(), a + 1e-4), 1 - 1e-4)
    eta0 = -math.log(max((ybar - a) / (1 - a), 1e-6))
    denom = float(m @ m)
    th = (eta0 / denom) * m if denom > 0 else np.zeros(X.shape[1])
    ll, p, q = _ll(X, th, y, a, ridge)
    for _ in range(iters):
        d = -(1 - a) * q
        u = y / p - (1 - y) / (1 - p)
        grad = X.T @ (u * d) - 2 * ridge * th
        h = -(y / p**2 + (1 - y) / (1 - p)**2) * d * d - u * d
        H = (X * h[:, None]).T @ X - 2 * ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H - 1e-7 * np.eye(X.shape[1]), grad)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        t, improved = 1.0, False
        for _bt in range(30):                       # backtracking line search
            cand = th - t * step
            llc, pc, qc = _ll(X, cand, y, a, ridge)
            if llc > ll + 1e-12:
                improved, th, ll, p, q = True, cand, llc, pc, qc
                break
            t *= 0.5
        if not improved or np.max(np.abs(t * step)) < 1e-10:
            break
    return th, ll


CHI2_1_90 = 2.70554          # one-sided alpha = 0.05 on a single contrast


def lrt(cols, y, k, i_hi, i_lo):
    """One-sided LRT of theta[i_hi] > theta[i_lo]. Returns (contrast, reject)."""
    X = np.column_stack(cols)
    a = 0.0 * k          # the guessing floor is already inside qgar (see item_features)
    th, ll1 = _fit(X, y, a)
    Xn = X.copy()
    Xn[:, i_lo] = Xn[:, i_lo] + Xn[:, i_hi]
    Xn = np.delete(Xn, i_hi, axis=1)
    _, ll0 = _fit(Xn, y, a)
    c = th[i_hi] - th[i_lo]
    return c, bool(c > 0 and 2 * (ll1 - ll0) > CHI2_1_90)


STATS = {
    # name: (columns, index of the composition column, index of the reference column)
    # T_cross   the published op-type contrast: cross-structure resolutions against every other
    #           resolution, at matched slice depth and matched op count.
    # T_crossW  the obvious repair — the x class's write-history load entered as a free
    #           nuisance. Valid against a read-history effect LINEAR in the write count.
    # T_crossW2 adds a quadratic term: any smooth read-history effect vanishing at zero writes.
    # T_crossWD adds retrieval distance as well.
    # T_rend    the pure RENDERING contrast — dynamic-but-unmoved resolutions against static
    #           ones. Same read cell, same returned value, same write history, same distance;
    #           the two differ only in which phrase the prompt uses. Has no support wherever the
    #           cell renders every reference dynamically.
    "T_cross":   (("nw", "nzu", "nx"), 2, 1),
    "T_crossW":  (("nw", "nzu", "nx", "W"), 2, 1),
    "T_crossW2": (("nw", "nzu", "nx", "W", "W2"), 2, 1),
    "T_crossWD": (("nw", "nzu", "nx", "W", "D"), 2, 1),
    "T_rend":    (("nw", "nz", "nu", "nx"), 2, 1),
}


def experiment(F, cov, p, R, n, rng, live):
    idx = rng.integers(0, int(F["n"]), size=(R, n))
    Y = (rng.random((R, n)) < p[idx]).astype(np.float64)
    k = int(F["k"])
    out = {s: [0, 0.0] for s in live}
    for r in range(R):
        ii = idx[r]
        for s in live:
            names, hi, lo = STATS[s]
            c, rej = lrt([cov[nm][ii] for nm in names], Y[r], k, hi, lo)
            out[s][0] += rej
            out[s][1] += c
    return {s: (v[0] / R, v[1] / R) for s, v in out.items()}, float(Y.mean())


def bind_tiebreak(F, eps, delta, R, ns, rng):
    """The one place the two executors are NOT the same map: the DECOUPLED bind component.

    Its retrieval reads a cell written ``w`` times, so an interference null fires on it; a
    composition deficit, which fires only on a reference rendered "at this point", cannot —
    the arm renders none. The arm is therefore the only tie-break, and this is the sample size
    it needs: a two-proportion test of bind-arm accuracy against the composition executor's.
    """
    w = F["bind_w"]
    p_null = (1 - eps) ** 2 * (1 - delta * (w >= 1))
    p_alt = np.full(w.shape, (1 - eps) ** 2)
    out = {}
    for n in ns:
        idx = rng.integers(0, len(w), size=(R, n))
        a0 = (rng.random((R, n)) < p_null[idx]).mean(1)
        a1 = (rng.random((R, n)) < p_alt[idx]).mean(1)
        se = np.sqrt(np.clip(a0 * (1 - a0) + a1 * (1 - a1), 1e-12, None) / n)
        out[n] = float(np.mean((a1 - a0) / se > 1.645))
    return out, float(p_null.mean()), float(p_alt.mean())


# ---------------------------------------------------------------------------
# Every executor is scaled to cost the SAME accuracy, so a rejection cannot be a reading of how
# hard the executor is. ``base`` fixes the uniform slip rate; each deficit is then dialled until
# the composed cell drops by ``drop``.
BASE_ACC = 0.90
CONFIGS = [
    ("N1 uniform slip",              "eps",    False),
    ("N2 read slip ~ write count",   "wlin",   False),
    ("N3 read slip ~ distance",      "dlin",   False),
    ("N4 stale on overwritten cell", "wjump",  False),
    ("A1 stated-map fallback",       "gamma",  True),
    ("A2 garbled cross reference",   "garble", True),
]
DROPS = (0.10, 0.20, 0.30)


def amplification(F, eps):
    """How much more sensitive the composed arm is to the interference rate than the component
    that could calibrate it.

    The composed arm's answer depends on ~nx cross-structure reads, the decoupled bind
    component's on one retrieval, so a unit of interference costs the two arms very different
    amounts of accuracy. Calibrating the composed arm's excess therefore needs the component
    measured to a precision the composed arm itself sets, and that ratio is a sample-size
    multiplier of roughly the squared amplification.
    """
    w = F["bind_w"]
    rows = []
    for d in (0.005, 0.01, 0.02):
        comp = 1.0 - p_correct(F, eps=eps, wjump=d).mean() / p_correct(F, eps=eps).mean()
        bind = float(np.mean(d * (w >= 1)))
        rows.append((d, comp, bind, comp / max(bind, 1e-9)))
    return rows


def report(R, cells, reps, seed=7):
    rng = np.random.default_rng(seed)
    res = {}
    for cell in cells:
        F = load(cell)
        cov = covariates(F)
        k, L, m = int(F["k"]), int(F["L"]), CELLS[cell][3]
        items = cell_items(cell)
        live = [s for s in STATS if all(cov[nm].max() > 0 for nm in STATS[s][0])]
        print(f"\n=== {cell}  (k={k}, L={L}, {int(F['n'])} items) " + "=" * 34, flush=True)

        # ---- 1. support -----------------------------------------------------
        dyn, st = F["kind"] == RES_DYN, F["kind"] == RES_ST
        sl = F["qgar"] > 0
        print("\n[support] resolution ops on the answer's dependency slice (qgar>0):")
        tot = float(np.sum((dyn | st) & sl))
        print(f"   rendered 'at the start' : {float(np.sum(st & sl))/tot:.3f}"
              f"   rendered 'at this point': {float(np.sum(dyn & sl))/tot:.3f}")
        for lo, hi, nm in ((0, 0, "w=0"), (1, 1, "w=1"), (2, 2, "w=2"), (3, 999, "w>=3")):
            sel = dyn & sl & (F["w"] >= lo) & (F["w"] <= hi)
            nsel = float(np.sum(sel))
            mv = float(np.sum(sel & (F["moved"] == 1)))
            print(f"     dyn {nm:5s}: {nsel/tot:.3f} of resolutions, "
                  f"P(needs running state) = {mv/max(nsel,1):.3f}")
        print(f"   per-item slice: writes {cov['nw'].mean():.1f}, z {cov['nz'].mean():.2f}, "
              f"u {cov['nu'].mean():.2f}, x {cov['nx'].mean():.1f}, W {cov['W'].mean():.1f}; "
              f"corr(x,W)={np.corrcoef(cov['nx'], cov['W'])[0,1]:.3f}", flush=True)

        # ---- 2. the identity ------------------------------------------------
        print("\n[identity] stated-fallback composition deficit vs interference null, per item,"
              " under common random numbers:")
        sub = items[:200]
        for g in (0.005, 0.01, 0.02):
            pc = noisy_acc(sub, k, m, 2000, np.random.default_rng(99), eps=0.002, gamma=g)
            ph = noisy_acc(sub, k, m, 2000, np.random.default_rng(99), eps=0.002, wjump=g)
            ac = p_correct(F, eps=0.002, gamma=g)
            ah = p_correct(F, eps=0.002, wjump=g)
            print(f"   gamma=delta={g:<6}: replay acc {pc.mean():.4f} / {ph.mean():.4f} "
                  f"(max|dP| {np.abs(pc - ph).max():.2e}, at the 2000-replay noise floor "
                  f"{np.sqrt(pc.mean()*(1-pc.mean())/2000)*3:.2e}); the two fire on the same op "
                  f"set with the same wrong value, so algebraically max|dP| = "
                  f"{np.abs(ac - ah).max():.1e}", flush=True)

        # ---- 3. the statistics ---------------------------------------------
        print(f"\n[tests] R={R}, alpha=0.05 one-sided LRT; N* must not fire, A* must")
        print(f"   {'executor':32s} {'acc':>6s} {'n':>5s} "
              + " ".join(f"{s:>10s}" for s in live), flush=True)
        cellres = {}
        eps0 = solve_noisy(items, k, m, "eps", 0.0, BASE_ACC, rng)
        knobs = {}
        for label, knob, is_alt in CONFIGS:
            for drop in DROPS:
                if knob == "eps":
                    kw = dict(eps=solve_noisy(items, k, m, "eps", 0.0, BASE_ACC - drop, rng))
                else:
                    kw = {"eps": eps0,
                          knob: solve_noisy(items, k, m, knob, eps0, BASE_ACC - drop, rng)}
                    knobs[(knob, drop)] = kw[knob]
                p = noisy_acc(items, k, m, reps, rng, **kw)
                for n in (100, 200, 500):
                    out, _ = experiment(F, cov, p, R, n, rng, live)
                    cellres[f"{label}|{drop}|{n}"] = out
                    tag = f"{label} -{drop:.2f}" if n == 100 else ""
                    print(f"   {tag:32s} {p.mean():6.3f} {n:5d} "
                          + " ".join(f"{out[s][0]:10.3f}" for s in live), flush=True)

        print("   the interference null and the composition deficit calibrate to the same "
              "knob value at every cost:")
        for drop in DROPS:
            print(f"     -{drop:.2f}: delta={knobs[('wjump', drop)]:.5f}  "
                  f"gamma={knobs[('gamma', drop)]:.5f}", flush=True)

        # ---- 4. the only tie-break ------------------------------------------
        print("\n[tie-break] the DECOUPLED bind component is the only arm the two differ on.")
        for d, comp, bind, amp in amplification(F, eps0):
            print(f"   delta={d:<6} accuracy cost: composed {comp:.3f}, bind component "
                  f"{bind:.4f} -> amplification {amp:5.1f}x  (calibrating it to the precision "
                  f"the composed arm resolves needs ~{amp**2:.0f}x the items)")
        print("   power of a two-proportion test on the bind component alone:")
        for d in (0.005, 0.01, 0.02):
            pw, a0, a1 = bind_tiebreak(F, eps0, d, R, (100, 200, 500, 2000, 10000), rng)
            print(f"   delta={d:<6} bind acc {a0:.4f} (null) vs {a1:.4f} (composition); "
                  + " ".join(f"n={n}:{v:.2f}" for n, v in pw.items()), flush=True)
        res[cell] = cellres
    with open(os.path.join(CACHE, "s5bind_composition_statistic.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


# ---------------------------------------------------------------------------
# What a construct would need, measured on a synthetic op stream.
#
# The defect above is in the ABLATION, not in the composition. s5_bind's uncoupled control is
# "resolve the same reference against the INITIAL structure", so
#     (the reference witnesses composition)
#         <=> (the coupled and uncoupled resolutions differ)
#         <=> (the referenced cell has been written since the start),
# and the right-hand side is a read-history predicate. The fix is to move the ablation from the
# TIME INDEX to the SOURCE STRUCTURE: keep every reference dynamic and vary only which structure
# it reads. A swap (which updates P) may name its operand either
#     CROSS  "... and the agent who holds o2 at this point"          -> needs B: composition
#     SAME   "... and the agent whose role at this point is r5"      -> needs only P
# and a give (which updates B) has the mirror pair. Both are live reads of overwritten cells at
# matched write counts and matched retrieval distances, so an interference effect is common to
# the two classes and cancels in theta_x - theta_z, while a solver that cannot hold the other
# structure fails only on CROSS.
#
# This probe builds that stream directly at the op level — no renderer, no sampler — and runs the
# same statistic on it, to check the dissociation is real before anyone builds it.
# ---------------------------------------------------------------------------
def design_stream(k, m, L, rng, p_cross=0.5):
    """One item of the SOURCE-STRUCTURE ablation, in the shape ``noisy_acc`` consumes.

    Every reference is dynamic; the ablation moves the SOURCE, not the time index. A swap
    (which updates P) reads either the holder map (CROSS: composition) or the role map's own
    inverse (SAME: one structure); a give (which updates B) has the mirror pair. Both classes
    are live reads of overwritten cells at matched write counts, so an interference effect is
    common to them.
    """
    P0 = list(range(k))
    rng.shuffle(P0)
    P0inv = [0] * k
    for a, r in enumerate(P0):
        P0inv[r] = a
    B0 = [rng.randrange(k) for _ in range(m)]
    evs = []
    for _ in range(L):
        sw = rng.random() < 0.5
        cross = rng.random() < p_cross
        reads_B = cross if sw else not cross
        ref = rng.randrange(m) if reads_B else rng.randrange(k)
        evs.append((0 if sw else 1, rng.randrange(k) if sw else rng.randrange(m),
                    ref, True, reads_B, cross))
    P, Pi, B = list(P0), list(P0inv), list(B0)
    touch = [0] * k
    for ev in evs:
        kind, u, v, _, reads_B, _c = ev
        x = B[v] if reads_B else Pi[v]
        if kind == 0:
            ra, rx = P[u], P[x]
            if ra != rx:
                P[u], P[x] = rx, ra
                Pi[rx], Pi[ra] = u, x
                touch[u] += 1
                touch[x] += 1
        else:
            B[u] = x
    cand = [a for a in range(k) if touch[a] >= 2 and P[a] != P0[a]]
    if not cand:
        return None
    return (P0, B0, evs, rng.choice(cand))


def design_covariates(items, k, m, draws, rng):
    """The same dependency-slice covariates for a source-structure item, by the same
    one-at-a-time perturbation."""
    n = len(items)
    IDX, KIND, WCOL, QG, QS, CROSS = [], [], [], [], [], []
    for i, (P0, B0, evs, qa) in enumerate(items):
        P0inv = [0] * k
        for a, r in enumerate(P0):
            P0inv[r] = a
        P, Pi, B = list(P0), list(P0inv), list(B0)
        snaps, xs, ws, stale = [], [], [], []
        ogive, orole = [0] * m, [0] * k
        for ev in evs:
            snaps.append((tuple(P), tuple(Pi), tuple(B)))
            kind, u, v, _, reads_B, _c = ev
            x = B[v] if reads_B else Pi[v]
            xs.append(x)
            ws.append(ogive[v] if reads_B else orole[v])
            stale.append(B0[v] if reads_B else P0inv[v])
            if kind == 0:
                ra, rx = P[u], P[x]
                if ra != rx:
                    orole[ra] += 1
                    orole[rx] += 1
                    P[u], P[x] = rx, ra
                    Pi[rx], Pi[ra] = u, x
            else:
                ogive[u] += 1
                B[u] = x
        gold = P[qa]
        for j, ev in enumerate(evs):
            Ps, Pis, Bs = snaps[j]
            xt = xs[j]
            hit = 0
            for _ in range(draws):
                xp = rng.randrange(k - 1)
                if xp >= xt:
                    xp += 1
                Pc, Pic, Bc = list(Ps), list(Pis), list(Bs)
                _apply_gen(ev, Pc, Pic, Bc, xp)
                _replay_gen_from(Pc, Pic, Bc, evs, j + 1, P0inv, B0)
                hit += (Pc[qa] != gold)
            if stale[j] != xt:
                Pc, Pic, Bc = list(Ps), list(Pis), list(Bs)
                _apply_gen(ev, Pc, Pic, Bc, stale[j])
                _replay_gen_from(Pc, Pic, Bc, evs, j + 1, P0inv, B0)
                qs = float(Pc[qa] != gold)
            else:
                qs = 0.0
            KIND.append(RES_DYN); WCOL.append(ws[j]); CROSS.append(int(ev[5]))
            QG.append(hit / draws); QS.append(qs); IDX.append(i)
            Pc, Pic, Bc = list(Ps), list(Pis), list(Bs)
            _replay_gen_from(Pc, Pic, Bc, evs, j + 1, P0inv, B0)
            KIND.append(WRITE); WCOL.append(0); CROSS.append(0)
            QG.append(float(Pc[qa] != gold)); QS.append(0.0); IDX.append(i)
        KIND.append(READOUT); WCOL.append(0); CROSS.append(0)
        QG.append(1.0); QS.append(float(P0[qa] != gold)); IDX.append(i)
    F = dict(kind=np.array(KIND, np.int8), w=np.array(WCOL, np.float64),
             qgar=np.array(QG), qsta=np.array(QS), item=np.array(IDX, np.int32),
             n=np.int32(n), k=np.int32(k))
    kind, cross, w, qg = F["kind"], np.array(CROSS) == 1, F["w"], F["qgar"]
    dyn = kind == RES_DYN

    def agg(mask, weight=None):
        v = qg * mask if weight is None else qg * mask * weight
        return np.bincount(F["item"], weights=v, minlength=n)

    cov = dict(nw=agg((kind == WRITE) | (kind == READOUT)),
               nzu=agg(dyn & ~cross), nx=agg(dyn & cross),
               W=agg(dyn & cross, w), W2=agg(dyn & cross, w * w),
               D=agg(dyn & cross, w))
    return F, cov


def _apply_gen(ev, P, Pi, B, x):
    kind, u = ev[0], ev[1]
    if kind == 0:
        ra, rx = P[u], P[x]
        if ra != rx:
            P[u], P[x] = rx, ra
            Pi[rx], Pi[ra] = u, x
    else:
        B[u] = x


def _replay_gen_from(P, Pi, B, evs, j, P0inv, B0):
    for t in range(j, len(evs)):
        ev = evs[t]
        reads_B, dyn = ev[4], ev[3]
        x = (B[ev[2]] if dyn else B0[ev[2]]) if reads_B else (Pi[ev[2]] if dyn else P0inv[ev[2]])
        _apply_gen(ev, P, Pi, B, x)


def design_probe(k, m, L, n, R, reps, rng, seed=11):
    import random
    r = random.Random(seed)
    items = []
    while len(items) < n:
        it = design_stream(k, m, L, r)
        if it is not None:
            items.append(it)
    F, cov = design_covariates(items, k, m, 4, r)
    live = [s for s in STATS if s != "T_rend"]
    lo, hi = 0.0, 0.5
    for _ in range(14):
        mid = .5 * (lo + hi)
        lo, hi = ((mid, hi) if noisy_acc(items[:600], k, m, 192, rng, eps=mid).mean() > BASE_ACC
                  else (lo, mid))
    eps0 = .5 * (lo + hi)
    print(f"\n=== design probe: source-structure ablation (k={k}, m={m}, L={L}, {n} items) ===")
    print(f"   per-item slice: writes {cov['nw'].mean():.1f}, same-structure "
          f"{cov['nzu'].mean():.1f}, cross-structure {cov['nx'].mean():.1f}; "
          f"uniform slip {eps0:.5f}/op -> {BASE_ACC:.2f}", flush=True)
    print(f"   {'executor':32s} {'acc':>6s} {'n':>5s} " + " ".join(f"{s:>10s}" for s in live))
    for label, knob in (("N4 stale on overwritten cell", "wjump"),
                        ("A1 cross-structure fallback", "gamma")):
        for drop in DROPS:
            lo, hi = 0.0, 0.5
            for _ in range(14):
                mid = .5 * (lo + hi)
                a = noisy_acc(items[:600], k, m, 192, rng, **{"eps": eps0, knob: mid}).mean()
                lo, hi = (mid, hi) if a > BASE_ACC - drop else (lo, mid)
            p = noisy_acc(items, k, m, reps, rng, **{"eps": eps0, knob: .5 * (lo + hi)})
            for nn in (100, 200, 500):
                out, _ = experiment(F, cov, p, R, nn, rng, live)
                tag = f"{label} -{drop:.2f}" if nn == 100 else ""
                print(f"   {tag:32s} {p.mean():6.3f} {nn:5d} "
                      + " ".join(f"{out[s][0]:10.3f}" for s in live), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="report", choices=("features", "report", "design"))
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=800)
    ap.add_argument("--cells", default="frontier,local,ladder50")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if a.stage == "features":
        for c in a.cells.split(","):
            build(c, a.n, a.draws if CELLS[c][1] > 100 else max(a.draws, 4))
    elif a.stage == "design":
        rng = np.random.default_rng(3)
        for k, m, L in ((6, 6, 64), (12, 12, 192)):
            design_probe(k, m, L, 3000, a.R, a.reps, rng)
    else:
        report(a.R, a.cells.split(","), a.reps)
