"""theta_cross - theta_same on s5_bind_v3: its null under every composition-free solver, and
its power against a real composition deficit.

The statistic is registered in ``factworld.composition``; this script is the evidence that it
identifies. Everything runs on the GENERATED cells, parsed back off the rendered prompt by this
file's own parser and simulated by its own replay — no sampler internals, no meta.

  FEATURES   the answer's dependency slice, one record per op, with the op's class (CROSS reads
             the structure it does not write, SAME reads the one it does), the write count and
             retrieval distance of the cell it read, and its measured answer sensitivity.
  SOLVERS    seven composition-free executors and two composition deficits, each dialled to the
             SAME accuracy cost, so a rejection can never be a reading of how hard the executor
             is. Accuracies come from a direct vectorised noisy replay, not from the linearised
             model; only the covariates are one-at-a-time sensitivities on the clean trajectory.
  TESTS      the contrast and its three read-history repairs, at R resamples, n = 100/200/500,
             at the frontier cell and the local cell.

Nothing here trains or calls an API.

Run:  .venv-train/bin/python scripts/probe_s5bind_v3_statistic_20260731.py [--n 1200] [--R 1000]
"""
import argparse
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np                                                   # noqa: E402

from factworld.tasks import CANONICAL, generate                      # noqa: E402

CELLS = {"frontier": ("s5_bind_v3", 192), "local": ("s5_bind_local_v3", 64)}

SWAP, GIVE = 0, 1
SRC_P, SRC_B = 0, 1

RE_P0 = re.compile(r"(g\d+) points to (g\d+) at the start\.")
RE_B0 = re.compile(r"(o\d+) belongs to (g\d+) at the start\.")
RE_SW_B = re.compile(r"^swaps the pointers of (g\d+) and the agent (o\d+) belongs to at this point\.$")
RE_SW_P = re.compile(r"^swaps the pointers of (g\d+) and the agent (g\d+) points to at this point\.$")
RE_GV_P = re.compile(r"^gives (o\d+) to the agent (g\d+) points to at this point\.$")
RE_GV_B = re.compile(r"^gives (o\d+) to the agent (o\d+) belongs to at this point\.$")
RE_Q = re.compile(r"which agent does (g\d+) point to at the end\?")


def parse(prompt):
    """Rendered surface -> (P0, B0, events, queried agent) as small ints. Shares no code with
    the sampler. ``events`` are ``(kind, target, ref, src, cross, clause)`` where ``clause`` is
    the surface form used (0 = "points to", 1 = "belongs to") — the format confound the design
    anti-correlates with the class."""
    p0 = RE_P0.findall(prompt)
    b0 = RE_B0.findall(prompt)
    agents = sorted({g for g, _ in p0}, key=lambda s: int(s[1:]))
    objs = sorted({o for o, _ in b0}, key=lambda s: int(s[1:]))
    ai = {a: i for i, a in enumerate(agents)}
    oi = {o: i for i, o in enumerate(objs)}
    P0 = [0] * len(agents)
    for g, h in p0:
        P0[ai[g]] = ai[h]
    B0 = [0] * len(objs)
    for o, h in b0:
        B0[oi[o]] = ai[h]
    body = RE_Q.sub("", prompt[prompt.index(" s0 ") + 1:]).strip()
    parts = re.split(r"\bs(\d+) ", body)
    evs = []
    for i in range(1, len(parts) - 1, 2):
        t = parts[i + 1].strip()
        m = RE_SW_B.match(t)
        if m:
            evs.append((SWAP, ai[m.group(1)], oi[m.group(2)], SRC_B, 1, 1))
            continue
        m = RE_SW_P.match(t)
        if m:
            evs.append((SWAP, ai[m.group(1)], ai[m.group(2)], SRC_P, 0, 0))
            continue
        m = RE_GV_P.match(t)
        if m:
            evs.append((GIVE, oi[m.group(1)], ai[m.group(2)], SRC_P, 1, 0))
            continue
        m = RE_GV_B.match(t)
        if m:
            evs.append((GIVE, oi[m.group(1)], oi[m.group(2)], SRC_B, 0, 1))
            continue
        raise AssertionError(t)
    return P0, B0, evs, ai[RE_Q.search(prompt).group(1)]


def _apply(ev, P, B, x):
    if ev[0] == SWAP:
        P[ev[1]], P[x] = P[x], P[ev[1]]
    else:
        B[ev[1]] = x


def _replay_from(P, B, evs, j, P0, B0):
    for t in range(j, len(evs)):
        ev = evs[t]
        x = P[ev[2]] if ev[3] == SRC_P else B[ev[2]]
        _apply(ev, P, B, x)


def features(items, k, m, draws, rng):
    """Per-op covariates for every item: class, write count, retrieval distance, clause form and
    measured answer sensitivity, aggregated to the per-item columns the fit consumes."""
    n = len(items)
    cols = {c: np.zeros(n) for c in
            ("nw", "nz", "nx", "bz", "bx", "W", "W2", "D", "J")}
    per_kind = {SWAP: [np.zeros(n), np.zeros(n)], GIVE: [np.zeros(n), np.zeros(n)]}
    stats = {"cx": 0, "cz": 0, "wx": 0.0, "wz": 0.0, "dx": 0.0, "dz": 0.0,
             "sx": 0.0, "sz": 0.0}
    for i, (P0, B0, evs, qa) in enumerate(items):
        P, B = list(P0), list(B0)
        snaps, xs, ws, ds = [], [], [], []
        wcnt = {}
        last = {}
        for j, ev in enumerate(evs):
            snaps.append((list(P), list(B)))
            cell = (ev[3], ev[2])
            ws.append(wcnt.get(cell, 0))
            ds.append(j - last.get(cell, -1))
            x = P[ev[2]] if ev[3] == SRC_P else B[ev[2]]
            xs.append(x)
            if ev[0] == SWAP:
                P[ev[1]], P[x] = P[x], P[ev[1]]
                for g in (ev[1], x):
                    wcnt[(SRC_P, g)] = wcnt.get((SRC_P, g), 0) + 1
                    last[(SRC_P, g)] = j
            else:
                B[ev[1]] = x
                wcnt[(SRC_B, ev[1])] = wcnt.get((SRC_B, ev[1]), 0) + 1
                last[(SRC_B, ev[1])] = j
        gold = P[qa]
        L = len(evs)
        for j, ev in enumerate(evs):
            Ps, Bs = snaps[j]
            hit = 0
            for _ in range(draws):
                alt = int(rng.integers(0, k - 1))
                alt += alt >= xs[j]
                Pc, Bc = list(Ps), list(Bs)
                _apply(ev, Pc, Bc, alt)
                _replay_from(Pc, Bc, evs, j + 1, P0, B0)
                hit += int(Pc[qa] != gold)
            sens = hit / draws
            Pc, Bc = list(Ps), list(Bs)
            _replay_from(Pc, Bc, evs, j + 1, P0, B0)
            cols["nw"][i] += float(Pc[qa] != gold)              # the WRITE this event performs
            key = "nx" if ev[4] else "nz"
            cols[key][i] += sens
            cols["J"][i] += sens * (j / L)
            per_kind[ev[0]][ev[4]][i] += sens
            if ev[4]:
                cols["W"][i] += sens * ws[j]
                cols["W2"][i] += sens * ws[j] ** 2
                cols["D"][i] += sens * ds[j]
                stats["cx"] += 1
                stats["wx"] += ws[j]
                stats["dx"] += ds[j]
                stats["sx"] += sens
            else:
                stats["cz"] += 1
                stats["wz"] += ws[j]
                stats["dz"] += ds[j]
                stats["sz"] += sens
        cols["nw"][i] += 1.0
    # THE KIND-BALANCED class columns: each op divided by its event kind's mean CROSS slice
    # mass, so swaps and gives contribute equally and a surface-clause failure — which is CROSS
    # on a swap and SAME on a give — enters the two class columns at equal size and cancels.
    g = {kd: (n / per_kind[kd][1].sum() if per_kind[kd][1].sum() > 0 else 0.0)
         for kd in (SWAP, GIVE)}
    for kd in (SWAP, GIVE):
        cols["bz"] += g[kd] * per_kind[kd][0]
        cols["bx"] += g[kd] * per_kind[kd][1]
    stats["gswap"], stats["ggive"] = g[SWAP], g[GIVE]
    return cols, stats


def noisy_acc(items, k, m, reps, rng, eps=0.0, wlin=0.0, dlin=0.0, jlin=0.0, wjump=0.0,
              cap=0.0, fmt=0.0, gamma=0.0, garble=0.0, capsize=None):
    """P(correct) per item under one executor, by direct replay of ``reps`` noisy trajectories.

    Composition-free: ``eps`` uniform per-op slip; ``wlin`` read slip linear in the read cell's
    write count; ``dlin`` linear in the retrieval distance; ``jlin`` linear in the op's depth in
    the stream (a chain-length slip); ``wjump`` an overwritten cell returns its STALE value;
    ``cap`` the probability that a read goes through a BOUNDED WORKING SET of ``capsize`` cells
    (default k // 2) under LRU-by-last-write eviction, a miss returning the stated value — the
    SIZE is integer-valued and far too coarse to dial to a matched accuracy cost (at k = m = 6
    it jumps from 0.90 to 0.28 between 12 slots and 10), so the size is fixed at the bound the
    solver is named for and the continuous knob is how often the bound bites;
    ``fmt`` an extra slip on one SURFACE CLAUSE ("belongs to"),
    which spans half of each class because the clause-to-class map flips between event kinds.
    Composition deficits: ``gamma`` a CROSS reference resolves against the STATED map;
    ``garble`` a CROSS reference resolves to a wrong agent outright.
    """
    out = np.empty(len(items))
    ar = np.arange(reps)
    for it, (P0, B0, evs, qa) in enumerate(items):
        P0a, B0a = np.array(P0), np.array(B0)
        P = np.tile(P0a, (reps, 1))
        B = np.tile(B0a, (reps, 1))
        ow = np.zeros((reps, k + m), np.int64)                # write counts, P cells then B
        lw = np.full((reps, k + m), -1, np.int64)             # last-write index
        L = len(evs)
        for j, ev in enumerate(evs):
            kind, tgt, ref, src, cross, clause = ev
            base = 0 if src == SRC_P else k
            if src == SRC_P:
                x = P[:, ref].copy()
                stale = np.full(reps, P0a[ref])
            else:
                x = B[:, ref].copy()
                stale = np.full(reps, B0a[ref])
            w, last = ow[:, base + ref], lw[:, base + ref]
            r = wjump * (w >= 1) + gamma * cross
            if cap:
                # in the working set iff among the ``capsize`` most recently written cells
                size = capsize if capsize is not None else max(1, k // 2)
                rank = (lw > last[:, None]).sum(1)
                r = r + (rank >= size) * cap
            if np.any(r):
                x = np.where(rng.random(reps) < np.clip(r, 0, 1), stale, x)
            e = eps + wlin * w + dlin * (j - last) + jlin * (j / L) \
                + garble * cross + fmt * clause
            bad = rng.random(reps) < e
            if bad.any():
                alt = rng.integers(0, k - 1, reps)
                x = np.where(bad, alt + (alt >= x), x)
            drop = rng.random(reps) < eps
            if kind == SWAP:
                pa, px = P[ar, tgt], P[ar, x]
                go = ~drop
                P[go, tgt] = px[go]
                P[go, x[go]] = pa[go]
                for g in (tgt,):
                    ow[:, g] += 1
                    lw[:, g] = j
                ow[ar, x] += 1
                lw[ar, x] = j
            else:
                B[~drop, tgt] = x[~drop]
                ow[:, k + tgt] += 1
                lw[:, k + tgt] = j
        Pc, Bc = list(P0), list(B0)
        _replay_from(Pc, Bc, evs, 0, P0, B0)
        gold = Pc[qa]
        ans = P[:, qa].copy()
        bad = rng.random(reps) < eps
        if bad.any():
            alt = rng.integers(0, k - 1, reps)
            ans = np.where(bad, alt + (alt >= ans), ans)
        out[it] = float(np.mean(ans == gold))
    return out


# --- the fit ------------------------------------------------------------------------------
def _ll(X, th, y, ridge):
    q = np.exp(-np.clip(X @ th, -30, 30))
    p = np.clip(q, 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))) - ridge * float(th @ th), p


def _fit(X, y, ridge=1e-6, iters=60):
    mcol = X.mean(0)
    ybar = min(max(y.mean(), 1e-4), 1 - 1e-4)
    denom = float(mcol @ mcol)
    th = (-math.log(ybar) / denom) * mcol if denom > 0 else np.zeros(X.shape[1])
    ll, p = _ll(X, th, y, ridge)
    for _ in range(iters):
        u = y / p - (1 - y) / (1 - p)
        d = -p
        grad = X.T @ (u * d) - 2 * ridge * th
        h = -(y / p**2 + (1 - y) / (1 - p)**2) * d * d - u * d
        H = (X * h[:, None]).T @ X - (2 * ridge + 1e-7) * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        t, improved = 1.0, False
        for _bt in range(30):
            cand = th - t * step
            llc, pc = _ll(X, cand, y, ridge)
            if llc > ll + 1e-12:
                th, ll, p, improved = cand, llc, pc, True
                break
            t *= 0.5
        if not improved or np.max(np.abs(t * step)) < 1e-10:
            break
    return th, ll


CHI2 = 2.70554


def lrt(cols, y, i_hi, i_lo):
    X = np.column_stack(cols)
    th, ll1 = _fit(X, y)
    Xn = X.copy()
    Xn[:, i_lo] += Xn[:, i_hi]
    Xn = np.delete(Xn, i_hi, axis=1)
    _t, ll0 = _fit(Xn, y)
    c = th[i_hi] - th[i_lo]
    return c, bool(c > 0 and 2 * (ll1 - ll0) > CHI2)


STATS = {
    "T_kind":    (("nw", "bz", "bx"), 2, 1),          # THE PRIMARY: kind-balanced
    "T_kindW":   (("nw", "bz", "bx", "W"), 2, 1),
    "T_kindWD":  (("nw", "bz", "bx", "W", "D"), 2, 1),
    "T_cross":   (("nw", "nz", "nx"), 2, 1),          # raw, kept as the diagnostic
    "T_crossW":  (("nw", "nz", "nx", "W"), 2, 1),
    "T_crossWD": (("nw", "nz", "nx", "W", "D"), 2, 1),
}

BASE_ACC = 0.90
DROPS = (0.10, 0.20, 0.30)
CONFIGS = [
    ("N1 uniform slip",              "eps",   False),
    ("N2 read slip ~ write count",   "wlin",  False),
    ("N3 read slip ~ distance",      "dlin",  False),
    ("N4 read slip ~ stream depth",  "jlin",  False),
    ("N5 stale on overwritten cell", "wjump", False),
    ("N6 bounded capacity + LRU",    "cap",   False),
    ("N7 surface-clause slip",       "fmt",   False),
    ("A1 stated-map fallback",       "gamma", True),
    ("A2 garbled cross reference",   "garble", True),
]


def solve(items, k, m, knob, eps, target, rng, reps=160, sub=400, hi=0.5):
    lo = 0.0
    it = items[:sub]
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        kw = {"eps": mid} if knob == "eps" else {"eps": eps, knob: mid}
        if noisy_acc(it, k, m, reps, rng, **kw).mean() > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def experiment(cols, p, R, n, rng, live, k):
    idx = rng.integers(0, len(p), size=(R, n))
    Y = (rng.random((R, n)) < p[idx]).astype(np.float64)
    out = {s: [0, 0.0] for s in live}
    for r in range(R):
        ii = idx[r]
        for s in live:
            names, hi, lo = STATS[s]
            c, rej = lrt([cols[nm][ii] for nm in names], Y[r], hi, lo)
            out[s][0] += rej
            out[s][1] += c
    return {s: (v[0] / R, v[1] / R) for s, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--cells", default="local,frontier")
    ap.add_argument("--stats", default="T_kind,T_kindWD,T_cross,T_crossWD",
                    help="which registered statistics to run (default: the primary, its "
                         "read-history repair, and the raw contrast with its repair)")
    a = ap.parse_args()
    rng = np.random.default_rng(20260731)
    live = a.stats.split(",")
    for cell in a.cells.split(","):
        name, L = CELLS[cell]
        spec = CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        ex = generate(spec, "test", n=a.n, length=L)
        items = [parse(e.prompt) for e in ex]
        cols, st = features(items, k, m, a.draws, rng)
        print(f"\n=== {cell}: {name} @ L={L}  (k={k}, m={m}, {len(items)} items) "
              + "=" * 20, flush=True)
        print(f"[class balance] cross ops {st['cx']} vs same ops {st['cz']} "
              f"({st['cx'] / (st['cx'] + st['cz']):.3f} cross)")
        print(f"[matching] mean write count of the read cell: cross {st['wx'] / st['cx']:.2f} "
              f"same {st['wz'] / st['cz']:.2f}   |   mean retrieval distance: cross "
              f"{st['dx'] / st['cx']:.2f} same {st['dz'] / st['cz']:.2f}")
        print(f"[slice] per item: writes {cols['nw'].mean():.1f}, same {cols['nz'].mean():.1f}, "
              f"cross {cols['nx'].mean():.1f}; mean op sensitivity cross "
              f"{st['sx'] / st['cx']:.3f} same {st['sz'] / st['cz']:.3f}")
        print(f"[kind balance] swap and give slice masses differ by "
              f"{st['ggive'] / st['gswap']:.1f}x, which is why the raw contrast does not cancel "
              f"a surface-clause failure; the balanced columns divide each op by its kind's "
              f"mean cross mass")
        eps0 = solve(items, k, m, "eps", 0.0, BASE_ACC, rng)
        print(f"[calibration] uniform slip {eps0:.5f}/op -> {BASE_ACC:.2f}; every executor below "
              f"is dialled to the SAME accuracy cost")
        print(f"\n   {'executor':32s} {'acc':>6s} {'n':>5s} "
              + " ".join(f"{s:>10s}" for s in live), flush=True)
        for label, knob, _is_alt in CONFIGS:
            for drop in DROPS:
                if knob == "eps":
                    kw = {"eps": solve(items, k, m, "eps", 0.0, BASE_ACC - drop, rng)}
                elif knob == "cap":
                    kw = {"eps": eps0, "cap": solve(items, k, m, "cap", eps0,
                                                    BASE_ACC - drop, rng, hi=1.0)}
                else:
                    kw = {"eps": eps0,
                          knob: solve(items, k, m, knob, eps0, BASE_ACC - drop, rng)}
                p = noisy_acc(items, k, m, a.reps, rng, **kw)
                for nn in (100, 200, 500):
                    out = experiment(cols, p, a.R, nn, rng, live, k)
                    tag = f"{label} -{drop:.2f}" if nn == 100 else ""
                    print(f"   {tag:32s} {p.mean():6.3f} {nn:5d} "
                          + " ".join(f"{out[s][0]:10.3f}" for s in live), flush=True)


if __name__ == "__main__":
    main()
