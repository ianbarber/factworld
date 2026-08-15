"""theta_cross - theta_same on s5_bind_v3: its SIZE under every composition-free executor we can
name, and its power against a real composition deficit.

The statistic is registered in ``factworld.composition``; this script is the evidence that it
identifies. Everything runs on the GENERATED cells, parsed back off the rendered prompt by this
file's own parser and simulated by its own replay — no sampler internals, no meta.

  FEATURES   the answer's dependency slice, one record per op, with the op's class (CROSS reads
             the structure it does not write, SAME reads the one it does), the read cell's write
             count, retrieval distance, derivation depth and distinct-value count, the surface
             clause, and the measured answer sensitivity.
  SOLVERS    the composition-free family — uniform slip; slip linear in the write count, the
             retrieval distance, the stream depth, the derivation depth or the distinct-value
             count; stale-value intrusion; surface-clause slip; kind slip; FIFO and LRU at
             several capacities; and HARD FORGETTING HORIZONS, whose cutoff falls inside a
             distance bin — plus two real deficits. Every one is dialled to the same accuracy
             cost, so a rejection can never be a reading of how hard the executor is.
  TESTS      the registered statistics at R resamples, n = 250/500/2000, at both cells. SIZE IS
             REPORTED AS A NUMBER at every configuration.

The two shipped bugs are reproducible from here with ``--shipped``: ``T_kind_ship`` weights the
class columns by the CROSS mass alone and ``T_kindW_ship`` sums the nuisance over the CROSS ops
alone, which is what the package did before.

Nothing here trains or calls an API.

Run:  .venv-train/bin/python scripts/probe_s5bind_v3_statistic_20260731.py
"""
import argparse
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np                                                   # noqa: E402

from factworld.tasks import CANONICAL, generate                      # noqa: E402

CELLS = {"local": ("s5_bind_local_v3", 64), "frontier": ("s5_bind_v3", 192)}

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


# --- the item tensor, and the read-history features of the clean trajectory ----------------
def to_arrays(items, k, m):
    """One cell's items as flat arrays, plus the per-event read-history of the cell each event
    reads: write count, retrieval distance, DERIVATION DEPTH (how many writes deep the value in
    that cell is), DISTINCT VALUES it has held, and its stated value."""
    n, L = len(items), len(items[0][2])
    A = {c: np.zeros((n, L), np.int32) for c in
         ("KIND", "TGT", "RCOL", "CROSS", "CLAUSE", "STALE", "W", "D", "DEP", "NV")}
    A["P0"] = np.zeros((n, k), np.int16)
    A["B0"] = np.zeros((n, m), np.int16)
    A["QA"] = np.zeros(n, np.int64)
    gold = np.zeros(n, np.int16)
    for i, (P0, B0, evs, qa) in enumerate(items):
        assert len(evs) == L, "ragged cell"
        A["P0"][i], A["B0"][i], A["QA"][i] = P0, B0, qa
        P, B = list(P0), list(B0)
        wcnt, last, dep, seen = {}, {}, {}, {}
        for j, (kind, tgt, ref, src, cross, clause) in enumerate(evs):
            cell = (src, ref)
            A["KIND"][i, j], A["TGT"][i, j] = kind, tgt
            A["RCOL"][i, j] = ref + (k if src == SRC_B else 0)
            A["CROSS"][i, j], A["CLAUSE"][i, j] = cross, clause
            A["STALE"][i, j] = P0[ref] if src == SRC_P else B0[ref]
            A["W"][i, j] = wcnt.get(cell, 0)
            A["D"][i, j] = j - last.get(cell, -1)
            A["DEP"][i, j] = dep.get(cell, 0)
            A["NV"][i, j] = len(seen.get(cell, ()))
            x = P[ref] if src == SRC_P else B[ref]
            if kind == SWAP:
                P[tgt], P[x] = P[x], P[tgt]
                da, dx = dep.get((SRC_P, tgt), 0), dep.get((SRC_P, x), 0)
                for g, dg in ((tgt, dx), (x, da)):
                    c = (SRC_P, g)
                    wcnt[c] = wcnt.get(c, 0) + 1
                    last[c] = j
                    dep[c] = 1 + max(dg, A["DEP"][i, j])
                    seen.setdefault(c, set()).add(P[g])
            else:
                B[tgt] = x
                c = (SRC_B, tgt)
                wcnt[c] = wcnt.get(c, 0) + 1
                last[c] = j
                dep[c] = 1 + A["DEP"][i, j]
                seen.setdefault(c, set()).add(x)
        gold[i] = P[qa]
    A["gold"] = gold
    return A


# --- one vectorised executor over (items x trajectories) -----------------------------------
def simulate(A, k, m, R, rng, knobs, force=None, chunk=None):
    """Final answers, shape (n_items, R), under one executor.

    Composition-free knobs: ``eps`` uniform per-op slip (and per-write loss); ``wlin`` slip
    linear in the read cell's write count; ``dlin`` in its retrieval distance; ``jlin`` in the
    op's depth in the stream; ``plin`` in the derivation depth of the value it reads; ``vlin`` in
    the number of distinct values that cell has held; ``wjump`` an overwritten cell returns its
    STATED value; ``fmt`` an extra slip on the "belongs to" clause, which spans half of each
    class because the clause-to-class map flips between kinds; ``kslip`` an extra slip on gives.
    ``cap``/``policy`` a HARD bounded working set of ``cap`` cells under LRU (by last write) or
    FIFO (by insertion) eviction, a miss returning the stated value. ``horizon`` a HARD
    forgetting horizon: any read of a cell last written more than H events ago returns the stated
    value, and nothing else changes — the executor holds both structures identically.
    Composition deficits: ``gamma`` a CROSS reference resolves against the STATED map; ``garble``
    a CROSS reference resolves to a wrong agent outright.

    ``force`` is ``(mode, targets)`` for the sensitivity slice: trajectory r garbles (mode
    'ref') or drops the write of (mode 'write') exactly event ``targets[r]``, and nothing else
    happens to it.
    """
    n, L = A["KIND"].shape
    eps = knobs.get("eps", 0.0)
    cap, policy = knobs.get("cap", 0), knobs.get("policy", "lru")
    horizon = knobs.get("horizon", 0)
    need_ins = bool(cap) and policy == "fifo"
    fmode, ftgt = force if force is not None else (None, None)
    out = np.empty((n, R), np.int16)
    chunk = chunk or max(1, int(6e6 // (R * (k + m))))
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        c = hi - lo
        M = np.empty((c, R, k + m), np.int16)
        M[:, :, :k] = A["P0"][lo:hi, None, :]
        M[:, :, k:] = A["B0"][lo:hi, None, :]
        OW = np.zeros((c, R, k + m), np.int32)
        LW = np.full((c, R, k + m), -1, np.int32)
        INS = np.full((c, R, k + m), -1, np.int32) if need_ins else None
        for t in range(L):
            kd = A["KIND"][lo:hi, t]
            is_swap = (kd == SWAP)[:, None]
            rc = np.broadcast_to(A["RCOL"][lo:hi, t][:, None, None], (c, R, 1))
            x = np.take_along_axis(M, rc, 2)[:, :, 0].astype(np.int64)
            w = np.take_along_axis(OW, rc, 2)[:, :, 0]
            lastw = np.take_along_axis(LW, rc, 2)[:, :, 0]
            age = t - lastw
            stale = np.broadcast_to(A["STALE"][lo:hi, t][:, None], (c, R))
            cross = A["CROSS"][lo:hi, t][:, None]
            # --- reads that come back with the STATED value instead of the live one
            miss = np.zeros((c, R), bool)
            r = knobs.get("wjump", 0.0) * (w >= 1) + knobs.get("gamma", 0.0) * cross
            if np.any(r):
                miss |= rng.random((c, R)) < np.clip(r, 0, 1)
            if horizon:
                miss |= age > horizon
            if cap:
                key = LW if policy == "lru" else INS
                mine = lastw if policy == "lru" else np.take_along_axis(INS, rc, 2)[:, :, 0]
                miss |= (key > mine[:, :, None]).sum(2) >= cap
            if miss.any():
                x = np.where(miss, stale, x)
            # --- reads that come back a wrong agent
            e = (eps + knobs.get("wlin", 0.0) * w + knobs.get("dlin", 0.0) * age
                 + knobs.get("jlin", 0.0) * (t / L) + knobs.get("plin", 0.0) * A["DEP"][lo:hi, t][:, None]
                 + knobs.get("vlin", 0.0) * A["NV"][lo:hi, t][:, None]
                 + knobs.get("garble", 0.0) * cross
                 + knobs.get("fmt", 0.0) * A["CLAUSE"][lo:hi, t][:, None]
                 + knobs.get("kslip", 0.0) * (kd == GIVE)[:, None])
            bad = rng.random((c, R)) < e if np.any(e) else np.zeros((c, R), bool)
            if fmode == "ref":
                bad = bad | (ftgt[None, :] == t)
            if bad.any():
                alt = rng.integers(0, k - 1, (c, R))
                x = np.where(bad, alt + (alt >= x), x)
            drop = rng.random((c, R)) < eps if eps else np.zeros((c, R), bool)
            if fmode == "write":
                drop = drop | (ftgt[None, :] == t)
            # --- the write: a swap exchanges two P cells, a give overwrites one B cell
            tg = A["TGT"][lo:hi, t]
            c1 = np.broadcast_to(np.where(kd == SWAP, tg, k + tg)[:, None, None], (c, R, 1))
            c2 = np.where(is_swap, x, (k + tg)[:, None])[:, :, None]
            pa = np.take_along_axis(M, c1, 2)[:, :, 0]
            px = np.take_along_axis(M, c2, 2)[:, :, 0]
            v1 = np.where(drop, pa, np.where(is_swap, px, x))
            v2 = np.where(drop, px, np.where(is_swap, pa, x))
            np.put_along_axis(M, c1, v1[:, :, None].astype(np.int16), 2)
            np.put_along_axis(M, c2, v2[:, :, None].astype(np.int16), 2)
            np.put_along_axis(OW, c1, (np.take_along_axis(OW, c1, 2) + 1), 2)
            np.put_along_axis(OW, c2, (np.take_along_axis(OW, c2, 2) + is_swap[:, :, None]), 2)
            np.put_along_axis(LW, c1, np.full((c, R, 1), t, np.int32), 2)
            np.put_along_axis(LW, c2, np.where(is_swap[:, :, None], t,
                                               np.take_along_axis(LW, c2, 2)), 2)
            if need_ins:
                # FIFO: a cell re-enters the working set when it is written while absent, and
                # keeps its insertion time while resident. A cell that has never been written is
                # ABSENT, not resident — its insertion time is -1 and the `cur >= 0` test is what
                # says so; without it nothing is ever inserted, every rank is 0, and the bound
                # never bites at any capacity.
                for cc in (c1, c2):
                    cur = np.take_along_axis(INS, cc, 2)
                    res = ((INS > cur).sum(2, keepdims=True) < cap) & (cur >= 0)
                    np.put_along_axis(INS, cc, np.where(res, cur, t), 2)
        qa = np.broadcast_to(A["QA"][lo:hi, None, None], (c, R, 1))
        ans = np.take_along_axis(M, qa, 2)[:, :, 0].astype(np.int64)
        if eps:
            bad = rng.random((c, R)) < eps
            alt = rng.integers(0, k - 1, (c, R))
            ans = np.where(bad, alt + (alt >= ans), ans)
        out[lo:hi] = ans
    return out


def accuracy(A, k, m, reps, rng, knobs):
    return (simulate(A, k, m, reps, rng, knobs) == A["gold"][:, None]).mean(1)


# --- the answer-sensitivity slice, and the class columns -----------------------------------
def sensitivities(A, k, m, draws, rng):
    """``(ref_sens, write_sens)``: for every event, the measured probability that garbling its
    reference — and, separately, that losing its write — changes the answer."""
    n, L = A["KIND"].shape
    tgt = np.repeat(np.arange(L), draws)
    ans = simulate(A, k, m, L * draws, rng, {}, force=("ref", tgt))
    ref_sens = (ans.reshape(n, L, draws) != A["gold"][:, None, None]).mean(2)
    ans = simulate(A, k, m, L, rng, {}, force=("write", np.arange(L)))
    return ref_sens, (ans != A["gold"][:, None]).astype(float)


def strata(A, n_bins):
    """The stratum of every op: its event kind, crossed with a retrieval-distance bin when
    ``n_bins > 1``. Bin edges are the pooled distance quantiles of this cell."""
    if n_bins > 1:
        edges = np.unique(np.quantile(A["D"], np.linspace(0, 1, n_bins + 1)[1:-1]))
        b = np.digitize(A["D"], edges)
    else:
        b = np.zeros_like(A["D"])
    return A["KIND"] * (b.max() + 1) + b, int(A["KIND"].max() + 1) * int(b.max() + 1)


def columns(A, sens, wsens, n_bins=1):
    """The per-item covariates of the fit.

    THE MODEL HAS ONE MASS COLUMN PER STRATUM AND ONE CONTRAST COLUMN. For stratum s, ``T{s}``
    is the item's total slice mass in s over BOTH classes and ``Dif`` is a fixed combination of
    the per-stratum class DIFFERENCES. The composition statistic is the coefficient on ``Dif``.

    WHY THE MASS COLUMNS ARE RAW AND NOT REWEIGHTED. Any hazard that is a function of the stratum
    alone — a per-kind slip, a distance-dependent slip, a forgetting horizon inside a bin — is
    then EXACTLY in the model's span with a zero coefficient on ``Dif``, so the fit returns zero
    by construction and not by cancellation. Divide the class columns by anything instead and the
    truth leaves the span the moment the divisor varies: the within-kind class MASSES are not
    equal here (a CROSS give's object cannot be referenced again until its pin dies, so its mean
    answer sensitivity is 0.129 against 0.248 for a SAME give), and a reweighting reports that
    imbalance as a coefficient.

    WHY ``Dif`` IS THE PRECISION-WEIGHTED COMBINATION. A surface-clause failure is CROSS on a
    swap and SAME on a give, so it loads on the ANTI-symmetric combination of the per-kind
    differences. Weighting the differences by ``w = Sigma^-1 1`` — Sigma the covariance of the
    difference columns — makes ``Dif`` exactly uncorrelated with every anti-symmetric combination
    (``w' Sigma v = 1' v = 0`` for any ``v`` summing to zero), while a real cross-only deficit
    loads on the symmetric one and survives. Equal weights only do that when the strata carry
    equal variance, and they do not: a swap resolution's mean sensitivity is 0.72 and a give's
    0.13.
    """
    cross = A["CROSS"].astype(bool)
    cols = {"nw": wsens.sum(1) + 1.0,
            "nz": (sens * ~cross).sum(1), "nx": (sens * cross).sum(1),
            "W": (sens * A["W"]).sum(1), "W2": (sens * A["W"] ** 2).sum(1),
            "D": (sens * A["D"]).sum(1)}
    st, ns = strata(A, n_bins)
    T, Dm = [], []
    for s in range(ns):
        sel = st == s
        T.append((sens * sel).sum(1))
        Dm.append((sens * sel * cross).sum(1) - (sens * sel * ~cross).sum(1))
    Dm = np.array(Dm)                                   # (strata, items)
    keep = [i for i in range(ns) if Dm[i].std() > 1e-9]
    Sig = np.cov(Dm[keep]) if len(keep) > 1 else np.array([[float(np.var(Dm[keep[0]]))]])
    Sig = np.atleast_2d(Sig) + 1e-9 * np.eye(len(keep))
    w = np.linalg.solve(Sig, np.ones(len(keep)))
    cols["Dif"] = w @ Dm[keep]
    cols["DifEq"] = Dm[keep].mean(0)
    for s in range(ns):
        cols[f"T{s}"] = T[s]
    cols["nstrata"] = ns
    # the shipped columns, for the before/after: the kind weight read off the CROSS mass alone
    # and applied to both class columns, and the nuisance summed over the CROSS ops alone.
    n = len(cols["nw"])
    mass_x = np.bincount(st.ravel(), weights=(sens * cross).ravel(), minlength=ns)
    gx = np.where(mass_x > 0, n / np.maximum(mass_x, 1e-12), 0.0)
    gwx = sens * gx[st]
    cols["bz_ship"] = (gwx * ~cross).sum(1)
    cols["bx_ship"] = (gwx * cross).sum(1)
    cols["W_ship"] = (sens * A["W"] * cross).sum(1)
    cols["D_ship"] = (sens * A["D"] * cross).sum(1)
    return cols


def matching_report(A, sens):
    """The within-kind class matching, on the exact scored items: the property the balanced
    contrast rests on, reported where the primary reads it rather than pooled."""
    out = {}
    cross = A["CROSS"].astype(bool)
    for kd, label in ((SWAP, "swap"), (GIVE, "give")):
        sel = A["KIND"] == kd
        row = {}
        for cl, mask in (("cross", sel & cross), ("same", sel & ~cross)):
            row[cl] = {"n": int(mask.sum()),
                       "d": float(A["D"][mask].mean()), "w": float(A["W"][mask].mean()),
                       "dep": float(A["DEP"][mask].mean()), "s": float(sens[mask].mean())}
        out[label] = row
    return out


# --- the fit: P(correct) = exp(-X theta), batched over resamples ---------------------------
CHI2 = 2.70554                       # one-sided alpha = 0.05 on a single contrast


def _ll(X, th, y, ridge):
    q = np.exp(-np.clip(np.matmul(X, th[:, :, None])[:, :, 0], -30, 30))
    p = np.clip(q, 1e-9, 1 - 1e-9)
    return (y * np.log(p) + (1 - y) * np.log(1 - p)).sum(1) - ridge * (th * th).sum(1), p


def _fit(X, y, ridge=1e-6, iters=30, th0=None):
    """Damped Newton on every resample at once. theta = 0 is a likelihood boundary, so the fit
    starts by matching the mean accuracy along the column means — the same start, step and
    backtrack as ``factworld.composition.fit``."""
    R, n, d = X.shape
    if th0 is not None:
        th = th0.copy()
    else:
        mcol = X.mean(1)
        ybar = np.clip(y.mean(1), 1e-4, 1 - 1e-4)
        den = (mcol * mcol).sum(1)
        th = np.where(den[:, None] > 0,
                      (-np.log(ybar) / np.maximum(den, 1e-12))[:, None] * mcol, 0.0)
    ll, p = _ll(X, th, y, ridge)
    eye = np.eye(d)[None, :, :]
    for _ in range(iters):
        u = y / p - (1 - y) / (1 - p)
        dp = -p
        grad = np.einsum("rnd,rn->rd", X, u * dp) - 2 * ridge * th
        h = -(y / p ** 2 + (1 - y) / (1 - p) ** 2) * dp * dp - u * dp
        H = np.einsum("rnd,rn,rne->rde", X, h, X) - (2 * ridge + 1e-7) * eye
        try:
            step = np.linalg.solve(H, grad[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break
        step = np.where(np.isfinite(step), step, 0.0)
        t = np.ones(R)
        done = np.zeros(R, bool)
        for _bt in range(12):
            cand = th - t[:, None] * step
            llc, pc = _ll(X, cand, y, ridge)
            take = (~done) & (llc > ll + 1e-12)
            if take.any():
                th[take], ll[take], p[take] = cand[take], llc[take], pc[take]
                done |= take
            if done.all():
                break
            t = np.where(done, t, t * 0.5)
        if not done.any() or np.max(np.abs(t[:, None] * step)) < 1e-10:
            break
    return th, ll


def lrt_drop(X, y, i):
    """One-sided LRT of theta[i] > 0, against the model with that column removed. The null fit
    warm-starts from the full fit, which is where it lands anyway."""
    th, ll1 = _fit(X, y)
    _t, ll0 = _fit(np.delete(X, i, axis=2), y, th0=np.delete(th, i, axis=1))
    c = th[:, i]
    return c, (c > 0) & (2 * (ll1 - ll0) > CHI2)


def lrt_merge(X, y, i_hi, i_lo):
    """One-sided LRT of theta[i_hi] > theta[i_lo] — the two-class form, kept so the before/after
    runs the estimator the package actually ran."""
    th, ll1 = _fit(X, y)
    Xn = X.copy()
    Xn[:, :, i_lo] += Xn[:, :, i_hi]
    c = th[:, i_hi] - th[:, i_lo]
    th0 = np.delete(th, i_hi, axis=1)
    th0[:, i_lo] = 0.5 * (th[:, i_hi] + th[:, i_lo])
    _t, ll0 = _fit(np.delete(Xn, i_hi, axis=2), y, th0=th0)
    return c, (c > 0) & (2 * (ll1 - ll0) > CHI2)


# name -> (fixed columns, contrast column or None, distance bins). ``None`` means the last two
# columns are class columns tested against each other — the shipped two-column form.
STATS = {
    "T_kind":        (("nw",), "Dif", 1),
    "T_strat":       (("nw",), "Dif", 4),
    "T_kindEq":      (("nw",), "DifEq", 1),
    "T_kindWD":      (("nw", "W", "D"), "Dif", 1),
    "T_cross":       (("nw", "nz", "nx"), None, 1),
    "T_kind_ship":   (("nw", "bz_ship", "bx_ship"), None, 1),
    "T_kindW_ship":  (("nw", "W_ship", "bz_ship", "bx_ship"), None, 1),
    "T_kindWD_ship": (("nw", "W_ship", "D_ship", "bz_ship", "bx_ship"), None, 1),
}


def design(cols, s):
    fixed, dif, _nb = STATS[s]
    names = list(fixed)
    if dif is not None:
        names += [f"T{j}" for j in range(cols["nstrata"])] + [dif]
    return names


def experiment(colsets, p, R, n, rng, live):
    idx = rng.integers(0, len(p), size=(R, n))
    Y = (rng.random((R, n)) < p[idx]).astype(np.float64)
    out = {}
    for s in live:
        cols = colsets[STATS[s][2]]
        names = design(cols, s)
        X = np.stack([cols[nm][idx] for nm in names], axis=2)
        c, rej = (lrt_drop(X, Y, len(names) - 1) if STATS[s][1] is not None
                  else lrt_merge(X, Y, len(names) - 1, len(names) - 2))
        out[s] = (float(rej.mean()), float(c.mean()))
    return out


# --- the executor family -------------------------------------------------------------------
BASE_ACC = 0.90
CONT = [
    ("N01 uniform per-op slip",        "eps"),
    ("N02 slip ~ write count",         "wlin"),
    ("N03 slip ~ retrieval distance",  "dlin"),
    ("N04 slip ~ stream depth",        "jlin"),
    ("N05 slip ~ derivation depth",    "plin"),
    ("N06 slip ~ distinct values",     "vlin"),
    ("N07 stale-value intrusion",      "wjump"),
    ("N08 surface-clause slip",        "fmt"),
    ("N09 kind slip (gives)",          "kslip"),
]
DEFICITS = [("A1 stated-map fallback", "gamma"), ("A2 garbled cross reference", "garble")]


def dial(A, k, m, knob, eps, target, rng, reps, sub, hi=1.0, steps=14):
    lo = 0.0
    sub_A = subset(A, sub)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        kw = {"eps": mid} if knob == "eps" else {"eps": eps, knob: mid}
        if accuracy(sub_A, k, m, reps, rng, kw).mean() > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def dial_int(A, k, m, key, values, eps, target, rng, reps, sub, extra=None):
    """Integer knobs (a capacity, a horizon) cannot be dialled to a target. Accuracy rises with
    both, so bisect for the crossing and take the neighbouring value that lands closest; the
    achieved accuracy is printed next to the result rather than assumed."""
    sub_A = subset(A, sub)
    acc = {}

    def at(i):
        if i not in acc:
            kw = dict(extra or {})
            kw.update({"eps": eps, key: values[i]})
            acc[i] = accuracy(sub_A, k, m, reps, rng, kw).mean()
        return acc[i]

    lo, hi = 0, len(values) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if at(mid) < target:
            lo = mid
        else:
            hi = mid
    return values[min((lo, hi), key=lambda i: abs(at(i) - target))]


def subset(A, n):
    out = {kk: (v[:n] if isinstance(v, np.ndarray) and v.shape and v.shape[0] == A["gold"].shape[0]
                else v) for kk, v in A.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="item pool per cell")
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=200, help="trajectories per item for P(correct)")
    ap.add_argument("--dial-reps", type=int, default=96)
    ap.add_argument("--dial-sub", type=int, default=300)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--sizes", default="250,500,2000")
    ap.add_argument("--drops", default="0.10,0.20,0.30")
    ap.add_argument("--cells", default="local,frontier")
    ap.add_argument("--stats", default="T_kind,T_strat,T_cross")
    ap.add_argument("--shipped", action="store_true",
                    help="also run the two shipped bugs (T_kind_ship, T_kindWD_ship)")
    ap.add_argument("--only", default="", help="comma-separated executor code prefixes to run")
    ap.add_argument("--match-reads", type=int, default=-1,
                    help="override TaskSpec.match_reads (0 = the unmatched stream)")
    ap.add_argument("--seed", type=int, default=20260731)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    live = a.stats.split(",")
    if a.shipped:
        live = live + ["T_kind_ship", "T_kindW_ship", "T_kindWD_ship"]
    sizes = [int(s) for s in a.sizes.split(",")]
    drops = [float(s) for s in a.drops.split(",")]

    for cell in a.cells.split(","):
        name, L = CELLS[cell]
        spec = CANONICAL[name]
        if a.match_reads >= 0:
            spec = spec.scaled(match_reads=a.match_reads)
        k, m = spec.k, spec.n_objects_active
        t0 = time.time()
        ex = generate(spec, "test", n=a.n, length=L)
        A = to_arrays([parse(e.prompt) for e in ex], k, m)
        sens, wsens = sensitivities(A, k, m, a.draws, rng)
        colsets = {nb: columns(A, sens, wsens, n_bins=nb)
                   for nb in sorted({STATS[s][2] for s in live})}
        print(f"\n=== {cell}: {name} @ L={L}  (k={k}, m={m}, {a.n} items, "
              f"match_reads={spec.match_reads}, {time.time() - t0:.0f}s) " + "=" * 12, flush=True)
        mr = matching_report(A, sens)
        for kd, row in mr.items():
            x, z = row["cross"], row["same"]
            def gp(p, q):
                return 100 * (p / q - 1) if q else 0.0
            print(f"  [within-kind {kd}] n {x['n']}/{z['n']}  distance {x['d']:.2f}/{z['d']:.2f} "
                  f"({gp(x['d'], z['d']):+.1f}%)  write count {x['w']:.2f}/{z['w']:.2f} "
                  f"({gp(x['w'], z['w']):+.1f}%)  derivation depth {x['dep']:.2f}/{z['dep']:.2f} "
                  f"({gp(x['dep'], z['dep']):+.1f}%)  slice mass {x['s']:.3f}/{z['s']:.3f} "
                  f"({gp(x['s'], z['s']):+.1f}%)")
        eps0 = dial(A, k, m, "eps", 0.0, BASE_ACC, rng, a.dial_reps, a.dial_sub)
        print(f"  [calibration] uniform slip {eps0:.5f}/op -> {BASE_ACC:.2f}; every executor "
              f"below is dialled to the same accuracy", flush=True)
        print(f"\n   {'executor':34s} {'acc':>6s} {'n':>5s} "
              + " ".join(f"{s:>13s}" for s in live), flush=True)

        caps = list(range(2, k + m + 1))
        hors = sorted({18} | set(range(2, 4 * L // 10, 2)))
        rows = []
        for drop in drops:
            tgt = BASE_ACC - drop
            for label, knob in CONT + DEFICITS:
                if knob == "eps":
                    kw = {"eps": dial(A, k, m, "eps", 0.0, tgt, rng, a.dial_reps, a.dial_sub)}
                else:
                    kw = {"eps": eps0,
                          knob: dial(A, k, m, knob, eps0, tgt, rng, a.dial_reps, a.dial_sub)}
                rows.append((f"{label} -{drop:.2f}", kw))
            for pol in ("lru", "fifo"):
                v = dial_int(A, k, m, "cap", caps, eps0, tgt, rng, a.dial_reps, a.dial_sub,
                             {"policy": pol})
                rows.append((f"{'N10' if pol == 'lru' else 'N11'} {pol.upper()} capacity {v} "
                             f"-{drop:.2f}", {"eps": eps0, "cap": v, "policy": pol}))
            v = dial_int(A, k, m, "horizon", hors, eps0, tgt, rng, a.dial_reps, a.dial_sub)
            rows.append((f"N12 hard horizon H={v} -{drop:.2f}", {"eps": eps0, "horizon": v}))
        rows.append(("N12 hard horizon H=18 (the one that broke it)",
                     {"eps": eps0, "horizon": 18}))
        if a.only:
            keep = tuple(a.only.split(","))
            rows = [(lb, kw) for lb, kw in rows if lb.startswith(keep)]
        for label, kw in rows:
            p = accuracy(A, k, m, a.reps, rng, kw)
            for j, nn in enumerate(sizes):
                out = experiment(colsets, p, a.R, nn, rng, live)
                tag = label if j == 0 else ""
                print(f"   {tag:34s} {p.mean():6.3f} {nn:5d} "
                      + " ".join(f"{out[s][0]:6.3f}{out[s][1]:+7.4f}" for s in live), flush=True)


if __name__ == "__main__":
    main()
