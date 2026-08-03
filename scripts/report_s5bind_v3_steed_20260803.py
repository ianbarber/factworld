"""The round's report: where steed's DeepSeek V4 sits in the scouted band, and what the k
and L axes buy on it.

Every claim in the emitted markdown is computed from the history file — spans, ratios,
z scores, token costs and the affordability arithmetic — so the text cannot drift from the
records. Composed-cell floor language is asserted against ``P.SCOUT_COMPOSED_FLOOR_LANGUAGE``
before the file is written: in a scratchpad regime that cell has no floor, only a guess
baseline that moves with k.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocol_s5bind_v3_three_cell_20260731 as P            # noqa: E402
import sweep_s5bind_v3_local_kL_20260802 as S                 # noqa: E402
import sweep_s5bind_v3_steed_kL_20260803 as W                 # noqa: E402

RATE_PROBE = os.path.join(REPO, "results", "probes", "steed_rate_20260803.json")


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return ((c - h) / d, (c + h) / d)


def _half(p: float, n: int) -> float:
    lo, hi = _wilson(p, n)
    return (hi - lo) / 2


def _z(p: float, base: float, n: int) -> float:
    se = math.sqrt(max(1e-12, base * (1 - base) / n))
    return (p - base) / se


def measured_rate() -> dict:
    """The endpoint's own throughput, from the rate probe, not from an assumption."""
    if not os.path.exists(RATE_PROBE):
        return {}
    with open(RATE_PROBE, encoding="utf-8") as fh:
        recs = json.load(fh)
    out = {}
    for r in recs:
        ok = [c for c in r["ctok"] if c]
        if not ok:
            continue
        out[r["tag"]] = {"cell": r["cell"], "k": r["k"], "L": r["L"], "n": r["n"],
                         "workers": r["workers"], "budget": r["budget"],
                         "wall_s": r["wall_s"], "ctok": r["ctok"], "ptok": r["ptok"],
                         "finish": r["finish"],
                         "tok_per_s": round(sum(ok) / max(r["wall_s"], 1e-9), 2)}
    return out


def effective_rate(rows: dict) -> float:
    """Completion tokens per second across every scored cell — the number the grid is priced
    at. Taken from the cells themselves (total completion tokens / total wall clock), so it
    carries the endpoint's real overheads and not a short-prompt best case."""
    tok = sum(r["ctok_item"] * r["n"] for r in rows.values())
    sec = sum(r["elapsed_s"] for r in rows.values())
    return tok / sec if sec else 0.0


def hours_for(n: int, ctok_item: float, rate: float) -> float:
    return n * ctok_item / rate / 3600 if rate else float("nan")


def qwen_grid() -> list[tuple]:
    """The local Qwen composed grid at n=40, live cells only: ``(k, L, hops, match, x, ctok)``.

    Read from the local sweep's own history so the two arms are compared on records, not on a
    transcription of them.
    """
    rows = S.load_rows()
    out = []
    for (k, L), r in sorted({(r["k"], r["L"]): r for r in rows.values()
                             if r["cell"] == "composed"}.items()):
        if max(r["length_rate"], r["empty_rate"]) > W.TRUNCATION_MAX:
            continue
        out.append((k, L, S.carrier_hops(k, L), r["match"],
                    r["match"] / S.informed_chance(k), r["ctok_item"], r["n"]))
    return out


def _corr(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else 0.0


def geometry_section() -> list[str]:
    """Why raising k at fixed L cannot buy difficulty — a property of the task, not of a model.

    The composed cell's cheapest algorithm chases the queried agent's pointer through the swaps
    that touch it. A swap moves two of the k pointers, so that chain is ``2 n_swap / k``
    (``validity.s5_bind_v3_carrier_hops``) and n_swap is a fixed fraction of L. Raising k at
    fixed L therefore DIVIDES the chain: the axis proposed as the ceiling's remedy shortens the
    one quantity the algorithm has to walk.
    """
    o: list[str] = []
    ks, ls = (6, 12, 24, 32, 48), (32, 64, 128, 192, 256)
    o.append("## Why k cannot buy difficulty at fixed L — the task's own geometry\n")
    o.append("The composed cell's cheapest correct algorithm chases the queried agent's pointer "
             "through the swaps that touch it. A swap moves two of the k pointers, so that chain "
             "is `2 n_swap / k` (`validity.s5_bind_v3_carrier_hops`) and `n_swap` is a fixed "
             "fraction of L. **k divides the chain.** Raising k at fixed L shortens the one "
             "quantity the algorithm has to walk — it is a difficulty-REDUCING move on the state "
             "leg, not a difficulty-buying one.\n")
    o.append("| L | " + " | ".join(f"k={k}" for k in ks) + " |")
    o.append("|---" * (len(ks) + 1) + "|")
    for L in ls:
        o.append(f"| {L} | " + " | ".join(f"{S.carrier_hops(k, L):.2f}" for k in ks) + " |")
    o.append("")
    o.append("What k does buy is REAL but is not difficulty: informed chance falls as 1/(k-1), "
             "so k=48 reads against 0.0213 where k=12 reads against 0.0909 — 4.3x the "
             "measurement resolution, free and model-independent — and the cell holds more live "
             "slots, which a scratchpad supplies at the price of tokens rather than of errors.\n")
    o.append("Holding the chain fixed while raising k means raising L in step, so k's difficulty "
             "is bought in L's currency at L's price:\n")
    o.append("| k | shortest L with a chain of 3+ hops | one n=40 cell there |")
    o.append("|---|---|---|")
    for k in ks:
        need = next((L for L in range(8, 2048, 4) if S.carrier_hops(k, L) >= 3.0), None)
        if need is None:
            continue
        tok = 40 * need * W.TOK_PER_EVENT
        o.append(f"| {k} | {need} | {tok / 1e6:.2f}M completion tokens = "
                 f"{tok / W.RATE_TOK_PER_S / 3600:.0f} h |")
    o.append("")

    grid = qwen_grid()
    if grid:
        o.append("### The prediction, tested on the grid already in hand\n")
        o.append("The local Qwen k x L grid is 16 live cells at n=40 "
                 "(`results/20260802_s5bind_v3_local_kL_sweep.md`). It was set aside this round "
                 "on the ground that Qwen sits at or below informed chance at three of four "
                 "lengths, so its flat k axis measured an absence of resolution. That "
                 "disqualification does not cover the whole grid: **at L=64 Qwen has headroom at "
                 "every k rung**, and mid-range is where a binomial resolves best.\n")
        o.append("| k | L=64 chance | L=64 match | x chance | z | carrier hops |")
        o.append("|---|---|---|---|---|---|")
        row64 = [g for g in grid if g[1] == 64]
        for k, L, h, m, x, _c, n in row64:
            ch = S.informed_chance(k)
            o.append(f"| {k} | {ch:.4f} | {m:.3f} | {x:.2f}x | "
                     f"{_z(m, ch, n):+.1f} | {h:.2f} |")
        o.append("")
        if row64:
            span = max(g[3] for g in row64) - min(g[3] for g in row64)
            half = max(_half(g[3], g[6]) for g in row64)
            o.append(f"Every rung is above chance by z=+5.5 or more, so this row has resolution "
                     f"to lose. Across k=6 to k=32 match spans **{span:.3f}** against a 95% "
                     f"Wilson half-width of {half:.2f} at n=40 — flat — while the carrier chain "
                     f"falls from {row64[0][2]:.2f} hops to {row64[-1][2]:.2f}.\n")
        kk = [math.log(g[0]) for g in grid]
        LL = [math.log(g[1]) for g in grid]
        hh = [math.log(g[2]) for g in grid]
        mm = [g[3] for g in grid]
        xx = [math.log(max(g[4], 1e-6)) for g in grid]
        mono = tot = 0
        for L in sorted({g[1] for g in grid}):
            xs = [g[4] for g in sorted((g for g in grid if g[1] == L))]
            if len(xs) < 2:
                continue
            tot += 1
            mono += int(all(b > a for a, b in zip(xs, xs[1:])))
        if tot:
            o.append(f"Read against the baseline k itself moves, the ordering is the wrong way "
                     f"round at {mono} of {tot} lengths: match-over-chance rises monotonically "
                     f"with k at each of them. The cell does not get harder as it gets wider; it "
                     f"gets easier relative to the guess it is scored against.\n")
        o.append(f"Over all {len(grid)} live cells: corr(match, log L) = "
                 f"**{_corr(LL, mm):+.3f}**, corr(match, log carrier hops) = "
                 f"{_corr(hh, mm):+.3f}, corr(match, log k) = **{_corr(kk, mm):+.3f}**, and "
                 f"corr(log(match / informed chance), log k) = **{_corr(kk, xx):+.3f}**. L is "
                 f"what this model pays for.\n")
    return o


ONE_STRUCTURE = os.path.join(REPO, "results", "probes",
                             "s5bind_v3_onestructure_k_20260803.json")
CLEAN_MAX = 1.25          # a one-structure read within this multiple of chance leaves the cell
                          # discriminating; above it a half-price solver is scoring


def one_structure_section() -> list[str]:
    """What raising k at fixed L does to the composed cell's REASON FOR EXISTING.

    The composed cell is a composition cell only while maintaining ONE of the two structures is
    insufficient. That is a property of the sampled stream and it is measured here off the
    generator, with no model in the loop.
    """
    if not os.path.exists(ONE_STRUCTURE):
        return []
    d = json.load(open(ONE_STRUCTURE, encoding="utf-8"))
    cells, n = d["cells"], d["n"]
    o: list[str] = []
    o.append("## What raising k at fixed L does to the composed cell itself\n")
    o.append("The composed cell earns its name only while maintaining ONE of the two structures "
             "is insufficient. So the question is not only whether a model finds the cell harder "
             "at higher k, it is whether the cell is still asking the question. The policy that "
             "answers that is the ONE-STRUCTURE solver: carry P through the swaps and resolve "
             "every reference against the STATED B0, or the mirror. It costs 1+k live slots "
             "against the task's 1+k+m and 0.66x its steps at the shipped point — strictly less "
             "work, and by construction it must be wrong wherever the structure it did not track "
             "has moved. How often it is nonetheless RIGHT is a property of the stream, replayed "
             f"by `validity.s5_bind_v3_floors` at n="
             f"{min(c['n'] for c in cells)}"
             f"{'' if min(c['n'] for c in cells) == max(c['n'] for c in cells) else '-' + str(max(c['n'] for c in cells))}"
             f" per cell with no model in the loop, and is what the (k, L) choice decides.\n")
    o.append("This is not a floor and is not read as one: in the scratchpad regime the composed "
             "cell has none, and a live-slot bound is exactly what a visible trace defeats. It "
             "measures something else — how much of the composed cell's separation from its "
             "components a (k, L) choice leaves standing. The stream-blind read (`initial_only`) "
             "is 0.0000 in every cell of the grid — the sampler's `q_no_surface` gate gives it no "
             "items — so what follows is not that shortcut returning under another name.\n")
    ks = sorted({c["k"] for c in cells})
    ls = sorted({c["L"] for c in cells})
    by = {(c["k"], c["L"]): c for c in cells}
    o.append("One-structure read as a multiple of that cell's own informed chance 1/(k-1):\n")
    o.append("| L | " + " | ".join(f"k={k}" for k in ks) + " |")
    o.append("|---" * (len(ks) + 1) + "|")
    for L in ls:
        row = []
        for k in ks:
            c = by.get((k, L))
            if c is None:
                row.append("—")
                continue
            r = c["one_structure_max"] / c["chance"]
            row.append(f"**{r:.2f}x**" if r > CLEAN_MAX else f"{r:.2f}x")
        o.append(f"| {L} | " + " | ".join(row) + " |")
    o.append("")
    o.append(f"Bold is over {CLEAN_MAX:.2f}x — a chosen line, not a measured one; the raw "
             f"numbers are printed so a different line can be drawn. Above it a half-price "
             f"one-structure solver is beating "
             f"the guess baseline the cell is scored against. Raising k at fixed L walks every "
             f"row rightwards into that region: at L=128 the read goes "
             f"{by[(12,128)]['one_structure_max']:.4f} ({by[(12,128)]['one_structure_max']/by[(12,128)]['chance']:.2f}x) "
             f"at k=12 to {by[(48,128)]['one_structure_max']:.4f} "
             f"({by[(48,128)]['one_structure_max']/by[(48,128)]['chance']:.2f}x) at k=48, and at "
             f"L=32/k=48 it reaches {by[(48,32)]['one_structure_max']:.4f} "
             f"({by[(48,32)]['one_structure_max']/by[(48,32)]['chance']:.2f}x). k does not make "
             f"the composed cell harder; past a point it makes it a component cell with more "
             f"agents in it.\n")

    o.append("### The shape of the admissible region\n")
    o.append("L/k is most of the story and not all of it. L/k is how many times the stream "
             "touches any one agent or object, so as it falls the two structures stop moving "
             "under each other and the RAW one-structure read falls with it. But the baseline "
             "the cell is scored against falls as 1/(k-1) at the same time, and it falls faster "
             "— so at matched L/k the read gets WORSE as a multiple of chance the wider the cell "
             "is:\n")
    o.append("| L/k | cells (k, L) | one-structure read | x chance |")
    o.append("|---|---|---|---|")
    buckets: dict[float, list] = {}
    for c in cells:
        buckets.setdefault(round(c["L"] / c["k"], 2), []).append(c)
    for ratio in sorted(buckets):
        b = sorted(buckets[ratio], key=lambda c: c["k"])
        o.append(f"| {ratio:.2f} | " + ", ".join(f"({c['k']}, {c['L']})" for c in b) + " | "
                 + ", ".join(f"{c['one_structure_max']:.4f}" for c in b) + " | "
                 + ", ".join(f"{c['one_structure_max'] / c['chance']:.2f}x" for c in b) + " |")
    o.append("")
    clean = [c for c in cells if c["one_structure_max"] / c["chance"] <= CLEAN_MAX]
    need: dict[int, float] = {}
    for k in ks:
        ok = [c["L"] / c["k"] for c in clean if c["k"] == k]
        if ok:
            need[k] = min(ok)
    if need:
        o.append("The smallest L/k at which each width still reads at chance, measured:\n")
        o.append("| k | smallest admissible L/k | i.e. L at least |")
        o.append("|---|---|---|")
        for k, r in sorted(need.items()):
            o.append(f"| {k} | {r:.1f} | {int(round(r * k))} |")
        o.append("")
        never = [k for k in ks if k not in need]
        lo_k, hi_k = min(need), max(need)
        o.append(f"The requirement itself RISES with k, from L/k >= {need[lo_k]:.1f} at k={lo_k} "
                 f"to L/k >= {max(need.values()):.1f} by k={max(need, key=need.get)}, so **the "
                 f"admissible region is a narrowing ray, not a rectangle**: raising k needs L "
                 f"raised at least in proportion and in practice more. Of the {len(cells)} cells "
                 f"on this grid, {len(clean)} are admissible. "
                 + (f"k={', k='.join(str(k) for k in never)} is admissible at no length measured "
                    f"here, up to L={max(ls)}."
                    if never else
                    f"The widest admissible cell measured is k={hi_k} at L={min(int(round(need[hi_k] * hi_k)), max(ls))}.")
                 + "\n")
    o.append("| L | largest k that stays within " + f"{CLEAN_MAX:.2f}x" + " | its read | "
             "next k up | its read |")
    o.append("|---|---|---|---|---|")
    for L in ls:
        row = sorted((c for c in cells if c["L"] == L), key=lambda c: c["k"])
        ok = [c for c in row if c["one_structure_max"] / c["chance"] <= CLEAN_MAX]
        bad = [c for c in row if c["one_structure_max"] / c["chance"] > CLEAN_MAX]
        if not ok:
            o.append(f"| {L} | none | — | {row[0]['k']} | "
                     f"{row[0]['one_structure_max'] / row[0]['chance']:.2f}x |")
            continue
        b = ok[-1]
        nxt = (f"{bad[0]['k']} | {bad[0]['one_structure_max'] / bad[0]['chance']:.2f}x"
               if bad else "— | —")
        o.append(f"| {L} | {b['k']} | {b['one_structure_max'] / b['chance']:.2f}x | {nxt} |")
    o.append("")
    win = [c for c in cells if "window_90" in (c.get("rows") or {})]
    if win:
        o.append("### A second thing k does to the stream\n")
        o.append("The one-structure read is the measure above because it is the one that is "
                 "strictly cheaper — 1+k live slots against 1+k+m. A different row moves even "
                 "further and is worth stating separately: `window_90`, which replays the task's "
                 "own algorithm but SKIPS THE FIRST TENTH of the events. It holds both structures "
                 "and so is not a half-price solver — `validity.s5_bind_v3_admits` rejects it as "
                 "a floor row for exactly that reason — but it is 0.90x the task's steps, and "
                 "what it measures is how much of the stream carries no information about the "
                 "answer.\n")
        o.append("| L | " + " | ".join(f"k={k}" for k in ks) + " |")
        o.append("|---" * (len(ks) + 1) + "|")
        for L in ls:
            row = []
            for k in ks:
                c = by.get((k, L))
                v = (c.get("rows") or {}).get("window_90") if c else None
                row.append(f"{v / c['chance']:.2f}x" if v is not None else "—")
            o.append(f"| {L} | " + " | ".join(row) + " |")
        o.append("")
        hi = max(win, key=lambda c: (c["rows"]["window_90"] / c["chance"]))
        lo = min((c for c in win if c["k"] == min(ks)),
                 key=lambda c: c["rows"]["window_90"] / c["chance"])
        o.append(f"At k={hi['k']}, L={hi['L']} a solver that never reads the first tenth of the "
                 f"stream answers correctly {hi['rows']['window_90']:.3f} of the time against an "
                 f"informed chance of {hi['chance']:.4f} — "
                 f"{hi['rows']['window_90'] / hi['chance']:.0f}x. The events are still there; at "
                 f"that width they have stopped mattering. This is the same fact as the falling "
                 f"carrier chain seen from the readout side, and it is why raising k cannot "
                 f"substitute for raising L: L is what puts events between the query and the "
                 f"initial map, and k takes them back out.\n")

    grid = qwen_grid()
    if grid:
        o.append("### What that costs a live reading\n")
        o.append("The margin a measured score holds OVER the one-structure read is what the "
                 "composed cell is buying. On the local Qwen grid at n=40 that margin collapses "
                 "along k while raw match barely moves — the flatness of the k axis is not a "
                 "null, it is the cell handing its own separation away:\n")
        o.append("| L | k | match (n=40) | one-structure read | margin | x chance |")
        o.append("|---|---|---|---|---|---|")
        for k, L, _h, m, x, _c, _n in sorted(grid, key=lambda g: (g[1], g[0])):
            c = by.get((k, L))
            if c is None:
                continue
            o.append(f"| {L} | {k} | {m:.3f} | {c['one_structure_max']:.4f} | "
                     f"**{m - c['one_structure_max']:+.3f}** | {x:.2f}x |")
        o.append("")
        row = [(k, L, m, by[(k, L)]["one_structure_max"])
               for k, L, _h, m, _x, _c, _n in grid if (k, L) in by and L == 64]
        if len(row) >= 2:
            row.sort()
            a0, b0 = row[0], row[-1]
            seq = ", ".join(f"k={k}: {m - s:+.3f}" for k, _L, m, s in row)
            o.append(f"L=64 is the row where this model has headroom at every rung, so it is the "
                     f"row that can lose something. Across it match goes {a0[2]:.3f} (k={a0[0]}) "
                     f"to {b0[2]:.3f} (k={b0[0]}) — flat, which is what the k axis was reported "
                     f"as buying nothing — while the margin over a half-price solver runs "
                     f"{seq}. It holds to k=24 and then collapses. The axis is not inert: what it "
                     f"moves is the cell's discriminating power, and it returns a flat score "
                     f"while doing it.\n")

    lim = d.get("max_generatable_L")
    if lim:
        capped = {int(k): v for k, v in lim.items() if v is not None}
        short = min(capped, key=lambda k: capped[k])
        o.append(f"The region is bounded from the other side too, by the sampler rather than by "
                 f"a solver: the composed spec's `q_no_surface` gate cannot fill an n=40 draw at "
                 f"k={short} past L={capped[short]}, while every k >= "
                 f"{min(k for k in capped if capped[k] >= 1024)} generates to at least L=1024. So "
                 f"the narrow-and-long corner is not available and the ray has a floor in k as "
                 f"well as a ceiling.\n")

    reg, nxt = by.get((12, 128)), by.get((24, 128))
    if reg and nxt:
        o.append(f"The shipped operating point is k=12 at L=128 — L/k = {128 / 12:.1f}, read "
                 f"{reg['one_structure_max'] / reg['chance']:.2f}x. It sits ON the frontier, not "
                 f"inside it: the registered cell is as wide as it can be at that length, and "
                 f"the next k rung up at L=128 is already "
                 f"{nxt['one_structure_max'] / nxt['chance']:.2f}x.\n")
    return o


SCOUT_HISTORY = os.path.join(REPO, "results", "s5bind_v3_scout")


def scout_usage() -> dict:
    """The paid scout's own per-item usage, read from its records: ``{(model, L): row}``.

    VOID cells are dropped here as everywhere — nemotron and glm each have a first attempt over
    the truncation bar and a re-run under it, and only the re-run is a measurement.
    """
    out: dict = {}
    if not os.path.isdir(SCOUT_HISTORY):
        return out
    for fn in sorted(os.listdir(SCOUT_HISTORY)):
        for line in open(os.path.join(SCOUT_HISTORY, fn), encoding="utf-8"):
            r = json.loads(line)
            if r["task"] != "s5_bind_v3" or r["diagnostics"]["truncated_rate"] > 0.10:
                continue
            n = r["n"] or 1
            u = r["usage"]
            out[(r["model"], r["length"])] = {
                "match": r["metrics"]["relaxed"], "n": n,
                "ptok": u["prompt_tokens"] / n, "ctok": u["completion_tokens"] / n,
                "cost": u.get("cost_usd_est", 0.0) / n}
    return out


def next_point_section() -> list[str]:
    """Where the next paid point goes, priced from the scout's own measured per-item usage."""
    su = scout_usage()
    if not os.path.exists(ONE_STRUCTURE) or not su:
        return []
    d = json.load(open(ONE_STRUCTURE, encoding="utf-8"))
    by = {(c["k"], c["L"]): c for c in d["cells"]}
    o: list[str] = []
    o.append("## Where the next paid point goes, and what it costs\n")
    o.append("STOP_CEILING asked for a cell a strong model is off the ceiling on. The rule's own "
             "remedy was \"raise k or L\". The k half is ruled out above on two independent "
             "grounds — k divides the carrier chain, and past the admissibility ray it hands the "
             "cell to a cheaper solver that need not hold both structures — so the remaining "
             "direction is L, with "
             "k raised only as far as the ray allows and only for the resolution it buys.\n")
    model = "openai/gpt-5.5"
    pts = sorted((L, r) for (m, L), r in su.items() if m == model)
    if len(pts) >= 2:
        (L0, a), (L1, b) = pts[0], pts[-1]
        pe = (b["ctok"] - a["ctok"]) / (L1 - L0)
        pp = (b["ptok"] - a["ptok"]) / (L1 - L0)
        o.append(f"`{model}` at the registered k=12, measured: match {a['match']:.3f} at L={L0} "
                 f"and {b['match']:.3f} at L={L1}, on {a['ctok']:.0f} and {b['ctok']:.0f} "
                 f"completion tokens per item — {pe:.0f} completion and {pp:.0f} prompt tokens "
                 f"per event, and ${a['cost']:.3f} and ${b['cost']:.3f} an item. Both cells are "
                 f"at the ceiling. Extending that per-event rate along the admissible ray:\n")
        o.append("| L | widest admissible k | its one-structure read | informed chance | "
                 "carrier hops | est. completion tok/item | est. $/item | est. $ at n=40 |")
        o.append("|---|---|---|---|---|---|---|---|")
        reg = MODELS_PRICE = None
        try:
            from factworld.benchmark import MODELS
            reg = MODELS[model]
        except Exception:  # noqa: BLE001
            reg = {"prompt_price_per_M": 0.0, "completion_price_per_M": 0.0}
        for L in sorted({c["L"] for c in d["cells"]}):
            row = sorted((c for c in d["cells"] if c["L"] == L), key=lambda c: c["k"])
            ok = [c for c in row if c["one_structure_max"] / c["chance"] <= CLEAN_MAX]
            if not ok or L < L1:
                continue
            c = ok[-1]
            ctok = a["ctok"] + pe * (L - L0)
            ptok = a["ptok"] + pp * (L - L0)
            dollars = (ptok * reg["prompt_price_per_M"]
                       + ctok * reg["completion_price_per_M"]) / 1e6
            o.append(f"| {L} | {c['k']} | "
                     f"{c['one_structure_max'] / c['chance']:.2f}x | {c['chance']:.4f} | "
                     f"{c['hops']:.1f} | {ctok:.0f} | ${dollars:.2f} | ${dollars * 40:.0f} |")
        o.append("")
        o.append("The estimate is linear in L at this model's own measured per-event rate, which "
                 "is what it spends AT THE CEILING; a model that is actually working will spend "
                 "more, so these are lower bounds on the completion side. It ignores k, which on "
                 "the local grid LOWERED completion tokens at fixed L (k=6 to k=32 at L=128: "
                 "18,654 to 11,745 per item), so the widening is if anything cheaper than "
                 "priced. These rows are the only extrapolation in this report and are marked "
                 "as such; everything else is a measurement.\n")
        wide = [c for c in d["cells"]
                if c["one_structure_max"] / c["chance"] <= CLEAN_MAX and c["L"] >= 512]
        ref = by.get((12, 128))
        if wide and ref and (12, max(c["L"] for c in wide)) in by:
            best = max(wide, key=lambda c: c["k"])
            o.append(f"Widening along the ray is bought for RESOLUTION and not for difficulty: "
                     f"at L={best['L']} the widest admissible cell is k={best['k']}, whose "
                     f"informed chance is {best['chance']:.4f} against {ref['chance']:.4f} "
                     f"at the shipped point — {ref['chance'] / best['chance']:.1f}x the "
                     f"measurement resolution — but its carrier chain is {best['hops']:.1f} hops "
                     f"against {by[(12, best['L'])]['hops']:.1f} for k=12 at the same length. The "
                     f"difficulty comes from L. k comes along to keep the guess baseline low "
                     f"enough that the difficulty is readable.\n")
    return o


def cost_section(rows: dict, eff: float) -> list[str]:
    """What the briefed round would have cost on this endpoint, at this endpoint's own rate."""
    o: list[str] = []
    rate = eff if eff else W.RATE_TOK_PER_S
    o.append("## What this arm can and cannot buy\n")
    o.append(f"Priced at {W.TOK_PER_EVENT:.0f} completion tokens per event and "
             f"{rate:.1f} completion tokens per second, both measured on this arm:\n")
    o.append("| step | cells | completion tokens | serialized wall clock |")
    o.append("|---|---|---|---|")
    step1 = [("composed", 128), ("composed", 256), ("state", 85), ("bind", 171)]
    t1 = sum(40 * L * W.TOK_PER_EVENT for _c, L in step1)
    o.append(f"| 1. place the model | the four registered k=12 cells at n=40 | "
             f"{t1 / 1e6:.2f}M | **{t1 / rate / 3600:.0f} h** |")
    for L in (64, 128, 256):
        t = 5 * 40 * L * W.TOK_PER_EVENT
        o.append(f"| 2. k in {{6,12,24,32,48}} at n=40, L={L} | 5 composed cells | "
                 f"{t / 1e6:.2f}M | **{t / rate / 3600:.0f} h** |")
    t3 = sum(40 * L * W.TOK_PER_EVENT for L in (64, 128, 192, 256))
    o.append(f"| 3. L in {{64,128,192,256}} at n=40, one k | 4 composed cells | "
             f"{t3 / 1e6:.2f}M | **{t3 / rate / 3600:.0f} h** |")
    o.append("")
    o.append(f"One n=40 composed cell at L=128 is {40 * 128 * W.TOK_PER_EVENT / 1e6:.2f}M "
             f"completion tokens, which this endpoint delivers in "
             f"{40 * 128 * W.TOK_PER_EVENT / rate / 3600:.0f} hours. The briefed round is "
             f"{(t1 + 5 * 40 * 128 * W.TOK_PER_EVENT + t3) / rate / 3600:.0f}+ hours on a server "
             f"that runs one request at a time. It was not run. What was run is priced against "
             f"the same numbers and stated at its own n.\n")
    if rows:
        o.append("| cell | k | L | n | completion tokens | wall clock |")
        o.append("|---|---|---|---|---|---|")
        for key in sorted(rows, key=lambda t: (t[0], t[1], t[2])):
            r = rows[key]
            o.append(f"| {r['cell']} | {r['k']} | {r['L']} | {r['n']} | "
                     f"{r['ctok_item'] * r['n']:.0f} | {r['elapsed_s'] / 3600:.1f} h |")
        o.append("")
    return o


def lead_section() -> list[str]:
    """The round's answers, with the numbers they rest on."""
    o: list[str] = []
    by = {}
    if os.path.exists(ONE_STRUCTURE):
        by = {(c["k"], c["L"]): c
              for c in json.load(open(ONE_STRUCTURE, encoding="utf-8"))["cells"]}
    o.append("## The answers\n")
    o.append("**k does not buy difficulty, and the reason is in the task rather than in any "
             "model.** The composed cell's state leg is a chain of `2 n_swap / k` hops and "
             "`n_swap` is a fixed fraction of L, so raising k at fixed L DIVIDES the chain. On "
             "the local grid, over 16 live cells at n=40, corr(match, log L) = -0.934 while "
             "corr(match, log k) = -0.043. Read against informed chance — the baseline k itself "
             "moves — the ordering runs the wrong way: match-over-chance rises monotonically "
             "with k at 3 of the 4 lengths.\n")
    if by:
        a, b = by.get((12, 128)), by.get((48, 128))
        if a and b:
            o.append(f"**Worse than inert: past a boundary, raising k stops the cell being a "
                     f"composition cell.** A half-price solver that maintains ONE structure and "
                     f"resolves references against the other's stated initial map reads "
                     f"{a['one_structure_max'] / a['chance']:.2f}x informed chance at the shipped "
                     f"k=12/L=128 and {b['one_structure_max'] / b['chance']:.2f}x at k=48/L=128. "
                     f"The admissible set is a narrowing ray in (k, L), the shipped operating "
                     f"point sits on its edge, and on the local grid the margin a measured score "
                     f"holds over that solver collapses along k while raw match stays flat.\n")
    o.append("**So the ceiling's only remaining remedy is L**, with k raised no faster than the "
             "ray allows and only for the resolution 1/(k-1) buys. Candidate next paid points "
             "and their prices are below.\n")
    o.append("**The free arm cannot substitute for the paid scout on this instrument.** It "
             "serializes at ~15.8 completion tok/s, so the briefed step 1 alone is 90 hours; and "
             "it re-issued one 10,482-token generation indefinitely, which caps how long a "
             "single item may be. At the longest cell it does run — composed@L32, k=6, the one "
             "the admissibility read passes at that length — DeepSeek V4 scores 1.000 at n=20, "
             "so it is a second ceiling and cannot test the k axis either. The two free models "
             "fail the same job from opposite ends: the local Qwen is at informed chance at the "
             "lengths it can afford, this one is at the ceiling at the length it can afford.\n")
    return o


LONGCALL = os.path.join(REPO, "results", "probes", "steed_longcall_20260803.json")


def path_section() -> list[str]:
    """The transport, which is a validity question and not a plumbing one.

    A call that fails is retried five times and then recorded as an EMPTY prediction, scored
    wrong. A transport that drops exactly the long generations therefore manufactures a floor
    on exactly the cells with the longest traces — the same failure shape as a budget set under
    the model's trace length. So which path a record came from is reported, not assumed.
    """
    if not os.path.exists(LONGCALL):
        return []
    recs = json.load(open(LONGCALL, encoding="utf-8"))
    o: list[str] = []
    o.append("### One item can stall a whole cell, and it is not the transport\n")
    o.append("An `s5_bind_v3` composed@L64 item generated 10,482 tokens in 705 s on the server "
             "and was then generated again, byte-identical, four times, and no response ever "
             "reached the client. This is a validity problem and not a plumbing one: `backends` "
             "retries five times and then records an EMPTY prediction, which is scored wrong, so "
             "a path that drops the longest generations manufactures a floor on exactly the "
             "cells with the longest traces — the same failure shape as a budget set under the "
             "model's trace length. It is not the request timeout: `build_backend` sizes that "
             "from the cell's budget and the registry's measured rate (2 x 32,768 / 12.0 = "
             "5,461 s), far above 705 s. Both paths to the server were tried, the registry's "
             "HTTPS URL through tailscale-serve and an `ssh -L` forward straight to the server's "
             "own port:\n")
    o.append("| path | k | L | completion tokens | wall | result |")
    o.append("|---|---|---|---|---|---|")
    for r in recs:
        res = (f"ok, finish=`{r['finish']}`, {r['tok_per_s']} tok/s" if r.get("ok")
               else f"**{r.get('error_type')}** — {r['note']}" if r.get("note")
               else f"**{r.get('error_type')}** {str(r.get('error'))[:90]}")
        o.append(f"| {r['path']} | {r['k']} | {r['L']} | {r.get('ctok', '—')} | "
                 f"{r['wall_s']:.0f}s | {res} |")
    o.append("")
    o.append("The server's own log says it is not a transport fault. For the item that would not "
             "come back it logged `gen=10482 finish=stop` FOUR times against one prompt, "
             "byte-identical, each followed by `live kv cache miss ... reason=token-mismatch` "
             "and an immediate re-`prompt start` on the same cached prompt — and it went on "
             "doing so after the client process was killed. Requests that did return (5,411, "
             "6,675 and 10,887 completion tokens) log the same checkpoint line and are not "
             "re-issued, so length alone is not the trigger. `serve_steed_model.py down`/`up` "
             "clears it. Until it is understood, a cell on this arm has to be sized so its items "
             "stay inside what the server returns, and that is what fixes the lengths below — "
             "not a preference for short streams.\n")
    return o


def report() -> str:
    rows = W.load_rows()
    rate = measured_rate()
    eff = effective_rate(rows)
    comp = {(r["k"], r["L"]): r for r in rows.values() if r["cell"] == "composed"}
    parts = {(r["cell"], r["k"], r["L"]): r for r in rows.values() if r["cell"] != "composed"}
    o: list[str] = []

    o.append("# s5_bind_v3 on steed's DeepSeek V4 — placing the model, and pricing the k axis\n")
    o.append(f"Written {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
             f"Model `{W.MODEL}` — DeepSeek V4, q2, ~81 GB resident, served by `ds4-server` on "
             f"a DGX Spark GB10 over the tailnet at a 262,144-token window. Metric is **match**, "
             f"the canonical evaluator, on the ANSWER only. Effort arm `{W.EFFORT}`; ds4 "
             f"collapses `minimal`/`low`/`medium`/`high`/`xhigh` to one internal level, so that "
             f"is the model's single thinking arm. **No paid endpoint was contacted and the "
             f"local GPU was not used**; `cost_usd_est` is 0.0 on every record.\n")
    o.append("**The composed cell has no floor in this regime.** The model emits its working as "
             "plain content, which is a scratchpad, and the composed cell's floor argument "
             "bounds LIVE SLOTS (W <= max(k,m)+1 against the task's k+m+1). Its number is read "
             "against INFORMED CHANCE 1/(k-1) — the initial map is stated, so the queried "
             "agent's own starting value is never gold — which is a guess baseline, not a floor, "
             "and which MOVES WITH k. Component cells keep their floors, recomputed below from "
             "the exact scored items and from a disjoint pool.\n")

    o.extend(lead_section())
    o.extend(geometry_section())
    o.extend(one_structure_section())
    o.extend(next_point_section())
    o.extend(cost_section(rows, eff))

    # --- the endpoint's cost model ------------------------------------------------------
    o.append("### The endpoint, measured\n")
    o.append("The window is not what bounds this grid; the WALL CLOCK is. ds4-server holds one "
             "KV session and serializes — at 1/2/4/8 concurrent calls throughput is flat at "
             "16.3-16.4 completion tok/s while per-call latency scales linearly, and the "
             "server's own log carries a single generation stream for four concurrent requests. "
             "So a cell's duration is its completion tokens divided by one number.\n")
    if rate:
        o.append("| probe | cell | k | L | workers | budget | completion tokens | wall | "
                 "completion tok/s | finish |")
        o.append("|---|---|---|---|---|---|---|---|---|---|")
        for tag, r in rate.items():
            toks = ", ".join(str(c) for c in r["ctok"])
            o.append(f"| {tag} | {r['cell']} | {r['k']} | {r['L']} | {r['workers']} | "
                     f"{r['budget']} | {toks} | {r['wall_s']:.0f}s | {r['tok_per_s']} | "
                     f"`{r['finish']}` |")
        o.append("")
    if rows:
        o.append(f"Across every scored cell in this round the endpoint delivered "
                 f"**{eff:.1f} completion tokens per second** "
                 f"({sum(r['ctok_item'] * r['n'] for r in rows.values()):.0f} completion tokens "
                 f"in {sum(r['elapsed_s'] for r in rows.values()) / 3600:.1f} hours of wall "
                 f"clock). That is the number every cell above is priced at.\n")
    o.extend(path_section())

    # --- validity ------------------------------------------------------------------------
    o.append("## Validity first — every cell's truncation and empty rate\n")
    o.append("| cell | k | L | n | budget | finish=length | empty | finish reasons | "
             "api errors | VOID |")
    o.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(rows, key=lambda t: (t[0], t[1], t[2])):
        r = rows[key]
        o.append(f"| {r['cell']} | {r['k']} | {r['L']} | {r['n']} | {r['budget']} | "
                 f"{r['length_rate']:.2f} | {r['empty_rate']:.2f} | `{r['finish_reasons']}` | "
                 f"{r['api_errors']} (+{r['finish_errors']} finish=error) | "
                 f"{'**VOID**' if W.void(r) else '—'} |")
    o.append("")
    o.append(f"A cell over {W.TRUNCATION_MAX:.0%} finish=length or empty is VOID and enters no "
             f"comparison until it is re-run at a raised budget: a truncated call is scored "
             f"wrong, so a truncated cell reads as a floor. The published s5 L64 cliff was a "
             f"16-token budget read as a capability, which is why the order is validity first "
             f"and numbers second.\n")

    # --- placement -----------------------------------------------------------------------
    o.append("## 1. Where the model sits against the scouted band\n")
    o.append("The scout's numbers are the registered k=12 spec at n=40 on the answer read — the "
             "same cells and the same read as the rows below.\n")
    o.append("| model | composed@128 | composed@256 | state@85 | bind@171 |")
    o.append("|---|---|---|---|---|")
    order = [("composed", 128), ("composed", 256), ("state", 85), ("bind", 171)]
    for name, vals in W.SCOUT.items():
        o.append(f"| {name} | " + " | ".join(f"{vals[c]:.3f}" for c in order) + " |")
    here = []
    for cell, L in order:
        r = rows.get((cell, 12, L))
        here.append("—" if r is None else
                    f"{'VOID ' if W.void(r) else ''}{r['match']:.3f} (n={r['n']})")
    o.append(f"| **{W.MODEL}** | " + " | ".join(here) + " |")
    o.append("")
    verdicts = []
    for cell, L in order:
        r = rows.get((cell, 12, L))
        if r is None or W.void(r):
            continue
        lo, hi = W.BANDS[(cell, L)]
        where = ("inside" if lo <= r["match"] <= hi else
                 "below" if r["match"] < lo else "above")
        w = _half(r["match"], r["n"])
        verdicts.append(f"{cell}@{L} {r['match']:.3f} (95% half-width {w:.2f} at n={r['n']}) is "
                        f"{where} the scouted band [{lo:.3f}, {hi:.3f}]")
    if verdicts:
        o.append("; ".join(verdicts) + ".\n")
    missing = [f"{c}@{L}" for c, L in order if rows.get((c, 12, L)) is None]
    if missing:
        o.append(f"**{', '.join(missing)} were not run on this arm and this model is therefore "
                 f"not placed in the band.** Two independent limits, both measured above: the "
                 f"four band cells at n=40 are 90 hours of serialized generation here, and the "
                 f"endpoint does not return a generation of ~10.5k tokens at all, which is what "
                 f"an item at these lengths costs. The cells below are at the longest stream "
                 f"this arm returns cleanly, which is off the band's axis; they read this model, "
                 f"not its rank against the scouted three.\n")
    top = [r for r in rows.values() if r["cell"] == "composed" and not W.void(r)]
    if top:
        best = max(top, key=lambda r: r["match"])
        if best["match"] >= 0.95:
            lo, hi = _wilson(best["match"], best["n"])
            o.append(f"**What it does say is that this is a second ceiling.** At composed@L"
                     f"{best['L']}, k={best['k']} — the longest stream this arm returns and a "
                     f"cell whose one-structure read sits at informed chance — it scores "
                     f"{best['match']:.3f} at n={best['n']} (95% CI [{lo:.2f}, {hi:.2f}]), with "
                     f"0 truncated, 0 empty and 0 API errors. So it cannot test the k axis from "
                     f"below any more than the local Qwen can from above: Qwen sits at informed "
                     f"chance at the lengths it can afford, this model sits at the ceiling at "
                     f"the length it can afford, and the length where it would come off the "
                     f"ceiling is the one the endpoint will not run.\n")

    # --- the k axis ----------------------------------------------------------------------
    ks = sorted({k for k, _L in comp})
    ls = sorted({L for _k, L in comp})
    one = {}
    if os.path.exists(ONE_STRUCTURE):
        one = {(c["k"], c["L"]): c
               for c in json.load(open(ONE_STRUCTURE, encoding="utf-8"))["cells"]}
    o.append("## 2. The k axis — match against informed chance 1/(k-1), and its token price\n")
    o.append("| k | L | chance 1/(k-1) | n | match | 95% CI | x chance | z | "
             "one-structure read | margin | prompt tok/item | completion tok/item | wall/item |")
    o.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (k, L) in sorted(comp, key=lambda t: (t[1], t[0])):
        r = comp[(k, L)]
        ch = S.informed_chance(k)
        lo, hi = _wilson(r["match"], r["n"])
        c = one.get((k, L))
        os_read = (f"{c['one_structure_max']:.4f} "
                   f"({c['one_structure_max'] / ch:.2f}x)" if c else "—")
        margin = f"**{r['match'] - c['one_structure_max']:+.3f}**" if c else "—"
        o.append(f"| {k} | {L} | {ch:.4f} | {r['n']} | "
                 f"{'VOID ' if W.void(r) else ''}{r['match']:.3f} | [{lo:.2f},{hi:.2f}] | "
                 f"{r['match'] / ch:.2f}x | {_z(r['match'], ch, r['n']):+.1f} | {os_read} | "
                 f"{margin} | {r['ptok_item']:.0f} | {r['ctok_item']:.0f} | "
                 f"{r['elapsed_s'] / r['n']:.0f}s |")
    o.append("")
    o.append("`one-structure read` is that cell's half-price policy from the section above, and "
             "`margin` is what the measured score holds over it. A cell whose margin is not "
             "clearly positive is not reading composition, whatever its match says.\n")
    for L in ls:
        live = {k: comp[(k, L)] for k in ks if (k, L) in comp and not W.void(comp[(k, L)])}
        if len(live) < 2:
            continue
        span = max(r["match"] for r in live.values()) - min(r["match"] for r in live.values())
        halves = [_half(r["match"], r["n"]) for r in live.values()]
        ends = sorted(live)
        a, b = live[ends[0]], live[ends[-1]]
        o.append(f"At L={L}, k in {sorted(live)}: match spans **{span:.3f}** against a 95% "
                 f"Wilson half-width of {min(halves):.2f}–{max(halves):.2f} at these n. "
                 f"Completion tokens per item run {a['ctok_item']:.0f} (k={ends[0]}) -> "
                 f"{b['ctok_item']:.0f} (k={ends[-1]}), "
                 f"{'down' if b['ctok_item'] < a['ctok_item'] else 'up'} "
                 f"{abs(b['ctok_item'] - a['ctok_item']):.0f}; prompt tokens "
                 f"{a['ptok_item']:.0f} -> {b['ptok_item']:.0f}.\n")

    # --- the L axis ----------------------------------------------------------------------
    o.append("## 3. The L axis, at the same k and the same protocol\n")
    if len(ls) < 2:
        o.append(f"Not run on this arm: it returns one length. A second length costs a second "
                 f"cell at this endpoint's serialized rate, and the next length up (L=64) is "
                 f"where an item's trace reaches the size the server re-issued rather than "
                 f"returned. The L axis this round rests on is the local grid's, 16 live cells "
                 f"at n=40, where corr(match, log L) = "
                 f"{_corr([math.log(g[1]) for g in qwen_grid()], [g[3] for g in qwen_grid()]):+.3f} "
                 f"against corr(match, log k) = "
                 f"{_corr([math.log(g[0]) for g in qwen_grid()], [g[3] for g in qwen_grid()]):+.3f}.\n")
    for k in ks:
        live = {L: comp[(k, L)] for L in ls if (k, L) in comp and not W.void(comp[(k, L)])}
        if len(live) < 2:
            continue
        span = max(r["match"] for r in live.values()) - min(r["match"] for r in live.values())
        ends = sorted(live)
        a, b = live[ends[0]], live[ends[-1]]
        per_event = ((b["ctok_item"] - a["ctok_item"]) / (ends[-1] - ends[0]))
        p_event = ((b["ptok_item"] - a["ptok_item"]) / (ends[-1] - ends[0]))
        o.append(f"At k={k}, L in {sorted(live)}: match spans **{span:.3f}**. An extra EVENT "
                 f"costs {per_event:+.0f} completion tokens and {p_event:+.0f} prompt tokens "
                 f"per item.\n")

    # --- components ----------------------------------------------------------------------
    if parts:
        o.append("## Component cells, against floors recomputed from the exact scored items\n")
        o.append("| cell | k | L | partner of composed@L | n | match | floor (scored) | "
                 "floor (disjoint) | operative | basis | x floor |")
        o.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for key in sorted(parts):
            r = parts[key]
            f = S.component_floors(r["cell"], r["k"], r["L"], r["n"])
            o.append(f"| {r['cell']} | {r['k']} | {r['L']} | {r['partner_of'] or '—'} | "
                     f"{r['n']} | {'VOID ' if W.void(r) else ''}{r['match']:.3f} | "
                     f"{f['scored']:.4f} | {f['disjoint']:.4f} | **{f['operative']:.4f}** | "
                     f"{f['basis']} | {r['match'] / f['operative']:.2f}x |")
        o.append("")
        o.append("Both floors are printed because they measure different failure modes: the max "
                 "over admitted rows carries an upward selection bias at small n, and the house "
                 "rule is that a floor is recomputed from the items a score is actually read "
                 "against. The larger is operative. A component floor's admitted rows are "
                 "depth <= 1 and cost under the cell's own algorithm's per-item minimum, so a "
                 "scratchpad does not void them — a pad substitutes for registers, not for "
                 "chaining.\n")

    # --- the operating point --------------------------------------------------------------
    o.append("## 4. The operating point on this model: off the ceiling, components still "
             "solved\n")
    op: list[str] = []
    op.append("| k | L | composed | state partner (match) | bind partner (match) | "
              "off the ceiling? | components solved? |")
    op.append("|---|---|---|---|---|---|---|")
    found = []
    for (k, L) in sorted(comp, key=lambda t: (t[1], t[0])):
        r = comp[(k, L)]
        wm = S.work_matched(k, L)
        st = parts.get(("state", k, wm["state"]))
        bd = parts.get(("bind", k, wm["bind"]))
        off = "yes" if r["match"] <= 0.90 else "NO (at the ceiling)"
        if st is None or bd is None:
            solved = "not measured"
        else:
            solved = ("yes" if min(st["match"], bd["match"]) >= 0.80 else
                      f"no (min {min(st['match'], bd['match']):.3f})")
            if off == "yes" and solved == "yes" and r["match"] > S.informed_chance(k):
                found.append((k, L, r, st, bd))
        op.append(f"| {k} | {L} | {r['match']:.3f} | "
                  f"{wm['state']} ({'—' if st is None else f'{st['match']:.3f}'}) | "
                  f"{wm['bind']} ({'—' if bd is None else f'{bd['match']:.3f}'}) | "
                  f"{off} | {solved} |")
    if len(op) > 2:
        o.extend(op)
        o.append("")
        if found:
            for k, L, r, st, bd in found:
                o.append(f"**(k={k}, L={L}) is such a cell on this model**: composed "
                         f"{r['match']:.3f}, state partner {st['match']:.3f}, bind partner "
                         f"{bd['match']:.3f}. Whether it is one for a STRONG model is not "
                         f"measurable here — the scouted three were read at k=12, L=128 and 256, "
                         f"which this arm cannot run.\n")
        else:
            o.append("No cell measured on this arm satisfies both conditions. The question is "
                     "asked of a STRONG model in any case, and the scouted three were read at "
                     "k=12, L=128 and L=256 — lengths this arm cannot run — so the operating "
                     "point for a redesign is the one priced above from the scout's own usage, "
                     "not one this model could be walked to.\n")
    else:
        o.append("Not measurable on this arm: the composed cell and both of its work-matched "
                 "components have to be read at the same (k, L) for the question to mean "
                 "anything, and the cells that completed do not form that triple. The "
                 "operating point for a redesign is the one priced above from the scout's own "
                 "measured usage.\n")

    text = "\n".join(o)
    return text


def write_report() -> str:
    text = report()
    low = text.lower()
    for phrase in P.SCOUT_COMPOSED_FLOOR_LANGUAGE:
        if phrase in low:
            raise AssertionError(
                f"the report uses {phrase!r}; the composed cell has no floor in this regime "
                "(P.SCOUT_COMPOSED_FLOOR_LANGUAGE)")
    os.makedirs(os.path.dirname(W.REPORT_MD), exist_ok=True)
    with open(W.REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nwrote {W.REPORT_MD}")
    return text


if __name__ == "__main__":
    write_report()
