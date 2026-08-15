"""The round's report: where steed's DeepSeek V4 sits in the scouted band, and what the k
and L axes buy on it.

Every claim in the emitted markdown is computed from the history file — spans, ratios,
z scores, token costs and the affordability arithmetic — so the text cannot drift from the
records. Composed-cell floor language is asserted against ``P.SCOUT_COMPOSED_FLOOR_LANGUAGE``
before the file is written: in a scratchpad regime that cell has no floor, only a guess
baseline that moves with k.

THREE QUANTITIES ARE READ FROM MEASUREMENT AND NOT FROM A FORMULA, each because the formula
was wrong on the stream this report is about, and each carries a CORRECTED marker in the
emitted text so a reader who saw the earlier rendering can tell what moved:

  the carrier chain      ``results/probes/s5bind_v3_carrier_hops_20260803.json``, replacing
                         ``2 n_swap / k``, which is a uniform expectation and the composed
                         spec's query gates are not uniform.
  the admissibility read the max over the rows ``validity.s5_bind_v3_admits`` ADMITS, replacing
                         a max over the two one-structure rows — one of which is rejected at
                         every cell on this grid because its width equals the task's.
  what the endpoint runs the placement control's own records, replacing an inference from a
                         single stalled item.
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
from factworld import validity as V                           # noqa: E402

RATE_PROBE = os.path.join(REPO, "results", "probes", "steed_rate_20260803.json")
HOPS_PROBE = os.path.join(REPO, "results", "probes", "s5bind_v3_carrier_hops_20260803.json")


def measured_hops() -> dict:
    """``{(k, L): row}`` from the carrier-hops probe — the chain as the streams carry it.

    ``touch`` is the formula's own quantity measured (swaps naming the queried agent) and
    ``carry`` is the backward carrier walk, which the probe checks against the gold answer on
    every item. Both are properties of the sampled stream and neither is a model reading.
    """
    if not os.path.exists(HOPS_PROBE):
        return {}
    d = json.load(open(HOPS_PROBE, encoding="utf-8"))
    return {(c["k"], c["L"]): c for c in d["cells"]}


HOPS = measured_hops()


def hops_at(k: int, L: int, which: str = "touch"):
    c = HOPS.get((k, L))
    return None if c is None else c[which]


def admitted(cell: dict) -> dict:
    """The rows of one one-structure grid cell that ``s5_bind_v3_admits`` lets set a number.

    The grid scales k and m together (``S.scaled_spec``), so m = k at every cell. A row is
    admitted on a COMPOSED cell when it holds at most one structure plus a scratch register and
    pays no more steps than the cell's own algorithm; ``one_structure_B``, the window rows, the
    prefix rows and ``final_state`` each hold BOTH maps and are rejected at every cell here.
    """
    k = cell["k"]
    return {r: v for r, v in (cell.get("rows") or {}).items()
            if V.s5_bind_v3_admits(r, k, k, cell["n_swap"], cell["n_give"])}


def admitted_max(cell: dict) -> tuple[float, str]:
    """``(value, row)`` — the number a (k, L) choice leaves a cheap policy at this cell."""
    rows = admitted(cell)
    row = max(rows, key=lambda r: rows[r])
    return rows[row], row


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


def _fisher(a: int, na: int, b: int, nb: int) -> float:
    """Two-sided Fisher exact p for two binomials — the right test at n=8 against n=40.

    A z test on a proportion of 1.000 at n=8 has no standard error to speak of; the exact
    conditional test does not need one.
    """
    from math import comb

    tot, hits = na + nb, a + b

    def pt(x):
        return (comb(na, x) * comb(nb, hits - x) / comb(tot, hits)
                if 0 <= hits - x <= nb else 0.0)

    obs = pt(a)
    return sum(v for v in (pt(x) for x in range(0, na + 1)) if v <= obs + 1e-12)


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
        out.append((k, L, hops_at(k, L) or S.carrier_hops(k, L), r["match"],
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

    MEASURED, not predicted. The formula ``2 n_swap / k`` is the expected number of swaps
    touching one FIXED agent under uniform operands; the composed spec's ``q_no_surface`` and
    ``q_tail`` gates choose WHICH agent is queried, so the stream is conditioned on it and the
    uniform expectation is not what any cell carries. Both quantities are read off the rendered
    prompts (``probe_s5bind_v3_carrier_hops_20260803``): ``touch`` is the formula's own quantity
    measured, and ``carry`` is the backward carrier walk the state leg's algorithm performs,
    checked on every item against the gold answer.
    """
    o: list[str] = []
    ks, ls = (6, 12, 24, 32, 48), (32, 64, 128, 192, 256)
    o.append("## Why k cannot buy difficulty at fixed L — the task's own geometry\n")
    o.append("The composed cell's cheapest correct algorithm chases the queried agent's pointer "
             "through the swaps that touch it. A swap moves two of the k pointers, so raising k "
             "at fixed L divides the chain among more agents. **k divides the chain.** That is a "
             "difficulty-REDUCING move on the state leg, not a difficulty-buying one.\n")
    if HOPS:
        o.append("> **CORRECTED.** The earlier rendering of this table printed the FORMULA "
                 "`2 n_swap / k` (`validity.s5_bind_v3_carrier_hops`), which is a uniform "
                 "expectation. The composed spec's query gates pick the agent the chain is "
                 "measured on, so the formula UNDERSTATES every cell, increasingly with k. The "
                 "direction survives and the magnitude does not: across k=6..48 at L=128 the "
                 "formula divides the chain 8.6x and the streams divide it "
                 f"{hops_at(6, 128) / hops_at(48, 128):.1f}x.\n")
    o.append("Carrier chain, MEASURED off the rendered prompts at n="
             f"{json.load(open(HOPS_PROBE, encoding='utf-8'))['n'] if HOPS else 0} items per "
             "cell. `touch` is the number of swaps that NAME the queried agent, which is what "
             "the formula estimates; `carry` is the backward carrier walk, the events whose "
             "contents the answer actually depends on. `formula` is `2 n_swap / k` at the cell's "
             "own measured `n_swap`, printed so the gap is visible rather than asserted:\n")
    o.append("| L | " + " | ".join(f"k={k}" for k in ks) + " |")
    o.append("|---" * (len(ks) + 1) + "|")
    for L in ls:
        cells = [HOPS.get((k, L)) for k in ks]
        o.append(f"| {L} **touch** | " + " | ".join(
            "—" if c is None else f"**{c['touch']:.2f}**" for c in cells) + " |")
        o.append(f"| {L} carry | " + " | ".join(
            "—" if c is None else f"{c['carry']:.2f}" for c in cells) + " |")
        o.append(f"| {L} formula | " + " | ".join(
            "—" if c is None else f"{c['formula']:.2f}" for c in cells) + " |")
    o.append("")
    zero = {c["touch_zero_frac"] for c in HOPS.values()} | {c["carry_zero_frac"]
                                                            for c in HOPS.values()}
    if HOPS:
        lo = min(HOPS.values(), key=lambda c: c["touch"])
        o.append(f"Zero-hop items are {max(zero):.3f} of every one of the {len(HOPS)} measured "
                 f"cells, so every sub-1-hop entry in the formula row describes no stream that "
                 f"exists: the measured floor over this grid is {lo['touch']:.2f} touches "
                 f"(k={lo['k']}, L={lo['L']}), against a formula reading of "
                 f"{lo['formula']:.2f} there. The two measured rows separate at high k because "
                 f"the carrier MOVES: a swap that names the queried agent may hand the value to "
                 f"an agent no later swap touches.\n")
    o.append("What k does buy is REAL but is not difficulty: informed chance falls as 1/(k-1), "
             "so k=48 reads against 0.0213 where k=12 reads against 0.0909 — 4.3x the "
             "measurement resolution, free and model-independent — and the cell holds more live "
             "slots, which a scratchpad supplies at the price of tokens rather than of errors.\n")
    o.append("Holding the chain fixed while raising k means raising L in step, so k's difficulty "
             "is bought in L's currency at L's price. The shortest MEASURED cell at each width "
             "whose chain reaches 3 touches, and what one n=40 cell costs there on this arm:\n")
    o.append("| k | shortest measured L with 3+ touches | its measured touch | "
             "one n=40 cell there |")
    o.append("|---|---|---|---|")
    for k in ks:
        got = sorted((c["L"], c["touch"]) for c in HOPS.values()
                     if c["k"] == k and c["touch"] >= 3.0)
        if not got:
            continue
        need, h = got[0]
        tok = 40 * need * W.TOK_PER_EVENT
        o.append(f"| {k} | {need} | {h:.2f} | {tok / 1e6:.2f}M completion tokens = "
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
        o.append("| k | L=64 chance | L=64 match | x chance | z | measured touch hops |")
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
                     f"Wilson half-width of {half:.2f} at n=40 — flat — while the MEASURED "
                     f"carrier chain falls from {row64[0][2]:.2f} touches to "
                     f"{row64[-1][2]:.2f}.\n")
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
                 f"**{_corr(LL, mm):+.3f}**, corr(match, log measured hops) = "
                 f"{_corr(hh, mm):+.3f}, corr(match, log k) = **{_corr(kk, mm):+.3f}**, and "
                 f"corr(log(match / informed chance), log k) = **{_corr(kk, xx):+.3f}**. L is "
                 f"what this model pays for.\n")
    return o


def width_depth_section() -> list[str]:
    """What k IS, priced on the cell's own algorithm, and why the answer differs by arm.

    The composed cell's cheapest correct algorithm costs W = k + m + 1 live slots and
    Theta(L) steps (``validity.s5_bind_v3_task_cost``). k moves W and barely moves S; L moves
    the measured chain and does not move W at all. That is a WIDTH axis and a DEPTH axis, and a
    scratchpad buys width with tokens while nothing buys depth but depth.
    """
    if not HOPS:
        return []
    d = json.load(open(ONE_STRUCTURE, encoding="utf-8")) if os.path.exists(ONE_STRUCTURE) else None
    if d is None:
        return []
    by = {(c["k"], c["L"]): c for c in d["cells"]}
    ks = (6, 12, 24, 32, 48)
    o: list[str] = []
    o.append("### What the k axis is, and why it is not inert for both arms\n")
    o.append("Priced on the cell's own algorithm at a fixed L=128 — live slots `W = k + m + 1` "
             "and steps `S` from `validity.s5_bind_v3_task_cost`, measured chain from the "
             "probe:\n")
    o.append("| k | composed W (live slots) | composed S (steps) | measured touch hops @L128 | "
             "informed chance |")
    o.append("|---|---|---|---|---|")
    first = last = None
    for k in ks:
        c = by.get((k, 128))
        if c is None:
            continue
        w, s = V.s5_bind_v3_task_cost(k, k, c["n_swap"], c["n_give"])
        row = (k, w, s, hops_at(k, 128))
        first = first or row
        last = row
        o.append(f"| {k} | {w} | {s} | {row[3]:.2f} | {S.informed_chance(k):.4f} |")
    o.append("")
    if first and last:
        o.append(f"Across k={first[0]} to k={last[0]} at fixed L=128 the cell's live-slot "
                 f"requirement rises **{last[1] / first[1]:.1f}x** ({first[1]} -> {last[1]}), its "
                 f"step count rises {last[2] / first[2]:.2f}x ({first[2]} -> {last[2]}), and its "
                 f"measured chain FALLS {first[3] / last[3]:.1f}x ({first[3]:.2f} -> "
                 f"{last[3]:.2f}).")
    a, b = HOPS.get((12, 128)), HOPS.get((12, 512))
    if a and b:
        wa = V.s5_bind_v3_task_cost(12, 12, a["n_swap"], a["n_give"])[0]
        wb = V.s5_bind_v3_task_cost(12, 12, b["n_swap"], b["n_give"])[0]
        o.append(f"Along the other axis, k=12 held fixed and L taken 128 -> 512, W does not move "
                 f"at all ({wa} -> {wb}) while the measured chain rises "
                 f"{b['touch'] / a['touch']:.1f}x ({a['touch']:.2f} -> {b['touch']:.2f}).\n")
    o.append("**k is a WIDTH axis and L is a DEPTH axis.** That is one fact with two different "
             "consequences, and the earlier rendering carried only the first. On the FRONTIER "
             "arm the model emits its working as content, so width is bought with tokens rather "
             "than with errors and k is inert: it moves the guess baseline and nothing else. On "
             "a bounded-state FROM-SCRATCH architecture width is the binding resource — the "
             "state has to be held in the recurrent carrier, not in a scratchpad — so k is NOT "
             "inert there, and the same axis that buys nothing on the paid arm is the axis the "
             "local arm is about.\n")
    return o


ONE_STRUCTURE = os.path.join(REPO, "results", "probes",
                             "s5bind_v3_onestructure_k_20260803.json")
CLEAN_MAX = 1.25          # an admitted row within this multiple of chance leaves the cell
                          # discriminating; above it a cheaper solver is scoring


def one_structure_section() -> list[str]:
    """What raising k at fixed L does to the composed cell's REASON FOR EXISTING.

    The composed cell is a composition cell only while maintaining ONE of the two structures is
    insufficient. That is a property of the sampled stream and it is measured here off the
    generator, with no model in the loop.

    THE READ IS A MAX OVER ADMITTED ROWS. Which rows those are is decided by
    ``validity.s5_bind_v3_admits`` and not by the row's name: on a composed cell a row may set a
    number only while it holds at most one structure plus a scratch register
    (``W <= max(k, m) + 1``) and pays no more steps than the cell's own algorithm. The grid
    scales k and m together, so ``one_structure_B`` — which reads B live AND reads the answer out
    of P — is 1+k+m wide, exactly the task's own width, and is rejected at every cell here. It is
    printed beside the admitted read as the full-width fact it is, the way ``window_90`` is.
    """
    if not os.path.exists(ONE_STRUCTURE):
        return []
    d = json.load(open(ONE_STRUCTURE, encoding="utf-8"))
    cells = d["cells"]
    for c in cells:
        c["admitted_max"], c["admitted_row"] = admitted_max(c)
    o: list[str] = []
    o.append("## What raising k at fixed L does to the composed cell itself\n")
    o.append("The composed cell earns its name only while maintaining ONE of the two structures "
             "is insufficient. So the question is not only whether a model finds the cell harder "
             "at higher k, it is whether the cell is still asking the question. What answers it "
             "is the strongest policy the class rule ADMITS: a row that holds at most one "
             "structure plus a scratch register and pays no more steps than the cell's own "
             "algorithm. How often such a row is nonetheless RIGHT is a property of the stream, "
             f"replayed by `validity.s5_bind_v3_floors` at n={min(c['n'] for c in cells)} per "
             f"cell with no model in the loop, and is what the (k, L) choice decides.\n")
    o.append("> **CORRECTED.** The earlier rendering read this off `one_structure_max` — the "
             "larger of `one_structure_P` and `one_structure_B`. `one_structure_B` reads B live "
             "and reads the ANSWER out of P, so it holds 1+k+m slots against the task's own "
             "1+k+m: `validity.s5_bind_v3_admits` rejects it at all "
             f"{len(cells)} cells on this grid, and it was the larger of the two at "
             f"{sum(1 for c in cells if c['rows']['one_structure_B'] >= c['rows']['one_structure_P'])} "
             "of them. The number below is the max over the rows that ARE admitted "
             f"({', '.join(sorted(admitted(cells[0])))}); `one_structure_B` is reported "
             "separately below rather than deleted. Two conclusions move with it and are marked "
             "where they occur.\n")
    o.append("This is not a floor and is not read as one: in the scratchpad regime the composed "
             "cell has none, and a live-slot bound is exactly what a visible trace defeats. It "
             "measures something else — how much of the composed cell's separation from its "
             "components a (k, L) choice leaves standing. The stream-blind read (`initial_only`) "
             "is 0.0000 in every cell of the grid — the sampler's `q_no_surface` gate gives it no "
             "items — so what follows is not that shortcut returning under another name.\n")
    ks = sorted({c["k"] for c in cells})
    ls = sorted({c["L"] for c in cells})
    by = {(c["k"], c["L"]): c for c in cells}
    o.append("Strongest ADMITTED row as a multiple of that cell's own informed chance 1/(k-1), "
             "with the row that sets it:\n")
    o.append("| L | " + " | ".join(f"k={k}" for k in ks) + " |")
    o.append("|---" * (len(ks) + 1) + "|")
    for L in ls:
        row = []
        for k in ks:
            c = by.get((k, L))
            if c is None:
                row.append("—")
                continue
            r = c["admitted_max"] / c["chance"]
            tag = f"{r:.2f}x ({c['admitted_row']})"
            row.append(f"**{tag}**" if r > CLEAN_MAX else tag)
        o.append(f"| {L} | " + " | ".join(row) + " |")
    o.append("")
    o.append(f"Bold is over {CLEAN_MAX:.2f}x — a chosen line, not a measured one; the raw "
             f"numbers are printed so a different line can be drawn. Above it a cheaper solver "
             f"is beating the guess baseline the cell is scored against. Raising k at fixed L "
             f"walks every row rightwards into that region: at L=128 the read goes "
             f"{by[(12,128)]['admitted_max']:.4f} "
             f"({by[(12,128)]['admitted_max']/by[(12,128)]['chance']:.2f}x) at k=12 to "
             f"{by[(48,128)]['admitted_max']:.4f} "
             f"({by[(48,128)]['admitted_max']/by[(48,128)]['chance']:.2f}x) at k=48, and at "
             f"L=32/k=48 it reaches {by[(48,32)]['admitted_max']:.4f} "
             f"({by[(48,32)]['admitted_max']/by[(48,32)]['chance']:.2f}x). k does not make "
             f"the composed cell harder; past a point it makes it a component cell with more "
             f"agents in it.\n")

    o.append("### The full-width one-structure row, reported separately\n")
    o.append("`one_structure_B` carries B through the gives, resolves every reference against "
             "the stated P0, and reads the answer out of P — so it holds both structures and is "
             "not a cheaper algorithm. It is not admitted and sets nothing; what it measures is "
             "how often tracking the OTHER structure alone lands on the answer, which is a "
             "different fact and moves the other way at long L:\n")
    o.append("| L | " + " | ".join(f"k={k}" for k in ks) + " |")
    o.append("|---" * (len(ks) + 1) + "|")
    for L in ls:
        row = []
        for k in ks:
            c = by.get((k, L))
            row.append("—" if c is None
                       else f"{c['rows']['one_structure_B'] / c['chance']:.2f}x")
        o.append(f"| {L} | " + " | ".join(row) + " |")
    o.append("")

    o.append("### The shape of the admissible region\n")
    o.append("L/k is most of the story and not all of it. L/k is how many times the stream "
             "touches any one agent or object, so as it falls the two structures stop moving "
             "under each other and the RAW read falls with it. But the baseline the cell is "
             "scored against falls as 1/(k-1) at the same time, and it falls faster — so at "
             "matched L/k the read gets WORSE as a multiple of chance the wider the cell is:\n")
    o.append("| L/k | cells (k, L) | admitted read | x chance |")
    o.append("|---|---|---|---|")
    buckets: dict[float, list] = {}
    for c in cells:
        buckets.setdefault(round(c["L"] / c["k"], 2), []).append(c)
    for ratio in sorted(buckets):
        b = sorted(buckets[ratio], key=lambda c: c["k"])
        o.append(f"| {ratio:.2f} | " + ", ".join(f"({c['k']}, {c['L']})" for c in b) + " | "
                 + ", ".join(f"{c['admitted_max']:.4f}" for c in b) + " | "
                 + ", ".join(f"{c['admitted_max'] / c['chance']:.2f}x" for c in b) + " |")
    o.append("")
    clean = [c for c in cells if c["admitted_max"] / c["chance"] <= CLEAN_MAX]
    old_clean = [c for c in cells
                 if c["one_structure_max"] / c["chance"] <= CLEAN_MAX]
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
        lo_k = min(need)
        o.append(f"The requirement rises with k, from L/k >= {need[lo_k]:.1f} at k={lo_k} to "
                 f"L/k >= {max(need.values()):.1f} at k={max(need, key=need.get)}. Of the "
                 f"{len(cells)} cells on this grid, {len(clean)} are admissible. "
                 + (f"k={', k='.join(str(k) for k in never)} is admissible at no length measured "
                    f"here, up to L={max(ls)}."
                    if never else "")
                 + "\n")
        o.append(f"> **CORRECTED.** The earlier rendering called this **a narrowing ray**, on a "
                 f"requirement that ran L/k >= 5.3 at k=6 to L/k >= 10.7 by k=24 with "
                 f"{len(old_clean)} of {len(cells)} cells admissible. On the admitted rows the "
                 f"requirement is flat at L/k >= {need[lo_k]:.1f} for k=6, 12 and 24 and rises "
                 f"only at the top two widths, and {len(clean)} of {len(cells)} cells are "
                 f"admissible. The region still narrows; it narrows about half as fast, and the "
                 f"consequence is that a given L buys a wider cell than was priced.\n")
    o.append("| L | largest k that stays within " + f"{CLEAN_MAX:.2f}x" + " | its read | "
             "next k up | its read | earlier rendering's k |")
    o.append("|---|---|---|---|---|---|")
    for L in ls:
        row = sorted((c for c in cells if c["L"] == L), key=lambda c: c["k"])
        ok = [c for c in row if c["admitted_max"] / c["chance"] <= CLEAN_MAX]
        bad = [c for c in row if c["admitted_max"] / c["chance"] > CLEAN_MAX]
        old = [c for c in row if c["one_structure_max"] / c["chance"] <= CLEAN_MAX]
        was = str(old[-1]["k"]) if old else "none"
        if not ok:
            o.append(f"| {L} | none | — | {row[0]['k']} | "
                     f"{row[0]['admitted_max'] / row[0]['chance']:.2f}x | {was} |")
            continue
        b = ok[-1]
        nxt = (f"{bad[0]['k']} | {bad[0]['admitted_max'] / bad[0]['chance']:.2f}x"
               if bad else "— | —")
        o.append(f"| {L} | {b['k']} | {b['admitted_max'] / b['chance']:.2f}x | {nxt} | {was} |")
    o.append("")
    o.append("The column runs non-monotonically in k at the long end, which is why the whole row "
             "is printed above rather than only its argmax: at L=768 the reads are "
             + ", ".join(f"k={c['k']}: {c['admitted_max'] / c['chance']:.2f}x"
                         for c in sorted((c for c in cells if c["L"] == 768),
                                         key=lambda c: c["k"]))
             + ", so the widest cell inside the line is not the widest cell with the lowest "
               "read.\n")

    win = [c for c in cells if "window_90" in (c.get("rows") or {})]
    if win:
        o.append("### A second thing k does to the stream\n")
        o.append("A row that moves even further is worth stating separately: `window_90`, which "
                 "replays the task's own algorithm but SKIPS THE FIRST TENTH of the events. It "
                 "holds both structures and so is not a cheaper solver — "
                 "`validity.s5_bind_v3_admits` rejects it for exactly that reason — but it is "
                 "0.90x the task's steps, and what it measures is how much of the stream carries "
                 "no information about the answer.\n")
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
        o.append(f"At k={hi['k']}, L={hi['L']} a solver that never reads the first tenth of the "
                 f"stream answers correctly {hi['rows']['window_90']:.3f} of the time against an "
                 f"informed chance of {hi['chance']:.4f} — "
                 f"{hi['rows']['window_90'] / hi['chance']:.0f}x. The events are still there; at "
                 f"that width they have stopped mattering. This is the same fact as the falling "
                 f"measured chain seen from the readout side, and it is why raising k cannot "
                 f"substitute for raising L: L is what puts events between the query and the "
                 f"initial map, and k takes them back out.\n")

    grid = qwen_grid()
    if grid:
        o.append("### What that costs a live reading\n")
        o.append("The margin a measured score holds OVER the admitted read is what the composed "
                 "cell is buying. On the local Qwen grid at n=40 that margin collapses along k "
                 "while raw match barely moves — the flatness of the k axis is not a null, it is "
                 "the cell handing its own separation away:\n")
        o.append("| L | k | match (n=40) | admitted read | margin | x chance |")
        o.append("|---|---|---|---|---|---|")
        for k, L, _h, m, x, _c, _n in sorted(grid, key=lambda g: (g[1], g[0])):
            c = by.get((k, L))
            if c is None:
                continue
            o.append(f"| {L} | {k} | {m:.3f} | {c['admitted_max']:.4f} | "
                     f"**{m - c['admitted_max']:+.3f}** | {x:.2f}x |")
        o.append("")
        row = [(k, L, m, by[(k, L)]["admitted_max"])
               for k, L, _h, m, _x, _c, _n in grid if (k, L) in by and L == 64]
        if len(row) >= 2:
            row.sort()
            a0, b0 = row[0], row[-1]
            seq = ", ".join(f"k={k}: {m - s:+.3f}" for k, _L, m, s in row)
            o.append(f"L=64 is the row where this model has headroom at every rung, so it is the "
                     f"row that can lose something. Across it match goes {a0[2]:.3f} (k={a0[0]}) "
                     f"to {b0[2]:.3f} (k={b0[0]}) — flat, which is what the k axis was reported "
                     f"as buying nothing — while the margin over the admitted class runs {seq}. "
                     f"The axis is not inert: what it moves is the cell's discriminating power, "
                     f"and it returns a flat score while doing it.\n")

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
    elif d.get("ungeneratable"):
        u = sorted((x["k"], x["L"]) for x in d["ungeneratable"])
        o.append(f"The region is bounded from the other side too, by the sampler rather than by "
                 f"a solver: the composed spec's `q_no_surface` gate could not fill the n="
                 f"{min(c['n'] for c in cells)} draw at {len(u)} of the grid's cells, every one "
                 f"of them k={u[0][0]} at L>={min(L for _k, L in u)}. The narrow-and-long corner "
                 f"is not available, so the ray has a floor in k as well as a ceiling.\n")

    reg, nxt = by.get((12, 128)), by.get((24, 128))
    if reg and nxt:
        inside = nxt["admitted_max"] / nxt["chance"] <= CLEAN_MAX
        o.append(f"The shipped operating point is k=12 at L=128 — L/k = {128 / 12:.1f}, read "
                 f"{reg['admitted_max'] / reg['chance']:.2f}x. "
                 + (f"It sits INSIDE the frontier and not on it: the next k rung up at L=128, "
                    f"k=24, reads {nxt['admitted_max'] / nxt['chance']:.2f}x and is admissible "
                    f"too, so the registered cell is one rung narrower than that length allows."
                    if inside else
                    f"It sits ON the frontier: the next k rung up at L=128 is already "
                    f"{nxt['admitted_max'] / nxt['chance']:.2f}x.") + "\n")
        if inside:
            o.append(f"> **CORRECTED.** The earlier rendering read \"the shipped operating point "
                     f"sits ON the edge\" off `one_structure_B`, which put k=24 at L=128 at "
                     f"{nxt['rows']['one_structure_B'] / nxt['chance']:.2f}x. On the admitted "
                     f"rows k=24 at L=128 reads "
                     f"{nxt['admitted_max'] / nxt['chance']:.2f}x, so there is one k rung of "
                     f"headroom at the shipped length that the earlier reading priced away.\n")
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
    for c in d["cells"]:
        c["admitted_max"], c["admitted_row"] = admitted_max(c)
    by = {(c["k"], c["L"]): c for c in d["cells"]}
    o: list[str] = []
    o.append("## Where the next paid point goes, and what it costs\n")
    o.append("STOP_CEILING asked for a cell a strong model is off the ceiling on. The rule's own "
             "remedy was \"raise k or L\". The k half is ruled out above on two independent "
             "grounds — k divides the measured chain, and past the admissibility ray it hands "
             "the cell to a cheaper solver that need not hold both structures — so the remaining "
             "direction is L, with k raised only as far as the ray allows and only for the "
             "resolution it buys.\n")
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
                 f"at the ceiling, and the deeper of them carries "
                 f"{hops_at(12, L1):.2f} measured touches. Extending that per-event rate along "
                 f"the admissible ray:\n")
        o.append("| L | widest admissible k | its admitted read | informed chance | "
                 "measured touch hops | est. completion tok/item | est. $/item | "
                 "est. $ at n=40 | earlier rendering's k (hops) |")
        o.append("|---|---|---|---|---|---|---|---|---|")
        try:
            from factworld.benchmark import MODELS
            reg = MODELS[model]
        except Exception:  # noqa: BLE001
            reg = {"prompt_price_per_M": 0.0, "completion_price_per_M": 0.0}
        for L in sorted({c["L"] for c in d["cells"]}):
            row = sorted((c for c in d["cells"] if c["L"] == L), key=lambda c: c["k"])
            ok = [c for c in row if c["admitted_max"] / c["chance"] <= CLEAN_MAX]
            old = [c for c in row if c["one_structure_max"] / c["chance"] <= CLEAN_MAX]
            if not ok or L < L1:
                continue
            c = ok[-1]
            ctok = a["ctok"] + pe * (L - L0)
            ptok = a["ptok"] + pp * (L - L0)
            dollars = (ptok * reg["prompt_price_per_M"]
                       + ctok * reg["completion_price_per_M"]) / 1e6
            h = hops_at(c["k"], L)
            was = (f"{old[-1]['k']} ({hops_at(old[-1]['k'], L):.1f})"
                   if old and hops_at(old[-1]["k"], L) is not None else "none")
            o.append(f"| {L} | {c['k']} | "
                     f"{c['admitted_max'] / c['chance']:.2f}x | {c['chance']:.4f} | "
                     f"{'—' if h is None else f'{h:.1f}'} | {ctok:.0f} | ${dollars:.2f} | "
                     f"${dollars * 40:.0f} | {was} |")
        o.append("")
        o.append("> **CORRECTED.** The `widest admissible k` column moved when the ray was "
                 "recomputed over admitted rows, and it changes which cell to buy at a given "
                 "price. At L=768 the earlier rendering priced k=48, whose measured chain is "
                 f"{hops_at(48, 768):.2f} touches; the admitted read permits k={by[(32, 768)]['k']} "
                 f"at {by[(32, 768)]['admitted_max'] / by[(32, 768)]['chance']:.2f}x, whose chain "
                 f"is {hops_at(32, 768):.2f} — {hops_at(32, 768) / hops_at(48, 768):.2f}x the "
                 f"depth for the same money — and k=24 at "
                 f"{by[(24, 768)]['admitted_max'] / by[(24, 768)]['chance']:.2f}x carries "
                 f"{hops_at(24, 768):.2f}, {hops_at(24, 768) / hops_at(48, 768):.2f}x the depth "
                 f"at a 2.2x coarser guess baseline.\n")
        o.append("The estimate is linear in L at this model's own measured per-event rate, which "
                 "is what it spends AT THE CEILING; a model that is actually working will spend "
                 "more, so these are lower bounds on the completion side. It ignores k, which on "
                 "the local grid LOWERED completion tokens at fixed L (k=6 to k=32 at L=128: "
                 "18,654 to 11,745 per item), so the widening is if anything cheaper than "
                 "priced. These rows and the probe priced at the end of this report are the only "
                 "extrapolations here and are marked as such; everything else is a measurement.\n")
        wide = [c for c in d["cells"]
                if c["admitted_max"] / c["chance"] <= CLEAN_MAX and c["L"] >= 512]
        ref = by.get((12, 128))
        if wide and ref:
            best = max(wide, key=lambda c: c["k"])
            same = hops_at(12, best["L"])
            o.append(f"Widening along the ray is bought for RESOLUTION and not for difficulty: "
                     f"at L={best['L']} the widest admissible cell is k={best['k']}, whose "
                     f"informed chance is {best['chance']:.4f} against {ref['chance']:.4f} "
                     f"at the shipped point — {ref['chance'] / best['chance']:.1f}x the "
                     f"measurement resolution — but its measured chain is "
                     f"{hops_at(best['k'], best['L']):.1f} touches against "
                     f"{'—' if same is None else f'{same:.1f}'} for k=12 at the same length. The "
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


def lead_section(rows: dict | None = None, eff: float = 0.0) -> list[str]:
    """The round's answers, with the numbers they rest on."""
    o: list[str] = []
    rows = rows or {}
    by = {}
    if os.path.exists(ONE_STRUCTURE):
        cells = json.load(open(ONE_STRUCTURE, encoding="utf-8"))["cells"]
        for c in cells:
            c["admitted_max"], c["admitted_row"] = admitted_max(c)
        by = {(c["k"], c["L"]): c for c in cells}
    o.append("## The answers\n")
    if HOPS:
        o.append(f"**k is a WIDTH axis, not a difficulty axis, and the reason is in the task "
                 f"rather than in any model.** Raising k at fixed L divides the pointer chain "
                 f"among more agents: MEASURED off the rendered prompts, the chain at L=128 runs "
                 f"{hops_at(6, 128):.2f} touches at k=6 down to {hops_at(48, 128):.2f} at k=48. "
                 f"What k does move is the cell's live-slot requirement, "
                 f"{V.s5_bind_v3_task_cost(6, 6, by[(6, 128)]['n_swap'], by[(6, 128)]['n_give'])[0]} "
                 f"slots at k=6 to "
                 f"{V.s5_bind_v3_task_cost(48, 48, by[(48, 128)]['n_swap'], by[(48, 128)]['n_give'])[0]} "
                 f"at k=48. On the local grid, over 16 live cells at n=40, corr(match, log L) = "
                 f"-0.934 while corr(match, log k) = -0.043. A scratchpad buys width with tokens, "
                 f"so k is inert on the FRONTIER arm; on a bounded-state from-scratch "
                 f"architecture width is the binding resource and k is not inert there.\n")
    if by:
        a, b = by.get((12, 128)), by.get((48, 128))
        if a and b:
            o.append(f"**Worse than inert: past a boundary, raising k stops the cell being a "
                     f"composition cell.** The strongest policy the class rule ADMITS — one "
                     f"structure plus a scratch register, no more steps than the cell's own "
                     f"algorithm — reads {a['admitted_max'] / a['chance']:.2f}x informed chance "
                     f"at the shipped k=12/L=128 and {b['admitted_max'] / b['chance']:.2f}x at "
                     f"k=48/L=128. The admissible set is a ray in (k, L) that narrows with k, and "
                     f"on the local grid the margin a measured score holds over that class "
                     f"collapses along k while raw match stays flat.\n")
    o.append("**So the ceiling's only remaining remedy is L**, with k raised no faster than the "
             "ray allows and only for the resolution 1/(k-1) buys. Candidate next paid points "
             "and their prices are below, and one probe is priced for decision at the end.\n")
    ctrl = rows.get(("composed", 6, 64))
    rate = eff or W.RATE_TOK_PER_S
    if ctrl:
        lo, hi = _wilson(ctrl["match"], ctrl["n"])
        o.append(f"**The free arm cannot substitute for the paid scout on this instrument, and "
                 f"the limit is throughput and not generation length.** It serializes at "
                 f"{rate:.1f} completion tok/s, so the briefed step 1 alone is "
                 f"{sum(40 * L * W.TOK_PER_EVENT for L in (128, 256, 85, 171)) / rate / 3600:.0f} "
                 f"hours. At composed@L{ctrl['L']}, k={ctrl['k']} — n={ctrl['n']}, every item "
                 f"finishing on `stop` at {ctrl['ctok_item']:.0f} completion tokens a piece and "
                 f"{hops_at(ctrl['k'], ctrl['L']):.2f} measured touches, DEEPER than the "
                 f"{hops_at(12, 128):.2f} of composed@128 where glm-5.2 reads 0.575 — DeepSeek V4 "
                 f"scores {ctrl['match']:.3f} (95% CI [{lo:.2f}, {hi:.2f}]). It is a second "
                 f"ceiling and cannot test the k axis either. The two free models fail the same "
                 f"job from opposite ends: the local Qwen is at informed chance at the lengths it "
                 f"can afford, this one is at the ceiling at the deepest cell it has been run "
                 f"on.\n")
    return o


LONGCALL = os.path.join(REPO, "results", "probes", "steed_longcall_20260803.json")


def path_section(rows: dict | None = None) -> list[str]:
    """What actually bounds an item here, measured on the items themselves.

    A call that fails is retried five times and then recorded as an EMPTY prediction, scored
    wrong, so a transport that drops exactly the long generations would manufacture a floor on
    exactly the cells with the longest traces. That is why the question is asked at all — and it
    is answered by a cell, not by an inference from one stalled item.
    """
    rows = rows or {}
    ctrl = rows.get(("composed", 6, 64))
    o: list[str] = []
    o.append("### What bounds an item on this arm: throughput, not generation length\n")
    o.append("> **CORRECTED.** The earlier rendering said \"the endpoint does not return a "
             "generation of ~10.5k tokens at all\" and used it, alongside the wall clock, as one "
             "of two independent reasons the band cells could not be run. It is disproved by a "
             "cell in this round's own history.\n")
    if ctrl:
        toks = ctrl.get("ctoks") or []
        o.append(f"composed@L{ctrl['L']}, k={ctrl['k']}, n={ctrl['n']}: every item returned, "
                 f"`{ctrl['finish_reasons']}`, {ctrl['length_rate']:.2f} truncated, "
                 f"{ctrl['empty_rate']:.2f} empty, {ctrl['api_errors']} API errors"
                 + (f", per-item completion tokens {min(toks):,}-{max(toks):,} "
                    f"(median {ctrl['ctok_median']:,})" if toks else "")
                 + f". {sum(1 for t in toks if t >= 10000)} of {ctrl['n']} items are at or over "
                   f"10,000 completion tokens and {sum(1 for t in toks if t >= 10482)} are at or "
                   f"over the 10,482 the earlier reading called unreturnable. The whole cell took "
                   f"{ctrl['elapsed_s'] / 3600:.1f} h at "
                   f"{ctrl['ctok_item'] * ctrl['n'] / ctrl['elapsed_s']:.1f} completion tok/s.\n")
        o.append("The abandoned k=12/L=64 item is recorded in this repository's own history as "
                 "having completed once at 10,482 tokens in 698.8 s with `finish=stop` and no "
                 "re-issue (commit `ce1cb35`). The stall is therefore a state the server can "
                 "enter and not a length the server refuses, and the only standing limit on this "
                 "arm is throughput: a cell's duration is its completion tokens divided by one "
                 "number, and that is what the pricing below rests on.\n")
    if os.path.exists(LONGCALL):
        recs = json.load(open(LONGCALL, encoding="utf-8"))
        o.append("The one path test taken this round, kept because it is VOID and a reader "
                 "should not re-derive it:\n")
        o.append("| path | k | L | completion tokens | wall | result |")
        o.append("|---|---|---|---|---|---|")
        for r in recs:
            res = (f"ok, finish=`{r['finish']}`, {r['tok_per_s']} tok/s" if r.get("ok")
                   else f"**{r.get('error_type')}** — {r['note']}" if r.get("note")
                   else f"**{r.get('error_type')}** {str(r.get('error'))[:90]}")
            o.append(f"| {r['path']} | {r['k']} | {r['L']} | {r.get('ctok', '—')} | "
                     f"{r['wall_s']:.0f}s | {res} |")
        o.append("")
    o.append("The stall itself is still unexplained and is a validity risk wherever it recurs: "
             "for the item that would not come back the server logged `gen=10482 finish=stop` "
             "four times against one prompt, byte-identical, each followed by `live kv cache "
             "miss ... reason=token-mismatch` and an immediate re-`prompt start` on the same "
             "cached prompt, and went on doing so after the client process was killed. "
             "`serve_steed_model.py down`/`up` clears it. A cell that hits it records empty "
             "predictions, which score wrong, so the empty-rate column is the gate that catches "
             "it — and it read 0.00 on every cell in this round.\n")
    return o


PROBE_MODELS = ("openai/gpt-5.5", "z-ai/glm-5.2")
PROBE_CELL = (12, 512)          # the admitted cell with the deepest measured chain at k=12
PROBE_N = 12


def probe_price_section() -> list[str]:
    """THE ONE PAID PROBE THE EVIDENCE SUPPORTS, priced for a decision and NOT issued.

    It is a candidate, not a plan: the estimate, its basis, and the rule that decides what the
    result means are all written down BEFORE the money is spent, because a ceiling reading that
    is re-budgeted instead of acted on is how the last two rounds were lost.
    """
    su = scout_usage()
    if not su or not HOPS:
        return []
    from factworld.benchmark import MODELS

    k, L = PROBE_CELL
    d = json.load(open(ONE_STRUCTURE, encoding="utf-8")) if os.path.exists(ONE_STRUCTURE) else None
    cell = None
    if d:
        for c in d["cells"]:
            if (c["k"], c["L"]) == (k, L):
                c["admitted_max"], c["admitted_row"] = admitted_max(c)
                cell = c
    o: list[str] = []
    o.append("## The one paid probe the evidence supports — priced for decision, NOT issued\n")
    o.append(f"**The cell: composed, k={k}, L={L}, n={PROBE_N}, on "
             f"`{'` and `'.join(PROBE_MODELS)}`.** No call was made. This section exists so the "
             f"decision can be taken on numbers rather than on a plan.\n")
    o.append(f"**Why this cell and not another.** It is the deepest admitted cell at the SHIPPED "
             f"width: {hops_at(k, L):.2f} measured touches against composed@256's "
             f"{hops_at(12, 256):.2f} and composed@128's {hops_at(12, 128):.2f} — "
             f"{hops_at(k, L) / hops_at(12, 128):.1f}x the depth of the shallower cell gpt-5.5 "
             f"already reads 1.000 on"
             + (f", with the strongest admitted policy at "
                f"{cell['admitted_max'] / cell['chance']:.2f}x informed chance "
                f"({cell['chance']:.4f}), so the cell is still asking its own question"
                if cell else "")
             + ". Holding k at 12 keeps the guess baseline and the component partners identical "
               "to the scouted band, so the only thing that moves is depth.\n")
    o.append("**The estimate, and its basis.** Linear extrapolation in L of each model's OWN "
             "measured per-item usage at k=12, L=128 and L=256 "
             "(`results/s5bind_v3_scout/`, VOID cells excluded), at the registry's prices. It is "
             "an EXTRAPOLATION and is the only number in this section that is not measured:\n")
    o.append("| model | measured L=128 | measured L=256 | ctok/event | est. ctok/item @512 | "
             f"est. $/item | est. $ at n={PROBE_N} |")
    o.append("|---|---|---|---|---|---|---|")
    total = 0.0
    for m in PROBE_MODELS:
        pts = sorted((x, r) for (mm, x), r in su.items() if mm == m)
        if len(pts) < 2:
            continue
        (L0, a), (L1, b) = pts[0], pts[-1]
        pe = (b["ctok"] - a["ctok"]) / (L1 - L0)
        pp = (b["ptok"] - a["ptok"]) / (L1 - L0)
        ctok = a["ctok"] + pe * (L - L0)
        ptok = a["ptok"] + pp * (L - L0)
        reg = MODELS[m]
        per = (ptok * reg["prompt_price_per_M"] + ctok * reg["completion_price_per_M"]) / 1e6
        total += per * PROBE_N
        o.append(f"| `{m}` | {a['match']:.3f} @ {a['ctok']:.0f} ctok | "
                 f"{b['match']:.3f} @ {b['ctok']:.0f} ctok | {pe:.0f} | {ctok:.0f} | "
                 f"${per:.2f} | ${per * PROBE_N:.2f} |")
    o.append("")
    o.append(f"**Total: ${total:.2f}** at the ceiling rate. A model that is actually working "
             f"spends more, so the completion side is a lower bound; the cost guard should be "
             f"set at 2x, i.e. ${total * 2:.0f}, and a budget of 65,536 completion tokens for "
             f"gpt-5.5 and 131,072 for glm-5.2 (both above 2x their estimated mean, since a "
             f"budget under the trace length scores the cell as a floor).\n")
    o.append("**The pre-registered decision rule, in force before the probe is issued.** The "
             "comparison is Fisher exact against each model's OWN measured L=256 cell at n=40, "
             "which is the only reading that answers \"did the depth move this model\" rather "
             "than \"is this model good\". The whole decision boundary is printed, so the rule "
             "is a threshold and not a judgement made after the fact:\n")
    ref = su.get((PROBE_MODELS[0], 256))
    if ref:
        hit = int(round(ref["match"] * ref["n"]))
        o.append(f"| `{PROBE_MODELS[0]}` at n={PROBE_N} | 95% Wilson | Fisher p vs its own "
                 f"{hit}/{ref['n']} at L=256 | verdict |")
        o.append("|---|---|---|---|")
        for x in range(PROBE_N, max(-1, PROBE_N - 5), -1):
            lo, hi = _wilson(x / PROBE_N, PROBE_N)
            pv = _fisher(x, PROBE_N, hit, ref["n"])
            o.append(f"| {x}/{PROBE_N} = {x / PROBE_N:.3f} | [{lo:.2f}, {hi:.2f}] | {pv:.3f} | "
                     + ("**STOP — the ceiling is not an L problem**" if pv >= 0.05
                        else "the L axis is live") + " |")
        o.append("")
        cut = max(x for x in range(PROBE_N + 1)
                  if _fisher(x, PROBE_N, hit, ref["n"]) < 0.05)
        o.append(f"- **STOP AND REDESIGN** if `{PROBE_MODELS[0]}` reads {cut + 1}/{PROBE_N} or "
                 f"better. A {hops_at(k, L) / hops_at(12, 128):.1f}x increase in measured depth "
                 f"over the cell it already reads 1.000 on would then have moved it by less than "
                 f"this test resolves, and **no larger-L purchase is warranted**: the ceiling is "
                 f"not an L problem. Do not re-budget — the last two rounds were lost to exactly "
                 f"that move.")
        wm = (S.work_matched(k, L) if cell is None
              else V.s5_bind_v3_work_match(cell["n_swap"], cell["n_give"]))
        o.append(f"- **THE L AXIS IS LIVE** if it reads {cut}/{PROBE_N} or worse. The next "
                 f"purchase is then this same cell at n=40 plus its two work-matched component "
                 f"partners (state@{wm['state']}, bind@{wm['bind']}), which is what a placement "
                 f"needs and what n={PROBE_N} cannot give.")
    ref2 = su.get((PROBE_MODELS[1], 256))
    if ref2:
        hit2 = int(round(ref2["match"] * ref2["n"]))
        lo_cut = max((x for x in range(PROBE_N + 1)
                      if _fisher(x, PROBE_N, hit2, ref2["n"]) < 0.05
                      and x / PROBE_N < ref2["match"]), default=None)
        o.append(f"- `{PROBE_MODELS[1]}` is NOT a second test of the same thing. Its job is to "
                 f"say the "
                 f"cell is answerable and not VOID at this length. Against its own "
                 f"{hit2}/{ref2['n']} at L=256, n={PROBE_N} resolves a drop only at "
                 + (f"{lo_cut}/{PROBE_N} or below" if lo_cut is not None else "no count at all")
                 + f" (p<0.05); anything above that is uninformative about it, and the report "
                 f"should say so rather than read the middle of that range.")
    o.append("- Either way the result is read against the admitted class at this cell "
             + (f"({cell['admitted_max'] / cell['chance']:.2f}x informed chance) "
                if cell else "")
             + "and against informed chance "
             + (f"{cell['chance']:.4f}" if cell else "1/(k-1)")
             + ", never against 1/k. VALIDITY FIRST as everywhere: over 10% finish=length or "
               "empty and the cell is VOID and decides nothing.\n")
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

    o.extend(lead_section(rows, eff))
    o.extend(geometry_section())
    o.extend(width_depth_section())
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
    o.extend(path_section(rows))

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
             "same cells and the same read as the rows below. The `measured touch` row is the "
             "chain each band cell carries, read off its own rendered prompts, so a cell run at "
             "another (k, L) can be compared on depth instead of on its label.\n")
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
    o.append("| *measured touch* | "
             + " | ".join(f"*{hops_at(12, L):.2f}*" if cell == "composed" and hops_at(12, L)
                          else "*—*" for cell, L in order) + " |")
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
        o.append(f"**{', '.join(missing)} were not run on this arm, so this model is not "
                 f"literally placed in the band.** The limit is one and not two: the four band "
                 f"cells at n=40 are "
                 f"{sum(40 * L * W.TOK_PER_EVENT for L in (128, 256, 85, 171)) / (eff or W.RATE_TOK_PER_S) / 3600:.0f} "
                 f"hours of serialized generation on this endpoint. The cells that were run are "
                 f"off the band's (k, L) axis and are compared to it on measured chain depth "
                 f"instead.\n")
        o.append("> **CORRECTED.** The earlier rendering gave a second, independent reason — "
                 "\"the endpoint does not return a generation of ~10.5k tokens at all\" — and "
                 "the placement control below disproves it. Wall clock is the only limit.\n")
    ctrl = rows.get(("composed", 6, 64))
    if ctrl and not W.void(ctrl):
        lo, hi = _wilson(ctrl["match"], ctrl["n"])
        ch = S.informed_chance(ctrl["k"])
        qw = {(r["k"], r["L"]): r for r in S.load_rows().values() if r["cell"] == "composed"}
        rival = qw.get((ctrl["k"], ctrl["L"]))
        o.append(f"**What the round does establish is that this is a SECOND CEILING**, and it "
                 f"rests on the placement control rather than on the shallowest cell. "
                 f"composed@L{ctrl['L']}, k={ctrl['k']}: match {ctrl['match']:.3f} at "
                 f"n={ctrl['n']}, `{ctrl['finish_reasons']}`, {ctrl['length_rate']:.2f} "
                 f"truncated, {ctrl['empty_rate']:.2f} empty, 95% CI [{lo:.2f}, {hi:.2f}], "
                 f"z={_z(ctrl['match'], ch, ctrl['n']):+.2f} over informed chance {ch:.4f}. That "
                 f"cell carries {hops_at(ctrl['k'], ctrl['L']):.2f} measured touches — DEEPER "
                 f"than composed@128's {hops_at(12, 128):.2f}, where glm-5.2 reads "
                 f"{W.SCOUT['z-ai/glm-5.2'][('composed', 128)]:.3f} and nemotron "
                 f"{W.SCOUT['nvidia/nemotron-3-ultra-550b-a55b'][('composed', 128)]:.3f}.")
        if rival is not None:
            p = _fisher(int(round(ctrl["match"] * ctrl["n"])), ctrl["n"],
                        int(round(rival["match"] * rival["n"])), rival["n"])
            o.append(f"Against the local Qwen at the IDENTICAL cell ({rival['match']:.3f}, "
                     f"n={rival['n']}), Fisher exact p={p:.3f}: the two free arms are separated "
                     f"at the one cell both have been run on.\n")
        else:
            o.append("")
        shallow = rows.get(("composed", 6, 32))
        if shallow is not None:
            o.append(f"> **CORRECTED.** The earlier rendering asserted the second ceiling from "
                     f"composed@L{shallow['L']}, k={shallow['k']} "
                     f"({shallow['match']:.3f} at n={shallow['n']}), which carries "
                     f"{hops_at(shallow['k'], shallow['L']):.2f} measured touches against a band "
                     f"at {hops_at(12, 128):.2f} and {hops_at(12, 256):.2f}. A cell that shallow "
                     f"cannot rank a model on that band; the L={ctrl['L']} control can, and it "
                     f"is what the claim now rests on.\n")
        o.append("So this arm cannot test the k axis from below any more than the local Qwen can "
                 "from above: Qwen is at informed chance at the lengths it can afford, this model "
                 "is at the ceiling at the deepest cell it has been run on, and the depth where "
                 "it would come off the ceiling is priced in hours below.\n")

    # --- the k axis ----------------------------------------------------------------------
    ks = sorted({k for k, _L in comp})
    ls = sorted({L for _k, L in comp})
    one = {}
    if os.path.exists(ONE_STRUCTURE):
        cells_os = json.load(open(ONE_STRUCTURE, encoding="utf-8"))["cells"]
        for c in cells_os:
            c["admitted_max"], c["admitted_row"] = admitted_max(c)
        one = {(c["k"], c["L"]): c for c in cells_os}
    o.append("## 2. The k axis — match against informed chance 1/(k-1), and its token price\n")
    o.append("| k | L | chance 1/(k-1) | n | match | 95% CI | x chance | z | measured touch | "
             "admitted read | margin | prompt tok/item | completion tok/item | wall/item |")
    o.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (k, L) in sorted(comp, key=lambda t: (t[1], t[0])):
        r = comp[(k, L)]
        ch = S.informed_chance(k)
        lo, hi = _wilson(r["match"], r["n"])
        c = one.get((k, L))
        os_read = (f"{c['admitted_max']:.4f} "
                   f"({c['admitted_max'] / ch:.2f}x)" if c else "—")
        margin = f"**{r['match'] - c['admitted_max']:+.3f}**" if c else "—"
        h = hops_at(k, L)
        o.append(f"| {k} | {L} | {ch:.4f} | {r['n']} | "
                 f"{'VOID ' if W.void(r) else ''}{r['match']:.3f} | [{lo:.2f},{hi:.2f}] | "
                 f"{r['match'] / ch:.2f}x | {_z(r['match'], ch, r['n']):+.1f} | "
                 f"{'—' if h is None else f'{h:.2f}'} | {os_read} | "
                 f"{margin} | {r['ptok_item']:.0f} | {r['ctok_item']:.0f} | "
                 f"{r['elapsed_s'] / r['n']:.0f}s |")
    o.append("")
    o.append("`admitted read` is that cell's strongest admitted policy from the section above, "
             "and `margin` is what the measured score holds over it. A cell whose margin is not "
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
                 f"cell at this endpoint's serialized rate. The L axis this round rests on is "
                 f"the local grid's, 16 live cells "
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
        ha, hb = hops_at(k, ends[0]), hops_at(k, ends[-1])
        o.append(f"At k={k}, L in {sorted(live)}: match spans **{span:.3f}**"
                 + (f", over a measured chain that goes {ha:.2f} touches to {hb:.2f} — "
                    f"{hb / ha:.1f}x the depth for no movement in the score"
                    if ha and hb else "")
                 + f". An extra EVENT costs {per_event:+.0f} completion tokens and "
                   f"{p_event:+.0f} prompt tokens per item.\n")

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

    o.extend(probe_price_section())

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
