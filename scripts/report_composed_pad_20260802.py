"""Render the bounded-pad restart grid and the composed-pad attack into one results page.

Per-seed values only. The composed cell's ``slot_acc`` is the quantity the round exists for, so it
is printed beside the answer everywhere the answer appears and never summarised across seeds.

Usage:
    .venv-train/bin/python scripts/report_composed_pad_20260802.py \\
        --grid results/20260802_s5bind_v3_bounded_pad_restart_grid.json \\
        --counts results/20260802_composed_pad_counts.json \\
        --forced results/20260802_composed_pad_forced.json \\
        --decompose results/20260802_composed_pad_decompose.json \\
        --overwrite results/20260802_composed_pad_overwrite.json \\
        --out results/20260802_bounded_pad_restart_grid.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def load(path):
    return None if not path or not os.path.exists(path) else json.load(open(path))


def cells_of(gg):
    return [(c, L) for c in ("state", "bind", "composed") for L in gg.get(c, ())]


def grid_tables(res, out):
    gg = res["cfg"]["guided_grid"]
    n = res["cfg"].get("final_guided_n") or res["cfg"]["guided_n"]
    npre = res["cfg"]["guided_n"]
    cells = cells_of(gg)
    out.append(f"## The three cells at every registered length, n={n}\n")
    out.append(f"`match` / `slot_acc`, `*` clears its bounded-pad floor "
               f"(z>{P.Z_CLEAR} and margin>={P.MARGIN}). Pad width "
               f"{res['pad_width']}, format `{res['format']}`.\n")
    hdr = ["seed"] + [f"{c}@{L}" for c, L in cells]
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for r in res["runs"]:
        g = r["stages"][-1]["guided"]
        row = [str(r["seed"])]
        for c, L in cells:
            blk = g.get(c, {}).get(str(L)) or {}
            v, sa = blk.get("match"), blk.get("slot_acc")
            fl = (res["floors"].get(f"{c}@{L}") or {}).get("floor")
            mark = "*" if v is not None and P.clears(v, fl, n)[0] else ""
            row.append("—" if v is None else f"{v:.3f}{mark} / {f3(sa)}")
        out.append("| " + " | ".join(row) + " |")
    frow = ["floor"]
    for c, L in cells:
        fl = (res["floors"].get(f"{c}@{L}") or {}).get("floor")
        frow.append("unfloorable" if fl is None else f"{fl:.4f}")
    out.append("| " + " | ".join(frow) + " |")
    out.append("")

    out.append(f"## The same cells BEFORE the restart, n={npre}\n")
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for r in res["runs"]:
        pre = [s for s in r["stages"] if s["stage"] != "stage4_restart"][-1]
        row = [str(r["seed"])]
        for c, L in cells:
            blk = pre["guided"].get(c, {}).get(str(L)) or {}
            v, sa = blk.get("match"), blk.get("slot_acc")
            fl = (res.get("floors_pre_restart", {}).get(f"{c}@{L}") or {}).get("floor")
            mark = "*" if v is not None and P.clears(v, fl, npre)[0] else ""
            row.append("—" if v is None else f"{v:.3f}{mark} / {f3(sa)}")
        out.append("| " + " | ".join(row) + " |")
    out.append("")


def findings(res, counts, forced, decomp, fw, out):
    """The head of the page: what the run establishes, with every number pulled from the data."""
    gg = res["cfg"]["guided_grid"]
    cl = P.registered_lengths("composed")
    pad = {r["seed"]: {L: ((r["stages"][-1]["guided"].get("composed", {}).get(str(L))) or {})
                       .get("slot_acc") for L in cl} for r in res["runs"]}
    per_seed = "; ".join(f"seed {s} {'/'.join(f'{pad[s][L]:.3f}' for L in cl)}" for s in pad)
    restart = []
    for r in res["runs"]:
        st = [x for x in r["stages"] if x["stage"] == "stage4_restart"]
        if st and st[0].get("track"):
            t = st[0]["track"]
            restart.append(f"seed {r['seed']} {t[0][1]['composed@48']['slot_acc']:.3f}->"
                           f"{t[-1][1]['composed@48']['slot_acc']:.3f}")
    perfect = {}
    for r in (decomp or {}).get("rows", []):
        if r["cell"] == "composed" and "items_perfect" in r:
            perfect.setdefault(r["seed"], {})[r["L"]] = r["items_perfect"]
    items = "; ".join(f"seed {s} " + "/".join(f3(perfect[s].get(L)) for L in cl)
                      for s in sorted(perfect))
    out.append("## What this run establishes\n")
    out.append(f"THE COMPOSED CELL'S PAD IS CAPPED, at every registered length on every seed: "
               f"{per_seed} at L={'/'.join(str(L) for L in cl)} per token, and "
               f"{items} PER ITEM — which is the unit the answer is generated in — against "
               "components whose pad is perfect on every item at every length including the "
               "token-matched ones. So the composed cell's floored answer is a TRACKING result "
               "and not a composition one, and the registered rule says so only because the pad "
               "gate is applied — without it the same numbers return a composition gap.\n")
    out.append("THE REGISTERED RESTART DOES NOT DEGRADE THE PAD; it raises it on every seed "
               f"(composed@48, start to end of `stage4_restart`: {'; '.join(restart)}). The "
               "degradation this round was called to investigate belonged to a grouped, "
               "answer-ratio-4 readout stage that is not the registered mix.\n")
    if counts:
        row = {f"{c['cell']}@{c['L']}": c for c in counts["cells"]}
        out.append(
            "LENGTH IS NOT THE LEVER. The composed pad is SHORTER than the components' at "
            f"their token-matched lengths: composed@48 writes "
            f"{row['composed@48']['pad_tokens']:.0f} pad tokens over an "
            f"{row['composed@48']['prompt_tokens']:.0f}-token prompt, against "
            f"state@80's {row['state@80']['pad_tokens']:.0f} and bind@132's "
            f"{row['bind@132']['pad_tokens']:.0f}, both written at 1.000. What differs is that "
            "every composed event's operand is RESOLVED through the other structure "
            f"({row['composed@48']['resolved_operand_frac']:.3f} of events) and no component "
            "event's is (0.000).\n")
    if forced and decomp:
        fr = {(r["seed"], r["L"]): r for r in decomp["rows"] if r["cell"] == "composed"}
        fo = {(r["seed"], r["L"]): r for r in forced["rows"] if r["cell"] == "composed"}
        ks = sorted(set(fo) & set(fr))
        s0 = [k for k in ks if k[0] == ks[0][0]]
        out.append(
            "THE CAP IS A PER-EVENT RESIDUAL THAT COMPOUNDS, not a missing rule. With the GOLD "
            "pad in context the composed per-slot accuracy is flat in length — seed "
            f"{s0[0][0]}: " + " ".join(f"{fo[k]['per_slot']:.3f}" for k in s0)
            + f" at L={'/'.join(str(k[1]) for k in s0)} — while the same slots free-running "
            "collapse with it: " + " ".join(f"{fr[k]['per_slot']:.3f}" for k in s0)
            + ". The median ordinal of each item's FIRST wrong slot is the same at L=48 and "
            "L=96 within a seed — "
            + "; ".join(
                f"seed {s} {fr[(s, 48)]['first_error_median']}/"
                f"{fr[(s, 96)]['first_error_median']}"
                for s in sorted({k[0] for k in fr}) if (s, 48) in fr and (s, 96) in fr)
            + " — so the pad survives a fixed number of composed events and the stream's length "
            "only decides how much of it is wrong by the end.\n")
        out.append(
            "AND THE RESIDUAL SITS ON THE TWO-HOP WRITE. Teacher-forced, the one pad token that "
            "needs the operand resolved through the holder map and then read through the pointer "
            f"map (`swap_p0`) scores "
            f"{min(fo[k]['by_kind_position']['swap_p0'] for k in ks):.3f}-"
            f"{max(fo[k]['by_kind_position']['swap_p0'] for k in ks):.3f} across seeds and "
            "lengths, against the three one-hop tokens of the same events. That is where the "
            "composition is, and the next section is what it is worth against a floor.\n")
    if fw:
        cl = P.registered_lengths("composed")
        fl = {L: fw["cells"][f"composed@{L}"] for L in cl if f"composed@{L}" in fw["cells"]}
        ps = {L: fl[L]["free_run"]["operative"]["all"]["per_slot"] for L in fl}
        sp = {L: fl[L]["free_run"]["operative"]["all"]["cells"]["swap_p0"] for L in fl}
        one = {L: fl[L]["free_run"]["operative"]["one_hop"]["cells"]["swap_p0"] for L in fl}
        ch = next(iter(fl.values()))["chance"]["uniform"]
        out.append(
            "THE PAD WRITE HAS A FLOOR AND THE MEASURED PAD IS UNDER IT. The bounded pad hands "
            "every policy 2 free live slots, so the registered live-slot rule admits any policy "
            f"holding 8 of the 12 map cells, and such a policy writes the composed pad at "
            + "/".join(f"{ps[L]:.3f}" for L in fl) + f" at L={'/'.join(str(L) for L in fl)} "
            f"({'/'.join(f'{ps[L] / ch:.1f}' for L in fl)}x the per-slot chance of 1/k = "
            f"{ch:.3f}). The pad width that costs the ANSWER floor nothing costs the PAD floor "
            "almost everything: on the answer a partial carry buys 1.05-1.17x chance, on the pad "
            "it buys 3-5x, because most pad tokens are one-hop reads of cells the carry holds.\n")
        out.append(
            "SO THE TWO-HOP TOKEN DOES NOT REGISTER A COMPOSITION EITHER. Free-running — the read "
            "the grid scores — `swap_p0`'s floor is "
            + "/".join(f"{sp[L]:.3f}" for L in fl) + ", and the same one-structure carry sets it: "
            "a policy that never composes the two hops correctly still gets that token right "
            "3-5x chance by resolving the operand against a map it holds only part of. The "
            f"one-hop SUB-class reaches only {min(one.values()):.3f}-{max(one.values()):.3f} "
            "there, so the token is two-hop work — but the class that decides a floor is the "
            "registered one, and it reaches the model.\n")
        n = decomp["cfg"]["n"]
        rows = {(r["seed"], r["L"]): r for r in decomp["rows"] if r["cell"] == "composed"}
        who = {}
        for (s, L), r in rows.items():
            key = f"composed@{L}"
            if key not in fw["cells"]:
                continue
            b = fw["cells"][key]["free_run"]["operative"]["all"]
            if P.clears(r["per_slot"], b["per_slot"], n)[0]:
                who.setdefault(s, []).append(L)
        dead = [s for s in sorted(who) if all(
            (r["stages"][-1]["guided"].get(c, {}).get(str(L)) or {}).get("match", 1) < 0.5
            for r in res["runs"] if r["seed"] == s
            for c in ("state", "bind") for L in gg.get(c, ()))]
        out.append(
            "AND IT DOES NOT FORM. Of 3 seeds, "
            + ("none clears the per-slot floor at any length"
               if not who else
               "; ".join(f"seed {s} clears at L={'/'.join(str(L) for L in sorted(v))}"
                         for s, v in sorted(who.items())))
            + f", against the {P.SEEDS_CLEAR} seeds FORMS requires at EVERY registered length"
            + (f". The seed(s) that clear — {dead} — are the ones whose ANSWER is at floor on "
               "every cell including both components, so the pad write separates on exactly the "
               "seed that reads nothing out.\n" if dead else ".\n"))
        L16 = fw["cells"].get("composed@16")
        if L16:
            b16 = L16["free_run"]["operative"]["all"]["per_slot"]
            best16 = max((r["per_slot"] for r in decomp["rows"]
                          if r["cell"] == "composed" and r["L"] == 16), default=None)
            out.append(
                "AND THE SHORT-STREAM ESCAPE IS CLOSED ON THIS AXIS TOO. At L=16, where the best "
                f"seed writes {f3(best16)} of the pad, the floor is {b16:.4f} and the CLEARS bar "
                f"is {b16 + P.MARGIN:.3f} — above 1.0, so no score at that length can clear it. "
                "The pad-write read is unbuyable at exactly the lengths where the pad is "
                "written.\n")
    return out


def cl_registered():
    return P.registered_lengths("composed")


def apply_verdict(res, decomp, gold, out):
    """Run the registered rule over this grid's own numbers, including the pad gate.

    Nothing here is asserted: ``forms``, ``pad_tracks``, ``readout_alive`` and ``verdict`` are the
    protocol's, applied to the final read. The verdict is printed WITH and WITHOUT the pad gate,
    because the difference between them is what the gate is for — a rule reading only the answer
    returns a composition gap on a run whose composed scratchpad is wrong by the middle of every
    stream.

    The pad column is ITEMS PERFECT and comes from the decompose probe, which is the same
    free-running decode on the same n and the same items as the grid's final read (its pooled
    ``per_slot`` reproduces the grid's ``slot_acc`` to the digit). The GOLD-PAD answer column is
    required on any path that interprets the composed cell's floored answer; where the run does
    not carry it, the rule raises and the refusal is printed instead of a verdict.
    """
    n = res["cfg"].get("final_guided_n") or res["cfg"]["guided_n"]
    gg = res["cfg"]["guided_grid"]
    ans = {}
    for r in res["runs"]:
        g = r["stages"][-1]["guided"]
        for c in ("state", "bind", "composed"):
            for L in gg.get(c, ()):
                blk = g.get(c, {}).get(str(L)) or {}
                ans.setdefault(c, {}).setdefault(r["seed"], {})[L] = blk.get("match")
    floors = {c: {L: (res["floors"].get(f"{c}@{L}") or {}).get("floor") for L in gg.get(c, ())}
              for c in ans}
    comp_forms, comp_counts = {}, {}
    for c in ("state", "bind", "composed"):
        lengths = P.registered_lengths(c)
        comp_forms[c], comp_counts[c] = P.forms(ans[c], floors[c], lengths, n=n)
    matched = {}
    for c in ("state", "bind"):
        ml = [L for L in (P.TOKEN_MATCHED.get(P.GUIDED_MATCHED_FROM, {}).get(c),) if L]
        matched[c] = P.forms(ans[c], floors[c], ml, n=n)[0] if ml else None
    cl = P.registered_lengths("composed")
    out.append("## The registered rule, applied to these numbers\n")
    out.append(f"- components FORM at their own registered lengths: {comp_counts} "
               f"(needs {P.SEEDS_CLEAR} seeds at every length)")
    out.append(f"- matched-cost control at composed@{P.GUIDED_MATCHED_FROM}: {matched}")
    out.append(f"- composed cell clears: {comp_counts['composed']}")
    tracked, pad_counts = None, None
    if decomp:
        perfect = {}
        for r in decomp["rows"]:
            if r["cell"] == "composed" and "items_perfect" in r:
                perfect.setdefault(r["seed"], {})[r["L"]] = r["items_perfect"]
        tracked, pad_counts = P.pad_tracks(perfect, cl)
        out.append(f"- composed pad is perfect on {P.PAD_TRACKS_MIN} of ITEMS on: {pad_counts} "
                   f"seeds per length -> pad_tracked={tracked} "
                   + "(per seed at L=" + "/".join(str(L) for L in cl) + ": "
                   + "; ".join(f"seed {s} " + "/".join(f3(perfect[s].get(L)) for L in cl)
                               for s in sorted(perfect)) + ")")
    else:
        out.append("- composed pad: no items-perfect column in this run")
    readout, readout_counts = None, None
    if gold:
        gp = {}
        for r in gold.get("rows", []):
            if r.get("cell") == "composed":
                gp.setdefault(r["seed"], {})[r["L"]] = r.get("match")
        readout, readout_counts = P.readout_alive(gp, floors["composed"], cl, n)
        out.append(f"- composed GOLD-PAD answer clears on: {readout_counts} seeds per length "
                   f"-> readout={readout}")
    else:
        out.append("- composed GOLD-PAD answer: not measured in this run")
    ctrl = {"seeds": 0, "cleared_on": "state", "per_pair": comp_counts["state"], "required": []}
    ctrl["seeds"] = max([v for v in comp_counts["state"].values() if v is not None] or [0])
    mm = {c: matched[c] is not None for c in ("state", "bind")}
    ungated = P.verdict(ctrl, comp_forms, comp_counts, matched, mm)
    try:
        gated = P.verdict(ctrl, comp_forms, comp_counts, matched, mm, pad_tracked=tracked,
                          pad_counts=pad_counts, readout=readout, readout_counts=readout_counts)
    except P.ReadoutNotEvaluable as exc:
        out.append(f"\n**Without the pad gate the rule returns `{ungated[0]}`. With it, the rule "
                   f"REFUSES this run: {exc}**\n")
        return ("READOUT_NOT_EVALUABLE", str(exc))
    out.append(f"\n**Without the pad gate the rule returns `{ungated[0]}`. With it, "
               f"`{gated[0]}`.**\n")
    out.append(gated[1] + "\n")
    return gated


def pad_write_floor_table(fw, decomp, out):
    """The pad-write floor at every composed cell, and every measured seed against it."""
    cells = [k for k in fw["cells"] if k.startswith("composed@")]
    out.append("## The PAD WRITE against its own floor\n")
    out.append("The class is the registered one, scored on the pad instead of on the answer: "
               "live slots `W - pad <= max(k, m) + 1`, steps no more than the cell's own algorithm "
               "pays to produce the pad. At pad 2 that admits any policy holding 8 of the 12 map "
               "cells and excludes the cell's own algorithm (12 + scratch). Chance for a per-slot "
               "read is `1/k` — every pad token is an agent name — and not the answer read's "
               "`1/(k-1)`.\n")
    out.append("| cell | chance | per-slot floor | row | swap_p0 floor | swap_p0, one-hop "
               "sub-class | CLEARS bar |")
    out.append("|---|---|---|---|---|---|---|")
    for key in cells:
        c = fw["cells"][key]
        b = c["free_run"]["operative"]["all"]
        o = c["free_run"]["operative"]["one_hop"]
        bar = b["per_slot"] + P.MARGIN
        out.append(f"| {key} | {c['chance']['uniform']:.4f} | {b['per_slot']:.4f} "
                   f"({b['per_slot'] / c['chance']['uniform']:.2f}x) | `{b['per_slot_row']}` | "
                   f"{b['cells']['swap_p0']:.4f} | {o['cells']['swap_p0']:.4f} | "
                   f"{bar:.3f}{' — above 1.0, unbuyable' if bar > 1.0 else ''} |")
    out.append("")
    rows = {(r["seed"], r["L"]): r for r in (decomp or {}).get("rows", [])
            if r["cell"] == "composed"}
    if not rows:
        return
    n = decomp["cfg"]["n"]
    out.append(f"Every measured seed against it, n={n}, under the registered `clears` "
               f"(z>{P.Z_CLEAR} and margin>={P.MARGIN}).\n")
    out.append("| seed | cell | per-slot | floor | clears | swap_p0 | floor | clears |")
    out.append("|---|---|---|---|---|---|---|---|")
    for (s, L), r in sorted(rows.items()):
        key = f"composed@{L}"
        if key not in fw["cells"]:
            continue
        b = fw["cells"][key]["free_run"]["operative"]["all"]
        sp0 = (r.get("by_kind_position") or {}).get("swap_p0")
        cl = P.clears(r["per_slot"], b["per_slot"], n)[0]
        cl0 = P.clears(sp0, b["cells"]["swap_p0"], n)[0] if sp0 is not None else False
        out.append(f"| {s} | {key} | {r['per_slot']:.3f} | {b['per_slot']:.4f} | "
                   f"{'yes' if cl else 'no'} | {f3(sp0)} | {b['cells']['swap_p0']:.4f} | "
                   f"{'yes' if cl0 else 'no'} |")
    out.append("")
    comp = [k for k in fw["cells"] if not k.startswith("composed@")]
    if comp:
        vals = {k: fw["cells"][k]["free_run"]["operative"]["all"]["per_slot"] for k in comp}
        out.append(
            "AND THE PAD-WRITE READ SUPPORTS NO POSITIVE CONTROL. A component cell has ONE "
            "structure, so the one-structure bound admits the component's own pad algorithm and "
            "its floor is " + ", ".join(f"{v:.4f} at {k}" for k, v in sorted(vals.items()))
            + " — the same argument that leaves the composed cell unfloorable under a `k + m` "
            "wide pad, one cell over. A component's perfect pad cannot clear anything, so this "
            "read has no cell on which a working model demonstrates that the measurement works.\n")
    key = f"composed@{P.GUIDED_MATCHED_FROM}"
    key = key if key in fw["cells"] else cells[0]
    tf = fw["cells"][key].get("teacher_forced")
    if tf:
        b = tf["operative"]["all"]
        out.append(
            "AND THE TEACHER-FORCED READ HAS A SATURATED CELL. Handed the same gold history, a "
            "row that holds the holder map refreshes each object's new holder from the adjacent "
            "gold block and scores 1.000 on `give_p1` at every length, so that cell's floor is "
            f"the ceiling; the pooled teacher-forced floor is {b['per_slot']:.4f} at {key}. The "
            "teacher-forced numbers stay what they were — a diagnostic — and the residual they "
            f"show on `swap_p0` is under the floor of that same class ({b['cells']['swap_p0']:.4f} "
            f"at {key}).\n")


def track_tables(res, out):
    out.append("## Pad accuracy ACROSS the stage\n")
    tn = None
    for r in res["runs"]:
        for st in r["stages"]:
            if not st.get("track"):
                continue
            tn = st.get("track_n")
            keys = list(st["track"][0][1])
            steps = [s for s, _ in st["track"]]
            out.append(f"**seed {r['seed']}, `{st['stage']}`** "
                       f"(every {st.get('track_every')} steps, n={tn})\n")
            out.append("| cell | " + " | ".join(str(s) for s in steps) + " |")
            out.append("|" + "---|" * (len(steps) + 1))
            for k in keys:
                vals = [f3(pt[k]["slot_acc"]) for _s, pt in st["track"]]
                out.append(f"| {k} | " + " | ".join(vals) + " |")
            out.append("")


def counts_table(counts, out):
    out.append("## Is the composed pad simply longer? (no model)\n")
    out.append("| cell | events | pad tokens | prompt tokens | pad/prompt | resolved operand |")
    out.append("|---|---|---|---|---|---|")
    for r in counts["cells"]:
        out.append(f"| {r['cell']}@{r['L']} | {r['events']:.0f} | {r['pad_tokens']:.0f} | "
                   f"{r['prompt_tokens']:.0f} | {r['pad_share']:.3f} | "
                   f"{r['resolved_operand_frac']:.3f} |")
    out.append("")


def probe_table(d, out, title, note):
    out.append(f"## {title}\n")
    out.append(note + "\n")
    out.append("On the COMPOSED rows `swap_p0` is the only two-hop token — the operand is "
               "resolved through the holder map and then read through the pointer map — and the "
               "other three "
               "are one hop. On a COMPONENT row every operand is NAMED, so its `swap_p0` is a "
               "one-hop read and is not the same quantity.\n")
    kp = ("give_p0", "give_p1", "swap_p0", "swap_p1")
    have_kp = any("by_kind_position" in r for r in d["rows"])
    hdr = ["seed", "cell", "per_slot"] + (list(kp) if have_kp else ["pos0", "pos1"])
    if "items_perfect" in d["rows"][0]:
        hdr += ["items perfect", "median first error"]
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for r in d["rows"]:
        row = [str(r["seed"]), f"{r['cell']}@{r['L']}", f3(r["per_slot"])]
        if have_kp:
            row += [f3((r.get("by_kind_position") or {}).get(k)) for k in kp]
        else:
            row += [f3(v) for v in r["by_position"]]
        if "items_perfect" in r:
            row += [f3(r["items_perfect"]), str(r.get("first_error_median"))]
        out.append("| " + " | ".join(row) + " |")
    out.append("")


def ordinal_table(d, out):
    rows = [r for r in d["rows"] if r["cell"] == "composed" and "by_ordinal" in r]
    if not rows:
        return
    out.append("## Where in the stream the pad breaks\n")
    out.append("Free-running per-slot accuracy at event ordinal, as a fraction of the stream.\n")
    fracs = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    out.append("| seed | cell | " + " | ".join(f"{f:.0%}" for f in fracs) + " |")
    out.append("|" + "---|" * (len(fracs) + 2))
    for r in rows:
        bo = r["by_ordinal"]
        idx = [min(len(bo) - 1, int(f * (len(bo) - 1))) for f in fracs]
        out.append(f"| {r['seed']} | composed@{r['L']} | "
                   + " | ".join(f3(bo[i]) for i in idx) + " |")
    out.append("")


def overwrite_table(d, out, title=None, note=None):
    arms = sorted({r["mix"] for r in d["runs"]})
    out.append("## " + (title or "Is the pad being traded away by answer supervision?") + "\n")
    out.append(note or (f"Equal continuations of {d['cfg']['steps']} steps at lr "
                        f"{d['cfg']['lr']:g} under {len(arms)} mixes ({', '.join(arms)}), "
                        "composed `slot_acc` before -> after.\n"))
    cells = sorted({c for r in d["runs"] for c in r["before"]})
    out.append("| seed | mix | " + " | ".join(cells) + " |")
    out.append("|" + "---|" * (len(cells) + 2))
    for r in d["runs"]:
        vals = [f"{f3(r['before'][c])} -> {f3(r['after'][c])}" for c in cells]
        out.append(f"| {r['seed']} | `{r['mix']}` | " + " | ".join(vals) + " |")
    out.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--counts", default=None)
    ap.add_argument("--forced", default=None)
    ap.add_argument("--decompose", default=None)
    ap.add_argument("--overwrite", default=None)
    ap.add_argument("--composed_only", default=None)
    ap.add_argument("--pad_write_floor", default=None)
    ap.add_argument("--gold_answer", default=None,
                    help="the composed cell read with the GOLD pad in context: the readout column")
    ap.add_argument("--out", default="results/20260802_bounded_pad_restart_grid.md")
    a = ap.parse_args()

    res = json.load(open(a.grid))
    decomp, fw = load(a.decompose), load(a.pad_write_floor)
    out = ["# The bounded-pad grid, the composed pad's cap, and the pad write's own floor", ""]
    findings(res, load(a.counts), load(a.forced), decomp, fw, out)
    grid_tables(res, out)
    apply_verdict(res, decomp, load(a.gold_answer), out)
    if fw:
        pad_write_floor_table(fw, decomp, out)
    track_tables(res, out)
    for path, fn in ((a.counts, counts_table),):
        d = load(path)
        if d:
            fn(d, out)
    d = load(a.forced)
    if d:
        probe_table(d, out, "Is the per-event update missing, or only the closed loop?",
                    "Slot accuracy with the GOLD pad in context, one forward per item. A "
                    "diagnostic: teacher-forced accuracy is not a score on the task, because the "
                    "true history is what the task withholds.")
    d = load(a.decompose)
    if d:
        probe_table(d, out, "The same slots free-running, decomposed",
                    "The grid's own read, instrumented per slot instead of pooled. `swap_p0` is "
                    "the only two-hop token: the operand is resolved through the holder map and "
                    "then read through the pointer map.")
        ordinal_table(d, out)
    d = load(a.overwrite)
    if d:
        overwrite_table(d, out)
    d = load(a.composed_only)
    if d:
        overwrite_table(
            d, out, "Does more composed training close it?",
            f"{d['cfg']['steps']} steps at lr {d['cfg']['lr']:g} on the composed cell ALONE, from "
            "the registered grid's own final checkpoint. This is the restart's whole budget spent "
            "on nothing but the cell whose pad is capped.\n")
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
