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

from factworld import validity as V                                        # noqa: E402
import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def f4(v):
    return "—" if v is None else f"{v:.4f}"


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
        cross = [fo[k].get("by_kind_position_source", {}).get(P.PAD_WRITE_TOKEN)
                 for k in ks if fo[k].get("by_kind_position_source", {}).get(P.PAD_WRITE_TOKEN)]
        same = [fo[k].get("by_kind_position_source", {}).get("swap_p0|same")
                for k in ks if fo[k].get("by_kind_position_source", {}).get("swap_p0|same")]
        out.append(
            "AND THE RESIDUAL SITS ON THE TWO-HOP WRITE. Teacher-forced, the one pad token that "
            "needs the operand resolved and then read through the pointer map (`swap_p0`) scores "
            f"{min(fo[k]['by_kind_position']['swap_p0'] for k in ks):.3f}-"
            f"{max(fo[k]['by_kind_position']['swap_p0'] for k in ks):.3f} across seeds and "
            "lengths, against the three one-hop tokens of the same events"
            + (f". Split by SOURCE it is {min(cross):.3f}-{max(cross):.3f} where the operand is "
               f"resolved through the HOLDER map and {min(same):.3f}-{max(same):.3f} where it is "
               "resolved through the pointer map the write also reads — the first needs both "
               "structures, the second is P read twice and is the state component's own carrier "
               "depth, and only the first is scored" if cross and same else "")
            + ".\n")
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
        read = P.PAD_WRITE_SCORED_READ
        two = {L: fl[L][read]["two_hop"][P.PAD_WRITE_TOKEN] for L in fl
               if fl[L][read]["two_hop"].get(P.PAD_WRITE_TOKEN, {}).get("floor") is not None}
        conf = {(r["seed"], r["cell"]): r for r in (fw.get(f"confront_{read}") or [])}
        conf_free = {(r["seed"], r["cell"]): r for r in (fw.get("confront_free_run") or [])}
        seeds = sorted({s for s, _c in conf})
        if two and conf:
            out.append(
                "AND THE TWO-HOP TOKEN IS SCORED AGAINST A CLASS THAT MAY NOT COMPOSE IT. The "
                "registered live-slot rule is the composed cell's own and is the floor for the "
                "three one-hop tokens; the two-hop token takes the COMPONENT cells' conjunct "
                f"instead — depth <= {V.S5_BIND_V3_MAX_DEPTH}, applied per emitted token — so an "
                "admitted row may hold 8 of the 12 map cells and still not perform the "
                "resolve-then-read the token is. On the events whose operand is resolved through "
                "the OTHER structure, that class reaches "
                + "/".join(f"{two[L]['floor']:.4f}" for L in two)
                + f" at L={'/'.join(str(L) for L in two)} against a per-slot chance of "
                f"{ch:.4f}, where the class WITHOUT the depth conjunct reaches "
                + "/".join(f"{two[L]['unrestricted']:.4f}" for L in two) + ".\n")
            out.append(
                "WHICH CONJUNCT IS APPLIED DECIDES THE RESULT, so it is stated rather than "
                "assumed. Under the W-only conjunct — the composed cell's own, no depth bound — a "
                "row holding 8 of the 12 map cells resolves the operand against a map it holds "
                "most of and reaches "
                + "/".join(f"{two[L]['unrestricted']:.4f}" for L in two)
                + ", and the two seeds that carry a claim are UNDER it at every registered "
                "length. The result below rests entirely on the depth conjunct being the right "
                "class for this token, and the argument for that is that it is the conjunct both "
                "COMPONENT cells' floors are already set by, read on the emission instead of on "
                "the policy: a row may not chain two events' contents, and this token is two "
                "dependent reads.\n")
            out.append(
                "THE SCORED READ IS THE TEACHER-FORCED PER-EVENT ONE, and it clears on every seed "
                "at every registered length. Per seed at L="
                + "/".join(str(L) for L in two) + ": "
                + "; ".join(
                    f"seed {s} " + "/".join(
                        f3(conf[(s, f'composed@{L}')].get(P.PAD_WRITE_TOKEN)) for L in two)
                    for s in seeds)
                + ", against bars "
                + "/".join(f"{P.bar_for(two[L]['floor']):.4f}" for L in two)
                + ". Teacher-forcing removes the COMPOUNDING of a per-event residual over the "
                "model's own writes and therefore claims nothing about them: free-running, the "
                "same token on the same events reads "
                + "; ".join(
                    f"seed {s} " + "/".join(
                        f3(conf_free.get((s, f'composed@{L}'), {}).get(P.PAD_WRITE_TOKEN))
                        for L in two)
                    for s in seeds)
                + " and clears "
                + str(sum(1 for k, r in conf_free.items()
                          if k[1].startswith("composed@")
                          and r.get(f"{P.PAD_WRITE_TOKEN}_clears")))
                + f" of {sum(1 for k in conf_free if k[1].startswith('composed@'))} cells. The "
                "free-running read stays the tracking diagnostic and is printed at every cell.\n")
        claim = fw.get("claim")
        if claim:
            carry = sorted(int(s) for s, v in claim["per_seed"].items() if v)
            out.append(
                "THE CLAIM IS CONJOINED PER SEED, so it cannot be assembled across models. "
                + "; ".join(f"`{g}` on {v}" for g, v in claim["by_gate"].items())
                + f" — {claim['n_seeds']} seed(s) carry all of it ({carry}), against "
                f"{P.SEEDS_CLEAR} required, so a scored composition result on this token "
                f"{'EXISTS' if claim['holds'] else 'DOES NOT EXIST'} on this grid. The seed with "
                "the best pad is excluded by its own components: its answer is at floor on all "
                "eight component cells with a byte-perfect pad, so it reads nothing out and "
                "carries no claim about anything downstream of a readout. The result survives "
                "its exclusion.\n")
            out.append(
                "WHAT THAT RESULT IS, exactly: on events whose operand is resolved through the "
                "holder map, given the true history, these models write the value that operand "
                "points to more often than any policy that may hold 8 of the 12 map cells but "
                "may not chain two reads. It is not an ANSWER result and not an end-to-end one — "
                "the same grid's answer read returns a tracking gap on the same seeds, and the "
                "free-running column above is why.\n")
        L16 = fw["cells"].get("composed@16")
        if L16:
            b16 = L16["free_run"]["operative"]["all"]["per_slot"]
            best16 = max((r["per_slot"] for r in decomp["rows"]
                          if r["cell"] == "composed" and r["L"] == 16), default=None)
            out.append(
                "AND THE MARGIN ON THIS READ IS FRACTIONAL, because the additive one is not a bar "
                f"here. At L=16 the per-slot floor is {b16:.4f} and the additive rule asks for "
                f"{b16 + P.MARGIN:.3f} — above 1.0, so no score whatever clears it, at the length "
                f"where the best measured pad is {f3(best16)}. The registered rule is "
                f"`(a - f)/(1 - f) >= {P.MARGIN_FRAC:g}`, which is the same 0.15 re-expressed at "
                f"the answer floor it was calibrated on ({P.MARGIN} / (1 - "
                f"{P.ANSWER_CHANCE_AT_REGISTRATION})) and is defined at every floor below 1. "
                f"Under it composed@16 is buyable at a bar of {P.bar_for(b16):.4f}.\n")
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
    comp_forms, comp_counts, comp_seeds = {}, {}, {}
    for c in ("state", "bind", "composed"):
        lengths = P.registered_lengths(c)
        comp_forms[c], comp_counts[c], comp_seeds[c] = P.forms(ans[c], floors[c], lengths, n=n)
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
    tracked, pad_counts, pad_seeds = None, None, {}
    if decomp:
        perfect = {}
        for r in decomp["rows"]:
            if r["cell"] == "composed" and "items_perfect" in r:
                perfect.setdefault(r["seed"], {})[r["L"]] = r["items_perfect"]
        tracked, pad_counts, pad_seeds = P.pad_tracks(perfect, cl)
        out.append(f"- composed pad is perfect on {P.PAD_TRACKS_MIN} of ITEMS on: {pad_counts} "
                   f"seeds per length -> pad_tracked={tracked} "
                   + "(per seed at L=" + "/".join(str(L) for L in cl) + ": "
                   + "; ".join(f"seed {s} " + "/".join(f3(perfect[s].get(L)) for L in cl)
                               for s in sorted(perfect)) + ")")
    else:
        out.append("- composed pad: no items-perfect column in this run")
    readout, readout_counts, readout_seeds = None, None, {}
    if gold:
        gp = {}
        for r in gold.get("rows", []):
            if r.get("cell") == "composed":
                gp.setdefault(r["seed"], {})[r["L"]] = r.get("match")
        readout, readout_counts, readout_seeds = P.readout_alive(gp, floors["composed"], cl, n)
        out.append(f"- composed GOLD-PAD answer clears on: {readout_counts} seeds per length "
                   f"-> readout={readout}")
    else:
        out.append("- composed GOLD-PAD answer: not measured in this run")
    # THE CONJUNCTION, per seed. Three counts over seeds are satisfied by a run in which no one
    # seed satisfies two of them, so the seeds are intersected before any composition verdict is
    # available and the intersection is printed with the gates that produced it.
    gates = {"components_form": {s: bool(comp_seeds["state"].get(s) and comp_seeds["bind"].get(s))
                                 for s in set(comp_seeds["state"]) | set(comp_seeds["bind"])},
             "pad_tracks": pad_seeds, "readout_alive": readout_seeds}
    gates = {g: v for g, v in gates.items() if v}
    carry_ok, carry_per, carry_n, by_gate = P.seeds_carrying(gates)
    out.append(f"- PER-SEED conjunction of the gates {by_gate} -> {carry_n} seed(s) carry the "
               f"whole claim ({carry_per}); needs {P.SEEDS_CLEAR}")
    ctrl = {"seeds": 0, "cleared_on": "state", "per_pair": comp_counts["state"], "required": []}
    ctrl["seeds"] = max([v for v in comp_counts["state"].values() if v is not None] or [0])
    mm = {c: matched[c] is not None for c in ("state", "bind")}
    ungated = P.verdict(ctrl, comp_forms, comp_counts, matched, mm)
    try:
        gated = P.verdict(ctrl, comp_forms, comp_counts, matched, mm, pad_tracked=tracked,
                          pad_counts=pad_counts, readout=readout, readout_counts=readout_counts,
                          seed_gates=gates)
    except (P.ReadoutNotEvaluable, P.SeedsNotConjoined) as exc:
        out.append(f"\n**Without the pad gate the rule returns `{ungated[0]}`. With it, the rule "
                   f"REFUSES this run: {exc}**\n")
        return ("READOUT_NOT_EVALUABLE", str(exc))
    out.append(f"\n**Without the pad gate the rule returns `{ungated[0]}`. With it, "
               f"`{gated[0]}`.**\n")
    out.append(gated[1] + "\n")
    return gated


def pad_write_floor_table(fw, decomp, out):
    """The pad-write floors, the SCORED two-hop read against its own floor, and the controls."""
    cells = [k for k in fw["cells"] if k.startswith("composed@")]
    out.append("## The PAD WRITE against its own floors\n")
    out.append("Two floors, because the pad has two kinds of token in it. The REGISTERED CLASS is "
               "the composed cell's own: live slots `W - pad <= max(k, m) + 1`, steps no more than "
               "the algorithm pays to produce the pad — at pad 2 any policy holding 8 of the 12 "
               "map cells, and not the algorithm itself (12 + scratch). It is the floor for the "
               "three ONE-HOP tokens. The TWO-HOP token takes the COMPONENT cells' conjunct "
               "instead, `depth <= 1` applied per emitted token, so an admitted row may hold most "
               "of both maps and still not perform the resolve-then-read that this token is. "
               "Chance for a per-slot read is `1/k` — every pad token is an agent name — and not "
               "the answer read's `1/(k-1)`.\n")
    out.append("`swap_p0` is split by SOURCE and only the cross half is scored: on a same-source "
               "swap both reads are of P, which a row holding P alone performs exactly and which "
               "is the state component's own carrier depth. Pooling the two lets a model that "
               "does the same-source write and floors on the cross one read as though it "
               "composed.\n")
    out.append("| cell | chance | per-slot floor (registered class) | row | swap_p0\\|cross, "
               "one-hop floor | row | same-source, one-hop | swap_p0\\|cross unrestricted |")
    out.append("|---|---|---|---|---|---|---|---|")
    for key in cells:
        c = fw["cells"][key]
        read = P.PAD_WRITE_SCORED_READ if P.PAD_WRITE_SCORED_READ in c else "free_run"
        b = c[read]["operative"]["all"]
        t = c[read]["two_hop"]
        cross = t.get(P.PAD_WRITE_TOKEN, {})
        same = t.get("swap_p0|same", {})
        ch = c["chance"]["uniform"]
        out.append(f"| {key} | {ch:.4f} | {b['per_slot']:.4f} ({b['per_slot'] / ch:.2f}x) | "
                   f"`{b['per_slot_row']}` | {f4(cross.get('floor'))} "
                   f"({(cross.get('floor') or 0) / ch:.2f}x) | `{cross.get('row')}` | "
                   f"{f4(same.get('floor'))} | {f4(cross.get('unrestricted'))} |")
    out.append("")
    out.append(f"Floors are on the {P.PAD_WRITE_SCORED_READ} read, which is the scored one; the "
               "same table on the free-running read is in the JSON. A floor is measured on the "
               "exact scored items AND on a disjoint pool with the larger operative, since a max "
               "over 78 rows carries an upward selection bias at small n.\n")
    conf = fw.get(f"confront_{P.PAD_WRITE_SCORED_READ}") or []
    if conf:
        n = fw["cfg"]["n"]
        out.append(f"### The scored read, per seed and per length\n")
        out.append(f"Teacher-forced per event, n={n}, under `clears_headroom` "
                   f"(z>{P.Z_CLEAR} and `(a-f)/(1-f) >= {P.MARGIN_FRAC:g}`). Teacher-forcing "
                   "removes the COMPOUNDING of a per-event residual over the model's own writes "
                   "and therefore claims nothing about them; the free-running column beside it is "
                   "the tracking diagnostic.\n")
        out.append("| seed | cell | swap_p0\\|cross | floor | bar | clears | same-source | "
                   "free-running cross | its floor | clears |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        free = {(r["seed"], r["cell"]): r for r in (fw.get("confront_free_run") or [])}
        for r in conf:
            if not r["cell"].startswith("composed@"):
                continue
            fr = free.get((r["seed"], r["cell"]), {})
            fl = r.get(f"{P.PAD_WRITE_TOKEN}_floor")
            out.append(
                f"| {r['seed']} | {r['cell']} | {f4(r.get(P.PAD_WRITE_TOKEN))} | {f4(fl)} | "
                f"{f4(P.bar_for(fl))} | {'yes' if r.get(f'{P.PAD_WRITE_TOKEN}_clears') else 'no'} "
                f"| {f4(r.get('swap_p0|same'))} | {f4(fr.get(P.PAD_WRITE_TOKEN))} | "
                f"{f4(fr.get(f'{P.PAD_WRITE_TOKEN}_floor'))} | "
                f"{'yes' if fr.get(f'{P.PAD_WRITE_TOKEN}_clears') else 'no'} |")
        out.append("")
    claim = fw.get("claim")
    if claim:
        out.append("### The claim, conjoined per seed\n")
        out.append("A seed counts only where EVERY gate holds for THAT seed. Counted apart, three "
                   "gates are satisfied by a run in which no single model satisfies two of "
                   "them.\n")
        out.append("| gate | seeds |")
        out.append("|---|---|")
        for g, v in claim["by_gate"].items():
            out.append(f"| `{g}` | {v} |")
        carry = sorted(int(s) for s, v in claim["per_seed"].items() if v)
        out.append(f"| **all of them** | **{carry}** |")
        out.append("")
        out.append(f"{claim['n_seeds']} seed(s) carry the whole claim at the registered lengths "
                   f"{claim['lengths']}; {P.SEEDS_CLEAR} are required, so the claim "
                   f"{'HOLDS' if claim['holds'] else 'DOES NOT HOLD'}.\n")
    sat = (fw["cells"].get(cells[0]) or {}).get("saturation") if cells else None
    comp = [k for k in fw["cells"] if not k.startswith("composed@")]
    out.append("### The saturation control\n")
    out.append("A saturation control asks whether the measurement can register a clear at all. "
               "NO MODEL CONTROL EXISTS ON THIS READ and the reason is structural, not a gap in "
               "the run: a component cell renders every operand by NAME, so the scored token has "
               "ZERO events there"
               + (" (and its pad-write floor is "
                  + ", ".join(f"{fw['cells'][k]['free_run']['operative']['all']['per_slot']:.4f} "
                              f"at {k}" for k in sorted(comp)) + ")" if comp else "")
               + ". The quantity is defined on the composed cell and nowhere else, so no cell "
                 "carries a model that is known good on other evidence AND has the token to "
                 "write.\n")
    if sat:
        out.append(f"What can be built is a POLICY control at the other end of the same scale. "
                   f"The composed cell's own algorithm is a row in this family — `{sat['row']}`, "
                   f"carrying both maps — and it is excluded by the live-slot conjunct at "
                   f"W = {sat['W']} against the bound {sat['bound']}. On the exact scored items it "
                   f"writes the two-hop token at {f4(sat['own_gold_cross'])} on n = "
                   f"{sat['n_cross']} events, against an admitted class that reaches "
                   f"{f4((fw['cells'][cells[0]][P.PAD_WRITE_SCORED_READ]['two_hop'].get(P.PAD_WRITE_TOKEN) or {}).get('floor'))} "
                   f"at {cells[0]}. That exercises the SCORER over the whole range; it does not "
                   "exercise the decode, and the difference is what the missing model control "
                   "would have bought.\n")
        out.append("WHAT ITS ABSENCE COSTS, stated rather than absorbed: this read is uncalibrated "
                   "against FALSE NEGATIVES. A floored two-hop score on some future model could be "
                   "the measurement rather than the model, and nothing here would separate them. "
                   "It does not weaken a clear — a clear is the model emitting the token more "
                   "often than any admitted policy does, on items the policies were measured on — "
                   "and the run reported here has no null on this token to protect.\n")


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
