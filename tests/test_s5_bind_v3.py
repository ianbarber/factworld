"""s5_bind_v3 — the SOURCE-STRUCTURE composition rung.

Two maps into agents run over one event stream (P: agents -> agents, rewritten by swaps;
B: objects -> agents under last-write-wins, rewritten by gives) and every event names its second
operand LIVE through one of them. The construct rests on properties this file pins one by one:

  the ablation is the SOURCE, not the time  — every reference is dynamic; within an event kind
                                              the CROSS and SAME renderings are identical in
                                              whitespace-token count and in register;
  the classes are matched                   — balanced in count, and matched in the write count
                                              and retrieval distance of the cell they read,
                                              which is what makes an interference effect common
                                              to the two and cancel in the contrast;
  the prompt alone determines gold          — an independent parser and replay reproduce the
                                              answer, so nothing is carried in meta;
  the floor is a ONE-STRUCTURE class        — a row may set the floor only if it holds at most
                                              one structure (W <= max(k,m)+1 under the stated W
                                              convention) and pays no more steps than the task,
                                              so the block-drop continuum, the partial-carry
                                              continuum and the demand resolver are all out by
                                              one argument;
  the COMPONENT rule is the same move       — its own algorithm holds no structure, so the bound
                                              goes on composition DEPTH (at most ONE hop against
                                              a carrier chain of 2 n_swap / k) and steps are held
                                              under that algorithm's MINIMUM per-item cost, not
                                              its mean. Both halves are attacked here by sweeping
                                              the truncated walk and the truncated give-scan over
                                              their whole parameter and asserting the admitted
                                              set is at chance — a test of the RULE, not of the
                                              registry;
  the cost convention is stated             — one step is a named thing, a backward walk IS
                                              charged for the events it scans and rejects, and
                                              the counter implements exactly that.

Runs with zero dependencies:  python3 tests/test_s5_bind_v3.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter
from dataclasses import fields, replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld import composition as C  # noqa: E402
from factworld import tasks as TK  # noqa: E402
from factworld.render import Renderer  # noqa: E402
from factworld.tokenizer import Tokenizer  # noqa: E402
from factworld.validity import (  # noqa: E402
    S5_BIND_V3_CHANCE_ROWS,
    S5_BIND_V3_REFERENCE_ROWS,
    S5_BIND_V3_ROWS,
    S5_BIND_V3_SURFACE_FEATURES,
    S5_BIND_V3_SURFACE_FEATURE_ACC,
    S5_BIND_V3_SURFACE_FEATURE_DEPTH,
    S5_BIND_V3_TRUNCATION_ROWS,
    S5_BIND_V3_MAX_DEPTH,
    floor_eligible,
    one_structure_bound,
    S5_BIND_V3_CKPT_ROWS,
    s5_bind_v3_admits,
    s5_bind_v3_block_drop,
    s5_bind_v3_carrier_hops,
    s5_bind_v3_ckpt_copy_per_slot,
    s5_bind_v3_ckpt_floors,
    s5_bind_v3_ckpt_lag,
    s5_bind_v3_ckpt_preds,
    s5_bind_v3_classify,
    s5_bind_v3_family_floors,
    s5_bind_v3_family_rows,
    s5_bind_v3_floor_basis,
    s5_bind_v3_floors,
    s5_bind_v3_give_scan,
    s5_bind_v3_guided_admits,
    s5_bind_v3_is_named,
    s5_bind_v3_needs,
    s5_bind_v3_operative_floor,
    s5_bind_v3_pad_admits,
    s5_bind_v3_pad_floorable,
    s5_bind_v3_pad_floors,
    s5_bind_v3_pad_max_width,
    s5_bind_v3_pad_operative_floor,
    s5_bind_v3_pad_reach,
    s5_bind_v3_partial_carry,
    s5_bind_v3_partial_carry_profile,
    s5_bind_v3_query_kind,
    s5_bind_v3_row_cost,
    s5_bind_v3_row_depth,
    s5_bind_v3_shape,
    s5_bind_v3_slot_profile,
    s5_bind_v3_surface_bound,
    s5_bind_v3_surface_depth,
    s5_bind_v3_surface_impls,
    s5_bind_v3_surface_loaded,
    s5_bind_v3_surface_price,
    s5_bind_v3_slot_moves,
    s5_bind_v3_task_cost,
    s5_bind_v3_task_cost_min,
    s5_bind_v3_task_depth,
    s5_bind_v3_trace_floor_basis,
    s5_bind_v3_trace_is_answer,
    s5_bind_v3_trace_operative_floor,
    s5_bind_v3_trace_slot,
    s5_bind_v3_trunc_walk,
    s5_bind_v3_work_match,
)
from factworld.world import Event  # noqa: E402

COMPOSED = ("s5_bind_v3", "s5_bind_local_v3")            # the STATE-query composed cells
COMPONENTS = ("s5_bind_v3_state", "s5_bind_v3_bind",
              "s5_bind_local_v3_state", "s5_bind_local_v3_bind")
ALL = COMPOSED + COMPONENTS

# The operative floor is a MAX over the admitted rows, so at a finite n it carries an upward
# selection bias of order the largest row's standard error even when every row sits at chance.
# The bound is stated as a multiple of informed chance and has to leave room for it: at
# N_FLOOR = 500 and k = 12 one row's standard error alone is 0.14 of chance.
N_FLOOR = 500
FLOOR_RATIO_MAX = 1.35

# Re-pinned when the components' grids moved to the WORK-MATCHED pairing. Every length that both
# grids contain hashes identically (s5_bind_local_v3_state@48, s5_bind_v3_bind@128), and neither
# composed cell moved at all, so the sampler is untouched and only the registered lengths changed.
GOLDENS = {
    "s5_bind_v3": {128: "a3f301a6769fe938", 192: "c0b0d78567ce1b29", 256: "147f0b5a10fc916f"},
    "s5_bind_v3_state": {43: "22fff0151054681f", 64: "296cf3e5fc8f13a5",
                         85: "8f2e28bebeafb168"},
    "s5_bind_v3_bind": {85: "a605d49fe1d077f5", 128: "1582867b26e59b3a",
                        171: "ae6827e0784bd43c"},
    "s5_bind_local_v3": {48: "f47aa0f745e19a77", 64: "571f81b8ecda65b9",
                         96: "1ef58f59227c3736"},
    "s5_bind_local_v3_state": {17: "ce56e584a72e3594", 23: "9777f0e1c64ee34f",
                               34: "611fa288c1c7d3ef", 48: "056763639c7c4882",
                               80: "7bbaa8887e2f9b49", 128: "4dc43c513befb989"},
    "s5_bind_local_v3_bind": {31: "6a66d9d1d1b00e89", 41: "92110436351c3995",
                              62: "f8757eb2cd008ab7"},
}


def _hash(examples) -> str:
    return hashlib.sha256(
        "\n".join(f"{e.prompt}\t{e.answer}" for e in examples).encode()).hexdigest()[:16]


def _statements(prompt: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.?]*[.?]", prompt) if s.strip()]


def _short(name, n=40):
    spec = TK.CANONICAL[name]
    return spec, TK.generate(spec, "test", n=n, length=spec.eval_lengths[0])


# --- registry -----------------------------------------------------------------------------

def test_registry_contract():
    for name in ALL:
        spec = TK.CANONICAL[name]
        assert spec.name == name and spec.family == "s5_bind" and spec.source_ablation
        assert spec.kind == "experimental" and name not in TK.REPORTED
        assert spec.n_objects_active <= spec.k
        assert TK.spec_for(name) is spec
    # the composed arms mix both event kinds and render references; each component contains
    # exactly one kind and names its operands
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        assert spec.event_kinds == "both" and not spec.named_operands
        assert 0.0 < spec.p_cross < 1.0 and spec.query_arm == "state"
    for name in COMPONENTS:
        spec = TK.CANONICAL[name]
        assert spec.named_operands and spec.event_kinds in ("swap", "give")
        assert spec.query_arm == ("state" if spec.event_kinds == "swap" else "bind")


def test_p_swap_is_the_write_count_matching():
    """The sampler's p_swap is not a taste knob: a swap moves TWO pointers and a give writes ONE
    object, so the two structures' cells accumulate writes at equal rates iff
    2 p_swap / k = (1 - p_swap) / m. Every composed spec sits on that solution."""
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        want = k / (k + 2.0 * m)
        assert abs(spec.p_swap - want) < 1e-9, f"{name}: p_swap={spec.p_swap}, want {want}"


def test_the_component_grids_are_work_matched_to_the_composed_one():
    """THE PAIRING THE COMPARISON RESTS ON. A composed stream of length L holds p_swap L swaps
    and (1 - p_swap) L gives, so a component read at the composed cell's OWN L carries 1/p_swap
    (state) or 1/(1 - p_swap) (retrieval) times the work — the confound that made a 5.7-hop
    composed cell readable against a 16-hop state cell. Each component's registered grid is
    therefore the composed cell's own event counts, and the two are then equal on the carrier
    chain, which is what "same depth" means here. The count is a sample mean, so the registered
    partner is pinned to a fixed probe and matched here to within one event."""
    P = _protocol()
    for composed, state, bind in (("s5_bind_v3", "s5_bind_v3_state", "s5_bind_v3_bind"),
                                  ("s5_bind_local_v3", "s5_bind_local_v3_state",
                                   "s5_bind_local_v3_bind")):
        cspec = TK.CANONICAL[composed]
        for L in cspec.eval_lengths:
            ex = TK.generate(cspec, "test", n=P.WORK_PROBE_N, length=L)
            ns, ng = s5_bind_v3_shape(ex)
            want = s5_bind_v3_work_match(ns, ng)
            for key, comp in (("state", state), ("bind", bind)):
                near = min(abs(want[key] - x) for x in TK.CANONICAL[comp].eval_lengths)
                assert near <= 1, (composed, L, key, want[key],
                                   TK.CANONICAL[comp].eval_lengths)
            # and no p_swap removes the need for the pairing: both legs are strictly under L
            assert 0 < ns < L and 0 < ng < L, (composed, L, ns, ng)
            sex = TK.generate(TK.CANONICAL[state], "test", n=40, length=want["state"])
            sns, _sng = s5_bind_v3_shape(sex)
            assert abs(s5_bind_v3_carrier_hops(cspec.k, ns)
                       - s5_bind_v3_carrier_hops(cspec.k, sns)) < 0.35, (composed, L)


def test_source_ablation_defaults_off_everywhere_else():
    for reg in (TK.CANONICAL, TK.RETIRED):
        for name, spec in reg.items():
            if name in ALL:
                continue
            assert not spec.source_ablation, name
            assert spec.event_kinds == "both" and not spec.named_operands, name


def test_frozen_stream_goldens():
    for name, per_len in GOLDENS.items():
        spec = TK.CANONICAL[name]
        assert tuple(per_len) == tuple(spec.eval_lengths)
        for L, want in per_len.items():
            got = _hash(TK.generate(spec, "test", n=25, length=L))
            assert got == want, f"{name}@L{L}: frozen-spec immutability VIOLATED ({got})"


def test_determinism():
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        for split, L in (("test", spec.eval_lengths[0]), ("train", None)):
            a = TK.generate(spec, split, n=20, length=L)
            b = TK.generate(spec, split, n=20, length=L)
            assert [(x.prompt, x.answer, x.meta) for x in a] == \
                   [(x.prompt, x.answer, x.meta) for x in b], f"{name} {split}"


# --- the ablation moves the SOURCE and nothing else ---------------------------------------

def test_cross_and_same_are_identical_in_length_and_register():
    """Within an event kind the two classes are the same number of whitespace tokens and the
    same shape — "the agent {slot} {verb} to at this point". That is the whole ablation: only
    which structure the reference reads varies."""
    for name in COMPOSED:
        _spec, ex = _short(name)
        by_kind: dict[str, set[int]] = {}
        seen = Counter()
        for e in ex:
            for st in _statements(e.prompt):
                toks = Renderer.normalize(st).split()
                if "swaps" in toks:
                    kind = "swap"
                elif "gives" in toks:
                    kind = "give"
                else:
                    continue
                by_kind.setdefault(kind, set()).add(len(toks))
                seen[(kind, "belongs" in toks)] += 1
                # the register: the reference is always "the agent X <verb> to at this point"
                i = toks.index("agent")
                assert toks[i + 2] in ("points", "belongs") and toks[i + 3] == "to", st
                assert toks[i + 4:i + 7] == ["at", "this", "point"], st
        for kind, lens in by_kind.items():
            assert len(lens) == 1, f"{name}: {kind} renders at {sorted(lens)} tokens"
        # both classes present in both kinds
        for kind in ("swap", "give"):
            assert seen[(kind, True)] > 0 and seen[(kind, False)] > 0, f"{name} {kind}"


def test_every_reference_is_live():
    """There is no static reading. A v3 prompt never says "at the start" except in the stated
    fact block, so no reference is a header lookup and the composition class cannot be a
    read-history predicate."""
    for name in COMPOSED:
        _spec, ex = _short(name, n=10)
        for e in ex:
            for st in _statements(e.prompt):
                toks = Renderer.normalize(st).split()
                if "swaps" in toks or "gives" in toks:
                    assert "start" not in toks, st
                    assert toks[-4:-1] == ["at", "this", "point"], st


def test_the_clause_to_class_map_flips_between_event_kinds():
    """A pure SURFACE-FORM failure cannot load on the contrast: the "belongs to" clause is CROSS
    on a swap and SAME on a give, so a solver that is simply worse at one clause slips on half
    of each class."""
    for name in COMPOSED:
        _spec, ex = _short(name, n=20)
        for e in ex:
            rec = C.read(e.prompt)
            for ev in rec["events"]:
                kind, _t, _r, src = ev
                if kind == C.SWAP:
                    assert C.is_cross(ev) == (src == "B")
                else:
                    assert C.is_cross(ev) == (src == "P")


def _class_read_history(ex):
    """Per (kind, class): count, mean retrieval distance, mean write count of the cell read —
    recomputed off the rendered prompt."""
    cnt, wsum, dsum = Counter(), Counter(), Counter()
    for e in ex:
        rec = C.read(e.prompt)
        P, B = dict(rec["P0"]), dict(rec["B0"])
        wcnt, last = {}, {}
        for j, ev in enumerate(rec["events"]):
            key = (ev[0], "cross" if C.is_cross(ev) else "same")
            cell = (ev[3], ev[2])
            cnt[key] += 1
            wsum[key] += wcnt.get(cell, 0)
            dsum[key] += j - last.get(cell, -1)
            x = C._resolve(ev, P, B, rec["P0"], rec["B0"])
            if ev[0] == C.SWAP:
                P[ev[1]], P[x] = P[x], P[ev[1]]
                for g in (ev[1], x):
                    wcnt[("P", g)] = wcnt.get(("P", g), 0) + 1
                    last[("P", g)] = j
            else:
                B[ev[1]] = x
                wcnt[("B", ev[1])] = wcnt.get(("B", ev[1]), 0) + 1
                last[("B", ev[1])] = j
    return cnt, wsum, dsum


def test_the_classes_are_matched_WITHIN_KIND():
    """The property the statistic rests on, measured where the primary reads it.

    POOLING HIDES IT. The primary is kind-balanced, so what has to be matched is the read history
    inside each event kind — and the two kinds carry the two clauses in opposite directions, so a
    within-kind gap can cancel in the pooled column and still load on the contrast at full size.
    The sampler matches the two candidate reference cells at the draw (TaskSpec.match_reads); this
    is that rule's outcome on the exact scored items."""
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        ex = TK.generate(spec, "test", n=150, length=spec.eval_lengths[0])
        cnt, wsum, dsum = _class_read_history(ex)
        for kind in (C.SWAP, C.GIVE):
            nx, nz = cnt[(kind, "cross")], cnt[(kind, "same")]
            assert 0.45 <= nx / (nx + nz) <= 0.55, f"{name} {kind}: cross share"
            dx, dz = dsum[(kind, "cross")] / nx, dsum[(kind, "same")] / nz
            wx, wz = wsum[(kind, "cross")] / nx, wsum[(kind, "same")] / nz
            assert abs(dx - dz) / max(dx, dz) < 0.15, \
                f"{name} {kind}: distances {dx:.2f} vs {dz:.2f}"
            assert abs(wx - wz) / max(wx, wz) < 0.15, \
                f"{name} {kind}: write counts {wx:.2f} vs {wz:.2f}"


def test_the_matched_draw_is_what_closes_the_within_kind_gap():
    """Turning the matching off leaves a gap several times the matched one, in OPPOSITE
    directions on the two kinds — which is why it survived a pooled check."""
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        assert spec.match_reads >= 1
        L = spec.eval_lengths[0]
        gaps = {}
        for tag, s in (("on", spec), ("off", spec.scaled(match_reads=0))):
            cnt, _w, dsum = _class_read_history(TK.generate(s, "test", n=150, length=L))
            gaps[tag] = {kd: (dsum[(kd, "cross")] / cnt[(kd, "cross")])
                         / (dsum[(kd, "same")] / cnt[(kd, "same")]) - 1.0
                         for kd in (C.SWAP, C.GIVE)}
        assert gaps["off"][C.SWAP] > 0.05 and gaps["off"][C.GIVE] < -0.05, gaps["off"]
        for kd in (C.SWAP, C.GIVE):
            assert abs(gaps["on"][kd]) < abs(gaps["off"][kd]), (name, kd, gaps)


def test_no_pin_keeps_the_cross_class_pure():
    """A CROSS reference onto an object whose current holder a P-only solver could already
    compute asks nothing of the other structure. TaskSpec.no_pin refuses exactly that event; the
    density it holds at zero is measured off the rendered prompt."""
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        for gate, want in ((True, 0.0), (False, 0.15)):
            ex = TK.generate(spec.scaled(no_pin=gate), "test", n=60,
                             length=spec.eval_lengths[0])
            live = tot = 0
            for e in ex:
                rec = C.read(e.prompt)
                P, B = dict(rec["P0"]), dict(rec["B0"])
                prov = {o: None for o in rec["B0"]}
                for kind, tgt, ref, src in rec["events"]:
                    x = C._resolve((kind, tgt, ref, src), P, B, rec["P0"], rec["B0"])
                    if kind == C.SWAP:
                        if src == "B":
                            tot += 1
                            live += int(prov[ref] is not None)
                        P[tgt], P[x] = P[x], P[tgt]
                        for g in (tgt, x):
                            for o in prov:
                                if prov[o] == g:
                                    prov[o] = None
                    else:
                        B[tgt] = x
                        prov[tgt] = ref if src == "P" else prov[ref]
            rate = live / tot
            if gate:
                assert rate == 0.0, f"{name}: no_pin leaves {rate:.3f} derivable cross reads"
            else:
                assert rate > want, f"{name}: the channel the gate closes is only {rate:.3f}"


# --- gold comes from the prompt -----------------------------------------------------------

def test_prompt_alone_determines_the_answer():
    """An independent parser and replay — no meta, no sampler — reproduce gold on every item."""
    for name in ALL:
        spec = TK.CANONICAL[name]
        for L in spec.eval_lengths[:2]:
            ex = TK.generate(spec, "test", n=40, length=L)
            for e in ex:
                rec = C.read(e.prompt)
                assert rec is not None, name
                assert C.answer_of(rec, C.replay(rec)) == e.answer, f"{name}@{L}"


def test_query_gates_hold_on_every_item():
    for name in COMPOSED + ("s5_bind_v3_state", "s5_bind_local_v3_state"):
        spec = TK.CANONICAL[name]
        L = spec.eval_lengths[0]
        tail_lo = L - max(1, int(round(spec.q_tail * L)))
        for e in TK.generate(spec, "test", n=40, length=L):
            rec = C.read(e.prompt)
            q = rec["query"][1]
            P, B = dict(rec["P0"]), dict(rec["B0"])
            moves, last = 0, -1
            for j, ev in enumerate(rec["events"]):
                x = C._resolve(ev, P, B, rec["P0"], rec["B0"])
                if ev[0] == C.SWAP:
                    if q in (ev[1], x):
                        moves += 1
                        last = j
                    P[ev[1]], P[x] = P[x], P[ev[1]]
                else:
                    B[ev[1]] = x
            assert moves >= 2, f"{name}: queried agent moved {moves} times"
            assert P[q] != rec["P0"][q], f"{name}: answer equals the stated pointer"
            assert last >= tail_lo, f"{name}: last move at {last}, gate {tail_lo}"


def test_answer_balance():
    for name in ALL:
        spec = TK.CANONICAL[name]
        ans = [e.answer for e in TK.generate(spec, "test", n=300, length=spec.eval_lengths[0])]
        top = Counter(ans).most_common(1)[0][1] / len(ans)
        assert top < 2.5 / spec.k, f"{name}: majority answer {top:.3f}"


def test_event_trace_replays_to_the_scored_answer():
    for name in ("s5_bind_local_v3", "s5_bind_local_v3_state", "s5_bind_local_v3_bind"):
        spec = TK.CANONICAL[name]
        assert spec.event_trace
        for e in TK.generate(spec, "test", n=10, length=spec.eval_lengths[0]):
            snaps = e.meta["trace"].split()
            width = spec.k + spec.n_objects_active
            assert len(snaps) == width * e.length
            final = snaps[-width:]
            rec = C.read(e.prompt)
            agents = sorted(set(rec["P0"]) | set(rec["B0"].values()), key=lambda s: int(s[1:]))
            objs = sorted(rec["B0"], key=lambda s: int(s[1:]))
            P, B = C.replay(rec)
            if spec.query_arm == "state":
                assert final[:spec.k] == [P[a] for a in agents[:spec.k]]
            else:
                assert final[spec.k:] == [B[o] for o in objs]


# --- surfaces -----------------------------------------------------------------------------

def test_every_surface_round_trips_through_the_parser():
    r = Renderer()
    for name in ALL:
        spec = TK.CANONICAL[name]
        for e in TK.generate(spec, "test", n=8, length=spec.eval_lengths[0]):
            for st in _statements(e.prompt):
                rec = r.parse(st)
                assert rec is not None, st
                if rec["type"] == "event":
                    assert r.render_event(rec["event"], step=rec["step"]) == st
                elif rec["type"] == "pointer":
                    assert r.render_pointer(rec["agent"], rec["target"]) == st
                elif rec["type"] == "belongs":
                    assert r.render_belongs(rec["object"], rec["holder"]) == st
                else:
                    assert rec["type"] == "query" and rec["family"].startswith("s5bind3_")


def test_pre_existing_surfaces_are_untouched():
    """The v3 router keys on tokens no other statement type in the suite emits, so adding it
    moved no other family's parse."""
    r = Renderer()
    for kind, args in (("give", ("o0", "g1")), ("swap_role", ("g0", "g1")),
                       ("swap_a0_ref", ("g0", "g1")), ("swap_roles_now", ("g0", "o1")),
                       ("give_role_start", ("o0", "r1"))):
        s = r.render_event(Event(kind, args), step="s0")
        assert r.parse(s)["event"].kind == kind, s
    assert r.parse(r.render_role("g0", "r1"))["type"] == "role"
    assert r.parse(r.render_query("state_hard", target="g0"))["family"] == "state_hard"


def test_tokenizer_covers_every_surface_with_no_unknowns():
    worlds = [TK.build_world(TK.CANONICAL[n])[0] for n in ALL]
    tok = Tokenizer.build(worlds, Renderer(), max_step=256)
    seen = 0
    for name in ALL:
        spec = TK.CANONICAL[name]
        for L in spec.eval_lengths:
            for e in TK.generate(spec, "test", n=4, length=L):
                for text in (e.prompt, e.answer, e.meta.get("trace", ""),
                             e.meta.get("interleaved_prompt", "")):
                    if not text:
                        continue
                    ids = tok.encode(text)
                    seen += len(ids)
                    assert tok.unk_id not in ids, f"{name}@{L}: <unk> in {text[:80]!r}"
                    assert tok.decode(ids) == text, f"{name}@{L}: decode round-trip"
    assert seen > 100_000


# --- the W convention and the one-structure floor class -------------------------------------

def test_the_w_convention_is_one_scratch_plus_the_structures_the_row_needs():
    """W1-W5 in factworld.validity, made checkable: W = 1 + (k if the row needs P) + (m if it
    needs B), the scratch register is charged to every row INCLUDING the task, and a row needs a
    structure when it reads it live OR when the answer comes out of it."""
    k, m, ns, ng = 12, 12, 43, 85
    guess = ("uniform", "uniform_non_initial", "initial_only")
    scan = ("last_write_1hop", "last_swap_ref", "uniform_anti_surface")
    for row in S5_BIND_V3_ROWS:
        for query in ("state", "bind"):
            np_, nb = s5_bind_v3_needs(row, query)
            w, _s = s5_bind_v3_row_cost(row, k, m, ns, ng, query)
            if row in guess:                     # holds only its own answer
                want = 1
            elif row in scan:                    # one carrier plus the scratch register
                want = 2
            else:                                # 1 + the structures it needs; final_state twice
                want = 1 + (k if np_ else 0) + (m if nb else 0)
                if row == "final_state":
                    want += k + m
            assert w == want, (row, query, w, want)
    # the task pays the same scratch register, so the task and the rows are comparable at all
    wt, _st = s5_bind_v3_task_cost(k, m, ns, ng)
    assert wt == k + m + 1
    assert s5_bind_v3_task_cost(k, m, ns, ng, named=True)[0] == 2


def test_one_structure_b_holds_both_maps_on_a_state_query():
    """The mis-costing the rule replaces. one_structure_B resolves out of B, but its replay
    writes P on every swap and a STATE query reads the answer out of P, so it holds k + m + 1
    and is not a floor row. On a BIND query the same policy holds m + 1 and is."""
    k, m, ns, ng = 12, 12, 43, 85
    assert s5_bind_v3_row_cost("one_structure_B", k, m, ns, ng, "state")[0] == k + m + 1
    assert s5_bind_v3_row_cost("one_structure_B", k, m, ns, ng, "bind")[0] == m + 1
    assert not s5_bind_v3_classify(k, m, ns, ng, query="state")["one_structure_B"]
    assert s5_bind_v3_classify(k, m, ns, ng, query="bind")["one_structure_B"]
    # and the mirror, so the rule is not one-sided
    assert s5_bind_v3_row_cost("one_structure_P", k, m, ns, ng, "state")[0] == k + 1
    assert s5_bind_v3_row_cost("one_structure_P", k, m, ns, ng, "bind")[0] == k + m + 1


def test_the_bound_is_one_structure_and_both_continua_go_out_by_it():
    """ONE inequality closes both continua. Every block-drop member and every partial-carry
    member with j >= 1 holds both structures, so W exceeds max(k,m)+1; j = 0 does not, and j = 0
    IS one_structure_P."""
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        ex = TK.generate(spec, "test", n=30, length=spec.eval_lengths[0])
        ns, ng = s5_bind_v3_shape(ex)
        bound = one_structure_bound(k, m)
        assert bound == max(k, m) + 1 < s5_bind_v3_task_cost(k, m, ns, ng)[0]
        cls = s5_bind_v3_classify(k, m, ns, ng)
        for row in S5_BIND_V3_TRUNCATION_ROWS:
            w, _s = s5_bind_v3_row_cost(row, k, m, ns, ng)
            assert w == k + m + 1 > bound and not cls[row], row
        # the partial-carry family, priced by W2 of the convention: j cells cost j slots
        for j in range(m + 1):
            assert (k + j + 1 <= bound) == (j == 0), (j, k + j + 1, bound)
        assert cls["one_structure_P"], "j = 0 must stay admitted — it is the registered row"


def test_the_admitted_rows_all_hold_at_most_one_structure():
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        ex = TK.generate(spec, "test", n=30, length=spec.eval_lengths[0])
        ns, ng = s5_bind_v3_shape(ex)
        _wt, st = s5_bind_v3_task_cost(k, m, ns, ng)
        cls = s5_bind_v3_classify(k, m, ns, ng)
        admitted = [r for r, ok in cls.items() if ok]
        assert set(admitted) >= {"one_structure_P", "stated_reference", "last_swap_ref",
                                 "uniform_non_initial"}
        assert "one_structure_B" not in admitted, "it holds P as well on a state query"
        for row in admitted:
            w, s = s5_bind_v3_row_cost(row, k, m, ns, ng)
            assert floor_eligible(w, s, one_structure_bound(k, m), st), (row, w, s)


def test_the_component_arms_separate_on_steps_and_not_on_slots():
    """A component's cheapest correct algorithm already holds NO structure (W = 2), so the
    one-structure bound is vacuous and the axis that separates is STEPS. That admits the one-hop
    read on the STATE component — the carrier walk truncated after one hop, which stops early —
    and excludes it on the RETRIEVAL component, where the same read IS the whole algorithm."""
    for name in COMPONENTS:
        spec = TK.CANONICAL[name]
        ex = TK.generate(spec, "test", n=20, length=spec.eval_lengths[0])
        assert s5_bind_v3_is_named(ex)
        ns, ng = s5_bind_v3_shape(ex)
        k, m = spec.k, spec.n_objects_active
        query = spec.query_arm
        cls = s5_bind_v3_classify(k, m, ns, ng, True, query)
        assert not cls["one_structure_P"], "a component floor row may not carry a map"
        assert cls["uniform_non_initial"] and cls["initial_only"]
        assert cls["last_write_1hop"] == (query == "state"), name


def test_a_definitional_floor_is_labelled_and_a_measured_one_is_not():
    """The thing this replaces printed 1.00x on every component cell as though a policy had been
    measured up to it. On the retrieval component it IS a definition and the basis says so; on
    the state component a real row bounds it and the basis says that instead."""
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        ex = TK.generate(spec, "test", n=20, length=spec.eval_lengths[0])
        ns, ng = s5_bind_v3_shape(ex)
        assert s5_bind_v3_floor_basis(spec.k, spec.n_objects_active, ns, ng, False) == "measured"
    for name, want in (("s5_bind_v3_state", "measured"), ("s5_bind_v3_bind", "chance"),
                       ("s5_bind_local_v3_state", "measured"),
                       ("s5_bind_local_v3_bind", "chance")):
        spec = TK.CANONICAL[name]
        ex = TK.generate(spec, "test", n=40, length=spec.eval_lengths[0])
        ns, ng = s5_bind_v3_shape(ex)
        got = s5_bind_v3_floor_basis(spec.k, spec.n_objects_active, ns, ng, True,
                                     spec.query_arm)
        assert got == want, f"{name}: floor basis {got}, want {want}"
    # what would have bounded the retrieval component is its own algorithm, which ties it
    spec = TK.CANONICAL["s5_bind_v3_bind"]
    fl = s5_bind_v3_floors(TK.generate(spec, "test", n=40, length=spec.eval_lengths[0]),
                           spec.k, spec.n_objects_active)
    assert fl["last_write_1hop"] > 0.9


def test_every_admitted_member_of_the_partial_carry_family_is_at_chance():
    """The j-profile is what makes the bound checkable rather than asserted: every ADMITTED
    member sits at chance and the family only climbs above it where the bound has excluded it."""
    spec = TK.CANONICAL["s5_bind_local_v3"]
    k, m = spec.k, spec.n_objects_active
    ex = TK.generate(spec, "test", n=300, length=64)
    prof = s5_bind_v3_partial_carry_profile(ex, m)
    chance = 1.0 / (k - 1)
    assert prof[0] <= FLOOR_RATIO_MAX * chance, f"j=0 is admitted and must be at chance: {prof}"
    assert prof[m] == 1.0, "j = m carries both maps and is the oracle"
    assert prof[m - 1] > 2.0 * chance, f"the excluded end must be far above chance: {prof}"
    for j in range(m + 1):
        admitted = k + j + 1 <= one_structure_bound(k, m)
        assert admitted == (j == 0)
        if admitted:
            assert prof[j] <= FLOOR_RATIO_MAX * chance, (j, prof)


# --- the COMPONENT rule: one hop, and the algorithm's MINIMUM cost ---------------------------

def test_the_component_bound_is_one_hop_against_a_chain_of_many():
    """The gap in kind on the axis a component separates on. The state component's own algorithm
    chains the whole carrier walk — 2 n_swap / k hops — and a floor row may chain ONE, exactly as
    a composed floor row may hold one structure against two. On the retrieval component the
    algorithm is itself one hop, so the bound is vacuous there and says so.

    Checked at EVERY registered length, because the work-matched grid puts the shallowest state
    rung at the composed cell's own carrier chain — 5.67 hops at k=6/L=17 and 7.17 at k=12/L=43 —
    so the gap is 5x at the shallow end rather than the 8x the composed cell's own length gave."""
    for name in COMPONENTS:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        depths = []
        for L in spec.eval_lengths:
            ex = TK.generate(spec, "test", n=40, length=L)
            ns, ng = s5_bind_v3_shape(ex)
            depths.append(s5_bind_v3_task_depth(k, m, ns, ng, True, spec.query_arm))
            if spec.query_arm == "state":
                assert depths[-1] == round(2 * ns / k), (name, L, depths[-1])
            else:
                assert depths[-1] == S5_BIND_V3_MAX_DEPTH == 1, (name, L, depths[-1])
        if spec.query_arm == "state":
            assert min(depths) >= 5 * S5_BIND_V3_MAX_DEPTH, (name, depths)
            assert max(depths) >= 10 * S5_BIND_V3_MAX_DEPTH, (name, depths)
    # and on a composed cell the depth bound is not what is applied: its own rule is on W
    for name in COMPOSED:
        spec, ex = _short(name, 40)
        ns, ng = s5_bind_v3_shape(ex)
        assert s5_bind_v3_task_depth(spec.k, spec.n_objects_active, ns, ng) == ns + ng
        assert s5_bind_v3_classify(spec.k, spec.n_objects_active, ns, ng)["one_structure_P"]


def test_the_step_bound_is_the_algorithms_minimum_and_not_its_mean():
    """One word, and it is the whole retrieval floor. A truncated give-scan reading fewer events
    than the algorithm's MEAN scan is far above chance and is EXACTLY that algorithm on the items
    it answers — it pays what the algorithm pays there. Only the per-item MINIMUM excludes it."""
    for name in ("s5_bind_v3_bind", "s5_bind_local_v3_bind"):
        spec = TK.CANONICAL[name]
        L = spec.eval_lengths[0]
        ex = TK.generate(spec, "test", n=N_FLOOR, length=L)
        k, m = spec.k, spec.n_objects_active
        ns, ng = s5_bind_v3_shape(ex)
        mean = s5_bind_v3_task_cost(k, m, ns, ng, True, "bind")[1]
        smin = s5_bind_v3_task_cost_min(k, m, ns, ng, True, "bind")
        assert smin < mean, (name, smin, mean)
        d_over = (mean - 3) // 2                     # under the mean, over the minimum
        row = f"give_scan_d{d_over}"
        _w, s = s5_bind_v3_row_cost(row, k, m, ns, ng, "bind")
        assert smin <= s <= mean, (name, s, smin, mean)
        assert not s5_bind_v3_admits(row, k, m, ns, ng, True, "bind"), name
        assert s5_bind_v3_give_scan(ex, d_over) > 2.0 / (k - 1), name


def test_no_truncation_of_a_cells_own_algorithm_is_admitted_on_either_axis():
    """THE TEST OF THE RULE, not of the registry: it sweeps the four truncation families over
    their whole parameter rather than checking the rows somebody registered.

    W axis (composed cell): partial carry at every j >= 1 and block drop at every (width,
    position) hold both structures and are out. STEP axis (component cell): every member that can
    chain more than one hop is out however its parameter is written, and every member the rule
    ADMITS sits at chance — which is the claim the floor rests on."""
    for name in COMPOSED:                            # the W axis
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        bound = one_structure_bound(k, m)
        for j in range(1, m + 1):                    # carry P and j of the m holder cells
            assert not floor_eligible(k + j + 1, 0, bound, 10 ** 9), (name, j)
        for _w in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.9):   # every block-drop member
            assert not floor_eligible(k + m + 1, 0, bound, 10 ** 9), name
    for name in COMPONENTS:                          # the STEP axis
        spec = TK.CANONICAL[name]
        L = spec.eval_lengths[0]
        ex = TK.generate(spec, "test", n=N_FLOOR, length=L)
        k, m = spec.k, spec.n_objects_active
        ns, ng = s5_bind_v3_shape(ex)
        q, ch = spec.query_arm, 1.0 / (k - 1)
        step = max(1, L // 16)
        if q == "state":
            rows = [f"trunc_walk_T{t}" for t in range(1, L, step)]
            rows += [f"trunc_walk_drop{c}" for c in range(1, L, step)]
        else:
            rows = [f"give_scan_d{d}" for d in range(1, L + 1, step)]
        assert len(rows) >= 16
        for row in rows:
            kind = row.rstrip("0123456789")
            p = int(row[len(kind):])
            admitted = s5_bind_v3_admits(row, k, m, ns, ng, True, q)
            if s5_bind_v3_row_depth(row, q, L) > S5_BIND_V3_MAX_DEPTH:
                assert not admitted, f"{name}: {row} chains more than one hop and is admitted"
            if kind == "give_scan_d":
                acc = s5_bind_v3_give_scan(ex, p)
            else:
                acc = s5_bind_v3_trunc_walk(ex, p if kind == "trunc_walk_T" else max(0, L - p))
            if admitted:
                assert acc <= FLOOR_RATIO_MAX * ch, \
                    f"{name}: {row} is ADMITTED and reads {acc / ch:.2f}x chance"


def test_the_two_parameterisations_of_the_truncated_walk_get_one_verdict():
    """The reason the rule is on hops and not on steps. At one cell ``trunc_walk_drop{c}`` and
    ``trunc_walk_T{L-c}`` are the same policy; a rule that reads the step cost as a constant
    admits the second and excludes the first. Cost, depth and verdict must agree on both."""
    for name in ("s5_bind_v3_state", "s5_bind_local_v3_state"):
        spec = TK.CANONICAL[name]
        L = spec.eval_lengths[0]
        ex = TK.generate(spec, "test", n=120, length=L)
        k, m = spec.k, spec.n_objects_active
        ns, ng = s5_bind_v3_shape(ex)
        for c in (1, 2, 5, 17, L // 2, L - 1):
            a, b = f"trunc_walk_drop{c}", f"trunc_walk_T{L - c}"
            assert s5_bind_v3_row_cost(a, k, m, ns, ng) == s5_bind_v3_row_cost(b, k, m, ns, ng)
            assert s5_bind_v3_row_depth(a, "state", L) == s5_bind_v3_row_depth(b, "state", L)
            assert (s5_bind_v3_admits(a, k, m, ns, ng, True, "state")
                    == s5_bind_v3_admits(b, k, m, ns, ng, True, "state")), (name, c)
        # and the excluded end is the cell's own algorithm with ONE event dropped, one step under
        # it: that is what has to be visible in the profile rather than hidden by a step rule
        assert not s5_bind_v3_admits("trunc_walk_drop1", k, m, ns, ng, True, "state")
        smin = s5_bind_v3_task_cost_min(k, m, ns, ng, True, "state")
        assert s5_bind_v3_row_cost("trunc_walk_drop1", k, m, ns, ng)[1] == smin - 1
        assert s5_bind_v3_trunc_walk(ex, L - 1) > 3.0 / (k - 1)


def test_the_retrieval_floor_is_the_samplers_window_to_the_event():
    """Why the retrieval component's floor is informed chance, as a proof rather than a
    definition: the sampler pins the queried object's resolving write at least L - 1 - hi events
    from the end, and the rule admits exactly the budgets that cannot reach it."""
    for name in ("s5_bind_v3_bind", "s5_bind_local_v3_bind"):
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        for L in spec.eval_lengths:
            ex = TK.generate(spec, "test", n=N_FLOOR, length=L)
            ns, ng = s5_bind_v3_shape(ex)
            _lo, hi = TK.s5_bind_v3_bind_window(L)
            reach = L - hi                          # the first budget that can read the write
            assert s5_bind_v3_give_scan(ex, reach - 1) == 0.0, (name, L)
            assert s5_bind_v3_give_scan(ex, reach) > 0.0, (name, L)
            assert s5_bind_v3_admits(f"give_scan_d{reach - 1}", k, m, ns, ng, True, "bind")
            assert not s5_bind_v3_admits(f"give_scan_d{reach}", k, m, ns, ng, True, "bind")
            # so every admitted family member resolves nothing at all
            fam = s5_bind_v3_family_floors(ex, k, m, True, "bind")
            cls = s5_bind_v3_classify(k, m, ns, ng, True, "bind", rows=tuple(fam))
            assert fam and all(v == 0.0 for r, v in fam.items() if cls[r]), (name, L)


def test_the_retrieval_scan_is_priced_at_the_window_and_the_two_counters_agree():
    """``validity`` and ``composition.cost_isolated_bind`` price the same algorithm on the same
    items. The pricing this replaces read the scan as L / (n_give / m) = m and understated it
    5.7x at L = 256 — 27 steps against a measured 152.4.

    The mean carries a ``+ m`` tail term that overshoots on the SHALLOWEST retrieval rung of each
    grid — the work-matched partner of the shortest composed cell — so the envelope is pinned at
    12% there and at 2% on every other rung, rather than one loose number covering both. Only the
    MINIMUM enters admission, and it is exact."""
    for name in COMPONENTS:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        for L in spec.eval_lengths:
            ex = TK.generate(spec, "test", n=200, length=L)
            ns, ng = s5_bind_v3_shape(ex)
            want = s5_bind_v3_task_cost(k, m, ns, ng, True, spec.query_arm)[1]
            cost = (C.cost_isolated_bind if spec.query_arm == "bind" else C.cost_isolated_state)
            got = sum(cost(C.read(e.prompt), k, m)[0] for e in ex) / len(ex)
            tol = 0.12 if (spec.query_arm == "bind" and L == spec.eval_lengths[0]) else 0.02
            assert abs(got - want) <= tol * want, f"{name}@{L}: validity {want}, counter {got}"
            assert s5_bind_v3_task_cost_min(k, m, ns, ng, True, spec.query_arm) <= want


def test_depth_is_a_property_of_the_policy_and_an_unpriced_row_raises():
    """THE DEFAULT WAS THE DEFECT. ``s5_bind_v3_row_depth`` used to be a name table with
    ``return 1 << 30`` as its fallback, so any row the table did not list was silently EXCLUDED —
    and exclusion lowers a floor, which is the direction that invalidates a "cleared the floor"
    reading. An unpriced policy now stops the classification instead of quietly passing it.

    Every registered row and every swept family member still gets a depth, so the raise cannot
    fire on the registry itself."""
    for row in ("a_policy_nobody_priced", "trunc_walk", "give_scan", "window_", "surface"):
        try:
            s5_bind_v3_row_depth(row, "state", 64)
        except KeyError:
            continue
        raise AssertionError(f"{row!r} was given a depth it has no basis for")
    for name in ALL:
        spec = TK.CANONICAL[name]
        L = spec.eval_lengths[0]
        ex = TK.generate(spec, "test", n=20, length=L)
        k, m = spec.k, spec.n_objects_active
        ns, ng = s5_bind_v3_shape(ex)
        named, q = s5_bind_v3_is_named(ex), s5_bind_v3_query_kind(ex)
        rows = tuple(s5_bind_v3_floors(ex, k, m)) + \
            s5_bind_v3_family_rows(k, m, ns, ng, named, q) + ("surface_ranker",)
        for row in rows:
            assert s5_bind_v3_row_depth(row, q, ns + ng) >= 0, (name, row)
    # and the depth ordering is the policies', not the names': a full replay chains the stream,
    # a half-window chains half of it, the one-hop read chains one, a guess chains none.
    assert s5_bind_v3_row_depth("uniform", "state", 64) == 0
    assert s5_bind_v3_row_depth("last_write_1hop", "state", 64) == 1
    assert s5_bind_v3_row_depth("window_50", "state", 64) == 32
    assert s5_bind_v3_row_depth("one_structure_P", "state", 64) == 64


def test_mutating_the_rankers_feature_tuple_moves_its_verdict_through_the_rule():
    """The ranker's DEPTH and REGISTERS are read off the features its weights load on, so the
    verdict has to move when the feature set does. Priced by the row NAME it could not: the name
    is constant while the policy is not.

    Three feature sets at one composed cell and one component cell: the whole registered set
    (six per-candidate accumulators — excluded), the landmark-and-stated subset (no accumulator
    — admitted), and the stated-only subset (which is also depth 0)."""
    feats = S5_BIND_V3_SURFACE_FEATURES

    def only(names):
        return {n: (1.0 if n in names else 0.0) for n in feats}

    landmarks = [n for n, a in zip(feats, S5_BIND_V3_SURFACE_FEATURE_ACC) if a is None]
    stated = [n for n, d in zip(feats, S5_BIND_V3_SURFACE_FEATURE_DEPTH) if d == 0]
    assert len(feats) == len(S5_BIND_V3_SURFACE_FEATURE_DEPTH) == \
        len(S5_BIND_V3_SURFACE_FEATURE_ACC)
    assert s5_bind_v3_surface_depth() == 1 and s5_bind_v3_surface_depth(only(stated)) == 0
    assert len(s5_bind_v3_surface_loaded(only(landmarks))) == len(landmarks)
    for name, want in (("s5_bind_local_v3", True), ("s5_bind_local_v3_state", True)):
        spec = TK.CANONICAL[name]
        L = spec.eval_lengths[0]
        ex = TK.generate(spec, "test", n=20, length=L)
        k, m = spec.k, spec.n_objects_active
        ns, ng = s5_bind_v3_shape(ex)
        named, q = s5_bind_v3_is_named(ex), s5_bind_v3_query_kind(ex)
        full = s5_bind_v3_admits("surface_ranker", k, m, ns, ng, named, q)
        lean = s5_bind_v3_admits("surface_ranker", k, m, ns, ng, named, q, only(landmarks))
        assert not full, f"{name}: the 25-feature ranker is admitted"
        assert lean is want, f"{name}: the landmark-only ranker verdict is {lean}"
        # the cost moves with the feature set too, not only the verdict
        w_full, s_full = s5_bind_v3_row_cost("surface_ranker", k, m, ns, ng, q, named)
        w_lean, s_lean = s5_bind_v3_row_cost("surface_ranker", k, m, ns, ng, q, named,
                                             only(landmarks))
        assert (w_full, s_full) != (w_lean, s_lean), name


def test_the_fitted_ranker_is_not_admitted_at_any_cell_and_the_price_says_why():
    """The row was priced W = 2 and one backward scan while holding six k-entry accumulators and
    argmaxing over k candidates. Neither end of its register/pass trade-off is admitted anywhere:
    one pass holds 1 + 7k registers, over the one-structure bound on a composed cell and over 2
    on a component one, and the register-lean implementation pays 2kL steps, over the composed
    task's own cost and over each component's per-item minimum."""
    for name in ALL:
        spec = TK.CANONICAL[name]
        for L in spec.eval_lengths:
            ex = TK.generate(spec, "test", n=20, length=L)
            k, m = spec.k, spec.n_objects_active
            ns, ng = s5_bind_v3_shape(ex)
            named, q = s5_bind_v3_is_named(ex), s5_bind_v3_query_kind(ex)
            price = s5_bind_v3_surface_price(k, m, ns, ng, named, q)
            impls = s5_bind_v3_surface_impls(k, m, ns, ng, q)
            assert not price["admitted"], f"{name}@{L}: {price}"
            assert price["A"] == 6, (name, L, price["A"])
            one_pass = max(impls, key=lambda d: d["W"])
            lean = min(impls, key=lambda d: d["W"])
            assert one_pass["W"] == 1 + k * 7 and one_pass["passes"] == 1
            assert lean["passes"] == k and lean["S"] == 2 * k * (ns + ng)
            assert one_pass["W"] > price["W_max"] and lean["S"] >= price["S_max"]
            # and it is out on REGISTERS/STEPS, never on depth — the surface set chains one hop
            assert price["depth"] == 1


def _protocol():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import protocol_s5bind_v3_three_cell_20260731 as P
    return P


def _driver():
    _protocol()
    import experiment_s5bind_v3_three_cell_local_20260731 as E
    return E


def test_the_positive_control_is_evaluated_on_the_grid_its_read_covers_or_raises():
    """THE DEFECT THIS CLOSES. The control was "the state component at L=16" and it was applied
    to the GUIDED read, whose grid is GUIDED_LENGTHS and does not contain 16. The guided arm
    never evaluated that cell, so the seed count was 0 and the rule reported a MISSING CELL as a
    model at floor and voided the run.

    A control is now declared as (read, cell, length) pairs, evaluated only where the read covers
    them, and it RAISES where none was measured — a raise cannot be mistaken for an abort.

    The guided pairs are each component's WORK-MATCHED partner of the composed cell's guided
    length, because that is the only length the guided read evaluates a component at."""
    P = _protocol()
    ws, wb = P.WORK_MATCHED[48]["state"], P.WORK_MATCHED[48]["bind"]
    grid = {"state": [16, ws], "bind": [16, wb], "composed": [16, 48]}
    assert P.control_grid("plain", grid) == (("state", 16), ("bind", 16))
    assert P.control_grid("guided", grid) == (("state", ws), ("bind", wb))
    floors = {"state": {16: 0.2, ws: 0.2}, "bind": {16: 0.2, wb: 0.2},
              "composed": {16: 0.5, 48: 0.23}}
    # the arm this was written against: measured at the work-matched lengths only, state at floor
    # there, retrieval at 1.000. The control is a DISJUNCTION over the components, so it is the
    # retrieval cell that licenses reading the rest — the single-cell control called the arm void.
    guided = {"state": {0: {ws: 0.21}, 1: {ws: 0.19}},
              "bind": {0: {wb: 1.0}, 1: {wb: 1.0}},
              "composed": {0: {48: 0.49}, 1: {48: 0.35}}}
    ctrl = P.evaluate_control("guided", guided, floors, grid, P.N_GUIDED)
    assert ctrl["seeds"] == 2 and ctrl["cleared_on"] == f"bind@{wb}", ctrl
    assert ctrl["per_pair"][f"state@{ws}"] == 0, ctrl
    # the same arm read against the OLD control length is not a failure, it is unevaluable
    try:
        P.evaluate_control("plain", guided, floors, grid, P.N_EVAL)
    except P.ControlNotEvaluable:
        pass
    else:
        raise AssertionError("a control at a length the arm never ran must raise")
    # and a control whose cells are not on the run's grid at all raises too
    try:
        P.evaluate_control("guided", guided, floors, {"composed": [48]}, P.N_GUIDED)
    except P.ControlNotEvaluable:
        pass
    else:
        raise AssertionError("a control with no cell on the grid must raise")


def test_the_verdict_refuses_a_bare_control_count_and_v1_needs_the_matched_control():
    """Two clauses, both of which the run showed do not say what they were meant to say.

    A bare seed count reports "at floor" and "never measured" with the same 0, so ``verdict``
    takes the evaluated control and refuses an int. And V1's whole claim is "beyond the step
    multiplier", which is exactly what the matched-cost control establishes — so an ABSENT
    matched control is not a pass, it is V1_UNCONTROLLED."""
    P = _protocol()
    ctrl = {"seeds": 2, "cleared_on": "bind@48", "per_pair": {"bind@48": 2}, "required": []}
    forms = {"state": True, "bind": True, "composed": False}
    counts = {c: {48: 2} for c in forms}
    try:
        P.verdict(2, forms, counts, {"state": None, "bind": None}, {})
    except P.ControlNotEvaluable:
        pass
    else:
        raise AssertionError("verdict() must refuse a bare seed count")
    code, _why = P.verdict(ctrl, forms, counts, {"state": None, "bind": None},
                           {"state": False, "bind": False})
    assert code == "V1_UNCONTROLLED", code
    code, _why = P.verdict(ctrl, forms, counts, {"state": True, "bind": True},
                           {"state": True, "bind": True})
    assert code == "V1_COMPOSITION_GAP", code
    code, _why = P.verdict(ctrl, forms, counts, {"state": True, "bind": False},
                           {"state": True, "bind": True})
    assert code == "V3_GAP_IS_THE_COST", code
    # the control is a DISJUNCTION over the components, and it gates everything
    code, _why = P.verdict({**ctrl, "seeds": 0}, forms, counts, {"state": True, "bind": True},
                           {"state": True, "bind": True})
    assert code == "V5_HARNESS_NULL", code


def test_the_matched_cost_control_is_declared_and_an_unreachable_one_is_absent():
    """The matched-cost control is a first-class requirement with named lengths, so "the composed
    cell is harder beyond the step multiplier" is testable. Where the sampler cannot build the
    matched length the requirement is ABSENT rather than approximated by a shorter cell."""
    P = _protocol()
    matched = {48: {"state": {"L": 80, "reachable": True}, "bind": {"L": 132,
                                                                    "reachable": True}},
               64: {"state": {"L": 108, "reachable": True}, "bind": {"L": None,
                                                                     "reachable": False}},
               96: {"state": {"L": 160, "reachable": True}, "bind": {"L": None,
                                                                     "reachable": False}}}
    req = P.matched_required(matched, lengths=(48, 64, 96))
    assert req["state"] == [80, 108, 160]
    assert req["bind"] == [132], "an unreachable matched length must not be substituted"
    # and the GUIDED read runs each component at its WORK-MATCHED partner (its registered grid)
    # plus the token-matched control, without which it could never reach V1: its own grid would
    # never cover a length longer than the composed cell's.
    ws, wb = P.WORK_MATCHED[48]["state"], P.WORK_MATCHED[48]["bind"]
    gg = P.guided_grid(matched, lengths=(48,))
    assert gg["state"] == sorted([ws, 80]) and gg["bind"] == sorted([wb, 132]), gg
    assert gg["composed"] == [48], gg
    assert max(gg["composed"]) < max(gg["state"]), "the control has to be the LONGER component"
    assert min(gg["state"]) < min(gg["composed"]), "the work-matched partner is the SHORTER one"


def test_the_two_pairings_answer_different_questions_and_both_are_registered():
    """TOKEN-matching sets the multiplier to 1.00 by construction — that is what makes it a
    control. WORK-matching leaves whatever the composed cell's extra structure costs, and that is
    the number the depth-matched comparison is against. Reading the components at the composed
    cell's own length reports a third number and is the one that hid the confound."""
    P = _protocol()
    assert P.PAIRINGS == ("work", "tokens")
    assert P.REGISTERED_PAIRING == "work" and P.MATCHED_PAIRING == "tokens"
    for L in P.LOCAL_LENGTHS:
        assert P.WORK_MATCHED[L]["state"] < L < P.TOKEN_MATCHED[L]["state"], L
    for cell in ("state", "bind"):
        assert P.registered_lengths(cell) == tuple(
            sorted({P.WORK_MATCHED[L][cell] for L in P.LOCAL_LENGTHS}))
        assert set(P.registered_lengths(cell)) <= set(
            TK.CANONICAL[P.LOCAL_CELLS[cell]].eval_lengths), cell
    assert P.registered_lengths("composed") == P.LOCAL_LENGTHS
    # the state component's grid also carries rungs ABOVE its work-matched ones — it is the leg
    # with a depth axis — and those are reported, not required: requiring a ladder built to span
    # the component's range would make V4 the standing verdict.
    for cell in ("state", "bind"):
        assert set(P.registered_lengths(cell)) <= set(P.PROFILE_LENGTHS[cell]), cell
        assert tuple(P.PROFILE_LENGTHS[cell]) == TK.CANONICAL[P.LOCAL_CELLS[cell]].eval_lengths
    assert len(P.PROFILE_LENGTHS["state"]) > len(P.registered_lengths("state"))
    # the stored record may predate the work pairing; a flat field is the TOKEN pairing and is
    # labelled as that rather than left unnamed
    flat = {"48": {"state": {"L": 80}}}
    assert P.as_pairings(flat) == {"tokens": {48: {"state": {"L": 80}}}}
    assert P.as_pairings({"work": {}, "tokens": {}}) == {"work": {}, "tokens": {}}


def test_the_retrieval_component_is_a_gate_at_every_setting_its_spec_can_take():
    """WHY IT IS REGISTERED AS A GATE AND NOT AS A DIFFICULTY AXIS. Its own algorithm chains ONE
    hop and holds ONE carrier plus a scratch register at every k, m and write density, and every
    admitted row is a guess, so the floor is informed chance on the 'chance' basis everywhere.
    What the knobs move is the answer space and the scan the sampler's window forces — recall
    difficulty — and the from-scratch arm reads the cell 1.000 at every length to L = 132."""
    seen = set()
    for k in (6, 8, 12, 16):
        for m in (2, 3, 4, 6, 8, 12, 16):
            if m > k:
                continue
            for L in (16, 48, 96, 171):
                assert s5_bind_v3_task_depth(k, m, 0, L, True, "bind") == S5_BIND_V3_MAX_DEPTH
                assert s5_bind_v3_task_cost(k, m, 0, L, True, "bind")[0] == 2
                assert s5_bind_v3_floor_basis(k, m, 0, L, True, "bind") == "chance"
                seen.add((k, m, L))
    assert len(seen) == 88, len(seen)
    # and the contrast: the STATE component's own algorithm chains the whole carrier walk, so the
    # same knobs DO move its depth — that is what makes it the leg with a range to register.
    assert s5_bind_v3_task_depth(6, 6, 17, 0, True, "state") == 6
    assert s5_bind_v3_task_depth(6, 6, 128, 0, True, "state") == 43


def test_no_retrieval_query_arm_is_registrable_under_this_samplers_window():
    """WHY THE COMPOSED QUERY IS A STATE QUERY, measured rather than argued.

    The bind query does need both maps: read the SAME composed stream with a bind query and both
    one-structure replays — P carried live against the stated holder map, and its mirror — fall to
    informed chance by the deepest registered length, as they do on the state query at every
    length (the B-only read decays with L: 0.370 / 0.297 / 0.257 / 0.203 at k=6/L=48/64/80/96 and
    0.303 / 0.170 / 0.103 at k=12/L=128/192/256). What disqualifies the arm is the TAIL. The
    sampler pins the queried object's resolving write into [0.1L, 0.75L], and that pin is what
    PROVES the retrieval component's floor; under it no event past 0.75L can move a bind answer,
    so a solver holding both maps and replaying only a prefix of the stream is exact. The state
    query's own gate puts the queried agent's last move inside the final 10%, so the same
    truncation is at chance there.

    A floor-proved retrieval component needs the resolving write far from the end; a query that
    reads the whole stream needs it near the end. One sampler cannot do both.
    """
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        chance = 1.0 / (spec.k - 1)
        L = spec.eval_lengths[-1]
        bindq = replace(spec, query_arm="bind", stream_name=f"{spec.stream_name}_bindq_probe")
        state_ex = TK.generate(spec, "test", n=150, length=L)
        bind_ex = TK.generate(bindq, "test", n=150, length=L)
        # both maps are needed on BOTH queries at the deepest registered length
        for ex, arm in ((state_ex, "state"), (bind_ex, "bind")):
            assert all(C.read(e.prompt)["query"][0] == arm for e in ex), (name, arm)
            for mode in ("P_live", "B_live"):
                hit = sum(int(C.answer_of(C.read(e.prompt),
                                          C.replay(C.read(e.prompt), mode=mode)) == e.answer)
                          for e in ex)
                assert hit / len(ex) <= 1.6 * chance, f"{name}/{arm}: {mode} {hit / len(ex):.3f}"

        def prefix(ex, f):
            hit = 0
            for e in ex:
                rec = C.read(e.prompt)
                n = len(rec["events"])
                hit += int(C.answer_of(rec, C.replay(rec, drop=(int(round(f * n)), n)))
                           == e.answer)
            return hit / len(ex)

        # ... and only the STATE query needs the tail
        assert prefix(bind_ex, 0.90) >= 0.99, (name, prefix(bind_ex, 0.90))
        assert prefix(state_ex, 0.90) <= 2.0 * chance, (name, prefix(state_ex, 0.90))
    # so no such arm is registered: every composed cell in the registry carries a state query
    for reg in (TK.CANONICAL, TK.RETIRED):
        for nm, sp in reg.items():
            if sp.source_ablation and sp.event_kinds == "both" and not sp.named_operands:
                assert sp.query_arm == "state", f"{nm}: a bind-query composed arm is registered"


def test_reference_rows_are_dropped_on_a_component_cell():
    for name in COMPONENTS:
        spec = TK.CANONICAL[name]
        fl = s5_bind_v3_floors(TK.generate(spec, "test", n=40, length=spec.eval_lengths[0]),
                               spec.k, spec.n_objects_active)
        for row in S5_BIND_V3_REFERENCE_ROWS:
            assert row not in fl, f"{name}: {row} reproduces the oracle and is not a floor"


def test_floors_are_recomputed_from_the_exact_items():
    """A floor row is a property of the cell, not a constant: the same policy on a different
    length gives a different number, and the number depends only on the items."""
    spec = TK.CANONICAL["s5_bind_local_v3"]
    a = s5_bind_v3_floors(TK.generate(spec, "test", n=200, length=48), 6, 6)
    b = s5_bind_v3_floors(TK.generate(spec, "test", n=200, length=48), 6, 6)
    assert a == b
    assert a["initial_only"] == 0.0                       # the gate forbids the stated answer


def test_the_operative_floor_is_near_informed_chance_on_every_scored_cell():
    for name in ALL:
        spec = TK.CANONICAL[name]
        for L in spec.eval_lengths:
            ex = TK.generate(spec, "test", n=N_FLOOR, length=L)
            ns, ng = s5_bind_v3_shape(ex)
            named = s5_bind_v3_is_named(ex)
            query = s5_bind_v3_query_kind(ex)
            fl = s5_bind_v3_floors(ex, spec.k, spec.n_objects_active)
            op = s5_bind_v3_operative_floor(fl, spec.k, spec.n_objects_active, ns, ng, named,
                                            query)
            chance = 1.0 / (spec.k - 1)
            assert op is not None and op / chance <= FLOOR_RATIO_MAX, \
                f"{name}@{L}: operative floor {op:.4f} = {op / chance:.2f}x chance"


def test_the_surface_read_is_dead_where_the_gate_is_set_and_priced_where_it_is_not():
    """q_no_surface empties ``last_swap_ref`` and registers the closed-form exclusion that
    striking it hands a guesser. Where the gate is off the row is measured instead, and the
    exclusion is not registered — a row that is merely low on this sample is not a rejection."""
    for name in COMPOSED:
        spec = TK.CANONICAL[name]
        fl = s5_bind_v3_floors(TK.generate(spec, "test", n=N_FLOOR, length=spec.eval_lengths[-1]),
                               spec.k, spec.n_objects_active)
        if spec.q_no_surface:
            assert fl["last_swap_ref"] == 0.0, f"{name}: the gate must empty the surface read"
            assert fl["uniform_anti_surface"] > 1.0 / (spec.k - 1) - 1e-9, \
                f"{name}: striking an emptied answer must be priced"
        else:
            assert fl["last_swap_ref"] > 0.0, f"{name}: an ungated cell measures the row"
            assert "uniform_anti_surface" not in fl


def test_the_surface_bound_is_fitted_on_one_sample_and_scored_on_a_disjoint_one():
    """The one-at-a-time sweep reported a best of 1.08x chance and did not contain the 1.41x
    rule; a ranker over the whole feature set, scored out of sample, is the honest bound. It is
    deterministic (no draw), and it is scored on items the fit never saw."""
    spec = TK.CANONICAL["s5_bind_local_v3"]
    both = TK.generate(spec, "test", n=600, length=64)
    fit, ho = both[:300], both[300:]
    assert not ({e.prompt for e in fit} & {e.prompt for e in ho})
    a = s5_bind_v3_surface_bound(fit, spec.k, held_out=ho)
    b = s5_bind_v3_surface_bound(fit, spec.k, held_out=ho)
    assert a["held_out"] == b["held_out"], "the bound must not depend on a seed"
    assert a["n_fit"] == 300 and a["n_held_out"] == 300
    assert a["in_sample"] >= a["held_out"] - 0.05, (a["in_sample"], a["held_out"])
    assert set(a["weights"]) == set(S5_BIND_V3_SURFACE_FEATURES)
    # THE FIT BUDGET SHIPS WITH THE NUMBER. 300 items is inside the range where the held-out
    # curve is still climbing, so the estimate says so, and a blocked fit reports the spread its
    # budget leaves rather than one figure that hides it.
    assert a["fit_at_least_min"] is False and a["n_fit"] < 2000
    c = s5_bind_v3_surface_bound(fit, spec.k, held_out=ho, blocks=2)
    assert len(c["blocks"]) == 2 and c["n_per_block"] == 150
    assert c["block_spread"] == max(c["blocks"]) - min(c["blocks"])


def test_the_profile_is_a_curve_over_slots_with_the_bound_drawn_across_it():
    """A cell's floor is reported as a profile, not a number: the partial-carry family fills
    every W from k+1 to k+m+1 and the bound sits at max(k,m)+1, so a continuum is visible."""
    spec = TK.CANONICAL["s5_bind_local_v3"]
    k, m = spec.k, spec.n_objects_active
    ex = TK.generate(spec, "test", n=200, length=64)
    prof = s5_bind_v3_slot_profile(ex, k, m, named=False, query="state")
    assert all(r["axis"] == "W" for r in prof), "a composed cell separates on live slots"
    by_w = {r["W"]: r for r in prof}
    assert set(range(k + 1, k + m + 2)) <= set(by_w), sorted(by_w)
    assert by_w[k + m + 1]["acc"] == 1.0, "the top of the profile is the task itself"
    admitted = [w for w, r in by_w.items() if r["admitted"]]
    assert max(admitted) == one_structure_bound(k, m) == max(k, m) + 1
    assert all(w > max(admitted) for w, r in by_w.items() if not r["admitted"] and w > 2)


def test_what_the_excluded_block_drop_family_actually_reads():
    """The exclusion is a cost argument, so what the continuum READS is measured, not assumed.

    It is a smooth function of the width, exactly as "do (1 - w) of the task" should be: a
    one-event drop reads ~4x chance because it is 98% of the task, and the family decays to
    chance as the width grows. No sampler gate is doing this work and none is needed — the
    class rule takes the whole continuum at once, at every width, because every member carries
    both maps.
    """
    spec = TK.CANONICAL["s5_bind_local_v3"]
    ex = TK.generate(spec, "test", n=200, length=64)
    chance = 1.0 / (spec.k - 1)
    prof = {w: max(s5_bind_v3_block_drop(ex, w, p) for p in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95))
            for w in (0.02, 0.05, 0.10, 0.15, 0.25, 0.50)}
    widths = sorted(prof)
    for a, b in zip(widths, widths[1:]):
        assert prof[b] <= prof[a] + 0.02, f"non-monotone in width: {prof}"
    assert prof[0.02] > 2.5 * chance, "a one-event drop should read far above chance"
    assert prof[0.25] < 1.35 * chance and prof[0.50] < 1.35 * chance, prof


# --- the cost convention ------------------------------------------------------------------

def test_the_convention_charges_a_backward_walk_for_what_it_scans():
    """The rule the number used to depend on, made testable: the state leg's walk pays for every
    event it passes, gives included, so its cost tracks L and not the chain length."""
    assert C.CHARGE_SCANNED_EVENTS
    spec = TK.CANONICAL["s5_bind_local_v3"]
    for L in (48, 96):
        ex = TK.generate(spec, "test", n=30, length=L)
        r = C.cost_report(ex, spec.k, spec.n_objects_active)
        # 2 steps per event scanned, plus the hits, the header read and the emit
        assert 2 * L <= r["state_S"] <= 2 * L + L / 2 + 4, (L, r["state_S"])
        assert r["state_W"] == 2 and r["composed_W"] == spec.k + spec.n_objects_active


def test_the_composed_pass_costs_what_the_rule_says():
    spec = TK.CANONICAL["s5_bind_v3"]
    ex = TK.generate(spec, "test", n=20, length=128)
    k, m = spec.k, spec.n_objects_active
    for e in ex:
        rec = C.read(e.prompt)
        ns = sum(1 for x in rec["events"] if x[0] == C.SWAP)
        ng = len(rec["events"]) - ns
        s, w = C.cost_composed(rec, k, m)
        assert w == k + m
        assert s == (k + m) + 6 * ns + 3 * ng + 1


def test_the_free_retrieval_convention_gives_a_different_number():
    """Which is the point of stating the rule: the multiplier is a function of the convention,
    so the convention is named and the alternative's number is reported next to it."""
    spec = TK.CANONICAL["s5_bind_local_v3"]
    r = C.cost_report(TK.generate(spec, "test", n=30, length=64), 6, 6)
    charged = r["composed_S"] / r["state_S"]
    free = r["composed_S_free"] / r["state_S_free"]
    assert 1.5 < charged < 3.0 and free > 3 * charged


# --- the statistic ------------------------------------------------------------------------

def test_the_statistic_is_registered_in_the_package():
    """It reports alongside match, from factworld and not from a script."""
    spec = TK.CANONICAL["s5_bind_local_v3"]
    ex = TK.generate(spec, "test", n=40, length=48)
    out = C.contrast(ex, [1] * 30 + [0] * 10, draws=1, seed=3)
    for key in ("contrast", "reject", "acc", "slice_cross", "slice_same",
                "class_balance_cross", "mean_write_count_cross", "mean_write_count_same"):
        assert key in out, key
    assert 0.45 <= out["class_balance_cross"] <= 0.55
    wx, wz = out["mean_write_count_cross"], out["mean_write_count_same"]
    assert abs(wx - wz) / max(wx, wz) < 0.10


def test_the_diagnostic_is_stratified_and_its_mass_columns_are_raw():
    """Within an event kind the class IS the reference clause — CROSS is "belongs to" on a swap
    and "points to" on a give — so a solver that is simply worse at one clause slips on
    swap-CROSS and give-SAME, which is the ANTI-symmetric combination of the two kinds' class
    differences. The design answers that with two properties, both pinned here.

    THE MASS COLUMNS ARE RAW, so any hazard that is a function of the stratum is exactly in the
    model's span with a zero contrast. Reweighting the class columns instead only works where the
    within-kind class masses are equal, and they are not: a CROSS give's object is pinned and
    cannot be referenced again until the pin dies, so its slice mass is about half a SAME give's.

    THE CONTRAST IS PRECISION-WEIGHTED, which makes it exactly uncorrelated with every
    anti-symmetric combination of the per-stratum differences."""
    assert C.DIAGNOSTIC_STAT == "T_kind" and C.STATS["T_kind"] == ((), 1)
    assert not hasattr(C, "PRIMARY_STAT"), "the contrast is a diagnostic, not a primary"
    spec = TK.CANONICAL["s5_bind_local_v3"]
    ex = TK.generate(spec, "test", n=120, length=48)
    ops = [C.op_slice(C.read(e.prompt), draws=1) for e in ex]
    of, ns = C.strata(ops, 1)
    assert ns == 2 and {of(op) for item in ops for op in item if op["cls"] != "write"} == {0, 1}
    T, dif = C._stratum_columns(ops, 1)
    # the mass column of a stratum is that stratum's whole slice mass, both classes, unweighted
    for i, item in enumerate(ops):
        for s in range(2):
            want = sum(op["sens"] for op in item if op["cls"] != "write" and of(op) == s)
            assert abs(T[i][s] - want) < 1e-9
    # the within-kind class masses are NOT equal, which is what forbids a reweighting
    mass = {}
    for item in ops:
        for op in item:
            if op["cls"] != "write":
                mass[(op["kind"], op["cls"])] = mass.get((op["kind"], op["cls"]), 0.0) + op["sens"]
    assert mass[("give", "cross")] < 0.7 * mass[("give", "same")]
    # and the contrast column is uncorrelated with the anti-symmetric direction
    Dm = [[0.0, 0.0] for _ in ops]
    for i, item in enumerate(ops):
        for op in item:
            if op["cls"] != "write":
                Dm[i][of(op)] += op["sens"] if op["cls"] == "cross" else -op["sens"]
    anti = [row[0] - row[1] for row in Dm]
    n = len(ops)
    md, ma = sum(dif) / n, sum(anti) / n
    cov = sum((dif[i] - md) * (anti[i] - ma) for i in range(n)) / (n - 1)
    sd = (C._var(dif) * C._var(anti)) ** 0.5
    assert abs(cov) < 0.02 * sd, f"contrast correlates with the clause direction: {cov / sd:.3f}"


def test_the_nuisance_columns_run_over_both_classes():
    """A read-history repair is a property of the cell READ, not of the class reading it, so the
    load it puts on an item is the sum over all of that item's resolutions. Summed over the CROSS
    reads alone it absorbs the cross half of the effect and leaves the same half loading on the
    SAME column — a repair that moves the contrast toward the defect it claims to remove."""
    spec = TK.CANONICAL["s5_bind_local_v3"]
    for e in TK.generate(spec, "test", n=12, length=48):
        ops = C.op_slice(C.read(e.prompt), draws=1)
        cov = C._cov_from_ops(ops)
        reads = [op for op in ops if op["cls"] != "write"]
        assert abs(cov["W"] - sum(op["sens"] * op["w"] for op in reads)) < 1e-9
        assert abs(cov["D"] - sum(op["sens"] * op["d"] for op in reads)) < 1e-9
        assert cov["W"] > sum(op["sens"] * op["w"] for op in reads if op["cls"] == "cross")


def test_a_cell_reports_the_structure_switch_diagnostic_alongside_match():
    """Registered in factworld, not in a script: evaluating a source-structure cell returns
    theta_cross - theta_same next to the canonical match, with the class balance and the
    write-count matching that make it valid. It is reported under ``structure_switch`` and
    NOT under ``composition``: within a kind the class label is the printed clause, so the
    key a caller reads has to say what a rejection licenses."""
    from factworld.runner import evaluate_task

    spec = TK.CANONICAL["s5_bind_local_v3"]
    ex = TK.generate(spec, "test", n=40, length=48)
    gold = {e.prompt: e.answer for e in ex}

    class _Backend:
        name = "oracle-with-errors"

        def generate(self, prompts, max_new_tokens=8, stop_at=None):
            return [gold[p] if j % 4 else "g0." for j, p in enumerate(prompts)]

    out = evaluate_task(_Backend(), spec, n=40, length=48, composition_draws=1)
    assert "composition" not in out, "the key must not read as a composition measure"
    assert "structure_switch" in out and out["structure_switch"]["n"] == 40
    assert out["structure_switch"]["stat"] == C.DIAGNOSTIC_STAT
    assert out["structure_switch"]["identifies"] == "structure_switch"
    assert out["structure_switch"]["acc"] == out["overall"]
    assert isinstance(out["structure_switch"]["reject"], bool)
    # a component arm has no cross class, so it reports match alone
    comp = TK.CANONICAL["s5_bind_local_v3_state"]
    ex2 = TK.generate(comp, "test", n=20, length=48)
    gold = {e.prompt: e.answer for e in ex2}
    out2 = evaluate_task(_Backend(), comp, n=20, length=48, composition_draws=1)
    assert out2["structure_switch"]["slice_cross"] == 0.0


def test_the_slice_carries_both_classes_on_every_item():
    spec = TK.CANONICAL["s5_bind_local_v3"]
    for e in TK.generate(spec, "test", n=20, length=64):
        cov = C.item_covariates(C.read(e.prompt), draws=2)
        assert cov["nx"] > 0 and cov["nz"] > 0 and cov["nw"] > 0


def test_a_uniform_slip_does_not_move_the_contrast():
    """The null the design exists to make true: with one slip rate per op and nothing else, the
    two classes are the same operation at the same cost and the fitted contrast sits at zero."""
    import random

    spec = TK.CANONICAL["s5_bind_local_v3"]
    ex = TK.generate(spec, "test", n=250, length=48)
    rng = random.Random(11)
    cov = [C.item_covariates(C.read(e.prompt), draws=2, rng=rng) for e in ex]
    y = []
    for row in cov:
        q = 2.718281828 ** (-0.02 * (row["nw"] + row["nz"] + row["nx"]))
        y.append(float(rng.random() < q))
    names = C.TWO_CLASS_STATS["T_cross"]
    c, _rej = C.lrt([[r[n] for r in cov] for n in names], y, len(names) - 1, len(names) - 2)
    assert abs(c) < 0.05, f"uniform slip moved the contrast to {c:.4f}"


def test_the_statistic_has_zero_power_against_a_single_structure_carrier():
    """THE STATISTIC IS A STRUCTURE-SWITCH DIAGNOSTIC AND NOT A COMPOSITION MEASURE, pinned so the
    composition claim cannot be quietly re-made.

    A solver that carries P alone, or B alone, is exactly what a composition measure has to
    detect. It is not detected here at any n, because within a kind the class label IS the
    printed clause: a one-structure solver fails on {swap CROSS, give SAME}, sign-flipped across
    the kinds, which is the anti-symmetric direction the kind-balancing annihilates. At the
    registered setting — k=6/L=64, n=800 — neither carrier rejects, and the contrast points the
    wrong way. ``contrast`` carries ``identifies`` so a caller cannot read it as composition.
    """
    spec = TK.CANONICAL["s5_bind_local_v3"]
    ex = TK.generate(spec, "test", n=800, length=64)
    recs = [C.read(e.prompt) for e in ex]
    for mode in ("P_live", "B_live"):
        correct = [int(C.answer_of(r, C.replay(r, mode)) == e.answer)
                   for r, e in zip(recs, ex)]
        assert 0.05 < sum(correct) / len(correct) < 0.5, mode      # a real, costly deficit
        for seed in (0, 1):
            out = C.contrast(ex, correct, seed=seed)
            assert out["identifies"] == "structure_switch"
            assert not out["reject"], \
                f"{mode}: the statistic rejected on a single-structure carrier ({out})"
            assert out["contrast"] < 0.0, f"{mode}: contrast {out['contrast']:.4f}"


# --- the trace read -------------------------------------------------------------------------

TRACE_CELLS = (("s5_bind_local_v3_state", 17), ("s5_bind_local_v3_state", 80),
               ("s5_bind_local_v3_bind", 31), ("s5_bind_local_v3_bind", 132),
               ("s5_bind_local_v3", 48))


def _agents_objs(spec):
    w, _r = TK.build_world(spec)
    return list(w.agents[:spec.k]), list(w.objects[:spec.n_objects_active])


def test_the_trace_read_scores_the_same_gold_as_the_answer_read():
    """T1, and everything else about the trace read rests on it: the FINAL checkpoint's value for
    the queried slot IS the gold answer. If it were not, the two reads would score different
    quantities and no floor row's number would transfer between them."""
    for name, L in TRACE_CELLS:
        spec = TK.CANONICAL[name]
        agents, objs = _agents_objs(spec)
        ex = TK.generate(spec, "test", n=200, length=L)
        agree, n = s5_bind_v3_trace_is_answer(ex, spec.k, spec.n_objects_active, agents, objs)
        assert n == len(ex) and agree == n, f"{name}@{L}: {agree}/{n}"
        v = s5_bind_v3_trace_slot(ex[0], spec.k, spec.n_objects_active, agents, objs)
        assert f"{v}." == ex[0].answer


def test_the_copier_is_zero_on_the_trace_read_and_the_gate_is_why():
    """THE OBVIOUS CHEAP POLICY ON A CHECKPOINT READ IS TO COPY THE PREVIOUS CHECKPOINT, and a
    trace that never moves ends at the STATED value. The query gate requires the queried slot to
    move at least twice and to end different from what the prompt states, so that policy is wrong
    on every item — 0.000 and not "near chance" — at every registered cell.

    The move distribution is measured beside it rather than left at "at least two"."""
    for name, L in TRACE_CELLS:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        agents, objs = _agents_objs(spec)
        ex = TK.generate(spec, "test", n=200, length=L)
        ck = s5_bind_v3_ckpt_floors(ex)
        assert ck["ckpt_copy_prev"] == 0.0, f"{name}@{L}: {ck['ckpt_copy_prev']}"
        mv = s5_bind_v3_slot_moves(ex, k, m, agents, objs)
        assert mv["n"] == len(ex) and mv["min"] >= 2, f"{name}@{L}: {mv['min']}"
        assert mv["mean"] >= 2.5, f"{name}@{L}: {mv['mean']}"
        # and the one-hop reads of the LAST event, which are admitted, stay at or under chance
        for row in ("ckpt_last_event_operand", "ckpt_last_event_target"):
            if row in ck:
                assert ck[row] <= 1.35 / (k - 1), f"{name}@{L}/{row}: {ck[row]}"


def test_every_checkpoint_row_is_priced_and_admitted_so_it_could_move_a_floor():
    """The cheap checkpoint rows are not excluded by construction: each has a registered depth and
    cost and each is ADMITTED at every cell, so if one ever read above chance the floor would
    move. An unpriced row would raise; a silently excluded one could not raise the floor."""
    for name, L in TRACE_CELLS:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        ex = TK.generate(spec, "test", n=40, length=L)
        ns, ng = s5_bind_v3_shape(ex)
        named, query = s5_bind_v3_is_named(ex), s5_bind_v3_query_kind(ex)
        for row in S5_BIND_V3_CKPT_ROWS:
            assert s5_bind_v3_row_depth(row, query, L) <= S5_BIND_V3_MAX_DEPTH
            assert s5_bind_v3_admits(row, k, m, ns, ng, named, query), f"{name}@{L}/{row}"
            assert s5_bind_v3_guided_admits(row, k, m, ns, ng, named, query), f"{name}@{L}/{row}"
        preds = s5_bind_v3_ckpt_preds(ex[0].prompt)
        assert set(preds) == set(S5_BIND_V3_CKPT_ROWS)


def test_the_guided_class_is_the_plain_class_on_a_component_cell():
    """T2 removes the LIVE-SLOT axis, because the guided protocol requires the whole of P and B to
    be written out at every event and so hands those slots to every policy. On a COMPONENT cell
    that removes nothing: its W bound is 2 and every depth <= 1 row already costs 2, so the guided
    class and the plain class admit exactly the same rows and the floors are the same number."""
    for name, L in TRACE_CELLS:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        ex = TK.generate(spec, "test", n=N_FLOOR, length=L)
        ns, ng = s5_bind_v3_shape(ex)
        named, query = s5_bind_v3_is_named(ex), s5_bind_v3_query_kind(ex)
        if not named:
            continue
        rows = tuple(S5_BIND_V3_ROWS) + tuple(S5_BIND_V3_CKPT_ROWS) + ("surface_ranker",) + \
            s5_bind_v3_family_rows(k, m, ns, ng, named, query)
        for row in rows:
            a = s5_bind_v3_admits(row, k, m, ns, ng, named, query)
            t = s5_bind_v3_guided_admits(row, k, m, ns, ng, named, query)
            assert a == t, f"{name}@{L}/{row}: plain {a} guided {t}"
        fl = dict(s5_bind_v3_floors(ex, k, m))
        fl.update(s5_bind_v3_family_floors(ex, k, m, named, query))
        assert (s5_bind_v3_trace_operative_floor({**fl, **s5_bind_v3_ckpt_floors(ex)},
                                                 k, m, ns, ng, named, query)
                >= s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, query))
        assert s5_bind_v3_trace_floor_basis(k, m, ns, ng, named, query) in ("measured", "chance")


def test_the_composed_cell_is_unfloorable_on_BOTH_reads_under_a_guided_protocol():
    """THE COMPOSED CELL'S FLOOR ARGUMENT IS ENTIRELY A LIVE-SLOT ARGUMENT — W <= max(k, m) + 1
    against the task's k + m + 1, with a step bound the task itself satisfies — and a scratchpad
    protocol hands out exactly those slots. What is left admits the task, so the operative floor
    is None rather than a max over a class that contains the answer.

    IT IS THE PROTOCOL AND NOT THE READ THAT DECIDES THIS, so the guided ANSWER channel gets the
    same treatment as the trace channel. Pinning both here is the point: the earlier revision
    voided the floor on the trace only, and the guided answer score was then read as clearing
    0.234 at composed@48 — a number that does not hold under the format that produced the score.
    ``s5_bind_v3_operative_floor`` returns the SAME thing as its trace wrapper, at every composed
    cell and for every floor dict.

    The price is measured, not asserted: the best both-maps policy strictly cheaper than the task
    (drop one block of events, replay the rest) reads far above the plain protocol's floor."""
    for name in ("s5_bind_local_v3", "s5_bind_v3"):
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        L = spec.eval_lengths[0]
        ex = TK.generate(spec, "test", n=200, length=L)
        ns, ng = s5_bind_v3_shape(ex)
        named, query = s5_bind_v3_is_named(ex), s5_bind_v3_query_kind(ex)
        assert not named
        fl = dict(s5_bind_v3_floors(ex, k, m))
        both = {**fl, **s5_bind_v3_ckpt_floors(ex)}
        plain = s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, query)
        assert plain is not None, f"{name}@{L}: the PLAIN protocol's floor must survive"
        # both channels, one rule: the answer read under guided=True and the trace wrapper
        for floors in (fl, both):
            assert s5_bind_v3_operative_floor(floors, k, m, ns, ng, named, query,
                                              guided=True) is None, f"{name}@{L}: guided ANSWER"
            assert s5_bind_v3_trace_operative_floor(floors, k, m, ns, ng, named,
                                                    query) is None, f"{name}@{L}: guided TRACE"
        assert s5_bind_v3_floor_basis(k, m, ns, ng, named, query, guided=True) == "unfloorable"
        assert s5_bind_v3_trace_floor_basis(k, m, ns, ng, named, query) == "unfloorable"
        assert s5_bind_v3_floor_basis(k, m, ns, ng, named, query) != "unfloorable"
        pad = s5_bind_v3_pad_reach(ex)
        assert pad > plain + 0.3, f"{name}@{L}: pad {pad:.3f} against plain floor {plain:.3f}"
        assert pad > 2.5 * plain, f"{name}@{L}: pad {pad:.3f} is {pad / plain:.2f}x plain floor"


def test_the_component_cells_are_unaffected_by_the_guided_protocol():
    """THE RETRACTION IS THE COMPOSED CELL'S ONLY. A component's class rule is depth <= 1 AND cost
    strictly under that cell's own algorithm's minimum, and a pad substitutes for REGISTERS, not
    for CHAINING — so nothing the guided format hands out is on either axis. Every component cell
    keeps a floor on both channels, its basis stays 'measured' or 'chance', and the guided number
    is the plain number (or above it, where the checkpoint-shaped rows the protocol makes
    available are merged in — never below).

    A pad reach is not defined there and must not be reported: the block-drop family carries both
    maps, and a component has one."""
    for name, L in TRACE_CELLS:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        ex = TK.generate(spec, "test", n=N_FLOOR, length=L)
        ns, ng = s5_bind_v3_shape(ex)
        named, query = s5_bind_v3_is_named(ex), s5_bind_v3_query_kind(ex)
        if not named:
            continue
        fl = dict(s5_bind_v3_floors(ex, k, m))
        fl.update(s5_bind_v3_family_floors(ex, k, m, named, query))
        plain = s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, query)
        guided = s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, query, guided=True)
        assert guided is not None, f"{name}@{L}: a component must keep its floor"
        assert abs(guided - plain) < 1e-12, f"{name}@{L}: {guided:.4f} against {plain:.4f}"
        with_ckpt = s5_bind_v3_operative_floor({**fl, **s5_bind_v3_ckpt_floors(ex)},
                                               k, m, ns, ng, named, query, guided=True)
        assert with_ckpt >= plain - 1e-12, f"{name}@{L}: {with_ckpt:.4f} below {plain:.4f}"
        for g in (False, True):
            assert s5_bind_v3_floor_basis(k, m, ns, ng, named, query,
                                          guided=g) in ("measured", "chance"), f"{name}@{L}"


def test_the_protocol_and_the_driver_agree_that_a_guided_composed_cell_has_no_floor():
    """THE REPO CONTRADICTED ITSELF ON EXACTLY THIS and the contradiction is what the test pins.
    ``cell_floor(guided=True)`` must return no floor on a composed cell and a pad reach instead,
    must return the component cells unchanged, and ``verdict`` must refuse to read a composition
    result off it: with the composed cell unfloorable the verdict is V0_COMPOSED_UNFLOORABLE and
    never V2_NO_GAP_HERE (the retracted one) and never V1, which would report a gap measured
    against a floor that does not exist."""
    P = _protocol()
    fl = P.cell_floor(TK.CANONICAL["s5_bind_local_v3"], 48, n_eval=64, n_fit=200, n_score=400,
                      guided=True)
    assert fl["floor"] is None and fl["basis"] == "unfloorable" and fl["protocol"] == "guided"
    assert fl["pad_reach"] is not None and fl["pad_reach"] > fl["floor_plain"] + 0.3
    fs = P.cell_floor(TK.CANONICAL["s5_bind_local_v3_state"], 17, n_eval=64, n_fit=200,
                      n_score=400, guided=True)
    assert fs["floor"] is not None and fs["pad_reach"] is None and fs["basis"] == "measured"
    # FORMS counts a length with no floor as None, not as 0: an unfloorable cell and a floored
    # one must not report the same number.
    ok, counts = P.forms({0: {48: 0.9}, 1: {48: 0.9}}, {48: None}, (48,), n=128)
    assert ok is False and counts == {48: None}
    ctrl = {"seeds": 2, "cleared_on": "state@17", "per_pair": {"state@17": 2}, "required": []}
    forms_ok = {"state": True, "bind": True, "composed": False}
    counts = {c: {48: 2} for c in forms_ok}
    code, why = P.verdict(ctrl, forms_ok, counts, {"state": True, "bind": True},
                          {"state": True, "bind": True}, composed_floored=False,
                          pad_reach=fl["pad_reach"])
    assert code == "V0_COMPOSED_UNFLOORABLE", code
    assert f"{fl['pad_reach']:.3f}" in why
    # and the same inputs WITH a floor still reach the registered table
    code, _why = P.verdict(ctrl, {**forms_ok, "composed": True}, counts,
                           {"state": True, "bind": True}, {"state": True, "bind": True})
    assert code == "V2_NO_GAP_HERE", code


def test_a_record_with_no_guided_floors_does_not_fall_back_to_the_plain_one():
    """THE RETRACTED FLOOR HAD A SECOND WAY BACK IN. A results file written before the guided
    floors were measured separately carries none, and the fallback was the PLAIN floors — so the
    guided composed score was read against 0.204 and the rule returned V2_NO_GAP_HERE through
    ``protocol.read_results`` after the runner's own path had stopped doing so.

    The fallback keeps the COMPONENT records, whose class is the same under either protocol, and
    returns the composed cell UNFLOORABLE. There is no input to the guided read that produces a
    composed floor."""
    E = _driver()
    plain = {"state@17": {"floor": 0.20}, "bind@31": {"floor": 0.20},
             "composed@48": {"floor": 0.204}}
    for guided in (None, {}):
        got = E._guided_records(guided, plain)
        assert got["composed@48"]["floor"] is None, got["composed@48"]
        assert got["composed@48"]["floor_plain"] == 0.204
        assert got["state@17"]["floor"] == 0.20 and got["bind@31"]["floor"] == 0.20
    # a record measured under the guided protocol is used as it stands
    measured = {"composed@48": {"floor": None, "pad_reach": 0.72, "protocol": "guided"},
                "state@17": {"floor": 0.219, "protocol": "guided"}}
    assert E._guided_records(measured, plain) is measured


def test_the_lag_family_is_the_copier_with_the_true_trace_and_is_never_admitted():
    """"Copy the previous checkpoint" scores above zero only if the copier is handed the TRUE
    trace, and then it is the cell's own algorithm stopped j events early. On the RETRIEVAL cell
    the sampler pins the resolving write at or below 0.75L, so every small j reads 1.000 — the
    slot has been constant for the last quarter of the stream. On the STATE and COMPOSED cells
    the gate pulls the last move into the final tenth and j = 1 already loses items.

    Neither is admitted: on a state or composed cell the row is depth L - j."""
    spec = TK.CANONICAL["s5_bind_local_v3_bind"]
    ex = TK.generate(spec, "test", n=200, length=62)
    assert s5_bind_v3_ckpt_lag(ex, 1) == 1.0 and s5_bind_v3_ckpt_lag(ex, 9) == 1.0
    assert s5_bind_v3_ckpt_lag(ex, 33) < 0.5
    for name, L, hi in (("s5_bind_local_v3_state", 17, 0.55), ("s5_bind_local_v3", 48, 0.90)):
        sp = TK.CANONICAL[name]
        e2 = TK.generate(sp, "test", n=200, length=L)
        lag1 = s5_bind_v3_ckpt_lag(e2, 1)
        assert 0.2 < lag1 < hi, f"{name}@{L}: lag1 {lag1:.3f}"
        assert s5_bind_v3_row_depth(f"trunc_walk_drop{1}", "state", L) > S5_BIND_V3_MAX_DEPTH


def test_the_per_slot_checkpoint_diagnostic_is_read_against_a_copier_not_against_chance():
    """A swap moves 2 of the k + m slots and a give moves 1, so a model that re-emits its previous
    checkpoint scores 1 - (2 n_swap + n_give) / ((k + m) L) per slot — 0.80-0.92 here, against
    1/k = 0.167. Reading the per-slot number against 1/k, or against the "frozen half" 0.583,
    calls a score BELOW a policy that never updates anything a partial trace."""
    for name, L in TRACE_CELLS:
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        agents, objs = _agents_objs(spec)
        ex = TK.generate(spec, "test", n=128, length=L)
        ns, ng = s5_bind_v3_shape(ex)
        got = s5_bind_v3_ckpt_copy_per_slot(ex, k, m, agents, objs)
        closed = 1.0 - (2 * ns + ng) / ((k + m) * float(ns + ng))
        assert abs(got - closed) < 0.05, f"{name}@{L}: {got:.4f} against {closed:.4f}"
        assert got > 0.75 and got > 4 * (1.0 / k), f"{name}@{L}: {got:.4f}"


def test_the_trace_read_is_a_from_scratch_arm_instrument_and_the_protocol_enforces_it():
    """A frontier model has no checkpoint stream this harness generates — its visible trace is
    prose under its own budget — so a slot read out of it is a different quantity per model. The
    restriction is code and not prose: ``assert_trace_read`` RAISES rather than falling back to
    the answer read, because a silent fallback would put two quantities in one column."""
    P = _protocol()
    assert P.TRACE_READ_ARMS == ("local",) and P.FRONTIER_READS == ("answer",)
    assert P.TRACE_READ_REQUIRES == "guided"
    assert P.assert_trace_read("local") is True
    for arm in ("frontier", "openrouter", None):
        try:
            P.assert_trace_read(arm)
        except P.TraceReadNotAvailable:
            pass
        else:
            raise AssertionError(f"the trace read was allowed on arm={arm!r}")
    try:
        P.assert_trace_read("local", read="plain")
    except P.TraceReadNotAvailable:
        pass
    else:
        raise AssertionError("the trace read was allowed off the guided protocol")


def test_a_floor_is_recomputed_on_the_items_the_read_actually_scores():
    """THE GUIDED READ SCORES 128 ITEMS AND THE PLAIN READ 1000, and the operative floor is a max
    over rows on the exact scored set, so the two are different numbers on the same rows: 0.250
    at state@80 and 0.234 at composed@48 against 0.207 and 0.201. A guided score read against the
    plain read's floor is read against a different item set."""
    P = _protocol()
    for name, L, want in (("s5_bind_local_v3_state", 80, 32 / 128),
                          ("s5_bind_local_v3", 48, 30 / 128)):
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        big = TK.generate(spec, "test", n=P.N_GUIDED + 1000, length=L)
        scored = big[:P.N_GUIDED]
        ns, ng = s5_bind_v3_shape(scored)
        named, query = s5_bind_v3_is_named(scored), s5_bind_v3_query_kind(scored)
        fl = dict(s5_bind_v3_floors(scored, k, m))
        fl.update(s5_bind_v3_family_floors(scored, k, m, named, query))
        got = s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, query)
        assert abs(got - want) < 1e-6, f"{name}@{L}: {got:.4f} against {want:.4f}"
        fb = dict(s5_bind_v3_floors(big[P.N_GUIDED:], k, m))
        assert got > s5_bind_v3_operative_floor(fb, k, m, ns, ng, named, query), (
            f"{name}@{L}: the small-n max is not above the large-n one; the selection bias this "
            "pins has gone")


def test_a_pad_of_zero_is_the_plain_rule_on_every_row_at_every_cell():
    """The bounded-pad rule is a GENERALISATION and not a second rule.

    If ``pad=0`` disagreed with ``s5_bind_v3_admits`` anywhere, the bounded protocol's floors and
    the plain protocol's would be two different objects and the comparison between them would be
    an artefact of which function was called.
    """
    for name, L in (("s5_bind_local_v3", 48), ("s5_bind_local_v3", 96),
                    ("s5_bind_local_v3_state", 17), ("s5_bind_local_v3_state", 80),
                    ("s5_bind_local_v3_bind", 31)):
        spec = TK.CANONICAL[name]
        k, m = spec.k, spec.n_objects_active
        ex = TK.generate(spec, "test", n=200, length=L)
        ns, ng = s5_bind_v3_shape(ex)
        named, query = s5_bind_v3_is_named(ex), s5_bind_v3_query_kind(ex)
        rows = (list(S5_BIND_V3_ROWS) + list(S5_BIND_V3_CKPT_ROWS) + ["surface_ranker"]
                + list(s5_bind_v3_family_rows(k, m, ns, ng, named, query)))
        for r in rows:
            assert (s5_bind_v3_admits(r, k, m, ns, ng, named, query)
                    == s5_bind_v3_pad_admits(r, k, m, ns, ng, named, query, pad=0)), (
                f"{name}@{L}: pad=0 disagrees with the plain rule on {r!r}")


def test_the_composed_floor_survives_exactly_the_pads_narrower_than_one_structure():
    """The single inequality the bounded-pad protocol rests on, checked as a boundary.

    A pad of w slots is w free live slots, so the task's W = k + m + 1 costs k + m + 1 - w of a
    policy's own and the class excludes it iff that exceeds max(k, m) + 1, i.e. iff w < min(k, m).
    The boundary is what matters: at w = min(k, m) the task TIES the bound and is admitted, so
    "narrower than one structure" would be off by one and the cell would be unfloorable at the
    width a half-and-half format hands out.
    """
    for k, m in ((6, 6), (12, 12), (8, 5)):
        assert s5_bind_v3_pad_max_width(k, m) == min(k, m) - 1
        for w in range(0, k + m + 1):
            want = w < min(k, m)
            assert s5_bind_v3_pad_floorable(k, m, w) is want, (k, m, w)
            # a COMPONENT cell is floored at every width: its rule never used the W axis
            assert s5_bind_v3_pad_floorable(k, m, w, named=True) is True, (k, m, w)


def test_the_partial_carry_family_is_what_a_pad_lets_in_and_it_is_priced():
    """A pad of w admits exactly ``partial_carry_j`` for j <= w, and that family sets the floor.

    Both halves matter. If the family were not priced at all it would be silently excluded and the
    bounded floor would be reported too LOW, which is the direction that manufactures a cleared
    floor; if it were admitted at every j the pad would admit the task by another name.
    """
    spec = TK.CANONICAL["s5_bind_local_v3"]
    k, m = spec.k, spec.n_objects_active
    ex = TK.generate(spec, "test", n=200, length=48)
    ns, ng = s5_bind_v3_shape(ex)
    for w in range(0, m + 1):
        for j in range(m + 1):
            got = s5_bind_v3_pad_admits(f"partial_carry_j{j}", k, m, ns, ng, False, "state", w)
            assert got is (j <= w), f"pad={w}: partial_carry_j{j} admitted={got}"
    assert s5_bind_v3_row_cost("partial_carry_j3", k, m, ns, ng, "state")[0] == k + 3 + 1


def test_the_bounded_pad_floor_is_the_plain_floor_up_to_width_two_and_rises_after():
    """The measurement the format width is chosen on, pinned at the k = 6 operating point.

    ``pad <= 2`` costs the composed cell's floor NOTHING — it is the plain protocol's own number —
    and every wider floorable pad is strictly worse. That is the whole reason the registered
    format is two slots and not five: five is floorable and useless (the bar at composed@48 is
    0.763), two is floorable and free.
    """
    spec = TK.CANONICAL["s5_bind_local_v3"]
    k, m = spec.k, spec.n_objects_active
    ex = TK.generate(spec, "test", n=128, length=48)
    ns, ng = s5_bind_v3_shape(ex)
    fl = s5_bind_v3_pad_floors(ex, k, m, False, "state")
    plain = s5_bind_v3_operative_floor(fl, k, m, ns, ng, False, "state")
    at = {w: s5_bind_v3_pad_operative_floor(fl, k, m, ns, ng, False, "state", pad=w)
          for w in (0, 1, 2, 3, 4, 5, 6, 12)}
    assert at[1] == at[2] == plain, at
    assert at[3] > at[2] and at[4] > at[3] and at[5] > at[4], at
    assert at[6] is None and at[12] is None, at
    # and the component cells do not move at any width — a pad buys registers, not chaining
    for name, L in (("s5_bind_local_v3_state", 17), ("s5_bind_local_v3_bind", 31)):
        cs = TK.CANONICAL[name]
        cex = TK.generate(cs, "test", n=128, length=L)
        cns, cng = s5_bind_v3_shape(cex)
        cq = s5_bind_v3_query_kind(cex)
        cfl = s5_bind_v3_pad_floors(cex, cs.k, cs.n_objects_active, True, cq)
        vals = {w: s5_bind_v3_pad_operative_floor(cfl, cs.k, cs.n_objects_active, cns, cng,
                                                  True, cq, pad=w)
                for w in (0, 1, 2, 5, 6, 12)}
        assert len(set(vals.values())) == 1, f"{name}@{L}: the pad moved a component floor {vals}"


def _run() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                fails += 1
                print(f"  FAIL  {name}: {exc}")
    print("OK" if not fails else f"{fails} FAILURES")
    return fails


if __name__ == "__main__":
    sys.exit(_run())
