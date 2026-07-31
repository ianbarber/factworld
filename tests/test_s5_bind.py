"""s5_bind — the mutual-reference composition rung.

Two structures run over one interleaved event stream (P: agents -> roles, permuted by the
swaps; B: objects -> agents, rewritten by the gives) and every event names its second operand
THROUGH the other structure. The whole construct rests on four properties, and this file pins
each of them:

  the arms are ONE item stream        — the composed cell, both components and the capacity
                                        control are the same worlds, events and queries, so a
                                        difference between arms is within-item;
  the coupling ablation is free       — the coupled and decoupled renderings of an item differ
                                        by two tokens per referenced event and NOT in length,
                                        and at rho=0 they are byte-identical;
  the prompt alone determines gold    — replaying the rendered sentences reproduces the answer,
                                        so nothing is carried in meta;
  the floor is a resource class       — a row may set the floor only if it carries no map, so
                                        the whole block-drop continuum is out at once rather
                                        than member by member, and the max over what is left
                                        sits within a stated multiple of the informed chance;
  the pin channel is closed           — no_pin holds pin density at zero, so no item is
                                        answerable by two state-free surface retrievals;
  the block-drop family is dead       — chain_max_gap bounds the queried agent's dependency
                                        chain in time, so every drop of width >= w_min lands on
                                        an event that can change the answer.

Runs with zero dependencies:  python3 tests/test_s5_bind.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter
from dataclasses import fields
from math import log

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld import tasks as TK  # noqa: E402
from factworld.render import Renderer  # noqa: E402
from factworld.tokenizer import Tokenizer  # noqa: E402
from factworld.validity import (  # noqa: E402
    S5_BIND_ADVERSARIES,
    S5_BIND_CHANCE_ROWS,
    S5_BIND_COUPLED_ONLY_ROWS,
    S5_BIND_MAP_CARRYING_ROWS,
    S5_BIND_ROW_SLOTS,
    S5_BIND_ROWS,
    S5_BIND_TRUNCATION_ROWS,
    S5_BIND_WINDOWS,
    operative_floor,
    s5_bind_block_drop,
    s5_bind_chain,
    s5_bind_floors,
    s5_bind_runs,
    s5_bind_operative_floor,
    s5_bind_pin_density,
    s5_bind_preds,
    s5_bind_read,
    _sb_answer,
    _sb_pin_chain,
    _sb_run,
)
from factworld.world import Event  # noqa: E402

ARMS = ("s5_bind_v2", "s5_bind_v2_state", "s5_bind_v2_bind", "s5_bind_v2_map")
LOCAL = ("s5_bind_local_v2", "s5_bind_local_v2_state")
ALL = ARMS + LOCAL
COUPLING_BLIND = ("stale_resolution", "one_leg_B", "one_leg_P", "final_state_resolution")
WINDOWS = S5_BIND_TRUNCATION_ROWS

# The operative floor is a MAX over a dozen registered rows, so at a finite n it carries an
# upward selection bias of order the largest row's standard error even when every row sits at
# chance: at N_FLOOR = 600 and k = 12 that is about 0.25 * chance on its own. The bound is
# stated as a multiple of the informed chance 1/(k-1) and has to leave room for it. Measured
# at n = 3000 the scored cells read 1.03 / 1.03 / 1.15 (k=12, L=128/192/256) and 1.11 / 1.09
# (k=6, L=48/64).
N_FLOOR = 600
FLOOR_RATIO_MAX = 1.45

# Frozen streams: same (spec, split, length, idx) -> identical example, forever.
GOLDENS = {
    "s5_bind_v2": {128: "38e62694d969a363", 192: "481d2b911881248b", 256: "bc60c4a900b04a2a"},
    "s5_bind_v2_state": {128: "3a0a8f810d39b09c", 192: "7b054e8717274d9a", 256: "0ed5e7279b64e7fe"},
    "s5_bind_v2_bind": {128: "ce4ee7fefdcbb640", 192: "7b5fafd44f46d1d6", 256: "5afe941df3d1a526"},
    "s5_bind_v2_map": {128: "b97bcbe20a418b46", 192: "b65225a399331af4", 256: "78e5d780ec422dd8"},
    "s5_bind_local_v2": {48: "8dcdc6b9b49b0f9a", 64: "0ac09d59ddae6cdd"},
    "s5_bind_local_v2_state": {48: "b109dcde2f3757d4", 64: "853971690e132d1e"},
}


def _hash(examples) -> str:
    return hashlib.sha256(
        "\n".join(f"{e.prompt}\t{e.answer}" for e in examples).encode()).hexdigest()[:16]


def _statements(prompt: str) -> list[str]:
    """The prompt back as the statements it was rendered from (no content id holds a '.' or
    '?', so the split is exact)."""
    return [s.strip() for s in re.findall(r"[^.?]*[.?]", prompt)]


# --- registry ---------------------------------------------------------------------------

def test_registry_contract():
    for name in ALL:
        spec = TK.CANONICAL[name]
        assert spec.name == name and spec.family == "s5_bind"
        assert spec.kind == "experimental" and name not in TK.REPORTED
        assert spec.n_objects_active <= spec.k            # the stated holder map is injective
        assert TK.spec_for(name) is spec
    # ONE item stream per operating point: the arms share a stream_name and differ ONLY in
    # what is rendered and what is asked.
    for group in (ARMS, LOCAL):
        base = TK.CANONICAL[group[0]]
        for name in group[1:]:
            other = TK.CANONICAL[name]
            assert other.stream_name == base.stream_name
            for f in (fl.name for fl in fields(TK.TaskSpec)):
                if f not in ("name", "coupled", "query_arm"):
                    assert getattr(other, f) == getattr(base, f), f"{name}.{f}"
    assert {TK.CANONICAL[n].query_arm for n in ARMS} == {"state", "bind", "state_all"}
    assert [TK.CANONICAL[n].coupled for n in ARMS] == [True, False, False, False]


def test_stream_name_defaults_to_name_everywhere_else():
    """The pairing mechanism is opt-in: every spec outside this family keys its RNG on its own
    name exactly as before, which is why adding the field moved no stream."""
    for reg in (TK.CANONICAL, TK.RETIRED):
        for name, spec in reg.items():
            if spec.family != "s5_bind":
                assert spec.stream_name is None, name


def test_frozen_stream_goldens():
    for name, per_len in GOLDENS.items():
        spec = TK.CANONICAL[name]
        assert tuple(per_len) == tuple(spec.eval_lengths)
        for L, want in per_len.items():
            got = _hash(TK.generate(spec, "test", n=25, length=L))
            assert got == want, f"{name}@L{L}: frozen-spec immutability VIOLATED ({got})"


def test_determinism():
    for name in ("s5_bind_v2", "s5_bind_local_v2"):
        spec = TK.CANONICAL[name]
        for split, L in (("test", spec.eval_lengths[0]), ("train", None)):
            a = TK.generate(spec, split, n=25, length=L)
            b = TK.generate(spec, split, n=25, length=L)
            assert [(x.prompt, x.answer, x.meta) for x in a] == \
                   [(x.prompt, x.answer, x.meta) for x in b], f"{name} {split}"


# --- pairing and the coupling ablation ----------------------------------------------------

def _body(prompt: str) -> list[str]:
    """The statements before the query, with the temporal phrase neutralised — what two arms
    of the same item must agree on exactly."""
    return [s.replace(Renderer.AT_POINT, Renderer.AT_START) for s in _statements(prompt)[:-1]]


def test_arms_are_one_item_stream():
    """Index i is the same world, the same events and the same two queries in every arm."""
    spec = TK.CANONICAL["s5_bind_v2"]
    arms = TK.s5_bind_arms(spec, "test", n=60, length=64)
    ref = arms[(True, "state")]
    for lst in arms.values():
        for a, b in zip(ref, lst):
            assert a.meta["q_state"] == b.meta["q_state"]
            assert a.meta["q_bind"] == b.meta["q_bind"]
            assert a.meta["n_swap"] == b.meta["n_swap"]
            assert a.meta["writes"] == b.meta["writes"]
            assert a.meta["last_write_pos"] == b.meta["last_write_pos"]
            assert _body(a.prompt) == _body(b.prompt)
    # the whole-map readout's slot for the queried agent IS the state arm's answer, item by
    # item — the control and the component read the same P, so a gap between them is capacity
    # and not a different world
    for st, mp in zip(arms[(False, "state")], arms[(False, "state_all")]):
        agents = re.search(r"each of ([^?]*) have", mp.prompt).group(1).split(", ")
        slot = agents.index(st.meta["q_state"])
        assert mp.answer.rstrip(".").split()[slot] == st.answer.rstrip(".")


def test_coupled_and_decoupled_have_identical_token_counts():
    """The ablation moves two tokens per referenced event ("this point" <-> "the start") and
    nothing else, so prompt length cannot explain an arm difference."""
    for name, L in (("s5_bind_v2", 64), ("s5_bind_local_v2", 32)):
        spec = TK.CANONICAL[name]
        arms = TK.s5_bind_arms(spec, "test", n=60, length=L,
                               arms=((True, "state"), (False, "state")))
        c, d = arms[(True, "state")], arms[(False, "state")]
        for a, b in zip(c, d):
            ta, tb = a.prompt.split(), b.prompt.split()
            assert len(ta) == len(tb), f"{name}: arm token counts differ"
            assert sum(1 for x, y in zip(ta, tb) if x != y) == 2 * a.meta["n_ref"]
            assert a.length == b.length


def test_rho_zero_makes_the_arms_byte_identical():
    """The dose-response's origin. At rho=0 no event is referenced, the coupled rendering IS
    the decoupled one, and the two golds coincide — so the coupling knob, not some other
    difference between the arms, is what the ablation moves."""
    spec = TK.CANONICAL["s5_bind_v2"].scaled(rho_p=0.0, rho_b=0.0)
    arms = TK.s5_bind_arms(spec, "test", n=40, length=64,
                           arms=((True, "state"), (False, "state")))
    c, d = arms[(True, "state")], arms[(False, "state")]
    assert all(a.prompt == b.prompt and a.answer == b.answer for a, b in zip(c, d))
    assert all(a.meta["n_ref"] == 0 for a in c)
    # and at rho=1 the coupling changes the answer on most items
    full = TK.s5_bind_arms(TK.CANONICAL["s5_bind_v2"], "test", n=200, length=64,
                           arms=((True, "state"), (False, "state")))
    changed = sum(a.answer != b.answer
                  for a, b in zip(full[(True, "state")], full[(False, "state")])) / 200
    assert changed > 0.75, changed


# --- item validity ------------------------------------------------------------------------

def test_query_gates_hold_on_every_item():
    for name, L in (("s5_bind_v2", 64), ("s5_bind_local_v2", 32)):
        spec = TK.CANONICAL[name]
        for e in TK.generate(spec, "test", n=100, length=L):
            read = s5_bind_read(e.prompt)
            events = read["events"]
            assert len(events) == L
            q_state, q_bind = e.meta["q_state"], e.meta["q_bind"]
            # the state query: the queried agent moved >= 2x and does not end where it started
            # moved at least twice under BOTH semantics — recounted from the surface, since
            # the queried agent may be reached only as the REFERENCED operand of a swap
            B = dict(read["B0"])
            P0inv = {v: k for k, v in read["P0"].items()}
            Pinv = dict(P0inv)
            P = dict(read["P0"])
            touched, last = 0, -1
            for i, (k, x, y, dyn) in enumerate(events):
                if k == "swap":
                    b = (B if dyn else read["B0"])[y]
                    if q_state in (x, b):
                        touched += 1
                        last = i
                    P[x], P[b] = P[b], P[x]
                    Pinv = {v: kk for kk, v in P.items()}
                else:
                    B[x] = (Pinv if dyn else P0inv)[y]
            assert touched == e.meta["touch"] >= 2
            # and its LAST carrier event is inside the final q_tail of the stream, so the
            # events a prefix policy discards are load-bearing (TaskSpec.q_tail)
            assert last >= L - max(1, round(spec.q_tail * L)), (name, last)
            if spec.query_arm == "state":                 # and does not end where it started
                assert read["P0"][q_state] != e.answer.rstrip(".")
            # the bind query: >= 2 writes and the resolving one inside [0.1L, 0.75L]
            writes = [i for i, (k, x, _y, _d) in enumerate(events)
                      if k == "give" and x == q_bind]
            assert len(writes) >= 2
            assert L // 10 <= writes[-1] <= int(0.75 * L)
            # no event is a no-op under EITHER semantics: a swap never names its own agent
            # through the stated holder map, and a give never restates a holder
            for k, x, y, _d in events:
                if k == "swap":
                    assert read["B0"][y] != x


def test_prompt_alone_determines_the_answer():
    """Replaying the rendered sentences under their own temporal phrases reproduces gold on
    every arm — the surface carries the whole computation, and nothing is smuggled in meta."""
    for name in ALL:
        spec = TK.CANONICAL[name]
        for L in spec.eval_lengths[:2]:
            for e in TK.generate(spec, "test", n=40, length=L):
                read = s5_bind_read(e.prompt)
                assert _sb_answer(read, _sb_run(read, "surface")) == e.answer, name


def test_answer_balance():
    for name in ("s5_bind_v2", "s5_bind_local_v2"):
        spec = TK.CANONICAL[name]
        exs = TK.generate(spec, "test", n=600, length=spec.eval_lengths[0])
        firsts = [e.answer.rstrip(".").split()[0] for e in exs]
        cnt, n, k = Counter(firsts), len(exs), spec.k
        kl = sum((v / n) * log((v / n) * k) for v in cnt.values()) - (k - 1) / (2 * n)
        assert len(cnt) == k, f"{name}: {len(cnt)}/{k} answers occur"
        assert kl < 0.02, f"{name}: KL excess {kl:.4f}"
        assert max(cnt.values()) / n < 1.0 / k + 0.05


def test_event_trace_replays_to_the_scored_answer():
    """The dense supervision is the same computation, checkpointed: the last state in the
    trace is the one the query reads."""
    spec = TK.CANONICAL["s5_bind_local_v2"]
    for e in TK.generate(spec, "test", n=30, length=32):
        cells = e.meta["trace"].split()
        width = spec.k + spec.n_objects_active
        assert len(cells) == 32 * width
        final_P = cells[-width:][: spec.k]
        agents = [f"g{i}" for i in range(spec.k)]
        assert final_P[agents.index(e.meta["q_state"])] == e.answer.rstrip(".")
        assert e.meta["interleaved_prompt"].endswith(e.prompt.split(". ")[-1])


# --- rendering ------------------------------------------------------------------------------

def test_every_surface_round_trips_through_the_parser():
    """A prior proposal failed here: four of five surfaces did not parse back."""
    r = Renderer()
    cases = [
        (r.render_role("g0", "r3", when=Renderer.AT_START),
         {"type": "role", "agent": "g0", "role": "r3", "step": None, "when": Renderer.AT_START}),
        (r.render_holder("o2", "g1", when=Renderer.AT_START),
         {"type": "holder", "object": "o2", "holder": "g1", "step": None,
          "when": Renderer.AT_START}),
        (r.render_event(Event("swap_roles_now", ("g4", "o2")), step="s0"),
         {"type": "event", "event": Event("swap_roles_now", ("g4", "o2")), "step": "s0"}),
        (r.render_event(Event("swap_roles_start", ("g4", "o2")), step="s0"),
         {"type": "event", "event": Event("swap_roles_start", ("g4", "o2")), "step": "s0"}),
        (r.render_event(Event("give_role_now", ("o3", "r5")), step="s1"),
         {"type": "event", "event": Event("give_role_now", ("o3", "r5")), "step": "s1"}),
        (r.render_event(Event("give_role_start", ("o3", "r5")), step="s1"),
         {"type": "event", "event": Event("give_role_start", ("o3", "r5")), "step": "s1"}),
        (r.render_query("s5bind_state", target="g4"),
         {"type": "query", "family": "s5bind_state", "target": "g4", "step": None}),
        (r.render_query("s5bind_bind", target="o2"),
         {"type": "query", "family": "s5bind_bind", "target": "o2", "step": None}),
        (r.render_query("s5bind_state_all", targets=["g0", "g1", "g2"]),
         {"type": "query", "family": "s5bind_state_all", "targets": ("g0", "g1", "g2"),
          "step": None}),
    ]
    for text, want in cases:
        assert r.parse(text) == want, text
    # and every statement of a real prompt, in every arm, parses back to a typed record
    for name in ALL:
        spec = TK.CANONICAL[name]
        for e in TK.generate(spec, "test", n=5, length=spec.eval_lengths[0]):
            recs = [r.parse(s) for s in _statements(e.prompt)]
            assert recs[-1]["type"] == "query"
            assert all(rec["type"] in ("role", "holder", "event") for rec in recs[:-1])
            assert sum(rec["type"] == "event" for rec in recs) == spec.eval_lengths[0]


def test_the_temporal_phrase_is_the_only_thing_the_two_readings_differ_in():
    r = Renderer()
    for kind_now, kind_start, args in (("swap_roles_now", "swap_roles_start", ("g4", "o2")),
                                       ("give_role_now", "give_role_start", ("o3", "r5"))):
        a = r.render_event(Event(kind_now, args), step="s0").split()
        b = r.render_event(Event(kind_start, args), step="s0").split()
        assert len(a) == len(b)
        assert [i for i, (x, y) in enumerate(zip(a, b)) if x != y] and \
               sum(1 for x, y in zip(a, b) if x != y) == 2


def test_pre_existing_surfaces_are_untouched():
    """The mutual-reference branch runs before every other shape, so this is the regression
    lock on the shapes it precedes."""
    r = Renderer()
    assert r.parse(r.render_role("g0", "r3")) == \
        {"type": "role", "agent": "g0", "role": "r3", "step": None}
    assert r.parse(r.render_holder("o2", "g1")) == \
        {"type": "holder", "object": "o2", "holder": "g1", "step": None}
    # Pinned as the parser ALREADY behaves, not as it ideally would: the pointer-map swap and
    # cycle route to the generic role-permutation records (only the referenced form carries a
    # kind of its own), and the mutual-reference branch must not disturb that.
    for e, want in ((Event("swap_a0", ("g1", "g2")), Event("swap_role", ("g1", "g2"))),
                    (Event("swap_a0_ref", ("g3", "g4")), Event("swap_a0_ref", ("g3", "g4"))),
                    (Event("cycle_a0", ("g5", "g6", "g7")),
                     Event("cycle_roles", ("g5", "g6", "g6", "g7", "g7", "g5"))),
                    (Event("give", ("o0", "g1")), Event("give", ("o0", "g1"))),
                    (Event("swap_role", ("g1", "g2")), Event("swap_role", ("g1", "g2"))),
                    (Event("cycle_roles", ("g1", "g2", "g3")),
                     Event("cycle_roles", ("g1", "g2", "g3")))):
        assert r.parse(r.render_event(e, step="s0"))["event"] == want, e.kind
    assert r.parse(r.render_query("state_hard", target="g2"))["family"] == "state_hard"
    assert r.parse(r.render_query("state_easy", target="o2"))["family"] == "state_easy"


def test_tokenizer_covers_every_surface_with_no_unknowns():
    """A 17-26% unknown rate on another family invalidated a whole local battery; the
    mutual-reference grammar adds five surfaces and a comma-separated slot enumeration."""
    for name in ALL:
        spec = TK.CANONICAL[name]
        world, renderer = TK.build_world(spec)
        tok = Tokenizer.build([world], renderer)
        for split, lengths in (("train", (None,)), ("test", spec.eval_lengths)):
            for L in lengths:
                for e in TK.generate(spec, split, n=4, length=L):
                    for s in [e.prompt, e.answer] + [v for v in e.meta.values()
                                                     if isinstance(v, str)]:
                        ids = tok.encode(s)
                        assert tok.unk_id not in ids, f"{name}: <unk> in {s[:120]!r}"
                        assert tok.decode(ids) == s, f"{name}: round trip broken"


# --- floors ----------------------------------------------------------------------------------

def test_floors_are_recomputed_from_the_exact_items():
    spec = TK.CANONICAL["s5_bind_v2"]
    exs = TK.generate(spec, "test", n=300, length=64)
    fl = s5_bind_floors(exs, spec.k)
    assert set(fl) == set(S5_BIND_ROWS)
    assert fl["uniform"] == 1.0 / spec.k and fl["uniform_non_initial"] == 1.0 / (spec.k - 1)
    # the no-op policy is exactly zero: the query gates force gold != the stated value
    assert fl["initial_only"] == 0.0
    # a shuffled copy of the same items gives the same rows (a property of the item set)
    assert s5_bind_floors(list(reversed(exs)), spec.k) == fl


def test_the_operative_floor_is_the_max_over_every_registered_row():
    """What the floor has to be once the pin channel is closed.

    Both families that could set it are registered — the recency windows (simulate the task
    exactly from the stated maps over the last fraction of the stream) and the zero-state pin
    chain — and the operative floor is the max over every registered row, so no registered
    policy sits above it. On the scored grid that max lands within a stated multiple of the
    informed chance 1/(k-1), which is what the construct claims: there is no policy cheaper
    than the forward pass. A floor BELOW chance is a mis-report — impossible here because the
    chance row is itself registered — and a floor far above it is an open shortcut.

    The earlier form of this test asserted the floor was at least 1.5x chance, which pinned the
    open channel as a requirement: the corrected construct fails it by design.
    """
    assert not set(WINDOWS) & set(S5_BIND_ADVERSARIES)   # class-excluded, see the cost test
    assert "pin_chain" in S5_BIND_ADVERSARIES
    for name in ("s5_bind_v2", "s5_bind_local_v2"):
        spec = TK.CANONICAL[name]
        chance = 1.0 / (spec.k - 1)
        for L in spec.eval_lengths:
            fl = s5_bind_floors(TK.generate(spec, "test", n=N_FLOOR, length=L), spec.k)
            op = s5_bind_operative_floor(fl)
            assert op == operative_floor(fl, S5_BIND_ADVERSARIES)
            assert all(v <= op for r, v in fl.items() if r in S5_BIND_ADVERSARIES), f"{name}@{L}"
            assert op >= chance, f"{name}@{L}: floor {op:.4f} below chance {chance:.4f}"
            assert op <= FLOOR_RATIO_MAX * chance, \
                f"{name}@{L}: floor {op:.4f} is {op / chance:.2f}x chance {chance:.4f}"


def test_the_family_dispatches_its_own_registered_set():
    """The mis-report the named accessor exists to prevent: the two adversary families share
    exactly one row name, so a fixed pointer-map default found ``uniform`` in a mutual-reference
    floor dict and returned 1/k — a floor BELOW this family's own informed chance."""
    from factworld.validity import S5_CHAIN_ADVERSARIES, registered_for  # noqa: PLC0415

    spec = TK.CANONICAL["s5_bind_v2"]
    fl = s5_bind_floors(TK.generate(spec, "test", n=100, length=64), spec.k)
    assert set(S5_CHAIN_ADVERSARIES) & set(fl) == {"uniform"}
    assert registered_for(fl) is S5_BIND_ADVERSARIES
    assert operative_floor(fl) == s5_bind_operative_floor(fl) >= fl["uniform_non_initial"]


def test_no_pin_closes_the_state_free_reset_channel():
    """The defect the fix closes, measured on the same cell with the knob off and on.

    A dynamic give pins its object's holder to the role it names; a later dynamic swap naming
    that object then writes that role onto its own agent, and the two references cancel the
    state. With the channel open a 2-retrieval zero-state policy reads well above chance and
    does NOT decay with length — which is what the window rows were reading. With ``no_pin``
    the sampler never emits the second event of such a pair: pin density is exactly zero and
    the policy falls to chance or below.

    The contrast is stated WITHIN the cell rather than as a ratio to chance, because at k=6
    chance is 0.200 and the ratio compresses; the absolute drop is the same size in both.

    It is SMALLER than it was before ``chain_max_gap``, and the reason is not that the channel
    weakened: pin density with the knob off is unchanged at 0.26-0.29. The steer often puts the
    queried agent on the REFERENCED side of its last carrier event, and the pin walk keys on the
    named side, so the policy has no chain to read on those items and falls back to its one-hop
    default. The stream property is the primary assertion here for that reason.
    """
    for name in ("s5_bind_v2", "s5_bind_local_v2"):
        spec = TK.CANONICAL[name]
        chance = 1.0 / (spec.k - 1)
        L = spec.eval_lengths[0]
        assert spec.no_pin
        acc = {}
        for closed, s in ((False, spec.scaled(no_pin=False)), (True, spec)):
            exs = TK.generate(s, "test", n=N_FLOOR, length=L)
            acc[closed] = (
                s5_bind_pin_density(exs),
                sum(_sb_pin_chain(s5_bind_read(e.prompt)) == e.answer for e in exs) / len(exs))
        (open_density, open_hits), (density, hits) = acc[False], acc[True]
        assert open_density > 0.2, f"{name}: channel not open in the control ({open_density:.4f})"
        assert density == 0.0, f"{name}: {density:.4f} of dynamic swaps still ride a pin"
        assert open_hits > chance, f"{name}: pin_chain {open_hits:.4f} with the channel open"
        assert hits <= chance, f"{name}: pin_chain {hits:.4f} vs chance {chance:.4f}"
        assert open_hits - hits >= 0.05, f"{name}: {open_hits:.4f} -> {hits:.4f}"


def test_no_pin_is_set_on_the_family_and_defaults_off_everywhere_else():
    """Appended and defaulted, so the constraint moved only this family's streams."""
    assert TK.TaskSpec("x", "s5_bind").no_pin is False
    for reg in (TK.CANONICAL, TK.RETIRED):
        for name, spec in reg.items():
            assert spec.no_pin is (spec.family == "s5_bind"), name


def test_the_truncation_family_is_measured_at_matched_budgets():
    """Truncation is a two-sided family and both sides pay the same price.

    window_f plays the LAST T = f*L events from the stated maps; prefix_f plays the FIRST T
    exactly and reads the true maps out there. Measuring one half and not the other prices the
    same T events differently at the two ends of the stream, which is what left prefix_90
    unmeasured at 4x the operative floor. Neither half is a FLOOR row — see the cost-class test.
    """
    assert len(S5_BIND_TRUNCATION_ROWS) == 2 * len(S5_BIND_WINDOWS)
    for f in S5_BIND_WINDOWS:
        tag = int(round(f * 100))
        assert f"window_{tag}" in S5_BIND_MAP_CARRYING_ROWS
        assert f"prefix_{tag}" in S5_BIND_MAP_CARRYING_ROWS
    spec = TK.CANONICAL["s5_bind_v2"]
    for e in TK.generate(spec, "test", n=20, length=64):
        read = s5_bind_read(e.prompt)
        preds = s5_bind_preds(e.prompt)
        L = len(read["events"])
        for f in S5_BIND_WINDOWS:
            T, tag = max(1, int(round(f * L))), int(round(f * 100))
            assert preds[f"window_{tag}"] == _sb_answer(
                read, _sb_run(read, "surface", start=L - T))
            assert preds[f"prefix_{tag}"] == _sb_answer(
                read, _sb_run(read, "surface", end=T))


def test_q_tail_closes_the_prefix_half():
    """The defect the gate closes, measured on the same cell with the knob off and on.

    The state gate asked that the queried agent MOVE but never said when, so its carrier chain
    typically finished mid-stream: simulating the task exactly and stopping 10% early was simply
    right on 45% of items at L=128, against an operative floor of 0.098. With q_tail the last
    carrier event is inside the final decile — the events that policy discards — and every
    prefix cut falls to chance or below.
    """
    for name in ("s5_bind_v2", "s5_bind_local_v2"):
        spec = TK.CANONICAL[name]
        assert spec.q_tail == 0.1
        chance = 1.0 / (spec.k - 1)
        L = spec.eval_lengths[0]
        rows = {}
        for gated, s in ((False, spec.scaled(q_tail=0.0, chain_max_gap=0.0)), (True, spec)):
            rows[gated] = s5_bind_floors(TK.generate(s, "test", n=N_FLOOR, length=L), spec.k)
        assert rows[False]["prefix_90"] > 2.0 * chance, \
            f"{name}: the hole is not reproduced ({rows[False]['prefix_90']:.4f})"
        for row in S5_BIND_TRUNCATION_ROWS:
            assert rows[True][row] <= FLOOR_RATIO_MAX * chance, \
                f"{name}: {row} {rows[True][row]:.4f} vs chance {chance:.4f}"


def test_q_tail_is_set_on_the_family_and_defaults_off_everywhere_else():
    """Appended and defaulted, so the gate moved only this family's streams."""
    assert TK.TaskSpec("x", "s5_bind").q_tail == 0.0
    for reg in (TK.CANONICAL, TK.RETIRED, TK.CALIBRATION):
        for name, spec in reg.items():
            assert spec.q_tail == (0.1 if spec.family == "s5_bind" else 0.0), name
    # the gate is a bound on an INDEX, matched to the tightest registered truncation budget:
    # exactly the events prefix_90 discards
    for L in (48, 64, 128, 192, 256):
        spec = TK.CANONICAL["s5_bind_v2"]
        assert TK._s5_bind_tail_lo(spec, L) == max(1, int(round(0.9 * L)))
        assert TK._s5_bind_tail_lo(spec.scaled(q_tail=0.0), L) == -1


def test_a_floor_row_carries_no_map():
    """The RESOURCE CLASS that decides registration, made checkable.

    The composed cell's cheapest correct algorithm carries P, its inverse and B — 2k + m live
    slots — so a row may set a floor only if its own live-slot count does not grow with k. That
    is one rule rather than a per-member judgement, which is what the block-drop family defeats:
    a per-item step threshold that admits one member admits the continuum around it.
    """
    assert set(S5_BIND_ROW_SLOTS) == set(S5_BIND_ADVERSARIES)
    assert not set(S5_BIND_ROW_SLOTS) & set(S5_BIND_MAP_CARRYING_ROWS)
    assert set(S5_BIND_ADVERSARIES) | set(S5_BIND_MAP_CARRYING_ROWS) == set(S5_BIND_ROWS)
    # bounded independently of k: the largest admitted row walks one carrier and a scratch
    assert max(S5_BIND_ROW_SLOTS.values()) <= 3
    # and the excluded ones are excluded because they carry a map, not because they are
    # inaccurate: on the decoupled retrieval arm they read ~1.000 and are still not floors
    assert "final_state_resolution" in S5_BIND_MAP_CARRYING_ROWS
    assert set(S5_BIND_TRUNCATION_ROWS) <= set(S5_BIND_MAP_CARRYING_ROWS)


def test_chain_max_gap_bounds_every_off_chain_run_after_the_first():
    """The gate, read back off the rendered surface: from the chain's first event on, no run of
    consecutive off-chain events is as long as w_min, the trailing run included, so no block of
    width w_min starting there fits inside one.

    The LEADING run is deliberately not bounded — see TaskSpec.chain_max_gap — and this pins
    that too: it is still drawn from the free distribution, which is what keeps the stated role
    of an early operand from being a one-retrieval shortcut.
    """
    for name in ("s5_bind_v2", "s5_bind_local_v2"):
        spec = TK.CANONICAL[name]
        assert spec.chain_max_gap
        for L in spec.eval_lengths:
            w_min = max(1, int(round(spec.chain_max_gap * L)))
            assert TK._s5_bind_gap_limit(spec, L) == w_min - 1
            leads = []
            for e in TK.generate(spec, "test", n=150, length=L):
                read = s5_bind_read(e.prompt)
                lead, rest = s5_bind_runs(read)
                assert rest < w_min, f"{name}@{L}: run {rest} >= {w_min} after the first event"
                leads.append(lead)
                assert s5_bind_chain(read)
            assert max(leads) >= w_min, f"{name}@{L}: the leading run looks bounded"


def test_the_block_drop_family_is_dead_at_and_above_w_min():
    """What the gate buys, over the family and not over a registered subset of it.

    Every block of width >= w_min drops an event that can change the answer, so the whole
    (position, width) surface above w_min sits at chance — including the six budgets the
    registered truncation rows sample, and including the late-interior positions the registered
    rows missed. Below w_min the class rule stands alone; that residual is measured by
    scripts/probe_s5bind_block_drop_20260730.py and is not folded into the floor.
    """
    positions = (0.0, 0.25, 0.5, 0.75, 0.85, 0.95, 1.0)
    for name in ("s5_bind_v2", "s5_bind_local_v2"):
        spec = TK.CANONICAL[name]
        chance = 1.0 / (spec.k - 1)
        L = spec.eval_lengths[0]
        exs = TK.generate(spec, "test", n=400, length=L)
        for width in (spec.chain_max_gap, 0.15, 0.25, 0.5):
            for pos in positions:
                acc = s5_bind_block_drop(exs, width, pos)
                assert acc <= FLOOR_RATIO_MAX * chance, \
                    f"{name}@{L}: drop {width:.2f}L @ {pos:.2f} reads {acc:.4f}"


def test_the_gate_reproduces_the_hole_when_it_is_off():
    """The defect the gate closes, on the same cell with the knob off. Without it the family's
    interior beats the operative floor by more than the registered endpoints did, which is why
    registering endpoints could never have closed it."""
    spec = TK.CANONICAL["s5_bind_v2"].scaled(chain_max_gap=0.0)
    L = 128
    exs = TK.generate(spec, "test", n=400, length=L)
    chance = 1.0 / (spec.k - 1)
    endpoints = max(s5_bind_block_drop(exs, 0.1, p) for p in (0.0, 1.0))
    interior = max(s5_bind_block_drop(exs, 0.1, p) for p in (0.85, 0.9, 0.95))
    assert interior > 2.0 * chance, f"the hole is not reproduced ({interior:.4f})"
    assert interior > 1.5 * endpoints, f"interior {interior:.4f} vs endpoints {endpoints:.4f}"


def test_chain_max_gap_is_set_on_the_family_and_defaults_off_everywhere_else():
    """Appended and defaulted, so the gate moved only this family's streams — and it is refused
    on a rho_ladder spec, whose five rungs have five different coupled trajectories."""
    assert TK.TaskSpec("x", "s5_bind").chain_max_gap == 0.0
    for reg in (TK.CANONICAL, TK.RETIRED, TK.CALIBRATION):
        for name, spec in reg.items():
            want = spec.family == "s5_bind" and name in ALL
            assert (spec.chain_max_gap > 0.0) is want, name
    for name in ALL:
        # w_min lands at 6-13 events on the k=12 grid and 5-6 on the k=6 one: the smallest
        # fraction whose w_min at the cell's shortest length still leaves the stream a free
        # swap:give mix
        spec = TK.CANONICAL[name]
        for L in spec.eval_lengths:
            assert 5 <= int(round(spec.chain_max_gap * L)) <= 13
    bad = TK.CALIBRATION["s5_bind_v2_lad100"].scaled(chain_max_gap=0.05)
    try:
        TK.generate(bad, "test", n=1, length=64)
    except ValueError as exc:
        assert "rho_ladder" in str(exc)
    else:
        raise AssertionError("a ladder spec accepted chain_max_gap")


def test_the_gate_does_not_buy_down_the_step_multiplier():
    """The gate is a constraint on WHEN the chain is touched, not on how much state is live: the
    composed cell still costs a forward pass carrying both maps, the components still admit
    their sparse walks, and the ratio between them holds. Steering re-targets a swap the stream
    was going to contain rather than inserting one, which is what keeps the swap:give mix — the
    thing both costs are priced on — from moving.

    Costed on the register machine the family is measured on: a swap costs one resolution plus
    four map writes under the composed forward pass and a give one resolution plus one write,
    while the decoupled component walks its answer's role backward at five steps per chain
    event over two live slots.
    """
    for name, L in (("s5_bind_v2", 192), ("s5_bind_local_v2", 48)):
        spec = TK.CANONICAL[name]
        ratio = {}
        for gated, s in ((False, spec.scaled(chain_max_gap=0.0)), (True, spec)):
            comp = dec = 0
            exs = TK.generate(s.scaled(coupled=True), "test", n=150, length=L)
            for e in exs:
                ev = s5_bind_read(e.prompt)["events"]
                n_swap = sum(1 for x in ev if x[0] == "swap")
                comp += 5 * n_swap + 2 * (len(ev) - n_swap)
            for e in TK.generate(s.scaled(coupled=False), "test", n=150, length=L):
                dec += 5 * len(s5_bind_chain(s5_bind_read(e.prompt))) + 2
            ratio[gated] = comp / dec
        assert ratio[True] >= 0.9 * ratio[False], \
            f"{name}@{L}: multiplier {ratio[False]:.2f} -> {ratio[True]:.2f}"
        assert ratio[True] >= 3.0, f"{name}@{L}: multiplier {ratio[True]:.2f}"


def test_coupling_blind_policies_reach_chance_on_the_long_cells():
    """Resolving every reference against the stated maps — the decoupled algorithm run on a
    coupled item — has to be worth nothing, or the composed cell is the component cell."""
    spec = TK.CANONICAL["s5_bind_v2"]
    for L in spec.eval_lengths:
        fl = s5_bind_floors(TK.generate(spec, "test", n=N_FLOOR, length=L), spec.k)
        for row in COUPLING_BLIND + ("pin_chain", "last_swap_1hop"):
            assert fl[row] <= 1.5 * fl["uniform_non_initial"], f"{row}@{L}: {fl[row]:.3f}"


def test_coupled_only_rows_are_dropped_on_a_decoupled_rendering():
    """With every reference rendered "at the start" each of those policies IS the oracle on the
    query it is defined for, so printing it would be a correctness check wearing a floor's
    clothes. The window and stated rows are defined on both renderings and stay."""
    assert set(COUPLING_BLIND) | {"pin_chain"} == set(S5_BIND_COUPLED_ONLY_ROWS)
    for name in ("s5_bind_v2_state", "s5_bind_v2_bind"):
        spec = TK.CANONICAL[name]
        exs = TK.generate(spec, "test", n=100, length=64)
        fl = s5_bind_floors(exs, spec.k)
        assert not set(S5_BIND_COUPLED_ONLY_ROWS) & set(fl)
        assert set(WINDOWS) <= set(fl)
    # they are dropped because they are exact, not because they are unmeasurable: stale
    # resolution reproduces gold on both decoupled queries, and the pin chain reproduces it on
    # the retrieval one (a static give's recipient IS the stated holder of the role it names)
    for e in TK.generate(TK.CANONICAL["s5_bind_v2_state"], "test", n=100, length=64):
        read = s5_bind_read(e.prompt)
        assert _sb_answer(read, _sb_run(read, "stale")) == e.answer
    for e in TK.generate(TK.CANONICAL["s5_bind_v2_bind"], "test", n=100, length=64):
        assert _sb_pin_chain(s5_bind_read(e.prompt)) == e.answer


def test_the_truncation_rows_are_measured_but_not_a_floor_on_either_rendering():
    """A floor row has to be in a strictly LOWER resource class than the task. On the decoupled
    retrieval arm the task is one content-addressed lookup and a truncated pass is 0.9L events,
    so both halves read ~1.000 by doing an order of magnitude more work than the cell asks for.
    The class rule reaches that case with no per-rendering exemption, which is why the ``coupled``
    argument no longer changes the max."""
    spec = TK.CANONICAL["s5_bind_v2_bind"]
    fl = s5_bind_floors(TK.generate(spec, "test", n=100, length=128), spec.k)
    assert not spec.coupled
    assert fl["prefix_90"] > 0.9 and fl["window_90"] > 0.9
    op = s5_bind_operative_floor(fl, coupled=False)
    assert op <= 1.5 / (spec.k - 1), f"a truncation row set the component floor ({op:.3f})"
    assert s5_bind_operative_floor(fl, coupled=True) == op


def test_the_anti_pin_guess_is_registered_chance_and_is_not_sampling_noise():
    """no_pin makes pin_chain an ANTI-predictor — the sampler rejects the very event on which it
    would have been right — so a guesser who strikes out its answer as well as the stated one is
    choosing uniformly over k-2 answers that carry more than (k-2)/(k-1) of the mass. The row is
    CHANCE, not a shortcut: its edge comes from the generator's rejection rule rather than from
    the item, so it enters the number a score is read against and not the suite gate.

    It is not a small-sample artifact either. The row is computed in closed form per item, so its
    only sampling error is that of pin_chain's own accuracy, and pin_chain sits many standard
    errors BELOW chance on every scored cell.
    """
    from math import sqrt  # noqa: PLC0415

    assert "uniform_anti_pin" in S5_BIND_CHANCE_ROWS
    assert "uniform_anti_pin" in S5_BIND_ADVERSARIES
    for name in ("s5_bind_v2", "s5_bind_local_v2"):
        spec = TK.CANONICAL[name]
        k, chance = spec.k, 1.0 / (spec.k - 1)
        n = N_FLOOR
        fl = s5_bind_floors(TK.generate(spec, "test", n=n, length=spec.eval_lengths[-1]), k)
        anti, pin = fl["uniform_anti_pin"], fl["pin_chain"]
        assert anti > chance
        # bounded above by striking one certainly-wrong answer — it cannot run away
        assert anti <= chance * (k - 1) / (k - 2) + 1e-9
        # and the reason is measured: pin_chain is many SE below chance, not near it
        se = sqrt(chance * (1 - chance) / n)
        assert (chance - pin) / se > 2.0, f"{name}: pin_chain {pin:.4f} vs chance {chance:.4f}"


def test_the_whole_map_readout_has_its_own_chance_row():
    """The capacity control's answer is a permutation of the k roles, so 1/k is not its
    chance level and uniform-over-non-initial is not defined for it."""
    spec = TK.CANONICAL["s5_bind_v2_map"]
    fl = s5_bind_floors(TK.generate(spec, "test", n=100, length=64), spec.k)
    assert "uniform_non_initial" not in fl
    assert fl["uniform"] < 1e-8                          # 1/12!
    assert fl["initial_only"] == 0.0


def test_registered_rows_reach_the_suite_gate_column():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import validate_suite  # noqa: PLC0415

    spec = TK.CANONICAL["s5_bind_v2"]
    fl = s5_bind_floors(TK.generate(spec, "test", n=50, length=64), spec.k)
    registered = {n for n in fl if n in S5_BIND_ADVERSARIES and n not in S5_BIND_CHANCE_ROWS}
    assert registered <= set(validate_suite.S5_BIND_SHORTCUTS)
    assert set(validate_suite.S5_BIND_SHORTCUTS) <= set(S5_BIND_ADVERSARIES)
    assert not set(validate_suite.S5_BIND_SHORTCUTS) & set(S5_BIND_CHANCE_ROWS)
    assert not set(validate_suite.S5_BIND_SHORTCUTS) & set(S5_BIND_MAP_CARRYING_ROWS)
    assert set(validate_suite.S5_BIND_DIAGNOSTIC_ROWS) == set(WINDOWS)
    assert "s5_bind" in validate_suite.S5_BIND_FAMILIES


def test_suite_gate_passes_on_every_registered_cell():
    """The condition scripts/validate_suite.py applies, at the length it applies it."""
    import validate_suite  # noqa: PLC0415

    for name in ALL:
        spec = TK.CANONICAL[name]
        exs = TK.generate(spec, "test", n=200, length=spec.eval_lengths[-1])
        fl = s5_bind_floors(exs, spec.k)
        gated = [n for n in validate_suite.S5_BIND_SHORTCUTS if n in fl]
        assert max(fl[n] for n in gated) < 0.5, f"{name}: a gating row clears 0.5"
        if spec.coupled and spec.chain_max_gap:
            lim = validate_suite.S5_BIND_DIAGNOSTIC_MAX / (spec.k - 1)
            for row in validate_suite.S5_BIND_DIAGNOSTIC_ROWS:
                assert fl[row] < lim, f"{name}: {row} {fl[row]:.3f} is not dead"


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
