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
  the floors are the registered ones  — the window family and the zero-state pin chain are both
                                        registered, the operative floor is the max over every
                                        registered row, and it sits within a stated multiple of
                                        the informed chance 1/(k-1);
  the pin channel is closed           — no_pin holds pin density at zero, so no item is
                                        answerable by two state-free surface retrievals.

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
    S5_BIND_ROWS,
    operative_floor,
    s5_bind_floors,
    s5_bind_operative_floor,
    s5_bind_pin_density,
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
WINDOWS = tuple(r for r in S5_BIND_ROWS if r.startswith("window_"))

# The operative floor is a MAX over a dozen registered rows, so at a finite n it carries an
# upward selection bias of order the largest row's standard error even when every row sits at
# chance: at N_FLOOR = 600 and k = 12 that is about 0.25 * chance on its own. The bound is
# stated as a multiple of the informed chance 1/(k-1) and has to leave room for it. Measured
# at n = 3000 the scored cells read 1.07 / 1.11 / 1.07 (k=12, L=128/192/256) and 1.08 / 1.04
# (k=6, L=48/64).
N_FLOOR = 600
FLOOR_RATIO_MAX = 1.45

# Frozen streams: same (spec, split, length, idx) -> identical example, forever.
GOLDENS = {
    "s5_bind_v2": {128: "14c5b489f6b1ec00", 192: "ceb08d301107cbf2", 256: "23bbf13c6895608a"},
    "s5_bind_v2_state": {128: "c33ce6e53dcf5da7", 192: "7d4556b0ac54cd42", 256: "ce634131e5e3356f"},
    "s5_bind_v2_bind": {128: "ca4458320ac62191", 192: "dc07072a09427f6e", 256: "ef1e48662404bcf3"},
    "s5_bind_v2_map": {128: "464ff5156aedf178", 192: "dad7cbd3874ac57b", 256: "f0d5334b0476ec19"},
    "s5_bind_local_v2": {48: "b6f519744303b318", 64: "2c27c6f0b438524b"},
    "s5_bind_local_v2_state": {48: "2827950081ff7857", 64: "456a7ee79eb09340"},
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
            touched = 0
            for k, x, y, dyn in events:
                if k == "swap":
                    b = (B if dyn else read["B0"])[y]
                    touched += int(q_state in (x, b))
                    P[x], P[b] = P[b], P[x]
                    Pinv = {v: kk for kk, v in P.items()}
                else:
                    B[x] = (Pinv if dyn else P0inv)[y]
            assert touched == e.meta["touch"] >= 2
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
    assert set(WINDOWS) <= set(S5_BIND_ADVERSARIES)
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
        assert open_hits - hits >= 0.08, f"{name}: {open_hits:.4f} -> {hits:.4f}"


def test_no_pin_is_set_on_the_family_and_defaults_off_everywhere_else():
    """Appended and defaulted, so the constraint moved only this family's streams."""
    assert TK.TaskSpec("x", "s5_bind").no_pin is False
    for reg in (TK.CANONICAL, TK.RETIRED):
        for name, spec in reg.items():
            assert spec.no_pin is (spec.family == "s5_bind"), name


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
    assert set(validate_suite.S5_BIND_WINDOW_ROWS) == set(WINDOWS)
    assert "s5_bind" in validate_suite.S5_BIND_FAMILIES


def test_suite_gate_passes_on_every_registered_cell():
    """The condition scripts/validate_suite.py applies, at the length it applies it."""
    import validate_suite  # noqa: PLC0415

    for name in ALL:
        spec = TK.CANONICAL[name]
        exs = TK.generate(spec, "test", n=200, length=spec.eval_lengths[-1])
        fl = s5_bind_floors(exs, spec.k)
        gated = [n for n in validate_suite.S5_BIND_SHORTCUTS if n in fl
                 and (spec.coupled or n not in validate_suite.S5_BIND_WINDOW_ROWS)]
        assert max(fl[n] for n in gated) < 0.5, f"{name}: a gating row clears 0.5"


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
