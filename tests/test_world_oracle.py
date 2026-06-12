"""Instrument validity: the oracle must be correct by construction (gate: 100%),
and the world must not bake in leakage shortcuts (the validity fixes from the adversarial read).

Runs with zero dependencies:  python3 tests/test_world_oracle.py
or under pytest:               uv run --with pytest pytest -q
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.config import WorldConfig  # noqa: E402
from factworld.oracle import Oracle  # noqa: E402
from factworld.world import Event, World  # noqa: E402


def _world(seed: int = 0, **kw) -> World:
    return World(WorldConfig(seed=seed, **kw))


# --- determinism --------------------------------------------------------------------------

def test_world_is_deterministic():
    w1, w2 = _world(0), _world(0)
    assert w1.attrs == w2.attrs
    assert w1.initial_holder == w2.initial_holder
    assert w1.initial_assignment == w2.initial_assignment
    assert w1.entity_freq == w2.entity_freq
    assert w1.sample_hard_chain(16, "ep3") == w2.sample_hard_chain(16, "ep3")
    assert w1.sample_easy_chain(16, "ep3") == w2.sample_easy_chain(16, "ep3")


def test_different_seeds_differ():
    assert _world(0).attrs != _world(1).attrs


# --- recall + the value-token leakage fix (R4-i) ------------------------------------------

def test_recall_in_shared_vocab():
    w = _world()
    o = Oracle(w)
    pool = set(w.value_vocab)
    for ent in w.entities:
        for attr in w.attribute_names:
            assert o.recall(ent, attr) in pool


def test_value_tokens_are_opaque_and_shared():
    w = _world()
    o = Oracle(w)
    # a value token never encodes its attribute name
    for v in w.value_vocab:
        assert re.fullmatch(r"v\d+", v)
    # the same shared pool serves every attribute (no attribute -> answer-set partition to exploit)
    realised = {a: set() for a in w.attribute_names}
    for ent in w.entities:
        for a in w.attribute_names:
            realised[a].add(o.recall(ent, a))
    a0, a1 = w.attribute_names[0], w.attribute_names[1]
    assert realised[a0] & realised[a1]  # heavy overlap across attributes


# --- entity-frequency scaffolding (R5) ----------------------------------------------------

def test_entity_frequency_is_skewed_and_normalised():
    w = _world()  # freq_skew = 1.0
    assert abs(sum(w.entity_freq.values()) - 1.0) < 1e-9
    vals = sorted(w.entity_freq.values())
    assert vals[-1] > vals[0] * 5  # clearly long-tailed
    u = list(_world(freq_skew=0.0).entity_freq.values())
    assert max(u) - min(u) < 1e-9  # uniform when skew = 0


# --- easy-state + the holder-domain leakage fix (R4-ii) -----------------------------------

def test_easy_last_write_wins_and_as_of_t():
    w = _world()
    o = Oracle(w)
    obj = w.objects[0]
    ev = [Event("move", (obj, "loc1")), Event("give", (obj, "g2")), Event("move", (obj, "loc3"))]
    assert o.easy_holder(ev, obj) == "loc3"
    assert o.easy_holder(ev, obj, t=1) == "loc1"
    assert o.easy_holder(ev, obj, t=2) == "g2"
    other = w.objects[1]
    assert o.easy_holder(ev, other) == w.initial_holder[other]


def test_easy_trace_consistency():
    w = _world()
    o = Oracle(w)
    obj = w.objects[0]
    ev = w.sample_easy_chain(20, "t")
    trace = o.easy_trace(ev, obj)
    assert len(trace) == len(ev) + 1
    for t in range(len(ev) + 1):
        assert o.easy_holder(ev, obj, t=t) == trace[t]


def test_holder_domain_is_union_no_event_type_leak():
    w = _world()
    ev = w.sample_easy_chain(2000, "leak")
    holders, agents, locs = set(w.holders), set(w.agents), set(w.locations)
    assert all(e.args[1] in holders for e in ev)
    move_t = {e.args[1] for e in ev if e.kind == "move"}
    give_t = {e.args[1] for e in ev if e.kind == "give"}
    # both kinds reach agents AND locations -> the holder value cannot reveal the event kind
    assert move_t & agents and move_t & locs
    assert give_t & agents and give_t & locs


# --- hard-state (S_k composition) ---------------------------------------------------------

def test_hard_assignment_is_always_a_permutation():
    w = _world()
    o = Oracle(w)
    for length in (0, 1, 4, 16, 64):
        a = o.hard_assignment(w.sample_hard_chain(length, f"perm{length}"))
        assert set(a.keys()) == set(w.agents)
        assert sorted(a.values()) == sorted(w.roles)


def test_swap_is_an_involution():
    w = _world()
    o = Oracle(w)
    g0, g1 = w.agents[0], w.agents[1]
    ev = [Event("swap_role", (g0, g1)), Event("swap_role", (g0, g1))]
    assert o.hard_assignment(ev) == w.initial_assignment


def test_hard_chain_then_inverse_returns_to_identity():
    w = _world()
    o = Oracle(w)
    for length in (1, 5, 16, 32):
        ev = w.sample_hard_chain(length, f"inv{length}")
        inv = [e.inverse() for e in reversed(ev)]
        assert o.hard_assignment(ev + inv) == w.initial_assignment


def test_hard_trace_consistency():
    w = _world()
    o = Oracle(w)
    ev = w.sample_hard_chain(24, "tr")
    trace = o.hard_trace(ev)
    assert len(trace) == len(ev) + 1
    assert trace[0] == w.initial_assignment
    for t in range(len(ev) + 1):
        assert o.hard_assignment(ev, t=t) == trace[t]
    for t in range(len(ev)):
        assert Oracle._apply(trace[t], ev[t]) == trace[t + 1]


def test_cycle_semantics_and_order():
    w = _world()
    o = Oracle(w)
    g = w.agents
    init = w.initial_assignment
    a = o.hard_assignment([Event("cycle_roles", (g[0], g[1], g[2]))])
    assert a[g[0]] == init[g[2]]
    assert a[g[1]] == init[g[0]]
    assert a[g[2]] == init[g[1]]
    for ag in g[3:]:
        assert a[ag] == init[ag]
    assert o.hard_assignment([Event("cycle_roles", (g[0], g[1], g[2]))] * 3) == init


def test_hard_cycles_have_min_length():
    # R4-iii: no degenerate 2-cycles (which would equal a transposition)
    for k in (3, 4, 5):
        w = _world(k=k)
        for e in w.sample_hard_chain(2000, "cyc"):
            if e.kind == "cycle_roles":
                assert len(e.args) >= w.config.min_cycle_len >= 3


# --- auxiliary operator-worlds: disjoint symbol sets (R2) ---------------------------------

def test_namespaced_worlds_have_disjoint_symbols():
    target = World(WorldConfig(seed=0, id_namespace=""))
    aux = World(WorldConfig(seed=0, id_namespace="aux1_"))
    assert set(target.agents).isdisjoint(aux.agents)
    assert set(target.roles).isdisjoint(aux.roles)
    assert set(target.entities).isdisjoint(aux.entities)
    # ... yet the operator algebra is identical, so an operator learned on aux applies to target
    o_t, o_a = Oracle(target), Oracle(aux)
    assert sorted(o_t.hard_assignment([]).values()) == sorted(target.roles)
    assert sorted(o_a.hard_assignment([]).values()) == sorted(aux.roles)


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
