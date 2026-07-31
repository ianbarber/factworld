"""M5 — validity-gate regression lock + a sanity check that the leakage detector actually detects.

Runs with zero dependencies:  python3 tests/test_validity.py
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.baselines import answer_kl, naive_bayes_accuracy  # noqa: E402
from factworld.config import WorldConfig  # noqa: E402
from factworld.eval import EvalItem, hard_suite  # noqa: E402
from factworld.oracle import Oracle  # noqa: E402
from factworld.render import Renderer  # noqa: E402
from factworld.validity import run_gate  # noqa: E402
from factworld.world import World  # noqa: E402


def test_gate_passes_on_the_default_instrument():
    r = run_gate(n_dist=1200, n_leak=300)
    assert r["passed"], r["checks"]
    h = r["families"]["state_hard"]                      # hard rung must require composition
    assert h["naive_bayes"] <= h["floor"] + 0.10
    assert h["recency"] <= h["floor"] + 0.10
    assert h["identity_guess"] <= h["floor"] + 0.10      # no structural small-t shortcut
    assert all(f["kl_excess"] < 0.02 for f in r["families"].values())  # balanced


def test_hard_asof_t_samples_past_the_mixing_threshold():
    w = World(WorldConfig(seed=0))
    items = hard_suite(w, Oracle(w), (16, 32), 100, seed="x", as_of_t=True)
    assert items and all(it.t >= min(w.k, it.length) for it in items)


def test_naive_bayes_detects_a_planted_leak():
    # gold is a deterministic function of the entity, entities reused across the split -> recoverable
    items = [EvalItem("recall", f"v{i % 20}", entity=f"e{i % 20}", attribute="a0") for i in range(400)]
    assert naive_bayes_accuracy(items, Renderer())["recall"] > 0.9


def test_naive_bayes_near_floor_on_random_labels():
    rng = random.Random(0)
    items = [EvalItem("recall", f"v{rng.randrange(20)}", entity=f"e{i}", attribute="a0") for i in range(400)]
    assert naive_bayes_accuracy(items, Renderer())["recall"] < 0.15   # ~ 1/20 floor, no leak


def test_answer_kl_zero_on_uniform_high_on_skew():
    space = [f"v{i}" for i in range(5)]
    assert answer_kl([f"v{i % 5}" for i in range(1000)], space) < 0.01
    assert answer_kl(["v0"] * 900 + ["v1"] * 100, space) > 0.5


def test_a0_event_reader_covers_both_grammars_in_stream_order():
    """The pointer-map adversaries read the rendered events, so the reader has to return
    every event, in the order it was rendered, in the canonical AND the compact grammar."""
    from factworld.render import Renderer
    from factworld.tasks import CANONICAL, generate
    from factworld.validity import a0_events
    from factworld.world import Event

    spec = CANONICAL["s5_chain_v4"]
    for ex in generate(spec, "test", n=5, length=32):
        events = a0_events(ex.prompt)
        assert len(events) == 32
        assert [i for i, (k, _) in enumerate(events) if k == "ref"] == \
            list(ex.meta["ref_positions"])
    r = Renderer()
    trio = [Event("swap_a0", ("g1", "g2")), Event("swap_a0_ref", ("g3", "g4")),
            Event("cycle_a0", ("g5", "g6", "g7"))]
    canonical = " ".join(r.render_event(e, step=f"s{i}") for i, e in enumerate(trio))
    compact = ("s0 swaps a0: g1 and g2. s1 swaps a0: g3 and whose a0 is g4. "
               "s2 cycles a0: g5 -> g6 -> g7.")
    want = [("swap", ("g1", "g2")), ("ref", ("g3", "g4")), ("cycle", ("g5", "g6", "g7"))]
    assert a0_events(canonical) == want
    assert a0_events(compact) == want


def test_initial_ref_adversary_is_exact_where_the_map_has_not_moved():
    """It is a real policy, not a random guess: resolving references against the initial map
    is CORRECT while the map still agrees with the facts, which is why it has to be measured
    on the scored stream rather than assumed to sit at chance."""
    from factworld.tasks import CANONICAL, generate
    from factworld.validity import s5_chain_ref_pred

    spec = CANONICAL["s5_chain_v4"].scaled(name="ref_probe_L1", eval_lengths=(1,),
                                           conditional_rate=1.0, chain_depth=1)
    exs = generate(spec, "test", n=200, length=1)
    assert all(s5_chain_ref_pred(e.prompt) == e.answer for e in exs)


def test_backward_hop_reads_the_stated_map_and_nothing_else():
    """f_0^{-1}(start) is recovered from the fact block alone, so it is defined on a prompt
    with no events at all and is unchanged by the event grammar the stream is rendered in.
    (Reported as a diagnostic; it is not a registered adversary — see the test below.)"""
    from factworld.tasks import CANONICAL, generate
    from factworld.validity import _A0_FACT_RE, s5_chain_shallow_preds

    spec = CANONICAL["s5_chain_v4"]
    plain = generate(spec, "test", n=20, length=32)
    compact = generate(spec.scaled(compact_events=True), "test", n=20, length=32)
    for exs in (plain, compact):
        for ex in exs:
            inv0 = {v: a for a, v in _A0_FACT_RE.findall(ex.prompt)}
            assert s5_chain_shallow_preds(ex.prompt)["initial_map_backhop"] == \
                f"{inv0[ex.meta['start']]}."
    assert [s5_chain_shallow_preds(e.prompt)["initial_map_backhop"] for e in plain] == \
        [s5_chain_shallow_preds(e.prompt)["initial_map_backhop"] for e in compact]


def test_no_backhop_row_where_there_is_no_event_stream():
    """With no events the stated map IS the final map, so one reverse lookup is the ORACLE
    exactly when the depth is k-1 — chain_v2's longer eval length. That is the cheap-direction
    property the chain staircase prices at k=2d+1, not a floor, so the row is dropped for the
    same reason the chase row is."""
    from factworld.tasks import CANONICAL, generate
    from factworld.validity import s5_chain_floors, s5_chain_shallow_preds

    chain = CANONICAL["chain_v2"]
    at_depth = {}
    for d in chain.eval_lengths:
        exs = generate(chain, "test", n=50, length=d)
        at_depth[d] = sum(
            s5_chain_shallow_preds(e.prompt)["initial_map_backhop"] == e.answer
            for e in exs) / len(exs)
        f = s5_chain_floors(exs, chain.k, has_events=False)
        assert "initial_map_backhop" not in f and "initial_map_chase" not in f
    assert at_depth == {4: 0.0, 5: 1.0}                  # k=6: exact at depth k-1, never else


def test_backhop_is_reported_but_not_registered_as_an_adversary():
    """The fixed-offset policies f_0^j(start) partition the answer, so every member's null is
    uniform-over-non-start and a max over a registered subset of them measures selection. The
    backhop is one such member and is named by nothing in the task, so it is measured and
    reported while ``operative_floor`` ignores it however large it reads. The chase stays
    registered because the query names it: it is the query's own computation against the
    stated map."""
    from factworld.validity import (
        S5_CHAIN_ADVERSARIES,
        S5_CHAIN_CHANCE_ROWS,
        S5_CHAIN_OFFSET_ROWS,
        S5_CHAIN_ROWS,
        operative_floor,
    )

    assert "initial_map_backhop" in S5_CHAIN_ROWS
    assert "initial_map_backhop" not in S5_CHAIN_ADVERSARIES
    assert "initial_map_chase" in S5_CHAIN_ADVERSARIES
    assert set(S5_CHAIN_ADVERSARIES) < set(S5_CHAIN_ROWS)
    assert set(S5_CHAIN_OFFSET_ROWS) <= set(S5_CHAIN_ROWS)
    assert set(S5_CHAIN_CHANCE_ROWS) <= set(S5_CHAIN_ADVERSARIES)
    # uniform_non_start is always registered, so the floor can never fall below the family's
    # common expectation — and an unregistered member never raises it
    floors = {"initial_map_chase": 0.10, "initial_map_backhop": 0.99, "echo": 0.0,
              "uniform_non_start": 0.20, "uniform": 1 / 6}
    assert operative_floor(floors) == 0.20


def test_the_truncation_family_and_the_pin_chain_are_registered_s5_bind_floors():
    """Both families that could set the mutual-reference floor are registered.

    The truncation family simulates the task exactly over T = f*L of its L events — the last T
    (window_f, carrying the stated maps in at the cut) or the first T (prefix_f, reading the
    true maps out at the cut). BOTH halves are registered at every budget: at a given f they pay
    the same T events, so registering one and not the other prices the same purchase differently
    at the two ends of the stream. Each half is monotone in its cut — cut 1.0 is the oracle by
    construction — so its max is always the largest cut and never a selection statistic, which
    is what separates it from the pointer-map fixed-offset family. The pin chain is the
    zero-state policy that reads the give -> swap reference pair, and it is what the window
    rows were tracking before ``TaskSpec.no_pin`` closed that channel.

    With the channel open the window rows sat 2-3x the informed chance and did not decay with
    length; with it closed every registered row lands at chance on the scored grid, which is
    the property the construct claims.
    """
    from factworld.tasks import CANONICAL, generate
    from factworld.validity import (
        S5_BIND_ADVERSARIES,
        S5_BIND_CHANCE_ROWS,
        S5_BIND_COUPLED_ONLY_ROWS,
        S5_BIND_ROWS,
        S5_BIND_TRUNCATION_ROWS,
        S5_BIND_WINDOWS,
        s5_bind_floors,
        s5_bind_operative_floor,
        s5_bind_pin_density,
    )

    windows = tuple(r for r in S5_BIND_ROWS if r.startswith(("window_", "prefix_")))
    assert windows == S5_BIND_TRUNCATION_ROWS
    assert len(windows) == 2 * len(S5_BIND_WINDOWS)
    assert set(windows) <= set(S5_BIND_ADVERSARIES)
    assert "pin_chain" in S5_BIND_ADVERSARIES
    assert set(S5_BIND_CHANCE_ROWS) <= set(S5_BIND_ADVERSARIES)
    assert set(S5_BIND_COUPLED_ONLY_ROWS) <= set(S5_BIND_ROWS)
    spec = CANONICAL["s5_bind_v2"]
    exs = generate(spec, "test", n=400, length=spec.eval_lengths[0])
    fl = s5_bind_floors(exs, spec.k)
    op = s5_bind_operative_floor(fl)
    assert s5_bind_pin_density(exs) == 0.0
    assert op >= fl["uniform_non_initial"]                # the chance row is registered
    assert op <= 1.6 * fl["uniform_non_initial"]
    assert fl["pin_chain"] <= fl["uniform_non_initial"]
    # both halves of the truncation family land there too, which is the property the query
    # gates buy: the object's resolving write makes the head load-bearing, q_tail the tail
    for row in S5_BIND_TRUNCATION_ROWS:
        assert fl[row] <= 1.6 * fl["uniform_non_initial"], f"{row}: {fl[row]:.4f}"
    # the same cell with the channel open: the window rows lift far off chance and the
    # zero-state policy reads more than twice it
    open_fl = s5_bind_floors(generate(spec.scaled(no_pin=False), "test", n=400,
                                      length=spec.eval_lengths[0]), spec.k)
    assert open_fl["pin_chain"] >= 2 * fl["uniform_non_initial"]
    assert s5_bind_operative_floor(open_fl) > 1.6 * fl["uniform_non_initial"]


def test_the_operative_floor_resolves_the_family_from_the_rows():
    """The mis-report a fixed default produced: the two adversary families share exactly one
    row name, so the pointer-map default found ``uniform`` in a mutual-reference floor dict and
    returned 1/k — a floor BELOW that family's own informed chance 1/(k-1). Dispatch is now on
    the rows only one family can emit, and an unresolvable dict raises."""
    from factworld.tasks import CANONICAL, generate
    from factworld.validity import (
        S5_BIND_ADVERSARIES,
        S5_CHAIN_ADVERSARIES,
        operative_floor,
        registered_for,
        s5_bind_floors,
        s5_bind_operative_floor,
    )

    spec = CANONICAL["s5_bind_v2"]
    fl = s5_bind_floors(generate(spec, "test", n=100, length=spec.eval_lengths[0]), spec.k)
    assert set(S5_CHAIN_ADVERSARIES) & set(fl) == {"uniform"}
    assert registered_for(fl) is S5_BIND_ADVERSARIES
    assert operative_floor(fl) == s5_bind_operative_floor(fl) >= fl["uniform_non_initial"]
    chain = {"initial_map_chase": 0.10, "echo": 0.0, "uniform_non_start": 0.20, "uniform": 1 / 6}
    assert registered_for(chain) is S5_CHAIN_ADVERSARIES
    assert operative_floor(chain) == 0.20
    for ambiguous in ({"uniform": 0.1}, {}, {"echo": 0.0, "pin_chain": 0.1}):
        try:
            registered_for(ambiguous)
        except ValueError:
            continue
        raise AssertionError(f"{ambiguous}: resolved a floor dict it cannot tell apart")
    assert operative_floor({}) is None                   # no cell, no floor — the prior contract


def test_s5_bind_reader_sees_only_what_the_prompt_says():
    """Every mutual-reference policy reads the rendered sentences, so the reader has to
    recover the two stated maps, the events in rendered order, and each event's temporal
    phrase — and replaying that reproduces the gold exactly, on both renderings."""
    from factworld.tasks import CANONICAL, generate
    from factworld.validity import _sb_answer, _sb_run, s5_bind_read

    for name in ("s5_bind_v2", "s5_bind_v2_state", "s5_bind_v2_bind", "s5_bind_v2_map"):
        spec = CANONICAL[name]
        for e in generate(spec, "test", n=10, length=64):
            read = s5_bind_read(e.prompt)
            assert len(read["events"]) == 64
            assert len(read["P0"]) == spec.k and len(read["B0"]) == spec.n_objects_active
            assert all(d for _k, _x, _y, d in read["events"]) == spec.coupled
            assert _sb_answer(read, _sb_run(read, "surface")) == e.answer, name
    assert s5_bind_read("what is a0 of g3? (1 hops)") is None


def test_every_registered_shortcut_reaches_the_suite_gate_column():
    """The drift the hardcoded tuple allowed: a row registered in factworld.validity never
    reached scripts/validate_suite.py's shallow-adversary column. The column is now derived,
    so registering one is enough."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import validate_suite  # noqa: E402

    from factworld.tasks import CANONICAL, generate
    from factworld.validity import (
        S5_CHAIN_ADVERSARIES,
        S5_CHAIN_CHANCE_ROWS,
        s5_chain_floors,
    )

    spec = CANONICAL["s5_chain_v4"]
    rows = s5_chain_floors(generate(spec, "test", n=50, length=32), spec.k)
    registered = {n for n in rows
                  if n in S5_CHAIN_ADVERSARIES and n not in S5_CHAIN_CHANCE_ROWS}
    assert registered <= set(validate_suite.S5_CHAIN_SHORTCUTS)
    assert set(validate_suite.S5_CHAIN_SHORTCUTS) <= set(S5_CHAIN_ADVERSARIES)
    assert not set(validate_suite.S5_CHAIN_SHORTCUTS) & set(S5_CHAIN_CHANCE_ROWS)
    assert "initial_map_backhop" not in validate_suite.S5_CHAIN_SHORTCUTS


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
