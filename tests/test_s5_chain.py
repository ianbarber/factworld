"""s5_chain task validity: deterministic generation, no-wrap gate, explicit rendering, the
distinct_path gate (echo/fixed-hop floors at exactly 0), and the v4 state-referencing events —
what they force about the algorithm, the floor they create, and the v3 defect they answer."""
import hashlib
import math
from collections import Counter

import pytest

from factworld.tasks import CANONICAL, RETIRED, generate, spec_for
from factworld.validity import _A0_FACT_RE, _A0_QUERY_RE, a0_events

SCORED = "s5_chain_v4"

# Frozen streams: same (spec, split, length, idx) -> identical example, forever. The retired
# and pre-existing entries are the values HEAD produced before the v4 construct existed, so
# they pin frozen-spec immutability across the change as well as after it.
PINNED = {
    "s5_chain_v1": {32: "670fa4e06d1ac93ae44f3dbba59ef02c",
                    64: "e25b96d1ac5c7e805d8709c2e64f70ff"},
    "s5_chain_v2": {32: "1b1a905e12a5c4975da1006d128f5a84",
                    64: "e7e81618863620cea79e054019e43874",
                    96: "aff5d4cc33b3fe83ffa9871f4f1ac702"},
    "s5_chain_v3": {32: "f2db10f109f27d2db1431411ade8c0bf",
                    64: "4aca0c3f4e251bfcf139ef62f4a56695",
                    96: "e78eee97eb89576fa5853ae03db586d9"},
    "s5_chain_local_v1": {4: "bd560c3096aeed7a69c9c9a88046c7bf",
                          8: "61be7642604c3754a801e935788d1007"},
    "s5_chain_local_v2": {4: "014ea7e49803dbf0c89f0f50dae26036",
                          8: "ede9e1acb1de40336ec6c561b6492c44"},
    "s5_chain_local_v2_path": {4: "82bc30f2bf327ec373c8419a69bc6aa3",
                               8: "b1405e787f3647c9c8d878b3db107f64"},
    "s5_chain_typed_v1": {4: "8f5931113e8abb36d7476a39ee4c7323",
                          8: "dc23d406b1aba7af2d417cef1449d237"},
    "s5_chain_v4": {32: "069828f7023f04da90c747bdcd7df394",
                    64: "bc3d5893a68e7371b3bd04ab5ccaa1b3",
                    96: "558f57d3a817ec3e1f638bd628af82dd"},
    "s5_chain_local_v4": {16: "870e74d35f8c8c1b1b47b88a35a0370b",
                          24: "439029840c42cee9f0d79560f10cc4fd"},
    "s5_chain_local_v4_path": {16: "0dab49252ffb57820fc3eca634f5d9c3",
                               24: "85764a87ca0ae32650f98173177619bf"},
}


def _hash(examples) -> str:
    return hashlib.sha256(
        "\n".join(f"{e.prompt}\t{e.answer}" for e in examples).encode()).hexdigest()[:32]


def backward_walk(prompt):
    """Answer the query by pushing ONE symbol backward through the event list.

    f' = f∘(a b) for a swap of a and b, so f_L = f_0∘σ_1∘…∘σ_L and
    f_L(x) = f_0(σ_1(…σ_L(x))): carry a single token, never form the map. Returns None
    when an event's operands are not readable off the sentence — the only state a
    referenced operand needs is the very map this walk exists to avoid building.
    """
    m = _A0_QUERY_RE.search(prompt)
    nxt0 = dict(_A0_FACT_RE.findall(prompt))
    events = a0_events(prompt)
    node = m.group(2)
    for _ in range(m.group(1).count("a0 of")):
        y = node
        for kind, args in reversed(events):
            if kind == "ref":
                return None
            if kind == "swap":
                a, b = args
                y = b if y == a else a if y == b else y
            else:
                a, b, c = args
                y = b if y == a else c if y == b else a if y == c else y
        node = nxt0[y]
    return f"{node}."


def test_generation_deterministic():
    spec = spec_for(SCORED)
    a = generate(spec, "test", n=3, length=32)
    b = generate(spec, "test", n=3, length=32)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.answer for x in a] == [x.answer for x in b]
    assert [x.meta for x in a] == [x.meta for x in b]


def test_no_wrap_gate():
    spec = spec_for(SCORED).scaled(chain_depth=spec_for(SCORED).k)
    with pytest.raises(ValueError, match="wraps"):
        generate(spec, "test", n=1, length=8)


def test_explicit_value_update_rendering():
    spec = spec_for(SCORED)
    ex = generate(spec, "test", n=1, length=32)[0]
    assert "swaps the values of" in ex.prompt and "cycles a0 simultaneously" in ex.prompt
    assert "old a0" in ex.prompt
    assert "the a0 of the agent whose a0 is currently " in ex.prompt
    assert f"({spec.chain_depth} hops)" in ex.prompt


def test_path_consistency():
    """The gold answer is the last element of the stored path."""
    spec = spec_for(SCORED)
    for ex in generate(spec, "test", n=10, length=32):
        assert ex.answer == f"{ex.meta['path'][-1]}."
        assert len(ex.meta["path"]) == ex.meta["depth"] + 1


def test_distinct_path_gate():
    """Every query path visits depth+1 DISTINCT agents, so the degenerate echo strategy
    (answer the queried agent) and every fixed-hop heuristic score exactly 0 and item
    difficulty is uniform. The retired v2 stream fails this (echo floor 0.16-0.32)."""
    spec = spec_for(SCORED)
    for L in spec.eval_lengths:
        for ex in generate(spec, "test", n=25, length=L):
            path = ex.meta["path"]
            assert len(set(path)) == len(path) == ex.meta["depth"] + 1
            assert path[-1] != ex.meta["start"]
            for hop in range(spec.chain_depth):          # every FIXED hop, not just echo
                assert path[hop] != path[-1]


def test_gate_leaves_the_item_distribution_alone():
    """The resampling loop conditions on the final map having a cycle longer than the depth
    — 0.68 of first draws at depth/k = 1/2 — and on nothing a prompt shows: gold and start
    are uniform over all k agents to within the finite-sample KL bias, the same as the
    ungated sampler."""
    spec = spec_for(SCORED)
    n, L = 1000, 64
    for gated in (spec, spec.scaled(name="ungated_probe", distinct_path=False)):
        exs = generate(gated, "test", n=n, length=L)
        operands, kinds = Counter(), Counter()
        for e in exs:
            for kind, args in a0_events(e.prompt):
                kinds[kind] += 1
                operands.update(args)
        for field, total in ((Counter(e.answer for e in exs), n),
                             (Counter(e.meta["start"] for e in exs), n),
                             (operands, sum(operands.values()))):
            assert len(field) == spec.k                  # every agent occurs
            kl = sum((c / total) * math.log((c / total) * spec.k) for c in field.values())
            assert kl - (spec.k - 1) / (2 * total) < 0.02   # the suite's balance threshold
        # the event mix is the one drawn: rate, then an even split of the remainder
        assert abs(kinds["ref"] / (n * L) - spec.conditional_rate) < 0.01
        assert abs(kinds["swap"] - kinds["cycle"]) / (n * L) < 0.01


def test_v2_stream_admits_echo_items():
    """Defect documentation: the retired v2 stream contains items whose gold equals
    the queried start agent (the echo floor that motivated the v3 gate)."""
    spec = RETIRED["s5_chain_v2"]
    exs = generate(spec, "test", n=25, length=32)
    assert any(ex.meta["path"][-1] == ex.meta["start"] for ex in exs)


def test_v3_is_answered_by_one_symbol_pushed_backward():
    """Defect documentation, the reason v3 is retired.

    Its events permute the DOMAIN of the pointer map, so the whole task is a single symbol
    walked backward through the event list and then looked up in the STATED initial map:
    exact, at every length, carrying log2(k) bits rather than the log2(k!) of the
    permutation. The walk needs the event list in reverse, which an attention model over
    the full context can do and a streaming recurrent model cannot.
    """
    for name in ("s5_chain_v1", "s5_chain_v2", "s5_chain_v3"):
        spec = RETIRED[name]
        for L in spec.eval_lengths:
            exs = generate(spec, "test", n=50, length=L)
            assert all(backward_walk(e.prompt) == e.answer for e in exs), (name, L)
    local = spec_for("s5_chain_local_v2")
    assert all(backward_walk(e.prompt) == e.answer
               for e in generate(local, "test", n=50, length=8))


def test_v4_admits_no_backward_walk():
    """A referenced operand has no identity until the map has been evaluated forward to it,
    so the backward walk cannot start on ANY scored item."""
    spec = spec_for(SCORED)
    for L in spec.eval_lengths:
        assert all(backward_walk(e.prompt) is None
                   for e in generate(spec, "test", n=50, length=L))


def test_forced_forward_prefix_reaches_the_end_of_the_stream():
    """How much of the stream must be evaluated forward: the position of the LAST referenced
    event. At rate 0.25 it sits at 0.909 / 0.953 / 0.969 of the stream at L=32/64/96 (n=2000
    per cell), and every item carries at least one reference. Doubling the rate buys 0.059 of
    that at the shortest length and 0.021 at the longest, which is what makes 0.25 the
    setting rather than 0.5."""
    spec = spec_for(SCORED)
    n = 2000
    for L, want in ((32, 0.90), (64, 0.95), (96, 0.96)):
        exs = generate(spec, "test", n=n, length=L)
        assert all(e.meta["ref_positions"] for e in exs)
        frac = sum(e.meta["ref_positions"][-1] + 1 for e in exs) / n / L
        assert frac > want, (L, frac)
        dense = generate(spec.scaled(name="rate_probe", conditional_rate=0.5),
                         "test", n=n, length=L)
        dense_frac = sum(e.meta["ref_positions"][-1] + 1 for e in dense) / n / L
        assert dense_frac - frac < 0.07, (L, frac, dense_frac)


def test_initial_ref_adversary_sits_at_chance_on_the_scored_stream():
    """The new shallow policy the construct creates: resolve every state reference against
    the STATED INITIAL map, which recovers an ordinary event list up front and hands the
    whole task back to the backward walk. It is exact on unconditional events and on any
    reference the map has not yet drifted past, so it has to be measured, not assumed.

    At n=8000 it reads 0.039 / 0.028 / 0.026 / 0.027 at L=32/64/96/128 against a chance of
    1/31 = 0.032 (the answer space is the non-start agents) — 1.23x chance at the shortest
    scored length, where four events of drift is all a reference has to survive, and at or
    under chance from L=64 on. The bound here is loose enough that the 2sd sampling width at
    this n (0.25x chance) cannot reach it: what the construct has to buy is that no shallow
    policy comes near DOUBLE chance, and every registered row has to clear that bound, not
    just this one.
    """
    from factworld.validity import S5_CHAIN_ROWS, operative_floor, s5_chain_floors

    spec = spec_for(SCORED)
    for L in spec.eval_lengths:
        f = s5_chain_floors(generate(spec, "test", n=2000, length=L), spec.k)
        assert set(f) == set(S5_CHAIN_ROWS), (L, f)
        assert f["echo"] == 0.0
        assert f["initial_ref_resolution"] <= 1.75 / spec.k, (L, f)
        assert f["initial_map_chase"] <= 1.75 / spec.k, (L, f)
        assert operative_floor(f) <= 1.75 / spec.k, (L, f)


def test_backward_hop_is_a_fixed_offset_and_never_sets_a_floor():
    """f_0^{-1}(start), a single BACKWARD lookup in the fact block that reads no event at all,
    is the j=k-1 member of the fixed-offset family f_0^j(start) — and that family PARTITIONS
    the answer.

    The stated map is a single k-cycle and distinct_path forces gold != start, so the k-1
    offsets hit each non-start agent exactly once per item and their accuracies sum to exactly
    1. Every member's null is therefore 1/(k-1) = 0.0323, not 1/k, and the max over any subset
    of them is a selection statistic: at n=5000 the expected family max is 0.0376.

    Measured on the scored stream at n=5000, k=32: the backhop reads 0.0402 / 0.0334 / 0.0304
    / 0.0290 at L=32/64/96/128, i.e. +3.2 sd on its own null at L=32 and at or under it from
    L=64. But at L=64/96/128 the LARGEST member of the family is an unregistered offset
    (0.0382 at j=13, 0.0382 at j=17, 0.0370 at j=17/23), above the backhop every time — which
    is what a max over an arbitrary subset of an exchangeable family looks like, and why the
    row is reported without being registered.
    """
    from factworld.validity import (
        S5_CHAIN_ADVERSARIES,
        operative_floor,
        s5_chain_floors,
        s5_chain_offset_accuracies,
    )

    spec = spec_for(SCORED)
    n, null = 5000, 1.0 / (spec.k - 1)
    for L in (32, 64, 96, 128):
        ex = generate(spec, "test", n=n, length=L)
        f = s5_chain_floors(ex, spec.k)
        off = s5_chain_offset_accuracies(ex, spec.k)
        assert set(off) == set(range(1, spec.k))
        assert abs(sum(off.values()) - 1.0) < 1e-9, (L, sum(off.values()))
        assert abs(f["uniform_non_start"] - null) < 1e-9
        assert abs(off[spec.chain_depth] - f["initial_map_chase"]) < 1e-9
        assert abs(off[spec.k - 1] - f["initial_map_backhop"]) < 1e-9
        # the floor is the max over the REGISTERED rows, which this row is not in — at L=32
        # that is what keeps the 0.0402 out of it
        assert "initial_map_backhop" not in S5_CHAIN_ADVERSARIES
        assert operative_floor(f) == max(v for name, v in f.items()
                                         if name in S5_CHAIN_ADVERSARIES), (L, f)
        if L == 32:
            assert f["initial_map_backhop"] > operative_floor(f)
        if L > 32:
            best = max(off, key=lambda j: off[j])
            assert best not in (spec.chain_depth, spec.k - 1), (L, best)
            assert off[best] > f["initial_map_backhop"], (L, off[best])


def test_the_shortest_scored_length_stays_scored_on_the_gate_margin():
    """L=32 carries the highest operative floor of the scored grid and stays scored.

    0.0398 (initial-ref resolution) against 0.0323-0.0334 at L=64/96/128 at n=5000. The
    decision does not rest on that 0.0064 gap: the initial-map rows belong to a partition, so
    comparing per-length maxima compares selection draws, and against the full 31-member
    family the L=32 maximum exceeds the longer lengths' by only 0.0020. What keeps L=32 is the
    gate margin — 0.0398 is 1.23x the 1/31 chance level against the 2x bound the suite gates
    on, so the shortest scored length is no closer to shallow-solvable than the rest of the
    grid.
    """
    from factworld.validity import (
        operative_floor,
        s5_chain_floors,
        s5_chain_offset_accuracies,
    )

    spec = spec_for(SCORED)
    n, lengths = 5000, (32, 64, 96, 128)
    items = {L: generate(spec, "test", n=n, length=L) for L in lengths}
    fl = {L: operative_floor(s5_chain_floors(items[L], spec.k)) for L in lengths}
    fam = {L: max(s5_chain_offset_accuracies(items[L], spec.k).values()) for L in lengths}

    assert fl[32] == max(fl.values())
    assert fl[32] < 2.0 / (spec.k - 1)                              # the gate the suite applies
    assert fl[32] < 1.3 / (spec.k - 1)                              # and well inside it
    # the registered-row gap overstates what the length grid buys: against the full family the
    # shortest length's advantage is a third of it
    assert fam[32] - max(fam[L] for L in (64, 96, 128)) < \
        0.5 * (fl[32] - max(fl[L] for L in (64, 96, 128)))


def test_ref_row_is_absent_where_no_item_carries_a_reference():
    """Without a reference to mis-resolve the policy IS the oracle, so the row is dropped
    rather than printed as a floor no arm can clear — as the chase row is where there are
    no events. The retired family's floors are therefore what they always were."""
    from factworld.validity import s5_chain_floors

    for name, L in (("s5_chain_v3", 96), ("s5_chain_local_v2", 8)):
        spec = spec_for(name)
        f = s5_chain_floors(generate(spec, "test", n=100, length=L), spec.k)
        assert "initial_ref_resolution" not in f
        assert set(f) == {"initial_map_chase", "initial_map_backhop", "echo",
                          "uniform_non_start", "uniform"}


def test_reference_rendering_round_trips_to_what_the_sentence_encodes():
    """The rendered sentence encodes (named slot, referenced VALUE) and the parser recovers
    exactly that. The second slot is not in the sentence: resolving the value against the
    stated initial map gives the wrong agent on items where the map has moved."""
    from factworld.render import Renderer

    spec = spec_for(SCORED)
    r = Renderer()
    moved = 0
    for ex in generate(spec, "test", n=50, length=64):
        nxt0 = dict(_A0_FACT_RE.findall(ex.prompt))
        # the reference clause states a value with no possessive, so it does not read as a
        # fact: the stated map is still exactly k entries
        assert len(nxt0) == spec.k
        inv0 = {v: a for a, v in nxt0.items()}
        sentences = [s + "." for s in ex.prompt.split(". ") if "whose a0 is currently" in s]
        assert len(sentences) == len(ex.meta["ref_positions"])
        for pos, sentence in zip(ex.meta["ref_positions"], sentences):
            rec = r.parse(sentence)
            assert rec["type"] == "event" and rec["event"].kind == "swap_a0_ref"
            named, value = rec["event"].args
            assert a0_events(sentence) == [("ref", (named, value))]
            assert sentence.startswith(f"s{pos} ")
            moved += int(inv0[value] != named and inv0[value] is not None)
    assert moved > 0


def test_frozen_streams_are_byte_identical():
    """Every registered and retired s5_chain spec, against hashes pinned before the v4
    construct existed. The conditional draw is short-circuited at rate 0.0, so a spec
    without references draws exactly the numbers it drew before the knob was added."""
    for name, per_len in PINNED.items():
        spec = spec_for(name)
        for L, want in per_len.items():
            assert _hash(generate(spec, "test", n=25, length=L)) == want, f"{name}@L{L}"


def test_registry_contract():
    """The WHOLE s5_chain family is retired and every version stays generable.

    The family is retired on a defect the shortcut fixes do not reach: on the published v3
    battery the top eleven models have zero pairwise separations at n=25, so the ranked cell
    orders by noise. So no version is scored, no version is in REPORTED, and the family
    contributes no ``kind="benchmark"`` spec — while every one of them still resolves, because
    the published cells must stay reproducible.
    """
    from factworld import tasks as TK

    spec = TK.spec_for(SCORED)
    assert SCORED not in TK.REPORTED and spec.kind == "retired"
    assert spec.distinct_path and spec.conditional_rate == 0.25
    assert spec.k == 32 and spec.chain_depth == 16 and spec.chain_depth < spec.k
    assert sum(1 for s in TK.CANONICAL.values()
               if s.family == "s5_chain" and s.kind == "benchmark") == 0
    for name in ("s5_chain_v1", "s5_chain_v2", "s5_chain_v3", "s5_chain_local_v1", SCORED):
        assert name not in TK.CANONICAL and name not in TK.REPORTED
        assert TK.RETIRED[name].kind == "retired"
        assert TK.spec_for(name) is TK.RETIRED[name]
    # v4 carries v3's knobs apart from the construct, the operating point it buys, and the
    # RNG-stream tag every new spec pins at introduction
    v3, v4 = TK.RETIRED["s5_chain_v3"], spec
    differing = {f for f in v3.__dataclass_fields__
                 if getattr(v3, f) != getattr(v4, f)}
    assert differing == {"name", "version", "k", "chain_depth", "conditional_rate"}


def test_local_arms_run_the_scored_construct():
    """The local family exists so the same algorithm is trainable from scratch, so it differs
    from the scored spec only in what a from-scratch operating point forces: breadth, depth,
    lengths, supervision, and the rate the floors demand at those lengths."""
    from factworld import tasks as TK

    scored = TK.spec_for(SCORED)   # retired 2026-08-12: the headline that did not
    #                                separate; the local arms still run its construct
    for name in ("s5_chain_local_v4", "s5_chain_local_v4_path"):
        arm = TK.CANONICAL[name]
        assert arm.kind == "experimental" and name not in TK.REPORTED
        # the arms are experimental and the construct they run is retired, so the field diff
        # against it carries ``kind``; that is the retirement and not a construct difference
        assert arm.conditional_rate == 0.5 and arm.distinct_path
        differing = {f for f in scored.__dataclass_fields__
                     if getattr(scored, f) != getattr(arm, f)}
        assert differing <= {"name", "kind", "k", "chain_depth", "conditional_rate",
                             "train_lengths", "eval_lengths", "event_trace", "worked_trace"}
    # the two arms are the supervision-density contrast and nothing else
    dense, path = TK.CANONICAL["s5_chain_local_v4"], TK.CANONICAL["s5_chain_local_v4_path"]
    assert {f for f in dense.__dataclass_fields__
            if getattr(dense, f) != getattr(path, f)} == {"name", "event_trace"}
    # against the v2 family: same k, depth and supervision; the construct and the lengths it
    # needs are what move (a reference is mis-resolvable only once the map has drifted)
    v2 = TK.spec_for("s5_chain_local_v2")
    assert {f for f in dense.__dataclass_fields__ if getattr(dense, f) != getattr(v2, f)} == \
        {"name", "version", "kind", "conditional_rate", "train_lengths", "eval_lengths"}


def test_local_arm_floors_are_set_by_the_pre_existing_adversaries():
    """At the registered local lengths the reference policy is weaker than the initial-map
    chase, so the local cells are read against a floor of the same kind as the v2 family's:
    chance at L=16 (the chase is 0.134, under the 1/(k-1) = 0.143 the offsets average to) and
    the chase at L=24 (0.156)."""
    from factworld.validity import operative_floor, s5_chain_floors

    spec = CANONICAL["s5_chain_local_v4"]
    for L in spec.eval_lengths:
        f = s5_chain_floors(generate(spec, "test", n=2000, length=L), spec.k)
        assert f["initial_ref_resolution"] < f["initial_map_chase"], (L, f)
        assert operative_floor(f) in (f["initial_map_chase"], f["uniform_non_start"]), (L, f)
    f16 = s5_chain_floors(generate(spec, "test", n=2000, length=16), spec.k)
    f24 = s5_chain_floors(generate(spec, "test", n=2000, length=24), spec.k)
    assert operative_floor(f16) == f16["uniform_non_start"]
    assert abs(f16["uniform_non_start"] - 1.0 / (spec.k - 1)) < 1e-9
    assert operative_floor(f24) == f24["initial_map_chase"] > f24["uniform_non_start"]


def test_typed_values_rejects_state_references():
    spec = spec_for("s5_chain_typed_v1").scaled(conditional_rate=0.25)
    with pytest.raises(ValueError, match="conditional_rate"):
        generate(spec, "test", n=1, length=4)


def test_committed_answer_extraction():
    """A reasoning endpoint that spills working into the visible completion commits to
    its single-token final line (real sonnet xhigh shapes); map-dump tails, truncated
    working, and every single-line emission commit to nothing and pass through."""
    from factworld.tasks import committed_answer, score_relaxed

    spill = ("g15=g7\n\nNow tracing 8 hops from g11:\n1. g11 → g4\n2. g4 → g13\n"
             "3. g13 → g1\n\n**g10**")
    assert committed_answer(spill) == "g10"
    assert score_relaxed(committed_answer(spill), "g10.") == 1
    labeled = "working...\n1. g12 → g8\n\n**Answer: g11**"
    assert committed_answer(labeled) == "g11"
    # a map-dump last line has two content tokens: no commitment, scored as-is (0)
    dump = "working...\n- g14→g7\n- g15→g12"
    assert committed_answer(dump) == dump
    assert score_relaxed(committed_answer(dump), "g12.") == 0
    # truncated working: no single-token line, no credit
    cut = "tracking the map:\n**Following the 8-h"
    assert committed_answer(cut) == cut
    # single-line: clean answers are score-invariant; multi-token streams commit nothing
    assert score_relaxed(committed_answer("g5."), "g5.") == 1
    assert committed_answer("g3. <eos> g7 g0") == "g3. <eos> g7 g0"
    # single-line hop path ending in the bolded answer (real sonnet L128 shape)
    hop = "g5 → g8 → g0 → g2 → g14 → g15 → g6 → g13 → **g7**"
    assert committed_answer(hop) == "g7"
    # prose commitment (real muse shape): last emphasized span carries the answer
    prose = "trace...\nso 8 applications of `a0` starting from `g14` ends at **g15** ."
    assert committed_answer(prose) == "g15"
    # lone token inside a trailing code fence (real muse shape)
    fenced = "g1→g13→g8→g14\n\n```\ng14\n```"
    assert committed_answer(fenced) == "g14"
    # copula lead-in
    assert committed_answer("working...\nThe answer is g10.") == "g10"


def test_trace_mode_scoring_cuts_at_eos():
    """A local model emits scratchpad, answer, <eos>, then budget-filling junk.
    evaluate_task must cut at <eos> so last_n scores the committed answer, not
    the junk tail (the pre-fix behavior read every trace-mode sweep as chance)."""
    from factworld.runner import evaluate_task

    spec = spec_for("s5_chain_local_v2")
    exs = generate(spec, "test", n=3, length=4)

    class OracleWithJunk:
        name = "oracle-junk"
        def generate(self, prompts, max_new_tokens, stop_at=None):
            return [f"{e.meta['trace']} {e.answer} <eos> g0 g1 g2" for e in exs]

    res = evaluate_task(OracleWithJunk(), spec, split="test", n=3, length=4)
    assert res["metrics"]["last_n"]["overall"] == 1.0


def test_typed_values_answer_a_different_token_type():
    """The typed-value ablation: slots are agents, values are roles, so no token is ambiguous
    between the two positions — the one structural property s5 has and s5_chain does not."""
    from factworld.render import classify

    spec = spec_for("s5_chain_typed_v1")
    for L in spec.eval_lengths:
        for ex in generate(spec, "test", n=25, length=L):
            assert classify(ex.answer.rstrip(".")) == "r"
            assert classify(ex.meta["start"]) == "g"
            assert ex.meta["depth"] == 1 and ex.meta["typed_values"] is True
            assert ex.meta["path"] == [ex.meta["start"], ex.answer.rstrip(".")]
            # facts map agents to roles; the event stream still names only agents
            assert f"{ex.meta['start']}'s a0 is " in ex.prompt
            assert "swaps the values of" in ex.prompt or "cycles a0 simultaneously" in ex.prompt


def test_typed_values_echo_floor_is_structurally_zero():
    """distinct_path buys echo=0 in the untyped task; typing buys it by construction, so the
    gate is neither applied nor needed here."""
    from factworld.validity import s5_chain_floors, s5_chain_offset_accuracies

    spec = spec_for("s5_chain_typed_v1")
    assert spec.distinct_path is False
    for L in spec.eval_lengths:
        items = generate(spec, "test", n=200, length=L)
        floors = s5_chain_floors(items, spec.k)
        assert floors["echo"] == 0.0
        # the stated map runs agents -> roles, so its inverse is not defined at an agent:
        # the backward hop has no answer here rather than a wrong one
        assert floors["initial_map_backhop"] == 0.0
        # and the fixed-offset family does not exist here at all: the walk leaves the map
        # after one hop, so there is nothing to partition and no 1/(k-1) null
        assert s5_chain_offset_accuracies(items, spec.k) == {}
        assert floors["uniform"] == 1.0 / spec.k
        # answer space is the roles, so the query start is not a candidate: no 1/(k-1) discount
        assert floors["uniform_non_start"] == 1.0 / spec.k


def test_typed_values_is_depth_one_only():
    spec = spec_for("s5_chain_typed_v1").scaled(chain_depth=2)
    with pytest.raises(ValueError, match="depth-1 construct"):
        generate(spec, "test", n=1, length=4)


def test_typed_and_untyped_initial_maps_have_different_structure():
    """The second documented difference, entailed by the type split rather than chosen: the
    untyped initial map is a single k-cycle (no fixed points, every agent reachable from every
    other), the typed one a uniform agent->role bijection, for which cycles are undefined."""
    import re

    fact = re.compile(r"\b(g\d+)'s a0 is ([a-z]+\d+)\.")
    untyped = spec_for("s5_chain_local_v2").scaled(chain_depth=1)
    for ex in generate(untyped, "test", n=10, length=4):
        nxt = dict(fact.findall(ex.prompt))
        assert set(nxt) == set(nxt.values())                # endo-map on the agent set
        node, seen = next(iter(nxt)), 0
        while True:                                         # one cycle covering every slot
            node, seen = nxt[node], seen + 1
            if node == next(iter(nxt)):
                break
        assert seen == len(nxt) == untyped.k

    typed = spec_for("s5_chain_typed_v1")
    for ex in generate(typed, "test", n=10, length=4):
        m = dict(fact.findall(ex.prompt))
        assert len(m) == typed.k and len(set(m.values())) == typed.k
        assert not (set(m) & set(m.values()))               # disjoint pools: no cycles at all


def test_typed_and_untyped_arms_differ_in_exactly_the_documented_fields():
    """The contrast is only readable if the differences are enumerated.

    Whole-dataclass comparison rather than a hand-listed subset: a field added later that
    silently diverges between the arms would slip past a subset check and confound the
    ablation. Three fields may differ, and each is documented in _ex_s5_chain_typed:
    ``name`` (it keys the RNG stream), ``typed_values`` (the intended change), and
    ``distinct_path`` (typing makes the echo adversary type-invalid, so the gate is neither
    applied nor needed — at the cost of a different chance level, 1/k against 1/(k-1)).
    """
    import dataclasses

    typed = dataclasses.asdict(spec_for("s5_chain_typed_v1"))
    untyped = dataclasses.asdict(spec_for("s5_chain_local_v2").scaled(chain_depth=1))
    assert set(typed) == set(untyped)
    differing = {f for f in typed if typed[f] != untyped[f]}
    assert differing == {"name", "typed_values", "distinct_path"}
    assert typed["typed_values"] and not untyped["typed_values"]
    assert untyped["distinct_path"] and not typed["distinct_path"]


def test_typed_values_supervision_shapes():
    """event_trace dumps the whole agent->role map per event; start_trace emits the queried
    slot's current value per event and builds the interleaved training prompt."""
    spec = spec_for("s5_chain_typed_v1")
    L = 4
    ex = generate(spec, "test", n=1, length=L)[0]
    toks = ex.meta["trace"].split()
    assert len(toks) == L * spec.k + spec.chain_depth
    assert toks[-1] == ex.meta["start"]

    st = generate(spec.scaled(start_trace=True), "test", n=1, length=L)[0]
    trace = st.meta["trace"].split()
    assert len(trace) == L + 1
    assert trace[L - 1] == st.answer.rstrip(".")          # last checkpoint IS the depth-1 gold
    assert st.meta["interleaved_prompt"].endswith(st.prompt.split("what is")[-1].strip())


def test_typed_values_does_not_perturb_the_untyped_stream():
    """Frozen-spec immutability: the typed builder is a separate function reached only via
    spec.typed_values, so the registered s5_chain streams are untouched."""
    spec = spec_for("s5_chain_local_v2")
    a = generate(spec, "test", n=5, length=4)
    generate(spec_for("s5_chain_typed_v1"), "test", n=5, length=4)
    b = generate(spec, "test", n=5, length=4)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.answer for x in a] == [x.answer for x in b]


def test_event_trace_checkpoints():
    """local_v2 dense supervision: the trace carries one full a0-map checkpoint
    (k agents) per event, then the query path prefix."""
    spec = spec_for("s5_chain_local_v2")
    L = 4
    ex = generate(spec, "test", n=1, length=L)[0]
    toks = ex.meta["trace"].split()
    assert len(toks) == L * spec.k + spec.chain_depth
    # the trace tail is the query path prefix and chains to the gold answer
    assert toks[-spec.chain_depth] == ex.meta["start"]
    assert toks[-1] == ex.meta["path"][-2]
