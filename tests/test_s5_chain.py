"""s5_chain task validity: deterministic generation, no-wrap gate, explicit rendering,
and the v3 distinct_path gate (echo/fixed-hop floors at exactly 0)."""
import pytest

from factworld.tasks import CANONICAL, RETIRED, generate


def test_generation_deterministic():
    spec = CANONICAL["s5_chain_v3"]
    a = generate(spec, "test", n=3, length=32)
    b = generate(spec, "test", n=3, length=32)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.answer for x in a] == [x.answer for x in b]
    assert [x.meta["path"] for x in a] == [x.meta["path"] for x in b]


def test_no_wrap_gate():
    spec = CANONICAL["s5_chain_v3"].scaled(chain_depth=16)
    with pytest.raises(ValueError, match="wraps"):
        generate(spec, "test", n=1, length=8)


def test_explicit_value_update_rendering():
    spec = CANONICAL["s5_chain_v3"]
    ex = generate(spec, "test", n=1, length=8)[0]
    assert "swaps the values of" in ex.prompt or "cycles a0 simultaneously" in ex.prompt
    assert "old a0" in ex.prompt or "swaps the values of" in ex.prompt
    assert "(8 hops)" in ex.prompt


def test_path_consistency():
    """The gold answer is the last element of the stored path."""
    spec = CANONICAL["s5_chain_v3"]
    for ex in generate(spec, "test", n=10, length=32):
        assert ex.answer == f"{ex.meta['path'][-1]}."
        assert len(ex.meta["path"]) == ex.meta["depth"] + 1


def test_distinct_path_gate():
    """v3 validity gate: every query path visits depth+1 DISTINCT agents, so the
    degenerate echo strategy (answer the queried agent) and every fixed-hop
    heuristic score exactly 0 and item difficulty is uniform. The retired v2
    stream fails this (its echo floor measured 0.16-0.32)."""
    spec = CANONICAL["s5_chain_v3"]
    for L in spec.eval_lengths:
        for ex in generate(spec, "test", n=25, length=L):
            path = ex.meta["path"]
            assert len(set(path)) == len(path) == ex.meta["depth"] + 1
            assert path[-1] != ex.meta["start"]


def test_v2_stream_admits_echo_items():
    """Defect documentation: the retired v2 stream contains items whose gold equals
    the queried start agent (the echo floor that motivated the v3 gate)."""
    spec = RETIRED["s5_chain_v2"]
    exs = generate(spec, "test", n=25, length=32)
    assert any(ex.meta["path"][-1] == ex.meta["start"] for ex in exs)


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

    spec = CANONICAL["s5_chain_local_v2"]
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

    spec = CANONICAL["s5_chain_typed_v1"]
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
    from factworld.validity import s5_chain_floors

    spec = CANONICAL["s5_chain_typed_v1"]
    assert spec.distinct_path is False
    for L in spec.eval_lengths:
        floors = s5_chain_floors(generate(spec, "test", n=200, length=L), spec.k)
        assert floors["echo"] == 0.0
        assert floors["uniform"] == 1.0 / spec.k
        # answer space is the roles, so the query start is not a candidate: no 1/(k-1) discount
        assert floors["uniform_non_start"] == 1.0 / spec.k


def test_typed_values_is_depth_one_only():
    spec = CANONICAL["s5_chain_typed_v1"].scaled(chain_depth=2)
    with pytest.raises(ValueError, match="depth-1 construct"):
        generate(spec, "test", n=1, length=4)


def test_typed_and_untyped_initial_maps_have_different_structure():
    """The second documented difference, entailed by the type split rather than chosen: the
    untyped initial map is a single k-cycle (no fixed points, every agent reachable from every
    other), the typed one a uniform agent->role bijection, for which cycles are undefined."""
    import re

    fact = re.compile(r"\b(g\d+)'s a0 is ([a-z]+\d+)\.")
    untyped = CANONICAL["s5_chain_local_v2"].scaled(chain_depth=1)
    for ex in generate(untyped, "test", n=10, length=4):
        nxt = dict(fact.findall(ex.prompt))
        assert set(nxt) == set(nxt.values())                # endo-map on the agent set
        node, seen = next(iter(nxt)), 0
        while True:                                         # one cycle covering every slot
            node, seen = nxt[node], seen + 1
            if node == next(iter(nxt)):
                break
        assert seen == len(nxt) == untyped.k

    typed = CANONICAL["s5_chain_typed_v1"]
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

    typed = dataclasses.asdict(CANONICAL["s5_chain_typed_v1"])
    untyped = dataclasses.asdict(CANONICAL["s5_chain_local_v2"].scaled(chain_depth=1))
    assert set(typed) == set(untyped)
    differing = {f for f in typed if typed[f] != untyped[f]}
    assert differing == {"name", "typed_values", "distinct_path"}
    assert typed["typed_values"] and not untyped["typed_values"]
    assert untyped["distinct_path"] and not typed["distinct_path"]


def test_typed_values_supervision_shapes():
    """event_trace dumps the whole agent->role map per event; start_trace emits the queried
    slot's current value per event and builds the interleaved training prompt."""
    spec = CANONICAL["s5_chain_typed_v1"]
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
    spec = CANONICAL["s5_chain_local_v2"]
    a = generate(spec, "test", n=5, length=4)
    generate(CANONICAL["s5_chain_typed_v1"], "test", n=5, length=4)
    b = generate(spec, "test", n=5, length=4)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.answer for x in a] == [x.answer for x in b]


def test_event_trace_checkpoints():
    """local_v2 dense supervision: the trace carries one full a0-map checkpoint
    (k agents) per event, then the query path prefix."""
    spec = CANONICAL["s5_chain_local_v2"]
    L = 4
    ex = generate(spec, "test", n=1, length=L)[0]
    toks = ex.meta["trace"].split()
    assert len(toks) == L * spec.k + spec.chain_depth
    # the trace tail is the query path prefix and chains to the gold answer
    assert toks[-spec.chain_depth] == ex.meta["start"]
    assert toks[-1] == ex.meta["path"][-2]
