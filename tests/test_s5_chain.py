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
    # single-line answers (API clean form and local streams) pass through untouched
    assert committed_answer("g5.") == "g5."
    assert committed_answer("g3. <eos> g7 g0") == "g3. <eos> g7 g0"
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
