"""M2 invariants: render<->parse is an exact round-trip (the ground-truth re-parse check),
and eval-suite gold is the oracle's answer, derived from the KB and never from rendered text.

Runs with zero dependencies:  python3 tests/test_eval_render.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.config import WorldConfig  # noqa: E402
from factworld.eval import easy_suite, hard_suite, recall_suite  # noqa: E402
from factworld.oracle import Oracle  # noqa: E402
from factworld.render import Renderer, classify  # noqa: E402
from factworld.world import World  # noqa: E402


def _wo():
    w = World(WorldConfig(seed=0))
    return w, Oracle(w), Renderer()


# --- render <-> parse round-trip (the no-leakage / ground-truth contract) ------------------

def test_fact_roundtrip():
    w, o, r = _wo()
    for e in w.entities[:50]:
        for a in w.attribute_names:
            v = o.recall(e, a)
            assert r.parse(r.render_fact(e, a, v)) == {
                "type": "fact", "entity": e, "attribute": a, "value": v
            }


def test_easy_event_roundtrip_with_and_without_steps():
    w, o, r = _wo()
    ev = w.sample_easy_chain(60, "x")
    for i, e in enumerate(ev):
        for step in (None, f"s{i}"):
            rec = r.parse(r.render_event(e, step=step))
            assert rec["type"] == "event" and rec["event"] == e and rec["step"] == step


def test_hard_event_roundtrip():
    w, o, r = _wo()
    ev = w.sample_hard_chain(80, "y")
    for i, e in enumerate(ev):
        rec = r.parse(r.render_event(e, step=f"s{i}"))
        assert rec["event"] == e and rec["step"] == f"s{i}"


def test_history_roundtrip_both_modes():
    w, o, r = _wo()
    for sampler in (w.sample_easy_chain, w.sample_hard_chain):
        ev = sampler(64, "h")
        for with_steps in (False, True):
            assert r.parse_history(r.render_history(ev, with_steps=with_steps)) == list(ev)


def test_render_is_deterministic():
    """Rendering is deterministic (one fixed phrasing per statement type)."""
    w, o, r = _wo()
    facts = [r.render_fact(e, a, o.recall(e, a)) for e in w.entities[:200] for a in w.attribute_names]
    facts2 = [r.render_fact(e, a, o.recall(e, a)) for e in w.entities[:200] for a in w.attribute_names]
    assert facts == facts2                                   # deterministic
    assert all(f.endswith(".") for f in facts)               # attached-punctuation format


def test_query_roundtrip():
    w, o, r = _wo()
    assert r.parse(r.render_query("recall", entity="e3", attribute="a1")) == {
        "type": "query", "family": "recall", "entity": "e3", "attribute": "a1"
    }
    assert r.parse(r.render_query("state_easy", target="o2"))["family"] == "state_easy"
    q = r.parse(r.render_query("state_hard", target="g2", t=5))
    assert q["family"] == "state_hard" and q["target"] == "g2" and q["step"] == "s4"


def test_state_easy_query_is_unambiguous():
    """state_easy query should ask for the final holder, not every holder, and round-trip."""
    w, o, r = _wo()
    q = r.render_query("state_easy", target="o2")
    assert "final holder" in q or "holds" in q
    assert r.parse(q)["family"] == "state_easy"
    assert r.parse(q)["target"] == "o2"


def test_composite_recall_query_not_misrouted():
    """The composite recall query contains 'holder' but must parse as recall, not state_easy."""
    w, o, r = _wo()
    q = r.render_query("recall", attribute="a0", entity="the holder of o3")
    assert r.parse(q)["family"] == "recall"


# --- eval gold == oracle (truth is KB-derived) --------------------------------------------

def test_recall_gold_matches_oracle():
    w, o, r = _wo()
    for it in recall_suite(w, o, 200, seed="s"):
        assert it.gold == o.recall(it.entity, it.attribute)
        assert it.freq_band in (0, 1, 2)


def test_easy_gold_matches_oracle():
    w, o, r = _wo()
    for as_of in (False, True):
        for it in easy_suite(w, o, (8, 16), 20, seed="s", as_of_t=as_of):
            assert it.gold == o.easy_holder(list(it.episode.events), it.target, it.t)
            if as_of:
                assert 1 <= it.t <= it.length


def test_hard_gold_matches_oracle():
    w, o, r = _wo()
    for as_of in (False, True):
        for it in hard_suite(w, o, (16, 32), 20, seed="s", as_of_t=as_of):
            assert it.gold == o.hard_role(list(it.episode.events), it.target, it.t)


def test_as_of_t_query_references_correct_step():
    w, o, r = _wo()
    for it in hard_suite(w, o, (32,), 5, seed="z", as_of_t=True):
        q = r.parse(r.render_query("state_hard", target=it.target, t=it.t))
        assert q["step"] == f"s{it.t - 1}"
        assert o.hard_role(list(it.episode.events), it.target, it.t) == it.gold


# --- markdown-strip normalization (F1: '**g22**' must score like 'g22') --------------------

def _tokens_path_relaxed(pred: str, gold: str) -> int:
    # exactly the tokens-path scoring pipeline (runner.evaluate_task / benchmark cells)
    from factworld.tasks import score_relaxed
    return score_relaxed(Renderer.normalize(pred), Renderer.normalize(gold))


def test_normalize_strips_markdown_emphasis_from_token_edges():
    from factworld.tasks import score_contains, score_exact, score_last_n, score_relaxed
    gold = "g22."  # attached-punctuation gold, exactly as chain cells render it
    gold_n = Renderer.normalize(gold)
    # the exact live sonnet chain_nowrap pattern, plus backtick and underscore decoration
    for pred in ("**g22**", "**g22**.", "`g22`", "_g22_", "**g22", "g22**", "** g22 **"):
        pred_n = Renderer.normalize(pred)
        assert score_relaxed(pred_n, gold_n) == 1, pred
        assert score_contains(pred_n, gold_n) == 1, pred
        assert score_last_n(pred_n, gold_n) == 1, pred
        # exact agrees with the same pred minus its decoration (semantics otherwise unchanged)
        undecorated = pred.replace("*", "").replace("`", "").replace("_", "")
        assert score_exact(pred_n, gold_n) == score_exact(Renderer.normalize(undecorated), gold_n), pred
    # multi-token composite answer decorated as a span
    assert _tokens_path_relaxed("**g30 v73**", "g30 v73.") == 1


def test_normalize_markdown_strip_is_edge_only():
    # internal underscores are token-structural (namespaced ids), never stripped
    assert Renderer.normalize("aux1_g0") == "aux1_g0"
    assert classify("aux1_g0") == "g"
    # non-markdown text is normalized exactly as before
    assert Renderer.normalize("g9's a0 is v26.") == "g9 's a0 is v26 ."
    assert Renderer.normalize("s1 gives o0 to g0.") == "s1 gives o0 to g0 ."


def test_prose_buried_answer_still_scores_zero():
    # prefix-commit is intentional: a correct answer buried mid-prose scores 0 (the kimi
    # chain d128 pattern — markdown stripping must NOT relax positional matching)
    pred = "Counting to index 128 we land on g20 in this cycle, so the answer follows."
    assert _tokens_path_relaxed(pred, "g20.") == 0
    # ...whereas a committed (first-token) markdown answer scores 1
    assert _tokens_path_relaxed("**g20** is the answer", "g20.") == 1


def test_s5_content_tokens_strip_markdown():
    # s5_concrete.score goes through tasks.content_tokens -> Renderer.normalize
    from factworld.tasks import content_tokens
    assert content_tokens("**Driver**.") == ["Driver"]
    assert content_tokens("Driver") == ["Driver"]
    # first-content-token commit still fails when prose precedes the answer
    assert content_tokens("The role is Driver")[0] != "Driver"


def test_eval_layer_is_kb_derived_not_text():
    # architectural no-leakage guarantee: the eval module knows nothing about rendering
    import factworld.eval as E
    assert not hasattr(E, "Renderer")
    w, o, r = _wo()
    for it in recall_suite(w, o, 50, seed="k"):
        assert isinstance(it.gold, str) and it.gold in w.value_vocab


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
