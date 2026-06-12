"""M3 part 2 invariants: role/holder assertions round-trip, the aux operator-worlds carry the
oracle's worked traces / final answers over disjoint symbols, and `build_corpus` assembles the
operator-learning split (target histories-without-finals + aux supervision) — and the M4
tokenizer round-trips the whole assembled corpus.

Runs with zero dependencies:  python3 tests/test_corpus_aux.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.config import WorldConfig  # noqa: E402
from factworld.corpus import aux_operator_documents, build_corpus  # noqa: E402
from factworld.eval import Episode  # noqa: E402
from factworld.oracle import Oracle  # noqa: E402
from factworld.render import Renderer, classify  # noqa: E402
from factworld.tokenizer import Tokenizer  # noqa: E402
from factworld.world import World  # noqa: E402


def _aux(seed=0, ns="aux0_"):
    w = World(WorldConfig(seed=seed, id_namespace=ns, n_entities=0, n_attributes=0, value_vocab_size=1))
    return w, Oracle(w), Renderer()


def test_role_and_holder_assertions_roundtrip():
    r = Renderer()
    for step in (None, "s4"):
        assert r.parse(r.render_role("g1", "r3", step=step)) == {
            "type": "role", "agent": "g1", "role": "r3", "step": step}
        assert r.parse(r.render_holder("o2", "loc5", step=step)) == {
            "type": "holder", "object": "o2", "holder": "loc5", "step": step}
        assert r.parse(r.render_holder("o2", "g4", step=step)) == {
            "type": "holder", "object": "o2", "holder": "g4", "step": step}


def test_aux_hard_doc_is_the_oracle_worked_trace():
    w, o, r = _aux()
    ep = Episode("state_hard", tuple(w.sample_hard_chain(8, "x")), 8, "x")
    docs = aux_operator_documents(w, o, r, [ep])
    assert len(docs) == 1 and docs[0].kind == "aux_trace"
    trace = o.hard_trace(ep.events)
    expected = []
    for i, e in enumerate(ep.events):
        expected.append(r.render_event(e, step=f"s{i}"))
        for ag in w.agents:
            expected.append(r.render_role(ag, trace[i + 1][ag], step=f"s{i}"))
    assert docs[0].text == " ".join(expected)
    for stmt in expected:                                  # every role line names the post-event state
        rec = r.parse(stmt)
        if rec["type"] == "role":
            assert trace[int(rec["step"][1:]) + 1][rec["agent"]] == rec["role"]


def test_aux_easy_doc_states_oracle_finals():
    w, o, r = _aux()
    ep = Episode("state_easy", tuple(w.sample_easy_chain(12, "y")), 12, "y")
    doc = aux_operator_documents(w, o, r, [ep])[0]
    assert doc.kind == "aux_answer"
    expected = r.render_history(ep.events, with_steps=True) + [
        r.render_holder(obj, o.easy_holder(list(ep.events), obj)) for obj in w.objects]
    assert doc.text == " ".join(expected)


def test_aux_symbols_are_namespaced_and_disjoint():
    w, o, r = _aux()
    ep = Episode("state_hard", tuple(w.sample_hard_chain(6, "z")), 6, "z")
    doc = aux_operator_documents(w, o, r, [ep])[0]
    target = World(WorldConfig(seed=0))
    tgt_syms = set(target.agents) | set(target.roles) | set(target.objects) | set(target.locations)
    content = [tok for tok in doc.text.split() if classify(tok)]
    assert content and all(tok.startswith("aux0_") or tok.startswith("s") for tok in content)
    assert not (set(content) & tgt_syms)


def test_build_corpus_assembles_operator_learning_split():
    w = World(WorldConfig(seed=0))
    c = build_corpus(w, Oracle(w), Renderer(), n_recall_docs=50, easy_lengths=(8,),
                     hard_lengths=(16,), n_state_episodes=5, n_aux_worlds=2)
    kinds = Counter(d.kind for d in c.documents)
    assert kinds["recall_fact"] == 50
    assert kinds["history_only"] > 0 and kinds["aux_trace"] > 0 and kinds["aux_answer"] > 0
    assert len(c.worlds) == 1 + 2                          # target + aux
    # target hard histories leak no finals (no role tokens); aux traces DO carry roles
    for d in c.documents:
        if d.kind == "history_only" and d.meta["ns"] == "" and d.meta["family"] == "state_hard":
            assert all(classify(tok) != "r" for tok in d.text.split())
        if d.kind == "aux_trace":
            assert any(classify(tok) == "r" for tok in d.text.split())


def test_tokenizer_roundtrips_the_whole_corpus():
    w = World(WorldConfig(seed=0))
    c = build_corpus(w, Oracle(w), Renderer(), n_recall_docs=30, easy_lengths=(8,),
                     hard_lengths=(16,), n_state_episodes=3, n_aux_worlds=2)
    tok = Tokenizer.build(c.worlds, Renderer())
    for d in c.documents:
        ids = tok.encode(d.text)
        assert tok.unk_id not in ids                       # closed vocab: no unknowns
        assert tok.decode(ids) == d.text                   # exact round-trip


def test_scenario_id_is_compositional_and_tokenizes():
    w = World(WorldConfig(seed=0))
    r = Renderer()
    tok = Tokenizer.build([w], r)
    for idx in (0, 42, 137, 9999):
        s = r.render_scenario(idx)
        assert s == "scn " + " ".join("#" + c for c in str(idx).zfill(4))
        ids = tok.encode(s)
        assert tok.unk_id not in ids and tok.decode(ids) == s
    # digits are SHARED across scenarios (no per-scenario token) -> ids overlap in vocab
    assert set(tok.encode(r.render_scenario(11))) & set(tok.encode(r.render_scenario(22)))


def test_build_corpus_deterministic():
    w = World(WorldConfig(seed=0))
    kw = dict(n_recall_docs=20, easy_lengths=(8,), hard_lengths=(16,), n_state_episodes=3, n_aux_worlds=2)
    a = build_corpus(w, Oracle(w), Renderer(), **kw)
    b = build_corpus(w, Oracle(w), Renderer(), **kw)
    assert [d.text for d in a.documents] == [d.text for d in b.documents]


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
