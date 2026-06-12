"""M3 invariants: recall documents realise the three Kim-et-al. data properties as
*measured* knobs, and target history documents leak no finals.

Runs with zero dependencies:  python3 tests/test_corpus.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.config import WorldConfig  # noqa: E402
from factworld.corpus import history_only_documents, recall_documents  # noqa: E402
from factworld.eval import Episode  # noqa: E402
from factworld.render import Renderer, classify  # noqa: E402
from factworld.world import World  # noqa: E402


def _wr():
    return World(WorldConfig(seed=0)), Renderer()


def test_recall_docs_follow_zipf_frequency():
    w, r = _wr()
    docs = recall_documents(w, r, 5000, seed="z", repetition=1, inconsistency_rate=0.0)
    band_tot = Counter(d.meta["freq_band"] for d in docs)
    # the frequent band (2) dominates the rare band (0) under Zipf sampling
    assert band_tot[2] > band_tot[0] * 3


def test_inconsistency_rate_is_approximately_configured():
    w, r = _wr()
    docs = recall_documents(w, r, 8000, seed="i", repetition=2, inconsistency_rate=0.01)
    frac = sum(d.meta["inconsistent"] for d in docs) / len(docs)
    assert 0.005 <= frac <= 0.02


def test_intra_doc_repetition_count():
    w, r = _wr()
    docs = recall_documents(w, r, 50, seed="r", repetition=3, inconsistency_rate=0.0)
    # a consistent doc states each of the n_attributes facts `repetition` times
    n_values = sum(1 for tok in docs[0].text.split() if classify(tok) == "v")
    assert n_values == 3 * w.config.n_attributes


def test_history_only_is_events_only_no_answer_leak():
    w, r = _wr()
    for fam, sampler in (("state_easy", w.sample_easy_chain), ("state_hard", w.sample_hard_chain)):
        ep = Episode(fam, tuple(sampler(20, "x")), 20, "x")
        history_only_documents(w, r, [ep])  # smoke: builds
        lines = r.render_history(ep.events, with_steps=True)
        assert all(r.parse(line)["type"] == "event" for line in lines)  # no answer statement
        if fam == "state_hard":  # the answer would be a role token; none may appear
            assert all(classify(tok) != "r" for tok in " ".join(lines).split())


def test_corpus_is_deterministic():
    w, r = _wr()
    a = recall_documents(w, r, 200, seed="d")
    b = recall_documents(w, r, 200, seed="d")
    assert [d.text for d in a] == [d.text for d in b]
    assert [d.meta for d in a] == [d.meta for d in b]


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
