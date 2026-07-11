"""Tests for commutative_v1 — the commutative (abelian) rung of the state-tracking ladder.

Per-entity dial-rotation accumulation mod k_positions=5: every event on the queried agent
matters, order does not (retrieval < last-write < commutative < non-abelian). This file pins
the construct in the house idiom (cf. tests/test_composite_v2.py):

  (a) registry contract — kind=experimental, out of REPORTED, version pinned "1.1",
      SUITE_VERSION unchanged;
  (b) determinism — regenerating gives identical (prompt, answer, meta);
  (c) goldens pinned — sha256 of the n=25 test streams at L16/L64, frozen from day one
      (same (spec, split, length, idx) -> identical example, forever);
  (d) oracle gold == (initial + sum of the queried agent's amounts) mod 5, recomputed from
      the RENDERED prompt (the label-cannot-leak round-trip); w_q >= 2 forced;
  (e) shallow-adversary floors — the four one-liner cheats (initial-only / last-turn-only /
      entity-blind-sum / count-mod-k) all <= 2x chance (0.4), majority <= 0.25, answer
      histogram uniform within binomial noise;
  (f) render <-> parse round-trip for the new grammar (turn_dial event incl. the singular
      "1 click" form, dial assertion, state_comm query; classify types pN);
  (g) worked trace matches oracle.comm_trace;
  (h) train-split smoke.

Run directly:  .venv-api/bin/python tests/test_commutative_v1.py
Run with pytest: .venv-api/bin/python -m pytest tests/test_commutative_v1.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld import tasks as TK
from factworld.render import Renderer, classify
from factworld.validity import comm_shallow_accuracy
from factworld.world import Event

# The canonical rendered grammar for the commutative rung (see factworld/render.py).
_TURN_RE = re.compile(r"\bs\d+ turns (g\d+)'s dial (\d+) clicks?\.")
_DIAL_RE = re.compile(r"\b(g\d+)'s dial is at p(\d+)\.")

LENGTHS = (16, 64)
K = 5                      # k_positions: chance floor 0.200
GATE_BOUND = 2.0 / K       # each shallow adversary must stay <= 2x chance

# Stream hashes pinned at introduction (suite 1.1): commutative_v1 is a frozen spec from
# day one — same (spec, split, length, idx) -> identical example, forever.
V1_GOLDENS = {
    16: "814a63b42aed7d0d107847816a9d49f6fb68cf4df7dadc1a068d738981022e10",
    64: "5355b5b1abc4a1cee3dd507ffbb7eb2c3b80fd103d96d19bcd6983d8ac7506ad",
}


def _hash(examples) -> str:
    return hashlib.sha256("\n".join(f"{e.prompt}\t{e.answer}" for e in examples).encode()).hexdigest()


def _gen(length, n, split="test", spec=None):
    spec = spec or TK.CANONICAL["commutative_v1"]
    return TK.generate(spec, split, n=n, length=length)


# --- (a) registry contract ---------------------------------------------------------------

def test_registry_contract():
    assert "commutative_v1" in TK.CANONICAL
    spec = TK.CANONICAL["commutative_v1"]
    assert spec.kind == "experimental"                 # not scored until calibrated (like s5_v1)
    assert "commutative_v1" not in TK.REPORTED
    assert spec.family == "commutative"
    assert spec.version == "1.1"                       # stream version pinned at introduction
    assert spec.k_positions == 5 and spec.n_objects_active == 4
    assert spec.train_lengths == (4, 8, 16) and spec.eval_lengths == (16, 32, 64)
    assert TK.spec_for("commutative_v1") is spec
    assert TK.SUITE_VERSION == "1.1"                   # experimental kind: no suite bump


# --- (b) determinism ---------------------------------------------------------------------

def test_determinism():
    spec = TK.CANONICAL["commutative_v1"]
    for split, L in (("test", 64), ("train", None)):
        a = TK.generate(spec, split, n=30, length=L)
        b = TK.generate(spec, split, n=30, length=L)
        assert [(x.prompt, x.answer, x.meta) for x in a] == \
               [(x.prompt, x.answer, x.meta) for x in b], f"{split}: nondeterministic"


# --- (c) goldens pinned ------------------------------------------------------------------

def test_goldens_pinned():
    for L, want in V1_GOLDENS.items():
        assert _hash(_gen(L, n=25)) == want, f"commutative_v1@L{L}: stream drifted"


# --- (d) oracle gold == prompt resolution ------------------------------------------------

def test_gold_matches_prompt_resolution():
    for L in LENGTHS:
        for e in _gen(L, n=100):
            target = e.meta["target"]
            initials = {g: int(d) for g, d in _DIAL_RE.findall(e.prompt)}
            turns = [(g, int(a)) for g, a in _TURN_RE.findall(e.prompt)]
            assert len(turns) == L, "every event must render as a turn line"
            mine = [a for g, a in turns if g == target]
            assert e.meta["w"] == len(mine) >= 2, "w_q >= 2 is a design guarantee"
            got = (initials[target] + sum(mine)) % K
            assert e.answer == f"p{got}.", f"gold mismatch at L{L}"
            assert e.meta["initial"] == f"p{initials[target]}"
            if L >= 8:
                assert any(g != target for g, _a in turns), "distractor turns must exist"
            assert all(1 <= a <= K - 1 for _g, a in turns), "amounts nonzero mod k"


# --- (e) shallow adversaries at floor ----------------------------------------------------

def test_shallow_baselines_at_floor():
    for L in LENGTHS:
        test = _gen(L, n=400)
        acc = comm_shallow_accuracy(test, K)
        assert set(acc) == {"initial_only", "last_turn_only", "entity_blind_sum", "count_mod_k"}
        for name, a in acc.items():
            assert a <= GATE_BOUND, f"{name}@L{L}: {a:.3f} > 2x chance {GATE_BOUND}"
        firsts = [e.answer for e in test]
        maj = Counter(firsts).most_common(1)[0][1] / len(test)
        assert maj <= 0.25, f"majority {maj:.3f} > 0.25 @L{L}"
        hist = Counter(firsts)
        for p in (f"p{i}." for i in range(K)):
            frac = hist[p] / len(test)
            assert 0.13 <= frac <= 0.27, f"answer {p} holds {frac:.3f} @L{L} (gold not uniform?)"


# --- (f) render <-> parse round-trip -----------------------------------------------------

def test_render_roundtrip():
    r = Renderer()
    line = r.render_event(Event("turn_dial", ("g3", "2")), step="s0")
    assert line == "s0 turns g3's dial 2 clicks."
    parsed = r.parse(line)
    assert parsed["type"] == "event" and parsed["step"] == "s0"
    assert parsed["event"] == Event("turn_dial", ("g3", "2"))
    # singular click form
    line1 = r.render_event(Event("turn_dial", ("g1", "1")), step="s4")
    assert line1 == "s4 turns g1's dial 1 click."
    assert r.parse(line1)["event"] == Event("turn_dial", ("g1", "1"))
    # dial assertion (initial-condition line)
    dial = r.render_dial("g2", "p4")
    assert dial == "g2's dial is at p4."
    assert r.parse(dial) == {"type": "dial", "agent": "g2", "position": "p4", "step": None}
    # query, final and as-of-t
    q = r.render_query("state_comm", target="g3")
    assert q == "what position is g3's dial?"
    assert r.parse(q) == {"type": "query", "family": "state_comm", "target": "g3", "step": None}
    qt = r.render_query("state_comm", target="g3", t=5)
    assert r.parse(qt) == {"type": "query", "family": "state_comm", "target": "g3", "step": "s4"}
    # token typing: pN is a typed atomic token; a bare digit is not
    assert classify("p3") == "p" and classify("3") is None


# --- (g) worked trace --------------------------------------------------------------------

def test_worked_trace():
    from factworld.oracle import Oracle
    spec = TK.CANONICAL["commutative_v1"].scaled(worked_trace=True)
    w, r = TK.build_world(spec)
    oracle = Oracle(w)
    for e in _gen(16, n=25, spec=spec):
        trace = e.meta["trace"].split()
        assert len(trace) == 16, "trace = position after each of the L events"
        # recompute the trajectory from the rendered prompt
        initials = {g: f"p{d}" for g, d in _DIAL_RE.findall(e.prompt)}
        events = [Event("turn_dial", (g, a)) for g, a in _TURN_RE.findall(e.prompt)]
        want = oracle.comm_trace(initials, events, e.meta["target"])[1:]
        assert trace == want
        assert trace[-1] == e.answer.rstrip(".")


# --- (h) train-split smoke ---------------------------------------------------------------

def test_train_split_smoke():
    exs = TK.generate(TK.CANONICAL["commutative_v1"], "train", n=60)
    assert len(exs) == 60 and all(e.answer for e in exs)
    for e in exs:
        assert e.length in (4, 8, 16)
        assert e.meta["w"] >= 2


if __name__ == "__main__":
    for fn in [test_registry_contract, test_determinism, test_goldens_pinned,
               test_gold_matches_prompt_resolution, test_shallow_baselines_at_floor,
               test_render_roundtrip, test_worked_trace, test_train_split_smoke]:
        fn()
        print(f"{fn.__name__}: ok")
