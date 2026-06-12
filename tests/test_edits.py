"""Append-edit invariants: the append-changes-answer control flips the target's gold, the
specificity (non-changing) append leaves it, and the diff manifest exactly partitions targets.

Runs with zero dependencies:  python3 tests/test_edits.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.config import WorldConfig  # noqa: E402
from factworld.edits import append_pair  # noqa: E402
from factworld.eval import Episode  # noqa: E402
from factworld.oracle import Oracle  # noqa: E402
from factworld.world import World  # noqa: E402


def _wo():
    w = World(WorldConfig(seed=0))
    return w, Oracle(w)


def _easy_ep(w, L=16, s="e"):
    return Episode("state_easy", tuple(w.sample_easy_chain(L, s)), L, s)


def _hard_ep(w, L=16, s="h"):
    return Episode("state_hard", tuple(w.sample_hard_chain(L, s)), L, s)


def test_changing_append_flips_the_target_answer():
    w, o = _wo()
    cases = [(_easy_ep(w), w.objects[3]), (_hard_ep(w), w.agents[2])]
    for ep, target in cases:
        before, after, edit = append_pair(w, o, ep, target, changing=True)
        assert before.gold != after.gold
        assert target in edit.manifest["changed"]
        assert edit.manifest["changed"][target] == [before.gold, after.gold]
        assert after.episode.events == ep.events + (edit.event,)
        assert after.length == ep.length + 1


def test_nonchanging_append_is_specific_but_real():
    w, o = _wo()
    cases = [(_easy_ep(w), w.objects[3]), (_hard_ep(w), w.agents[2])]
    for ep, target in cases:
        before, after, edit = append_pair(w, o, ep, target, changing=False)
        assert before.gold == after.gold
        assert target in edit.manifest["unchanged"]
        assert len(edit.manifest["changed"]) >= 1  # it did edit *something* (another target)


def test_manifest_is_a_partition_of_all_targets():
    w, o = _wo()
    for ep, targets in [(_easy_ep(w), set(w.objects)), (_hard_ep(w), set(w.agents))]:
        target = next(iter(targets))
        _, _, edit = append_pair(w, o, ep, target, changing=True)
        ch, un = set(edit.manifest["changed"]), set(edit.manifest["unchanged"])
        assert ch.isdisjoint(un)
        assert ch | un == targets


def test_append_is_deterministic():
    w, o = _wo()
    ep, t = _hard_ep(w), w.agents[0]
    a = append_pair(w, o, ep, t, changing=True)
    b = append_pair(w, o, ep, t, changing=True)
    assert a[2].event == b[2].event and a[1].gold == b[1].gold


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
