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


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
