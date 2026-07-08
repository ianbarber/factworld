"""chain_v1 depth-validity gate — regression lock for the wrap defect.

The chain pointer map is a single k-cycle, so at depth >= k gold collapses to
nxt^(depth mod k)(start) and "depth" is not being measured (depth ≡ 0 mod k is
the identity). This locks in: (1) generating at depth >= k RAISES unless the
caller opts into wrap semantics via spec.scaled(chain_allow_wrap=True); (2) the
no-wrap deep-chain protocol spec.scaled(k=depth+2) is genuinely valid — gold is
reachable only through depth distinct hops; (3) the shipped shallow protocol
(eval_lengths (4, 5), train_lengths (2, 3)) is unchanged.

Runs with zero dependencies:  python3 tests/test_chain_validity.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.tasks import CANONICAL, generate  # noqa: E402

SPEC = CANONICAL["chain_v1"]

# The renderer has exactly one fact phrasing ("{e}'s a0 is {v}."), so the pointer map can be
# re-parsed from the prompt and the example verified independently of the generator's own walk.
_FACT = re.compile(r"(\S+)'s a0 is (\S+)\.")


def _pointer_map(prompt: str) -> dict[str, str]:
    return {e: v for e, v in _FACT.findall(prompt)}


def _walk(nxt: dict[str, str], start: str, depth: int) -> tuple[str, list[str]]:
    """Follow the parsed map: (final node, full visit path incl. start)."""
    cur, path = start, [start]
    for _ in range(depth):
        cur = nxt[cur]
        path.append(cur)
    return cur, path


def test_depth_at_or_beyond_k_raises():
    for depth in (SPEC.k, SPEC.k + 1, 12, 64):                 # k=6: boundary, +1, and the old run depths
        try:
            generate(SPEC, "test", n=1, length=depth)
        except ValueError as e:
            assert "design" in str(e) and str(SPEC.k) in str(e)
        else:
            raise AssertionError(f"depth {depth} >= k {SPEC.k} must raise ValueError")


def test_wrap_opt_in_generates_but_is_degenerate():
    # explicit opt-in works — and demonstrates WHY the gate exists: depth ≡ 0 (mod k) is the identity
    ex = generate(SPEC.scaled(chain_allow_wrap=True), "test", n=1, length=2 * SPEC.k)[0]
    assert ex.meta["start"] == ex.answer.rstrip(".").strip()   # gold == start: depth not measured


def test_scaled_no_wrap_is_valid_at_depth_16_and_64():
    # k=2*depth+1 is the benchmark protocol: no wrap, AND the backward walk
    # costs depth+1 hops (k=depth+2 would put gold a constant 2 reverse
    # lookups from start — the same algebraic collapse as the wrap, offset -2).
    for depth in (16, 64):
        spec = SPEC.scaled(k=2 * depth + 1)
        for ex in generate(spec, "test", n=10, length=depth):
            start, gold = ex.meta["start"], ex.answer.rstrip(".").strip()
            assert gold != start                               # never the identity shortcut
            nxt = _pointer_map(ex.prompt)
            assert len(nxt) == 2 * depth + 1                   # one fact per agent in the k-pool
            final, path = _walk(nxt, start, depth)
            assert final == gold                               # gold re-derived from the rendered facts
            assert len(set(path)) == depth + 1                 # no revisit before the answer
            # backward distance from start to gold is k - depth = depth + 1 > depth
            back, cur = 0, start
            while cur != gold:
                cur = {v: a for a, v in nxt.items()}[cur]
                back += 1
            assert back == depth + 1                           # reverse walk is never cheaper


def test_scaled_no_wrap_is_deterministic():
    spec = SPEC.scaled(k=18)
    a = generate(spec, "test", n=5, length=16)
    b = generate(spec, "test", n=5, length=16)
    assert [(e.prompt, e.answer) for e in a] == [(e.prompt, e.answer) for e in b]


def test_shipped_shallow_protocol_unchanged():
    # eval_lengths (4, 5) and train_lengths (2, 3) are all < k=6: no gate, still valid examples
    for depth in SPEC.eval_lengths:
        for ex in generate(SPEC, "test", n=10, length=depth):
            start, gold = ex.meta["start"], ex.answer.rstrip(".").strip()
            assert gold != start
            final, path = _walk(_pointer_map(ex.prompt), start, depth)
            assert final == gold and len(set(path)) == depth + 1
    train = generate(SPEC, "train", n=20)
    assert len(train) == 20 and all(e.answer for e in train)


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
