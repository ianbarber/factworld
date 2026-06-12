"""Validity gate over the CANONICAL task suite — guarantees cover the headline tasks.

For every canonical task, certify that no shallow baseline clears floor on the held-out test split:
  - oracle-consistency: gold answers resolve through the symbolic oracle (true by construction; we assert
    the answer span is well-formed and the answer-token type is as expected).
  - answer balance: the majority-answer baseline ≈ floor (no dominant-class shortcut).
  - recency shortcut: predicting the last in-prompt token of the answer's type (e.g. the final-give agent)
    stays near floor — the composition tasks are not solvable by "copy the most recent entity".
A task PASSES if majority and recency accuracy are both well below 0.5 (near the 1/#answers floor).

  .venv/bin/python scripts/validate_suite.py
"""
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import tasks as TK          # noqa: E402
from factworld.render import classify      # noqa: E402  (atomic-token type by prefix: g/v/r/o/...)

N = 500


def positional_pred(prompt: str, ans_type: str, which: str):
    """The first/last token in the prompt whose type matches the answer's type — a fixed-POSITION shortcut.
    `which='last'` is the recency shortcut; `which='first'` catches 'the answer is always the first
    fact's value' (e.g. a query whose target is always rendered first)."""
    toks = prompt.split()
    it = reversed(toks) if which == "last" else toks
    for t in it:
        if classify(t) == ans_type:
            return t
    return None


def main():
    print(f"Validity gate over CANONICAL suite (n={N} held-out test, '.'=at eval_lengths[-1])\n")
    print(f"  {'task':<18} {'#ans':>5} {'floor':>6} {'majority':>9} {'recency':>8} {'firstpos':>9}   verdict")
    all_ok = True
    for name, spec in TK.CANONICAL.items():
        test = TK.generate(spec, "test", n=N, length=spec.eval_lengths[-1])
        firsts = [e.answer.split()[0] for e in test]
        assert all(e.answer.split()[-1] == "." for e in test), f"{name}: answer not '.'-terminated"
        atype = classify(firsts[0])                      # answer-token type (g/v/r)
        assert all(classify(f) == atype for f in firsts), f"{name}: inconsistent answer-token type"
        distinct = len(set(firsts))
        floor = 1.0 / distinct
        majority = Counter(firsts).most_common(1)[0][1] / N
        recency = sum(positional_pred(e.prompt, atype, "last") == f for e, f in zip(test, firsts)) / N
        firstpos = sum(positional_pred(e.prompt, atype, "first") == f for e, f in zip(test, firsts)) / N
        ok = majority < 0.5 and recency < 0.5 and firstpos < 0.5
        all_ok &= ok
        print(f"  {name:<18} {distinct:>5} {floor:>6.3f} {majority:>9.3f} {recency:>8.3f} {firstpos:>9.3f}   {'PASS' if ok else 'FLAG'}")
    print(f"\nSUITE VALIDITY: {'PASS — no shallow/recency/position shortcut clears floor on any canonical task' if all_ok else 'FLAG — investigate'}")


if __name__ == "__main__":
    main()
