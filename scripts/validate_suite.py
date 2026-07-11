"""Validity gate over the CANONICAL task suite — guarantees cover the headline tasks.

For every canonical task, certify that no shallow baseline clears floor on the held-out test split:
  - oracle-consistency: gold answers resolve through the symbolic oracle (true by construction; we assert
    the answer span is well-formed and the answer-token type is as expected).
  - answer balance: the majority-answer baseline ≈ floor (no dominant-class shortcut).
  - recency shortcut: predicting the last in-prompt token of the answer's type (e.g. the final-give agent)
    stays near floor — the composition tasks are not solvable by "copy the most recent entity".
  - STRONG recency shortcut (binding/composite only): the full-answer heuristic "last give-event's
    recipient" (+ "that holder's stated a0 fact" for composite) — see factworld.validity. Every
    registered binding/composite task uses the last_write_uniform (v2) sampler, so this baseline is
    GATED (must stay near floor). The recency-defective v1 family — where this heuristic scored
    ~0.34@L16 on composite_copy_v1 / ~0.4 on binding_v1 — is RETIRED (tasks.RETIRED, issue #11):
    excluded from the suite run here; its known-shortcut annotation lives on the RETIRED dict.
A task PASSES if majority, recency, first-position and (where defined) strong-recency accuracy are
all well below 0.5 (near the 1/#answers floor).

  .venv/bin/python scripts/validate_suite.py
"""
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import tasks as TK          # noqa: E402
from factworld.render import Renderer, classify  # noqa: E402  (atomic-token type by prefix: g/v/r/o/...)
from factworld.validity import comm_shallow_accuracy, strong_recency_accuracy  # noqa: E402

N = 500

# The strong recency baseline only has a defined prediction on the give-stream families.
STRONG_REC_FAMILIES = ("binding", "composite")

# The commutative family gets its own four shallow adversaries (initial-only / last-turn-only /
# entity-blind-sum / count-mod-k, factworld.validity.comm_shallow_accuracy); the MAX of the four
# fills the strongrec-style column and folds into the verdict.
COMM_FAMILIES = ("commutative",)


def positional_pred(prompt: str, ans_type: str, which: str):
    """The first/last token in the prompt whose type matches the answer's type — a fixed-POSITION shortcut.
    `which='last'` is the recency shortcut; `which='first'` catches 'the answer is always the first
    fact's value'. Operates on the normalized (detached-punctuation) form so attached-punctuation
    tokens like `v109.` / `g0's` are classified correctly."""
    toks = Renderer.normalize(prompt).split()
    it = reversed(toks) if which == "last" else toks
    for t in it:
        if classify(t) == ans_type:
            return t
    return None


def main():
    print(f"Validity gate over CANONICAL suite (n={N} held-out test, at eval_lengths[-1]; "
          f"RETIRED specs excluded — see tasks.RETIRED)\n")
    print(f"  {'task':<22} {'#ans':>5} {'floor':>6} {'majority':>9} {'recency':>8} {'firstpos':>9} {'strongrec':>10}   verdict")
    all_ok = True
    for name, spec in TK.CANONICAL.items():
        test = TK.generate(spec, "test", n=N, length=spec.eval_lengths[-1])
        # normalize answers so the check is format-agnostic (attached `.` -> ` .`)
        ans_norm = [Renderer.normalize(e.answer) for e in test]
        firsts = [a.split()[0] for a in ans_norm]
        assert all(a.split()[-1] == "." for a in ans_norm), f"{name}: answer not '.'-terminated"
        atype = classify(firsts[0])                      # answer-token type (g/v/r)
        assert all(classify(f) == atype for f in firsts), f"{name}: inconsistent answer-token type"
        distinct = len(set(firsts))
        floor = 1.0 / distinct
        majority = Counter(firsts).most_common(1)[0][1] / N
        recency = sum(positional_pred(e.prompt, atype, "last") == f for e, f in zip(test, firsts)) / N
        firstpos = sum(positional_pred(e.prompt, atype, "first") == f for e, f in zip(test, firsts)) / N
        ok = majority < 0.5 and recency < 0.5 and firstpos < 0.5
        if spec.family in STRONG_REC_FAMILIES:
            # every registered give-stream task is a v2 (last_write_uniform) spec: GATED.
            assert spec.last_write_uniform, \
                f"{name}: non-uniform (v1) sampler in CANONICAL — v1 specs belong in RETIRED"
            strongrec = strong_recency_accuracy(test, spec.family)
            ok &= strongrec < 0.5
            srec_col = f"{strongrec:>10.3f}"
        elif spec.family in COMM_FAMILIES:
            # commutative rung: the strongest of the four dial-fold shallow adversaries.
            strongrec = max(comm_shallow_accuracy(test, spec.k_positions).values())
            ok &= strongrec < 0.5
            srec_col = f"{strongrec:>10.3f}"
        else:
            srec_col = f"{'—':>10}"
        all_ok &= ok
        print(f"  {name:<22} {distinct:>5} {floor:>6.3f} {majority:>9.3f} {recency:>8.3f} {firstpos:>9.3f} {srec_col}   {'PASS' if ok else 'FLAG'}")
    print(f"\nSUITE VALIDITY: {'PASS — no shallow/recency/position shortcut clears floor on any canonical task' if all_ok else 'FLAG — investigate'}")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
