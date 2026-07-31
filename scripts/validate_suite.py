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
  - SHALLOW-ADVERSARY floor (chain / s5_chain): the largest of the registered pointer-map
    adversaries — initial-map chase, initial-ref resolution, echo (factworld.validity), i.e. the
    cell's operative floor with the two chance rows removed, since chance is not a shortcut.
    The initial-map backhop is measured by s5_chain_floors but is not registered and so is not
    in this column: it is an unnamed member of a fixed-offset family whose accuracies sum to 1,
    so its null is uniform-over-non-start and a max over it measures selection.
  - SHALLOW-ADVERSARY floor (s5_bind): the largest registered mutual-reference policy
    (factworld.validity.S5_BIND_ADVERSARIES) — the coupling-blind rows, the wrong-time row, the
    zero-state pin chain, the stated/one-hop rows, and the recency-window family. The window
    rows enter the GATE, and the operative floor, only on a coupled rendering: a windowed policy
    still maintains both maps, so it is cheaper than the task exactly where the task's own
    cheapest correct algorithm reads the whole stream, which is the coupled arm. On a decoupled
    arm the retrieval component is one content-addressed lookup, a windowed policy is more
    expensive than the task rather than a shortcut, and the row reads 1.000 by doing the work —
    measured, printed, and neither gated nor counted. Every row is printed for every cell in the
    s5_bind floor block below the table, next to the operative floor and its ratio to the
    informed chance 1/(k-1): with TaskSpec.no_pin closing the state-free reset channel that
    ratio is ~1 on every scored cell, and a cell that drifts off it has an open shortcut.
A task PASSES if majority, recency, first-position and (where defined) strong-recency accuracy are
all well below 0.5 (near the 1/#answers floor). The pointer-map families answer over a much larger
space than 2, so their column is gated against their own chance level instead: no shallow policy
may reach twice 1/#answers. That gate reads the registered spec at its longest eval length; a
rescaled cell (the local sweep runs k=4..8 at L=4..8) carries its own floor rows, and a score
there is read against that cell's operative floor rather than against chance.

  .venv/bin/python scripts/validate_suite.py
"""
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import tasks as TK          # noqa: E402
from factworld.render import Renderer, classify  # noqa: E402  (atomic-token type by prefix: g/v/r/o/...)
from factworld.validity import (  # noqa: E402
    S5_BIND_ADVERSARIES,
    S5_BIND_CHANCE_ROWS,
    S5_BIND_ROWS,
    S5_CHAIN_ADVERSARIES,
    S5_CHAIN_CHANCE_ROWS,
    comm_shallow_accuracy,
    s5_bind_floors,
    s5_bind_operative_floor,
    s5_chain_floors,
    strong_recency_accuracy,
)

N = 500

# The strong recency baseline only has a defined prediction on the give-stream families.
STRONG_REC_FAMILIES = ("binding", "composite")

# The commutative family gets its own four shallow adversaries (initial-only / last-turn-only /
# entity-blind-sum / count-mod-k, factworld.validity.comm_shallow_accuracy); the MAX of the four
# fills the strongrec-style column and folds into the verdict.
COMM_FAMILIES = ("commutative",)

# The pointer-map families get theirs (factworld.validity.s5_chain_floors); the MAX fills the
# same column. DERIVED, not restated: the registered adversaries minus the chance rows, since
# uniform and uniform-over-non-start are what a shortcut has to BEAT, not shortcuts. Registering
# a new adversary in factworld.validity therefore reaches this column with no edit here, and a
# row that cannot set a floor cannot enter the gate either.
S5_CHAIN_FAMILIES = ("s5_chain", "chain")
S5_CHAIN_SHORTCUTS = tuple(n for n in S5_CHAIN_ADVERSARIES if n not in S5_CHAIN_CHANCE_ROWS)

# The mutual-reference family, derived the same way: registering a row in factworld.validity is
# enough to put it in this column, and a row that cannot set a floor cannot enter the gate.
S5_BIND_FAMILIES = ("s5_bind",)
S5_BIND_SHORTCUTS = tuple(n for n in S5_BIND_ADVERSARIES if n not in S5_BIND_CHANCE_ROWS)
# The recency-window rows gate only where the task's own cheapest correct algorithm reads the
# whole stream — the coupled rendering. See the module docstring.
S5_BIND_WINDOW_ROWS = tuple(n for n in S5_BIND_ROWS if n.startswith("window_"))


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
    bind_rows = {}
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
        elif spec.family in S5_CHAIN_FAMILIES:
            # pointer-map rung: the strongest registered shallow policy, gated against this
            # task's own chance level. Rows the stream cannot support are absent, not zero
            # (no events -> no chase, no references -> no ref resolution).
            fl = s5_chain_floors(test, spec.k, has_events=(spec.family == "s5_chain"))
            strongrec = max([fl[n] for n in S5_CHAIN_SHORTCUTS if n in fl], default=0.0)
            ok &= strongrec < 2.0 * floor
            srec_col = f"{strongrec:>10.3f}"
        elif spec.family in S5_BIND_FAMILIES:
            # mutual-reference rung: every registered policy, recomputed from these exact items.
            fl = s5_bind_floors(test, spec.k)
            gated = [n for n in S5_BIND_SHORTCUTS if n in fl
                     and (spec.coupled or n not in S5_BIND_WINDOW_ROWS)]
            strongrec = max([fl[n] for n in gated], default=0.0)
            ok &= strongrec < 0.5
            srec_col = f"{strongrec:>10.3f}"
            bind_rows[name] = (fl, s5_bind_operative_floor(fl, coupled=spec.coupled), gated,
                               spec.eval_lengths[-1], spec.k)
        else:
            srec_col = f"{'—':>10}"
        all_ok &= ok
        print(f"  {name:<22} {distinct:>5} {floor:>6.3f} {majority:>9.3f} {recency:>8.3f} {firstpos:>9.3f} {srec_col}   {'PASS' if ok else 'FLAG'}")
    if bind_rows:
        # The mutual-reference floors sit far above chance, so every row is printed: the
        # operative floor (the max over all registered rows) is what a score is read against,
        # while the gate reads only the rows marked '*' — see the module docstring.
        print(f"\n  s5_bind registered floors (n={N} at eval_lengths[-1]; "
              f"'*' = enters the gate, 'op' = the number a score is read against, "
              f"'op/ch' = op over the informed chance 1/(k-1))")
        print("    " + f"{'task':<24}{'L':>5}" + "".join(f"{r[:10]:>12}" for r in S5_BIND_ROWS)
              + f"{'op':>9}{'op/ch':>8}")
        for name, (fl, op, gated, L, k) in bind_rows.items():
            cells = "".join(
                (f"{fl[r]:>11.3f}{'*' if r in gated else ' '}" if r in fl else f"{'—':>12}")
                for r in S5_BIND_ROWS)
            ratio = op / fl["uniform_non_initial"] if "uniform_non_initial" in fl else None
            print(f"    {name:<24}{L:>5}" + cells + f"{op:>9.3f}"
                  + (f"{ratio:>8.2f}" if ratio is not None else f"{'—':>8}"))
    print(f"\nSUITE VALIDITY: {'PASS — no shallow/recency/position shortcut clears floor on any canonical task' if all_ok else 'FLAG — investigate'}")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
