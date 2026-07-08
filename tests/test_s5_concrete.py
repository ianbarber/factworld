"""Tests for factworld.s5_concrete — prompt identity, determinism, and score semantics.

The expected system/user strings below were captured VERBATIM from
``python scripts/experiment_s5_framing.py --print-samples`` (L8, example idx 0) BEFORE
the renderings were extracted into factworld/s5_concrete.py.  They pin the prompts so
benchmark results stay comparable with docs/openrouter/s5-*.jsonl.

Run directly:  python3 tests/test_s5_concrete.py
Run with pytest: python3 -m pytest tests/test_s5_concrete.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.s5_concrete import gen_examples, render_prompt, score, FRAMINGS
from factworld.tasks import CANONICAL, _world
from factworld.s5_concrete import gen_problems

# --- expected L8/idx0 prompts, captured from the pre-refactor script ----------

SYS_ABSTRACT = (
    "You are taking a short test. Answer with only the requested value, no explanation. "
    "For 'what role does X have?' answer with only a role token "
    "(r0, r1, r2, r3, or r4) followed by a period. Example: 'r2 .'"
)
SYS_CONCRETE = (
    "You are taking a short test. Answer with only the requested value, no explanation. "
    "The jobs are Manager, Chef, Driver, Clerk, Guard. For 'what job does X have?' "
    "answer with only the job name followed by a period. Example: 'Driver .'"
)
USER_V0_ABSTRACT_L8 = (
    "s0 swaps g0 and g2. s1 swaps g4 and g2. s2 swaps g2 and g4. s3 swaps g1 and g0. "
    "s4 cycles roles: g3 -> g2 -> g4. s5 swaps g3 and g1. "
    "s6 cycles roles: g0 -> g4 -> g2 -> g1. s7 cycles roles: g0 -> g3 -> g1 -> g4. "
    "what role does g2 have?"
)
USER_ABSTRACT_STATED_L8 = (
    "Initially g0 has role r0, g1 has r1, g2 has r2, g3 has r3, g4 has r4. "
    + USER_V0_ABSTRACT_L8
)
USER_CONCRETE_L8 = (
    "Five people — Alice, Bob, Cara, Dan, Eva — each hold one job. "
    "Initially: Alice is Manager, Bob is Chef, Cara is Driver, Dan is Clerk, Eva is Guard. "
    "s0 Alice and Cara swap jobs. s1 Eva and Cara swap jobs. s2 Cara and Eva swap jobs. "
    "s3 Bob and Alice swap jobs. "
    "s4 job rotation: Dan takes Eva's job, Cara takes Dan's job, Eva takes Cara's job. "
    "s5 Dan and Bob swap jobs. "
    "s6 job rotation: Alice takes Bob's job, Eva takes Alice's job, Cara takes Eva's job, "
    "Bob takes Cara's job. "
    "s7 job rotation: Alice takes Eva's job, Dan takes Alice's job, Bob takes Dan's job, "
    "Eva takes Bob's job. "
    "what job does Cara have?"
)


def test_gen_examples_concrete_l8_matches_captured_prompts():
    s, u, g = gen_examples(8, 1, framing="concrete")[0]
    assert s == SYS_CONCRETE
    assert u == USER_CONCRETE_L8
    assert g == "Manager"


def test_gen_examples_abstract_stated_l8_matches_captured_prompts():
    s, u, g = gen_examples(8, 1, framing="abstract_stated")[0]
    assert s == SYS_ABSTRACT
    assert u == USER_ABSTRACT_STATED_L8
    assert g == "r0"


def test_render_prompt_v0_abstract_l8_matches_captured_prompts():
    # floor-control framing (not exposed via gen_examples aliases, used by the script)
    spec = CANONICAL["s5_v1"]
    w, _r, oracle = _world(spec)
    events, agent, gold = gen_problems(spec, w, oracle, 8, 1)[0]
    s, u, g = render_prompt("V0_abstract", events, agent, gold)
    assert s == SYS_ABSTRACT
    assert u == USER_V0_ABSTRACT_L8
    assert g == "r0"


def test_gen_examples_deterministic():
    for framing in ("concrete", "abstract_stated"):
        a = gen_examples(16, 5, framing=framing)
        b = gen_examples(16, 5, framing=framing)
        assert a == b
    # prefix stability: n=2 is a prefix of n=5 (same episode seeds by idx)
    assert gen_examples(16, 5, "concrete")[:2] == gen_examples(16, 2, "concrete")


def test_gen_examples_same_problem_across_framings():
    # same episode -> abstract gold role token maps to the concrete job word
    from factworld.s5_concrete import JOBS
    conc = gen_examples(8, 5, "concrete")
    abst = gen_examples(8, 5, "abstract_stated")
    for (_, _, gj), (_, _, gr) in zip(conc, abst):
        assert JOBS[gr] == gj


def test_gen_examples_rejects_unknown_framing():
    try:
        gen_examples(8, 1, framing="V2_bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown framing")


def test_gen_examples_accepts_internal_framing_names():
    assert gen_examples(8, 1, "V1_concrete") == gen_examples(8, 1, "concrete")
    assert FRAMINGS == ("V0_abstract", "V0_abstract_stated_init", "V1_concrete")


def test_score_semantics():
    # exactly the old script's semantics: content_tokens strips punctuation, keeps case
    assert score("Driver .", "Driver") == {"relaxed": 1, "contains": 1}
    assert score("Driver", "Driver") == {"relaxed": 1, "contains": 1}
    assert score("Driver.", "Driver") == {"relaxed": 1, "contains": 1}
    assert score("driver x", "Driver") == {"relaxed": 0, "contains": 0}  # case-sensitive
    assert score("The answer is Driver.", "Driver") == {"relaxed": 0, "contains": 1}
    assert score("", "Driver") == {"relaxed": 0, "contains": 0}
    assert score("r2 .", "r2") == {"relaxed": 1, "contains": 1}
    assert score("r0 r2", "r2") == {"relaxed": 0, "contains": 1}


def test_script_wrapper_reexports_module():
    # the experiment script must be a thin wrapper over the module (single source of truth)
    import importlib
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    mod = importlib.import_module("experiment_s5_framing")
    import factworld.s5_concrete as s5
    assert mod.render_prompt is s5.render_prompt
    assert mod.score is s5.score
    assert mod.NAMES is s5.NAMES and mod.JOBS is s5.JOBS


# --- stdlib runner -----------------------------------------------------------

def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
