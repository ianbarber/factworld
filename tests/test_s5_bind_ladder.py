"""The coupling-dose ladder (tasks.CALIBRATION).

Five properties, one per test:

  the ladder is ONE item stream        — the skeleton-first sampler draws the coupling variate
                                          once per event slot, so index i is the same world,
                                          the same events and the same queries at every dose;
                                          the five prompts differ only in which sentences say
                                          "at this point", at equal whitespace-token counts;
  the doses are nested                  — the referenced set at a lower dose is a subset of the
                                          set at a higher one, so the ladder is a filtration of
                                          one skeleton and not five unrelated renderings;
  the bottom rung is an identity control — at rho=0 nothing is rendered "at this point", so the
                                          coupled and decoupled readings of an item are the
                                          same string and the dose-response starts at zero;
  the dose is the only thing that moves  — every other field is the composed cell's;
  calibration cells are never scored     — outside CANONICAL, outside REPORTED, resolvable by
                                          name.

Runs with zero dependencies:  python3 tests/test_s5_bind_ladder.py
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import fields, replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld import tasks as TK  # noqa: E402

LADDER = ("s5_bind_v2_lad00", "s5_bind_v2_lad25", "s5_bind_v2_lad50",
          "s5_bind_v2_lad75", "s5_bind_v2_lad100")
L = 192
N_PAIR = 500
_WHEN = re.compile(r"\bat (?:this point|the start)\b")


def _cells(n=N_PAIR):
    return {nm: TK.generate(TK.CALIBRATION[nm], "test", n=n, length=L) for nm in LADDER}


def _body(prompt: str) -> str:
    """The prompt with every temporal phrase blanked — everything the dose must NOT move."""
    return _WHEN.sub("<when>", prompt)


def _dynamic_positions(prompt: str) -> set[int]:
    return {i for i, ph in enumerate(_WHEN.findall(prompt)) if ph == "at this point"}


def test_the_ladder_is_one_item_stream():
    """The pairing the ladder exists for: same skeleton, same stated maps, same queries, and
    prompts that differ only in the temporal phrases — checked on every item of a 500-item
    sample at all five doses."""
    cells = _cells()
    ref = cells[LADDER[-1]]
    for i in range(N_PAIR):
        want_body = _body(ref[i].prompt)
        want_len = len(ref[i].prompt.split())
        want_when = len(_WHEN.findall(ref[i].prompt))
        for nm in LADDER:
            ex = cells[nm][i]
            assert _body(ex.prompt) == want_body, f"{nm}[{i}]: body moved with the dose"
            assert len(ex.prompt.split()) == want_len, f"{nm}[{i}]: prompt length moved"
            assert len(_WHEN.findall(ex.prompt)) == want_when, f"{nm}[{i}]: event count moved"
    # gold is the one thing the dose is allowed to move, and it does move
    moved = sum(len({cells[nm][i].answer for nm in LADDER}) > 1 for i in range(N_PAIR))
    assert moved > 0.9 * N_PAIR, f"only {moved}/{N_PAIR} items change answer across the ladder"


def test_the_doses_are_nested():
    cells = _cells(n=120)
    for i in range(120):
        sets = [_dynamic_positions(cells[nm][i].prompt) for nm in LADDER]
        for lo, hi in zip(sets, sets[1:]):
            assert lo <= hi, f"item {i}: the referenced set is not nested in the dose"


def test_bottom_rung_is_an_identity_control():
    spec = TK.CALIBRATION["s5_bind_v2_lad00"]
    c = TK.generate(spec, "test", n=25, length=L)
    d = TK.generate(replace(spec, coupled=False), "test", n=25, length=L)
    for x, y in zip(c, d):
        assert x.prompt == y.prompt and x.answer == y.answer
        assert x.meta["n_ref"] == 0


def test_the_dose_is_the_only_field_that_moves():
    base = TK.CANONICAL["s5_bind_v2"]
    doses = []
    for name in LADDER:
        spec = TK.CALIBRATION[name]
        assert spec.family == "s5_bind" and spec.coupled and spec.query_arm == "state"
        assert spec.no_pin and spec.rho_p == spec.rho_b
        assert spec.rho_ladder == TK._S5_BIND_LADDER and spec.rho_p in spec.rho_ladder
        doses.append(spec.rho_p)
        for f in (fl.name for fl in fields(TK.TaskSpec)):
            if f not in ("name", "rho_p", "rho_b", "rho_ladder", "stream_name", "eval_lengths"):
                assert getattr(spec, f) == getattr(base, f), f"{name}.{f}"
    assert doses == list(TK._S5_BIND_LADDER)
    # the referenced fraction is the dose, item by item
    for name, rho in zip(LADDER, doses):
        exs = TK.generate(TK.CALIBRATION[name], "test", n=40, length=L)
        share = sum(e.meta["n_ref"] for e in exs) / (40 * L)
        assert abs(share - rho) < 0.05, f"{name}: {share:.3f} referenced, dose {rho}"


def test_calibration_cells_are_generable_and_never_scored():
    for name in LADDER:
        assert name not in TK.CANONICAL and name not in TK.REPORTED
        assert TK.spec_for(name) is TK.CALIBRATION[name]
        exs = TK.generate(TK.CALIBRATION[name], "test", n=10, length=L)
        assert len({e.prompt for e in exs}) == 10
        assert all(e.answer.endswith(".") for e in exs)


def test_the_default_sampler_is_untouched():
    """rho_ladder is opt-in: with it unset the original draw order runs, which is what keeps
    every already-frozen s5_bind stream byte-identical."""
    for reg in (TK.CANONICAL, TK.RETIRED):
        for name, spec in reg.items():
            assert spec.rho_ladder == (), name
    unpaired = replace(TK.CANONICAL["s5_bind_v2"], name="unpaired_probe",
                       stream_name="s5_bind_v2", rho_p=0.5, rho_b=0.5)
    a = TK.generate(unpaired, "test", n=8, length=64)
    b = TK.generate(replace(unpaired, rho_p=1.0, rho_b=1.0), "test", n=8, length=64)
    # the defect the paired sampler fixes, pinned so it cannot be reintroduced silently:
    # under the default sampler the dose is consumed inside the rejection loop, so two doses
    # are different items
    assert sum(_body(x.prompt) == _body(y.prompt) for x, y in zip(a, b)) == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
