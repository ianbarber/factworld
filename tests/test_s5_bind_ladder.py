"""The coupling-dose ladder (tasks.CALIBRATION).

Four properties, one per test:

  the top rung IS the composed cell    — same fields, same stream_name, byte-identical items,
                                          so the ladder's full dose is the headline cell and
                                          not a near-copy of it;
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
import sys
from dataclasses import fields, replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld import tasks as TK  # noqa: E402

LADDER = ("s5_bind_v2_rho00", "s5_bind_v2_rho25", "s5_bind_v2_rho50",
          "s5_bind_v2_rho75", "s5_bind_v2_rho100")
L = 192


def test_top_rung_is_the_composed_cell_itself():
    a = TK.generate(TK.CALIBRATION["s5_bind_v2_rho100"], "test", n=30, length=L)
    b = TK.generate(TK.CANONICAL["s5_bind_v2"], "test", n=30, length=L)
    assert [(e.prompt, e.answer) for e in a] == [(e.prompt, e.answer) for e in b]


def test_bottom_rung_is_an_identity_control():
    spec = TK.CALIBRATION["s5_bind_v2_rho00"]
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
        assert spec.stream_name == base.stream_name and spec.no_pin
        assert spec.rho_p == spec.rho_b
        doses.append(spec.rho_p)
        for f in (fl.name for fl in fields(TK.TaskSpec)):
            if f not in ("name", "rho_p", "rho_b", "eval_lengths"):
                assert getattr(spec, f) == getattr(base, f), f"{name}.{f}"
    assert doses == [0.0, 0.25, 0.5, 0.75, 1.0]
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
