"""s5_bind — the RETIRED temporal family, kept generable.

The construct ran two structures over one interleaved event stream and ablated the TIME INDEX:
the same reference rendered "at this point" (resolve against the running map) or "at the start"
(against the stated one). It is retired because that ablation cannot identify composition. The
stated structure is by definition the one before any write, so

    (a reference witnesses composition)  <=>  (its referenced cell has been written since the
                                               start)

and the right-hand side is a read-history predicate. Measured on these streams,
P(the two readings differ | write count of the read cell = 0) = 0.000 and = 1.000 at one write,
over 16,386/31,276 and 8,825/19,933 dependency-slice resolutions at the two scored cells. A
composition-free bounded-capacity solver therefore rejects the op-type contrast at type-I rates
at or above the real deficit's power, and the one contrast clean of read history has exactly
zero power. The SOURCE-STRUCTURE family (tests/test_s5_bind_v3.py) replaces it.

What this file pins: the specs are retired and out of every scored path, they stay generable and
byte-identical from retirement on, the reason is recorded where a reader will find it, and the
``chain_max_gap`` steer is gone from the dataclass.

Runs with zero dependencies:  python3 tests/test_s5_bind.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld import tasks as TK  # noqa: E402
from factworld.validity import (  # noqa: E402
    S5_BIND_ADVERSARIES,
    s5_bind_floors,
    s5_bind_operative_floor,
    s5_bind_pin_density,
)

TEMPORAL = ("s5_bind_v2", "s5_bind_v2_state", "s5_bind_v2_bind", "s5_bind_v2_map",
            "s5_bind_local_v2", "s5_bind_local_v2_state")
LADDER = tuple(f"s5_bind_v2_lad{s}" for s in ("00", "25", "50", "75", "100"))

# Pinned at retirement, not at the pre-retirement values: removing TaskSpec.chain_max_gap moved
# these streams. Nothing reproduces against them — the family was never scored and never
# published — and the specs stay generable so the temporal reader and its floor rows keep a cell
# to run on (tests/test_validity.py).
RETIRED_GOLDENS = {
    "s5_bind_v2": {128: "95db0338a0567784", 192: "153edc1c97dd74e5", 256: "7c6e8f1770d25d56"},
    "s5_bind_v2_state": {128: "421f23d9a7c0b923", 192: "1908f2626443fe92",
                         256: "69f5b92ae40ae729"},
    "s5_bind_v2_bind": {128: "ca4458320ac62191", 192: "cf214b641852abf7",
                        256: "907208d3458f8ec6"},
    "s5_bind_v2_map": {128: "464ff5156aedf178", 192: "d1e56e324b005fa1",
                       256: "4d013e8e5680ea10"},
    "s5_bind_local_v2": {48: "03533f8c242232cf", 64: "91a0056a1c2976ba"},
    "s5_bind_local_v2_state": {48: "afd55d6129af10f9", 64: "8a956658af65892a"},
    "s5_bind_v2_lad00": {192: "8d277a3e6a4b98c5"},
    "s5_bind_v2_lad25": {192: "e6048977605805f7"},
    "s5_bind_v2_lad50": {192: "e49086f6f9cff43d"},
    "s5_bind_v2_lad75": {192: "3609674f33b07daa"},
    "s5_bind_v2_lad100": {192: "1ab178a52110f7b3"},
}


def _hash(examples) -> str:
    return hashlib.sha256(
        "\n".join(f"{e.prompt}\t{e.answer}" for e in examples).encode()).hexdigest()[:16]


def test_the_temporal_family_is_retired_and_out_of_every_scored_path():
    for name in TEMPORAL + LADDER:
        assert name not in TK.CANONICAL, f"{name} is still in the scored registry"
        assert name not in TK.CALIBRATION
        spec = TK.RETIRED[name]
        assert spec.kind == "retired" and name not in TK.REPORTED
        assert TK.spec_for(name) is spec
    assert not TK.CALIBRATION, "the ladder calibrated the retired ablation and went with it"


def test_the_retirement_reason_is_recorded_where_a_reader_looks():
    """The instrument-over-frozen-specs rule: a defective version is retired with its reason,
    not silently dropped."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "factworld", "tasks.py")).read()
    block = src[src.index("---- the s5_bind TEMPORAL family"):]
    block = block[:block.index('"s5_bind_v2":')]
    for phrase in ("CANNOT IDENTIFY COMPOSITION", "written since the start",
                   "bounded-capacity", "s5_bind_v3"):
        assert phrase in block, phrase


def test_frozen_stream_goldens():
    for name, per_len in RETIRED_GOLDENS.items():
        spec = TK.RETIRED[name]
        assert tuple(per_len) == tuple(spec.eval_lengths)
        for L, want in per_len.items():
            got = _hash(TK.generate(spec, "test", n=25, length=L))
            assert got == want, f"{name}@L{L}: retired-spec immutability VIOLATED ({got})"


def test_the_chain_max_gap_steer_is_gone():
    """It could not close the block-drop family — the family is continuous in (position, width)
    and non-monotone in both — and it charged the construct's own comparison: the steer followed
    the coupled trajectory only, so the coupled arm's carrier chain ran 11.8 -> 27.6 events while
    the decoupled reading of the same stream stayed at 12.2. What replaces it is a class rule
    (factworld.validity.floor_eligible), not a gate."""
    names = {f.name for f in fields(TK.TaskSpec)}
    assert "chain_max_gap" not in names
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "factworld", "tasks.py")).read()
    assert "chain_max_gap" not in src.replace("``chain_max_gap`` steer", "")


def test_the_retired_cells_still_generate_and_the_reader_still_runs_on_them():
    spec = TK.RETIRED["s5_bind_v2"]
    ex = TK.generate(spec, "test", n=200, length=spec.eval_lengths[0])
    assert len({e.prompt for e in ex}) == len(ex)
    fl = s5_bind_floors(ex, spec.k)
    assert set(fl) & set(S5_BIND_ADVERSARIES)
    assert s5_bind_operative_floor(fl) is not None
    assert s5_bind_pin_density(ex) == 0.0                 # no_pin still holds on the old stream


def _run() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                fails += 1
                print(f"  FAIL  {name}: {exc}")
    print("OK" if not fails else f"{fails} FAILURES")
    return fails


if __name__ == "__main__":
    sys.exit(_run())
