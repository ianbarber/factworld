"""Tests for the v2 give-stream sampler (composite_copy_v2 / binding_v2, TaskSpec.last_write_uniform).

The v1 sampler drew every event's object uniformly from the active set, so the queried object's
resolving write sat ~Geometric(1/4) from the stream END at every L — a one-line recency heuristic
(last give's recipient [+ that holder's fact]) scored ~0.33@L16, and L added distractor volume, not
binding depth. The v2 sampler chooses the queried object FIRST and places its last write uniformly
over [floor(0.1*L), L-2]. This file pins BOTH sides of the fix:

  (a) resolving-write distance-from-end is ~Uniform (coarse KS: each quartile holds 15-35%);
  (b) the strong recency heuristic is <= 2x chance on v2 AND still well above it on v1 (contrast pinned);
  (c) no event after the resolving write touches the queried object; position bounds hold;
  (d) generation is deterministic;
  (e) every pre-change v1 spec — now in tasks.RETIRED (issue #11: one clean version per task in
      the scored registry) — is BYTE-IDENTICAL to the goldens captured at git HEAD
      (tests/goldens_prechange.json) — frozen-spec immutability survives retirement;
  (f) the oracle gold equals last-write-wins resolution recomputed from the rendered prompt;
  plus pinned v2 stream hashes so the new specs are frozen from day one.

Run directly:  .venv-api/bin/python tests/test_composite_v2.py
Run with pytest: .venv-api/bin/python -m pytest tests/test_composite_v2.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld import tasks as TK
from factworld.validity import strong_recency_accuracy, strong_recency_pred

GOLDENS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens_prechange.json")

# The canonical renderer grammar for the give-stream and the facts block (see factworld/render.py).
_GIVE_RE = re.compile(r"\bs\d+ gives (o\d+) to (g\d+)\.")
_FACT_RE = re.compile(r"\b(g\d+)'s a0 is (v\d+)\.")

# Distribution-check sample size per (spec, L). 400, not 200: at L16 the support has only 14
# discrete values (uneven 4/3/4/3 quartile bins, expected 0.286/0.214) and the fixed n=200 draw
# happens to put quartile 3 of binding_v2@L16 at 0.135 — pure discreteness + sampling noise (the
# per-position histogram is flat; n=400/800 quartiles are all in [0.20, 0.30]). Generation is
# deterministic, so these checks are exact regressions, not flaky statistics.
N_DIST = 400
LENGTHS = (16, 64)

# v2 stream hashes pinned at introduction (suite 1.1): composite_copy_v2 / binding_v2 are frozen
# specs from day one — same (spec, split, length, idx) -> identical example, forever.
V2_GOLDENS = {
    "composite_copy_v2": {
        16: "844236cf73425f6e00dd6ed1c4b96da4a69b46629e5db794078d244e09d21c71",
        64: "35373f6f89f88190c1de25efa31d5d189c22cce1b0af165aee05a573520e6ec5",
    },
    "binding_v2": {
        16: "0856f87b88042f37ee5f9c9e427971f2be45873bdbca4ef529746651edf05980",
        64: "1cccd3021dfe51151912b1bb5ccb0c9ab6c021d029818040e9fce573aaa92f90",
    },
}


def _hash(examples) -> str:
    return hashlib.sha256("\n".join(f"{e.prompt}\t{e.answer}" for e in examples).encode()).hexdigest()


def _gen(name, length, n=N_DIST):
    # spec_for: v2 names resolve in CANONICAL, retired v1 names in RETIRED (defect contrast)
    return TK.generate(TK.spec_for(name), "test", n=n, length=length)


# --- (e) frozen-spec immutability: pre-change goldens byte-identical -------------------------

def test_v1_goldens_byte_identical():
    with open(GOLDENS) as fh:
        gold = json.load(fh)
    assert gold["captured_at_suite_version"] == "1.0"
    for name, per_len in gold["hashes"].items():
        # canonical specs resolve in CANONICAL; the retired v1 family stays generable +
        # frozen in RETIRED (the goldens cover every pre-change spec)
        spec = TK.spec_for(name)
        # the RNG-stream version must stay pinned at "1.0" for v1 specs even as SUITE_VERSION moves
        assert spec.version == "1.0", f"{name}: stream version drifted to {spec.version}"
        for L, want in per_len.items():
            got = _hash(TK.generate(spec, "test", n=gold["n"], length=int(L)))
            assert got == want, f"{name}@L{L}: frozen-spec immutability VIOLATED"


def test_registry_contract():
    for name in ("composite_copy_v2", "binding_v2"):
        assert name in TK.CANONICAL and name in TK.REPORTED
        assert TK.CANONICAL[name].kind == "benchmark"
        assert TK.CANONICAL[name].last_write_uniform is True
        assert TK.CANONICAL[name].version == "1.1"
    # v2 mirrors v1 on every knob except the sampler (and name/version/kind)
    for v1, v2 in (("composite_copy_v1", "composite_copy_v2"), ("binding_v1", "binding_v2")):
        a, b = TK.RETIRED[v1], TK.CANONICAL[v2]
        skip = {"name", "version", "last_write_uniform", "kind"}
        for f in a.__dataclass_fields__:
            if f not in skip:
                assert getattr(a, f) == getattr(b, f), f"{v2}.{f} != {v1}.{f}"
    assert TK.SUITE_VERSION == "1.1"
    # the v1 family is RETIRED (issue #11): out of CANONICAL/REPORTED, kind demoted,
    # still generable via RETIRED / spec_for (historical reproduction + this file)
    for name in ("binding_v1", "binding_load_v1", "composite_v1",
                 "composite_copy_v1", "composite_copy_scale_v1"):
        assert name not in TK.CANONICAL and name not in TK.REPORTED
        assert name in TK.RETIRED and TK.RETIRED[name].kind == "retired"
        assert not TK.RETIRED[name].last_write_uniform
        assert TK.RETIRED[name].version == "1.0"   # frozen stream, forever
        assert TK.spec_for(name) is TK.RETIRED[name]
    # ONE version per task family in the scored registry
    assert sum(1 for n in TK.CANONICAL if TK.CANONICAL[n].family == "binding") == 1
    assert sum(1 for n in TK.CANONICAL if TK.CANONICAL[n].family == "composite") == 1


def test_v2_goldens_pinned():
    for name, per_len in V2_GOLDENS.items():
        for L, want in per_len.items():
            assert _hash(_gen(name, L, n=25)) == want, f"{name}@L{L}: v2 stream drifted"


# --- (a) resolving-write distance-from-end ~ Uniform ------------------------------------------

def test_position_distribution_uniform():
    for name in ("composite_copy_v2", "binding_v2"):
        for L in LENGTHS:
            exs = _gen(name, L)
            lo, hi = L // 10, L - 2                     # the design support for p
            positions = [e.meta["last_write_pos"] for e in exs]
            assert all(lo <= p <= hi for p in positions), f"{name}@L{L}: p out of [{lo},{hi}]"
            # coarse KS: quartiles of the support each hold 15-35% (exact 25% up to discreteness)
            quart = Counter((p - lo) * 4 // (hi - lo + 1) for p in positions)
            for q in range(4):
                frac = quart[q] / len(exs)
                assert 0.15 <= frac <= 0.35, f"{name}@L{L}: quartile {q} holds {frac:.3f}"
            # distance-from-end therefore scales with L: it reaches deep into the stream
            dists = [L - 1 - p for p in positions]
            assert min(dists) >= 1                       # never the final event
            assert max(dists) >= 0.75 * L                # deep writes actually occur


# --- (b) the strong recency heuristic: ~chance on v2, the known shortcut on v1 -----------------

def test_recency_heuristic_v2_at_chance_v1_pinned():
    pool = TK.CANONICAL["composite_copy_v2"].recall_pool         # 16 -> chance ~1/16
    comp_bound = 2.0 / pool                                       # <= 2x chance = 0.125
    k = TK.CANONICAL["binding_v2"].k                              # 5 agents -> chance 1/5
    bind_bound = 2.0 / k                                          # <= 2x chance = 0.4
    for L in LENGTHS:
        c2 = strong_recency_accuracy(_gen("composite_copy_v2", L), "composite")
        b2 = strong_recency_accuracy(_gen("binding_v2", L), "binding")
        assert c2 <= comp_bound, f"composite_copy_v2@L{L}: recency heuristic {c2:.3f} > {comp_bound}"
        assert b2 <= bind_bound, f"binding_v2@L{L}: recency heuristic {b2:.3f} > {bind_bound}"
    # the CONTRAST: the same heuristic still clears its known level on the v1 sampler
    # (measured 0.325@L16 / 0.225@L64 composite, 0.425@L16 / 0.345@L64 binding at this n; the
    # documented ~0.34/~0.21 figures are from the adversarial review's sample)
    assert strong_recency_accuracy(_gen("composite_copy_v1", 16), "composite") >= 0.25
    assert strong_recency_accuracy(_gen("composite_copy_v1", 64), "composite") >= 0.15
    assert strong_recency_accuracy(_gen("binding_v1", 16), "binding") >= 0.35
    assert strong_recency_accuracy(_gen("binding_v1", 64), "binding") >= 0.30


# --- (c) stream structure: nothing after p writes the queried object --------------------------

def test_no_write_after_resolving_position():
    for name in ("composite_copy_v2", "binding_v2"):
        for L in LENGTHS:
            for e in _gen(name, L):
                gives = _GIVE_RE.findall(e.prompt)
                assert len(gives) == L
                p = e.meta["last_write_pos"]
                writes = [i for i, (o, _) in enumerate(gives) if o == e.meta["obj"]]
                assert writes and max(writes) == p, f"{name}@L{L}: last write not at p"
                assert p <= L - 2, f"{name}@L{L}: resolving write is the final event"


# --- (d) determinism ---------------------------------------------------------------------------

def test_determinism():
    for name in ("composite_copy_v2", "binding_v2"):
        spec = TK.CANONICAL[name]
        for split, L in (("test", 64), ("train", None)):
            a = TK.generate(spec, split, n=30, length=L)
            b = TK.generate(spec, split, n=30, length=L)
            assert [(x.prompt, x.answer, x.meta) for x in a] == \
                   [(x.prompt, x.answer, x.meta) for x in b], f"{name} {split}: nondeterministic"


# --- (f) oracle gold == last-write-wins recomputed from the rendered prompt --------------------

def test_gold_matches_prompt_resolution():
    for L in LENGTHS:
        for e in _gen("composite_copy_v2", L, n=100):
            gives = _GIVE_RE.findall(e.prompt)
            holder = [h for o, h in gives if o == e.meta["obj"]][-1]   # last-write-wins from the prompt
            facts = dict(_FACT_RE.findall(e.prompt))
            assert e.answer == f"{holder} {facts[holder]}."
            assert e.meta["holder"] == holder
        for e in _gen("binding_v2", L, n=100):
            gives = _GIVE_RE.findall(e.prompt)
            holder = [h for o, h in gives if o == e.meta["obj"]][-1]
            assert e.answer == f"{holder}."
    # and the heuristic-adversary itself parses the same grammar (sanity on the gate's baseline)
    ex = _gen("composite_copy_v2", 16, n=1)[0]
    pred = strong_recency_pred(ex.prompt, "composite")
    assert pred is not None and re.fullmatch(r"g\d+ v\d+\.", pred)


def test_train_split_smoke():
    """Shortest train length (L=4): p in [0, 2] is non-degenerate and generation succeeds."""
    for name in ("composite_copy_v2", "binding_v2"):
        exs = TK.generate(TK.CANONICAL[name], "train", n=60)
        assert len(exs) == 60 and all(e.answer for e in exs)
        for e in exs:
            L = e.length
            assert L // 10 <= e.meta["last_write_pos"] <= L - 2


if __name__ == "__main__":
    for fn in [test_v1_goldens_byte_identical, test_registry_contract, test_v2_goldens_pinned,
               test_position_distribution_uniform, test_recency_heuristic_v2_at_chance_v1_pinned,
               test_no_write_after_resolving_position, test_determinism,
               test_gold_matches_prompt_resolution, test_train_split_smoke]:
        fn()
        print(f"{fn.__name__}: ok")
