"""Regression tests for scripts/render_benchmark.py.

Builds a synthetic history fixture that conforms EXACTLY to the C3 record contract
(every field the runner will emit, all five facets x 3 fake models plus sanity and
floor-control rows), renders it into a temp directory, and checks every artefact:
PNG magic bytes, SVG XML validity, HTML table content, CSV shape, and the
latest-record-wins dedup rule.

Run directly:  python3 tests/test_render_benchmark.py
Run with pytest: python3 -m pytest tests/test_render_benchmark.py

Requires matplotlib (``pip install 'factworld[bench]'``); tests skip without it.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

try:
    import matplotlib  # noqa: F401
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

if HAS_MPL:
    import render_benchmark as RB

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

MODELS = {
    # slug: (tier-ish behavior for the fixture, base accuracy)
    "testlab/model-a": ("frontier_pair", 0.95),
    "testlab/model-b": ("cheap_reasoner", 0.78),
    "testlab/model-c": ("non_reasoning", 0.55),
}


def _record(model, facet, task, length, n, relaxed, *, ts="2026-07-05T10:00:00+00:00",
            effort=None, leg=None, rendering=None, max_new_tokens=64, stop_at=".",
            exact=None, contains=None, last_n=None, empty_rate=0.0, api_errors=0):
    """One C3-conformant history record (no prompt text in examples)."""
    reasoning_on = effort not in (None, "none")
    if reasoning_on:
        max_new_tokens, stop_at = 8192, None  # protocol rule for reasoning-on cells
    k = round(relaxed * n)
    examples = [{"gold": f"g{i % 6}", "pred": f"g{i % 6}" if i < k else "g9",
                 "relaxed": 1 if i < k else 0} for i in range(min(n, 5))]
    return {
        "run_id": "fixture-20260705",
        "ts": ts,
        "git_commit": "deadbeef",
        "suite_version": "v1",
        "model": model,
        "served_models": [model + ":served"],
        "providers": ["FixtureProvider"],
        "facet": facet,
        "task": task,
        "length": length,
        "n": n,
        "settings": {
            "effort": effort,
            "max_new_tokens": max_new_tokens,
            "stop_at": stop_at,
            "rendering": rendering,
            "format_prompt": None,
            "n_shot": 0,
            **({"leg": leg} if leg is not None else {}),
        },
        "metrics": {
            "relaxed": relaxed,
            "exact": exact if exact is not None else max(0.0, relaxed - 0.1),
            "contains": contains if contains is not None else min(1.0, relaxed + 0.02),
            "last_n": last_n,
        },
        "diagnostics": {
            "empty_rate": empty_rate,
            "api_errors": api_errors,
            "finish_reasons": {"stop": n - api_errors, **({"error": api_errors} if api_errors else {})},
        },
        "usage": {
            "prompt_tokens": 200 * n,
            "completion_tokens": (900 if reasoning_on else 8) * n,
            "reasoning_tokens": (850 if reasoning_on else 0) * n,
            "cost_usd_est": round(0.0005 * n, 4),
        },
        "elapsed_s": 12.5,
        "examples": examples,
    }


def make_fixture_history(path):
    """Write a synthetic history covering all five facets x 3 models + sanity/floor rows."""
    recs = []
    for model, (tier, base) in MODELS.items():
        # Facet 1: dose_response @ L16 across efforts.
        if tier == "cheap_reasoner":
            efforts = [None, "low", "medium", "high"]
        elif tier == "frontier_pair":
            efforts = [None, "high"]
        else:
            efforts = [None]  # non_reasoning: one default arm only
        for i, eff in enumerate(efforts):
            acc = min(1.0, base - 0.25 + 0.12 * i)
            recs.append(_record(model, "dose_response", "composite_copy_v1", 16, 100,
                                round(acc, 2),
                                effort="none" if (eff is None and tier != "non_reasoning") else eff))
        # Facet 2: composite_length, efforts none vs high.
        for eff in (["none", "high"] if tier != "non_reasoning" else [None]):
            for j, L in enumerate([16, 64, 128, 256, 512]):
                boost = 0.15 if eff == "high" else 0.0
                acc = max(0.05, min(1.0, base + boost - 0.12 * j))
                recs.append(_record(model, "composite_length", "composite_copy_v1", L, 30,
                                    round(acc, 2), effort=eff))
        # Facet 3: s5_concrete with reasoning.
        for j, L in enumerate([4, 16, 32, 64, 128, 256]):
            acc = max(0.0, min(1.0, base + 0.1 - 0.16 * j))
            recs.append(_record(model, "s5_concrete", "s5", L, 30, round(acc, 2),
                                effort="high" if tier != "non_reasoning" else None,
                                rendering="concrete"))
        # Facet 4: chain_depth with reasoning.
        for j, D in enumerate([4, 8, 12, 16, 24, 32, 48]):
            acc = max(0.0, min(1.0, base + 0.15 - 0.14 * j))
            recs.append(_record(model, "chain_depth", "chain_v1", D, 30, round(acc, 2),
                                effort="high" if tier != "non_reasoning" else None,
                                empty_rate=0.03 * j / 6, api_errors=1 if j == 6 else 0))
        # Facet 5: decomposition legs.
        for leg, acc in [("binding_only", min(1.0, base + 0.05)),
                         ("end_to_end", max(0.0, base - 0.3)),
                         ("scaffolded", min(1.0, base - 0.05))]:
            recs.append(_record(model, "decomposition", "composite_copy_v1", 16, 100,
                                round(acc, 2), leg=leg,
                                effort="high" if tier != "non_reasoning" else None))
        # Sanity rows + floor control (s5 abstract token rendering, reasoning off).
        recs.append(_record(model, "sanity", "recall_copy_v1", 4, 30, 1.0))
        recs.append(_record(model, "sanity", "conflict_v1", 4, 30, round(base, 2)))
        recs.append(_record(model, "floor", "s5", 16, 30, 0.1,
                            rendering="abstract_stated"))
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs


# --- tests --------------------------------------------------------------------

def test_wilson_interval():
    if not HAS_MPL:
        return
    lo, hi = RB.wilson_interval(50, 100)
    assert abs(lo - 0.4038) < 0.002 and abs(hi - 0.5962) < 0.002
    assert RB.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = RB.wilson_interval(30, 30)
    assert hi > 0.999 and 0.85 < lo < 0.90  # 30/30 -> [0.886, 1.0]
    lo, hi = RB.wilson_interval(0, 30)
    assert lo < 1e-9 and 0.0 < hi < 0.15


def test_latest_record_wins():
    if not HAS_MPL:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history.jsonl")
        old = _record("testlab/model-a", "dose_response", "composite_copy_v1", 16, 100,
                      0.10, effort="high", ts="2026-07-01T00:00:00+00:00")
        new = _record("testlab/model-a", "dose_response", "composite_copy_v1", 16, 100,
                      0.90, effort="high", ts="2026-07-05T00:00:00+00:00")
        other = _record("testlab/model-a", "dose_response", "composite_copy_v1", 16, 100,
                        0.50, effort="none", ts="2026-07-02T00:00:00+00:00")
        with open(path, "w") as fh:
            for r in (new, old, other):  # out of order on purpose
                fh.write(json.dumps(r) + "\n")
        recs = RB.load_latest(path)
        assert len(recs) == 2  # high deduped, none kept
        high = [r for r in recs if r["settings"]["effort"] == "high"]
        assert high[0]["metrics"]["relaxed"] == 0.90


def test_headline_scalars():
    if not HAS_MPL:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history.jsonl")
        make_fixture_history(path)
        recs = RB.load_latest(path)
        # model-a: dose_response high arm exists.
        dr, eff = RB.headline_dose_response(recs, "testlab/model-a")
        assert eff == "high" and dr is not None
        # model-c has no high arm -> falls back to best arm.
        _, eff_c = RB.headline_dose_response(recs, "testlab/model-c")
        assert eff_c == "default"
        # Horizons respect the 0.8 threshold and ignore the abstract floor row.
        s5 = RB.headline_horizon(recs, "s5_concrete", "testlab/model-a")
        assert s5 in (4, 16, 32, 64, 128, 256)
        legs = RB.headline_decomposition(recs, "testlab/model-b")
        assert set(legs) == {"binding_only", "end_to_end", "scaffolded"}
        assert all(v is not None for v in legs.values())


def test_render_end_to_end():
    if not HAS_MPL:
        return
    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "history.jsonl")
        out = os.path.join(tmp, "out")
        make_fixture_history(history)
        written = RB.render(history, out)

        expected = ["results.md", "results.csv", "index.html"]
        for base in ["fig_dose_response", "fig_composite_length", "fig_s5_horizon",
                     "fig_chain_depth", "fig_decomposition"]:
            expected += [base + ".png", base + ".svg"]
        for name in expected:
            p = os.path.join(out, name)
            assert os.path.exists(p), f"missing {name}"
            assert p in written

        # PNG magic bytes.
        for name in [n for n in expected if n.endswith(".png")]:
            with open(os.path.join(out, name), "rb") as fh:
                assert fh.read(8) == PNG_MAGIC, f"bad PNG magic in {name}"

        # SVG parses as XML.
        for name in [n for n in expected if n.endswith(".svg")]:
            ET.parse(os.path.join(out, name))

        # HTML contains the sortable table, every model, and inline SVG figures.
        with open(os.path.join(out, "index.html"), encoding="utf-8") as fh:
            page = fh.read()
        assert '<table class="sortable">' in page
        assert "<svg" in page
        for m in MODELS:
            assert m in page

        # Markdown has the three sections and relaxed listed before diagnostics metrics.
        with open(os.path.join(out, "results.md"), encoding="utf-8") as fh:
            md = fh.read()
        for section in ["## Headline", "## Diagnostics per cell", "## Full per-cell results"]:
            assert section in md
        assert md.index("relaxed") < md.index("exact")

        # CSV: header + one row per latest cell, relaxed column always populated.
        with open(os.path.join(out, "results.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(RB.load_latest(history))
        assert all(r["relaxed"] != "" for r in rows)
        assert all(r["model"] in MODELS for r in rows)


if __name__ == "__main__":
    for fn in [test_wilson_interval, test_latest_record_wins,
               test_headline_scalars, test_render_end_to_end]:
        fn()
        print(f"{fn.__name__}: ok")
