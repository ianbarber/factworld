"""Regression tests for scripts/render_benchmark.py.

Builds a synthetic history fixture that conforms EXACTLY to the C3 record contract
(every field the runner will emit: the five v1 facets plus the v2 zero_budget cells
with contract diagnostics / escalation and the chain_nowrap staircase, x 3 fake
roster models plus one model dropped from the roster, and sanity/floor rows),
renders it into a temp directory, and checks every artefact: PNG magic bytes, SVG
XML validity, HTML table content, CSV shape, the composition headline
(current roster only, instant/thinking regime grouping, escalated-canonical
values, recalibrated † daggers, the composition gap and replicate-noise columns,
the thinking state-stress scores with '⊘ >budget' censoring, ‡ cap-escape,
the recency-heuristic + object-filter floor rows, s5@128 ctok, n/a-vs-—,
finish_errors), the archived-models section, and the latest-record-wins dedup
rule.

RB.CURRENT_ROSTER is patched to the fixture roster (the real one is
factworld.benchmark.MODELS): the headline shows roster models only and
ARCHIVED_MODEL renders in '## Archived models (dropped from the roster)'.

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

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

MODELS = {
    # slug: (tier-ish behavior for the fixture, base accuracy)
    "testlab/model-a": ("frontier_pair", 0.95),
    "testlab/model-b": ("cheap_reasoner", 0.78),
    "testlab/model-c": ("non_reasoning", 0.55),
}
# Model dropped from the roster: in history but not in CURRENT_ROSTER -> renders
# only in the '## Archived models (dropped from the roster)' section + per-cell tables.
ARCHIVED_MODEL = "testlab/model-d"
ALL_MODELS = set(MODELS) | {ARCHIVED_MODEL}

if HAS_MPL:
    import render_benchmark as RB
    # Stand-in for factworld.benchmark.MODELS: the fixture models are the roster.
    RB.CURRENT_ROSTER = frozenset(MODELS)

# Escalation payload mirroring the real runner (aggregates only, no per-example data).
ESCALATION_L64 = {
    "max_new_tokens": 512,
    "length_rate": 0.09,
    "first_attempt": {
        "max_new_tokens": 96,
        "relaxed": 0.38,
        "length_rate": 0.44,
        "diagnostics": {
            "empty_rate": 0.44, "api_errors": 0,
            "finish_reasons": {"stop": 56, "length": 44},
            "contract_rate": 0.56, "covert_cot_rate": 0.03, "rtok_leak_rate": 0.33,
        },
        "usage": {"prompt_tokens": 38934, "completion_tokens": 10712,
                  "reasoning_tokens": 33},
    },
}


def _record(model, facet, task, length, n, relaxed, *, ts="2026-07-05T10:00:00+00:00",
            effort=None, leg=None, rendering=None, max_new_tokens=64, stop_at=".",
            exact=None, contains=None, last_n=None, empty_rate=0.0, api_errors=0,
            contract=None, contract_rate=None, covert_cot_rate=None,
            rtok_leak_rate=None, escalated=None, escalation=None, reasoning_tokens=None,
            ex_ctok=None, ex_rtok=None, ex_finish="stop", n_error_examples=0,
            finish_reasons=None):
    """One C3-conformant history record (no prompt text in examples)."""
    reasoning_on = effort not in (None, "none") and not contract
    if reasoning_on:
        max_new_tokens, stop_at = 8192, None  # protocol rule for reasoning-on cells
    k = round(relaxed * n)
    ctok = ex_ctok if ex_ctok is not None else (900 if reasoning_on else 8)
    rtok = ex_rtok if ex_rtok is not None else 0
    examples = [{"gold": f"g{i % 6}", "pred": f"g{i % 6}" if i < k else "g9",
                 "relaxed": 1 if i < k else 0, "ctok": ctok, "rtok": rtok,
                 "finish": "error" if i < n_error_examples else ex_finish}
                for i in range(min(n, 5))]
    diagnostics = {
        "empty_rate": empty_rate,
        "api_errors": api_errors,
        "finish_reasons": finish_reasons if finish_reasons is not None else
        {"stop": n - api_errors, **({"error": api_errors} if api_errors else {})},
    }
    if contract_rate is not None:  # zero_budget contract diagnostics
        diagnostics.update({
            "contract_rate": contract_rate,
            "covert_cot_rate": covert_cot_rate if covert_cot_rate is not None else 0.0,
            "rtok_leak_rate": rtok_leak_rate if rtok_leak_rate is not None else 0.0,
        })
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
            **({"contract": contract} if contract is not None else {}),
        },
        "metrics": {
            "relaxed": relaxed,
            "exact": exact if exact is not None else max(0.0, relaxed - 0.1),
            "contains": contains if contains is not None else min(1.0, relaxed + 0.02),
            "last_n": last_n,
        },
        "diagnostics": diagnostics,
        "usage": {
            "prompt_tokens": 200 * n,
            "completion_tokens": (900 if reasoning_on else 8) * n,
            "reasoning_tokens": (reasoning_tokens if reasoning_tokens is not None
                                 else (850 if reasoning_on else 0) * n),
            "cost_usd_est": round(0.0005 * n, 4),
        },
        "elapsed_s": 12.5,
        **({"escalated": escalated} if escalated is not None else {}),
        **({"escalation": escalation} if escalation is not None else {}),
        "examples": examples,
    }


def make_fixture_history(path):
    """Write a synthetic history covering the five v1 facets plus the v2 facets
    (zero_budget with contract diagnostics + one escalated cell, chain_nowrap with
    a budget-censored failure and a cap-escaping cell) x 3 models, one archived
    v1-only roster model, and sanity/floor rows."""
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
        # Facet 3: s5_concrete with reasoning (includes the L128 matched
        # efficiency cell — reasoning-on models spend 900 ctok/call, model-c 8;
        # model-b's L256 state-stress cell is majority finish=length -> the
        # headline renders it '⊘ >budget', not a number).
        for j, L in enumerate([4, 16, 32, 64, 128, 256]):
            acc = max(0.0, min(1.0, base + 0.1 - 0.16 * j))
            kw = {}
            if tier == "cheap_reasoner" and L == 256:
                kw.update(finish_reasons={"length": 20, "stop": 10},
                          ex_finish="length")
            recs.append(_record(model, "s5_concrete", "s5", L, 30, round(acc, 2),
                                effort="high" if tier != "non_reasoning" else None,
                                rendering="concrete", **kw))
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
        # Facet 6 (v2): zero_budget — reasoning off under a one-line answer contract.
        # model-c plays the "cannot disable reasoning" role (off-arm = minimal -> *);
        # model-b plays kimi/sonnet:
        #   L64 plain     escalated cell (canonical = first attempt 0.38, the 0.96
        #                 escalated value is a diagnostic; auto-†)
        #   L16 plain     rtok leak (per-example mean rtok 4 > 2 -> †)
        #   binding L16   visible working (per-example median ctok 60 > 32 -> †)
        #   e2e L16       clean (the replicate/test-retest arm)
        zb_effort = "minimal" if tier == "non_reasoning" else "none"
        for L, leg, acc in [(16, None, base - 0.15), (64, None, base - 0.3),
                            (16, "binding_only", base), (16, "end_to_end", base - 0.2)]:
            kw = dict(effort=zb_effort, leg=leg, max_new_tokens=96, stop_at=None,
                      contract=True, contract_rate=0.97, covert_cot_rate=0.02,
                      rtok_leak_rate=0.05, reasoning_tokens=0, escalated=False)
            if tier == "cheap_reasoner":
                if L == 64 and leg is None:
                    kw.update(escalated=True, escalation=ESCALATION_L64)
                    acc = 0.96  # metrics carry the ESCALATED (diagnostic) value
                elif leg is None:
                    kw.update(ex_rtok=4)
                elif leg == "binding_only":
                    kw.update(ex_ctok=60)
            recs.append(_record(model, "zero_budget", "composite_copy_v1", L, 100,
                                round(max(0.05, min(1.0, acc)), 2), **kw))
        # Facet 7 (v2): chain_nowrap staircase (k=2d+1, so d128 is the k=257
        # state-stress cell) — model-a scores 0.96 at d128; model-b's d32 is
        # majority finish=length (plus one per-example finish=error with
        # api_errors=0) and its d128 escapes the token cap (per-example ctok
        # 9000 > 8192 -> ‡, score 0.13‡); model-c's d128 is majority
        # finish=length -> the headline renders '⊘ >budget', not a number.
        for j, D in enumerate([16, 32, 64, 128]):
            drop = 0.03 if tier == "frontier_pair" else 0.25
            acc = max(0.0, min(1.0, base + 0.1 - drop * j))
            kw = dict(effort="high" if tier != "non_reasoning" else None,
                      escalated=False)
            if tier == "cheap_reasoner" and D == 32:
                kw.update(finish_reasons={"length": 20, "stop": 4, "error": 1},
                          ex_finish="length", n_error_examples=1)
            if tier == "cheap_reasoner" and D == 128:
                kw.update(ex_ctok=9000)
            if tier == "non_reasoning" and D == 128:
                kw.update(finish_reasons={"length": 15, "stop": 10},
                          ex_finish="length")
            recs.append(_record(model, "chain_nowrap", "chain_v1", D, 25,
                                round(acc, 2), **kw))
        # Sanity rows + floor control (s5 abstract token rendering, reasoning off).
        recs.append(_record(model, "sanity", "recall_copy_v1", 4, 30, 1.0))
        recs.append(_record(model, "sanity", "conflict_v1", 4, 30, round(base, 2)))
        recs.append(_record(model, "floor", "s5", 16, 30, 0.1,
                            rendering="abstract_stated"))
    # Archived v1-only roster row: no record in any v2 facet.
    recs.append(_record(ARCHIVED_MODEL, "dose_response", "composite_copy_v1", 16, 100,
                        0.50, effort="high"))
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs


def _fixture_records():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history.jsonl")
        make_fixture_history(path)
        return RB.load_latest(path)


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
    recs = _fixture_records()
    # model-a: dose_response high arm exists.
    dr, eff = RB.headline_dose_response(recs, "testlab/model-a")
    assert eff == "high" and dr is not None
    # model-c has no high arm -> falls back to best arm.
    _, eff_c = RB.headline_dose_response(recs, "testlab/model-c")
    assert eff_c == "default"
    # State-stress lookups hit the named cell and ignore the abstract floor row.
    s5 = RB.stress_cell(recs, "s5_concrete", "testlab/model-a", 256)
    assert s5 is not None and s5["settings"]["rendering"] == "concrete"
    assert RB.stress_cell(recs, "s5_concrete", "testlab/model-a", 999) is None
    legs = RB.headline_decomposition(recs, "testlab/model-b")
    assert set(legs) == {"binding_only", "end_to_end", "scaffolded"}
    assert all(v is not None for v in legs.values())
    # Ladder rung 1: recall from the sanity recall_copy_v1 cell (instant regime).
    assert abs(RB.headline_recall(recs, "testlab/model-a") - 1.0) < 1e-9
    assert RB.headline_recall(recs, ARCHIVED_MODEL) is None


def test_escalated_canonical():
    """F2: the first attempt at the base budget is CANONICAL for escalated cells;
    the escalated value renders as a marked diagnostic and the cell auto-daggers."""
    if not HAS_MPL:
        return
    recs = _fixture_records()
    esc = RB.zero_budget_cell(recs, "testlab/model-b", 64, None)
    assert RB.is_escalated(esc)
    assert RB.canonical_relaxed(esc) == 0.38          # NOT the escalated 0.96
    lo, hi = RB._ci(esc)
    wlo, whi = RB.wilson_interval(38, 100)            # CI from the same n
    assert abs(lo - wlo) < 1e-9 and abs(hi - whi) < 1e-9
    assert RB.zb_value_str(esc) == "0.38 (0.96 @512)†"
    assert "escalated @512 diagnostic 0.96" in RB.cell_note(esc)
    assert "canonical = first attempt @96" in RB.cell_note(esc)
    # Non-escalated cells are untouched.
    plain = RB.zero_budget_cell(recs, "testlab/model-a", 64, None)
    assert not RB.is_escalated(plain)
    assert RB.canonical_relaxed(plain) == plain["metrics"]["relaxed"]
    assert RB.zb_value_str(plain) == "0.65"


def test_zero_budget_marks():
    """F3: † recalibrated to the canonical attempt (ctok line 32, rtok/call > 2,
    escalation auto-dagger); * kept for effort=minimal; F9: missing cell -> n/a."""
    if not HAS_MPL:
        return
    recs = _fixture_records()
    # * mark: model-c's off-arm ran effort=minimal (per-cell and model-level).
    assert RB.model_effort_minimal(recs, "testlab/model-c")
    assert not RB.model_effort_minimal(recs, "testlab/model-a")
    cell_c = RB.zero_budget_cell(recs, "testlab/model-c", 16, "binding_only")
    assert RB.zb_marks(cell_c, model_minimal=True) == "*"
    # † via per-example rtok leak (mean 4 > 2) on the published attempt.
    assert RB.zb_marks(RB.zero_budget_cell(recs, "testlab/model-b", 16, None)) == "†"
    # † via visible working (per-example median ctok 60 > 32).
    assert RB.zb_marks(
        RB.zero_budget_cell(recs, "testlab/model-b", 16, "binding_only")) == "†"
    # † via escalation (auto-dagger).
    assert RB.zb_marks(RB.zero_budget_cell(recs, "testlab/model-b", 64, None)) == "†"
    # Clean cells stay unmarked: model-b's replicate arm and everything of model-a.
    assert RB.zb_marks(RB.zero_budget_cell(recs, "testlab/model-b", 16, "end_to_end")) == ""
    cell_a = RB.zero_budget_cell(recs, "testlab/model-a", 16, None)
    assert RB.zb_marks(cell_a) == ""
    # F9: a cell that never ran renders n/a (not —).
    assert RB.zb_value_str(None) == "n/a"


def test_stress_scores_and_censoring():
    """Thinking state-stress cells render plain scores at the named settings
    (chain d128 = staircase k=257, s5 @L256); a majority-finish=length cell
    renders '⊘ >budget' (not measurable at this budget — never a number);
    ‡ propagates on cap-escaping cells; 'n/a' when the cell never ran."""
    if not HAS_MPL:
        return
    recs = _fixture_records()
    assert RB.CHAIN_STRESS_K == 257  # staircase k=2d+1 at d128
    chain = lambda m: RB.stress_value_str(
        RB.stress_cell(recs, "chain_nowrap", m, RB.CHAIN_STRESS_DEPTH))
    s5 = lambda m: RB.stress_value_str(
        RB.stress_cell(recs, "s5_concrete", m, RB.S5_STRESS_LENGTH))
    assert chain("testlab/model-a") == "0.96"
    assert chain("testlab/model-b") == "0.13‡"      # cap-escape mark propagates
    assert chain("testlab/model-c") == "⊘ >budget"  # majority finish=length
    assert chain(ARCHIVED_MODEL) == "n/a"           # facet never run
    assert s5("testlab/model-a") == "0.25"
    assert s5("testlab/model-b") == "⊘ >budget"
    assert s5("testlab/model-c") == "0.00"
    assert RB.stress_value_str(None) == "n/a"


def test_composition_gap_and_replicate_noise():
    """THE headline statistic: gap = binding_only@L16 - composed@L16, rendered
    +0.NN with the input cells' marks propagated; replicate noise = per-model
    |composed - replicate| @L16 rendered ±0.NN; 'n/a' when a cell never ran."""
    if not HAS_MPL:
        return
    recs = _fixture_records()
    assert RB.composition_gap_str(recs, "testlab/model-a") == "+0.15"
    assert RB.composition_gap_str(recs, "testlab/model-b") == "+0.15†"  # † inputs
    assert RB.composition_gap_str(recs, "testlab/model-c") == "+0.15*"  # minimal
    assert RB.composition_gap_str(recs, ARCHIVED_MODEL) == "n/a"
    assert RB.model_replicate_noise_str(recs, "testlab/model-a") == "±0.05"
    assert RB.model_replicate_noise_str(recs, "testlab/model-c") == "±0.05"
    assert RB.model_replicate_noise_str(recs, ARCHIVED_MODEL) == "n/a"


def test_cap_escape_and_finish_errors():
    """F5: ‡ when per-example ctok > max_new_tokens on >10% of calls;
    F8: per-example finish=='error' surfaces even where api_errors is 0."""
    if not HAS_MPL:
        return
    recs = _fixture_records()

    def chain_cell(m, d):
        return next(r for r in RB.by_facet(recs, "chain_nowrap")
                    if r["model"] == m and r["length"] == d)

    escapee = chain_cell("testlab/model-b", 128)   # 9000 ctok vs 8192 cap
    assert RB.cap_escape(escapee)
    assert "‡ cap-escape" in RB.cell_note(escapee)
    assert not RB.cap_escape(chain_cell("testlab/model-a", 128))
    # Escalated cells compare against the ESCALATED budget (their examples come
    # from the rerun), so a 96-token first attempt does not false-positive.
    esc = RB.zero_budget_cell(recs, "testlab/model-b", 64, None)
    assert RB._effective_cap(esc) == 512
    # finish=error in per-example data with diagnostics.api_errors == 0.
    err_cell = chain_cell("testlab/model-b", 32)
    assert err_cell["diagnostics"]["api_errors"] == 0
    assert RB.finish_error_count(err_cell) == 1
    assert RB.finish_error_count(chain_cell("testlab/model-a", 32)) == 0
    # Budget-censored detection prefers the aggregate finish_reasons (full n).
    assert RB.majority_finish_length(err_cell)


def test_efficiency_matched_cell():
    """F10: efficiency = s5@128 ctok/call (the matched cell), not ctok/solve."""
    if not HAS_MPL:
        return
    recs = _fixture_records()
    assert abs(RB.headline_efficiency(recs, "testlab/model-a") - 900.0) < 1e-9
    assert abs(RB.headline_efficiency(recs, "testlab/model-b") - 900.0) < 1e-9
    assert abs(RB.headline_efficiency(recs, "testlab/model-c") - 8.0) < 1e-9
    assert RB.headline_efficiency(recs, ARCHIVED_MODEL) is None  # never ran -> n/a


def test_replicate_noise():
    """F6: |plain@L16 - replicate@L16| max across models = the run-to-run noise bar
    (the fixture separates every pair by exactly 0.05)."""
    if not HAS_MPL:
        return
    recs = _fixture_records()
    noise = RB.replicate_noise(recs)
    assert noise is not None and abs(noise - 0.05) < 1e-6
    note = RB.replicate_note(recs)
    assert "test-retest" in note and "0.05" in note and "IDENTICAL" in note


def test_recency_heuristic():
    """F7: the render-time recency-heuristic floor reproduces the judge-verified
    values on the exact deterministic items (0.34@L16 / 0.21@L64 / binding 0.34).
    The floor is parameterized by the task the zero_budget records used and the
    row is labelled with it."""
    if not HAS_MPL:
        return
    vals = RB.recency_heuristic("composite_copy_v1", 100)
    assert vals is not None
    assert abs(vals["composite_16"] - 0.34) < 1e-9
    assert abs(vals["composite_64"] - 0.21) < 1e-9
    assert abs(vals["binding_16"] - 0.34) < 1e-9
    recs = _fixture_records()
    assert RB.zb_latest_task(recs) == "composite_copy_v1"
    # headline order: [label, recall —, binding@16, composed@16, composed@64,
    # gap —, replicate noise —, chain —, s5 —, ctok —]
    row = RB.heuristic_row(recs)
    assert row == [RB.heuristic_label("composite_copy_v1"),
                   "—", "0.34", "0.34", "0.21", "—", "—", "—", "—", "—"]
    # an unknown task (e.g. v2-task records rendered against an older factworld)
    # degrades to no floor row rather than a wrong-task floor
    assert RB.recency_heuristic("composite_copy_v999", 100) is None


def test_recency_heuristic_v2_floor_near_chance():
    """The uniform-last-write v2 sampler is the shortcut fix: the same one-line
    recency heuristic must collapse toward chance on composite_copy_v2 items.
    Tolerated as a flagged skip until the generator agent lands the v2 spec."""
    if not HAS_MPL:
        return
    import factworld.tasks as TK
    if "composite_copy_v2" not in TK.CANONICAL:
        print("FLAG: composite_copy_v2 not in tasks.CANONICAL yet (generator agent "
              "pending) — v2 floor values unverified", file=sys.stderr)
        return
    v1 = RB.recency_heuristic("composite_copy_v1", 100)
    v2 = RB.recency_heuristic("composite_copy_v2", 100)
    assert v2 is not None
    # v1's geometric(1/4) recency mass put the floor at 0.34/0.21; the v2 uniform
    # placement leaves the last event's recipient ~1/pool correct.
    assert v2["composite_16"] < v1["composite_16"] / 2
    assert v2["composite_64"] < v1["composite_64"] / 2
    assert v2["composite_16"] <= 0.15 and v2["composite_64"] <= 0.15
    assert v2["binding_16"] <= 0.15


def test_object_filter_floor():
    """The object-filter floor: E[1/w] over the exact deterministic items, where w
    is the number of writes to the queried object — the score of filtering the
    stream by the queried object and picking a RANDOM write (no last-write-wins
    resolution). Inherent shallow floor: well above chance at small L, ~1/L decay.
    The binding leg derives from the same items, so its floor equals composite@L16.
    The v2 floor row is what small-L zero-budget cells are read against."""
    if not HAS_MPL:
        return
    v1 = RB.object_filter_floor("composite_copy_v1", 100)
    assert v1 is not None
    assert abs(v1["composite_16"] - 0.2909047619047617) < 1e-9
    assert abs(v1["composite_64"] - 0.06582200301642682) < 1e-9
    assert v1["binding_16"] == v1["composite_16"]
    # unknown task degrades to no floor row rather than a wrong-task floor
    assert RB.object_filter_floor("composite_copy_v999", 100) is None
    # fixture history is v1-task: the row is labelled with the records' task and
    # fills the instant score columns only
    recs = _fixture_records()
    assert RB.object_filter_row(recs) == [
        RB.object_filter_label("composite_copy_v1"),
        "—", "0.29", "0.29", "0.07", "—", "—", "—", "—", "—"]
    import factworld.tasks as TK
    if ("composite_copy_v2" not in TK.CANONICAL
            and "composite_copy_v2" not in getattr(TK, "RETIRED", {})):
        print("FLAG: composite_copy_v2 spec unavailable — v2 object-filter floor "
              "unverified", file=sys.stderr)
        return
    v2 = RB.object_filter_floor("composite_copy_v2", 100)
    assert v2 is not None
    # uniform last-write placement leaves MORE (earlier) writes to the queried
    # object in play than v1's geometric recency mass; both decay ~1/L
    assert abs(v2["composite_16"] - 0.4085119047619049) < 1e-9
    assert abs(v2["composite_64"] - 0.1481572065275625) < 1e-9
    assert v2["binding_16"] == v2["composite_16"]
    assert v2["composite_64"] < v2["composite_16"] / 2


def _append_v2_task_records(path):
    """Append NEWER composite_copy_v2 zero_budget records for model-a (the state
    of history right after the first post-switch benchmark run): plain L16/L64,
    binding_only, and the renamed 'replicate' test-retest leg."""
    v2 = []
    for L, leg, acc in [(16, None, 0.30), (64, None, 0.10),
                        (16, "binding_only", 0.85), (16, "replicate", 0.28)]:
        v2.append(_record("testlab/model-a", "zero_budget", "composite_copy_v2",
                          L, 100, acc, ts="2026-07-09T00:00:00+00:00",
                          effort="none", leg=leg, max_new_tokens=96, stop_at=None,
                          contract=True, contract_rate=0.99, covert_cot_rate=0.0,
                          rtok_leak_rate=0.0, reasoning_tokens=0, escalated=False))
    with open(path, "a", encoding="utf-8") as fh:
        for r in v2:
            fh.write(json.dumps(r) + "\n")


def test_zero_budget_task_versioning():
    """A history mixing v1-task zero-budget cells with newer v2-task records:
    the headline/figure use the LATEST task's records only, columns carry the
    version tag, the floor row follows the task, and the archived v1-task cells
    stay in the per-cell tables."""
    if not HAS_MPL:
        return
    import factworld.tasks as TK
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history.jsonl")
        make_fixture_history(path)
        _append_v2_task_records(path)
        recs = RB.load_latest(path)

        assert RB.zb_latest_task(recs) == "composite_copy_v2"
        assert RB.headline_columns("composite_copy_v2")[3] == \
            "instant: composed @L16 (relaxed, v2)"
        # v1 and v2 cells coexist in the latest records (task is in the dedup key)
        zb_tasks = {r.get("task") for r in RB.by_facet(recs, "zero_budget")}
        assert zb_tasks == {"composite_copy_v1", "composite_copy_v2"}
        # task-filtered lookup + the replicate/end_to_end leg alias
        a16 = RB.zero_budget_cell(recs, "testlab/model-a", 16, None,
                                  task="composite_copy_v2")
        assert a16["metrics"]["relaxed"] == 0.30
        rep = RB.zero_budget_cell(recs, "testlab/model-a", 16, "replicate",
                                  task="composite_copy_v2")
        assert rep["settings"]["leg"] == "replicate"
        rep_v1 = RB.zero_budget_cell(recs, "testlab/model-a", 16, "replicate",
                                     task="composite_copy_v1")
        assert rep_v1["settings"]["leg"] == "end_to_end"  # pre-F6 name still found
        # headline (order: model, recall, binding, composed@16, composed@64,
        # gap, replicate noise, chain, s5, ctok): model-a publishes the v2
        # records; model-b (v1-task only) is n/a
        rows = RB.headline_rows(recs)
        row_a = next(r for r in rows if r[0].startswith("testlab/model-a"))
        assert row_a[2] == "0.85" and row_a[3] == "0.30"
        assert row_a[4] == "0.10"
        assert row_a[5] == "+0.55"   # gap = binding 0.85 - composed 0.30
        assert row_a[6] == "±0.02"   # |composed 0.30 - replicate 0.28|
        row_b = next(r for r in rows if r[0].startswith("testlab/model-b"))
        assert row_b[3] == "n/a" and row_b[4] == "n/a"
        assert row_b[5] == "n/a" and row_b[6] == "n/a"
        note = RB._zb_mixed_task_note(recs, "composite_copy_v2")
        assert "composite_copy_v1" in note and "latest task" in note
        # replicate noise pairs within the latest task only: |0.30 - 0.28|
        assert abs(RB.replicate_noise(recs) - 0.02) < 1e-9
        # floor row follows the records' task; absent (never wrong-task) while the
        # generator agent has not landed the v2 spec
        hr = RB.heuristic_row(recs)
        if "composite_copy_v2" in TK.CANONICAL:
            assert hr is not None and hr[0] == RB.heuristic_label("composite_copy_v2")
        else:
            assert hr is None
            print("FLAG: composite_copy_v2 not in tasks.CANONICAL yet — floor row "
                  "omitted for the v2-task table", file=sys.stderr)
        # full render: v2-tagged headline, mixed-task note, archived v1 cells in
        # the per-cell tables
        out = os.path.join(tmp, "out")
        RB.render(path, out)
        with open(os.path.join(out, "results.md"), encoding="utf-8") as fh:
            md = fh.read()
        assert "task **composite_copy_v2**" in md
        assert "instant: composed @L16 (relaxed, v2)" in md
        assert "use the latest task's records (composite_copy_v2)" in md
        full = md[md.index("## Full per-cell results"):]
        assert "| zero_budget | composite_copy_v1 |" in full   # archived cells kept
        assert "| zero_budget | composite_copy_v2 |" in full


def test_breadth_and_k_fixed_arms():
    """v3 rung settings keys: cells carrying settings.breadth (pool rung) or
    settings.k_fixed (fixed-breadth chain) are SEPARATE arms — own dedup key,
    B=/k_fixed= arm labels, excluded from the canonical headline lookups,
    stress columns and (via canonical_arm) figures — and history renders the
    per-(m,L) breadth floor note."""
    if not HAS_MPL:
        return
    zb = _record("testlab/model-a", "zero_budget", "composite_copy_v2", 16, 100,
                 0.60, ts="2026-07-09T00:00:00+00:00", effort="none", leg=None,
                 max_new_tokens=96, stop_at=None, contract=True, contract_rate=1.0)
    rung = json.loads(json.dumps(zb))
    rung["settings"]["breadth"] = 64          # NEWER record on a different rung:
    rung["ts"] = "2026-07-10T00:00:00+00:00"  # must not dedup/shadow the plain cell
    rung["metrics"]["relaxed"] = 0.20
    stair = _record("testlab/model-a", "chain_nowrap", "chain_v1", 16, 25, 0.95,
                    effort="high")
    fixed = _record("testlab/model-a", "chain_nowrap", "chain_v1", 128, 25, 0.95,
                    effort="high")
    fixed["settings"]["k_fixed"] = 257
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in (zb, rung, stair, fixed):
                fh.write(json.dumps(r) + "\n")
        recs = RB.load_latest(path)
        # dedup: every rung survives as its own arm
        assert len(recs) == 4
        # helpers: canonical vs rung arms
        assert RB.breadth_of(zb) == RB.CANONICAL_BREADTH == 16
        assert RB.breadth_of(rung) == 64 and not RB.canonical_arm(rung)
        assert RB.k_fixed_of(fixed) == 257 and not RB.canonical_arm(fixed)
        assert RB.canonical_arm(zb) and RB.canonical_arm(stair)
        # arm labels name the rung
        assert "B=64" in RB.arm_label(rung) and "B=" not in RB.arm_label(zb)
        assert "k_fixed=257" in RB.arm_label(fixed)
        # headline lookups stay canonical: the newer B=64 record neither shadows
        # nor replaces the plain zero-budget cell...
        cell = RB.zero_budget_cell(recs, "testlab/model-a", 16, None,
                                   task="composite_copy_v2")
        assert cell["metrics"]["relaxed"] == 0.60
        # ...and the d128 fixed-k cell must not feed the headline chain column
        # (stress_cell reads canonical staircase arms only -> n/a at d128)
        assert RB.stress_cell(recs, "chain_nowrap", "testlab/model-a",
                              RB.CHAIN_STRESS_DEPTH) is None
        assert RB.stress_cell(recs, "chain_nowrap", "testlab/model-a", 16) is not None
        # full render: rung rows stay in the per-cell tables, labelled, with the
        # per-(m, L) floor note
        out = os.path.join(tmp, "out")
        RB.render(path, out)
        with open(os.path.join(out, "results.md"), encoding="utf-8") as fh:
            md = fh.read()
        assert "B=64" in md and "k_fixed=257" in md
        assert RB.BREADTH_FLOOR_NOTE in md         # per-(m, L) floor note rendered
        head = md[md.index("## Headline"):md.index("## Diagnostics per cell")]
        assert "B=64" not in head                  # rungs never enter the headline
        with open(os.path.join(out, "results.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert {"breadth", "k_fixed"} <= set(rows[0])
        assert {r["breadth"] for r in rows} == {"", "64"}
        assert {r["k_fixed"] for r in rows} == {"", "257"}


def test_object_filter_floor_at_per_rung():
    """Per-rung floor hook: E[1/w] moves with m and L only — recomputed at a pool
    rung it stays FLAT (the pool changes lookup candidates, not the queried
    object's write count) while moving ~1/L in length; the row label is keyed by
    (task, L, m) and says so."""
    if not HAS_MPL:
        return
    # canonical rungs reproduce the headline floor values exactly
    f16 = RB.object_filter_floor_at("composite_copy_v2", 16)
    f64 = RB.object_filter_floor_at("composite_copy_v2", 64)
    assert abs(f16 - 0.4085119047619049) < 1e-9
    assert abs(f64 - 0.1481572065275625) < 1e-9
    assert RB.object_filter_floor_at("composite_copy_v2", 16,
                                     breadth=RB.CANONICAL_BREADTH) == f16
    # flat across pool rungs at fixed (m, L) — up to item-sampling noise...
    f16_b64 = RB.object_filter_floor_at("composite_copy_v2", 16, breadth=64)
    f64_b64 = RB.object_filter_floor_at("composite_copy_v2", 64, breadth=64)
    assert abs(f16_b64 - f16) < 0.06
    assert abs(f64_b64 - f64) < 0.04
    # ...while moving with L (the ~1/L decay survives at every rung)
    assert f64_b64 < f16_b64 / 2
    # unknown task degrades to None (no wrong-task floor), like the other floors
    assert RB.object_filter_floor_at("composite_copy_v999", 16) is None
    # labels are per (task, L, m) and flag the flatness across rungs
    label = RB.object_filter_floor_label_at("composite_copy_v2", 64)
    assert label == ("object-filter floor (composite_copy_v2 @L64, m=4; "
                     "flat across pool rungs)")


def test_archived_roster():
    """Roster split: models not in the current roster (factworld.benchmark.MODELS,
    patched to the fixture roster) are archived — headline excludes them and they
    render in the archived-models section with the v1-facet columns."""
    if not HAS_MPL:
        return
    recs = _fixture_records()
    assert RB.archived_roster(recs, ARCHIVED_MODEL)
    for m in MODELS:
        assert not RB.archived_roster(recs, m)
    assert RB.archived_models(recs) == [ARCHIVED_MODEL]
    assert set(RB.roster_models(recs)) == set(MODELS)
    assert RB.archived_model_rows(recs) == [
        [ARCHIVED_MODEL, "0.50 @ high", "—", "— / — / —"]]
    # no mixing: the headline has a row per roster model + the two floor rows
    rows = RB.headline_rows(recs)
    assert [r[0] for r in rows] == sorted(MODELS) + [
        RB.heuristic_label("composite_copy_v1"),
        RB.object_filter_label("composite_copy_v1")]


def test_profile_values_and_figure():
    """fig_profiles inputs: one profile per CURRENT-ROSTER model over the six
    axes (binding, composed@L16, gap inverted, chain d128, s5 @L256, s5@128 ctok
    inverted); marks carried from the input cells; ⊘ budget-censored and
    never-ran cells are gaps ('censored'/'missing' — never zeros); norms are
    roster min-max positions with inverted axes flipped so 1.0 = roster-best."""
    if not HAS_MPL:
        return
    recs = _fixture_records()
    values = RB.profile_normalized(RB.profile_values(recs))
    # roster only — the archived model gets no panel
    assert set(values) == set(MODELS)
    labels = RB.PROFILE_AXIS_LABELS
    assert labels == ["binding @L16", "composed @L16", "gap (inv)",
                      "chain d128", "s5 @L256", "s5@128 ctok (inv)"]
    a, b, c = (values[f"testlab/model-{s}"] for s in "abc")
    # instant cells carry the headline marks (model-b binding †; model-c *)
    assert b["binding @L16"]["marks"] == "†"
    assert c["binding @L16"]["marks"] == "*"
    assert a["binding @L16"]["marks"] == ""
    # the gap inherits marks from either input cell and renders signed
    assert b["gap (inv)"]["marks"] == "†" and b["gap (inv)"]["display"] == "+0.15"
    # censored cells are gaps, not zeros: model-b s5 @L256 and model-c chain d128
    # are majority finish=length in the fixture
    assert b["s5 @L256"]["status"] == "censored"
    assert "norm" not in b["s5 @L256"]
    assert c["chain d128"]["status"] == "censored"
    # ‡ cap-escape carries on stress cells (model-b chain d128, 9000 > 8192)
    assert b["chain d128"]["marks"] == "‡"
    # every measurable norm is a [0, 1] roster position...
    for d in values.values():
        for cell in d.values():
            if cell["status"] == "ok":
                assert 0.0 <= cell["norm"] <= 1.0
    # ...with inverted axes flipped: model-c's 8-ctok efficiency cell is the
    # roster-best (norm 1.0), the 900-ctok reasoners the worst (0.0)
    assert abs(c["s5@128 ctok (inv)"]["norm"] - 1.0) < 1e-9
    assert abs(a["s5@128 ctok (inv)"]["norm"] - 0.0) < 1e-9
    # binding stays score-ordered: model-a (0.95+0.05->1.0 capped) >= b >= c
    assert (a["binding @L16"]["norm"] >= b["binding @L16"]["norm"]
            >= c["binding @L16"]["norm"])
    # the figure itself renders both formats
    with tempfile.TemporaryDirectory() as tmp:
        paths = RB.fig_profiles(recs, tmp)
        assert [os.path.basename(p) for p in paths] == \
            ["fig_profiles.png", "fig_profiles.svg"]
        with open(paths[0], "rb") as fh:
            assert fh.read(8) == PNG_MAGIC
        ET.parse(paths[1])


def test_render_end_to_end():
    if not HAS_MPL:
        return
    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "history.jsonl")
        out = os.path.join(tmp, "out")
        make_fixture_history(history)
        written = RB.render(history, out)

        expected = ["results.md", "results.csv", "index.html"]
        for base in ["fig_zero_budget", "fig_profiles", "fig_dose_response",
                     "fig_composite_length", "fig_s5_horizon", "fig_chain_depth",
                     "fig_chain_nowrap", "fig_decomposition"]:
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
        for m in ALL_MODELS:
            assert m in page
        assert "finish_errors" in page
        # regime grouping header row + roster split + floor note mirror in HTML
        assert "Archived models (dropped from the roster)" in page
        assert "v1 archived facets (pre-redesign)" in page
        assert '<th colspan="6">instant (reasoning off, answer contract)</th>' in page
        assert '<th colspan="3">thinking</th>' in page
        assert "composition instrument" in page
        assert "object-filter floor" in page
        assert "Read small-L zero-budget cells against the object-filter floor" in page
        assert "horizon" not in page.lower()  # banned vocabulary

        # Markdown has the five sections and relaxed listed before diagnostics metrics.
        with open(os.path.join(out, "results.md"), encoding="utf-8") as fh:
            md = fh.read()
        for section in ["## Headline (current roster)",
                        "## Archived models (dropped from the roster)",
                        "## v1 archived facets (pre-redesign)",
                        "## Diagnostics per cell", "## Full per-cell results"]:
            assert section in md
        assert md.index("relaxed") < md.index("exact")

        # Composition headline: regime-prefixed columns — recall (sanity), state
        # tracking, composed @L16/@L64, the composition gap, replicate noise,
        # then the thinking state-stress scores at named settings (with the
        # task-version tag read from the records — this fixture is a v1-task
        # history), cleanliness marks + footnotes, ⊘ censoring, s5@128 ctok.
        cols = RB.headline_columns("composite_copy_v1")
        assert cols == [
            "Model",
            "instant: recall (sanity, recall_copy_v1)",
            "instant: state tracking (binding_only @L16, v1)",
            "instant: composed @L16 (relaxed, v1)",
            "instant: composed @L64 (v1)",
            "instant: composition gap (binding_only - composed @L16)",
            "instant: replicate noise (|composed - replicate| @L16)",
            "thinking: chain d128 (chain_nowrap, k=257, relaxed)",
            "thinking: s5 @L256 (s5_concrete, relaxed)",
            "thinking: s5@128 ctok",
        ]
        for col in cols:
            assert col in md, f"missing headline column {col!r}"
        assert "task **composite_copy_v1**" in md  # headline names the zb task
        assert "composition instrument" in md      # two-regime framing note
        assert "composition deficit" in md         # the gap footnote
        assert "end_to_end @L16" not in md.split("## Diagnostics per cell")[0]
        assert "off-arm ran effort=minimal" in md
        assert "visible working on the canonical attempt" in md
        assert "cap-escape" in md
        assert "⊘ >budget" in md                   # censored state-stress cells
        assert "not measurable at this budget" in md
        assert "run-to-run noise bar" in md        # replicate noise note (max 0.05)
        assert "0.05" in md
        # banned vocabulary is gone from every label, note and footnote
        for word in ("horizon", "wall", "knee", "cliff"):
            assert word not in md.lower(), f"banned word {word!r} in results.md"
        head = md[md.index("## Headline"):
                  md.index("## Archived models (dropped from the roster)")]
        # 'dose' terminology is purged from all active labels/prose (the historical
        # facet name survives only in the archived tables below).
        assert "dose" not in head.lower()
        row_a = next(l for l in head.splitlines() if l.startswith("| testlab/model-a |"))
        assert row_a.split("|")[-2].strip().isdigit()  # s5@128 ctok renders numerically
        assert row_a.split("|")[2].strip() == "1.00"   # ladder rung 1: sanity recall
        row_b = next(l for l in head.splitlines() if l.startswith("| testlab/model-b |"))
        assert "†" in row_b                        # recalibrated dagger
        assert "0.38 (0.96 @512)†" in row_b        # escalated cell: canonical + diagnostic
        row_c = next(l for l in head.splitlines() if l.startswith("| testlab/model-c |"))
        assert "*" in row_c                        # effort=minimal quarantine mark
        row_a_clean = row_a.replace("| testlab/model-a |", "|")
        assert "†" not in row_a_clean and "*" not in row_a_clean
        # Dropped-roster model: NOT in the headline (no mixing) — it renders in the
        # archived-models section with its v1-facet columns.
        assert ARCHIVED_MODEL not in head
        dropped = md[md.index("## Archived models (dropped from the roster)"):
                     md.index("## v1 archived facets (pre-redesign)")]
        row_d = next(l for l in dropped.splitlines()
                     if l.startswith(f"| {ARCHIVED_MODEL} |"))
        assert "0.50 @ high" in row_d
        # ...and the pre-redesign facet table holds roster models only.
        prered = md[md.index("## v1 archived facets (pre-redesign)"):
                    md.index("## Diagnostics per cell")]
        assert ARCHIVED_MODEL not in prered
        # Floor rows computed at render time on the exact items of the task the
        # records used (labelled with the task version): the recency heuristic and
        # the object-filter floor (E[1/w]), with the read-small-L-against-floor note.
        heur = next(l for l in head.splitlines()
                    if l.startswith(f"| {RB.heuristic_label('composite_copy_v1')} |"))
        assert "0.34" in heur and "0.21" in heur
        objf = next(l for l in head.splitlines()
                    if l.startswith(f"| {RB.object_filter_label('composite_copy_v1')} |"))
        assert "0.29" in objf and "0.07" in objf
        assert "Read small-L zero-budget cells against the object-filter floor" in head
        # Diagnostics table surfaces finish_errors and the ‡ per-cell note.
        diag = md[md.index("## Diagnostics per cell"):md.index("## Full per-cell results")]
        assert "finish_errors" in diag
        assert "‡ cap-escape" in md
        # Full per-cell table shows the CANONICAL relaxed for the escalated cell.
        full = md[md.index("## Full per-cell results"):]
        esc_row = next(l for l in full.splitlines()
                       if l.startswith("| testlab/model-b | zero_budget") and "| 64 |" in l)
        assert "0.38 [" in esc_row and "escalated @512 diagnostic 0.96" in esc_row

        # CSV: header + one row per latest cell, canonical relaxed always populated,
        # plus the v2 diagnostics columns on zero_budget rows.
        with open(os.path.join(out, "results.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(RB.load_latest(history))
        assert all(r["relaxed"] != "" for r in rows)
        assert all(r["model"] in ALL_MODELS for r in rows)
        for col in ["contract_rate", "covert_cot_rate", "rtok_leak_rate", "rtok_per_call",
                    "escalated", "escalated_relaxed", "cap_escape", "finish_errors"]:
            assert col in rows[0], f"missing csv column {col!r}"
        zb_rows = [r for r in rows if r["facet"] == "zero_budget"]
        assert zb_rows and all(r["contract_rate"] != "" for r in zb_rows)
        esc_rows = [r for r in zb_rows if r["escalated"] == "True"]
        assert esc_rows and esc_rows[0]["relaxed"] == "0.38"       # canonical
        assert esc_rows[0]["escalated_relaxed"] == "0.96"          # diagnostic
        err_rows = [r for r in rows if r["facet"] == "chain_nowrap"
                    and r["model"] == "testlab/model-b" and r["length"] == "32"]
        assert err_rows[0]["finish_errors"] == "1"
        assert err_rows[0]["api_errors"] == "0"
        cap_rows = [r for r in rows if r["facet"] == "chain_nowrap"
                    and r["model"] == "testlab/model-b" and r["length"] == "128"]
        assert cap_rows[0]["cap_escape"] == "True"
        v1_rows = [r for r in rows if r["facet"] == "dose_response"]
        assert all(r["contract_rate"] == "" for r in v1_rows)  # v1 cells stay blank


if __name__ == "__main__":
    for fn in [test_wilson_interval, test_latest_record_wins, test_headline_scalars,
               test_escalated_canonical, test_zero_budget_marks,
               test_stress_scores_and_censoring,
               test_composition_gap_and_replicate_noise,
               test_cap_escape_and_finish_errors,
               test_efficiency_matched_cell, test_replicate_noise,
               test_recency_heuristic, test_recency_heuristic_v2_floor_near_chance,
               test_object_filter_floor, test_zero_budget_task_versioning,
               test_breadth_and_k_fixed_arms, test_object_filter_floor_at_per_rung,
               test_archived_roster, test_profile_values_and_figure,
               test_render_end_to_end]:
        fn()
        print(f"{fn.__name__}: ok")
