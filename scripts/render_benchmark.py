"""Render the recurring frontier-benchmark history into blog-ready tables, figures and HTML.

Reads ``results/benchmark/history.jsonl`` (one JSON record per cell, contract C3 — see
scripts/run_frontier_benchmark.py), keeps the LATEST record per
``(model, facet, task, length, {effort, leg, rendering})`` key, and writes into
``docs/benchmark/``:

  results.md    v2 headline (zero_budget + chain/s5 horizons + s5@128 ctok) with
                cleanliness footnotes, the recency-heuristic floor row, a
                "v1 (archived facets)" legacy headline, diagnostics and full
                per-cell markdown tables. Escalated zero-budget cells publish the
                CANONICAL first attempt at the shared base budget; the escalated
                rerun renders as a marked diagnostic.
  results.csv   flat per-cell export (all metrics + diagnostics incl. contract_rate /
                covert_cot_rate / rtok_leak_rate / escalated + usage)
  fig_*.png/.svg  facet figures, one per facet with data (PNG 150 dpi for blog upload,
                  SVG for the HTML page); chain_depth cells past chain_v1's design gate
                  (depth >= k=6, cycle wrap) are excluded from figures and headline horizons
                  and marked INVALID in the tables
  index.html    self-contained static page (inline CSS, inline SVG, sortable table)

Relaxed match is the canonical metric and is listed first everywhere; exact/contains/last_n
are diagnostics. Confidence intervals are Wilson 95% (pure python, no scipy).

Requires matplotlib (``pip install 'factworld[bench]'``); everything else is stdlib.

Usage:
    python scripts/render_benchmark.py \
        --history results/benchmark/history.jsonl --out docs/benchmark/
"""
from __future__ import annotations

import argparse
import csv
import functools
import html
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - environment guard
    print(
        "render_benchmark.py needs matplotlib: pip install 'factworld[bench]'",
        file=sys.stderr,
    )
    raise

# Okabe-Ito colorblind-safe palette (cycled if more models than colors).
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]
# "minimal" is the off-arm substitute for models that cannot disable reasoning
# (Gemini 3 rejects effort=none), so it sits next to "none" on the dose axis.
EFFORT_ORDER = ["default", "none", "minimal", "low", "medium", "high"]
LEG_ORDER = ["binding_only", "end_to_end", "scaffolded"]
HORIZON_THRESHOLD = 0.8
# chain_v1 builds a single k=6 pointer cycle; its design gate requires depth < k
# (factworld/tasks.py: "Depths stay < k so the cycle never wraps"). Cells run at
# depth >= 6 wrapped the cycle (gold == start agent at depths 12/24/48; effective
# difficulty depth mod 6), so they measure the wrapped task, not depth. They stay
# visible in the per-cell/diagnostics tables with an explicit marker but are
# excluded from the chain figure and any headline horizon. Depth scaling reports
# under the `chain_nowrap` facet (scaled no-wrap variant), which reuses the same
# figure code.
CHAIN_CYCLE_K = 6
CHAIN_INVALID_MARK = "INVALID (k=6 cycle wrap — task redesigned as chain_nowrap)"
FIGSIZE = (7.0, 4.4)  # readable at ~700px blog width
DPI = 150

# --- v2 zero-budget headline ---------------------------------------------------
# zero_budget cells run composite_copy_v1 with reasoning off (effort=none, or
# "minimal" where the model cannot disable reasoning) under a hard one-line answer
# contract (settings.contract=true). Cleanliness marks quarantine cells whose
# CANONICAL attempt was not actually "no visible working". Answers are 8-11 tokens,
# so the visible-working line sits at ~3x the answer length:
CTOK_WORKING_LINE = 32.0     # median (per-example) / mean ctok per call above this -> †
RTOK_LEAK_PER_CALL = 2.0     # mean reasoning tokens per call on the published attempt -> †
CAP_ESCAPE_RATE = 0.10       # fraction of calls with ctok > max_new_tokens -> ‡
S5_EFF_LENGTH = 128          # matched efficiency cell: s5_concrete @ L128 (F10)
V2_FACETS = ("zero_budget", "chain_nowrap")  # roster rows with none of these are archived
ZB_HEADLINE_CELLS = [        # (length, leg) in headline column order
    (16, None), (64, None), (16, "binding_only"), (16, "end_to_end"),
]
HEADLINE_COLUMNS = [
    "Model",
    "zero-budget composite @L16 (relaxed)",
    "zero-budget composite @L64",
    "binding_only @L16",
    "replicate @L16 (test-retest)",
    f"chain horizon (chain_nowrap, max depth, relaxed >= {HORIZON_THRESHOLD})",
    f"s5 horizon (max L, relaxed >= {HORIZON_THRESHOLD})",
    "s5@128 ctok",
]
HEURISTIC_LABEL = "recency heuristic (floor)"
ZB_FOOTNOTES = [
    "(*) off-arm ran effort=minimal (model cannot disable reasoning).",
    "(†) visible working on the canonical attempt: median (per-example) or mean "
    f"ctok/call > {CTOK_WORKING_LINE:.0f} (~3x the 8-11 token answers), mean "
    f"rtok/call > {RTOK_LEAK_PER_CALL:.0f} on the published attempt, or the cell "
    "needed a budget escalation — measures short visible working, not in-weights.",
    "(‡) cap-escape: per-example ctok exceeded settings.max_new_tokens on "
    f">{CAP_ESCAPE_RATE:.0%} of calls (the provider did not enforce the cap); token "
    "counts and budget comparisons for those cells are not cap-comparable.",
    "(x.xx @512) escalated diagnostic: the cell was rerun once at an escalated token "
    "budget after majority finish=length; the CANONICAL number is the first attempt "
    "at the shared base budget — the escalated value is a marked diagnostic, not the "
    "headline.",
    f"{HEURISTIC_LABEL}: one-line floor recomputed at render time on the exact "
    "deterministic items — answer the LAST event's recipient plus that holder's "
    "fact (binding leg: the last recipient).",
    "n/a = facet/cell not run for this model; — = run, but no qualifying value.",
]
HORIZON_NOTE = (
    "Horizons marked >=N are censored lower bounds: either the max tested depth/length "
    "still qualifies, or ('budget-censored') the first FAILING cell above N was majority "
    "finish=length — the model ran out of token budget there, not necessarily ability. "
    "Horizons marked (borderline) are threshold calls: the first failing cell's Wilson "
    "CI crosses the 0.8 line (e.g. a 0.72 [0.52, 0.86] cell), so N vs the next tested "
    "length is not statistically resolved.")
EFFICIENCY_NOTE = (
    f"s5@128 ctok: completion tokens per call on the matched s5_concrete L{S5_EFF_LENGTH} "
    "cell (run by every current-roster model). This replaces ctok/solve, which averaged "
    "only over cells a model SOLVED and therefore rewarded models that failed early "
    "(selection bias: the published 2.7x opus-vs-kimi ctok/solve gap is ~1.4x on the "
    "matched cell).")
# v1-only facets whose headline scalars are kept in a separate archived table.
V1_ARCHIVED_FACETS = ("dose_response", "composite_length", "decomposition")
ARCHIVED_COLUMNS = [
    "Model", "dose_response (relaxed)", "composite_length (relaxed @ L512, high)",
    "decomposition (bind / e2e / scaffold)",
]


# --- statistics ---------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for k successes out of n trials (pure python)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _ci(rec) -> tuple[float, float]:
    """Wilson CI on the CANONICAL relaxed value (first attempt for escalated cells)."""
    n = rec.get("n") or 0
    r = canonical_relaxed(rec)
    if r is None or n <= 0:
        return (0.0, 1.0)
    return wilson_interval(round(r * n), n)


# --- canonical attempt (escalation handling, F2) --------------------------------

def escalation_first(rec) -> dict | None:
    """The stored first-attempt aggregates of an escalated cell, or None."""
    esc = rec.get("escalation")
    if rec.get("escalated") and isinstance(esc, dict):
        fa = esc.get("first_attempt")
        if isinstance(fa, dict) and fa.get("relaxed") is not None:
            return fa
    return None


def is_escalated(rec) -> bool:
    return escalation_first(rec) is not None


def canonical_relaxed(rec):
    """Canonical relaxed for a cell. For escalated zero-budget cells the FIRST
    attempt at the shared base budget is canonical (the escalated 512-token rerun
    was a confound against models answering at 96 — it renders as a diagnostic)."""
    fa = escalation_first(rec)
    if fa is not None:
        return fa.get("relaxed")
    return (rec.get("metrics") or {}).get("relaxed")


def _ex_vals(rec, key) -> list:
    return [e[key] for e in rec.get("examples") or [] if e.get(key) is not None]


def canonical_ctok_per_call(rec):
    """Visible completion tokens per call of the CANONICAL attempt: median of the
    per-example ctok when present, else the aggregate mean; for escalated cells the
    first attempt only stores aggregates, so its mean is used."""
    fa = escalation_first(rec)
    n = rec.get("n") or 0
    if fa is not None:
        c = (fa.get("usage") or {}).get("completion_tokens")
        return c / n if n and c is not None else None
    cs = _ex_vals(rec, "ctok")
    if cs:
        return statistics.median(cs)
    c = (rec.get("usage") or {}).get("completion_tokens")
    return c / n if n and c is not None else None


def canonical_rtok_per_call(rec):
    """Mean reasoning tokens per call of the published (canonical) attempt."""
    fa = escalation_first(rec)
    n = rec.get("n") or 0
    if fa is None:
        rs = _ex_vals(rec, "rtok")
        if rs:
            return sum(rs) / len(rs)
        src = rec
    else:
        src = fa
    rt = (src.get("usage") or {}).get("reasoning_tokens")
    return rt / n if n and rt is not None else None


def _effective_cap(rec):
    """The token cap the stored per-example data actually ran under (the escalated
    budget for escalated cells, whose examples come from the rerun)."""
    if rec.get("escalated") and isinstance(rec.get("escalation"), dict):
        return rec["escalation"].get("max_new_tokens") or _settings(rec).get("max_new_tokens")
    return _settings(rec).get("max_new_tokens")


def cap_escape(rec) -> bool:
    """True when per-example ctok exceeds the cell's token cap on > CAP_ESCAPE_RATE
    of calls (the provider did not enforce max_new_tokens) — marked ‡ (F5)."""
    cap = _effective_cap(rec)
    cs = _ex_vals(rec, "ctok")
    if not cap or not cs:
        return False
    return sum(1 for c in cs if c > cap) / len(cs) > CAP_ESCAPE_RATE


def majority_finish_length(rec) -> bool:
    """True when most of the cell's calls ended with finish=length (budget cutoff)."""
    fr = (rec.get("diagnostics") or {}).get("finish_reasons") or {}
    n = rec.get("n") or 0
    if fr and n:
        return fr.get("length", 0) * 2 > n
    fins = _ex_vals(rec, "finish")
    return bool(fins) and sum(1 for f in fins if f == "length") * 2 > len(fins)


def finish_error_count(rec) -> int:
    """Per-example finish=='error' count (12 v2 calls carry it while
    diagnostics.api_errors stays 0 — F8); falls back to the aggregate."""
    per_ex = sum(1 for f in _ex_vals(rec, "finish") if f == "error")
    agg = ((rec.get("diagnostics") or {}).get("finish_reasons") or {}).get("error", 0)
    return max(per_ex, agg)


# --- history loading ----------------------------------------------------------

def _settings(rec) -> dict:
    return rec.get("settings") or {}


def cell_key(rec) -> tuple:
    """Dedup key: (model, facet, task, length, hash of {effort, leg, rendering})."""
    s = _settings(rec)
    arm = json.dumps(
        {"effort": s.get("effort"), "leg": s.get("leg"), "rendering": s.get("rendering")},
        sort_keys=True,
    )
    return (rec.get("model"), rec.get("facet"), rec.get("task"), rec.get("length"), arm)


def load_latest(history_path: str) -> list[dict]:
    """Parse history.jsonl and keep the latest record per cell key (by ts, then file order)."""
    latest: dict[tuple, tuple] = {}
    with open(history_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            order = (rec.get("ts") or "", i)
            key = cell_key(rec)
            if key not in latest or order >= latest[key][0]:
                latest[key] = (order, rec)
    recs = [v[1] for v in latest.values()]
    recs.sort(key=lambda r: (r.get("model") or "", r.get("facet") or "",
                             r.get("task") or "", r.get("length") or 0,
                             json.dumps(_settings(r), sort_keys=True)))
    return recs


def by_facet(records, facet):
    return [r for r in records if r.get("facet") == facet]


def chain_invalid(rec) -> bool:
    """True for chain_depth cells past chain_v1's design gate (depth >= k=6)."""
    return rec.get("facet") == "chain_depth" and (rec.get("length") or 0) >= CHAIN_CYCLE_K


def cell_note(rec) -> str:
    notes = []
    if chain_invalid(rec):
        notes.append(CHAIN_INVALID_MARK)
    if is_escalated(rec):
        notes.append(
            f"escalated @{_effective_cap(rec)} diagnostic "
            f"{_fmt((rec.get('metrics') or {}).get('relaxed'))}; canonical = first "
            f"attempt @{_settings(rec).get('max_new_tokens')}")
    if cap_escape(rec):
        notes.append("‡ cap-escape")
    return "; ".join(notes) or "—"


def archived_roster(records, model) -> bool:
    """True for v1-only roster rows: the model has no record in any v2 facet."""
    return not any(r.get("model") == model and r.get("facet") in V2_FACETS
                   for r in records)


def models_of(records) -> list[str]:
    return sorted({r.get("model") for r in records if r.get("model")})


def arm_label(rec) -> str:
    """Compact human label for the non-key settings of a cell."""
    s = _settings(rec)
    parts = []
    if s.get("leg"):
        parts.append(f"leg={s['leg']}")
    if s.get("rendering"):
        parts.append(f"rendering={s['rendering']}")
    if s.get("contract"):
        parts.append("contract")
    parts.append(f"effort={s.get('effort') or 'default'}")
    return ", ".join(parts)


# --- headline scalars ---------------------------------------------------------

def headline_dose_response(records, model):
    """Relaxed at effort=high, falling back to the best arm if high is absent."""
    cells = [r for r in by_facet(records, "dose_response") if r["model"] == model]
    if not cells:
        return None, ""
    high = [r for r in cells if _settings(r).get("effort") == "high"]
    if high:
        return high[0]["metrics"]["relaxed"], "high"
    best = max(cells, key=lambda r: r["metrics"]["relaxed"] or 0.0)
    return best["metrics"]["relaxed"], _settings(best).get("effort") or "default"


def headline_composite_length(records, model):
    """Relaxed at L512 with effort=high (falls back to the longest high-effort cell)."""
    cells = [r for r in by_facet(records, "composite_length")
             if r["model"] == model and _settings(r).get("effort") == "high"]
    if not cells:
        cells = [r for r in by_facet(records, "composite_length") if r["model"] == model]
    if not cells:
        return None
    at512 = [r for r in cells if r.get("length") == 512]
    pick = at512[0] if at512 else max(cells, key=lambda r: r.get("length") or 0)
    return pick["metrics"]["relaxed"]


def headline_horizon_censored(records, facet, model,
                              exclude_renderings=("abstract_stated",)):
    """(max length with canonical relaxed >= HORIZON_THRESHOLD, censor kind).

    Censor kind: 'n/a' (facet never run), 'tested' (the max TESTED length still
    qualifies), 'budget' (the first FAILING cell above the horizon was majority
    finish=length — from-below censoring, F4), 'borderline' (the first failing
    cell's Wilson CI crosses HORIZON_THRESHOLD, so the cutoff between N and the
    next tested length is not statistically resolved), or None (uncensored)."""
    cells = [r for r in by_facet(records, facet)
             if r["model"] == model
             and _settings(r).get("rendering") not in exclude_renderings
             and not chain_invalid(r)]
    if not cells:
        return None, "n/a"
    good = [r["length"] for r in cells
            if (canonical_relaxed(r) or 0.0) >= HORIZON_THRESHOLD]
    if not good:
        return None, None
    horizon = max(good)
    if horizon == max(r["length"] for r in cells):
        return horizon, "tested"
    failing = [r for r in cells if r["length"] > horizon
               and (canonical_relaxed(r) or 0.0) < HORIZON_THRESHOLD]
    first_fail = min(failing, key=lambda r: r["length"])
    if majority_finish_length(first_fail):
        return horizon, "budget"
    if _ci(first_fail)[1] >= HORIZON_THRESHOLD:
        return horizon, "borderline"
    return horizon, None


def headline_horizon(records, facet, model, exclude_renderings=("abstract_stated",)):
    """Max length with relaxed >= HORIZON_THRESHOLD, or None if no cell qualifies."""
    return headline_horizon_censored(records, facet, model, exclude_renderings)[0]


def _horizon_str(records, facet, model):
    horizon, kind = headline_horizon_censored(records, facet, model)
    if kind == "n/a":
        return "n/a"
    if horizon is None:
        return "—"
    if kind == "tested":
        return f">={horizon}"
    if kind == "budget":
        return f">={horizon} (budget-censored)"
    if kind == "borderline":
        return f"{horizon} (borderline)"
    return str(horizon)


def headline_decomposition(records, model):
    """(binding_only, end_to_end, scaffolded) relaxed triple; None where a leg is missing."""
    cells = [r for r in by_facet(records, "decomposition") if r["model"] == model]
    out = {}
    for leg in LEG_ORDER:
        matches = [r for r in cells if _settings(r).get("leg") == leg]
        out[leg] = matches[0]["metrics"]["relaxed"] if matches else None
    return out


def zero_budget_cell(records, model, length, leg=None):
    """The model's zero_budget cell at (length, leg), or None."""
    cells = [r for r in by_facet(records, "zero_budget")
             if r["model"] == model and r.get("length") == length
             and _settings(r).get("leg") == leg]
    return cells[0] if cells else None


def model_effort_minimal(records, model) -> bool:
    """True when the model's off-arm ran effort=minimal (cannot disable reasoning)."""
    return any(r.get("model") == model and _settings(r).get("effort") == "minimal"
               for r in records)


def zb_marks(rec, model_minimal=False) -> str:
    """Cleanliness marks for one zero-budget headline value: '' / '*' / '†' / '*†'.

    * — the cell (or the model's off-arm generally) ran effort=minimal; also applied
        to missing cells of minimal-only models so the quarantine stays visible.
    † — recalibrated from the CANONICAL attempt's data (F3): median (per-example)
        or mean visible ctok/call > CTOK_WORKING_LINE (~3x the answer length), mean
        rtok/call > RTOK_LEAK_PER_CALL on the published attempt, or the cell needed
        a budget escalation (auto-dagger, F2) — measures short visible working, not
        in-weights.
    """
    marks = ""
    if model_minimal or (rec is not None and _settings(rec).get("effort") == "minimal"):
        marks += "*"
    if rec is not None:
        ctok = canonical_ctok_per_call(rec)
        rtok = canonical_rtok_per_call(rec)
        if (is_escalated(rec)
                or (ctok is not None and ctok > CTOK_WORKING_LINE)
                or (rtok is not None and rtok > RTOK_LEAK_PER_CALL)):
            marks += "†"
    return marks


def zb_value_str(rec, model_minimal=False) -> str:
    """One zero-budget headline cell: canonical value + escalated diagnostic suffix
    + cleanliness marks; 'n/a' when the cell never ran (F9)."""
    if rec is None:
        return "n/a"
    val = _fmt(canonical_relaxed(rec))
    if is_escalated(rec):
        val += (f" ({_fmt((rec.get('metrics') or {}).get('relaxed'))} "
                f"@{_effective_cap(rec)})")
    return val + zb_marks(rec, model_minimal)


def zb_model_marks(records, model) -> str:
    """Union of cleanliness marks over the model's zero_budget cells (figure labels)."""
    minimal = model_effort_minimal(records, model)
    seen = set()
    for r in by_facet(records, "zero_budget"):
        if r["model"] == model:
            seen.update(zb_marks(r, minimal))
    if not seen and minimal:
        seen.add("*")
    return "".join(c for c in "*†" if c in seen)


def headline_efficiency(records, model):
    """s5@128 ctok: completion tokens per call on the matched s5_concrete cell at
    L=S5_EFF_LENGTH — the cell every current-roster model runs, replacing the
    selection-biased ctok/solve (F10). None when the cell never ran."""
    cells = [r for r in by_facet(records, "s5_concrete")
             if r["model"] == model and r.get("length") == S5_EFF_LENGTH
             and _settings(r).get("rendering") != "abstract_stated"]
    if not cells:
        return None
    r = cells[0]
    n = r.get("n") or 0
    ctok = (r.get("usage") or {}).get("completion_tokens")
    return ctok / n if n > 0 and ctok is not None else None


@functools.lru_cache(maxsize=8)
def recency_heuristic(n: int = 100):
    """Zero-budget floor (F7): score the one-line recency heuristic — answer with
    the LAST event's recipient and that holder's stated fact (binding-leg analog:
    the last recipient IS the holder guess) — on the exact deterministic
    composite_copy_v1 items, regenerated locally (pure stdlib, no API).

    Returns {"composite_16", "composite_64", "binding_16"} or None when the
    factworld package is unavailable."""
    try:
        from factworld import tasks as TK
    except Exception:  # pragma: no cover - environment guard
        return None
    spec = TK.CANONICAL["composite_copy_v1"]

    def scores(length):
        comp = bind = 0
        for e in TK.generate(spec, "test", n=n, length=length):
            recips = re.findall(r"gives o\d+ to (g\d+)\.", e.prompt)
            facts = dict(re.findall(r"(g\d+)'s a0 is (v\d+)\.", e.prompt))
            if not recips:
                continue
            last = recips[-1]
            comp += TK.score_relaxed(f"{last} {facts.get(last, '')}", e.answer)
            bind += int(last == e.meta.get("holder"))
        return comp / n, bind / n

    c16, b16 = scores(16)
    c64, _ = scores(64)
    return {"composite_16": c16, "composite_64": c64, "binding_16": b16}


def heuristic_row(records):
    """The 'recency heuristic' row for the zero-budget headline table, or None."""
    n = max((r.get("n") or 0 for r in by_facet(records, "zero_budget")), default=0)
    vals = recency_heuristic(n or 100)
    if vals is None:
        return None
    return [HEURISTIC_LABEL,
            _fmt(vals["composite_16"]), _fmt(vals["composite_64"]),
            _fmt(vals["binding_16"]), _fmt(vals["composite_16"]),
            "—", "—", "—"]


def replicate_noise(records):
    """Max observed |plain@L16 - replicate@L16| across models (F6): the end_to_end
    leg builds prompts IDENTICAL to the plain composite cell, so the pair is a
    test-retest replicate and the max |delta| is the run-to-run noise bar."""
    deltas = []
    for m in models_of(records):
        a = zero_budget_cell(records, m, 16, None)
        b = zero_budget_cell(records, m, 16, "end_to_end")
        if a is not None and b is not None:
            va, vb = canonical_relaxed(a), canonical_relaxed(b)
            if va is not None and vb is not None:
                deltas.append(abs(va - vb))
    return max(deltas) if deltas else None


def replicate_note(records) -> str:
    noise = replicate_noise(records)
    noise_s = "n/a" if noise is None else f"{noise:.2f}"
    return (
        "replicate @L16 (test-retest): the zero_budget end_to_end leg builds prompts "
        "IDENTICAL to the plain composite @L16 cell (same runner path), so the column "
        "is a replicate, not a distinct measurement; max observed |plain - replicate| "
        f"across models = {noise_s} — read that as the run-to-run noise bar on the "
        "headline numbers. Future runs keep this arm intentionally as leg='replicate'.")


def headline_rows(records):
    """One row per model for the primary (v2) headline table, HEADLINE_COLUMNS order.
    v1-only roster rows are labelled '(archived roster)'; cells that never ran say
    'n/a' (distinct from '—' = run but no qualifying value, F9)."""
    rows = []
    for m in models_of(records):
        minimal = model_effort_minimal(records, m)
        label = f"{m} (archived roster)" if archived_roster(records, m) else m
        row = [label]
        for length, leg in ZB_HEADLINE_CELLS:
            r = zero_budget_cell(records, m, length, leg)
            row.append(zb_value_str(r, minimal))
        row.append(_horizon_str(records, "chain_nowrap", m))
        row.append(_horizon_str(records, "s5_concrete", m))
        eff = headline_efficiency(records, m)
        row.append("n/a" if eff is None else f"{eff:.0f}")
        rows.append(row)
    hr = heuristic_row(records)
    if hr is not None:
        rows.append(hr)
    return rows


def archived_headline_rows(records):
    """Legacy v1 headline scalars, only for models with data in an archived facet."""
    rows = []
    for m in models_of(records):
        if not any(r["model"] == m
                   for facet in V1_ARCHIVED_FACETS for r in by_facet(records, facet)):
            continue
        dr, dr_eff = headline_dose_response(records, m)
        cl = headline_composite_length(records, m)
        legs = headline_decomposition(records, m)
        rows.append([m,
                     "—" if dr is None else f"{dr:.2f} @ {dr_eff}",
                     _fmt(cl),
                     " / ".join(_fmt(legs[leg]) for leg in LEG_ORDER)])
    return rows


# --- markdown / csv -----------------------------------------------------------

def _fmt(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def _settings_block(records) -> list[str]:
    """Distinct (effort, max_new_tokens, stop_at) combos observed, from the records."""
    combos = sorted({
        (str(_settings(r).get("effort") or "default"),
         str(_settings(r).get("max_new_tokens")),
         str(_settings(r).get("stop_at")))
        for r in records
    })
    lines = ["## Settings", "",
             "Canonical metric: **relaxed** match. exact / contains / last_n are diagnostics.",
             f"Horizon threshold: relaxed >= {HORIZON_THRESHOLD}.",
             "Error bars / intervals: Wilson 95% CI.", "",
             "Observed generation settings (effort -> max_new_tokens, stop_at):", ""]
    for effort, mnt, stop in combos:
        lines.append(f"- effort={effort}: max_new_tokens={mnt}, stop_at={stop}")
    lines.append("")
    return lines


def _display_path(path: str) -> str:
    """Path relative to the repo when inside it, else the path as given."""
    rel = os.path.relpath(os.path.abspath(path), REPO)
    return path if rel.startswith("..") else rel


def write_results_md(records, out_path, history_path):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# FactWorld frontier benchmark — results",
        "",
        f"Generated {now} from `{_display_path(history_path)}` "
        f"({len(records)} latest cells).",
        "",
    ]
    lines += _settings_block(records)

    lines += ["## Headline", "",
              "Zero-budget cells: composite_copy_v1 with reasoning off (effort=none) under a "
              "one-line answer contract (settings.contract=true); relaxed match. Escalated "
              "cells show the CANONICAL first attempt at the shared base budget, with the "
              "escalated rerun as a parenthesised diagnostic.", "",
              "| " + " | ".join(HEADLINE_COLUMNS) + " |",
              "|" + "---|" * len(HEADLINE_COLUMNS)]
    for row in headline_rows(records):
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    for note in ZB_FOOTNOTES:
        lines += [note, ""]
    lines += [replicate_note(records), "", HORIZON_NOTE, "", EFFICIENCY_NOTE, ""]
    lines += [
        "Chain horizons come from the `chain_nowrap` facet only. `chain_v1` builds a single "
        f"k={CHAIN_CYCLE_K} pointer cycle and measures depth only for depths < k "
        '(`factworld/tasks.py`: "Depths stay < k so the cycle never wraps"); `chain_depth` cells '
        f"at depth >= {CHAIN_CYCLE_K} wrapped the cycle (gold == start agent at depths 12/24/48; "
        "effective difficulty depth mod 6), measure the wrapped task rather than depth, and are "
        f"marked `{CHAIN_INVALID_MARK}` in the tables below and excluded from the chain figure.",
        "",
    ]

    archived = archived_headline_rows(records)
    if archived:
        lines += ["## v1 (archived facets)", "",
                  "Legacy headline columns for the v1-only facets "
                  f"({', '.join(V1_ARCHIVED_FACETS)}); superseded by the zero-budget "
                  "headline above. Per-cell rows remain in the tables below.", "",
                  "| " + " | ".join(ARCHIVED_COLUMNS) + " |",
                  "|" + "---|" * len(ARCHIVED_COLUMNS)]
        for row in archived:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    lines += ["## Diagnostics per cell", "",
              "finish_errors counts per-example finish=='error' calls (surfaced even where "
              "diagnostics.api_errors is 0).", "",
              "| Model | Facet | Task | Length | Arm | empty_rate | api_errors | "
              "finish_errors | reasoning_tokens | finish_reasons | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        d = r.get("diagnostics") or {}
        u = r.get("usage") or {}
        fr = d.get("finish_reasons") or {}
        fr_s = ", ".join(f"{k}:{v}" for k, v in sorted(fr.items())) or "—"
        lines.append(
            f"| {r['model']} | {r['facet']} | {r['task']} | {r.get('length', '—')} | "
            f"{arm_label(r)} | {_fmt(d.get('empty_rate'), 3)} | {d.get('api_errors', '—')} | "
            f"{finish_error_count(r)} | {u.get('reasoning_tokens', '—')} | {fr_s} | "
            f"{cell_note(r)} |")
    lines.append("")

    lines += ["## Full per-cell results", "",
              "relaxed is the CANONICAL value (first attempt for escalated cells; the "
              "escalated diagnostic is in the note column). ‡ = cap-escape (see headline "
              "footnotes).", "",
              "| Model | Facet | Task | Length | Arm | n | relaxed [95% CI] | exact | contains | "
              "last_n | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        mt = r.get("metrics") or {}
        lo, hi = _ci(r)
        lines.append(
            f"| {r['model']} | {r['facet']} | {r['task']} | {r.get('length', '—')} | "
            f"{arm_label(r)} | {r.get('n', '—')} | {_fmt(canonical_relaxed(r))} "
            f"[{lo:.2f}, {hi:.2f}] | {_fmt(mt.get('exact'))} | {_fmt(mt.get('contains'))} | "
            f"{_fmt(mt.get('last_n'))} | {cell_note(r)} |")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


CSV_FIELDS = [
    "run_id", "ts", "git_commit", "suite_version", "model", "served_models", "providers",
    "facet", "task", "length", "n", "effort", "leg", "rendering", "max_new_tokens",
    "stop_at", "format_prompt", "n_shot", "relaxed", "relaxed_ci_lo", "relaxed_ci_hi",
    "exact", "contains", "last_n", "empty_rate", "api_errors", "finish_errors",
    "contract_rate", "covert_cot_rate", "rtok_leak_rate", "rtok_per_call", "escalated",
    "escalated_relaxed", "cap_escape", "finish_reasons",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd_est", "elapsed_s",
    "note",
]


def write_results_csv(records, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            s, mt = _settings(r), r.get("metrics") or {}
            d, u = r.get("diagnostics") or {}, r.get("usage") or {}
            lo, hi = _ci(r)
            w.writerow({
                "run_id": r.get("run_id"), "ts": r.get("ts"),
                "git_commit": r.get("git_commit"), "suite_version": r.get("suite_version"),
                "model": r.get("model"),
                "served_models": ";".join(r.get("served_models") or []),
                "providers": ";".join(r.get("providers") or []),
                "facet": r.get("facet"), "task": r.get("task"),
                "length": r.get("length"), "n": r.get("n"),
                "effort": s.get("effort"), "leg": s.get("leg"),
                "rendering": s.get("rendering"),
                "max_new_tokens": s.get("max_new_tokens"), "stop_at": s.get("stop_at"),
                "format_prompt": s.get("format_prompt"), "n_shot": s.get("n_shot"),
                # relaxed is the CANONICAL value (first attempt for escalated cells);
                # the escalated rerun exports as escalated_relaxed (a diagnostic).
                "relaxed": canonical_relaxed(r),
                "relaxed_ci_lo": round(lo, 4), "relaxed_ci_hi": round(hi, 4),
                "exact": mt.get("exact"), "contains": mt.get("contains"),
                "last_n": mt.get("last_n"),
                "empty_rate": d.get("empty_rate"), "api_errors": d.get("api_errors"),
                "finish_errors": finish_error_count(r),
                "contract_rate": d.get("contract_rate"),
                "covert_cot_rate": d.get("covert_cot_rate"),
                "rtok_leak_rate": d.get("rtok_leak_rate"),
                "rtok_per_call": canonical_rtok_per_call(r),
                "escalated": r.get("escalated"),
                "escalated_relaxed": mt.get("relaxed") if is_escalated(r) else None,
                "cap_escape": cap_escape(r),
                "finish_reasons": json.dumps(d.get("finish_reasons") or {}, sort_keys=True),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": u.get("reasoning_tokens"),
                "cost_usd_est": u.get("cost_usd_est"), "elapsed_s": r.get("elapsed_s"),
                "note": (lambda note: "" if note == "—" else note)(cell_note(r)),
            })


# --- figures ------------------------------------------------------------------

def _color_map(models):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    ax.set_ylim(-0.03, 1.05)


def _caption(fig, cells):
    ns = sorted({r.get("n") for r in cells if r.get("n")})
    n_s = f"n={ns[0]}" if len(ns) == 1 else f"n={ns[0]}–{ns[-1]}" if ns else "n=?"
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fig.text(0.01, 0.005,
             f"{n_s} per cell; relaxed match (canonical); error bars: Wilson 95% CI; {date}",
             fontsize=7, color="#555555")


def _save(fig, out_dir, name):
    png = os.path.join(out_dir, f"{name}.png")
    svg = os.path.join(out_dir, f"{name}.svg")
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]


def _line_ci(ax, xs, cells, color, label, linestyle="-", offset=0.0):
    ys = [canonical_relaxed(r) for r in cells]
    los, his = zip(*[_ci(r) for r in cells])
    # clamp at 0: the stored relaxed mean can differ from the reconstructed k/n
    # by a float epsilon, which would hand matplotlib a negative error bar
    yerr = [[max(0.0, y - lo) for y, lo in zip(ys, los)],
            [max(0.0, hi - y) for y, hi in zip(ys, his)]]
    xs = [x + offset for x in xs]
    ax.errorbar(xs, ys, yerr=yerr, color=color, label=label, linestyle=linestyle,
                marker="o", markersize=4, linewidth=1.4, capsize=2.5, elinewidth=0.9)


def fig_dose_response(records, out_dir):
    cells = by_facet(records, "dose_response")
    if not cells:
        return []
    models = models_of(cells)
    colors = _color_map(models)
    efforts = [e for e in EFFORT_ORDER
               if any((_settings(r).get("effort") or "default") == e for r in cells)]
    xpos = {e: i for i, e in enumerate(efforts)}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, m in enumerate(models):
        mine = sorted((r for r in cells if r["model"] == m),
                      key=lambda r: xpos[_settings(r).get("effort") or "default"])
        if not mine:
            continue
        xs = [xpos[_settings(r).get("effort") or "default"] for r in mine]
        _line_ci(ax, xs, mine, colors[m], m, offset=(i - len(models) / 2) * 0.02)
    ax.set_xticks(range(len(efforts)), efforts)
    ax.set_xlabel("reasoning effort")
    ax.set_ylabel("relaxed accuracy")
    ax.set_title("Dose response: composite_copy_v1 @ L16 vs reasoning effort")
    _style_axes(ax)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out_dir, "fig_dose_response")


def fig_composite_length(records, out_dir):
    cells = by_facet(records, "composite_length")
    if not cells:
        return []
    models = models_of(cells)
    colors = _color_map(models)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for m in models:
        # effort=None (non-reasoning default arms) and "minimal" (Gemini's
        # closest off-arm) are grouped with "none" (dashed).
        for efforts, style in ((("high",), "-"), (("none", "minimal", None), "--")):
            mine = sorted((r for r in cells
                           if r["model"] == m and _settings(r).get("effort") in efforts),
                          key=lambda r: r["length"])
            if not mine:
                continue
            label = m if style == "-" or not any(
                _settings(r).get("effort") == "high" for r in cells if r["model"] == m
            ) else None
            _line_ci(ax, [r["length"] for r in mine], mine, colors[m], label, style)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["length"] for r in cells}),
                  [str(x) for x in sorted({r["length"] for r in cells})])
    ax.set_xlabel("composite length L (log scale)")
    ax.set_ylabel("relaxed accuracy")
    ax.set_title("composite_copy_v1 vs length (solid: effort=high, dashed: reasoning off)")
    _style_axes(ax)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out_dir, "fig_composite_length")


def _horizon_fig(records, out_dir, facet, name, xlabel, title):
    cells = [r for r in by_facet(records, facet)
             if _settings(r).get("rendering") != "abstract_stated"]
    if not cells:
        return []
    models = models_of(cells)
    colors = _color_map(models)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for m in models:
        mine = sorted((r for r in cells if r["model"] == m), key=lambda r: r["length"])
        if not mine:
            continue
        _line_ci(ax, [r["length"] for r in mine], mine, colors[m], m)
        for r in mine:
            y = canonical_relaxed(r) or 0.0
            # F4: hollow marker — the cell FAILS with majority finish=length, so
            # the failure is budget-censored (ran out of tokens, not ability).
            if y < HORIZON_THRESHOLD and majority_finish_length(r):
                ax.plot([r["length"]], [y], marker="o", markersize=6,
                        markerfacecolor="white", markeredgecolor=colors[m],
                        linestyle="none", zorder=5)
            # F5: ‡ — >10% of the cell's calls escaped the token cap.
            if cap_escape(r):
                ax.annotate("‡", (r["length"], y), textcoords="offset points",
                            xytext=(4, 5), fontsize=9, color=colors[m])
    ax.axhline(HORIZON_THRESHOLD, color="#888888", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    xt = sorted({r["length"] for r in cells})
    ax.set_xticks(xt, [str(x) for x in xt])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relaxed accuracy")
    ax.set_title(title)
    _style_axes(ax)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    fig.text(0.01, 0.03,
             "hollow: failing cell majority finish=length (budget-censored); "
             "‡: >10% of calls escaped the token cap",
             fontsize=7, color="#555555")
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out_dir, name)


def fig_s5_horizon(records, out_dir):
    return _horizon_fig(records, out_dir, "s5_concrete", "fig_s5_horizon",
                        "permutation sequence length (log scale)",
                        "S5 concrete rendering: relaxed vs length "
                        f"(dotted: horizon threshold {HORIZON_THRESHOLD})")


def _fig_chain(records, out_dir, facet, task_label):
    """Chain figure over a chain facet; chain_depth cells past the design gate
    (depth >= k=6, cycle wrap) are excluded — they measure the wrapped task."""
    valid = [r for r in records if not chain_invalid(r)]
    return _horizon_fig(valid, out_dir, facet, f"fig_{facet}",
                        "chain depth (log scale)",
                        f"{task_label}: relaxed vs depth "
                        f"(dotted: horizon threshold {HORIZON_THRESHOLD})")


def fig_chain_depth(records, out_dir):
    return _fig_chain(records, out_dir, "chain_depth",
                      f"chain_v1 (depths < k={CHAIN_CYCLE_K})")


def fig_chain_nowrap(records, out_dir):
    return _fig_chain(records, out_dir, "chain_nowrap", "chain_nowrap")


ZB_GROUPS = [  # (bar label, length, leg) — headline zero-budget cells
    ("composite L16", 16, None),
    ("composite L64", 64, None),
    ("binding_only L16", 16, "binding_only"),
    ("replicate L16 (test-retest)", 16, "end_to_end"),
]


def fig_zero_budget(records, out_dir):
    cells = by_facet(records, "zero_budget")
    if not cells:
        return []

    def composite_l64(m):
        r = zero_budget_cell(records, m, 64, None)
        return (canonical_relaxed(r) or 0.0) if r else -1.0

    models = sorted(models_of(cells), key=composite_l64, reverse=True)
    group_colors = {label: PALETTE[j] for j, (label, _, _) in enumerate(ZB_GROUPS)}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    width = 0.2
    for j, (label, length, leg) in enumerate(ZB_GROUPS):
        xs, ys, errs, hatched, escapes = [], [], [[], []], [], []
        for i, m in enumerate(models):
            r = zero_budget_cell(records, m, length, leg)
            if r is None:
                continue
            y = canonical_relaxed(r)  # first attempt for escalated cells (F2)
            lo, hi = _ci(r)
            xs.append(i + (j - 1.5) * width)
            ys.append(y)
            errs[0].append(max(0.0, y - lo))
            errs[1].append(max(0.0, hi - y))
            hatched.append("*" in zb_marks(r, model_effort_minimal(records, m)))
            escapes.append(cap_escape(r))
        if not xs:
            continue
        bars = ax.bar(xs, ys, width=width, color=group_colors[label], label=label,
                      yerr=errs, capsize=2.5, error_kw={"elinewidth": 0.9})
        for rect, flag, esc in zip(bars, hatched, escapes):
            if flag:  # asterisked (effort=minimal) cells get hatched bars
                rect.set_hatch("///")
                rect.set_edgecolor("#555555")
            if esc:   # ‡: >10% of the cell's calls escaped the token cap (F5)
                ax.annotate("‡", (rect.get_x() + rect.get_width() / 2,
                                  rect.get_height()),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=9, color="#333333")
    ax.set_xticks(range(len(models)),
                  [m + zb_model_marks(records, m) for m in models],
                  fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("relaxed accuracy")
    ax.set_title("Zero budget: composite_copy_v1, reasoning off, one-line answer contract")
    _style_axes(ax)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    fig.text(0.01, 0.03,
             "canonical first-attempt values (escalated reruns are table diagnostics); "
             "hatched / *: off-arm ran effort=minimal; †: visible working on the "
             "canonical attempt; ‡: cap-escape; sorted by composite @L64",
             fontsize=7, color="#555555")
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, out_dir, "fig_zero_budget")


def fig_decomposition(records, out_dir):
    cells = by_facet(records, "decomposition")
    if not cells:
        return []
    models = models_of(cells)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    width = 0.26
    leg_colors = dict(zip(LEG_ORDER, PALETTE[:3]))
    for j, leg in enumerate(LEG_ORDER):
        xs, ys, errs = [], [], [[], []]
        for i, m in enumerate(models):
            mine = [r for r in cells if r["model"] == m and _settings(r).get("leg") == leg]
            if not mine:
                continue
            r = mine[0]
            y = r["metrics"]["relaxed"]
            lo, hi = _ci(r)
            xs.append(i + (j - 1) * width)
            ys.append(y)
            errs[0].append(max(0.0, y - lo))
            errs[1].append(max(0.0, hi - y))
        if xs:
            ax.bar(xs, ys, width=width, color=leg_colors[leg], label=leg,
                   yerr=errs, capsize=2.5, error_kw={"elinewidth": 0.9})
    ax.set_xticks(range(len(models)), models, fontsize=7)
    ax.set_ylabel("relaxed accuracy")
    ax.set_title("Decomposition: routing legs (binding_only / end_to_end / scaffolded)")
    _style_axes(ax)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out_dir, "fig_decomposition")


FIGURES = [fig_zero_budget, fig_dose_response, fig_composite_length, fig_s5_horizon,
           fig_chain_depth, fig_chain_nowrap, fig_decomposition]


# --- html ---------------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
 max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.45}
h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:2rem}
table{border-collapse:collapse;font-size:0.78rem;width:100%;margin:0.8rem 0}
th,td{border:1px solid #ddd;padding:3px 6px;text-align:left}
th{background:#f4f4f4;cursor:pointer;user-select:none;white-space:nowrap}
th:hover{background:#e8e8e8}
tr:nth-child(even){background:#fafafa}
figure{margin:1.2rem 0}figure svg{max-width:100%;height:auto}
.small{color:#555;font-size:0.8rem}
"""

_SORT_JS = """
document.querySelectorAll("table.sortable th").forEach(function (th, idx) {
  th.addEventListener("click", function () {
    var table = th.closest("table");
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    var dir = th.dataset.dir === "asc" ? -1 : 1;
    Array.prototype.forEach.call(table.querySelectorAll("th"), function (h) {
      delete h.dataset.dir;
    });
    th.dataset.dir = dir === 1 ? "asc" : "desc";
    rows.sort(function (a, b) {
      var x = a.cells[idx].textContent.trim();
      var y = b.cells[idx].textContent.trim();
      var nx = parseFloat(x), ny = parseFloat(y);
      if (!isNaN(nx) && !isNaN(ny)) return (nx - ny) * dir;
      return x.localeCompare(y) * dir;
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
  });
});
"""


def _inline_svg(svg_path: str) -> str:
    with open(svg_path, encoding="utf-8") as fh:
        text = fh.read()
    # Strip the XML prolog / DOCTYPE so the SVG can be embedded inline.
    start = text.find("<svg")
    return text[start:] if start >= 0 else text


def _html_table(headers, rows, sortable=False):
    cls = ' class="sortable"' if sortable else ""
    out = [f"<table{cls}><thead><tr>"]
    out += [f"<th>{html.escape(h)}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def write_index_html(records, out_dir, svg_paths, history_path):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    head_rows = headline_rows(records)
    archived = archived_headline_rows(records)
    archived_html = ""
    if archived:
        archived_html = (
            "<h2>v1 (archived facets)</h2>\n"
            f'<p class="small">Legacy headline columns for the v1-only facets '
            f"({html.escape(', '.join(V1_ARCHIVED_FACETS))}); superseded by the "
            "zero-budget headline above.</p>\n"
            + _html_table(ARCHIVED_COLUMNS, archived))

    cell_rows = []
    for r in records:
        mt = r.get("metrics") or {}
        d = r.get("diagnostics") or {}
        u = r.get("usage") or {}
        lo, hi = _ci(r)
        cell_rows.append([
            r["model"], r["facet"], r["task"], r.get("length", "—"), arm_label(r),
            r.get("n", "—"), _fmt(canonical_relaxed(r)), f"[{lo:.2f}, {hi:.2f}]",
            _fmt(mt.get("exact")), _fmt(mt.get("contains")), _fmt(mt.get("last_n")),
            _fmt(d.get("empty_rate"), 3), d.get("api_errors", "—"),
            finish_error_count(r), u.get("reasoning_tokens", "—"), cell_note(r),
        ])

    figures_html = "".join(
        f"<figure>{_inline_svg(p)}</figure>" for p in svg_paths if p.endswith(".svg")
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FactWorld frontier benchmark</title>
<style>{_CSS}</style>
</head>
<body>
<h1>FactWorld frontier benchmark</h1>
<p class="small">Generated {now} from <code>{html.escape(os.path.basename(history_path))}</code>
({len(records)} latest cells). Canonical metric: <strong>relaxed</strong> match;
exact / contains / last_n are diagnostics. Intervals: Wilson 95% CI.
Horizons: max length/depth with relaxed &ge; {HORIZON_THRESHOLD}.
Chain horizons come from the <code>chain_nowrap</code> facet only: <code>chain_v1</code> builds a
single k={CHAIN_CYCLE_K} pointer cycle and measures depth only for depths &lt; k
(<code>factworld/tasks.py</code> design gate), so <code>chain_depth</code> cells at depth &ge;
{CHAIN_CYCLE_K} wrapped the cycle, measure the wrapped task rather than depth, and are marked
{html.escape(CHAIN_INVALID_MARK)} below and excluded from the chain figure.</p>
<h2>Headline</h2>
<p class="small">Zero-budget cells: composite_copy_v1 with reasoning off (effort=none) under a
one-line answer contract (settings.contract=true); relaxed match. Escalated cells show the
CANONICAL first attempt at the shared base budget, with the escalated rerun as a
parenthesised diagnostic.</p>
{_html_table(HEADLINE_COLUMNS, head_rows)}
<p class="small">{"<br>".join(html.escape(n) for n in ZB_FOOTNOTES)}<br>
{html.escape(replicate_note(records))}<br>
{html.escape(HORIZON_NOTE)}<br>
{html.escape(EFFICIENCY_NOTE)}</p>
{archived_html}
<h2>Figures</h2>
{figures_html}
<h2>All cells</h2>
<p class="small">Click a column header to sort. relaxed is the canonical value
(first attempt for escalated cells); finish_errors counts per-example finish=='error'
calls even where api_errors is 0.</p>
{_html_table(["Model", "Facet", "Task", "Length", "Arm", "n", "relaxed", "95% CI",
              "exact", "contains", "last_n", "empty_rate", "api_errors",
              "finish_errors", "reasoning_tokens", "note"], cell_rows, sortable=True)}
<script>{_SORT_JS}</script>
</body>
</html>
"""
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out_path


# --- entry point ----------------------------------------------------------------

def render(history_path: str, out_dir: str) -> list[str]:
    """Render every output; returns the list of files written."""
    records = load_latest(history_path)
    if not records:
        raise SystemExit(f"no records in {history_path}")
    os.makedirs(out_dir, exist_ok=True)
    written = []

    md_path = os.path.join(out_dir, "results.md")
    write_results_md(records, md_path, history_path)
    written.append(md_path)

    csv_path = os.path.join(out_dir, "results.csv")
    write_results_csv(records, csv_path)
    written.append(csv_path)

    fig_paths = []
    for fn in FIGURES:
        fig_paths.extend(fn(records, out_dir))
    written.extend(fig_paths)

    written.append(write_index_html(records, out_dir, fig_paths, history_path))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history", default=os.path.join(REPO, "results", "benchmark", "history.jsonl"),
                    help="path to the benchmark history JSONL (contract C3)")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "benchmark"),
                    help="output directory for tables, figures and index.html")
    args = ap.parse_args(argv)
    for path in render(args.history, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
