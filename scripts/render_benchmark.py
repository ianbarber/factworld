"""Render the recurring frontier-benchmark history into blog-ready tables, figures and HTML.

Reads ``results/benchmark/history.jsonl`` (one JSON record per cell, contract C3 — see
scripts/run_frontier_benchmark.py), keeps the LATEST record per
``(model, facet, task, length, {effort, leg, rendering})`` key, and writes into
``docs/benchmark/``:

  results.md    composition headline for the CURRENT roster
                (factworld.benchmark.MODELS) — 'instant (reasoning off, answer
                contract)' columns (recall sanity, state tracking, composed @L16/
                @L64, the COMPOSITION GAP, replicate noise) plus 'thinking'
                state-stress scores (chain d128 at k=257, s5 @L256) and s5@128
                ctok — with cleanliness footnotes, the recency-heuristic and
                object-filter floor rows, an "Archived models (dropped from the
                roster)" section for models no longer on the roster, a "v1
                archived facets (pre-redesign)" legacy headline, diagnostics and
                full per-cell markdown tables. Escalated zero-budget cells
                publish the CANONICAL first attempt at the shared base budget;
                the escalated rerun renders as a marked diagnostic.
  results.csv   flat per-cell export (all metrics + diagnostics incl. contract_rate /
                covert_cot_rate / rtok_leak_rate / escalated + usage)
  fig_*.png/.svg  facet figures, one per facet with data, plus fig_profiles — a
                  small-multiples panel per roster model showing its normalized
                  position on each axis (binding, composed@L16, gap inverted,
                  chain d128, s5 @L256, s5@128 ctok inverted; missing/⊘ cells are
                  gaps, not zeros) — (PNG 150 dpi for blog upload,
                  SVG for the HTML page); chain_depth cells past chain_v1's design gate
                  (depth >= k=6, cycle wrap) are excluded from figures and the headline
                  columns and marked INVALID in the tables
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
# (Gemini 3 rejects effort=none), so it sits next to "none" on the effort axis.
EFFORT_ORDER = ["default", "none", "minimal", "low", "medium", "high"]
LEG_ORDER = ["binding_only", "end_to_end", "scaffolded"]
REF_THRESHOLD = 0.8  # dotted reference line in the score-vs-depth/length figures
# chain_v1 builds a single k=6 pointer cycle; its design gate requires depth < k
# (factworld/tasks.py: "Depths stay < k so the cycle never wraps"). Cells run at
# depth >= 6 wrapped the cycle (gold == start agent at depths 12/24/48; effective
# difficulty depth mod 6), so they measure the wrapped task, not depth. They stay
# visible in the per-cell/diagnostics tables with an explicit marker but are
# excluded from the chain figure and any headline column. Depth scaling reports
# under the `chain_nowrap` facet (scaled no-wrap variant), which reuses the same
# figure code.
CHAIN_CYCLE_K = 6
CHAIN_INVALID_MARK = "INVALID (k=6 cycle wrap — task redesigned as chain_nowrap)"
FIGSIZE = (7.0, 4.4)  # readable at ~700px blog width
DPI = 150

# --- v3 working-set-breadth settings keys ---------------------------------------
# Cells may carry settings["breadth"] (the pool rung B: the runner executes
# CANONICAL[task].scaled(k=2*B, recall_pool=B)) and chain cells settings["k_fixed"]
# (chain_v1.scaled(k=k_fixed), fixed breadth instead of the k=2d+1 staircase).
# Both are sentinel-dropped at their canonical values (B=16 == composite_copy_v2's
# recall_pool / no k_fixed), so records WITHOUT the keys are canonical cells.
# Non-canonical rungs are distinct arms: they get their own dedup key + arm label
# and are EXCLUDED from the canonical headline columns and figures
# (they will publish under their own breadth tables at Checkpoint 1).
CANONICAL_BREADTH = 16
BREADTH_FLOOR_NOTE = (
    "Per-rung floors for breadth cells: the object-filter floor E[1/w] moves with "
    "m (active objects) and L only — w counts writes to the QUERIED object, which "
    "the pool size does not touch — so the floor is FLAT across pool rungs at a "
    "given (m, L); rows are labelled per (task, L, m), not per rung.")


def breadth_of(rec) -> int:
    """The cell's pool rung B (settings.breadth; CANONICAL_BREADTH when absent)."""
    return _settings(rec).get("breadth") or CANONICAL_BREADTH


def k_fixed_of(rec):
    """The cell's fixed chain cycle size, or None (staircase / non-chain cells)."""
    return _settings(rec).get("k_fixed")


def canonical_arm(rec) -> bool:
    """True for cells on the canonical rungs (no breadth override, no fixed-k
    chain): only these feed the headline columns and figures."""
    return breadth_of(rec) == CANONICAL_BREADTH and not k_fixed_of(rec)

# --- v2 zero-budget headline ---------------------------------------------------
# zero_budget cells run a composite_copy task with reasoning off (effort=none, or
# "minimal" where the model cannot disable reasoning) under a hard one-line answer
# contract (settings.contract=true). The TASK VERSION is read from the records:
# the registry switched the facet from composite_copy_v1 to composite_copy_v2
# (uniform last-write placement — v1's recency shortcut fix), so a history can mix
# v1-task and v2-task zero-budget cells. The headline/figure use the LATEST task's
# records only (see ``zb_latest_task``) and label the columns with that task;
# older-task zero-budget cells remain in the per-cell tables.
# Cleanliness marks quarantine cells whose
# CANONICAL attempt was not actually "no visible working". Answers are 8-11 tokens,
# so the visible-working line sits at ~3x the answer length:
CTOK_WORKING_LINE = 32.0     # median (per-example) / mean ctok per call above this -> †
RTOK_LEAK_PER_CALL = 2.0     # mean reasoning tokens per call on the published attempt -> †
CAP_ESCAPE_RATE = 0.10       # fraction of calls with ctok > max_new_tokens -> ‡
S5_EFF_LENGTH = 128          # matched efficiency cell: s5_concrete @ L128 (F10)
# Thinking state-stress cells: chain d128 on the chain_nowrap staircase (k=2d+1,
# so d128 IS the fixed-breadth k=257 cell across the roster) and s5 @L256.
CHAIN_STRESS_DEPTH = 128
CHAIN_STRESS_K = 2 * CHAIN_STRESS_DEPTH + 1  # staircase k=2d+1 -> 257
S5_STRESS_LENGTH = 256
# Instant component-stress cells beyond the composite headline (facets
# recall_load / chain_instant): single-query deferred recall under working-set
# load (the pool scales with the length) and the chain d16 off arm of the
# staircase (the same items as the thinking d16 cell). Floors are first-class
# rows: the uniform guess over the recall pool and over the staircase agent set.
RECALL_LOAD_LENGTH = 64                       # recall_copy_v1 @L64, pool == L
CHAIN_INSTANT_DEPTH = 16
CHAIN_INSTANT_K = 2 * CHAIN_INSTANT_DEPTH + 1  # staircase k=2d+1 -> 33
CENSORED_CELL = "⊘ >budget"  # majority finish=length: not measurable at this budget
# The runner's test-retest leg was renamed end_to_end -> replicate (review F6);
# "replicate" headline lookups also accept the pre-rename leg name in old records.
REPLICATE_LEG_ALIASES = ("replicate", "end_to_end")
# (length, leg) score columns of the instant regime: state tracking (binding)
# before the composed cells; the replicate arm feeds the noise column instead.
ZB_HEADLINE_CELLS = [
    (16, "binding_only"), (16, None), (64, None),
]


def _current_roster() -> frozenset:
    """Slugs of the CURRENT benchmark roster (factworld.benchmark.MODELS). The
    headline shows these models only; any other model in history is archived
    (dropped from the roster) and renders in its own section — no mixing."""
    try:
        from factworld.benchmark import MODELS
    except Exception:  # pragma: no cover - environment guard
        return frozenset()
    return frozenset(MODELS)


CURRENT_ROSTER = _current_roster()


def _task_ver(zb_task) -> str:
    """Short version tag of a zero-budget task name ('composite_copy_v2' -> 'v2')."""
    return (zb_task or "").rsplit("_", 1)[-1] if zb_task else "?"


# Header-row regouping of headline_columns: (group label, colspan). The two
# regimes are the owner's framing: 'instant' = reasoning off, one-line answer
# contract (in-weights composition); 'thinking' = generous reasoning budget.
HEADLINE_GROUPS = [("", 1),
                   ("instant (reasoning off, answer contract)", 6),
                   ("thinking", 3)]
COMPOSITION_NOTE = (
    "The benchmark is a composition instrument: recall and state tracking are the "
    "component abilities, and 'instant' cells (reasoning off, hard one-line answer "
    "contract) measure whether the model composes them in-weights — the composition "
    "gap column is the deficit. 'thinking' cells measure composition with reasoning: "
    "~ceiling at canonical settings for this roster, so the state-stress columns "
    f"(chain d{CHAIN_STRESS_DEPTH} at k={CHAIN_STRESS_K}, s5 @L{S5_STRESS_LENGTH}) "
    "carry the thinking discrimination.")


def headline_columns(zb_task) -> list[str]:
    """Headline column headers, regime-prefixed, composition-statistic order:
    recall (sanity) -> state tracking -> composed @L16/@L64 -> composition gap ->
    replicate noise -> thinking state-stress scores -> efficiency. The instant
    columns carry the task version of the records they were built from (the
    LATEST zero-budget task in history)."""
    ver = _task_ver(zb_task)
    return [
        "Model",
        "instant: recall (sanity, recall_copy_v1)",
        f"instant: state tracking (binding_only @L16, {ver})",
        f"instant: composed @L16 (relaxed, {ver})",
        f"instant: composed @L64 ({ver})",
        "instant: composition gap (binding_only - composed @L16)",
        "instant: replicate noise (|composed - replicate| @L16)",
        f"thinking: chain d{CHAIN_STRESS_DEPTH} (chain_nowrap, k={CHAIN_STRESS_K}, relaxed)",
        f"thinking: s5 @L{S5_STRESS_LENGTH} (s5_concrete, relaxed)",
        "thinking: s5@128 ctok",
    ]


HEURISTIC_LABEL = "recency heuristic (floor"  # completed with the task by heuristic_label
OBJECT_FILTER_LABEL = "object-filter floor"   # completed with the task by object_filter_label


def heuristic_label(zb_task) -> str:
    """Floor-row label, versioned by the task its items were regenerated from."""
    return f"{HEURISTIC_LABEL}, {zb_task})"


def object_filter_label(zb_task) -> str:
    """Object-filter floor-row label, versioned like heuristic_label."""
    return f"{OBJECT_FILTER_LABEL} ({zb_task})"
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
    f"{HEURISTIC_LABEL}, <task>): one-line floor recomputed at render time on the "
    "exact deterministic items of the task named in the row label (the same task "
    "as the zero-budget columns) — answer the LAST event's recipient plus that "
    "holder's fact (binding leg: the last recipient).",
    f"{OBJECT_FILTER_LABEL} (<task>): E[1/w] recomputed at render time on the same "
    "exact items — for each item, 1/(number of writes to the queried object): a "
    "reader that filters events by the queried object but picks a RANDOM write "
    "(no last-write-wins resolution) scores this with no state tracking at all; "
    "the binding leg derives from the same items, so its floor is the same 1/w.",
    "n/a = facet/cell not run for this model; — = run, but no qualifying value.",
]
FLOOR_NOTE = (
    "Read small-L zero-budget cells against the object-filter floor, not chance: "
    "the floor is inherent to last-write-wins (filter the stream to the queried "
    "object, guess among its w writes) and decays only ~1/L, so it sits well above "
    "chance at L16 — a score near the floor row shows object filtering, not state "
    "tracking; genuine last-write resolution has to clear it.")
GAP_NOTE = (
    "composition gap = state tracking (binding_only @L16) - composed @L16, marks from "
    "either input cell propagated. recall|holder is ~1.0 for every roster model (the "
    "scaffolded leg), so if composition were free the composed cell would match the "
    "binding leg; the gap is the composition deficit.")
CENSOR_NOTE = (
    f"{CENSORED_CELL} = not measurable at this budget: the cell's calls were majority "
    "finish=length (the token budget ran out before an answer), so the cell has no "
    "score at these settings.")
EFFICIENCY_NOTE = (
    f"s5@128 ctok: completion tokens per call on the matched s5_concrete L{S5_EFF_LENGTH} "
    "cell (run by every current-roster model). This replaces ctok/solve, which averaged "
    "only over cells a model SOLVED and therefore rewarded models that failed early "
    "(selection bias: the published 2.7x opus-vs-kimi ctok/solve gap is ~1.4x on the "
    "matched cell).")
# v1-only facets whose headline scalars are kept in separate archived tables
# (the historical facet name dose_response stays in the archived headers only —
# 'dose' terminology is purged from all active labels/prose).
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
    """Dedup key: (model, facet, task, length, hash of {effort, leg, rendering,
    breadth, k_fixed}) — every breadth/fixed-k rung is its own arm (records
    without the v3 keys read as the canonical rung)."""
    s = _settings(rec)
    arm = json.dumps(
        {"effort": s.get("effort"), "leg": s.get("leg"), "rendering": s.get("rendering"),
         "breadth": s.get("breadth"), "k_fixed": s.get("k_fixed")},
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
    """True when the model is NOT on the current roster (factworld.benchmark.MODELS):
    it renders in the '## Archived models (dropped from the roster)' section, never
    in the headline. An empty roster (factworld unavailable) archives nothing."""
    return bool(CURRENT_ROSTER) and model not in CURRENT_ROSTER


def models_of(records) -> list[str]:
    return sorted({r.get("model") for r in records if r.get("model")})


def roster_models(records) -> list[str]:
    """Models in history that are on the current roster (headline rows)."""
    return [m for m in models_of(records) if not archived_roster(records, m)]


def archived_models(records) -> list[str]:
    """Models in history that were dropped from the roster (archived section)."""
    return [m for m in models_of(records) if archived_roster(records, m)]


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
    if s.get("breadth"):  # non-canonical pool rung (absent == B=16, canonical)
        parts.append(f"B={s['breadth']}")
    if s.get("k_fixed"):  # fixed-breadth chain (vs the k=2d+1 staircase)
        parts.append(f"k_fixed={s['k_fixed']}")
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


def stress_cell(records, facet, model, length,
                exclude_renderings=("abstract_stated",)):
    """The model's thinking state-stress cell at (facet, length) — canonical arm
    only (breadth/fixed-k rungs are separate arms), abstract-token floor rendering
    and wrapped chain_depth cells excluded. None when the cell never ran."""
    cells = [r for r in by_facet(records, facet)
             if r["model"] == model and r.get("length") == length
             and _settings(r).get("rendering") not in exclude_renderings
             and not chain_invalid(r)
             and canonical_arm(r)]
    return cells[0] if cells else None


def stress_value_str(rec) -> str:
    """One thinking state-stress headline cell: the plain relaxed score at the
    named setting, ‡ when the cell escaped the token cap; CENSORED_CELL when the
    cell's calls were majority finish=length (not measurable at this budget —
    never a number); 'n/a' when the cell never ran."""
    if rec is None:
        return "n/a"
    if majority_finish_length(rec):
        return CENSORED_CELL
    val = _fmt(canonical_relaxed(rec))
    if cap_escape(rec):
        val += "‡"
    return val


def headline_decomposition(records, model):
    """(binding_only, end_to_end, scaffolded) relaxed triple; None where a leg is missing."""
    cells = [r for r in by_facet(records, "decomposition") if r["model"] == model]
    out = {}
    for leg in LEG_ORDER:
        matches = [r for r in cells if _settings(r).get("leg") == leg]
        out[leg] = matches[0]["metrics"]["relaxed"] if matches else None
    return out


def zb_latest_task(records):
    """The task version the headline zero-budget columns publish: the task of the
    NEWEST zero_budget record (ties broken by task name, so v2 wins over v1 at
    equal ts). A history that mixes v1-task cells with v2-task cells publishes
    the latest task's cells only; the older task's cells stay in the per-cell
    tables. None when the facet never ran."""
    cells = by_facet(records, "zero_budget")
    if not cells:
        return None
    newest = max(cells, key=lambda r: (r.get("ts") or "", r.get("task") or ""))
    return newest.get("task")


def _zb_mixed_task_note(records, zb_task) -> str:
    """A headline note when history mixes zero-budget task versions, else ''."""
    others = sorted({r.get("task") for r in by_facet(records, "zero_budget")
                     if r.get("task") and r.get("task") != zb_task})
    if not others:
        return ""
    return (f"History also contains zero-budget cells on {', '.join(others)}; the "
            f"zero-budget columns below use the latest task's records ({zb_task}) "
            "only — the archived task's cells remain in the per-cell tables.")


def zero_budget_cell(records, model, length, leg=None, task=None):
    """The model's zero_budget cell at (length, leg), or None. ``task`` (when
    given) restricts to one task version — the headline passes the latest one.
    leg="replicate" also matches the pre-F6 leg name "end_to_end" (the same
    test-retest arm before the rename), preferring a true replicate record."""
    legs = REPLICATE_LEG_ALIASES if leg == "replicate" else (leg,)
    for want in legs:
        cells = [r for r in by_facet(records, "zero_budget")
                 if r["model"] == model and r.get("length") == length
                 and _settings(r).get("leg") == want
                 and (task is None or r.get("task") == task)
                 and canonical_arm(r)]  # breadth rungs never shadow the headline
        if cells:
            return cells[0]
    return None


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


def zb_model_marks(records, model, task=None) -> str:
    """Union of cleanliness marks over the model's zero_budget cells (figure
    labels); ``task`` restricts to one task version (the figure's)."""
    minimal = model_effort_minimal(records, model)
    seen = set()
    for r in by_facet(records, "zero_budget"):
        if r["model"] == model and (task is None or r.get("task") == task):
            seen.update(zb_marks(r, minimal))
    if not seen and minimal:
        seen.add("*")
    return "".join(c for c in "*†" if c in seen)


def composition_gap_str(records, model, zb_task=None) -> str:
    """THE headline statistic: composition gap = state tracking (binding_only
    @L16) - composed @L16, rendered '+0.NN' with the input cells' cleanliness
    marks propagated. recall|holder is ~1.0 across the roster (scaffolded leg),
    so a free composer would score the composed cell at its binding leg; the gap
    is the composition deficit. 'n/a' when either cell never ran."""
    if zb_task is None:
        zb_task = zb_latest_task(records)
    bind = zero_budget_cell(records, model, 16, "binding_only", task=zb_task)
    comp = zero_budget_cell(records, model, 16, None, task=zb_task)
    if bind is None or comp is None:
        return "n/a"
    vb, vc = canonical_relaxed(bind), canonical_relaxed(comp)
    if vb is None or vc is None:
        return "—"
    minimal = model_effort_minimal(records, model)
    seen = zb_marks(bind, minimal) + zb_marks(comp, minimal)
    marks = "".join(c for c in "*†" if c in seen)
    return f"{vb - vc:+.2f}{marks}"


def model_replicate_noise_str(records, model, zb_task=None) -> str:
    """Per-model replicate noise: |composed@L16 - replicate@L16| (the replicate
    leg builds prompts IDENTICAL to the composed cell — test-retest), rendered
    '±0.NN'. 'n/a' when either cell never ran."""
    if zb_task is None:
        zb_task = zb_latest_task(records)
    a = zero_budget_cell(records, model, 16, None, task=zb_task)
    b = zero_budget_cell(records, model, 16, "replicate", task=zb_task)
    if a is None or b is None:
        return "n/a"
    va, vb = canonical_relaxed(a), canonical_relaxed(b)
    if va is None or vb is None:
        return "—"
    return f"±{abs(va - vb):.2f}"


INSTANT_STRESS_NOTE = (
    "Two instant cells beyond the composite headline, same protocol (reasoning "
    "off, one-line answer contract, 96-token cap; marks and escalated "
    "diagnostics as in the headline). recall_load scales the recall pool with "
    f"the length (recall_copy_v1 @L{RECALL_LOAD_LENGTH}, "
    f"pool {RECALL_LOAD_LENGTH}, n=50): single-query deferred recall under "
    f"working-set load. chain_instant runs chain_v1 d{CHAIN_INSTANT_DEPTH} on "
    f"the same k={CHAIN_INSTANT_K} staircase items as the thinking "
    f"d{CHAIN_INSTANT_DEPTH} cell (n=25): the within-item regime contrast for "
    "depth. The floor row is the uniform guess over the answer pool; escalated "
    "cells show the CANONICAL first attempt with the escalated rerun as a "
    "parenthesised diagnostic.")


def instant_stress_cell(records, facet, model, length):
    """The model's canonical-arm cell in an instant stress facet
    (recall_load / chain_instant), or None."""
    cells = [r for r in by_facet(records, facet)
             if r["model"] == model and r.get("length") == length
             and canonical_arm(r)]
    return cells[0] if cells else None


def instant_stress_columns() -> list[str]:
    """Column headers of the instant stress table, regime-prefixed like the
    headline columns."""
    return [
        "Model",
        f"instant: recall under load (recall_load, recall_copy_v1 "
        f"pool-{RECALL_LOAD_LENGTH} @L{RECALL_LOAD_LENGTH})",
        f"instant: chain d{CHAIN_INSTANT_DEPTH} (chain_instant, chain_v1, "
        f"k={CHAIN_INSTANT_K})",
    ]


def instant_stress_rows(records):
    """One row per CURRENT-ROSTER model for the instant stress table, plus the
    uniform-guess floor row (floors are first-class rows). [] when neither
    facet has records (older histories render no section)."""
    if not (by_facet(records, "recall_load")
            or by_facet(records, "chain_instant")):
        return []
    rows = []
    for m in roster_models(records):
        minimal = model_effort_minimal(records, m)
        rows.append([
            m,
            zb_value_str(instant_stress_cell(
                records, "recall_load", m, RECALL_LOAD_LENGTH), minimal),
            zb_value_str(instant_stress_cell(
                records, "chain_instant", m, CHAIN_INSTANT_DEPTH), minimal),
        ])
    rows.append([
        "uniform-guess floor (chance)",
        f"{1 / RECALL_LOAD_LENGTH:.3f} (1/{RECALL_LOAD_LENGTH})",
        f"{1 / CHAIN_INSTANT_K:.3f} (1/{CHAIN_INSTANT_K})",
    ])
    return rows


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


def headline_recall(records, model):
    """The ladder's first rung: the sanity recall_copy_v1 cell's canonical relaxed
    (instant regime — reasoning off). None when the cell never ran."""
    cells = [r for r in by_facet(records, "sanity")
             if r["model"] == model and r.get("task") == "recall_copy_v1"]
    return canonical_relaxed(cells[0]) if cells else None


def _task_spec(task):
    """TaskSpec lookup tolerant of the v1-family retirement (issue #11): scored
    specs live in tasks.CANONICAL; retired v1 specs move to tasks.RETIRED but
    their historical records still need render-time floors. None when the
    factworld package (or the requested spec) is unavailable."""
    try:
        from factworld import tasks as TK
    except Exception:  # pragma: no cover - environment guard
        return None
    spec = TK.CANONICAL.get(task)
    if spec is None:
        spec = getattr(TK, "RETIRED", {}).get(task)
    return spec


@functools.lru_cache(maxsize=8)
def recency_heuristic(task: str = "composite_copy_v1", n: int = 100):
    """Zero-budget floor (F7): score the one-line recency heuristic — answer with
    the LAST event's recipient and that holder's stated fact (binding-leg analog:
    the last recipient IS the holder guess) — on the exact deterministic items of
    ``task`` (the task the zero_budget records actually ran; looked up in
    CANONICAL or, once the v1 family retires, RETIRED), regenerated locally
    (pure stdlib, no API). v1 and v2 share the prompt grammar, so the same
    extraction applies; on v2 the floor sits near chance by construction.

    Returns {"composite_16", "composite_64", "binding_16"} or None when the
    factworld package (or the requested task spec) is unavailable."""
    spec = _task_spec(task)
    if spec is None:  # e.g. v2-task records rendered against an older factworld
        return None
    from factworld import tasks as TK

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


@functools.lru_cache(maxsize=8)
def object_filter_floor(task: str = "composite_copy_v2", n: int = 100):
    """Shallow zero-budget floor: E[1/w] over the exact deterministic items of
    ``task`` — for each item, 1/(number of give-events that write the queried
    object). A reader that FILTERS the stream to the queried object's writes but
    picks one uniformly at random (no last-write-wins resolution) is correct on
    the holder with probability >= 1/w, so E[1/w] is the score of pure object
    filtering with zero state tracking. Inherent to last-write-wins: it sits well
    above chance at small L and decays ~1/L. The binding leg derives from the
    SAME items (same w per item), so its floor equals the composite one at the
    same L; the replicate leg is the identical composite@L16 prompt.

    Returns {"composite_16", "composite_64", "binding_16"} or None when the
    factworld package (or the requested task spec) is unavailable."""
    spec = _task_spec(task)
    if spec is None:
        return None
    from factworld import tasks as TK

    def floor(length):
        tot = 0.0
        for e in TK.generate(spec, "test", n=n, length=length):
            writes = len(re.findall(rf"gives {e.meta.get('obj')} to ", e.prompt))
            if writes:
                tot += 1.0 / writes
        return tot / n

    f16, f64 = floor(16), floor(64)
    return {"composite_16": f16, "composite_64": f64, "binding_16": f16}


@functools.lru_cache(maxsize=64)
def object_filter_floor_at(task: str, length: int, breadth: int | None = None,
                           n: int = 100):
    """Per-rung floor hook for breadth cells: E[1/w] on the exact deterministic
    items of ``task`` at ``length`` under pool rung ``breadth`` (the runner's
    scaled(k=2*B, recall_pool=B) spec; None/16 = the canonical spec).

    The floor MOVES WITH m (n_objects_active) and L ONLY: w counts give-events
    that write the QUERIED object, which depend on the active-object working set
    and the stream length, never on the pool size — so across pool rungs at a
    fixed (m, L) the floor is FLAT (up to item-sampling noise; the rung changes
    which deterministic items are drawn, not the w distribution). Label floor
    rows per (task, L, m) via ``object_filter_floor_label_at``, one row per
    (m, L), shared by every rung. None when the spec is unavailable."""
    spec = _task_spec(task)
    if spec is None:
        return None
    if breadth and breadth != CANONICAL_BREADTH:
        spec = spec.scaled(k=2 * breadth, recall_pool=breadth)
    from factworld import tasks as TK
    tot = 0.0
    for e in TK.generate(spec, "test", n=n, length=length):
        writes = len(re.findall(rf"gives {e.meta.get('obj')} to ", e.prompt))
        if writes:
            tot += 1.0 / writes
    return tot / n


def object_filter_floor_label_at(task: str, length: int) -> str:
    """Row label for a per-rung breadth floor: keyed by (task, L, m) — the knobs
    the floor actually moves with — and explicitly flat across pool rungs."""
    spec = _task_spec(task)
    m = spec.n_objects_active if spec is not None else "?"
    return (f"{OBJECT_FILTER_LABEL} ({task} @L{length}, m={m}; "
            f"flat across pool rungs)")


def _floor_row(label, vals):
    """One floor row in headline-column order. Floors apply to the instant score
    columns only; recall, the gap/noise columns and the thinking columns render
    '—' (the floor's binding and composed values coincide by construction, so its
    gap is 0 by definition, not a measurement)."""
    return [label, "—",
            _fmt(vals["binding_16"]), _fmt(vals["composite_16"]),
            _fmt(vals["composite_64"]),
            "—", "—", "—", "—", "—"]


def _zb_floor_n(records, zb_task) -> int:
    return max((r.get("n") or 0 for r in by_facet(records, "zero_budget")
                if r.get("task") == zb_task), default=0) or 100


def heuristic_row(records, zb_task=None):
    """The 'recency heuristic' row for the zero-budget headline table, or None.
    Regenerates the floor on the task the zero_budget records actually used
    (``zb_task``, default: the latest one in history) and labels the row with it,
    so the row renders v1 floors for a v1-task history and v2 floors once
    uniform-last-write records land."""
    if zb_task is None:
        zb_task = zb_latest_task(records)
    if zb_task is None:
        return None
    vals = recency_heuristic(zb_task, _zb_floor_n(records, zb_task))
    if vals is None:
        return None
    return _floor_row(heuristic_label(zb_task), vals)


def object_filter_row(records, zb_task=None):
    """The 'object-filter floor' row (E[1/w] on the exact items), or None; the
    same task/labeling rules as heuristic_row."""
    if zb_task is None:
        zb_task = zb_latest_task(records)
    if zb_task is None:
        return None
    vals = object_filter_floor(zb_task, _zb_floor_n(records, zb_task))
    if vals is None:
        return None
    return _floor_row(object_filter_label(zb_task), vals)


def replicate_noise(records, zb_task=None):
    """Max observed |plain@L16 - replicate@L16| across models (F6): the replicate
    leg (recorded as end_to_end before the F6 rename) builds prompts IDENTICAL to
    the plain composite cell, so the pair is a test-retest replicate and the max
    |delta| is the run-to-run noise bar. Computed within the latest task version."""
    if zb_task is None:
        zb_task = zb_latest_task(records)
    deltas = []
    for m in models_of(records):
        a = zero_budget_cell(records, m, 16, None, task=zb_task)
        b = zero_budget_cell(records, m, 16, "replicate", task=zb_task)
        if a is not None and b is not None:
            va, vb = canonical_relaxed(a), canonical_relaxed(b)
            if va is not None and vb is not None:
                deltas.append(abs(va - vb))
    return max(deltas) if deltas else None


def replicate_note(records) -> str:
    noise = replicate_noise(records)
    noise_s = "n/a" if noise is None else f"{noise:.2f}"
    return (
        "replicate noise: the zero_budget replicate leg (recorded as end_to_end "
        "before the F6 rename) builds prompts IDENTICAL to the composed @L16 cell "
        "(same runner path), so |composed - replicate| is a test-retest delta; max "
        f"across models = {noise_s} — read that as the run-to-run noise bar on the "
        "headline numbers (including the gap column). Future runs keep this arm "
        "intentionally as leg='replicate'.")


def headline_rows(records):
    """One row per CURRENT-ROSTER model for the headline table, headline_columns
    (composition-statistic) order — archived models render in their own section,
    never here. The zero-budget columns use the LATEST zero-budget task's records only
    (older-task cells render 'n/a' here but stay in the per-cell tables); cells
    that never ran say 'n/a' (distinct from '—' = run but no qualifying value,
    F9). The recency-heuristic and object-filter floor rows come last."""
    zb_task = zb_latest_task(records)
    rows = []
    for m in roster_models(records):
        minimal = model_effort_minimal(records, m)
        row = [m]
        rec = headline_recall(records, m)
        row.append("n/a" if rec is None else _fmt(rec))
        for length, leg in ZB_HEADLINE_CELLS:
            r = zero_budget_cell(records, m, length, leg, task=zb_task)
            row.append(zb_value_str(r, minimal))
        row.append(composition_gap_str(records, m, zb_task))
        row.append(model_replicate_noise_str(records, m, zb_task))
        row.append(stress_value_str(
            stress_cell(records, "chain_nowrap", m, CHAIN_STRESS_DEPTH)))
        row.append(stress_value_str(
            stress_cell(records, "s5_concrete", m, S5_STRESS_LENGTH)))
        eff = headline_efficiency(records, m)
        row.append("n/a" if eff is None else f"{eff:.0f}")
        rows.append(row)
    for floor in (heuristic_row(records, zb_task), object_filter_row(records, zb_task)):
        if floor is not None:
            rows.append(floor)
    return rows


def _v1_facet_row(records, m):
    """One ARCHIVED_COLUMNS row of v1-facet headline scalars for one model."""
    dr, dr_eff = headline_dose_response(records, m)
    cl = headline_composite_length(records, m)
    legs = headline_decomposition(records, m)
    return [m,
            "—" if dr is None else f"{dr:.2f} @ {dr_eff}",
            _fmt(cl),
            " / ".join(_fmt(legs[leg]) for leg in LEG_ORDER)]


def archived_headline_rows(records):
    """'v1 archived facets (pre-redesign)' rows: CURRENT-ROSTER models with data
    in an archived v1 facet (archived MODELS render in their own section)."""
    return [_v1_facet_row(records, m) for m in roster_models(records)
            if any(r["model"] == m
                   for facet in V1_ARCHIVED_FACETS for r in by_facet(records, facet))]


def archived_model_rows(records):
    """'Archived models (dropped from the roster)' rows: every model in history
    that is not in factworld.benchmark.MODELS, with its v1-facet columns."""
    return [_v1_facet_row(records, m) for m in archived_models(records)]


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
             f"Figures draw a dotted reference line at relaxed {REF_THRESHOLD}.",
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

    zb_task = zb_latest_task(records)
    headline_cols = headline_columns(zb_task)
    lines += ["## Headline (current roster)", "",
              "Current roster only (factworld.benchmark.MODELS); models dropped from "
              "the roster render in the archived-models section below.",
              "",
              COMPOSITION_NOTE,
              "",
              f"Instant cells: task **{zb_task or 'n/a'}** with reasoning "
              "off (effort=none) under a one-line answer contract "
              "(settings.contract=true); relaxed match. Escalated cells show the "
              "CANONICAL first attempt at the shared base budget, with the escalated "
              "rerun as a parenthesised diagnostic.",
              ""]
    note = _zb_mixed_task_note(records, zb_task)
    if note:
        lines += [note, ""]
    lines += ["| " + " | ".join(headline_cols) + " |",
              "|" + "---|" * len(headline_cols)]
    for row in headline_rows(records):
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines += ["", FLOOR_NOTE, ""]
    if any(not canonical_arm(r) for r in records):
        # breadth/fixed-k rung cells exist: they are separate arms (labelled
        # B=... / k_fixed=... in the per-cell tables, excluded from the headline
        # columns/figures above) and their floors are per-(m, L), not per-rung.
        lines += [BREADTH_FLOOR_NOTE, ""]
    for note in ZB_FOOTNOTES:
        lines += [note, ""]
    lines += [GAP_NOTE, "", replicate_note(records), "", CENSOR_NOTE, "",
              EFFICIENCY_NOTE, ""]
    lines += [
        "The chain column reads the `chain_nowrap` facet only (staircase k=2d+1, so the "
        f"d{CHAIN_STRESS_DEPTH} cell is k={CHAIN_STRESS_K}). `chain_v1` builds a single "
        f"k={CHAIN_CYCLE_K} pointer cycle and measures depth only for depths < k "
        '(`factworld/tasks.py`: "Depths stay < k so the cycle never wraps"); `chain_depth` cells '
        f"at depth >= {CHAIN_CYCLE_K} wrapped the cycle (gold == start agent at depths 12/24/48; "
        "effective difficulty depth mod 6), measure the wrapped task rather than depth, and are "
        f"marked `{CHAIN_INVALID_MARK}` in the tables below and excluded from the chain figure.",
        "",
    ]

    stress = instant_stress_rows(records)
    if stress:
        cols = instant_stress_columns()
        lines += ["## Instant stress rows (recall under load; chain d16)", "",
                  INSTANT_STRESS_NOTE, "",
                  "| " + " | ".join(cols) + " |",
                  "|" + "---|" * len(cols)]
        for row in stress:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    dropped = archived_model_rows(records)
    if dropped:
        lines += ["## Archived models (dropped from the roster)", "",
                  "Models present in history but no longer in "
                  "factworld.benchmark.MODELS, with their v1-facet columns "
                  "(historical facet names). Their per-cell rows — any facet — "
                  "remain in the tables below.", "",
                  "| " + " | ".join(ARCHIVED_COLUMNS) + " |",
                  "|" + "---|" * len(ARCHIVED_COLUMNS)]
        for row in dropped:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    archived = archived_headline_rows(records)
    if archived:
        lines += ["## v1 archived facets (pre-redesign)", "",
                  "Legacy headline columns for the pre-redesign v1-only facets "
                  f"({', '.join(V1_ARCHIVED_FACETS)}), current-roster models only; "
                  "superseded by the ladder headline above. Per-cell rows remain "
                  "in the tables below.", "",
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
    "facet", "task", "length", "n", "effort", "leg", "rendering", "breadth", "k_fixed",
    "max_new_tokens",
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
                # v3 rung keys (blank on canonical cells — sentinel-dropped)
                "breadth": s.get("breadth"), "k_fixed": s.get("k_fixed"),
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
    ax.set_title("v1 archived facet (pre-redesign): composite_copy_v1 @L16 vs reasoning effort")
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


def _stress_fig(records, out_dir, facet, name, xlabel, title):
    """Score-vs-depth/length figure for a thinking state-stress facet."""
    cells = [r for r in by_facet(records, facet)
             if _settings(r).get("rendering") != "abstract_stated"
             and canonical_arm(r)]  # one line per model: canonical rungs only
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
            # F4: hollow marker — the cell's calls were majority finish=length,
            # so it is not measurable at this budget (tokens, not ability).
            if y < REF_THRESHOLD and majority_finish_length(r):
                ax.plot([r["length"]], [y], marker="o", markersize=6,
                        markerfacecolor="white", markeredgecolor=colors[m],
                        linestyle="none", zorder=5)
            # F5: ‡ — >10% of the cell's calls escaped the token cap.
            if cap_escape(r):
                ax.annotate("‡", (r["length"], y), textcoords="offset points",
                            xytext=(4, 5), fontsize=9, color=colors[m])
    ax.axhline(REF_THRESHOLD, color="#888888", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    xt = sorted({r["length"] for r in cells})
    ax.set_xticks(xt, [str(x) for x in xt])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relaxed accuracy")
    ax.set_title(title, fontsize=10)
    _style_axes(ax)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    fig.text(0.01, 0.03,
             "hollow: majority finish=length — not measurable at this budget; "
             "‡: >10% of calls escaped the token cap",
             fontsize=7, color="#555555")
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out_dir, name)


def fig_s5_stress(records, out_dir):
    return _stress_fig(records, out_dir, "s5_concrete", "fig_s5_horizon",
                       "permutation sequence length (log scale)",
                       "s5_concrete (thinking): relaxed score vs length "
                       f"(dotted: {REF_THRESHOLD} reference)")


def _fig_chain(records, out_dir, facet, task_label):
    """Chain figure over a chain facet; chain_depth cells past the design gate
    (depth >= k=6, cycle wrap) are excluded — they measure the wrapped task."""
    valid = [r for r in records if not chain_invalid(r)]
    return _stress_fig(valid, out_dir, facet, f"fig_{facet}",
                       "chain depth (log scale)",
                       f"{task_label}: relaxed score vs depth "
                       f"(dotted: {REF_THRESHOLD} reference)")


def fig_chain_depth(records, out_dir):
    return _fig_chain(records, out_dir, "chain_depth",
                      f"chain_v1 (depths < k={CHAIN_CYCLE_K})")


def fig_chain_nowrap(records, out_dir):
    return _fig_chain(records, out_dir, "chain_nowrap",
                      "chain_nowrap (staircase k=2d+1)")


ZB_GROUPS = [  # (bar label, length, leg) — the composition figure's cells: the
    # state-tracking leg before the composed cells (the leg-vs-composed delta is
    # the composition gap); "replicate" also matches the pre-F6 leg name
    # end_to_end (REPLICATE_LEG_ALIASES)
    ("state tracking (binding_only) L16", 16, "binding_only"),
    ("composed L16", 16, None),
    ("composed L64", 64, None),
    ("replicate L16 (test-retest)", 16, "replicate"),
]


def fig_zero_budget(records, out_dir):
    """The composition figure: component leg (state tracking), composed cells and
    the per-model composition gap annotation, instant regime. Like the headline
    table it shows the LATEST zero-budget task's cells only (canonical rungs —
    breadth-rung cells are separate arms); an archived task's cells stay in the
    per-cell tables."""
    zb_task = zb_latest_task(records)
    cells = [r for r in by_facet(records, "zero_budget")
             if r.get("task") == zb_task and canonical_arm(r)]
    if not cells:
        return []

    def composite_l64(m):
        r = zero_budget_cell(records, m, 64, None, task=zb_task)
        return (canonical_relaxed(r) or 0.0) if r else -1.0

    models = sorted(models_of(cells), key=composite_l64, reverse=True)
    group_colors = {label: PALETTE[j] for j, (label, _, _) in enumerate(ZB_GROUPS)}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    width = 0.2
    for j, (label, length, leg) in enumerate(ZB_GROUPS):
        xs, ys, errs, hatched, escapes = [], [], [[], []], [], []
        for i, m in enumerate(models):
            r = zero_budget_cell(records, m, length, leg, task=zb_task)
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
    # gap annotation just above each model's bars: binding_only@L16 -
    # composed@L16 (the composition deficit, the headline statistic)
    for i, m in enumerate(models):
        bind = zero_budget_cell(records, m, 16, "binding_only", task=zb_task)
        comp = zero_budget_cell(records, m, 16, None, task=zb_task)
        if bind is None or comp is None:
            continue
        vb, vc = canonical_relaxed(bind), canonical_relaxed(comp)
        if vb is None or vc is None:
            continue
        tops = [_ci(r)[1] for r in (zero_budget_cell(records, m, L, leg, task=zb_task)
                                    for _, L, leg in ZB_GROUPS) if r is not None]
        ax.annotate(f"gap {vb - vc:+.2f}", (i, max(tops) + 0.03), ha="center",
                    fontsize=6.5, color="#333333")
    ax.set_xticks(range(len(models)),
                  [m + zb_model_marks(records, m, task=zb_task) for m in models],
                  fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("relaxed accuracy")
    ax.set_title(f"Composition (instant): {zb_task}, reasoning off, "
                 "one-line answer contract", fontsize=10)
    _style_axes(ax)
    ax.set_ylim(-0.03, 1.14)  # _style_axes resets ylim; keep annotation headroom
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    fig.text(0.01, 0.05,
             "gap = state tracking (binding_only) - composed @L16; canonical "
             "first-attempt values (escalated reruns are table diagnostics);\n"
             "hatched / *: off-arm ran effort=minimal; †: visible working on the "
             "canonical attempt; ‡: cap-escape; sorted by composed @L64",
             fontsize=7, color="#555555")
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
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


# --- profile small-multiples (per-model axis positions) -------------------------
# One panel per CURRENT-ROSTER model showing its normalized position on each
# benchmark axis (present per-axis ranks and profiles, never a single scalar —
# AGENTS.md). Inverted axes (the composition gap and the s5@128 ctok efficiency
# column: smaller is better) are negated before normalization so the right-hand
# end is always the roster-best. Missing cells and ⊘ budget-censored cells render
# as gaps (no bar) with their mark, never as zeros.

PROFILE_AXES = [  # (short label, inverted)
    ("binding @L16", False),
    ("composed @L16", False),
    ("gap (inv)", True),
    (f"chain d{CHAIN_STRESS_DEPTH}", False),
    (f"s5 @L{S5_STRESS_LENGTH}", False),
    ("s5@128 ctok (inv)", True),
]
PROFILE_AXIS_LABELS = [a for a, _ in PROFILE_AXES]


def _profile_cell(value, display, marks="", status="ok"):
    return {"value": value, "display": display, "marks": marks, "status": status}


def _profile_score_cell(rec, minimal=False):
    """Profile cell from a score-valued record: the canonical relaxed value with
    the cell's cleanliness marks; 'censored' (⊘, not measurable at this budget)
    when the cell's calls were majority finish=length; 'missing' when the cell
    never ran."""
    if rec is None:
        return _profile_cell(None, "n/a", status="missing")
    if majority_finish_length(rec):
        return _profile_cell(None, CENSORED_CELL, status="censored")
    v = canonical_relaxed(rec)
    if v is None:
        return _profile_cell(None, "—", status="missing")
    if rec.get("facet") == "zero_budget":
        marks = zb_marks(rec, minimal)
    else:
        marks = "‡" if cap_escape(rec) else ""
    return _profile_cell(v, f"{v:.2f}", marks)


def profile_values(records):
    """{model: {axis label: cell}} over PROFILE_AXES for the CURRENT roster.
    Cells carry value / display / marks / status ('ok' | 'censored' | 'missing');
    profile_normalized adds 'norm'. The instant cells (binding, composed, gap)
    read the LATEST zero-budget task only, marks propagated as in the headline;
    the gap needs both input cells and inherits their marks."""
    zb_task = zb_latest_task(records)
    out = {}
    for m in roster_models(records):
        minimal = model_effort_minimal(records, m)
        bcell = _profile_score_cell(
            zero_budget_cell(records, m, 16, "binding_only", task=zb_task), minimal)
        ccell = _profile_score_cell(
            zero_budget_cell(records, m, 16, None, task=zb_task), minimal)
        if bcell["status"] == "ok" and ccell["status"] == "ok":
            g = bcell["value"] - ccell["value"]
            marks = "".join(c for c in "*†" if c in bcell["marks"] + ccell["marks"])
            gcell = _profile_cell(g, f"{g:+.2f}", marks)
        else:
            gcell = _profile_cell(None, "n/a", status="missing")
        eff = headline_efficiency(records, m)
        ecell = (_profile_cell(eff, f"{eff:.0f}") if eff is not None
                 else _profile_cell(None, "n/a", status="missing"))
        out[m] = dict(zip(PROFILE_AXIS_LABELS, [
            bcell, ccell, gcell,
            _profile_score_cell(
                stress_cell(records, "chain_nowrap", m, CHAIN_STRESS_DEPTH)),
            _profile_score_cell(
                stress_cell(records, "s5_concrete", m, S5_STRESS_LENGTH)),
            ecell,
        ]))
    return out


def profile_normalized(values):
    """Adds 'norm' to every measurable cell: its min-max position across the
    roster on that axis (inverted axes negated first, so 1.0 is always the
    roster-best end); an axis with a single measurable value normalizes to 1.0."""
    for label, invert in PROFILE_AXES:
        cells = [d[label] for d in values.values() if d[label]["status"] == "ok"]
        oriented = [(-c["value"] if invert else c["value"]) for c in cells]
        if not oriented:
            continue
        lo, hi = min(oriented), max(oriented)
        for c, o in zip(cells, oriented):
            c["norm"] = (o - lo) / (hi - lo) if hi > lo else 1.0
    return values


def fig_profiles(records, out_dir):
    """Small-multiples profile grid: one panel per current-roster model, its
    normalized position on each axis as a horizontal bar (right = roster-best;
    inverted axes flipped first) with the raw value + cleanliness marks
    alongside; missing / ⊘ budget-censored cells are gaps, never zeros."""
    values = profile_normalized(profile_values(records))
    if not values or all(c["status"] != "ok"
                         for d in values.values() for c in d.values()):
        return []
    zb_task = zb_latest_task(records)
    models = list(values)
    ncols = 3
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True,
                             figsize=(FIGSIZE[0], 1.6 * nrows + 1.0))
    grid = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    n_ax = len(PROFILE_AXIS_LABELS)
    ys = list(range(n_ax))[::-1]  # first axis on top
    for k, (ax, m) in enumerate(zip(grid, models)):
        for y, label in zip(ys, PROFILE_AXIS_LABELS):
            c = values[m][label]
            color = PALETTE[PROFILE_AXIS_LABELS.index(label) % len(PALETTE)]
            if c["status"] == "ok":
                ax.barh(y, max(c["norm"], 0.015), height=0.62, color=color)
                ax.text(min(c["norm"], 1.0) + 0.03, y, c["display"] + c["marks"],
                        va="center", fontsize=6, color="#333333")
            else:  # a gap, not a zero: no bar, the mark text only
                txt = CENSORED_CELL if c["status"] == "censored" else c["display"]
                ax.text(0.03, y, txt, va="center", fontsize=6, color="#999999")
        ax.set_yticks(ys)
        ax.set_yticklabels(PROFILE_AXIS_LABELS if k % ncols == 0 else [""] * n_ax,
                           fontsize=6)
        ax.set_ylim(-0.6, n_ax - 0.4)
        ax.set_xlim(0, 1.45)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"], fontsize=6)
        ax.set_title(m.split("/", 1)[-1] + zb_model_marks(records, m, task=zb_task),
                     fontsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2)
    for ax in grid[len(models):]:  # unused trailing panels
        ax.set_visible(False)
    fig.suptitle("Per-model profiles: normalized axis positions "
                 "(right = roster-best; inverted axes flipped)", fontsize=9)
    fig.text(0.01, 0.012,
             "bar = min-max position across the roster on that axis; the raw value "
             "+ marks is printed beside it (instant marks as in the headline; ‡ "
             "cap-escape).\n"
             f"gap and s5@128 ctok are inverted (smaller better); {CENSORED_CELL} "
             "and n/a cells are gaps, not zeros; instant cells read task "
             f"{zb_task or 'n/a'}.",
             fontsize=6.5, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    return _save(fig, out_dir, "fig_profiles")


FIGURES = [fig_zero_budget, fig_profiles, fig_dose_response, fig_composite_length,
           fig_s5_stress, fig_chain_depth, fig_chain_nowrap, fig_decomposition]


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


def _html_table(headers, rows, sortable=False, groups=None):
    """``groups`` (label, colspan) pairs add a grouping header row above the
    column headers (the headline's instant/thinking regime split); only used on
    non-sortable tables (the sort JS indexes the flat header row)."""
    cls = ' class="sortable"' if sortable else ""
    out = [f"<table{cls}><thead>"]
    if groups:
        out.append("<tr>" + "".join(
            f'<th colspan="{span}">{html.escape(label)}</th>'
            for label, span in groups) + "</tr>")
    out.append("<tr>")
    out += [f"<th>{html.escape(h)}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def write_index_html(records, out_dir, svg_paths, history_path):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    zb_task = zb_latest_task(records)
    headline_cols = headline_columns(zb_task)
    mixed_note = _zb_mixed_task_note(records, zb_task)
    head_rows = headline_rows(records)
    dropped = archived_model_rows(records)
    dropped_html = ""
    if dropped:
        dropped_html = (
            "<h2>Archived models (dropped from the roster)</h2>\n"
            '<p class="small">Models present in history but no longer in '
            "factworld.benchmark.MODELS, with their v1-facet columns (historical "
            "facet names). Their per-cell rows — any facet — remain in the tables "
            "below.</p>\n"
            + _html_table(ARCHIVED_COLUMNS, dropped))
    archived = archived_headline_rows(records)
    archived_html = ""
    if archived:
        archived_html = (
            "<h2>v1 archived facets (pre-redesign)</h2>\n"
            f'<p class="small">Legacy headline columns for the pre-redesign '
            f"v1-only facets ({html.escape(', '.join(V1_ARCHIVED_FACETS))}), "
            "current-roster models only; superseded by the ladder headline "
            "above.</p>\n"
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
The chain column reads the <code>chain_nowrap</code> facet only (staircase k=2d+1, so the
d{CHAIN_STRESS_DEPTH} cell is k={CHAIN_STRESS_K}): <code>chain_v1</code> builds a
single k={CHAIN_CYCLE_K} pointer cycle and measures depth only for depths &lt; k
(<code>factworld/tasks.py</code> design gate), so <code>chain_depth</code> cells at depth &ge;
{CHAIN_CYCLE_K} wrapped the cycle, measure the wrapped task rather than depth, and are marked
{html.escape(CHAIN_INVALID_MARK)} below and excluded from the chain figure.</p>
<h2>Headline (current roster)</h2>
<p class="small">Current roster only (factworld.benchmark.MODELS); models dropped from the
roster render in the archived-models section below. {html.escape(COMPOSITION_NOTE)}</p>
<p class="small">Instant cells: task <strong>{html.escape(zb_task or "n/a")}</strong>
with reasoning off (effort=none) under a one-line answer contract (settings.contract=true);
relaxed match. Escalated cells show the CANONICAL first attempt at the shared base budget,
with the escalated rerun as a parenthesised diagnostic.
{(" " + html.escape(mixed_note)) if mixed_note else ""}</p>
{_html_table(headline_cols, head_rows, groups=HEADLINE_GROUPS)}
<p class="small">{html.escape(FLOOR_NOTE)}</p>
<p class="small">{"<br>".join(html.escape(n) for n in ZB_FOOTNOTES)}<br>
{html.escape(GAP_NOTE)}<br>
{html.escape(replicate_note(records))}<br>
{html.escape(CENSOR_NOTE)}<br>
{html.escape(EFFICIENCY_NOTE)}</p>
{dropped_html}
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
