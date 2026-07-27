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
                archived facets (pre-redesign)" legacy headline, then the full
                per-cell table, diagnostics, and a quarantined provenance
                section for INVALID chain_depth cells. Escalated zero-budget
                cells publish the CANONICAL first attempt at the shared base
                budget; the escalated rerun renders as a marked diagnostic.
                An instant cell whose canonical attempt shows PERVASIVE covert
                reasoning (rtok on > 50% of calls) renders as an explicit upper
                bound '≤x†' and takes no part in orderings; a gap whose
                state-tracking input sits at the object-filter floor renders
                '—ᶠ' (floor − floor ≈ 0 by construction, not a measurement).
                The s5_chain ranking carries the two per-call diagnostics
                (worked calls, event-blind); a 'Repeat runs' section lists every
                cell run more than once at identical settings with its
                run-to-run spread, and the thinking noise bar is read off the
                current-roster cells there that take part in orderings — a
                marked cell's spread is engagement or budget variance and
                reports in the ᵘ footnote instead.
  results.csv   flat per-cell export (all metrics + diagnostics incl. contract_rate /
                covert_cot_rate / rtok_leak_rate / escalated + usage, the per-cell
                work rate with its two conditional match scores, the s5_chain
                event-blind rate, the truncation rate, and the repeat-run count
                and spread)
  fig_*.png/.svg  facet figures, one per facet with data, plus fig_profiles — a
                  small-multiples panel per roster model showing its normalized
                  position on each axis (binding, composed@L16, gap inverted,
                  chain d128, s5 @L256, s5@128 mean ctok inverted; missing/⊘ cells are
                  gaps, not zeros) — (PNG 150 dpi for blog upload,
                  SVG for the HTML page); chain_depth cells past chain_v1's design gate
                  (depth >= k=6, cycle wrap) are excluded from figures and the headline
                  columns and marked INVALID in the tables
  index.html    self-contained static page (inline CSS, inline SVG, sortable table)

After rendering, the repo README's marked frontier block (the tables between
``<!-- FRONTIER_TABLE_START -->`` and ``<!-- FRONTIER_TABLE_END -->``) is
rewritten from the same headline data in compact cell form (escalation
diagnostics collapse to the canonical value + mark; raised-budget cells carry
ʳ; '⊘ >budget' renders ⊘), and closes with the block's ONE marks legend —
generated from the rendered cells, so it lists exactly the marks the tables use
and no others — a README without the markers is untouched.

EXCLUSION FROM ORDERINGS is one rule with one implementation: ``exclusion_mark``
names the mark (⊘ budget censoring, ≤x† pervasive covert reasoning, ᵘ unworked
answers) and ``ordering_value`` returns the number a cell contributes to a sort,
which for a marked cell is nothing. Marked rows therefore sort last on their
name, never on their score, in every table and in fig_bench_headline, where they
plot below a rule in grey with their mark on the tick label: published, not
ranked. The rule reaches the tiebreaks and the derived statistics too: a marked
row's ctok is not read as a tiebreak, and the published thinking noise bar reads
unmarked cells only.

Thinking cells carry two per-call diagnostics beside the score. WORK RATE is the
fraction of a cell's calls whose completion exceeds the working line, with match
conditional on working and on not working; a cell whose unworked calls are both
frequent and materially less accurate than its worked calls renders the ᵘ mark
and takes no part in orderings. It measures the model UNDER THIS HARNESS'S SYSTEM
PROMPT, not the model alone: a three-arm probe over identical items moves
gpt-5.6-sol's work rate from 0.68 to 0.96 by deleting two clauses of that prompt
(PROMPT_PROBE_NOTE, results/probes/). EVENT-BLIND RATE (s5_chain) is the fraction
of ELIGIBLE predictions equal to the depth-hop dereference of the INITIAL pointer
map — the answer of a reader that takes the fact block and skips the event
stream. Eligible = the blind answer differs from the gold answer under **match**
(``event_blind_counts``); the published roster-wide total (``event_blind_note``)
sums those same counts over the current roster's scored cells only, since a
published number describes the published roster.

The canonical evaluator is **match** — strip a trailing period from both sides and compare
the model's first len(gold) whitespace tokens to the gold answer; binary per item, no partial
credit (`factworld.tasks.score_relaxed`). Containment renders as the one diagnostic column;
stored record keys keep their historical names (metrics.relaxed/exact/contains/last_n — no
history rewrite). Confidence intervals are Wilson 95% (pure python, no scipy).

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


def canonical_rung(rec) -> bool:
    """True for cells on the canonical pool/chain rungs (no breadth override, no
    fixed-k chain)."""
    return breadth_of(rec) == CANONICAL_BREADTH and not k_fixed_of(rec)


def off_protocol_prompt(rec) -> bool:
    """True when the cell ran under a system prompt that is not one the instrument
    is defined against. The runner sentinel-drops ``settings.system_prompt_fp`` at
    the canonical prompts (factworld.benchmark.with_system_prompt), so a scored
    cell carries no fingerprint and any fingerprint present names another prompt."""
    return bool(_settings(rec).get("system_prompt_fp"))


def canonical_arm(rec) -> bool:
    """True for cells on the canonical arm — canonical rungs and the protocol's
    system prompt. Only these feed the headline columns and figures; an
    off-protocol prompt is a different measurement, so such a cell stays in the
    per-cell tables and never shadows the scored one."""
    return canonical_rung(rec) and not off_protocol_prompt(rec)

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
# Symmetric contamination rule: an instant cell whose CANONICAL attempt emitted
# reasoning tokens on MORE THAN HALF its calls is pervasively covert — its score
# renders as the explicit upper bound '≤x†' and, like ⊘ budget-censored cells,
# it takes no part in figure sorts or cross-model ordering prose.
PERVASIVE_RTOK_RATE = 0.5
# Gap cells where the state-tracking (binding) input sits at the object-filter
# floor render this mark instead of a number: floor − floor ≈ 0 by construction.
GAP_FLOOR_MARK = "—ᶠ"
S5_EFF_LENGTH = 128          # matched efficiency cell: s5_concrete @ L128 (F10)
# Thinking state-stress cells: chain d128 on the chain_nowrap staircase (k=2d+1,
# so d128 IS the fixed-breadth k=257 cell across the roster) and s5 @L256, at the
# standard thinking budget; cells rerun at a raised budget (--budget-override)
# render the budget with the number: '1.00 @32,768tok (raised budget)'.
THINKING_BUDGET = 16384
CHAIN_STRESS_DEPTH = 128
CHAIN_STRESS_K = 2 * CHAIN_STRESS_DEPTH + 1  # staircase k=2d+1 -> 257
S5_STRESS_LENGTH = 256
# The s5_chain headline runs at the maximum supported reasoning effort; cells at
# any other effort are probes and never enter the ranking or its aggregates.
S5_CHAIN_EFFORT = "xhigh"
# Instant component-stress cells beyond the composite headline (facets
# recall_load / chain_instant): single-query deferred recall under working-set
# load (the pool scales with the length) and the chain d16 off arm of the
# staircase (the same items as the thinking d16 cell). Floors are first-class
# rows: the uniform guess over the recall pool and over the staircase agent set.
RECALL_LOAD_LENGTH = 64                       # recall_copy_v1 @L64, pool == L
CHAIN_INSTANT_DEPTH = 16
CHAIN_INSTANT_K = 2 * CHAIN_INSTANT_DEPTH + 1  # staircase k=2d+1 -> 33
CENSORED_CELL = "⊘ >budget"  # majority finish=length: not measurable at this budget
# Facets of the instant regime (reasoning off, one-line answer contract). Short
# completions are the POINT there — the work-rate diagnostics below apply to the
# thinking regime only, where a short completion means the model did not work.
INSTANT_FACETS = {"zero_budget", "sanity", "recall_load", "chain_instant", "gap_stability"}

# --- work rate (thinking cells) -------------------------------------------------
# WORK_LINE splits a thinking cell's calls into worked and unworked. The signal is
# ctok (completion tokens), NOT rtok (reasoning tokens): rtok accounting is not
# comparable across providers — on this history the median ctok-minus-rtok is 5184
# for sonnet-5, 4184 for opus and 1675 for fable but 2-17 for every other model,
# and glm reports rtok=0 on correct 8k-token answers. ctok is reported on the same
# basis everywhere, so it is the only cross-provider work signal.
#
# The line is set at 512 completion tokens. The answer is one token ('gNN'), and
# the thinking cells that split at all split with a literal gap in the sorted ctok
# distribution — gpt-5.6-sol's s5_chain cells jump 257->1136 at L32, 180->2366 at
# L64, 187->3101 at L96 and 160->3690 at L128 — so 512 falls inside the empty
# region at every length, and any line in the hundreds gives the same split.
# Sensitivity: the nearest call on the other side of the line anywhere in the
# s5_chain history is grok-4.5's 538-token completion at L64, so a line above ~530
# starts counting worked calls as unworked; 512 is the widest margin available.
WORK_LINE = 512
# A cell is marked when BOTH hold: more than UNWORKED_RATE of its calls are
# unworked, AND its unworked calls score materially below its worked calls
# (disjoint Wilson intervals). The rate alone is not enough — a model that answers
# a shallow cell correctly without visible working is measured, not contaminated.
# UNWORKED_RATE reuses CAP_ESCAPE_RATE's line: the project already treats a
# per-call anomaly on more than a tenth of a cell's calls as contamination.
UNWORKED_RATE = 0.10
# Marked cells render 'x.xxᵘ' and, like ⊘ and ≤x†, take no part in orderings.
UNWORKED_MARK = "ᵘ"
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
    """Full headline column headers (instant + thinking), kept for backwards
    compatibility with callers that need one combined list."""
    return instant_headline_columns(zb_task) + thinking_headline_columns()[1:]


def instant_headline_columns(zb_task) -> list[str]:
    """Instant-regime headline columns. The state-tracking and composed columns
    carry the task version of the records they were built from (the LATEST
    zero-budget task in history)."""
    ver = _task_ver(zb_task)
    return [
        "Model",
        "instant: recall (sanity, recall_copy_v1)",
        f"instant: state tracking (binding_only @L16, {ver})",
        f"instant: composed @L16 (match, {ver})",
        f"instant: composed @L64 ({ver})",
        "instant: composition gap (binding_only - composed @L16)",
        "instant: replicate noise (|composed - replicate| @L16)",
    ]


def thinking_headline_columns() -> list[str]:
    """Thinking-regime headline columns: the state-stress cells plus an
    efficiency column measured on the matched s5_concrete L128 cell."""
    return [
        "Model",
        f"thinking: chain d{CHAIN_STRESS_DEPTH} (chain_nowrap, k={CHAIN_STRESS_K}, match)",
        f"thinking: s5 @L{S5_STRESS_LENGTH} (s5_concrete, match)",
        "thinking: s5@128 mean ctok/call",
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
    "(†, trigger 1 — visible working) the canonical attempt's completion carries "
    "short visible working instead of a bare answer: median (per-example) or mean "
    f"ctok (completion tokens) per call > {CTOK_WORKING_LINE:.0f} (~3x the 8-11 "
    "token answers), or the cell needed a budget escalation.",
    "(†, trigger 2 — covert reasoning) the model reasoned despite effort=none: mean "
    f"rtok (reasoning tokens) per call > {RTOK_LEAK_PER_CALL:.0f} on the published "
    f"attempt. Where MORE THAN {PERVASIVE_RTOK_RATE:.0%} of the canonical attempt's "
    "calls carry reasoning tokens the covert reasoning is pervasive and the cell "
    "renders as the explicit upper bound ≤x†.",
    "(‡) cap-escape: per-example ctok exceeded settings.max_new_tokens on "
    f">{CAP_ESCAPE_RATE:.0%} of calls (the provider did not enforce the cap); token "
    "counts and budget comparisons for those cells are not cap-comparable.",
    "(diag x.xx @512tok) escalated diagnostic: the cell was rerun once at an "
    "escalated token budget after majority finish=length; the CANONICAL number is "
    "the first attempt at the shared base budget — the escalated value is a marked "
    "diagnostic, not the headline.",
    f"({GAP_FLOOR_MARK}) gap not interpretable where the state-tracking component "
    "sits at the floor: the binding cell's Wilson CI overlaps the object-filter "
    "floor's, so the composed cell is floor-shaped too and binding − composed "
    "reads floor − floor ≈ 0 by construction.",
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
SYMMETRY_NOTE = (
    "⊘ = not measurable at this budget; ≤x† = upper bound, covert reasoning on most "
    "calls; neither participates in orderings.")
NOTATION_NOTE = (
    "Notation: `@Ln` = stream length (events, or hops for chain depth d); "
    "`@Ntok` = a completion-token budget. Instant escalations render "
    "`(diag x.xx @512tok)`; thinking cells rerun at a raised budget render it "
    "with the number, e.g. `1.00 @32,768tok (raised budget)`.")
THINKING_NOISE_NOTE = (
    "Thinking columns: n=25 per cell; Wilson intervals ≈ ±0.15–0.19 — differences "
    "under ~0.2 are not an ordering.")


def noise_bar_spreads(records):
    """Run-to-run spreads that set the thinking noise bar: current-roster thinking
    cells, run more than once at identical settings, that take part in orderings.

    A marked cell is excluded here for the reason it is excluded everywhere else —
    its repeat spread measures what the mark names (engagement under ᵘ, budget
    under ⊘), not the instrument's measurement noise — and the exclusion uses the
    one implementation, ``exclusion_mark`` over the cell as rendered."""
    roster = set(roster_models(records))
    return [replicate_spread(r) for r in records
            if thinking_cell(r) and r.get("model") in roster
            and replicate_spread(r) is not None
            and not out_of_ordering(stress_value_str(r))]


def marked_noise_spreads(records):
    """The same spreads for the current roster's MARKED thinking cells — the
    engagement/budget variance the noise bar does not read."""
    roster = set(roster_models(records))
    return [replicate_spread(r) for r in records
            if thinking_cell(r) and r.get("model") in roster
            and replicate_spread(r) is not None
            and out_of_ordering(stress_value_str(r))]


def thinking_noise_note(records) -> str:
    """The thinking-regime noise bar, read off the repeat runs in the history
    rather than quoted: the widest run-to-run spread among the cells that set it
    (``noise_bar_spreads``). Falls back to the interval-only note when no such
    cell was repeated."""
    spreads = noise_bar_spreads(records)
    if not spreads:
        return THINKING_NOISE_NOTE
    return (f"Thinking columns: n=25 per cell; Wilson intervals ≈ ±0.15–0.19. "
            f"{len(spreads)} current-roster thinking cells that take part in "
            f"orderings were run more than once at identical settings (Repeat runs "
            f"below) and their run-to-run spreads reach {max(spreads):.2f} — read a "
            "difference smaller than that as noise, not an ordering. Marked cells "
            "set no part of this bar: their repeat spread is variance in what the "
            "mark names, not measurement noise.")
NONCOMPARABLE_NOTE = (
    "Numbers in this table are on retired v1 tasks/facets (pre-redesign samplers "
    "and settings) and are NOT comparable to the current headline.")
EFFICIENCY_NOTE = (
    f"s5@128 mean ctok/call: the cell's total completion tokens divided by n on the "
    f"matched s5_concrete L{S5_EFF_LENGTH} cell "
    "(run by every current-roster model). This replaces ctok/solve, which averaged "
    "only over cells a model SOLVED and therefore rewarded models that failed early "
    "(selection bias: the published 2.7x opus-vs-kimi ctok/solve gap is ~1.4x on the "
    "matched cell).")
S5_EFFICIENCY_NOTE = (
    "S5 efficiency ranking: models sorted by s5 @L256 score, then by s5@128 mean "
    "completion tokens per call (lower is better) on the matched s5_concrete "
    f"L{S5_EFF_LENGTH} cell (the cell every current-roster model runs). "
    "At s5 @L256 several models hit 1.00, so token efficiency is the practical discriminator.")
CTOK_STATISTIC_NOTE = (
    "Two completion-token statistics appear, and they are named apart everywhere. "
    "**mean ctok/call** is the cell's usage total divided by n — the efficiency "
    "columns and the report quote this one, because it is the token spend a "
    "practitioner pays. **median ctok/call** is the per-example median; it is used "
    "only as the instant regime's visible-working trigger (†), where a handful of "
    "long completions must not drag a cell of bare answers over the line.")
WORK_RATE_NOTE = (
    f"worked calls: the fraction of a cell's calls whose completion exceeds {WORK_LINE} "
    "completion tokens, with (match | worked / match | unworked) beside it where "
    "the cell has calls of both kinds. The signal is completion tokens, not "
    "reasoning tokens: reasoning-token accounting is not comparable across "
    "providers (on this history the median ctok-minus-rtok is 5184 for sonnet-5, "
    "4184 for opus and 1675 for fable, but 2-17 for every other model, and glm "
    "reports rtok=0 on correct 8k-token answers). Cells run before per-example "
    "token logging report n/a.")
UNWORKED_FOOTNOTE = (
    f"({UNWORKED_MARK}) unworked answers on a large fraction of calls; the cell "
    "measures engagement, not capability. Set when more than "
    f"{UNWORKED_RATE:.0%} of a thinking cell's calls fall below the "
    f"{WORK_LINE}-token working line AND those calls score materially below the "
    "cell's worked calls (disjoint Wilson 95% intervals) — the rate alone is not "
    "enough, since answering a shallow cell correctly without visible working is a "
    f"measurement. Like {CENSORED_CELL} and ≤x†, a marked cell takes no part in "
    "orderings.")


def unworked_footnote(records) -> str:
    """UNWORKED_FOOTNOTE plus the engagement variance the marked cells carry: the
    widest run-to-run spread among the roster's marked thinking cells run more
    than once at identical settings. It is reported here rather than in the
    thinking noise bar because it varies with how often the model engaged, which
    is what the mark names."""
    spreads = marked_noise_spreads(records)
    if not spreads:
        return UNWORKED_FOOTNOTE
    tail = (" Engagement moves between runs: the marked cells repeated at "
            f"identical settings spread up to {max(spreads):.2f} run to run")
    bar = noise_bar_spreads(records)
    if bar:
        tail += (f", against {max(bar):.2f} across the cells that set the thinking "
                 "noise bar")
    return UNWORKED_FOOTNOTE + tail + "."
TRUNC_FOOTNOTE = (
    "(trunc 0.NN) the cell's truncation rate, where some calls ended finish=length: "
    "those calls score 0 whatever the model knew, so the published score is a lower "
    f"bound. Above 0.50 the cell renders {CENSORED_CELL} instead of a number.")
EVENT_BLIND_NOTE = (
    "event-blind: the fraction of a cell's predictions equal to the 8-hop "
    "dereference of the INITIAL pointer map — the answer a model gives if it reads "
    "the fact block and skips the whole event stream. Items where that answer "
    "coincides with the gold answer are dropped (it coincides at chance, 0.063-0.078 "
    "across lengths against 1/16), so the rate names which cheaper task the model "
    "substituted rather than crediting luck.")
# The probe behind PROMPT_PROBE_NOTE: three arms over the identical deterministic
# items, differing only in the system prompt (results/probes/, script in scripts/).
# Its event-blind rates are quoted through the PUBLISHED column's eligibility rule
# (``event_blind_counts``: drop the items whose blind answer is the gold answer),
# not through the probe script's own unfiltered hits/n — one item of the L64 stream
# has blind == gold, so the two rules differ by 24 vs 25 in the denominator and by
# that item in every arm's numerator. tests/test_render_benchmark.py recomputes the
# quoted numbers from the raw probe rows.
PROMPT_PROBE_PATH = "results/probes/sol_system_prompt_20260727.json"
PROMPT_PROBE_NOTE = (
    "The work rate is a property of the model AND of the system prompt the "
    "benchmark sends, not of the model alone. Three arms over the identical "
    "deterministic items (openai/gpt-5.6-sol, s5_chain_v3 @L64, n=25, top effort, "
    "49,152-token budget) differ only in that prompt. Under the scored protocol "
    'prompt ("You are taking a short test... no explanation") the model matches '
    "0.68, works 0.68 of its calls and answers event-blind on 0.33; with the two "
    "clauses that read as instructions to spend less effort removed and the "
    "identical answer-format contract kept, it matches 0.96, works 0.96 and "
    "answers event-blind on 0.04 — 7 of the 8 calls the scored prompt left "
    "unworked are worked under the neutral one. The third arm sends no system "
    "prompt at all: it works 0.96 and matches 0.84, and that 0.84 is a format "
    "reading rather than a composition one — all 24 of its worked calls carry the "
    "gold value, three of them committed in LaTeX (`**Answer: \\(g15\\)**`, "
    "`\\boxed{g0}`) that the committed-answer rule does not read. The event-blind "
    "rates run through the published column's eligibility rule, so they read "
    "against it. Every scored thinking cell carries the scored prompt, so the "
    "completion-token columns are token spend under an instruction to be brief. "
    f"One model, one length, n=25: `{PROMPT_PROBE_PATH}`.")
# v1-only facets whose headline scalars are kept in separate archived tables
# (the historical facet name dose_response stays in the archived headers only —
# 'dose' terminology is purged from all active labels/prose).
V1_ARCHIVED_FACETS = ("dose_response", "composite_length", "decomposition")
ARCHIVED_COLUMNS = [
    "Model", "dose_response (match)", "composite_length (match @ L512, high)",
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
    """MEDIAN visible completion tokens per call of the CANONICAL attempt (the
    aggregate mean where per-example ctok is absent; escalated cells store only
    aggregates for the first attempt, so their mean is used).

    This is the instant regime's visible-working trigger (†) and nothing else — a
    median so that a few long completions cannot drag a cell of bare answers over
    CTOK_WORKING_LINE. The efficiency columns and the report quote the MEAN
    ctok/call (usage total / n): those name token spend, and they are labelled
    'mean ctok/call' wherever they render (CTOK_STATISTIC_NOTE)."""
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


def rtok_any_rate(rec):
    """Fraction of the CANONICAL attempt's calls that emitted ANY reasoning
    tokens (per-example rtok > 0); None when per-example data is unavailable
    (escalated cells store only aggregates for the canonical first attempt)."""
    if escalation_first(rec) is not None:
        return None
    rs = _ex_vals(rec, "rtok")
    if not rs:
        return None
    return sum(1 for x in rs if x > 0) / len(rs)


def pervasive_covert(rec) -> bool:
    """True when the canonical attempt shows PERVASIVE covert reasoning (rtok on
    > PERVASIVE_RTOK_RATE of calls): the cell's score is an upper bound on
    in-weights ability — rendered '≤x†', excluded from figure sorts and from any
    cross-model ordering (the symmetric twin of ⊘ budget censoring)."""
    rate = rtok_any_rate(rec)
    return rate is not None and rate > PERVASIVE_RTOK_RATE


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


def truncation_rate(rec):
    """Fraction of the cell's calls that ended finish=length — the token budget ran
    out before an answer, so those calls score 0 whatever the model knew. Read at
    every level: a cell is censored (⊘) only above 50%, but a cell truncating on a
    fifth of its calls publishes a score that is a lower bound. None when the cell
    records no finish reasons."""
    fr = (rec.get("diagnostics") or {}).get("finish_reasons") or {}
    n = rec.get("n") or 0
    if fr and n:
        return fr.get("length", 0) / n
    fins = _ex_vals(rec, "finish")
    return sum(1 for f in fins if f == "length") / len(fins) if fins else None


def majority_finish_length(rec) -> bool:
    """True when most of the cell's calls ended with finish=length (budget cutoff)."""
    rate = truncation_rate(rec)
    return rate is not None and rate > 0.5


def thinking_cell(rec) -> bool:
    """True for cells of the thinking regime (reasoning on). The instant facets and
    the effort=none/minimal arms are excluded: there a short completion is the
    answer contract being honoured, not a model declining to work."""
    return (rec.get("facet") not in INSTANT_FACETS
            and _settings(rec).get("effort") not in ("none", "minimal"))


def work_stats(rec):
    """Per-call work split of a cell from its stored per-example ctok:
    {"rate", "worked", "unworked", "acc_worked", "acc_unworked"} — the fraction of
    calls above WORK_LINE plus match conditional on working and on not working.

    None when the record predates per-example token logging (records written
    before 2026-07-17 carry no ctok/rtok/finish per example); the diagnostic then
    renders not-available, never 0."""
    ex = [e for e in rec.get("examples") or []
          if e.get("ctok") is not None and e.get("relaxed") is not None]
    if not ex:
        return None
    worked = [e for e in ex if e["ctok"] > WORK_LINE]
    unworked = [e for e in ex if e["ctok"] <= WORK_LINE]

    def acc(items):
        return sum(e["relaxed"] for e in items) / len(items) if items else None

    return {"rate": len(worked) / len(ex), "worked": len(worked),
            "unworked": len(unworked), "acc_worked": acc(worked),
            "acc_unworked": acc(unworked)}


def unworked_bound(rec) -> bool:
    """True when a thinking cell's score measures engagement rather than capability:
    more than UNWORKED_RATE of its calls carry no working AND those calls score
    materially below the worked ones (disjoint Wilson 95% intervals). Marked ᵘ and,
    like ⊘ budget censoring and ≤x† covert reasoning, excluded from orderings.

    The two conditions are both needed. A low work rate on its own says the model
    answered without visible working, which on a shallow cell is a measurement; the
    accuracy split is what shows the cell is a mixture of a model that solved the
    item and a model that guessed."""
    if not thinking_cell(rec):
        return False
    st = work_stats(rec)
    if st is None or not st["worked"] or not st["unworked"]:
        return False
    if 1.0 - st["rate"] <= UNWORKED_RATE:
        return False
    wlo, _whi = wilson_interval(round(st["acc_worked"] * st["worked"]), st["worked"])
    _ulo, uhi = wilson_interval(round(st["acc_unworked"] * st["unworked"]),
                                st["unworked"])
    return uhi < wlo


def work_cell_str(rec) -> str:
    """One work-rate table cell: '1.00' when every call carries working, else the
    worked fraction with its two conditional match scores, '0.56 (1.00/0.09)' =
    (match | worked / match | unworked). 'n/a' before per-example token logging."""
    st = work_stats(rec) if rec is not None else None
    if st is None:
        return "n/a"
    if st["unworked"] == 0:
        return f"{st['rate']:.2f}"
    return (f"{st['rate']:.2f} ({_fmt(st['acc_worked'])}/"
            f"{_fmt(st['acc_unworked'])})")


# --- event-blind rate (s5_chain) ------------------------------------------------
# The shallow-baseline discipline the project applies to ITEMS (the recency
# heuristic, the object-filter floor), applied to PREDICTIONS: name the cheaper
# task a model substituted for the one it was asked. For s5_chain the cheaper task
# is "read the fact block, skip the event stream" — dereference the query start
# chain_depth times through the INITIAL pointer map. The items are deterministic,
# so the blind answer is regenerated at render time from the same stream the cell
# ran on. On this suite that answer coincides with the gold answer at chance
# (0.063-0.078 over 1000 regenerated items per length, chance 1/16), so items where
# the two coincide are dropped and the rest are a clean read.

@functools.lru_cache(maxsize=64)
def event_blind_items(task: str, length: int, n: int):
    """[(blind answer, gold answer)] over the exact deterministic items of an
    s5_chain task: the blind answer follows meta['depth'] a0 hops from
    meta['start'] through the pointer map STATED IN THE FACT BLOCK, ignoring every
    event. () when the spec is unavailable or is not an s5_chain task."""
    spec = _task_spec(task)
    if spec is None or getattr(spec, "family", None) != "s5_chain":
        return ()
    from factworld import tasks as TK
    out = []
    for e in TK.generate(spec, "test", n=n, length=length):
        initial = dict(re.findall(r"(g\d+)'s a0 is (g\d+)\.", e.prompt))
        cur = e.meta.get("start")
        for _ in range(e.meta.get("depth") or 0):
            cur = initial.get(cur)
            if cur is None:
                break
        out.append((cur, e.answer))
    return tuple(out)


def event_blind_counts(rec):
    """(hits, eligible) for one cell, or None when the cell carries no such
    reading. ELIGIBILITY, the one rule every event-blind number on every surface
    uses: an item counts when its blind answer differs from its gold answer under
    the canonical **match** evaluator (where the two coincide the diagnostic
    cannot separate the two readings), and a HIT is a prediction that matches the
    blind answer under the same evaluator. None for non-s5_chain cells, and
    whenever the regenerated stream does not reproduce the cell's stored gold
    answers item for item — the diagnostic reports nothing rather than something
    derived from a different stream."""
    if rec.get("facet") != "s5_chain":
        return None
    ex = rec.get("examples") or []
    items = event_blind_items(rec.get("task"), rec.get("length") or 0, len(ex))
    if not items or len(items) != len(ex):
        return None
    from factworld import tasks as TK

    def norm(s):
        return (s or "").strip().rstrip(".")

    if any(norm(e.get("gold")) != norm(gold) for e, (_b, gold) in zip(ex, items)):
        return None
    hits = eligible = 0
    for e, (blind, gold) in zip(ex, items):
        if blind is None or TK.score_relaxed(blind, gold):
            continue  # ineligible: the shallow answer IS the gold answer
        eligible += 1
        hits += TK.score_relaxed(e.get("pred") or "", blind)
    return hits, eligible


def event_blind_rate(rec):
    """Fraction of a cell's ELIGIBLE predictions that are the event-blind answer
    (``event_blind_counts``); None when the cell carries no such reading."""
    counts = event_blind_counts(rec)
    if counts is None or not counts[1]:
        return None
    return counts[0] / counts[1]


def event_blind_scoped_cells(records):
    """The cells a PUBLISHED event-blind aggregate reads: the current roster's
    scored s5_chain cells — the facet's registry task (retired task versions carry
    no published number), the canonical arm, the published effort, every length.
    Archived models are excluded for the same reason they are excluded from the
    headline: a published number describes the published roster."""
    try:
        from factworld.benchmark import FACETS
        task = FACETS.get("s5_chain", {}).get("task")
    except Exception:  # pragma: no cover - environment guard
        task = None
    roster = set(roster_models(records))
    return [r for r in by_facet(records, "s5_chain")
            if r.get("model") in roster and canonical_arm(r)
            and (task is None or r.get("task") == task)
            and _settings(r).get("effort") == S5_CHAIN_EFFORT]


def event_blind_aggregate(records):
    """Roster-wide event-blind totals over ``event_blind_scoped_cells``:
    {"cells", "hits", "eligible", "by_model"} with by_model[m] = (hits, eligible).
    {} when no scoped cell carries a reading."""
    hits = eligible = cells = 0
    by_model = defaultdict(lambda: [0, 0])
    for r in event_blind_scoped_cells(records):
        counts = event_blind_counts(r)
        if counts is None:
            continue
        cells += 1
        hits += counts[0]
        eligible += counts[1]
        by_model[r["model"]][0] += counts[0]
        by_model[r["model"]][1] += counts[1]
    if not eligible:
        return {}
    return {"cells": cells, "hits": hits, "eligible": eligible,
            "by_model": {m: tuple(v) for m, v in by_model.items()}}


def event_blind_note(records) -> str:
    """EVENT_BLIND_NOTE plus the roster-wide total read off the same eligibility
    rule, so the column is read against what the rest of the roster does: the
    published counts, the heaviest single model's share of them, and the rate
    over the remaining models against the 1/16 chance rate. The base note alone
    when no scoped cell carries a reading."""
    agg = event_blind_aggregate(records)
    if not agg:
        return EVENT_BLIND_NOTE
    top, (t_hits, _t_elig) = max(agg["by_model"].items(), key=lambda kv: kv[1][0])
    rest_hits = agg["hits"] - t_hits
    rest_elig = agg["eligible"] - _t_elig
    tail = (f" Across the roster's {agg['cells']} scored s5_chain cells the blind "
            f"answer is given on {agg['hits']} of {agg['eligible']:,} eligible "
            f"items, {t_hits} of them by {top}")
    if rest_elig:
        tail += (f"; over the other {len(agg['by_model']) - 1} models the rate is "
                 f"{rest_hits} of {rest_elig:,} ({rest_hits / rest_elig:.3f}), "
                 "against 0.063 for a uniform guess")
    return EVENT_BLIND_NOTE + tail + "."


# --- repeat runs ----------------------------------------------------------------

def replicate_values(rec) -> list:
    """Canonical match of every run of this cell at identical settings, in run
    order (attached by load_latest). Length 1 for a cell run once."""
    return [v for v in (rec.get("_replicates") or []) if v is not None]


def replicate_spread(rec):
    """max - min over the cell's repeat runs, or None when it ran once."""
    vals = replicate_values(rec)
    return max(vals) - min(vals) if len(vals) > 1 else None


def replicate_str(rec) -> str:
    """' (3 runs, spread 0.24)' for a cell run more than once at identical
    settings; '' otherwise. The published number is the last run — the suffix is
    what says so."""
    spread = replicate_spread(rec)
    if spread is None:
        return ""
    return f" ({len(replicate_values(rec))} runs, spread {spread:.2f})"


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
    breadth, k_fixed, system_prompt_fp}) — every breadth/fixed-k rung is its own
    arm (records without the v3 keys read as the canonical rung).

    ``system_prompt_fp`` is in the arm for the same reason the RUNNER's resume key
    carries it (factworld.benchmark.settings_hash): a cell run under a different
    system prompt is a different measurement, and without the key an off-protocol
    record would dedup into — and publish as — the scored cell. The runner
    sentinel-drops the key at the canonical prompts, so every scored record carries
    no fingerprint here and keys exactly as it always did."""
    s = _settings(rec)
    arm = json.dumps(
        {"effort": s.get("effort"), "leg": s.get("leg"), "rendering": s.get("rendering"),
         "breadth": s.get("breadth"), "k_fixed": s.get("k_fixed"),
         "system_prompt_fp": s.get("system_prompt_fp")},
        sort_keys=True,
    )
    return (rec.get("model"), rec.get("facet"), rec.get("task"), rec.get("length"), arm)


def replicate_key(rec) -> tuple:
    """Repeat-run key: the cell key plus the token budget. Runs that share it are
    test-retest repeats of one cell; a rerun at a DIFFERENT budget is a budget
    rerun, not a repeat, and keeps its own group (the raised-budget cells that
    turn a ⊘ into a score would otherwise read as a 1.00-wide spread)."""
    return cell_key(rec) + (_settings(rec).get("max_new_tokens"),)


def load_latest(history_path: str) -> list[dict]:
    """Parse history.jsonl and keep the latest record per cell key (by ts, then file
    order). Every kept record carries ``_replicates``: the canonical match of each
    run of that cell at identical settings, in run order — a cell run three times
    publishes its last run and says so (see ``replicate_str``)."""
    latest: dict[tuple, tuple] = {}
    runs: dict[tuple, list] = defaultdict(list)
    with open(history_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            order = (rec.get("ts") or "", i)
            key = cell_key(rec)
            runs[replicate_key(rec)].append((order, rec))
            if key not in latest or order >= latest[key][0]:
                latest[key] = (order, rec)
    recs = [v[1] for v in latest.values()]
    for rec in recs:
        rec["_replicates"] = [canonical_relaxed(r)
                              for _o, r in sorted(runs[replicate_key(rec)],
                                                  key=lambda x: x[0])]
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
            f"escalated @{_effective_cap(rec)}tok diagnostic "
            f"{_fmt((rec.get('metrics') or {}).get('relaxed'))}; canonical = first "
            f"attempt @{_settings(rec).get('max_new_tokens')}tok")
    if cap_escape(rec):
        notes.append("‡ cap-escape")
    if unworked_bound(rec):
        notes.append(f"{UNWORKED_MARK} unworked answers on a large fraction of calls")
    trunc = truncation_rate(rec)
    if trunc:
        notes.append(f"truncation {trunc:.2f}")
    spread = replicate_spread(rec)
    if spread is not None:
        notes.append(f"{len(replicate_values(rec))} runs, spread {spread:.2f}")
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


def instant_excluded(model: str) -> bool:
    """True when the registry structurally skips all instant-regime facets for
    this model (e.g. grok-4.5, muse-spark-1.1, kimi-k2.6). Such models render
    only in the thinking headline, not the instant one."""
    if model not in CURRENT_ROSTER:
        return False
    try:
        from factworld.benchmark import MODELS
    except Exception:  # pragma: no cover - environment guard
        return False
    if model not in MODELS:
        return False
    skipped = set(MODELS[model].get("skip_facets", ()))
    return INSTANT_FACETS.issubset(skipped)


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
    if s.get("system_prompt_fp"):  # off-protocol prompt (canonical ones carry no key)
        parts.append(f"sysprompt={s['system_prompt_fp']}")
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


def _latest_chain_task(records, facet):
    """Latest chain task version among records for a chain facet (v2 over v1)."""
    cells = by_facet(records, facet)
    if not cells:
        return None
    newest = max(cells, key=lambda r: (r.get("ts") or "", r.get("task") or ""))
    return newest.get("task")


def stress_cell(records, facet, model, length,
                exclude_renderings=("abstract_stated",), effort=None):
    """The model's thinking state-stress cell at (facet, length) — canonical arm
    only (breadth/fixed-k rungs are separate arms), abstract-token floor rendering
    and wrapped chain_depth cells excluded. None when the cell never ran.

    Only records for the facet's CURRENT task version are used, so retired
    chain_v1/s5_chain_v1-v2 history does not shadow the current chain_v2/s5_chain_v3 results.
    ``effort`` pins the facet's canonical effort arm (e.g. s5_chain's xhigh), so
    off-protocol effort probes in history never shadow the headline cell."""
    cells = [r for r in by_facet(records, facet)
             if r["model"] == model and r.get("length") == length
             and _settings(r).get("rendering") not in exclude_renderings
             and (effort is None or _settings(r).get("effort") == effort)
             and not chain_invalid(r)
             and canonical_arm(r)]
    if not cells:
        return None
    from factworld.benchmark import FACETS
    facet_task = FACETS.get(facet, {}).get("task")
    if facet_task:
        cells = [r for r in cells if r.get("task") == facet_task]
    # latest-timestamp-wins (e.g. an effort-rerun supersedes the earlier attempt)
    return max(cells, key=lambda r: r.get("ts") or "") if cells else None


def _raised_budget_suffix(rec) -> str:
    """' @32,768tok (raised budget)' for a thinking cell rerun above its PLANNED
    budget (--budget-override); '' otherwise. The plan is the facet's per-length
    budget (falling back to its max_new_tokens, then THINKING_BUDGET), so facets
    whose standard budgets exceed 16,384 — s5_chain sizes budgets so truncation
    stays a rounding error — do not mark every cell raised."""
    cap = _settings(rec).get("max_new_tokens")
    from factworld.benchmark import FACETS
    fc = FACETS.get(rec.get("facet"), {})
    planned = fc.get("budgets", {}).get(rec.get("length")) or fc.get("max_new_tokens") or THINKING_BUDGET
    if cap and cap > planned:
        return f" @{cap:,}tok (raised budget)"
    return ""


def stress_value_str(rec) -> str:
    """One thinking state-stress headline cell: the plain match score at the
    named setting, ‡ when the cell escaped the token cap, ᵘ when the cell's
    unworked calls make the score a measure of engagement rather than capability;
    CENSORED_CELL when the cell's calls were majority finish=length (not
    measurable at this budget — never a number); a raised budget publishes with
    the number ('1.00 @32,768tok (raised budget)'); partial truncation and repeat
    runs publish as parenthesised rates ('0.52 (trunc 0.48)', '0.60 (3 runs,
    spread 0.24)'); 'n/a' when the cell never ran."""
    if rec is None:
        return "n/a"
    if majority_finish_length(rec):
        return CENSORED_CELL + _raised_budget_suffix(rec) + replicate_str(rec)
    val = _fmt(canonical_relaxed(rec))
    if cap_escape(rec):
        val += "‡"
    if unworked_bound(rec):
        val += UNWORKED_MARK
    val += _raised_budget_suffix(rec)
    trunc = truncation_rate(rec)
    if trunc:
        val += f" (trunc {trunc:.2f})"
    return val + replicate_str(rec)


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
                or pervasive_covert(rec)
                or (ctok is not None and ctok > CTOK_WORKING_LINE)
                or (rtok is not None and rtok > RTOK_LEAK_PER_CALL)):
            marks += "†"
    return marks


def zb_value_str(rec, model_minimal=False) -> str:
    """One zero-budget headline cell: canonical value + escalated diagnostic suffix
    + cleanliness marks; pervasively covert cells render as the explicit upper
    bound '≤x†'; 'n/a' when the cell never ran (F9)."""
    if rec is None:
        return "n/a"
    val = _fmt(canonical_relaxed(rec))
    if pervasive_covert(rec):
        val = "≤" + val
    if is_escalated(rec):
        val += (f" (diag {_fmt((rec.get('metrics') or {}).get('relaxed'))} "
                f"@{_effective_cap(rec)}tok)")
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


def binding_floor_bound(records, model, zb_task=None) -> bool:
    """True when the model's state-tracking (binding_only @L16) cell sits at the
    object-filter floor statistically: its Wilson CI overlaps the floor's Wilson
    CI at the same n. For these models the composed cell is floor-shaped too, so
    the gap is not a composition measurement (floor − floor ≈ 0 by construction)
    and renders GAP_FLOOR_MARK. Derived from the data, never hardcoded."""
    if zb_task is None:
        zb_task = zb_latest_task(records)
    bind = zero_budget_cell(records, model, 16, "binding_only", task=zb_task)
    if bind is None or zb_task is None:
        return False
    n = _zb_floor_n(records, zb_task)
    vals = object_filter_floor(zb_task, n)
    if vals is None:
        return False
    flo, fhi = wilson_interval(round(vals["binding_16"] * n), n)
    blo, bhi = _ci(bind)
    return blo <= fhi and flo <= bhi


def composition_gap_str(records, model, zb_task=None) -> str:
    """THE headline statistic: composition gap = state tracking (binding_only
    @L16) - composed @L16, rendered '+0.NN' with the input cells' cleanliness
    marks propagated. recall|holder is ~1.0 across the roster (scaffolded leg),
    so a free composer would score the composed cell at its binding leg; the gap
    is the composition deficit. GAP_FLOOR_MARK where the binding input sits at
    the object-filter floor (not interpretable); 'n/a' when either cell never
    ran."""
    if zb_task is None:
        zb_task = zb_latest_task(records)
    bind = zero_budget_cell(records, model, 16, "binding_only", task=zb_task)
    comp = zero_budget_cell(records, model, 16, None, task=zb_task)
    if bind is None or comp is None:
        return "n/a"
    vb, vc = canonical_relaxed(bind), canonical_relaxed(comp)
    if vb is None or vc is None:
        return "—"
    if binding_floor_bound(records, model, zb_task):
        return GAP_FLOOR_MARK
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
    f"working-set load. chain_instant runs chain_v2 d{CHAIN_INSTANT_DEPTH} on "
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
        f"instant: chain d{CHAIN_INSTANT_DEPTH} (chain_instant, chain_v2, "
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
    """s5@128 mean ctok/call: the cell's total completion tokens divided by n on the
    matched s5_concrete cell at
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


def s5_efficiency_rows(records):
    """S5 efficiency ranking: models that solve (or attempt) s5 @L256 and the
    matched completion-token efficiency of the L128 cell. Sorted by s5 @L256 score
    descending, then by s5@128 mean ctok per call ascending. The matched L128 cell is
    the fairest per-model efficiency comparison because every current-roster model
    runs it. A model whose @L256 cell is marked sorts last on its NAME: the mark
    takes the row out of the ordering, so the ctok tiebreak is not read either
    (mirroring ``sort_thinking_rows``)."""
    rows = []
    for m in roster_models(records):
        score_rec = stress_cell(records, "s5_concrete", m, S5_STRESS_LENGTH)
        eff_rec = stress_cell(records, "s5_concrete", m, S5_EFF_LENGTH)
        if eff_rec is None:
            continue
        n = eff_rec.get("n") or 0
        ctok = (eff_rec.get("usage") or {}).get("completion_tokens")
        score = stress_value_str(score_rec)
        ctok_per = ctok / n if n > 0 and ctok is not None else None
        excluded = out_of_ordering(score)
        score_num = ordering_value(score)
        rows.append((
            (1 if excluded else 0,
             -(score_num if score_num is not None else 0),
             float("inf") if excluded or ctok_per is None else ctok_per,
             m),
            [m,
             score,
             "n/a" if ctok_per is None else f"{ctok_per:.0f}"],
        ))
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows]


S5_CHAIN_STRESS_LENGTH = 96
S5_CHAIN_EXT_LENGTH = 128
S5_CHAIN_EFF_LENGTH = 64


# The marks that take a cell out of every ordering, with the plain-language
# reason each one gives. One dict, so the tables, the figure tick labels and the
# README legend cannot disagree about which marks these are or what they mean.
EXCLUSION_MARKS = {
    UNWORKED_MARK: "unworked answers on a large fraction of calls",
    "⊘": "not measurable at this budget",
    "≤": "upper bound: covert reasoning on most calls",
}


def exclusion_mark(cell) -> str:
    """The mark that takes a rendered cell out of orderings, '' when the cell
    takes part: ⊘ budget censoring, ≤x† pervasive covert reasoning, ᵘ unworked
    answers — the same symmetric rule for every mark that says the number is not
    a capability measurement."""
    s = str(cell)
    if UNWORKED_MARK in s:
        return UNWORKED_MARK
    if s.startswith(CENSORED_CELL):
        return "⊘"
    if s.startswith("≤"):
        return "≤"
    return ""


def out_of_ordering(cell: str) -> bool:
    """True for a rendered cell carrying an exclusion mark. Such rows sort to the
    bottom of every ranking and their value never places them there."""
    return bool(exclusion_mark(cell))


def ordering_value(cell):
    """The number a rendered cell contributes to a sort, or None when it
    contributes none. A marked cell (⊘, ≤x†, ᵘ) contributes none: its row sorts
    to the bottom and its value never places it there, so removing the mark's
    cause can only move the row, never reshuffle the models above it. Tables and
    figures still show the number — excluded from the ordering is not excluded
    from the page."""
    return None if out_of_ordering(cell) else _numeric_value(cell)


def s5_chain_rows(records):
    """s5_chain ranking: the single composite stressor (non-abelian pointer-map state
    tracking composed with serial dereference). Sorted by the @L96 score (the full-roster
    cell), then by the @L128 top-cluster separator, then by s5_chain@64 mean ctok per call
    ascending (the matched L64 cell every current-roster model runs); a cell marked ᵘ or ⊘
    contributes no value to the sort (``ordering_value``), so a model whose @L96 cell is
    marked sorts last on its NAME — neither tiebreak is read for it, since both would
    order it on numbers read off a cell the mark took out of the ordering. Two diagnostic
    columns read the @L96 cell: the work rate (with match conditional on working and on not
    working) and the event-blind rate. Models without an L128 cell render — there."""
    rows = []
    for m in roster_models(records):
        score_rec = stress_cell(records, "s5_chain", m, S5_CHAIN_STRESS_LENGTH,
                                effort=S5_CHAIN_EFFORT)
        ext_rec = stress_cell(records, "s5_chain", m, S5_CHAIN_EXT_LENGTH,
                              effort=S5_CHAIN_EFFORT)
        eff_rec = stress_cell(records, "s5_chain", m, S5_CHAIN_EFF_LENGTH,
                              effort=S5_CHAIN_EFFORT)
        if score_rec is None and eff_rec is None:
            continue
        score = stress_value_str(score_rec)
        ext = "—" if ext_rec is None else stress_value_str(ext_rec)
        n = (eff_rec.get("n") or 0) if eff_rec else 0
        ctok = (eff_rec.get("usage") or {}).get("completion_tokens") if eff_rec else None
        ctok_per = ctok / n if n > 0 and ctok is not None else None
        excluded = out_of_ordering(score)
        score_num = ordering_value(score)
        ext_num = None if excluded else ordering_value(ext)
        blind = event_blind_rate(score_rec) if score_rec is not None else None
        rows.append((
            (1 if excluded else 0,
             -(score_num if score_num is not None else 0),
             -(ext_num if ext_num is not None else -1),
             float("inf") if excluded or ctok_per is None else ctok_per,
             m),
            [m,
             score,
             ext,
             work_cell_str(score_rec),
             "n/a" if blind is None else f"{blind:.2f}",
             "n/a" if ctok_per is None else f"{ctok_per:.0f}"],
        ))
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows]


REPEAT_COLUMNS = ["Model", "Facet", "Task", "Length", "Arm", "runs", "match per run",
                  "spread", "published (last run)"]
REPEAT_NOTE = (
    "Cells run more than once at identical settings (same model, facet, task, "
    "length, arm and token budget). The tables above publish the LAST run; the "
    "spread column is the run-to-run range of the same cell. A rerun at a "
    "different token budget is a budget rerun, not a repeat, and is not listed "
    "here.")


def repeat_run_rows(records):
    """One row per cell run more than once at identical settings, widest spread
    first. [] when every cell ran once."""
    rows = []
    for r in records:
        vals = replicate_values(r)
        if len(vals) < 2:
            continue
        rows.append([r.get("model"), r.get("facet"), r.get("task"),
                     str(r.get("length", "—")), arm_label(r), str(len(vals)),
                     " / ".join(f"{v:.2f}" for v in vals),
                     f"{max(vals) - min(vals):.2f}",
                     _fmt(canonical_relaxed(r))])
    rows.sort(key=lambda x: (-float(x[7]), x[0] or "", x[1] or "", x[3]))
    return rows


def s5_chain_repeat_line(records) -> str:
    """Plain line naming the s5_chain cells of the ranking table that were run more
    than once at identical settings, with every run's match. '' when there are
    none — the ranking then has no repeat runs behind it."""
    parts = []
    for m in roster_models(records):
        for length in (S5_CHAIN_STRESS_LENGTH, S5_CHAIN_EXT_LENGTH,
                       S5_CHAIN_EFF_LENGTH):
            rec = stress_cell(records, "s5_chain", m, length, effort=S5_CHAIN_EFFORT)
            if rec is None:
                continue
            vals = replicate_values(rec)
            if len(vals) < 2:
                continue
            parts.append(f"{m} @L{length} "
                         + "/".join(f"{v:.2f}" for v in vals))
    if not parts:
        return ""
    return ("Repeat runs at identical settings, the table publishing the last run: "
            + "; ".join(parts) + ".")


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
        "in earlier runs) builds prompts IDENTICAL to the composed @L16 cell "
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


def _numeric_value(cell):
    """First numeric value in a rendered cell, or None. Strips marks and status text."""
    s = str(cell).replace("n/a", "").replace("—", "")
    m = re.search(r"(\d+\.\d+|\.\d+|\d+)", s)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def _is_floor_row(label):
    return str(label).startswith(HEURISTIC_LABEL) or str(label).startswith(OBJECT_FILTER_LABEL)


def sort_instant_rows(rows):
    """Sort model rows by composed @L16 descending; a cell carrying a mark that
    takes it out of orderings (⊘, ≤x†, ᵘ) contributes no value, so its row sorts
    last on its name. Floor rows sit at the bottom."""
    models = [r for r in rows if not _is_floor_row(r[0])]
    floors = [r for r in rows if _is_floor_row(r[0])]
    # composed @L16 is index 3 in headline rows
    def key(r):
        v = ordering_value(r[3])
        return (out_of_ordering(r[3]), v is None, -(v or 0), str(r[0]))
    models.sort(key=key)
    return models + floors


def sort_thinking_rows(rows):
    """Sort model rows by s5 @L256 descending, then s5@128 mean ctok ascending; a
    cell carrying a mark that takes it out of orderings (⊘, ≤x†, ᵘ) contributes no
    value, so its row sorts last on its name. Floor rows sit at the bottom."""
    models = [r for r in rows if not _is_floor_row(r[0])]
    floors = [r for r in rows if _is_floor_row(r[0])]
    # s5 @L256 is index 8, ctok is index 9
    def key(r):
        s5 = ordering_value(r[8])
        ctok = _numeric_value(r[9]) if s5 is not None else None
        return (out_of_ordering(r[8]), s5 is None, -(s5 or 0),
                ctok if ctok is not None else float("inf"), str(r[0]))
    models.sort(key=key)
    return models + floors


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


# --- README frontier block (§2 rot-proofing) ------------------------------------
# The README's "Benchmarking the frontier" table regenerates from the SAME data
# as the results.md headline: the block between the two marker comments is
# rewritten on every render, so it can never drift from history. Cells use the
# README's compact conventions (below); recall sanity, replicate noise and the
# ctok column stay in results.md / the report.

README_FRONTIER_START = "<!-- FRONTIER_TABLE_START -->"
README_FRONTIER_END = "<!-- FRONTIER_TABLE_END -->"
README_INSTANT_COLUMNS = ["Model", "binding @L16", "composed @L16",
                          "composed @L64", "gap"]
README_THINKING_COLUMNS = ["Model", "chain d128", "s5 @L256", "s5@128 mean ctok/call"]
# headline_rows column indices the README tables keep (model is 0).
README_INSTANT_KEEP = (0, 2, 3, 4, 5)   # model, binding, composed@L16/@L64, gap
README_THINKING_KEEP = (0, 7, 8, 9)     # model, chain d128, s5 @L256, s5@128 mean ctok
_DIAG_SUFFIX = re.compile(r" \(diag \d+\.\d+ @[\d,]+tok\)")
_RAISED_SUFFIX = re.compile(r" @[\d,]+tok \(raised budget\)")
_RUNS_SUFFIX = re.compile(r" \(\d+ runs, spread \d\.\d\d\)")
RAISED_MARK = "ʳ"  # compact raised-budget mark; the legend names the budget itself
# The README's ONE marks legend, generated with the tables so a mark can never be
# in the block without a line explaining it (or explained without appearing). Each
# entry is (mark, gloss); ``_readme_mark_legend`` keeps the entries whose mark is
# actually in the rendered block, in this order.
README_MARKS = [
    ("†", "visible working or covert reasoning on the canonical attempt"),
    ("≤x†", "an explicit upper bound: covert reasoning on most calls"),
    (UNWORKED_MARK, "unworked answers on a large fraction of calls; the cell "
                    "measures engagement, not capability"),
    ("⊘", "not measurable at this budget (majority of calls finish=length)"),
    ("(trunc 0.NN)", "the fraction of the cell's calls that ran out of budget "
                     "before an answer; those calls score 0, so the cell's score "
                     "is a lower bound"),
    ("*", "off-arm ran effort=minimal (the endpoint cannot disable reasoning)"),
    (RAISED_MARK, "single rerun at a raised token budget"),
    ("‡", "provider ignored the token cap"),
    (GAP_FLOOR_MARK, "gap not interpretable: the binding input sits at the "
                     "object-filter floor"),
    ("n/a", "cell not run"),
    ("—", "not applicable to this row"),
]
README_ORDERING_MARKS = ("≤x†", UNWORKED_MARK, "⊘")
README_ORDERING_LINE = (
    "{subject}: {pronoun} published, but {take} no part in any ordering — not in "
    "these tables' sorts, not in the figures.")


def _readme_compact(cell: str) -> str:
    """README compact form of one headline cell: escalation diagnostics collapse
    to the canonical value + mark ('0.62 (diag 0.76 @512tok)†' -> '0.62†'),
    raised-budget reruns render the ʳ mark ('1.00 @32,768tok (raised budget)' ->
    '1.00ʳ'; a censored raised cell -> '⊘ʳ'), the repeat-run suffix drops (the
    repeat-run line under the table carries every run's value); '⊘ >budget' -> '⊘'.
    The truncation rate does NOT compact to a mark — a cell truncating on a fifth
    of its calls and one truncating on a single call are not the same finding, so
    the rate travels to every surface. ≤x†, ᵘ, —ᶠ, n/a and the *, †, ‡ marks pass
    through unchanged."""
    raised = bool(_RAISED_SUFFIX.search(cell))
    cell = _RUNS_SUFFIX.sub("", cell)
    cell = _RAISED_SUFFIX.sub("", cell)
    cell = _DIAG_SUFFIX.sub("", cell)
    cell = cell.replace(CENSORED_CELL, "⊘")
    return cell + (RAISED_MARK if raised else "")


def _readme_row_label(label: str) -> str:
    """Model column of the README table: floor rows drop the task-version tag
    (the README block is one table; the tag lives in results.md) and italicize."""
    if label.startswith(HEURISTIC_LABEL):
        return "*recency heuristic (floor)*"
    if label.startswith(OBJECT_FILTER_LABEL):
        return "*object-filter floor*"
    return label


def _readme_table_lines(columns, keep, rows):
    """Build one markdown table from headline rows sliced by ``keep``."""
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "---|" * len(columns)]
    for row in rows:
        label = str(row[keep[0]])
        # floor rows belong to the instant table only
        if label.startswith(HEURISTIC_LABEL) or label.startswith(OBJECT_FILTER_LABEL):
            if keep[0] != 0 or columns[1] != "binding @L16":
                continue
        lines.append("| " + " | ".join(
            [_readme_row_label(label)] + [_readme_compact(str(row[i])) for i in keep[1:]]
        ) + " |")
    return lines


# How each legend entry is detected in the rendered block. Substring matching
# would over-fire (every † inside a ≤x†, the italic asterisks of a floor row's
# label), so each mark names the shape it actually takes in a cell.
_MARK_PATTERNS = {
    "†": r"†",
    "≤x†": r"≤",
    UNWORKED_MARK: re.escape(UNWORKED_MARK),
    "⊘": r"⊘",
    "(trunc 0.NN)": r"\(trunc \d\.\d\d\)",
    "*": r"\d\*",
    RAISED_MARK: re.escape(RAISED_MARK),
    "‡": r"‡",
    GAP_FLOOR_MARK: re.escape(GAP_FLOOR_MARK),
    "n/a": r"n/a",
    "—": r"\| — \|",
}


def _raised_budgets(*rowsets) -> list:
    """The distinct token budgets behind the ʳ mark in the block ('98,304'),
    read off the uncompacted cells so the legend names the budget that was
    actually run instead of a number written down once and left to rot."""
    out = []
    for rows in rowsets:
        for row in rows:
            for cell in row:
                m = _RAISED_SUFFIX.search(str(cell))
                if m:
                    tok = re.search(r"@([\d,]+)tok", m.group(0)).group(1)
                    if tok not in out:
                        out.append(tok)
    return out


def _readme_mark_legend(table_lines, raised) -> list:
    """The block's ONE marks legend: the README_MARKS entries whose mark appears
    in the rendered tables, in fixed order, plus the ordering rule when a mark
    that leaves orderings is among them. [] when the block carries no marks."""
    body = "\n".join(line for line in table_lines if line.startswith("|"))
    items, marks = [], []
    for mark, gloss in README_MARKS:
        if not re.search(_MARK_PATTERNS[mark], body):
            continue
        if mark == RAISED_MARK and raised:
            gloss += " (" + " / ".join(f"{b} tokens" for b in raised) + ")"
        marks.append(mark)
        items.append(f"- `{mark}` {gloss}.")
    if not items:
        return []
    lines = ["**Marks**", ""] + items
    excl = [f"`{m}`" for m in README_ORDERING_MARKS if m in marks]
    if excl:
        if len(excl) == 1:
            fields = {"subject": f"A cell marked {excl[0]} is not a capability "
                                 "measurement", "pronoun": "it is", "take": "takes"}
        else:
            joined = " and ".join([", ".join(excl[:-1]), excl[-1]])
            fields = {"subject": f"Cells marked {joined} are not capability "
                                 "measurements", "pronoun": "they are",
                      "take": "take"}
        lines += ["", README_ORDERING_LINE.format(**fields)]
    return lines


def update_readme_frontier(records, readme_path=None) -> bool:
    """Rewrite the README's marked frontier block (README_FRONTIER_START/_END)
    from the rendered records: the s5_chain headline ranking first, then the
    component tables (instant composition, thinking state stress), then the marks
    legend for exactly the marks those tables use. No-op — returns False, file
    untouched — when the file or either marker is absent, so histories rendered
    outside the repo never grow tables.

    Returns True only when the file was actually rewritten, so a caller that
    reports the rewrite reports it when it happened: re-rendering an unchanged
    history returns False."""
    if readme_path is None:
        readme_path = os.path.join(REPO, "README.md")
    if not os.path.exists(readme_path):
        return False
    with open(readme_path, encoding="utf-8") as fh:
        text = fh.read()
    start = text.find(README_FRONTIER_START)
    end = text.find(README_FRONTIER_END)
    if start < 0 or end < start:
        return False
    rows = list(headline_rows(records))
    s5c_lines = []
    s5c = s5_chain_rows(records)
    if s5c:
        s5c_lines = ["**s5_chain**", "",
                     "| Model | s5_chain @L96 | @L128 | worked calls @L96 | "
                     "event-blind @L96 | mean ctok/call @L64 |",
                     "|---|---|---|---|---|---|"]
        for m, score, ext, work, blind, ctok in s5c:
            s5c_lines.append(f"| {m} | {_readme_compact(score)} | "
                             f"{_readme_compact(ext)} | {work} | {blind} | {ctok} |")
        s5c_lines += ["", WORK_RATE_NOTE, "", PROMPT_PROBE_NOTE, "",
                      event_blind_note(records), ""]
        repeat_line = s5_chain_repeat_line(records)
        if repeat_line:
            s5c_lines += [repeat_line, ""]
    instant_lines = ["**Component: instant composition (reasoning off, answer contract)**", ""]
    instant_rows = [r for r in sort_instant_rows(rows) if not instant_excluded(r[0])]
    instant_lines += _readme_table_lines(README_INSTANT_COLUMNS, README_INSTANT_KEEP,
                                          instant_rows)
    thinking_lines = ["", "**Components: thinking state stress (reasoning on)**", ""]
    thinking_lines += _readme_table_lines(README_THINKING_COLUMNS, README_THINKING_KEEP,
                                           sort_thinking_rows(rows))
    lines = s5c_lines + instant_lines + thinking_lines
    legend = _readme_mark_legend(lines, _raised_budgets(s5c, rows))
    if legend:
        lines += [""] + legend
    block = README_FRONTIER_START + "\n" + "\n".join(lines) + "\n" + README_FRONTIER_END
    new = text[:start] + block + text[end + len(README_FRONTIER_END):]
    if new == text:
        return False
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(new)
    return True


# --- markdown / csv -----------------------------------------------------------

def _fmt(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


MATCH_DEFINITION = (
    "Canonical metric: **match** — strip a trailing period from both sides and "
    "compare the model's first len(gold) whitespace tokens to the gold answer; "
    "binary per item, no partial credit (`factworld.tasks.score_relaxed`). "
    "Containment is the one published diagnostic.")


def _settings_block(records) -> list[str]:
    """Distinct (effort, max_new_tokens, stop_at) combos observed, from the
    records, each annotated with the facets that ran under it."""
    combos = defaultdict(set)
    for r in records:
        s = _settings(r)
        key = (str(s.get("effort") or "default"), str(s.get("max_new_tokens")),
               str(s.get("stop_at")))
        combos[key].add(r.get("facet") or "?")
    lines = ["## Settings", "",
             MATCH_DEFINITION,
             f"Figures draw a dotted reference line at match {REF_THRESHOLD}.",
             "Error bars / intervals: Wilson 95% CI.", "",
             "Observed generation settings (effort -> max_new_tokens, stop_at; "
             "annotated with the facets that ran under each combo):", ""]
    for (effort, mnt, stop), facets in sorted(combos.items()):
        lines.append(f"- effort={effort}: max_new_tokens={mnt}, stop_at={stop} — "
                     f"facets: {', '.join(sorted(facets))}")
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
    inst_cols = instant_headline_columns(zb_task)
    think_cols = thinking_headline_columns()
    lines += ["## Instant headline (current roster)", "",
              "Current roster only (factworld.benchmark.MODELS); models dropped from "
              "the roster render in the archived-models section below.",
              "",
              COMPOSITION_NOTE,
              "",
              f"Instant cells: task **{zb_task or 'n/a'}** with reasoning "
              "off (effort=none) under a one-line answer contract "
              "(settings.contract=true); match. Escalated cells show the "
              "CANONICAL first attempt at the shared base budget, with the escalated "
              "rerun as a parenthesised diagnostic.",
              "",
              NOTATION_NOTE,
              ""]
    note = _zb_mixed_task_note(records, zb_task)
    if note:
        lines += [note, ""]
    lines += ["| " + " | ".join(inst_cols) + " |",
              "|" + "---|" * len(inst_cols)]
    for row in sort_instant_rows(list(headline_rows(records))):
        if instant_excluded(row[0]):
            continue
        lines.append("| " + " | ".join(str(c) for c in row[:len(inst_cols)]) + " |")
    lines += ["", FLOOR_NOTE, ""]
    if any(not canonical_rung(r) for r in records):
        # breadth/fixed-k rung cells exist: they are separate arms (labelled
        # B=... / k_fixed=... in the per-cell tables, excluded from the headline
        # columns/figures above) and their floors are per-(m, L), not per-rung.
        lines += [BREADTH_FLOOR_NOTE, ""]
    for note in ZB_FOOTNOTES:
        lines += [note, ""]
    lines += [SYMMETRY_NOTE, "", GAP_NOTE, "", replicate_note(records), ""]

    lines += ["## Thinking headline (current roster)", "",
              "Thinking-regime state-stress cells (effort=high): chain d128 is a pointer "
              f"chase 128 hops deep at fixed breadth k={CHAIN_STRESS_K}; s5 @L256 is non-abelian "
              "state tracking over 256 events. The s5@128 mean ctok/call column measures efficiency on the "
              "matched L128 cell that every current-roster model runs.",
              "",
              NOTATION_NOTE,
              ""]
    lines += ["| " + " | ".join(think_cols) + " |",
              "|" + "---|" * len(think_cols)]
    for row in sort_thinking_rows(list(headline_rows(records))):
        # floor rows belong to the instant table only
        label = str(row[0])
        if label.startswith(HEURISTIC_LABEL) or label.startswith(OBJECT_FILTER_LABEL):
            continue
        lines.append("| " + " | ".join(str(c) for c in [row[0]] + row[len(inst_cols):]) + " |")
    lines += ["", thinking_noise_note(records), "", EFFICIENCY_NOTE, "", CTOK_STATISTIC_NOTE, "",
              PROMPT_PROBE_NOTE, "", unworked_footnote(records), "", TRUNC_FOOTNOTE, ""]

    s5_eff = s5_efficiency_rows(records)
    if s5_eff:
        lines += ["## S5 efficiency ranking", "",
                  S5_EFFICIENCY_NOTE, "",
                  "| Model | s5 @L256 | s5@128 mean ctok/call |",
                  "|---|---|---|"]
        for r in s5_eff:
            lines.append("| " + " | ".join(r) + " |")
        lines.append("")

    s5c = s5_chain_rows(records)
    if s5c:
        lines += [
            "## s5_chain ranking (headline)", "",
            "s5_chain is the headline composite stressor: k=16 agents with an a0 pointer map, "
            "L order-sensitive swap/cycle events on the pointer targets, then an 8-hop serial "
            "dereference query (`what is a0 of ... of gX? (8 hops)`). Every item is gated so the "
            "query path visits 9 distinct agents: answering the queried agent, or any fixed hop, "
            "scores exactly 0, and chance is 1/16. Protocol: maximum supported reasoning effort "
            "(xhigh), budgets sized so truncation stays a rounding error, n=25 per cell. Sorted "
            "by the @L96 score (the full-roster cell), then by the @L128 top-cluster separator, "
            "then by mean completion tokens per call on the matched @L64 cell. A cell marked "
            f"`{UNWORKED_MARK}` or `{CENSORED_CELL}` takes no part in the ordering: its row "
            "sorts last on its name whatever its score, here and in the figure.",
            "",
            "| Model | s5_chain @L96 | @L128 | worked calls @L96 | event-blind @L96 | "
            "s5_chain@64 mean ctok/call |",
            "|---|---|---|---|---|---|"]
        for r in s5c:
            lines.append("| " + " | ".join(r) + " |")
        lines += ["", WORK_RATE_NOTE, "", PROMPT_PROBE_NOTE, "",
                  event_blind_note(records), "", unworked_footnote(records), ""]
        repeat_line = s5_chain_repeat_line(records)
        if repeat_line:
            lines += [repeat_line, ""]

    lines += [
        "The chain column reads the `chain_nowrap` facet only (staircase k=2d+1, so the "
        f"d{CHAIN_STRESS_DEPTH} cell is k={CHAIN_STRESS_K}). `chain_v2` builds a single "
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
                  f"(historical facet names). {NONCOMPARABLE_NOTE} Their per-cell "
                  "rows — any facet — remain in the tables below.", "",
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
                  f"superseded by the ladder headline above. {NONCOMPARABLE_NOTE} "
                  "Per-cell rows remain in the tables below.", "",
                  "| " + " | ".join(ARCHIVED_COLUMNS) + " |",
                  "|" + "---|" * len(ARCHIVED_COLUMNS)]
        for row in archived:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    valid = [r for r in records if not chain_invalid(r)]
    invalid = [r for r in records if chain_invalid(r)]

    def _cell_row(r) -> str:
        mt = r.get("metrics") or {}
        lo, hi = _ci(r)
        return (
            f"| {r['model']} | {r['facet']} | {r['task']} | {r.get('length', '—')} | "
            f"{arm_label(r)} | {r.get('n', '—')} | {_fmt(canonical_relaxed(r))} "
            f"[{lo:.2f}, {hi:.2f}] | {_fmt(mt.get('contains'))} | {cell_note(r)} |")

    CELL_HEADER = ("| Model | Facet | Task | Length | Arm | n | match [95% CI] | "
                   "containment (diagnostic) | note |",
                   "|---|---|---|---|---|---|---|---|---|")
    lines += ["## Full per-cell results", "",
              "match is the CANONICAL value (first attempt for escalated cells; the "
              "escalated diagnostic is in the note column). ‡ = cap-escape (see headline "
              "footnotes). INVALID chain_depth cells are quarantined in the provenance "
              "section at the end.", "",
              *CELL_HEADER]
    for r in valid:
        lines.append(_cell_row(r))
    lines.append("")

    lines += ["## Diagnostics per cell", "",
              "finish_errors counts per-example finish=='error' calls (surfaced even where "
              "diagnostics.api_errors is 0). ctok = completion tokens; rtok = reasoning "
              "tokens. The worked-calls, event-blind and truncation columns are the "
              "per-cell diagnostics described under the headline tables; instant cells "
              "render — for worked calls, where a short completion is the answer "
              "contract rather than a model declining to work.",
              "", WORK_RATE_NOTE, "", PROMPT_PROBE_NOTE, "",
              event_blind_note(records), "",
              "| Model | Facet | Task | Length | Arm | empty_rate | api_errors | "
              "finish_errors | reasoning_tokens | worked calls | event-blind | "
              "truncation | finish_reasons | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in valid:
        d = r.get("diagnostics") or {}
        u = r.get("usage") or {}
        fr = d.get("finish_reasons") or {}
        fr_s = ", ".join(f"{k}:{v}" for k, v in sorted(fr.items())) or "—"
        blind = event_blind_rate(r)
        trunc = truncation_rate(r)
        lines.append(
            f"| {r['model']} | {r['facet']} | {r['task']} | {r.get('length', '—')} | "
            f"{arm_label(r)} | {_fmt(d.get('empty_rate'), 3)} | {d.get('api_errors', '—')} | "
            f"{finish_error_count(r)} | {u.get('reasoning_tokens', '—')} | "
            f"{work_cell_str(r) if thinking_cell(r) else '—'} | "
            f"{'—' if blind is None else f'{blind:.2f}'} | {_fmt(trunc)} | {fr_s} | "
            f"{cell_note(r)} |")
    lines.append("")

    repeats = repeat_run_rows(records)
    if repeats:
        lines += ["## Repeat runs", "", REPEAT_NOTE, "",
                  "| " + " | ".join(REPEAT_COLUMNS) + " |",
                  "|" + "---|" * len(REPEAT_COLUMNS)]
        for row in repeats:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    if invalid:
        lines += ["## Provenance: INVALID chain_depth cells (wrapped k=6 cycle)", "",
                  f"These cells ran chain_v1 past its design gate (depth >= k={CHAIN_CYCLE_K}, "
                  "so the pointer cycle wrapped): they measure the wrapped task, not depth, "
                  "and are excluded from every figure and headline column above. They are "
                  "kept here as provenance only; the redesigned facet is chain_nowrap.", "",
                  *CELL_HEADER]
        for r in invalid:
            lines.append(_cell_row(r))
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# Rendered-surface names: the canonical metric column exports as `match` (the
# stored record key stays metrics.relaxed — no history rewrite) and `contains`
# is the one diagnostic; the exact / last_n diagnostics are not exported.
CSV_FIELDS = [
    "run_id", "ts", "git_commit", "suite_version", "model", "served_models", "providers",
    "facet", "task", "length", "n", "effort", "leg", "rendering", "breadth", "k_fixed",
    "max_new_tokens",
    "stop_at", "format_prompt", "n_shot", "match", "match_ci_lo", "match_ci_hi",
    "contains", "empty_rate", "api_errors", "finish_errors",
    "contract_rate", "covert_cot_rate", "rtok_leak_rate", "rtok_per_call",
    "rtok_any_rate", "escalated",
    "escalated_match", "cap_escape", "finish_reasons",
    # per-call diagnostics: the work split (ctok, not rtok — see WORK_RATE_NOTE),
    # the s5_chain event-blind rate, truncation, and the repeat-run spread
    "work_rate", "match_worked", "match_unworked", "unworked_bound",
    "event_blind_rate", "truncation_rate", "runs", "replicate_matches",
    "replicate_spread",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "elapsed_s",
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
            work = work_stats(r) or {}
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
                # match is the CANONICAL value (metrics.relaxed in the stored
                # record; first attempt for escalated cells); the escalated
                # rerun exports as escalated_match (a diagnostic).
                "match": canonical_relaxed(r),
                "match_ci_lo": round(lo, 4), "match_ci_hi": round(hi, 4),
                "contains": mt.get("contains"),
                "empty_rate": d.get("empty_rate"), "api_errors": d.get("api_errors"),
                "finish_errors": finish_error_count(r),
                "contract_rate": d.get("contract_rate"),
                "covert_cot_rate": d.get("covert_cot_rate"),
                "rtok_leak_rate": d.get("rtok_leak_rate"),
                "rtok_per_call": canonical_rtok_per_call(r),
                "rtok_any_rate": rtok_any_rate(r),
                "escalated": r.get("escalated"),
                "escalated_match": mt.get("relaxed") if is_escalated(r) else None,
                "cap_escape": cap_escape(r),
                "finish_reasons": json.dumps(d.get("finish_reasons") or {}, sort_keys=True),
                # blank (not 0) where the record predates per-example token logging
                "work_rate": work.get("rate"),
                "match_worked": work.get("acc_worked"),
                "match_unworked": work.get("acc_unworked"),
                "unworked_bound": unworked_bound(r),
                "event_blind_rate": event_blind_rate(r),
                "truncation_rate": truncation_rate(r),
                "runs": len(replicate_values(r)) or None,
                "replicate_matches": ";".join(f"{v:.4f}" for v in replicate_values(r)),
                "replicate_spread": replicate_spread(r),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": u.get("reasoning_tokens"),
                "elapsed_s": r.get("elapsed_s"),
                "note": (lambda note: "" if note == "—" else note)(cell_note(r)),
            })


# --- figures ------------------------------------------------------------------

def roster_cells(cells):
    """Roster filter for figures: models no longer in factworld.benchmark.MODELS
    are dropped from every figure (their rows stay in the archived section and
    the per-cell tables). No-op when the roster is unavailable (empty)."""
    if not CURRENT_ROSTER:
        return list(cells)
    return [r for r in cells if r.get("model") in CURRENT_ROSTER]


def _color_map(models):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


# Legends sit OUTSIDE the axes (right margin) so they never cover data; the
# companion tight_layout rect reserves the margin and LEGEND_FIGSIZE widens the
# canvas so the axes keep their readable width.
LEGEND_KW = dict(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                 frameon=False)
LEGEND_RECT_RIGHT = 0.76  # tight_layout right edge that leaves legend room
LEGEND_FIGSIZE = (9.2, 4.4)


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
             f"{n_s} per cell; match (canonical); error bars: Wilson 95% CI; "
             f"current roster only; {date}",
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
    cells = roster_cells(by_facet(records, "dose_response"))
    if not cells:
        return []
    models = models_of(cells)
    colors = _color_map(models)
    efforts = [e for e in EFFORT_ORDER
               if any((_settings(r).get("effort") or "default") == e for r in cells)]
    xpos = {e: i for i, e in enumerate(efforts)}
    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)
    for i, m in enumerate(models):
        mine = sorted((r for r in cells if r["model"] == m),
                      key=lambda r: xpos[_settings(r).get("effort") or "default"])
        if not mine:
            continue
        xs = [xpos[_settings(r).get("effort") or "default"] for r in mine]
        _line_ci(ax, xs, mine, colors[m], m, offset=(i - len(models) / 2) * 0.02)
    ax.set_xticks(range(len(efforts)), efforts)
    ax.set_xlabel("reasoning effort")
    ax.set_ylabel("match accuracy")
    ax.set_title("v1 archived facet (pre-redesign): composite_copy_v1 @L16 vs reasoning effort",
                 fontsize=10, loc="left")
    _style_axes(ax)
    ax.legend(**LEGEND_KW)
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.03, LEGEND_RECT_RIGHT, 1))
    return _save(fig, out_dir, "fig_dose_response")


def fig_composite_length(records, out_dir):
    cells = roster_cells(by_facet(records, "composite_length"))
    if not cells:
        return []
    models = models_of(cells)
    colors = _color_map(models)
    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)
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
    ax.set_ylabel("match accuracy")
    ax.set_title("composite_copy_v1 vs length (solid: effort=high, dashed: reasoning off)",
                 fontsize=10, loc="left")
    _style_axes(ax)
    ax.legend(**LEGEND_KW)
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.03, LEGEND_RECT_RIGHT, 1))
    return _save(fig, out_dir, "fig_composite_length")


def _stress_fig(records, out_dir, facet, name, xlabel, title):
    """Score-vs-depth/length figure for a thinking state-stress facet."""
    cells = roster_cells(
        r for r in by_facet(records, facet)
        if _settings(r).get("rendering") != "abstract_stated"
        and canonical_arm(r))  # one line per model: canonical arm only
    if not cells:
        return []
    models = models_of(cells)
    colors = _color_map(models)
    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)
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
            # ᵘ — unworked answers on a large fraction of calls: the point measures
            # engagement, not capability, and takes no part in orderings.
            if unworked_bound(r):
                ax.annotate(UNWORKED_MARK, (r["length"], y), textcoords="offset points",
                            xytext=(-8, 5), fontsize=9, color=colors[m])
    ax.axhline(REF_THRESHOLD, color="#888888", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    xt = sorted({r["length"] for r in cells})
    ax.set_xticks(xt, [str(x) for x in xt])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("match accuracy")
    ax.set_title(title, fontsize=10, loc="left")
    _style_axes(ax)
    ax.legend(**LEGEND_KW)
    fig.text(0.01, 0.03,
             "hollow: majority finish=length — not measurable at this budget; "
             "‡: >10% of calls escaped the token cap;\n"
             f"{UNWORKED_MARK}: unworked answers on a large fraction of calls — the "
             "cell measures engagement, not capability",
             fontsize=7, color="#555555")
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.06, LEGEND_RECT_RIGHT, 1))
    return _save(fig, out_dir, name)


def fig_s5_stress(records, out_dir):
    return _stress_fig(records, out_dir, "s5_concrete", "fig_s5_horizon",
                       "permutation sequence length (log scale)",
                       "s5_concrete (thinking): match score vs length "
                       f"(dotted: {REF_THRESHOLD} reference)")


def fig_s5_chain(records, out_dir):
    """The headline figure: s5_chain score vs event-stream length, current task
    version at the facet's canonical xhigh arm only (retired-stream and
    off-protocol effort-probe cells excluded)."""
    from factworld.benchmark import FACETS
    task = FACETS.get("s5_chain", {}).get("task")
    mine = [r for r in records
            if r.get("task") == task and _settings(r).get("effort") == "xhigh"]
    return _stress_fig(mine, out_dir, "s5_chain", "fig_s5_chain",
                       "permutation events before the 8-hop dereference (log scale)",
                       "s5_chain (headline): match score vs length "
                       f"(dotted: {REF_THRESHOLD} reference)")


def _fig_chain(records, out_dir, facet, task_label):
    """Chain figure over a chain facet; chain_depth cells past the design gate
    (depth >= k=6, cycle wrap) are excluded — they measure the wrapped task."""
    valid = [r for r in records if not chain_invalid(r)]
    return _stress_fig(valid, out_dir, facet, f"fig_{facet}",
                       "chain depth (log scale)",
                       f"{task_label}: match score vs depth "
                       f"(dotted: {REF_THRESHOLD} reference)")


def bench_headline_rows(records):
    """The figure's rows, [(model, {length: record}, exclusion mark)], in plot
    order: the ranked models in table order first, then the models whose @L96
    cell carries a mark that takes it out of orderings (ᵘ, ⊘, ≤x†), which are
    PLOTTED BUT NOT RANKED. Their scores decide nothing — not their own row's
    position, not any other row's — so the figure and the table agree on what a
    mark means. Models with no plottable cell at either length drop out."""
    rows = []
    for name, score, *_rest in s5_chain_rows(records):
        recs = {L: stress_cell(records, "s5_chain", name, L, effort=S5_CHAIN_EFFORT)
                for L in (S5_CHAIN_STRESS_LENGTH, S5_CHAIN_EXT_LENGTH)}
        if all(r is None or majority_finish_length(r) for r in recs.values()):
            continue
        rows.append((name, recs, exclusion_mark(score)))
    return [r for r in rows if not r[2]] + [r for r in rows if r[2]]


def fig_bench_headline(records, out_dir):
    """The FactWorldBench noise view: one row per model, s5_chain match at L96
    (filled) and L128 (open) with Wilson 95% intervals — the visual reading
    guide for the README table (overlapping intervals are not an ordering).

    Rows come from ``bench_headline_rows``: unranked models sit below a rule at
    the foot of the figure, in grey, their tick label carrying the mark."""
    rows = bench_headline_rows(records)
    if not rows:
        return []
    ranked = [r for r in rows if not r[2]]
    unranked = [r for r in rows if r[2]]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.4))
    ys = range(len(rows))
    # Plot in ERROR space (1 - match) on a symlog axis so the 0.92-1.00 cluster
    # spreads out — a true log/logit axis cannot represent exact 1.00 scores.
    # Ticks are positioned in error space but LABELED as match, so the figure
    # reads in the table's units; linthresh = one error out of 25.
    ONE_ERR = 1 / 25
    for series, (length, dy, mfc) in {
        "@L96": (S5_CHAIN_STRESS_LENGTH, -0.13, None),
        "@L128": (S5_CHAIN_EXT_LENGTH, 0.13, "white"),
    }.items():
        for excluded, color in ((False, "#1f4e79"), (True, "#9a9a9a")):
            pts = [(y + dy, recs[length])
                   for y, (_n, recs, mark) in zip(ys, rows)
                   if bool(mark) is excluded and recs.get(length) is not None
                   and not majority_finish_length(recs[length])]
            if not pts:
                continue
            errs = [1 - canonical_relaxed(r) for _y, r in pts]
            cis = [_ci(r) for _y, r in pts]
            xerr = [[max(0.0, (1 - lo) - e) for e, (lo, _hi) in zip(errs, cis)],
                    [max(0.0, e - (1 - hi)) for e, (_lo, hi) in zip(errs, cis)]]
            ax.errorbar(errs, [y for y, _r in pts], xerr=[xerr[1], xerr[0]],
                        linestyle="none", marker="o", markersize=5,
                        markerfacecolor=mfc if mfc else color, color=color,
                        capsize=2.5, elinewidth=0.9,
                        label=None if excluded else series)
    ax.set_xscale("symlog", linthresh=ONE_ERR, linscale=0.6)
    tick_match = (1.00, 0.96, 0.92, 0.84, 0.72, 0.44)
    ax.set_xticks([1 - m for m in tick_match], [f"{m:.2f}" for m in tick_match])
    ax.set_xlim(0.62, -0.004)   # decreasing: best (match 1.00) on the right
    ax.set_yticks(list(ys), [f"{n} {mark}".strip() for n, _r, mark in rows])
    for label, (_n, _r, mark) in zip(ax.get_yticklabels(), rows):
        if mark:
            label.set_color("#666666")
    # models on y: _style_axes' score-axis ylim clamp would crop past row 1
    ax.set_ylim(len(rows) - 0.5, -0.5)   # decreasing limits: best model on top
    if unranked:
        ax.axhline(len(ranked) - 0.5, color="#999999", linewidth=0.7,
                   linestyle=(0, (4, 3)))
    ax.set_xlabel("match (log-spaced toward 1.00)")
    ax.set_title("s5_chain", fontsize=10, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    if unranked:
        marks = "; ".join(f"{m} {EXCLUSION_MARKS[m]}"
                          for m in sorted({m for _n, _r, m in unranked}))
        fig.text(0.01, 0.01,
                 f"below the rule: {marks} — plotted, not ranked",
                 fontsize=7, color="#555555", va="bottom")
        fig.tight_layout(rect=(0, 0.045, 1, 1))
    else:
        fig.tight_layout()
    return _save(fig, out_dir, "fig_bench_headline")


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


def model_pervasive_covert(records, model, zb_task=None) -> bool:
    """True when ANY of the model's headline zero-budget cells shows pervasive
    covert reasoning: the model's instant numbers are upper bounds (≤x†), so it
    is excluded from figure sort orderings (plotted hatched, at the end)."""
    if zb_task is None:
        zb_task = zb_latest_task(records)
    for length, leg in ZB_HEADLINE_CELLS:
        r = zero_budget_cell(records, model, length, leg, task=zb_task)
        if r is not None and pervasive_covert(r):
            return True
    return False


def fig_zero_budget(records, out_dir):
    """The composition figure: component leg (state tracking), composed cells and
    the per-model composition gap annotation, instant regime. Like the headline
    table it shows the LATEST zero-budget task's cells only (canonical rungs —
    breadth-rung cells are separate arms); an archived task's cells stay in the
    per-cell tables. Sorted by composed @L16 among non-pervasive models;
    pervasive-covert models (≤x† upper bounds) plot hatched at the end and take
    no part in the ordering (the symmetric twin of ⊘ exclusion)."""
    zb_task = zb_latest_task(records)
    cells = roster_cells(r for r in by_facet(records, "zero_budget")
                         if r.get("task") == zb_task and canonical_arm(r))
    if not cells:
        return []

    def composite_l16(m):
        r = zero_budget_cell(records, m, 16, None, task=zb_task)
        return (canonical_relaxed(r) or 0.0) if r else -1.0

    pervasive = {m: model_pervasive_covert(records, m, zb_task)
                 for m in models_of(cells)}
    # clean sort column: composed @L16, non-pervasive models first
    models = sorted(models_of(cells),
                    key=lambda m: (pervasive[m], -composite_l16(m)))
    group_colors = {label: PALETTE[j] for j, (label, _, _) in enumerate(ZB_GROUPS)}
    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)
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
            # hatched: effort=minimal (*) cells and pervasive-covert (≤x†) models
            hatched.append(pervasive[m] or
                           "*" in zb_marks(r, model_effort_minimal(records, m)))
            escapes.append(cap_escape(r))
        if not xs:
            continue
        bars = ax.bar(xs, ys, width=width, color=group_colors[label], label=label,
                      yerr=errs, capsize=2.5, error_kw={"elinewidth": 0.9})
        for rect, flag, esc in zip(bars, hatched, escapes):
            if flag:  # * (effort=minimal) or pervasive ≤x† cells: hatched bars
                rect.set_hatch("///")
                rect.set_edgecolor("#555555")
            if esc:   # ‡: >10% of the cell's calls escaped the token cap (F5)
                ax.annotate("‡", (rect.get_x() + rect.get_width() / 2,
                                  rect.get_height()),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=9, color="#333333")
    # gap annotation just above each model's bars: binding_only@L16 -
    # composed@L16 (the composition deficit, the headline statistic); —ᶠ where
    # the binding input sits at the object-filter floor (not interpretable)
    for i, m in enumerate(models):
        bind = zero_budget_cell(records, m, 16, "binding_only", task=zb_task)
        comp = zero_budget_cell(records, m, 16, None, task=zb_task)
        if bind is None or comp is None:
            continue
        vb, vc = canonical_relaxed(bind), canonical_relaxed(comp)
        if vb is None or vc is None:
            continue
        gap_s = (GAP_FLOOR_MARK if binding_floor_bound(records, m, zb_task)
                 else f"{vb - vc:+.2f}")
        tops = [_ci(r)[1] for r in (zero_budget_cell(records, m, L, leg, task=zb_task)
                                    for _, L, leg in ZB_GROUPS) if r is not None]
        ax.annotate(f"gap {gap_s}", (i, max(tops) + 0.03), ha="center",
                    fontsize=6.5, color="#333333")
    ax.set_xticks(range(len(models)),
                  [("≤ " if pervasive[m] else "") + m
                   + zb_model_marks(records, m, task=zb_task) for m in models],
                  fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("match accuracy")
    ax.set_title(f"Composition (instant): {zb_task}, reasoning off, "
                 "one-line answer contract", fontsize=10, loc="left")
    _style_axes(ax)
    ax.set_ylim(-0.03, 1.14)  # _style_axes resets ylim; keep annotation headroom
    ax.legend(**LEGEND_KW)
    fig.text(0.01, 0.035,
             "gap = state tracking (binding_only) - composed @L16 (—ᶠ: binding at the "
             "object-filter floor, gap not interpretable);\n"
             "canonical first-attempt values (escalated reruns are table diagnostics); "
             "sorted by composed @L16 among non-pervasive models;\n"
             "hatched: effort=minimal (*) or covert reasoning on most calls (≤, upper "
             "bounds, plotted last, excluded from the ordering); ‡: cap-escape",
             fontsize=7, color="#555555", va="bottom")
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.15, LEGEND_RECT_RIGHT, 1))
    return _save(fig, out_dir, "fig_zero_budget")


def fig_decomposition(records, out_dir):
    cells = roster_cells(by_facet(records, "decomposition"))
    if not cells:
        return []
    models = models_of(cells)
    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)
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
    ax.set_ylabel("match accuracy")
    ax.set_title("Decomposition: routing legs (binding_only / end_to_end / scaffolded)",
                 fontsize=10, loc="left")
    _style_axes(ax)
    ax.legend(**LEGEND_KW)
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.03, LEGEND_RECT_RIGHT, 1))
    return _save(fig, out_dir, "fig_decomposition")


# --- profile small-multiples (per-model axis positions) -------------------------
# One panel per CURRENT-ROSTER model showing its normalized position on each
# benchmark axis (present per-axis ranks and profiles, never a single scalar —
# AGENTS.md). Inverted axes (the composition gap and the s5@128 mean ctok efficiency
# column: smaller is better) are negated before normalization so the right-hand
# end is always the roster-best. Missing cells and ⊘ budget-censored cells render
# as gaps (no bar) with their mark, never as zeros.

PROFILE_AXES = [  # (short label, inverted)
    ("binding @L16", False),
    ("composed @L16", False),
    ("gap (inv)", True),
    (f"chain d{CHAIN_STRESS_DEPTH}", False),
    (f"s5 @L{S5_STRESS_LENGTH}", False),
    ("s5@128 mean ctok (inv)", True),
]
PROFILE_AXIS_LABELS = [a for a, _ in PROFILE_AXES]
PROFILE_AXES_INSTANT = PROFILE_AXES[:3]
PROFILE_AXES_THINKING = PROFILE_AXES[3:]
PROFILE_AXIS_LABELS_INSTANT = [a for a, _ in PROFILE_AXES_INSTANT]
PROFILE_AXIS_LABELS_THINKING = [a for a, _ in PROFILE_AXES_THINKING]


def _profile_cell(value, display, marks="", status="ok"):
    return {"value": value, "display": display, "marks": marks, "status": status}


def _profile_score_cell(rec, minimal=False):
    """Profile cell from a score-valued record: the canonical match value with
    the cell's cleanliness marks (pervasively covert instant cells display as
    the upper bound ≤x); 'censored' (⊘, not measurable at this budget) when the
    cell's calls were majority finish=length; 'missing' when the cell never
    ran."""
    if rec is None:
        return _profile_cell(None, "n/a", status="missing")
    if majority_finish_length(rec):
        return _profile_cell(None, CENSORED_CELL, status="censored")
    v = canonical_relaxed(rec)
    if v is None:
        return _profile_cell(None, "—", status="missing")
    if rec.get("facet") == "zero_budget":
        marks = zb_marks(rec, minimal)
        prefix = "≤" if pervasive_covert(rec) else ""
    else:
        marks = "‡" if cap_escape(rec) else ""
        prefix = ""
    return _profile_cell(v, f"{prefix}{v:.2f}", marks)


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
        if (bcell["status"] == "ok" and ccell["status"] == "ok"
                and binding_floor_bound(records, m, zb_task)):
            # gap not interpretable: the binding input sits at the floor
            gcell = _profile_cell(None, GAP_FLOOR_MARK, status="missing")
        elif bcell["status"] == "ok" and ccell["status"] == "ok":
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


def _fig_profile_grid(records, out_dir, name, models, axis_labels, title, footnote):
    """Render a small-multiples profile grid for a subset of models and axes."""
    values_all = profile_normalized(profile_values(records))
    if not values_all or all(c["status"] != "ok"
                             for d in values_all.values() for c in d.values()):
        return []
    values = {m: values_all[m] for m in models if m in values_all}
    if not values:
        return []
    zb_task = zb_latest_task(records)
    ncols = 3
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True,
                             figsize=(FIGSIZE[0], 1.6 * nrows + 1.0))
    grid = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    n_ax = len(axis_labels)
    ys = list(range(n_ax))[::-1]  # first axis on top
    for k, (ax, m) in enumerate(zip(grid, models)):
        if m not in values:
            ax.set_visible(False)
            continue
        for y, label in zip(ys, axis_labels):
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
        ax.set_yticklabels(axis_labels if k % ncols == 0 else [""] * n_ax,
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
    fig.suptitle(title, fontsize=9)
    fig.text(0.01, 0.012, footnote, fontsize=6.5, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    return _save(fig, out_dir, name)


def fig_profiles_instant(records, out_dir):
    """Instant-regime profile grid: only models with clean instant measurements."""
    models = [m for m in roster_models(records) if not instant_excluded(m)]
    footnote = (
        "bar = min-max position across the roster on that axis; raw value + marks "
        "printed beside it. gap is inverted (smaller better); "
        f"{CENSORED_CELL} and n/a cells are gaps, not zeros; "
        f"instant cells read task {zb_latest_task(records) or 'n/a'}."
    )
    return _fig_profile_grid(
        records, out_dir, "fig_profiles_instant", models,
        PROFILE_AXIS_LABELS_INSTANT,
        "Instant profiles: normalized axis positions (right = roster-best)",
        footnote,
    )


def fig_profiles_thinking(records, out_dir):
    """Thinking-regime profile grid: all current-roster models."""
    models = list(roster_models(records))
    footnote = (
        "bar = min-max position across the roster on that axis; raw value + marks "
        "printed beside it. s5@128 mean ctok/call is inverted (smaller better); "
        f"{CENSORED_CELL} and n/a cells are gaps, not zeros."
    )
    return _fig_profile_grid(
        records, out_dir, "fig_profiles_thinking", models,
        PROFILE_AXIS_LABELS_THINKING,
        "Thinking profiles: normalized axis positions (right = roster-best)",
        footnote,
    )


# Backward-compatible alias: returns both split figures.
def fig_profiles(records, out_dir):
    return fig_profiles_instant(records, out_dir) + fig_profiles_thinking(records, out_dir)


FIGURES = [fig_zero_budget, fig_profiles_instant, fig_profiles_thinking,
           fig_dose_response, fig_composite_length, fig_s5_stress,
           fig_s5_chain, fig_bench_headline, fig_chain_depth, fig_chain_nowrap,
           fig_decomposition]


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
    inst_cols = instant_headline_columns(zb_task)
    think_cols = thinking_headline_columns()
    mixed_note = _zb_mixed_task_note(records, zb_task)
    head_rows = list(headline_rows(records))
    dropped = archived_model_rows(records)
    dropped_html = ""
    if dropped:
        dropped_html = (
            "<h2>Archived models (dropped from the roster)</h2>\n"
            '<p class="small">Models present in history but no longer in '
            "factworld.benchmark.MODELS, with their v1-facet columns (historical "
            f"facet names). {html.escape(NONCOMPARABLE_NOTE)} Their per-cell rows "
            "— any facet — remain in the tables below.</p>\n"
            + _html_table(ARCHIVED_COLUMNS, dropped))
    archived = archived_headline_rows(records)
    archived_html = ""
    if archived:
        archived_html = (
            "<h2>v1 archived facets (pre-redesign)</h2>\n"
            f'<p class="small">Legacy headline columns for the pre-redesign '
            f"v1-only facets ({html.escape(', '.join(V1_ARCHIVED_FACETS))}), "
            "current-roster models only; superseded by the ladder headline "
            f"above. {html.escape(NONCOMPARABLE_NOTE)}</p>\n"
            + _html_table(ARCHIVED_COLUMNS, archived))

    cell_rows = []
    for r in records:
        mt = r.get("metrics") or {}
        d = r.get("diagnostics") or {}
        u = r.get("usage") or {}
        lo, hi = _ci(r)
        blind = event_blind_rate(r)
        cell_rows.append([
            r["model"], r["facet"], r["task"], r.get("length", "—"), arm_label(r),
            r.get("n", "—"), _fmt(canonical_relaxed(r)), f"[{lo:.2f}, {hi:.2f}]",
            _fmt(mt.get("contains")),
            _fmt(d.get("empty_rate"), 3), d.get("api_errors", "—"),
            finish_error_count(r), u.get("reasoning_tokens", "—"),
            work_cell_str(r) if thinking_cell(r) else "—",
            "—" if blind is None else f"{blind:.2f}",
            _fmt(truncation_rate(r)), cell_note(r),
        ])
    repeat_rows = repeat_run_rows(records)
    repeat_html = ""
    if repeat_rows:
        repeat_html = (
            "<h2>Repeat runs</h2>\n"
            f'<p class="small">{html.escape(REPEAT_NOTE)}</p>\n'
            + _html_table(REPEAT_COLUMNS, repeat_rows))

    figures_html = "".join(
        f"<figure>{_inline_svg(p)}</figure>" for p in svg_paths if p.endswith(".svg")
    )

    instant_rows = [row[:len(inst_cols)] for row in sort_instant_rows(head_rows)
                    if not instant_excluded(row[0])]
    thinking_rows = [
        [row[0]] + row[len(inst_cols):]
        for row in sort_thinking_rows(head_rows)
        if not str(row[0]).startswith(HEURISTIC_LABEL)
        and not str(row[0]).startswith(OBJECT_FILTER_LABEL)
    ]
    s5_eff_rows = s5_efficiency_rows(records)
    s5_eff_html = ""
    if s5_eff_rows:
        s5_eff_html = (
            "<h3>S5 efficiency ranking</h3>\n"
            f'<p class="small">{html.escape(S5_EFFICIENCY_NOTE)}</p>\n'
            + _html_table(["Model", "s5 @L256", "s5@128 mean ctok/call"],
                          s5_eff_rows)
        )
    s5c_rows = s5_chain_rows(records)
    s5c_repeat_line = s5_chain_repeat_line(records)
    s5c_html = ""
    if s5c_rows:
        s5c_html = (
            "<h3>s5_chain ranking (headline)</h3>\n"
            '<p class="small">s5_chain is the headline composite stressor: k=16 agents with an a0 '
            "pointer map, L order-sensitive swap/cycle events on the pointer targets, then an "
            "8-hop serial dereference query. Every item is gated so the query path visits 9 "
            "distinct agents — answering the queried agent, or any fixed hop, scores exactly 0, "
            "and chance is 1/16. Sorted by the @L96 score, then by the @L128 top-cluster "
            "separator, then by mean completion tokens per call on the matched @L64 cell. "
            f"A cell marked {UNWORKED_MARK} or {CENSORED_CELL} takes no part in the "
            "ordering: its row sorts last on its name whatever its score, here and in "
            "the figure.</p>\n"
            + _html_table(["Model", "s5_chain @L96", "@L128", "worked calls @L96",
                           "event-blind @L96", "s5_chain@64 mean ctok/call"],
                          s5c_rows)
            + f'<p class="small">{html.escape(WORK_RATE_NOTE)}<br>'
            + f'{html.escape(PROMPT_PROBE_NOTE)}<br>'
            + f'{html.escape(event_blind_note(records))}<br>'
            + f'{html.escape(unworked_footnote(records))}</p>'
            + (f'<p class="small">{html.escape(s5c_repeat_line)}</p>'
               if s5c_repeat_line else "")
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
({len(records)} latest cells). Canonical metric: <strong>match</strong> — strip a trailing
period from both sides and compare the model's first len(gold) whitespace tokens to the gold
answer; binary per item, no partial credit (<code>factworld.tasks.score_relaxed</code>).
Containment is the one published diagnostic. Intervals: Wilson 95% CI.
{html.escape(NOTATION_NOTE.replace("`", ""))}
The chain column reads the <code>chain_nowrap</code> facet only (staircase k=2d+1, so the
d{CHAIN_STRESS_DEPTH} cell is k={CHAIN_STRESS_K}): <code>chain_v2</code> builds a
single k={CHAIN_CYCLE_K} pointer cycle and measures depth only for depths &lt; k
(<code>factworld/tasks.py</code> design gate), so <code>chain_depth</code> cells at depth &ge;
{CHAIN_CYCLE_K} wrapped the cycle, measure the wrapped task rather than depth, and are marked
{html.escape(CHAIN_INVALID_MARK)} below and excluded from the chain figure.</p>
<h2>Instant headline (current roster)</h2>
<p class="small">Current roster only (factworld.benchmark.MODELS); models dropped from the
roster render in the archived-models section below. {html.escape(COMPOSITION_NOTE)}</p>
<p class="small">Instant cells: task <strong>{html.escape(zb_task or "n/a")}</strong>
with reasoning off (effort=none) under a one-line answer contract (settings.contract=true);
match. Escalated cells show the CANONICAL first attempt at the shared base budget,
with the escalated rerun as a parenthesised diagnostic.
{(" " + html.escape(mixed_note)) if mixed_note else ""}</p>
{_html_table(inst_cols, instant_rows)}
<p class="small">{html.escape(FLOOR_NOTE)}</p>
<p class="small">{"<br>".join(html.escape(n) for n in ZB_FOOTNOTES)}<br>
{html.escape(SYMMETRY_NOTE)}<br>
{html.escape(GAP_NOTE)}<br>
{html.escape(replicate_note(records))}</p>
<h2>Thinking headline (current roster)</h2>
<p class="small">Thinking-regime state-stress cells (effort=high): chain d128 is a pointer
chase 128 hops deep at fixed breadth k={CHAIN_STRESS_K}; s5 @L256 is non-abelian state tracking
over 256 events. The s5@128 mean ctok/call column measures efficiency on the matched L128 cell that every
current-roster model runs.</p>
{_html_table(think_cols, thinking_rows)}
<p class="small">{html.escape(thinking_noise_note(records))}<br>
{html.escape(EFFICIENCY_NOTE)}<br>
{html.escape(CTOK_STATISTIC_NOTE)}<br>
{html.escape(PROMPT_PROBE_NOTE)}<br>
{html.escape(unworked_footnote(records))}<br>
{html.escape(TRUNC_FOOTNOTE)}</p>
{s5_eff_html}
{s5c_html}
{dropped_html}
{archived_html}
{repeat_html}
<h2>Figures</h2>
{figures_html}
<h2>All cells</h2>
<p class="small">Click a column header to sort. match is the canonical value
(first attempt for escalated cells); finish_errors counts per-example finish=='error'
calls even where api_errors is 0. {html.escape(WORK_RATE_NOTE)} {html.escape(PROMPT_PROBE_NOTE)}
{html.escape(event_blind_note(records))}</p>
{_html_table(["Model", "Facet", "Task", "Length", "Arm", "n", "match", "95% CI",
              "containment (diagnostic)", "empty_rate", "api_errors",
              "finish_errors", "reasoning_tokens", "worked calls", "event-blind",
              "truncation", "note"], cell_rows, sortable=True)}
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
    if update_readme_frontier(load_latest(args.history)):
        print("rewrote the README.md frontier block")


if __name__ == "__main__":
    main()
