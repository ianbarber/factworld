"""Regression tests for scripts/render_blog_page.py.

Renders a fixture results.md (a faithful subset of the real artifact: the
generated stamp, the headline table with its literal-pipe replicate-noise
header, escalated parenthesised diagnostics, every cleanliness mark, ⊘-censored
cells and both floor rows, plus a trailing section the parser must not read
into) and checks the page: every section present, every published value a
verbatim match against the source, marks preserved, escalated diagnostics
dropped, floors kept, footer stamped — and that the parse-time guard raises on
drifted values instead of publishing them.

Run directly:   python3 tests/test_render_blog_page.py
Run with pytest: python3 -m pytest tests/test_render_blog_page.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import render_blog_page as BP  # noqa: E402

FIXTURE = """\
# FactWorld frontier benchmark — results

Generated 2026-07-12 01:20 UTC from `results/benchmark/history.jsonl` (544 latest cells).

## Settings

Canonical metric: **match** — strip a trailing period from both sides and compare the model's first len(gold) whitespace tokens to the gold answer; binary per item, no partial credit (`factworld.tasks.score_relaxed`). Containment is the one published diagnostic.

## Headline (current roster)

Current roster only (factworld.benchmark.MODELS).

| Model | instant: recall (sanity, recall_copy_v1) | instant: state tracking (binding_only @L16, v2) | instant: composed @L16 (match, v2) | instant: composed @L64 (v2) | instant: composition gap (binding_only - composed @L16) | instant: replicate noise (|composed - replicate| @L16) | thinking: chain d128 (chain_nowrap, k=257, match) | thinking: s5 @L256 (s5_concrete, match) | thinking: s5@128 ctok |
|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 0.78 | 0.72 | 0.43 | +0.06 | ±0.05 | 0.08 | 1.00 @32,768tok (raised budget) | 12683 |
| anthropic/claude-sonnet-5 | 0.97 | 0.77 | 0.62 (diag 0.76 @512tok)† | 0.32 (diag 0.66 @512tok)† | +0.15† | ±0.03 | 0.04 | 1.00 @32,768tok (raised budget) | 11866 |
| deepseek/deepseek-v4-pro | 1.00 | 0.51 | 0.44 | 0.19 | —ᶠ | ±0.00 | ⊘ >budget @32,768tok (raised budget) | ⊘ >budget | 10043 |
| google/gemini-3.5-flash | 1.00 | 0.66* | 0.64* | 0.28* | +0.02* | ±0.01 | 0.88 | 0.52 | 11022 |
| moonshotai/kimi-k2.6 | 1.00 | ≤0.94† | ≤0.77† | ≤0.93† | +0.17† | ±0.06 | 0.64‡ | 0.88 | 17418 |
| testlab/new-model-9 | 1.00 | 0.55 | 0.35 | 0.11 | +0.20 | ±0.02 | 0.40 | 0.90 | 9001 |
| recency heuristic (floor, composite_copy_v2) | — | 0.04 | 0.04 | 0.06 | — | — | — | — | — |
| object-filter floor (composite_copy_v2) | — | 0.41 | 0.41 | 0.15 | — | — | — | — | — |

Read small-L zero-budget cells against the object-filter floor, not chance.

## Instant stress rows (recall under load; chain d16)

A later table the headline parser must NOT read into.

| Model | instant: recall under load | instant: chain d16 |
|---|---|---|
| anthropic/claude-opus-4.8 | 1.00 | 0.00 (diag 0.96 @512tok)† |
"""

N_ROSTER = 6  # fixture roster rows (floors excluded)


def _page() -> str:
    return BP.render_page(FIXTURE)


def _table_rows(page: str) -> list[str]:
    """The blog results-table lines (header + body) of the rendered page."""
    section = page.split("## Results (current roster)")[1].split("##")[0]
    return [ln for ln in section.splitlines() if ln.startswith("|")]


def test_sections_present():
    page = _page()
    assert page.startswith("# ")                                # 1: title
    assert sum(1 for ln in page.splitlines()
               if ln.startswith("- **")) in (4, 5)              # 1: framing bullets
    assert "instant" in page.lower() and "thinking" in page.lower()
    assert "**match**" in page                                  # 1: the one metric
    assert "score_relaxed" in page                              # ...cited by function
    assert BP.TASKS_URL in page                                 # 1: worked-item pointer
    assert "## Results (current roster)" in page               # 2: table
    assert BP.COLUMN_DECODE_LINE in page                        # 2: column decode
    assert "`@Ln`" in page and "`@Ntok`" in page                # 2: notation legend
    assert "\nMarks: " in page                                  # 3: marks explainer
    assert "neither participates in orderings" in page          # 3: symmetric rule
    assert BP.THINKING_NOISE_LINE in page                       # 3: thinking noise bar
    assert "*Recency heuristic (floor)*:" in page               # 3: floor definitions
    assert "*Object-filter floor*:" in page
    assert "## Figures" in page                                 # 4: figure list
    for name, _caption, _alt in BP.FIGURES:
        assert f"`{name}`" in page
    assert page.count("*Alt text:*") == len(BP.FIGURES)
    assert "Last updated: 2026-07-12" in page                   # 5: footer
    assert f"{N_ROSTER} models" in page
    assert BP.REPO_URL in page
    # operator instructions stay OFF the page (review: figures intro + footer)
    assert "three commands" not in page
    assert "Upload from" not in page
    assert "run_frontier_benchmark" not in page


def test_kimi_note_roster_conditional():
    page = _page()
    assert BP.KIMI_NOTE in page                     # kimi is on the fixture roster
    no_kimi = FIXTURE.replace(
        "| moonshotai/kimi-k2.6 | 1.00 | ≤0.94† | ≤0.77† | ≤0.93† | +0.17† | "
        "±0.06 | 0.64‡ | 0.88 | 17418 |\n", "")
    assert BP.KIMI_NOTE not in BP.render_page(no_kimi)


def test_table_shape_and_names():
    rows = _table_rows(_page())
    assert rows[0] == ("| Model | State tracking (binding @L16) | Composed @L16 "
                       "| Composition gap | Chain d128 (thinking) "
                       "| S5 @L256 (thinking) |")
    body = rows[2:]
    assert len(body) == N_ROSTER + 2  # roster + the two floor rows
    assert body[0].startswith("| Claude Opus 4.8 |")   # known slug -> display name
    assert "| new-model-9 |" in body[5]                # unknown slug -> tail fallback


def test_every_value_appears_in_source():
    for row in _table_rows(_page())[2:]:
        for tok in BP.NUM_TOKEN.findall(row.split("|", 2)[2]):
            assert tok in FIXTURE, f"published {tok!r} not in source"


def test_marks_preserved_and_diagnostics_dropped():
    page = _page()
    assert "| 0.62† |" in page          # escalated cell: canonical value + mark
    assert "(diag 0.76 @512tok)" not in page  # ...with the rerun diagnostic dropped
    assert "| 0.66* |" in page and "| +0.02* |" in page
    assert "| ≤0.94† |" in page and "| +0.17† |" in page  # pervasive upper bound
    assert "| 0.64‡ |" in page
    assert "| —ᶠ |" in page              # floor-bound gap survives verbatim
    assert "1.00 @32,768tok (raised budget)" in page  # raised budget stated
    assert page.count("⊘ >budget") >= 2  # censored cells survive verbatim


def test_floor_rows_included():
    rows = _table_rows(_page())
    heur = [r for r in rows if r.startswith("| *recency heuristic (floor)* |")]
    objf = [r for r in rows if r.startswith("| *object-filter floor* |")]
    assert heur and heur[0].split(" | ")[1:3] == ["0.04", "0.04"]
    assert objf and objf[0].split(" | ")[1:3] == ["0.41", "0.41"]


def test_vocabulary():
    page = _page().lower()
    assert "work-in-progress" not in page
    # word-bounded so fig_s5_horizon (filename) and 'horizontal' stay legal
    for banned in ("walls?", "horizons?", "knees?", "cliffs?", "wip"):
        assert not re.search(rf"\b{banned}\b", page), banned


def _raises(fn, *args) -> bool:
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_verify_value_guards():
    ok = "0.62 (diag 0.76 @512tok)†"
    assert not _raises(BP.verify_value, "0.62†", ok, FIXTURE)
    # a number that is not in results.md must never publish
    assert _raises(BP.verify_value, "0.63†", "0.63†", FIXTURE)
    # dropping or inventing a mark must never publish
    assert _raises(BP.verify_value, "0.62", ok, FIXTURE)
    assert _raises(BP.verify_value, "0.78†", "0.78", FIXTURE)
    # a cell that was never in results.md must never publish
    assert _raises(BP.verify_value, "0.78", "0.78 fabricated", FIXTURE)


def test_malformed_source_raises():
    assert _raises(BP.render_page, "no stamp, no table")
    assert _raises(BP.render_page,
                   FIXTURE.replace("## Headline (current roster)", "## Something"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} tests passed")
