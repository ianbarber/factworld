"""The technical report's benchmark tables against `results/benchmark/history.jsonl`.

`reports/factworld.tex` hand-writes the two benchmark tables — FactWorldBench
(`tab:bench`: s5_chain_v3 match at L96 and L128, completion tokens per call at L64) and
the component stress cells (`tab:components`: chain d128, s5 @L256, completion tokens per
call at s5 L128). Every number and every mark in them is a claim about a specific record in
the history file, and nothing else in the repo checks that binding.

This module parses both tables out of the LaTeX and re-derives them from history:

* the truncated set (`\\marktrunc`) — cells with at least one finish=length call, which
  score wrong, so the cell is a lower bound on its score and its spend;
* the unworked set (`\\marku`) — `render_benchmark.unworked_bound`, imported rather than
  reimplemented so the report and the rendered pages cannot drift apart;
* the printed values themselves, against the same records.

Cell selection uses the renderer's own `load_latest` dedup (latest record per cell key) and
`stress_cell` (canonical arm, current task version, effort arm pinned where the facet pins
it), so the report is checked against exactly the records the benchmark pages publish.

Run directly:  python3 tests/test_report_marks.py
Run with pytest: python3 -m pytest tests/test_report_marks.py
"""
from __future__ import annotations

import os
import re
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

# render_benchmark imports matplotlib at module scope to draw the figures. None of the
# rules used here touch plotting, and this check must run wherever the report is edited,
# so a stub stands in when the figure dependency is absent.
try:  # pragma: no cover - environment guard
    import matplotlib  # noqa: F401
except ImportError:  # pragma: no cover - environment guard
    _mpl = types.ModuleType("matplotlib")
    _mpl.use = lambda *a, **k: None
    _plt = types.ModuleType("matplotlib.pyplot")
    _mpl.pyplot = _plt
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _plt

import render_benchmark as RB  # noqa: E402

TEX = os.path.join(REPO, "reports", "factworld.tex")
HISTORY = os.path.join(REPO, "results", "benchmark", "history.jsonl")

TRUNC_MARK = r"\marktrunc"
UNWORKED_MARK = r"\marku"

# (label, column index within the row after the model name) -> the history cell the column
# publishes: (facet, length, effort arm, kind). "score" columns print match; "spend"
# columns print the cell's completion tokens per call.
COLUMNS = {
    ("tab:bench", 0): ("s5_chain", 96, "xhigh", "score"),
    ("tab:bench", 1): ("s5_chain", 128, "xhigh", "score"),
    ("tab:bench", 2): ("s5_chain", 64, "xhigh", "spend"),
    ("tab:components", 0): ("chain_nowrap", 128, None, "score"),
    ("tab:components", 1): ("s5_concrete", 256, None, "score"),
    ("tab:components", 2): ("s5_concrete", 128, None, "spend"),
}


def _tex() -> str:
    with open(TEX, encoding="utf-8") as fh:
        return fh.read()


def _table_body(tex: str, label: str) -> str:
    """The rows between \\midrule and \\bottomrule of the table carrying `label`."""
    tables = [t for t in tex.split(r"\begin{table}") if f"\\label{{{label}}}" in t]
    assert len(tables) == 1, f"expected exactly one table labelled {label}, got {len(tables)}"
    body = tables[0].split(r"\midrule")[1].split(r"\bottomrule")[0]
    return body


def _rows(tex: str, label: str) -> list[list[str]]:
    """Parsed data rows: [model, cell, cell, cell] with LaTeX cell text kept verbatim."""
    out = []
    for line in _table_body(tex, label).split(r"\\"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 2 or not cells[0]:
            continue
        out.append(cells)
    return out


def _slug(name: str, roster) -> str:
    """The history slug for a table's model name. The tables shorten one slug
    (nvidia/nemotron-3-ultra for nvidia/nemotron-3-ultra-550b-a55b)."""
    hits = [m for m in roster if m == name or m.startswith(name)]
    assert len(hits) == 1, f"model name {name!r} matches {hits} in the roster"
    return hits[0]


def _records():
    return RB.load_latest(HISTORY)


def _cell(records, model, facet, length, effort):
    return RB.stress_cell(records, facet, model, length, effort=effort)


def _published():
    """Every (label, model, column) cell the two tables publish, with its record."""
    tex = _tex()
    records = _records()
    roster = RB._current_roster()
    assert roster, "factworld.benchmark.MODELS did not load"
    out = []
    for label in ("tab:bench", "tab:components"):
        for row in _rows(tex, label):
            model = _slug(row[0], roster)
            for i, text in enumerate(row[1:]):
                facet, length, effort, kind = COLUMNS[(label, i)]
                rec = _cell(records, model, facet, length, effort)
                out.append(((label, model, i), text, rec, kind))
    return out


def _marked(cells, mark):
    return {key for key, text, _rec, _kind in cells if mark in text}


def _assert_marks(printed, expected, mark):
    missing = sorted(expected - printed)
    spurious = sorted(printed - expected)
    assert not missing and not spurious, (
        f"{mark}: {len(missing)} cell(s) need the mark {missing}; "
        f"{len(spurious)} cell(s) carry it without cause {spurious}")


def test_every_published_cell_exists_in_history():
    cells = _published()
    assert len(cells) == 13 * 6, f"expected 13 models x 6 columns, got {len(cells)}"
    missing = [key for key, _t, rec, _k in cells if rec is None]
    assert not missing, f"table cells with no history record: {missing}"


def test_truncated_cells_are_marked():
    """A cell with any finish=length call publishes a lower bound and must carry the
    truncation mark; a cell with none must not."""
    cells = _published()
    expected = {key for key, _t, rec, _k in cells
                if rec is not None and (RB.truncation_rate(rec) or 0) > 0}
    _assert_marks(_marked(cells, TRUNC_MARK), expected, TRUNC_MARK)


def test_unworked_cells_are_marked():
    """The report's unworked mark is `render_benchmark.unworked_bound` — the same rule the
    rendered pages use, so a change to the rule cannot leave the report behind."""
    cells = _published()
    expected = {key for key, _t, rec, _k in cells
                if rec is not None and RB.unworked_bound(rec)}
    _assert_marks(_marked(cells, UNWORKED_MARK), expected, UNWORKED_MARK)


def test_no_censored_or_cap_escape_cell_is_published_bare():
    """The other two contamination marks (budget censoring, provider cap escape) do not
    apply to any published cell; if one starts to, the table has to say so."""
    cells = _published()
    contaminated = [key for key, _t, rec, _k in cells
                    if rec is not None
                    and (RB.majority_finish_length(rec) or RB.cap_escape(rec))]
    assert not contaminated, f"unmarked contaminated cells: {contaminated}"


def test_published_values_match_history():
    """The printed number of every cell, against the record it claims."""
    wrong = []
    for key, text, rec, kind in _published():
        printed = re.match(r"[0-9.]+", text)
        assert printed, f"{key}: no number in cell {text!r}"
        if kind == "score":
            actual = f"{RB.canonical_relaxed(rec):.2f}"
        else:
            n = rec.get("n") or 0
            ctok = (rec.get("usage") or {}).get("completion_tokens")
            actual = f"{ctok / n:.0f}"
        if printed.group(0) != actual:
            wrong.append((key, printed.group(0), actual))
    assert not wrong, f"cells whose printed value differs from history: {wrong}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all report-mark checks passed")
