"""Render docs/benchmark/results.md into a paste-ready blog page (markdown).

Reads the rendered benchmark artifact (``docs/benchmark/results.md``, the output
of scripts/render_benchmark.py) and emits the blog page to stdout — and to
``--out`` (e.g. docs/benchmark/page.md) when given. The page is a pure function
of that artifact:

  1. title + framing bullets (what FactWorld measures, the metric in one
     sentence, the two regimes in plain words, floors/marks, repo + full
     report + worked-item links)
  2. a column-decode line, then the simplified five-column results table for
     the CURRENT roster — values and cleanliness marks parsed from the
     headline table, never hardcoded; the escalated reruns' parenthesised
     diagnostics are dropped for the blog (the canonical value + marks stay);
     the floor rows are kept
  3. a one-line marks explainer (incl. the symmetric ⊘ / ≤x† contamination
     principle and the —ᶠ floor-bound gap), the kimi covert-reasoning note,
     the thinking noise bar, + the two floor definitions, one line each
  4. the figure list with one-line captions and alt text
  5. a "Last updated" footer (date from results.md's generated stamp, roster
     size, repo link)

Every published table value is verified at parse time against results.md
(numeric tokens must appear verbatim; marks must match the source cell exactly)
— the script fails loudly rather than publish a number that drifted from the
source artifact.

Pure stdlib. Usage:
    python scripts/render_blog_page.py                       # markdown on stdout
    python scripts/render_blog_page.py --out docs/benchmark/page.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(REPO, "docs", "benchmark", "results.md")
REPO_URL = "https://github.com/ianbarber/factworld"
REPO_LABEL = "github.com/ianbarber/factworld"
REPORT_URL = f"{REPO_URL}/blob/main/docs/benchmark/results.md"
TASKS_URL = f"{REPO_URL}/blob/main/docs/tasks.md"

HEADLINE_HEADING = "## Headline (current roster)"
FLOOR_PREFIXES = ("recency heuristic (floor", "object-filter floor")
# Headline columns the blog table publishes, found by substring in the source
# header (robust to the task-version tag and the exact d/L settings).
COLUMN_NEEDLES = {
    "binding": "state tracking (binding_only",
    "composed": "composed @L16 (match",
    "gap": "composition gap",
    "chain": "thinking: chain d",
    "s5": "thinking: s5 @L",
}
KEY_ORDER = ("binding", "composed", "gap", "chain", "s5")

MARK_CHARS = "*†‡⊘"
NUM_TOKEN = re.compile(r"[+±]?\d+(?:\.\d+)?")
# "0.62 (0.76 @512)†" -> the parenthesised escalated-rerun diagnostic.
ESCALATED_DIAG = re.compile(r"\s*\([^()]*@[^()]*\)")
TABLE_SEPARATOR = re.compile(r"^\|[-: |]+\|$")

# Friendly display names for known roster slugs; unknown slugs fall back to the
# slug's tail so a new model renders without touching this script.
DISPLAY_NAMES = {
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "moonshotai/kimi-k2.6": "Kimi K2.6",
    "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra",
    "openai/gpt-5.5": "GPT-5.5",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "qwen/qwen3.7-max": "Qwen3.7 Max",
    "x-ai/grok-4.5": "Grok 4.5",
    "z-ai/glm-5.2": "GLM-5.2",
}

TITLE = ("# Frontier models can recall and track state. "
         "Combining the two is what costs them.")

FRAMING = [
    "- **What it measures.** Two abilities every long task leans on — recalling "
    "a stated fact, and tracking state through a stream of updates — measured "
    "independently and then composed into a single two-hop question. Recall is "
    "cheap for every model; state tracking is established for six of the nine — "
    "and for those six, composition is where they diverge.",
    "- **One metric.** Every cell is **match**: strip a trailing period from "
    "both sides and compare the model's first len(gold) whitespace tokens to "
    "the gold answer — binary per item, no partial credit "
    "(`factworld.tasks.score_relaxed`).",
    "- **Two regimes, plain words.** *Instant* = reasoning off, one-line answer "
    "required: what the weights compute directly. *Thinking* = a generous "
    "reasoning budget: what the model can do given room to work. The two rank "
    "the roster in very different orders, so every score is regime-labelled and "
    "there is no single leaderboard number.",
    "- **Floors and marks.** Scores are read against shallow-heuristic floors — "
    "cheats run as first-class rows — and marks flag models whose numbers need "
    "a caveat. A score only counts when the cheats can't earn it.",
    f"- **Where everything lives.** Code, tasks and the add-a-model path: "
    f"[{REPO_LABEL}]({REPO_URL}); every number with per-cell confidence "
    f"intervals: [full results]({REPORT_URL}); a worked item — one full prompt, "
    f"its gold answer, and real model mistakes: [docs/tasks.md]({TASKS_URL}).",
]

COLUMN_DECODE_LINE = (
    "How to read the columns: the first three are *instant* cells (reasoning "
    "off) — hold state through a 16-event stream, answer the composed two-hop "
    "question on the same stream, and the gap between them; the last two are "
    "*thinking* cells (reasoning on) — a 128-hop pointer chase and a 256-event "
    "permutation stream. `@Ln` = stream length in events or hops; `@Ntok` = a "
    "completion-token budget (raised budgets are stated with the number).")

MARKS_LINE = (
    "Marks: `*` the model cannot fully disable reasoning; `†` visible working "
    "or covert reasoning leaked onto a supposedly instant attempt — read it as "
    "a soft upper bound; `≤x†` covert reasoning on *most* calls — the number "
    "is an explicit upper bound; `‡` the provider did not enforce the token "
    "cap, so token comparisons are off; `⊘ >budget` the token budget ran out "
    "before an answer — not measurable at that budget, which is different from "
    "a zero; `—ᶠ` the gap is not interpretable because the model's state "
    "tracking sits at the object-filter floor (floor − floor ≈ 0 by "
    "construction). ⊘ and ≤x† are the same principle from both sides: neither "
    "participates in orderings.")

KIMI_NOTE = (
    "Kimi's composed @L64 exceeding its @L16 is the covert-reasoning artifact "
    "(reasoning tokens on most calls despite reasoning off), not a length "
    "effect.")

THINKING_NOISE_LINE = (
    "Thinking columns: n=25 per cell; Wilson intervals ≈ ±0.15–0.19, and the "
    "one thinking test-retest pair moved 0.16 — differences under ~0.2 are not "
    "an ordering.")

FLOOR_LINES = [
    "*Recency heuristic (floor)*: answer with the last event's recipient — a "
    "one-line cheat with no state tracking at all.",
    "*Object-filter floor*: filter the stream to the queried object but pick a "
    "random one of its writes — a score near this row shows filtering, not "
    "state tracking.",
]

# (filename, one-line caption, alt text) — the page's figure list. Alt text
# stays qualitative on purpose: the numbers live in the table above, parsed
# fresh from results.md on every render.
FIGURES = [
    ("fig_zero_budget.png",
     "Components vs. composition with reasoning off: state tracking beside the "
     "composed two-hop cell — the annotated gap is what composing costs each "
     "model.",
     "Grouped bar chart of the current model roster with reasoning off. For "
     "each model, bars show state tracking at length 16, the composed two-hop "
     "task at lengths 16 and 64, and a test-retest replicate, with Wilson 95% "
     "error bars and the composition gap annotated (a floor marker where the "
     "gap is not interpretable). Models are ordered by the composed length-16 "
     "score; models whose instant numbers are upper bounds from covert "
     "reasoning are hatched and placed last, outside the ordering. The legend "
     "sits outside the plot area."),
    ("fig_profiles_instant.png",
     "Instant-regime profile grid: one panel per instant-measured model, with "
     "binding, composed score, and composition gap side by side.",
     "Small-multiples panel, one chart per instant-measured model. Horizontal "
     "bars give each model's normalized position on binding at length 16, "
     "composed at length 16, and composition gap (inverted), with raw values "
     "printed beside the bars; unmeasurable cells are gaps, not zeros."),
    ("fig_profiles_thinking.png",
     "Thinking-regime profile grid: one panel per roster model, with "
     "pointer-chase depth, S5 at length 256, and token efficiency side by side.",
     "Small-multiples panel, one chart per current-roster model. Horizontal "
     "bars give each model's normalized position on chain depth 128, S5 at "
     "length 256, and completion tokens on the matched S5 cell (inverted), "
     "with raw values printed beside the bars; unmeasurable cells are gaps, "
     "not zeros."),
    ("fig_chain_nowrap.png",
     "Pointer chases with reasoning on: score vs. chain depth — the "
     "instant-regime leaders are not the leaders here.",
     "Line chart of match accuracy versus pointer-chase depth, 16 to 128 on "
     "a log scale, with reasoning enabled; one line per current-roster model "
     "(legend outside the plot area), hollow markers where the token budget "
     "ran out before an answer."),
    ("fig_s5_horizon.png",
     "Permutation state with reasoning on: tracking five people across five "
     "jobs through up to 256 swap/cycle events.",
     "Line chart of match accuracy versus permutation stream length, 16 to "
     "256 on a log scale, with reasoning enabled; one line per current-roster "
     "model (legend outside the plot area), hollow markers where the token "
     "budget ran out before an answer."),
]

FIGURES_INTRO = "Four figures carry the shape of the results:"


# --- parsing results.md ---------------------------------------------------------

def parse_generated_date(text: str) -> str:
    """The YYYY-MM-DD of results.md's 'Generated <stamp>' line."""
    m = re.search(r"^Generated (\d{4}-\d{2}-\d{2})", text, re.MULTILINE)
    if m is None:
        raise ValueError("results.md has no 'Generated <date>' stamp")
    return m.group(1)


def _split_row(line: str) -> list[str]:
    """Cells of one markdown table row. Splits on ' | ' (space-pipe-space), not
    bare '|': the source header contains literal pipes inside a cell
    ('(|composed - replicate| @L16)') that a bare split would shear through."""
    s = line.strip()
    return [c.strip() for c in s[1:-1].split(" | ")]


def _table_after(text: str, heading: str) -> tuple[list[str], list[list[str]]]:
    """(header cells, body rows) of the first markdown table under ``heading``."""
    start = text.find(heading)
    if start < 0:
        raise ValueError(f"results.md has no {heading!r} section")
    header, body, in_table = None, [], False
    for line in text[start:].splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            in_table = True
            if TABLE_SEPARATOR.match(s):
                continue
            cells = _split_row(s)
            if header is None:
                header = cells
            else:
                body.append(cells)
        elif in_table:
            break
    if header is None or not body:
        raise ValueError(f"no table under {heading!r}")
    return header, body


def parse_headline(text: str):
    """(header, column indices, roster rows, floor rows) of the headline table.
    Rows are (label, {key: source cell}) with keys from COLUMN_NEEDLES."""
    header, body = _table_after(text, HEADLINE_HEADING)
    idx = {}
    for key, needle in COLUMN_NEEDLES.items():
        hits = [i for i, h in enumerate(header) if needle in h]
        if not hits:
            raise ValueError(f"headline table has no {needle!r} column")
        idx[key] = hits[0]
    roster, floors = [], []
    for row in body:
        if len(row) != len(header):
            raise ValueError(f"ragged headline row: {row[0]!r}")
        cells = {key: row[i] for key, i in idx.items()}
        (floors if row[0].startswith(FLOOR_PREFIXES) else roster).append(
            (row[0], cells))
    if not roster:
        raise ValueError("headline table has no roster rows")
    return header, idx, roster, floors


# --- publishing guards ----------------------------------------------------------

def _marks(cell: str) -> str:
    return "".join(c for c in MARK_CHARS if c in cell)


def simplify_cell(cell: str) -> str:
    """Blog cell: the canonical value + marks only. The escalated rerun's
    parenthesised diagnostic ('0.62 (0.76 @512)†' in results.md) is dropped —
    the canonical number is the headline, the rerun stays in the full report."""
    return ESCALATED_DIAG.sub("", cell).strip()


def verify_value(published: str, source_cell: str, results_text: str) -> None:
    """Parse-time guard on one published table cell: every numeric token must
    appear verbatim in results.md, the cleanliness marks must be exactly the
    source cell's, and the source cell itself must be in results.md. Raises
    ValueError rather than publish a number that drifted from the source."""
    for tok in NUM_TOKEN.findall(published):
        if tok not in results_text:
            raise ValueError(
                f"published value {tok!r} (cell {published!r}) is not in results.md")
    if _marks(published) != _marks(source_cell):
        raise ValueError(
            f"marks drifted: published {published!r} vs source {source_cell!r}")
    if source_cell not in results_text:
        raise ValueError(f"source cell {source_cell!r} is not in results.md")


# --- page assembly ---------------------------------------------------------------

def display_name(slug: str) -> str:
    return DISPLAY_NAMES.get(slug, slug.split("/", 1)[-1])


def floor_display(label: str) -> str:
    if label.startswith("recency heuristic (floor"):
        return "*recency heuristic (floor)*"
    return "*object-filter floor*"


def page_columns(header: list[str], idx: dict) -> list[str]:
    """Blog table headers, with the lengths/depth read off the source headers
    (so a re-tuned benchmark relabels the page automatically)."""
    def grab(pattern, key, default):
        m = re.search(pattern, header[idx[key]])
        return m.group(1) if m else default

    l_bind = grab(r"@L(\d+)", "binding", "16")
    depth = grab(r"chain d(\d+)", "chain", "128")
    l_s5 = grab(r"s5 @L(\d+)", "s5", "256")
    return ["Model",
            f"State tracking (binding @L{l_bind})",
            f"Composed @L{l_bind}",
            "Composition gap",
            f"Chain d{depth} (thinking)",
            f"S5 @L{l_s5} (thinking)"]


def table_lines(header, idx, roster, floors, results_text) -> list[str]:
    cols = page_columns(header, idx)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for label, cells in roster + floors:
        is_floor = label.startswith(FLOOR_PREFIXES)
        out = [floor_display(label) if is_floor else display_name(label)]
        for key in KEY_ORDER:
            published = simplify_cell(cells[key])
            verify_value(published, cells[key], results_text)
            out.append(published)
        lines.append("| " + " | ".join(out) + " |")
    return lines


def footer_line(date: str, n_models: int) -> str:
    return (f"Last updated: {date} · {n_models} models · data, methodology, and "
            f"the add-a-model path: [{REPO_LABEL}]({REPO_URL}).")


def render_page(results_text: str) -> str:
    """The full page markdown — a pure function of results.md's text."""
    date = parse_generated_date(results_text)
    header, idx, roster, floors = parse_headline(results_text)
    lines = [TITLE, ""]
    lines += FRAMING + [""]
    lines += ["## Results (current roster)", ""]
    lines += [COLUMN_DECODE_LINE, ""]
    lines += table_lines(header, idx, roster, floors, results_text) + [""]
    lines += [MARKS_LINE, ""]
    if any("kimi" in label for label, _ in roster):  # roster-conditional note
        lines += [KIMI_NOTE, ""]
    lines += [THINKING_NOISE_LINE, ""]
    for floor_line in FLOOR_LINES:
        lines += [floor_line, ""]
    lines += ["## Figures", "", FIGURES_INTRO, ""]
    for i, (name, caption, alt) in enumerate(FIGURES, 1):
        lines.append(f"{i}. `{name}` — {caption}")
        lines.append(f"   *Alt text:* {alt}")
    lines += ["", "---", "", footer_line(date, len(roster)), ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Render docs/benchmark/results.md into the blog page markdown.")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="rendered results.md to read (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="also write the page here (e.g. docs/benchmark/page.md)")
    args = ap.parse_args(argv)
    with open(args.results, encoding="utf-8") as fh:
        results_text = fh.read()
    page = render_page(results_text)
    sys.stdout.write(page)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(page)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
