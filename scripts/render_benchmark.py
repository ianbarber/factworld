"""Render the recurring frontier-benchmark history into blog-ready tables, figures and HTML.

Reads ``results/benchmark/history.jsonl`` (one JSON record per cell, contract C3 — see
scripts/run_frontier_benchmark.py), keeps the LATEST record per
``(model, facet, task, length, {effort, leg, rendering})`` key, and writes into
``docs/benchmark/``:

  results.md    headline + diagnostics + full per-cell markdown tables
  results.csv   flat per-cell export (all metrics + diagnostics + usage)
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
import html
import json
import math
import os
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
    n = rec.get("n") or 0
    r = (rec.get("metrics") or {}).get("relaxed")
    if r is None or n <= 0:
        return (0.0, 1.0)
    return wilson_interval(round(r * n), n)


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
    return CHAIN_INVALID_MARK if chain_invalid(rec) else "—"


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


def headline_horizon(records, facet, model, exclude_renderings=("abstract_stated",)):
    """Max length with relaxed >= HORIZON_THRESHOLD, or None if no cell qualifies."""
    cells = [r for r in by_facet(records, facet)
             if r["model"] == model
             and _settings(r).get("rendering") not in exclude_renderings]
    good = [r["length"] for r in cells
            if (r["metrics"].get("relaxed") or 0.0) >= HORIZON_THRESHOLD]
    return max(good) if good else None


def headline_decomposition(records, model):
    """(binding_only, end_to_end, scaffolded) relaxed triple; None where a leg is missing."""
    cells = [r for r in by_facet(records, "decomposition") if r["model"] == model]
    out = {}
    for leg in LEG_ORDER:
        matches = [r for r in cells if _settings(r).get("leg") == leg]
        out[leg] = matches[0]["metrics"]["relaxed"] if matches else None
    return out


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
    models = models_of(records)
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
              "| Model | dose_response (relaxed) | composite_length (relaxed @ L512, high) | "
              "s5 horizon (max L, relaxed >= 0.8) | chain horizon (chain_nowrap, max depth, "
              "relaxed >= 0.8) | decomposition (bind / e2e / scaffold) |",
              "|---|---|---|---|---|---|"]
    for m in models:
        dr, dr_eff = headline_dose_response(records, m)
        cl = headline_composite_length(records, m)
        s5 = headline_horizon(records, "s5_concrete", m)
        ch = headline_horizon(records, "chain_nowrap", m)
        legs = headline_decomposition(records, m)
        dr_s = "—" if dr is None else f"{dr:.2f} @ {dr_eff}"
        triple = " / ".join(_fmt(legs[leg]) for leg in LEG_ORDER)
        lines.append(f"| {m} | {dr_s} | {_fmt(cl)} | {s5 if s5 is not None else '—'} | "
                     f"{ch if ch is not None else '—'} | {triple} |")
    lines += [
        "",
        "Chain horizons come from the `chain_nowrap` facet only. `chain_v1` builds a single "
        f"k={CHAIN_CYCLE_K} pointer cycle and measures depth only for depths < k "
        '(`factworld/tasks.py`: "Depths stay < k so the cycle never wraps"); `chain_depth` cells '
        f"at depth >= {CHAIN_CYCLE_K} wrapped the cycle (gold == start agent at depths 12/24/48; "
        "effective difficulty depth mod 6), measure the wrapped task rather than depth, and are "
        f"marked `{CHAIN_INVALID_MARK}` in the tables below and excluded from the chain figure.",
        "",
    ]

    lines += ["## Diagnostics per cell", "",
              "| Model | Facet | Task | Length | Arm | empty_rate | api_errors | "
              "reasoning_tokens | finish_reasons | note |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        d = r.get("diagnostics") or {}
        u = r.get("usage") or {}
        fr = d.get("finish_reasons") or {}
        fr_s = ", ".join(f"{k}:{v}" for k, v in sorted(fr.items())) or "—"
        lines.append(
            f"| {r['model']} | {r['facet']} | {r['task']} | {r.get('length', '—')} | "
            f"{arm_label(r)} | {_fmt(d.get('empty_rate'), 3)} | {d.get('api_errors', '—')} | "
            f"{u.get('reasoning_tokens', '—')} | {fr_s} | {cell_note(r)} |")
    lines.append("")

    lines += ["## Full per-cell results", "",
              "| Model | Facet | Task | Length | Arm | n | relaxed [95% CI] | exact | contains | "
              "last_n | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        mt = r.get("metrics") or {}
        lo, hi = _ci(r)
        lines.append(
            f"| {r['model']} | {r['facet']} | {r['task']} | {r.get('length', '—')} | "
            f"{arm_label(r)} | {r.get('n', '—')} | {_fmt(mt.get('relaxed'))} "
            f"[{lo:.2f}, {hi:.2f}] | {_fmt(mt.get('exact'))} | {_fmt(mt.get('contains'))} | "
            f"{_fmt(mt.get('last_n'))} | {cell_note(r)} |")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


CSV_FIELDS = [
    "run_id", "ts", "git_commit", "suite_version", "model", "served_models", "providers",
    "facet", "task", "length", "n", "effort", "leg", "rendering", "max_new_tokens",
    "stop_at", "format_prompt", "n_shot", "relaxed", "relaxed_ci_lo", "relaxed_ci_hi",
    "exact", "contains", "last_n", "empty_rate", "api_errors", "finish_reasons",
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
                "relaxed": mt.get("relaxed"),
                "relaxed_ci_lo": round(lo, 4), "relaxed_ci_hi": round(hi, 4),
                "exact": mt.get("exact"), "contains": mt.get("contains"),
                "last_n": mt.get("last_n"),
                "empty_rate": d.get("empty_rate"), "api_errors": d.get("api_errors"),
                "finish_reasons": json.dumps(d.get("finish_reasons") or {}, sort_keys=True),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": u.get("reasoning_tokens"),
                "cost_usd_est": u.get("cost_usd_est"), "elapsed_s": r.get("elapsed_s"),
                "note": CHAIN_INVALID_MARK if chain_invalid(r) else "",
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
    ys = [r["metrics"]["relaxed"] for r in cells]
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
    ax.axhline(HORIZON_THRESHOLD, color="#888888", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    xt = sorted({r["length"] for r in cells})
    ax.set_xticks(xt, [str(x) for x in xt])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relaxed accuracy")
    ax.set_title(title)
    _style_axes(ax)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    _caption(fig, cells)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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


FIGURES = [fig_dose_response, fig_composite_length, fig_s5_horizon,
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
    models = models_of(records)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    head_rows = []
    for m in models:
        dr, dr_eff = headline_dose_response(records, m)
        cl = headline_composite_length(records, m)
        s5 = headline_horizon(records, "s5_concrete", m)
        ch = headline_horizon(records, "chain_nowrap", m)
        legs = headline_decomposition(records, m)
        head_rows.append([
            m,
            "—" if dr is None else f"{dr:.2f} @ {dr_eff}",
            _fmt(cl),
            s5 if s5 is not None else "—",
            ch if ch is not None else "—",
            " / ".join(_fmt(legs[leg]) for leg in LEG_ORDER),
        ])

    cell_rows = []
    for r in records:
        mt = r.get("metrics") or {}
        d = r.get("diagnostics") or {}
        u = r.get("usage") or {}
        lo, hi = _ci(r)
        cell_rows.append([
            r["model"], r["facet"], r["task"], r.get("length", "—"), arm_label(r),
            r.get("n", "—"), _fmt(mt.get("relaxed")), f"[{lo:.2f}, {hi:.2f}]",
            _fmt(mt.get("exact")), _fmt(mt.get("contains")), _fmt(mt.get("last_n")),
            _fmt(d.get("empty_rate"), 3), d.get("api_errors", "—"),
            u.get("reasoning_tokens", "—"), cell_note(r),
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
{_html_table(["Model", "dose_response (relaxed)", "composite_length (relaxed @ L512, high)",
              "s5 horizon", "chain horizon (chain_nowrap)",
              "decomposition (bind / e2e / scaffold)"], head_rows)}
<h2>Figures</h2>
{figures_html}
<h2>All cells</h2>
<p class="small">Click a column header to sort.</p>
{_html_table(["Model", "Facet", "Task", "Length", "Arm", "n", "relaxed", "95% CI",
              "exact", "contains", "last_n", "empty_rate", "api_errors",
              "reasoning_tokens", "note"], cell_rows, sortable=True)}
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
