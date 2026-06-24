"""Merge multiple FactWorld grid JSON outputs into one markdown table.

Use this to mix OpenRouter API results (from ``eval_openrouter_grid.py``) with
local or HuggingFace model results (from ``eval_model.py --json_out``).

Examples:
    python scripts/eval_model.py composite_copy_v1 --backend local \\
        --arch gdn_hybrid --d_model 320 --steps 8000 --n 50 \\
        --json_out results/local-gdn.json

    python scripts/merge_grid_results.py \\
        docs/openrouter-results.json results/local-gdn.json \\
        --out docs/combined-results.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _model_label(model: str) -> str:
    return model.split("/")[-1]


def main():
    ap = argparse.ArgumentParser(
        description="Merge FactWorld grid JSON outputs into a markdown table."
    )
    ap.add_argument("inputs", nargs="+", help="Grid JSON files to merge.")
    ap.add_argument("--out", default="docs/combined-results.md",
                    help="Markdown output path.")
    ap.add_argument("--title", default="FactWorld Combined Model Grid",
                    help="Table title.")
    a = ap.parse_args()

    all_rows: list[dict] = []
    for path in a.inputs:
        with open(path) as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise SystemExit(f"{path}: expected a JSON list of rows")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows to merge")

    tasks = sorted({r["task"] for r in all_rows})
    model_order = []
    seen = set()
    for r in all_rows:
        m = r["model"]
        if m not in seen:
            seen.add(m)
            model_order.append(m)

    pivot: dict[str, dict[str, dict[str, list[float]]]] = {
        m: {t: {"exact": [], "relaxed": []} for t in tasks} for m in model_order
    }
    for r in all_rows:
        pivot[r["model"]][r["task"]]["exact"].append(r["accuracy_exact"])
        pivot[r["model"]][r["task"]]["relaxed"].append(r["accuracy_relaxed"])

    lines = [
        f"# {a.title}",
        "",
        f"Evaluated at {datetime.now(timezone.utc).isoformat()}.",
        f"n varies per row; see source JSONs. Position-strict exact match.",
        "",
        "## Exact-match results",
        "",
        "| model | " + " | ".join(tasks) + " |",
        "| " + " | ".join(["---"] * (len(tasks) + 1)) + " |",
    ]
    for model in model_order:
        accs = []
        for t in tasks:
            vals = pivot[model][t]["exact"]
            accs.append(f"{sum(vals) / len(vals):.3f}" if vals else "N/A")
        lines.append(f"| {_model_label(model)} | " + " | ".join(accs) + " |")

    lines += [
        "",
        "## Relaxed results (whitespace / period invariant)",
        "",
        "| model | " + " | ".join(tasks) + " |",
        "| " + " | ".join(["---"] * (len(tasks) + 1)) + " |",
    ]
    for model in model_order:
        accs = []
        for t in tasks:
            vals = pivot[model][t]["relaxed"]
            accs.append(f"{sum(vals) / len(vals):.3f}" if vals else "N/A")
        lines.append(f"| {_model_label(model)} | " + " | ".join(accs) + " |")

    lines += [
        "",
        "## Sources",
        "",
    ]
    for path in a.inputs:
        lines.append(f"- `{path}`")
    lines.append("")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
