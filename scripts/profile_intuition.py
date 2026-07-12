"""Spearman check of the benchmark's profile axes against the owner's pinned
intuitive ranking (docs/benchmark/profiles-analysis.md is the written note;
re-run this script after a benchmark cycle to refresh its numbers).

Reads the same history + axis definitions as scripts/render_benchmark.py
(profile_values / PROFILE_AXES: binding @L16, composed @L16, gap inverted,
chain d128, s5 @L256, s5@128 ctok inverted). For each axis it computes the
Spearman rank correlation between the axis (oriented so higher = better) and
the pinned ranking, over the models measurable on that axis — censored (⊘)
and missing cells are dropped, never scored as zeros. Kimi's instant cells are
daggered (covert working, cap not enforced), so every correlation is reported
both with and without kimi. It then scores every 2-axis combination (mean of
the two normalized positions, models measurable on both) the same way.

No API calls; pure history read. Usage:
    python scripts/profile_intuition.py [--history results/benchmark/history.jsonl]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import render_benchmark as RB  # noqa: E402

# The owner's pinned intuitive ranking, best first (grok is unrepresented on the
# roster and skipped). Kept in sync by hand — it is a prior, not a measurement.
PINNED = [
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.8",
    "z-ai/glm-5.2",
    "anthropic/claude-sonnet-5",
    "moonshotai/kimi-k2.6",
    "google/gemini-3.5-flash",
    "qwen/qwen3.7-max",
    "deepseek/deepseek-v4-pro",
    "nvidia/nemotron-3-ultra-550b-a55b",
]
DAGGERED = "moonshotai/kimi-k2.6"  # † instant cells: reported with and without


def _avg_ranks(xs) -> list[float]:
    """Average ranks (1 = smallest), ties averaged."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys) -> float | None:
    """Spearman rho via Pearson on average ranks; None below 3 pairs."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = _avg_ranks(xs), _avg_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def _axis_pairs(values, label, invert, models):
    """(intuition_goodness, oriented_value) pairs over measurable models."""
    pairs = []
    for m in models:
        cell = values.get(m, {}).get(label)
        if cell and cell["status"] == "ok":
            pairs.append((-PINNED.index(m),
                          -cell["value"] if invert else cell["value"]))
    return pairs


def _rho_str(pairs) -> str:
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    return "n<3" if rho is None else f"{rho:+.2f} (n={len(pairs)})"


def _combo_pairs(values, la, lb, models):
    """(intuition_goodness, mean of the two normalized positions) pairs over
    models measurable on BOTH axes (norms already orient inverted axes)."""
    pairs = []
    for m in models:
        ca, cb = values[m][la], values[m][lb]
        if ca["status"] == "ok" and cb["status"] == "ok":
            pairs.append((-PINNED.index(m), (ca["norm"] + cb["norm"]) / 2))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history",
                    default=os.path.join(REPO, "results", "benchmark", "history.jsonl"))
    args = ap.parse_args(argv)
    records = RB.load_latest(args.history)
    values = RB.profile_normalized(RB.profile_values(records))
    roster = [m for m in PINNED if m in values]
    missing = sorted(set(values) - set(PINNED))
    if missing:
        print(f"WARNING: roster models missing from PINNED (excluded): {missing}")
    no_kimi = [m for m in roster if m != DAGGERED]

    print("Spearman rho vs the pinned intuitive ranking, per axis")
    print("(oriented so higher = better; censored/missing cells dropped)\n")
    print("| axis | rho (all measurable) | rho (without kimi †) |")
    print("|---|---|---|")
    for label, invert in RB.PROFILE_AXES:
        all_p = _axis_pairs(values, label, invert, roster)
        nk_p = _axis_pairs(values, label, invert, no_kimi)
        print(f"| {label} | {_rho_str(all_p)} | {_rho_str(nk_p)} |")

    print("\nTop 2-axis combinations (mean of normalized positions, "
          "models measurable on both):\n")
    combos = []
    for (la, _), (lb, _) in itertools.combinations(RB.PROFILE_AXES, 2):
        p_all = _combo_pairs(values, la, lb, roster)
        p_nk = _combo_pairs(values, la, lb, no_kimi)
        rho = spearman([p[0] for p in p_all], [p[1] for p in p_all])
        if rho is not None:
            combos.append((rho, la, lb, len(p_all), p_nk))
    combos.sort(reverse=True)
    print("| combination | rho (all measurable) | rho (without kimi †) |")
    print("|---|---|---|")
    for rho, la, lb, n, p_nk in combos[:5]:
        print(f"| {la} + {lb} | {rho:+.2f} (n={n}) | {_rho_str(p_nk)} |")


if __name__ == "__main__":
    main()
