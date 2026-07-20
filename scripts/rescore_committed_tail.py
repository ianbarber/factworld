#!/usr/bin/env python
"""Rescore history.jsonl from stored preds after the committed-answer extraction fix.

Reasoning endpoints that spill working into the visible completion state the answer on
the FINAL line; prefix scoring read the working instead (sonnet xhigh s5_chain: match
0.56 vs contains 0.92). ``tasks.committed_answer`` now extracts the single-token final
line before scoring. This script deterministically recomputes every affected record's
per-example ``relaxed`` (and cell metric) from the STORED gold/pred pairs — no API
calls — and rewrites the history atomically in place (git preserves the pre-fix file).

Safety rails (mirroring scripts/rescore_history.py):
  * Scope: non-contract task cells only (settings.contract falsy, facet not
    s5_concrete/zero_budget) — the paths that score via ``evaluate_task``'s pipeline.
  * For every candidate record the OLD pipeline (normalize, no extraction) must
    REPRODUCE the stored per-example ``relaxed``; otherwise the record is left
    byte-identical and reported.
  * Changed records get ``"rescored": "committed-tail 2026-07-20"`` appended to any
    existing marker; ts/usage/diagnostics are untouched (renderer dedup unperturbed).
  * Unchanged records are written back byte-identical.

Usage: python scripts/rescore_committed_tail.py [--history PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.render import Renderer  # noqa: E402
from factworld.benchmark import REASONING_EFFORTS  # noqa: E402
from factworld.tasks import committed_answer, score_relaxed  # noqa: E402

SKIP_FACETS = {"s5_concrete", "zero_budget", "gap_stability"}
MARKER = "committed-tail 2026-07-20"


def _old(pred: str, gold: str) -> int:
    return score_relaxed(Renderer.normalize(pred.split("<eos>")[0]),
                         Renderer.normalize(gold))


def _new(pred: str, gold: str) -> int:
    return score_relaxed(Renderer.normalize(committed_answer(pred.split("<eos>")[0])),
                         Renderer.normalize(gold))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="results/benchmark/history.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out_lines, changed, skipped_unreproduced = [], [], []
    for line in open(a.history, encoding="utf-8"):
        rec = json.loads(line)
        settings = rec.get("settings") or {}
        examples = rec.get("examples") or []
        if (rec.get("facet") in SKIP_FACETS or settings.get("contract")
                or settings.get("effort") not in REASONING_EFFORTS
                or not examples or "gold" not in examples[0]):
            # Reasoning arms only: crediting a spilled trace's final line on an
            # instant arm would break the in-weights semantics.
            out_lines.append(line)
            continue
        if any(_old(ex.get("pred") or "", ex["gold"]) != ex.get("relaxed")
               for ex in examples if ex.get("finish") != "length"):
            skipped_unreproduced.append((rec.get("run_id"), rec.get("model"),
                                         rec.get("task"), rec.get("length")))
            out_lines.append(line)
            continue
        new_scores = [0 if ex.get("finish") == "length"
                      else _new(ex.get("pred") or "", ex["gold"]) for ex in examples]
        if new_scores == [ex.get("relaxed") for ex in examples]:
            out_lines.append(line)
            continue
        old_metric = rec["metrics"].get("relaxed")
        for ex, s in zip(examples, new_scores):
            ex["relaxed"] = s
        rec["metrics"]["relaxed"] = round(sum(new_scores) / len(new_scores), 4)
        prev = rec.get("rescored")
        rec["rescored"] = f"{prev}; {MARKER}" if prev else MARKER
        changed.append((rec.get("model"), rec.get("task"), rec.get("length"),
                        settings.get("effort"), old_metric, rec["metrics"]["relaxed"]))
        out_lines.append(json.dumps(rec) + "\n")

    for m, t, L, eff, o, n in changed:
        print(f"  {m} {t} L{L} [{eff}]: {o} -> {n}")
    for key in skipped_unreproduced:
        print(f"  !! unreproduced (left untouched): {key}")
    print(f"{len(changed)} records rescored, {len(skipped_unreproduced)} unreproduced.")
    if a.dry_run or not changed:
        return 0
    d = os.path.dirname(os.path.abspath(a.history))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.writelines(out_lines)
    os.replace(tmp, a.history)
    print(f"rewrote {a.history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
