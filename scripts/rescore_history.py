#!/usr/bin/env python
"""Rescore results/benchmark/history.jsonl from stored preds after the markdown-strip
scorer fix (F1).

The tokens-path scorers (chain / s5 / sanity / composite cells; ``runner.evaluate_task``
and ``s5_concrete.score`` via ``Renderer.normalize`` / ``tasks.content_tokens``) did not
strip markdown emphasis, so a correct answer decorated as ``**g22**`` scored 0. The fix
lives in ``Renderer.normalize`` (edge-strip of ``*_```). This script deterministically
recomputes every affected record's per-example ``relaxed`` (and the cell metrics) from
the STORED gold/pred pairs — no API calls — and rewrites the history file atomically,
in place (git preserves the pre-fix file, so the in-place rewrite is the
provenance-correct move).

Safety rails:
  * For every record we first REPRODUCE the stored per-example ``relaxed`` with the
    pre-fix scorer. A record whose stored scores the old scorer cannot reproduce was
    scored by some other (legacy) rule and is left byte-identical, and reported.
  * zero_budget cells are immune by construction (the contract extractor already strips
    markdown off the answer span); they are verified, never modified. A zero_budget cell
    that WOULD change is a loud error.
  * Changed records get a top-level ``"rescored": "markdown-strip 2026-07-08"`` marker;
    everything else (usage, diagnostics, ts, per-example ctok/rtok/finish) is kept
    intact — in particular ``ts`` is NOT bumped, so renderer dedup is not perturbed.
  * Unchanged records are written back byte-identical (the original line verbatim).

Usage:
    python scripts/rescore_history.py [--history results/benchmark/history.jsonl] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.render import Renderer  # noqa: E402
from factworld.tasks import (  # noqa: E402
    score_contains,
    score_exact,
    score_last_n,
    score_relaxed,
)

RESCORED_TAG = "markdown-strip 2026-07-08"


# --- the PRE-FIX normalizer, replicated verbatim for the reproduction check ----------
def _normalize_old(text: str) -> str:
    text = text.strip()
    text = re.sub(r"([a-zA-Z0-9]+)'s\b", r"\1 's", text)
    text = re.sub(r"(?<=\S)([.,?!])", r" \1", text)
    return text


def _normalize_new(text: str) -> str:
    return Renderer.normalize(text)


def _content_tokens(text: str, normalize) -> list[str]:
    return [t for t in normalize(text).split() if t != "."]


# --- per-family scoring, parameterized by the normalizer ------------------------------
def _score_tokens(pred: str, gold: str, normalize) -> dict:
    """runner.evaluate_task path: normalize both sides, all four scorers."""
    p, g = normalize(pred), normalize(gold)
    return {"relaxed": score_relaxed(p, g), "exact": score_exact(p, g),
            "contains": score_contains(p, g), "last_n": score_last_n(p, g)}


def _score_s5(pred: str, gold: str, normalize) -> dict:
    """s5_concrete.score path: first content token / membership."""
    ct = _content_tokens(pred, normalize)
    first = ct[0] if ct else ""
    return {"relaxed": int(first == gold), "contains": int(gold in ct)}


def _score_binding_prefix(pred: str, gold: str, normalize) -> dict:
    """zero_budget binding_only leg: the span must COMMIT to the holder as its first
    content token (prefix, not membership)."""
    holder = gold.strip().rstrip(".")
    return {"relaxed": int(_content_tokens(pred, normalize)[:1] == [holder])}


def _family(rec: dict) -> str:
    if rec.get("task") == "s5":
        return "s5"                          # s5_concrete / floor facets
    if rec.get("settings", {}).get("leg") == "binding_only":
        return "binding_prefix"              # zero_budget / decomposition binding leg
    return "tokens"                          # chain_* / sanity / composite / zero_budget span


_SCORERS = {"tokens": _score_tokens, "s5": _score_s5, "binding_prefix": _score_binding_prefix}


def _cell_id(rec: dict) -> str:
    leg = rec.get("settings", {}).get("leg")
    return (f"{rec.get('model')} {rec.get('facet')}/{rec.get('task')} L={rec.get('length')}"
            + (f" leg={leg}" if leg else "") + f" [{rec.get('run_id')}]")


def rescore_record(rec: dict) -> tuple[dict | None, str | None]:
    """Return (updated_record_or_None, note). None means the record is unchanged."""
    examples = rec.get("examples")
    if not examples:
        return None, "no per-example data — skipped"
    fam = _family(rec)
    score = _SCORERS[fam]

    # idempotency: an already-rescored record must verify against the FIXED scorer.
    if rec.get("rescored") == RESCORED_TAG:
        bad = sum(1 for ex in examples
                  if score(ex["pred"], ex["gold"], _normalize_new)["relaxed"] != ex["relaxed"])
        if bad:
            raise AssertionError(f"already-rescored record does not verify: {_cell_id(rec)}")
        return None, None

    # 1. reproduction check: the OLD scorer must reproduce every stored relaxed value.
    mismatch = sum(1 for ex in examples
                   if score(ex["pred"], ex["gold"], _normalize_old)["relaxed"] != ex["relaxed"])
    if mismatch:
        return None, (f"stored scores NOT reproducible by the pre-fix {fam} scorer "
                      f"({mismatch}/{len(examples)} examples differ) — legacy scoring rule, "
                      f"left untouched")

    # 2. rescore with the fixed scorer.
    new_scores = [score(ex["pred"], ex["gold"], _normalize_new) for ex in examples]
    changed = [i for i, (ex, ns) in enumerate(zip(examples, new_scores))
               if ns["relaxed"] != ex["relaxed"]]

    # zero_budget is immune by construction (extractor strips markdown): verify, never touch.
    if rec.get("facet") == "zero_budget":
        if changed:
            raise AssertionError(
                f"zero_budget cell would change under the fix — investigate: {_cell_id(rec)} "
                f"examples {changed}")
        return None, None

    if not changed:
        return None, None

    n = len(examples)
    old_relaxed = rec["metrics"]["relaxed"]
    for i in changed:
        examples[i]["relaxed"] = new_scores[i]["relaxed"]
    new_metrics = dict(rec["metrics"])
    new_metrics["relaxed"] = sum(ns["relaxed"] for ns in new_scores) / n
    # diagnostic metrics: recompute only the ones this record actually carries.
    for name in ("exact", "contains", "last_n"):
        if rec["metrics"].get(name) is not None and all(name in ns for ns in new_scores):
            new_metrics[name] = sum(ns[name] for ns in new_scores) / n
    rec["metrics"] = new_metrics
    rec["rescored"] = RESCORED_TAG
    note = (f"relaxed {old_relaxed:.2f} -> {new_metrics['relaxed']:.2f} "
            f"({len(changed)} example(s): "
            f"{sum(1 for i in changed if new_scores[i]['relaxed'])} up, "
            f"{sum(1 for i in changed if not new_scores[i]['relaxed'])} down)")
    return rec, note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", default="results/benchmark/history.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not rewrite")
    args = ap.parse_args()

    with open(args.history, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    out_lines: list[str] = []
    n_changed = 0
    skipped: list[str] = []
    print(f"rescoring {len(lines)} records in {args.history} (fix: {RESCORED_TAG})\n")
    for line in lines:
        rec = json.loads(line)
        updated, note = rescore_record(rec)
        if updated is None:
            out_lines.append(line)                     # byte-identical passthrough
            if note is not None:
                skipped.append(f"  SKIP  {_cell_id(rec)}: {note}")
        else:
            out_lines.append(json.dumps(updated, ensure_ascii=True))
            n_changed += 1
            print(f"  CHANGED {_cell_id(rec)}: {note}")

    if skipped:
        print(f"\n{len(skipped)} record(s) left untouched (not reproducible / no examples):")
        for s in skipped:
            print(s)
    print(f"\n{n_changed} record(s) rescored, {len(lines) - n_changed} unchanged "
          f"(zero_budget verified immune).")

    if args.dry_run:
        print("dry run — history file not modified.")
        return
    d = os.path.dirname(os.path.abspath(args.history)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".history_rescore_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        os.replace(tmp, args.history)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(f"wrote {args.history} atomically.")


if __name__ == "__main__":
    main()
