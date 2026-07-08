"""V2 pilot 1 — chain_nowrap TRUE-depth check (PR #10 amendment pilots).

Runs the no-wrap chain staircase (chain_v1.scaled(k=depth+2)) at depths 16 and 32,
n=15, effort=high, max_new_tokens=16384, on four models. Questions:
  1. Does TRUE depth separate glm / kimi / gpt-5.4 / opus (v1's "glm cliff at 32"
     was measured on the invalid wrapped chain)?
  2. Shortcut fingerprint: for each wrong pred, which hop-count m (pred ==
     nxt^m(start)) did the model actually compute? m=0 identity, m=1 recency, etc.
  3. ctok-vs-depth slope for cost projection to d64/128.

Pre-spend validity gate: asserts gold != start on every generated item.
Appends C3-style records to results/v2_pilots/pilot1_chain_nowrap.jsonl (resume by
(model, length): a cell already in the file is skipped). Records carry per-call
{ctok, rtok, finish} in examples.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from factworld import tasks as TK
from factworld.benchmark import _settings
from run_frontier_benchmark import (
    _git_commit,
    append_record,
    build_backend,
    execute_cell,
)

PILOT_MODELS = (
    # kimi last: its d32 traces run past 10 minutes per call, so it must not
    # serialize ahead of the fast models (2026-07-08 relaunch order).
    "z-ai/glm-5.2",
    "openai/gpt-5.4",
    "anthropic/claude-opus-4.8",
    "moonshotai/kimi-k2.6",
)
DEPTHS = (16, 32)
N = 15
MAX_NEW_TOKENS = 16384
OUT = os.path.join(REPO, "results", "v2_pilots", "pilot1_chain_nowrap.jsonl")

_FACT_RE = re.compile(r"(g\d+)'s a0 is (g\d+)\.")


def shortcut_fingerprint(depth: int, n: int) -> list[dict]:
    """Rebuild each item's pointer map so preds can be mapped to a hop count."""
    spec = TK.CANONICAL["chain_v1"].scaled(k=depth + 2)
    items = []
    for ex in TK.generate(spec, "test", n=n, length=depth):
        nxt = dict(_FACT_RE.findall(ex.prompt))
        start = ex.meta["start"]
        walk, cur = [start], start
        for _ in range(len(nxt)):
            cur = nxt[cur]
            walk.append(cur)
        assert walk[depth] == ex.answer.rstrip(".").strip(), "oracle mismatch"
        items.append({"start": start, "walk": walk})
    return items


def hop_of(pred: str, walk: list[str]) -> int | None:
    """The hop count m with walk[m] == pred's first token (None if off-cycle)."""
    tok = pred.strip().rstrip(".").split()
    tok = tok[0] if tok else ""
    # first occurrence: walk repeats after k, and we only walked k hops
    return walk.index(tok) if tok in walk else None


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    # validity gate BEFORE any spend: no item may have gold == start
    for depth in DEPTHS:
        spec = TK.CANONICAL["chain_v1"].scaled(k=depth + 2)
        for ex in TK.generate(spec, "test", n=N, length=depth):
            assert ex.answer.rstrip(".").strip() != ex.meta["start"], (
                f"gold==start at depth {depth}: {ex.meta}")
    print(f"validity gate OK: gold != start for all {N} items at depths {DEPTHS}")

    done: set[tuple] = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((rec["model"], rec["length"]))

    run_id = f"v2pilot1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    git_commit = _git_commit()
    for model in PILOT_MODELS:
        for depth in DEPTHS:
            if (model, depth) in done:
                print(f"SKIP (resume): {model} d{depth}")
                continue
            cell = {
                "facet": "chain_nowrap", "task": "chain_v1", "length": depth,
                "n": N,
                "settings": _settings("high", max_new_tokens=MAX_NEW_TOKENS),
            }
            backend = build_backend(model, cell, api_key,
                                    "https://openrouter.ai/api/v1", 8)
            try:
                rec = execute_cell(backend, model, cell, n=N, run_id=run_id,
                                   git_commit=git_commit)
            except Exception as exc:  # noqa: BLE001 — keep the other cells running
                print(f"FAILED {model} d{depth}: {exc}")
                continue
            # fingerprint: hop count per pred against the item's regenerated walk
            walks = shortcut_fingerprint(depth, N)
            for ex, item in zip(rec["examples"], walks):
                ex["hop"] = hop_of(ex["pred"], item["walk"])
            append_record(OUT, rec)
            u, d = rec["usage"], rec["diagnostics"]
            ctoks = [e["ctok"] for e in rec["examples"] if e["ctok"]]
            print(f"{model} d{depth}: relaxed={rec['metrics']['relaxed']:.3f} "
                  f"empty={d['empty_rate']:.2f} err={d['api_errors']} "
                  f"finish={d['finish_reasons']} "
                  f"mean_ctok={sum(ctoks)/max(1,len(ctoks)):.0f} "
                  f"${u['cost_usd_est']:.2f} [{rec['elapsed_s']:.0f}s]", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
