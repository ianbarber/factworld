"""V3 probe P3 — fixed-k chain control (composition-as-axis square completion).

The existing chain_nowrap staircase confounds depth with breadth (k = 2d+1): the
d16 cells ran on a 33-cycle, the d128 cells on a 257-cycle. This probe runs chain
d16 AT k_fixed=257 (16 hops over a 257-cycle: the d128 cells' breadth at the d16
cells' depth), completing the 2x2 square with the existing history cells:

    d16@k33   (glm 0.96 / qwen 1.00 / flash 1.00)   d16@k257   <- THIS PROBE
    d128@k33  (invalid: wrap gate)                  d128@k257  (glm 0.36 / qwen 0.96 / flash 0.88)

Pre-registered signature (Checkpoint 1): if d16@k257 passes while a model's
d128@k257 fails, depth survives as an axis at fixed breadth; if d16@k257 FAILS,
the old chain column was breadth all along.

Protocol matches the chain_nowrap facet exactly (effort=high, max_new_tokens
16384, tokens answer mode, no contract): n=15, models glm / qwen / gemini-flash.
Records are C3-conformant (execute_cell) + a per-record ``floors`` context and
the pilot-1 shortcut fingerprint (hop count per pred). Output goes to
results/v3_probes/p3_chain_fixedk.jsonl — never history.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from factworld import tasks as TK
from factworld.benchmark import _settings, spec_for_cell
from run_frontier_benchmark import (
    _git_commit,
    append_record,
    build_backend,
    execute_cell,
)

PROBE_MODELS = (
    "z-ai/glm-5.2",
    "qwen/qwen3.7-max",
    "google/gemini-3.5-flash",
)
DEPTH = 16
K_FIXED = 257
N = 15
MAX_NEW_TOKENS = 16384  # identical to the chain_nowrap facet budget
OUT = os.path.join(REPO, "results", "v3_probes", "p3_chain_fixedk.jsonl")

_FACT_RE = re.compile(r"(g\d+)'s a0 is (g\d+)\.")


def probe_items(n: int) -> list[dict]:
    """Regenerate the exact probe items with their full pointer walks (pre-spend
    validity gate + shortcut fingerprint): asserts the map is a single 257-cycle
    presented in full, gold == nxt^16(start) recomputed from the RENDERED prompt,
    and gold != start."""
    spec = spec_for_cell("chain_v1", DEPTH, k_fixed=K_FIXED)
    assert spec.k == K_FIXED
    items = []
    for ex in TK.generate(spec, "test", n=n, length=DEPTH):
        nxt = dict(_FACT_RE.findall(ex.prompt))
        assert len(nxt) == K_FIXED, f"{len(nxt)} facts != k_fixed {K_FIXED}"
        start = ex.meta["start"]
        walk, cur = [start], start
        for _ in range(len(nxt)):
            cur = nxt[cur]
            walk.append(cur)
        assert len(set(walk[:-1])) == K_FIXED, "pointer map is not a single cycle"
        assert walk[K_FIXED] == start, "cycle does not close"
        gold = ex.answer.rstrip(".").strip()
        assert walk[DEPTH] == gold, "gold != nxt^depth(start) from the rendered prompt"
        assert gold != start, "gold == start (identity leak)"
        items.append({"start": start, "walk": walk})
    return items


def hop_of(pred: str, walk: list[str]) -> int | None:
    """The hop count m with walk[m] == pred's first token (None if off-cycle)."""
    tok = pred.strip().rstrip(".").split()
    tok = tok[0] if tok else ""
    return walk.index(tok) if tok in walk else None


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    walks = probe_items(N)  # validity gate BEFORE any spend
    print(f"validity gate OK: {N} items, single {K_FIXED}-cycle rendered in full, "
          f"gold recomputes as hop {DEPTH}, gold != start")

    done: set[tuple] = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((rec["model"], rec["length"], rec["settings"].get("k_fixed")))

    run_id = f"v3probe_p3_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    git_commit = _git_commit()
    for model in PROBE_MODELS:
        if (model, DEPTH, K_FIXED) in done:
            print(f"SKIP (resume): {model} d{DEPTH}@k{K_FIXED}")
            continue
        cell = {
            "facet": "chain_fixedk_probe", "task": "chain_v1", "length": DEPTH,
            "n": N,
            "settings": _settings("high", max_new_tokens=MAX_NEW_TOKENS,
                                  k_fixed=K_FIXED),
        }
        backend = build_backend(model, cell, api_key,
                                "https://openrouter.ai/api/v1", 8)
        try:
            rec = execute_cell(backend, model, cell, n=N, run_id=run_id,
                               git_commit=git_commit)
        except Exception as exc:  # noqa: BLE001 — keep the other models running
            print(f"FAILED {model} d{DEPTH}@k{K_FIXED}: {exc}")
            continue
        for ex, item in zip(rec["examples"], walks):
            ex["hop"] = hop_of(ex["pred"], item["walk"])
        # floor context: an off-task guess is 1-of-k over the presented cycle
        rec["floors"] = {"uniform_cycle": round(1 / K_FIXED, 5)}
        append_record(OUT, rec)
        u, d = rec["usage"], rec["diagnostics"]
        ctoks = [e["ctok"] for e in rec["examples"] if e["ctok"]]
        rtoks = [e["rtok"] for e in rec["examples"] if e["rtok"]]
        print(f"{model} d{DEPTH}@k{K_FIXED}: relaxed={rec['metrics']['relaxed']:.3f} "
              f"empty={d['empty_rate']:.2f} err={d['api_errors']} "
              f"finish={d['finish_reasons']} "
              f"ctok_med={statistics.median(ctoks) if ctoks else 0:.0f} "
              f"rtok_med={statistics.median(rtoks) if rtoks else 0:.0f} "
              f"${u['cost_usd_est']:.2f} [{rec['elapsed_s']:.0f}s]", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
