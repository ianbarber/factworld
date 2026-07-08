"""Analysis for the v2 pilots (PR #10): prints per-pilot tables from the JSONL.

Usage: .venv-api/bin/python scripts/experiment_v2_pilot_analyze.py [1|2|3 ...]
Reads results/v2_pilots/pilot{1,2,3}_*.jsonl; no API calls.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(REPO, "results", "v2_pilots")

# OpenRouter live pricing checked 2026-07-08 ($/M prompt, $/M completion) — the
# registry's glm price is stale ($0.56/$1.76), so spend is computed from these.
LIVE = {
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "anthropic/claude-opus-4.8": (5.0, 25.0),
    "moonshotai/kimi-k2.6": (0.66, 3.41),
    "x-ai/grok-4.20": (1.25, 2.50),
    "z-ai/glm-5.2": (0.93, 3.00),
    "openai/gpt-5.4": (2.50, 15.00),
}


def load(name: str) -> list[dict]:
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def live_cost(rec: dict) -> float:
    pp, cp = LIVE[rec["model"]]
    u = rec["usage"]
    return u["prompt_tokens"] / 1e6 * pp + u["completion_tokens"] / 1e6 * cp


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def pilot1() -> None:
    recs = load("pilot1_chain_nowrap.jsonl")
    print("=== PILOT 1: chain_nowrap TRUE depth (n=15, effort=high, cap 16384) ===")
    total = 0.0
    by_model: dict[str, dict[int, dict]] = {}
    for r in recs:
        by_model.setdefault(r["model"], {})[r["length"]] = r
        total += live_cost(r)
    for model, cells in by_model.items():
        for depth, r in sorted(cells.items()):
            rel = r["metrics"]["relaxed"]
            lo, hi = wilson(rel, r["n"])
            ctoks = [e["ctok"] for e in r["examples"] if e["ctok"]]
            hops = Counter(e.get("hop") for e in r["examples"] if not e["relaxed"])
            fr = r["diagnostics"]["finish_reasons"]
            print(f"{model:<28} d{depth:<3} relaxed={rel:.2f} [{lo:.2f},{hi:.2f}] "
                  f"mean_ctok={sum(ctoks)/max(1,len(ctoks)):.0f} "
                  f"max_ctok={max(ctoks) if ctoks else 0} finish={fr} "
                  f"wrong-hop-fingerprint={dict(hops) or '-'}")
        # ctok-vs-depth slope + projection
        if 16 in cells and 32 in cells:
            m16 = [e["ctok"] for e in cells[16]["examples"] if e["ctok"]]
            m32 = [e["ctok"] for e in cells[32]["examples"] if e["ctok"]]
            a, b = sum(m16) / len(m16), sum(m32) / len(m32)
            slope = (b - a) / 16
            pp, cp = LIVE[model]
            for d in (64, 128):
                proj = b + slope * (d - 32)
                cost25 = 25 * proj / 1e6 * cp  # n=25 facet cell, completion only
                print(f"  {model} ctok slope={slope:.0f}/hop -> d{d} ~{proj:.0f} ctok/call "
                      f"(~${cost25:.2f}/cell completion at n=25)")
    print(f"PILOT 1 live spend: ${total:.2f}\n")


def pilot2() -> None:
    recs = load("pilot2_contract.jsonl")
    print("=== PILOT 2: capped-contract effort=none (composite_copy_v1@L16, n=50) ===")
    total = 0.0
    for r in recs:
        d, m = r["diagnostics"], r["metrics"]
        total += live_cost(r)
        lo, hi = wilson(m["relaxed"], r["n"])
        line = (f"{r['model']:<28} relaxed={m['relaxed']:.2f} [{lo:.2f},{hi:.2f}] "
                f"contains={m['contains']:.2f} contract={d['contract_rate']:.2f} "
                f"covert={d['covert_cot_rate']:.2f} leak={d['rtok_leak_rate']:.2f} "
                f"finish={d['finish_reasons']}")
        if r["escalated"]:
            f = r["escalation"]["first_attempt"]
            line += f" | ESC first@96: relaxed={f['relaxed']:.2f} len_rate={f['length_rate']:.2f}"
        print(line)
    print(f"PILOT 2 live spend: ${total:.2f}\n")


def pilot3() -> None:
    recs = load("pilot3_anthropic_budget.jsonl")
    print("=== PILOT 3: Anthropic thinking budget, s5_concrete@L128 (n=15) ===")
    total = 0.0
    for r in recs:
        m = r["metrics"]
        total += live_cost(r)
        lo, hi = wilson(m["relaxed"], r["n"])
        rtoks = [e["rtok"] or 0 for e in r["examples"]]
        ctoks = [e["ctok"] or 0 for e in r["examples"]]
        fr = r["diagnostics"]["finish_reasons"]
        print(f"{r['model']:<28} {r['condition']:<13} relaxed={m['relaxed']:.2f} "
              f"[{lo:.2f},{hi:.2f}] contains={m['contains']:.2f} "
              f"rtok mean={sum(rtoks)/len(rtoks):.0f} max={max(rtoks)} "
              f"ctok mean={sum(ctoks)/len(ctoks):.0f} finish={fr}")
    print(f"PILOT 3 live spend: ${total:.2f}\n")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    for w in which:
        {"1": pilot1, "2": pilot2, "3": pilot3}[w]()
