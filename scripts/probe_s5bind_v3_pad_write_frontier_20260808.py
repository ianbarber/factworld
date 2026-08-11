"""CAN A SHIPPED MODEL WRITE THE BOUNDED PAD? The pad-write cell, asked of an API model.

WHY. The composed cell's only live axis is the PAD WRITE — the answer is generated from the
model's own pad, so a per-item perfect pad makes that context byte-identical to the gold-pad one
and the answer registers the pad and nothing else. Everything measured on that axis so far is a
from-scratch model on one architecture, and a floored reading there has two explanations that the
local arm alone cannot separate: the architecture cannot do it, or the CONSTRUCT cannot be done.
This asks a competent shipped model the same question, on the same items, scored the same way.

WHAT IS ASKED. The item's own prompt, plus an instruction to emit one line per step holding the
pad block that step writes. The block is the registered format's:

    swap step   the pointers of the two agents whose pointers CHANGED, after the swap: the agent
                the step names first, then the other one. (Which is ``before2``'s pair — the
                pre-values of the two written cells — printed in the order ``moved2`` prints it.)
    give step   under ``before2`` the agent the object belonged to BEFORE the step; under
                ``moved2`` that value and, first, the agent it belongs to after.

WHAT IS SCORED. Per pad token, split by ``"cell|source"`` exactly as the local read is, with
``swap_p0|cross`` — the one token whose write is two dependent reads — reported apart. The
comparison is against the FREE-RUNNING depth-<=1 floor: the model writes its own pad and reads
its own pad back, so the gold-pad address space does not exist for it and the event lines and the
header do (``validity.s5_bind_v3_pad_fixed_reads``, spaces without ``pad``).

    .venv-api/bin/python scripts/probe_s5bind_v3_pad_write_frontier_20260808.py \\
        --spec s5_bind_local_v3 --lengths 16,32 --n 20 --effort medium \\
        --out results/20260808_pad_write_frontier.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import tasks as TK                                          # noqa: E402
from factworld import validity as V                                        # noqa: E402
from factworld.composition import SWAP, read as _read                      # noqa: E402

BASE_URL = "https://steed.tailc4bb6.ts.net/v1"
TOK_PER_S = 14.0                 # steed's measured LONG-output decode rate, the slow one
MODEL = "deepseek-v4-flash"
RE_LINE = re.compile(r"^\s*s(\d+)\s*[:.\)]\s*(.+?)\s*$", re.M)
RE_TOK = re.compile(r"\bg\d+\b")

INSTRUCTION = {
    "before2": (
        "After the facts above, a numbered list of steps s0 onward changes them.\n"
        "Work through the steps in order and write ONE line per step, in order:\n"
        "  for a SWAP step:  s<i>: <a> <b>   where <a> is the pointer of the agent the step\n"
        "        names FIRST and <b> is the pointer of the OTHER agent whose pointer the swap\n"
        "        changed, both AFTER the swap.\n"
        "  for a GIVE step:  s<i>: <a>       where <a> is the agent the object belonged to\n"
        "        BEFORE the step.\n"
        "Every value is an agent name. Output only the lines, nothing else."),
    "moved2": (
        "After the facts above, a numbered list of steps s0 onward changes them.\n"
        "Work through the steps in order and write ONE line per step, in order:\n"
        "  for a SWAP step:  s<i>: <a> <b>   where <a> is the pointer of the agent the step\n"
        "        names FIRST and <b> is the pointer of the OTHER agent whose pointer the swap\n"
        "        changed, both AFTER the swap.\n"
        "  for a GIVE step:  s<i>: <a> <b>   where <a> is the agent the object belongs to AFTER\n"
        "        the step and <b> is the agent it belonged to BEFORE it.\n"
        "Every value is an agent name. Output only the lines, nothing else."),
}


def build_prompt(ex, fmt):
    body = ex.prompt.rsplit(".", 1)[0] if False else ex.prompt
    # drop the trailing question: this cell asks for the pad, not the answer
    body = re.sub(r"\s*(which agent does .*?\?)\s*$", "", body, flags=re.I | re.S)
    return f"{body}\n\n{INSTRUCTION[fmt]}"


def parse_blocks(text, n_events):
    """``{step: [tokens]}`` from the reply, last line per step wins."""
    out: dict = {}
    for mt in RE_LINE.finditer(text or ""):
        i = int(mt.group(1))
        if 0 <= i < n_events:
            out[i] = RE_TOK.findall(mt.group(2))
    return out


def score_one(rec, gold, blocks, fmt):
    hit: Counter = Counter()
    tot: Counter = Counter()
    empty = 0
    for i, (kind, _t, _r, src) in enumerate(rec["events"]):
        names = V.s5_bind_v3_pad_cells(kind, fmt)
        source = V.s5_bind_v3_pad_event_source(kind, src)
        got = blocks.get(i, [])
        if not got:
            empty += 1
        for p, cell in enumerate(names):
            key = f"{cell}|{source}"
            tot[key] += 1
            if p < len(got) and got[p] == gold[i][p]:
                hit[key] += 1
    return hit, tot, empty


def ask(client, prompt, effort, max_tokens, stream=True):
    """One call, STREAMED.

    Two transport walls sit in front of a long generation on this endpoint and both drop the SLOW
    items, which are the hard ones, so a cell that loses them is biased toward the model:
      the CLIENT timeout, sized from max_tokens at the measured long-output rate by the caller;
      the tailscale-serve PROXY, which returns 502 or drops the connection when a request produces
      nothing for long enough. At ~15 tok/s a 13,000-token reply is 15 minutes of silence on a
      non-streaming call, and the L = 48 cell lost half its items to exactly that. Streaming keeps
      bytes moving, so the proxy sees a live response throughout.
    """
    for attempt in range(3):
        try:
            if not stream:
                r = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, seed=0, max_tokens=max_tokens,
                    extra_body={"reasoning_effort": effort})
                ch = (r.choices or [None])[0]
                return ("" if ch is None else (ch.message.content or ""),
                        None if ch is None else ch.finish_reason)
            parts, fin = [], None
            with client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, seed=0, max_tokens=max_tokens, stream=True,
                    extra_body={"reasoning_effort": effort}) as chunks:
                for ch in chunks:
                    for c in (ch.choices or []):
                        if c.delta is not None and c.delta.content:
                            parts.append(c.delta.content)
                        if c.finish_reason:
                            fin = c.finish_reason
            return "".join(parts), fin
        except Exception as exc:                                   # noqa: BLE001
            if attempt == 2:
                return "", f"error:{type(exc).__name__}:{exc}"
            time.sleep(8 * (attempt + 1))
    return "", "error"


def main():
    from openai import OpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="s5_bind_local_v3")
    ap.add_argument("--lengths", default="16,32")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--fmt", default="before2")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="per-request seconds; 0 sizes it from max_tokens at TOK_PER_S")
    ap.add_argument("--no_stream", action="store_true",
                    help="non-streaming call; the tailscale proxy drops these on long generations")
    ap.add_argument("--max_missing", type=float, default=0.02,
                    help="a cell whose blocks are missing above this rate is VOID, not scored")
    ap.add_argument("--floor_n", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    timeout = a.timeout or (a.max_tokens / TOK_PER_S + 120.0)
    client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("STEED_API_KEY", "none"),
                    timeout=timeout, max_retries=0)
    spec = TK.spec_for(a.spec)
    k, m = spec.k, spec.n_objects_active
    res = {"model": MODEL, "spec": a.spec, "fmt": a.fmt, "effort": a.effort,
           "k": k, "m": m, "n": a.n, "cells": {},
           "generated": datetime.now(timezone.utc).isoformat()}
    for L in [int(z) for z in a.lengths.split(",")]:
        exs = TK.generate(spec, "test", n=a.n, length=L)
        items = []
        for e in exs:
            rec = _read(e.prompt)
            g = None if rec is None else V.s5_bind_v3_pad_gold(rec, a.fmt)
            if g is not None:
                items.append((e, rec, g))
        prompts = [build_prompt(e, a.fmt) for e, _r, _g in items]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            replies = list(pool.map(
                lambda p: ask(client, p, a.effort, a.max_tokens, not a.no_stream), prompts))
        hit, tot = Counter(), Counter()
        empty_blocks = n_empty = 0
        fins: Counter = Counter()
        for (e, rec, g), (txt, fin) in zip(items, replies):
            fins[str(fin)] += 1
            blocks = parse_blocks(txt, len(rec["events"]))
            if not blocks:
                n_empty += 1
            h, t, eb = score_one(rec, g, blocks, a.fmt)
            hit.update(h)
            tot.update(t)
            empty_blocks += eb
        # the FREE-RUNNING floor: the model writes and reads its own pad, so the gold-pad address
        # space does not exist for it; the event lines and the header do
        # ... and it is HELD OUT: the max over a five-figure family is selection-inflated at a
        # few hundred items (free-running it is attained by a HEADER address at chance), so the
        # member is chosen on one half of a disjoint pool and scored on the other.
        pool_items = TK.generate(spec, "test", n=2 * a.floor_n, length=L)
        spaces = tuple(z for z in V.S5_BIND_V3_PAD_SPACES if z != "pad")
        fit = V.s5_bind_v3_pad_write_scores(pool_items[:a.floor_n], k, m, pad=2, rows=(),
                                            fmt=a.fmt)["fixed_reads"]["keys"]
        held = V.s5_bind_v3_pad_write_scores(pool_items[a.floor_n:], k, m, pad=2, rows=(),
                                             fmt=a.fmt, fixed_members=fit)["fixed_reads"]
        fr = V.s5_bind_v3_pad_fixed_reads(
            pool_items[a.floor_n:], parts=V.S5_BIND_V3_PAD_CLOSED_PARTS, top=4, fmt=a.fmt,
            spaces=spaces)
        fr["in_sample"] = fr["best"]
        fr["best"] = held["best"]
        missing = empty_blocks / max(1, sum(tot.values()))
        cell = {"n_items": len(items), "seconds": round(time.time() - t0, 1),
                "timeout_s": timeout,
                # A CELL IS VOID ON COMPLETENESS, not scored down. Absent blocks count as misses
                # in ``acc``, so a cell that lost items to timeouts reports a number that is a
                # completeness artifact; it is marked rather than quietly compared to a floor.
                "void": bool(missing > a.max_missing or n_empty),
                "void_reason": ("missing_blocks %.3f > %.3f" % (missing, a.max_missing)
                                if missing > a.max_missing else
                                (f"{n_empty} empty replies" if n_empty else None)),
                "finish": dict(fins), "empty_replies": n_empty,
                "missing_blocks": empty_blocks / max(1, sum(tot.values())),
                "acc": {key: hit[key] / tot[key] for key in sorted(tot) if tot[key]},
                "n_tokens": {key: tot[key] for key in sorted(tot)},
                "free_run_floor": fr["best"], "chance": 1.0 / k}
        res["cells"][f"{a.spec}@{L}"] = cell
        print(f"\n{a.spec}@{L}  k={k} n={len(items)}  {cell['seconds']}s  "
              f"finish={dict(fins)} empty={n_empty} missing_blocks={cell['missing_blocks']:.3f}"
              + ("   *** VOID: " + cell["void_reason"] + " ***" if cell["void"] else ""))
        for key, v in cell["acc"].items():
            fl = (fr["best"].get(key) or {}).get("acc")
            bar = "" if fl is None else f"   floor {fl:.4f} [{fr['best'][key]['member']}]"
            print(f"   {key:18s} {v:.4f}  (n={tot[key]}){bar}")
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    if a.out:
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
