"""What steed's DeepSeek V4 endpoint actually is, and what it can be asked for.

Two questions had to be answered before ``steed/deepseek-v4-flash`` could carry a roster
slug, and both are answered from the WIRE rather than from the /v1/models listing.

IDENTITY (``identity``). steed's ``/v1/models`` lists two ids — ``deepseek-v4-flash`` and
``deepseek-v4-pro`` — and gives BOTH the display name "DeepSeek V4 Flash". A slug per
advertised id would put the same weights on the board twice under two names, and a sweep
across the two would read as a model comparison. Greedy decoding (temperature 0, fixed
seed) on prompts chosen to diverge if anything about the weights or the sampling path
differs: a FactWorld composed item, a long arithmetic chain, a free-form continuation, and
one thinking-on generation run out to several hundred tokens (a trace is the most
sensitive probe of identity — every token conditions the next). Each prompt is called
THREE times: flash, flash again, pro. The flash/flash pair is the self-consistency
baseline the flash/pro comparison is read against; without it, "identical" and
"deterministic" are not distinguishable.

BUDGET (``budget``). The context is a TOTAL, prompt plus completion, so a completion budget
is only sound relative to a MEASURED prompt length. This mode sends the longest prompt of
each planned cell's own scored draw with ``max_tokens=1`` and records the
``usage.prompt_tokens`` the server counted — the tokenizer's number, not a chars/token
estimate — then reports the largest completion budget every cell can be given with the
whole draw still inside the window the registry plans against. The cells are the
s5_bind_v3 k x L grid (scripts/sweep_s5bind_v3_local_kL_20260802.py), which is what this
round runs.

Examples:
    .venv-api/bin/python scripts/probe_steed_ds4_20260802.py identity
    .venv-api/bin/python scripts/probe_steed_ds4_20260802.py budget
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factworld import benchmark as B  # noqa: E402
from factworld import tasks as TK  # noqa: E402

SLUG = "steed/deepseek-v4-flash"
BASE_URL = B.MODELS[SLUG]["base_url"]
IDS = ("deepseek-v4-flash", "deepseek-v4-pro")
OUT = os.path.join(REPO, "results", "probes", "steed_ds4_identity_20260802.json")
OUT_BUDGET = os.path.join(REPO, "results", "probes", "steed_ds4_budget_20260802.json")


def chat(model: str, user: str, *, max_tokens: int, effort: str,
         system: str = "Answer the question with only the requested value or values.",
         timeout: float = 1800.0) -> dict:
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": 0.0, "seed": 7,
            "reasoning_effort": effort}
    req = urllib.request.Request(f"{BASE_URL}/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    ch = d["choices"][0]
    return {"model_echo": d.get("model"),
            "content": ch["message"].get("content") or "",
            "finish_reason": ch.get("finish_reason"),
            "usage": d.get("usage"),
            "elapsed_s": round(time.time() - t0, 2)}


def prompts() -> list[dict]:
    item = TK.generate(TK.CANONICAL["s5_bind_v3"], "test", n=1, length=64)[0]
    return [
        {"name": "factworld_composed_L64", "effort": "none", "max_tokens": 64,
         "user": item.prompt},
        {"name": "arithmetic_chain", "effort": "none", "max_tokens": 128,
         "user": "Compute 7*13 + 91/7 - 5*5, then multiply the result by 3. "
                 "Show the intermediate values."},
        {"name": "free_continuation", "effort": "none", "max_tokens": 128,
         "user": "Continue this exactly ten more words: The quantizer folded the"},
        {"name": "thinking_trace", "effort": "high", "max_tokens": 400,
         "user": "A frog starts at 0 on a number line and jumps +3, -1, +3, -1, ... "
                 "How many jumps until it first passes 20? Reason it out."},
    ]


def budget(args) -> int:
    """Measure every planned cell's prompt length and price the completion budget on it."""
    import run_frontier_benchmark as RFB
    import sweep_s5bind_v3_local_kL_20260802 as S

    limit = B.MODELS[SLUG]["max_model_len"]
    rows = []
    for L in S.LS:
        for k in S.KS:
            wm = S.work_matched(k, L)
            for cell, length in (("composed", L), ("state", wm["state"]), ("bind", wm["bind"])):
                spec = S.scaled_spec(cell, k)
                items = TK.generate(spec, "test", n=args.n, length=length)
                # the LONGEST item of the scored draw is what the budget must fit
                ex = max(items, key=lambda e: len(e.prompt))
                c = {"facet": S.FACET, "cell": cell, "task": S.TASKS[cell],
                     "length": length, "n": args.n,
                     "settings": B._settings("high", max_new_tokens=1024)}
                sysp = RFB.system_prompt_for(c)
                d = chat(IDS[0], ex.prompt, max_tokens=1, effort="none", system=sysp)
                ptok = d["usage"]["prompt_tokens"]
                row = {"cell": cell, "k": k, "composed_L": L, "length": length,
                       "prompt_chars": len(sysp) + len(ex.prompt),
                       "prompt_tokens": ptok,
                       "chars_per_token": round((len(sysp) + len(ex.prompt)) / ptok, 3),
                       "headroom_tokens": limit - ptok}
                rows.append(row)
                print(f"  {cell:9s} k={k:2d} L={length:3d} (composed L{L:3d})  "
                      f"ptok={ptok:6d}  headroom={row['headroom_tokens']:6d}", flush=True)

    worst = max(rows, key=lambda r: r["prompt_tokens"])
    out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "slug": SLUG,
           "context_total": limit, "n_per_cell": args.n,
           "max_prompt_tokens": worst["prompt_tokens"],
           "max_prompt_cell": {kk: worst[kk] for kk in ("cell", "k", "composed_L", "length")},
           "max_servable_completion_budget": limit - worst["prompt_tokens"],
           "rows": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\ncontext {limit} TOTAL; longest planned prompt {worst['prompt_tokens']} tokens "
          f"({worst['cell']} k={worst['k']} L={worst['length']}); every cell fits a "
          f"completion budget of {limit - worst['prompt_tokens']}")
    print(f"  written {args.out}")
    return 0


def identity(args) -> int:
    rows = []
    for p in prompts():
        # flash twice (the self-consistency baseline), then pro
        calls = {}
        for label, model in (("flash_a", IDS[0]), ("flash_b", IDS[0]), ("pro", IDS[1])):
            calls[label] = chat(model, p["user"], max_tokens=p["max_tokens"],
                                effort=p["effort"])
            print(f"  {p['name']:24s} {label:8s} "
                  f"ctok={calls[label]['usage']['completion_tokens']:4d} "
                  f"finish={calls[label]['finish_reason']} "
                  f"[{calls[label]['elapsed_s']}s]", flush=True)
        row = {"prompt": p["name"], "effort": p["effort"], "max_tokens": p["max_tokens"],
               "prompt_chars": len(p["user"]),
               "flash_self_identical": calls["flash_a"]["content"] == calls["flash_b"]["content"],
               "flash_pro_identical": calls["flash_a"]["content"] == calls["pro"]["content"],
               "calls": calls}
        rows.append(row)
        print(f"  {p['name']:24s} flash/flash={row['flash_self_identical']} "
              f"flash/pro={row['flash_pro_identical']}", flush=True)

    deterministic = all(r["flash_self_identical"] for r in rows)
    same = all(r["flash_pro_identical"] for r in rows)
    verdict = ("ALIAS: no observable difference" if deterministic and same else
               "DISTINCT: outputs differ where decoding is reproducible" if deterministic
               else "INCONCLUSIVE: decoding is not reproducible on this endpoint")
    out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base_url": BASE_URL,
           "ids": list(IDS), "deterministic": deterministic,
           "flash_pro_identical_everywhere": same, "verdict": verdict, "rows": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n{verdict}\n  written {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    pi = sub.add_parser("identity", help="is deepseek-v4-pro a different model?")
    pi.add_argument("--out", default=OUT)
    pi.set_defaults(fn=identity)
    pb = sub.add_parser(
        "budget",
        help=f"measured prompt lengths vs the {B.MODELS[SLUG]['max_model_len']}-token "
             f"window the registry plans against")
    pb.add_argument("--out", default=OUT_BUDGET)
    pb.add_argument("--n", type=int, default=40,
                    help="Draw size the longest prompt is taken over (the sweep's n).")
    pb.set_defaults(fn=budget)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
