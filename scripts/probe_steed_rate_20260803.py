"""What one s5_bind_v3 item COSTS on steed, in tokens and in seconds.

The registration round measured the endpoint on short generations (16.3-16.4 completion
tok/s, flat across 1/2/4/8 workers) and bounded the LONG-generation rate only from below
(one composed@L128 item still running at 45 minutes under a 32,768-token cap, so < 12.1
tok/s). Those two numbers decide whether this round's grid is affordable at all, and they
disagree by more than a factor the plan can absorb, so they are re-taken here on the exact
prompts the sweep sends.

Three things are measured, in the order they bind:
  rate       one item per (cell, k, L), timed end to end, tokens from the server's usage.
  workers    the same item count at 1 and at 4 concurrent calls: if throughput is flat the
             grid is priced at one call at a time, which is what makes or breaks it.
  budget     completion tokens actually spent, so the sweep's per-length budgets are set
             from THIS model's trace length and not from the local Qwen's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_frontier_benchmark as RFB  # noqa: E402
from factworld import benchmark as B  # noqa: E402
from factworld import tasks as TK  # noqa: E402

MODEL = "steed/deepseek-v4-flash"
TASKS = {"composed": "s5_bind_v3", "state": "s5_bind_v3_state", "bind": "s5_bind_v3_bind"}
OUT = os.path.join(REPO, "results", "probes", "steed_rate_20260803.json")


def spec_for(cell: str, k: int):
    return TK.CANONICAL[TASKS[cell]].scaled(k=k, n_objects=k, n_objects_active=k)


def build_prompts(cell: str, k: int, L: int, n: int, seed_off: int = 0):
    items = TK.generate(spec_for(cell, k), "test", n=n + seed_off, length=L)
    return items[seed_off:]


def one_call(client, model_name, prompt, budget, effort, sysprompt):
    msgs = []
    if sysprompt:
        msgs.append({"role": "system", "content": sysprompt})
    msgs.append({"role": "user", "content": prompt})
    kw = dict(model=model_name, messages=msgs, max_tokens=budget, temperature=0.0)
    if effort:
        kw["reasoning_effort"] = effort
    t0 = time.time()
    try:
        r = client.chat.completions.create(**kw)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "wall_s": round(time.time() - t0, 1)}
    dt = time.time() - t0
    ch = r.choices[0]
    u = r.usage
    return {"wall_s": round(dt, 1), "ctok": u.completion_tokens, "ptok": u.prompt_tokens,
            "finish": ch.finish_reason, "tok_per_s": round(u.completion_tokens / max(dt, 1e-9), 2),
            "text_tail": (ch.message.content or "")[-160:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["composed"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--lengths", nargs="+", type=int, default=[128])
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="rate")
    a = ap.parse_args()

    from openai import OpenAI
    base_url, key_env = B.endpoint_for(MODEL, B.DEFAULT_BASE_URL)
    client = OpenAI(api_key=os.environ.get(key_env) or "no-key", base_url=base_url,
                    timeout=7200, max_retries=0)
    model_name = B.MODELS[MODEL].get("model_name", MODEL)

    records = []
    for cell in a.cells:
        for L in a.lengths:
            items = build_prompts(cell, a.k, L, a.n)
            probe_cell = {"facet": "s5_bind_v3_kl_sweep", "cell": cell, "task": TASKS[cell],
                          "length": L, "n": a.n,
                          "settings": B._settings(a.effort, max_new_tokens=a.budget)}
            sysprompt = RFB.system_prompt_for(probe_cell)
            prompts = [it.prompt for it in items]
            t0 = time.time()
            if a.workers > 1:
                with ThreadPoolExecutor(max_workers=a.workers) as ex:
                    res = list(ex.map(lambda p: one_call(client, model_name, p, a.budget,
                                                         a.effort, sysprompt), prompts))
            else:
                res = [one_call(client, model_name, p, a.budget, a.effort, sysprompt)
                       for p in prompts]
            wall = time.time() - t0
            ok = [r for r in res if "ctok" in r]
            rec = {"tag": a.tag, "cell": cell, "k": a.k, "L": L, "n": a.n,
                   "workers": a.workers, "budget": a.budget, "effort": a.effort,
                   "wall_s": round(wall, 1),
                   "agg_tok_per_s": round(sum(r["ctok"] for r in ok) / max(wall, 1e-9), 2),
                   "ctok": [r.get("ctok") for r in res],
                   "ptok": [r.get("ptok") for r in res],
                   "finish": [r.get("finish") for r in res],
                   "per_call_s": [r.get("wall_s") for r in res],
                   "errors": [r.get("error") for r in res if r.get("error")],
                   "tails": [r.get("text_tail") for r in res],
                   "ts": datetime.now(timezone.utc).isoformat()}
            records.append(rec)
            print(json.dumps({kk: vv for kk, vv in rec.items() if kk != "tails"}, indent=1),
                  flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prev = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            prev = json.load(fh)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(prev + records, fh, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
