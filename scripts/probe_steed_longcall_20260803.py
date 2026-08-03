"""Does a LONG generation survive the path? One call, one answer, both paths comparable.

Symptom this exists to explain: an s5_bind_v3 composed@L64 item generated 10,482 tokens in
705 s on the server three times in a row, byte-identical, with no server-side error and a new
identical prompt arriving each time. That is a client-side retry loop (``backends`` retries
``APIError``/``APIConnectionError``/``InternalServerError``/``ValueError`` five times and then
returns an empty prediction, which is scored WRONG) — so a cell of long items would have come
back empty and read as a floor.

It is not the request timeout: ``build_backend`` sizes that from the cell's budget and the
registry's measured rate (2 x 32,768 / 12.0 = 5,461 s here), far above 705 s. The two
candidates left are the HTTPS reverse proxy in front of the server (tailscale-serve) and the
client's own parsing of a large body. This script separates them by making the SAME call over
each path with retries OFF, so a failure is reported instead of hidden:

  --path tailnet   https://steed.tailc4bb6.ts.net/v1   (through tailscale serve)
  --path tunnel    http://127.0.0.1:PORT/v1            (ssh -L to the server's own port)

The prompt is a real s5_bind_v3 composed item, so the generation length is the one that fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_frontier_benchmark as RFB                       # noqa: E402
import sweep_s5bind_v3_local_kL_20260802 as S              # noqa: E402
from factworld import benchmark as B                       # noqa: E402
from factworld import tasks as TK                          # noqa: E402

OUT = os.path.join(REPO, "results", "probes", "steed_longcall_20260803.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["tailnet", "tunnel"], default="tunnel")
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--item", type=int, default=0)
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--timeout", type=float, default=7200.0)
    a = ap.parse_args()

    from openai import OpenAI
    if a.path == "tunnel":
        base_url = f"http://127.0.0.1:{a.port}/v1"
    else:
        base_url, _ = B.endpoint_for("steed/deepseek-v4-flash", B.DEFAULT_BASE_URL)

    spec = S.scaled_spec("composed", a.k)
    items = TK.generate(spec, "test", n=a.item + 1, length=a.length)
    prompt = items[a.item].prompt
    cell = {"facet": "s5_bind_v3_kl_sweep", "cell": "composed", "task": "s5_bind_v3",
            "length": a.length, "n": 1, "settings": B._settings("high", max_new_tokens=a.budget)}
    system = RFB.system_prompt_for(cell)

    client = OpenAI(api_key="no-key", base_url=base_url, timeout=a.timeout, max_retries=0)
    t0 = time.time()
    rec = {"path": a.path, "base_url": base_url, "k": a.k, "L": a.length, "item": a.item,
           "budget": a.budget, "client_timeout_s": a.timeout,
           "ts": datetime.now(timezone.utc).isoformat()}
    try:
        r = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            max_tokens=a.budget, reasoning_effort="high")
        dt = time.time() - t0
        rec.update({"ok": True, "wall_s": round(dt, 1),
                    "ctok": r.usage.completion_tokens, "ptok": r.usage.prompt_tokens,
                    "finish": r.choices[0].finish_reason,
                    "tok_per_s": round(r.usage.completion_tokens / dt, 2),
                    "gold": items[a.item].answer,
                    "tail": (r.choices[0].message.content or "")[-200:]})
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "wall_s": round(time.time() - t0, 1),
                    "error_type": type(exc).__name__, "error": str(exc)[:400],
                    "status_code": getattr(exc, "status_code", None)})
    print(json.dumps(rec, indent=1), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prev = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else []
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(prev + [rec], fh, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
