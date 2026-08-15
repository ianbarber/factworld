"""Bring the STEED roster arm up and down (DeepSeek V4 Flash on a DGX Spark, no paid endpoint).

The steed arm is a first-class ``factworld.benchmark.MODELS`` entry whose ``base_url``
points at steed over the tailnet. This script is the only supported way to start and stop
the server behind that entry, and it reads the entry itself for the URL, the served model
name and the context length — so the server and the registry cannot disagree about what is
being measured. It uses NO GPU on this box: steed is a separate machine, so the arm runs
beside a local training job.

    UP      .venv-api/bin/python scripts/serve_steed_model.py up
            If the endpoint already answers with the registry's model at the registry's
            context length, does nothing. Otherwise starts the on-demand unit over SSH
            (``systemctl --user start ds4-server``) and POLLS /v1/models until it answers.
            A cold load of the ~81 GB model takes several minutes.

    STATUS  .venv-api/bin/python scripts/serve_steed_model.py status
            Endpoint reachability, the ids served, the context length, the systemd unit
            state, and whether all of that matches the registry. Exit 0 only when it does.

    DOWN    .venv-api/bin/python scripts/serve_steed_model.py down
            ``systemctl --user stop ds4-server`` over SSH, then polls until the endpoint
            stops answering. The ~81 GB model is resident while the unit runs, which is
            why the unit is on-demand rather than enabled at login.

WHY EVERY SSH CALL LOOKS LIKE THIS. ``BatchMode=yes`` (never prompt), an explicit argv
list with no shell on this side, and stdin from /dev/null: an ssh that inherits the
launching shell's stdin eats the caller's input and can leave the terminal wedged. The
remote command is a systemd unit name, never a pattern — ``pkill -f`` on a server name
matches the launching shell itself, which is how a launch has been killed by its own stop
command before. Nothing here can signal a process on this machine.

WHY IT FAILS LOUDLY. Every wait is bounded and every failure prints what to look at on
steed (``journalctl --user -u ds4-server -n 50``). A silent retry loop against an endpoint
that will never answer is the failure this round already paid for once: an over-large
max_tokens produced 51 minutes of retries instead of an error.

CONCURRENCY IS 1, AND MEASURED. The unit passes no ``--batched-session``, so ds4-server
allocates a single KV session and serializes requests: at 1/2/4/8 concurrent calls
throughput was flat at 16.3-16.4 completion tok/s while per-call latency scaled linearly
(7.2 s -> 57.7 s). The registry entry carries ``max_workers: 1`` and the runner clamps to
it; ``status`` re-reports the number so the two cannot drift apart silently. Long
generations run slower than short ones — one s5_bind_v3 composed item at L=128 was still
generating after 45 minutes at a 32,768-token cap — so on this endpoint a cell's
completion budget is also its per-item duration, and that, not the context window, is what
bounds a battery here.

THE WINDOW IS THE UNIT FILE'S, NOT THIS REPO'S. ``--ctx`` lives in steed's systemd unit
and is tuned there; the server has answered at 65,536, 393,216 and 262,144 within one
evening, and a restart is what picks a change up. So the registry's ``max_model_len`` is a
FLOOR the instrument plans against, ``status``/``up`` refuse only a window BELOW it, and
what the server reports live is what the runner checks each cell against. The window also
decides whether the reasoning arm ``max`` is a rung of its own: ds4 serves it only at
393,216 or more and otherwise decodes it as the high band.

Examples:
    .venv-api/bin/python scripts/serve_steed_model.py up
    .venv-api/bin/python scripts/serve_steed_model.py status
    .venv-api/bin/python scripts/run_frontier_benchmark.py \\
        --models steed/deepseek-v4-flash --facets sanity
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld.benchmark import MODELS, endpoint_for  # noqa: E402

# The registry slug this script serves. Everything else is read off its entry.
SLUG = "steed/deepseek-v4-flash"
SSH_HOST = os.environ.get("DS4_HOST", "steed")
UNIT = os.environ.get("DS4_UNIT", "ds4-server")
JOURNAL_HINT = f"on {SSH_HOST}:  journalctl --user -u {UNIT} -n 50"


# --- registry-derived settings -------------------------------------------------

def settings(slug: str = SLUG) -> dict:
    """URL / served model name / context length for the steed arm, off the registry."""
    reg = MODELS[slug]
    base_url, key_env = endpoint_for(slug)
    return {
        "slug": slug,
        "base_url": base_url,
        "model_name": reg["model_name"],
        "max_model_len": reg["max_model_len"],
        "max_workers": reg.get("max_workers"),
        "key_env": key_env,
        "key_optional": bool(reg.get("api_key_optional")),
    }


def api_key(cfg: dict) -> str | None:
    """The endpoint's key, or None when it needs none.

    steed is tailnet-only and checks no key, so a missing var is not an error here
    (``api_key_optional`` in the registry). It is still read and sent when set, so
    putting auth on the box later changes nothing on this side.
    """
    key = os.environ.get(cfg["key_env"])
    if not key and not cfg["key_optional"]:
        raise SystemExit(f"{cfg['key_env']} not set (required for {cfg['slug']})")
    return key


# --- health --------------------------------------------------------------------

def health(cfg: dict, key: str | None, timeout: float = 5.0) -> dict:
    """``{alive, ids, served, context_len, error}`` for the endpoint right now.

    /v1/models is both the readiness check and the authority on the window: the
    server does not answer it until the model is loaded. ``ids`` is every id the
    listing advertises, which on this server is NOT the set of distinct models —
    deepseek-v4-pro is an alias of deepseek-v4-flash
    (results/probes/steed_ds4_identity_20260802.json) — so what is checked is that
    the REGISTRY's model_name is among them.
    """
    out = {"alive": False, "ids": [], "served": None, "context_len": None, "error": None}
    req = urllib.request.Request(f"{cfg['base_url']}/models")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cards = json.loads(resp.read().decode("utf-8")).get("data") or []
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["alive"] = True
    out["ids"] = [c.get("id") for c in cards]
    for card in cards:
        if card.get("id") == cfg["model_name"]:
            out["served"] = card["id"]
            tp = card.get("top_provider") or {}
            out["context_len"] = card.get("context_length") or tp.get("context_length")
            break
    return out


def conflict(cfg: dict, h: dict) -> str | None:
    """Why a live server is NOT the one the registry describes (None if it is).

    The context window is a FLOOR, not an equality (``context_is_minimum``):
    steed's ``--ctx`` lives in a unit file outside this repo and is tuned there.
    A window LARGER than the registry plans against is safe — every budget was
    checked against the smaller number — so only a smaller one is a conflict.
    """
    if h["served"] != cfg["model_name"]:
        return f"serves {h['ids']}, registry expects {cfg['model_name']!r}"
    if h["context_len"] is None:
        return "the listing declares no context length"
    if h["context_len"] < cfg["max_model_len"]:
        return (f"served context length {h['context_len']} is BELOW the "
                f"{cfg['max_model_len']} every planned budget was checked "
                f"against — raise --ctx in the unit or lower max_model_len")
    return None


# --- ssh -----------------------------------------------------------------------

def ssh(*remote: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Run one command on steed. Never a shell on this side, never a pattern on that one.

    stdin is /dev/null so this cannot consume the launching shell's input, and the
    remote argument is a literal systemd unit name — a ``pkill -f`` style pattern
    would match the launching shell itself.
    """
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", SSH_HOST, *remote],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)


def unit_state() -> str:
    p = ssh("systemctl", "--user", "is-active", UNIT)
    return (p.stdout or p.stderr or "").strip() or "unknown"


# --- commands ------------------------------------------------------------------

def cmd_status(cfg: dict, args) -> int:
    key = api_key(cfg)
    h = health(cfg, key)
    print(f"slug           {cfg['slug']}")
    print(f"endpoint       {cfg['base_url']}")
    print(f"unit           {SSH_HOST}:{UNIT} {unit_state()}")
    print(f"alive          {h['alive']}" + (f"  ({h['error']})" if h["error"] else ""))
    print(f"advertised ids {h['ids']}")
    print(f"served         {h['served']}  (registry: {cfg['model_name']})")
    print(f"context length {h['context_len']}  (registry plans against at least "
          f"{cfg['max_model_len']}, TOTAL prompt+completion)")
    print(f"max workers    {cfg['max_workers']}  (measured; the server holds one KV session)")
    if not h["alive"]:
        print(f"\nNOT READY. Start it:  {sys.argv[0]} up\n{JOURNAL_HINT}")
        return 1
    why = conflict(cfg, h)
    if why:
        print(f"\nCONFLICT: {why}")
        return 1
    print("\nREADY (registry and server agree)")
    return 0


def cmd_up(cfg: dict, args) -> int:
    key = api_key(cfg)
    h = health(cfg, key)
    if h["alive"]:
        why = conflict(cfg, h)
        if why:
            print(f"steed is up but {why}. Refusing to measure a different server "
                  f"under this slug; nothing was started or stopped.")
            return 1
        print(f"already up: {h['served']} @ {h['context_len']} tokens")
        return 0

    print(f"waking {SSH_HOST}:{UNIT} over SSH (a cold load of the ~81 GB model takes "
          f"several minutes)...", flush=True)
    p = ssh("systemctl", "--user", "start", UNIT)
    if p.returncode != 0:
        print(f"could not start {UNIT} on {SSH_HOST} (exit {p.returncode})")
        if p.stderr.strip():
            print(f"  {p.stderr.strip()}")
        print(f"{JOURNAL_HINT}")
        return 1

    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        time.sleep(args.poll)
        h = health(cfg, key)
        if h["alive"]:
            break
        state = unit_state()
        if state in ("failed", "inactive"):
            # The unit gave up: polling on would be a silent retry against a
            # server that is never coming.
            print(f"{UNIT} is {state} — it is not starting.\n{JOURNAL_HINT}")
            return 1
        print(f"  waiting ({int(deadline - time.time())}s left, unit {state})", flush=True)
    if not h["alive"]:
        print(f"{cfg['base_url']} did not answer within {args.startup_timeout}s "
              f"(last error: {h['error']}).\n{JOURNAL_HINT}")
        return 1
    why = conflict(cfg, h)
    if why:
        print(f"came up, but {why}.\n{JOURNAL_HINT}")
        return 1
    print(f"READY: {h['served']} @ {h['context_len']} tokens "
          f"(max_workers {cfg['max_workers']})")
    return 0


def cmd_down(cfg: dict, args) -> int:
    key = api_key(cfg)
    p = ssh("systemctl", "--user", "stop", UNIT)
    if p.returncode != 0:
        print(f"could not stop {UNIT} on {SSH_HOST} (exit {p.returncode})")
        if p.stderr.strip():
            print(f"  {p.stderr.strip()}")
        return 1
    deadline = time.time() + args.stop_timeout
    while time.time() < deadline:
        if not health(cfg, key, timeout=3.0)["alive"]:
            print(f"stopped ({UNIT} on {SSH_HOST}); the model is no longer resident")
            return 0
        time.sleep(args.poll)
    print(f"{UNIT} was stopped but {cfg['base_url']} still answers after "
          f"{args.stop_timeout}s.\n{JOURNAL_HINT}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("up", "status", "down"))
    ap.add_argument("--startup-timeout", type=float, default=900.0,
                    help="Seconds to wait for a cold load before failing (default 900).")
    ap.add_argument("--stop-timeout", type=float, default=120.0)
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args()
    cfg = settings()
    return {"up": cmd_up, "status": cmd_status, "down": cmd_down}[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
