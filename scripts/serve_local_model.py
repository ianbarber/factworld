"""Bring the LOCAL roster arm up and down (vLLM, one GPU, no paid endpoint).

The local arm is a first-class ``factworld.benchmark.MODELS`` entry whose
``base_url`` points at this machine. This script is the only supported way to
start and stop the server behind that entry, and it reads the entry itself for
the host/port, the served model name and the context length — so the server and
the registry cannot disagree about what is being measured.

    UP      .venv-serve/bin/python scripts/serve_local_model.py up
            Waits for the GPU to be free (the card is single: training and
            serving cannot share it), launches vLLM in its own process session,
            polls /health until the server answers, and prints the served
            context length + GPU memory in use. Idempotent: a server that is
            already healthy at the registry's model/length is left alone; one
            healthy at a DIFFERENT model or length is reported as a conflict and
            nothing is started.

    STATUS  .venv-serve/bin/python scripts/serve_local_model.py status
            pid, health, served model + context length, GPU memory in use.
            Exit 0 only when the server is healthy at the registry's settings.

    DOWN    .venv-serve/bin/python scripts/serve_local_model.py down
            SIGTERM to the recorded process GROUP (the server is launched with
            start_new_session=True, so its group contains the engine-core
            workers and nothing else), then SIGKILL after --kill-timeout.
            NEVER pkill: ``pkill -f "[v]llm serve"`` matches the launching shell
            and has killed the launch itself.

    LOGS    .venv-serve/bin/python scripts/serve_local_model.py logs [-n N]

Every command needs the key env var, so source the env first:

    set -a; source .env; set +a

Both --api-key and the runner's key come from that one var (the registry entry's
``api_key_env``); the server rejects requests without it, so a misconfigured
client fails with 401 instead of silently talking to the wrong endpoint.

Once it is up, the arm runs through the ordinary runner and writes ordinary C3
records (at cost 0.0):

    .venv-api/bin/python scripts/run_frontier_benchmark.py \\
        --models local/qwen3.6-35b-a3b-nvfp4 --facets sanity

Weights come from HF_HUB_CACHE=/mnt/nas/hf-cache/hub, which is NFS. ``up`` polls
/health rather than sleeping, so it returns as soon as the server is actually
ready and fails loudly at --startup-timeout otherwise. Measured on this machine
(RTX 5090, 32,607 MiB): 240 s from launch to healthy on a cold compile cache
(39 s weight load, 47 s torch.compile, the rest warmup and CUDA-graph capture)
and 125 s on a warm one (31 s load, 4.9 s compile); 27,894 MiB of the card in
use at --gpu-util 0.90, of which 20.4 GiB is weights; 283,648 tokens of KV cache at the
served max_model_len of 131,072 — more than two full-length sequences;
~200-240 generated tokens/s single-stream.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld.benchmark import MODELS, endpoint_for  # noqa: E402

# The registry slug this script serves. Everything else is read off its entry.
SLUG = "local/qwen3.6-35b-a3b-nvfp4"
VLLM_BIN = os.path.join(REPO, ".venv-serve", "bin", "vllm")
HF_HUB_CACHE = "/mnt/nas/hf-cache/hub"
PID_PATH = os.path.join(REPO, "logs", "vllm_local_qwen.pid")
LOG_PATH = os.path.join(REPO, "logs", "vllm_local_qwen.log")


# --- registry-derived settings -------------------------------------------------

def settings(slug: str = SLUG) -> dict:
    """Host/port/model/context length for the local arm, read off the registry."""
    reg = MODELS[slug]
    base_url, key_env = endpoint_for(slug)
    host_port = base_url.split("//", 1)[1].split("/", 1)[0]
    host, _, port = host_port.partition(":")
    return {
        "slug": slug,
        "base_url": base_url,
        "host": host,
        "port": int(port or 80),
        "model_name": reg["model_name"],
        "hf_repo": reg.get("hf_repo") or reg["model_name"],
        "max_model_len": reg["max_model_len"],
        "key_env": key_env,
    }


def api_key(cfg: dict) -> str:
    key = os.environ.get(cfg["key_env"])
    if not key:
        raise SystemExit(
            f"{cfg['key_env']} not set. The server requires it and so does the runner "
            f"(factworld.benchmark.MODELS[{cfg['slug']!r}]['api_key_env']). "
            f"Run:  set -a; source .env; set +a")
    return key


# --- health --------------------------------------------------------------------

def _get(url: str, key: str | None, timeout: float = 5.0):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return resp.status, body


def health(cfg: dict, key: str) -> dict:
    """``{alive, served, max_model_len, error}`` for the endpoint right now.

    ``alive`` is /health answering 200 (vLLM's real readiness endpoint — it does
    not answer until the engine has finished loading), ``served``/``max_model_len``
    come from /v1/models, which is the authority on the context window the server
    actually built (``--max-model-len`` can be clamped by the engine).
    """
    out = {"alive": False, "served": None, "max_model_len": None, "error": None}
    root = f"http://{cfg['host']}:{cfg['port']}"
    try:
        status, _ = _get(f"{root}/health", None)
        out["alive"] = status == 200
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        _status, body = _get(f"{cfg['base_url']}/models", key)
        cards = json.loads(body).get("data") or []
        for card in cards:
            if card.get("id") == cfg["model_name"]:
                out["served"] = card["id"]
                out["max_model_len"] = card.get("max_model_len")
                break
        else:
            out["served"] = cards[0]["id"] if cards else None
            out["max_model_len"] = cards[0].get("max_model_len") if cards else None
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, KeyError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def conflict(cfg: dict, h: dict) -> str | None:
    """Why a live server is NOT the one the registry describes (None if it is)."""
    if h["served"] != cfg["model_name"]:
        return f"serving {h['served']!r}, registry expects {cfg['model_name']!r}"
    if h["max_model_len"] != cfg["max_model_len"]:
        return (f"served context length {h['max_model_len']}, registry expects "
                f"{cfg['max_model_len']}")
    return None


# --- GPU ------------------------------------------------------------------------

def gpu_state() -> dict:
    """``{used_mib, total_mib, free_mib, compute_apps}`` from nvidia-smi.

    ``compute_apps`` lists CUDA compute processes only — the desktop's Xorg /
    gnome-shell are graphics contexts and never appear, so an empty list means
    the card is free for a job (not that nothing is on the display).
    """
    def smi(args):
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    mem = smi(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    used, total = (int(x) for x in mem.splitlines()[0].split(","))
    apps = []
    raw = smi(["--query-compute-apps=pid,process_name,used_memory",
               "--format=csv,noheader,nounits"])
    for line in raw.splitlines():
        if not line.strip():
            continue
        pid, name, mib = (p.strip() for p in line.split(",", 2))
        try:
            apps.append({"pid": int(pid), "name": name, "used_mib": int(mib)})
        except ValueError:
            # a row nvidia-smi could not fill in ("[N/A]") still means the card
            # is occupied, which is the only thing the wait loop needs
            apps.append({"pid": -1, "name": name, "used_mib": 0})
    return {"used_mib": used, "total_mib": total, "free_mib": total - used,
            "compute_apps": apps}


def wait_for_gpu(need_mib: int, timeout_s: float, poll_s: float = 15.0) -> dict:
    """Poll until no compute process holds the card and ``need_mib`` is free.

    The GPU is single: a sibling's training and this server cannot share it, and
    competing for it corrupts both. Polling (never pre-empting) is the protocol.

    The free-memory bar defaults to ``gpu_util * total`` because that is what
    vLLM will try to take; desktop graphics contexts count against it, so if the
    display is holding a lot the fix is a lower --gpu-util, not a longer wait.
    """
    deadline = time.time() + timeout_s
    announced = False
    while True:
        g = gpu_state()
        others = g["compute_apps"]
        if not others and g["free_mib"] >= need_mib:
            return g
        if not announced:
            print(f"waiting for the GPU: {len(others)} compute process(es), "
                  f"{g['free_mib']} MiB free of {g['total_mib']} (need {need_mib})",
                  flush=True)
            announced = True
        if time.time() >= deadline:
            raise SystemExit(
                f"GPU still busy after {timeout_s:.0f}s: "
                f"{[(a['pid'], a['name'], a['used_mib']) for a in others]}, "
                f"{g['free_mib']} MiB free (need {need_mib}). Nothing launched. "
                f"If the shortfall is the desktop rather than a job, lower "
                f"--gpu-util (and --need-mib with it).")
        time.sleep(poll_s)


# --- process handling -------------------------------------------------------------

def read_pid() -> int | None:
    """The recorded server pid if that process is still alive AND is our server.

    The cmdline check is what makes ``down`` safe after a pid is recycled: a
    stale pidfile pointing at some unrelated process must never be signalled.
    """
    try:
        with open(PID_PATH, encoding="utf-8") as fh:
            pid = int(fh.read().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmdline = fh.read().decode("utf-8", "replace").replace("\0", " ")
    except FileNotFoundError:
        return None
    return pid if ("vllm" in cmdline and "serve" in cmdline) else None


def launch(cfg: dict, key: str, gpu_util: float, max_num_seqs: int,
           extra: list[str]) -> int:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    cmd = [
        VLLM_BIN, "serve", cfg["hf_repo"],
        "--served-model-name", cfg["model_name"],
        "--host", cfg["host"], "--port", str(cfg["port"]),
        "--max-model-len", str(cfg["max_model_len"]),
        "--gpu-memory-utilization", str(gpu_util),
        "--max-num-seqs", str(max_num_seqs),
        "--api-key", key,
        *extra,
    ]
    env = {**os.environ, "HF_HUB_CACHE": HF_HUB_CACHE}
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                  f"{' '.join(c if c != key else '<api-key>' for c in cmd)} ===\n")
        log.flush()
        # start_new_session: the server becomes its own session/group leader, so
        # ``down`` can signal the whole group (engine-core workers included)
        # without the signal reaching this script or the shell that ran it.
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=REPO, start_new_session=True)
    with open(PID_PATH, "w", encoding="utf-8") as fh:
        fh.write(str(proc.pid))
    return proc.pid


def wait_healthy(cfg: dict, key: str, pid: int, timeout_s: float,
                 poll_s: float = 5.0) -> dict:
    """Poll /health until the server answers, or the process dies, or we time out."""
    deadline = time.time() + timeout_s
    while True:
        h = health(cfg, key)
        if h["alive"]:
            return h
        if read_pid() is None:
            raise SystemExit(f"vLLM exited during startup (pid {pid}); "
                             f"last log lines:\n{tail(LOG_PATH, 30)}")
        if time.time() >= deadline:
            raise SystemExit(f"server not healthy after {timeout_s:.0f}s "
                             f"(pid {pid} still running); last log lines:\n"
                             f"{tail(LOG_PATH, 30)}")
        time.sleep(poll_s)


def tail(path: str, n: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except FileNotFoundError:
        return f"(no log at {path})"


def _same_group(pid_a: int, pid_b: int) -> bool:
    try:
        return os.getpgid(pid_a) == os.getpgid(pid_b)
    except (ProcessLookupError, PermissionError):
        return False


def report(cfg: dict, h: dict, pid: int | None) -> None:
    g = gpu_state()
    # The CUDA context belongs to the EngineCore worker, not to the launched
    # front-end process, so the server's own VRAM is found by process GROUP.
    mine = [a for a in g["compute_apps"]
            if pid is not None and _same_group(a["pid"], pid)]
    print(f"  pid            {pid}")
    print(f"  endpoint       {cfg['base_url']}  (health "
          f"{'ok' if h['alive'] else 'DOWN'})")
    print(f"  served model   {h['served']}")
    print(f"  context length {h['max_model_len']} tokens (registry "
          f"{cfg['max_model_len']})")
    print(f"  gpu memory     {g['used_mib']} MiB used of {g['total_mib']} "
          f"({g['free_mib']} MiB free)" +
          (f"; this server {mine[0]['used_mib']} MiB" if mine else ""))


# --- commands ---------------------------------------------------------------------

def cmd_up(a) -> int:
    cfg = settings()
    key = api_key(cfg)
    pid = read_pid()
    h = health(cfg, key)
    if h["alive"]:
        why = conflict(cfg, h)
        if why:
            raise SystemExit(
                f"a DIFFERENT server already answers {cfg['base_url']}: {why}. "
                f"Nothing started — stop it first (`down`) or fix the registry.")
        print("already up (idempotent no-op)")
        report(cfg, h, pid)
        return 0
    if pid is not None:
        raise SystemExit(f"pid {pid} is a live vllm process but the endpoint is not "
                         f"healthy ({h['error']}). Run `down` first.")
    if not os.path.exists(VLLM_BIN):
        raise SystemExit(f"{VLLM_BIN} not found (expected vLLM in .venv-serve)")
    if not shutil.which("nvidia-smi"):
        raise SystemExit("nvidia-smi not found; refusing to launch blind")

    total = gpu_state()["total_mib"]
    need = int(a.gpu_util * total) if a.need_mib is None else a.need_mib
    g = wait_for_gpu(need, a.gpu_timeout)
    print(f"gpu free: {g['free_mib']} MiB of {g['total_mib']}, no compute processes",
          flush=True)
    pid = launch(cfg, key, a.gpu_util, a.max_num_seqs, a.extra or [])
    print(f"launched pid {pid} -> {LOG_PATH} (polling /health, timeout "
          f"{a.startup_timeout:.0f}s)", flush=True)
    t0 = time.time()
    h = wait_healthy(cfg, key, pid, a.startup_timeout)
    why = conflict(cfg, h)
    if why:
        raise SystemExit(f"server came up but does not match the registry: {why}")
    print(f"healthy in {time.time() - t0:.0f}s")
    report(cfg, h, pid)
    return 0


def cmd_status(a) -> int:
    cfg = settings()
    key = os.environ.get(cfg["key_env"])
    h = health(cfg, key)
    pid = read_pid()
    report(cfg, h, pid)
    if not h["alive"]:
        print(f"  error          {h['error']}")
        return 1
    why = conflict(cfg, h)
    if why:
        print(f"  MISMATCH       {why}")
        return 1
    return 0


def cmd_down(a) -> int:
    cfg = settings()
    pid = read_pid()
    if pid is None:
        print("no live server recorded (pidfile absent, stale, or not a vllm process)")
        if os.path.exists(PID_PATH):
            os.remove(PID_PATH)
        return 0
    pgid = os.getpgid(pid)
    if pgid == os.getpgid(0):
        # Cannot happen with start_new_session, but a wrong group here would
        # signal this script and its parent shell — exactly the pkill accident.
        raise SystemExit(f"refusing to signal process group {pgid}: it is our own")
    print(f"SIGTERM to process group {pgid} (pid {pid})")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + a.kill_timeout
    while read_pid() is not None and time.time() < deadline:
        time.sleep(1.0)
    if read_pid() is not None:
        print(f"still alive after {a.kill_timeout:.0f}s: SIGKILL")
        os.killpg(pgid, signal.SIGKILL)
        time.sleep(2.0)
    if os.path.exists(PID_PATH):
        os.remove(PID_PATH)
    g = gpu_state()
    print(f"down. gpu memory {g['used_mib']} MiB used of {g['total_mib']}, "
          f"{len(g['compute_apps'])} compute process(es)")
    return 0


def cmd_logs(a) -> int:
    print(tail(LOG_PATH, a.lines), end="")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="wait for the GPU, launch vLLM, poll /health")
    up.add_argument("--gpu-util", type=float, default=0.90, dest="gpu_util",
                    help="vLLM --gpu-memory-utilization (default 0.90).")
    up.add_argument("--max-num-seqs", type=int, default=8, dest="max_num_seqs",
                    help="Concurrent sequences (default 8 = the runner's "
                         "--max-workers).")
    up.add_argument("--need-mib", type=int, default=None, dest="need_mib",
                    help="Free-VRAM bar to wait for (default: gpu_util * total).")
    up.add_argument("--gpu-timeout", type=float, default=7200.0, dest="gpu_timeout",
                    help="Give up waiting for a free card after this many seconds.")
    up.add_argument("--startup-timeout", type=float, default=1800.0,
                    dest="startup_timeout",
                    help="Give up waiting for /health after this many seconds "
                         "(a cold 22 GB NFS load takes minutes).")
    up.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="Extra flags passed through to `vllm serve`.")
    up.set_defaults(func=cmd_up)

    st = sub.add_parser("status", help="pid, health, served length, GPU memory")
    st.set_defaults(func=cmd_status)

    dn = sub.add_parser("down", help="SIGTERM the recorded process group, then SIGKILL")
    dn.add_argument("--kill-timeout", type=float, default=60.0, dest="kill_timeout")
    dn.set_defaults(func=cmd_down)

    lg = sub.add_parser("logs", help="tail the server log")
    lg.add_argument("-n", "--lines", type=int, default=40)
    lg.set_defaults(func=cmd_logs)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
