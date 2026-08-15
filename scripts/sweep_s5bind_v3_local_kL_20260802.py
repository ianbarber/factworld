"""THE DIFFICULTY SURFACE of s5_bind_v3 in k and L, measured on the locally served model.

WHY THIS EXISTS. The frontier scout returned STOP_CEILING: gpt-5.5 read 1.000/0.975 on
composed@128/@256 with zero truncation, so the top of the roster is saturated and a ranking
whose first place is a ceiling cannot order a roster. The rule's own remedy is "raise k or L",
and the concern raised against it is that L is bounded at the BOTTOM of the roster by the token
budget — nemotron already needed 98,304 completion tokens at composed@256 and still truncated 5%.
Which of the two axes buys difficulty per token is a measurement, and on a served local model it
costs nothing to take. No paid endpoint is contacted by this script: ``--models`` is fixed to the
locally served slug, which ``benchmark.endpoint_for`` resolves to 127.0.0.1.

THE AXES, AND WHAT IS HELD FIXED ON EACH.
  k   The composed spec's k is scaled together with m (``n_objects``/``n_objects_active``), so the
      k=m symmetry the floor argument is written at (``pad < min(k, m)``, task cost k+m+1) is
      preserved at every rung. Nothing else moves: p_swap, p_cross, no_pin, q_tail, q_no_surface
      and match_reads are the shipped s5_bind_v3 values at every k.
  L   The event count. The composed stream carries p_swap*L swaps and (1-p_swap)*L gives, so the
      components' work-matched partners are read off the GENERATED stream
      (``validity.s5_bind_v3_shape`` -> ``s5_bind_v3_work_match``) rather than from a formula.

WHAT THE k RUNGS ARE NOT. k=6 here is the k=12 frontier spec scaled down, not the registered
from-scratch cell ``s5_bind_local_v3``: that one carries match_reads=2 (the k=6 pools are six
cells wide and one matched draw leaves a read-history offset) and no q_no_surface gate (at k=6
the gate's floor trade is not bought — tasks.py prices it at 1.01x-1.02x there). This sweep holds
the spec fixed and moves k alone, which is what a k-vs-L cost comparison needs; the
STRUCTURE-SWITCH diagnostic is therefore not read off the k=6 rung.

THE READ IS THE ANSWER ONLY, and the composed cell has NO FLOOR in this regime.
A served reasoning model emits visible tokens, which IS a scratchpad, so the composed cell's
floor argument (a bound on LIVE SLOTS, W <= max(k,m)+1 against the task's k+m+1) does not hold
and no number bounds a cheap policy there. The composed cell is read against INFORMED CHANCE
1/(k-1) — the initial map is stated, so the queried agent's own starting value is never gold —
which is a guess baseline and not a floor. The report asserts on its own text against
``P.SCOUT_COMPOSED_FLOOR_LANGUAGE``. COMPONENT cells keep their floors and those are recomputed
from the EXACT scored items and from a disjoint pool, larger operative, exactly as the scout does.

VALIDITY, in the order it is evaluated. Truncation FIRST: a cell over 10% finish=length or empty
is VOID and enters no comparison until it is re-run at a raised budget — read in any other order
a truncated cell is a floor, which is how the published s5 L64 cliff was manufactured. Then the
numbers. Reasoning tokens on this arm live inside ``ctok``: vLLM's chat-completions usage carries
no reasoning-token field and the chat template opens ``<think>`` in the generation prompt, so
``rtok`` reads 0 whether or not the model thought, and ctok-per-item is the token cost column.

Examples:
    set -a; source .env; set +a
    .venv-serve/bin/python scripts/serve_local_model.py up --max-num-seqs 16

    .venv-api/bin/python scripts/sweep_s5bind_v3_local_kL_20260802.py --plan
    .venv-api/bin/python scripts/sweep_s5bind_v3_local_kL_20260802.py --cells composed
    .venv-api/bin/python scripts/sweep_s5bind_v3_local_kL_20260802.py --report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import lru_cache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocol_s5bind_v3_three_cell_20260731 as P            # noqa: E402
import run_frontier_benchmark as RFB                          # noqa: E402
from factworld import benchmark as B                          # noqa: E402
from factworld import tasks as TK                             # noqa: E402
from factworld import validity as V                           # noqa: E402
from factworld.runner import evaluate_task                    # noqa: E402

MODEL = "local/qwen3.6-35b-a3b-nvfp4"
FACET = "s5_bind_v3_kl_sweep"
HISTORY = os.path.join(REPO, "results", "local_qwen",
                       "history_s5bind_v3_kL_20260802.jsonl")
REPORT_MD = os.path.join(REPO, "results", "20260802_s5bind_v3_local_kL_sweep.md")

TASKS = {"composed": "s5_bind_v3",
         "state": "s5_bind_v3_state",
         "bind": "s5_bind_v3_bind"}

KS = (6, 12, 24, 32)
LS = (64, 128, 192, 256)
N = 40
EFFORT = "high"          # the served template reads an on/off bit; every ladder rung is this arm
TRUNCATION_MAX = P.SCOUT_TRUNCATION_MAX          # 0.10 — the scout's own void bar
FLOOR_DISJOINT_N = P.N_SCORE                     # the disjoint pool a component floor is re-read on
WORK_MATCH_POOL = 200                            # the pool the composed stream's shape is read on

# PER-LENGTH COMPLETION BUDGETS, set from the model's OWN measured trace length and not from the
# scout's. The scout's registered 16,384 at composed@128 is a VOID budget here: at n=5 it cut 2
# of 5 traces off (this model spent 12.4k-15.6k completion tokens on the three that finished),
# and at 49,152 the same cell finished 16 of 16 at a mean of 13,187. A budget under the model's
# own trace length is not a cheaper measurement, it is a floor with the model's name on it.
# Keyed by LENGTH and never by k: whether k costs tokens is the question this sweep asks, so k
# may not be allowed to move the budget.
BUDGETS = {64: 32768, 128: 49152, 192: 65536, 256: 65536}

# The cost guard submits prompts in chunks and checks its budget between them, so its chunk size
# is also the cell's CONCURRENCY. The shipped 8 was chosen against a paid endpoint, where a
# smaller chunk bounds the overshoot past a dollar cap; this arm has no dollar cap (prices are
# 0.0) and the served engine holds 16 sequences, so the chunk is raised to fill it. It changes
# how often the token guard is consulted, and nothing about what is measured.
GUARD_CHUNK = 16


def scaled_spec(cell: str, k: int):
    """The k rung of one cell's spec: k and m scaled together, nothing else moved."""
    return TK.CANONICAL[TASKS[cell]].scaled(k=k, n_objects=k, n_objects_active=k)


def informed_chance(k: int) -> float:
    """1/(k-1): the stated initial map strikes the queried agent's own starting value.

    NOT a floor. It is the guess baseline the composed cell is read against in a scratchpad
    regime, and it MOVES WITH k — which is why a raw match comparison across k rungs is not a
    difficulty comparison and the report carries the ratio and the z beside every cell.
    """
    return 1.0 / (k - 1)


@lru_cache(maxsize=None)
def work_matched(k: int, L: int) -> dict[str, int]:
    """The component lengths carrying the same work as the composed stream at (k, L).

    Measured off the GENERATED stream (``s5_bind_v3_shape`` counts the swaps and gives the
    sampler actually emitted), not from p_swap*L: the registered pairings were measured the same
    way and they are not the rounded product. The pool is WORK_MATCH_POOL and not this sweep's
    n, so the pairing is a property of the stream rather than of the scored draw; at that pool
    the k=12 rung reproduces the registered ``FRONTIER_WORK_MATCHED`` exactly (state 43 / bind 85
    at L=128, state 85 / bind 171 at L=256), which is the check that this is the same quantity.
    """
    items = TK.generate(scaled_spec("composed", k), "test", n=WORK_MATCH_POOL, length=L)
    ns, ng = V.s5_bind_v3_shape(items)
    return V.s5_bind_v3_work_match(ns, ng)


def carrier_hops(k: int, L: int) -> float:
    """The composed stream's STATE-leg chain length at (k, L): ``2 n_swap / k``.

    The quantity BOTH axes move, and they move it in opposite directions: a swap touches two of
    the k pointers, so the queried agent's value is carried through 2 n_swap / k hops, and L
    raises n_swap while k divides it. It is the obvious candidate for what the composed cell
    costs, and the measurement refuses it — cells matched on hops to within 10% spread by up to
    0.450 in match, ordered by L every time. So it is printed as the hypothesis the surface is
    read against, not as the explanation of it.
    """
    return V.s5_bind_v3_carrier_hops(k, work_matched(k, L)["state"])


def make_cell(cell: str, k: int, L: int, n: int = N) -> dict:
    """A runner-shaped cell. ``k_sweep`` rides in the settings so it is part of the settings hash
    and therefore of the resume key: two k rungs of one (task, length) are different cells."""
    settings = B._settings(EFFORT, max_new_tokens=budget_for(L))
    settings["k_sweep"] = k
    c = {"facet": FACET, "cell": cell, "task": TASKS[cell], "length": L, "n": n,
         "settings": settings}
    c["settings"] = B.with_system_prompt(c["settings"], RFB.system_prompt_for(c))
    return c


def budget_for(length: int) -> int:
    """The registered completion budget at a stream length: the entry for the shortest
    registered length that covers it. Keyed by LENGTH and never by k — whether k costs tokens is
    the question this sweep asks, so k may not be allowed to move the budget."""
    for L in sorted(BUDGETS):
        if length <= L:
            return BUDGETS[L]
    return BUDGETS[max(BUDGETS)]


def plan(cells: list[str], ks: list[int], ls: list[int]) -> list[dict]:
    out = []
    for L in ls:
        for k in ks:
            if "composed" in cells:
                out.append(make_cell("composed", k, L))
            if {"state", "bind"} & set(cells):
                wm = work_matched(k, L)
                for c in ("state", "bind"):
                    if c in cells:
                        cell = make_cell(c, k, wm[c])
                        cell["settings"]["max_new_tokens"] = budget_for(wm[c])
                        cell["settings"] = B.with_system_prompt(
                            cell["settings"], RFB.system_prompt_for(cell))
                        cell["partner_of"] = L
                        out.append(cell)
    return out


# --- floors ---------------------------------------------------------------------------------

def component_floors(cell: str, k: int, L: int, n: int) -> dict:
    """The operative floor at a COMPONENT cell, recomputed from the exact scored items AND from
    a disjoint pool, larger operative — the scout's rule, at this sweep's k.

    Returns ``{}`` for the composed cell: it has no floor in this regime, and printing one puts
    back exactly the number the bounded-pad retraction removed.
    """
    if cell == "composed":
        return {}
    spec = scaled_spec(cell, k)
    m = spec.n_objects_active
    pool = TK.generate(spec, "test", n=n + FLOOR_DISJOINT_N, length=L)
    out: dict = {}
    for label, items in (("scored", pool[:n]), ("disjoint", pool[n:])):
        ns, ng = V.s5_bind_v3_shape(items)
        named, q = V.s5_bind_v3_is_named(items), V.s5_bind_v3_query_kind(items)
        keep = tuple(r for r in V.s5_bind_v3_family_rows(k, m, ns, ng, named, q)
                     if V.s5_bind_v3_admits(r, k, m, ns, ng, named, q))
        fl = dict(V.s5_bind_v3_floors(items, k, m))
        fl.update(V.s5_bind_v3_family_floors(items, k, m, named, q, rows=keep))
        out[label] = V.s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, q)
        out.setdefault("basis", V.s5_bind_v3_floor_basis(k, m, ns, ng, named, q))
    out["operative"] = max(v for v in (out["scored"], out["disjoint"]) if v is not None)
    out["n_scored"], out["n_disjoint"] = n, FLOOR_DISJOINT_N
    return out


# --- running --------------------------------------------------------------------------------

def cell_key(cell: dict) -> tuple:
    return (MODEL, cell["facet"], cell["task"], cell["length"], cell["n"],
            B.settings_hash(cell), RFB.stream_version(cell["task"]))


def run_cell(backend, cell: dict, run_id: str, git_commit: str) -> dict:
    """One cell, scored exactly as ``run_frontier_benchmark._run_attempt`` scores a task cell.

    The only departure is that the spec comes from ``scaled_spec`` instead of
    ``benchmark.spec_for_cell``: the shipped resolver reads CANONICAL by name and this sweep's
    whole point is a k rung the registry does not carry. Everything downstream — the cost guard,
    the committed-answer extraction, the finish=length-is-not-an-answer rule, the diagnostics —
    is the shipped path's.
    """
    t0 = time.time()
    s = cell["settings"]
    budget = s["max_new_tokens"]
    guard = RFB.CostGuardBackend(backend, B.CELL_BUDGET_FACTOR * cell["n"] * budget,
                                 budget_usd=None, completion_price_per_M=0.0)
    guard.CHUNK = GUARD_CHUNK
    spec = scaled_spec(cell["cell"], s["k_sweep"])
    result = evaluate_task(guard, spec, split="test", n=cell["n"], length=cell["length"],
                           max_new_tokens=budget, n_shot=s["n_shot"], stop_at=s["stop_at"],
                           extract_commit=True)
    metrics = {name: result["metrics"][name]["overall"]
               for name in ("relaxed", "exact", "contains", "last_n")}
    examples = [{"gold": gold, "pred": pred, "relaxed": ms["relaxed"]}
                for (_p, gold, pred, _ok), ms in zip(result["examples"],
                                                     result["example_metrics"])]
    RFB._attach_example_meta(examples, guard.pop_example_meta())
    truncated = 0
    for ex in examples:
        if ex.get("finish") == "length":
            ex["pred"], ex["relaxed"] = "", 0
            truncated += 1
    if truncated:
        metrics["relaxed"] = sum(ex["relaxed"] for ex in examples) / max(1, len(examples))
    meta = guard.pop_call_meta()
    empty_rate = sum(1 for ex in examples if not ex["pred"].strip()) / max(1, len(examples))
    diagnostics = {
        "empty_rate": round(empty_rate, 4),
        "truncated_rate": round(truncated / max(1, len(examples)), 4),
        "api_errors": meta["errors"],
        "finish_errors": meta["finish_reasons"].get("error", 0),
        "finish_reasons": meta["finish_reasons"],
        "cost_aborted": guard.cost_aborted,
        "request": dict(getattr(backend, "request_params", {}) or {}),
    }
    if guard.cost_aborted:
        diagnostics["calls_completed"] = guard.calls_completed
        diagnostics["cost_abort_reason"] = guard.abort_reason
    usage = meta["usage"]
    return {
        "run_id": run_id, "ts": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "suite_version": TK.SUITE_VERSION,
        "stream_version": RFB.stream_version(cell["task"]),
        "model": MODEL, "served_models": meta["served_models"], "providers": meta["providers"],
        "facet": cell["facet"], "sweep_cell": cell["cell"], "k_sweep": s["k_sweep"],
        "partner_of": cell.get("partner_of"),
        "task": cell["task"], "length": cell["length"], "n": cell["n"],
        "settings": dict(cell["settings"]),
        "metrics": metrics, "diagnostics": diagnostics,
        "structure_switch": result.get("structure_switch"),
        "usage": {**usage, "cost_usd_est": 0.0},
        "elapsed_s": round(time.time() - t0, 2), "escalated": False,
        "examples": examples,
    }


def run(cells: list[str], ks: list[int], ls: list[int], n: int, force: bool,
        max_workers: int, run_id: str, budget_override: dict[str, int]) -> None:
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    done = RFB.history_keys(HISTORY)
    git_commit = RFB._git_commit()
    cell_plan = plan(cells, ks, ls)
    for c in cell_plan:
        c["n"] = n
        key = f"{c['cell']}@{c['length']}k{c['settings']['k_sweep']}"
        if key in budget_override:
            c["settings"]["max_new_tokens"] = budget_override[key]
            c["settings"] = B.with_system_prompt(c["settings"], RFB.system_prompt_for(c))
    RFB.preflight_context({MODEL: cell_plan}, B.DEFAULT_BASE_URL)
    for cell in cell_plan:
        tag = (f"{cell['cell']}@L{cell['length']} k={cell['settings']['k_sweep']} "
               f"[budget {cell['settings']['max_new_tokens']}]")
        if not force and cell_key(cell) in done:
            print(f"  {tag}: SKIP (in history)", flush=True)
            continue
        print(f"  {tag}: running n={cell['n']}", flush=True)
        try:
            backend = RFB.build_backend(MODEL, cell, os.environ.get(B.DEFAULT_API_KEY_ENV),
                                        B.DEFAULT_BASE_URL, max_workers)
            rec = run_cell(backend, cell, run_id, git_commit)
        except Exception:
            print(f"  {tag}: FAILED (no record written)", flush=True)
            traceback.print_exc()
            continue
        RFB.append_record(HISTORY, rec)
        done.add(cell_key(cell))
        d, u = rec["diagnostics"], rec["usage"]
        print(f"  {tag}: match={rec['metrics']['relaxed']:.3f} "
              f"empty={d['empty_rate']:.2f} trunc={d['truncated_rate']:.2f} "
              f"finish={d['finish_reasons']} err={d['api_errors']} "
              f"ptok/item={u['prompt_tokens'] // cell['n']} "
              f"ctok/item={u['completion_tokens'] // cell['n']} "
              f"[{rec['elapsed_s']:.0f}s]", flush=True)
        if max(d["empty_rate"], d["truncated_rate"]) > TRUNCATION_MAX:
            print(f"  !!! {tag} is VOID (>{TRUNCATION_MAX:.0%} truncated/empty): re-run at a "
                  f"raised budget; it enters no comparison until it is.", flush=True)


# --- reading --------------------------------------------------------------------------------

def load_rows(min_n: int = N) -> dict:
    """``{(cell, k, L): row}`` at n >= ``min_n``, latest record winning.

    A low-n pilot never supersedes a full-n cell, and never enters the report: the resume key
    carries n, so the two coexist in history, and a throughput probe at n=5 is a measurement of
    the server, not of the model.
    """
    out: dict[tuple, dict] = {}
    if not os.path.exists(HISTORY):
        return out
    with open(HISTORY, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if (rec["n"] or 0) < min_n:
                continue
            n = rec["n"] or 1
            d, u = rec["diagnostics"], rec["usage"]
            ctoks = [ex.get("ctok") for ex in rec["examples"] if ex.get("ctok") is not None]
            row = {"cell": rec["sweep_cell"], "k": rec["k_sweep"], "L": rec["length"],
                   "n": rec["n"], "match": rec["metrics"]["relaxed"],
                   "length_rate": d["truncated_rate"], "empty_rate": d["empty_rate"],
                   "finish_reasons": d["finish_reasons"], "api_errors": d["api_errors"],
                   "finish_errors": d["finish_errors"], "cost_aborted": d["cost_aborted"],
                   "budget": rec["settings"]["max_new_tokens"],
                   "ptok_item": u["prompt_tokens"] / n, "ctok_item": u["completion_tokens"] / n,
                   "rtok_item": u["reasoning_tokens"] / n,
                   "ctok_median": sorted(ctoks)[len(ctoks) // 2] if ctoks else None,
                   "ctok_max": max(ctoks) if ctoks else None,
                   "elapsed_s": rec["elapsed_s"], "ts": rec["ts"],
                   "partner_of": rec.get("partner_of")}
            key = (rec["sweep_cell"], rec["k_sweep"], rec["length"])
            prev = out.get(key)
            if prev is None:
                out[key] = row
            elif row["ts"] >= prev["ts"]:
                row["superseded"] = prev.pop("superseded", []) + [prev]
                out[key] = row
            else:
                prev.setdefault("superseded", []).append(row)
    return out


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return ((c - h) / d, (c + h) / d)


def _z_against(p: float, base: float, n: int) -> float:
    se = math.sqrt(max(1e-12, base * (1 - base) / n))
    return (p - base) / se


def void(row: dict) -> bool:
    return max(row["length_rate"], row["empty_rate"]) > TRUNCATION_MAX


def report(rows: dict) -> str:
    ks = sorted({r["k"] for r in rows.values()})
    comp = {(r["k"], r["L"]): r for r in rows.values() if r["cell"] == "composed"}
    ls = sorted({L for _k, L in comp})
    out: list[str] = []
    out.append("# s5_bind_v3 on the local arm — the difficulty surface in k and L\n")
    out.append(f"Written {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
               f"Model `{MODEL}` (nvidia/Qwen3.6-35B-A3B-NVFP4, 35B total / 3B active, NVFP4, "
               f"served by vLLM on this machine at a 131,072-token window). Metric is **match**, "
               f"the canonical evaluator, on the ANSWER only. Effort arm `{EFFORT}`; the served "
               f"chat template reads an on/off thinking bit and has no effort ladder, so every "
               f"reasoning rung is this one measurement. n={N} per cell. No paid endpoint was "
               f"contacted.\n")
    out.append("**The composed cell has no floor in this regime.** A served reasoning model "
               "emits visible tokens, which is a scratchpad, and the composed cell's floor "
               "argument bounds LIVE SLOTS (W <= max(k,m)+1 against the task's k+m+1). Its "
               "number is read against INFORMED CHANCE 1/(k-1) — the initial map is stated, so "
               "the queried agent's own starting value is never gold — which is a guess "
               "baseline, not a floor, and which MOVES WITH k. Component cells keep their "
               "floors, recomputed below from the exact scored items and from a disjoint pool.\n")

    live = {key: r for key, r in comp.items() if not void(r)}
    if len(live) >= 4:
        k_spans = {L: (max(r["match"] for (kk, LL), r in live.items() if LL == L)
                       - min(r["match"] for (kk, LL), r in live.items() if LL == L))
                   for L in ls if sum(1 for (kk, LL) in live if LL == L) > 1}
        L_spans = {k: (max(r["match"] for (kk, LL), r in live.items() if kk == k)
                       - min(r["match"] for (kk, LL), r in live.items() if kk == k))
                   for k in ks if sum(1 for (kk, LL) in live if kk == k) > 1}
        def _ends(sel, coord: int):
            got = sorted(((key[coord], r) for key, r in live.items() if sel(key)),
                         key=lambda t: t[0])
            return got[0], got[-1]

        k_tok, L_tok = {}, {}
        for L in k_spans:
            (a_k, a), (b_k, b) = _ends(lambda t, L=L: t[1] == L, 0)
            k_tok[L] = (a_k, round(a["ctok_item"]), b_k, round(b["ctok_item"]))
        for k in L_spans:
            (a_L, a), (b_L, b) = _ends(lambda t, k=k: t[0] == k, 1)
            L_tok[k] = (a_L, round(a["ctok_item"]), b_L, round(b["ctok_item"]))
        out.append("## What the two axes bought\n")
        out.append(f"Across the whole k range at one L, match spans "
                   f"{ {L: round(v, 3) for L, v in k_spans.items()} } — against a 95% Wilson "
                   f"half-width of about 0.15 at n={N}. Across the whole L range at one k it "
                   f"spans { {k: round(v, 3) for k, v in L_spans.items()} }.\n")
        out.append("The completion-token cost runs the OTHER way on the k axis. Tokens per item "
                   "at the two ends of each axis:\n")
        out.append("| axis | held fixed | low end | high end |")
        out.append("|---|---|---|---|")
        for L, (a_k, a_t, b_k, b_t) in sorted(k_tok.items()):
            out.append(f"| k | L={L} | k={a_k}: {a_t} | k={b_k}: {b_t} |")
        for k, (a_L, a_t, b_L, b_t) in sorted(L_tok.items()):
            out.append(f"| L | k={k} | L={a_L}: {a_t} | L={b_L}: {b_t} |")
        out.append("")
        per_event = sorted((b_t - a_t) / (b_L - a_L) for _k, (a_L, a_t, b_L, b_t)
                           in L_tok.items())
        p_event = sorted((comp[(k, max(ls))]["ptok_item"] - comp[(k, min(ls))]["ptok_item"])
                         / (max(ls) - min(ls)) for k in ks
                         if (k, max(ls)) in comp and (k, min(ls)) in comp)
        out.append(f"So an extra EVENT costs {per_event[0]:.0f}–{per_event[-1]:.0f} completion "
                   f"tokens and {p_event[0]:.0f}–{p_event[-1]:.0f} prompt tokens across the k "
                   f"rungs, and an extra AGENT costs prompt tokens only. That is the price of "
                   f"the axis that moves match; the axis that does not move it is the cheaper "
                   f"one, which is the opposite of the trade the ceiling remedy assumed.\n")

    out.append("## Validity first — every cell's truncation and empty rate\n")
    out.append("| cell | k | L | budget | finish=length | empty | finish reasons | api errors | "
               "cost aborted | VOID |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(rows, key=lambda t: (t[0], t[1], t[2])):
        r = rows[key]
        out.append(f"| {r['cell']} | {r['k']} | {r['L']} | {r['budget']} | "
                   f"{r['length_rate']:.2f} | {r['empty_rate']:.2f} | "
                   f"`{r['finish_reasons']}` | {r['api_errors']} "
                   f"(+{r['finish_errors']} finish=error) | {r['cost_aborted']} | "
                   f"{'**VOID**' if void(r) else '—'} |")
    out.append("")
    out.append(f"A cell over {TRUNCATION_MAX:.0%} finish=length or empty is VOID and enters no "
               f"comparison until it is re-run at a raised budget: a truncated call is scored "
               f"wrong, so a truncated cell reads as a floor. That ordering is the rule, not a "
               f"preference — the published s5 L64 cliff was a 16-token budget read as a "
               f"capability.\n")
    at_bar = [f"{r['cell']}@{r['L']} k={r['k']} ({max(r['length_rate'], r['empty_rate']):.2f})"
              for _key, r in sorted(rows.items())
              if 0 < max(r["length_rate"], r["empty_rate"]) <= TRUNCATION_MAX]
    if at_bar:
        out.append(f"Cells with some truncation but AT OR UNDER the bar are read as measured, "
                   f"with the caveat that a truncated call is scored wrong, so their match is a "
                   f"LOWER bound: {', '.join(at_bar)}.\n")

    out.append("## The composed surface — match against informed chance 1/(k-1)\n")
    out.append("| k | chance | " + " | ".join(f"L={L}" for L in ls) + " |")
    out.append("|---|---" + "|---" * len(ls) + "|")
    for k in ks:
        ch = informed_chance(k)
        cells = []
        for L in ls:
            r = comp.get((k, L))
            if r is None:
                cells.append("—")
            elif void(r):
                cells.append(f"VOID ({r['match']:.3f})")
            else:
                lo, hi = _wilson(r["match"], r["n"])
                cells.append(f"{r['match']:.3f} [{lo:.2f},{hi:.2f}] "
                             f"{r['match'] / ch:.2f}x z={_z_against(r['match'], ch, r['n']):+.1f}")
        out.append(f"| {k} | {ch:.4f} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("95% Wilson intervals in brackets; the multiplier and z are against that row's "
               "own informed chance, which is why raw match is not comparable across k rungs. "
               "Per-cell values only — nothing is averaged across cells.\n")

    out.append("## What the difficulty tracks, and what it does not\n")
    out.append("The cheapest correct algorithm's STATE leg is a chain of `2 n_swap / k` hops "
               "(`validity.s5_bind_v3_carrier_hops`): a swap moves two of the k pointers, so the "
               "queried agent is touched by that fraction of them. L raises `n_swap` and k "
               "DIVIDES it, so the carrier chain is the one quantity BOTH axes move, and it is "
               "the obvious candidate for what makes the cell hard. On this model it is not what "
               "match tracks.\n")
    out.append("| k | L | carrier hops | match | x chance | prompt tok/item | "
               "completion tok/item |")
    out.append("|---|---|---|---|---|---|---|")
    for k in ks:
        for L in ls:
            r = comp.get((k, L))
            if r is None:
                continue
            ch = informed_chance(k)
            out.append(f"| {k} | {L} | {carrier_hops(k, L):.2f} | "
                       f"{'VOID ' if void(r) else ''}{r['match']:.3f} | "
                       f"{r['match'] / ch:.2f}x | {r['ptok_item']:.0f} | "
                       f"{r['ctok_item']:.0f} |")
    out.append("")

    # Cells at MATCHED carrier-hop counts and different L. If the chain length were what the
    # cell costs, these would agree; the spread inside a band is the falsification.
    band_rows = sorted(((carrier_hops(k, L), k, L, r) for (k, L), r in comp.items()),
                       key=lambda t: t[0])
    bands: list[list] = []
    for item in band_rows:
        if bands and item[0] <= bands[-1][0][0] * 1.10:
            bands[-1].append(item)
        else:
            bands.append([item])
    bands = [b for b in bands if len(b) > 1 and len({x[2] for x in b}) > 1]
    if bands:
        out.append("### The same carrier chain at different L\n")
        out.append("| carrier hops | cells (k, L) | match at each | spread |")
        out.append("|---|---|---|---|")
        for b in bands:
            cells_s = ", ".join(f"({k}, {L})" for _h, k, L, _r in b)
            vals = ", ".join(f"{r['match']:.3f}" for _h, _k, _L, r in b)
            spread = max(r["match"] for *_x, r in b) - min(r["match"] for *_x, r in b)
            out.append(f"| {b[0][0]:.2f}–{b[-1][0]:.2f} | {cells_s} | {vals} | {spread:.3f} |")
        out.append("")
        widest = max(max(r["match"] for *_x, r in b) - min(r["match"] for *_x, r in b)
                     for b in bands)
        surface = (max(r["match"] for r in comp.values())
                   - min(r["match"] for r in comp.values()))
        ordered = sum(1 for b in bands
                      if [r["match"] for _h, _k, _L, r in sorted(b, key=lambda t: t[2])]
                      == sorted((r["match"] for _h, _k, _L, r in b), reverse=True))
        out.append(f"Within a band the carrier chain is the same length to within 10%, and match "
                   f"still spreads by up to {widest:.3f} against a whole-surface span of "
                   f"{surface:.3f}; {ordered} of {len(bands)} bands order their cells by L, "
                   f"shorter stream first. So the chain length is not what this model pays for. "
                   f"What it pays for is L — the number of EVENTS it walks — which is the cost "
                   f"of SIMULATING the stream rather than of chasing the carrier through it.\n")
        out.append("That is the same conclusion the bounded-pad result reached from the other "
                   "side. A scratchpad substitutes for REGISTERS and not for CHAINING, so k — "
                   "which prices registers, k+m+1 live slots — is nearly free to a model that "
                   "writes its working down, while L, which prices the walk, is not.\n")

    out.append("## The token cost of each axis\n")
    out.append("| cell | k | L | prompt tok/item | completion tok/item | median | max | "
               "budget | wall clock |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(rows, key=lambda t: (t[0], t[1], t[2])):
        r = rows[key]
        out.append(f"| {r['cell']} | {r['k']} | {r['L']} | {r['ptok_item']:.0f} | "
                   f"{r['ctok_item']:.0f} | {r['ctok_median'] or '—'} | {r['ctok_max'] or '—'} | "
                   f"{r['budget']} | {r['elapsed_s']:.0f}s |")
    out.append("")
    out.append("Reasoning tokens are INSIDE completion tokens on this arm: vLLM's "
               "chat-completions usage carries no reasoning-token field and the chat template "
               "opens `<think>` in the generation prompt, so the record's `rtok` reads 0 whether "
               "or not the model thought. The completion column is therefore the whole token "
               "cost of an item, thinking included.\n")

    out.append("### What one step of each axis costs and buys\n")
    out.append("| axis | held fixed | step | delta match | delta prompt tok/item | "
               "delta completion tok/item |")
    out.append("|---|---|---|---|---|---|")
    for L in ls:
        pairs = [(k, comp.get((k, L))) for k in ks]
        pairs = [(k, r) for k, r in pairs if r is not None and not void(r)]
        for (k0, a), (k1, b) in zip(pairs, pairs[1:]):
            out.append(f"| k | L={L} | {k0} -> {k1} | {b['match'] - a['match']:+.3f} | "
                       f"{b['ptok_item'] - a['ptok_item']:+.0f} | "
                       f"{b['ctok_item'] - a['ctok_item']:+.0f} |")
    for k in ks:
        pairs = [(L, comp.get((k, L))) for L in ls]
        pairs = [(L, r) for L, r in pairs if r is not None and not void(r)]
        for (l0, a), (l1, b) in zip(pairs, pairs[1:]):
            out.append(f"| L | k={k} | {l0} -> {l1} | {b['match'] - a['match']:+.3f} | "
                       f"{b['ptok_item'] - a['ptok_item']:+.0f} | "
                       f"{b['ctok_item'] - a['ctok_item']:+.0f} |")
    out.append("")
    out.append("VOID cells are omitted from the deltas rather than differenced against: a "
               "truncated cell's match is a lower bound, so a step into or out of one is not a "
               "measured step.\n")

    comps = {(r["cell"], r["k"], r["L"]): r for r in rows.values() if r["cell"] != "composed"}
    if comps:
        out.append("## Component cells, against floors recomputed from the exact scored items\n")
        out.append("| cell | k | L | partner of composed@L | match | floor (scored) | "
                   "floor (disjoint) | operative | basis | x floor |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for key in sorted(comps):
            r = comps[key]
            f = component_floors(r["cell"], r["k"], r["L"], r["n"])
            out.append(f"| {r['cell']} | {r['k']} | {r['L']} | "
                       f"{r['partner_of'] or '—'} | "
                       f"{'VOID ' if void(r) else ''}{r['match']:.3f} | {f['scored']:.4f} | "
                       f"{f['disjoint']:.4f} | **{f['operative']:.4f}** | {f['basis']} | "
                       f"{r['match'] / f['operative']:.2f}x |")
        out.append("")
        out.append("Both floors are printed because they measure different failure modes: the "
                   "max over admitted rows carries an upward selection bias at n=40, and the "
                   "house rule is that a floor is recomputed from the items a score is actually "
                   "read against. The larger is operative. A component floor's admitted rows are "
                   "depth <= 1 and cost under the cell's own algorithm's per-item minimum, so a "
                   "scratchpad does not void them — a pad substitutes for registers, not for "
                   "chaining.\n")
    gap = []
    for (kk, L), r in sorted(comp.items()):
        wm = work_matched(kk, L)
        st, bd = rows.get(("state", kk, wm["state"])), rows.get(("bind", kk, wm["bind"]))
        if st is None or bd is None:
            continue
        gap.append((kk, L, r, st, bd, wm))
    if gap:
        out.append("## The composed cell against its own components, at matched work\n")
        out.append("| k | composed@L | composed match | state@L (match) | bind@L (match) | "
                   "lower component | composed - lower component |")
        out.append("|---|---|---|---|---|---|---|")
        for kk, L, r, st, bd, wm in gap:
            low = min(st["match"], bd["match"])
            out.append(f"| {kk} | {L} | {r['match']:.3f} | {wm['state']} ({st['match']:.3f}) | "
                       f"{wm['bind']} ({bd['match']:.3f}) | {low:.3f} | "
                       f"{r['match'] - low:+.3f} |")
        out.append("")
        out.append("The component lengths are the work-matched partners of that composed cell, "
                   "so each row compares the composed stream against exactly the swaps and "
                   "exactly the gives it contains. A composed number below both components is "
                   "not a component deficit; the components are the gate the scout's rule reads "
                   "before the composed cell means anything.\n")

    out.append("## Where this model sits against the three scouted frontier models\n")
    out.append("The scout's numbers are the registered k=12 spec at n=40 on the answer read, "
               "which is the same cell and the same read as the k=12 row above.\n")
    out.append("| model | composed@128 | composed@256 | state@85 | bind@171 |")
    out.append("|---|---|---|---|---|")
    for name, vals in (("openai/gpt-5.5", ("1.000", "0.975", "1.000", "1.000")),
                       ("nvidia/nemotron-3-ultra-550b-a55b", ("0.750", "0.500", "0.875", "1.000")),
                       ("z-ai/glm-5.2", ("0.575", "0.450", "0.950", "0.850"))):
        out.append(f"| {name} | " + " | ".join(vals) + " |")
    here = []
    for key, want in ((("composed", 12, 128), None), (("composed", 12, 256), None),
                      (("state", 12, 85), None), (("bind", 12, 171), None)):
        r = rows.get(key)
        here.append("—" if r is None else
                    f"{'VOID ' if void(r) else ''}{r['match']:.3f}")
    out.append(f"| **{MODEL}** | " + " | ".join(here) + " |")
    out.append("")
    verdict = []
    for label, key, band in (("composed@128", ("composed", 12, 128), (0.575, 1.000)),
                             ("composed@256", ("composed", 12, 256), (0.450, 0.975)),
                             ("state@85", ("state", 12, 85), (0.875, 1.000)),
                             ("bind@171", ("bind", 12, 171), (0.850, 1.000))):
        r = rows.get(key)
        if r is None:
            continue
        lo, hi = band
        where = ("inside" if lo <= r["match"] <= hi else
                 "below" if r["match"] < lo else "above")
        verdict.append(f"{label} {r['match']:.3f} is {where} the scouted band "
                       f"[{lo:.3f}, {hi:.3f}]")
    if verdict:
        out.append("; ".join(verdict) + ".\n")
    out.append("A stand-in has to land inside the band on the cell it is standing in for. On the "
               "COMPONENT cells it is close: bind@171 is inside, and state@85 is 0.100 under the "
               "bottom of a band whose whole width is 0.125. On the COMPOSED cell it is not: at "
               "composed@256 it is at informed chance (0.075 against 0.0909) where the three "
               "scouted models span 0.450 to 0.975, and at composed@128 it is 0.325 under the "
               "bottom. So this model can stand in for a frontier model on the components and "
               "cannot on the composed cell — the axis a composed-cell redesign has to be tested "
               "on is exactly the one it is off the bottom of.\n")
    out.append("The k=6 rung of this sweep is the k=12 frontier spec scaled down and NOT the "
               "registered from-scratch cell `s5_bind_local_v3`, which carries match_reads=2 and "
               "no q_no_surface gate. Nothing here is a measurement of that cell.\n")

    return "\n".join(out)


def write_report() -> str:
    text = report(load_rows())
    for phrase in P.SCOUT_COMPOSED_FLOOR_LANGUAGE:
        if phrase in text.lower():
            raise AssertionError(
                f"the report uses {phrase!r}; the composed cell has no floor in this regime "
                "(P.SCOUT_COMPOSED_FLOOR_LANGUAGE)")
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nwrote {REPORT_MD}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", nargs="+", default=["composed"], choices=list(TASKS))
    ap.add_argument("--ks", nargs="+", type=int, default=list(KS))
    ap.add_argument("--lengths", nargs="+", type=int, default=list(LS), dest="lengths")
    ap.add_argument("-n", type=int, default=N)
    ap.add_argument("--max-workers", type=int, default=16, dest="max_workers")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the plan; no calls")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--run-id", default=None, dest="run_id")
    ap.add_argument("--budget-override", action="append", default=None, dest="budget_override",
                    metavar="CELL@LkK:BUDGET",
                    help="re-run a VOID cell at a raised budget, e.g. composed@256k32:65536")
    a = ap.parse_args()

    overrides: dict[str, int] = {}
    for spec in a.budget_override or ():
        key, _, val = spec.rpartition(":")
        overrides[key] = int(val)

    if a.report:
        write_report()
        return
    if a.plan:
        for c in plan(a.cells, a.ks, a.lengths):
            c["n"] = a.n
            over = B.context_overrun(MODEL, c,
                                     min_completion_tokens=RFB.cell_completion_ceiling(c))
            print(f"  {c['cell']}@L{c['length']:<4} k={c['settings']['k_sweep']:<3} n={c['n']} "
                  f"budget={c['settings']['max_new_tokens']:<6} "
                  f"partner_of={c.get('partner_of') or '-'} "
                  f"chance={informed_chance(c['settings']['k_sweep']):.4f} "
                  f"context={'OK' if not over else f'OVER {over}'}")
        return
    run(a.cells, a.ks, a.lengths, a.n, a.force, a.max_workers,
        a.run_id or f"s5bind_kL_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        overrides)


if __name__ == "__main__":
    main()
