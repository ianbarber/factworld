"""THE k AXIS OF s5_bind_v3, measured on a model that has HEADROOM on the composed cell.

WHY THIS EXISTS. The paid scout returned STOP_CEILING: gpt-5.5 read 1.000/0.975 on
composed@128/@256, so the top of the roster is saturated and the rule's own remedy is
"raise k or L". The local Qwen sweep then found the k axis flat (match spans 0.05-0.15
across k in {6,12,24,32} at fixed L, inside the n=40 Wilson half-width) while the L axis
spanned 0.325-0.625. But that flatness was read at three of four lengths where Qwen sits
AT OR BELOW informed chance: an axis cannot be shown to buy nothing where there is nothing
left to lose. This sweep re-asks the question on steed's DeepSeek V4, at the length where
THIS model has the most headroom, chosen from its own placement numbers and not in advance.

THE ARM. ``steed/deepseek-v4-flash`` — a DGX Spark GB10 the owner runs, reached over the
tailnet. Prices are 0.0 and no paid endpoint is contacted; the local GPU is untouched.
The endpoint SERIALIZES (one KV session, ~16.5 completion tok/s warm), so a cell's cost is
its completion tokens divided by that rate, and the wall clock — not the 262,144-token
window — is what bounds this grid. Budgets are therefore set from THIS model's measured
trace length (results/probes/steed_rate_20260803.json), not from the local Qwen's, and n
is chosen per cell against the same arithmetic.

THE READ IS THE ANSWER ONLY, and the composed cell has NO FLOOR in this regime. The model
emits its working as plain content (no ``<think>`` delimiter, so ``rtok`` reads 0 and the
completion column is the whole token cost), which IS a scratchpad, so the composed cell's
floor argument — a bound on LIVE SLOTS, W <= max(k,m)+1 against the task's k+m+1 — does
not hold. The composed number is read against INFORMED CHANCE 1/(k-1), a guess baseline
that MOVES WITH k, which is why raw match is not comparable across k rungs and every cell
carries its ratio and z. COMPONENT cells keep their floors, recomputed from the exact
scored items and from a disjoint pool, larger operative.

VALIDITY FIRST. A cell over 10% finish=length or empty is VOID and enters no comparison
until it is re-run at a raised budget: a truncated call is scored wrong, so a truncated
cell reads as a floor. That is how the published s5 L64 cliff was manufactured.

Examples:
    .venv-api/bin/python scripts/serve_steed_model.py status
    .venv-api/bin/python scripts/sweep_s5bind_v3_steed_kL_20260803.py --plan
    .venv-api/bin/python scripts/sweep_s5bind_v3_steed_kL_20260803.py --stage place
    .venv-api/bin/python scripts/sweep_s5bind_v3_steed_kL_20260803.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocol_s5bind_v3_three_cell_20260731 as P            # noqa: E402
import run_frontier_benchmark as RFB                          # noqa: E402
import sweep_s5bind_v3_local_kL_20260802 as S                 # noqa: E402
from factworld import benchmark as B                          # noqa: E402
from factworld import tasks as TK                             # noqa: E402
from factworld.runner import evaluate_task                    # noqa: E402

MODEL = "steed/deepseek-v4-flash"
FACET = "s5_bind_v3_kl_sweep"
HISTORY = os.path.join(REPO, "results", "steed", "history_s5bind_v3_kL_20260803.jsonl")
REPORT_MD = os.path.join(REPO, "results", "20260803_s5bind_v3_steed_kL_sweep.md")

TASKS = S.TASKS
EFFORT = "high"          # ds4 collapses minimal..xhigh to one internal level; this is that arm
TRUNCATION_MAX = P.SCOUT_TRUNCATION_MAX
GUARD_CHUNK = 8

# The scouted band this model is placed against — the registered k=12 cells at n=40,
# answer read, from the paid scout that returned STOP_CEILING.
SCOUT = {
    "openai/gpt-5.5": {("composed", 128): 1.000, ("composed", 256): 0.975,
                       ("state", 85): 1.000, ("bind", 171): 1.000},
    "nvidia/nemotron-3-ultra-550b-a55b": {("composed", 128): 0.750, ("composed", 256): 0.500,
                                          ("state", 85): 0.875, ("bind", 171): 1.000},
    "z-ai/glm-5.2": {("composed", 128): 0.575, ("composed", 256): 0.450,
                     ("state", 85): 0.950, ("bind", 171): 0.850},
}
BANDS = {("composed", 128): (0.575, 1.000), ("composed", 256): (0.450, 0.975),
         ("state", 85): (0.875, 1.000), ("bind", 171): (0.850, 1.000)}

# PER-LENGTH COMPLETION BUDGETS, in completion tokens, set from THIS model's measured trace
# length. Keyed by the SHORTEST registered length that covers the cell's own length and never
# by k: whether k costs tokens is the question, so k may not be allowed to move the budget.
# The anchor is measured, not assumed — state@L43 k=12 spent 5,411 completion tokens and
# finished on `stop` (results/probes/steed_rate_20260803.json), about half what the local Qwen
# spends at the same cell — and these sit at roughly 3x that per-event rate.
# A budget is also a DURATION on a serialized endpoint: at 15.7 tok/s a 32,768-token budget is
# 35 minutes for a single item that runs to the cap. But a budget UNDER the model's trace
# length is worse than slow, it is a whole cell scored as a floor, and a VOID cell costs the
# hours it took plus the hours of the re-run. So these are generous by design: the cost of the
# headroom is only paid by the items that actually use it.
BUDGETS = {24: 16384, 48: 24576, 96: 32768, 160: 49152, 320: 65536}

# Completion tokens per EVENT, measured on this arm, used only to price a plan before it is
# run (--plan) and never to score anything. state@L43 spent 5,411 (126/event) and finished on
# `stop`; two bind@L42 items spent 6,675 (159/event) and over 12,288. The spread is the point:
# this model's trace length is both long and heavy-tailed, so a budget near the mean turns a
# cell VOID and a cell that VOIDs costs its own hours plus the re-run's.
TOK_PER_EVENT = 200.0
RATE_TOK_PER_S = 15.8      # measured warm, sustained to 9k of generated context


def budget_for(length: int) -> int:
    for L in sorted(BUDGETS):
        if length <= L:
            return BUDGETS[L]
    return BUDGETS[max(BUDGETS)]


def make_cell(cell: str, k: int, L: int, n: int) -> dict:
    settings = B._settings(EFFORT, max_new_tokens=budget_for(L))
    settings["k_sweep"] = k
    c = {"facet": FACET, "cell": cell, "task": TASKS[cell], "length": L, "n": n,
         "settings": settings}
    c["settings"] = B.with_system_prompt(c["settings"], RFB.system_prompt_for(c))
    return c


def composed_cell(k: int, L: int, n: int) -> dict:
    return make_cell("composed", k, L, n)


def component_cells(k: int, L: int, n: int, which=("state", "bind")) -> list[dict]:
    """The work-matched component partners of composed@(k, L), read off the GENERATED stream."""
    wm = S.work_matched(k, L)
    out = []
    for c in which:
        cell = make_cell(c, k, wm[c], n)
        cell["partner_of"] = L
        out.append(cell)
    return out


# --- running --------------------------------------------------------------------------------

def cell_key(cell: dict) -> tuple:
    return (MODEL, cell["facet"], cell["task"], cell["length"], cell["n"],
            B.settings_hash(cell), RFB.stream_version(cell["task"]))


def run_cell(backend, cell: dict, run_id: str, git_commit: str) -> dict:
    """One cell, scored exactly as the shipped runner scores a task cell.

    The only departure is the spec: ``S.scaled_spec`` instead of ``benchmark.spec_for_cell``,
    because the registry resolves CANONICAL by name and this sweep's point is a k rung the
    registry does not carry. Everything downstream — the cost guard, the committed-answer
    extraction, the finish=length-is-not-an-answer rule, the diagnostics — is the shipped path.
    """
    t0 = time.time()
    s = cell["settings"]
    budget = s["max_new_tokens"]
    guard = RFB.CostGuardBackend(backend, B.CELL_BUDGET_FACTOR * cell["n"] * budget,
                                 budget_usd=None, completion_price_per_M=0.0)
    guard.CHUNK = GUARD_CHUNK
    spec = S.scaled_spec(cell["cell"], s["k_sweep"])
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


def run(cell_plan: list[dict], force: bool, run_id: str) -> None:
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    done = RFB.history_keys(HISTORY)
    git_commit = RFB._git_commit()
    RFB.preflight_context({MODEL: cell_plan}, B.DEFAULT_BASE_URL)
    for cell in cell_plan:
        tag = (f"{cell['cell']}@L{cell['length']} k={cell['settings']['k_sweep']} "
               f"n={cell['n']} [budget {cell['settings']['max_new_tokens']}]")
        if not force and cell_key(cell) in done:
            print(f"  {tag}: SKIP (in history)", flush=True)
            continue
        print(f"  {tag}: running  ({datetime.now(timezone.utc).isoformat(timespec='seconds')})",
              flush=True)
        try:
            backend = RFB.build_backend(MODEL, cell, os.environ.get(B.DEFAULT_API_KEY_ENV),
                                        B.DEFAULT_BASE_URL, 1)
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

def load_rows(min_n: int = 1) -> dict:
    """``{(cell, k, L): row}``, latest record winning, and never a lower-n record over a
    higher-n one: the resume key carries n, so a pilot and a full cell coexist in history."""
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
                   "contains": rec["metrics"]["contains"],
                   "length_rate": d["truncated_rate"], "empty_rate": d["empty_rate"],
                   "finish_reasons": d["finish_reasons"], "api_errors": d["api_errors"],
                   "finish_errors": d["finish_errors"], "cost_aborted": d["cost_aborted"],
                   "budget": rec["settings"]["max_new_tokens"],
                   "ptok_item": u["prompt_tokens"] / n, "ctok_item": u["completion_tokens"] / n,
                   "ctok_median": sorted(ctoks)[len(ctoks) // 2] if ctoks else None,
                   "ctok_max": max(ctoks) if ctoks else None,
                   # the per-item spread, not just its summary: whether this endpoint returns a
                   # long generation is answered by the items, and a median hides exactly that
                   "ctoks": sorted(ctoks),
                   "elapsed_s": rec["elapsed_s"], "ts": rec["ts"],
                   "partner_of": rec.get("partner_of")}
            key = (rec["sweep_cell"], rec["k_sweep"], rec["length"])
            prev = out.get(key)
            if prev is None or (row["n"], row["ts"]) >= (prev["n"], prev["ts"]):
                out[key] = row
    return out


def void(row: dict) -> bool:
    return max(row["length_rate"], row["empty_rate"]) > TRUNCATION_MAX


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default=None,
                    choices=["place", "k", "L", "custom"],
                    help="place: the four registered k=12 cells. k: the k rung sweep. "
                         "L: the L axis at the sweep k.")
    ap.add_argument("--cells", nargs="+", default=["composed"], choices=list(TASKS))
    ap.add_argument("--ks", nargs="+", type=int, default=[12])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128], dest="lengths")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the plan; no calls")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--run-id", default=None, dest="run_id")
    ap.add_argument("--budget-override", action="append", default=None, dest="budget_override",
                    metavar="CELL@LkK:BUDGET")
    ap.add_argument("--base-url", default=None, dest="base_url",
                    help="reach the SAME server on another path, e.g. an ssh -L forward to its "
                         "own port. The registry's HTTPS URL goes through a reverse proxy that "
                         "drops generations past ~700 s, which the backend retries five times "
                         "and then records as an empty prediction. The URL actually used is "
                         "recorded in diagnostics.request, so a record always says which path "
                         "produced it.")
    a = ap.parse_args()

    if a.base_url:
        # Point the registry entry at the other path for this process only. Everything else
        # about the arm — model name, effort map, worker cap, context policy, prices — is
        # unchanged, because it is the same server.
        B.MODELS[MODEL] = {**B.MODELS[MODEL], "base_url": a.base_url}
        print(f"base_url override: {a.base_url}", flush=True)

    if a.report:
        import report_s5bind_v3_steed_20260803 as R
        R.write_report()
        return

    cell_plan: list[dict] = []
    if a.stage == "place":
        # The four registered k=12 cells the scout read, in the order they decide things:
        # the composed cell first (it is what places the model in the band), then the
        # components (the gate a composed number is read behind).
        cell_plan = [composed_cell(12, 128, a.n),
                     *component_cells(12, 128, a.n, ("bind",)),
                     *component_cells(12, 256, a.n, ("state", "bind")),
                     composed_cell(12, 256, a.n)]
    elif a.stage == "k":
        for k in a.ks:
            for L in a.lengths:
                cell_plan.append(composed_cell(k, L, a.n))
    elif a.stage == "L":
        for L in a.lengths:
            for k in a.ks:
                cell_plan.append(composed_cell(k, L, a.n))
    else:
        for L in a.lengths:
            for k in a.ks:
                if "composed" in a.cells:
                    cell_plan.append(composed_cell(k, L, a.n))
                comps = [c for c in ("state", "bind") if c in a.cells]
                if comps:
                    cell_plan.extend(component_cells(k, L, a.n, tuple(comps)))

    overrides: dict[str, int] = {}
    for spec in a.budget_override or ():
        key, _, val = spec.rpartition(":")
        overrides[key] = int(val)
    for c in cell_plan:
        key = f"{c['cell']}@{c['length']}k{c['settings']['k_sweep']}"
        if key in overrides:
            c["settings"]["max_new_tokens"] = overrides[key]
            c["settings"] = B.with_system_prompt(c["settings"], RFB.system_prompt_for(c))

    if a.plan:
        cap_ctok = exp_ctok = 0
        for c in cell_plan:
            over = B.context_overrun(MODEL, c,
                                     min_completion_tokens=RFB.cell_completion_ceiling(c))
            cap = c["n"] * c["settings"]["max_new_tokens"]
            exp = c["n"] * min(c["settings"]["max_new_tokens"],
                               c["length"] * TOK_PER_EVENT)
            cap_ctok += cap
            exp_ctok += exp
            print(f"  {c['cell']}@L{c['length']:<4} k={c['settings']['k_sweep']:<3} "
                  f"n={c['n']} budget={c['settings']['max_new_tokens']:<6} "
                  f"partner_of={c.get('partner_of') or '-'} "
                  f"chance={S.informed_chance(c['settings']['k_sweep']):.4f} "
                  f"expect={exp / RATE_TOK_PER_S / 3600:.1f}h "
                  f"cap={cap / RATE_TOK_PER_S / 3600:.1f}h "
                  f"context={'OK' if not over else f'OVER {over}'}")
        print(f"  {len(cell_plan)} cells: expect {exp_ctok / RATE_TOK_PER_S / 3600:.1f} h at "
              f"{TOK_PER_EVENT:.0f} completion tokens/event and {RATE_TOK_PER_S} tok/s; "
              f"{cap_ctok / RATE_TOK_PER_S / 3600:.1f} h if every item ran to its cap")
        return

    run(cell_plan, a.force,
        a.run_id or f"s5bind_steed_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
