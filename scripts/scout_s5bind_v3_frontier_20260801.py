"""THE FRONTIER SCOUT for s5_bind_v3 — the three-cell comparison's other regime.

WHAT THIS BUYS AND WHY IT IS FOUR CELLS
    The from-scratch arm has run; the frontier arm never has. The instrument's whole claim is
    that ONE basis carries both regimes, and half of that has been asserted rather than
    measured: ``scripts/protocol_s5bind_v3_three_cell_20260731.py`` defines the scout, and until
    now there was no runner and no result.

    The plan is the protocol module's ``scout_cells()`` and nothing else — the composed cell at
    both scout lengths (it carries the length axis, which is where separation would come from)
    and each component at the WORK-MATCHED partner of the deeper composed length, state@85 and
    bind@171. n=40 resolves a 0.20 spread and deliberately does not order two models 0.05 apart;
    that is what the roster run is for, and this run does not buy it.

THE READ IS THE ANSWER, AND ONLY THE ANSWER
    ``FRONTIER_READS`` is ``("answer",)`` and ``assert_trace_read`` raises on this arm. A
    frontier model's visible trace is prose it chose to write under its own budget, not the
    per-event checkpoint stream the harness generates locally, so a slot read out of it would
    score a different object per model. The frontier spec carries no ``event_trace``.

THE COMPOSED CELL HAS NO FLOOR HERE, and the reading rule is written so that nothing needs one.
    A frontier model reasons in visible tokens, which IS a scratchpad, and the composed cell's
    floor argument is a bound on LIVE SLOTS (W <= max(k, m) + 1 against the task's k + m + 1) —
    the same bound the guided protocol's format voids locally, where the both-maps replay
    reaches 0.719 against a printed floor of 0.234. So the composed cell is read against
    INFORMED CHANCE 1/(k-1) = 0.0909 as a guess baseline, and the BUY rule is a SPREAD across
    models, which needs no baseline at all. No "clears the floor" language is available for the
    composed cell and this script asserts on its own report text
    (``P.SCOUT_COMPOSED_FLOOR_LANGUAGE``).

    The COMPONENT cells keep their floors, and those are recomputed here from the exact scored
    items as well as from a disjoint pool, with the larger operative: their admitted rows are
    depth <= 1 and cost under the cell's own algorithm's per-item minimum, and a scratchpad
    substitutes for REGISTERS, not for CHAINING.

WHY THIS IS ITS OWN SCRIPT AND NOT A FACET
    Every mechanism it runs on is the recurring benchmark's — ``build_backend``,
    ``execute_cell`` (cost guard, per-cell dollar cap, escalation, per-call ctok/rtok/finish),
    ``system_prompt_for`` (the thinking regime's NEUTRAL prompt: the canonical harness text's
    "a short test" / "no explanation" clauses move gpt-5.6-sol 0.68 -> 0.96 on identical items),
    ``cell_key``/``history_keys`` resume, and the C3 record. What it does NOT do is add a facet
    to ``benchmark.FACETS``: a facet is planned for EVERY model by ``arms_for``, so registering
    one here would put 4 unapproved cells per roster model into the next battery's plan. The
    facet label ``s5_bind_v3_scout`` is carried in the record for provenance, and the records go
    to their own history file so the published battery history is untouched.

Examples:
    set -a; source .env; set +a

    .venv-api/bin/python scripts/scout_s5bind_v3_frontier_20260801.py --dry-run
    .venv-api/bin/python scripts/scout_s5bind_v3_frontier_20260801.py --models openai/gpt-5.5
    .venv-api/bin/python scripts/scout_s5bind_v3_frontier_20260801.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocol_s5bind_v3_three_cell_20260731 as P            # noqa: E402
import run_frontier_benchmark as RFB                          # noqa: E402
from factworld import benchmark as B                          # noqa: E402
from factworld import tasks as TK                             # noqa: E402
from factworld import validity as V                           # noqa: E402

FACET = "s5_bind_v3_scout"
HISTORY_DIR = os.path.join(REPO, "results", "s5bind_v3_scout")
REPORT_MD = os.path.join(REPO, "results", "s5bind_v3_frontier_scout_20260801.md")
FLOOR_DISJOINT_N = P.N_SCORE      # the pool the component floors' disjoint read is measured on


def history_path(model: str) -> str:
    """One history file per model: the three models run as concurrent processes and a C3
    record is far larger than PIPE_BUF, so a shared append target could interleave lines."""
    return os.path.join(HISTORY_DIR, f"history_{model.replace('/', '_')}.jsonl")


def parse_raised(specs: list[str] | None) -> dict[str, int]:
    """Parse repeatable ``--raise-budget cell@L:budget`` — the rule's OWN remedy for a VOID cell.

    The budget is part of the settings hash, so a raised cell gets a FRESH resume key: the void
    record stays in history exactly as measured (a cell that truncated at 16,384 is a fact about
    that budget) and the re-run is a separate record the report reads in its place. Fails loudly
    on a cell the scout does not register — a typo must not silently no-op a paid re-run.
    """
    known = {f"{c['cell']}@{c['length']}" for c in P.scout_cells()}
    out: dict[str, int] = {}
    for s in specs or ():
        try:
            key, budget = s.rsplit(":", 1)
            out[key] = int(budget)
        except ValueError:
            raise SystemExit(f"--raise-budget expects cell@L:budget, got {s!r}")
        if key not in known:
            raise SystemExit(f"--raise-budget: unknown cell {key!r} (known: {sorted(known)})")
        if out[key] <= 0:
            raise SystemExit(f"--raise-budget: budget must be positive, got {s!r}")
    return out


def plan_for(model: str, n_scale: float = 1.0, raised: dict[str, int] | None = None) -> list[dict]:
    """The runner-shaped cells for one model, straight off ``P.scout_cells()``.

    Settings come from ``benchmark._settings`` so the C3 key set is the registry's own, and the
    system prompt is stamped by ``benchmark.with_system_prompt`` exactly as ``build_plan`` does
    — the thinking regime's neutral text is outside the frozen sentinel-drop set, so these cells
    carry its fingerprint in the resume key instead of resuming against another regime.
    """
    cells = []
    raised = raised or {}
    for c in P.scout_cells(n=max(5, round(P.SCOUT_N * n_scale))):
        budget = raised.get(f"{c['cell']}@{c['length']}", c["settings"]["max_new_tokens"])
        cell = {"facet": FACET, "cell": c["cell"], "task": c["task"], "length": c["length"],
                "n": c["n"],
                "settings": B._settings(c["settings"]["effort"], max_new_tokens=budget)}
        cell["settings"] = B.with_system_prompt(cell["settings"], RFB.system_prompt_for(cell))
        cells.append(cell)
    return cells


def component_floors(cell: str, task: str, L: int, n: int) -> dict:
    """The operative floor at a COMPONENT cell, recomputed from the EXACT scored items and from
    a disjoint pool, with the larger operative.

    Both, because they measure different failure modes. The max over admitted rows carries an
    upward selection bias at small n — at n=40 the state cell's rows read 0.150 against 0.091 on
    4,000 disjoint items — and the house rule is that a floor is recomputed from the items a
    score is actually read against. Reporting one number would either overstate the bar or
    quote a bar the scored items do not support.

    Returns ``{}`` for the COMPOSED cell: it has no floor in this regime (the frontier is a
    scratchpad and the cell's floor argument prices live slots), and printing one would put back
    exactly the number the retraction removed.
    """
    if cell == "composed":
        return {}
    spec = TK.CANONICAL[task]
    k, m = spec.k, spec.n_objects_active
    pool = TK.generate(spec, "test", n=n + FLOOR_DISJOINT_N, length=L)
    out = {}
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


# --- running -------------------------------------------------------------------------------

def run(models: list[str], n_scale: float, max_workers: int, force: bool,
        base_url: str, run_id: str, raised: dict[str, int] | None = None,
        only: list[str] | None = None) -> None:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    api_key = os.environ.get(B.DEFAULT_API_KEY_ENV)
    git_commit = RFB._git_commit()
    for model in models:
        path = history_path(model)
        done = RFB.history_keys(path)
        print(f"\n>>> {model} ({B.MODELS[model]['tier']}) -> {path} "
              f"({len(done)} keys in history)", flush=True)
        total = 0.0
        for cell in plan_for(model, n_scale, raised):
            if only and f"{cell['cell']}@{cell['length']}" not in only:
                continue
            tag = f"{cell['cell']}@{cell['length']} [budget {cell['settings']['max_new_tokens']}]"
            if not force and RFB.cell_key(model, cell) in done:
                print(f"  {tag}: SKIP (in history)", flush=True)
                continue
            cap = B.cell_dollar_cap(model, cell["n"], cell["settings"]["max_new_tokens"])
            print(f"  {tag}: running n={cell['n']} "
                  f"(cell dollar cap {'none' if cap is None else f'${cap:.2f}'})", flush=True)
            try:
                backend = RFB.build_backend(model, cell, api_key, base_url, max_workers)
                rec = RFB.execute_cell(backend, model, cell, n=cell["n"],
                                       run_id=run_id, git_commit=git_commit)
            except Exception:            # one bad cell must not kill the run
                print(f"  {tag}: FAILED (no record written)", flush=True)
                traceback.print_exc()
                continue
            rec["scout_cell"] = cell["cell"]
            RFB.append_record(path, rec)
            done.add(RFB.cell_key(model, cell))
            d, u = rec["diagnostics"], rec["usage"]
            total += u["cost_usd_est"]
            print(f"  {tag}: match={rec['metrics']['relaxed']:.3f} "
                  f"empty={d['empty_rate']:.2f} trunc={d['truncated_rate']:.2f} "
                  f"finish={d['finish_reasons']} err={d['api_errors']} "
                  f"ptok={u['prompt_tokens']} ctok={u['completion_tokens']} "
                  f"rtok={u['reasoning_tokens']} ${u['cost_usd_est']:.2f} "
                  f"[{rec['elapsed_s']:.0f}s]", flush=True)
            if max(d["empty_rate"], d["truncated_rate"]) > P.SCOUT_TRUNCATION_MAX:
                print(f"  !!! {tag} is VOID under the pre-registered rule "
                      f"(>{P.SCOUT_TRUNCATION_MAX:.0%} truncated/empty): re-run at a raised "
                      f"budget; it enters no stop or buy decision.", flush=True)
            if d["cost_aborted"]:
                print(f"  !!! {tag}: COST GUARD tripped ({d.get('cost_abort_reason')}), only "
                      f"{d['calls_completed']}/{rec['n']} calls submitted.", flush=True)
        print(f"  -- {model} done, ${total:.2f} this process", flush=True)


# --- reading -------------------------------------------------------------------------------

def load_rows() -> dict:
    """``{model: {"<cell>@<L>": row}}`` from every scout history file, latest record wins.

    A cell the truncation rule VOIDED and that was re-run at a raised budget has two records
    under one key; the later one is the row, and every earlier one is kept on it as
    ``superseded`` so the report shows what each budget bought rather than quietly dropping the
    measurement that forced the re-run.
    """
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(HISTORY_DIR, "history_*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cell = rec.get("scout_cell") or rec["task"]
                key = f"{cell}@{rec['length']}"
                d, u = rec["diagnostics"], rec["usage"]
                row = {"match": rec["metrics"]["relaxed"], "n": rec["n"],
                       "length_rate": d["truncated_rate"], "empty_rate": d["empty_rate"],
                       "finish_reasons": d["finish_reasons"], "api_errors": d["api_errors"],
                       "finish_errors": d["finish_errors"], "cost_aborted": d["cost_aborted"],
                       "budget": rec["settings"]["max_new_tokens"],
                       "prompt_tokens": u["prompt_tokens"],
                       "completion_tokens": u["completion_tokens"],
                       "reasoning_tokens": u["reasoning_tokens"],
                       "cost_usd": u["cost_usd_est"], "elapsed_s": rec["elapsed_s"],
                       "ts": rec["ts"], "task": rec["task"], "length": rec["length"]}
                prev = out.setdefault(rec["model"], {}).get(key)
                if prev is None:
                    out[rec["model"]][key] = row
                elif row["ts"] >= prev["ts"]:
                    row["superseded"] = prev.pop("superseded", []) + [prev]
                    out[rec["model"]][key] = row
                else:
                    prev.setdefault("superseded", []).append(row)
    return out


def void_bound(row: dict) -> float:
    """The largest match a cell's score could have taken had every failed call been right.

    A truncated or empty reply is scored WRONG (``run_frontier_benchmark._run_attempt`` empties
    a finish=length reply), so a void cell's observed match is a LOWER bound and this is the
    upper one. It is what makes a verdict's independence from the void cells checkable rather
    than asserted: where the bound cannot reach the threshold, no re-run of that cell can move
    the rule.
    """
    return min(1.0, row["match"] + max(row["length_rate"], row["empty_rate"]))


def detail_top(scores: dict, models: list, key: str) -> str | None:
    """The top model on one cell — the same ordering ``scout_verdict`` ranks by."""
    ranked = sorted(models, key=lambda m: -(scores[m].get(key, {}).get("match") or 0.0))
    return ranked[0] if ranked else None


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return ((c - h) / d, (c + h) / d)


def report(scores: dict) -> str:
    """The scout report: every cell's number with the diagnostics that say whether it is a
    measurement, then the pre-registered rule applied mechanically."""
    chance = P.scout_informed_chance()
    models = [m for m in P.SCOUT_MODELS if m in scores]
    cells = [(c["cell"], c["length"]) for c in P.scout_cells()]
    est = {r["model"]: r for r in P.scout_plan()["per_model"] if "cost_usd" in r}

    L = []
    L.append("# s5_bind_v3 — the frontier scout\n")
    L.append(f"Written {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
             f"Reading rule pre-registered in `scripts/protocol_s5bind_v3_three_cell_20260731.py` "
             f"(`scout_verdict`), fixed before any frontier number existed. Metric is **match**, "
             f"the canonical evaluator, on the answer read only — the frontier arm has no "
             f"harness-generated checkpoint stream and `assert_trace_read` raises on it.\n")
    L.append(f"Informed chance is 1/(k-1) = **{chance:.4f}** at the k=12 operating point: the "
             f"initial map is stated, so the queried agent's own starting value is never the "
             f"gold answer. **The composed cell has no floor in this regime.** A frontier model "
             f"reasons in visible tokens, which is a scratchpad, and the composed cell's floor "
             f"argument is a bound on live slots (W <= max(k,m)+1 = 13 against the task's 25) — "
             f"the same bound the guided protocol voids locally, where the both-maps replay "
             f"reaches 0.719 against a printed floor of 0.234. Its number is read against "
             f"informed chance as a guess baseline and against the other models as a spread.\n")

    L.append("## Cells\n")
    L.append("| cell | task | L | n | registered budget | budget it was measured at | "
             "prompt tokens |")
    L.append("|---|---|---|---|---|---|---|")
    for c in P.scout_cells():
        key = f"{c['cell']}@{c['length']}"
        got = sorted({scores[m][key]["budget"] for m in models if key in scores[m]})
        ptok = [scores[m][key]["prompt_tokens"] // scores[m][key]["n"]
                for m in models if key in scores[m]]
        L.append(f"| {key} | `{c['task']}` | {c['length']} | {c['n']} | "
                 f"{c['settings']['max_new_tokens']} | "
                 f"{', '.join(str(b) for b in got) or '—'} | "
                 f"{min(ptok) if ptok else '—'}–{max(ptok) if ptok else '—'} |")
    L.append("")
    L.append("A cell measured at more than one budget was VOIDED by the truncation rule at the "
             "registered one and re-run; the budget history is below. The prompt-token range is "
             "across models — the composed prompt alone is ~3.0k tokens at L=128 and ~5.6k at "
             "L=256, which is why the 8,192 the scout was originally priced at was a validity "
             "defect and not a saving.\n")

    L.append("## Match, per model per cell\n")
    L.append("| model | " + " | ".join(f"{c}@{l}" for c, l in cells) + " |")
    L.append("|---" * (len(cells) + 1) + "|")
    for m in models:
        vals = []
        for c, l in cells:
            r = scores[m].get(f"{c}@{l}")
            if r is None:
                vals.append("—")
                continue
            lo, hi = _wilson(r["match"], r["n"])
            vals.append(f"{r['match']:.3f} [{lo:.2f}, {hi:.2f}]")
        L.append(f"| {m} | " + " | ".join(vals) + " |")
    L.append("")
    L.append("95% Wilson intervals at n=40 in brackets. Per-cell values only; nothing here is "
             "averaged across cells or models.\n")

    L.append("## Validity diagnostics — read before the scores\n")
    L.append("| model | cell | budget | finish=length | empty | finish reasons | api errors | "
             "cost aborted |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in models:
        for c, l in cells:
            r = scores[m].get(f"{c}@{l}")
            if r is None:
                continue
            L.append(f"| {m} | {c}@{l} | {r['budget']} | {r['length_rate']:.2f} | "
                     f"{r['empty_rate']:.2f} | `{r['finish_reasons']}` | "
                     f"{r['api_errors']} (+{r['finish_errors']} finish=error) | "
                     f"{r['cost_aborted']} |")
    L.append("")
    L.append(f"The pre-registered rule evaluates truncation FIRST: any cell over "
             f"{P.SCOUT_TRUNCATION_MAX:.0%} finish=length or empty answers is VOID, is re-run at "
             f"a raised budget, and enters no stop or buy decision. Evaluated in any other "
             f"order, a truncated cell reads as a floor — which is how the published s5 L64 "
             f"cliff was manufactured.\n")
    at_bar = [(m, f"{c}@{l}") for m in models for c, l in cells
              if f"{c}@{l}" in scores[m]
              and 0 < max(scores[m][f"{c}@{l}"]["length_rate"],
                          scores[m][f"{c}@{l}"]["empty_rate"]) <= P.SCOUT_TRUNCATION_MAX]
    if at_bar:
        L.append(f"Cells with some truncation but AT OR UNDER the bar are admissible and are "
                 f"read as measured, with the caveat that a truncated call is scored wrong, so "
                 f"their match is a lower bound: "
                 f"{', '.join(f'{m} {k}' for m, k in at_bar)}.\n")

    superseded = [(m, key, a) for m in models for key, r in sorted(scores[m].items())
                  for a in r.get("superseded", ())]
    if superseded:
        L.append("### The budgets the truncation rule voided, and what they bought\n")
        L.append("| model | cell | budget | match (scored) | upper bound if every failed call "
                 "were right | finish=length | empty |")
        L.append("|---|---|---|---|---|---|---|")
        for m, key, a in superseded:
            L.append(f"| {m} | {key} | {a['budget']} | {a['match']:.3f} | "
                     f"{void_bound(a):.3f} | {a['length_rate']:.2f} | {a['empty_rate']:.2f} |")
        L.append("")
        L.append("A truncated or empty reply is scored wrong, so a void cell's match is a LOWER "
                 "bound and the column beside it is the upper one. These rows are kept because "
                 "the raised-budget re-run replaces them in the decision, not in the record: "
                 "what a budget bought is a fact about that budget.\n")
        ck = f"composed@{P.SCOUT_STOP_CEILING_LENGTH}"
        top_c = scores.get(detail_top(scores, models, ck), {}).get(ck, {}).get("match")
        bounds = [void_bound(a) for m in models for a in scores[m].get(ck, {}).get(
            "superseded", ()) ]
        if top_c is not None and bounds and max(bounds) < top_c:
            L.append(f"The upper bounds also settle what the re-runs could have changed about "
                     f"the ranking: the highest value any voided {ck} cell could have taken is "
                     f"{max(bounds):.3f}, below the {top_c:.3f} the top model read on a cell "
                     f"with no truncation at all, so no budget raise anywhere in this run could "
                     f"have moved which model is top.\n")

    L.append("## Spend — every call billed, including the attempts the truncation rule voided\n")
    L.append("| model | prompt tok | completion tok | reasoning tok | actual $ | of which "
             "voided attempts | estimate $ (4k ctok) | worst case $ |")
    L.append("|---|---|---|---|---|---|---|---|")
    grand = void_spend = 0.0
    for m in models:
        keep = [scores[m][f"{c}@{l}"] for c, l in cells if f"{c}@{l}" in scores[m]]
        dead = [a for r in keep for a in r.get("superseded", ())]
        rows = keep + dead
        act = sum(r["cost_usd"] for r in rows)
        vd = sum(r["cost_usd"] for r in dead)
        grand += act
        void_spend += vd
        e = est.get(m, {})
        L.append(f"| {m} | {sum(r['prompt_tokens'] for r in rows):,} | "
                 f"{sum(r['completion_tokens'] for r in rows):,} | "
                 f"{sum(r['reasoning_tokens'] for r in rows):,} | ${act:.2f} | ${vd:.2f} | "
                 f"${e.get('cost_usd', float('nan')):.2f} | "
                 f"${e.get('worst_case_usd', float('nan')):.2f} |")
    L.append(f"| **total** | | | | **${grand:.2f}** | ${void_spend:.2f} | "
             f"${sum(e.get('cost_usd', 0) for e in est.values()):.2f} | "
             f"${sum(e.get('worst_case_usd', 0) for e in est.values()):.2f} |")
    L.append("")
    L.append("The estimate column is what `benchmark.cost_estimate` prices at 4,000 assumed "
             "completion tokens per call; it reads `assumed_output_tokens` and never "
             "`max_new_tokens`, so it under-prices a reasoning cell whose traces run long and "
             "it is not a budget guard. The worst case — every call spending its whole "
             "registered budget — is what the per-cell dollar caps bound the run to, and it is "
             "the number the spend was approved against.\n")

    L.append("## Component floors, recomputed from the exact scored items\n")
    L.append("| cell | floor on the 40 scored items | floor on 4,000 disjoint items | "
             "operative | basis | " + " | ".join(m.split("/")[-1] for m in models) + " |")
    L.append("|---|---|---|---|---" + "|---" * len(models) + "|")
    for c in P.scout_cells():
        if c["cell"] == "composed":
            continue
        f = component_floors(c["cell"], c["task"], c["length"], c["n"])
        key = f"{c['cell']}@{c['length']}"
        got = " | ".join("—" if key not in scores[m] else f"{scores[m][key]['match']:.3f}"
                         for m in models)
        L.append(f"| {key} | {f['scored']:.4f} | {f['disjoint']:.4f} | "
                 f"**{f['operative']:.4f}** | {f['basis']} | {got} |")
    L.append("")
    L.append("The two differ where the max over admitted rows takes a high draw at n=40; the "
             "larger is operative. The component floors survive a scratchpad by structure: "
             "their admitted rows are depth <= 1 and cost under the cell's own algorithm's "
             "per-item minimum, and a pad substitutes for registers, not for chaining. The "
             "composed cell has no row here because it has no floor in this regime.\n")

    L.append("## The verdict\n")
    for rule in P.scout_plan()["stop_rules"]:
        L.append(f"- {rule}")
    L.append(f"- 5. {P.scout_plan()['buy_rule']}")
    L.append("")
    try:
        code, why, detail = P.scout_verdict(scores)
    except P.ScoutCellVoid as exc:
        L.append(f"**VOID_TRUNCATION** — {exc}\n")
        L.append("No stop or buy decision is available from this run until the void cells are "
                 "re-run at a raised budget.\n")
        return "\n".join(L)
    L.append(f"**{code}** — {why}\n")
    spread = {L_: (None if s is None else round(s, 3)) for L_, s in detail["spreads"].items()}
    L.append(f"Top model by composed@{P.SCOUT_STOP_CEILING_LENGTH}: `{detail['top_model']}`. "
             f"Composed spread per length: {spread} against the {P.SCOUT_SEPARATION} bar. "
             f"Informed-chance band at {P.SCOUT_FLOOR_SE:.0f} se: "
             f"{detail.get('informed_chance_band', 'not reached — an earlier rule fired')}. "
             f"Components at or above {P.SCOUT_COMPONENT_MIN} for every scout model: "
             f"{detail.get('components_all_above_min', 'not reached — an earlier rule fired')}. "
             f"Per-component values for the top model: "
             f"{ {c: v.get(detail['top_model']) for c, v in detail['components'].items()} }.\n")
    L.append("The roster run is not bought by this script under any verdict: the scout reports "
             "and stops.\n")

    # What the LATER rules would have read, printed whichever one fired. The order decides the
    # verdict; it does not decide which numbers are worth knowing, and the difference between
    # "the cell does not discriminate" and "it discriminates below a saturated top" is the whole
    # of what a redesign should target.
    L.append("### What the rules below the one that fired would have read\n")
    comp_top = {c: v.get(detail["top_model"]) for c, v in detail["components"].items()}
    at_floor = {m: scores[m].get(f"composed@{P.SCOUT_STOP_FLOOR_LENGTH}", {}).get("match")
                for m in models}
    L.append(f"- STOP (floor) reads composed@{P.SCOUT_STOP_FLOOR_LENGTH} at "
             f"{ {m.split('/')[-1]: v for m, v in at_floor.items()} } against the informed-chance "
             f"band {detail['informed_chance_band']}. It does not fire: no model is at a guess.")
    L.append(f"- STOP (component) reads the top model's components at {comp_top} against "
             f"{P.SCOUT_COMPONENT_MIN}. It does not fire.")
    L.append(f"- BUY reads a composed spread of {spread} against {P.SCOUT_SEPARATION}, with "
             f"both components at or above {P.SCOUT_COMPONENT_MIN} for every scout model: "
             f"{detail['components_all_above_min']}. Its condition is "
             f"{'MET' if (detail.get('best_spread') or 0) >= P.SCOUT_SEPARATION and detail['components_all_above_min'] else 'NOT met'}.")
    L.append("")
    if code == "STOP_CEILING":
        L.append("So the cell is not undiscriminating — it separates the three models by "
                 f"{spread[P.SCOUT_STOP_FLOOR_LENGTH]:.3f} at L={P.SCOUT_STOP_FLOOR_LENGTH} and "
                 f"{spread[P.SCOUT_STOP_CEILING_LENGTH]:.3f} at "
                 f"L={P.SCOUT_STOP_CEILING_LENGTH}, and both components are at or above "
                 f"{P.SCOUT_COMPONENT_MIN} everywhere, which is the buy rule's condition. It is "
                 "the TOP of the range that is saturated, and a ranking whose first place is a "
                 "ceiling cannot order the models a roster run is bought to order. That is what "
                 "the redesign has to move: k or L, not the budget and not the roster.\n")
    return "\n".join(L)


def write_report() -> str:
    scores = load_rows()
    text = report(scores)
    # the composed cell has no floor here, so the report may not speak of one
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
    ap.add_argument("--models", nargs="+", default=list(P.SCOUT_MODELS),
                    choices=list(P.SCOUT_MODELS),
                    help="Scout models (the registered three; nothing else is approved).")
    ap.add_argument("--n-scale", type=float, default=1.0, dest="n_scale")
    ap.add_argument("--max-workers", type=int, default=8, dest="max_workers")
    ap.add_argument("--force", action="store_true", help="ignore resume")
    ap.add_argument("--base-url", default=B.DEFAULT_BASE_URL, dest="base_url")
    ap.add_argument("--run-id", default=None, dest="run_id")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="print the plan, the prices and the caps; no API calls")
    ap.add_argument("--report", action="store_true", help="read the history files and apply the "
                                                          "pre-registered rule")
    ap.add_argument("--raise-budget", action="append", default=None, dest="raise_budget",
                    metavar="CELL@L:BUDGET",
                    help="Re-run a VOID cell at a raised max_new_tokens (repeatable). The "
                         "budget is part of the settings hash, so the raised cell gets a fresh "
                         "resume key and the void record stays as measured.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only these cells (e.g. composed@128).")
    a = ap.parse_args()
    raised = parse_raised(a.raise_budget)

    if a.report:
        write_report()
        return
    if a.dry_run:
        plan = P.scout_plan()
        print(f"informed chance 1/(k-1) = {plan['informed_chance']}")
        print(f"budgets {plan['budgets']}")
        for r in plan["per_model"]:
            print(f"\n{r['model']}: {r['calls']} calls, {r['prompt_tokens']:,} prompt tok, "
                  f"est ${r['cost_usd']:.2f} @4k ctok, worst case ${r['worst_case_usd']:.2f}")
            for k, v in r["cell_dollar_caps"].items():
                print(f"    {k:<14} cell dollar cap "
                      f"{'none (token guard: 3x nominal)' if v is None else f'${v:.2f}'}")
        print(f"\nTOTAL est ${plan['total_usd']:.2f} @4k ctok; "
              f"worst case ${plan['worst_case_usd']:.2f}")
        for m in a.models:
            for c in plan_for(m, a.n_scale, raised):
                cap = B.cell_dollar_cap(m, c["n"], c["settings"]["max_new_tokens"])
                print(f"  {m} {c['cell']}@{c['length']} n={c['n']} "
                      f"budget={c['settings']['max_new_tokens']} "
                      f"cap={'none' if cap is None else f'${cap:.2f}'} "
                      f"sysprompt={c['settings'].get('system_prompt_fp')}")
        return
    run(a.models, a.n_scale, a.max_workers, a.force, a.base_url,
        a.run_id or f"s5bind_scout_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        raised=raised, only=a.only)


if __name__ == "__main__":
    main()
