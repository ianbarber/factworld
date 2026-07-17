"""Run the recurring frontier-model benchmark and append C3 records to history.

Executes the cell plan from ``factworld.benchmark.arms_for`` (contract C4) against
each model's API endpoint — OpenRouter by default; a registry entry carrying
``{"base_url", "api_key_env"}`` runs against its direct vendor endpoint with its
own key env (``factworld.benchmark.endpoint_for`` — the muse-spark slot) — one
model at a time, one cell at a time (examples fan out concurrently inside
``APIBackend``). Each completed cell appends ONE crash-safe JSONL record to
the history file (contract C3) — metrics, diagnostics (empty-pred rate, api errors,
finish reasons), token usage and an estimated cost. Every example record carries the
per-call ``{ctok, rtok, finish}`` (completion tokens, reasoning tokens, finish
reason) alongside ``{gold, pred, relaxed}`` so pass-at-budget / tokens-at-matched-
accuracy views are computable post hoc. Resume is automatic: any cell whose (model,
facet, task, length, n, settings_hash, stream_version) key already has a history
record is skipped (latest-wins dedup lives in scripts/render_benchmark.py). n is
part of the key so a low-n scouting pass (--n-scale) never satisfies resume for the
full-n cell; the version component is the cell's SPEC stream version (not the
global suite version), so registry/suite bumps never invalidate cells whose pinned
example streams are unchanged.

Protocol rules:
  - Reasoning-on cells run with a >=8192-token budget and stop_at=None (smaller
    budgets manufactured the old "s5 L64 cliff" / "chain floor" as truncation
    artifacts — see results/s5_horizon_recheck_20260705.jsonl). A cell whose
    empty-pred rate exceeds 0.5 gets a loud truncation-suspect warning but the
    record is kept.
  - chain_nowrap is a STAIRCASE: depth d runs chain_v2.scaled(k=2*d+1), so the
    pointer cycle never wraps AND the backward walk costs d+1 hops (k=d+2 would
    leave gold a constant 2 reverse lookups from start). Breadth grows with
    depth by design.
  - zero_budget cells (settings.contract) append a hard one-line answer contract
    to every prompt and score the span after the LAST "Answer:" line of the
    visible output; per-cell contract_rate / covert_cot_rate / rtok_any_rate /
    rtok_mean_per_call diagnose whether the effort=none arm is actually clean
    for that model.
  - The zero_budget "replicate" leg is a TEST-RETEST duplicate of the plain cell:
    the runner builds the IDENTICAL prompt on purpose (review F6 — the old
    "end_to_end" leg was this prompt mislabeled as a distinct measurement); its
    |delta| vs the plain cell is the quoted run-to-run noise bar.
  - Model-aware cutoff handling (iterated, review F2/F3): a zero_budget cell
    whose finish=length rate exceeds 10% is rerun at escalating budgets
    (96 -> 512 -> 2048, at most 2 escalations). The FIRST attempt (at the
    planned budget) is the CANONICAL number; escalated attempts are marked
    diagnostics. EVERY attempt's metrics/diagnostics/examples are recorded
    verbatim under ``escalation.attempts`` (the first attempt's per-example
    data must be complete — the renderer publishes it), and the record's
    top-level metrics/examples are the last attempt's (escalated=true flags
    them as the diagnostic view). Usage/cost cover all attempts.
  - Per-cell spend guard (the grok-build lesson): once a cell's cumulative
    visible completion tokens exceed CELL_BUDGET_FACTOR * n * max_new_tokens —
    or, for expensive models (completion price >= $10/M: opus, gpt-5.5,
    sonnet), once its completion-priced spend exceeds the per-cell DOLLAR cap
    max($2.50, nominal n * max_new_tokens completion spend) — no further calls
    are submitted; what completed is recorded with diagnostics.cost_aborted=true
    (+ cost_abort_reason "ctok"/"usd") and a loud warning.
  - v3 working-set-breadth rungs: a cell may carry settings["breadth"] (the pool
    rung B; the task runs at CANONICAL[task].scaled(k=2*B, recall_pool=B)) or,
    for chain cells, settings["k_fixed"] (chain_v2.scaled(k=k_fixed), fixed
    breadth instead of the staircase). Both keys are sentinel-dropped at their
    canonical values (B=16 / no k_fixed) so pre-breadth history resume keys are
    unchanged; when present (non-canonical) they are part of the settings hash,
    so every rung resumes independently.
  - finish_reason=="error" calls are counted into diagnostics.finish_errors
    (distinct from api_errors, the exception-path count — review F8 found 12
    finish=error calls invisible at api_errors=0) and warned about loudly.

Examples:
    set -a; source .env; set +a

    # Print the full plan + cost estimate, no API calls
    .venv-api/bin/python scripts/run_frontier_benchmark.py --dry-run

    # Scouting pass at 1/5 the per-facet n
    .venv-api/bin/python scripts/run_frontier_benchmark.py --n-scale 0.2

    # Re-run the drift canary even where history already has its cells
    .venv-api/bin/python scripts/run_frontier_benchmark.py \\
        --models z-ai/glm-5.2 --canary
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factworld import s5_concrete as S5
from factworld import tasks as TK
from factworld.backends import APIBackend, ResponsesBackend
from factworld.benchmark import (
    CANARY_MODEL,
    CELL_BUDGET_FACTOR,
    COVERT_COT_CTOK_THRESHOLD,
    DEFAULT_API_KEY_ENV,
    FACETS,
    MODELS,
    REASONING_EFFORTS,
    arms_for,
    cell_dollar_cap,
    cost_estimate,
    endpoint_for,
    settings_hash,
    spec_for_cell,
)
from factworld.render import Renderer
from factworld.runner import evaluate_task

# Reuse the grid script's system-prompt plumbing (composite format instruction) and
# the autoregressive experiment's binding-leg prompt builder, per the design.
from eval_openrouter_grid import (  # noqa: E402
    COMPOSITE_FORMAT_PROMPT,
    _build_system_prompt,
)
from experiment_autoregressive import binding_prompt, scaffold_prompt  # noqa: E402

# The canonical published grid's base system prompt (docs/openrouter/results-natural.jsonl
# records this verbatim for every non-composite cell). NOTE: this is deliberately NOT
# eval_openrouter_grid.DEFAULT_SYSTEM_PROMPT — that constant later had the composite
# holder-name instruction folded into it, which leaks the two-token answer format into
# single-answer tasks (models then answer 'g5 v37' on recall/conflict and score 0 on the
# canonical relaxed prefix match).
BASE_SYSTEM_PROMPT = (
    "You are taking a short test. Answer each question with only the requested "
    "value or values, no explanation. Use the same spelling as in the question."
)

# Grid mechanics: --composite_format appends the two-token format instruction for
# composite tasks (format-fair across models that don't guess the output shape).
# composite_copy_v2 (the zero_budget task since the uniform-last-write fix) has
# the same prompt/answer shape as v1, so it takes the same format instruction.
TASK_PROMPTS = {
    "composite_copy_v1": COMPOSITE_FORMAT_PROMPT,
    "composite_copy_v2": COMPOSITE_FORMAT_PROMPT,
    "composite_v1": COMPOSITE_FORMAT_PROMPT,
}

S5_FACETS = ("s5_concrete",)  # cells rendered via factworld.s5_concrete

# --- zero-budget answer contract ----------------------------------------------------
# Every contract cell's prompt ends with a hard one-line contract; extraction takes
# the span after the LAST "Answer:" line of the visible output, so a model that
# emits working (sonnet's visible scratchpad, kimi's covert in-content CoT) and THEN
# the contract line still scores its committed answer.
CONTRACT_LINE_COMPOSITE = "Reply with only one line: Answer: <holder> <value>"
CONTRACT_LINE_BINDING = "Reply with only one line: Answer: <holder>"
CONTRACT_LINE_VALUE = "Reply with only one line: Answer: <value>"
CONTRACT_LINE_AGENT = "Reply with only one line: Answer: <agent>"
# Plain (leg None / replicate) contract cells pick the line by the spec family:
# composite answers are "<holder> <value>"; recall_load answers a single value
# token; chain_instant answers a single agent token. Unsupported families fail
# loudly (KeyError) rather than shipping a mismatched contract.
CONTRACT_LINES_BY_FAMILY = {
    "composite": CONTRACT_LINE_COMPOSITE,
    "recall": CONTRACT_LINE_VALUE,
    "chain": CONTRACT_LINE_AGENT,
}
_ANSWER_LINE_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)

# Model-aware cutoff handling (owner requirement; iterated per review — a single
# 512 escalation left kimi L16 with 9 residual cap-outs): while finish=length
# exceeds this fraction of a zero_budget cell's calls, rerun the cell at the next
# escalation budget (96 -> 512 -> 2048, at most len(ESCALATION_BUDGETS) reruns).
# The FIRST attempt stays canonical (review F2); escalated attempts are marked
# diagnostics, each recorded in full under escalation.attempts.
LENGTH_ESCALATION_THRESHOLD = 0.10
ESCALATION_BUDGETS = (512, 2048)


def extract_contract_answer(text: str) -> str | None:
    """The answer span after the LAST ``Answer:`` line of ``text`` (None if absent).

    Case-insensitive; the span is stripped of whitespace and surrounding markdown
    emphasis (``**Answer:** g3 v9`` -> ``g3 v9``). An ``Answer:`` line with an
    empty span does not count as parseable.
    """
    spans = [m.strip().strip("*_` ") for m in _ANSWER_LINE_RE.findall(text)]
    spans = [s for s in spans if s]
    return spans[-1] if spans else None

EMPTY_META = {
    "calls": 0, "errors": 0,
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
    "served_models": [], "providers": [], "finish_reasons": {},
}


class CostGuardBackend:
    """Per-cell spend guard (the grok-build lesson: a pinned generator ignored the
    16384 cap and emitted ~256k visible ctok per call — a cell must not be able to
    spend unboundedly past its nominal budget).

    Wraps any ``ModelBackend`` and submits prompts in chunks; after each chunk the
    cell's cumulative visible completion tokens are compared against
    ``budget_ctok`` (CELL_BUDGET_FACTOR * n * max_new_tokens) AND — when
    ``budget_usd`` is set (expensive models, see ``cell_dollar_cap``) — their
    completion-priced dollar cost against the per-cell DOLLAR cap. Once either is
    exceeded, no further chunks are submitted: the remaining prompts get empty
    predictions and ``finish="cost_aborted"`` per-example meta, ``cost_aborted``
    is set so the record can flag what completed, and ``abort_reason`` records
    which guard tripped ("ctok" or "usd"). Backends without per-example meta
    (FunctionBackend) cannot be measured, so the guard is inert for them (it
    passes calls through untracked and ``pop_example_meta`` returns None).

    Cumulative ctok persists across ``generate`` calls on one guard instance; a
    fresh guard is built per attempt with that attempt's budget (an aborted
    attempt also blocks any further escalation — see ``execute_cell``).
    """

    CHUNK = 8  # calls submitted between budget checks

    def __init__(self, backend, budget_ctok: int, budget_usd: float | None = None,
                 completion_price_per_M: float = 0.0):
        self.backend = backend
        self.budget_ctok = budget_ctok
        self.budget_usd = budget_usd
        self.completion_price_per_M = completion_price_per_M
        self.cost_aborted = False
        self.abort_reason: str | None = None  # "ctok" | "usd" once tripped
        self.calls_completed = 0
        self._cum_ctok = 0
        self._has_meta = hasattr(backend, "pop_example_meta")
        self._ex_meta: list[dict] | None = None
        self._call_meta: dict | None = None

    @property
    def name(self) -> str:
        return getattr(self.backend, "name", "unknown")

    def generate(self, prompts, max_new_tokens, stop_at=None):
        preds: list[str] = []
        ex_meta: list[dict] = []
        call_meta = copy.deepcopy(EMPTY_META)
        i = 0
        while i < len(prompts) and not self.cost_aborted:
            chunk = prompts[i:i + self.CHUNK]
            preds.extend(self.backend.generate(chunk, max_new_tokens, stop_at))
            self.calls_completed += len(chunk)
            if self._has_meta:
                metas = self.backend.pop_example_meta()
                ex_meta.extend(metas)
                self._cum_ctok += sum(m["completion_tokens"] or 0 for m in metas)
            if hasattr(self.backend, "pop_call_meta"):
                self._merge_call_meta(call_meta, self.backend.pop_call_meta())
            i += len(chunk)
            if self._has_meta:
                if self._cum_ctok > self.budget_ctok:
                    self.cost_aborted, self.abort_reason = True, "ctok"
                elif (self.budget_usd is not None
                      and self._cum_ctok / 1e6 * self.completion_price_per_M
                      > self.budget_usd):
                    # per-cell DOLLAR cap (completion-priced; usage completion
                    # tokens already include reasoning) — expensive models only.
                    self.cost_aborted, self.abort_reason = True, "usd"
        n_missing = len(prompts) - len(preds)
        if n_missing:
            preds.extend([""] * n_missing)
            ex_meta.extend([{"completion_tokens": None, "reasoning_tokens": None,
                             "finish_reason": "cost_aborted"}] * n_missing)
        self._ex_meta = ex_meta if self._has_meta else None
        self._call_meta = call_meta
        return preds

    @staticmethod
    def _merge_call_meta(acc: dict, meta: dict) -> None:
        acc["calls"] += meta["calls"]
        acc["errors"] += meta["errors"]
        for key in acc["usage"]:
            acc["usage"][key] += meta["usage"][key]
        for key in ("served_models", "providers"):
            acc[key] = list(dict.fromkeys(acc[key] + meta[key]))
        for reason, count in meta["finish_reasons"].items():
            acc["finish_reasons"][reason] = acc["finish_reasons"].get(reason, 0) + count

    def pop_example_meta(self):
        meta, self._ex_meta = self._ex_meta, None
        return meta

    def pop_call_meta(self):
        meta, self._call_meta = self._call_meta, None
        return meta if meta is not None else copy.deepcopy(EMPTY_META)


# --- plan / resume ----------------------------------------------------------------

def stream_version(task: str) -> str:
    """The RNG-stream version of a cell's task: ``spec.version`` (frozen at spec
    introduction — see tasks._STREAM_V1), NOT the global TK.SUITE_VERSION.

    The suite version moves whenever the registry changes (1.0 -> 1.1 added the
    v2 specs), but a spec's example stream is immutable forever, so keying resume
    on the global would spuriously invalidate EVERY existing cell on a suite bump
    (chain_nowrap/s5/sanity cells are unchanged — their streams are pinned).
    Facet tasks that are not TaskSpecs (task "s5", rendered via
    factworld.s5_concrete off the pinned s5_v1 spec) carry the v1 stream version.
    """
    spec = TK.CANONICAL.get(task) or TK.RETIRED.get(task)
    return spec.version if spec is not None else TK._STREAM_V1


def cell_key(model: str, cell: dict) -> tuple:
    """Resume key: skip a cell if ANY history record already carries this key.

    Includes the cell's n so a low-n scouting run (--n-scale) does not mark the
    full-n cell as done. The version component is the PER-SPEC stream version
    (``stream_version``), so a suite bump only invalidates cells whose task
    stream actually changed (i.e. genuinely new specs).
    """
    return (model, cell["facet"], cell["task"], cell["length"], cell["n"],
            settings_hash(cell), stream_version(cell["task"]))


def history_keys(history_path: str) -> set[tuple]:
    """Read the history file and return the set of already-run cell keys.

    Records written since the per-spec resume key carry ``stream_version``;
    older records fall back to their recorded ``suite_version``, which equals
    the stream version for every pre-1.1 record (all their tasks are pinned at
    the "1.0" stream).
    """
    keys: set[tuple] = set()
    if not os.path.exists(history_path):
        return keys
    with open(history_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn tail line from a crashed run
            keys.add((rec.get("model"), rec.get("facet"), rec.get("task"),
                      rec.get("length"), rec.get("n"),
                      settings_hash({"settings": rec.get("settings") or {}}),
                      rec.get("stream_version") or rec.get("suite_version")))
    return keys


def should_skip(model: str, cell: dict, done: set[tuple], force: bool, canary: bool) -> bool:
    if force:
        return False
    if canary and model == CANARY_MODEL:
        return False  # drift canary: always re-run glm cells
    return cell_key(model, cell) in done


def parse_budget_overrides(specs: list[str] | None) -> dict[tuple[str, int], int]:
    """Parse repeatable ``--budget-override facet:length:budget`` specs.

    Returns {(facet, length): max_new_tokens}. The override REPLACES the planned
    cell's max_new_tokens, which is part of the settings hash — so an overridden
    cell gets a FRESH resume key (it re-runs even when the planned-budget cell is
    in history) and the renderer's latest-ts dedup shows the raised-budget record
    in place of the old one, with no renderer changes. Fails loudly on malformed
    specs or unknown facets (a typo must not silently no-op a paid run).
    """
    out: dict[tuple[str, int], int] = {}
    for s in specs or ():
        parts = s.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--budget-override expects facet:length:budget, got {s!r}")
        facet, length_s, budget_s = parts
        if facet not in FACETS:
            raise SystemExit(f"--budget-override: unknown facet {facet!r} "
                             f"(known: {', '.join(FACETS)})")
        try:
            key, budget = (facet, int(length_s)), int(budget_s)
        except ValueError:
            raise SystemExit(f"--budget-override expects integer length/budget, got {s!r}")
        if budget <= 0:
            raise SystemExit(f"--budget-override: budget must be positive, got {s!r}")
        out[key] = budget
    return out


def build_plan(models: list[str], facets: list[str] | None, n_scale: float,
               lengths: list[int] | None = None,
               budget_overrides: dict[tuple[str, int], int] | None = None) -> dict[str, list[dict]]:
    """Per-model cell lists with the scouting multiplier applied (n floor of 5).

    ``lengths`` (from --lengths) keeps only cells at those lengths/depths;
    ``budget_overrides`` (from --budget-override) replaces matching cells'
    max_new_tokens — see parse_budget_overrides for the resume-key semantics.
    """
    plan = {}
    for model in models:
        cells = [c for c in arms_for(model) if facets is None or c["facet"] in facets]
        if lengths is not None:
            cells = [c for c in cells if c["length"] in lengths]
        for c in cells:
            c["n"] = max(5, round(c["n"] * n_scale))
            if budget_overrides:
                override = budget_overrides.get((c["facet"], c["length"]))
                if override is not None:
                    c["settings"]["max_new_tokens"] = override
        plan[model] = cells
    return plan


# --- backend construction -----------------------------------------------------------

def system_prompt_for(cell: dict) -> str:
    """The per-cell system prompt (constant across a cell's examples)."""
    if cell["facet"] in S5_FACETS:
        # The framing-specific system prompt from the single source of truth
        # (identical for every example/length of a framing).
        return S5.gen_examples(4, 1, framing=cell["settings"]["rendering"])[0][0]
    if cell["settings"]["leg"] == "binding_only":
        # Holder-only leg: the composite two-token format instruction ("g3 v9")
        # would contradict the one-token contract line, so use the bare base prompt.
        return BASE_SYSTEM_PROMPT
    return _build_system_prompt(BASE_SYSTEM_PROMPT, cell["task"], TASK_PROMPTS)


def build_backend(model: str, cell: dict, api_key: str, base_url: str, max_workers: int) -> APIBackend:
    settings = cell["settings"]
    reg = MODELS[model]
    # Per-model direct endpoint (muse-spark readiness): a registry entry may
    # carry {"base_url", "api_key_env"}; endpoint_for resolves them against the
    # CLI --base-url + OPENROUTER_API_KEY defaults. A direct-endpoint model
    # reads its key from its own env var — a missing key fails loudly HERE
    # (once, before any call) instead of 401-ing n times inside the backend.
    base_url, key_env = endpoint_for(model, default_base_url=base_url)
    if key_env != DEFAULT_API_KEY_ENV:
        api_key = os.environ.get(key_env)
        if not api_key:
            raise SystemExit(f"{key_env} not set (required for {model})")
    extra_body: dict = {}
    # OpenRouter-style endpoints accept a reasoning-effort block; direct vendor
    # endpoints (e.g. OpenAI chat completions for gpt-5.6-sol) reject it and
    # use a top-level reasoning_effort parameter instead.
    reasoning_effort: str | None = None
    if settings["effort"] is not None:
        if reg.get("supports_reasoning_effort", True):
            extra_body["reasoning"] = {"effort": settings["effort"]}
        rev = reg.get("reasoning_effort_values")
        if rev:
            reasoning_effort = rev.get(settings["effort"])
    if (reg["open_weights"] and reg.get("quantization_filter", True)
            and not reg.get("base_url")):
        # Quantization filter is only meaningful for open-weight models (C4);
        # models whose endpoints don't declare a quantization opt out via the
        # registry's quantization_filter flag (the filter 404s them otherwise).
        # The provider block is an OPENROUTER routing option — never sent to a
        # direct vendor endpoint (which would reject the unknown field).
        extra_body["provider"] = {"require_parameters": False,
                                  "quantizations": ["fp8", "bf16", "fp16"]}
    backend_cls = ResponsesBackend if reg.get("responses_endpoint") else APIBackend
    return backend_cls(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_workers=max_workers,
        system_prompt=system_prompt_for(cell),
        extra_body=extra_body or None,
        # raw mode for contract cells (the runner regex-extracts the last
        # "Answer:" line itself); words mode only for the concrete/natural-word
        # s5 facet (contract C1); tokens mode otherwise.
        answer_mode=("raw" if cell["settings"].get("contract")
                     else "words" if cell["facet"] == "s5_concrete" else "tokens"),
        model_name=reg.get("model_name"),
        max_completion_tokens=reg.get("max_completion_tokens", False),
        reasoning_model=reg.get("reasoning_model", False),
        reasoning_effort=reasoning_effort,
        # 30-minute request timeout: 16k-token reasoning cells (kimi at depth 32,
        # sonnet with a 16k thinking budget) legitimately exceed the openai
        # client's default 600s, which otherwise times out and re-bills the
        # generation once per retry (v2 pilot 2026-07-08).
        timeout=1800.0,
    )


# --- cell execution ------------------------------------------------------------------

def _cell_spec(cell: dict):
    """The TaskSpec this cell runs (single source of truth:
    factworld.benchmark.spec_for_cell — breadth rungs via settings["breadth"]
    (scaled(k=2*B, recall_pool=B)), fixed-k chains via settings["k_fixed"]
    (chain_v2.scaled(k=k_fixed)), the chain_nowrap staircase k=2d+1 otherwise)."""
    s = cell["settings"]
    return spec_for_cell(cell["task"], cell["length"],
                         breadth=s.get("breadth"), k_fixed=s.get("k_fixed"))


def _run_s5_cell(backend, cell, n) -> tuple[dict, list[dict], list[str]]:
    settings = cell["settings"]
    triples = S5.gen_examples(cell["length"], n, framing=settings["rendering"])
    prompts = [user for _sys, user, _gold in triples]
    golds = [gold for _sys, _user, gold in triples]
    preds = backend.generate(prompts, max_new_tokens=settings["max_new_tokens"],
                             stop_at=settings["stop_at"])
    scores = [S5.score(p, g) for p, g in zip(preds, golds)]
    metrics = {
        "relaxed": sum(s["relaxed"] for s in scores) / n,
        "exact": None,
        "contains": sum(s["contains"] for s in scores) / n,
        "last_n": None,
    }
    examples = [{"gold": g, "pred": p, "relaxed": s["relaxed"]}
                for g, p, s in zip(golds, preds, scores)]
    return metrics, examples, preds


def _run_zero_budget_cell(backend, cell, n) -> tuple[dict, list[dict], list[str], float]:
    """One answer-contract cell (zero_budget legs None / binding_only / replicate,
    plus the single-answer contract facets recall_load and chain_instant).

    Prompts end with the hard contract line (family-matched for plain cells —
    see CONTRACT_LINES_BY_FAMILY); the backend runs in ``raw`` answer
    mode and the prediction is the span after the LAST "Answer:" line of the
    visible output (empty when no line parses — the contract miss also counts as
    an empty pred). Scoring:
      leg None / replicate  — the unmodified task prompt, canonical relaxed
                              on the extracted span (plus the diagnostic scorers).
                              The replicate leg is INTENTIONALLY the identical
                              prompt to the plain cell: a test-retest arm whose
                              |delta| vs the plain cell is the quoted run-to-run
                              noise bar (review F6 — do not "fix" this into a
                              distinct measurement).
      binding_only          — query rewritten to ask only for the holder
                              (experiment_autoregressive protocol); score = the
                              extracted span's FIRST content token is the holder
                              (relaxed-style prefix, NOT membership — a membership
                              scorer has a 100% false-positive rate against a
                              holder-dump reply "Answer: g0 g1 g2 ...", which kimi
                              exploited live at effort=none).
      scaffolded            — the resolved holder is INJECTED into the prompt
                              ("(the holder is gX)", experiment_autoregressive
                              scaffold_prompt), so only the recall leg remains
                              (recall|holder — the E1b upper-bound leg). Gold =
                              the value alone; same prefix-commit scoring as
                              binding_only (a value dump must not score).
    Returns (metrics, examples, preds, contract_rate).
    """
    settings = cell["settings"]
    spec = _cell_spec(cell)  # breadth rungs: scaled(k=2*B, recall_pool=B)
    examples_in = TK.generate(spec, "test", n=n, length=cell["length"])
    leg = settings["leg"]
    if leg == "binding_only":
        rewritten = [binding_prompt(e, spec.name) for e in examples_in]
        # binding_prompt returns the UNMODIFIED (prompt, answer) for task names
        # outside its allowlist — a silent fallback that would mislabel the leg
        # as binding while measuring the full composite. Fail loudly instead.
        for e, (p, g) in zip(examples_in, rewritten):
            if (p, g) == (e.prompt, e.answer):
                raise ValueError(
                    f"binding_prompt did not rewrite the query for task "
                    f"{spec.name!r} (is it missing from its task allowlist?)")
        base_prompts = [p for p, _g in rewritten]
        golds = [g for _p, g in rewritten]
        contract_line = CONTRACT_LINE_BINDING
    elif leg == "scaffolded":
        base_prompts, golds = [], []
        for e in examples_in:
            p = scaffold_prompt(e, spec.name)
            # scaffold_prompt is a silent no-op outside its task allowlist (or when
            # meta['holder'] is missing) — that would mislabel the full composite as
            # the recall-given-holder leg. Fail loudly instead (binding_only lesson).
            if p == e.prompt:
                raise ValueError(
                    f"scaffold_prompt did not inject the holder for task {spec.name!r} "
                    f"(missing from its allowlist, or meta['holder'] absent?)")
            gold_ct = TK.content_tokens(e.answer)
            if len(gold_ct) < 2:
                raise ValueError(
                    f"scaffolded leg needs a 2-token (holder, value) answer, got {e.answer!r}")
            base_prompts.append(p)
            golds.append(f"{gold_ct[1]}.")   # gold = the VALUE alone (recall|holder)
        contract_line = CONTRACT_LINE_VALUE
    elif leg in (None, "replicate"):
        base_prompts = [e.prompt for e in examples_in]
        golds = [e.answer for e in examples_in]
        contract_line = CONTRACT_LINES_BY_FAMILY[spec.family]
    else:
        raise ValueError(f"unknown zero_budget leg {leg!r}")
    prompts = [f"{p}\n{contract_line}" for p in base_prompts]
    raw = backend.generate(prompts, max_new_tokens=settings["max_new_tokens"],
                           stop_at=settings["stop_at"])
    spans = [extract_contract_answer(t) for t in raw]
    preds = [s if s is not None else "" for s in spans]
    contract_rate = sum(1 for s in spans if s is not None) / n

    rel: list[int] = []
    diag = {"exact": [], "contains": [], "last_n": []}
    for e, gold, pred in zip(examples_in, golds, preds):
        if leg == "binding_only":
            # Prefix match, not membership: the span must COMMIT to the holder as
            # its first content token, so a dump of every candidate holder on the
            # answer line scores like the guess it is (kimi exploited membership
            # scoring live: 12/50 degenerate all-giver loops, all scored 1).
            holder = e.meta.get("holder")
            rel.append(int(holder is not None
                           and TK.content_tokens(pred)[:1] == [holder]))
        elif leg == "scaffolded":
            # Prefix-commit on the value (gold is "<value>."), tolerating ONE leading
            # echo of the INJECTED holder: the prompt both injects "(the holder is gX)"
            # and asks the composite question, so a legitimate reply is often
            # "gX vY" (measured live: opus answered the correct value 100/100 in that
            # shape and prefix-only scored it 0.05). Only the exact injected holder is
            # stripped; a value dump still commits to its (wrong) first value, and a
            # wrong-holder echo is not excused.
            pred_ct = TK.content_tokens(pred)
            if pred_ct[:1] == [e.meta.get("holder")]:
                pred_ct = pred_ct[1:]
            rel.append(int(pred_ct[:1] == TK.content_tokens(gold)[:1]))
        else:
            pred_n, gold_n = Renderer.normalize(pred), Renderer.normalize(gold)
            rel.append(TK.score_relaxed(pred_n, gold_n))
            diag["exact"].append(TK.score_exact(pred_n, gold_n))
            diag["contains"].append(TK.score_contains(pred_n, gold_n))
            diag["last_n"].append(TK.score_last_n(pred_n, gold_n))
    metrics = {"relaxed": sum(rel) / n}
    for name in ("exact", "contains", "last_n"):
        metrics[name] = (sum(diag[name]) / n) if diag[name] else None
    examples = [{"gold": g, "pred": p, "relaxed": r}
                for g, p, r in zip(golds, preds, rel)]
    return metrics, examples, preds, contract_rate


def _run_task_cell(backend, cell, n) -> tuple[dict, list[dict], list[str]]:
    """Canonical-task cells (chain_nowrap / sanity) via the same ``evaluate_task``
    path as scripts/eval_openrouter_grid.py.

    Chain cells WITHOUT settings["k_fixed"] run the STAIRCASE: depth d over a
    (2d+1)-cycle — never wraps, and the backward walk costs d+1 hops so no
    direction is cheaper than the measured depth (k=d+2 would leave gold a
    constant 2 reverse lookups from start). With settings["k_fixed"] the cycle
    size is pinned (chain_v2.scaled(k=k_fixed)): d hops at FIXED breadth, the
    composition-as-axis arm."""
    settings = cell["settings"]
    spec = _cell_spec(cell)
    result = evaluate_task(
        backend, spec, split="test", n=n, length=cell["length"],
        max_new_tokens=settings["max_new_tokens"], n_shot=settings["n_shot"],
        stop_at=settings["stop_at"],
    )
    metrics = {name: result["metrics"][name]["overall"]
               for name in ("relaxed", "exact", "contains", "last_n")}
    preds = [pred for _p, _g, pred, _ok in result["examples"]]
    examples = [{"gold": gold, "pred": pred, "relaxed": ms["relaxed"]}
                for (_prompt, gold, pred, _ok), ms in zip(result["examples"],
                                                          result["example_metrics"])]
    return metrics, examples, preds


def _attach_example_meta(examples: list[dict], ex_meta: list[dict] | None) -> None:
    """Write per-call ``{ctok, rtok, finish}`` into each example record (in place).

    ``ex_meta`` comes from ``APIBackend.pop_example_meta`` and is order-aligned
    with the prompts (hence with ``examples``); backends without per-example
    metadata (FunctionBackend and friends) get explicit Nones so the example
    schema stays uniform.
    """
    if ex_meta is not None and len(ex_meta) == len(examples):
        for ex, m in zip(examples, ex_meta):
            ex["ctok"] = m["completion_tokens"]
            ex["rtok"] = m["reasoning_tokens"]
            ex["finish"] = m["finish_reason"]
    else:
        for ex in examples:
            ex["ctok"] = ex["rtok"] = ex["finish"] = None


def _run_attempt(backend, cell: dict, n: int, max_new_tokens: int | None = None,
                 model: str | None = None) -> dict:
    """One generation pass over a cell (``max_new_tokens`` overrides the planned
    budget for an escalation rerun). The backend is wrapped in a per-attempt
    ``CostGuardBackend`` (token budget CELL_BUDGET_FACTOR * n * max_new_tokens,
    plus the per-cell DOLLAR cap from ``cell_dollar_cap`` for expensive models —
    max($2.50, the attempt's nominal n * max_new_tokens completion spend)).
    Returns metrics/examples/diagnostics, the popped call meta, and the
    finish=length rate that drives escalation."""
    run_cell = cell
    if max_new_tokens is not None:
        run_cell = {**cell, "settings": {**cell["settings"], "max_new_tokens": max_new_tokens}}
    attempt_budget = run_cell["settings"]["max_new_tokens"]
    guard = CostGuardBackend(
        backend, CELL_BUDGET_FACTOR * n * attempt_budget,
        budget_usd=(cell_dollar_cap(model, n, attempt_budget) if model else None),
        completion_price_per_M=MODELS.get(model, {}).get("completion_price_per_M", 0.0))
    contract_rate = None
    if run_cell["settings"].get("contract"):
        metrics, examples, preds, contract_rate = _run_zero_budget_cell(guard, run_cell, n)
    elif run_cell["facet"] in S5_FACETS:
        metrics, examples, preds = _run_s5_cell(guard, run_cell, n)
    else:
        metrics, examples, preds = _run_task_cell(guard, run_cell, n)

    _attach_example_meta(examples, guard.pop_example_meta())
    # Truncated (finish=length) outputs are not committed answers: score them as empty.
    # This prevents a mid-reasoning scratchpad's first token from being scored as a
    # real answer (review: the v1 s5_chain pilot had a false positive this way).
    truncated = 0
    for i, ex in enumerate(examples):
        if ex.get("finish") == "length":
            ex["pred"] = ""
            ex["relaxed"] = 0
            if i < len(preds):
                preds[i] = ""
            truncated += 1
    if truncated:
        metrics["relaxed"] = sum(ex["relaxed"] for ex in examples) / max(1, len(examples))
    meta = guard.pop_call_meta()  # already a fresh dict: the escalation path mutates usage
    empty_rate = sum(1 for p in preds if not p.strip()) / max(1, len(preds))
    length_rate = (sum(1 for ex in examples if ex["finish"] == "length")
                   / max(1, len(examples)))
    diagnostics = {
        "empty_rate": round(empty_rate, 4),
        "truncated_rate": round(truncated / max(1, len(examples)), 4),
        "api_errors": meta["errors"],
        # finish=error calls are NOT exception-path api_errors (review F8: 12
        # finish=error calls sat invisible at api_errors=0) — count them apart.
        "finish_errors": meta["finish_reasons"].get("error", 0),
        "finish_reasons": meta["finish_reasons"],
        "cost_aborted": guard.cost_aborted,
    }
    if guard.cost_aborted:
        diagnostics["calls_completed"] = guard.calls_completed
        # which guard tripped: "ctok" (3x token envelope) or "usd" (per-cell
        # dollar cap, expensive models only — see cell_dollar_cap).
        diagnostics["cost_abort_reason"] = guard.abort_reason
    if contract_rate is not None:
        # Zero-budget cleanliness diagnostics (owner gate for the battery): is the
        # effort=none arm actually reasoning-free and contract-compliant here?
        ctoks = [ex["ctok"] for ex in examples]
        rtoks = [ex["rtok"] for ex in examples if ex["rtok"] is not None]
        diagnostics["contract_rate"] = round(contract_rate, 4)
        diagnostics["covert_cot_rate"] = round(
            sum(1 for c in ctoks if c is not None and c > COVERT_COT_CTOK_THRESHOLD)
            / max(1, len(ctoks)), 4)
        # rtok_any_rate (was rtok_leak_rate, renamed per review F9: "leak" implied
        # the renderer's per-call magnitude dagger, but this is the fraction of
        # calls with ANY reasoning tokens); rtok_mean_per_call is the magnitude.
        diagnostics["rtok_any_rate"] = round(
            sum(1 for r in rtoks if r) / max(1, len(rtoks)), 4)
        diagnostics["rtok_mean_per_call"] = (
            round(sum(rtoks) / len(rtoks), 2) if rtoks else None)
    return {"metrics": metrics, "examples": examples, "diagnostics": diagnostics,
            "meta": meta, "length_rate": length_rate,
            "max_new_tokens": run_cell["settings"]["max_new_tokens"]}


def execute_cell(backend, model: str, cell: dict, *, n: int, run_id: str,
                 git_commit: str) -> dict:
    """Run one cell and return the C3-conformant history record (not yet written).

    Works with any ``ModelBackend``; per-call diagnostics/usage come from
    ``pop_call_meta``/``pop_example_meta`` when the backend provides them
    (APIBackend), else zeros/Nones. Every attempt runs behind a per-attempt
    ``CostGuardBackend`` spend guard (diagnostics.cost_aborted flags a tripped
    cell — see ``_run_attempt``).

    Contract cells get model-aware cutoff handling (C3, iterated per review
    F2/F3): while the finish=length rate exceeds LENGTH_ESCALATION_THRESHOLD,
    the cell reruns at the next ESCALATION_BUDGETS entry (96 -> 512 -> 2048, at
    most 2 reruns; a cost-aborted attempt stops escalation). The FIRST attempt
    at the planned budget is the CANONICAL number for the renderer; every
    attempt (including the first and the final) is recorded IN FULL — metrics,
    diagnostics, per-example ctok/rtok/finish — under ``escalation.attempts``
    (``escalation.first_attempt`` keeps the legacy summary alias of attempt 0
    for renderer compat). The record's top-level metrics/examples are the LAST
    attempt's, marked escalated=true so the renderer treats them as the
    diagnostic view, and usage/cost cover ALL attempts. The record's
    ``settings`` stay the planned ones so the resume key is unchanged.
    """
    t0 = time.time()
    attempt = _run_attempt(backend, cell, n, model=model)
    attempts = [attempt]
    for budget in ESCALATION_BUDGETS:
        if not (cell["settings"].get("contract")
                and attempt["length_rate"] > LENGTH_ESCALATION_THRESHOLD
                and not attempt["diagnostics"]["cost_aborted"]):
            break
        attempt = _run_attempt(backend, cell, n, max_new_tokens=budget, model=model)
        attempts.append(attempt)
    escalation = None
    if len(attempts) > 1:
        attempt_recs = [{
            "attempt": i,
            "max_new_tokens": a["max_new_tokens"],
            "relaxed": a["metrics"]["relaxed"],
            "length_rate": round(a["length_rate"], 4),
            "metrics": a["metrics"],
            "diagnostics": a["diagnostics"],
            # snapshot: the spend-honesty pass below mutates the LAST attempt's
            # meta usage in place, and this entry must keep per-attempt numbers.
            "usage": dict(a["meta"]["usage"]),
            # per-example data for EVERY attempt: the renderer publishes the
            # FIRST attempt as canonical (review F2), so its examples must be
            # complete here, not just the last attempt's at the top level.
            "examples": a["examples"],
        } for i, a in enumerate(attempts)]
        escalation = {
            "max_new_tokens": attempt["max_new_tokens"],
            "length_rate": round(attempt["length_rate"], 4),
            "attempts": attempt_recs,
            # legacy alias (pre-attempts renderer schema): summary of attempt 0.
            "first_attempt": {k: attempt_recs[0][k] for k in
                              ("max_new_tokens", "relaxed", "length_rate",
                               "diagnostics", "usage")},
        }
        # Spend honesty: the published usage/cost cover every attempt; served
        # models/providers are the first-seen union across them.
        for prev in attempts[:-1]:
            for key in attempt["meta"]["usage"]:
                attempt["meta"]["usage"][key] += prev["meta"]["usage"][key]
        for key in ("served_models", "providers"):
            attempt["meta"][key] = list(dict.fromkeys(
                [m for a in attempts for m in a["meta"][key]]))
    elapsed = time.time() - t0

    meta = attempt["meta"]
    reg = MODELS.get(model, {})
    usage = meta["usage"]
    cost = (usage["prompt_tokens"] / 1e6 * reg.get("prompt_price_per_M", 0.0)
            + usage["completion_tokens"] / 1e6 * reg.get("completion_price_per_M", 0.0))
    record = {
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "suite_version": TK.SUITE_VERSION,
        # the resume-key version component (per-spec stream version, NOT the
        # suite version — see stream_version/history_keys)
        "stream_version": stream_version(cell["task"]),
        "model": model,
        "served_models": meta["served_models"],
        "providers": meta["providers"],
        "facet": cell["facet"],
        "task": cell["task"],
        "length": cell["length"],
        "n": n,
        # The PLANNED settings (not the escalated budget): the resume key must
        # keep matching this cell on the next run.
        "settings": dict(cell["settings"]),
        "metrics": attempt["metrics"],
        "diagnostics": attempt["diagnostics"],
        "usage": {**usage, "cost_usd_est": round(cost, 4)},
        "elapsed_s": round(elapsed, 2),
        "escalated": escalation is not None,
        # gold/pred/relaxed + per-call ctok/rtok/finish — NO prompt text (prompts
        # are deterministic and regenerable; keeping them out of history reduces
        # the contamination surface).
        "examples": attempt["examples"],
    }
    if escalation is not None:
        record["escalation"] = escalation
    return record


def append_record(history_path: str, record: dict) -> None:
    """Crash-safe single-line JSONL append (mkdir -p the parent first)."""
    parent = os.path.dirname(os.path.abspath(history_path))
    os.makedirs(parent, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# --- reporting ------------------------------------------------------------------------

def _arm_label(cell: dict) -> str:
    s = cell["settings"]
    parts = [f"effort={s['effort'] or 'default'}"]
    if s["leg"]:
        parts.append(f"leg={s['leg']}")
    if s["rendering"]:
        parts.append(f"rendering={s['rendering']}")
    if s.get("breadth"):  # non-canonical pool rung (sentinel-dropped at B=16)
        parts.append(f"B={s['breadth']}")
    if s.get("k_fixed"):  # fixed-breadth chain (vs the k=2d+1 staircase)
        parts.append(f"k_fixed={s['k_fixed']}")
    return " ".join(parts)


def print_plan(plan, done, assumed_output_tokens, force, canary):
    """Dry-run: per-model cell tables with per-cell/total cost estimates."""
    grand_cost = grand_calls = grand_cells = grand_skipped = 0
    for model, cells in plan.items():
        reg = MODELS[model]
        print(f"\n>>> {model} ({reg['tier']})")
        print(f"  {'facet':<17} {'task':<19} {'L':>4} {'n':>4} {'arm':<42} {'est_$':>8}")
        model_cost = model_calls = model_skipped = 0
        for cell in cells:
            est = cost_estimate(model, [cell], assumed_output_tokens)
            skip = should_skip(model, cell, done, force, canary)
            marker = "  SKIP (in history)" if skip else ""
            print(f"  {cell['facet']:<17} {cell['task']:<19} {cell['length']:>4} "
                  f"{cell['n']:>4} {_arm_label(cell):<42} {est['cost_usd']:>8.2f}{marker}")
            if skip:
                model_skipped += 1
            else:
                model_cost += est["cost_usd"]
                model_calls += est["calls"]
        n_run = len(cells) - model_skipped
        print(f"  -- {n_run} cells to run ({model_skipped} skipped by resume), "
              f"{model_calls} calls, est ${model_cost:.2f}")
        grand_cost += model_cost
        grand_calls += model_calls
        grand_cells += n_run
        grand_skipped += model_skipped
    print(f"\nTOTAL: {grand_cells} cells to run ({grand_skipped} skipped by resume), "
          f"{grand_calls} calls, est ${grand_cost:.2f} "
          f"(assumed {assumed_output_tokens} output tokens per reasoning call, "
          f"{','.join(REASONING_EFFORTS)} arms)")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --- entry point ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run the recurring frontier benchmark (contracts C3/C4).")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS),
                    help="Registry model slugs to run (default: all).")
    ap.add_argument("--facets", nargs="+", default=None, choices=list(FACETS),
                    help="Facets to run (default: all, including sanity rows). "
                         "chain_nowrap is a STAIRCASE: depth d runs "
                         "chain_v2.scaled(k=2*d+1), so breadth grows with depth.")
    ap.add_argument("--n-scale", type=float, default=1.0, dest="n_scale",
                    help="Scouting multiplier applied to each facet's n (floor 5).")
    ap.add_argument("--lengths", nargs="+", type=int, default=None,
                    help="Keep only cells at these lengths/depths (default: all).")
    ap.add_argument("--budget-override", action="append", default=None,
                    dest="budget_override", metavar="FACET:LENGTH:BUDGET",
                    help="Replace matching cells' max_new_tokens (repeatable). The "
                         "budget is part of the settings hash, so overridden cells "
                         "get fresh resume keys and re-run; the renderer's latest-ts "
                         "dedup then displays the raised-budget record.")
    ap.add_argument("--run-id", default=None, dest="run_id",
                    help="Run identifier (default: bench_<UTC stamp>).")
    ap.add_argument("--history", default=os.path.join(REPO, "results", "benchmark", "history.jsonl"),
                    help="History JSONL path (contract C3).")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Print the full cell plan + cost estimates, no API calls.")
    ap.add_argument("--max-workers", type=int, default=8, dest="max_workers",
                    help="Concurrent API calls per cell (default: 8).")
    ap.add_argument("--canary", action="store_true",
                    help=f"Force-rerun {CANARY_MODEL} cells even if present in history.")
    ap.add_argument("--force", action="store_true",
                    help="Force-rerun every selected cell (ignore resume).")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1", dest="base_url",
                    help="OpenRouter-compatible API base URL.")
    ap.add_argument("--assumed-output-tokens", type=int, default=2000, dest="assumed_output_tokens",
                    help="Per-call completion-token assumption for reasoning cells (cost estimate).")
    a = ap.parse_args()

    run_id = a.run_id or f"bench_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    plan = build_plan(a.models, a.facets, a.n_scale, lengths=a.lengths,
                      budget_overrides=parse_budget_overrides(a.budget_override))
    done = history_keys(a.history)
    n_done = sum(1 for m, cells in plan.items() for c in cells
                 if should_skip(m, c, done, a.force, a.canary))
    print(f"run_id={run_id} history={a.history} "
          f"({len(done)} keys in history; {n_done} planned cells already present)")

    if a.dry_run:
        print_plan(plan, done, a.assumed_output_tokens, a.force, a.canary)
        return

    api_key = os.environ.get(DEFAULT_API_KEY_ENV)
    if not api_key and any(endpoint_for(m)[1] == DEFAULT_API_KEY_ENV for m in plan):
        # only required when some selected model actually runs via OpenRouter;
        # direct-endpoint models resolve their own key env in build_backend.
        raise SystemExit(f"{DEFAULT_API_KEY_ENV} not set")
    git_commit = _git_commit()

    total_cost = 0.0
    for model, cells in plan.items():
        print(f"\n>>> {model} ({MODELS[model]['tier']})", flush=True)
        skipped = 0
        for cell in cells:
            if should_skip(model, cell, done, a.force, a.canary):
                skipped += 1
                continue
            tag = f"{cell['facet']}/{cell['task']}@L{cell['length']} [{_arm_label(cell)}]"
            try:
                backend = build_backend(model, cell, api_key, a.base_url, a.max_workers)
                rec = execute_cell(backend, model, cell, n=cell["n"],
                                   run_id=run_id, git_commit=git_commit)
            except Exception:  # noqa: BLE001 — one bad cell must not kill the run
                print(f"  {tag}: FAILED (no record written)", flush=True)
                traceback.print_exc()
                continue
            append_record(a.history, rec)
            done.add(cell_key(model, cell))
            d, u = rec["diagnostics"], rec["usage"]
            total_cost += u["cost_usd_est"]
            extras = ""
            if "contract_rate" in d:
                extras = (f" contract={d['contract_rate']:.2f} "
                          f"covert={d['covert_cot_rate']:.2f} rtok_any={d['rtok_any_rate']:.2f}")
            if rec["escalated"]:
                n_att = len(rec["escalation"]["attempts"])
                extras += (f" ESCALATED->{rec['escalation']['max_new_tokens']} "
                           f"({n_att} attempts; first attempt stays canonical)")
            print(f"  {tag}: relaxed={rec['metrics']['relaxed']:.3f} "
                  f"empty={d['empty_rate']:.2f} err={d['api_errors']} "
                  f"rtok={u['reasoning_tokens']} ${u['cost_usd_est']:.2f} "
                  f"[{rec['elapsed_s']:.1f}s]{extras}", flush=True)
            if d["empty_rate"] > 0.5:
                print(f"  !!! WARNING {tag}: empty_rate={d['empty_rate']:.2f} > 0.5 — "
                      f"truncation suspect (check finish_reasons={d['finish_reasons']}); "
                      f"record kept.", flush=True)
            if d["finish_errors"] > 0:
                print(f"  !!! WARNING {tag}: {d['finish_errors']} calls finished with "
                      f"finish_reason=error (api_errors={d['api_errors']} — these are "
                      f"NOT exception-path errors); scores for those calls are not "
                      f"real measurements. Record kept.", flush=True)
            if d["cost_aborted"]:
                why = ("its per-cell DOLLAR cap (cell_dollar_cap: max($2.50, "
                       "nominal n*max_new_tokens completion spend))"
                       if d.get("cost_abort_reason") == "usd" else
                       f"{CELL_BUDGET_FACTOR}x its n*max_new_tokens ctok envelope")
                print(f"  !!! WARNING {tag}: COST GUARD tripped — cell exceeded "
                      f"{why}; only {d['calls_completed']}/{rec['n']} calls "
                      f"submitted, the rest recorded as finish=cost_aborted.",
                      flush=True)
        print(f"  -- done ({skipped} cells skipped by resume)", flush=True)
    print(f"\nrun {run_id} complete: total est cost ${total_cost:.2f}; history -> {a.history}")


if __name__ == "__main__":
    main()
