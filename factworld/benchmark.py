"""Registry for the recurring frontier-model benchmark (contract C4).

This module is the single source of truth for WHAT the recurring benchmark runs:

  - ``MODELS``: OpenRouter slug -> tier, per-million pricing, open_weights flag.
  - ``TIERS``: how much of the reasoning-effort sweep each tier gets.
  - ``FACETS``: the scored facets plus the sanity rows — task, lengths/depths,
    default n, and per-facet arm policy (v2 roster 2026-07-08: zero_budget
    answer-contract battery, s5_concrete mid-band, chain_nowrap staircase, sanity).
  - ``arms_for(model_slug)``: the exact list of cell dicts the runner executes.
  - ``settings_hash(cell)``: stable resume key for a cell's settings.
  - ``cost_estimate(model_slug, cells)``: price a cell plan before running it.

Protocol rule (learned 2026-07-05, see results/s5_horizon_recheck_20260705.jsonl and
results/chain_reasoning_pilot*_20260705.*): every reasoning-on cell (effort in
low/medium/high) uses ``max_new_tokens=8192`` and ``stop_at=None`` — smaller budgets
manufactured the published "s5 L64 cliff" and "chain floor" as truncation artifacts.

Effort encoding: ``None`` means "default" — the reasoning parameter is omitted from
the request entirely (non-reasoning models, and any default arm); the string
``"none"`` sends ``{"reasoning": {"effort": "none"}}`` to explicitly disable
chain-of-thought on reasoning-capable models.

Pure stdlib (plus sibling pure-stdlib factworld modules for prompt-size estimates).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

# --- protocol constants ---------------------------------------------------------

REASONING_EFFORTS = ("low", "medium", "high")   # arms that actually think
REASONING_MAX_NEW_TOKENS = 8192                 # protocol rule: never truncate thinking
DEFAULT_MAX_NEW_TOKENS = 2048                   # non-thinking arms (grid-script default)
# zero-budget battery: tight completion budget + hard answer contract. The runner
# escalates a cell up to twice (96 -> 512 -> 2048) while finish=length exceeds 10%
# of its calls; the FIRST attempt (at this budget) stays the canonical number and
# the escalated attempts are marked diagnostics, so a fixed cap never silently
# zeros a verbose model and an escalated budget never inflates the headline.
ZERO_BUDGET_MAX_NEW_TOKENS = 96
# Per-cell spend guard (the grok-build lesson: a pinned generator ignored the
# 16384 cap and emitted ~256k ctok per call, 23/25 calls at d128). A cell's
# cumulative visible completion tokens may not exceed
# CELL_BUDGET_FACTOR * n * max_new_tokens; past that the runner stops submitting
# new calls, records what completed, and flags the cell cost_aborted.
CELL_BUDGET_FACTOR = 3
# per-call visible-completion-token threshold above which an effort=none reply is
# counted as covert in-content CoT (kimi at effort=none averaged ~2762 ctok/call;
# clean contract answers are tens of tokens).
COVERT_COT_CTOK_THRESHOLD = 350

# Cost-estimate assumptions: non-reasoning arms answer in a few tokens; the
# synthetic token-dense prompts (g12/a0/v45) tokenize at roughly 3 chars/token.
NON_REASONING_OUTPUT_TOKENS = 64
CHARS_PER_TOKEN = 3
SYSTEM_PROMPT_EST_TOKENS = 90


# --- registry ---------------------------------------------------------------------

# slug -> tier, OpenRouter pricing (USD per million tokens), open_weights (the
# fp8/bf16/fp16 quantization filter is only meaningful for open-weight models).
MODELS = {
    "anthropic/claude-opus-4.8": {
        "tier": "frontier_pair", "prompt_price_per_M": 5.0,
        "completion_price_per_M": 25.0, "open_weights": False},
    "anthropic/claude-sonnet-5": {
        "tier": "frontier_pair", "prompt_price_per_M": 2.0,
        "completion_price_per_M": 10.0, "open_weights": False},
    "openai/gpt-5.5": {
        "tier": "frontier_pair", "prompt_price_per_M": 5.0,
        "completion_price_per_M": 30.0, "open_weights": False},
    # openai/gpt-5.4 and google/gemini-3.1-pro-preview DROPPED 2026-07-08 (owner
    # decision: one flagship per vendor; Google is pushing flash).
    # no_reasoning_effort: Gemini 3 endpoints reject effort=none outright
    # ("Reasoning is mandatory ... cannot be disabled", 400); effort=minimal is
    # the closest off-arm (0 reasoning tokens on flash).
    "google/gemini-3.5-flash": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 1.5,
        "completion_price_per_M": 9.0, "open_weights": False,
        "no_reasoning_effort": "minimal"},
    # x-ai is UNREPRESENTED (owner decision 2026-07-09): no current xAI endpoint
    # is cleanly measurable on this suite. Mainline grok (4.20 AND 4.3, verified
    # on both): the endpoint bio-safety filter deterministically blocks ~56% of
    # the g/v-token composite prompts (finish_reason=content_filter,
    # SAFETY_CHECK_TYPE_BIO — the token soup reads as gene/variant nomenclature;
    # see results/v2_pilots/pilot2_contract.jsonl). grok-build-0.1 (dropped after
    # one cycle; its chain/s5 records remain in history as an archived model):
    # cannot disable reasoning, its "minimal" emits 4-15k reasoning tokens, and
    # its provider pins reasoning at ~256k tokens ignoring the requested cap
    # (chain d128: 17 truncations + 7 errors + 1 empty stop = zero scoreable
    # completions for ~$11). Re-probe when xAI ships new endpoints.
    "qwen/qwen3.7-max": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 1.25,
        "completion_price_per_M": 3.75, "open_weights": False},
    # drift canary: cheapest full-sweep reasoner, re-run each cycle (--canary).
    # pricing re-verified against https://openrouter.ai/api/v1/models 2026-07-08
    # (was $0.56/$1.76 — stale; live is $0.93/$3.00 per M).
    "z-ai/glm-5.2": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 0.93,
        "completion_price_per_M": 3.0, "open_weights": True},
    "moonshotai/kimi-k2.6": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 0.66,
        "completion_price_per_M": 3.41, "open_weights": True},
    "deepseek/deepseek-v4-pro": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 0.435,
        "completion_price_per_M": 0.87, "open_weights": True},
    # quantization_filter off: no OpenRouter endpoint for this slug declares
    # fp8/bf16/fp16, so the filter 404s ("No endpoints found"); the served
    # provider is recorded per cell instead.
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 0.5,
        "completion_price_per_M": 2.2, "open_weights": True,
        "quantization_filter": False},
    # meta-llama/llama-4-maverick DROPPED 2026-07-07 (owner decision); the
    # non_reasoning tier is currently empty but kept for future roster additions.
    # Candidate additions (noted, NOT added pending a pricing/behavior sanity pass;
    # OpenRouter list prices 2026-07-08): anthropic/claude-fable-5 ($10/$50 per M,
    # newest Anthropic tier), moonshotai/kimi-k2.7-code ($0.74/$3.50 per M).
    # muse spark: not on OpenRouter (watch item).
}

CANARY_MODEL = "z-ai/glm-5.2"

# tier -> reasoning capability + the effort sweep a "dose"-policy facet would get
# (no current facet uses "dose"; the policy machinery is kept for future sweeps).
TIERS = {
    "cheap_reasoner": {"reasoning": True, "dose_efforts": ("none", "low", "medium", "high")},
    "frontier_pair": {"reasoning": True, "dose_efforts": ("none", "high")},
    "non_reasoning": {"reasoning": False, "dose_efforts": (None,)},
}

# Facet definitions. ``efforts`` is a policy resolved per tier by ``_facet_efforts``:
#   "dose" -> the tier's full effort sweep
#   "pair" -> none vs high (the reasoning on/off contrast)
#   "on"   -> high only (facets defined WITH reasoning: s5_concrete, chain_nowrap)
#   "off"  -> none only (reasoning explicitly disabled: zero_budget, sanity)
# Non-reasoning models resolve every policy to the single default arm (effort=None).
# Task "s5" cells are rendered via factworld.s5_concrete (gold is a job word for
# "concrete", a role token for "abstract_stated"); all other tasks are CANONICAL specs.
# Per-cell budget resolution: ``budgets[length]`` raises the floor for thinking arms
# only; a facet-level ``max_new_tokens`` applies to every arm (the zero_budget cap,
# the chain_nowrap 16384 thinking budget); otherwise the protocol defaults apply.
FACETS = {
    # zero-budget battery: reasoning explicitly off, tight completion budget, and a
    # hard answer contract appended to every prompt ("Reply with only one line:
    # Answer: ..."); scoring extracts the LAST "Answer:" line of the visible output
    # so models that emit working before the contract line still score their answer.
    # ``cells`` lists explicit (length, leg) pairs: the plain composite at L16/L64
    # (leg None), the binding_only decomposition leg at L16, and the replicate leg
    # at L16. The replicate leg is a TEST-RETEST duplicate of the plain L16 cell:
    # the runner builds the IDENTICAL prompt on purpose (adversarial review F6 —
    # the old "end_to_end" leg was this same prompt mislabeled as a distinct
    # measurement); its |delta| vs the plain cell is the run-to-run noise bar
    # quoted next to the headline. The leg stays in the settings hash so the
    # replicate cell resumes/re-runs independently of the plain cell.
    # Per-cell diagnostics gate publication: contract_rate, covert_cot_rate,
    # rtok_any_rate / rtok_mean_per_call, finish_errors, cost_aborted, and the
    # iterated finish=length escalation (see the runner).
    # Task is composite_copy_v2 (adversarial-review fix): v1 drew every event
    # uniformly from the 4 active objects, leaving the queried object's last
    # write ~geometric(1/4) from the stream END regardless of L — a one-line
    # recency heuristic scored 0.34@L16/0.21@L64. v2 places the queried
    # object's last write UNIFORMLY over [0.1*L, L-2] (interference from the
    # other objects continues to the end), so L is a genuine binding-depth axis
    # and the recency floor drops to ~chance. The binding_only leg derives from
    # the SAME v2 items via binding_prompt. Task is part of the resume key
    # (cell_key includes cell["task"]), so every v2-task cell gets a fresh key
    # by construction — v1-task history records never satisfy resume for it.
    # composite_copy_v1 is retired (tasks.RETIRED, issue #11): generable for
    # historical reproduction only, never scored.
    "zero_budget": {
        "task": "composite_copy_v2", "n": 100,
        "cells": ((16, None), (64, None), (16, "binding_only"), (16, "replicate")),
        "format_prompt": "composite", "efforts": "off",
        "contract": True, "max_new_tokens": ZERO_BUDGET_MAX_NEW_TOKENS},
    # s5 mid-band with reasoning on (owner decision 2026-07-07): L16-64 saturate for
    # reasoning models under the concrete rendering, so only the discriminating
    # lengths remain. Budgets: reasoning traces scale with the permutation horizon;
    # the shared 8192 cap truncates strong models at L128+ (opus/sonnet
    # finish_reason=length with 0 visible answer at 8192), so both cells get 16384.
    "s5_concrete": {
        "task": "s5", "lengths": (128, 256), "n": 25,
        "rendering": "concrete", "efforts": "on",
        "budgets": {128: 16384, 256: 16384}},
    # no-wrap deep chains, replacing the invalid wrap-era chain_depth facet (its
    # k=6 cycle wrapped at depth >= 6, collapsing gold to nxt^(depth mod 6)).
    # STAIRCASE protocol: each depth d runs chain_v1.scaled(k=2*d+1). k must
    # exceed d (the wrap gate), but k=d+2 would leave its own constant shortcut:
    # on a single complete k-cycle, d forward hops == (k-d) BACKWARD hops, so
    # k=d+2 puts gold always exactly 2 reverse lookups from start. k=2d+1 prices
    # the backward walk at d+1 hops — no direction is cheaper than the measured
    # depth. Breadth (k agents) grows with depth by design; read the axis as
    # "d hops over 2d+1 agents", not d hops at fixed breadth.
    "chain_nowrap": {
        "task": "chain_v1", "lengths": (16, 32, 64, 128), "n": 25,
        "efforts": "on", "max_new_tokens": 16384},
    # sanity rows: cheap positive controls at each task's first eval length.
    "sanity": {
        "tasks": (("recall_copy_v1", 6), ("conflict_v1", 4)), "n": 30,
        "efforts": "off"},
}


def _facet_efforts(policy: str, tier: dict) -> tuple:
    """Resolve a facet's effort policy for a tier (see FACETS docstring)."""
    if not tier["reasoning"]:
        return (None,)  # no reasoning param at all: one default arm
    if policy == "dose":
        return tier["dose_efforts"]
    if policy == "pair":
        return ("none", "high")
    if policy == "on":
        return ("high",)
    if policy == "off":
        return ("none",)
    raise ValueError(f"unknown effort policy {policy!r}")


def _settings(effort, *, rendering=None, format_prompt=None, leg=None,
              max_new_tokens=None, contract=False) -> dict:
    """One cell's settings dict (contract C3 keys, always all present)."""
    reasoning_on = effort in REASONING_EFFORTS
    if max_new_tokens is None:
        max_new_tokens = REASONING_MAX_NEW_TOKENS if reasoning_on else DEFAULT_MAX_NEW_TOKENS
    return {
        "effort": effort,
        "max_new_tokens": max_new_tokens,
        "stop_at": None,
        "rendering": rendering,
        "format_prompt": format_prompt,
        "n_shot": 0,
        "leg": leg,
        # zero-budget battery: hard "Answer: ..." contract line appended to every
        # prompt + last-Answer-line extraction (part of the resume key).
        "contract": contract,
    }


def arms_for(model_slug: str) -> list[dict]:
    """The full cell plan for one model: list of {facet, task, length, n, settings}.

    Tier policy: "dose"-policy facets would give cheap_reasoner the full effort
    sweep and frontier_pair none+high only (no current facet uses "dose");
    non_reasoning gets a single default arm per (facet, task, length, leg) and
    never receives a reasoning parameter.

    Facets listed in the model's ``skip_facets`` registry field are dropped here
    (structurally — not by CLI discipline): grok-build's "minimal" is not
    minimal, so its zero_budget off-arm is known-contaminated and never planned.
    """
    reg = MODELS[model_slug]
    tier = TIERS[reg["tier"]]
    cells: list[dict] = []
    skip = set(reg.get("skip_facets", ()))
    for facet_name, fc in FACETS.items():
        if facet_name in skip:
            continue
        if "cells" in fc:
            # explicit (length, leg) pairs (zero_budget mixes plain + leg cells)
            triples = tuple((fc["task"], L, leg) for L, leg in fc["cells"])
        else:
            tasks = fc.get("tasks") or tuple((fc["task"], L) for L in fc["lengths"])
            legs = fc.get("legs", (None,))
            triples = tuple((t, L, leg) for t, L in tasks for leg in legs)
        for effort in _facet_efforts(fc["efforts"], tier):
            # Models that cannot disable reasoning substitute their closest
            # off-arm (e.g. Gemini 3: "minimal"); recorded truthfully in settings.
            if effort == "none":
                effort = reg.get("no_reasoning_effort", "none")
            for task, length, leg in triples:
                budget = fc.get("budgets", {}).get(length)
                if budget is not None and effort not in REASONING_EFFORTS:
                    budget = None  # per-length raises only apply to thinking arms
                if budget is None:
                    budget = fc.get("max_new_tokens")  # facet-level cap, any arm
                cells.append({
                    "facet": facet_name,
                    "task": task,
                    "length": length,
                    "n": fc["n"],
                    "settings": _settings(
                        effort,
                        rendering=fc.get("rendering"),
                        format_prompt=fc.get("format_prompt"),
                        leg=leg,
                        max_new_tokens=budget,
                        contract=fc.get("contract", False),
                    ),
                })
    return cells


def settings_hash(cell: dict) -> str:
    """Stable 10-hex-char hash of a cell's settings (the resume key component).

    Hashes the sorted-key JSON dump of ``cell["settings"]``, so it is invariant to
    dict insertion order and identical after a JSON round-trip through history.jsonl.

    A falsy ``contract`` flag is dropped before hashing: history records written
    before the flag existed (no ``contract`` key) and post-flag non-contract cells
    (``contract: false``) hash identically, so the resume keys of every already-run
    cell survive the schema addition. ``contract: true`` cells hash distinctly.
    """
    settings = {k: v for k, v in cell["settings"].items()
                if k != "contract" or v}
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


# --- cost estimation --------------------------------------------------------------

@lru_cache(maxsize=None)
def _prompt_tokens_est(task: str, length: int, rendering: str | None) -> int:
    """Rough prompt-token count for one example of (task, length, rendering).

    Generates one deterministic example and estimates tokens at CHARS_PER_TOKEN
    (the synthetic g/v/r token soup tokenizes densely). Cached: the dry-run plan
    touches each (task, length) once, not once per model.
    """
    if task == "s5":
        from . import s5_concrete
        sysp, user, _gold = s5_concrete.gen_examples(length, 1, framing=rendering)[0]
        return max(1, (len(sysp) + len(user)) // CHARS_PER_TOKEN)
    from . import tasks as TK
    spec = TK.CANONICAL[task]
    if spec.family == "chain" and length >= spec.k:
        # chain_nowrap staircase: depth d runs over a (2d+1)-cycle — no wrap, and
        # the backward walk costs d+1 hops so neither direction beats depth d
        # (generating at depth >= k raises the wrap validity gate otherwise).
        spec = spec.scaled(k=2 * length + 1)
    ex = TK.generate(spec, "test", n=1, length=length)[0]
    return SYSTEM_PROMPT_EST_TOKENS + max(1, len(ex.prompt) // CHARS_PER_TOKEN)


def cost_estimate(model_slug: str, cells: list[dict], assumed_output_tokens: int = 2000) -> dict:
    """Price a cell plan for one model.

    Reasoning-on cells (effort in REASONING_EFFORTS) are assumed to emit
    ``assumed_output_tokens`` completion tokens per call (thinking included);
    other cells NON_REASONING_OUTPUT_TOKENS. Returns
    ``{calls, prompt_tokens, completion_tokens, cost_usd}``.
    """
    reg = MODELS[model_slug]
    calls = prompt_tokens = completion_tokens = 0
    for cell in cells:
        n = cell["n"]
        s = cell["settings"]
        per_prompt = _prompt_tokens_est(cell["task"], cell["length"], s.get("rendering"))
        per_out = assumed_output_tokens if s["effort"] in REASONING_EFFORTS else NON_REASONING_OUTPUT_TOKENS
        calls += n
        prompt_tokens += n * per_prompt
        completion_tokens += n * per_out
    cost = (prompt_tokens / 1e6 * reg["prompt_price_per_M"]
            + completion_tokens / 1e6 * reg["completion_price_per_M"])
    return {
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 4),
    }
