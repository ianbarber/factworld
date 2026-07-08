"""Registry for the recurring frontier-model benchmark (contract C4).

This module is the single source of truth for WHAT the recurring benchmark runs:

  - ``MODELS``: OpenRouter slug -> tier, per-million pricing, open_weights flag.
  - ``TIERS``: how much of the reasoning-effort sweep each tier gets.
  - ``FACETS``: the five scored facets plus the sanity and floor-control rows —
    task, lengths/depths, default n, and per-facet arm policy.
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
    "openai/gpt-5.4": {
        "tier": "frontier_pair", "prompt_price_per_M": 2.5,
        "completion_price_per_M": 15.0, "open_weights": False},
    # NOTE: preview slug — revisit when the stable gemini-3.1-pro ships.
    # no_reasoning_effort: Gemini 3 endpoints reject effort=none outright
    # ("Reasoning is mandatory ... cannot be disabled", 400); effort=minimal is
    # the closest off-arm (0 reasoning tokens on flash, ~85 on pro).
    "google/gemini-3.1-pro-preview": {
        "tier": "frontier_pair", "prompt_price_per_M": 2.0,
        "completion_price_per_M": 12.0, "open_weights": False,
        "no_reasoning_effort": "minimal"},
    "google/gemini-3.5-flash": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 1.5,
        "completion_price_per_M": 9.0, "open_weights": False,
        "no_reasoning_effort": "minimal"},
    "x-ai/grok-4.3": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 1.25,
        "completion_price_per_M": 2.5, "open_weights": False},
    "qwen/qwen3.7-max": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 1.25,
        "completion_price_per_M": 3.75, "open_weights": False},
    # drift canary: cheapest full-sweep reasoner, re-run each cycle (--canary).
    "z-ai/glm-5.2": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 0.56,
        "completion_price_per_M": 1.76, "open_weights": True},
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
    "meta-llama/llama-4-maverick": {
        "tier": "non_reasoning", "prompt_price_per_M": 0.15,
        "completion_price_per_M": 0.6, "open_weights": True},
}

CANARY_MODEL = "z-ai/glm-5.2"

# tier -> reasoning capability + the dose_response effort sweep it gets.
TIERS = {
    "cheap_reasoner": {"reasoning": True, "dose_efforts": ("none", "low", "medium", "high")},
    "frontier_pair": {"reasoning": True, "dose_efforts": ("none", "high")},
    "non_reasoning": {"reasoning": False, "dose_efforts": (None,)},
}

# Facet definitions. ``efforts`` is a policy resolved per tier by ``_facet_efforts``:
#   "dose" -> the tier's full dose_response sweep
#   "pair" -> none vs high (the reasoning on/off contrast)
#   "on"   -> high only (facets defined WITH reasoning: s5_concrete, chain_depth)
#   "off"  -> none only (reasoning explicitly disabled)
# Non-reasoning models resolve every policy to the single default arm (effort=None).
# Task "s5" cells are rendered via factworld.s5_concrete (gold is a job word for
# "concrete", a role token for "abstract_stated"); all other tasks are CANONICAL specs.
FACETS = {
    "dose_response": {
        "task": "composite_copy_v1", "lengths": (16,), "n": 50,
        "format_prompt": "composite", "efforts": "dose"},
    "composite_length": {
        "task": "composite_copy_v1", "lengths": (16, 64, 128, 512), "n": 30,
        "format_prompt": "composite", "efforts": "pair"},
    # budgets: reasoning traces scale with the permutation horizon; the shared
    # 8192 cap truncates strong models at L128+ (opus/sonnet finish_reason=length
    # with 0 visible answer at 8192), so the two heaviest cells get 16384. A model
    # that still truncates there (empty_rate high) is read as "needs >16k tokens
    # of thinking at that horizon" — the diagnostic column, not the score, tells
    # that story. L4 is omitted: every scouted model scores 1.00 (L16 anchors).
    "s5_concrete": {
        "task": "s5", "lengths": (16, 32, 64, 128, 256), "n": 25,
        "rendering": "concrete", "efforts": "on",
        "budgets": {128: 16384, 256: 16384}},
    # depth 64: kimi and gemini-pro saturate depth 48 in scouting.
    "chain_depth": {
        "task": "chain_v1", "lengths": (4, 8, 12, 16, 24, 32, 48, 64), "n": 30,
        "efforts": "on"},
    "decomposition": {
        "task": "composite_copy_v1", "lengths": (16,), "n": 50,
        "legs": ("binding_only", "end_to_end", "scaffolded"),
        "format_prompt": "composite", "efforts": "off"},
    # sanity rows: cheap positive controls at each task's first eval length.
    "sanity": {
        "tasks": (("recall_copy_v1", 6), ("conflict_v1", 4)), "n": 30,
        "efforts": "off"},
    # floor control: the OLD abstract token rendering with reasoning off — the regime
    # where the pre-2026-07-05 "s5 wall" lived. Kept as a row, excluded from horizons.
    "floor": {
        "task": "s5", "lengths": (16,), "n": 30,
        "rendering": "abstract_stated", "efforts": "off"},
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
              max_new_tokens=None) -> dict:
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
    }


def arms_for(model_slug: str) -> list[dict]:
    """The full cell plan for one model: list of {facet, task, length, n, settings}.

    Tier policy: cheap_reasoner gets the full effort sweep in dose_response,
    frontier_pair gets none+high only, non_reasoning gets a single default arm per
    (facet, task, length) and never receives a reasoning parameter.
    """
    reg = MODELS[model_slug]
    tier = TIERS[reg["tier"]]
    cells: list[dict] = []
    for facet_name, fc in FACETS.items():
        tasks = fc.get("tasks") or tuple((fc["task"], L) for L in fc["lengths"])
        legs = fc.get("legs", (None,))
        for effort in _facet_efforts(fc["efforts"], tier):
            # Models that cannot disable reasoning substitute their closest
            # off-arm (e.g. Gemini 3: "minimal"); recorded truthfully in settings.
            if effort == "none":
                effort = reg.get("no_reasoning_effort", "none")
            for task, length in tasks:
                for leg in legs:
                    budget = fc.get("budgets", {}).get(length)
                    if budget is not None and effort not in REASONING_EFFORTS:
                        budget = None  # per-length raises only apply to thinking arms
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
                        ),
                    })
    return cells


def settings_hash(cell: dict) -> str:
    """Stable 10-hex-char hash of a cell's settings (the resume key component).

    Hashes the sorted-key JSON dump of ``cell["settings"]``, so it is invariant to
    dict insertion order and identical after a JSON round-trip through history.jsonl.
    """
    payload = json.dumps(cell["settings"], sort_keys=True, separators=(",", ":"))
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
    ex = TK.generate(TK.CANONICAL[task], "test", n=1, length=length)[0]
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
