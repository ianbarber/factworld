"""Registry for the recurring frontier-model benchmark (contract C4).

This module is the single source of truth for WHAT the recurring benchmark runs:

  - ``MODELS``: OpenRouter slug -> tier, per-million pricing, open_weights flag.
  - ``TIERS``: how much of the reasoning-effort sweep each tier gets.
  - ``FACETS``: the scored facets plus the sanity rows — task, lengths/depths,
    default n, and per-facet arm policy (v2 roster 2026-07-08: zero_budget
    answer-contract battery, s5_concrete mid-band, chain_nowrap staircase, sanity;
    2026-07-10: recall_load pool-64 instant cell, chain_instant d16 off arm).
  - ``arms_for(model_slug)``: the exact list of cell dicts the runner executes.
  - ``endpoint_for(model_slug)``: the (base_url, api_key_env) a model's backend
    is built against — per-model direct endpoints (the muse-spark slot),
    defaulting to OpenRouter + OPENROUTER_API_KEY.
  - ``settings_hash(cell)``: stable resume key for a cell's settings.
  - ``cost_estimate(model_slug, cells)``: price a cell plan before running it.
  - ``spec_for_cell(task, length, breadth, k_fixed)``: the TaskSpec a cell runs —
    the v3 working-set-breadth rungs (settings["breadth"]: scaled(k=2*B,
    recall_pool=B)) and fixed-k chains (settings["k_fixed"]) resolve here, shared
    by the runner and the cost estimator.
  - ``cell_dollar_cap(model_slug, n, max_new_tokens)``: the per-cell dollar cap
    the runner's cost guard enforces for expensive models.

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

REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")   # arms that actually think
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
# Working-set-breadth axis (v3): a cell's settings may carry ``breadth`` — the pool
# rung B, running the task at CANONICAL[task].scaled(k=2*B, recall_pool=B). The
# anchor is composite_copy_v2 itself (k=32/pool16), so B=16 IS the canonical spec;
# the key is SENTINEL-DROPPED at B=16 (omitted from settings, ignored by
# settings_hash) so every pre-breadth history record's resume key is unchanged.
CANONICAL_BREADTH = 16
# Per-cell DOLLAR cap (in addition to the token-based CostGuard): the token guard
# alone permits CELL_BUDGET_FACTOR (3x) a cell's nominal completion budget, which
# on a frontier thinking cell (e.g. 32768 tokens x n=25 x 3 on opus) is ~$61. For
# models at or above the price threshold the runner also caps a cell's completion
# spend at max(CELL_DOLLAR_CAP_MIN_USD, its NOMINAL budget n*max_new_tokens priced
# at the completion rate) — see cell_dollar_cap.
CELL_DOLLAR_CAP_MIN_USD = 2.50
CELL_DOLLAR_CAP_PRICE_THRESHOLD = 10.0  # completion $/M at or above which the cap applies

# Cost-estimate assumptions: non-reasoning arms answer in a few tokens; the
# synthetic token-dense prompts (g12/a0/v45) tokenize at roughly 3 chars/token.
NON_REASONING_OUTPUT_TOKENS = 64
CHARS_PER_TOKEN = 3
SYSTEM_PROMPT_EST_TOKENS = 90


# --- registry ---------------------------------------------------------------------

# slug -> tier, OpenRouter pricing (USD per million tokens), open_weights (the
# fp8/bf16/fp16 quantization filter is only meaningful for open-weight models).
#
# Endpoint keys (OPTIONAL): an entry may carry {"base_url": str,
# "api_key_env": str, "responses_endpoint": bool} for a model served OFF
# OpenRouter (a direct vendor endpoint). ``endpoint_for`` resolves them,
# defaulting to OpenRouter + OPENROUTER_API_KEY; the runner builds each model's
# backend against the resolved endpoint and skips the OpenRouter-specific
# provider/quantization request options. ``responses_endpoint`` selects
# ``ResponsesBackend`` instead of ``APIBackend`` for endpoints that speak the
# OpenAI Responses API (e.g. Meta Model API /v1/responses).
MODELS = {
    "anthropic/claude-opus-4.8": {
        "tier": "frontier_pair", "prompt_price_per_M": 5.0,
        "completion_price_per_M": 25.0, "open_weights": False},
    "anthropic/claude-sonnet-5": {
        "tier": "frontier_pair", "prompt_price_per_M": 2.0,
        "completion_price_per_M": 10.0, "open_weights": False},
    # Routed directly to the OpenAI API since 2026-07-18 (same vendor serving as
    # the prior OpenRouter route; switched when the OpenRouter account exhausted
    # its credits mid-battery). Same direct-endpoint pattern as gpt-5.6-sol.
    "openai/gpt-5.5": {
        "tier": "frontier_pair", "prompt_price_per_M": 5.0,
        "completion_price_per_M": 30.0, "open_weights": False,
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-5.5",
        "max_completion_tokens": True,
        "reasoning_model": True,
        "supports_reasoning_effort": False,
        "reasoning_effort_values": {"low": "low", "medium": "medium", "high": "high",
                                     "xhigh": "xhigh", "max": "max"}},
    # ADDED 2026-07-12 (issue #15). Pricing verified against
    # https://openrouter.ai/api/v1/models 2026-07-12 ($5/$30 per M; the -pro
    # variant is the same price and NOT what we run). effort=none probe clean:
    # finish=stop, rtok=0, 10 visible ctok, well-formed contract answer
    # (results/probes/new_models_20260712.jsonl).
    # Routed directly to OpenAI; the OpenRouter slug is kept as the registry key
    # for roster consistency, but the literal model name sent to the API is
    # "gpt-5.6-sol" without the provider prefix. It is a reasoning model that
    # rejects max_tokens in favor of max_completion_tokens, does not accept
    # temperature/top_p overrides, and does not accept the OpenRouter-style
    # reasoning-effort extra body.
    "openai/gpt-5.6-sol": {
        "tier": "frontier_pair", "prompt_price_per_M": 5.0,
        "completion_price_per_M": 30.0, "open_weights": False,
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-5.6-sol",
        "max_completion_tokens": True,
        "reasoning_model": True,
        "supports_reasoning_effort": False,
        "reasoning_effort_values": {"low": "low", "medium": "medium", "high": "high",
                                     "xhigh": "xhigh", "max": "max"}},
    # openai/gpt-5.4 and google/gemini-3.1-pro-preview DROPPED 2026-07-08 (owner
    # decision: one flagship per vendor; Google is pushing flash).
    # no_reasoning_effort: Gemini 3 endpoints reject effort=none outright
    # ("Reasoning is mandatory ... cannot be disabled", 400); effort=minimal is
    # the closest off-arm (0 reasoning tokens on flash).
    "google/gemini-3.5-flash": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 1.5,
        "completion_price_per_M": 9.0, "open_weights": False,
        "no_reasoning_effort": "minimal"},
    # ADDED 2026-07-13. Muse Spark 1.1 is served directly by the Meta Model API
    # (not OpenRouter) and speaks the OpenAI Responses API. The endpoint cannot
    # disable reasoning; even effort=minimal emits ~5-9k reasoning tokens per
    # call, so the 96-token instant contract cells are structurally unmeasurable
    # (the model produces no visible answer within the cap). It therefore runs
    # only the thinking facets, like x-ai/grok-4.5. Pricing from Meta's
    # public-preview announcement: $1.25/$4.25 per M (verified 2026-07-13).
    "muse-spark-1.1": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 1.25,
        "completion_price_per_M": 4.25, "open_weights": False,
        "base_url": "https://api.meta.ai/v1",
        "api_key_env": "META_API_KEY",
        "responses_endpoint": True,
        "skip_facets": ("zero_budget", "recall_load", "chain_instant",
                        "sanity", "gap_stability")},
    # x-ai REJOINS via grok-4.5, THINKING FACETS ONLY (probes 2026-07-12,
    # results/probes/new_models_20260712.jsonl; issue #15). History: x-ai was
    # unrepresented 2026-07-09..12 — mainline grok (4.20 AND 4.3) had a
    # bio-safety filter deterministically blocking ~56% of the g/v-token
    # composite prompts (finish_reason=content_filter, SAFETY_CHECK_TYPE_BIO —
    # the token soup reads as gene/variant nomenclature; see
    # results/v2_pilots/pilot2_contract.jsonl), and grok-build-0.1 (dropped
    # after one cycle; archived records remain in history) pinned reasoning at
    # ~256k tokens ignoring caps. grok-4.5 probe outcomes:
    #   - filter CLEAN: 3 composite_copy_v2 contract prompts + 1 chain d16
    #     prompt all finish=stop with well-formed answers (no content_filter).
    #   - NO instant regime: effort=none is rejected 400 ("Reasoning is
    #     mandatory for this endpoint and cannot be disabled") and
    #     effort=minimal is NOT a clean off-arm (547 rtok on an L16 composite —
    #     past the 350-ctok covert-CoT bar; the Gemini-flash "minimal"
    #     substitution does not transfer). Hence skip_facets on every
    #     "off"-policy facet: the answer-contract battery and the sanity rows
    #     are structurally unplanned, and grok-4.5 carries no instant numbers.
    #   - max_tokens does NOT bound reasoning (256-cap call billed 759 ctok,
    #     1024-cap billed 1328; finish=stop both) — but traces self-terminate
    #     (~0.5-1.3k on L16 probes), NOT the grok-build ~256k pinning; the
    #     per-cell CostGuard is the effective spend bound.
    "x-ai/grok-4.5": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 2.0,
        "completion_price_per_M": 6.0, "open_weights": False,
        "skip_facets": ("zero_budget", "recall_load", "chain_instant",
                        "sanity", "gap_stability")},
    "qwen/qwen3.7-max": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 1.25,
        "completion_price_per_M": 3.75, "open_weights": False},
    # drift canary: cheapest full-sweep reasoner, re-run each cycle (--canary).
    # pricing re-verified against https://openrouter.ai/api/v1/models 2026-07-08
    # (was $0.56/$1.76 — stale; live is $0.93/$3.00 per M).
    "z-ai/glm-5.2": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 0.93,
        "completion_price_per_M": 3.0, "open_weights": True},
    # INSTANT FACETS EXCLUDED: kimi-k2.6 emits reasoning tokens on 65-89% of
    # effort=none calls despite the answer contract, and its provider does not
    # enforce the requested token cap, so its instant numbers are explicit upper
    # bounds rather than in-weights measurements. It runs in the thinking regime
    # only, like grok-4.5 and muse-spark-1.1.
    "moonshotai/kimi-k2.6": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 0.66,
        "completion_price_per_M": 3.41, "open_weights": True,
        "skip_facets": ("zero_budget", "recall_load", "chain_instant",
                        "sanity", "gap_stability")},
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
    # OpenRouter list prices re-checked 2026-07-12):
    #   - fablet: NOT YET SHIPPED — only anthropic/claude-fable-5 is listed
    #     ($10/$50 per M, newest Anthropic tier); watch for the smaller variant.
    #   - moonshotai/kimi-k2.7-code ($0.74/$3.50 per M).
    #   - (muse-spark-1.1 was added 2026-07-13 via the Meta Model API; kept
    #     here as provenance that this slot is now live.)
}

CANARY_MODEL = "z-ai/glm-5.2"

# Default API endpoint (every current roster model is served via OpenRouter).
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"


def endpoint_for(model_slug: str, default_base_url: str = DEFAULT_BASE_URL) -> tuple[str, str]:
    """``(base_url, api_key_env)`` for a model's API endpoint.

    Registry entries may carry ``{"base_url": str, "api_key_env": str}`` for a
    model served OFF OpenRouter (a direct vendor endpoint — the muse-spark
    slot: base_url=<vendor endpoint> + api_key_env="MUSE_API_KEY"). Models
    without the keys resolve to ``default_base_url`` (the runner passes its
    --base-url, defaulting to OpenRouter) + OPENROUTER_API_KEY.
    """
    reg = MODELS.get(model_slug) or {}
    return (reg.get("base_url") or default_base_url,
            reg.get("api_key_env") or DEFAULT_API_KEY_ENV)

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
    # The scaffolded leg (issue #11 re-measure, 2026-07-10) completes the E1b decomposition
    # triple on v2 items: query unchanged but the resolved holder is injected into the prompt
    # ("(the holder is gX)", experiment_autoregressive.scaffold_prompt), so only the recall
    # leg remains. Gold = the value; scored prefix-commit like binding_only (membership
    # scoring has a 100% false-positive rate against a value dump). This is a positive-control
    # ceiling row (predicted ~1.0) bought ONCE to anchor the v2 gap definition — the frontier
    # report's "recall|holder 0.98-1.00" currently rests on the archived v1 decomposition
    # facet; exempt from "never buy predicted-ceiling cells" the same way sanity/recall_load
    # are. binding_only@L16 and composed@L16/L64 already exist on v2 for the whole roster
    # (bench_v2_zb2_20260709), so this is the only missing leg.
    "zero_budget": {
        "task": "composite_copy_v2", "n": 100,
        "cells": ((16, None), (64, None), (16, "binding_only"), (16, "replicate"),
                  (16, "scaffolded")),
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
    # recall under load: the recall COMPONENT measured at working-set breadth in
    # the instant regime. The legacy frontier recall evidence is the sanity row
    # only (recall_copy_v1 @L6, pool 6 — near ceiling for this roster), so the
    # composition profile had no under-load recall cell. recall_copy_v1's pool is
    # min(length, k) (tasks._ex_recall: non-memorized recall samples a pool of
    # min(L, #agents) agents), so the pool-64 cell runs at L=64 and spec_for_cell
    # scales the agent pool k up to the length (k=64 -> pool exactly 64; chance
    # 1/64). Instant protocol as zero_budget: effort none, hard one-line answer
    # contract, 96-token cap, same escalation machinery. No breadth settings key:
    # (task recall_copy_v1, L=64) is a distinct cell from the sanity row (L=6),
    # so resume keys are fresh by construction and sanity is byte-identical.
    "recall_load": {
        "task": "recall_copy_v1", "lengths": (64,), "n": 50,
        "efforts": "off", "contract": True,
        "max_new_tokens": ZERO_BUDGET_MAX_NEW_TOKENS},
    # no-wrap deep chains, replacing the invalid wrap-era chain_depth facet (its
    # k=6 cycle wrapped at depth >= 6, collapsing gold to nxt^(depth mod 6)).
    # STAIRCASE protocol: each depth d runs chain_v2.scaled(k=2*d+1). k must
    # exceed d (the wrap gate), but k=d+2 would leave its own constant shortcut:
    # on a single complete k-cycle, d forward hops == (k-d) BACKWARD hops, so
    # k=d+2 puts gold always exactly 2 reverse lookups from start. k=2d+1 prices
    # the backward walk at d+1 hops — no direction is cheaper than the measured
    # depth. Breadth (k agents) grows with depth by design; read the axis as
    # "d hops over 2d+1 agents", not d hops at fixed breadth.
    "chain_nowrap": {
        "task": "chain_v2", "lengths": (16, 32, 64, 128), "n": 25,
        "efforts": "on", "max_new_tokens": 16384},
    # chain d16 INSTANT arm: the within-item regime contrast for recall∘recall
    # composition. Same staircase spec as the chain_nowrap d16 thinking cell
    # (chain_v2.scaled(k=2*16+1=33) via spec_for_cell, same deterministic items
    # and n, chance ~1/33), but reasoning off under the answer contract, so the
    # instant-vs-thinking contrast is within-item. A dedicated facet, not an
    # extra chain_nowrap arm: effort policies are facet-wide and this off arm
    # runs at d16 only (an "off" arm at d32-128 would buy predicted floor cells).
    "chain_instant": {
        "task": "chain_v2", "lengths": (16,), "n": 25,
        "efforts": "off", "contract": True,
        "max_new_tokens": ZERO_BUDGET_MAX_NEW_TOKENS},
    # sanity rows: cheap positive controls at each task's first eval length.
    "sanity": {
        "tasks": (("recall_copy_v1", 6), ("conflict_v1", 4)), "n": 30,
        "efforts": "off"},
    # EXPERIMENTAL (issue #18, owner-approved 2026-07-11): commutative_v1 thinking
    # @L64 across the roster, n=25 (matches the calibration protocol in
    # scripts/experiment_commutative_frontier.py — glm 0.52 / deepseek 0.80 live in
    # results/commutative_frontier/runs.jsonl and are REUSED, not re-bought).
    # Pre-registered promotion bar: >=3 CI-separated tiers -> headline state-stress
    # column; otherwise this stays an experimental report row. No renderer section
    # reads this facet yet (by_facet ignores it), so rendering is unchanged.
    "commutative": {
        "task": "commutative_v1", "lengths": (64,), "n": 25,
        "efforts": "on"},
    # s5_chain — THE headline composite stressor: non-abelian pointer-map state
    # tracking composed with an 8-hop serial dereference (k=16 agents; length =
    # number of swap/cycle events). Runs the distinct_path-gated v3 stream (echo
    # and fixed-hop floors 0, chance 1/16). Protocol: every model at its maximum
    # supported reasoning effort (xhigh; OpenRouter maps down to high where xhigh
    # is unsupported). Per-length budgets are sized so finish=length truncation —
    # scored as wrong — stays a rounding error, not a ranking confound (deepseek/
    # nemotron/glm truncated 16-28% of calls at the old 16-24k budgets).
    # Rendered by render_benchmark.s5_chain_rows (README + report ranking table).
    "s5_chain": {
        "task": "s5_chain_v3", "lengths": (32, 64, 96, 128), "n": 25,
        "efforts": "xhigh", "max_new_tokens": 32768,
        "budgets": {32: 32768, 64: 49152, 96: 65536, 128: 98304}},
    # EXPERIMENTAL (issue #16a, owner-approved 2026-07-11): gap stability — the
    # composed and binding_only legs at a SECOND operating point (L32, instant,
    # contract, n=50) for the gap-interpretable models, to test whether the
    # zero_budget gap ORDERING (binding − composed) holds off the L16 anchor.
    # Same protocol as zero_budget in every other respect; a separate facet so
    # the canonical zero_budget rows/renderer are untouched.
    "gap_stability": {
        "task": "composite_copy_v2", "n": 50,
        "cells": ((32, None), (32, "binding_only")),
        "format_prompt": "composite", "efforts": "off",
        "contract": True, "max_new_tokens": ZERO_BUDGET_MAX_NEW_TOKENS},
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
    if policy == "xhigh":
        return ("xhigh",)
    if policy == "off":
        return ("none",)
    raise ValueError(f"unknown effort policy {policy!r}")


def _settings(effort, *, rendering=None, format_prompt=None, leg=None,
              max_new_tokens=None, contract=False, breadth=None, k_fixed=None) -> dict:
    """One cell's settings dict (contract C3 keys, always all present).

    The v3 breadth/depth extension keys are OPTIONAL and sentinel-dropped at their
    canonical values (``breadth`` at CANONICAL_BREADTH/None, ``k_fixed`` at None):
    they are OMITTED from the dict entirely so canonical cells keep the exact
    settings (and resume keys) of pre-breadth history. When present they are part
    of the settings hash (see settings_hash).

      breadth  — pool rung B: run the task at CANONICAL[task].scaled(k=2*B,
                 recall_pool=B) (composite tasks; B=16 IS canonical
                 composite_copy_v2).
      k_fixed  — fixed-breadth chain: chain_v2.scaled(k=k_fixed) — d hops over a
                 FIXED k-cycle, replacing the staircase k=2d+1 (k_fixed must
                 exceed the depth; tasks.py's wrap gate raises otherwise).
    """
    reasoning_on = effort in REASONING_EFFORTS
    if max_new_tokens is None:
        max_new_tokens = REASONING_MAX_NEW_TOKENS if reasoning_on else DEFAULT_MAX_NEW_TOKENS
    settings = {
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
    if breadth is not None and breadth != CANONICAL_BREADTH:
        settings["breadth"] = breadth
    if k_fixed is not None:
        settings["k_fixed"] = k_fixed
    return settings


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
        # v3 breadth/depth knobs (no current facet sets them, so plans and resume
        # keys are byte-identical): ``breadths`` lists the pool rungs B a facet
        # runs (each cell repeats per rung; the canonical rung's key is
        # sentinel-dropped), ``k_fixed`` pins the chain cycle size instead of the
        # staircase k=2d+1.
        breadths = fc.get("breadths", (CANONICAL_BREADTH,))
        k_fixed = fc.get("k_fixed")
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
                for breadth in breadths:
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
                            breadth=breadth,
                            k_fixed=k_fixed,
                        ),
                    })
    return cells


def settings_hash(cell: dict) -> str:
    """Stable 10-hex-char hash of a cell's settings (the resume key component).

    Hashes the sorted-key JSON dump of ``cell["settings"]``, so it is invariant to
    dict insertion order and identical after a JSON round-trip through history.jsonl.

    Sentinel-dropped keys keep every already-run cell's resume key valid across
    schema additions; the keys hash distinctly whenever they carry a
    NON-canonical value:

      - a falsy ``contract`` flag: history written before the flag existed (no
        ``contract`` key) and post-flag non-contract cells (``contract: false``)
        hash identically; ``contract: true`` cells hash distinctly.
      - ``breadth`` at the canonical pool rung (CANONICAL_BREADTH == 16) or
        falsy: the plan omits the key at canonical B, and an explicit
        ``breadth: 16`` must still hash like pre-breadth history (breadth=16 IS
        canonical composite_copy_v2). Note breadth=16 is TRUTHY — a plain falsy
        drop would not cover it, hence the explicit sentinel.
      - a falsy ``k_fixed``: staircase chain cells (k=2d+1) keep their keys; a
        fixed-k chain cell (``k_fixed: 257``) hashes distinctly.
    """
    _drop = {
        "contract": lambda v: not v,
        "breadth": lambda v: not v or v == CANONICAL_BREADTH,
        "k_fixed": lambda v: not v,
    }
    settings = {k: v for k, v in cell["settings"].items()
                if k not in _drop or not _drop[k](v)}
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


# --- spec resolution (single source of truth for the runner + cost estimator) ------

def spec_for_cell(task: str, length: int, breadth: int | None = None,
                  k_fixed: int | None = None):
    """The TaskSpec a cell actually runs (shared by the runner's cell execution
    and ``_prompt_tokens_est`` so prompts and prices never diverge).

      - ``breadth`` (pool rung B, composite tasks): CANONICAL[task].scaled(
        k=2*B, recall_pool=B). Anchored so B=CANONICAL_BREADTH (16) resolves to
        the canonical composite_copy_v2 knobs (k=32/pool16) — scaling at the
        canonical rung is a no-op by construction.
      - ``k_fixed`` (chain family): chain_v2.scaled(k=k_fixed) — d hops over a
        FIXED k-cycle (composition at fixed breadth). tasks.py's wrap gate
        raises at generation time if k_fixed <= depth.
      - chain without k_fixed: the no-wrap STAIRCASE k=2*length+1 when the depth
        reaches the spec's cycle (breadth grows with depth by design).
      - non-memorized recall (recall_copy_v1): the pool is min(length, k)
        (tasks._ex_recall), so a length past the spec's agent count scales k up
        to the length — pool == L exactly (the recall_load facet's pool-64 cell
        is L=64 -> k=64). Lengths within the canonical k (the sanity row's L=6
        < k=8) resolve to the canonical spec unchanged.
    """
    from . import tasks as TK
    spec = TK.CANONICAL[task]
    if breadth:
        spec = spec.scaled(k=2 * breadth, recall_pool=breadth)
    if spec.family == "recall" and not spec.memorized_recall and length > spec.k:
        spec = spec.scaled(k=length)
    if spec.family == "chain":
        if k_fixed:
            spec = spec.scaled(k=k_fixed)
        elif length >= spec.k:
            # chain_nowrap staircase: depth d runs over a (2d+1)-cycle — no wrap,
            # and the backward walk costs d+1 hops so neither direction beats
            # depth d (generating at depth >= k raises the wrap gate otherwise).
            spec = spec.scaled(k=2 * length + 1)
    return spec


# --- cost estimation --------------------------------------------------------------

@lru_cache(maxsize=None)
def _prompt_tokens_est(task: str, length: int, rendering: str | None,
                       breadth: int | None = None, k_fixed: int | None = None) -> int:
    """Rough prompt-token count for one example of (task, length, rendering,
    breadth rung, fixed chain k).

    Generates one deterministic example — via ``spec_for_cell``, so breadth rungs
    (more facts + a bigger recipient pool) and fixed-k chains (k facts at any
    depth) are priced on the exact spec the runner executes — and estimates
    tokens at CHARS_PER_TOKEN (the synthetic g/v/r token soup tokenizes
    densely). Cached: the dry-run plan touches each distinct combination once,
    not once per model.
    """
    if task == "s5":
        from . import s5_concrete
        sysp, user, _gold = s5_concrete.gen_examples(length, 1, framing=rendering)[0]
        return max(1, (len(sysp) + len(user)) // CHARS_PER_TOKEN)
    from . import tasks as TK
    spec = spec_for_cell(task, length, breadth=breadth, k_fixed=k_fixed)
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
        per_prompt = _prompt_tokens_est(cell["task"], cell["length"], s.get("rendering"),
                                        s.get("breadth"), s.get("k_fixed"))
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


def cell_dollar_cap(model_slug: str, n: int, max_new_tokens: int) -> float | None:
    """Per-cell DOLLAR cap for expensive models, or None (token guard suffices).

    Applies to models whose completion price is at or above
    CELL_DOLLAR_CAP_PRICE_THRESHOLD ($10/M — opus, gpt-5.5, sonnet on the current
    roster). Cap = max(CELL_DOLLAR_CAP_MIN_USD, the cell's NOMINAL completion
    budget ``n * max_new_tokens`` priced at the completion rate): the token-based
    CostGuard alone permits CELL_BUDGET_FACTOR (3x) the nominal budget — ~$61 for
    a 32768-token x n=25 thinking cell on opus — so the dollar cap holds an
    expensive cell to what it would legitimately cost with every call at its full
    budget, while the $2.50 floor keeps tight cells (e.g. the 96-token
    zero-budget battery, nominal ~$0.24 on opus) from being aborted by a handful
    of cap-escaping verbose calls. The runner prices the guard on completion
    tokens (usage.completion_tokens already includes reasoning); prompt spend is
    deterministic and priced by the dry-run estimate instead.
    """
    reg = MODELS.get(model_slug)
    if reg is None or reg["completion_price_per_M"] < CELL_DOLLAR_CAP_PRICE_THRESHOLD:
        return None
    nominal = n * max_new_tokens / 1e6 * reg["completion_price_per_M"]
    return max(CELL_DOLLAR_CAP_MIN_USD, nominal)
