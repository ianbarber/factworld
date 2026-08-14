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
    is built against — per-model direct endpoints (the muse-spark slot, and the
    locally served arm), defaulting to OpenRouter + OPENROUTER_API_KEY.
  - ``context_limit(model_slug)`` / ``context_overrun(model_slug, cell)``: the
    served context window of a self-hosted entry and the cells whose completion
    budget cannot fit it (checked before any call — see context_overrun).
  - ``settings_hash(cell)``: stable resume key for a cell's settings.
  - ``with_system_prompt(settings, text)``: stamp a cell's RESOLVED system prompt
    into its settings as a fingerprint, sentinel-dropped at the frozen
    SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS (the system prompt is part of what a
    cell measures, and the thinking and instant regimes take different texts — see
    CANONICAL_SYSTEM_PROMPT_FINGERPRINTS).
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
# System-prompt axis. The system prompt is not a knob the plan chooses — the runner
# derives it per cell from the facet/task/leg and the cell's REGIME
# (run_frontier_benchmark.system_prompt_for) off a small set of fixed texts. It is
# nonetheless part of what a cell measures: on identical s5_chain_v3 L64 items
# (n=25, effort xhigh, same endpoint and budget), removing two clauses of the base
# prompt that read as instructions to spend less effort — "You are taking a short
# test" and "no explanation" — while keeping the identical answer-format contract
# moved gpt-5.6-sol from 0.68 to 0.96 match, and dropped the rate of answers that
# dereference the initial map while ignoring the event stream from 0.33 to 0.04
# (results/probes/sol_system_prompt_20260727.json). That measurement is one model,
# one length, n=25 — it bounds nothing about the rest of the roster.
#
# The two regimes therefore take different texts, because they measure different
# things. Thinking cells (effort in REASONING_EFFORTS) carry the NEUTRAL prompt:
# the answer-format contract with those two clauses removed, so the score is what
# the model computes rather than how much effort the instruction elicited. Instant
# cells (effort none/minimal, hard one-line contract, 96-token cap) keep the base
# text: there, suppressing reasoning IS the measurement — "what the weights alone
# compute" is defined by pairing the suppressive instruction with the off arm.
#
# An edit to a scored system prompt is a change of measurement regime, and the
# resume key tracks it. The runner stamps each cell with
# settings["system_prompt_fp"] = the fingerprint of the RESOLVED prompt; the key is
# SENTINEL-DROPPED at SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS (both from the
# settings dict and from settings_hash) exactly like ``breadth`` at
# CANONICAL_BREADTH, so every cell measured before that set was frozen keeps a
# byte-identical resume key and no paid cell is invalidated, while any other prompt
# hashes distinctly and re-runs.
#
# CANONICAL is the set of texts the instrument is defined against; every planned
# cell resolves to one of them:
#   60766724c1  the base test prompt — instant cells whose answer is one token:
#               recall_load, chain_instant, sanity, and the holder-only legs of
#               zero_budget and gap_stability
#   8b02734258  the base prompt + the composite two-token format instruction —
#               the composite legs of zero_budget and gap_stability
#   27d71cb774  the s5_concrete framing prompt (its own answer-format contract is
#               inline in the text factworld.s5_concrete generates)
#   04153d7439  the neutral prompt — thinking cells: s5_chain, chain_nowrap,
#               commutative
#
# SENTINEL_DROP is the FROZEN set the drop is anchored at, defined by what it does:
# these three fingerprints are omitted from the key, so the records written before
# 2026-07-27 — all of which were measured under one of these three texts — keep the
# keys they were written with. It is CLOSED: nothing is ever added to it, and it is
# NOT "whatever history.jsonl currently contains". Once the neutral battery is
# bought, history holds neutral-prompt records too; adding 04153d7439 here would
# strike the fingerprint from those cells' keys and resume them against the
# base-prompt records the split exists to keep apart. A cell measured under a text
# outside this set carries its fingerprint permanently, which is what keeps the two
# regimes' records distinguishable inside one history file. The set is likewise
# independent of CANONICAL: a text that stops being planned stays in the drop set,
# because the records taken under it keep their keys either way.
# These are LITERALS on purpose: deriving them from the live text would move the
# sentinel with any edit and defeat the check. tests/test_benchmark_registry.py
# pins CANONICAL to the live resolved prompts, so an edit fails loudly, and pins
# SENTINEL_DROP to exactly these three strings, so nothing can be added. A new or
# edited prompt goes into CANONICAL alone — staying out of the drop set is what
# gives the cells measured under it fresh resume keys.
CANONICAL_SYSTEM_PROMPT_FINGERPRINTS = frozenset({
    "60766724c1", "8b02734258", "27d71cb774", "04153d7439",
})
SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS = frozenset({
    "60766724c1", "8b02734258", "27d71cb774",
})
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
#
# Self-hosted entries (``local_served``: a machine the owner runs — this box or
# another on the tailnet) additionally carry ``max_model_len``, the TOTAL window
# their server is started at and the number every planned budget is checked
# against, plus ``serve_hint``, the command that brings that server up, quoted
# in the error when it does not answer. Four further keys are theirs in practice:
# ``api_key_optional`` (the endpoint checks no key, so a missing ``api_key_env``
# is not an error — the value is still sent when the var is set),
# ``max_workers`` (a MEASURED ceiling on concurrent calls; the runner clamps its
# --max-workers to it, so a server holding one session is not hammered),
# ``generation_tok_per_s`` (the MEASURED completion rate, from which the runner
# sizes each request's timeout — a timeout under a cell's generation time is a
# retry loop, not an error), and ``context_is_minimum`` (the declared window is a
# floor rather than an equality, for a server whose --ctx this repo does not own).
MODELS = {
    "anthropic/claude-opus-4.8": {
        "tier": "frontier_pair", "prompt_price_per_M": 5.0,
        "completion_price_per_M": 25.0, "open_weights": False},
    # ADDED 2026-07-24 (roster refresh). Thinking-only: effort=none rejected 400
    # ("Reasoning is mandatory"), and effort=minimal still reasons (rtok=26 on a
    # 96-token contract probe) — no clean off-arm, so every instant facet is
    # skipped structurally like grok-4.5/muse. Its endpoint content-filters the
    # v-token composite contract prompts (3/3 finish=content_filter, the
    # mainline-grok blocker shape) — those are all instant cells, so the skip
    # list already covers them; the headline s5_chain prompts probe clean (3/3
    # stop, correct single-token answers). Cap respected. Pricing verified
    # against /api/v1/models 2026-07-24 ($10/$50 per M).
    "anthropic/claude-fable-5": {
        "tier": "frontier_pair", "prompt_price_per_M": 10.0,
        "completion_price_per_M": 50.0, "open_weights": False,
        "skip_facets": ("zero_budget", "recall_load", "chain_instant",
                        "sanity", "gap_stability")},
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
    # SWITCHED to the native Responses API 2026-07-24 (the Chat Completions shim
    # hides GPT-5.6's ``max`` level; at xhigh the two APIs score the same).
    # Protocol decision (owner, 2026-07-24): the scored top arm is the SHARED
    # xhigh level for every model — cross-model fairness over per-vendor
    # ceilings — so xhigh maps literally. The higher Responses-only ``max``
    # level is a documented probe finding, not a scored arm: on identical
    # s5_chain L96 items it reads 0.88 at 8,360 rtok/call vs 0.60-0.72 at
    # ~3.4-4.2k for xhigh on every route (results/probes/
    # sol_responses_20260724.json, sol_openrouter_xhigh_20260724.json).
    "openai/gpt-5.6-sol": {
        "tier": "frontier_pair", "prompt_price_per_M": 5.0,
        "completion_price_per_M": 30.0, "open_weights": False,
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-5.6-sol",
        "responses_endpoint": True,
        "reasoning_model": True,
        "supports_reasoning_effort": False,
        "reasoning_effort_values": {"low": "low", "medium": "medium", "high": "high",
                                     "xhigh": "xhigh", "max": "max"}},
    # openai/gpt-5.4 and google/gemini-3.1-pro-preview DROPPED 2026-07-08 (owner
    # decision: one flagship per vendor; Google is pushing flash).
    # no_reasoning_effort: Gemini 3 endpoints reject effort=none outright
    # ("Reasoning is mandatory ... cannot be disabled", 400); effort=minimal is
    # the closest off-arm (0 reasoning tokens on flash).
    # ADDED 2026-07-24, replacing gemini-3.5-flash (DROPPED same day: superseded
    # by version — the 3.5 slug still routes separately, so old cells were never
    # silently upgraded; explicit-version policy). Same endpoint behavior as 3.5:
    # effort=none rejected 400, effort=minimal is a clean off-arm (rtok=0,
    # contract obeyed). Pricing verified against /api/v1/models 2026-07-24.
    "google/gemini-3.6-flash": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 1.5,
        "completion_price_per_M": 7.5, "open_weights": False,
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
    # ADDED 2026-07-24, replacing kimi-k2.6 (DROPPED same day: superseded by
    # version; its cells render in the archived section). Unlike k2.6, k3's
    # effort=none arm probes CLEAN — rtok=0, contract obeyed, cap respected
    # (results/probes/new_models_20260724.jsonl) — so it runs the full battery
    # including instant. Pricing verified against /api/v1/models 2026-07-24.
    # quantization_filter off: no OpenRouter endpoint for this slug declares a
    # quantization, so the open-weights fp8/bf16/fp16 filter 404s every call
    # (verified 2026-07-24; the first battery attempt lost all cells to it).
    "moonshotai/kimi-k3": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 3.0,
        "completion_price_per_M": 15.0, "open_weights": True,
        "quantization_filter": False},
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
    # LOCAL ARM (2026-08-02): nvidia/Qwen3.6-35B-A3B-NVFP4 served by vLLM 0.26.0
    # on this machine's RTX 5090 (scripts/serve_local_model.py brings it up and
    # down; that script reads THIS entry for the host/port, the served name and
    # the context length, so server and registry cannot disagree). A roster
    # entry that happens to be local, not a side path: it resolves through
    # endpoint_for like the muse-spark slot and therefore inherits every runner
    # diagnostic — empty rate, finish reasons, contract/covert-CoT rates, the
    # regime-scoped system prompt, the cost guard and the resume keys.
    # PRICING IS ZERO, and zero is a measured fact here, not a missing field:
    # the endpoint is this machine, so usage * 0.0 records cost_usd_est 0.0 on
    # every cell (cell_dollar_cap returns None below the $10/M threshold, and
    # the ctok guard still bounds a runaway generator).
    # Reasoning: vLLM takes the OpenAI-style TOP-LEVEL reasoning_effort
    # parameter (supports_reasoning_effort False keeps the OpenRouter
    # extra_body block off the wire), and maps it to the chat template's
    # enable_thinking = (effort != "none"). "none" MUST stay in the map: an
    # unmapped arm sends no parameter at all and the template then defaults to
    # thinking ON, which would silently turn every instant cell into a thinking
    # cell. The template reads only that on/off bit — it has no effort ladder —
    # so low/medium/high/xhigh/max are the same measurement on this model, and
    # a battery should buy one of them, not a sweep.
    # ONE DIAGNOSTIC READS DIFFERENTLY HERE: vLLM's chat-completions usage has no
    # reasoning-token field, so every call records rtok 0 and the rtok_any_rate /
    # rtok_mean_per_call columns are 0 for this arm whether or not it thought —
    # they are not evidence that it did not. The thinking tokens are inside
    # ctok (the template opens <think> in the generation prompt and the answer
    # follows </think>, which backends.APIBackend already splits on), so
    # covert_cot_rate — visible ctok past COVERT_COT_CTOK_THRESHOLD — is the
    # diagnostic that still bites, and the off arm is structural rather than
    # instructed: at effort "none" the template emits a CLOSED empty think block,
    # so there is no trace to suppress.
    # max_model_len is the context the server is started with and the cap the
    # runner checks every planned budget against (context_limit / the runner's
    # preflight): 131,072 covers the largest budget the registry can plan
    # (s5_chain@L128, 98,304 completion tokens) plus its prompt.
    "local/qwen3.6-35b-a3b-nvfp4": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 0.0,
        "completion_price_per_M": 0.0, "open_weights": True,
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "LOCAL_VLLM_API_KEY",
        "model_name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "supports_reasoning_effort": False,
        "reasoning_effort_values": {"none": "none", "minimal": "minimal",
                                    "low": "low", "medium": "medium",
                                    "high": "high", "xhigh": "xhigh",
                                    "max": "max"},
        "local_served": True,
        "serve_hint": ".venv-serve/bin/python scripts/serve_local_model.py up",
        "max_model_len": 131072},
    # STEED ARM (2026-08-02): DeepSeek V4 Flash (IQ2XXS q2 GGUF, ~81 GB resident)
    # served by ds4-server on steed, a DGX Spark GB10 reached over the tailnet
    # (scripts/serve_steed_model.py brings it up and down and reads THIS entry).
    # Registered exactly like the local vLLM arm — {base_url, api_key_env}
    # resolved by endpoint_for — so it inherits every runner diagnostic: empty
    # rate, finish reasons, contract/covert-CoT rates, the regime-scoped system
    # prompt, the resume keys and the cost guard. Prices are 0.0 as a measured
    # fact (the endpoint is a machine the owner runs), so records carry
    # cost_usd_est 0.0 rather than an absent cost field. It uses no GPU on THIS
    # box, so it runs alongside a local training job.
    # ONE SLUG, NOT TWO. steed's /v1/models advertises deepseek-v4-flash AND
    # deepseek-v4-pro, both with the display name "DeepSeek V4 Flash". The pair
    # is an ALIAS, not two models: the server's send_models emits both ids
    # unconditionally while the loaded engine picks its own id, and on the wire
    # the two are indistinguishable — greedy decoding (temperature 0, fixed
    # seed) is reproducible on this endpoint (flash vs flash byte-identical on
    # every probe) and flash vs pro is byte-identical too, out to a 400-token
    # thinking trace (results/probes/steed_ds4_identity_20260802.json). Only
    # -flash is registered; a -pro slug would put one model on the board twice
    # and a sweep across the two would read as a model comparison.
    # NO AUTH: the endpoint is tailnet-only and checks no key, so api_key_env is
    # OPTIONAL here (api_key_optional). The var is still read and sent when set,
    # so putting auth on the box later needs no code change; a missing key is
    # not an error the way META_API_KEY's absence is.
    # THREE REASONING ARMS, NOT SEVEN, AND THE THIRD DEPENDS ON THE WINDOW.
    # ds4-server parses reasoning_effort and collapses minimal/low/medium/high/
    # xhigh to ONE internal level (they decode byte-identically), so buying that
    # band twice buys the same measurement twice. "none" is a genuine off arm and
    # MUST stay mapped: the server's default think mode is on, so an unmapped arm
    # would send nothing and silently turn every instant cell into a thinking
    # cell. "max" is a separate level, but ds4 serves it only when the server's
    # context is at least 393,216 and otherwise decodes it as the high band — so
    # whether the roster's top arm is one rung or two is a property of how the
    # server was STARTED, and preflight_context reads that number live.
    # supports_reasoning_effort False keeps the OpenRouter extra_body block off
    # the wire; the value rides the top-level parameter.
    # Reasoning tokens read like the local arm's: the chat-completions usage has
    # no reasoning-token field, so rtok is 0 whether or not it thought. There is
    # no <think> delimiter either — the working is plain content — so the answer
    # extractors see it: on the sanity row 5 of 30 answers were correct values
    # inside a sentence ("The a0 of g5 is v37 ."), match 0.833 against
    # containment 1.000. covert_cot_rate and the contract cells' extraction are
    # the diagnostics that bite here, not rtok.
    # CONCURRENCY 1, MEASURED. The unit passes no --batched-session, so the
    # server allocates a single KV session and serializes: at 1/2/4/8 concurrent
    # calls throughput was flat at 16.3-16.4 completion tok/s while wall time and
    # per-call latency scaled linearly with the worker count (7.2 s -> 57.7 s at
    # 8). Parallelism buys nothing here and only walks calls toward the request
    # timeout, so max_workers caps the runner at 1.
    # THE BINDING CONSTRAINT IS WALL CLOCK, NOT CONTEXT, and a cell's budget is
    # also its per-item duration. Short generations run at 16.3-16.4 completion
    # tok/s; a long one runs slower, because the KV grows under it — one
    # s5_bind_v3 composed item at L=128 was still generating after 45 minutes at
    # a 32,768-token cap, which bounds the long-output rate below 12.1 tok/s.
    # generation_tok_per_s therefore carries the SLOW rate, not the fast one, and
    # build_backend sizes each request's timeout from the cell's own budget
    # against it: a timeout under the budget's generation time does not fail the
    # call, it RETRIES it (an openai timeout is an APIConnectionError) and re-runs
    # the whole generation up to five times. Read the same arithmetic before
    # planning a battery here: at 12 tok/s a 32,768-token budget is 45 minutes an
    # item and an n=40 cell is 30 hours.
    # max_model_len is the TOTAL window, prompt plus completion, and here it is a
    # FLOOR rather than an equality (context_is_minimum). steed's --ctx lives in
    # a unit file outside this repo and is tuned there — the server has been seen
    # at 65,536, 393,216 and 262,144 within one evening — so the registry declares
    # the smallest window observed, budgets are planned against that, and
    # preflight_context reads the live number and uses it when it is larger. Only
    # a server smaller than the declared floor is a fault. The longest prompt this
    # instrument plans is 5,083 tokens (s5_bind_v3 composed, k=32, L=256, measured
    # on the server's own tokenizer — results/probes/steed_ds4_budget_20260802.json),
    # so within this floor a completion budget of 53,346 fits every cell of the
    # k x L grid and the window is not what bounds it.
    "steed/deepseek-v4-flash": {
        "tier": "cheap_reasoner", "prompt_price_per_M": 0.0,
        "completion_price_per_M": 0.0, "open_weights": True,
        "base_url": "https://steed.tailc4bb6.ts.net/v1",
        "api_key_env": "STEED_DS4_API_KEY",
        "api_key_optional": True,
        "model_name": "deepseek-v4-flash",
        "supports_reasoning_effort": False,
        "reasoning_effort_values": {"none": "none", "minimal": "minimal",
                                    "low": "low", "medium": "medium",
                                    "high": "high", "xhigh": "xhigh",
                                    "max": "max"},
        "local_served": True,
        "serve_hint": ".venv-api/bin/python scripts/serve_steed_model.py up",
        "max_workers": 1,
        "generation_tok_per_s": 12.0,
        "context_is_minimum": True,
        "max_model_len": 65536},
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


# Prompt-side safety factor for the context check below. _prompt_tokens_est
# counts at CHARS_PER_TOKEN and can under-count a given tokenizer, so the check
# reserves twice the estimate: the guard exists to stop a budget that cannot fit
# at all, and it must not pass a cell that then 400s at the first call.
CONTEXT_PROMPT_HEADROOM = 2.0


def context_limit(model_slug: str) -> int | None:
    """The served context window (prompt + completion) of a model, or None.

    Only a self-hosted entry declares one (``max_model_len``, the length its
    server is started with — see the local arm). For a vendor endpoint the
    window is the vendor's business and the value is None, which makes
    ``context_overrun`` inert.
    """
    return (MODELS.get(model_slug) or {}).get("max_model_len")


def context_overrun(model_slug: str, cell: dict, *, limit: int | None = None,
                    min_completion_tokens: int = 0) -> tuple[int, int] | None:
    """``(tokens_needed, limit)`` when a cell cannot fit the context, else None.

    A completion budget larger than the served context is not a slow cell, it is
    a cell that cannot run: vLLM rejects the request 4xx, and while the backend
    now fails such a request fast (backends.APIBackend: a non-429 4xx is not
    retried — the lesson of ~1,000 rejected calls over 51 minutes), every call
    still returns an empty prediction that scores as wrong. So the plan is
    checked against the limit BEFORE any call is made.

    ``limit`` overrides the registry value with what a live server reports (the
    engine may clamp ``--max-model-len``); ``min_completion_tokens`` raises the
    budget to cover a cell whose runner may rerun it at a bigger one (the
    contract-cell escalation ladder).
    """
    limit = limit if limit is not None else context_limit(model_slug)
    if not limit:
        return None
    s = cell["settings"]
    budget = max(s["max_new_tokens"], min_completion_tokens)
    prompt = _prompt_tokens_est(cell["task"], cell["length"], s.get("rendering"),
                                s.get("breadth"), s.get("k_fixed"), s.get("k_sweep"))
    needed = budget + int(CONTEXT_PROMPT_HEADROOM * prompt)
    return (needed, limit) if needed > limit else None

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
        "budgets": {128: 16384, 256: 32768}},  # L256 raised to the s5_chain rule
        # (budgets sized so truncation stays a rounding error): at 16,384 four
        # models needed single raised reruns and two were majority-truncated.
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
    # s5_chain — a RETIRED facet: its cells stay in history and in results.md and are
    # reproducible, and facet_retired keeps it out of every plan. Non-abelian pointer-map state
    # tracking composed with a 16-hop serial dereference (k=32 agents; length =
    # number of swap/cycle events). Runs the distinct_path-gated v4 stream, where
    # a quarter of the events name an operand by reference to the RUNNING map, so
    # no event's identity is fixed until the map has been evaluated forward to it
    # — echo 0 under the gate, and every registered shallow adversary at or
    # below 0.040: the operative floor is 0.0398 at L=32 (supplied by
    # initial-ref resolution) and 0.0323-0.0334 at L=64/96/128 (n=5000),
    # against a chance of 1/31 = 0.0323 for a guesser that has learned only
    # "never answer the queried agent" and the 2x-chance bound the suite gates
    # on. The initial-map backhop is measured but NOT registered: it is one of
    # 31 fixed offsets through the stated map whose accuracies sum to exactly
    # 1, so its null is that same 0.0323 and a max over it measures selection
    # (validity.S5_CHAIN_ADVERSARIES). s5_chain_v3 is retired (tasks.RETIRED): its
    # events permute the map's domain, so one symbol pushed BACKWARD through the
    # event list answers the query exactly, which an attention model over the full
    # context can do and a streaming recurrent model cannot. Task is part of the
    # resume key, so v4 cells key fresh by construction and no v3 record satisfies
    # them. Protocol: the SHARED xhigh arm for every model — cross-model fairness
    # over per-vendor ceilings, so a vendor level ABOVE xhigh (gpt-5.6-sol's
    # Responses-only ``max``) is a probe finding and not a scored arm; see the
    # gpt-5.6-sol registry entry. Per-length budgets are sized so finish=length
    # truncation — scored as wrong — stays a rounding error, not a ranking
    # confound (deepseek/nemotron/glm truncated 16-28% of calls at the old 16-24k
    # budgets); they are the budgets the v3 battery ran, and v4's deeper
    # dereference over a wider map is the reason a battery reads the truncation
    # column before its scores.
    # Rendered by render_benchmark.s5_chain_rows (README + report ranking table).
    "s5_chain": {
        "task": "s5_chain_v4", "lengths": (32, 64, 96, 128), "n": 25,
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
        if facet_name in skip or facet_retired(facet_name):
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


def system_prompt_fingerprint(text: str) -> str:
    """10-hex-char fingerprint of a resolved system prompt (same digest shape as
    ``settings_hash``)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def with_system_prompt(settings: dict, system_prompt: str) -> dict:
    """``settings`` carrying the fingerprint of the cell's RESOLVED system prompt.

    SENTINEL-DROPPED at the frozen drop set: when the resolved text is one of
    SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS the key is omitted entirely, so such
    a run's settings — and the history record built from them — stay
    byte-identical to the records taken before the set was frozen and the resume
    key is unchanged. Any other prompt carries ``system_prompt_fp``, which
    ``settings_hash`` keys on, so cells under it get fresh keys and run. The
    neutral thinking prompt is in CANONICAL and not in the drop set, so thinking
    cells stamp it and re-plan while instant cells keep their keys — and it stays
    out of the drop set after those cells are bought, or their records would
    resume against the base-prompt ones.
    """
    fp = system_prompt_fingerprint(system_prompt)
    if fp in SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS:
        return {k: v for k, v in settings.items() if k != "system_prompt_fp"}
    return {**settings, "system_prompt_fp": fp}


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
      - ``system_prompt_fp`` at one of the frozen
        SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS: the system prompt materially
        changes what a cell measures, so a prompt outside that set must produce a
        fresh key — while the cells measured under those three texts, which is
        every record written before the set was frozen, keep the keys they were
        written with. The set never grows, so a cell's key never changes under it.
    """
    _drop = {
        "contract": lambda v: not v,
        "breadth": lambda v: not v or v == CANONICAL_BREADTH,
        "k_fixed": lambda v: not v,
        "system_prompt_fp":
            lambda v: not v or v in SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS,
    }
    settings = {k: v for k, v in cell["settings"].items()
                if k not in _drop or not _drop[k](v)}
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


# --- spec resolution (single source of truth for the runner + cost estimator) ------

def spec_for_cell(task: str, length: int, breadth: int | None = None,
                  k_fixed: int | None = None, k_sweep: int | None = None):
    """The TaskSpec a cell actually runs (shared by the runner's cell execution
    and ``_prompt_tokens_est`` so prompts and prices never diverge).

      - ``k_sweep`` (s5_bind family): CANONICAL[task].scaled(k=K, n_objects=K,
        n_objects_active=K) — the agent/object count moved together, which is
        what the k-vs-L difficulty sweep varies and what the family's floor
        argument is written at (pad < min(k, m)). It is the only knob that
        changes an s5_bind prompt's LENGTH at fixed L, so the context guard
        prices the rung that runs rather than the canonical k=12 spec. None
        (every registry facet) leaves the spec untouched.
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
    # spec_for, not CANONICAL: a RETIRED spec must stay RESOLVABLE so a historical cell can be
    # re-rendered and reproduced. It must not be PLANNED, and that is a separate rule --
    # ``facet_retired`` drops those facets out of the plan, so nothing buys a retired cell.
    spec = TK.spec_for(task)
    if k_sweep and spec.family == "s5_bind":
        spec = spec.scaled(k=k_sweep, n_objects=k_sweep, n_objects_active=k_sweep)
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
                       breadth: int | None = None, k_fixed: int | None = None,
                       k_sweep: int | None = None) -> int:
    """Rough prompt-token count for one example of (task, length, rendering,
    breadth rung, fixed chain k, s5_bind k rung).

    Generates one deterministic example — via ``spec_for_cell``, so breadth rungs
    (more facts + a bigger recipient pool), fixed-k chains (k facts at any
    depth) and s5_bind k rungs (k agents and k objects, both stated in the
    initial map) are priced on the exact spec the runner executes — and estimates
    tokens at CHARS_PER_TOKEN (the synthetic g/v/r token soup tokenizes
    densely). Cached: the dry-run plan touches each distinct combination once,
    not once per model.
    """
    if task == "s5":
        from . import s5_concrete
        sysp, user, _gold = s5_concrete.gen_examples(length, 1, framing=rendering)[0]
        return max(1, (len(sysp) + len(user)) // CHARS_PER_TOKEN)
    from . import tasks as TK
    spec = spec_for_cell(task, length, breadth=breadth, k_fixed=k_fixed, k_sweep=k_sweep)
    ex = TK.generate(spec, "test", n=1, length=length)[0]
    return SYSTEM_PROMPT_EST_TOKENS + max(1, len(ex.prompt) // CHARS_PER_TOKEN)


def facet_retired(facet_name: str) -> bool:
    """Is every task this facet would run RETIRED (or unregistered)?

    A facet naming a retired spec is history, not a plan: its cells are in
    ``results/benchmark/history.jsonl`` and are still rendered and reproducible,
    but a NEW battery must not buy them. The check is on the registry rather than
    on a hand-kept list of dead facet names, so retiring a spec is the only edit
    retiring its facet takes.
    """
    from . import tasks as TK

    fc = FACETS[facet_name]
    names = ({t for t, _L in fc["tasks"]} if fc.get("tasks")
             else {fc["task"]} if fc.get("task") else set())
    if not names:
        return False
    live = {n for n in names
            if n in TK.CANONICAL and TK.CANONICAL[n].kind != "retired"}
    return not live


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
                                        s.get("breadth"), s.get("k_fixed"), s.get("k_sweep"))
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
