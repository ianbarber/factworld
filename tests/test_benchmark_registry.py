"""Tests for factworld/benchmark.py (contract C4) and the runner's cell execution.

Registry: tier -> arm expansion (v2 facets: zero_budget contract battery,
s5_concrete mid-band, chain_nowrap staircase, sanity), stable settings_hash
(including pre-contract-flag history compatibility), sane cost estimates.
Runner: FunctionBackend-driven mini-runs of ``execute_cell`` writing C3-conformant
records (per-example ctok/rtok/finish, contract diagnostics, finish=length
escalation) to a tmp history file (no API), plus the resume-key round trip.
Provenance: the resolved system prompt is scoped to the cell's regime (thinking
cells neutral, instant cells the base test prompt) and fingerprinted into the
resume key (sentinel-dropped at a frozen, closed set of three texts, so no
already-run cell is invalidated), and the resolved vendor request — effort string, endpoint,
base URL, model name — is recorded per cell without being keyed.

Run directly:  .venv-api/bin/python tests/test_benchmark_registry.py
Run with pytest: .venv-api/bin/python -m pytest tests/test_benchmark_registry.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import factworld.benchmark as B
import factworld.tasks as TK
from factworld.backends import FunctionBackend

import run_frontier_benchmark as RFB

SETTINGS_KEYS = {"effort", "max_new_tokens", "stop_at", "rendering",
                 "format_prompt", "n_shot", "leg", "contract"}

# The zero_budget facet targets composite_copy_v2 (uniform last-write placement,
# killing the retired v1 sampler's recency shortcut); the v1 family lives in
# tasks.RETIRED (issue #11) and every planned facet task resolves in CANONICAL.
assert "composite_copy_v2" in TK.CANONICAL


def _runnable(cell: dict) -> dict:
    """Every planned cell's task now resolves in CANONICAL (kept as a seam so the
    item-generating tests read explicitly; asserts the invariant)."""
    assert cell["task"] in TK.CANONICAL or cell["task"] == "s5"
    return cell
C3_TOP_KEYS = {"run_id", "ts", "git_commit", "suite_version", "stream_version", "model", "served_models",
               "providers", "facet", "task", "length", "n", "settings", "metrics",
               "diagnostics", "usage", "elapsed_s", "escalated", "examples"}
EXAMPLE_KEYS = {"gold", "pred", "relaxed", "ctok", "rtok", "finish"}


def _efforts(cells, facet):
    return [c["settings"]["effort"] for c in cells if c["facet"] == facet]


# --- registry -------------------------------------------------------------------

def test_registry_shape():
    assert len(B.MODELS) == 15
    # roster decisions 2026-07-07/08/09: maverick, gpt-5.4, gemini-3.1-pro
    # dropped; the UNMEASURABLE x-ai endpoints stay off the roster (mainline
    # grok bio-filtered on composite prompts; grok-build dropped for provider
    # pathology — pinned ~256k reasoning ignoring caps, cannot disable
    # reasoning). 2026-07-12 (issue #15): grok-4.5 and gpt-5.6-sol added.
    # 2026-07-13: muse-spark-1.1 added via direct Meta Model API endpoint.
    for dropped in ("meta-llama/llama-4-maverick", "openai/gpt-5.4",
                    "google/gemini-3.1-pro-preview", "x-ai/grok-4.3",
                    "x-ai/grok-4.20", "x-ai/grok-build-0.1"):
        assert dropped not in B.MODELS
    # issue #15 additions, pricing verified against /api/v1/models 2026-07-12
    sol = B.MODELS["openai/gpt-5.6-sol"]
    assert sol["tier"] == "frontier_pair"
    assert (sol["prompt_price_per_M"], sol["completion_price_per_M"]) == (5.0, 30.0)
    assert not sol.get("skip_facets") and not sol.get("no_reasoning_effort")
    grok = B.MODELS["x-ai/grok-4.5"]
    assert grok["tier"] == "cheap_reasoner"
    assert (grok["prompt_price_per_M"], grok["completion_price_per_M"]) == (2.0, 6.0)
    # thinking facets only: no clean off-arm exists (effort=none rejected 400;
    # "minimal" emits ~547 rtok — past the covert-CoT bar), so every
    # "off"-policy facet is structurally skipped.
    assert set(grok["skip_facets"]) == {"zero_budget", "recall_load",
                                        "chain_instant", "sanity",
                                        "gap_stability"}
    assert "no_reasoning_effort" not in grok  # a substituted off-arm would be dirty
    muse = B.MODELS["muse-spark-1.1"]
    assert muse["tier"] == "cheap_reasoner"
    assert (muse["prompt_price_per_M"], muse["completion_price_per_M"]) == (1.25, 4.25)
    assert muse["base_url"] == "https://api.meta.ai/v1"
    assert muse["api_key_env"] == "META_API_KEY"
    assert muse.get("responses_endpoint") is True
    assert set(muse["skip_facets"]) == {"zero_budget", "recall_load",
                                         "chain_instant", "sanity", "gap_stability"}
    for slug, reg in B.MODELS.items():
        assert reg["tier"] in B.TIERS, slug
        assert isinstance(reg["open_weights"], bool)
        # Prices are always present, because a missing price and a zero price
        # mean different things: a metered endpoint must carry a real rate, and
        # the self-hosted arm's rate is zero as a measured fact.
        if reg.get("local_served"):
            assert (reg["prompt_price_per_M"], reg["completion_price_per_M"]) == (0.0, 0.0)
        else:
            assert reg["prompt_price_per_M"] > 0 and reg["completion_price_per_M"] > 0
    # commutative / gap_stability / s5_chain: EXPERIMENTAL facets (owner-approved
    # 2026-07-11/#18/#16a and 2026-07-16) — no renderer section reads them yet.
    assert set(B.FACETS) == {"zero_budget", "recall_load", "s5_concrete",
                             "chain_nowrap", "chain_instant", "sanity",
                             "commutative", "gap_stability", "s5_chain"}


def test_arms_for_facets():
    opus = B.arms_for("anthropic/claude-opus-4.8")

    # zero_budget: explicit (length, leg) cells, effort none, tight budget, contract on.
    # The "replicate" leg (review F6, replacing the mislabeled "end_to_end") is a
    # deliberate test-retest duplicate of the plain L16 cell.
    # The scaffolded leg (issue #11 re-measure): recall-given-holder on the v2 items,
    # completing the E1b decomposition triple (a one-shot positive-control ceiling row).
    zb = [c for c in opus if c["facet"] == "zero_budget"]
    assert [(c["length"], c["settings"]["leg"]) for c in zb] == [
        (16, None), (64, None), (16, "binding_only"), (16, "replicate"),
        (16, "scaffolded")]
    assert not any(c["settings"]["leg"] == "end_to_end" for c in zb)
    for c in zb:
        # composite_copy_v2: the uniform-last-write sampler (v1's queried-object
        # last write sat ~geometric(1/4) from the stream end; a recency one-liner
        # scored 0.34@L16). All other knobs inherited from the v1 battery.
        assert c["task"] == "composite_copy_v2" and c["n"] == 100
        assert c["settings"]["effort"] == "none"
        assert c["settings"]["contract"] is True
        assert c["settings"]["max_new_tokens"] == B.ZERO_BUDGET_MAX_NEW_TOKENS == 96

    # s5_concrete: mid-band only (ceiling lengths dropped), reasoning on; L128 at the
    # 16384 default, L256 raised to 32768 (the truncation-as-rounding-error rule).
    s5 = [c for c in opus if c["facet"] == "s5_concrete"]
    assert [c["length"] for c in s5] == [128, 256]
    for c in s5:
        assert c["n"] == 25 and c["settings"]["effort"] == "high"
        assert c["settings"]["max_new_tokens"] == (16384 if c["length"] == 128 else 32768)
        assert c["settings"]["rendering"] == "concrete"
        assert c["settings"]["contract"] is False

    # chain_nowrap: the four staircase depths, reasoning on, 16384 budget.
    chain = [c for c in opus if c["facet"] == "chain_nowrap"]
    assert [c["length"] for c in chain] == [16, 32, 64, 128]
    for c in chain:
        assert c["task"] == "chain_v2" and c["n"] == 25
        assert c["settings"]["effort"] == "high"
        assert c["settings"]["max_new_tokens"] == 16384

    # recall_load: the pool-64 instant recall cell (recall under working-set
    # load; pool = min(L, k) with spec_for_cell scaling k to L). Contract
    # protocol identical to zero_budget (effort none, 96-token cap).
    rl = [c for c in opus if c["facet"] == "recall_load"]
    assert [(c["task"], c["length"], c["n"]) for c in rl] == [("recall_copy_v1", 64, 50)]
    assert rl[0]["settings"]["effort"] == "none"
    assert rl[0]["settings"]["contract"] is True
    assert rl[0]["settings"]["max_new_tokens"] == B.ZERO_BUDGET_MAX_NEW_TOKENS

    # chain_instant: the d16 off arm of the chain staircase (within-item regime
    # contrast with the chain_nowrap d16 thinking cell — same spec k=33, same n).
    ci = [c for c in opus if c["facet"] == "chain_instant"]
    assert [(c["task"], c["length"], c["n"]) for c in ci] == [("chain_v2", 16, 25)]
    assert ci[0]["settings"]["effort"] == "none"
    assert ci[0]["settings"]["contract"] is True
    assert ci[0]["settings"]["max_new_tokens"] == B.ZERO_BUDGET_MAX_NEW_TOKENS

    # sanity: unchanged positive controls, effort none, default budget.
    sanity = [c for c in opus if c["facet"] == "sanity"]
    assert [(c["task"], c["length"]) for c in sanity] == [("recall_copy_v1", 6),
                                                          ("conflict_v1", 4)]
    for c in sanity:
        assert c["settings"]["effort"] == "none"
        assert c["settings"]["max_new_tokens"] == B.DEFAULT_MAX_NEW_TOKENS

    # cheap_reasoner expands to the same plan shape (no dose facet remains).
    glm = B.arms_for("z-ai/glm-5.2")
    assert [(c["facet"], c["task"], c["length"], c["settings"]["leg"]) for c in glm] == \
           [(c["facet"], c["task"], c["length"], c["settings"]["leg"]) for c in opus]


def test_arm_settings_protocol():
    """Reasoning-on cells: >=8192 tokens, no stop; every cell has all C3 settings
    keys; zero_budget cells carry the 96-token cap + contract flag."""
    for slug in ("z-ai/glm-5.2", "anthropic/claude-opus-4.8"):
        for cell in B.arms_for(slug):
            s = cell["settings"]
            assert set(s) == SETTINGS_KEYS
            assert s["stop_at"] is None
            assert s["n_shot"] == 0
            assert isinstance(s["contract"], bool)
            if s["effort"] in B.REASONING_EFFORTS:
                assert s["max_new_tokens"] >= B.REASONING_MAX_NEW_TOKENS
            elif s["contract"]:  # zero_budget / recall_load / chain_instant
                assert s["max_new_tokens"] == B.ZERO_BUDGET_MAX_NEW_TOKENS
            else:
                assert s["max_new_tokens"] == B.DEFAULT_MAX_NEW_TOKENS
            assert cell["n"] == B.FACETS[cell["facet"]]["n"]


def test_gemini_off_arm_is_minimal():
    """Gemini 3 cannot disable reasoning: its "none" arms substitute "minimal"."""
    for slug in ("google/gemini-3.6-flash",):
        efforts = {c["settings"]["effort"] for c in B.arms_for(slug)}
        assert "none" not in efforts
        assert "minimal" in efforts
    # models without the override keep the true off arm
    assert "none" in {c["settings"]["effort"] for c in B.arms_for("z-ai/glm-5.2")}


def test_settings_hash_stable():
    cell = B.arms_for("z-ai/glm-5.2")[0]
    h = B.settings_hash(cell)
    assert len(h) == 10 and all(ch in "0123456789abcdef" for ch in h)
    # invariant to dict key order
    reordered = {"settings": dict(reversed(list(cell["settings"].items())))}
    assert B.settings_hash(reordered) == h
    # invariant to a JSON round trip (how the resume path recomputes it)
    roundtrip = {"settings": json.loads(json.dumps(cell["settings"]))}
    assert B.settings_hash(roundtrip) == h
    # sensitive to a settings change
    changed = {"settings": {**cell["settings"], "effort": "medium"}}
    assert B.settings_hash(changed) != h


def test_settings_hash_contract_flag_compat():
    """History written BEFORE the contract flag existed must keep its resume keys:
    contract=False hashes identically to a settings dict without the key; only
    contract=True changes the hash."""
    s5_cell = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "s5_concrete")
    assert s5_cell["settings"]["contract"] is False
    legacy = {k: v for k, v in s5_cell["settings"].items() if k != "contract"}
    assert B.settings_hash({"settings": legacy}) == B.settings_hash(s5_cell)
    flagged = {"settings": {**s5_cell["settings"], "contract": True}}
    assert B.settings_hash(flagged) != B.settings_hash(s5_cell)


def test_settings_hash_breadth_and_k_fixed_sentinels():
    """v3 rung keys are SENTINEL-DROPPED at their canonical values (breadth=16 —
    truthy, so the falsy drop alone would not cover it — and k_fixed absent/None):
    canonical-valued keys hash identically to the key being absent, so every
    pre-breadth history record's resume key is unchanged; truthy NON-canonical
    values hash distinctly, so each rung resumes independently."""
    zb = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "zero_budget")
    h = B.settings_hash(zb)
    # the plan omits the keys entirely at canonical values
    assert "breadth" not in zb["settings"] and "k_fixed" not in zb["settings"]
    # direction 1: canonical / falsy values are dropped -> identical hash
    assert B.settings_hash(
        {"settings": {**zb["settings"], "breadth": B.CANONICAL_BREADTH}}) == h
    assert B.settings_hash(
        {"settings": {**zb["settings"], "breadth": None, "k_fixed": None}}) == h
    # direction 2: truthy non-canonical values are hashed -> distinct keys per rung
    b64 = B.settings_hash({"settings": {**zb["settings"], "breadth": 64}})
    b4 = B.settings_hash({"settings": {**zb["settings"], "breadth": 4}})
    assert len({h, b64, b4}) == 3
    # JSON round trip (how history_keys recomputes the hash) is stable for rungs
    rt = json.loads(json.dumps({**zb["settings"], "breadth": 64}))
    assert B.settings_hash({"settings": rt}) == b64

    chain = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "chain_nowrap")
    hc = B.settings_hash(chain)
    assert B.settings_hash({"settings": {**chain["settings"], "k_fixed": None}}) == hc
    assert B.settings_hash({"settings": {**chain["settings"], "k_fixed": 257}}) != hc


def test_canonical_system_prompt_fingerprints_pin_the_live_prompts():
    """The CANONICAL set must be exactly the prompts the planned cells resolve to.

    This is the tripwire for the system-prompt axis. The prompt is a measured
    protocol parameter — on identical s5_chain_v3 L64 items, dropping the "short
    test"/"no explanation" clauses moved gpt-5.6-sol from 0.68 to 0.96 match — so
    an edit to any scored prompt is a re-measurement. If this test fails because a
    prompt text changed, add the new fingerprint to CANONICAL and nowhere else:
    keeping it out of the sentinel drop set is exactly what gives the affected
    cells fresh resume keys instead of resuming against measurements taken under
    the old text."""
    live = {}
    for slug in B.MODELS:
        for cell in B.arms_for(slug):
            sp = RFB.system_prompt_for(cell)
            live.setdefault(B.system_prompt_fingerprint(sp), sp)
    assert set(live) == set(B.CANONICAL_SYSTEM_PROMPT_FINGERPRINTS), (
        "the planned cells' system prompts no longer match the canonical set; "
        "an edited prompt is a different measurement regime — see the docstring")
    # the four texts, pinned by content so a silent rewording cannot pass
    assert live["60766724c1"] == RFB.BASE_SYSTEM_PROMPT
    assert live["8b02734258"].startswith(RFB.BASE_SYSTEM_PROMPT)
    assert "g3 v9" in live["8b02734258"]          # composite two-token format leg
    assert "Driver" in live["27d71cb774"]         # s5_concrete "concrete" framing
    assert live["04153d7439"] == RFB.NEUTRAL_SYSTEM_PROMPT
    # the neutral text is the base text minus exactly the two suppressing clauses
    for clause in ("short test", "no explanation"):
        assert clause in RFB.BASE_SYSTEM_PROMPT
        assert clause not in RFB.NEUTRAL_SYSTEM_PROMPT
    assert "the same spelling as in the question" in RFB.NEUTRAL_SYSTEM_PROMPT

    # regime scoping: thinking cells neutral, instant cells the base test prompt
    for slug in B.MODELS:
        for cell in B.arms_for(slug):
            if cell["facet"] in RFB.S5_FACETS:
                continue
            sp = RFB.system_prompt_for(cell)
            if cell["settings"]["effort"] in B.REASONING_EFFORTS:
                assert sp.startswith(RFB.NEUTRAL_SYSTEM_PROMPT), (slug, cell["facet"])
            else:
                assert sp.startswith(RFB.BASE_SYSTEM_PROMPT), (slug, cell["facet"])
    s5_chain = next(c for c in B.arms_for("openai/gpt-5.6-sol")
                    if c["facet"] == "s5_chain")
    assert RFB.system_prompt_for(s5_chain) == RFB.NEUTRAL_SYSTEM_PROMPT


def test_sentinel_drop_set_is_frozen_and_closed():
    """The sentinel drop set is pinned to its three literals and never grows.

    It is defined by what it does — omit these fingerprints from the resume key, so
    the cells measured before it was frozen keep the keys they were written with —
    and NOT as "the prompts history.jsonl contains". Those two definitions agree
    today and diverge the moment a neutral-prompt battery is bought: history then
    holds that text too, and a set maintained against history would take the
    neutral fingerprint in, strike it from those cells' keys, and resume them
    against the base-prompt records the split exists to keep apart. The second half
    of this test states that consequence executably."""
    assert B.SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS == frozenset({
        "60766724c1", "8b02734258", "27d71cb774"}), (
        "the drop set is frozen at the pre-2026-07-27 texts and closed to "
        "additions; a new or edited prompt goes into CANONICAL alone")
    neutral_fp = B.system_prompt_fingerprint(RFB.NEUTRAL_SYSTEM_PROMPT)
    assert neutral_fp not in B.SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS

    cell = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "s5_chain")
    base = {"settings": B.with_system_prompt(cell["settings"], RFB.BASE_SYSTEM_PROMPT)}
    neutral = {"settings": {**cell["settings"], "system_prompt_fp": neutral_fp}}
    assert B.settings_hash(neutral) != B.settings_hash(base)
    frozen = B.SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS
    try:  # with the neutral fingerprint added, the two regimes share one key
        B.SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS = frozen | {neutral_fp}
        assert B.settings_hash(neutral) == B.settings_hash(base)
    finally:
        B.SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS = frozen


def test_settings_hash_system_prompt_sentinel():
    """The resolved system prompt is SENTINEL-DROPPED at the frozen drop set:
    cells under those three texts hash exactly like the pre-stamp history (both
    with the key absent and with such a fingerprint explicitly recorded), while any
    other prompt hashes distinctly — including after a JSON round trip through
    history.jsonl."""
    cell = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "s5_chain")
    h = B.settings_hash(cell)
    assert "system_prompt_fp" not in cell["settings"]

    # direction 1: a drop-set prompt -> key omitted, hash unchanged
    canonical = B.with_system_prompt(cell["settings"], RFB.BASE_SYSTEM_PROMPT)
    assert canonical == cell["settings"]
    assert B.settings_hash({"settings": canonical}) == h
    # a record that explicitly carries such a fingerprint still hashes the same
    for fp in B.SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS:
        assert B.settings_hash(
            {"settings": {**cell["settings"], "system_prompt_fp": fp}}) == h
    assert B.settings_hash(
        {"settings": {**cell["settings"], "system_prompt_fp": None}}) == h

    # direction 2: the neutral thinking prompt (same answer contract, the two
    # effort-suppressing clauses removed) is a different regime -> distinct hash
    neutral = RFB.NEUTRAL_SYSTEM_PROMPT
    fp = B.system_prompt_fingerprint(neutral)
    assert fp in B.CANONICAL_SYSTEM_PROMPT_FINGERPRINTS
    assert fp not in B.SENTINEL_DROP_SYSTEM_PROMPT_FINGERPRINTS
    edited = B.with_system_prompt(cell["settings"], neutral)
    assert edited["system_prompt_fp"] == fp
    h2 = B.settings_hash({"settings": edited})
    assert h2 != h
    # ...and the empty prompt is a third regime again
    assert B.settings_hash({"settings": B.with_system_prompt(cell["settings"], "")}) \
        not in (h, h2)
    # JSON round trip (how history_keys recomputes the hash) preserves both
    assert B.settings_hash({"settings": json.loads(json.dumps(edited))}) == h2
    assert B.settings_hash({"settings": json.loads(json.dumps(canonical))}) == h
    # stamping is idempotent: re-resolving to a canonical prompt clears a stale key
    assert "system_prompt_fp" not in B.with_system_prompt(edited, RFB.BASE_SYSTEM_PROMPT)


def test_build_plan_stamps_the_resolved_system_prompt():
    """The runner stamps every cell with the fingerprint of the prompt it resolves
    to, and the regime split lands in the resume keys: instant cells (and
    s5_concrete, which keeps its framing prompt) are byte-identical to arms_for, so
    no paid cell is invalidated; thinking cells carry the neutral fingerprint and
    key distinctly from the same cell measured under the base prompt."""
    model = "z-ai/glm-5.2"
    planned = RFB.build_plan([model], None, 1.0)[model]
    raw = B.arms_for(model)
    neutral_fp = B.system_prompt_fingerprint(RFB.NEUTRAL_SYSTEM_PROMPT)
    assert len(planned) == len(raw)
    n_thinking = 0
    for p, r in zip(planned, raw):
        thinking = (p["settings"]["effort"] in B.REASONING_EFFORTS
                    and p["facet"] not in RFB.S5_FACETS)
        if thinking:
            n_thinking += 1
            assert p["settings"]["system_prompt_fp"] == neutral_fp
            assert RFB.cell_key(model, p) != RFB.cell_key(model, r)
        else:
            assert set(p["settings"]) == SETTINGS_KEYS  # no key at measured prompts
            assert p["settings"] == r["settings"]
            assert RFB.cell_key(model, p) == RFB.cell_key(model, r)
    # chain_nowrap (4) + commutative (1) + s5_chain (4); s5_concrete keeps its own
    assert n_thinking == 9
    assert {p["facet"] for p, r in zip(planned, raw)
            if p["settings"].get("system_prompt_fp")} == {
        "chain_nowrap", "commutative", "s5_chain"}

    # the resume consequence, both directions, against records written the way
    # history holds them (settings verbatim, no fingerprint under the base prompt)
    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "history.jsonl")
        archived = [c for c in raw if c["facet"] in ("s5_chain", "zero_budget")]
        for cell in archived:
            rec = RFB.execute_cell(
                FunctionBackend(lambda ps, m, s: ["g3 ."] * len(ps), name="f"),
                model, {**cell, "n": 5}, n=5, run_id="t", git_commit="d")
            assert "system_prompt_fp" not in rec["settings"]
            RFB.append_record(history, rec)
        done = RFB.history_keys(history)
        for cell in archived:
            assert RFB.should_skip(model, {**cell, "n": 5}, done,
                                   force=False, canary=False)
        for cell in planned:
            if cell["facet"] not in ("s5_chain", "zero_budget"):
                continue
            skip = RFB.should_skip(model, {**cell, "n": 5}, done,
                                   force=False, canary=False)
            # instant zero_budget cells resume; thinking s5_chain cells re-plan
            assert skip == (cell["facet"] == "zero_budget"), cell["facet"]


def test_plan_has_no_rung_keys_and_facet_breadths_expand():
    """No current facet sets breadths/k_fixed, so every planned cell's settings
    are exactly the pre-breadth C3 keys (the --dry-run 0-cells acceptance rests
    on this); a facet that opts in via ``breadths`` expands one cell per rung
    with the canonical rung's key sentinel-dropped."""
    from unittest import mock
    for slug in B.MODELS:
        for cell in B.arms_for(slug):
            assert set(cell["settings"]) == SETTINGS_KEYS
    fake = {**B.FACETS["zero_budget"], "breadths": (4, B.CANONICAL_BREADTH, 64)}
    with mock.patch.dict(B.FACETS, {"zero_budget": fake}):
        cells = [c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "zero_budget"]
        assert len(cells) == 5 * 3  # 5 (length, leg) cells x 3 pool rungs
        rungs = {(c["length"], c["settings"]["leg"], c["settings"].get("breadth"))
                 for c in cells}
        # canonical rung: key omitted (None); non-canonical rungs carried verbatim
        assert {(16, None, 4), (16, None, None), (16, None, 64)} <= rungs
        keys = {B.settings_hash(c) for c in cells if c["settings"]["leg"] is None
                and c["length"] == 16}
        assert len(keys) == 3  # each rung is its own resume key


def test_prompt_tokens_est_breadth_and_k_fixed():
    """Cost estimation prices the exact spec the runner executes: breadth rungs
    (more facts + a bigger pool) and fixed-k chains (k facts at any depth)."""
    base = B._prompt_tokens_est("composite_copy_v2", 64, None)
    # canonical rung (B=16) is the canonical spec — identical estimate
    assert B._prompt_tokens_est("composite_copy_v2", 64, None, 16, None) == base
    b64 = B._prompt_tokens_est("composite_copy_v2", 64, None, 64, None)
    b4 = B._prompt_tokens_est("composite_copy_v2", 64, None, 4, None)
    assert b64 > base > b4  # pool 64 adds ~48 fact lines; pool 4 drops 12
    # B=128 (k=256 > the 128-value vocab) must generate: tasks.generate builds
    # fixed origins only for memorized_recall specs (stream-neutral guard —
    # _fixed_origins draws from its own rng namespace and is unused otherwise)
    b128 = B._prompt_tokens_est("composite_copy_v2", 64, None, 128, None)
    assert b128 > b64
    ex = TK.generate(B.spec_for_cell("composite_copy_v2", 64, breadth=128),
                     "test", n=1, length=64)[0]
    assert ex.answer and ex.meta.get("holder")
    # fixed-k chain: d16 over a 257-cycle prices ~like the d128 staircase's 257
    # facts (the staircase d16 runs k=33), bounded above by d128@k257's deeper query
    stair16 = B._prompt_tokens_est("chain_v2", 16, None)
    stair128 = B._prompt_tokens_est("chain_v2", 128, None)          # k=257
    d16_k257 = B._prompt_tokens_est("chain_v2", 16, None, None, 257)
    assert stair16 < d16_k257 <= stair128
    assert d16_k257 > 3 * stair16
    # the runner and the estimator resolve the SAME spec
    spec = B.spec_for_cell("chain_v2", 16, k_fixed=257)
    assert spec.k == 257
    spec2 = B.spec_for_cell("composite_copy_v2", 64, breadth=64)
    assert (spec2.k, spec2.recall_pool) == (128, 64)
    spec3 = B.spec_for_cell("composite_copy_v2", 64, breadth=16)
    assert spec3 == TK.CANONICAL["composite_copy_v2"]  # B=16 IS canonical


def test_cell_dollar_cap():
    """Per-cell dollar cap: expensive models (completion >= $10/M) get
    max($2.50, nominal n*max_new_tokens completion spend); cheap models None."""
    # opus chain thinking cell: 25 * 16384 * $25/M = $10.24 nominal
    assert B.cell_dollar_cap("anthropic/claude-opus-4.8", 25, 16384) == 10.24
    # tight zero-budget cell floors at $2.50 (nominal would be $0.24)
    assert B.cell_dollar_cap("anthropic/claude-opus-4.8", 100, 96) == 2.5
    # the motivating scenario: a 32768x25 frontier cell was ~$61 under the 3x
    # token guard alone; the dollar cap holds it to its nominal $20.48/$24.58
    assert B.cell_dollar_cap("anthropic/claude-opus-4.8", 25, 32768) == 20.48
    assert B.cell_dollar_cap("openai/gpt-5.5", 25, 32768) == 24.576
    assert B.cell_dollar_cap("anthropic/claude-sonnet-5", 25, 16384) == max(
        2.5, 25 * 16384 * 10 / 1e6)
    # gpt-5.6-sol prices like gpt-5.5 ($30/M) -> same caps apply
    assert B.cell_dollar_cap("openai/gpt-5.6-sol", 25, 32768) == 24.576
    # below the threshold: gemini flash ($9/M) and the cheap tier are uncapped
    assert B.cell_dollar_cap("google/gemini-3.6-flash", 25, 16384) is None
    assert B.cell_dollar_cap("z-ai/glm-5.2", 25, 32768) is None
    assert B.cell_dollar_cap("x-ai/grok-4.5", 25, 16384) is None  # $6/M completion
    assert B.cell_dollar_cap("not/on-roster", 25, 16384) is None


def test_endpoint_for_defaults_and_direct_entry():
    """Per-model direct-endpoint support: registry entries MAY carry
    {"base_url", "api_key_env"}; everything else resolves to the caller's
    default base URL + OPENROUTER_API_KEY. OpenRouter is the default; the live
    muse-spark entry is the one direct-endpoint exception."""
    from unittest import mock
    for slug in B.MODELS:
        if B.MODELS[slug].get("base_url"):
            continue
        assert "base_url" not in B.MODELS[slug] and "api_key_env" not in B.MODELS[slug]
        assert B.endpoint_for(slug) == (B.DEFAULT_BASE_URL, B.DEFAULT_API_KEY_ENV)
    # the runner's --base-url passes through for OpenRouter models
    assert B.endpoint_for("z-ai/glm-5.2", default_base_url="http://mirror:8080/v1") == \
        ("http://mirror:8080/v1", "OPENROUTER_API_KEY")
    # the live muse-spark direct entry overrides both, beating the CLI default
    assert B.endpoint_for("muse-spark-1.1", default_base_url="http://mirror:8080/v1") == \
        ("https://api.meta.ai/v1", "META_API_KEY")
    # a synthetic direct entry (the historical test shape) also overrides both
    fake = {"tier": "cheap_reasoner", "prompt_price_per_M": 1.0,
            "completion_price_per_M": 2.0, "open_weights": False,
            "base_url": "https://api.muse.example/v1", "api_key_env": "MUSE_API_KEY"}
    with mock.patch.dict(B.MODELS, {"muse/spark-1.1": fake}):
        assert B.endpoint_for("muse/spark-1.1", default_base_url="http://mirror:8080/v1") == \
            ("https://api.muse.example/v1", "MUSE_API_KEY")


def test_build_backend_direct_endpoint():
    """The runner builds each model's backend against endpoint_for's
    resolution: a fake direct entry gets its own base_url + key env (and never
    the OpenRouter-specific provider/quantization request options); a missing
    key env fails loudly; default models keep the CLI endpoint + OpenRouter key
    (with the open-weights quantization filter intact). Direct entries flagged
    ``responses_endpoint`` use ``ResponsesBackend``."""
    from unittest import mock
    cell = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "sanity")
    fake = {"tier": "cheap_reasoner", "prompt_price_per_M": 1.0,
            "completion_price_per_M": 2.0, "open_weights": True,  # would filter on OpenRouter
            "base_url": "https://api.muse.example/v1", "api_key_env": "MUSE_API_KEY",
            "responses_endpoint": True}
    with mock.patch.dict(B.MODELS, {"muse/spark-1.1": fake}), \
         mock.patch.dict(os.environ, {"MUSE_API_KEY": "muse-key"}):
        be = RFB.build_backend("muse/spark-1.1", cell, api_key="or-key",
                               base_url=B.DEFAULT_BASE_URL, max_workers=2)
        assert isinstance(be, RFB.ResponsesBackend)
        assert str(be.client.base_url).rstrip("/") == "https://api.muse.example/v1"
        assert be.client.api_key == "muse-key"
        # provider/quantization is an OpenRouter routing option: never sent to
        # a direct vendor endpoint, even for an open-weights model.
        assert "provider" not in (be.extra_body or {})
    # a direct-endpoint model whose key env is unset fails loudly, pre-call
    with mock.patch.dict(B.MODELS, {"muse/spark-1.1": fake}), \
         mock.patch.dict(os.environ):
        os.environ.pop("MUSE_API_KEY", None)
        try:
            RFB.build_backend("muse/spark-1.1", cell, api_key="or-key",
                              base_url=B.DEFAULT_BASE_URL, max_workers=2)
        except SystemExit as exc:
            assert "MUSE_API_KEY" in str(exc)
        else:
            raise AssertionError("expected SystemExit for the missing key env")
    # default (OpenRouter) model: CLI endpoint + passed key + quantization filter
    be2 = RFB.build_backend("z-ai/glm-5.2", cell, api_key="or-key",
                            base_url=B.DEFAULT_BASE_URL, max_workers=2)
    assert isinstance(be2, RFB.APIBackend)
    assert str(be2.client.base_url).rstrip("/") == B.DEFAULT_BASE_URL
    assert be2.client.api_key == "or-key"
    assert be2.extra_body["provider"]["quantizations"] == ["fp8", "bf16", "fp16"]


LOCAL_SLUG = "local/qwen3.6-35b-a3b-nvfp4"


def test_local_arm_is_a_roster_entry_not_a_side_path():
    """The locally served model enters through the SAME machinery as a paid one.

    What that buys, and what this pins: it resolves through ``endpoint_for`` to
    this machine's URL and its own key env; it plans the same facets as any
    cheap_reasoner (nothing is skipped structurally); its cost is zero rather
    than absent, so a record carries cost_usd_est 0.0; and — the one thing that
    could silently change the measurement — its effort=none arm resolves to a
    vendor value. vLLM maps reasoning_effort to the chat template's
    enable_thinking, so an UNMAPPED "none" would send no parameter, leave
    thinking on, and turn every instant cell into a thinking cell while still
    recording effort "none"."""
    reg = B.MODELS[LOCAL_SLUG]
    assert reg["local_served"] is True
    assert B.endpoint_for(LOCAL_SLUG, default_base_url="https://openrouter.ai/api/v1") == \
        (reg["base_url"], "LOCAL_VLLM_API_KEY")
    assert reg["base_url"].startswith("http://127.0.0.1"), "the endpoint must be this machine"
    assert not reg.get("skip_facets")
    assert {c["facet"] for c in B.arms_for(LOCAL_SLUG)} == set(B.FACETS)
    # zero cost, priced through the same path a paid model uses
    est = B.cost_estimate(LOCAL_SLUG, B.arms_for(LOCAL_SLUG))
    assert est["calls"] > 0 and est["cost_usd"] == 0.0
    assert B.cell_dollar_cap(LOCAL_SLUG, 25, 32768) is None  # below the $10/M threshold
    # every planned effort arm maps to a vendor reasoning_effort string
    for cell in B.arms_for(LOCAL_SLUG):
        req = RFB.resolved_request_params(LOCAL_SLUG, cell)
        assert req["reasoning_effort"] == cell["settings"]["effort"], cell["facet"]
        assert req["reasoning_extra_body"] is None  # not an OpenRouter endpoint
        assert req["endpoint"] == "chat_completions"
        assert req["model_name"] == reg["model_name"]
    off = [c for c in B.arms_for(LOCAL_SLUG) if c["settings"]["effort"] == "none"]
    assert off and all(
        RFB.resolved_request_params(LOCAL_SLUG, c)["reasoning_effort"] == "none"
        for c in off)


def test_context_guard_refuses_a_budget_the_server_cannot_hold():
    """A completion budget past the served window is a cell that cannot run.

    vLLM rejects it 4xx per call and the backend (correctly) does not retry a
    non-429 4xx, so the cell lands as empty predictions — which scores as a
    floor. The guard is therefore pre-call, at the plan level AND at the single
    choke point every backend is built through, and it is inert for vendor
    endpoints, whose window is not ours to declare."""
    from unittest import mock
    assert B.context_limit("z-ai/glm-5.2") is None
    cell = next(c for c in B.arms_for(LOCAL_SLUG)
                if c["facet"] == "s5_chain" and c["length"] == 128)
    assert cell["settings"]["max_new_tokens"] == 98304
    # the registry's declared window holds the largest budget the registry plans
    assert B.context_overrun(LOCAL_SLUG, cell) is None
    for c in B.arms_for(LOCAL_SLUG):
        assert B.context_overrun(LOCAL_SLUG, c) is None, c["facet"]
    # a server started at a smaller window is caught, and the message carries
    # both numbers
    over = B.context_overrun(LOCAL_SLUG, cell, limit=32768)
    assert over is not None and over[0] > 98304 and over[1] == 32768
    assert B.context_overrun("z-ai/glm-5.2", cell, limit=None) is None
    # the choke point: build_backend refuses rather than calling
    small = {**B.MODELS[LOCAL_SLUG], "max_model_len": 32768}
    with mock.patch.dict(B.MODELS, {LOCAL_SLUG: small}), \
         mock.patch.dict(os.environ, {"LOCAL_VLLM_API_KEY": "k"}):
        try:
            RFB.build_backend(LOCAL_SLUG, cell, api_key="k",
                              base_url=B.DEFAULT_BASE_URL, max_workers=2)
        except SystemExit as exc:
            assert "32768" in str(exc) and "98304" in str(exc)
        else:
            raise AssertionError("expected SystemExit for an unservable budget")
    # a contract cell is checked at the TOP of its escalation ladder, not at its
    # planned 96-token budget (the runner may rerun it there)
    zb = next(c for c in B.arms_for(LOCAL_SLUG) if c["facet"] == "zero_budget")
    assert RFB.cell_completion_ceiling(zb) == max(RFB.ESCALATION_BUDGETS)
    assert RFB.cell_completion_ceiling(cell) == 98304


STEED_SLUG = "steed/deepseek-v4-flash"


def test_steed_arm_is_a_roster_entry_not_a_side_path():
    """The DeepSeek V4 endpoint on steed enters through the SAME machinery as a paid one.

    It resolves through ``endpoint_for`` to steed's tailnet URL; it plans every
    facet a cheap_reasoner plans (nothing skipped structurally); its cost is zero
    rather than absent, so a record carries cost_usd_est 0.0; every effort arm
    resolves to a vendor string on the TOP-LEVEL parameter (an unmapped "none"
    would send nothing and leave ds4's default think mode ON, turning every
    instant cell into a thinking cell while still recording effort "none"); and
    ONE slug is registered, not one per advertised id — steed lists
    deepseek-v4-pro as well, and it is an alias of the same weights
    (results/probes/steed_ds4_identity_20260802.json)."""
    reg = B.MODELS[STEED_SLUG]
    assert reg["local_served"] is True
    assert B.endpoint_for(STEED_SLUG, default_base_url="https://openrouter.ai/api/v1") == \
        (reg["base_url"], "STEED_DS4_API_KEY")
    assert reg["base_url"].startswith("https://steed."), "the endpoint must be steed"
    assert reg["model_name"] == "deepseek-v4-flash"
    assert not any(s.endswith("deepseek-v4-pro") and s.startswith("steed/") for s in B.MODELS)
    assert not reg.get("skip_facets")
    assert {c["facet"] for c in B.arms_for(STEED_SLUG)} == set(B.FACETS)
    est = B.cost_estimate(STEED_SLUG, B.arms_for(STEED_SLUG))
    assert est["calls"] > 0 and est["cost_usd"] == 0.0
    assert B.cell_dollar_cap(STEED_SLUG, 25, 32768) is None
    for cell in B.arms_for(STEED_SLUG):
        req = RFB.resolved_request_params(STEED_SLUG, cell)
        assert req["reasoning_effort"] == cell["settings"]["effort"], cell["facet"]
        assert req["reasoning_extra_body"] is None   # not an OpenRouter endpoint
        assert req["endpoint"] == "chat_completions"
        assert req["model_name"] == "deepseek-v4-flash"
    off = [c for c in B.arms_for(STEED_SLUG) if c["settings"]["effort"] == "none"]
    assert off and all(
        RFB.resolved_request_params(STEED_SLUG, c)["reasoning_effort"] == "none"
        for c in off)


def test_steed_arm_needs_no_key_clamps_workers_and_sizes_its_own_timeout():
    """Three endpoint properties that are measurements, not defaults.

    The endpoint is tailnet-only and checks no key, so a missing env var is not
    the loud failure it is for muse-spark — but the var is still read and sent
    when set. The server holds ONE KV session (no --batched-session) and
    serializes, so the registry caps concurrency at the measured 1 no matter what
    --max-workers asks for. And its measured completion rate means a cell's budget
    IS its duration — one composed item was still generating after 45 minutes at a
    32,768-token cap — so the shared 30-minute timeout sits below the generation
    time of most thinking budgets, and an openai timeout is an APIConnectionError:
    the backend retries it, regenerating from scratch up to five times. So the
    timeout is sized from the cell's own budget against the SLOW rate."""
    from unittest import mock
    reg = B.MODELS[STEED_SLUG]
    assert reg["api_key_optional"] is True and reg["max_workers"] == 1
    small = next(c for c in B.arms_for(STEED_SLUG) if c["facet"] == "sanity")
    big = next(c for c in B.arms_for(STEED_SLUG)
               if c["facet"] == "s5_chain" and c["length"] == 32)
    with mock.patch.dict(os.environ, {}, clear=True):
        be = RFB.build_backend(STEED_SLUG, small, api_key=None,
                               base_url=B.DEFAULT_BASE_URL, max_workers=8)
        assert be.client.api_key and be.max_workers == 1
        assert be.client.timeout == RFB.DEFAULT_REQUEST_TIMEOUT_S
        be_big = RFB.build_backend(STEED_SLUG, big, api_key=None,
                                   base_url=B.DEFAULT_BASE_URL, max_workers=8)
        # 32,768 tokens at this endpoint's rate is ~45 minutes of generation; the
        # timeout must exceed it, not stop at the shared 30-minute floor.
        assert be_big.client.timeout > 32768 / reg["generation_tok_per_s"]
        assert be_big.client.timeout > RFB.DEFAULT_REQUEST_TIMEOUT_S
        # the key env is honoured when it IS set (adding auth needs no code change)
    with mock.patch.dict(os.environ, {"STEED_DS4_API_KEY": "k9"}, clear=True):
        assert RFB.build_backend(STEED_SLUG, small, api_key=None,
                                 base_url=B.DEFAULT_BASE_URL,
                                 max_workers=8).client.api_key == "k9"
    # a metered direct endpoint keeps the loud failure
    muse = next(c for c in B.arms_for("muse-spark-1.1"))
    with mock.patch.dict(os.environ, {}, clear=True):
        try:
            RFB.build_backend("muse-spark-1.1", muse, api_key=None,
                              base_url=B.DEFAULT_BASE_URL, max_workers=2)
        except SystemExit as exc:
            assert "META_API_KEY" in str(exc)
        else:
            raise AssertionError("a metered endpoint must still demand its key")


def test_declared_context_is_an_equality_locally_and_a_floor_on_steed():
    """Who owns the --ctx flag decides which direction of drift is a fault.

    This repo starts the local vLLM server, so its window is an EQUALITY the run
    enforces: a server at any other length is not the one being described.
    steed's --ctx lives in a unit file outside this repo and is tuned there (the
    endpoint has answered at 65,536, 393,216 and 262,144), so the registry number
    is a FLOOR: a larger live window is reported and then used for the per-cell
    checks, and only a smaller one aborts the run."""
    from unittest import mock
    assert B.MODELS[STEED_SLUG]["context_is_minimum"] is True
    assert not B.MODELS[LOCAL_SLUG].get("context_is_minimum")
    floor = B.context_limit(STEED_SLUG)
    cells = [c for c in B.arms_for(STEED_SLUG) if c["facet"] == "sanity"]

    def probe(served):
        return mock.patch.object(RFB, "served_context",
                                 lambda *a, **k: {"served": "x", "max_model_len": served})

    with probe(floor * 4):        # tuned up: fine, and the bigger number is used
        RFB.preflight_context({STEED_SLUG: cells}, B.DEFAULT_BASE_URL)
    with probe(floor):            # exactly the floor
        RFB.preflight_context({STEED_SLUG: cells}, B.DEFAULT_BASE_URL)
    with probe(floor // 2):       # below it: the budgets were planned on a lie
        try:
            RFB.preflight_context({STEED_SLUG: cells}, B.DEFAULT_BASE_URL)
        except SystemExit as exc:
            assert str(floor // 2) in str(exc) and "serve_steed_model" in str(exc)
        else:
            raise AssertionError("a window below the planned floor must abort")
    # The DRY RUN has no live number and a floor is not a claim about the server,
    # so a floor entry is not checked offline: checking the s5_chain budgets
    # against 65,536 would refuse cells the endpoint serves and abort the whole
    # roster's plan. On the live run those same cells are checked and pass.
    big = [c for c in B.arms_for(STEED_SLUG) if c["facet"] == "s5_chain"]
    assert any(RFB.cell_completion_ceiling(c) > floor for c in big)
    RFB.preflight_context({STEED_SLUG: big}, B.DEFAULT_BASE_URL, probe=False)

    # The plan check and the per-backend check must agree. build_backend is the
    # last line of defence and reads the registry when nothing has probed — which
    # on a floor entry refuses cells the endpoint can serve — so preflight hands
    # it the live number it observed. Without that the cell passes the plan and
    # then dies at run time.
    RFB._SERVED_CONTEXT.pop(STEED_SLUG, None)
    biggest = max(big, key=RFB.cell_completion_ceiling)
    with mock.patch.dict(os.environ, {}, clear=True):
        try:
            RFB.build_backend(STEED_SLUG, biggest, None, B.DEFAULT_BASE_URL, 1)
        except SystemExit:
            pass                      # conservative: nothing has probed yet
        else:
            raise AssertionError("unprobed, the floor must be what bounds it")
        with probe(floor * 4):
            RFB.preflight_context({STEED_SLUG: big}, B.DEFAULT_BASE_URL)
        assert RFB._SERVED_CONTEXT[STEED_SLUG] == floor * 4
        RFB.build_backend(STEED_SLUG, biggest, None, B.DEFAULT_BASE_URL, 1)
    RFB._SERVED_CONTEXT.pop(STEED_SLUG, None)
    # the local arm keeps the strict equality in BOTH directions
    local_cells = [c for c in B.arms_for(LOCAL_SLUG) if c["facet"] == "sanity"]
    with probe(B.context_limit(LOCAL_SLUG) * 2):
        try:
            RFB.preflight_context({LOCAL_SLUG: local_cells}, B.DEFAULT_BASE_URL)
        except SystemExit as exc:
            assert "serve_local_model" in str(exc)
        else:
            raise AssertionError("the local arm's window is an equality")


def test_prompt_tokens_est_prices_the_s5_bind_k_rung_that_runs():
    """k is the only knob that changes an s5_bind prompt's LENGTH at fixed L.

    The k x L sweep runs rungs the registry does not carry, so a context guard
    reading the canonical k=12 spec under-counts the k=32 prompt and would admit
    a budget the server cannot hold. ``k_sweep`` rides in settings (it is already
    part of the settings hash, hence of the resume key), and both the guard and
    the cost estimate resolve the spec through it. It is sentinel-absent on every
    registry facet, so no shipped cell's estimate moves."""
    canonical = B._prompt_tokens_est("s5_bind_v3", 256, None, None, None, None)
    wide = B._prompt_tokens_est("s5_bind_v3", 256, None, None, None, 32)
    narrow = B._prompt_tokens_est("s5_bind_v3", 256, None, None, None, 6)
    assert narrow < canonical < wide
    assert B._prompt_tokens_est("s5_bind_v3", 256, None, None, None, 12) == canonical
    # the guard flips on the rung's own prompt, not the canonical one
    settings = B._settings("high", max_new_tokens=53346)
    settings["k_sweep"] = 32
    cell = {"facet": "s5_bind_v3_kl_sweep", "task": "s5_bind_v3", "length": 256,
            "n": 40, "settings": settings}
    assert B.context_overrun(STEED_SLUG, cell, limit=65536) is None
    cell["settings"]["max_new_tokens"] = 53347
    assert B.context_overrun(STEED_SLUG, cell, limit=65536) is not None
    # non-s5_bind specs are untouched by the new argument
    for task, length in (("composite_copy_v2", 16), ("chain_v2", 32), ("s5_chain_v4", 64)):
        assert (B.spec_for_cell(task, length, k_sweep=32)
                == B.spec_for_cell(task, length))


def test_resolved_request_params_records_what_was_sent():
    """settings.effort is the PROTOCOL arm; the registry maps it to a vendor value,
    and that mapping has changed under a fixed arm name (gpt-5.6-sol's archived
    s5_chain_v3 batches all record "xhigh" but ran chat completions at xhigh, the
    Responses endpoint at max, and the Responses endpoint at xhigh). The resolution
    is recorded per cell; it is NOT part of the resume key."""
    from unittest import mock
    sol_cell = next(c for c in B.arms_for("openai/gpt-5.6-sol")
                    if c["facet"] == "s5_chain")
    req = RFB.resolved_request_params("openai/gpt-5.6-sol", sol_cell)
    assert req == {
        "endpoint": "responses",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-5.6-sol",          # the literal string sent, no prefix
        "effort_arm": "xhigh",
        "reasoning_effort": "xhigh",          # the vendor value
        "reasoning_extra_body": None,         # this endpoint rejects the block
    }

    # the exact drift the archive cannot resolve: same protocol arm, "max" sent
    sol_max = {**B.MODELS["openai/gpt-5.6-sol"],
               "reasoning_effort_values": {"low": "low", "medium": "medium",
                                           "high": "high", "xhigh": "max",
                                           "max": "max"}}
    key_before = RFB.cell_key("openai/gpt-5.6-sol", sol_cell)
    with mock.patch.dict(B.MODELS, {"openai/gpt-5.6-sol": sol_max}):
        drifted = RFB.resolved_request_params("openai/gpt-5.6-sol", sol_cell)
        # captured UNDER the patched mapping: the key must not read the resolution
        key_under_new_mapping = RFB.cell_key("openai/gpt-5.6-sol", sol_cell)
    assert drifted["reasoning_effort"] == "max"
    assert drifted != req
    # the resolution is diagnostics, not settings — the resume key is the protocol
    # cell, so the mapping change does not re-buy the roster.
    assert key_under_new_mapping == key_before
    assert "reasoning_effort" not in sol_cell["settings"]

    # OpenRouter models carry the effort in the extra body, not the top-level param
    glm_cell = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "s5_chain")
    glm = RFB.resolved_request_params("z-ai/glm-5.2", glm_cell)
    assert glm["endpoint"] == "chat_completions"
    assert glm["base_url"] == B.DEFAULT_BASE_URL
    assert glm["model_name"] == "z-ai/glm-5.2"      # no model_name override
    assert glm["reasoning_extra_body"] == {"effort": "xhigh"}
    assert glm["reasoning_effort"] is None
    # a --base-url mirror is recorded as the endpoint actually used
    assert RFB.resolved_request_params(
        "z-ai/glm-5.2", glm_cell, default_base_url="http://mirror:8080/v1"
    )["base_url"] == "http://mirror:8080/v1"

    # gemini's substituted off-arm is recorded as the value actually sent
    gem = next(c for c in B.arms_for("google/gemini-3.6-flash")
               if c["facet"] == "sanity")
    assert RFB.resolved_request_params("google/gemini-3.6-flash", gem)[
        "reasoning_extra_body"] == {"effort": "minimal"}
    # muse-spark: direct Meta endpoint, Responses API
    muse = next(c for c in B.arms_for("muse-spark-1.1") if c["facet"] == "s5_chain")
    mreq = RFB.resolved_request_params("muse-spark-1.1", muse)
    assert (mreq["endpoint"], mreq["base_url"]) == ("responses",
                                                    "https://api.meta.ai/v1")


def test_build_backend_matches_resolved_request_params():
    """build_backend builds FROM resolved_request_params and carries it on the
    backend, so what is recorded is what was sent."""
    from unittest import mock
    glm_cell = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "s5_chain")
    be = RFB.build_backend("z-ai/glm-5.2", glm_cell, api_key="or-key",
                           base_url=B.DEFAULT_BASE_URL, max_workers=2)
    req = RFB.resolved_request_params("z-ai/glm-5.2", glm_cell)
    assert be.request_params == req
    assert be.extra_body["reasoning"] == req["reasoning_extra_body"]
    assert be.reasoning_effort == req["reasoning_effort"]
    assert be.model_name == req["model_name"]
    assert str(be.client.base_url).rstrip("/") == req["base_url"]

    sol_cell = next(c for c in B.arms_for("openai/gpt-5.6-sol")
                    if c["facet"] == "s5_chain")
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        be2 = RFB.build_backend("openai/gpt-5.6-sol", sol_cell, api_key="or-key",
                                base_url=B.DEFAULT_BASE_URL, max_workers=2)
    req2 = RFB.resolved_request_params("openai/gpt-5.6-sol", sol_cell)
    assert isinstance(be2, RFB.ResponsesBackend)
    assert be2.request_params == req2
    assert be2.reasoning_effort == "xhigh"
    assert be2.model_name == "gpt-5.6-sol"
    assert "reasoning" not in (be2.extra_body or {})


def test_execute_cell_records_request_diagnostics():
    """The record carries the resolution under diagnostics.request (a copy, once
    per cell), and history_request_params/request_drift surface a resume hit whose
    LATEST record was made under a different resolution than the run would send
    now — superseded resolutions are history, not a live mismatch."""
    model = "openai/gpt-5.6-sol"
    cell = next(c for c in B.arms_for(model) if c["facet"] == "s5_chain")
    cell["n"] = 5
    backend = _MetaBackend([{"finish": "stop", "ctok": 20, "rtok": 8, "text": "g3 ."}])
    # the archived regime: same protocol arm, the Responses endpoint at "max"
    backend.request_params = {**RFB.resolved_request_params(model, cell),
                              "reasoning_effort": "max"}
    rec = RFB.execute_cell(backend, model, cell, n=5, run_id="t", git_commit="d")
    assert rec["diagnostics"]["request"]["reasoning_effort"] == "max"
    assert rec["diagnostics"]["request"]["endpoint"] == "responses"
    assert rec["diagnostics"]["request"]["model_name"] == "gpt-5.6-sol"
    assert rec["settings"]["effort"] == "xhigh"   # the PROTOCOL arm, as recorded
    _validate_c3(rec, cell, model)
    # a copy, not the backend's dict
    backend.request_params["reasoning_effort"] = "mutated"
    assert rec["diagnostics"]["request"]["reasoning_effort"] == "max"

    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "history.jsonl")
        RFB.append_record(history, rec)
        done = RFB.history_keys(history)
        recorded = RFB.history_request_params(history)
        # the record still satisfies resume (the resume key is the protocol cell)...
        assert RFB.should_skip(model, cell, done, force=False, canary=False)
        # ...but the mixing is not silent: the live registry resolves xhigh -> xhigh
        drift = RFB.request_drift(model, cell, recorded, B.DEFAULT_BASE_URL)
        assert drift["reasoning_effort"] == "max"

        # a LATER record made under the current resolution supersedes it: the
        # renderer publishes the latest, so the warning clears instead of firing
        # forever on a resolution the archive has already moved past.
        backend2 = _MetaBackend([{"finish": "stop", "ctok": 20, "rtok": 8,
                                  "text": "g3 ."}])
        backend2.request_params = RFB.resolved_request_params(model, cell)
        rec2 = RFB.execute_cell(backend2, model, cell, n=5, run_id="t2",
                                git_commit="d")
        assert rec2["ts"] > rec["ts"]
        RFB.append_record(history, rec2)
        recorded2 = RFB.history_request_params(history)
        assert recorded2[RFB.cell_key(model, cell)] == backend2.request_params
        assert RFB.request_drift(model, cell, recorded2, B.DEFAULT_BASE_URL) is None

        # latest is by ts, not by file position: an OLDER drifted record appended
        # last does not resurrect the warning
        stale = {**rec, "run_id": "t0", "ts": "2000-01-01T00:00:00+00:00"}
        RFB.append_record(history, stale)
        assert RFB.request_drift(model, cell,
                                 RFB.history_request_params(history),
                                 B.DEFAULT_BASE_URL) is None
        # ...while a NEWER drifted record does
        fresh = {**rec, "run_id": "t3", "ts": "2099-01-01T00:00:00+00:00"}
        RFB.append_record(history, fresh)
        assert RFB.request_drift(model, cell,
                                 RFB.history_request_params(history),
                                 B.DEFAULT_BASE_URL)["reasoning_effort"] == "max"

    # backends that resolve no vendor request (the FunctionBackend path) omit it
    plain = RFB.execute_cell(
        FunctionBackend(lambda ps, m, s: ["g3 ."] * len(ps), name="f"),
        model, cell, n=5, run_id="t", git_commit="d")
    assert "request" not in plain["diagnostics"]
    _validate_c3(plain, cell, model)
    assert RFB.history_request_params(os.path.join(tempfile.gettempdir(),
                                                   "factworld-no-such-history")) == {}


def test_cost_estimate_sane():
    glm_cells = [_runnable(c) for c in B.arms_for("z-ai/glm-5.2")]
    est = B.cost_estimate("z-ai/glm-5.2", glm_cells)
    assert est["calls"] == sum(c["n"] for c in glm_cells)
    assert est["prompt_tokens"] > 0 and est["completion_tokens"] > 0
    assert 0 < est["cost_usd"] < 50  # the cheap canary must stay cheap
    # more assumed thinking -> strictly more money
    more = B.cost_estimate("z-ai/glm-5.2", glm_cells, assumed_output_tokens=4000)
    assert more["cost_usd"] > est["cost_usd"]
    # same plan priced at opus rates costs more than at glm rates
    opus = B.cost_estimate("anthropic/claude-opus-4.8", glm_cells)
    assert opus["cost_usd"] > est["cost_usd"]
    # longer composite prompts cost more prompt tokens
    short = B._prompt_tokens_est("composite_copy_v2", 16, None)
    long = B._prompt_tokens_est("composite_copy_v2", 64, None)
    assert long > short * 2
    # chain estimates use the no-wrap staircase (k=2*depth+1), so depth 128 must
    # not trip chain_v2's wrap validity gate and deeper chains cost more.
    d16 = B._prompt_tokens_est("chain_v2", 16, None)
    d128 = B._prompt_tokens_est("chain_v2", 128, None)
    assert d128 > d16 * 3
    # composite_copy_v2 keeps v1's L/pool/prompt shape, so its prompt size is
    # comparable to the retired spec's (regenerated via RETIRED — _prompt_tokens_est
    # itself only prices CANONICAL tasks, which is all the registry plans).
    v1_ex = TK.generate(TK.RETIRED["composite_copy_v1"], "test", n=1, length=16)[0]
    v2_ex = TK.generate(TK.CANONICAL["composite_copy_v2"], "test", n=1, length=16)[0]
    assert 0.8 * len(v1_ex.prompt) < len(v2_ex.prompt) < 1.2 * len(v1_ex.prompt)


# --- contract extraction -------------------------------------------------------------

def test_extract_contract_answer():
    # last Answer: line wins (the sonnet/kimi fix: working THEN the contract line)
    text = "Let me think.\nAnswer: g1 v2 maybe? No.\nMore work...\nAnswer: g3 v9"
    assert RFB.extract_contract_answer(text) == "g3 v9"
    # markdown emphasis around the line is stripped
    assert RFB.extract_contract_answer("**Answer:** g3 v9") == "g3 v9"
    # case-insensitive
    assert RFB.extract_contract_answer("answer: g0") == "g0"
    # no line / empty span -> None (counts as a contract miss)
    assert RFB.extract_contract_answer("the holder is g3") is None
    assert RFB.extract_contract_answer("Answer:") is None
    assert RFB.extract_contract_answer("") is None


# --- runner cell execution (no API) ------------------------------------------------

def _validate_c3(rec, cell, model):
    expected = C3_TOP_KEYS | ({"escalation"} if rec["escalated"] else set())
    assert set(rec) == expected
    assert rec["model"] == model
    assert rec["facet"] == cell["facet"] and rec["task"] == cell["task"]
    assert rec["length"] == cell["length"] and rec["n"] == cell["n"]
    assert set(rec["settings"]) == SETTINGS_KEYS
    assert set(rec["metrics"]) == {"relaxed", "exact", "contains", "last_n"}
    assert rec["metrics"]["relaxed"] is not None  # relaxed is ALWAYS present
    diag_keys = {"empty_rate", "truncated_rate", "api_errors", "finish_errors",
                 "finish_reasons", "cost_aborted"}
    if "request" in rec["diagnostics"]:
        # what was actually sent (vendor effort, endpoint, base URL, model name):
        # present whenever the backend came from build_backend, absent on the
        # FunctionBackend test path, which resolves no vendor request at all.
        diag_keys |= {"request"}
    if rec["diagnostics"]["cost_aborted"]:
        diag_keys |= {"calls_completed", "cost_abort_reason"}
    if rec["settings"]["contract"]:
        diag_keys |= {"contract_rate", "covert_cot_rate", "rtok_any_rate",
                      "rtok_mean_per_call"}
    assert set(rec["diagnostics"]) == diag_keys
    assert set(rec["usage"]) == {"prompt_tokens", "completion_tokens",
                                 "reasoning_tokens", "cost_usd_est"}
    # per-example records: gold/pred/relaxed + per-call usage; NO prompt text
    assert rec["examples"] and all(set(e) == EXAMPLE_KEYS for e in rec["examples"])


def test_execute_cell_end_to_end():
    """FunctionBackend mini-run: s5_concrete + sanity + zero_budget cells -> valid
    C3 records in a tmp history file, and the resume key round-trips."""
    model = "z-ai/glm-5.2"
    cells = B.arms_for(model)
    s5_cell = next(c for c in cells if c["facet"] == "s5_concrete" and c["length"] == 128)
    sanity_cell = next(c for c in cells if c["facet"] == "sanity" and c["task"] == "recall_copy_v1")
    zb_cell = _runnable(next(c for c in cells if c["settings"]["leg"] == "binding_only"))
    for c in (s5_cell, sanity_cell, zb_cell):
        c["n"] = 5

    backend = FunctionBackend(lambda prompts, mnt, stop: ["Driver ."] * len(prompts),
                              name="fake")
    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "sub", "history.jsonl")  # exercises mkdir -p
        for cell in (s5_cell, sanity_cell, zb_cell):
            rec = RFB.execute_cell(backend, model, cell, n=cell["n"],
                                   run_id="bench_test", git_commit="deadbeef")
            _validate_c3(rec, cell, model)
            assert rec["escalated"] is False
            RFB.append_record(history, rec)

        with open(history) as fh:
            lines = [json.loads(line) for line in fh]
        assert len(lines) == 3

        # s5 cell: "Driver ." matches some golds; relaxed in [0,1]; s5 has no exact/last_n.
        s5_rec = lines[0]
        assert s5_rec["metrics"]["exact"] is None and s5_rec["metrics"]["last_n"] is None
        assert 0.0 <= s5_rec["metrics"]["relaxed"] <= 1.0
        assert s5_rec["diagnostics"]["empty_rate"] == 0.0
        # FunctionBackend has no per-example meta -> explicit Nones, keys still present.
        assert all(e["ctok"] is None and e["rtok"] is None and e["finish"] is None
                   for e in s5_rec["examples"])
        # sanity cell went through evaluate_task: all four metrics populated.
        assert all(v is not None for v in lines[1]["metrics"].values())
        # zero_budget leg record: "Driver ." has no Answer: line -> contract_rate 0,
        # empty preds, relaxed 0.
        zb_rec = lines[2]
        assert zb_rec["settings"]["leg"] == "binding_only"
        assert zb_rec["diagnostics"]["contract_rate"] == 0.0
        assert zb_rec["metrics"]["relaxed"] == 0.0
        assert zb_rec["diagnostics"]["empty_rate"] == 1.0

        # resume: every written cell's key is now recognized as done.
        done = RFB.history_keys(history)
        for cell in (s5_cell, sanity_cell, zb_cell):
            assert RFB.cell_key(model, cell) in done
            assert RFB.should_skip(model, cell, done, force=False, canary=False)
        # --force and --canary (for the glm canary model) bypass the skip.
        assert not RFB.should_skip(model, s5_cell, done, force=True, canary=False)
        assert not RFB.should_skip(model, s5_cell, done, force=False, canary=True)
        # a cell that never ran is not skipped
        other = next(c for c in cells if c["facet"] == "chain_nowrap")
        assert not RFB.should_skip(model, other, done, force=False, canary=False)


def test_zero_budget_contract_scoring():
    """Contract cells append the contract line to every prompt and score the span
    after the LAST Answer: line; an oracle answering through the contract gets 1.0."""
    model = "z-ai/glm-5.2"
    cells = B.arms_for(model)
    plain = _runnable(next(c for c in cells if c["facet"] == "zero_budget"
                           and c["settings"]["leg"] is None and c["length"] == 16))
    plain["n"] = 5

    spec = TK.CANONICAL[plain["task"]]
    gold = {e.prompt: e.answer for e in TK.generate(spec, "test", n=5, length=16)}
    seen_prompts = []

    def oracle(prompts, mnt, stop):
        seen_prompts.extend(prompts)
        out = []
        for p in prompts:
            base, line = p.rsplit("\n", 1)
            assert line == RFB.CONTRACT_LINE_COMPOSITE
            # emit working THEN the contract line (the sonnet-style transcript)
            out.append(f"tracking gives... let me check.\nAnswer: {gold[base].rstrip(' .')}")
        return out

    rec = RFB.execute_cell(FunctionBackend(oracle, name="oracle"), model, plain,
                           n=5, run_id="t", git_commit="d")
    assert len(seen_prompts) == 5
    assert rec["metrics"]["relaxed"] == 1.0
    assert rec["diagnostics"]["contract_rate"] == 1.0

    # the binding leg uses the holder-only contract line
    bind = _runnable(next(c for c in cells if c["settings"]["leg"] == "binding_only"))
    bind["n"] = 5
    lines = []
    rec2 = RFB.execute_cell(
        FunctionBackend(lambda ps, m, s: [lines.append(p.rsplit("\n", 1)[1]) or "x"
                                          for p in ps], name="probe"),
        model, bind, n=5, run_id="t", git_commit="d")
    assert set(lines) == {RFB.CONTRACT_LINE_BINDING}
    assert rec2["diagnostics"]["contract_rate"] == 0.0


def test_spec_for_cell_recall_load_scaling():
    """Non-memorized recall past the canonical agent pool scales k to the length
    (pool = min(L, k), so pool-64 needs k >= 64); the sanity row's L=6 resolves
    to the untouched canonical spec (its resume keys/streams are pinned)."""
    spec = B.spec_for_cell("recall_copy_v1", 64)
    assert (spec.k, spec.family, spec.memorized_recall) == (64, "recall", False)
    assert B.spec_for_cell("recall_copy_v1", 6) == TK.CANONICAL["recall_copy_v1"]
    # the generated pool is EXACTLY 64 distinct agents, distractors included
    import re
    for e in TK.generate(spec, "test", n=3, length=64):
        agents = re.findall(r"(g\d+)'s a0 is v\d+\.", e.prompt)
        assert len(agents) == len(set(agents)) == 64
    # memorized-recall controls never scale (fixed-map lookup, not load)
    assert B.spec_for_cell("recall_v1", 64) == TK.CANONICAL["recall_v1"]


def test_recall_load_and_chain_instant_contract_cells():
    """The new instant rows run the contract path with family-matched contract
    lines (recall -> <value>, chain -> <agent>); an oracle answering through the
    contract scores 1.0, and chain_instant runs the SAME staircase spec/items as
    the chain_nowrap d16 thinking cell (within-item regime contrast)."""
    model = "z-ai/glm-5.2"
    cells = B.arms_for(model)
    for facet, line in (("recall_load", RFB.CONTRACT_LINE_VALUE),
                        ("chain_instant", RFB.CONTRACT_LINE_AGENT)):
        cell = _runnable(next(c for c in cells if c["facet"] == facet))
        cell["n"] = 5
        spec = B.spec_for_cell(cell["task"], cell["length"])
        gold = {e.prompt: e.answer
                for e in TK.generate(spec, "test", n=5, length=cell["length"])}
        seen_lines = []

        def oracle(prompts, mnt, stop):
            out = []
            for p in prompts:
                base, last = p.rsplit("\n", 1)
                seen_lines.append(last)
                out.append(f"Answer: {gold[base].rstrip(' .')}")
            return out

        rec = RFB.execute_cell(FunctionBackend(oracle, name="oracle"), model, cell,
                               n=5, run_id="t", git_commit="d")
        assert set(seen_lines) == {line}, facet
        assert rec["metrics"]["relaxed"] == 1.0, facet
        assert rec["diagnostics"]["contract_rate"] == 1.0, facet
        _validate_c3(rec, cell, model)
    # within-item contrast: chain_instant resolves the same spec as the
    # chain_nowrap d16 staircase cell (k=2*16+1=33)
    assert B.spec_for_cell("chain_v2", 16).k == 33


def test_binding_leg_rejects_holder_dump():
    """A holder-dump reply ("Answer: zz g0 g1 g2 ...") must score 0 on the binding
    leg: the span's FIRST content token must be the holder (membership scoring had
    a 100% false-positive rate against this — kimi exploited it live)."""
    model = "z-ai/glm-5.2"
    bind = _runnable(next(c for c in B.arms_for(model)
                          if c["settings"]["leg"] == "binding_only"))
    bind["n"] = 5

    dump = "Answer: zz " + " ".join(f"g{i}" for i in range(64))  # every candidate holder
    rec = RFB.execute_cell(
        FunctionBackend(lambda ps, m, s: [dump] * len(ps), name="dump"),
        model, bind, n=5, run_id="t", git_commit="d")
    assert rec["diagnostics"]["contract_rate"] == 1.0  # well-formed contract line...
    assert rec["metrics"]["relaxed"] == 0.0            # ...but no committed holder

    # positive control: committing to the correct holder (with trailing period,
    # relaxed-style) scores 1.0.
    spec = TK.CANONICAL[bind["task"]]
    examples = TK.generate(spec, "test", n=5, length=16)
    from experiment_autoregressive import binding_prompt
    holder = {binding_prompt(e, spec.name)[0]: e.meta["holder"] for e in examples}
    rec2 = RFB.execute_cell(
        FunctionBackend(lambda ps, m, s: [f"Answer: {holder[p.rsplit(chr(10), 1)[0]]}."
                                          for p in ps], name="oracle"),
        model, bind, n=5, run_id="t", git_commit="d")
    assert rec2["metrics"]["relaxed"] == 1.0


class _MetaBackend:
    """FunctionBackend-alike exposing APIBackend's meta interface, with scriptable
    per-attempt finish reasons / token counts."""

    name = "fakemeta"

    def __init__(self, attempts):
        # attempts: list of dicts {"finish", "ctok", "rtok", "text"} consumed per generate call
        self.attempts = list(attempts)
        self.budgets = []
        self._last = None

    def generate(self, prompts, max_new_tokens, stop_at=None):
        self.budgets.append(max_new_tokens)
        self._last = (self.attempts.pop(0), len(prompts))
        att, n = self._last
        return [att["text"]] * n

    def pop_example_meta(self):
        att, n = self._last
        return [{"completion_tokens": att["ctok"], "reasoning_tokens": att["rtok"],
                 "finish_reason": att["finish"]}] * n

    def pop_call_meta(self):
        att, n = self._last
        return {"calls": n, "errors": 0,
                "usage": {"prompt_tokens": 10 * n, "completion_tokens": att["ctok"] * n,
                          "reasoning_tokens": att["rtok"] * n},
                "served_models": ["served/m"], "providers": ["P"],
                "finish_reasons": {att["finish"]: n}}


def _zb_cell(n=5):
    cell = _runnable(next(c for c in B.arms_for("z-ai/glm-5.2")
                          if c["facet"] == "zero_budget" and c["settings"]["leg"] is None
                          and c["length"] == 16))
    cell["n"] = n
    return cell


def test_escalation_on_length_cutoff():
    """A contract cell with finish=length > 10% reruns at the next escalation
    budget (96 -> 512) and stops once clean. The record keeps EVERY attempt in
    full under escalation.attempts (first attempt canonical per review F2), the
    top-level metrics/examples are the last attempt's (escalated=true), usage is
    summed across attempts, planned settings preserved for resume."""
    backend = _MetaBackend([
        {"finish": "length", "ctok": 96, "rtok": 0, "text": "truncated working with no answer line"},
        {"finish": "stop", "ctok": 400, "rtok": 2, "text": "long working...\nAnswer: g3 v9"},
    ])
    cell = _zb_cell()
    rec = RFB.execute_cell(backend, "z-ai/glm-5.2", cell, n=5,
                           run_id="t", git_commit="d")
    assert backend.budgets == [96, RFB.ESCALATION_BUDGETS[0]] == [96, 512]
    assert rec["escalated"] is True
    # published (top-level) diagnostics/examples come from the LAST attempt
    assert rec["diagnostics"]["finish_reasons"] == {"stop": 5}
    assert rec["diagnostics"]["contract_rate"] == 1.0
    assert all(e["finish"] == "stop" and e["ctok"] == 400 for e in rec["examples"])
    # rerun cleanliness diagnostics: 400 ctok > threshold, rtok leak present
    assert rec["diagnostics"]["covert_cot_rate"] == 1.0
    assert rec["diagnostics"]["rtok_any_rate"] == 1.0
    assert rec["diagnostics"]["rtok_mean_per_call"] == 2.0
    esc = rec["escalation"]
    assert esc["max_new_tokens"] == 512
    # attempts: BOTH attempts recorded in full, index-stamped, budgets in order
    assert [a["attempt"] for a in esc["attempts"]] == [0, 1]
    assert [a["max_new_tokens"] for a in esc["attempts"]] == [96, 512]
    first, last = esc["attempts"][0], esc["attempts"][-1]
    # the FIRST attempt (the canonical number) keeps its examples VERBATIM —
    # per-call ctok/rtok/finish and the empty preds of the truncated replies.
    assert len(first["examples"]) == 5
    assert all(e["finish"] == "length" and e["ctok"] == 96 and e["rtok"] == 0
               and e["pred"] == "" and e["relaxed"] == 0 for e in first["examples"])
    assert first["relaxed"] == 0.0 and first["metrics"]["relaxed"] == 0.0
    assert first["diagnostics"]["finish_reasons"] == {"length": 5}
    assert first["diagnostics"]["contract_rate"] == 0.0
    assert first["length_rate"] == 1.0
    # per-attempt usage stays per-attempt (NOT the summed record usage)
    assert first["usage"]["completion_tokens"] == 5 * 96
    assert last["usage"]["completion_tokens"] == 5 * 400
    assert last["examples"] == rec["examples"]
    # legacy alias for the pre-attempts renderer schema: summary of attempt 0
    fa = esc["first_attempt"]
    assert fa["max_new_tokens"] == 96
    assert fa["relaxed"] == 0.0
    assert fa["diagnostics"] == first["diagnostics"]
    assert "examples" not in fa
    # spend honesty: top-level usage covers BOTH attempts
    assert rec["usage"]["prompt_tokens"] == 100  # 2 attempts x 5 calls x 10
    assert rec["usage"]["completion_tokens"] == 5 * 96 + 5 * 400
    # resume key unchanged: the record's settings are the PLANNED ones, so the
    # written record satisfies the original cell's resume key.
    assert rec["settings"]["max_new_tokens"] == 96
    assert B.settings_hash(rec) == B.settings_hash(cell)
    _validate_c3(rec, cell, "z-ai/glm-5.2")


def test_escalation_iterates_to_2048():
    """Escalation is iterated (review: one-shot escalation left kimi L16 with 9
    residual cap-outs at 512): still >10% finish=length at 512 -> a second rerun
    at 2048, and it hard-stops there (at most 2 escalations)."""
    backend = _MetaBackend([
        {"finish": "length", "ctok": 96, "rtok": 0, "text": "cut"},
        {"finish": "length", "ctok": 512, "rtok": 0, "text": "still cut"},
        {"finish": "length", "ctok": 2048, "rtok": 0, "text": "STILL cut"},
    ])
    cell = _zb_cell()
    rec = RFB.execute_cell(backend, "z-ai/glm-5.2", cell, n=5,
                           run_id="t", git_commit="d")
    # 96 -> 512 -> 2048, then stop even though finish=length persists.
    assert backend.budgets == [96, 512, 2048]
    assert list(RFB.ESCALATION_BUDGETS) == [512, 2048]
    assert rec["escalated"] is True
    esc = rec["escalation"]
    assert esc["max_new_tokens"] == 2048 and esc["length_rate"] == 1.0
    assert [a["max_new_tokens"] for a in esc["attempts"]] == [96, 512, 2048]
    # every attempt carries complete per-example data
    for a, ctok in zip(esc["attempts"], (96, 512, 2048)):
        assert len(a["examples"]) == 5
        assert all(e["ctok"] == ctok and e["finish"] == "length" for e in a["examples"])
    # first_attempt alias still points at attempt 0
    assert esc["first_attempt"]["max_new_tokens"] == 96
    # usage covers all three attempts
    assert rec["usage"]["completion_tokens"] == 5 * (96 + 512 + 2048)
    assert rec["usage"]["prompt_tokens"] == 150
    _validate_c3(rec, cell, "z-ai/glm-5.2")


def test_no_escalation_below_threshold_or_without_contract():
    # 0/5 length finishes: no escalation, single generate call.
    backend = _MetaBackend([{"finish": "stop", "ctok": 20, "rtok": 0,
                             "text": "Answer: g3 v9"}])
    rec = RFB.execute_cell(backend, "z-ai/glm-5.2", _zb_cell(), n=5,
                           run_id="t", git_commit="d")
    assert backend.budgets == [96]
    assert rec["escalated"] is False and "escalation" not in rec
    assert rec["diagnostics"]["covert_cot_rate"] == 0.0
    assert rec["diagnostics"]["rtok_any_rate"] == 0.0
    assert rec["diagnostics"]["rtok_mean_per_call"] == 0.0
    assert rec["diagnostics"]["cost_aborted"] is False

    # non-contract cells never escalate even at 100% finish=length.
    sanity = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "sanity" and c["task"] == "recall_copy_v1")
    sanity["n"] = 5
    backend2 = _MetaBackend([{"finish": "length", "ctok": 96, "rtok": 0, "text": "v0"}])
    rec2 = RFB.execute_cell(backend2, "z-ai/glm-5.2", sanity, n=5,
                            run_id="t", git_commit="d")
    assert backend2.budgets == [2048]
    assert rec2["escalated"] is False
    assert all(e["finish"] == "length" for e in rec2["examples"])


def test_finish_errors_counted():
    """finish_reason=='error' calls are counted into diagnostics.finish_errors,
    SEPARATE from api_errors (review F8: 12 finish=error calls were invisible at
    api_errors=0)."""
    backend = _MetaBackend([{"finish": "error", "ctok": 0, "rtok": 0, "text": ""}])
    sanity = next(c for c in B.arms_for("z-ai/glm-5.2") if c["facet"] == "sanity" and c["task"] == "recall_copy_v1")
    sanity["n"] = 5
    rec = RFB.execute_cell(backend, "z-ai/glm-5.2", sanity, n=5,
                           run_id="t", git_commit="d")
    assert rec["diagnostics"]["finish_errors"] == 5
    assert rec["diagnostics"]["api_errors"] == 0  # the F8 signature
    assert rec["diagnostics"]["finish_reasons"] == {"error": 5}
    _validate_c3(rec, sanity, "z-ai/glm-5.2")

    # clean cells report 0 (the key is always present)
    backend2 = _MetaBackend([{"finish": "stop", "ctok": 5, "rtok": 0, "text": "v0"}])
    rec2 = RFB.execute_cell(backend2, "z-ai/glm-5.2", sanity, n=5,
                            run_id="t", git_commit="d")
    assert rec2["diagnostics"]["finish_errors"] == 0


class _RunawayBackend:
    """Meta-capable fake that emits a huge visible ctok on every call (the
    grok-build failure mode: a pinned generator ignoring the token cap)."""

    name = "runaway"

    def __init__(self, ctok_per_call, finish="stop", text="Answer: g3 v9"):
        self.ctok = ctok_per_call
        self.finish = finish
        self.text = text
        self.calls = 0
        self._last_n = 0

    def generate(self, prompts, max_new_tokens, stop_at=None):
        self.calls += len(prompts)
        self._last_n = len(prompts)
        return [self.text] * len(prompts)

    def pop_example_meta(self):
        return [{"completion_tokens": self.ctok, "reasoning_tokens": 0,
                 "finish_reason": self.finish}] * self._last_n

    def pop_call_meta(self):
        n = self._last_n
        return {"calls": n, "errors": 0,
                "usage": {"prompt_tokens": 10 * n, "completion_tokens": self.ctok * n,
                          "reasoning_tokens": 0},
                "served_models": ["served/m"], "providers": ["P"],
                "finish_reasons": {self.finish: n}}


def test_cost_guard_aborts_runaway_cell():
    """Once cumulative ctok exceeds CELL_BUDGET_FACTOR * n * max_new_tokens the
    guard stops submitting calls; what completed is recorded, the rest get empty
    preds with finish=cost_aborted, and the cell is flagged cost_aborted."""
    cell = _zb_cell(n=25)
    # budget = 3 * 25 * 96 = 7200 ctok; 5000 ctok/call trips after the 1st chunk (8 calls)
    backend = _RunawayBackend(ctok_per_call=5000)
    rec = RFB.execute_cell(backend, "z-ai/glm-5.2", cell, n=25,
                           run_id="t", git_commit="d")
    assert B.CELL_BUDGET_FACTOR == 3
    chunk = RFB.CostGuardBackend.CHUNK
    assert backend.calls == chunk == 8  # stopped submitting after one chunk
    d = rec["diagnostics"]
    assert d["cost_aborted"] is True
    assert d["cost_abort_reason"] == "ctok"  # the token envelope, not the $ cap
    assert d["calls_completed"] == chunk
    # completed calls recorded verbatim; the rest are explicit cost_aborted stubs
    done = [e for e in rec["examples"] if e["finish"] == "stop"]
    stub = [e for e in rec["examples"] if e["finish"] == "cost_aborted"]
    assert len(done) == chunk and len(stub) == 25 - chunk
    assert all(e["ctok"] == 5000 for e in done)
    assert all(e["ctok"] is None and e["pred"] == "" for e in stub)
    # usage only covers what actually ran
    assert rec["usage"]["completion_tokens"] == 5000 * chunk
    _validate_c3(rec, cell, "z-ai/glm-5.2")


def test_cost_guard_blocks_escalation():
    """A cost-aborted attempt must not escalate (escalating a runaway cell would
    multiply the spend), even when its finish=length rate exceeds the threshold."""
    backend = _RunawayBackend(ctok_per_call=5000, finish="length", text="cut")
    rec = RFB.execute_cell(backend, "z-ai/glm-5.2", _zb_cell(n=25), n=25,
                           run_id="t", git_commit="d")
    assert rec["diagnostics"]["cost_aborted"] is True
    assert backend.calls == RFB.CostGuardBackend.CHUNK  # no second attempt ran
    assert rec["escalated"] is False and "escalation" not in rec


def test_cost_guard_within_budget_is_inert():
    """Well-behaved cells run all n calls with cost_aborted=False (and backends
    without per-example meta are never aborted — nothing to measure)."""
    backend = _RunawayBackend(ctok_per_call=10)  # 25*10 << 7200
    rec = RFB.execute_cell(backend, "z-ai/glm-5.2", _zb_cell(n=25), n=25,
                           run_id="t", git_commit="d")
    assert backend.calls == 25
    assert rec["diagnostics"]["cost_aborted"] is False
    assert "calls_completed" not in rec["diagnostics"]
    assert all(e["finish"] == "stop" for e in rec["examples"])


def test_dollar_cap_guard_aborts_expensive_cell():
    """The per-cell DOLLAR cap (expensive models only): an opus chain cell whose
    provider ignores the token cap is stopped once its completion-priced spend
    exceeds max($2.50, nominal n*max_new_tokens) — well inside the 3x token
    envelope — while the same runaway on the cheap canary (no dollar cap, token
    envelope not exceeded) runs to completion."""
    model = "anthropic/claude-opus-4.8"
    cell = next(c for c in B.arms_for(model)
                if c["facet"] == "chain_nowrap" and c["length"] == 16)
    cell["n"] = 25
    # dollar cap $10.24 (= 25*16384*$25/M); trip point 409,600 ctok. 40k ctok/call:
    # chunk 1 (8 calls) = 320k ($8.00, under), chunk 2 (16) = 640k ($16 > $10.24)
    # -> abort on DOLLARS with the 3x token envelope (1,228,800 ctok) untouched.
    backend = _RunawayBackend(ctok_per_call=40000)
    rec = RFB.execute_cell(backend, model, cell, n=25, run_id="t", git_commit="d")
    d = rec["diagnostics"]
    assert d["cost_aborted"] is True
    assert d["cost_abort_reason"] == "usd"
    assert backend.calls == 2 * RFB.CostGuardBackend.CHUNK == 16
    assert d["calls_completed"] == 16
    assert rec["usage"]["completion_tokens"] == 40000 * 16 < B.CELL_BUDGET_FACTOR * 25 * 16384
    _validate_c3(rec, cell, model)

    glm_cell = next(c for c in B.arms_for("z-ai/glm-5.2")
                    if c["facet"] == "chain_nowrap" and c["length"] == 16)
    glm_cell["n"] = 25
    backend2 = _RunawayBackend(ctok_per_call=40000)  # 25*40k = 1.0M < 1.2288M budget
    rec2 = RFB.execute_cell(backend2, "z-ai/glm-5.2", glm_cell, n=25,
                            run_id="t", git_commit="d")
    assert rec2["diagnostics"]["cost_aborted"] is False
    assert backend2.calls == 25


def test_breadth_cell_uses_scaled_spec():
    """A zero_budget cell carrying settings['breadth']=B must run
    composite_copy_v2.scaled(k=2*B, recall_pool=B) — the exact items of the pool
    rung, not the canonical pool-16 prompts — and resume under its own key."""
    model = "z-ai/glm-5.2"
    cell = _zb_cell()
    plain_key = RFB.cell_key(model, cell)
    cell["settings"] = {**cell["settings"], "breadth": 8}
    assert RFB.cell_key(model, cell) != plain_key  # each rung is its own cell

    spec = TK.CANONICAL["composite_copy_v2"].scaled(k=16, recall_pool=8)
    gold = {e.prompt: e.answer for e in TK.generate(spec, "test", n=5, length=16)}

    def oracle(prompts, mnt, stop):
        out = []
        for p in prompts:
            base, line = p.rsplit("\n", 1)
            assert line == RFB.CONTRACT_LINE_COMPOSITE
            out.append(f"Answer: {gold[base].rstrip(' .')}")  # KeyError on wrong spec
        return out

    rec = RFB.execute_cell(FunctionBackend(oracle, name="oracle"), model, cell,
                           n=5, run_id="t", git_commit="d")
    assert rec["metrics"]["relaxed"] == 1.0
    assert rec["settings"]["breadth"] == 8  # recorded -> the rung resumes itself
    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "history.jsonl")
        RFB.append_record(history, rec)
        done = RFB.history_keys(history)
        assert RFB.cell_key(model, cell) in done
        assert plain_key not in done  # the canonical cell is NOT satisfied by a rung


def test_k_fixed_chain_cell_uses_fixed_spec():
    """A chain_nowrap cell carrying settings['k_fixed'] must run
    chain_v2.scaled(k=k_fixed) (fixed breadth at any depth) instead of the
    staircase k=2d+1, and resume under its own key."""
    model = "z-ai/glm-5.2"
    cell = next(c for c in B.arms_for(model)
                if c["facet"] == "chain_nowrap" and c["length"] == 16)
    cell["n"] = 5
    stair_key = RFB.cell_key(model, cell)
    cell["settings"] = {**cell["settings"], "k_fixed": 257}
    assert RFB.cell_key(model, cell) != stair_key

    spec = TK.CANONICAL["chain_v2"].scaled(k=257)
    gold = {e.prompt: e.answer for e in TK.generate(spec, "test", n=5, length=16)}

    def oracle(prompts, mnt, stop):
        return [gold[p] for p in prompts]  # KeyError if the staircase (k=33) ran

    rec = RFB.execute_cell(FunctionBackend(oracle, name="oracle"), model, cell,
                           n=5, run_id="t", git_commit="d")
    assert rec["metrics"]["relaxed"] == 1.0
    assert rec["settings"]["k_fixed"] == 257


def test_chain_nowrap_uses_scaled_spec():
    """chain_nowrap depth d must evaluate chain_v2.scaled(k=2*d+1) — the same
    examples the task module generates under the no-wrap protocol."""
    cell = next(c for c in B.arms_for("z-ai/glm-5.2")
                if c["facet"] == "chain_nowrap" and c["length"] == 32)
    cell["n"] = 5
    spec = TK.CANONICAL["chain_v2"].scaled(k=65)
    gold = {e.prompt: e.answer for e in TK.generate(spec, "test", n=5, length=32)}

    def oracle(prompts, mnt, stop):
        return [gold[p] for p in prompts]  # KeyError if the runner built other prompts

    rec = RFB.execute_cell(FunctionBackend(oracle, name="oracle"), "z-ai/glm-5.2",
                           cell, n=5, run_id="t", git_commit="d")
    assert rec["metrics"]["relaxed"] == 1.0


def test_resume_key_includes_n():
    """Regression: a low-n scouting cell (--n-scale) must NOT satisfy resume for the
    full-n cell — cell_key/history_keys include n (contract: effort, leg, rendering,
    n, budget, contract flag all in the resume key)."""
    model = "anthropic/claude-sonnet-5"
    full = _runnable(next(c for c in RFB.build_plan([model], ["zero_budget"], n_scale=1.0)[model]
                          if c["settings"]["leg"] is None and c["length"] == 16))
    scout = _runnable(next(c for c in RFB.build_plan([model], ["zero_budget"], n_scale=0.2)[model]
                           if c["settings"]["leg"] is None and c["length"] == 16))
    zb_n = B.FACETS["zero_budget"]["n"]
    assert full["n"] == zb_n and scout["n"] == max(5, int(zb_n * 0.2))
    assert RFB.cell_key(model, scout) != RFB.cell_key(model, full)

    # A history record written by the scouting run skips the scout cell only.
    backend = FunctionBackend(lambda prompts, mnt, stop: ["x"] * len(prompts), name="fake")
    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "history.jsonl")
        rec = RFB.execute_cell(backend, model, scout, n=scout["n"],
                               run_id="scout", git_commit="deadbeef")
        RFB.append_record(history, rec)
        done = RFB.history_keys(history)
        assert RFB.should_skip(model, scout, done, force=False, canary=False)
        assert not RFB.should_skip(model, full, done, force=False, canary=False)


def test_skip_facets_machinery():
    """skip_facets drops facets structurally in arms_for (grok-build was the
    motivating case; grok-4.5 is the LIVE case since 2026-07-12 — no clean
    off-arm, so every "off"-policy facet is unplanned and its cell plan is
    thinking facets only). The full-roster zero_budget plan covers the other
    11 instant-measured models (all except grok-4.5, muse-spark-1.1, and
    claude-fable-5)."""
    from unittest import mock
    # grok-4.5, muse-spark-1.1, and claude-fable-5 are thinking-only by endpoint
    # design (fable additionally content-filters the composite contract prompts).
    INSTANT_EXCLUDED = {"x-ai/grok-4.5", "muse-spark-1.1", "anthropic/claude-fable-5"}
    for slug in B.MODELS:
        if slug not in INSTANT_EXCLUDED:
            assert not B.MODELS[slug].get("skip_facets")
    for slug in INSTANT_EXCLUDED:
        cells = B.arms_for(slug)
        assert {c["facet"] for c in cells} == {"s5_concrete", "chain_nowrap",
                                                  "commutative", "s5_chain"}
        # every planned cell is a reasoning-ON arm (there is no off arm at all)
        assert {c["settings"]["effort"] for c in cells} == {"high", "xhigh"}
    # simulate a partial-skip model without mutating the real registry
    fake = {**B.MODELS["z-ai/glm-5.2"], "skip_facets": ("zero_budget",)}
    with mock.patch.dict(B.MODELS, {"z-ai/glm-5.2": fake}):
        cells = B.arms_for("z-ai/glm-5.2")
        assert not any(c["facet"] == "zero_budget" for c in cells)
        assert {c["facet"] for c in cells} == {"recall_load", "s5_concrete",
                                               "chain_nowrap", "chain_instant",
                                               "sanity", "commutative",
                                               "gap_stability", "s5_chain"}
    plan = RFB.build_plan(list(B.MODELS), ["zero_budget"], n_scale=1.0)
    assert sum(len(cells) for cells in plan.values()) == 60  # 12 models x 5 zero_budget cells
    assert sum(1 for cells in plan.values() if cells) == 12  # grok-4.5, muse-spark-1.1, fable-5 plan none


def test_v2_task_cells_get_fresh_resume_keys():
    """The zero_budget facet switched task to composite_copy_v2: the task is part
    of cell_key, so v1-task history records (same model/facet/length/n/settings)
    must NOT satisfy resume for the v2-task cells."""
    model = "z-ai/glm-5.2"
    v2_cells = [c for c in B.arms_for(model) if c["facet"] == "zero_budget"]
    assert v2_cells and all(c["task"] == "composite_copy_v2" for c in v2_cells)
    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "history.jsonl")
        with open(history, "w", encoding="utf-8") as fh:
            for c in v2_cells:  # a v1-task twin of every planned v2 cell
                fh.write(json.dumps({
                    "model": model, "facet": c["facet"],
                    "task": "composite_copy_v1", "length": c["length"],
                    "n": c["n"], "settings": c["settings"],
                    # what real v1 records carry (pre-1.1 history has no
                    # stream_version field; suite_version was "1.0")
                    "suite_version": "1.0",
                }) + "\n")
        done = RFB.history_keys(history)
        for c in v2_cells:
            twin = {**c, "task": "composite_copy_v1"}
            assert RFB.cell_key(model, twin) in done       # the twin IS in history
            assert RFB.cell_key(model, c) not in done      # ...but never the v2 cell
            assert not RFB.should_skip(model, c, done, force=False, canary=False)


def test_resume_key_uses_spec_stream_version_not_suite_version():
    """Regression (issue #11 landing): cell_key's version component is the CELL's
    spec stream version, NOT the global TK.SUITE_VERSION. The 1.0 -> 1.1 suite bump
    (which only ADDED the v2 specs) must not invalidate resume for cells whose
    example streams are pinned and unchanged (chain_nowrap / s5_concrete / sanity /
    any v1-stream cell): their keys must still match pre-bump history records that
    recorded suite_version "1.0"."""
    assert TK.SUITE_VERSION == "1.1"
    # per-task stream versions: pinned v1 streams stay "1.0"; chain_v2 is a new 1.1 stream
    assert RFB.stream_version("chain_v2") == "1.1"
    assert RFB.stream_version("chain_v1") == "1.0"  # retired, frozen
    assert RFB.stream_version("recall_copy_v1") == "1.0"
    assert RFB.stream_version("conflict_v1") == "1.0"
    assert RFB.stream_version("s5") == "1.0"              # s5_concrete facet task (non-TaskSpec)
    assert RFB.stream_version("composite_copy_v1") == "1.0"  # retired specs resolve too
    assert RFB.stream_version("composite_copy_v2") == "1.1"
    assert RFB.stream_version("binding_v2") == "1.1"

    model = "z-ai/glm-5.2"
    cells = B.arms_for(model)
    pinned = [c for c in cells if c["facet"] in ("s5_concrete", "sanity")]
    v2 = [c for c in cells if c["facet"] == "zero_budget"]
    assert pinned and v2
    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "history.jsonl")
        with open(history, "w", encoding="utf-8") as fh:
            for c in pinned + v2:  # records exactly as a pre-bump run wrote them
                fh.write(json.dumps({
                    "model": model, "facet": c["facet"], "task": c["task"],
                    "length": c["length"], "n": c["n"], "settings": c["settings"],
                    "suite_version": "1.0",   # the global at the time of the run
                }) + "\n")
        done = RFB.history_keys(history)
        # pinned-stream cells resume-hit across the suite bump...
        for c in pinned:
            assert RFB.cell_key(model, c) in done, (c["facet"], c["task"], c["length"])
            assert RFB.should_skip(model, c, done, force=False, canary=False)
        # ...but the genuinely-new v2-stream cells do not (stream "1.1" != "1.0")
        for c in v2:
            assert RFB.cell_key(model, c) not in done
        # new records carry stream_version explicitly and resume against themselves
        rec = RFB.execute_cell(FunctionBackend(lambda ps, m, st: ["x"] * len(ps), name="f"),
                               model, {**v2[0], "n": 5}, n=5, run_id="t", git_commit="d")
        assert rec["stream_version"] == "1.1" and rec["suite_version"] == TK.SUITE_VERSION
        RFB.append_record(history, rec)
        assert RFB.cell_key(model, {**v2[0], "n": 5}) in RFB.history_keys(history)


def test_build_plan_n_scale():
    plan = RFB.build_plan(["anthropic/claude-sonnet-5"], ["sanity"], n_scale=0.1)
    cells = plan["anthropic/claude-sonnet-5"]
    assert len(cells) == 2  # recall_copy_v1 + conflict_v1
    assert all(c["n"] == 5 for c in cells)  # 30 * 0.1 = 3 -> floor of 5
    plan2 = RFB.build_plan(["anthropic/claude-sonnet-5"], ["zero_budget"], n_scale=0.5)
    half = max(5, int(B.FACETS["zero_budget"]["n"] * 0.5))
    assert all(c["n"] == half for c in plan2["anthropic/claude-sonnet-5"])


def test_budget_override_and_lengths_filter():
    """--budget-override rewrites max_new_tokens for matching (facet, length)
    cells only, which changes the settings hash (fresh resume key); --lengths
    keeps only matching cells. Malformed/unknown specs fail loudly."""
    overrides = RFB.parse_budget_overrides(
        ["s5_concrete:256:49152", "chain_nowrap:128:32768"])
    assert overrides == {("s5_concrete", 256): 49152, ("chain_nowrap", 128): 32768}

    model = "anthropic/claude-opus-4.8"
    plan = RFB.build_plan([model], ["s5_concrete"], 1.0, budget_overrides=overrides)
    cells = plan[model]
    by_len = {c["length"]: c for c in cells}
    assert by_len[256]["settings"]["max_new_tokens"] == 49152
    assert by_len[128]["settings"]["max_new_tokens"] == 16384  # untouched
    # the override is part of the settings hash -> fresh resume key
    planned = next(c for c in B.arms_for(model)
                   if c["facet"] == "s5_concrete" and c["length"] == 256)
    assert B.settings_hash(by_len[256]) != B.settings_hash(planned)
    assert RFB.cell_key(model, by_len[256]) != RFB.cell_key(model, planned)

    # --lengths filter keeps only the requested lengths
    plan = RFB.build_plan([model], ["s5_concrete"], 1.0, lengths=[128])
    assert [c["length"] for c in plan[model]] == [128]

    # parse failures are loud (a typo must not silently no-op a paid run)
    for bad in ("s5_concrete:256", "nope:256:32768", "s5_concrete:x:32768",
                "s5_concrete:256:-1"):
        try:
            RFB.parse_budget_overrides([bad])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"expected SystemExit for {bad!r}")
    assert RFB.parse_budget_overrides(None) == {}


def test_experimental_facets_shape():
    """commutative (#18) and gap_stability (#16a): protocol knobs match the
    approved plans (calibration-matched n=25 thinking @L64; zero_budget-identical
    L32 contract cells at n=50)."""
    for slug in ("anthropic/claude-opus-4.8", "z-ai/glm-5.2"):
        cells = B.arms_for(slug)
        comm = [c for c in cells if c["facet"] == "commutative"]
        assert [(c["task"], c["length"], c["n"]) for c in comm] == \
               [("commutative_v1", 64, 25)]
        assert comm[0]["settings"]["effort"] == "high"
        assert comm[0]["settings"]["max_new_tokens"] == B.REASONING_MAX_NEW_TOKENS
        assert comm[0]["settings"]["contract"] is False
        gap = [c for c in cells if c["facet"] == "gap_stability"]
        assert [(c["task"], c["length"], c["settings"]["leg"], c["n"]) for c in gap] == \
               [("composite_copy_v2", 32, None, 50),
                ("composite_copy_v2", 32, "binding_only", 50)]
        for c in gap:
            assert c["settings"]["effort"] == "none"
            assert c["settings"]["contract"] is True
            assert c["settings"]["max_new_tokens"] == B.ZERO_BUDGET_MAX_NEW_TOKENS


if __name__ == "__main__":
    for fn in [test_registry_shape, test_arms_for_facets, test_arm_settings_protocol,
               test_gemini_off_arm_is_minimal, test_settings_hash_stable,
               test_settings_hash_contract_flag_compat,
               test_settings_hash_breadth_and_k_fixed_sentinels,
               test_canonical_system_prompt_fingerprints_pin_the_live_prompts,
               test_settings_hash_system_prompt_sentinel,
               test_build_plan_stamps_the_resolved_system_prompt,
               test_plan_has_no_rung_keys_and_facet_breadths_expand,
               test_prompt_tokens_est_breadth_and_k_fixed,
               test_local_arm_is_a_roster_entry_not_a_side_path,
               test_context_guard_refuses_a_budget_the_server_cannot_hold,
               test_steed_arm_is_a_roster_entry_not_a_side_path,
               test_steed_arm_needs_no_key_clamps_workers_and_sizes_its_own_timeout,
               test_declared_context_is_an_equality_locally_and_a_floor_on_steed,
               test_prompt_tokens_est_prices_the_s5_bind_k_rung_that_runs,
               test_cell_dollar_cap, test_endpoint_for_defaults_and_direct_entry,
               test_build_backend_direct_endpoint,
               test_resolved_request_params_records_what_was_sent,
               test_build_backend_matches_resolved_request_params,
               test_execute_cell_records_request_diagnostics,
               test_cost_estimate_sane,
               test_extract_contract_answer, test_execute_cell_end_to_end,
               test_zero_budget_contract_scoring, test_binding_leg_rejects_holder_dump,
               test_escalation_on_length_cutoff, test_escalation_iterates_to_2048,
               test_no_escalation_below_threshold_or_without_contract,
               test_finish_errors_counted, test_cost_guard_aborts_runaway_cell,
               test_cost_guard_blocks_escalation, test_cost_guard_within_budget_is_inert,
               test_dollar_cap_guard_aborts_expensive_cell,
               test_breadth_cell_uses_scaled_spec, test_k_fixed_chain_cell_uses_fixed_spec,
               test_chain_nowrap_uses_scaled_spec, test_resume_key_includes_n,
               test_skip_facets_machinery,
               test_v2_task_cells_get_fresh_resume_keys,
               test_resume_key_uses_spec_stream_version_not_suite_version,
               test_build_plan_n_scale, test_budget_override_and_lengths_filter,
               test_experimental_facets_shape]:
        fn()
        print(f"{fn.__name__}: ok")
