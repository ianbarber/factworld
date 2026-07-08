"""Tests for factworld/benchmark.py (contract C4) and the runner's cell execution.

Registry: tier -> arm expansion (v2 facets: zero_budget contract battery,
s5_concrete mid-band, chain_nowrap staircase, sanity), stable settings_hash
(including pre-contract-flag history compatibility), sane cost estimates.
Runner: FunctionBackend-driven mini-runs of ``execute_cell`` writing C3-conformant
records (per-example ctok/rtok/finish, contract diagnostics, finish=length
escalation) to a tmp history file (no API), plus the resume-key round trip.

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
C3_TOP_KEYS = {"run_id", "ts", "git_commit", "suite_version", "model", "served_models",
               "providers", "facet", "task", "length", "n", "settings", "metrics",
               "diagnostics", "usage", "elapsed_s", "escalated", "examples"}
EXAMPLE_KEYS = {"gold", "pred", "relaxed", "ctok", "rtok", "finish"}


def _efforts(cells, facet):
    return [c["settings"]["effort"] for c in cells if c["facet"] == facet]


# --- registry -------------------------------------------------------------------

def test_registry_shape():
    assert len(B.MODELS) == 10
    # roster decisions 2026-07-07/08: maverick, gpt-5.4, gemini-3.1-pro dropped;
    # x-ai entry is grok-build-0.1 (mainline grok bio-filtered on composite cells)
    for dropped in ("meta-llama/llama-4-maverick", "openai/gpt-5.4",
                    "google/gemini-3.1-pro-preview", "x-ai/grok-4.3",
                    "x-ai/grok-4.20"):
        assert dropped not in B.MODELS
    grok = B.MODELS["x-ai/grok-build-0.1"]
    # pricing verified against openrouter.ai/api/v1/models 2026-07-08
    assert grok["prompt_price_per_M"] == 1.0
    assert grok["completion_price_per_M"] == 2.0
    for slug, reg in B.MODELS.items():
        assert reg["tier"] in B.TIERS, slug
        assert reg["prompt_price_per_M"] > 0 and reg["completion_price_per_M"] > 0
        assert isinstance(reg["open_weights"], bool)
    assert set(B.FACETS) == {"zero_budget", "s5_concrete", "chain_nowrap", "sanity"}


def test_arms_for_facets():
    opus = B.arms_for("anthropic/claude-opus-4.8")

    # zero_budget: explicit (length, leg) cells, effort none, tight budget, contract on.
    zb = [c for c in opus if c["facet"] == "zero_budget"]
    assert [(c["length"], c["settings"]["leg"]) for c in zb] == [
        (16, None), (64, None), (16, "binding_only"), (16, "end_to_end")]
    for c in zb:
        assert c["task"] == "composite_copy_v1" and c["n"] == 100
        assert c["settings"]["effort"] == "none"
        assert c["settings"]["contract"] is True
        assert c["settings"]["max_new_tokens"] == B.ZERO_BUDGET_MAX_NEW_TOKENS == 96

    # s5_concrete: mid-band only (ceiling lengths dropped), reasoning on, 16384 budget.
    s5 = [c for c in opus if c["facet"] == "s5_concrete"]
    assert [c["length"] for c in s5] == [128, 256]
    for c in s5:
        assert c["n"] == 25 and c["settings"]["effort"] == "high"
        assert c["settings"]["max_new_tokens"] == 16384
        assert c["settings"]["rendering"] == "concrete"
        assert c["settings"]["contract"] is False

    # chain_nowrap: the four staircase depths, reasoning on, 16384 budget.
    chain = [c for c in opus if c["facet"] == "chain_nowrap"]
    assert [c["length"] for c in chain] == [16, 32, 64, 128]
    for c in chain:
        assert c["task"] == "chain_v1" and c["n"] == 25
        assert c["settings"]["effort"] == "high"
        assert c["settings"]["max_new_tokens"] == 16384

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
            elif cell["facet"] == "zero_budget":
                assert s["max_new_tokens"] == B.ZERO_BUDGET_MAX_NEW_TOKENS
            else:
                assert s["max_new_tokens"] == B.DEFAULT_MAX_NEW_TOKENS
            assert cell["n"] == B.FACETS[cell["facet"]]["n"]


def test_gemini_off_arm_is_minimal():
    """Gemini 3 cannot disable reasoning: its "none" arms substitute "minimal"."""
    for slug in ("google/gemini-3.5-flash",):
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


def test_cost_estimate_sane():
    glm_cells = B.arms_for("z-ai/glm-5.2")
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
    short = B._prompt_tokens_est("composite_copy_v1", 16, None)
    long = B._prompt_tokens_est("composite_copy_v1", 64, None)
    assert long > short * 2
    # chain estimates use the no-wrap staircase (k=2*depth+1), so depth 128 must
    # not trip chain_v1's wrap validity gate and deeper chains cost more.
    d16 = B._prompt_tokens_est("chain_v1", 16, None)
    d128 = B._prompt_tokens_est("chain_v1", 128, None)
    assert d128 > d16 * 3


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
    diag_keys = {"empty_rate", "api_errors", "finish_reasons"}
    if rec["settings"]["contract"]:
        diag_keys |= {"contract_rate", "covert_cot_rate", "rtok_leak_rate"}
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
    sanity_cell = next(c for c in cells if c["task"] == "recall_copy_v1")
    zb_cell = next(c for c in cells if c["settings"]["leg"] == "binding_only")
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
    plain = next(c for c in cells if c["facet"] == "zero_budget"
                 and c["settings"]["leg"] is None and c["length"] == 16)
    plain["n"] = 5

    spec = TK.CANONICAL["composite_copy_v1"]
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
    bind = next(c for c in cells if c["settings"]["leg"] == "binding_only")
    bind["n"] = 5
    lines = []
    rec2 = RFB.execute_cell(
        FunctionBackend(lambda ps, m, s: [lines.append(p.rsplit("\n", 1)[1]) or "x"
                                          for p in ps], name="probe"),
        model, bind, n=5, run_id="t", git_commit="d")
    assert set(lines) == {RFB.CONTRACT_LINE_BINDING}
    assert rec2["diagnostics"]["contract_rate"] == 0.0


def test_binding_leg_rejects_holder_dump():
    """A holder-dump reply ("Answer: zz g0 g1 g2 ...") must score 0 on the binding
    leg: the span's FIRST content token must be the holder (membership scoring had
    a 100% false-positive rate against this — kimi exploited it live)."""
    model = "z-ai/glm-5.2"
    bind = next(c for c in B.arms_for(model) if c["settings"]["leg"] == "binding_only")
    bind["n"] = 5

    dump = "Answer: zz " + " ".join(f"g{i}" for i in range(64))  # every candidate holder
    rec = RFB.execute_cell(
        FunctionBackend(lambda ps, m, s: [dump] * len(ps), name="dump"),
        model, bind, n=5, run_id="t", git_commit="d")
    assert rec["diagnostics"]["contract_rate"] == 1.0  # well-formed contract line...
    assert rec["metrics"]["relaxed"] == 0.0            # ...but no committed holder

    # positive control: committing to the correct holder (with trailing period,
    # relaxed-style) scores 1.0.
    spec = TK.CANONICAL["composite_copy_v1"]
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
    cell = next(c for c in B.arms_for("z-ai/glm-5.2")
                if c["facet"] == "zero_budget" and c["settings"]["leg"] is None
                and c["length"] == 16)
    cell["n"] = n
    return cell


def test_escalation_on_length_cutoff():
    """A contract cell with finish=length > 10% reruns ONCE at 512 and publishes
    the rerun (escalated=true, first attempt's diagnostics kept, usage summed,
    planned settings preserved for resume)."""
    backend = _MetaBackend([
        {"finish": "length", "ctok": 96, "rtok": 0, "text": "truncated working with no answer line"},
        {"finish": "stop", "ctok": 400, "rtok": 2, "text": "long working...\nAnswer: g3 v9"},
    ])
    cell = _zb_cell()
    rec = RFB.execute_cell(backend, "z-ai/glm-5.2", cell, n=5,
                           run_id="t", git_commit="d")
    assert backend.budgets == [96, RFB.ESCALATED_MAX_NEW_TOKENS] == [96, 512]
    assert rec["escalated"] is True
    # published diagnostics/examples come from the rerun
    assert rec["diagnostics"]["finish_reasons"] == {"stop": 5}
    assert rec["diagnostics"]["contract_rate"] == 1.0
    assert all(e["finish"] == "stop" and e["ctok"] == 400 for e in rec["examples"])
    # rerun cleanliness diagnostics: 400 ctok > threshold, rtok leak present
    assert rec["diagnostics"]["covert_cot_rate"] == 1.0
    assert rec["diagnostics"]["rtok_leak_rate"] == 1.0
    # first attempt is preserved with its own diagnostics
    esc = rec["escalation"]
    assert esc["max_new_tokens"] == 512
    assert esc["first_attempt"]["max_new_tokens"] == 96
    assert esc["first_attempt"]["diagnostics"]["finish_reasons"] == {"length": 5}
    assert esc["first_attempt"]["diagnostics"]["contract_rate"] == 0.0
    assert esc["first_attempt"]["relaxed"] == 0.0
    # spend honesty: usage covers BOTH attempts
    assert rec["usage"]["prompt_tokens"] == 100  # 2 attempts x 5 calls x 10
    assert rec["usage"]["completion_tokens"] == 5 * 96 + 5 * 400
    # resume key unchanged: the record's settings are the PLANNED ones, so the
    # written record satisfies the original cell's resume key.
    assert rec["settings"]["max_new_tokens"] == 96
    assert B.settings_hash(rec) == B.settings_hash(cell)
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
    assert rec["diagnostics"]["rtok_leak_rate"] == 0.0

    # non-contract cells never escalate even at 100% finish=length.
    sanity = next(c for c in B.arms_for("z-ai/glm-5.2") if c["task"] == "recall_copy_v1")
    sanity["n"] = 5
    backend2 = _MetaBackend([{"finish": "length", "ctok": 96, "rtok": 0, "text": "v0"}])
    rec2 = RFB.execute_cell(backend2, "z-ai/glm-5.2", sanity, n=5,
                            run_id="t", git_commit="d")
    assert backend2.budgets == [2048]
    assert rec2["escalated"] is False
    assert all(e["finish"] == "length" for e in rec2["examples"])


def test_chain_nowrap_uses_scaled_spec():
    """chain_nowrap depth d must evaluate chain_v1.scaled(k=2*d+1) — the same
    examples the task module generates under the no-wrap protocol."""
    cell = next(c for c in B.arms_for("z-ai/glm-5.2")
                if c["facet"] == "chain_nowrap" and c["length"] == 32)
    cell["n"] = 5
    spec = TK.CANONICAL["chain_v1"].scaled(k=65)
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
    full = next(c for c in RFB.build_plan([model], ["zero_budget"], n_scale=1.0)[model]
                if c["settings"]["leg"] is None and c["length"] == 16)
    scout = next(c for c in RFB.build_plan([model], ["zero_budget"], n_scale=0.2)[model]
                 if c["settings"]["leg"] is None and c["length"] == 16)
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


def test_build_plan_n_scale():
    plan = RFB.build_plan(["anthropic/claude-sonnet-5"], ["sanity"], n_scale=0.1)
    cells = plan["anthropic/claude-sonnet-5"]
    assert len(cells) == 2  # recall_copy_v1 + conflict_v1
    assert all(c["n"] == 5 for c in cells)  # 30 * 0.1 = 3 -> floor of 5
    plan2 = RFB.build_plan(["anthropic/claude-sonnet-5"], ["zero_budget"], n_scale=0.5)
    half = max(5, int(B.FACETS["zero_budget"]["n"] * 0.5))
    assert all(c["n"] == half for c in plan2["anthropic/claude-sonnet-5"])


if __name__ == "__main__":
    for fn in [test_registry_shape, test_arms_for_facets, test_arm_settings_protocol,
               test_gemini_off_arm_is_minimal, test_settings_hash_stable,
               test_settings_hash_contract_flag_compat, test_cost_estimate_sane,
               test_extract_contract_answer, test_execute_cell_end_to_end,
               test_zero_budget_contract_scoring, test_binding_leg_rejects_holder_dump,
               test_escalation_on_length_cutoff,
               test_no_escalation_below_threshold_or_without_contract,
               test_chain_nowrap_uses_scaled_spec, test_resume_key_includes_n,
               test_build_plan_n_scale]:
        fn()
        print(f"{fn.__name__}: ok")
