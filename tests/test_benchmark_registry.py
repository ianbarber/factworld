"""Tests for factworld/benchmark.py (contract C4) and the runner's cell execution.

Registry: tier -> arm expansion (effort sweeps, reasoning-on budgets), stable
settings_hash, sane cost estimates. Runner: a FunctionBackend-driven mini-run of
``execute_cell`` writing a C3-conformant record to a tmp history file (no API),
plus the resume-key round trip.

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
from factworld.backends import FunctionBackend

import run_frontier_benchmark as RFB

SETTINGS_KEYS = {"effort", "max_new_tokens", "stop_at", "rendering",
                 "format_prompt", "n_shot", "leg"}
C3_TOP_KEYS = {"run_id", "ts", "git_commit", "suite_version", "model", "served_models",
               "providers", "facet", "task", "length", "n", "settings", "metrics",
               "diagnostics", "usage", "elapsed_s", "examples"}


def _efforts(cells, facet):
    return [c["settings"]["effort"] for c in cells if c["facet"] == facet]


# --- registry -------------------------------------------------------------------

def test_registry_shape():
    assert len(B.MODELS) == 13
    for slug, reg in B.MODELS.items():
        assert reg["tier"] in B.TIERS, slug
        assert reg["prompt_price_per_M"] > 0 and reg["completion_price_per_M"] > 0
        assert isinstance(reg["open_weights"], bool)
    assert set(B.FACETS) == {"dose_response", "composite_length", "s5_concrete",
                             "chain_depth", "decomposition", "sanity", "floor"}


def test_arms_for_tiers():
    # non_reasoning: no reasoning arms at all — every effort is None (param omitted).
    llama = B.arms_for("meta-llama/llama-4-maverick")
    assert all(c["settings"]["effort"] is None for c in llama)
    assert {c["facet"] for c in llama} == set(B.FACETS)  # still covers every facet
    # one default arm per (facet, task, length), not one per (length, effort)
    n_lengths = len(B.FACETS["composite_length"]["lengths"])
    assert len(_efforts(llama, "composite_length")) == n_lengths

    # frontier_pair: dose_response gets none+high ONLY.
    opus = B.arms_for("anthropic/claude-opus-4.8")
    assert _efforts(opus, "dose_response") == ["none", "high"]
    assert _efforts(opus, "composite_length").count("high") == n_lengths
    assert set(_efforts(opus, "s5_concrete")) == {"high"}
    assert set(_efforts(opus, "chain_depth")) == {"high"}
    assert set(_efforts(opus, "decomposition")) == {"none"}
    assert set(_efforts(opus, "floor")) == {"none"}

    # cheap_reasoner: the full 4-effort sweep in dose_response.
    glm = B.arms_for("z-ai/glm-5.2")
    assert _efforts(glm, "dose_response") == ["none", "low", "medium", "high"]

    # decomposition expands to the three legs.
    legs = [c["settings"]["leg"] for c in glm if c["facet"] == "decomposition"]
    assert legs == ["binding_only", "end_to_end", "scaffolded"]


def test_arm_settings_protocol():
    """Reasoning-on cells: >=8192 tokens (per-length facet budgets may raise the
    floor, never lower it), no stop; every cell has all C3 settings keys."""
    for slug in ("z-ai/glm-5.2", "anthropic/claude-opus-4.8", "meta-llama/llama-4-maverick"):
        for cell in B.arms_for(slug):
            s = cell["settings"]
            assert set(s) == SETTINGS_KEYS
            assert s["stop_at"] is None
            assert s["n_shot"] == 0
            if s["effort"] in B.REASONING_EFFORTS:
                assert s["max_new_tokens"] >= B.REASONING_MAX_NEW_TOKENS
                expected = B.FACETS[cell["facet"]].get("budgets", {}).get(
                    cell["length"], B.REASONING_MAX_NEW_TOKENS)
                assert s["max_new_tokens"] == expected
            else:
                assert s["max_new_tokens"] == B.DEFAULT_MAX_NEW_TOKENS
            assert cell["n"] == B.FACETS[cell["facet"]]["n"]


def test_gemini_off_arm_is_minimal():
    """Gemini 3 cannot disable reasoning: its "none" arms substitute "minimal"."""
    for slug in ("google/gemini-3.5-flash", "google/gemini-3.1-pro-preview"):
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
    long = B._prompt_tokens_est("composite_copy_v1", 512, None)
    assert long > short * 5


# --- runner cell execution (no API) ------------------------------------------------

def _validate_c3(rec, cell, model):
    assert set(rec) == C3_TOP_KEYS
    assert rec["model"] == model
    assert rec["facet"] == cell["facet"] and rec["task"] == cell["task"]
    assert rec["length"] == cell["length"] and rec["n"] == cell["n"]
    assert set(rec["settings"]) == SETTINGS_KEYS
    assert set(rec["metrics"]) == {"relaxed", "exact", "contains", "last_n"}
    assert rec["metrics"]["relaxed"] is not None  # relaxed is ALWAYS present
    assert set(rec["diagnostics"]) == {"empty_rate", "api_errors", "finish_reasons"}
    assert set(rec["usage"]) == {"prompt_tokens", "completion_tokens",
                                 "reasoning_tokens", "cost_usd_est"}
    assert rec["examples"] and all(set(e) == {"gold", "pred", "relaxed"}
                                   for e in rec["examples"])  # no prompt text


def test_execute_cell_end_to_end():
    """FunctionBackend mini-run: s5_concrete + sanity + decomposition cells -> valid
    C3 records in a tmp history file, and the resume key round-trips."""
    model = "z-ai/glm-5.2"
    cells = B.arms_for(model)
    s5_cell = next(c for c in cells if c["facet"] == "s5_concrete" and c["length"] == 16)
    sanity_cell = next(c for c in cells if c["task"] == "recall_copy_v1")
    decomp_cell = next(c for c in cells if c["settings"]["leg"] == "binding_only")
    for c in (s5_cell, sanity_cell, decomp_cell):
        c["n"] = 5

    backend = FunctionBackend(lambda prompts, mnt, stop: ["Driver ."] * len(prompts),
                              name="fake")
    with tempfile.TemporaryDirectory() as tmp:
        history = os.path.join(tmp, "sub", "history.jsonl")  # exercises mkdir -p
        for cell in (s5_cell, sanity_cell, decomp_cell):
            rec = RFB.execute_cell(backend, model, cell, n=cell["n"],
                                   run_id="bench_test", git_commit="deadbeef")
            _validate_c3(rec, cell, model)
            RFB.append_record(history, rec)

        with open(history) as fh:
            lines = [json.loads(line) for line in fh]
        assert len(lines) == 3

        # s5 cell: "Driver ." matches some golds; relaxed in [0,1]; s5 has no exact/last_n.
        s5_rec = lines[0]
        assert s5_rec["metrics"]["exact"] is None and s5_rec["metrics"]["last_n"] is None
        assert 0.0 <= s5_rec["metrics"]["relaxed"] <= 1.0
        assert s5_rec["diagnostics"]["empty_rate"] == 0.0
        # sanity cell went through evaluate_task: all four metrics populated.
        assert all(v is not None for v in lines[1]["metrics"].values())
        # decomposition leg: relaxed present, per-leg record carries settings.leg.
        assert lines[2]["settings"]["leg"] == "binding_only"

        # resume: every written cell's key is now recognized as done.
        done = RFB.history_keys(history)
        for cell in (s5_cell, sanity_cell, decomp_cell):
            assert RFB.cell_key(model, cell) in done
            assert RFB.should_skip(model, cell, done, force=False, canary=False)
        # --force and --canary (for the glm canary model) bypass the skip.
        assert not RFB.should_skip(model, s5_cell, done, force=True, canary=False)
        assert not RFB.should_skip(model, s5_cell, done, force=False, canary=True)
        # a cell that never ran is not skipped
        other = next(c for c in cells if c["facet"] == "chain_depth")
        assert not RFB.should_skip(model, other, done, force=False, canary=False)


def test_resume_key_includes_n():
    """Regression: a low-n scouting cell (--n-scale) must NOT satisfy resume for the
    full-n cell — cell_key/history_keys include n (contract: effort, leg, rendering,
    n, budget all in the resume key)."""
    model = "anthropic/claude-sonnet-5"
    full = next(c for c in RFB.build_plan([model], ["dose_response"], n_scale=1.0)[model]
                if c["settings"]["effort"] == "high")
    scout = next(c for c in RFB.build_plan([model], ["dose_response"], n_scale=0.2)[model]
                 if c["settings"]["effort"] == "high")
    dose_n = B.FACETS["dose_response"]["n"]
    assert full["n"] == dose_n and scout["n"] == max(5, int(dose_n * 0.2))
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
    plan2 = RFB.build_plan(["anthropic/claude-sonnet-5"], ["dose_response"], n_scale=0.5)
    half = max(5, int(B.FACETS["dose_response"]["n"] * 0.5))
    assert all(c["n"] == half for c in plan2["anthropic/claude-sonnet-5"])


if __name__ == "__main__":
    for fn in [test_registry_shape, test_arms_for_tiers, test_arm_settings_protocol,
               test_settings_hash_stable, test_cost_estimate_sane,
               test_execute_cell_end_to_end, test_resume_key_includes_n,
               test_build_plan_n_scale]:
        fn()
        print(f"{fn.__name__}: ok")
