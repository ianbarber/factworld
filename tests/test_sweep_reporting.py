"""The local sweep's reporting and eval layer: floors, per-seed rendering, generation
budget, interleaved-slot alignment, and the guided free-run protocol.

These are the parts of scripts/sweep.py that decide what a local cell MEANS, so they are
tested without a GPU: sweep.py defers its torch import into ``run_one``, and the guided
free-run loop is exercised against a stub model that emits the gold continuation (an oracle
backend must score exactly 1.000 — the check that separates an eval-path defect from a real
null).

Runs with zero dependencies:  python3 tests/test_sweep_reporting.py
"""
from __future__ import annotations

import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import sweep  # noqa: E402
from factworld.tasks import CANONICAL, generate  # noqa: E402
from factworld.tokenizer import Tokenizer  # noqa: E402
from factworld.tasks import build_world  # noqa: E402
from factworld.validity import (  # noqa: E402
    operative_floor,
    s5_chain_floors,
    s5_chain_shallow_preds,
)

_LOCAL = CANONICAL["s5_chain_local_v2"]


# --------------------------------------------------------------------------- floors
_GRID = [(k, d) for k in (4, 5, 6, 8) for d in (1, 2)]


def test_registered_adversaries_are_well_formed_on_every_local_cell():
    """echo is zeroed by the distinct_path gate, uniform is 1/k, and non-start is 1/(k-1)."""
    for k, depth in _GRID:
        spec = _LOCAL.scaled(k=k, chain_depth=depth)
        for L in spec.eval_lengths:
            f = s5_chain_floors(generate(spec, "test", n=200, length=L), k)
            assert f["echo"] == 0.0, (k, depth, L)          # distinct_path gate
            assert f["uniform"] == 1.0 / k
            assert abs(f["uniform_non_start"] - 1.0 / (k - 1)) < 1e-9
            assert 0.0 <= f["initial_map_chase"] <= 1.0


def test_operative_floor_is_the_max_over_the_registered_adversaries():
    """The floor of a cell is that max, and NO single adversary supplies it everywhere.

    The initial-map chase is the biggest one on some cells and not on others: over the
    registered local grid, uniform-over-non-start wins 7 of the 16 cells, and on two of them
    the chase falls below even plain 1/k. Reading a score against one named row understates
    the floor wherever that row is not the largest.
    """
    chase_wins = other_wins = 0
    below_uniform = []
    for k, depth in _GRID:
        spec = _LOCAL.scaled(k=k, chain_depth=depth)
        for L in spec.eval_lengths:
            f = s5_chain_floors(generate(spec, "test", n=200, length=L), k)
            assert operative_floor(f) == max(f.values())
            assert operative_floor(f) >= f["initial_map_chase"]
            if f["initial_map_chase"] == max(f.values()):
                chase_wins += 1
            else:
                other_wins += 1
            if f["initial_map_chase"] < f["uniform"]:
                below_uniform.append((k, depth, L))
    assert chase_wins == 9 and other_wins == 7
    assert below_uniform == [(5, 2, 4), (6, 2, 8)]


def test_k6_depth2_floor_is_uniform_non_start_at_both_lengths():
    """The cell that already produced a formed seed. Its floor is 0.200 at L4 AND L8 — the
    non-start guesser — while the initial-map chase is 0.195 and 0.160; two of its three
    published seeds sat below it, which the run's own table did not show."""
    spec = _LOCAL.scaled(k=6, chain_depth=2)
    f4 = s5_chain_floors(generate(spec, "test", n=200, length=4), 6)
    f8 = s5_chain_floors(generate(spec, "test", n=200, length=8), 6)
    assert abs(f4["initial_map_chase"] - 0.195) < 1e-9
    assert abs(f8["initial_map_chase"] - 0.160) < 1e-9
    assert abs(operative_floor(f4) - 0.200) < 1e-9
    assert abs(operative_floor(f8) - 0.200) < 1e-9
    published = [0.155, 0.815, 0.170]                       # gdp_hybrid seeds 0/1/2 @L4
    assert sum(1 for x in published if x < operative_floor(f4)) == 2
    assert max(published) > 4 * operative_floor(f4)


def test_no_chase_row_where_there_is_no_event_stream():
    """In the `chain` family nothing moves the stated map, so the chase IS the oracle: it
    scores 1.000 and would render as a floor no arm can ever clear."""
    chain = CANONICAL["chain_v2"]
    ex = generate(chain, "test", n=50, length=4)
    assert s5_chain_floors(ex, chain.k)["initial_map_chase"] == 1.0
    floors = sweep.cell_floors(chain, 4, 50)
    assert "initial_map_chase" not in floors
    assert abs(operative_floor(floors) - 1.0 / (chain.k - 1)) < 1e-9


def test_shallow_preds_parse_both_event_grammars():
    """The adversary reads the FACTS, so the compact event grammar must not change it."""
    spec = _LOCAL.scaled(k=6, chain_depth=2)
    plain = generate(spec, "test", n=20, length=4)
    compact = generate(spec.scaled(compact_events=True), "test", n=20, length=4)
    for exs in (plain, compact):
        for ex in exs:
            preds = s5_chain_shallow_preds(ex.prompt)
            assert preds["echo"] == f"{ex.meta['start']}."
            assert preds["initial_map_chase"] is not None


def test_cell_floors_uses_the_scored_items():
    spec = _LOCAL.scaled(k=6, chain_depth=2)
    assert sweep.cell_floors(spec, 4, 200) == s5_chain_floors(
        generate(spec, "test", n=200, length=4), 6)
    assert sweep.cell_floors(CANONICAL["binding_v2"], 16, 20) == {}


# --------------------------------------------------------------------------- budget
def test_trace_budget_survives_a_spurious_checkpoint_row():
    """An event_trace model that emits ONE extra k-token checkpoint row must still fit its
    answer inside the budget. n_trace + 6 does not: it leaves 5 tokens past a 1-token answer,
    so any k >= 6 truncates a correct answer to 0."""
    for k in (4, 6, 8, 16):
        spec = _LOCAL.scaled(k=k)
        for L in (4, 8):
            n_trace = L * k + spec.chain_depth
            budget = sweep.trace_budget(spec, L, n_trace)
            assert budget >= n_trace + k + 2, (k, L)        # extra row + answer + <eos>
            assert budget > n_trace + 6


# ------------------------------------------------------------------ interleaved slots
def test_interleaved_slots_align_with_the_plain_prompt():
    """The interleaved prompt is the plain prompt with one checkpoint inserted per event."""
    for k, depth, compact in ((4, 1, False), (6, 2, False), (8, 1, True), (8, 2, True)):
        spec = _LOCAL.scaled(k=k, chain_depth=depth, start_trace=True, compact_events=compact)
        for L in spec.eval_lengths:
            for ex in generate(spec, "test", n=5, length=L):
                toks, slots = sweep._interleaved_slots(ex.prompt, ex.meta["interleaved_prompt"])
                assert len(slots) == L
                assert [toks[s] for s in slots] == ex.meta["trace"].split()[:L]


def test_depth_one_interleaved_answer_is_the_token_before_what():
    """Why the depth-1 interleaved arm needs guided eval: under this supervision the gold
    answer is a verbatim copy of the last checkpoint, which sits immediately before 'what' in
    every training document and is deleted from the free-running eval prompt."""
    spec = _LOCAL.scaled(k=8, chain_depth=1, start_trace=True)
    copies = 0
    for ex in generate(spec, "test", n=20, length=4):
        inter = ex.meta["interleaved_prompt"].split()
        plain = ex.prompt.split()
        # in TRAINING the token before "what" is the answer, with probability 1
        assert f"{inter[inter.index('what') - 1]}." == ex.answer
        # the free-running EVAL prompt has that position occupied by the last event's own
        # last token, so the copy rule the model was trained on reads off the wrong source
        before_what = plain[plain.index("what") - 1]
        assert before_what != inter[inter.index("what") - 1]
        copies += int(f"{before_what}." == ex.answer)
    assert copies == 0                                      # never accidentally still correct


# --------------------------------------------------------------------- guided free-run
class _Row:
    def __init__(self, value): self.value = value
    def float(self): return self
    def argmax(self): return self.value


class _Out:
    def __init__(self, value): self.value = value
    def __getitem__(self, _key): return _Row(self.value)


class _FakeTorch:
    """Just enough torch for sweep.guided_free_run_eval's decode loop."""
    bfloat16 = "bfloat16"

    @staticmethod
    def no_grad(): return contextlib.nullcontext()

    @staticmethod
    def autocast(*_a, **_k): return contextlib.nullcontext()

    @staticmethod
    def tensor(x, device=None): return x


class _OracleModel:
    """Emits the gold continuation: the next id of whichever expected sequence it is inside."""

    def __init__(self, expected, corrupt_slot=None):
        self.expected = expected                  # list[list[int]]
        self.corrupt_slot = corrupt_slot          # (seq_index, position) to answer wrongly
        self.calls = 0

    def eval(self): pass
    def train(self): pass

    def __call__(self, batch):
        prefix = batch[0]
        self.calls += 1
        # The sequence this prefix agrees with longest. Position-based rather than
        # exact-prefix, so the stub keeps answering after a deliberately corrupted slot has
        # been fed back into the context (which is the behaviour under test).
        best_i, best = 0, -1
        for si, seq in enumerate(self.expected):
            agree = 0
            for a, b in zip(prefix, seq):
                if a != b:
                    break
                agree += 1
            if agree > best:
                best_i, best = si, agree
        seq = self.expected[best_i]
        nxt = seq[min(len(prefix), len(seq) - 1)]
        if self.corrupt_slot == (best_i, len(prefix)):
            return _Out(nxt + 1)
        return _Out(nxt)


def _guided_fixture(spec, L, n):
    world, renderer = build_world(spec)
    tok = Tokenizer.build([world], renderer)
    expected = []
    for ex in generate(spec, "test", n=n, length=L):
        expected.append(tok.encode(f"{ex.meta['interleaved_prompt']} {ex.answer}") + [tok.eos_id])
    return tok, expected


def _run_guided(spec, L, n, model, tok):
    real = sys.modules.get("torch")
    sys.modules["torch"] = _FakeTorch
    try:
        return sweep.guided_free_run_eval(model, tok, spec, L, n, device="cpu")
    finally:
        if real is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = real


def test_guided_free_run_scores_an_oracle_model_at_one():
    """The eval-path sanity check: a backend emitting the gold continuation must score 1.000.
    A failure here is an eval defect, not a model result."""
    spec = _LOCAL.scaled(k=6, chain_depth=1, start_trace=True)
    tok, expected = _guided_fixture(spec, 4, 5)
    overall, ck, records = _run_guided(spec, 4, 5, _OracleModel(expected), tok)
    assert overall == 1.0 and ck == 1.0
    assert len(records) == 5 and all(ok for _i, _g, _p, ok, _gen, _gold in records)
    # generated checkpoints ride along so a below-threshold checkpoint_acc can be re-scored
    # under the reference's type-constrained rule without re-running the cell
    for _i, _g, _p, _ok, gen_ck, gold_ck in records:
        assert gen_ck == gold_ck and len(gen_ck) == 4


def test_guided_free_run_feeds_the_models_own_checkpoints():
    """A wrong checkpoint must be fed back (not silently corrected by the gold trace), so a
    model that mis-tracks one step is visible in checkpoint accuracy."""
    spec = _LOCAL.scaled(k=6, chain_depth=1, start_trace=True)
    tok, expected = _guided_fixture(spec, 4, 4)
    toks, slots = sweep._interleaved_slots(
        generate(spec, "test", n=4, length=4)[0].prompt,
        generate(spec, "test", n=4, length=4)[0].meta["interleaved_prompt"])
    clean, _ck, _r = _run_guided(spec, 4, 4, _OracleModel(expected), tok)
    model = _OracleModel(expected, corrupt_slot=(0, slots[0]))
    _overall, ck, _records = _run_guided(spec, 4, 4, model, tok)
    assert clean == 1.0
    assert ck < 1.0


def test_guided_free_run_needs_interleaved_supervision():
    spec = _LOCAL.scaled(k=6, chain_depth=1)               # no start_trace -> no interleaving
    tok, _ = _guided_fixture(spec.scaled(start_trace=True), 4, 1)
    try:
        _run_guided(spec, 4, 1, _OracleModel([]), tok)
    except ValueError as exc:
        assert "interleaved_prompt" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# ------------------------------------------------------------------------- reporting
def _fake_runs(values, length="4"):
    return [{"task": "s5_chain_local_v2", "arch": "gdp_hybrid", "seed": s,
             "lengths": {length: {"overall": v, "holder_acc": v, "value_acc": None,
                                  "answer_tokens": 1, "heldout_loss": 0.5,
                                  "floors": {"initial_map_chase": 0.195, "echo": 0.0,
                                             "uniform_non_start": 0.2, "uniform": 1 / 6}}}}
            for s, v in enumerate(values)]


def test_bimodal_cell_reports_its_per_seed_values():
    """The published k6/d2 cell rendered as '0.38±0.31 (0%)' — a converged seed shown as a
    floor. Per-seed values and min/median/max must both be present."""
    summary = sweep.aggregate(_fake_runs([0.155, 0.815, 0.170]))
    cell = summary["s5_chain_local_v2"]["gdp_hybrid"]["4"]
    assert cell["per_seed"] == [0.155, 0.815, 0.170]
    assert cell["max"] == 0.815 and cell["median"] == 0.170
    assert cell["p_converge"] == 0.0                       # the old summary's only signal


def _render(values):
    import pathlib
    import tempfile

    summary = sweep.aggregate(_fake_runs(values))
    cfg = {"tasks": ["s5_chain_local_v2"], "d_model": 320, "n_layers": 4, "steps": 8000,
           "seeds": [0, 1, 2], "train_n": 8000, "eval_n": 200}
    out = pathlib.Path(tempfile.mkdtemp()) / "s.md"
    sweep.write_markdown(summary, cfg, out)
    return out.read_text()


def test_markdown_shows_seeds_floors_and_drops_the_single_token_leg_column():
    text = _render([0.155, 0.815, 0.170])
    assert "0.15 0.81 0.17" in text                         # per-seed values, not a mean
    assert "floor: initial_map_chase" in text and "0.195" in text
    assert "holder/value" not in text                       # 1-token answers have no value leg
    assert "held-out loss" in text


def test_markdown_orders_floor_rows_by_value_and_bolds_the_operative_one():
    """A reader takes the first floor row as THE floor, so the largest has to be first — and
    on this cell the largest is not the initial-map chase."""
    lines = [ln for ln in _render([0.155, 0.815, 0.170]).splitlines() if "floor:" in ln]
    names = [ln.split("floor: ")[1].split("_")[0] for ln in lines]
    assert names == ["uniform", "initial", "uniform", "echo"]      # non_start, chase, 1/k, echo
    assert "uniform_non_start" in lines[0] and "**0.200**" in lines[0]
    assert "0.195" in lines[1] and "**" not in lines[1]            # chase is not the floor here


def test_prefix_decomp_reports_no_value_leg_for_single_token_answers():
    """'0.815 / 0.00' reads as a failed recall leg; a 1-token family has no second leg."""
    inspected = [("p", "g4.", "g4.", True), ("p", "g2.", "g7.", False)]
    d = sweep.prefix_decomp(inspected)
    assert d["value_acc"] is None and d["answer_tokens"] == 1
    two = [("p", "g4 v1.", "g4 v1.", True)]
    assert sweep.prefix_decomp(two)["value_acc"] == 1.0
    assert sweep.prefix_decomp(two)["answer_tokens"] == 2


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
