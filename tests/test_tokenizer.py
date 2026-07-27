"""M4 — atomic / closed-vocabulary tokenizer: exact whitespace round-trip,
unknown -> <unk>, determinism, fixed special ids, and bos/eos handling.

Runs with zero dependencies:  python3 tests/test_tokenizer.py
or under pytest:               uv run --with pytest pytest -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.config import WorldConfig  # noqa: E402
from factworld.render import Renderer  # noqa: E402
from factworld.tokenizer import Tokenizer  # noqa: E402
from factworld.world import World  # noqa: E402

_TARGET = World(WorldConfig(seed=0))
_AUX = World(WorldConfig(seed=0, id_namespace="aux0_"))
_R = Renderer()
_TOK = Tokenizer.build([_TARGET, _AUX], _R)


def _rendered_statements(world: World):
    """A large sample of in-vocab rendered statements from `world`."""
    out: list[str] = []

    # render_fact for many (entity, attr) pairs (and across paraphrase slots).
    for i in range(40):
        e = world.entities[i]
        a = world.attribute_names[i % len(world.attribute_names)]
        v = world.value_vocab[(i * 7) % len(world.value_vocab)]
        out.append(_R.render_fact(e, a, v))
        out.append(_R.render_fact(e, a, v, key=f"k{i}"))

    # render_history for easy + hard chains, lengths up to ~40, both with_steps modes.
    for L in (1, 4, 16, 32, 40):
        easy = world.sample_easy_chain(L, f"easy{L}")
        hard = world.sample_hard_chain(L, f"hard{L}")
        for chain in (easy, hard):
            for with_steps in (False, True):
                out.extend(_R.render_history(chain, with_steps=with_steps))

    # render_query, all three families, t=None and t set.
    e0, a0 = world.entities[0], world.attribute_names[0]
    obj, agent = world.objects[0], world.agents[0]
    out.append(_R.render_query("recall", entity=e0, attribute=a0))
    for t in (None, 1, 5, 12):
        out.append(_R.render_query("state_easy", target=obj, t=t))
        out.append(_R.render_query("state_hard", target=agent, t=t))
    return out


def test_roundtrip_target_and_aux():
    for world in (_TARGET, _AUX):
        for s in _rendered_statements(world):
            assert _TOK.decode(_TOK.encode(s)) == s, repr(s)


def test_no_unk_on_in_vocab_statements():
    for world in (_TARGET, _AUX):
        for s in _rendered_statements(world):
            assert _TOK.unk_id not in _TOK.encode(s), repr(s)


def test_foreign_token_maps_to_unk():
    ids = _TOK.encode("zzz999")
    assert _TOK.unk_id in ids
    # also inside a sentence
    assert _TOK.unk_id in _TOK.encode("what is zzz999 of e0 ?")


def test_build_is_deterministic():
    a = Tokenizer.build([_TARGET, _AUX], _R)
    b = Tokenizer.build([_TARGET, _AUX], _R)
    assert a.token_to_id == b.token_to_id


def test_specials_at_fixed_ids():
    assert _TOK.pad_id == 0
    assert _TOK.bos_id == 1
    assert _TOK.eos_id == 2
    assert _TOK.unk_id == 3
    assert _TOK.token_to_id["<pad>"] == 0
    assert _TOK.token_to_id["<bos>"] == 1
    assert _TOK.token_to_id["<eos>"] == 2
    assert _TOK.token_to_id["<unk>"] == 3


def test_vocab_size_consistency():
    assert _TOK.vocab_size == len(_TOK.token_to_id) == len(_TOK.id_to_token)
    # bijective: every id distinct.
    assert len(set(_TOK.token_to_id.values())) == _TOK.vocab_size


def test_bos_eos_handling():
    s = _R.render_query("recall", entity=_TARGET.entities[0], attribute=_TARGET.attribute_names[0])
    base = _TOK.encode(s)
    assert _TOK.encode(s, add_bos=True) == [_TOK.bos_id] + base
    assert _TOK.encode(s, add_eos=True) == base + [_TOK.eos_id]
    assert _TOK.encode(s, add_bos=True, add_eos=True) == [_TOK.bos_id] + base + [_TOK.eos_id]
    # decode without bos/eos still round-trips.
    assert _TOK.decode(_TOK.encode(s)) == s


def test_decode_skips_pad():
    s = _R.render_fact(_TARGET.entities[0], _TARGET.attribute_names[0], _TARGET.value_vocab[0])
    ids = _TOK.encode(s)
    padded = [_TOK.pad_id, _TOK.pad_id] + ids + [_TOK.pad_id]
    assert _TOK.decode(padded) == s


def test_rendering_has_no_unk():
    """Rendered documents must tokenize without <unk>."""
    r = Renderer()
    tok = Tokenizer.build([_TARGET, _AUX], r)

    # facts, events, queries
    e0, a0 = _TARGET.entities[0], _TARGET.attribute_names[0]
    v0 = _TARGET.value_vocab[0]
    obj, agent = _TARGET.objects[0], _TARGET.agents[0]

    docs = [
        r.render_fact(e0, a0, v0),
        " ".join(r.render_history(_TARGET.sample_easy_chain(8, "nat-e"), with_steps=True)),
        " ".join(r.render_history(_TARGET.sample_hard_chain(8, "nat-h"), with_steps=True)),
        r.render_query("recall", entity=e0, attribute=a0),
        r.render_query("state_easy", target=obj),
        r.render_query("state_hard", target=agent),
    ]
    for s in docs:
        ids = tok.encode(s)
        assert tok.unk_id not in ids, f"<unk> in: {s!r}"
        assert tok.decode(ids) == s


def _task_strings(spec):
    """Every string a local run feeds through the tokenizer for ``spec``.

    That is the prompt and the answer of both splits at every registered length, plus the
    meta strings the sweep builds training documents from (``trace``, ``interleaved_prompt``).
    s5_chain specs are additionally sampled under the two rendering/supervision ablations the
    local runs use (``compact_events``, ``start_trace``), since those change the surface
    grammar rather than the item stream.
    """
    from factworld.tasks import generate

    variants = [spec]
    if spec.family == "s5_chain":
        variants.append(spec.scaled(compact_events=True, start_trace=True))
    for sp in variants:
        for split, lengths in (("train", (None,)), ("test", sp.eval_lengths)):
            for L in lengths:
                for ex in generate(sp, split, n=6, length=L):
                    yield ex.prompt
                    yield ex.answer
                    for value in ex.meta.values():
                        if isinstance(value, str):
                            yield value


def test_every_canonical_task_round_trips_losslessly():
    """THE tokenizer contract, over the tasks that are actually trained locally.

    ``decode(encode(x)) == x`` and no ``<unk>`` for every string a local run sees. An
    <unk> here is silent input corruption: the model never observes the token the task
    turns on, so the cell measures the tokenizer rather than the architecture.
    """
    from factworld.tasks import CANONICAL, build_world

    for name, spec in CANONICAL.items():
        world, renderer = build_world(spec)
        tok = Tokenizer.build([world], renderer)
        for s in _task_strings(spec):
            assert tok.unk_id not in tok.encode(s), f"{name}: <unk> in {s!r}"
            assert tok.decode(tok.encode(s)) == s, f"{name}: round trip broken on {s!r}"


def test_retired_tasks_round_trip_losslessly():
    """Retired specs stay generable for historical reproduction, so they stay encodable."""
    from factworld.tasks import RETIRED, build_world

    for name, spec in RETIRED.items():
        world, renderer = build_world(spec)
        tok = Tokenizer.build([world], renderer)
        for s in _task_strings(spec):
            assert tok.unk_id not in tok.encode(s), f"{name}: <unk> in {s!r}"
            assert tok.decode(tok.encode(s)) == s, f"{name}: round trip broken on {s!r}"


def test_pointer_map_event_grammar_is_covered():
    """The s5_chain event vocabulary — the load-bearing tokens of the composite stressor.

    ``swaps the values of ...`` / ``cycles a0 simultaneously: ... takes ... old a0,`` and the
    compact ``swaps a0:`` / ``cycles a0:`` forms, plus the ``(N hops)`` query annotation.
    """
    for tk in ("values", "simultaneously:", "takes", "old", "a0,", "a0.", "a0:", "hops)",
               "(1", "(2", "(8", "(128"):
        assert tk in _TOK.token_to_id, tk


def test_holder_assertion_object_form_is_covered():
    """``g3 holds o0.`` glues the period to the object (the aux-world trace documents)."""
    s = _R.render_holder(_TARGET.objects[0], _TARGET.agents[0])
    assert _TOK.unk_id not in _TOK.encode(s)
    assert _TOK.decode(_TOK.encode(s)) == s


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
