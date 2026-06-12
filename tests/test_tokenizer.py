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


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
