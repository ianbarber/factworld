"""s5_chain task validity: deterministic generation, no-wrap gate, explicit rendering."""
import pytest

from factworld.tasks import CANONICAL, generate


def test_generation_deterministic():
    spec = CANONICAL["s5_chain_v2"]
    a = generate(spec, "test", n=3, length=32)
    b = generate(spec, "test", n=3, length=32)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.answer for x in a] == [x.answer for x in b]
    assert [x.meta["path"] for x in a] == [x.meta["path"] for x in b]


def test_no_wrap_gate():
    spec = CANONICAL["s5_chain_v2"].scaled(chain_depth=16)
    with pytest.raises(ValueError, match="wraps"):
        generate(spec, "test", n=1, length=8)


def test_explicit_value_update_rendering():
    spec = CANONICAL["s5_chain_v2"]
    ex = generate(spec, "test", n=1, length=8)[0]
    assert "swaps the values of" in ex.prompt or "cycles a0:" in ex.prompt
    assert "becomes" in ex.prompt
    assert "(8 hops)" in ex.prompt


def test_path_consistency():
    """The gold answer is the last element of the stored path."""
    spec = CANONICAL["s5_chain_v2"]
    for ex in generate(spec, "test", n=10, length=32):
        assert ex.answer == f"{ex.meta['path'][-1]}."
        assert len(ex.meta["path"]) == ex.meta["depth"] + 1
