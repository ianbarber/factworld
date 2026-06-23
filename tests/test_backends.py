"""Tests for the new backend layer.

The zero-dependency path (``FunctionBackend`` + ``evaluate_task``) runs with pure
stdlib.  Optional backends are exercised with import guards and mocks.

Run directly:  python3 tests/test_backends.py
Run with pytest: python3 -m pytest tests/test_backends.py
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factworld.backends import FunctionBackend, ModelBackend
from factworld.runner import evaluate_task
from factworld.tasks import CANONICAL, build_world, generate


# --- ModelBackend contract ---------------------------------------------------

def test_model_backend_is_abstract():
    try:
        ModelBackend()
    except TypeError as exc:
        msg = str(exc).lower()
        assert "abstract" in msg
        assert "generate" in msg
    else:
        raise AssertionError("ModelBackend should not be instantiable")


# --- FunctionBackend ---------------------------------------------------------

def test_function_backend_passes_arguments():
    calls = []

    def fn(prompts, max_new_tokens, stop_at):
        calls.append({"prompts": prompts, "max_new_tokens": max_new_tokens, "stop_at": stop_at})
        return [f"{p} -> answer" for p in prompts]

    backend = FunctionBackend(fn, name="recorder")
    out = backend.generate(["hello", "world"], max_new_tokens=7, stop_at=".")

    assert len(calls) == 1
    assert calls[0]["prompts"] == ["hello", "world"]
    assert calls[0]["max_new_tokens"] == 7
    assert calls[0]["stop_at"] == "."
    assert out == ["hello -> answer", "world -> answer"]
    assert backend.name == "recorder"


def test_function_backend_default_stop_at_none():
    calls = []

    def fn(prompts, max_new_tokens, stop_at):
        calls.append(stop_at)
        return prompts

    backend = FunctionBackend(fn)
    backend.generate(["a"], max_new_tokens=1)
    assert calls[-1] is None
    assert backend.name == "function"


# --- evaluate_task -----------------------------------------------------------

def test_evaluate_task_with_function_backend():
    spec = CANONICAL["recall_v1"]
    gold = {ex.prompt: ex.answer for ex in generate(spec, "test", n=10, length=4)}

    def oracle(prompts, max_new_tokens, stop_at):
        return [gold[p] for p in prompts]

    backend = FunctionBackend(oracle, name="oracle")
    result = evaluate_task(backend, spec, split="test", n=10, length=4, max_new_tokens=4)

    assert result["task"] == spec.name
    assert result["backend"] == "oracle"
    assert result["n"] == 10
    assert result["split"] == "test"
    assert result["length"] == 4
    assert result["overall"] == 1.0
    assert len(result["examples"]) == 10
    for prompt, answer, pred, correct in result["examples"]:
        assert isinstance(prompt, str) and isinstance(answer, str) and isinstance(pred, str)
        assert correct is True


# --- Optional backend import guards ------------------------------------------

def test_hf_backend_import_error():
    import factworld.backends as B

    with patch.dict(sys.modules, {"transformers": None, "torch": None}):
        reloaded = importlib.reload(B)
        try:
            reloaded.HFBackend("gpt2")
        except ImportError as exc:
            assert "transformers" in str(exc).lower()
        else:
            raise AssertionError("HFBackend should raise ImportError when dependencies are missing")


def test_api_backend_import_error():
    import factworld.backends as B

    with patch.dict(sys.modules, {"openai": None}):
        reloaded = importlib.reload(B)
        try:
            reloaded.APIBackend("test-model")
        except ImportError as exc:
            assert "openai" in str(exc).lower()
        else:
            raise AssertionError("APIBackend should raise ImportError when openai is missing")


# --- APIBackend mock ---------------------------------------------------------

def test_api_backend_mock():
    import factworld.backends as B

    message = types.SimpleNamespace(content=" g3")
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    response = types.SimpleNamespace(choices=[choice])
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    fake_openai = types.SimpleNamespace(OpenAI=lambda **kwargs: client)

    # Inject a fake ``openai`` module so the constructor does not need the real package.
    with patch.dict(sys.modules, {"openai": fake_openai}):
        backend = B.APIBackend("test-model", client=client)

    # APIBackend re-attaches the stop token when the API reports finish_reason == "stop".
    preds = backend.generate(["facts what is a0 of g3 ? : "], max_new_tokens=4, stop_at=".")
    assert preds == [" g3."]

    # Without a stop string the raw content is returned unchanged.
    preds = backend.generate(["facts what is a0 of g3 ? : "], max_new_tokens=4, stop_at=None)
    assert preds == [" g3"]


# --- LocalBackend (torch-dependent) -----------------------------------------

def test_local_backend_builds_tokenizer_if_torch_available():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        print(f"  skip test_local_backend_builds_tokenizer_if_torch_available (no torch: {exc})")
        return

    import factworld.backends as B

    spec = CANONICAL["recall_v1"]
    world, _ = build_world(spec)
    backend = B.LocalBackend([world], arch="transformer", d_model=32, n_layers=1, device="cpu")
    assert backend.tokenizer.vocab_size > 0
    assert backend.name == "local-transformer"


# --- stdlib runner -----------------------------------------------------------

def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
