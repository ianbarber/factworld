"""Tests for the new backend layer.

The zero-dependency path (``FunctionBackend`` + ``evaluate_task``) runs with pure
stdlib.  Optional backends are exercised with import guards and mocks.

Run directly:  python3 tests/test_backends.py
Run with pytest: python3 -m pytest tests/test_backends.py
"""
from __future__ import annotations

import builtins
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


def test_evaluate_task_groups_by_length():
    """Per-length accuracy is reported under ``by_length``."""
    spec = CANONICAL["recall_copy_v1"]
    gold = {ex.prompt: ex.answer for ex in generate(spec, "test", n=10, length=6)}

    def oracle(prompts, max_new_tokens, stop_at):
        return [gold.get(p, "v0 .") for p in prompts]

    backend = FunctionBackend(oracle, name="oracle")
    result = evaluate_task(backend, spec, split="test", n=10, length=6, max_new_tokens=4)

    assert 6 in result["by_length"]
    assert result["by_length"][6] == result["overall"]
    assert result["overall"] == 1.0


def test_evaluate_task_rejects_wrong_prediction_count():
    def short_backend(prompts, max_new_tokens, stop_at):
        return ["v0 ."]  # one answer for many prompts

    backend = FunctionBackend(short_backend, name="short")
    try:
        evaluate_task(backend, CANONICAL["recall_v1"], split="test", n=5, length=4)
    except RuntimeError as exc:
        assert "returned 1 predictions" in str(exc)
    else:
        raise AssertionError("evaluate_task should raise when backend returns wrong count")


# --- Optional backend import guards ------------------------------------------

def _block_import(name: str):
    """Return a context manager that makes ``builtins.__import__`` raise for ``name``."""
    real_import = builtins.__import__

    def fake_import(import_name, *args, **kwargs):
        if import_name == name or import_name.startswith(f"{name}."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(import_name, *args, **kwargs)

    return patch.object(builtins, "__import__", fake_import)


def test_hf_backend_import_error():
    from factworld.backends import HFBackend

    with _block_import("transformers"):
        try:
            HFBackend("gpt2")
        except ImportError as exc:
            assert "transformers" in str(exc).lower()
        else:
            raise AssertionError("HFBackend should raise ImportError when dependencies are missing")


def test_api_backend_import_error():
    from factworld.backends import APIBackend

    with _block_import("openai"):
        try:
            APIBackend("test-model")
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

    # APIBackend re-attaches the stop token when the API reports finish_reason == "stop",
    # strips common answer prefixes / surrounding whitespace, and normalizes a glued trailing
    # period to match FactWorld's atomic tokenizer.
    preds = backend.generate(["facts what is a0 of g3 ? : "], max_new_tokens=4, stop_at=".")
    assert preds == ["g3 ."]

    # Without a stop string the raw content is normalized (prefixes/whitespace stripped).
    preds = backend.generate(["facts what is a0 of g3 ? : "], max_new_tokens=4, stop_at=None)
    assert preds == ["g3"]

    # Natural-mode prose prefixes are stripped as well.
    message2 = types.SimpleNamespace(content="Let's track the swaps: r2")
    choice2 = types.SimpleNamespace(message=message2, finish_reason=None)
    response2 = types.SimpleNamespace(choices=[choice2])
    client2 = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: response2)
        )
    )
    with patch.dict(sys.modules, {"openai": fake_openai}):
        backend2 = B.APIBackend("test-model", client=client2)
    preds2 = backend2.generate(["prompt"], max_new_tokens=4, stop_at=None)
    assert preds2 == ["r2"]

    # Listing all holders should normalize to the last one.
    message3 = types.SimpleNamespace(content="The final holders are g0, g2, g4")
    choice3 = types.SimpleNamespace(message=message3, finish_reason=None)
    response3 = types.SimpleNamespace(choices=[choice3])
    client3 = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: response3)
        )
    )
    with patch.dict(sys.modules, {"openai": fake_openai}):
        backend3 = B.APIBackend("test-model", client=client3)
    preds3 = backend3.generate(["prompt"], max_new_tokens=4, stop_at=None)
    assert preds3 == ["g4"]


# --- LocalBackend (torch-dependent) -----------------------------------------

def test_local_backend_builds_tokenizer_if_torch_available():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        # Use pytest.skip when running under pytest, otherwise just return.
        try:
            import pytest
            pytest.skip(f"torch not installed: {exc}")
        except ImportError:
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
