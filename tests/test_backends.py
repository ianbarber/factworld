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


def _fake_client(response):
    """Build a fake OpenAI-style client whose ``create`` returns ``response``."""
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: response)
        )
    )


_FAKE_OPENAI = types.SimpleNamespace(OpenAI=lambda **kwargs: None)


def _make_backend(response, **kwargs):
    import factworld.backends as B

    client = _fake_client(response)
    with patch.dict(sys.modules, {"openai": _FAKE_OPENAI}):
        return B.APIBackend("test-model", client=client, **kwargs)


def test_api_backend_answer_mode_validation():
    try:
        _make_backend(None, answer_mode="sentences")
    except ValueError as exc:
        assert "answer_mode" in str(exc)
    else:
        raise AssertionError("APIBackend should reject unknown answer_mode")


def test_api_backend_words_mode():
    # 'Driver .' must survive intact as 'Driver' (tokens mode would keep 'Driver .').
    message = types.SimpleNamespace(content="Driver .")
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    response = types.SimpleNamespace(choices=[choice])
    backend = _make_backend(response, answer_mode="words")
    assert backend.generate(["who is p3 ?"], max_new_tokens=4, stop_at=None) == ["Driver"]

    # <think> blocks are still stripped; prose munging (colon split, comma rsplit,
    # prefix regexes) is skipped, and the pre-period span is kept.
    message2 = types.SimpleNamespace(content="<think>swap, swap: hm</think>\n Nurse, the last one.")
    choice2 = types.SimpleNamespace(message=message2, finish_reason="stop")
    response2 = types.SimpleNamespace(choices=[choice2])
    backend2 = _make_backend(response2, answer_mode="words")
    assert backend2.generate(["p"], max_new_tokens=4, stop_at=None) == ["Nurse, the last one"]

    # Unclosed think block -> no committed answer.
    message3 = types.SimpleNamespace(content="<think>still thinking")
    choice3 = types.SimpleNamespace(message=message3, finish_reason="length")
    response3 = types.SimpleNamespace(choices=[choice3])
    backend3 = _make_backend(response3, answer_mode="words")
    assert backend3.generate(["p"], max_new_tokens=4, stop_at=None) == [""]


def test_api_backend_tokens_mode_unchanged_with_explicit_kwarg():
    # answer_mode="tokens" must be byte-identical to the historical default path.
    message = types.SimpleNamespace(content="The final holders are g0, g2, g4")
    choice = types.SimpleNamespace(message=message, finish_reason=None)
    response = types.SimpleNamespace(choices=[choice])
    backend = _make_backend(response, answer_mode="tokens")
    assert backend.generate(["p"], max_new_tokens=4, stop_at=None) == ["g4"]


def test_api_backend_pop_call_meta_aggregates_and_clears():
    usage = types.SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=40,
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=25),
    )
    message = types.SimpleNamespace(content="g3 .")
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    response = types.SimpleNamespace(
        choices=[choice], usage=usage, model="test-model-2026", provider="FakeCloud"
    )
    backend = _make_backend(response)

    # Concurrent path: appends happen from ThreadPoolExecutor worker threads.
    backend.generate(["a", "b", "c"], max_new_tokens=4, stop_at=".")
    meta = backend.pop_call_meta()
    assert meta["calls"] == 3
    assert meta["errors"] == 0
    assert meta["usage"] == {"prompt_tokens": 300, "completion_tokens": 120, "reasoning_tokens": 75}
    assert meta["served_models"] == ["test-model-2026"]
    assert meta["providers"] == ["FakeCloud"]
    assert meta["finish_reasons"] == {"stop": 3}

    # pop clears: a second pop reports zero calls.
    meta2 = backend.pop_call_meta()
    assert meta2["calls"] == 0
    assert meta2["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    assert meta2["served_models"] == []
    assert meta2["finish_reasons"] == {}


def test_api_backend_pop_call_meta_openrouter_style_usage():
    # OpenRouter reports reasoning tokens directly on usage, with no
    # completion_tokens_details object.
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, reasoning_tokens=3)
    message = types.SimpleNamespace(content="g1")
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    response = types.SimpleNamespace(choices=[choice], usage=usage, model="served/model")
    backend = _make_backend(response)
    backend.generate(["p"], max_new_tokens=4, stop_at=None)
    meta = backend.pop_call_meta()
    assert meta["usage"]["reasoning_tokens"] == 3
    assert meta["providers"] == []  # provider field absent -> omitted, not None


def test_api_backend_error_path_records_meta():
    import factworld.backends as B

    def boom(**kwargs):
        # ValueError is in APIBackend's transient-fault except tuple
        # (JSONDecodeError is a ValueError), so this exercises the
        # retry-then-give-up path without needing real openai exceptions.
        raise ValueError("upstream returned garbage")

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=boom))
    )
    with patch.dict(sys.modules, {"openai": _FAKE_OPENAI}):
        backend = B.APIBackend("test-model", client=client)

    with patch("time.sleep"):  # skip retry backoff
        preds = backend.generate(["p"], max_new_tokens=4, stop_at=None)
    assert preds == [""]  # retry exhaustion still yields an empty prediction
    meta = backend.pop_call_meta()
    assert meta["calls"] == 1
    assert meta["errors"] == 1
    assert meta["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    assert meta["finish_reasons"] == {}


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
    n_passed = 0
    n_skipped = 0
    for fn in fns:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - pytest.skip raises a BaseException subclass
            if type(exc).__name__ == "Skipped":
                print(f"  skip {fn.__name__} ({exc})")
                n_skipped += 1
                continue
            raise
        print(f"  ok  {fn.__name__}")
        n_passed += 1
    print(f"\n{n_passed} tests passed, {n_skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
