# FactWorld Backend API & Usage Guide

This document describes how to evaluate arbitrary models against FactWorld
tasks.  For the project overview and paper reproduction scripts, see
[`README.md`](../README.md).

## The `ModelBackend` interface

A backend is any object with one method and one property:

```python
class ModelBackend:
    @property
    def name(self) -> str:
        """Short identifier for this backend instance."""
        ...

    def generate(self, prompts: list[str], max_new_tokens: int,
                 stop_at: str | None = None) -> list[str]:
        """Return one continuation string per prompt (prompts must not be included)."""
        ...
```

The returned strings are scored with ``factworld.tasks.score_exact``, which
compares whitespace-token sequences up to the length of the gold answer.  Extra
whitespace or trailing tokens are ignored, so a backend does not need to
terminate generation exactly at ``.``.

## Built-in backends

All built-in backends live in ``factworld.backends``.

### `LocalBackend`

Wraps a FactWorld ``HybridLM`` and the FactWorld atomic tokenizer.  The
constructor builds a fresh, randomly-initialized model; for the train-then-eval
path, train a model with ``factworld.train`` and assign it to ``backend.model``.

```python
from factworld import tasks as TK, train as T
from factworld.backends import LocalBackend

spec = TK.CANONICAL["composite_copy_v2"]
w, _ = TK.build_world(spec)
train = TK.generate(spec, "train", n=8000)
tok, docs, _ = T.prepare([f"{e.prompt}{e.answer}" for e in train], [], [w])
run = T.run("gdp_hybrid", tok, docs, [], steps=100, batch=32,
            d_model=128, n_layers=2, return_model=True, device="cuda")

backend = LocalBackend([w], arch="gdp_hybrid", model=run["model"], device="cuda")
```

Direct greedy generation:

```python
preds = backend.generate(["what is a0 of g3 ? : "], max_new_tokens=4, stop_at=".")
```

### `HFBackend`

Evaluates any HuggingFace ``AutoModelForCausalLM``.

```python
from factworld.backends import HFBackend

backend = HFBackend("meta-llama/Llama-2-7b-hf", device="cuda")
conts = backend.generate(["what is a0 of g3 ? : "], max_new_tokens=8, stop_at=".")
```

### `APIBackend`

OpenAI-compatible chat-completions (OpenAI, vLLM, ollama, etc.). Calls are
issued concurrently (``max_workers``) and a ``system_prompt`` can be provided
to steer output formatting.

```python
import os
from factworld.backends import APIBackend

backend = APIBackend(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    max_workers=4,
    system_prompt=(
        "Answer each question with only the requested value, no explanation."
    ),
)
conts = backend.generate(["what is a0 of g3 ? : "], max_new_tokens=8, stop_at=".")
```

For local endpoints, point ``base_url`` at your server:

```python
# vLLM / ollama / llama.cpp server
backend = APIBackend(
    model="llama3.1",
    base_url="http://localhost:8000/v1",
)
```

### `FunctionBackend`

Wrap an arbitrary Python callable for one-off baselines or CI smoke tests.

```python
from factworld.backends import FunctionBackend

backend = FunctionBackend(
    lambda prompts, n, stop: ["g0 ."] * len(prompts),
    name="always-g0",
)
```

## Composite output format

Composite-family tasks (e.g. `composite_copy_v2`) require a two-token answer span
(``<holder> <value> .``). Naive chat-model prompts tend to emit only the value,
so the built-in CLI appends an explicit format instruction for API and HF
backends:

```bash
python scripts/eval_model.py composite_copy_v2 --backend api --model gpt-4o-mini
```

To disable it (e.g. for ablations):

```bash
python scripts/eval_model.py composite_copy_v2 --backend api --model gpt-4o-mini --no-composite-format
```

When using ``APIBackend`` directly, include the instruction in the
``system_prompt``:

```python
backend = APIBackend(
    model="gpt-4o-mini",
    system_prompt=(
        "Answer each question with only the requested value, no explanation. "
        "For questions that ask 'what is a0 of the holder of ...', "
        "answer with the holder's name followed by the requested value, "
        "like 'g3 v9'."
    ),
)
```

## Custom backend by subclassing

Subclass ``ModelBackend`` and pass the instance to
``factworld.runner.evaluate_task``:

```python
from factworld.backends import ModelBackend
from factworld.runner import evaluate_task
from factworld.tasks import CANONICAL

class MyBackend(ModelBackend):
    @property
    def name(self):
        return "my-backend"

    def generate(self, prompts, max_new_tokens, stop_at=None):
        return [my_model.complete(p, max_tokens=max_new_tokens, stop=stop_at) for p in prompts]

spec = CANONICAL["composite_copy_v2"]
result = evaluate_task(MyBackend(), spec, n=50)
print(result["overall"])
```

## The `evaluate_task` runner

```python
from factworld.runner import evaluate_task

result = evaluate_task(
    backend,
    spec,                        # or a canonical task name string
    split="test",
    n=200,                       # examples to evaluate
    length=64,                   # None -> spec.eval_lengths[0]
    max_new_tokens=8,            # None -> max answer length + 2
)
```

Returns a dictionary:

```python
{
    "task": spec.name,
    "backend": backend.name,
    "n": 200,
    "split": "test",
    "length": 64,
    "by_length": {64: 0.42},
    "overall": 0.42,
    "examples": [
        (prompt, gold, pred, correct),
        ...
    ],
}
```

To evaluate at every OOD length, call once per length:

```python
for L in spec.eval_lengths:
    result = evaluate_task(backend, spec, n=50, length=L)
    print(f"test@L{L}: {result['overall']:.3f}")
```

## API cost tips

- Use a small ``--n`` for prototyping (``--n 10`` or ``--n 50``).
- Set ``--max_new_tokens`` low; FactWorld answers are short (1–4 tokens).
- Evaluate one length at a time with ``--length``.
- Run local endpoints (vLLM, ollama, llama.cpp server) to avoid per-token API
  costs during development.

## Smoke test

```bash
python scripts/eval_model.py recall_v1 --backend function --n 5
```

This uses the hardcoded function backend and should finish instantly without
any external dependencies.

---

Back to the main documentation: [`README.md`](../README.md)
