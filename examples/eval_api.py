"""Minimal example: evaluate an OpenAI-compatible API on one FactWorld task.

Install the API client first:
    pip install openai

This works with OpenAI, vLLM, llama.cpp-server, or any other provider that
exposes the chat-completions endpoint.

Run:
    export OPENAI_BASE_URL=http://localhost:8000/v1
    export OPENAI_API_KEY=not-needed-for-local-vllm
    python examples/eval_api.py
"""
from __future__ import annotations

import os

from factworld.backends import APIBackend
from factworld.runner import evaluate_task
from factworld.tasks import CANONICAL


def main() -> None:
    # Base URL of the API server.  For OpenAI itself use "https://api.openai.com/v1".
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

    # API key.  Many local servers (vLLM, llama.cpp) accept any non-empty string.
    api_key = os.getenv("OPENAI_API_KEY", "not-needed-for-local")

    # Model name as known by the serving engine (e.g. "gpt-4o", "meta-llama/Llama-2-7b-hf").
    model = os.getenv("OPENAI_MODEL", "gpt2")

    # 1. Build the backend.
    backend = APIBackend(model=model, base_url=base_url, api_key=api_key)

    # 2. Pick a small task for the smoke test.
    spec = CANONICAL["recall_v1"]

    # 3. Evaluate.
    result = evaluate_task(
        backend,
        spec,
        split="test",
        n=20,
        length=4,
        max_new_tokens=8,
    )

    print(f"task={result['task']}  backend={result['backend']}")
    print(f"accuracy={result['overall']:.3f}  ({len(result['examples'])} examples)")


if __name__ == "__main__":
    raise SystemExit(main())
