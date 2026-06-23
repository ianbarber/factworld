"""Minimal example: evaluate a Hugging Face model on one FactWorld task.

Install the optional HF dependencies first:
    pip install -e ".[hf]"

Then run:
    python examples/eval_hf.py
"""
from __future__ import annotations

from factworld.backends import HFBackend
from factworld.runner import evaluate_task
from factworld.tasks import CANONICAL


def main() -> None:
    # Pick a small, public causal-LM.  Smaller models are cheaper but will floor
    # on the harder composition tasks; this script just demonstrates the API.
    model_name = "gpt2"  # or "HuggingFaceTB/SmolLM2-135M" for a modern tiny LM

    # 1. Build the backend.  This downloads/loads the tokenizer and model once.
    backend = HFBackend(model_name)

    # 2. Pick a canonical task.  "recall_v1" is a positive control (memorized
    #    fixed map) and therefore cheap to run for a smoke test.
    spec = CANONICAL["recall_v1"]

    # 3. Evaluate on a small held-out test slice.
    result = evaluate_task(
        backend,
        spec,
        split="test",
        n=20,          # number of examples
        length=4,      # OOD / difficulty coordinate
        max_new_tokens=8,
    )

    print(f"task={result['task']}  backend={result['backend']}")
    print(f"accuracy={result['overall']:.3f}  ({len(result['examples'])} examples)")


if __name__ == "__main__":
    raise SystemExit(main())
