"""Casual CLI for evaluating external models on FactWorld tasks.

Supports local from-scratch training, HuggingFace transformers, OpenAI-compatible
APIs, and a hardcoded function backend for smoke tests.

Examples:
    python scripts/eval_model.py recall_v1 --backend function --n 5
    python scripts/eval_model.py composite_copy_v1 --backend api --model gpt-4o-mini --n 20
    python scripts/eval_model.py binding_v1 --backend hf --model meta-llama/Llama-2-7b-hf --n 50
    python scripts/eval_model.py conflict_v1 --backend local --arch gdp_hybrid --d_model 128 --n_layers 2 --steps 100
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld.backends import APIBackend, FunctionBackend, HFBackend, LocalBackend
from factworld.runner import evaluate_task


def _mock_model(prompts, max_new_tokens, stop_at=None):
    """Hardcoded smoke-test backend; always predicts a plausible-looking answer."""
    return ["g0 ."] * len(prompts)


def build_local_backend(spec, arch, d_model, n_layers, steps, seed, device):
    """Train a tiny local model on ``spec`` and wrap it for evaluation."""
    from factworld import train as T

    d_ff = 4 * d_model
    w, _ = TK.build_world(spec)
    train = TK.generate(spec, "train", n=8000)
    tok, docs, _ = T.prepare([f"{e.prompt}{e.answer}" for e in train], [], [w])
    run = T.run(arch, tok, docs, [], steps=steps, batch=32, d_model=d_model,
                n_layers=n_layers, d_ff=d_ff, seed=seed, return_model=True,
                device=device)
    backend = LocalBackend([w], arch=arch, model=run["model"], device=device)
    return backend


def main():
    ap = argparse.ArgumentParser(description="Evaluate a model on a FactWorld task.")
    ap.add_argument("task", choices=list(TK.CANONICAL), help="Canonical task name.")
    ap.add_argument("--backend", choices=["local", "hf", "api", "function"], required=True,
                    help="Which backend to use for generation.")
    ap.add_argument("--model", default=None,
                    help="Model name or path (used by hf/api; for local defaults to --arch).")
    ap.add_argument("--arch", default="gdp_hybrid",
                    help="Architecture for the local backend (default: gdp_hybrid).")
    ap.add_argument("--d_model", type=int, default=256, help="Model width for local backend.")
    ap.add_argument("--n_layers", type=int, default=4, help="Model depth for local backend.")
    ap.add_argument("--steps", type=int, default=4000, help="Training steps for local backend.")
    ap.add_argument("--device", default="cuda", help="Device for local/hf backends.")
    ap.add_argument("--base_url", default=None, help="API base URL (for api backend).")
    ap.add_argument("--api_key", default=None,
                    help="API key (for api backend; falls back to OPENAI_API_KEY).")
    ap.add_argument("--n", type=int, default=50, help="Number of eval examples per length.")
    ap.add_argument("--length", type=int, default=None, help="Override the eval length.")
    ap.add_argument("--max_new_tokens", type=int, default=16,
                    help="Generation budget per example.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    a = ap.parse_args()

    spec = TK.CANONICAL[a.task]

    if a.backend == "local":
        backend = build_local_backend(
            spec, a.arch, a.d_model, a.n_layers, a.steps, a.seed, a.device
        )
        label = f"local/{a.arch} d{a.d_model} x{a.n_layers}"
    elif a.backend == "hf":
        model_name = a.model or "gpt2"
        backend = HFBackend(model_name, device=a.device)
        label = f"hf/{model_name}"
    elif a.backend == "api":
        model_name = a.model or "gpt-4o-mini"
        backend = APIBackend(model=model_name, api_key=a.api_key, base_url=a.base_url)
        label = f"api/{model_name}"
    elif a.backend == "function":
        backend = FunctionBackend(_mock_model)
        label = "function/mock"
    else:
        raise ValueError(f"unknown backend: {a.backend}")

    lengths = [a.length] if a.length is not None else list(spec.eval_lengths)
    results = {
        L: evaluate_task(backend, spec, split="test", n=a.n, length=L,
                         max_new_tokens=a.max_new_tokens)
        for L in lengths
    }

    length_label = f"L{a.length}" if a.length is not None else "canonical lengths"
    print(f"\n{a.task} [{label}] @ {length_label} — position-strict exact match:")
    for L, result in results.items():
        total = len(result["examples"])
        correct = sum(1 for _, _, _, ok in result["examples"] if ok)
        print(f"  test@L{L}: {result['overall']:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
