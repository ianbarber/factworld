"""Casual CLI for evaluating external models on FactWorld tasks.

Supports local from-scratch training, HuggingFace transformers, OpenAI-compatible
APIs, and a hardcoded function backend for smoke tests.

Examples:
    python scripts/eval_model.py recall_v1 --backend function --n 5
    python scripts/eval_model.py composite_copy_v1 --backend api --model gpt-4o-mini --n 20
    python scripts/eval_model.py binding_v1 --backend hf --model meta-llama/Llama-2-7b-hf --n 50
    python scripts/eval_model.py conflict_v1 --backend local --arch gdp_hybrid --d_model 128 --n_layers 2 --steps 100

The API / HF backends automatically append a composite-output format instruction
for ``composite_copy_v1`` and ``composite_v1`` so chat models emit the required
``<holder> <value> .`` span (use ``--no-composite-format`` to disable).
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld.backends import APIBackend, FunctionBackend, HFBackend, LocalBackend
from factworld.runner import evaluate_task


DEFAULT_SYSTEM_PROMPT = (
    "You are taking a short test. Answer each question with only the requested "
    "value or values, no explanation. Use the same spelling as in the question."
)

COMPOSITE_FORMAT_PROMPT = (
    "For questions that ask 'what is a0 of the holder of ...', "
    "answer with the holder's name followed by the requested value, "
    "like 'g3 v9'."
)


def _mock_model(prompts, max_new_tokens, stop_at=None):
    """Hardcoded smoke-test backend; always predicts a plausible-looking answer."""
    return ["g0 ."] * len(prompts)


def _relaxed_score(pred: str, gold: str) -> int:
    """Tokenization-agnostic score: ignore whitespace and trailing periods."""
    pred_norm = pred.replace(".", "").replace(" ", "").strip()
    gold_norm = gold.replace(".", "").replace(" ", "").strip()
    return int(pred_norm == gold_norm)


def build_docs(examples, use_trace):
    """Training strings: prompt + (optional oracle worked-trace) + final answer."""
    docs = []
    for e in examples:
        trace = f"{e.meta['trace']} " if (use_trace and "trace" in e.meta) else ""
        docs.append(f"{e.prompt}{trace}{e.answer}")
    return docs


def build_local_backend(spec, arch, d_model, n_layers, steps, train_n, batch,
                        use_trace, seed, device):
    """Train a tiny local model on ``spec`` and wrap it for evaluation."""
    from factworld import train as T

    d_ff = 4 * d_model
    w, _ = TK.build_world(spec)
    train = TK.generate(spec, "train", n=train_n)
    tok, docs, _ = T.prepare(build_docs(train, use_trace), [], [w])
    run = T.run(arch, tok, docs, [], steps=steps, batch=batch, d_model=d_model,
                n_layers=n_layers, d_ff=d_ff, seed=seed, return_model=True,
                device=device)
    backend = LocalBackend([w], arch=arch, model=run["model"], tokenizer=tok,
                           device=device)
    return backend


def main():
    ap = argparse.ArgumentParser(description="Evaluate a model on a FactWorld task.")
    ap.add_argument("task", choices=list(TK.CANONICAL), help="Canonical task name.")
    ap.add_argument("--backend", choices=["local", "hf", "api", "function"], required=True,
                    help="Which backend to use for generation.")
    ap.add_argument("--model", default=None,
                    help="Model name or path (used by hf/api; ignored for local).")
    ap.add_argument("--arch", default="gdp_hybrid",
                    help="Architecture for the local backend (default: gdp_hybrid).")
    ap.add_argument("--d_model", type=int, default=256, help="Model width for local backend.")
    ap.add_argument("--n_layers", type=int, default=4, help="Model depth for local backend.")
    ap.add_argument("--steps", type=int, default=4000, help="Training steps for local backend.")
    ap.add_argument("--train_n", type=int, default=8000,
                    help="Number of training examples for local backend.")
    ap.add_argument("--batch", type=int, default=32, help="Training batch size for local backend.")
    ap.add_argument("--use_trace", action="store_true",
                    help="Train on the oracle worked-trace (if any) for local backend.")
    ap.add_argument("--device", default="cuda", help="Device for local/hf backends.")
    ap.add_argument("--base_url", default=None, help="API base URL (for api backend).")
    ap.add_argument("--api_key", default=None,
                    help="API key (for api backend; falls back to OPENAI_API_KEY).")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT,
                    help="System prompt for api/hf backends.")
    ap.add_argument("--composite_format", dest="composite_format",
                    action="store_true", default=True,
                    help="Append composite format instruction for composite tasks (default).")
    ap.add_argument("--no-composite-format", dest="composite_format",
                    action="store_false", help="Disable composite format instruction.")
    ap.add_argument("--n", type=int, default=50, help="Number of eval examples per length.")
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="Which split to evaluate.")
    ap.add_argument("--length", type=int, default=None, help="Override the eval length.")
    ap.add_argument("--max_new_tokens", type=int, default=16,
                    help="Generation budget per example.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--json_out", default=None,
                    help="Optional JSON output path (same schema as eval_openrouter_grid.py).")
    a = ap.parse_args()

    spec = TK.CANONICAL[a.task]

    system_prompt = a.system_prompt
    if a.composite_format and a.task in ("composite_copy_v1", "composite_v1"):
        system_prompt = f"{system_prompt} {COMPOSITE_FORMAT_PROMPT}"

    if a.backend == "local":
        backend = build_local_backend(
            spec, a.arch, a.d_model, a.n_layers, a.steps, a.train_n, a.batch,
            a.use_trace, a.seed, a.device
        )
        label = f"local/{a.arch} d{a.d_model} x{a.n_layers}"
    elif a.backend == "hf":
        model_name = a.model or "gpt2"
        backend = HFBackend(model_name, device=a.device, system_prompt=system_prompt)
        label = f"hf/{model_name}"
    elif a.backend == "api":
        model_name = a.model or "gpt-4o-mini"
        backend = APIBackend(model=model_name, api_key=a.api_key, base_url=a.base_url,
                             system_prompt=system_prompt)
        label = f"api/{model_name}"
    elif a.backend == "function":
        backend = FunctionBackend(_mock_model)
        label = "function/mock"
    else:
        raise ValueError(f"unknown backend: {a.backend}")

    lengths = [a.length] if a.length is not None else list(spec.eval_lengths)
    results = {
        L: evaluate_task(backend, spec, split=a.split, n=a.n, length=L,
                         max_new_tokens=a.max_new_tokens)
        for L in lengths
    }

    json_rows = []
    for L, result in results.items():
        examples = [
            {"prompt": p, "gold": g, "pred": pred, "exact": bool(ok),
             "relaxed": bool(_relaxed_score(pred, g))}
            for p, g, pred, ok in result["examples"]
        ]
        correct_exact = sum(e["exact"] for e in examples)
        correct_relaxed = sum(e["relaxed"] for e in examples)
        json_rows.append({
            "model": label,
            "task": a.task,
            "length": L,
            "n": len(examples),
            "system_prompt": system_prompt,
            "accuracy_exact": result["overall"],
            "accuracy_relaxed": correct_relaxed / len(examples),
            "correct_exact": correct_exact,
            "correct_relaxed": correct_relaxed,
            "elapsed": 0.0,
            "examples": examples,
        })

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(json_rows, f, indent=2)
        print(f"\nWrote JSON to {a.json_out}")

    length_label = f"L{a.length}" if a.length is not None else "canonical lengths"
    print(f"\n{a.task} [{label}] @ {length_label} — position-strict exact match:")
    for L, result in results.items():
        total = len(result["examples"])
        correct = sum(1 for _, _, _, ok in result["examples"] if ok)
        print(f"  test@L{L}: {result['overall']:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
