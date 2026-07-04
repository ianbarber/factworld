"""Run a FactWorld benchmark task end-to-end: train from scratch, evaluate with the canonical metric.

This is the single entry point that makes the frozen task suite (`factworld.tasks`) *runnable*: pick a
canonical task (or a scaled variant), train a model on its `train` split, and score its `test` splits at
each OOD length with the one canonical metric (relaxed match of the final answer span).

  .venv/bin/python scripts/run_benchmark.py composite_v1 --arch gdp_hybrid --d_model 256 --steps 4000
  # programmatic:
  from run_benchmark import run_task
  acc = run_task("composite_copy_v1", arch="gdp_hybrid", d_model=512, n_layers=8, steps=25000)
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld.backends import LocalBackend
from factworld.render import Renderer
from factworld.runner import evaluate_task


def build_docs(examples, use_trace):
    """Training strings: prompt + (optional oracle worked-trace) + final answer.

    Prompts end with '?' (attached), so a single space separates the query from
    the answer continuation.
    """
    docs = []
    for e in examples:
        trace = f" {e.meta['trace']} " if (use_trace and "trace" in e.meta) else " "
        docs.append(f"{e.prompt}{trace}{e.answer}")
    return docs


def run_task(name, *, spec=None, arch="gdp_hybrid", d_model=256, n_layers=4, d_ff=None, steps=4000,
             batch=32, train_n=8000, eval_n=200, seed=0, use_trace=False, device="cuda"):
    """Train on the task's `train` split; return {eval_length: canonical relaxed-match accuracy}."""
    import torch

    from factworld import train as T

    spec = spec or TK.CANONICAL[name]
    d_ff = d_ff or 4 * d_model
    w, r = TK.build_world(spec)
    train = TK.generate(spec, "train", n=train_n)
    tok, docs, _ = T.prepare(build_docs(train, use_trace), [], [w], renderer=r)
    run = T.run(arch, tok, docs, [], steps=steps, batch=batch, d_model=d_model, n_layers=n_layers,
                d_ff=d_ff, seed=seed, return_model=True, device=device)
    model = run["model"]

    backend = LocalBackend([w], arch=arch, model=model, tokenizer=tok, device=device)

    out = {}
    for L in spec.eval_lengths:
        result = evaluate_task(backend, spec, split="test", n=eval_n, length=L)
        out[L] = result["overall"]

    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TK.CANONICAL))
    ap.add_argument("--arch", default="gdp_hybrid")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_trace", action="store_true", help="train on the oracle worked-trace (if any)")
    a = ap.parse_args()
    acc = run_task(a.task, arch=a.arch, d_model=a.d_model, n_layers=a.n_layers, steps=a.steps,
                   seed=a.seed, use_trace=a.use_trace)
    print(f"\n{a.task} [{a.arch} d{a.d_model} x{a.n_layers}, {a.steps} steps] — relaxed match (canonical):")
    for L, v in acc.items():
        print(f"  test@L{L}: {v:.3f}")


if __name__ == "__main__":
    main()
