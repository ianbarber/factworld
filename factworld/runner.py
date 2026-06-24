"""High-level evaluation runner for FactWorld tasks.

The runner glues the frozen task suite (``factworld.tasks``) to any backend that
implements the ``ModelBackend`` interface in ``factworld.backends``. It is
intentionally thin: deterministically generate examples, ask the backend for
greedy continuations, and score them with the one canonical metric
(position-strict exact match of the answer span).
"""
from __future__ import annotations

from .backends import ModelBackend
from .tasks import (
    CANONICAL,
    TaskSpec,
    generate,
    score_contains,
    score_exact,
    score_last_n,
    score_relaxed,
)


def evaluate_task(
    backend: ModelBackend,
    task: str | TaskSpec,
    *,
    split: str = "test",
    n: int = 200,
    length: int | None = None,
    max_new_tokens: int | None = None,
) -> dict:
    """Evaluate ``backend`` on a single FactWorld task.

    Args:
        backend: a ``ModelBackend`` instance.
        task: either a canonical task name or a ``TaskSpec``.
        split: ``"train"`` or ``"test"``.
        n: number of examples to evaluate.
        length: explicit difficulty coordinate for the ``"test"`` split; uses the
            task's default ``eval_lengths[0]`` when ``None``.
        max_new_tokens: generation budget. If ``None``, set to
            ``max(len(e.answer.split()) + 2 for e in examples)``.

    Returns:
        A dictionary with task name, backend name, evaluation parameters,
        per-length accuracy, overall accuracy, a list of inspected examples
        as ``(prompt, gold, pred, correct)`` tuples, and a ``metrics`` dict
        with canonical (``exact``) and tokenizer-robust (``relaxed``,
        ``contains``, ``last_n``) scores.
    """
    if isinstance(task, str):
        spec = CANONICAL[task]
    else:
        spec = task

    examples = generate(spec, split, n=n, length=length)

    if max_new_tokens is None:
        max_new_tokens = max((len(e.answer.split()) + 2 for e in examples), default=4)

    prompts = [e.prompt for e in examples]
    preds = backend.generate(prompts, max_new_tokens=max_new_tokens, stop_at=".")
    if len(preds) != len(prompts):
        raise RuntimeError(
            f"backend {backend.name!r} returned {len(preds)} predictions "
            f"for {len(prompts)} prompts"
        )

    scorers = {
        "exact": score_exact,
        "relaxed": score_relaxed,
        "contains": score_contains,
        "last_n": score_last_n,
    }
    inspected: list[tuple[str, str, str, bool]] = []
    example_metrics: list[dict[str, int]] = []
    by_length: dict[int, dict[str, list[int]]] = {}
    totals: dict[str, int] = {name: 0 for name in scorers}
    for example, pred in zip(examples, preds):
        scores = {name: fn(pred, example.answer) for name, fn in scorers.items()}
        for name, val in scores.items():
            totals[name] += val
        inspected.append((example.prompt, example.answer, pred, bool(scores["exact"])))
        example_metrics.append({name: scores[name] for name in scorers if name != "exact"})
        by_length.setdefault(example.length, {name: [] for name in scorers})
        for name, val in scores.items():
            by_length[example.length][name].append(val)

    n_examples = len(examples)
    metrics = {
        name: {"overall": totals[name] / n_examples if n_examples else 0.0}
        for name in scorers
    }
    for length_key, length_scores in by_length.items():
        for name, vals in length_scores.items():
            metrics[name].setdefault("by_length", {})[length_key] = sum(vals) / len(vals)

    return {
        "task": spec.name,
        "backend": backend.name,
        "n": n,
        "split": split,
        "length": length,
        "by_length": metrics["exact"].get("by_length", {}),
        "overall": metrics["exact"]["overall"],
        "examples": inspected,
        "example_metrics": example_metrics,
        "metrics": metrics,
    }
