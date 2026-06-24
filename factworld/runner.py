"""High-level evaluation runner for FactWorld tasks.

The runner glues the frozen task suite (``factworld.tasks``) to any backend that
implements the ``ModelBackend`` interface in ``factworld.backends``. It is
intentionally thin: deterministically generate examples, ask the backend for
greedy continuations, and score them with the one canonical metric
(position-strict exact match of the answer span).
"""
from __future__ import annotations

from .backends import ModelBackend
from .tasks import CANONICAL, TaskSpec, generate, score_exact


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
        per-length accuracy, overall accuracy, and a list of inspected examples
        as ``(prompt, gold, pred, correct)`` tuples.
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

    inspected: list[tuple[str, str, str, bool]] = []
    by_length: dict[int, list[int]] = {}
    total_correct = 0
    for example, pred in zip(examples, preds):
        correct = score_exact(pred, example.answer)
        total_correct += correct
        inspected.append((example.prompt, example.answer, pred, bool(correct)))
        by_length.setdefault(example.length, []).append(correct)

    overall = total_correct / len(examples) if examples else 0.0
    by_length_acc = {length_key: sum(scores) / len(scores) for length_key, scores in by_length.items()}

    return {
        "task": spec.name,
        "backend": backend.name,
        "n": n,
        "split": split,
        "length": length,
        "by_length": by_length_acc,
        "overall": overall,
        "examples": inspected,
    }
