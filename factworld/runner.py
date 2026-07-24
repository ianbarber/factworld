"""High-level evaluation runner for FactWorld tasks.

The runner glues the frozen task suite (``factworld.tasks``) to any backend that
implements the ``ModelBackend`` interface in ``factworld.backends``. It is
intentionally thin: deterministically generate examples, ask the backend for
greedy continuations, and score them. The headline fields (``overall`` /
``by_length`` / the inspected ``correct`` flag) use the one canonical metric —
**relaxed match** (``tasks.CANONICAL_METRIC``); ``exact`` / ``contains`` /
``last_n`` are reported as diagnostics under ``metrics``.
"""
from __future__ import annotations

from dataclasses import replace

from .backends import ModelBackend
from .render import Renderer
from .tasks import (
    CANONICAL,
    CANONICAL_METRIC,
    Example,
    TaskSpec,
    generate,
    score_contains,
    score_exact,
    committed_answer,
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
    n_shot: int = 0,
    stop_at: str | None = ".",
    extract_commit: bool = False,
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
        n_shot: number of training demonstrations to prepend to each test prompt.
        stop_at: stop generation at this token; ``None`` disables early stopping.
        extract_commit: score a multi-line emission's committed final line
            (``tasks.committed_answer``) instead of its first tokens. Reasoning-arm
            cells only: in the instant regime visible working is a protocol leak,
            not an answer, so crediting a spilled trace's final line would break
            the in-weights semantics.

    Returns:
        A dictionary with task name, backend name, evaluation parameters,
        per-length accuracy, overall accuracy, a list of inspected examples
        as ``(prompt, gold, pred, correct)`` tuples (``correct`` reflects the
        canonical relaxed match), and a ``metrics`` dict with the canonical
        (``relaxed``) score plus diagnostics (``exact``, ``contains``,
        ``last_n``).
    """
    if isinstance(task, str):
        spec = CANONICAL[task]
    else:
        spec = task

    examples = generate(spec, split, n=n, length=length)

    if n_shot:
        # Use length-matched test examples as demonstrations (indices offset past the
        # scored set so there is no overlap). Format with explicit Q/A labels.
        all_test = generate(spec, split, n=n + n_shot, length=length)
        examples = all_test[:n]
        demos = all_test[n:]
        demo_text = "\n\n".join(f"Question: {d.prompt}\nAnswer: {d.answer}" for d in demos)
        examples = [
            replace(ex, prompt=f"{demo_text}\n\nQuestion: {ex.prompt}\nAnswer:")
            for ex in examples
        ]

    if max_new_tokens is None:
        max_new_tokens = max((len(e.answer.split()) + 2 for e in examples), default=4)

    prompts = [e.prompt for e in examples]
    preds = backend.generate(prompts, max_new_tokens=max_new_tokens, stop_at=stop_at)
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
        # A local model marks the end of its answer with <eos>; anything generated past it is
        # budget-filling continuation, not answer. Without this cut a trace-mode prediction
        # ("...scratchpad... g3. <eos> g7 g0 ...") scores its junk tail under last_n and reads
        # as chance no matter how often the committed answer is right.
        pred = pred.split("<eos>")[0]
        if extract_commit:
            # A reasoning endpoint that spills working into the visible completion commits
            # to the single-token final line, not to the working's first tokens (see
            # tasks.committed_answer; inert for single-line answers and local streams).
            pred = committed_answer(pred)
        # Normalize output (detach attached punctuation, expand contractions) to canonical
        # whitespace tokens before scoring. Both prediction and gold are normalized so attached-
        # punctuation answers (e.g. "g4.") score correctly against equally-attached gold.
        pred_norm = Renderer.normalize(pred)
        gold_norm = Renderer.normalize(example.answer)
        scores = {name: fn(pred_norm, gold_norm) for name, fn in scorers.items()}
        for name, val in scores.items():
            totals[name] += val
        # `correct` reflects the canonical (relaxed) metric so the inspected examples and the
        # headline `overall` agree on what "right" means; per-example exact/contains/last_n are
        # still available in `example_metrics` for diagnostics.
        inspected.append((example.prompt, example.answer, pred, bool(scores[CANONICAL_METRIC])))
        example_metrics.append(dict(scores))
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
        "canonical_metric": CANONICAL_METRIC,
        "by_length": metrics[CANONICAL_METRIC].get("by_length", {}),
        "overall": metrics[CANONICAL_METRIC]["overall"],
        "examples": inspected,
        "example_metrics": example_metrics,
        "metrics": metrics,
    }
