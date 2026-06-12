"""Shallow / leakage baselines (M5). A surprising win here = artifact, not result.

These are model-free probes run on the eval items. The gate's teeth: if a shallow, non-
compositional method answers the HARD state task above the random floor, the instrument has a
shortcut and a model "success" would not prove state-tracking. (Recall is memorisation and
easy-state has recency structure — those being shallow-solvable is calibration, not leakage;
the hard rung must require the Sₖ composition.)
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from .render import Renderer
from .world import World


def answer_space(world: World, family: str) -> list[str]:
    if family == "recall":
        return list(world.value_vocab)
    if family == "state_easy":
        return list(world.holders)
    if family == "state_hard":
        return list(world.roles)
    raise ValueError(family)


def random_floor(world: World, family: str) -> float:
    return 1.0 / len(answer_space(world, family))


def _by_family(items):
    g = defaultdict(list)
    for it in items:
        g[it.family].append(it)
    return g


def majority_accuracy(items) -> dict[str, float]:
    out = {}
    for fam, its in _by_family(items).items():
        out[fam] = Counter(i.gold for i in its).most_common(1)[0][1] / len(its)
    return out


def answer_kl(golds, space) -> float:
    """KL(empirical || uniform) in nats over the answer space (0 = perfectly balanced)."""
    n, k = len(golds), len(space)
    counts = Counter(golds)
    kl = 0.0
    for tok in space:
        p = counts.get(tok, 0) / n
        if p > 0:
            kl += p * math.log(p * k)
    return kl


def _context_tokens(item, renderer: Renderer) -> list[str]:
    if item.family == "recall":
        return [item.entity, item.attribute]
    toks = " ".join(renderer.render_history(item.episode.events, with_steps=True)).split()
    toks.append(item.target)
    if item.t is not None:
        toks.append(f"s{item.t - 1}")
    return toks


def recency_accuracy(items, world: World, renderer: Renderer) -> dict[str, float]:
    """Predict the answer-space token appearing LAST in the context; fallback = family majority."""
    out = {}
    for fam, its in _by_family(items).items():
        space = set(answer_space(world, fam))
        majority = Counter(i.gold for i in its).most_common(1)[0][0]
        correct = 0
        for it in its:
            pred = majority
            for tk in reversed(_context_tokens(it, renderer)):
                if tk in space:
                    pred = tk
                    break
            correct += pred == it.gold
        out[fam] = correct / len(its)
    return out


def naive_bayes_accuracy(items, renderer: Renderer, seed: int = 0, train_frac: float = 0.5) -> dict[str, float]:
    """Multinomial Naive Bayes over bag-of-context-tokens — the surface-feature leakage detector."""
    out = {}
    for fam, its in _by_family(items).items():
        rng = random.Random(f"nb|{fam}|{seed}")
        its = list(its)
        rng.shuffle(its)
        cut = int(len(its) * train_frac)
        train, test = its[:cut], its[cut:]
        if not train or not test:
            continue
        class_counts = Counter(i.gold for i in train)
        tok_counts: dict = defaultdict(Counter)
        vocab: set = set()
        for i in train:
            for tk in _context_tokens(i, renderer):
                tok_counts[i.gold][tk] += 1
                vocab.add(tk)
        classes = list(class_counts)
        V = len(vocab)
        total = {c: sum(tok_counts[c].values()) for c in classes}
        logprior = {c: math.log(class_counts[c] / len(train)) for c in classes}
        correct = 0
        for i in test:
            ctx = _context_tokens(i, renderer)
            best, best_lp = None, None
            for c in classes:
                lp = logprior[c]
                denom = total[c] + V
                tc = tok_counts[c]
                for tk in ctx:
                    lp += math.log((tc[tk] + 1) / denom)
                if best_lp is None or lp > best_lp:
                    best, best_lp = c, lp
            correct += best == i.gold
        out[fam] = correct / len(test)
    return out


def identity_baseline_accuracy(items, world: World) -> dict[str, float]:
    """Hard-state structural 'no-op' predictor: guess the target agent's identity (initial) role.
    Near floor once the chain / as-of-t is past the Sₖ mixing threshold; reported alongside the floor
    so a model is never credited for the trivially-identity small-t region (adversarial finding)."""
    out = {}
    for fam, its in _by_family(items).items():
        if fam == "state_hard":
            out[fam] = sum(world.initial_assignment[i.target] == i.gold for i in its) / len(its)
    return out


def objblind_recency_accuracy(items) -> dict[str, float]:
    """Easy-state target-AGNOSTIC predictor: guess the last event's destination, ignoring the query.
    Inflated at short chains (the queried object is often the most recent); reported so easy short-L
    accuracy isn't mistaken for tracking (adversarial finding)."""
    out = {}
    for fam, its in _by_family(items).items():
        if fam == "state_easy":
            correct = 0
            for i in its:
                prefix = i.episode.events if i.t is None else i.episode.events[: i.t]
                correct += bool(prefix) and prefix[-1].args[1] == i.gold
            out[fam] = correct / len(its)
    return out
