"""Eval-suite generator — task items emitted from the KB + oracle ONLY, never from rendered docs.

This module deliberately does not import the renderer: an EvalItem's ground truth comes from
`Oracle`, so there is no path by which a rendering artifact can leak into the labels. The
harness later *presents* an item (ICL: history in the prompt; IWL: history in weights), but the
item and its gold are fixed here from the world.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .oracle import Oracle
from .world import Event, World


@dataclass(frozen=True)
class Episode:
    family: str                 # "state_easy" | "state_hard"
    events: tuple[Event, ...]
    length: int
    seed: str


@dataclass(frozen=True)
class EvalItem:
    family: str                 # "recall" | "state_easy" | "state_hard"
    gold: str
    # recall
    entity: str | None = None
    attribute: str | None = None
    freq_band: int | None = None
    # state
    episode: Episode | None = None
    target: str | None = None   # object (easy) or agent (hard)
    t: int | None = None        # as-of-t (None => final state)
    length: int | None = None


def _freq_bands(world: World, n_bands: int) -> dict[str, int]:
    ranked = sorted(world.entities, key=lambda e: world.entity_freq[e])  # ascending frequency
    size = max(1, len(ranked) // n_bands)
    return {e: min(i // size, n_bands - 1) for i, e in enumerate(ranked)}


def recall_suite(world: World, oracle: Oracle, n_items: int, seed: str = "0", n_bands: int = 3) -> list[EvalItem]:
    rng = random.Random(f"eval|recall|{world.config.id_namespace}|{world.config.seed}|{seed}")
    band = _freq_bands(world, n_bands)
    items = []
    for _ in range(n_items):
        e = rng.choice(world.entities)
        a = rng.choice(world.attribute_names)
        items.append(EvalItem("recall", oracle.recall(e, a), entity=e, attribute=a, freq_band=band[e]))
    return items


def easy_suite(world: World, oracle: Oracle, lengths, n_per_length: int, seed: str = "0",
               as_of_t: bool = False) -> list[EvalItem]:
    rng = random.Random(f"eval|easy|{world.config.id_namespace}|{world.config.seed}|{seed}")
    items = []
    for L in lengths:
        for i in range(n_per_length):
            es = f"{seed}|{L}|{i}"
            events = tuple(world.sample_easy_chain(L, es))
            ep = Episode("state_easy", events, L, es)
            t = rng.randint(1, L) if as_of_t else None
            # query an object the presented history actually determines (touched within the prefix),
            # so the gold reflects last-write-wins tracking rather than an unobservable initial holder
            touched = sorted({e.args[0] for e in (events if t is None else events[:t])})
            obj = rng.choice(touched)
            gold = oracle.easy_holder(list(events), obj, t)
            items.append(EvalItem("state_easy", gold, episode=ep, target=obj, t=t, length=L))
    return items


def hard_suite(world: World, oracle: Oracle, lengths, n_per_length: int, seed: str = "0",
               as_of_t: bool = False) -> list[EvalItem]:
    rng = random.Random(f"eval|hard|{world.config.id_namespace}|{world.config.seed}|{seed}")
    items = []
    for L in lengths:
        for i in range(n_per_length):
            es = f"{seed}|{L}|{i}"
            events = tuple(world.sample_hard_chain(L, es))
            ep = Episode("state_hard", events, L, es)
            agent = rng.choice(world.agents)
            # as-of-t for hard samples t >= k: below the S_k mixing threshold a query is shallowly
            # solvable by "guess the identity role" (most agents still hold their identity start).
            t = rng.randint(min(world.k, L), L) if as_of_t else None
            gold = oracle.hard_role(list(events), agent, t)
            items.append(EvalItem("state_hard", gold, episode=ep, target=agent, t=t, length=L))
    return items
