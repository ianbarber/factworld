"""Edit semantics — the append round.

An append adds a single event to a state episode and produces an exact diff manifest
(changed / unchanged targets), so forgetting, specificity, and the *append-changes-answer*
control are measurable per individual fact: a memoriser of final states fails the after-query,
a transition-learner updates it. The manifest's `unchanged` set doubles as the specificity
(unchanged-neighbours) probe.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .eval import Episode, EvalItem
from .oracle import Oracle
from .world import Event, World


@dataclass(frozen=True)
class Edit:
    kind: str            # "append"
    event: Event
    manifest: dict       # {"changed": {target: [before, after]}, "unchanged": [target, ...]}


def _state(world: World, oracle: Oracle, family: str, events: list[Event]) -> dict[str, str]:
    if family == "state_easy":
        return {o: oracle.easy_holder(events, o) for o in world.objects}
    if family == "state_hard":
        return dict(oracle.hard_assignment(events))
    raise ValueError(f"not a state family: {family!r}")


def append_edit(world: World, oracle: Oracle, episode: Episode, new_event: Event) -> Edit:
    before = _state(world, oracle, episode.family, list(episode.events))
    after = _state(world, oracle, episode.family, list(episode.events) + [new_event])
    changed = {t: [before[t], after[t]] for t in before if before[t] != after[t]}
    unchanged = [t for t in before if before[t] == after[t]]
    return Edit("append", new_event, {"changed": changed, "unchanged": unchanged})


def _changing_event(world, oracle, episode, target, rng) -> Event:
    """An appended event guaranteed to flip `target`'s answer."""
    if episode.family == "state_easy":
        cur = oracle.easy_holder(list(episode.events), target)
        h = rng.choice([x for x in world.holders if x != cur])
        return Event("move" if rng.random() < 0.5 else "give", (target, h))
    # hard: assignment is a bijection, so swapping the target with ANY other agent changes its role
    other = rng.choice([g for g in world.agents if g != target])
    return Event("swap_role", (target, other))


def _nonchanging_event(world, oracle, episode, target, rng) -> Event:
    """A real append that changes some OTHER target but leaves `target` untouched (specificity)."""
    if episode.family == "state_easy":
        obj = rng.choice([o for o in world.objects if o != target])
        cur = oracle.easy_holder(list(episode.events), obj)
        h = rng.choice([x for x in world.holders if x != cur])
        return Event("move" if rng.random() < 0.5 else "give", (obj, h))
    # hard: permute two agents that are NOT the target -> target's role is untouched
    a, b = rng.sample([g for g in world.agents if g != target], 2)
    return Event("swap_role", (a, b))


def append_pair(world: World, oracle: Oracle, episode: Episode, target: str,
                seed: str = "0", changing: bool = True) -> tuple[EvalItem, EvalItem, Edit]:
    """Return (before_item, after_item, edit) for a one-event append on `target`."""
    rng = random.Random(
        f"edit|{world.config.id_namespace}|{world.config.seed}|{episode.seed}|{target}|{seed}|{changing}"
    )
    new_event = (_changing_event if changing else _nonchanging_event)(world, oracle, episode, target, rng)
    after_events = episode.events + (new_event,)
    after_ep = Episode(episode.family, after_events, episode.length + 1, f"{episode.seed}|+1")
    edit = append_edit(world, oracle, episode, new_event)

    answer = oracle.easy_holder if episode.family == "state_easy" else oracle.hard_role
    before = EvalItem(episode.family, answer(list(episode.events), target),
                      episode=episode, target=target, length=episode.length)
    after = EvalItem(episode.family, answer(list(after_events), target),
                     episode=after_ep, target=target, length=episode.length + 1)
    return before, after, edit
