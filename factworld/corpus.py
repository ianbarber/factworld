"""Corpus assembly — data properties + the operator-learning split.

Recall documents carry the three co-occurring properties Kim et al. (2510.02370) show are
preconditions for balanced parametric/in-context use: (1) skewed (Zipf) entity document
frequency, (2) intra-document repetition, (3) a small, *measured* dose of intra-document
inconsistency (~1%). All three are logged in `Document.meta` so the corpus is stratification-
capable (the "gap vanished under frequency stratification" falsification must be runnable).

The operator-learning split (Option A):
- TARGET state histories are emitted *without finals* (history-only) — the IWL-history-only
  measurement set: it tests parametric *application* of the operator, not endpoint memorisation.
- AUXILIARY operator-worlds (disjoint `id_namespace` symbol sets, minimal recall footprint)
  carry worked traces (hard) / final answers (easy) that *teach* the transition operator over
  name-binding rather than over one fixed 5-symbol table.
- Format is decorrelated: some answer-free history docs are also dropped into the aux worlds,
  so "has-trace" does not perfectly predict "aux-world".
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .eval import Episode, _freq_bands
from .oracle import Oracle
from .render import Renderer
from .world import World


@dataclass(frozen=True)
class Document:
    text: str
    kind: str            # "recall_fact" | "history_only" | "aux_trace" | "aux_answer"
    split: str           # "train" | "aux"
    meta: dict


@dataclass(frozen=True)
class Corpus:
    documents: list      # list[Document]
    worlds: list         # target + aux worlds — the closed token universe for the tokenizer


def recall_documents(world: World, renderer: Renderer, n_docs: int, seed: str = "0",
                     repetition: int = 2, inconsistency_rate: float = 0.01,
                     n_bands: int = 3) -> list[Document]:
    if inconsistency_rate > 0 and repetition < 2:
        raise ValueError("inconsistency injection needs repetition >= 2 so the correct value out-votes the wrong one")
    rng = random.Random(f"corpus|recall|{world.config.id_namespace}|{world.config.seed}|{seed}")
    entities = list(world.entities)
    weights = [world.entity_freq[e] for e in entities]   # (1) Zipf document frequency
    band = _freq_bands(world, n_bands)
    docs: list[Document] = []
    for d in range(n_docs):
        ent = rng.choices(entities, weights=weights, k=1)[0]
        statements = []
        for a in world.attribute_names:                  # (2) intra-document repetition
            v = world.attrs[ent][a]
            for rep in range(repetition):
                statements.append(renderer.render_fact(ent, a, v, key=f"{seed}|{d}|{a}|{rep}"))
        inconsistent = rng.random() < inconsistency_rate  # (3) measured ~1% inconsistency
        if inconsistent:
            a = rng.choice(world.attribute_names)
            wrong = rng.choice([x for x in world.value_vocab if x != world.attrs[ent][a]])
            statements.append(renderer.render_fact(ent, a, wrong, key=f"{seed}|{d}|inc"))
        rng.shuffle(statements)
        docs.append(Document(" ".join(statements), "recall_fact", "train",
                             {"entity": ent, "freq_band": band[ent], "inconsistent": inconsistent}))
    return docs


def history_only_documents(world: World, renderer: Renderer, episodes, with_steps: bool = True) -> list[Document]:
    """State histories with NO final/as-of-t answer stated (IWL-history-only)."""
    docs: list[Document] = []
    for ep in episodes:
        lines = renderer.render_history(ep.events, with_steps=with_steps)
        docs.append(Document(" ".join(lines), "history_only", "train",
                             {"family": ep.family, "length": ep.length,
                              "ns": world.config.id_namespace, "seed": ep.seed}))
    return docs


def aux_operator_documents(world: World, oracle: Oracle, renderer: Renderer, episodes) -> list[Document]:
    """Operator-teaching supervision: worked traces (hard) / final answers (easy)."""
    docs: list[Document] = []
    for ep in episodes:
        if ep.family == "state_hard":
            trace = oracle.hard_trace(ep.events)          # trace[i+1] = assignment AFTER event i
            lines: list[str] = []
            for i, e in enumerate(ep.events):
                lines.append(renderer.render_event(e, step=f"s{i}"))
                assign = trace[i + 1]
                for ag in world.agents:                   # state every agent's role after the step
                    lines.append(renderer.render_role(ag, assign[ag], step=f"s{i}"))
            kind = "aux_trace"
        elif ep.family == "state_easy":
            lines = renderer.render_history(ep.events, with_steps=True)
            for o in world.objects:                       # final holder of every object (the answer)
                lines.append(renderer.render_holder(o, oracle.easy_holder(list(ep.events), o)))
            kind = "aux_answer"
        else:
            raise ValueError(ep.family)
        docs.append(Document(" ".join(lines), kind, "aux",
                             {"family": ep.family, "length": ep.length,
                              "ns": world.config.id_namespace, "seed": ep.seed}))
    return docs


def _aux_world(target_config, ns: str) -> World:
    # operator-only world: disjoint symbols, minimal recall footprint (keeps objects/agents/roles,
    # so the embedding table isn't bloated by unused aux recall vocab)
    return World(replace(target_config, id_namespace=ns, n_entities=0, n_attributes=0, value_vocab_size=1))


def _episodes(world: World, family: str, sampler, lengths, n_each: int, tag: str) -> list[Episode]:
    eps: list[Episode] = []
    for length in lengths:
        for i in range(n_each):
            eps.append(Episode(family, tuple(sampler(length, f"{tag}|{i}")), length, f"{tag}|{family}|{length}|{i}"))
    return eps


def build_corpus(target_world: World, oracle: Oracle, renderer: Renderer, *,
                 n_recall_docs: int = 2000, easy_lengths=(8, 16), hard_lengths=(16, 32),
                 n_state_episodes: int = 50, n_aux_worlds: int = 8, seed: str = "0",
                 repetition: int = 2, inconsistency_rate: float = 0.01) -> Corpus:
    docs: list[Document] = []
    worlds: list[World] = [target_world]

    # 1. recall facts (target world, data properties)
    docs += recall_documents(target_world, renderer, n_recall_docs, seed=seed,
                             repetition=repetition, inconsistency_rate=inconsistency_rate)

    # 2. TARGET state histories WITHOUT finals (the IWL-history-only measurement set)
    tgt_eps = (_episodes(target_world, "state_easy", target_world.sample_easy_chain, easy_lengths, n_state_episodes, f"tgt|{seed}")
               + _episodes(target_world, "state_hard", target_world.sample_hard_chain, hard_lengths, n_state_episodes, f"tgt|{seed}"))
    docs += history_only_documents(target_world, renderer, tgt_eps)

    # 3. AUX operator-worlds (disjoint symbols) — traces (hard) / answers (easy) teach the operator
    for w in range(n_aux_worlds):
        aw = _aux_world(target_world.config, f"aux{w}_")
        ao = Oracle(aw)
        worlds.append(aw)
        aux_eps = (_episodes(aw, "state_easy", aw.sample_easy_chain, easy_lengths, n_state_episodes, f"aux{w}")
                   + _episodes(aw, "state_hard", aw.sample_hard_chain, hard_lengths, n_state_episodes, f"aux{w}"))
        docs += aux_operator_documents(aw, ao, renderer, aux_eps)
        docs += history_only_documents(aw, renderer, aux_eps[::4])  # decorrelate {has-trace, world}

    return Corpus(docs, worlds)
