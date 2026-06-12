"""Data pipeline — turn the world into training texts and eval probes for each condition.

Training texts (next-token LM):
  - recall facts (parametric recall);
  - AUX operator scenarios (disjoint symbols): "scn<id> : <history> <query> <answer> ." — teach the
    transition operator AND the query->answer format, with the scenario's history present;
  - TARGET scenarios: "scn<id> : <history> ." — history only, no answer (the IWL-memorised set).

Eval probes (single-answer-token, exact-match vs the oracle):
  - recall   : IWL (cloze on a fact; value in weights)   + ICL (fact stated in the prompt, then cloze);
  - state_*  : IWL (scn<id> + query; history in weights) + ICL (fresh history in the prompt, all lengths).

The scenario id binds an IWL query to a stored history via a marker + shared digit tokens
(Renderer.render_scenario), so there is no per-scenario embedding.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .config import WorldConfig
from .corpus import recall_documents
from .eval import Episode
from .oracle import Oracle
from .render import Renderer
from .world import World


@dataclass(frozen=True)
class Probe:
    condition: str            # "iwl" | "icl"
    family: str               # "recall" | "state_easy" | "state_hard"
    length: int | None
    prompt: str               # text ending right before the answer token
    gold: str


@dataclass(frozen=True)
class DataSpec:
    seed: int = 0
    n_recall: int = 3000
    n_aux_worlds: int = 6
    n_aux_scn: int = 600
    n_icl_demo: int = 800     # fresh target-symbol history->query->answer demos (teach format + in-context op)
    n_target_scn: int = 600
    n_eval: int = 200
    train_lengths: tuple = (16, 32)
    ood_lengths: tuple = (8, 48, 64, 96, 128)


def _touched(events):
    return sorted({e.args[0] for e in events})


def _q_and_gold(r, oracle, fam, ep, target, t=None):
    if fam == "state_hard":
        return r.render_query("state_hard", target=target, t=t), oracle.hard_role(list(ep.events), target, t)
    return r.render_query("state_easy", target=target, t=t), oracle.easy_holder(list(ep.events), target, t)


def _cloze(r, e, a, v, key):
    """A fact rendered up to (but not including) its value, plus the value token, for cloze recall."""
    toks = r.render_fact(e, a, v, key=key).split()
    vi = len(toks) - 2                       # value sits just before the final "."
    return " ".join(toks[:vi]), toks[vi]


def build(spec: DataSpec = DataSpec(), world_config: WorldConfig | None = None):
    cfg = world_config or WorldConfig(seed=spec.seed)
    world, oracle, r = World(cfg), None, Renderer()
    oracle = Oracle(world)
    rng = random.Random(f"data|{spec.seed}")

    width = max(4, len(str(spec.n_aux_scn + spec.n_target_scn)))
    id_pool = rng.sample(range(10 ** width), spec.n_aux_scn + spec.n_target_scn)  # unique, randomized
    pool_i = 0

    texts: list[str] = []
    probes: list[Probe] = []
    worlds = [world]

    # --- recall facts + cloze probes ---
    texts += [d.text for d in recall_documents(world, r, spec.n_recall, seed=str(spec.seed))]
    for e in rng.sample(list(world.entities), min(spec.n_eval, len(world.entities))):
        a = rng.choice(world.attribute_names)
        v = oracle.recall(e, a)
        cue, gold = _cloze(r, e, a, v, key=f"probe|{e}|{a}")
        probes.append(Probe("iwl", "recall", None, cue + " ", gold))
        ctx, _ = _cloze(r, e, a, v, key=f"ctx|{e}|{a}")            # a (possibly different) phrasing as context
        probes.append(Probe("icl", "recall", None, f"{ctx} {v} . {cue} ", gold))

    # --- aux operator scenarios (disjoint symbols): history + query + answer ---
    for w in range(spec.n_aux_worlds):
        aw = World(replace(cfg, id_namespace=f"aux{w}_", n_entities=0, n_attributes=0, value_vocab_size=1))
        ao = Oracle(aw)
        worlds.append(aw)
        for i in range(spec.n_aux_scn // spec.n_aux_worlds):
            fam = "state_hard" if i % 2 else "state_easy"
            L = rng.choice(spec.train_lengths)
            ev = tuple((aw.sample_hard_chain if fam == "state_hard" else aw.sample_easy_chain)(L, f"aux{w}|{i}"))
            ep = Episode(fam, ev, L, f"aux{w}|{i}")
            target = rng.choice(aw.agents if fam == "state_hard" else _touched(ev))
            q, g = _q_and_gold(r, ao, fam, ep, target)
            sid = r.render_scenario(id_pool[pool_i]); pool_i += 1
            hist = " ".join(r.render_history(ev, with_steps=True))
            texts.append(f"{sid} : {hist} {q} {g} .")

    # --- target ICL demonstrations: fresh episodes, history + query + answer (teach the answer
    # format and the in-context operator on TARGET symbols, at train lengths; these are fresh, not
    # the id-tagged IWL-eval scenarios, so they don't leak any memorised final) ---
    for i in range(spec.n_icl_demo):
        fam = "state_hard" if i % 2 else "state_easy"
        L = rng.choice(spec.train_lengths)
        ev = tuple((world.sample_hard_chain if fam == "state_hard" else world.sample_easy_chain)(L, f"icldemo|{i}"))
        ep = Episode(fam, ev, L, f"icldemo|{i}")
        target = rng.choice(world.agents if fam == "state_hard" else _touched(ev))
        q, g = _q_and_gold(r, oracle, fam, ep, target)
        texts.append(f"{' '.join(r.render_history(ev, with_steps=True))} {q} {g} .")

    # --- target scenarios: id-tagged history-only (IWL-memorised) ---
    target_scn = []
    for i in range(spec.n_target_scn):
        fam = "state_hard" if i % 2 else "state_easy"
        L = rng.choice(spec.train_lengths)
        ev = tuple((world.sample_hard_chain if fam == "state_hard" else world.sample_easy_chain)(L, f"tgt|{i}"))
        ep = Episode(fam, ev, L, f"tgt|{i}")
        sid = r.render_scenario(id_pool[pool_i]); pool_i += 1
        texts.append(f"{sid} : {' '.join(r.render_history(ev, with_steps=True))} .")
        target_scn.append((sid, ep, fam))

    # --- IWL state probes: query a memorised target scenario by id ---
    for sid, ep, fam in rng.sample(target_scn, min(spec.n_eval, len(target_scn))):
        target = rng.choice(world.agents if fam == "state_hard" else _touched(ep.events))
        q, g = _q_and_gold(r, oracle, fam, ep, target)
        probes.append(Probe("iwl", fam, ep.length, f"{sid} : {q} ", g))

    # --- ICL state probes: fresh histories in the prompt, across lengths ---
    lengths = spec.train_lengths + spec.ood_lengths
    per = max(1, spec.n_eval // (2 * len(lengths)))
    for fam in ("state_easy", "state_hard"):
        for L in lengths:
            for j in range(per):
                ev = tuple((world.sample_hard_chain if fam == "state_hard" else world.sample_easy_chain)(L, f"icl|{fam}|{L}|{j}"))
                ep = Episode(fam, ev, L, "icl")
                target = rng.choice(world.agents if fam == "state_hard" else _touched(ev))
                q, g = _q_and_gold(r, oracle, fam, ep, target)
                probes.append(Probe("icl", fam, L, f"{' '.join(r.render_history(ev, with_steps=True))} {q} ", g))

    return texts, probes, worlds
