"""S₅ role-permutation problems rendered as prompts — the single source of truth.

Same S₅ problems (same world state, same permutation sequences, same oracle gold)
rendered under three framings; the only thing that varies is the surface text, so a
change in accuracy attributes the gap to presentation, not computation:

  V0_abstract              — g/r tokens, "swaps"/"cycles roles", initial assignment UNSTATED
  V0_abstract_stated_init  — same, but "Initially g0 has role r0, ..." is given
  V1_concrete              — people + jobs ("Eva and Bob swap jobs", "Cara takes Eva's job, ...",
                             "what job does Cara have?"), initial stated

The events come from the task's deterministic sampler (``sample_hard_chain`` + ``_rng``
for the queried agent) and the gold from the symbolic oracle, so every framing is the
same problem, just re-worded.  A cycle is rendered explicitly ("X takes Y's job")
matching the oracle's semantics, so there is no arrow-direction ambiguity.

This module was extracted from ``scripts/experiment_s5_framing.py`` and reproduces its
prompts and episode seeds EXACTLY, so results are comparable with docs/openrouter/s5-*.jsonl.

Benchmark-facing API (contract C2):
    gen_examples(length, n, framing="concrete") -> [(system_prompt, user_prompt, gold_answer)]
        framing "concrete" -> V1_concrete (gold is the job word, e.g. "Driver")
        framing "abstract_stated" -> V0_abstract_stated_init (gold is the role token, e.g. "r2")
    score(pred, gold) -> {"relaxed": 0|1, "contains": 0|1}
        relaxed = first content token equals gold (canonical relaxed semantics for a
        one-token answer); case-sensitive, punctuation-insensitive via ``content_tokens``.
"""
from __future__ import annotations

from factworld.tasks import CANONICAL, _world, _rng
from factworld.tasks import content_tokens

NAMES = {"g0": "Alice", "g1": "Bob", "g2": "Cara", "g3": "Dan", "g4": "Eva"}
JOBS = {"r0": "Manager", "r1": "Chef", "r2": "Driver", "r3": "Clerk", "r4": "Guard"}

FRAMINGS = ("V0_abstract", "V0_abstract_stated_init", "V1_concrete")

# benchmark-facing aliases (contract C2) -> internal framing names
FRAMING_ALIASES = {"concrete": "V1_concrete", "abstract_stated": "V0_abstract_stated_init"}


def _name(g): return NAMES[g]
def _job(r): return JOBS[r]


def render_event_v0(e, i):
    if e.kind == "swap_role":
        x, y = e.args
        return f"s{i} swaps {x} and {y}."
    cyc = e.args
    return f"s{i} cycles roles: " + " -> ".join(cyc) + "."


def render_event_v1(e, i):
    if e.kind == "swap_role":
        x, y = e.args
        return f"s{i} {_name(x)} and {_name(y)} swap jobs."
    cyc = e.args
    m = len(cyc)
    # oracle semantics: a[c_i] = old[c_{(i-1) mod m}]  ->  c_i takes c_{i-1}'s old role
    parts = [f"{_name(cyc[i])} takes {_name(cyc[(i - 1) % m])}'s job" for i in range(m)]
    return f"s{i} job rotation: " + ", ".join(parts) + "."


INIT_ABSTRACT = "Initially g0 has role r0, g1 has r1, g2 has r2, g3 has r3, g4 has r4."
INIT_CONCRETE = ("Five people — Alice, Bob, Cara, Dan, Eva — each hold one job. "
                 "Initially: Alice is Manager, Bob is Chef, Cara is Driver, Dan is Clerk, Eva is Guard.")


def render_prompt(framing, events, agent, gold):
    """Return (system_prompt, user_prompt, gold_answer_token)."""
    if framing == "V0_abstract":
        sysp = ("You are taking a short test. Answer with only the requested value, no explanation. "
                "For 'what role does X have?' answer with only a role token "
                "(r0, r1, r2, r3, or r4) followed by a period. Example: 'r2 .'")
        hist = " ".join(render_event_v0(e, i) for i, e in enumerate(events))
        user = f"{hist} what role does {agent} have?"
        return sysp, user, gold
    if framing == "V0_abstract_stated_init":
        sysp = ("You are taking a short test. Answer with only the requested value, no explanation. "
                "For 'what role does X have?' answer with only a role token "
                "(r0, r1, r2, r3, or r4) followed by a period. Example: 'r2 .'")
        hist = " ".join(render_event_v0(e, i) for i, e in enumerate(events))
        user = f"{INIT_ABSTRACT} {hist} what role does {agent} have?"
        return sysp, user, gold
    if framing == "V1_concrete":
        sysp = ("You are taking a short test. Answer with only the requested value, no explanation. "
                "The jobs are Manager, Chef, Driver, Clerk, Guard. For 'what job does X have?' "
                "answer with only the job name followed by a period. Example: 'Driver .'")
        hist = " ".join(render_event_v1(e, i) for i, e in enumerate(events))
        user = f"{INIT_CONCRETE} {hist} what job does {_name(agent)} have?"
        return sysp, user, _job(gold)
    raise ValueError(framing)


def gen_problems(spec, w, oracle, length, n):
    """Deterministic (events, agent, gold) list — identical problems across framings."""
    out = []
    for idx in range(n):
        events = w.sample_hard_chain(length, episode_seed=f"{spec.name}|{idx}")
        agent = _rng(spec, "test", length, idx).choice(w.agents)
        gold = oracle.hard_role(events, agent)
        out.append((events, agent, gold))
    return out


def gen_examples(length: int, n: int, framing: str = "concrete"):
    """Rendered prompts for the benchmark: [(system_prompt, user_prompt, gold_answer)].

    ``framing`` is "concrete" (gold is the job word, e.g. "Driver") or
    "abstract_stated" (gold is the role token, e.g. "r2").  Problems and prompts are
    deterministic and byte-identical to scripts/experiment_s5_framing.py's V1_concrete /
    V0_abstract_stated_init renderings, so results are comparable across runs.
    """
    internal = FRAMING_ALIASES.get(framing, framing)
    if internal not in FRAMINGS:
        raise ValueError(f"unknown framing {framing!r}; expected one of "
                         f"{sorted(FRAMING_ALIASES)} or {list(FRAMINGS)}")
    spec = CANONICAL["s5_v1"]
    w, _r, oracle = _world(spec)
    return [render_prompt(internal, events, agent, gold)
            for events, agent, gold in gen_problems(spec, w, oracle, length, n)]


def score(pred_text, gold_token):
    # s5_concrete cells are always reasoning-arm, so a working-spilling emission is
    # scored on its committed final line (tasks.committed_answer; inert for the clean
    # single-line answers every measured cell has produced — 78/78 gap-free in history).
    from factworld.tasks import committed_answer
    ct = content_tokens(committed_answer(pred_text))
    first = ct[0] if ct else ""
    return {
        "relaxed": int(first == gold_token),          # first content token matches
        "contains": int(gold_token in ct),            # gold token appears anywhere
    }
