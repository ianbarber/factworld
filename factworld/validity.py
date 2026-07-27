"""Validity gate — proves the instrument has signal with ZERO training.

Pass conditions (thresholds tunable; existence non-negotiable):
- oracle == 100% (by construction);
- balanced answer distributions (EXCESS KL from uniform, above the finite-sample bias, below a
  threshold) so accuracy is meaningful and majority is no cheap win;
- hard-state NOT shallow-solvable: Naive-Bayes + recency near floor (surface leakage), AND the
  identity-guess "no-op" predictor near floor (the structural shortcut the identity-start
  convention could create at small t — adversarial finding; the as-of-t suite now samples t>=k).

The KL of an empirical distribution from uniform is biased upward by ~(k-1)/(2n) even when the
true distribution IS uniform; we subtract that bias and gate on the excess.

This module also hosts the STRONG recency baseline for the task suite (``strong_recency_pred`` /
``strong_recency_accuracy``, consumed by scripts/validate_suite.py): predict the LAST give-event's
recipient (binding), plus that holder's stated a0 fact (composite). This is the adversary that
exposed the v1 give-stream sampler (resolving write clustered near the stream end: ~0.34@L16 on
the now-RETIRED composite_copy_v1 — see tasks.RETIRED, issue #11) and that the registered
last_write_uniform v2 specs hold at ~chance (gated by scripts/validate_suite.py).

Run directly to print the report:  python3 -m factworld.validity
"""
from __future__ import annotations

import re
from collections import Counter

from .baselines import (
    answer_kl,
    answer_space,
    identity_baseline_accuracy,
    naive_bayes_accuracy,
    objblind_recency_accuracy,
    random_floor,
    recency_accuracy,
)
from .config import WorldConfig
from .eval import easy_suite, hard_suite, recall_suite
from .oracle import Oracle
from .render import Renderer
from .world import World

KL_EXCESS_TAU = 0.02     # nats above the finite-sample bias (k-1)/(2n)
LEAK_MARGIN = 0.10       # a shallow baseline within floor+this counts as "near floor"
MAJ_MARGIN = 0.05


def run_gate(seed: int = 0, n_dist: int = 2000, n_leak: int = 400,
             easy_lengths=(8, 16), hard_lengths=(16, 32)) -> dict:
    world = World(WorldConfig(seed=seed))
    oracle = Oracle(world)
    renderer = Renderer()

    # distribution samples (balance) — large so the KL estimate is low-bias
    recall_pop = [oracle.recall(e, a) for e in world.entities for a in world.attribute_names]
    easy_dist = easy_suite(world, oracle, easy_lengths, n_dist // len(easy_lengths), seed="dist")
    hard_dist = hard_suite(world, oracle, hard_lengths, n_dist // len(hard_lengths), seed="dist")
    golds = {"recall": recall_pop, "state_easy": [i.gold for i in easy_dist], "state_hard": [i.gold for i in hard_dist]}

    # leakage samples (NB / recency / structural baselines) — moderate; include as-of-t hard
    recall_leak = recall_suite(world, oracle, n_leak, seed="leak")
    easy_leak = easy_suite(world, oracle, easy_lengths, n_leak // len(easy_lengths), seed="leak")
    hard_leak = hard_suite(world, oracle, hard_lengths, n_leak // len(hard_lengths), seed="leak")
    hard_asof = hard_suite(world, oracle, hard_lengths, n_leak // len(hard_lengths), seed="asof", as_of_t=True)
    leak_items = recall_leak + easy_leak + hard_leak + hard_asof

    rec = recency_accuracy(leak_items, world, renderer)
    nb = naive_bayes_accuracy(leak_items, renderer)
    ident = identity_baseline_accuracy(leak_items, world).get("state_hard")
    objblind = objblind_recency_accuracy(leak_items).get("state_easy")

    families = {}
    for fam, g in golds.items():
        space = answer_space(world, fam)
        k, n = len(space), len(g)
        kl = answer_kl(g, space)
        families[fam] = {
            "floor": random_floor(world, fam),
            "answer_space": k,
            "kl_excess": kl - (k - 1) / (2 * n),     # subtract finite-sample bias of KL-from-uniform
            "majority": Counter(g).most_common(1)[0][1] / n,
            "recency": rec.get(fam),
            "naive_bayes": nb.get(fam),
        }
    families["state_hard"]["identity_guess"] = ident      # structural no-op baseline
    families["state_easy"]["objblind_recency"] = objblind  # target-agnostic recency (foil caveat)

    oracle_ok = (
        all(i.gold == oracle.recall(i.entity, i.attribute) for i in recall_leak)
        and all(i.gold == oracle.hard_role(list(i.episode.events), i.target, i.t) for i in hard_leak + hard_asof)
        and all(i.gold == oracle.easy_holder(list(i.episode.events), i.target, i.t) for i in easy_leak)
    )
    fh = families["state_hard"]["floor"]
    checks = {
        "oracle_100": oracle_ok,
        "balanced_distributions": all(families[x]["kl_excess"] < KL_EXCESS_TAU for x in families),
        "majority_near_floor": all(families[x]["majority"] <= families[x]["floor"] + MAJ_MARGIN for x in families),
        "hard_state_no_shallow_leak": (families["state_hard"]["recency"] <= fh + LEAK_MARGIN
                                       and families["state_hard"]["naive_bayes"] <= fh + LEAK_MARGIN),
        "hard_state_no_structural_shortcut": ident <= fh + LEAK_MARGIN,
    }
    return {"families": families, "checks": checks, "passed": all(checks.values())}


# ---------------------------------------------------------------------------
# STRONG recency baseline over rendered task-suite prompts (tasks.Example lists).
# Registered like the other shallow baselines: a heuristic ADVERSARY scored against the oracle gold
# (never a label source). It reads the canonical renderer grammar directly ("sN gives oX to gY."
# events, "gY's a0 is vZ." facts) — the exact one-liner a lazy model could implement.
# ---------------------------------------------------------------------------
_GIVE_RE = re.compile(r"\bs\d+ gives (o\d+) to (g\d+)\.")
_FACT_RE = re.compile(r"\b(g\d+)'s a0 is (v\d+)\.")


def strong_recency_pred(prompt: str, family: str) -> str | None:
    """The strong recency heuristic's answer for one rendered prompt.

    binding:   the LAST give-event's recipient ("whoever was given something most recently").
    composite: that recipient plus his stated a0 fact (the full 2-token composite answer).
    Returns the answer in canonical rendered form (attached trailing period) or None when the
    prompt has no give events / the family has no recency structure to exploit.
    """
    gives = _GIVE_RE.findall(prompt)
    if not gives:
        return None
    holder = gives[-1][1]
    if family == "binding":
        return f"{holder}."
    if family == "composite":
        facts = dict(_FACT_RE.findall(prompt))
        value = facts.get(holder)
        return f"{holder} {value}." if value is not None else None
    return None


def strong_recency_accuracy(examples, family: str) -> float:
    """Accuracy of ``strong_recency_pred`` over a list of tasks.Example — near the random floor on a
    valid binding/composite task; well above it under the retired v1 sampler's end-clustered
    resolving write (the defect-documentation tests pin that contrast via tasks.RETIRED)."""
    return sum(strong_recency_pred(e.prompt, family) == e.answer for e in examples) / len(examples)


# ---------------------------------------------------------------------------
# Commutative-rung shallow adversaries (commutative_v1), in the strong_recency_pred idiom:
# regexes over the canonical rendered grammar ("g3's dial is at p2." initials, "sN turns g3's
# dial 2 clicks." events), scored against the oracle gold — heuristic ADVERSARIES, never a
# label source. These are the four one-liner cheats the task design must (and does) depress:
# each is gated <= 2x chance by scripts/validate_suite.py + tests/test_commutative_v1.py.
# Expected floors (Monte-Carlo n=200k, k=5, m=4, w_q>=2; chance 0.200):
#   initial_only      0.224@L4 -> 0.201@L16 (analytic 1/k + (1-1/k)(-1/(k-1))^w)
#   last_turn_only    0.106@L4 -> 0.199@L16+ (w=2 leaves ONE nonzero residual, never ≡ 0 mod 5)
#   entity_blind_sum  ~0.200 (m=4 active dials force per-entity filtering)
#   count_mod_k       ~0.194-0.200 (amounts vary over {1..4}, so event count is uninformative)
# ---------------------------------------------------------------------------
_TURN_RE = re.compile(r"\bs\d+ turns (g\d+)'s dial (\d+) clicks?\.")
_DIAL_RE = re.compile(r"\b(g\d+)'s dial is at p(\d+)\.")
_COMM_QUERY_RE = re.compile(r"what position is (g\d+)'s dial\?")


def comm_shallow_preds(prompt: str, k: int) -> dict[str, str | None]:
    """The four commutative shallow adversaries' answers for one rendered prompt.

    Returns {name: answer} in canonical rendered form (attached trailing period), or None
    values when the prompt has no parseable dial structure.
      initial_only     — the queried agent's STATED initial position (the no-op / identity cheat)
      last_turn_only   — initial + the LAST matching amount only (the last-event cheat)
      entity_blind_sum — initial + sum of ALL amounts mod k, no per-entity filtering
      count_mod_k      — initial + (#matching events) mod k, ignoring amounts
    """
    names = ("initial_only", "last_turn_only", "entity_blind_sum", "count_mod_k")
    m = _COMM_QUERY_RE.search(prompt)
    initials = {g: int(d) for g, d in _DIAL_RE.findall(prompt)}
    if not m or m.group(1) not in initials:
        return {n: None for n in names}
    target = m.group(1)
    p0 = initials[target]
    turns = [(g, int(a)) for g, a in _TURN_RE.findall(prompt)]
    mine = [a for g, a in turns if g == target]
    preds = {
        "initial_only": p0,
        "last_turn_only": (p0 + (mine[-1] if mine else 0)) % k,
        "entity_blind_sum": (p0 + sum(a for _g, a in turns)) % k,
        "count_mod_k": (p0 + len(mine)) % k,
    }
    return {n: f"p{v}." for n, v in preds.items()}


def comm_shallow_accuracy(examples, k: int) -> dict[str, float]:
    """Accuracy of each commutative shallow adversary over a list of tasks.Example."""
    hits: Counter = Counter()
    names = ("initial_only", "last_turn_only", "entity_blind_sum", "count_mod_k")
    for e in examples:
        preds = comm_shallow_preds(e.prompt, k)
        for name in names:
            hits[name] += int(preds[name] == e.answer)
    return {name: hits[name] / len(examples) for name in names}


# ---------------------------------------------------------------------------
# s5_chain shallow adversaries (chain / s5_chain families), in the strong_recency_pred idiom:
# regexes over the canonical rendered grammar ("g6's a0 is g2." initial-map facts, the
# "what is a0 of ... g6? (N hops)" query), scored against the oracle gold.
#
# INITIAL-MAP CHASE ignores every event and chases the stated initial a0 map `depth` times.
# It is the shallow policy the task's own structure hands a model — the facts state a complete
# map, the query is a pure dereference of *a* map, and only the events make it the wrong map.
# Its value depends on (k, depth, L): more events move the final map further from the initial
# one, and a bigger k leaves fewer chances of coincidence, so it is a per-cell number, not a
# constant.
#
# UNIFORM-OVER-NON-START is chance for a guesser that has learned only "never answer the agent
# you were asked about": distinct_path guarantees gold != start, so that guesser scores
# 1/(k-1), not 1/k. ECHO (answer the queried agent) is exactly 0 under distinct_path and is
# reported so that gate is visible rather than assumed.
#
# NO SINGLE ROW IS THE FLOOR. The chase is not uniformly dominant: uniform-over-non-start
# exceeds it on 7 of the 16 registered local cells (k6/d1/L8, k6/d2/L4, k6/d2/L8, k5/d1/L4,
# k5/d2/L4, k4/d1/L8, k4/d2/L4), and on two of those — k5/d2/L4 (0.185) and k6/d2/L8 (0.160) —
# it sits below even plain 1/k (0.200 and 0.167). A cell's operative floor is the MAX over the
# registered adversaries (``operative_floor``), and that is the only number a score may be
# read against.
#
# Reference values, measured on the s5_chain_local_v2 item streams at eval_n=200 — the exact
# item count a local cell scores. The rows are a property of that item set, so they move with
# n: the k6/d2/L4 chase is 0.195 at n=200 and 0.215 at n=5000.
#
#     cell        chase   unif_non_start   operative
#     k8/d1/L4    0.335   0.143            0.335  (chase)
#     k8/d2/L4    0.235   0.143            0.235  (chase)
#     k6/d1/L4    0.275   0.200            0.275  (chase)
#     k6/d2/L4    0.195   0.200            0.200  (uniform_non_start)
#     k6/d2/L8    0.160   0.200            0.200  (uniform_non_start)
#     k4/d1/L4    0.335   0.333            0.335  (chase)
#     k4/d1/L8    0.325   0.333            0.333  (uniform_non_start)
#
# The chase is a shallow policy only where an event stream exists. The `chain` family has no
# events, so the initial map IS the final map and the chase is the ORACLE: it measures 1.000.
# ``has_events=False`` drops the row rather than printing an oracle score as a floor.
# ---------------------------------------------------------------------------
_A0_FACT_RE = re.compile(r"\b(g\d+)'s a0 is ([a-z]+\d+)\.")
_A0_QUERY_RE = re.compile(r"what is ((?:a0 of )+)(g\d+)\?")


def s5_chain_shallow_preds(prompt: str) -> dict[str, str | None]:
    """The s5_chain shallow adversaries' answers for one rendered prompt.

    Returns {name: answer} in canonical rendered form (attached trailing period), or None
    values when the prompt carries no parseable initial map / chain query.
      initial_map_chase — chase the STATED INITIAL a0 map `depth` times (ignore every event)
      echo              — answer the queried agent itself
    """
    names = ("initial_map_chase", "echo")
    m = _A0_QUERY_RE.search(prompt)
    if m is None:
        return {n: None for n in names}
    depth = m.group(1).count("a0 of")
    start = m.group(2)
    nxt0 = dict(_A0_FACT_RE.findall(prompt))
    node = start
    for _ in range(depth):
        node = nxt0.get(node)
        if node is None:
            return {"initial_map_chase": None, "echo": f"{start}."}
    return {"initial_map_chase": f"{node}.", "echo": f"{start}."}


S5_CHAIN_ADVERSARIES = ("initial_map_chase", "echo", "uniform_non_start", "uniform")


def operative_floor(floors: dict[str, float]) -> float | None:
    """The number a cell has to clear: the max over whichever adversaries are registered.

    Reading a score against one named row understates the floor wherever that row is not the
    largest, which is most of the low-k end of the local grid.
    """
    vals = [v for name, v in floors.items() if name in S5_CHAIN_ADVERSARIES and v is not None]
    return max(vals) if vals else None


def s5_chain_floors(examples, k: int, has_events: bool = True) -> dict[str, float]:
    """Shallow-adversary floors for a list of s5_chain/chain ``tasks.Example``.

    Recomputed from the exact deterministic items a cell scores, so a floor row is a
    property of that cell rather than a global constant. Take the max over the returned
    rows (``operative_floor``) — which row is largest varies by cell.

      initial_map_chase  — accuracy of the initial-map chase. Omitted when ``has_events`` is
                           False: with no events to ignore the chase reproduces the oracle
                           and scores 1.000, which is a correctness check, not a floor.
      echo               — accuracy of answering the queried agent (0 under distinct_path)
      uniform_non_start  — expected accuracy of guessing uniformly over the answer space
                           minus the queried agent; chance for a gated stream. Falls back to
                           plain uniform where the answer is a different token type from the
                           query start (the typed-value ablation), since the start is then
                           not a candidate answer.
      uniform            — 1/k, the answer space with nothing excluded
    """
    from .render import classify

    n = len(examples)
    if not n:
        return {}
    hits = Counter()
    non_start = 0.0
    for e in examples:
        preds = s5_chain_shallow_preds(e.prompt)
        for name, pred in preds.items():
            hits[name] += int(pred is not None and pred == e.answer)
        gold = e.answer.strip().rstrip(".")
        start = e.meta.get("start")
        if start is None or classify(gold) != classify(start):
            non_start += 1.0 / k                    # start is not in the answer space
        elif gold != start:
            non_start += 1.0 / max(1, k - 1)
    out = {
        "initial_map_chase": hits["initial_map_chase"] / n,
        "echo": hits["echo"] / n,
        "uniform_non_start": non_start / n,
        "uniform": 1.0 / k,
    }
    if not has_events:
        del out["initial_map_chase"]
    return out


def _fmt(report: dict) -> str:
    lines = ["FactWorld — validity gate", "=" * 46, ""]
    lines.append(f"{'family':<12}{'floor':>8}{'KLexc':>8}{'major':>8}{'recency':>9}{'n.bayes':>9}")
    for fam, m in report["families"].items():
        lines.append(f"{fam:<12}{m['floor']:>8.3f}{m['kl_excess']:>8.3f}"
                     f"{m['majority']:>8.3f}{m['recency']:>9.3f}{m['naive_bayes']:>9.3f}")
    lines.append("")
    lines.append(f"  hard identity-guess baseline : {report['families']['state_hard']['identity_guess']:.3f}  (floor 0.200)")
    lines.append(f"  easy object-blind recency    : {report['families']['state_easy']['objblind_recency']:.3f}  (floor 0.048)")
    lines.append("")
    for name, ok in report["checks"].items():
        lines.append(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    lines.append("")
    lines.append(f"GATE: {'PASSED' if report['passed'] else 'FAILED'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_fmt(run_gate()))
