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

Each task family's shallow adversaries live here too, one block per family, all in the same
idiom: regexes over the canonical rendered grammar, scored against the oracle gold, recomputed
from the exact items a cell scores. A cell's OPERATIVE floor is the max over the family's
registered rows (``operative_floor``, ``s5_bind_operative_floor``) and is the only number a
score may be read against — never 1/k, and never one named row.

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
# THE FIXED-OFFSET PARTITION fixes what any initial-map row can mean, so it comes first. The
# stated initial map f_0 is a single k-cycle (tasks._ex_s5_chain) and distinct_path forces
# gold != start, so the k-1 fixed-offset policies f_0^1(start) ... f_0^{k-1}(start) enumerate
# the k-1 non-start agents exactly once each: on every item exactly one of them is right, and
# their accuracies sum to exactly 1.0000 (measured at n=5000 on the scored stream at
# L=32/64/96/128, and at n=200 on all sixteen registered local cells). Three consequences:
#   - every member's null expectation is 1/(k-1) — `uniform_non_start`, an already-registered
#     row — and NOT 1/k;
#   - the MAX over the family is a SELECTION statistic. At k=32, n=5000 the expected family
#     max is 0.0376 (1.17x the 0.0323 per-member null) and its 95th percentile 0.0398 (1.23x),
#     so a member reading ~1.2x "chance" is what the partition produces on its own;
#   - which member is largest moves with the sample. On the scored stream the family max sits
#     at j=13 (0.0382) at L=64, j=17 (0.0382) at L=96 and j=17/23 (0.0370) at L=128; on the
#     local k6/d2 cell at j=+1/+5 (0.230, tied) at L=4 and j=+4 (0.235) at L=8.
# THE RULE THAT FOLLOWS: a member of this family sets a floor only where it beats
# `uniform_non_start`. It is enforced by construction rather than by a threshold —
# uniform_non_start is itself a registered row and ``operative_floor`` is a max, so a member
# below it contributes nothing — and it is why registering further members would price in no
# additional shortcut: it would only raise the floor by the family's selection spread.
#
# INITIAL-MAP CHASE ignores every event and chases the stated initial a0 map `depth` times. It
# is the j=depth member, registered because the query names it rather than because it measured
# high: it is the query's own computation run against the stated map instead of the final one
# — the facts state a complete map, the query is a pure dereference of *a* map, and only the
# events make it the wrong map. Its value depends on (k, depth, L): more events move the final
# map further from the initial one, and a bigger k leaves fewer chances of coincidence, so it
# is a per-cell number, not a constant. Where it is elevated it is elevated by that mechanism
# and by a wide margin (0.335 against a 0.143 null at k8/d1/L4); where the stream has moved
# the map it falls to the partition's own level or below — at k5/d2/L4 (0.185) and k6/d2/L8
# (0.160) it sits under even plain 1/k.
#
# INITIAL-MAP BACKHOP reads the same stated map in the other direction: f_0^{-1}(start) =
# f_0^{k-1}(start), one lookup, no event and no forward walk. It is MEASURED AND REPORTED as a
# diagnostic but is NOT a registered adversary, so it never sets a floor. j=k-1 is named by
# nothing in the task — it is one of the k-1 exchangeable offsets, singled out by its measured
# value — and that value does not survive replication: on the scored stream at L=32 the split
# reads 0.0402 (1.25x the 0.0323 null), while eight independent streams of the same
# construction (n=2000 each; spec.name, hence the item rng namespace, is all that differs)
# pool to 0.0368 (1.14x). The residue is real — the distinct_path gate keeps starts whose
# FINAL-map cycle exceeds the depth, so on a short stream, where the final map is still near
# the stated one, the stated predecessor carries some information about the answer — but
# 0.0368 sits below the initial-ref row's 0.0398 at that same length, so the backhop does not
# set the L=32 floor on its own account, and registering the split's 0.0402 would price its
# sampling excess into every score at that length. From L=64 on the row is at or under
# uniform_non_start (0.0334 / 0.0304 / 0.0290 against 0.0323) and there is nothing to register.
#
# UNIFORM-OVER-NON-START is chance for a guesser that has learned only "never answer the agent
# you were asked about": distinct_path guarantees gold != start, so that guesser scores
# 1/(k-1), not 1/k. It is also the fixed-offset partition's common expectation, so it is that
# family's floor contribution wherever no member beats it. ECHO (answer the queried agent) is
# exactly 0 under distinct_path and is reported so that gate is visible rather than assumed.
#
# NO SINGLE ROW IS THE FLOOR. The chase is not uniformly dominant: over the 16 registered local
# cells it is the largest registered row on 9 and uniform-over-non-start on 7. A cell's
# operative floor is the MAX over the registered adversaries (``operative_floor``), and that is
# the only number a score may be read against.
#
# Reference values, measured on the s5_chain_local_v2 item streams at eval_n=200 — the exact
# item count a local cell scores. The rows are a property of that item set, so they move with
# n: the k6/d2/L4 chase is 0.195 at n=200 and 0.215 at n=5000. The backhop column is the
# unregistered diagnostic, shown in parentheses so the partition's spread is visible beside the
# floor it does not set.
#
#     cell        chase  (backhop)  unif_non_start   operative
#     k8/d1/L4    0.335   0.130     0.143            0.335  (chase)
#     k8/d2/L4    0.235   0.090     0.143            0.235  (chase)
#     k6/d1/L4    0.275   0.215     0.200            0.275  (chase)
#     k6/d2/L4    0.195   0.230     0.200            0.200  (uniform_non_start)
#     k6/d2/L8    0.160   0.230     0.200            0.200  (uniform_non_start)
#     k4/d1/L4    0.335   0.320     0.333            0.335  (chase)
#     k4/d1/L8    0.325   0.330     0.333            0.333  (uniform_non_start)
#     k4/d2/L4    0.325   0.355     0.333            0.333  (uniform_non_start)
#
# BOTH INITIAL-MAP ROWS are defined only where an event stream exists. The `chain` family has
# no events, so the stated map IS the final map: the chase becomes the ORACLE and measures
# 1.000, and the backhop measures whether depth == k-1, in which case one reverse lookup is
# likewise exact (chain_v2 scores 1.000 at its depth-5 eval length and 0.000 at depth 4). The
# second is the cheap-BACKWARD-direction property the chain staircase prices by running each
# depth at k=2d+1; neither is a floor, so ``has_events=False`` drops both rows rather than
# printing an oracle score as one.
#
# INITIAL-REF RESOLUTION is the adversary that the state-referencing events
# (TaskSpec.conditional_rate) create, and it is the cheap algorithm those events exist to
# break. An event that names its operand as "the agent whose a0 is currently gX" has no
# identity until the map has been evaluated forward to it; resolving every such reference
# against the STATED INITIAL map instead recovers a complete, ordinary event list up front,
# after which the whole task is answerable by pushing one symbol backward through it —
# log2(k) bits of carried state, no map. The policy is exact on every unconditional event and
# on any reference the drifting map has not yet invalidated, so it is not a guess: it is the
# v3 algorithm applied to a v4 stream, and it has to sit at chance for the construct to be
# doing anything.
#
# Its row is dropped only where NO item carries a reference (a stream without them makes this
# policy the oracle, exactly as the chase is the oracle without events). Where SOME items
# carry none, the row is scored over all of them and reads high — correctly: a cheap policy
# really does answer a reference-free item, so a rate that leaves items reference-free raises
# the cell's floor rather than being free difficulty.
# ---------------------------------------------------------------------------
_A0_FACT_RE = re.compile(r"\b(g\d+)'s a0 is ([a-z]+\d+)\.")
_A0_QUERY_RE = re.compile(r"what is ((?:a0 of )+)(g\d+)\?")

# The pointer-map event grammar, canonical and compact (tasks.TaskSpec.compact_events), as the
# adversary reads it. Scanned by position so the event ORDER is the rendered order.
_A0_EVENT_FORMS = (
    ("swap",  re.compile(r"swaps the values of (g\d+)'s a0 and (g\d+)'s a0\.")),
    ("ref",   re.compile(r"swaps the values of (g\d+)'s a0 and the a0 of the agent "
                         r"whose a0 is currently (g\d+)\.")),
    ("cycle", re.compile(r"cycles a0 simultaneously: (g\d+)'s a0 takes (g\d+)'s old a0, "
                         r"g\d+'s a0 takes (g\d+)'s old a0, and g\d+'s a0 takes g\d+'s old a0\.")),
    ("swap",  re.compile(r"swaps a0: (g\d+) and (g\d+)\.")),
    ("ref",   re.compile(r"swaps a0: (g\d+) and whose a0 is (g\d+)\.")),
    ("cycle", re.compile(r"cycles a0: (g\d+) -> (g\d+) -> (g\d+)\.")),
)


def a0_events(prompt: str) -> list[tuple[str, tuple[str, ...]]]:
    """The pointer-map events of one rendered prompt, in stream order.

    ``("swap", (a, b))`` / ``("cycle", (a, b, c))`` name their operands outright;
    ``("ref", (a, v))`` names the second operand by the value it currently holds, so the
    slot it refers to is recoverable only from the map at that point in the stream.
    """
    found = []
    for kind, rx in _A0_EVENT_FORMS:
        for m in rx.finditer(prompt):
            found.append((m.start(), kind, m.groups()))
    found.sort()
    return [(kind, args) for _pos, kind, args in found]


def s5_chain_shallow_preds(prompt: str) -> dict[str, str | None]:
    """The s5_chain shallow adversaries' answers for one rendered prompt.

    Returns {name: answer} in canonical rendered form (attached trailing period), or None
    values when the prompt carries no parseable initial map / chain query.
      initial_map_chase   — chase the STATED INITIAL a0 map `depth` times (ignore every event)
      initial_map_backhop — ONE hop BACKWARD through the stated initial map, f_0^{-1}(start)
      echo                — answer the queried agent itself
    """
    names = ("initial_map_chase", "initial_map_backhop", "echo")
    m = _A0_QUERY_RE.search(prompt)
    if m is None:
        return {n: None for n in names}
    depth = m.group(1).count("a0 of")
    start = m.group(2)
    nxt0 = dict(_A0_FACT_RE.findall(prompt))
    inv0 = {v: a for a, v in nxt0.items()}
    back = inv0.get(start)                  # absent when values are a different type (typed arm)
    out: dict[str, str | None] = {
        "echo": f"{start}.",
        "initial_map_backhop": None if back is None else f"{back}.",
    }
    node = start
    for _ in range(depth):
        node = nxt0.get(node)
        if node is None:
            out["initial_map_chase"] = None
            return out
    out["initial_map_chase"] = f"{node}."
    return out


def s5_chain_offset_accuracies(examples, k: int) -> dict[int, float]:
    """Accuracy of every fixed-offset policy f_0^j(start), j = 1..k-1, over a list of items.

    The family the two initial-map rows belong to, measured so the partition is checkable
    rather than asserted. The stated map is a single k-cycle and distinct_path forces
    gold != start, so the k-1 offsets hit each non-start agent exactly once per item: exactly
    one is right on every item, the returned values sum to exactly 1.0, every member's null is
    1/(k-1) (``uniform_non_start``), and the max over any subset of them is a selection
    statistic rather than a shortcut.

    Returns {} where the stated map is not a permutation of the query's own token type (the
    typed-value ablation maps agents to roles, so the walk leaves the map after one hop).
    """
    hits: Counter = Counter()
    for e in examples:
        m = _A0_QUERY_RE.search(e.prompt)
        if m is None:
            return {}
        nxt0 = dict(_A0_FACT_RE.findall(e.prompt))
        gold = e.answer.strip().rstrip(".")
        node = m.group(2)
        for j in range(1, k):
            node = nxt0.get(node)
            if node is None:
                return {}
            hits[j] += int(node == gold)
    n = len(examples)
    return {j: hits[j] / n for j in range(1, k)} if n else {}


def s5_chain_ref_pred(prompt: str) -> str | None:
    """The initial-ref-resolution adversary's answer for one rendered prompt.

    Resolves every "the agent whose a0 is currently gX" against the STATED INITIAL map
    (f_0^{-1}(gX)) rather than the running one, then plays the stream out exactly and
    dereferences the resulting final map ``depth`` times. Returns None when the prompt
    carries no reference event — the policy is then the oracle, not a floor.
    """
    m = _A0_QUERY_RE.search(prompt)
    if m is None:
        return None
    events = a0_events(prompt)
    if not any(kind == "ref" for kind, _ in events):
        return None
    nxt = dict(_A0_FACT_RE.findall(prompt))
    inv0 = {v: a for a, v in nxt.items()}          # the map as STATED, inverted once, up front
    for kind, args in events:
        if kind == "ref":
            a, b = args[0], inv0.get(args[1])
            if b is None or a not in nxt or b not in nxt:
                return None
            nxt[a], nxt[b] = nxt[b], nxt[a]
        elif kind == "swap":
            a, b = args
            if a not in nxt or b not in nxt:
                return None
            nxt[a], nxt[b] = nxt[b], nxt[a]
        else:
            a, b, c = args
            if a not in nxt or b not in nxt or c not in nxt:
                return None
            nxt[a], nxt[b], nxt[c] = nxt[b], nxt[c], nxt[a]
    node = m.group(2)
    for _ in range(m.group(1).count("a0 of")):
        node = nxt.get(node)
        if node is None:
            return None
    return f"{node}."


# Every row ``s5_chain_floors`` can emit, in report order.
S5_CHAIN_ROWS = ("initial_map_chase", "initial_map_backhop", "initial_ref_resolution",
                 "echo", "uniform_non_start", "uniform")
# The rows that may SET a cell's floor. initial_map_backhop is measured and reported but is not
# here: it is an unnamed member of the fixed-offset partition documented above, so the max over
# it is a selection statistic and its floor contribution is already carried by uniform_non_start.
S5_CHAIN_ADVERSARIES = ("initial_map_chase", "initial_ref_resolution",
                        "echo", "uniform_non_start", "uniform")
# Chance, not shortcuts: what a shortcut has to BEAT. Removed wherever the question is "does a
# cheap policy do better than guessing" rather than "what must a score clear".
S5_CHAIN_CHANCE_ROWS = ("uniform_non_start", "uniform")
# The members of the fixed-offset family f_0^j(start) that carry a name of their own.
S5_CHAIN_OFFSET_ROWS = ("initial_map_chase", "initial_map_backhop")


def registered_for(floors: dict[str, float]):
    """The registered adversary set belonging to the family whose rows these are.

    Dispatch is on the rows only the family can emit, never on the shared ones: the two
    families both emit ``uniform``, so a set-containment test would call a one-row dict either
    family's. A dict whose rows name no family, or both, raises rather than resolving — the
    failure this replaces returned a number.
    """
    rows = set(floors)
    bind = rows & (set(S5_BIND_ROWS) - set(S5_CHAIN_ROWS))
    chain = rows & (set(S5_CHAIN_ROWS) - set(S5_BIND_ROWS))
    if bind and not chain:
        return S5_BIND_ADVERSARIES
    if chain and not bind:
        return S5_CHAIN_ADVERSARIES
    raise ValueError(
        f"cannot tell which family these floor rows belong to: {sorted(rows)}. Pass the "
        f"family's registered set explicitly (S5_CHAIN_ADVERSARIES / S5_BIND_ADVERSARIES).")


def operative_floor(floors: dict[str, float], registered=None) -> float | None:
    """The number a cell has to clear: the max over whichever adversaries are registered.

    Reading a score against one named row understates the floor wherever that row is not the
    largest, which is most of the low-k end of the local grid.

    Every family's chance row is registered, so this max can never fall below the family's own
    chance level — 1/(k-1) over the fixed-offset partition for the pointer-map family, and the
    same quantity as informed chance for the mutual-reference one. A row that does not beat
    chance therefore contributes nothing, which is what keeps the floor from tracking a
    family's selection noise. Rows outside the registered set (currently
    ``initial_map_backhop``) are reported for inspection and are ignored here.

    ``registered`` selects the family's registered set. It defaults to None, which resolves the
    family from the rows (``registered_for``): the two families share the row name ``uniform``
    and nothing else, so a fixed default silently reported CHANCE — a floor below the family's
    own informed chance — for every mutual-reference cell handed to it.
    """
    if not floors:
        return None                                      # no cell, no floor — as before
    if registered is None:
        registered = registered_for(floors)
    vals = [v for name, v in floors.items() if name in registered and v is not None]
    return max(vals) if vals else None


def s5_chain_floors(examples, k: int, has_events: bool = True) -> dict[str, float]:
    """Shallow-adversary floors for a list of s5_chain/chain ``tasks.Example``.

    Recomputed from the exact deterministic items a cell scores, so a floor row is a
    property of that cell rather than a global constant. Take the max over the REGISTERED
    rows (``operative_floor``, S5_CHAIN_ADVERSARIES) — which row is largest varies by cell,
    and not every returned row is registered.

      initial_map_chase  — accuracy of the initial-map chase, the j=depth member of the
                           fixed-offset partition. Omitted when ``has_events`` is False:
                           with no events to ignore the chase reproduces the oracle and
                           scores 1.000, which is a correctness check, not a floor.
      initial_map_backhop — accuracy of one backward hop through the stated initial map, the
                           j=k-1 member. REPORTED BUT NOT REGISTERED: the k-1 offsets
                           partition the answer, so this row's null is uniform_non_start and
                           the max over an arbitrary subset of them measures selection (see
                           the module comment). Omitted with ``has_events`` False for a
                           second reason: the stated map is then the final one and the row
                           reads 1.000 or 0.000 on whether depth == k-1, a property of the
                           query.
      initial_ref_resolution — accuracy of playing the stream out exactly with every state
                           reference resolved against the stated initial map. Omitted where
                           no item carries a reference event, for the same reason.
      echo               — accuracy of answering the queried agent (0 under distinct_path)
      uniform_non_start  — expected accuracy of guessing uniformly over the answer space
                           minus the queried agent; chance for a gated stream, and the common
                           expectation of every fixed-offset policy. Falls back to plain
                           uniform where the answer is a different token type from the query
                           start (the typed-value ablation), since the start is then not a
                           candidate answer.
      uniform            — 1/k, the answer space with nothing excluded
    """
    from .render import classify

    n = len(examples)
    if not n:
        return {}
    hits = Counter()
    non_start = 0.0
    has_refs = False
    for e in examples:
        preds = s5_chain_shallow_preds(e.prompt)
        for name, pred in preds.items():
            hits[name] += int(pred is not None and pred == e.answer)
        ref = s5_chain_ref_pred(e.prompt)
        has_refs |= ref is not None
        # A reference-free item has nothing for this policy to mis-resolve, so it answers it
        # exactly; counting that is what makes such items read as the floor they are.
        hits["initial_ref_resolution"] += 1 if ref is None else int(ref == e.answer)
        gold = e.answer.strip().rstrip(".")
        start = e.meta.get("start")
        if start is None or classify(gold) != classify(start):
            non_start += 1.0 / k                    # start is not in the answer space
        elif gold != start:
            non_start += 1.0 / max(1, k - 1)
    out = {
        "initial_map_chase": hits["initial_map_chase"] / n,
        "initial_map_backhop": hits["initial_map_backhop"] / n,
        "initial_ref_resolution": hits["initial_ref_resolution"] / n,
        "echo": hits["echo"] / n,
        "uniform_non_start": non_start / n,
        "uniform": 1.0 / k,
    }
    if not has_events:
        del out["initial_map_chase"]
        del out["initial_map_backhop"]
    if not has_refs:
        del out["initial_ref_resolution"]
    return out


# ---------------------------------------------------------------------------
# s5_bind shallow adversaries (the mutual-reference family), in the strong_recency_pred idiom:
# regexes over the canonical rendered grammar, scored against the oracle gold. Every policy
# below sees exactly what a model sees — the two stated maps, the event sentences in order, and
# each event's temporal phrase — and nothing from meta.
#
# WHAT THE FAMILY IS FOR. The construct's claim is that the coupled rendering forces a forward
# pass carrying both maps. Each registered row is a named way of NOT doing that, so the max over
# them is the number a score has to clear.
#
# COUPLING-BLIND (defined only where some event is rendered "at this point"; under a fully
# decoupled rendering each of these IS the oracle, so the row is dropped rather than printed as
# a floor):
#   stale_resolution  — resolve every reference against the STATED maps. This is the decoupled
#                       algorithm run on a coupled item, i.e. the best policy available to a
#                       model that tracks neither structure into the other, and 1 minus its
#                       accuracy is how often the coupling changes the answer at all. If it is
#                       high the composed cell is secretly the component cell.
#   one_leg_B         — feed B into P but never P into B (swap references read the running
#                       holder map; give recipients are resolved statically). Half the coupling.
#   one_leg_P         — the mirror: feed P into B but never B into P.
#   final_state_resolution — compute the true final maps, then replay the stream resolving every
#                       reference against THEM. A wrong-TIME policy, not a shallow one: it is
#                       what a model that forms one state and applies it everywhere would do.
#   pin_chain         — the state-free RESET channel the two reference forms compose into. A
#                       dynamic give, "give o to the agent whose role at this point is r",
#                       writes B[o] <- Pinv[r], so from that event until o is written again its
#                       holder is PINNED: whoever holds o has role r. A later dynamic swap,
#                       "swap the roles of a and the holder of o at this point", then sets
#                       P[a] <- r exactly, because selecting an agent by its role and reading
#                       that role back returns the role. The two references cancel the state,
#                       so the answer to a state query is two retrievals from the surface — the
#                       last swap naming the queried agent, then the last dynamic give to the
#                       object that swap names — and no map is carried at all. The bind query
#                       has the mirror form at three retrievals. Registering it is what makes
#                       the floor honest: a zero-state policy reading well above chance means
#                       the cell is not measuring composition on those items, and the row does
#                       not decay with length because the channel is length-free.
#
# TRUNCATION (window_50/75/90 and prefix_50/75/90): simulate the task EXACTLY, honouring every
# temporal phrase, over T = f*L of its L events — the LAST T (window, the stated maps carried in
# at L-T) or the FIRST T (prefix, the true maps read out at T). They are two positions of ONE
# family — drop a block of width (1-f)L at position p, play everything else — which is
# continuous in (position, width), non-monotone in both, and whose accuracy peaks in the
# INTERIOR and at the NARROW end, which is where none of the six registered budgets sits.
# Scanned over 19 positions x 8 widths at n=1500 on the un-gated stream, the best member reads
# 4.73 / 3.15 / 2.36x the informed chance at k=12, L=128/192/256 (operative floors 0.097 /
# 0.099 / 0.096), 2.74 / 2.19x at k=6, L=48/64, and 5.08 / 3.95x at k=16 — in every cell a
# width-0.05L block at position 0.85-0.90, while the registered endpoints sat within 1.2x of
# chance. A max over a finite subset of an exchangeable continuum is a selection statistic, so
# no set of registered members could ever have defined this family's floor, and three rounds of
# trying it each lost to a neighbour of the member registered.
#
# THEY ARE NOT FLOOR ROWS. Every member carries both maps across its budget, so W = 2k + m and
# the row sits in the TASK's own resource class — see S5_BIND_MAP_CARRYING_ROWS. They stay
# measured and printed: the gap between f=0.5 and f=0.9 is how much of the stream is
# load-bearing, at each end separately, and the two ends answer different gates. A window is
# beaten by making the stream's HEAD load-bearing (the queried object's resolving write is gated
# away from the first decile); a prefix by making its TAIL load-bearing, which is what
# TaskSpec.q_tail gates on the queried agent's last carrier event, and before that gate existed
# prefix_90 read 0.45/0.37/0.29 at k=12/L=128/192/256 against a 0.098-0.117 operative floor.
#
# NO SAMPLER GATE STANDS BEHIND THE EXCLUSION, and none is needed. Four rounds were lost trying
# to gate the family member by member; what closes it is the cost argument, applied to the whole
# continuum at once. See the s5_bind_v3 block below for the rule in its final form.
#
# ON A DECOUPLED RENDERING they are not shortcuts at all: the state component costs a sparse
# backward walk over one live symbol and the retrieval component costs three retrievals, so a
# truncated pass is an order of magnitude MORE expensive than the task. On the decoupled bind
# arm both halves read ~1.000 by doing the work. The class rule reaches that case for free — a
# map-carrying row is excluded on both renderings — which is why it replaced the earlier
# per-rendering exemption.
#
# ONE-HOP AND STATED:
#   initial_only      — answer the stated initial role / holder (the no-op policy).
#   last_swap_1hop    — the stated role of the other operand of the last swap naming the queried
#                       agent, resolved off the stated holder map (state query only).
#
# CHANCE, not shortcuts: uniform_non_initial = 1/(k-1), since the query gates force the answer
# to differ from the stated one, and uniform = 1/k. For the whole-map readout the answer is a
# permutation of the k roles, so its chance row is 1/k! and there is no non-initial variant.
#   uniform_anti_pin  — the same guess with pin_chain's answer struck out as well. no_pin makes
#                       that answer an ANTI-predictor rather than a neutral one: the sampler
#                       rejects exactly the swap on which the pin chain would have been right, so
#                       the row reads below chance and a guesser who knows which answer it names
#                       is choosing uniformly over k-2 roles that carry more than (k-2)/(k-1) of
#                       the probability. Computed in closed form per item — [gold survives the
#                       exclusion] / (k - |exclusion|), no guessing draw — so it carries no
#                       sampling noise of its own, which is what makes a 1.05-1.13x reading a
#                       property of the stream rather than an artifact of n. It is CHANCE, not a
#                       shortcut: it buys its edge from the generator's rejection rule and not
#                       from the item, so it belongs in the number a score is read against and
#                       not in the suite gate, exactly like the other two uniform rows.
#                       ONE exclusion, not a search over them. Striking any other row's answer
#                       as well is a different member of the same family and the family is not
#                       monotone: measured at n=3000 with the gates on, striking pin_chain reads
#                       1.03-1.13x chance, striking pin_chain AND prefix_90 reads 1.03-1.16x
#                       (better at k=6, WORSE at k=12/L=256, where prefix_90 is back at chance
#                       and the exclusion costs more denominator than it buys), and striking
#                       every registered row reads 0.55-1.05x. A max over subsets would
#                       therefore measure selection. What makes the pin exclusion registerable
#                       is that a sampler REJECTION rule stands behind it — no_pin refuses the
#                       event on which pin_chain would have been right — rather than a measured
#                       accuracy on this sample.
#
# min(component, control) IS A CEILING, NOT A NULL: none of these rows is a component score, and
# a component or capacity-control accuracy never belongs in this max.
# ---------------------------------------------------------------------------
_SB_ROLE_RE = re.compile(r"\b(g\d+) has role (r\d+) at the start\.")
_SB_HOLD_RE = re.compile(r"\b(g\d+) holds (o\d+) at the start\.")
_SB_SWAP_RE = re.compile(r"\bs\d+ swaps the roles of (g\d+) and the agent who holds (o\d+) "
                         r"at (this point|the start)\.")
_SB_GIVE_RE = re.compile(r"\bs\d+ gives (o\d+) to the agent whose role at (this point|the start) "
                         r"is (r\d+)\.")
_SB_Q_STATE_RE = re.compile(r"what role does (g\d+) have at the end\?")
_SB_Q_BIND_RE = re.compile(r"who is the holder of (o\d+) at the end\?")
_SB_Q_ALL_RE = re.compile(r"what role does each of ((?:g\d+, )+g\d+) have at the end\?")

# Every row ``s5_bind_floors`` can emit, in report order.
S5_BIND_ROWS = ("stale_resolution", "one_leg_B", "one_leg_P", "final_state_resolution",
                "pin_chain", "window_90", "window_75", "window_50",
                "prefix_90", "prefix_75", "prefix_50",
                "initial_only", "last_swap_1hop",
                "uniform_non_initial", "uniform_anti_pin", "uniform")
# ---------------------------------------------------------------------------
# THE RESOURCE CLASS. Which rows may set a floor is decided by what a row COSTS, not by how
# accurate it is and not by a per-item step threshold, which the block-drop family defeats: its
# members cost ~0.9x the task, so any threshold that admits one admits a continuum.
#
# On the step-counted register machine the family is measured on (content-addressed retrieval of
# a stated fact or an event record: one step; resolving an operand against a carried map: one
# step; each entry written to a carried map: one step), write W for the number of symbol
# registers a policy holds simultaneously and S for its steps. The composed cell's cheapest
# correct algorithm is a forward pass carrying P, its inverse and B: W = 2k + m and S = Theta(L).
#
#   A ROW MAY SET A FLOOR IFF W = O(1) IN k — it carries NO structure-sized state.
#
# That is the separation the construct claims: the coupled rendering forces a pass carrying both
# maps, so a policy that carries neither is a genuine shortcut however many events it reads,
# and a policy that carries both is doing the task's own work at a constant-factor discount.
# Classifying on W and not on S is what makes the rule survive the family: every block-drop, at
# every (position, width) with width < L, has W = 2k + m, so the whole continuum is excluded by
# one rule rather than member by member.
#
# WHAT THIS EXCLUDES. Near-oracle policies — play 0.99L events, carry both maps — which is
# right: they are the task, discounted. What the excluded continuum reads is measured and
# reported rather than folded into the floor.
#
# The rows that carry a map. final_state_resolution needs the true final maps before it can
# start, so it costs MORE than the task and could never have been a floor; the truncation rows
# each carry both maps across their budget. Both stay measured and printed, as diagnostics: the
# first is how a model that forms one state and applies it everywhere fails, and the gap between
# the truncation budgets is how much of the stream is load-bearing at each end.
S5_BIND_MAP_CARRYING_ROWS = ("final_state_resolution",) + tuple(
    r for r in S5_BIND_ROWS if r.startswith(("window_", "prefix_")))
# Live slots per row, as a function of k and m — the classification above, made checkable.
# The admitted rows walk one carrier (plus a scratch register) or do a bounded number of
# retrievals; nothing in them grows with k.
S5_BIND_ROW_SLOTS = {
    "stale_resolution": 2,        # backward carrier walk; both operands are stated facts
    "one_leg_B": 2,               # ditto, B read by content from the last give to the object
    "one_leg_P": 2,               # ditto, the mirror
    "pin_chain": 3,               # two surface retrievals for state, three for bind
    "initial_only": 1,
    "last_swap_1hop": 2,
    "uniform_non_initial": 1, "uniform_anti_pin": 2, "uniform": 1,
}
# The rows that may SET a cell's floor: every row that carries no map. Each is a named policy
# rather than a member of an exchangeable family, so the max over them is not a selection
# statistic — which is the second thing a floor row has to be.
S5_BIND_ADVERSARIES = tuple(r for r in S5_BIND_ROWS if r not in S5_BIND_MAP_CARRYING_ROWS)
S5_BIND_CHANCE_ROWS = ("uniform_non_initial", "uniform_anti_pin", "uniform")
# The two named positions of the block-drop family, at matched budgets. Derived from the row
# names so that measuring a further cut, at either end, reaches every consumer without an edit
# there — and every one of them lands in S5_BIND_MAP_CARRYING_ROWS by the same derivation, so a
# new cut is a diagnostic on arrival and cannot become a floor by being added.
S5_BIND_TRUNCATION_ROWS = tuple(r for r in S5_BIND_ROWS
                                if r.startswith(("window_", "prefix_")))
# Rows defined only where some event is rendered "at this point". Under a fully decoupled
# rendering each of them reproduces the oracle on the query it is defined for — the first four
# by resolving against the stated maps, which IS the decoupled semantics, and pin_chain because
# a static give's recipient is the stated holder of the named role, i.e. exactly the retrieval
# component's answer. Printing 1.000 as a floor would be a correctness check wearing a floor's
# clothes, so they are dropped rather than reported.
S5_BIND_COUPLED_ONLY_ROWS = ("stale_resolution", "one_leg_B", "one_leg_P",
                             "final_state_resolution", "pin_chain")
# The registered truncation budgets, as fractions of the stream length. Both halves are read at
# every one of them: window_f keeps the last f*L events, prefix_f the first f*L.
S5_BIND_WINDOWS = (0.9, 0.75, 0.5)


def s5_bind_read(prompt: str) -> dict | None:
    """The stated maps, the event stream in rendered order, and the query — read off one
    rendered prompt exactly as a model sees it.

    Returns None when the prompt is not a mutual-reference item. Each event is
    ``("swap", agent, obj, dynamic)`` or ``("give", obj, role, dynamic)``, where ``dynamic``
    is whether the sentence said "at this point" (resolve against the running map) rather than
    "at the start" (resolve against the stated one).
    """
    P0 = dict(_SB_ROLE_RE.findall(prompt))
    B0 = {o: g for g, o in _SB_HOLD_RE.findall(prompt)}
    if not P0 or not B0:
        return None
    found = []
    for m in _SB_SWAP_RE.finditer(prompt):
        found.append((m.start(), ("swap", m.group(1), m.group(2), m.group(3) == "this point")))
    for m in _SB_GIVE_RE.finditer(prompt):
        found.append((m.start(), ("give", m.group(1), m.group(3), m.group(2) == "this point")))
    found.sort()
    events = [e for _pos, e in found]
    m_all = _SB_Q_ALL_RE.search(prompt)
    if m_all is not None:
        query = ("state_all", tuple(m_all.group(1).split(", ")))
    else:
        m_state = _SB_Q_STATE_RE.search(prompt)
        m_bind = _SB_Q_BIND_RE.search(prompt)
        if m_state is not None:
            query = ("state", m_state.group(1))
        elif m_bind is not None:
            query = ("bind", m_bind.group(1))
        else:
            return None
    return {"P0": P0, "B0": B0, "events": events, "query": query}


def _sb_run(read: dict, mode: str = "surface", start: int = 0, end: int | None = None,
            final=None, drop: tuple[int, int] | None = None):
    """Play ``events[start:end]`` from the stated maps and return the resulting (P, B).

    ``start`` drops a PREFIX of the stream (the window policies, which carry the stated maps in
    at the cut); ``end`` drops a SUFFIX (the prefix policies, which are exact up to the cut and
    read the true maps out there). Both cost the events they play. ``drop=(lo, hi)`` skips an
    INTERIOR block instead and plays everything else, which is the general member of the family
    those two are the endpoints of.

    mode 'surface' honours each event's rendered temporal phrase (the exact semantics);
    'stale' resolves every reference against the stated maps; 'B_only' feeds B into P but not
    P into B; 'P_only' the mirror; 'final' resolves dynamic references against ``final``, the
    true final maps.
    """
    P0, B0 = read["P0"], read["B0"]
    P, B = dict(P0), dict(B0)
    P0inv = {v: k for k, v in P0.items()}
    Pinv = dict(P0inv)
    for i, (kind, x, y, dyn) in enumerate(read["events"][start:end], start):
        if drop is not None and drop[0] <= i < drop[1]:
            continue
        if kind == "swap":
            if mode == "stale" or (mode == "P_only") or not dyn:
                b = B0.get(y)
            elif mode == "final":
                b = final[1].get(y)
            else:                                    # 'surface' / 'B_only'
                b = B.get(y)
            if b is None or x not in P or b not in P:
                return None
            P[x], P[b] = P[b], P[x]
            Pinv = {v: k for k, v in P.items()}
        else:
            if mode == "stale" or (mode == "B_only") or not dyn:
                h = P0inv.get(y)
            elif mode == "final":
                h = final[0].get(y)
            else:                                    # 'surface' / 'P_only'
                h = Pinv.get(y)
            if h is None:
                return None
            B[x] = h
    return P, B


def _sb_answer(read: dict, maps) -> str | None:
    """The rendered answer a policy's final maps imply for this item's query."""
    if maps is None:
        return None
    P, B = maps
    kind, target = read["query"]
    if kind == "state":
        return None if target not in P else f"{P[target]}."
    if kind == "bind":
        return None if target not in B else f"{B[target]}."
    if any(a not in P for a in target):
        return None
    return " ".join(P[a] for a in target) + "."


def _sb_pin_chain(read: dict) -> str | None:
    """The zero-state PIN-CHAIN answer for this item's query, or None where it has none.

    Two surface retrievals for a state query, three for a bind query, and no map is carried:
    the give -> swap reference pair cancels the state (see the module comment). Both walks fall
    back to the one-hop stated read where the chain is absent, which is what a policy that
    looked for the channel and did not find it would answer.
    """
    events, P0, B0 = read["events"], read["P0"], read["B0"]
    kind, target = read["query"]
    if kind == "state":
        j = next((i for i in range(len(events) - 1, -1, -1)
                  if events[i][0] == "swap" and events[i][1] == target), None)
        if j is None:
            return None
        o, dyn = events[j][2], events[j][3]
        if dyn:
            for i in range(j - 1, -1, -1):
                if events[i][0] == "give" and events[i][1] == o:
                    # a dynamic give pins o's holder to the role it names; a static one names
                    # the role that holder had at the start. Both reads answer with that role.
                    return f"{events[i][2]}."
        h = B0.get(o)                                    # no pin: the stated holder's role
        return None if h is None or h not in P0 else f"{P0[h]}."
    if kind == "bind":
        P0inv = {v: kk for kk, v in P0.items()}
        g = next((i for i in range(len(events) - 1, -1, -1)
                  if events[i][0] == "give" and events[i][1] == target), None)
        if g is None:
            h = B0.get(target)
            return None if h is None else f"{h}."
        r, dyn = events[g][2], events[g][3]
        if dyn:
            # the last dynamic swap before the give whose referenced object was pinned to r:
            # that swap put its named agent on role r, so the give hands the object to it
            for i in range(g - 1, -1, -1):
                if events[i][0] != "swap" or not events[i][3]:
                    continue
                o2 = events[i][2]
                for jj in range(i - 1, -1, -1):
                    if events[jj][0] == "give" and events[jj][1] == o2:
                        if events[jj][3] and events[jj][2] == r:
                            return f"{events[i][1]}."
                        break
        h = P0inv.get(r)
        return None if h is None else f"{h}."
    return None                                          # the whole-map readout has no such row


def s5_bind_preds(prompt: str, windows=S5_BIND_WINDOWS) -> dict[str, str | None]:
    """Every s5_bind shallow policy's answer for one rendered prompt, in canonical rendered
    form (attached trailing period), or None where the prompt does not support the row."""
    read = s5_bind_read(prompt)
    names = [n for n in S5_BIND_ROWS if n not in S5_BIND_CHANCE_ROWS]
    if read is None:
        return {n: None for n in names}
    events = read["events"]
    L = len(events)
    out: dict[str, str | None] = {
        "stale_resolution": _sb_answer(read, _sb_run(read, "stale")),
        "one_leg_B": _sb_answer(read, _sb_run(read, "B_only")),
        "one_leg_P": _sb_answer(read, _sb_run(read, "P_only")),
        "initial_only": _sb_answer(read, ({**read["P0"]}, {**read["B0"]})),
        "pin_chain": _sb_pin_chain(read),
    }
    exact = _sb_run(read, "surface")
    out["final_state_resolution"] = (
        None if exact is None else
        _sb_answer(read, _sb_run(read, "final",
                                 final=({v: k for k, v in exact[0].items()}, exact[1]))))
    for f in windows:
        T = max(1, int(round(f * L)))                    # the budget, in events, both halves pay
        out[f"window_{int(round(f * 100))}"] = _sb_answer(read, _sb_run(read, "surface",
                                                                        start=max(0, L - T)))
        out[f"prefix_{int(round(f * 100))}"] = _sb_answer(read, _sb_run(read, "surface", end=T))
    out["last_swap_1hop"] = None
    if read["query"][0] == "state":
        for kind, x, y, _dyn in reversed(events):
            if kind == "swap" and x == read["query"][1]:
                partner = read["B0"].get(y)
                if partner is not None and partner in read["P0"]:
                    out["last_swap_1hop"] = f"{read['P0'][partner]}."
                break
    return {n: out.get(n) for n in names}


def _sb_anti_pin_chance(examples, k: int) -> float | None:
    """The ``uniform_anti_pin`` chance level: guess uniformly over the answers that are neither
    the stated one nor the one ``pin_chain`` names.

    In CLOSED FORM per item — the probability the guess is right is [gold survives the
    exclusion] / (k - |exclusion|) — so the row carries no draw of its own and a reading above
    1/(k-1) is a property of the stream. Under ``no_pin`` it always is: the sampler rejects the
    swap that would have made pin_chain right, so struck-out mass is mass the answer does not
    carry. None where the query has no stated answer to exclude (the whole-map readout).
    """
    hits: Counter = Counter()                            # struck-out count -> items surviving it
    n = 0
    for e in examples:
        read = s5_bind_read(e.prompt)
        if read is None:
            continue
        kind, target = read["query"]
        if kind == "state":
            stated = read["P0"].get(target)
        elif kind == "bind":
            stated = read["B0"].get(target)
        else:
            return None
        if stated is None:
            continue
        excl = {f"{stated}."}
        pc = _sb_pin_chain(read)
        if pc is not None:
            excl.add(pc)
        n += 1
        if e.answer not in excl:
            hits[max(1, k - len(excl))] += 1
    # summed by denominator rather than item by item, so the row is a property of the item SET
    # and not of the order it is handed in
    return sum(c / d for d, c in sorted(hits.items())) / n if n else None


def s5_bind_block_drop(examples, width: float, pos: float) -> float:
    """Accuracy of ONE member of the block-drop family: skip the ``width * L`` events starting
    at ``pos * (L - width * L)`` and play the rest exactly.

    window_f is the member at pos 0 with width 1-f, prefix_f the member at pos 1. The family is
    continuous in both arguments and every member carries both maps, so no member is a floor row
    (see S5_BIND_MAP_CARRYING_ROWS); this exists to MEASURE what the excluded continuum reads.
    """
    hits = n = 0
    for e in examples:
        read = s5_bind_read(e.prompt)
        if read is None:
            continue
        L = len(read["events"])
        w = max(1, int(round(width * L)))
        lo = int(round(pos * (L - w)))
        n += 1
        hits += int(_sb_answer(read, _sb_run(read, "surface", drop=(lo, lo + w))) == e.answer)
    return hits / n if n else 0.0


def s5_bind_chain(read: dict) -> list[int] | None:
    """The queried agent's dependency chain, off one rendered prompt: the indices of the events
    that MOVE the answer's role under the rendering's own semantics.

    At a swap both operands are on it — the role moves from one to the other — and the second
    operand is itself resolved through B, so the chain is what closes over both structures. It is
    a SUBSET of the events whose removal can change the answer (dropping the give that last wrote
    a referenced object changes the operand of a chain swap, and so on), which is the direction
    that makes it safe to gate on: bounding the gaps in this set bounds them in every superset.
    None where the prompt is not a single-slot state query.
    """
    if read is None or read["query"][0] != "state":
        return None
    P0, B0 = read["P0"], read["B0"]
    P, B = dict(P0), dict(B0)
    P0inv = {v: k for k, v in P0.items()}
    Pinv = dict(P0inv)
    chain: dict[str, list[int]] = {r: [] for r in P0.values()}
    for i, (kind, x, y, dyn) in enumerate(read["events"]):
        if kind == "swap":
            b = (B if dyn else B0).get(y)
            if b is None or b == x or x not in P or b not in P:
                continue
            chain[P[x]].append(i)
            chain[P[b]].append(i)
            P[x], P[b] = P[b], P[x]
            Pinv = {v: k for k, v in P.items()}
        else:
            h = (Pinv if dyn else P0inv).get(y)
            if h is not None:
                B[x] = h
    return chain.get(P.get(read["query"][1]))


def s5_bind_runs(read: dict) -> tuple[int, int] | None:
    """``(leading run, longest run after it)`` over the events OFF the queried agent's
    dependency chain — the two ends of the stream, which are not symmetric.

    A block of width w misses the chain iff it fits inside one of these runs, so every width-w
    block that starts at or after the chain's first event hits it iff the SECOND number is < w.
    The second number is the one a gate would have to bound; bounding the first hands a
    zero-state policy the answer, which is why the family is closed by cost and not by a gate.
    """
    idx = s5_bind_chain(read)
    if idx is None:
        return None
    L = len(read["events"])
    if not idx:
        return L, 0
    return idx[0], max([L - 1 - idx[-1]]
                       + [idx[j + 1] - idx[j] - 1 for j in range(len(idx) - 1)])


def s5_bind_pin_density(examples) -> float:
    """The fraction of dynamic swaps that ride a LIVE PIN, over a list of s5_bind Examples.

    A STREAM property rather than a policy accuracy, so it is not a floor row: it is the direct
    count of the events that make the ``pin_chain`` row work, read back off the rendered
    prompts. ``TaskSpec.no_pin`` holds it at exactly zero; without it, roughly a quarter to a
    third of dynamic swaps carry the channel and the floor does not decay with length.
    """
    pinned = dyn_swaps = 0
    for e in examples:
        read = s5_bind_read(e.prompt)
        if read is None:
            continue
        P, B = dict(read["P0"]), dict(read["B0"])
        P0inv = {v: k for k, v in read["P0"].items()}
        Pinv = dict(P0inv)
        pin: dict[str, str | None] = {}
        for kind, x, y, dyn in read["events"]:
            if kind == "swap":
                b = (B if dyn else read["B0"]).get(y)
                if b is None or x not in P or b not in P:
                    break
                if dyn:
                    dyn_swaps += 1
                    pinned += int(pin.get(y) is not None and P[b] == pin[y])
                P[x], P[b] = P[b], P[x]
                Pinv = {v: k for k, v in P.items()}
            else:
                h = (Pinv if dyn else P0inv).get(y)
                if h is None:
                    break
                B[x] = h
                pin[x] = y if dyn else None
    return pinned / dyn_swaps if dyn_swaps else 0.0


def s5_bind_operative_floor(floors: dict[str, float], coupled: bool = True) -> float | None:
    """The number an s5_bind cell has to clear: ``operative_floor`` over this family's
    registered rows.

    Named rather than left to the caller because the two adversary families share the row name
    ``uniform``, so a caller who forgets to pass the registered set gets chance rather than the
    floor (see ``registered_for``).

    The max runs over the rows that carry no map (``S5_BIND_ADVERSARIES``), which is the same
    set on both renderings: a map-carrying policy is in the task's own resource class on a
    coupled cell and an order of magnitude more expensive than the task on a decoupled one, and
    neither is a floor. ``coupled`` is kept because callers pass it and because the suite prints
    the class-excluded rows differently on the two renderings, but it no longer changes this max.
    """
    del coupled                                  # the class rule is rendering-independent
    return operative_floor(floors, S5_BIND_ADVERSARIES)


def s5_bind_floors(examples, k: int, windows=S5_BIND_WINDOWS) -> dict[str, float]:
    """Shallow-adversary floors for a list of s5_bind ``tasks.Example``.

    Recomputed from the exact deterministic items a cell scores, so every row is a property of
    that cell. Take the max over the registered rows —
    ``operative_floor(floors, S5_BIND_ADVERSARIES)`` — which is the only number a score may be
    read against; which row is largest varies with k, L and the coupling rates.

    The coupled-only rows (S5_BIND_COUPLED_ONLY_ROWS) are DROPPED where no event is rendered
    "at this point": with a fully decoupled stream each of them reproduces the oracle exactly,
    and printing 1.000 as a floor would be a correctness check wearing a floor's clothes. The
    window and stated rows are defined on both renderings.
    """
    from math import factorial

    n = len(examples)
    if not n:
        return {}
    names = [nm for nm in S5_BIND_ROWS if nm not in S5_BIND_CHANCE_ROWS]
    hits: Counter = Counter()
    defined: Counter = Counter()
    has_dyn = False
    is_all = False
    for e in examples:
        read = s5_bind_read(e.prompt)
        if read is not None:
            has_dyn |= any(d for _kind, _x, _y, d in read["events"])
            is_all |= read["query"][0] == "state_all"
        preds = s5_bind_preds(e.prompt, windows=windows)
        for nm in names:
            if preds[nm] is not None:
                defined[nm] += 1
                hits[nm] += int(preds[nm] == e.answer)
    out = {nm: hits[nm] / n for nm in names if defined[nm]}
    if not has_dyn:
        for nm in S5_BIND_COUPLED_ONLY_ROWS:
            out.pop(nm, None)
    if is_all:
        out["uniform"] = 1.0 / factorial(k)
    else:
        out["uniform_non_initial"] = 1.0 / max(1, k - 1)
        # the anti-pin guess follows pin_chain: where that row is dropped for reproducing the
        # oracle, striking its answer strikes the truth and the guess is not chance but the
        # worst policy available. Registered exactly where the row it strikes is.
        anti = _sb_anti_pin_chance(examples, k) if "pin_chain" in out else None
        if anti is not None:
            out["uniform_anti_pin"] = anti
        out["uniform"] = 1.0 / k
    return {nm: out[nm] for nm in S5_BIND_ROWS if nm in out}


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


# ---------------------------------------------------------------------------
# s5_bind_v3 — the SOURCE-STRUCTURE family, and the ONE-STRUCTURE floor class.
#
# WHAT A FLOOR ROW IS HERE. The construct claims the composed cell forces a forward pass
# carrying BOTH structures. A floor row is a named way of not doing that. Which rows may set a
# floor is decided by COST, not by accuracy.
#
# ===========================================================================================
# THE W CONVENTION — what a live slot is, stated before anything is classified by it
# ===========================================================================================
# ``factworld.composition`` states what one STEP is. Every admit/exclude verdict below turns on
# W as well, so W is stated here to the same standard: as a rule that can be applied to a policy
# by reading its code, and that a test can check.
#
#   W1  A LIVE SLOT HOLDS EXACTLY ONE SYMBOL. Symbols are the atomic ids the prompt uses
#       (agents, objects, roles). A policy's W is the number of slots whose contents it must
#       have available for later use, maximised over its execution.
#   W2  A CARRIED MAP OF c ENTRIES COSTS c SLOTS — one per cell it can answer for. A policy that
#       carries only j of a structure's cells pays j, not the structure's size: partial carry is
#       priced continuously, which is the whole point of stating this.
#   W3  SCRATCH IS CHARGED, uniformly. Every policy here holds one working register — the
#       resolved second operand — so every W below carries a +1. A policy that also walks a
#       CARRIER (the symbol a backward scan is chasing) pays for it too, which is what makes a
#       one-scan row W = 2 and a pure guess W = 1. The earlier rule charged the task no scratch
#       and one_structure_P one, which made the two incomparable.
#   W4  THE STATED FACT BLOCK IS FREE TO LEAVE IN THE PROMPT. It is content-addressed (one H
#       step per keyed read, ``factworld.composition``), so a policy may re-read it instead of
#       holding it, and pays steps rather than slots. This is what makes an uncached read a
#       priced alternative to a carried cell rather than an error.
#   W5  THE EVENT STREAM IS NOT ADDRESSABLE. A policy that needs event i's content scans to it
#       (one E + one C per event passed) and holds nothing for the scan.
#
# Under W1-W5 a row is one of three kinds, and every registered row is priced by which:
#     guess   holds only its own answer                                     W = 1
#     scan    walks one carrier back through the stream                     W = 2
#     carry   holds structures: 1 + (k if it needs P) + (m if it needs B)
# and a policy NEEDS a structure when it reads it live anywhere OR when the query's answer is
# read out of it. That second half is what one_structure_B was mis-costed on: it resolves its
# references out of B, but its replay writes P on every swap and a state query reads the answer
# out of P, so it holds k + m + 1 and not m + 1, and it is not a floor row on a state cell.
#
# ===========================================================================================
# THE CLASS RULE
# ===========================================================================================
# The earlier rule was PARETO: strictly cheaper than the task in W, no more expensive in S. Its
# S half is sound and is kept — it excludes a demand-driven resolver that carries no map but
# re-walks the event list per dependency, which on the temporal construct was exactly correct
# with a FALLING number of live frames and separated from the task only on steps, by 5,100x.
#
# Its W half is not. "Strictly cheaper than k + m + 1" closes the block-drop continuum, because
# every member of that family carries both maps — but it leaves a continuum open one step over:
#
#     CARRY P IN FULL AND ONLY j OF THE m HOLDER CELLS, allocated by first write, online, no
#     lookahead; an uncached holder read hits the stated fact block, which W4 prices at one step
#     and which one_structure_P (the j = 0 member) already does.
#
# Every member costs W = k + j + 1 and S no more than the task's, so strict-cheaper-than-the-task
# admitted every j up to m - 1. Measured on the shipped stream at k = 12, m = 12, L = 128, as a
# ratio to informed chance, j = 0 .. 12:
#     1.08 1.06 1.39 1.10 0.95 0.86 0.92 1.25 1.56 2.31 3.48 6.47 11.00
# so the admitted j = 11 member read 6.47x the chance the cell's floor was published at.
#
# THE RULE THAT CLOSES IT, and it closes the block-drop family by the same argument:
#
#     A ROW MAY SET A FLOOR IFF IT HOLDS AT MOST ONE STRUCTURE — W <= max(k, m) + 1 under the
#     convention above — AND IS NO MORE EXPENSIVE IN STEPS S THAN THE CELL'S CHEAPEST CORRECT
#     ALGORITHM.
#
# The argument is the composed cell's own defining requirement rather than a comparison against
# its cost: the claim is that answering needs BOTH structures carried, so a policy holding both
# is doing the composition — at a discount, or with a wrong time index, or over a truncated
# budget, but doing it — and cannot witness that the task is cheap. A policy holding part of the
# second structure is holding both structures; j = 1 is as much a violation as j = m. One
# inequality therefore closes both continua at once:
#     block-drop, every (position, width):  W = k + m + 1  >  max(k, m) + 1     excluded
#     partial carry, j = 1 .. m:            W = k + j + 1  >  max(k, m) + 1     excluded at m = k
#     partial carry, j = 0:                 W = k + 1     <= max(k, m) + 1      admitted, and it
#                                           IS one_structure_P, already registered.
# The j-profile above is exactly the check that this is not vacuous: every admitted member is at
# chance and every member above chance is excluded, with the boundary between them at j = 1.
#
# ===========================================================================================
# THE COMPONENT RULE — the same move on the axis a component separates on
# ===========================================================================================
# A component renders every second operand by name, so its cheapest correct algorithm already
# holds NO structure — the sparse backward carrier walk, W = 2 — and the one-structure bound is
# vacuous there. The strictness has to go somewhere else, and TWO earlier placements of it fail
# for the same reason the W half failed: they are thresholds on a continuous axis, so the
# continuum reopens one step over.
#
#   ``s_row < s_task``, steps against the task's MEAN. The backward carrier walk truncated to the
#   last L - c events costs 2(L - c) + 3 against the walk's 2L + 2, so every c >= 1 is cheaper
#   and admitted. It is not a cheap policy: it is the task with c events dropped, and it reads
#   9.46x / 9.36x / 9.28x informed chance at c = 1 and L = 128/192/256 (n = 800), decaying
#   smoothly (9.28 7.62 6.41 4.74 2.79 1.31 0.96 at c = 1 2 3 5 9 17 33, L = 256) to chance
#   around c = 33.
#
#   ``S = O(1) in L``, bounded lookback, tested by doubling the stream. It keeps the one-hop read
#   (S = 2k + 3) and excludes c-drop truncation written as such — but the SAME policy written as
#   an absolute budget, "read the last 127 events", has S = 2 min(127, L) + 3, which does not
#   move when the stream doubles. At L = 128 that policy IS the c = 1 walk and reads 9.35x. A
#   rule whose verdict depends on how a parameter is written down is not a rule.
#
# WHAT THE ACCURACIES SAY. On the state component a truncated walk's accuracy is governed by the
# UNREAD PREFIX, not by the budget: at every registered length, absolute budgets T = 1 .. L/2 sit
# at 0.10x-1.18x chance while c-drops of 1 .. 9 sit at 2.5x-9.4x. The two families are the same
# policy; what separates them is how many of the carrier's HOPS they compose.
#
# THE RULE THAT CLOSES IT, and it is the one-structure move on the axis this cell is made of:
#
#     A ROW MAY SET A COMPONENT CELL'S FLOOR IFF IT COMPOSES AT MOST ONE HOP — one event's
#     content resolved into its answer — AND COSTS STRICTLY LESS THAN THE CELL'S OWN ALGORITHM
#     ON EVERY ITEM, i.e. less than that algorithm's MINIMUM per-item cost, not its mean.
#
# Both halves are gaps in KIND and neither is a threshold:
#   DEPTH.  The state component's algorithm composes the whole carrier chain — 2 n_swap / k hops,
#           21 at L=128 and 43 at L=256 — and the bound is ONE, exactly as the composed cell's
#           bound is one structure against two. Hops are integers and a truncation of an
#           unbounded-depth walk still composes more than one of them at every budget that reads
#           more than one hop's worth of stream, so the WHOLE T-family goes out at once, in both
#           parameterisations, with no threshold to place.
#   COST.   Against the MINIMUM the algorithm can cost on any item, a row that ever RUNS that
#           algorithm to completion is excluded — it ties, and a policy that runs the task where
#           it can and abstains elsewhere is the task on a subset, not a cheaper policy. Against
#           the MEAN it was admitted. One word.
# The depth convention is D1-D3 below; the two halves are checked separately by
# ``s5_bind_v3_row_depth`` and ``s5_bind_v3_task_cost_min`` so each verdict is readable.
#
#   state component:     the bound bites. ``last_write_1hop`` — scan back to the last swap naming
#                        the queried agent, take its named partner, read that agent's stated
#                        pointer — is one hop and pays 2k + 3 against a minimum of 2L + 2, so it
#                        is ADMITTED, and it reads 0.98x informed chance at k=12/L=256 and 1.03x
#                        at k=6/L=96 (n = 4000). Every other admitted one-hop row measured there
#                        is at or below chance: the state-free surface read 0.97x / 0.75x, the
#                        truncated walk 0.10x-1.18x over absolute budgets. The operative floor is
#                        then 1.00x and 1.02x, and it is informed chance only to within the
#                        selection bias a max over rows carries: one row's standard error alone
#                        is 0.05x of chance at k=12 and n = 4000, so the number is pool-dependent
#                        at that size (1.00x-1.09x across pools and lengths) and is re-measured
#                        per pool. The whole c-drop continuum is excluded on DEPTH.
#   retrieval component: the depth bound is VACUOUS — that cell's own algorithm is itself one hop
#                        (a named give's recipient IS the answer), so there is nothing to hold at
#                        most one of, and the floor rests entirely on the cost half. It closes
#                        because the SAMPLER puts the answer out of reach: the queried object's
#                        resolving write is pinned into [L/10, 0.75L]
#                        (tasks.s5_bind_v3_bind_window), so it sits at least L - 1 - floor(0.75L)
#                        events from the end and a row cheaper than 2(L - floor(0.75L)) + 3
#                        cannot have read it. Measured at n = 4000, the truncated give-scan
#                        resolves NOTHING at every admitted budget — 0.000, not "near chance" —
#                        and resolves a positive fraction at the first excluded one (0.082 at
#                        k=12/L=256, 0.174 at k=6/L=96), at all six cells. The boundary is the
#                        window's, to the event. That cell's floor is informed chance and it is
#                        now a proof rather than a definition; ``s5_bind_v3_floor_basis`` still
#                        returns 'chance' there, because no admitted row that reads the item
#                        beats a guess.
#
# ===========================================================================================
# THE FITTED SURFACE RANKER IS NOT A FLOOR ROW, AT ANY CELL
# ===========================================================================================
# ``s5_bind_v3_surface_bound`` fits a multinomial logit over the 25 state-free features and
# scores it out of sample. It was priced W = 2 and one backward scan, and it set the published
# floor on the composed cells. That price is wrong, and it is wrong in both directions at once:
# ``s5_bind_v3_surface_features`` holds SIX per-candidate accumulators (mention counts, the first
# and last mention positions, and the three naming counts) and argmaxes over the k candidates, so
#   * one pass over the whole candidate set costs W = 1 + 7k live slots — 43 at k=6 and 85 at
#     k=12 — against the composed cells' one-structure bound of 7 and 13 and the component
#     cells' W = 2; and
#   * the register-lean implementation scores ONE candidate per pass, so it pays k passes,
#     S = 2 k L: 576 against the k=6/L=48 composed task's 208, and 6144 against the k=12/L=256
#     composed task's 1048.
# ``s5_bind_v3_surface_impls`` prices the whole trade-off between those two ends, generously (the
# landmark registers folded into the scratch register, later passes charged only their events),
# and NO point on it is admitted at any of the six cells. So the row is measured and printed as a
# DIAGNOSTIC — what the state-free surface information supports — beside the block-drop family,
# and it never enters ``s5_bind_v3_operative_floor``. Its price is recomputed from the weights
# each fit produces, so a leaner feature set is admitted where this one is not: with no
# accumulator feature carrying weight the policy IS the W = 2 one-scan read it was priced as, and
# the rule admits it on the composed cells and on the state component.
#
# ===========================================================================================
# THE DEPTH CONVENTION  (D1-D3, stated so "one hop" can be applied by reading a policy's code)
# ===========================================================================================
#   D1  A HOP is one event whose CONTENT the row resolves into its answer. A row's DEPTH is the
#       longest chain of read events in which each is located or resolved using the previous
#       one's output.
#   D2  LOCATING an event by a key the row already holds is free — "the last swap naming the
#       queried agent" is keyed by the query, which every policy has. Locating the NEXT event by
#       the previous one's output is a hop; that is what a carrier walk does and what a one-hop
#       read refuses.
#   D3  Resolving a symbol through the STATED fact block is not a hop: it is content-addressed
#       (W4) and reads no event.
# Under D1-D3 a guess is depth 0, the one-hop read and the state-free surface read are depth 1,
# a walk truncated to T events is depth T (its budget bounds its hops), and the forward pass a
# composed cell needs is depth L.
#
# THE W AXIS HAS NO FORCE UNDER A SCRATCHPAD PROTOCOL, AND IT IS THE PROTOCOL THAT DECIDES THAT,
# NOT THE READ. A protocol under which both maps can be — or must be — written down and replayed
# hands out the k + m live slots the one-structure bound prices, to every policy, with the task's
# own algorithm among them. Two protocols do it here: a frontier model's own scratchpad, and this
# repo's GUIDED protocol, whose format REQUIRES the whole of P then the whole of B at every event.
# Under either, every row of the profile is available, so the number a score is read against is
# the TOP of the profile — the block-drop continuum, 6.47x chance at the composed k=12/L=128
# cell — and not the admitted max. What the profile bounds there is what can be answered WITHOUT
# writing both maps down, which is a claim about the cheapest solution and not about the model.
#
# IT FOLLOWS ON BOTH CHANNELS. The trace read and the answer read differ only in which token
# carries the prediction, and the voided bound is not about tokens: a guided decode accumulates
# the generated checkpoints into the same context the answer is decoded from, so an answer
# emitted under that protocol is emitted by a policy that has both maps written down.
# ``s5_bind_v3_operative_floor(..., guided=True)`` therefore returns None on a composed cell,
# exactly as the trace read's wrapper does, and ``s5_bind_v3_pad_reach`` measures what the
# unfloorable class reaches so the retracted floor leaves a number rather than a blank. In that
# regime the composition evidence has to come from the within-cell contrast
# (``factworld.composition``) or from a within-run comparison, never from the floor. The profile
# has force under the PLAIN protocol, where a streaming model's state IS its W.
#
# Steps are counted by ``factworld.composition`` against the convention stated there (one step =
# a keyed header read, an event read, a map resolution, a map write, or a comparison — and a
# backward walk IS charged for the events it scans and rejects).
# ---------------------------------------------------------------------------
S5_BIND_V3_ROWS = ("stated_reference", "one_structure_P", "one_structure_B",
                   "final_state", "last_write_1hop", "last_swap_ref", "initial_only",
                   "window_90", "window_75", "window_50",
                   "prefix_90", "prefix_75", "prefix_50",
                   "uniform_non_initial", "uniform_anti_surface", "uniform")
S5_BIND_V3_CHANCE_ROWS = ("uniform_non_initial", "uniform_anti_surface", "uniform")
S5_BIND_V3_TRUNCATION_ROWS = tuple(r for r in S5_BIND_V3_ROWS
                                   if r.startswith(("window_", "prefix_")))
# Rows defined only where some event carries a REFERENCE. On a component cell every operand is
# named, so each of these reproduces the oracle exactly and is dropped rather than reported.
S5_BIND_V3_REFERENCE_ROWS = ("stated_reference", "one_structure_P", "one_structure_B",
                             "final_state")
S5_BIND_V3_WINDOWS = (0.9, 0.75, 0.5)


def s5_bind_v3_needs(row: str, query: str = "state") -> tuple[bool, bool]:
    """``(needs P, needs B)`` for one registered row at one query kind.

    A policy needs a structure when it reads it live anywhere OR when the answer is read out of
    it (W3 in the convention above). Written out per row rather than folded into the cost so the
    two halves of each verdict are separately readable — the mis-costing this replaces came from
    reading only the first half.
    """
    ans_p = query in ("state", "state_all")
    if row in ("uniform", "uniform_non_initial", "uniform_anti_surface", "initial_only",
               "last_write_1hop", "last_swap_ref"):
        return False, False                      # stated reads and surface scans hold no map
    if row == "stated_reference":                # resolves nothing live; carries the answer's map
        return ans_p, not ans_p
    if row == "one_structure_P":                 # reads P live; the answer's map on top of that
        return True, not ans_p
    if row == "one_structure_B":                 # reads B live; the answer's map on top of that
        return ans_p, True
    if row == "final_state" or row.startswith(("window_", "prefix_")):
        return True, True
    raise KeyError(row)


def _v3_bind_scan(m: int, L: int) -> tuple[int, int]:
    """``(minimum, mean)`` events a backward scan to the queried OBJECT's resolving write reads.

    The sampler pins that write into ``tasks.s5_bind_v3_bind_window``, so the scan is Theta(L)
    and not Theta(m): its distance from the end is at least ``L - 1 - hi``, and the last write
    below ``hi`` sits a further ~m events down (the writes to one object are ~Geometric(1/m)
    apart), so the mean is that minimum plus m.

    The earlier ``L / (n_give / m) = m`` priced the window out of existence and understated this
    cell's own algorithm 5.7x at L = 256 — 27 steps where ``composition.cost_isolated_bind``
    measures 152.4 on the same items.

    THE MEAN IS THE APPROXIMATION AND THE MINIMUM IS EXACT, which matters because only the
    minimum enters admission (``s5_bind_v3_task_cost_min``); the mean is reported cost. Against
    ``composition.cost_isolated_bind`` on the registered grid the mean is within 1% everywhere
    except the SHALLOWEST rung at each operating point, where the ``+ m`` tail overshoots: 4.1%
    high at k=12/L=85 and 10.7% high at k=6/L=31, against -0.9% to 0.2% at every deeper rung. The
    tail term is the gap between consecutive writes to one object, and on a short stream the
    query gate (two writes, the resolving one inside the window) selects objects written more
    often than 1/m of the time.
    """
    from .tasks import s5_bind_v3_bind_window

    _lo, hi = s5_bind_v3_bind_window(L)
    d_min = max(0, L - 1 - hi)               # the closest the window lets the write sit
    return min(L, d_min + 1), min(L, d_min + max(1, m))


def _v3_scan_len(k: int, m: int, n_swap: int, n_give: int, query: str) -> int:
    """Events a backward scan to "the last event naming the queried slot" passes, in expectation.

    The event stream is not addressable (W5), so the scan is charged for every event it reads
    and rejects. A STATE query scans for a swap naming one of k agents, so the expected distance
    from the end is the stream length over the number of such events per slot. A BIND query is
    not that: the sampler places the queried object's write, so the scan is ``_v3_bind_scan``.
    """
    L = n_swap + n_give
    if query not in ("state", "state_all"):
        return _v3_bind_scan(m, L)[1]
    per = n_swap / k
    return L if per <= 0 else min(L, int(round(L / per)))


S5_BIND_V3_FAMILY_PREFIXES = ("trunc_walk_T", "trunc_walk_drop", "give_scan_d")
S5_BIND_V3_MAX_DEPTH = 1                         # a component floor row composes at most one hop


def _v3_family(row: str) -> tuple[str, int] | None:
    """``(family, parameter)`` for a swept family member, or None for a named registry row.

    The two component families are registered with a parameter in the NAME so a whole continuum
    can be priced and plotted rather than summarised by whichever member somebody wrote down:
    ``trunc_walk_T{T}`` reads the last T events, ``trunc_walk_drop{c}`` leaves the first c
    unread, and ``give_scan_d{d}`` reads the last d for the queried object's write.
    ``surface_ranker`` is the fitted 25-feature state-free ranker (``s5_bind_v3_surface_bound``),
    priced here so its admission is decided by the same rule as everything else.
    """
    if row == "surface_ranker":
        return ("surface_ranker", 0)
    for pre in S5_BIND_V3_FAMILY_PREFIXES:
        if row.startswith(pre):
            tail = row[len(pre):]
            if tail.isdigit():
                return (pre, int(tail))
    return None


# DEPTH IS A PROPERTY OF THE POLICY, so it is read off what the policy does and never off what
# the row is called. The three groups below are the registry's policies written out; a name that
# is in none of them and matches no swept family has no depth here and ``s5_bind_v3_row_depth``
# RAISES on it. A default — the ``1 << 30`` this replaces — silently EXCLUDES an unlabelled
# policy, and exclusion pushes the floor DOWN, which is the direction that invalidates a "cleared
# the floor" reading. An unpriced policy has to stop the measurement, not quietly pass it.
S5_BIND_V3_DEPTH_0_ROWS = ("uniform", "uniform_non_initial", "initial_only", "ckpt_copy_prev")
S5_BIND_V3_DEPTH_1_ROWS = ("last_write_1hop", "last_swap_ref", "uniform_anti_surface",
                           "ckpt_last_event_operand", "ckpt_last_event_target")
S5_BIND_V3_REPLAY_DEPTH_ROWS = ("stated_reference", "one_structure_P", "one_structure_B",
                                "final_state")


def s5_bind_v3_row_depth(row: str, query: str = "state", length: int | None = None,
                         weights=None) -> int:
    """The POLICY's COMPOSITION DEPTH under D1-D3: the MOST events whose contents it can chain.

    A guess chains none; the one-hop read chains one and stops; a walk given T events can chain
    up to T; a replay chains the whole stream. The bound is the policy's, not the stream's, which
    is what makes it independent of how the parameter is written: at one cell
    ``trunc_walk_T{L-c}`` and ``trunc_walk_drop{c}`` are the same policy and get the same depth.

    The FITTED RANKER's depth is the max over the features its weights actually load on
    (``s5_bind_v3_surface_depth``), read off ``S5_BIND_V3_SURFACE_FEATURE_DEPTH`` rather than off
    the row name, so mutating the feature set moves the verdict. ``weights`` omitted prices the
    whole registered feature set.

    Raises:
        KeyError: the row names no policy this module prices. Returning a sentinel instead
            excludes it, and an excluded row cannot raise the floor.
    """
    fam = _v3_family(row)
    if fam is not None:
        kind, p = fam
        if kind == "trunc_walk_T":
            return p if length is None else min(p, length)
        if kind == "trunc_walk_drop":
            return (1 << 30) if length is None else max(0, length - p)
        if kind == "give_scan_d":
            return 1                             # one give, and its recipient IS the answer
        return s5_bind_v3_surface_depth(weights)
    if row in S5_BIND_V3_DEPTH_0_ROWS:
        return 0
    if row in S5_BIND_V3_DEPTH_1_ROWS:
        return 1
    if row in S5_BIND_V3_REPLAY_DEPTH_ROWS:
        return (1 << 30) if length is None else length
    if row.startswith("partial_carry_j") and row[len("partial_carry_j"):].isdigit():
        return (1 << 30) if length is None else length      # it replays the whole stream
    if row.startswith(("window_", "prefix_")) and row.split("_")[1].isdigit():
        f = int(row.split("_")[1]) / 100.0   # a replay of f*L events chains that many
        return (1 << 30) if length is None else max(1, int(round(f * length)))
    raise KeyError(f"{row!r} names no policy with a registered depth; price it before it is "
                   f"classified (an unpriced row would be silently excluded)")


def s5_bind_v3_carrier_hops(k: int, n_swap: int) -> float:
    """The STATE leg's chain length on a stream carrying ``n_swap`` swaps: ``2 n_swap / k``.

    A swap moves TWO of the k pointers, so the queried agent is touched by 2 n_swap / k of them
    and its value is carried through that many hops. It is a property of the STREAM and not of
    the cell's name, so it is the quantity a composed cell and a state component are comparable
    on: the composed cell at L carries p_swap L swaps and the state component at L carries L,
    and reading the two at the same L compares chains of different lengths.
    """
    return 2.0 * n_swap / max(1, k)


def s5_bind_v3_work_match(n_swap: int, n_give: int) -> dict[str, int]:
    """The COMPONENT lengths carrying the same amount of their own work as one composed stream.

    A component's stream is all of one kind, so the state component at length ``n_swap`` contains
    exactly the swaps the composed stream contains and the retrieval component at ``n_give``
    exactly its gives. Both are then equal on ``s5_bind_v3_carrier_hops`` and on write count.

    NO p_swap MAKES THE LENGTHS THEMSELVES MATCH. A composed stream of length L holds p L swaps
    and (1 - p) L gives, and both are strictly under L, so pairing a composed cell with components
    at its own L compares 1/p and 1/(1-p) times the work in the two legs whatever p is. Raising
    p_swap changes the ratio; only this pairing removes it.
    """
    return {"state": int(n_swap), "bind": int(n_give)}


def s5_bind_v3_task_depth(k: int, m: int, n_swap: int, n_give: int, named: bool = False,
                          query: str = "state") -> int:
    """THIS CELL's own algorithm's composition depth — what the one-hop bound is a gap against.

    The composed cell's forward pass chains every event. The STATE component's carrier walk
    chains the carrier's hops (``s5_bind_v3_carrier_hops``) — 21 at k=12/L=128 and 43 at L=256,
    measured 21.3 and 42.6. The RETRIEVAL component's algorithm chains ONE, which is why the
    bound is vacuous there and that cell's floor rests on cost.
    """
    if not named:
        return n_swap + n_give
    if query == "bind":
        return 1
    return int(round(s5_bind_v3_carrier_hops(k, n_swap)))


def s5_bind_v3_row_cost(row: str, k: int, m: int, n_swap: int, n_give: int,
                        query: str = "state", named: bool = False,
                        weights=None) -> tuple[int, int]:
    """``(W, S)`` for one row — registry or swept family member — under the W convention above
    and the step convention in ``factworld.composition``.

    W is ``1 + (k if the row needs P) + (m if it needs B)`` (``s5_bind_v3_needs``) for a row that
    carries a structure, 2 for a row that walks one carrier, and 1 for a row that holds only its
    own answer; the scratch register is the +1 and every row pays it, the task included.

    A swept family member is priced at the BUDGET it declares, which is exact on every member the
    rule admits: an admitted truncated walk reads its whole budget, and an admitted truncated
    give-scan never finds the write the sampler pinned, so it reads its whole budget too.

    ``surface_ranker`` has no single price: it is an ALGORITHM with a register/pass trade-off, so
    it is priced over its implementations (``s5_bind_v3_surface_impls``) and this returns the one
    the cell admits if any implementation is admitted, and the cheapest-W one otherwise.
    ``named`` and ``weights`` are only read for that row.
    """
    L = n_swap + n_give
    fam = _v3_family(row)
    if fam is not None:
        kind, p = fam
        if kind == "trunc_walk_T":               # read the last p events, following the carrier
            return 2, 2 * min(p, L) + 3
        if kind == "trunc_walk_drop":             # the same walk with the first p events unread
            return 2, 2 * max(0, L - p) + 3
        if kind == "give_scan_d":                 # read the last p events, then fall back
            return 2, 2 * min(p, L) + 3
        impl = s5_bind_v3_surface_price(k, m, n_swap, n_give, named, query, weights)
        return impl["W"], impl["S"]
    if row.startswith("partial_carry_j") and row[len("partial_carry_j"):].isdigit():
        # carry P in full and only j of the m holder cells, allocated by first write, online, no
        # lookahead; an uncached holder read hits the stated fact block (W4). W = k + j + 1, and
        # the replay itself is the task's, so S ties. It is priced here because a BOUNDED PAD
        # admits exactly j <= pad of this family and it is the family that sets the number there.
        return k + int(row[len("partial_carry_j"):]) + 1, \
            (k + m) + 6 * n_swap + 3 * n_give + 1
    if row == "ckpt_copy_prev":
        # emit the previous checkpoint, so the trace never moves: one keyed read of the stated
        # block and the answer register. The (k + m) L emission itself is protocol overhead the
        # cell's own algorithm pays too, so it is charged to nobody (T2).
        return 1, 2
    if row in ("ckpt_last_event_operand", "ckpt_last_event_target"):
        # the LAST event is located positionally and not by another event's output, so no
        # backward scan is charged (D2): read it, resolve one symbol through the stated block,
        # emit. This is the cheapest one-hop row available at any cell.
        return 2, 3
    needs_p, needs_b = s5_bind_v3_needs(row, query)
    scan = 2 * _v3_scan_len(k, m, n_swap, n_give, query) + 3
    if row in ("uniform", "uniform_non_initial", "initial_only"):
        return 1, 2
    if row in ("last_write_1hop", "last_swap_ref", "uniform_anti_surface"):
        # one backward scan to the last event naming the queried slot; the carrier and one
        # scratch register are all it holds, and it stops there rather than walking on.
        return 2, scan
    w = 1 + (k if needs_p else 0) + (m if needs_b else 0)
    if row in ("stated_reference", "one_structure_P"):
        # every cross reference is a keyed read of the stated block
        return w, k + 6 * n_swap + n_give + 1
    if row == "one_structure_B":
        return w, m + 3 * n_give + n_swap + 1
    if row == "final_state":
        # needs the true final maps before it can start: two passes, both maps, twice over
        return w + (k + m), 2 * ((k + m) + 6 * n_swap + 3 * n_give + 1)
    if row.startswith(("window_", "prefix_")):
        f = int(row.split("_")[1]) / 100.0
        return w, int((k + m) + f * (6 * n_swap + 3 * n_give) + 1)
    raise KeyError(row)


def one_structure_bound(k: int, m: int) -> int:
    """The most live slots a floor row may hold: one whole structure plus the scratch register.

    ``max(k, m) + 1``. A row above it is holding both structures — in full, or in full plus part
    of the other — and a policy that holds both is doing the composition the cell is about.
    """
    return max(k, m) + 1


def floor_eligible(w_row: int, s_row: int, w_max: int, s_max: int,
                   depth_row: int = 0, depth_max: int | None = None) -> bool:
    """The class rule, in the one form both cell kinds use: at most ONE unit of the resource the
    cell's difficulty is made of, and no more of the other than the cell's own algorithm needs.

    ``depth_max`` picks which axis carries the gap. It is None on a COMPOSED cell — the gap is on
    W (one structure against two) and steps are compared to the task's own cost, where a tie is
    allowed because a row holding one structure cannot be the task however long it runs. It is an
    integer on a COMPONENT cell — the gap is on composition DEPTH (one hop against the carrier
    chain), and steps are then compared to the task's MINIMUM per-item cost with NO tie, because
    a row that ever pays exactly what the algorithm pays has run the algorithm.
    """
    if w_row > w_max:
        return False
    if depth_max is None:
        return s_row <= s_max
    return depth_row <= depth_max and s_row < s_max


def s5_bind_v3_task_cost(k: int, m: int, n_swap: int, n_give: int, named: bool = False,
                         query: str = "state") -> tuple[int, int]:
    """``(W, S)`` for THIS CELL's cheapest correct algorithm, S in the MEAN over its items.

    The composed cell's is the forward pass carrying both maps plus the scratch register. A
    COMPONENT cell renders its second operand by name, so every event's identity is fixed on the
    surface: the STATE component is the sparse backward carrier walk, which under W5 pays for
    every event it passes and one resolve per carrier hop (2L + 2 n_swap/k + 2); the RETRIEVAL
    component stops at the last give naming the queried object and pays that scan
    (``_v3_bind_scan``). Both hold one carrier and one scratch register. Every number here is
    within a step of what ``composition.cost_report`` counts on the same items.
    """
    if named:
        L = n_swap + n_give
        if query == "bind":
            return 2, 2 * _v3_bind_scan(m, L)[1] + 3
        return 2, 2 * L + int(round(2 * n_swap / max(1, k))) + 2
    return k + m + 1, (k + m) + 6 * n_swap + 3 * n_give + 1


def s5_bind_v3_task_cost_min(k: int, m: int, n_swap: int, n_give: int, named: bool = False,
                             query: str = "state") -> int:
    """The LEAST this cell's own algorithm can pay on any ONE of its items — the number a
    component floor row has to come in strictly under.

    It is the mean that let the whole truncation continuum in: the c-drop walk beats the MEAN by
    2c and ties nothing, while against the minimum a row that ever runs the algorithm to
    completion is excluded. The state component's walk reads every event on every item, so its
    minimum is 2L + 2 (zero carrier hops); the retrieval component's scan is as short as the
    sampler's window lets it be, ``2 (L - floor(0.75 L)) + 3``.
    """
    if not named:
        return s5_bind_v3_task_cost(k, m, n_swap, n_give, named, query)[1]
    L = n_swap + n_give
    if query == "bind":
        return 2 * _v3_bind_scan(m, L)[0] + 3
    return 2 * L + 2


def s5_bind_v3_admits(row: str, k: int, m: int, n_swap: int, n_give: int, named: bool = False,
                      query: str = "state", weights=None) -> bool:
    """Whether ONE row — registry or swept family member — may set this cell's floor.

    ``weights`` is read only by ``surface_ranker``, whose depth and register cost are properties
    of the features its fit loads on; omitted, that row is priced over the whole registered
    feature set, which is what the fitted ranker uses.
    """
    if row == "surface_ranker":
        return s5_bind_v3_surface_price(k, m, n_swap, n_give, named, query, weights)["admitted"]
    w, s = s5_bind_v3_row_cost(row, k, m, n_swap, n_give, query, named, weights)
    wt, st = s5_bind_v3_task_cost(k, m, n_swap, n_give, named, query)
    if not named:
        return floor_eligible(w, s, one_structure_bound(k, m), st)
    return floor_eligible(w, s, wt, s5_bind_v3_task_cost_min(k, m, n_swap, n_give, named, query),
                          s5_bind_v3_row_depth(row, query, n_swap + n_give, weights),
                          S5_BIND_V3_MAX_DEPTH)


def s5_bind_v3_classify(k: int, m: int, n_swap: int, n_give: int, named: bool = False,
                        query: str = "state", rows=S5_BIND_V3_ROWS) -> dict[str, bool]:
    """Every row, classified at this cell's shape. ``rows`` takes swept family members too.

    COMPOSED cell: the one-structure bound, ``W <= max(k, m) + 1``, and no more steps than the
    task. The cell's difficulty is structure-sized memory, so that is the axis the gap goes on.
    COMPONENT cell: the cheapest correct algorithm already holds no structure, so the
    one-structure bound is vacuous and the gap goes on composition DEPTH — at most one hop —
    with steps then held strictly under the algorithm's MINIMUM per-item cost. That admits the
    one-hop read and the state-free surface family on the state component, excludes the whole
    truncated-walk continuum there on depth, and on the retrieval component (whose own algorithm
    is one hop) leaves only rows too short to reach the write the sampler has pinned.
    """
    return {row: s5_bind_v3_admits(row, k, m, n_swap, n_give, named, query) for row in rows}


def s5_bind_v3_floor_basis(k: int, m: int, n_swap: int, n_give: int, named: bool = False,
                           query: str = "state", guided: bool = False) -> str:
    """Whether this cell's operative floor is MEASURED, DEFINITIONAL, or absent.

    'measured'    — some REGISTERED row is a policy that reads the item, is admitted, and could
                    have come out anywhere; the floor is whatever it scored.
    'chance'      — every admitted registered row holds nothing and reads nothing (the guess
                    rows), so the max over them is the family's own chance level however the
                    items fall. Printing the resulting 1.00x as though a policy had been measured
                    up to it is what this exists to stop. It is the retrieval component's case:
                    the row that would bound it is that cell's own one-hop algorithm, which no
                    admitted row may pay for, and the swept give-scan family — which IS admitted
                    below the sampler's window — resolves nothing there and measures 0.000 at
                    every admitted budget.
    'unfloorable' — the COMPOSED cell under a scratchpad protocol (``guided=True``). Not a
                    missing measurement: the surviving class contains the task, so no number
                    bounds a cheap policy there. ``s5_bind_v3_pad_reach`` prices what that class
                    reaches.
    The swept families are measured separately (``s5_bind_v3_family_floors``) and printed in the
    profile; this label is about which REGISTERED rows the rule lets set the number.
    """
    if guided and not named:
        return "unfloorable"
    cls = ({r: s5_bind_v3_guided_admits(r, k, m, n_swap, n_give, named, query)
            for r in S5_BIND_V3_ROWS} if guided
           else s5_bind_v3_classify(k, m, n_swap, n_give, named, query))
    for row, ok in cls.items():
        if ok and row not in S5_BIND_V3_CHANCE_ROWS and row != "initial_only":
            return "measured"
    return "chance"


def s5_bind_v3_preds(prompt: str, windows=S5_BIND_V3_WINDOWS) -> dict[str, str | None]:
    """Every registered policy's answer for one rendered prompt, in canonical rendered form."""
    from .composition import GIVE, SWAP, answer_of, read, replay

    rec = read(prompt)
    names = [n for n in S5_BIND_V3_ROWS if n not in S5_BIND_V3_CHANCE_ROWS]
    if rec is None:
        return {n: None for n in names}
    evs = rec["events"]
    L = len(evs)
    exact = replay(rec)
    out: dict[str, str | None] = {
        "stated_reference": answer_of(rec, replay(rec, "stated")),
        "one_structure_P": answer_of(rec, replay(rec, "P_live")),
        "one_structure_B": answer_of(rec, replay(rec, "B_live")),
        "initial_only": answer_of(rec, (dict(rec["P0"]), dict(rec["B0"]))),
    }
    out["final_state"] = (None if exact is None else
                          answer_of(rec, _v3_final_pass(rec, exact)))
    for f in windows:
        T = max(1, int(round(f * L)))
        out[f"window_{int(round(f * 100))}"] = answer_of(rec, replay(rec, drop=(0, L - T)))
        out[f"prefix_{int(round(f * 100))}"] = answer_of(rec, replay(rec, drop=(T, L)))
    out["last_write_1hop"] = None
    out["last_swap_ref"] = _v3_last_swap_ref(rec)
    kind, target = rec["query"]
    if kind == "state":
        for kd, tgt, ref, src in reversed(evs):
            if kd == SWAP and tgt == target:
                x = ref if src == "N" else (rec["P0"] if src == "P" else rec["B0"]).get(ref)
                if x is not None and x in rec["P0"]:
                    out["last_write_1hop"] = f"{rec['P0'][x]}."
                break
    elif kind == "bind":
        for kd, tgt, ref, src in reversed(evs):
            if kd == GIVE and tgt == target:
                x = ref if src == "N" else (rec["P0"] if src == "P" else rec["B0"]).get(ref)
                if x is not None:
                    out["last_write_1hop"] = f"{x}."
                break
    return {n: out.get(n) for n in names}


def _v3_last_swap_ref(rec) -> str | None:
    """The STATE-FREE surface read: scan back to the last swap naming the queried agent as its
    FIRST operand and emit the agent its reference clause names — the slot itself on a SAME swap
    ("the agent g7 points to" -> g7), that object's stated holder on a CROSS one.

    One backward scan, W = 2 (the carrier and one scratch register), no map. It is a floor row on
    its own account and it is why ``TaskSpec.q_no_surface`` exists: the SAME branch reads high for
    an algebraic reason and not a sampler one. "Swap the pointers of a and the agent r points to"
    writes P[a] <- P(P(r)), which equals r exactly when r sits on a cycle of P of length 1 or 2,
    i.e. with probability 2/k under a permutation with no other structure — 1.83x the informed
    chance 1/(k-1) at k = 12 before the later stream dilutes it. Measured on the ungated stream
    at n = 3000, conditional on the branch: SAME 0.1279 (1.41x, z = +4.96) at k=12/L=128 and
    0.2503 (1.25x, z = +4.90) at k=6/L=64, against CROSS 0.0759 (0.84x) and 0.1711 (0.86x).
    The ~40-rule one-at-a-time sweep that preceded it reported a best of 1.08x and did not
    contain this rule.

    The CROSS branch reads through the STATED holder map because a CROSS swap's reference slot
    is an OBJECT and objects are not candidate answers — which is also why the leak is a
    one-branch property and the gate is written on one branch.

    Defined only for a state query: the give-side mirror on a component bind cell is the
    component's own algorithm and reads 1.000, which is a correctness check and not a floor.
    """
    from .composition import SWAP

    kind, target = rec["query"]
    if kind != "state":
        return None
    for kd, tgt, ref, src in reversed(rec["events"]):
        if kd != SWAP or tgt != target:
            continue
        if src == "B":
            h = rec["B0"].get(ref)
            return None if h is None else f"{h}."
        return f"{ref}."                             # 'P' (SAME) and 'N' (named) both name agents
    return None


def _v3_final_pass(rec, final):
    """Replay resolving every reference against the TRUE FINAL maps — a wrong-TIME policy, and
    the one that costs more than the task rather than less."""
    from .composition import SWAP

    P, B = dict(rec["P0"]), dict(rec["B0"])
    Pf, Bf = final
    for kind, tgt, ref, src in rec["events"]:
        x = ref if src == "N" else (Pf if src == "P" else Bf).get(ref)
        if x is None:
            return None
        if kind == SWAP:
            if tgt not in P or x not in P:
                return None
            P[tgt], P[x] = P[x], P[tgt]
        else:
            B[tgt] = x
    return P, B


def s5_bind_v3_floors(examples, k: int, m: int | None = None,
                      windows=S5_BIND_V3_WINDOWS) -> dict[str, float]:
    """Every registered row's accuracy on a list of source-structure ``tasks.Example``.

    Recomputed from the exact deterministic items a cell scores. Which of these rows may SET the
    cell's floor is the one-structure class rule (``s5_bind_v3_operative_floor``); the rest are
    printed as diagnostics.

    The reference-resolution rows are DROPPED on a component cell. A component renders every
    second operand by name, so "resolve the references against the stated maps" and "carry one
    structure" are both the exact algorithm there, and printing 1.000 as a floor would be a
    correctness check wearing a floor's clothes.
    """
    from math import factorial

    n = len(examples)
    if not n:
        return {}
    names = [nm for nm in S5_BIND_V3_ROWS if nm not in S5_BIND_V3_CHANCE_ROWS]
    hits: Counter = Counter()
    defined: Counter = Counter()
    is_all = False
    for e in examples:
        preds = s5_bind_v3_preds(e.prompt, windows=windows)
        for nm in names:
            if preds[nm] is not None:
                defined[nm] += 1
                hits[nm] += int(preds[nm] == e.answer)
        if "each of" in e.prompt:
            is_all = True
    out = {nm: hits[nm] / n for nm in names if defined[nm]}
    if s5_bind_v3_is_named(examples):
        for nm in S5_BIND_V3_REFERENCE_ROWS:
            out.pop(nm, None)
    if is_all:
        out["uniform"] = 1.0 / factorial(k)
    else:
        out["uniform_non_initial"] = 1.0 / max(1, k - 1)
        # The anti-surface guess follows the GATE, not the row: it is chance for a guesser who
        # strikes both the stated answer and the one ``last_swap_ref`` names. Registered exactly
        # where a sampler REJECTION rule stands behind the exclusion — detectable from the items
        # themselves, since q_no_surface forces the row to exactly zero hits — and not where the
        # row merely happens to be low on this sample, which would be selection.
        if out.get("last_swap_ref") == 0.0:
            anti = _v3_anti_surface_chance(examples, k)
            if anti is not None:
                out["uniform_anti_surface"] = anti
        out["uniform"] = 1.0 / k
    return {nm: out[nm] for nm in S5_BIND_V3_ROWS if nm in out}


def _v3_anti_surface_chance(examples, k: int) -> float | None:
    """The ``uniform_anti_surface`` chance level: guess uniformly over the answers that are
    neither the stated one nor the one ``last_swap_ref`` names.

    In CLOSED FORM per item — the probability the guess is right is [gold survives the exclusion]
    / (k - |exclusion|) — so the row carries no draw of its own and reads exactly what the two
    generator rejections leave. This is the PRICE of the surface gate: striking an answer the
    sampler has emptied hands a guesser the mass it used to carry. Summed by denominator, so the
    row is a property of the item set rather than of the order it is handed in.
    """
    from .composition import read

    hits: Counter = Counter()
    n = 0
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        kind, target = rec["query"]
        if kind == "state":
            stated = rec["P0"].get(target)
        elif kind == "bind":
            stated = rec["B0"].get(target)
        else:
            return None
        if stated is None:
            continue
        excl = {f"{stated}."}
        sr = _v3_last_swap_ref(rec)
        if sr is not None:
            excl.add(sr)
        n += 1
        if e.answer not in excl:
            hits[max(1, k - len(excl))] += 1
    return sum(c / d for d, c in sorted(hits.items())) / n if n else None


def s5_bind_v3_shape(examples) -> tuple[int, int]:
    """``(mean swaps, mean gives)`` over a cell's items — the shape the cost rule is evaluated
    at, read back off the rendered prompts."""
    from .composition import SWAP, read

    sw = gv = n = 0
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        n += 1
        s = sum(1 for x in rec["events"] if x[0] == SWAP)
        sw += s
        gv += len(rec["events"]) - s
    return (round(sw / max(1, n)), round(gv / max(1, n)))


def s5_bind_v3_is_named(examples) -> bool:
    """Whether this cell renders every second operand by NAME — i.e. whether it is a component
    arm. Read off the prompts, not off the spec."""
    from .composition import read

    for e in examples:
        rec = read(e.prompt)
        if rec is not None:
            return all(ev[3] == "N" for ev in rec["events"])
    return False


def s5_bind_v3_query_kind(examples) -> str:
    """``state`` / ``bind`` / ``state_all`` for a cell, read off its prompts. The class rule
    depends on it: which structure a row must carry includes the one the answer comes out of."""
    from .composition import read

    for e in examples:
        rec = read(e.prompt)
        if rec is not None:
            return rec["query"][0]
    return "state"


def s5_bind_v3_operative_floor(floors: dict[str, float], k: int, m: int,
                               n_swap: int, n_give: int, named: bool = False,
                               query: str = "state", guided: bool = False) -> float | None:
    """The number a source-structure cell has to clear under one PROTOCOL, or None where that
    protocol leaves the cell unfloorable.

    Under the PLAIN protocol (``guided=False``) it is the max over the rows the class rule ADMITS
    at this cell's shape: a row holding more than one structure, composing more than one hop, or
    paying what the cell's own algorithm pays, is a diagnostic and never enters the max.

    Under a SCRATCHPAD protocol (``guided=True``) the live-slot conjunct is void, because the
    format hands out the k + m slots it prices. On a COMPONENT cell that removes nothing — its
    rule is depth <= 1 and cost under the cell's own algorithm's minimum, and a pad substitutes
    for registers, not for chaining — so the number is unchanged. On a COMPOSED cell the live-slot
    conjunct is the whole of the registered class's first half and the step half is a bound the
    task itself satisfies, so what survives admits the task and there is no floor: this returns
    None, and ``s5_bind_v3_pad_reach`` measures how far the unfloorable class reaches.

    ``floors`` may carry swept family members as well as registry rows — every key is classified
    on its own cost, so a family enters the floor exactly where its members are admitted.
    """
    if guided:
        if not named:
            return None
        ok = {nm: s5_bind_v3_guided_admits(nm, k, m, n_swap, n_give, named, query)
              for nm in floors}
    else:
        ok = s5_bind_v3_classify(k, m, n_swap, n_give, named, query, rows=tuple(floors))
    vals = [v for nm, v in floors.items() if ok.get(nm) and v is not None]
    return max(vals) if vals else None


# ===========================================================================================
# THE BOUNDED PAD — the same protocol argument run at a WIDTH instead of at a flag
# ===========================================================================================
# ``guided`` above is binary because the shipped format is: it writes the whole of P then the
# whole of B after every event, so the pad it hands out is k + m slots wide and the composed
# cell's live-slot conjunct is void. The argument is not binary, and writing it at a width is what
# makes the retraction reversible: a pad of ``pad`` slots is ``pad`` free live slots (W3 — a pad
# substitutes for REGISTERS, and a policy may allocate them as it likes), so a row of true cost W
# costs W - pad of the policy's own and the class rule admits it iff
#
#     W - pad <= max(k, m) + 1.
#
# THE COMPOSED CELL IS FLOORED IFF THE TASK IS EXCLUDED, and its algorithm costs W = k + m + 1:
#
#     k + m + 1 - pad > max(k, m) + 1   <=>   pad < min(k, m).
#
# So ``pad = k + m`` (the shipped format) is unfloorable, ``pad = min(k, m)`` is unfloorable by a
# TIE, and every ``pad <= min(k, m) - 1`` restores the bound. ``s5_bind_v3_pad_max_width`` returns
# that last number; at the k = m = 6 local operating point it is 5.
#
# THE SAME INEQUALITY SETS THE FLOOR'S VALUE, and that half decides whether a floorable width is
# a USEFUL one. ``partial_carry_j`` costs W = k + j + 1, so a pad of ``pad`` admits exactly
# j <= pad, and that family is not flat in j. Measured at k = m = 6 on the exact 128 scored items
# and a disjoint 4000-item pool (``scripts/probe_s5bind_v3_bounded_pad_floor_20260802.py``):
#
#     pad    composed@48        composed@64        composed@96
#     1-2    0.2344 (1.17x)     0.2266 (1.13x)     0.2109 (1.05x)   = the PLAIN floor, unchanged
#     3      0.2891 (1.45x)     0.2578 (1.29x)     0.2109 (1.05x)
#     4      0.3906 (1.95x)     0.2734 (1.37x)     0.2109 (1.05x)
#     5      0.6132 (3.07x)     0.4609 (2.31x)     0.2969 (1.48x)   bar 0.763 — unbuyable
#     6+     unfloorable        unfloorable        unfloorable
#
# so the floorable range is pad <= 5 and the range that costs the floor NOTHING is pad <= 2.
# COMPONENT cells are unmoved at every width, measured as well as argued (state@17 0.2188,
# state@80 0.2500, bind@31 and bind@132 0.2000 at every pad from 1 to 12): their rule is depth and
# steps, and T3 holds — a pad stores values, it does not chain them, and it does not shorten a
# scan.
# ===========================================================================================
def s5_bind_v3_pad_max_width(k: int, m: int) -> int:
    """The widest pad under which a COMPOSED cell still has a floor: ``min(k, m) - 1``.

    At ``min(k, m)`` the task ties the one-structure bound and is admitted, which is why the
    boundary is stated as a width rather than as "narrower than one structure".
    """
    return min(k, m) - 1


def s5_bind_v3_pad_floorable(k: int, m: int, pad: int, named: bool = False) -> bool:
    """Whether a cell of this kind keeps a floor under a pad ``pad`` slots wide.

    A COMPONENT cell always does — its class rule never used the W axis. A COMPOSED cell does iff
    ``pad <= s5_bind_v3_pad_max_width(k, m)``.
    """
    return bool(named) or pad <= s5_bind_v3_pad_max_width(k, m)


def s5_bind_v3_pad_admits(row: str, k: int, m: int, n_swap: int, n_give: int,
                          named: bool = False, query: str = "state", pad: int = 0,
                          weights=None) -> bool:
    """Whether ONE row may set a floor under a pad ``pad`` slots wide.

    ``s5_bind_v3_admits`` with the live-slot conjunct relaxed by the pad and the depth and step
    conjuncts untouched (T3). ``pad=0`` is the plain protocol and returns exactly what
    ``s5_bind_v3_admits`` returns, which is what makes this a generalisation rather than a second
    rule; ``pad >= min(k, m)`` admits the task on a composed cell, which is what
    ``s5_bind_v3_pad_operative_floor`` turns into a None.
    """
    if row == "surface_ranker":
        impl = s5_bind_v3_surface_price(k, m, n_swap, n_give, named, query, weights)
        if named:
            return (impl["S"] < s5_bind_v3_task_cost_min(k, m, n_swap, n_give, named, query)
                    and s5_bind_v3_surface_depth(weights) <= S5_BIND_V3_MAX_DEPTH)
        return (impl["W"] - pad <= one_structure_bound(k, m)
                and impl["S"] <= s5_bind_v3_task_cost(k, m, n_swap, n_give, named, query)[1])
    w, s = s5_bind_v3_row_cost(row, k, m, n_swap, n_give, query, named, weights)
    wt, st = s5_bind_v3_task_cost(k, m, n_swap, n_give, named, query)
    if not named:
        return floor_eligible(w - pad, s, one_structure_bound(k, m), st)
    return floor_eligible(w, s, wt, s5_bind_v3_task_cost_min(k, m, n_swap, n_give, named, query),
                          s5_bind_v3_row_depth(row, query, n_swap + n_give, weights),
                          S5_BIND_V3_MAX_DEPTH)


def s5_bind_v3_pad_operative_floor(floors: dict[str, float], k: int, m: int,
                                   n_swap: int, n_give: int, named: bool = False,
                                   query: str = "state", pad: int = 0) -> float | None:
    """The number a score must clear under a BOUNDED-PAD protocol, or None where the pad is wide
    enough to admit the task.

    ``floors`` is expected to carry the ``partial_carry_j{j}`` rows as well as the registry and
    swept ones: the pad admits exactly j <= pad of that family and it is the family that sets the
    number at every pad above 2, so a caller that omits it under-reports the floor.
    ``s5_bind_v3_pad_floors`` builds the whole dict from a cell's items.
    """
    if not s5_bind_v3_pad_floorable(k, m, pad, named):
        return None
    vals = [v for nm, v in floors.items()
            if v is not None and s5_bind_v3_pad_admits(nm, k, m, n_swap, n_give, named, query,
                                                       pad)]
    return max(vals) if vals else None


def s5_bind_v3_pad_floors(examples, k: int, m: int, named: bool | None = None,
                          query: str | None = None) -> dict[str, float]:
    """Every row a bounded-pad floor maxes over, measured on one cell's exact items.

    The registry rows, the swept families, the checkpoint-shaped rows and the whole partial-carry
    profile in one dict, so ``s5_bind_v3_pad_operative_floor`` has the family that decides the
    number at every pad above 2 rather than silently omitting it.
    """
    if named is None:
        named = s5_bind_v3_is_named(examples)
    if query is None:
        query = s5_bind_v3_query_kind(examples)
    ns, ng = s5_bind_v3_shape(examples)
    fl = dict(s5_bind_v3_floors(examples, k, m))
    fl.update(s5_bind_v3_family_floors(examples, k, m, named, query,
                                       rows=s5_bind_v3_family_rows(k, m, ns, ng, named, query)))
    fl.update(s5_bind_v3_ckpt_floors(examples))
    if not named:
        for j, v in enumerate(s5_bind_v3_partial_carry_profile(examples, m)):
            fl[f"partial_carry_j{j}"] = v
    return fl


# --- the two component families, swept, so each cell carries its continuum ------------------
S5_BIND_V3_TRUNC_WALK_T = (1, 2, 3, 6, 12, 24, 48, 96)
S5_BIND_V3_TRUNC_WALK_DROP = (1, 2, 3, 5, 9, 17, 33, 65)
S5_BIND_V3_GIVE_SCAN_D = (1, 2, 4, 8, 16, 32, 64, 128)


def s5_bind_v3_family_rows(k: int, m: int, n_swap: int, n_give: int, named: bool = False,
                           query: str = "state") -> tuple[str, ...]:
    """The swept family members defined at this cell — empty on a composed cell, where no
    backward walk is available at all (no event's operand is known until the other structure has
    been evaluated forward to it).

    The state component carries the truncated carrier walk in BOTH parameterisations, absolute
    budget and c-drop, because they are the same policy and the earlier rules disagreed about
    which one they were looking at. The retrieval component carries the truncated give-scan, and
    the sweep includes the two members either side of the sampler's window so the exclusion
    boundary is visible rather than asserted.
    """
    if not named:
        return ()
    L = n_swap + n_give
    if query == "bind":
        d_min = _v3_bind_scan(m, L)[0]
        ds = set(S5_BIND_V3_GIVE_SCAN_D) | {d_min - 1, d_min, L // 2, L}
        return tuple(f"give_scan_d{d}" for d in sorted(d for d in ds if 1 <= d <= L))
    return (tuple(f"trunc_walk_T{t}" for t in sorted(S5_BIND_V3_TRUNC_WALK_T) if 1 <= t < L)
            + tuple(f"trunc_walk_drop{c}" for c in sorted(S5_BIND_V3_TRUNC_WALK_DROP)
                    if 1 <= c < L))


def s5_bind_v3_trunc_walk(examples, T: int) -> float | None:
    """THE TRUNCATED CARRIER WALK: scan back over the last ``T`` events following the carrier,
    then read the agent it lands on out of the stated pointer map.

    ``T = L`` is the state component's own algorithm and reads 1.000; every smaller T is that
    algorithm stopped early. Its accuracy is governed by the events it does NOT read, so an
    absolute budget and a c-drop behave completely differently at one cell — which is why the
    rule bounds the HOPS it composes and not the events it reads. None where the stream is not a
    named state component.
    """
    from .composition import SWAP, read

    hits = n = 0
    for e in examples:
        rec = read(e.prompt)
        if rec is None or rec["query"][0] != "state":
            return None
        evs = rec["events"]
        if any(ev[3] != "N" for ev in evs):
            return None
        carrier = rec["query"][1]
        for i in range(len(evs) - 1, max(-1, len(evs) - 1 - T), -1):
            kind, tgt, ref, _src = evs[i]
            if kind != SWAP:
                continue
            if carrier == tgt:
                carrier = ref
            elif carrier == ref:
                carrier = tgt
        v = rec["P0"].get(carrier)
        n += 1
        hits += int(v is not None and f"{v}." == e.answer)
    return hits / n if n else None


def s5_bind_v3_give_scan(examples, d: int) -> float | None:
    """THE TRUNCATED GIVE-SCAN: read back over the last ``d`` events for a give naming the
    queried object and answer its named recipient; where none is there, fall back to the stated
    holder, which the sampler guarantees is wrong.

    So this row measures exactly the fraction of items a ``d``-event lookback RESOLVES. A
    guessing version adds 1/(k-1) on the rest, which is what ``uniform_non_initial`` already
    prices, so nothing is hidden by the deterministic fallback. None off a named bind component.
    """
    from .composition import GIVE, read

    hits = n = 0
    for e in examples:
        rec = read(e.prompt)
        if rec is None or rec["query"][0] != "bind":
            return None
        evs = rec["events"]
        if any(ev[3] != "N" for ev in evs):
            return None
        target = rec["query"][1]
        pred = rec["B0"].get(target)
        for i in range(len(evs) - 1, max(-1, len(evs) - 1 - d), -1):
            kind, tgt, ref, _src = evs[i]
            if kind == GIVE and tgt == target:
                pred = ref
                break
        n += 1
        hits += int(pred is not None and f"{pred}." == e.answer)
    return hits / n if n else None


def s5_bind_v3_family_floors(examples, k: int, m: int, named: bool | None = None,
                             query: str | None = None, rows=None) -> dict[str, float]:
    """Every swept family member's accuracy on this cell's exact items, keyed by row name.

    Merged into ``s5_bind_v3_floors`` by a caller that wants the continuum in the floor and in
    the profile; each member is then admitted or excluded on its own cost, like any other row.
    ``rows`` restricts the sweep — a caller re-measuring only the ADMITTED end at a large n does
    not have to pay for the deep walks it has already excluded.
    """
    if named is None:
        named = s5_bind_v3_is_named(examples)
    if query is None:
        query = s5_bind_v3_query_kind(examples)
    if not named or not examples:
        return {}
    ns, ng = s5_bind_v3_shape(examples)
    out: dict[str, float] = {}
    for row in (s5_bind_v3_family_rows(k, m, ns, ng, named, query) if rows is None else rows):
        kind, p = _v3_family(row)
        if kind == "give_scan_d":
            v = s5_bind_v3_give_scan(examples, p)
        elif kind == "trunc_walk_T":
            v = s5_bind_v3_trunc_walk(examples, p)
        else:
            v = s5_bind_v3_trunc_walk(examples, max(0, (ns + ng) - p))
        if v is not None:
            out[row] = v
    return out


S5_BIND_V3_SURFACE_FEATURES = (
    "stated_answer", "echo", "same_ref", "same_ref_1hop", "same_ref_2hop",
    "cross_ref_holder", "cross_ref_holder_1hop", "named_partner", "named_partner_1hop",
    "stated_preimage", "stated_2hop", "prev_naming_swap_slot", "in_last_swap_sentence",
    "n_named_first", "n_ref_slot", "n_give_ref", "last_mention_pos", "first_mention_pos",
    "last_same_ref_anywhere", "last_swap_named_anywhere", "last_give_stated_holder",
    "last_fact_agent", "mention_count", "agent_index", "end_distance",
)

# THE RANKER'S PRICE, PER FEATURE, in the same order — so the row's depth and its registers are
# read off the features a fit loads on and not off the row's name.
#
# DEPTH under D1-D3. A stated-block read is depth 0 (D3: content-addressed, reads no event). A
# feature that reads ONE event located by a key the policy already holds — the query, or "the
# last swap", or "the end of the stream" — is depth 1 (D2). A count or a position over the whole
# stream is also depth 1: it reads many events and CHAINS none, and depth is the longest chain,
# not the number of reads. Nothing in this set is deeper, which is why the row's exclusion below
# is a REGISTER argument and not a depth one.
S5_BIND_V3_SURFACE_FEATURE_DEPTH = (
    0, 0, 1, 1, 1,
    1, 1, 1, 1,
    0, 0, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1,
    0, 1, 0, 1,
)
# REGISTERS. The name of the PER-CANDIDATE accumulator a feature has to keep live for the whole
# pass, or None where the feature is a stated read or a landmark event that O(1) shared registers
# cover. ``end_distance`` is an affine function of ``last_mention_pos`` and shares its
# accumulator, so the count below is over DISTINCT accumulators and never double-counts.
S5_BIND_V3_SURFACE_FEATURE_ACC = (
    None, None, None, None, None,
    None, None, None, None,
    None, None, None, None,
    "n_named", "n_ref", "n_give_ref", "last_pos", "first_pos",
    None, None, None,
    None, "count", None, "last_pos",
)


def s5_bind_v3_surface_loaded(weights=None) -> tuple[str, ...]:
    """The features a fit actually loads on: those with NON-ZERO weight, in registered order.

    ``weights`` may be the dict ``s5_bind_v3_surface_bound`` returns or a sequence in feature
    order; omitted, every registered feature counts, which is what the fitted ranker uses.
    """
    if weights is None:
        return tuple(S5_BIND_V3_SURFACE_FEATURES)
    if isinstance(weights, dict):
        vals = [weights.get(n, 0.0) for n in S5_BIND_V3_SURFACE_FEATURES]
    else:
        vals = list(weights)
    if len(vals) != len(S5_BIND_V3_SURFACE_FEATURES):
        raise ValueError(f"{len(vals)} weights for {len(S5_BIND_V3_SURFACE_FEATURES)} features")
    return tuple(n for n, w in zip(S5_BIND_V3_SURFACE_FEATURES, vals) if w)


def s5_bind_v3_surface_depth(weights=None) -> int:
    """The ranker row's composition depth: the MAX over the features it loads on."""
    idx = {n: i for i, n in enumerate(S5_BIND_V3_SURFACE_FEATURES)}
    return max((S5_BIND_V3_SURFACE_FEATURE_DEPTH[idx[n]]
                for n in s5_bind_v3_surface_loaded(weights)), default=0)


def s5_bind_v3_surface_impls(k: int, m: int, n_swap: int, n_give: int, query: str = "state",
                             weights=None) -> list[dict]:
    """EVERY IMPLEMENTATION of the fitted ranker, priced. The row has no single ``(W, S)``.

    Scoring the k candidates needs, per candidate, one register per DISTINCT accumulator the
    weighted features use plus one for its running score. A pass can therefore carry ``c``
    candidates in ``1 + c (A + 1)`` registers, and ``ceil(k / c)`` passes cover them all, each
    pass one E and one C per event: ``S = ceil(k / c) * 2 L``. ``c = k`` is the one-pass
    implementation (``W = 1 + k (A + 1)``, up to 7k + 1 at the full feature set) and ``c = 1`` is
    the register-lean one (``W = A + 2``, ``S = 2 k L``). Nothing in between is cheaper on both.

    Where the fit loads on NO accumulator feature (``A = 0``) the trade-off disappears: every
    feature is then a stated read or a landmark event, so one backward scan to the last event
    naming the queried slot — the scan ``last_write_1hop`` pays — plus one comparison per
    candidate covers it, at ``W = 2``. That IS the "W = 2 and one backward scan" price, and it is
    what the fitted 25-feature ranker was asserted to pay. It does not: that fit loads on six
    accumulators, where one pass costs ``W >= k + 1`` and the ``W = 2`` implementation costs k
    passes. Every price here is the generous reading — the landmark registers are folded into the
    scratch register and the later passes are charged only their events — because under-pricing a
    row ADMITS it and raises the floor, while over-pricing it lowers the floor.
    """
    L = n_swap + n_give
    idx = {n: i for i, n in enumerate(S5_BIND_V3_SURFACE_FEATURES)}
    accs = {S5_BIND_V3_SURFACE_FEATURE_ACC[idx[n]] for n in s5_bind_v3_surface_loaded(weights)}
    accs.discard(None)
    a = len(accs)
    if a == 0:
        scan = 2 * _v3_scan_len(k, m, n_swap, n_give, query) + 3
        return [{"c": k, "W": 2, "S": scan + k, "A": 0, "passes": 1,
                 "label": "landmarks only: one backward scan, one comparison per candidate"}]
    out = []
    for c in range(1, max(1, k) + 1):
        passes = -(-k // c)
        out.append({"c": c, "W": 1 + c * (a + 1), "S": passes * 2 * L, "A": a, "passes": passes,
                    "label": f"{c} candidate(s) per pass, {passes} pass(es)"})
    return out


def s5_bind_v3_surface_price(k: int, m: int, n_swap: int, n_give: int, named: bool = False,
                             query: str = "state", weights=None) -> dict:
    """The implementation of the fitted ranker THIS CELL admits, or the cheapest-W one if none is.

    The row is admitted iff SOME implementation is, because a policy may choose how to run; it is
    excluded when every point on its register/pass trade-off is out. On both cell kinds every
    point is out at the full feature set — see the module header — so the row is measured and
    printed as a diagnostic and never enters the operative floor.
    """
    depth = s5_bind_v3_surface_depth(weights)
    impls = s5_bind_v3_surface_impls(k, m, n_swap, n_give, query, weights)
    if not named:
        w_max = one_structure_bound(k, m)
        s_max = s5_bind_v3_task_cost(k, m, n_swap, n_give, named, query)[1]
        d_max = None
    else:
        w_max = s5_bind_v3_task_cost(k, m, n_swap, n_give, named, query)[0]
        s_max = s5_bind_v3_task_cost_min(k, m, n_swap, n_give, named, query)
        d_max = S5_BIND_V3_MAX_DEPTH
    for impl in sorted(impls, key=lambda d: (d["S"], d["W"])):
        if floor_eligible(impl["W"], impl["S"], w_max, s_max, depth, d_max):
            return {**impl, "depth": depth, "admitted": True,
                    "W_max": w_max, "S_max": s_max, "depth_max": d_max}
    cheap = min(impls, key=lambda d: (d["W"], d["S"]))
    return {**cheap, "depth": depth, "admitted": False,
            "W_max": w_max, "S_max": s_max, "depth_max": d_max}


def s5_bind_v3_surface_features(rec, agents) -> dict[str, list[float]]:
    """Per-candidate surface features for one item — the whole state-free candidate set at once.

    Every feature is a stated-block read, a count or a position, so nothing here carries a MAP —
    but six of them are per-candidate accumulators, and a policy that argmaxes over the k
    candidates in one pass holds all of them live. The row's price is therefore a register/pass
    trade-off rather than a number, and it is computed in ``s5_bind_v3_surface_impls`` off the
    two tables beside ``S5_BIND_V3_SURFACE_FEATURES``, in this order.
    """
    from .composition import GIVE, SWAP

    P0, B0, evs = rec["P0"], rec["B0"], rec["events"]
    q = rec["query"][1]
    inv0 = {v: a for a, v in P0.items()}
    L = max(1, len(evs))
    naming = [(i, ref, src) for i, (kd, t, ref, src) in enumerate(evs)
              if kd == SWAP and t == q]
    ls = naming[-1] if naming else None
    ls2 = naming[-2] if len(naming) > 1 else None

    def slot(entry):
        if entry is None:
            return None
        _i, ref, src = entry
        return B0.get(ref) if src == "B" else ref

    n_named = {a: 0 for a in agents}
    n_ref = {a: 0 for a in agents}
    n_give_ref = {a: 0 for a in agents}
    cnt = {a: 0 for a in agents}
    last_pos = {a: -1 for a in agents}
    first_pos = {a: -1 for a in agents}
    for i, (kd, t, ref, src) in enumerate(evs):
        touched = []
        if kd == SWAP and t in n_named:
            n_named[t] += 1
            touched.append(t)
        tok = ref if src != "B" else None
        if tok in n_ref:
            (n_ref if kd == SWAP else n_give_ref)[tok] += 1
            touched.append(tok)
        for a in touched:
            cnt[a] += 1
            last_pos[a] = i
            if first_pos[a] < 0:
                first_pos[a] = i
    last_swap = next(((t, ref) for kd, t, ref, _s in reversed(evs) if kd == SWAP), None)
    last_same = next((ref for kd, _t, ref, src in reversed(evs)
                      if kd == SWAP and src == "P"), None)
    last_give = next((ref for kd, _t, ref, _s in reversed(evs) if kd == GIVE), None)
    lg_holder = B0.get(last_give, last_give)
    slot_ls, slot_ls2 = slot(ls), slot(ls2)
    src_ls = ls[2] if ls else None
    last_fact = list(P0)[-1] if P0 else None
    out = {}
    for j, c in enumerate(agents):
        out[c] = [
            float(c == P0.get(q)), float(c == q),
            float(src_ls == "P" and c == slot_ls),
            float(src_ls == "P" and c == P0.get(slot_ls)),
            float(src_ls == "P" and c == P0.get(P0.get(slot_ls))),
            float(src_ls == "B" and c == slot_ls),
            float(src_ls == "B" and c == P0.get(slot_ls)),
            float(src_ls == "N" and c == slot_ls),
            float(src_ls == "N" and c == P0.get(slot_ls)),
            float(c == inv0.get(q)), float(c == P0.get(P0.get(q))),
            float(slot_ls2 is not None and c == slot_ls2),
            float(last_swap is not None and c in last_swap),
            n_named[c] / L, n_ref[c] / L, n_give_ref[c] / L,
            last_pos[c] / L, first_pos[c] / L,
            float(c == last_same), float(last_swap is not None and c == last_swap[0]),
            float(c == lg_holder), float(c == last_fact),
            cnt[c] / L, j / max(1, len(agents)), (L - 1 - last_pos[c]) / L,
        ]
    return out


S5_BIND_V3_SURFACE_FIT_MIN = 2000        # below this the held-out curve is still moving
S5_BIND_V3_SURFACE_BLOCKS = 2            # disjoint fit blocks whose spread is reported; a caller
                                         # passing a fit pool of BLOCKS * S5_BIND_V3_SURFACE_FIT_MIN
                                         # gets the spread AT the registered budget


def s5_bind_v3_surface_bound(examples, k: int, held_out=None, epochs: int = 250,
                             lr: float = 1.0, l2: float = 1e-3,
                             blocks: int = S5_BIND_V3_SURFACE_BLOCKS) -> dict | None:
    """THE STATE-FREE SURFACE FAMILY, as a FITTED RANKER scored out of sample. A DIAGNOSTIC.

    A multinomial logit over ``S5_BIND_V3_SURFACE_FEATURES`` is fitted on ``examples`` and scored
    on ``held_out``, a DISJOINT list of items from the same cell; with ``held_out`` omitted the
    given items are split in half. Returns the held-out accuracy, the in-sample one, the two
    sample sizes and the block-to-block spread, or None where the cell has no state query.

    IT IS NOT A FLOOR ROW. No implementation of it achieves a price the class rule admits, at
    either cell kind: six of its features are per-candidate accumulators, so one pass over the k
    candidates holds ``1 + 7k`` registers and the register-lean implementation pays k passes,
    ``S = 2 k L`` (``s5_bind_v3_surface_impls``). It is measured and printed beside the profile
    for the same reason the block-drop family is — so what the excluded policy actually reads is
    visible — and it never enters ``s5_bind_v3_operative_floor``.

    THE FIT SAMPLE IS THE BINDING CONSTRAINT, not the scored one, and it has to be at least
    ``S5_BIND_V3_SURFACE_FIT_MIN``: fitting 25 features on a few hundred items leaves the estimate
    biased DOWN, because the weights are poor and the ranker then scores below the best policy in
    the span. Measured against a fixed 4000-item held-out sample, the held-out accuracy climbs and
    then flattens — 1.12x / 1.21x / 1.23x / 1.21x / 1.21x informed chance at 250 / 500 / 1000 /
    2000 / 4000 fit items on the k=6/L=48 composed cell, and 1.10x / 1.13x / 1.18x / 1.12x /
    1.12x on the k=12/L=256 one. From 1000 on the movement is inside the spread of a fit at that
    budget, which is what ``blocks`` measures: it refits on that many DISJOINT equal blocks of
    the fit sample, scores each on the same held-out items, and reports the block-to-block spread
    beside the pooled number rather than shipping one figure that hides it.

    WHY A FIT AND NOT A SWEEP. Running candidate rules one at a time and reporting the largest is
    a selection statistic over an exchangeable family — the same trap the fixed-offset partition
    documents for the pointer-map family — and it is also weak: the ~40-rule sweep that preceded
    this reported a best of 1.08x chance and did not contain ``last_swap_ref``, which reads 1.41x
    conditional. A ranker fitted on one sample and scored on another is neither selected nor
    weak: it is an estimate of the best policy IN THE SPAN of these features, and the span is the
    whole state-free candidate set rather than one rule.
    """
    from .composition import read

    def prep(items):
        rows, golds = [], []
        for e in items:
            rec = read(e.prompt)
            if rec is None or rec["query"][0] != "state":
                continue
            agents = sorted(rec["P0"], key=lambda s: int(s[1:]))
            f = s5_bind_v3_surface_features(rec, agents)
            rows.append([(c, f[c]) for c in agents])
            golds.append(e.answer.strip().rstrip("."))
        return rows, golds

    rows, golds = prep(examples)
    if held_out is None:
        half = len(rows) // 2
        rows, golds, ho_rows, ho_golds = rows[:half], golds[:half], rows[half:], golds[half:]
    else:
        ho_rows, ho_golds = prep(held_out)
    if len(rows) < 40 or len(ho_rows) < 40:
        return None
    n_feat = len(S5_BIND_V3_SURFACE_FEATURES)
    w = _v3_fit_ranker(rows, golds, n_feat, epochs, lr, l2)
    per_block = []
    b = max(1, int(blocks))
    if b > 1 and len(rows) // b >= 40:
        step = len(rows) // b
        for i in range(b):
            lo, hi = i * step, (i + 1) * step
            wb = _v3_fit_ranker(rows[lo:hi], golds[lo:hi], n_feat, epochs, lr, l2)
            per_block.append(_v3_rank_acc(wb, ho_rows, ho_golds))
    return {"held_out": _v3_rank_acc(w, ho_rows, ho_golds),
            "in_sample": _v3_rank_acc(w, rows, golds),
            "n_fit": len(rows), "n_held_out": len(ho_rows),
            "fit_at_least_min": len(rows) >= S5_BIND_V3_SURFACE_FIT_MIN,
            "blocks": per_block, "n_per_block": (len(rows) // b) if per_block else None,
            "block_spread": (max(per_block) - min(per_block)) if per_block else None,
            "weights": dict(zip(S5_BIND_V3_SURFACE_FEATURES, w)),
            "chance": 1.0 / max(1, k - 1)}


def _v3_fit_ranker(rows, golds, n_feat, epochs, lr, l2):
    """Full-batch gradient ascent on the multinomial log-likelihood. Deterministic: no shuffle,
    no draw, so the bound is a property of the items and not of a seed."""
    from math import exp

    w = [0.0] * n_feat
    for _ep in range(epochs):
        grad = [0.0] * n_feat
        for cand, g in zip(rows, golds):
            zs = [sum(a * b for a, b in zip(w, f)) for _c, f in cand]
            mx = max(zs)
            es = [exp(z - mx) for z in zs]
            tot = sum(es)
            for (c, f), e in zip(cand, es):
                d = (1.0 if c == g else 0.0) - e / tot
                if d:
                    for j in range(n_feat):
                        if f[j]:
                            grad[j] += d * f[j]
        step = lr / len(rows)
        for j in range(n_feat):
            w[j] += step * (grad[j] - l2 * w[j])
    return w


def _v3_rank_acc(w, rows, golds) -> float:
    hit = 0
    for cand, g in zip(rows, golds):
        best = max(cand, key=lambda cf: sum(a * b for a, b in zip(w, cf[1])))
        hit += int(best[0] == g)
    return hit / max(1, len(rows))


def s5_bind_v3_partial_carry(examples, j: int) -> float:
    """One member of the PARTIAL-CARRY family: carry P in full and only ``j`` of the m holder
    cells, allocated by FIRST WRITE, online, with no lookahead; an uncached holder read hits the
    stated fact block (one H step, W4).

    ``j = 0`` is ``one_structure_P`` and ``j = m`` is the exact algorithm, so the family is the
    continuum between the cheapest one-structure policy and the task. It costs W = k + j + 1, so
    the one-structure bound admits exactly ``j = 0`` — which is the check that the bound is the
    rule the earlier strict-W rule should have been, and not a threshold placed after the fact.
    """
    from .composition import SWAP, read

    hits = n = 0
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        n += 1
        P, B0, cache = dict(rec["P0"]), rec["B0"], {}
        ok = True
        for kind, tgt, ref, src in rec["events"]:
            if src == "N":
                x = ref
            elif src == "P":
                x = P.get(ref)
            else:
                x = cache.get(ref, B0.get(ref))
            if x is None:
                ok = False
                break
            if kind == SWAP:
                if tgt not in P or x not in P:
                    ok = False
                    break
                P[tgt], P[x] = P[x], P[tgt]
            elif tgt in cache or len(cache) < j:
                cache[tgt] = x
        if not ok:
            continue
        qkind, target = rec["query"]
        if qkind == "state":
            pred = None if target not in P else f"{P[target]}."
        elif qkind == "bind":
            v = cache.get(target, B0.get(target))
            pred = None if v is None else f"{v}."
        else:
            pred = None
        hits += int(pred is not None and pred == e.answer)
    return hits / n if n else 0.0


def s5_bind_v3_partial_carry_profile(examples, m: int) -> list[float]:
    """The whole ``j = 0 .. m`` partial-carry profile for one cell, in j order."""
    return [s5_bind_v3_partial_carry(examples, j) for j in range(m + 1)]


def s5_bind_v3_width_profile(examples, widths=(0.02, 0.05, 0.10, 0.15, 0.25, 0.50),
                             positions=(0.0, 0.2, 0.4, 0.6, 0.8, 0.95)) -> dict[float, float]:
    """The BLOCK-DROP family's profile: the best member at each width, over positions.

    The other continuum the one-structure bound closes. Every member carries both maps, so every
    member costs W = k + m + 1 and the width axis is a step axis alone — which is why the family
    has to be reported as a profile beside the partial-carry one rather than summarised by the
    two endpoints that happen to be registered as rows.
    """
    return {w: max(s5_bind_v3_block_drop(examples, w, p) for p in positions) for w in widths}


def s5_bind_v3_slot_profile(examples, k: int, m: int, named: bool = False,
                            query: str | None = None) -> list[dict]:
    """THIS CELL'S FLOOR, AS A PROFILE OVER LIVE SLOTS — not as a single number.

    One record per distinct W the measured policy families occupy, each carrying the best
    accuracy available at that W, the policy that achieves it, and whether the class rule admits
    it. The operative floor is the last admitted row of the profile; everything above it is what
    the bound excludes, and it is printed so the exclusion can be judged rather than believed.

    READ IT UNDER THE PLAIN PROTOCOL. A streaming model with no scratchpad has W for its state, so
    the profile says what each state budget buys. IT HAS NO FORCE UNDER A SCRATCHPAD PROTOCOL, and
    what decides that is the PROTOCOL and not the model or the channel: where both maps can be —
    or, under this repo's guided format, must be — written down and replayed, every row is
    available to every policy and the number a score must clear is the TOP of the profile rather
    than the admitted max. On a COMPOSED cell that top is the task itself, so there is no floor at
    all and ``s5_bind_v3_operative_floor(..., guided=True)`` returns None on both of that
    protocol's reads. There the floor bounds the cheapest solution, not the model, and the
    composition evidence has to come from the within-cell contrast (``factworld.composition``) or
    from a within-run comparison instead.
    """
    if query is None:
        query = s5_bind_v3_query_kind(examples)
    ns, ng = s5_bind_v3_shape(examples)
    fl = dict(s5_bind_v3_floors(examples, k, m))
    fl.update(s5_bind_v3_family_floors(examples, k, m, named, query))
    cls = s5_bind_v3_classify(k, m, ns, ng, named, query, rows=tuple(fl))
    wt, st = s5_bind_v3_task_cost(k, m, ns, ng, named, query)
    # the profile runs along whichever resource the cell separates on: live slots on a composed
    # cell, steps on a component one, where every policy already holds two registers.
    axis = "S" if named else "W"
    at: dict[int, tuple[float, str, bool]] = {}

    def put(x, v, row, admitted):
        if x not in at or v > at[x][0]:
            at[x] = (v, row, admitted)

    for row, v in fl.items():
        w, s = s5_bind_v3_row_cost(row, k, m, ns, ng, query)
        put(s if named else w, v, row if cls[row] else row + " (excluded)", cls[row])
    if named:
        # the truncation continuum sits at the top of the STEP axis, one step under the cell's
        # own algorithm, and the profile has to show it there: it is what the depth bound
        # excludes and what a step bound against the mean admitted.
        smin = s5_bind_v3_task_cost_min(k, m, ns, ng, named, query)
        put(smin, 1.0, f"the component's own algorithm (min {smin}, mean {st})", False)
    else:
        # j = m is the task itself, so the family is reported up to m - 1 and the task named.
        for j in range(m):
            put(k + j + 1, s5_bind_v3_partial_carry(examples, j), f"partial_carry_j{j}",
                k + j + 1 <= one_structure_bound(k, m))
        put(k + m + 1, 1.0, "the task (both maps)", False)
    return [{"axis": axis, axis: x, "W": x if axis == "W" else wt,
             "acc": at[x][0], "row": at[x][1], "admitted": at[x][2]}
            for x in sorted(at)]


def s5_bind_v3_block_drop(examples, width: float, pos: float) -> float:
    """One member of the block-drop family: skip ``width * L`` events starting at
    ``pos * (L - width * L)`` and play the rest exactly.

    Every member carries both maps, so every member costs W = k + m + 1 and no member can be a
    floor row under the one-structure bound; this exists to MEASURE what the excluded continuum
    actually reads, so the exclusion can be judged rather than taken on faith.
    """
    from .composition import answer_of, read, replay

    hits = n = 0
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        L = len(rec["events"])
        w = max(1, int(round(width * L)))
        lo = int(round(pos * (L - w)))
        n += 1
        hits += int(answer_of(rec, replay(rec, drop=(lo, lo + w))) == e.answer)
    return hits / n if n else 0.0


# ===========================================================================================
# THE GUIDED PROTOCOL: what it does to a floor, on BOTH channels
# ===========================================================================================
# Under the GUIDED protocol the events are teacher-forced and the model generates every per-event
# checkpoint — the whole of P in agent order then the whole of B in object order, k + m tokens per
# event — and then the answer. It carries TWO reads of one gold. The TRACE read takes the LAST
# checkpoint's value for the queried slot; it removes the answer-emission channel, which is what
# it is for, since a model can hold the right value and emit a different token and two published
# nulls were that. The ANSWER read takes the emitted token.
#
# THE PROTOCOL IS THE DISCRIMINATOR, NOT THE READ. T2 below is a statement about the FORMAT: it
# holds for every policy decoding under it, whichever token the prediction is finally written to.
# The guided decode accumulates the generated checkpoints into the same context the answer comes
# out of, so the answer read under this protocol is an answer emitted by a policy that has both
# maps written down. Both reads therefore get the same treatment — ``s5_bind_v3_operative_floor``
# takes ``guided`` and returns None on a composed cell under it, and the trace entry points below
# are that call with ``guided=True`` fixed. An earlier revision voided the composed cell's floor
# on the trace channel only, which put a floor that does not hold beside the guided ANSWER score.
#
# THREE FACTS FIX WHAT A FLOOR UNDER IT CAN BE.
#
#   T1  THE FINAL CHECKPOINT'S QUERIED SLOT IS THE GOLD ANSWER, by construction of the trace
#       (``tasks._ex_s5_bind_v3`` builds ``meta["trace"]`` from the same replay the gold comes
#       from) and measured 1000/1000 at every registered cell by ``s5_bind_v3_trace_is_answer``.
#       So the trace read and the answer read score the SAME quantity, and every registered row's
#       trace score is its answer score to the item. Nothing about a floor row changes because
#       its prediction is delivered in a checkpoint slot rather than in the answer token.
#
#   T2  THE PROTOCOL GRANTS THE PAD. The guided format REQUIRES the whole of P and B to be
#       written out at every event, so the k + m live slots the one-structure bound prices are
#       handed to every policy — the cell's own algorithm included — and the emission is charged
#       to every policy equally, which is why steps are counted on RESOLUTION work exactly as on
#       the plain protocol. A bound on live slots cannot discriminate under a format that
#       supplies them, and the format is the same whichever token the prediction is read from.
#
#   T3  DEPTH AND STEPS SURVIVE THE PAD. Composition depth is a property of the policy under
#       D1-D3 — the longest chain of events whose contents it resolves into its answer — and a
#       notepad stores values rather than chaining them; a one-step updater invoked once per
#       event still chains L events and is depth L. Steps are charged for the events a policy
#       reads however it stores what it reads. So both axes read exactly as they do on the
#       answer read.
#
#       T3 IS PROCEDURE-LEVEL AND THAT IS LOAD-BEARING. It is the same convention the swept
#       families are priced under — ``trunc_walk_T{T}`` is depth T although each step of the walk
#       is one hop — so it is not a rule invented for this read. Priced per FORWARD PASS instead,
#       a guided one-step updater is depth 1 and the COMPONENT floors go with the composed one:
#       ``trunc_walk_drop1`` costs 2(L-1)+3, one step under the state component's own minimum of
#       2L+2, and reads 0.667 at state@17 and 0.688 at state@128 against a floor of 0.200. Both
#       axes would then be void and no cell would be floored under this protocol at all.
#
# WHAT FOLLOWS, PER CELL KIND, AND THE TWO ANSWERS ARE DIFFERENT.
#
#   COMPONENT CELLS ARE FLOORED, at the plain protocol's floor unchanged, on BOTH channels. Their
#   rule is depth <= 1 AND steps < the cell's own algorithm's MINIMUM per-item cost; its W bound
#   (2) admits no row the other two do not already admit, so T2 removes nothing. The state
#   component's floor is then the same admitted max, and the retrieval component's is still
#   informed chance PROVED by the sampler's pin — a pad substitutes for REGISTERS, not for
#   CHAINING, and it does not shorten a scan.
#
#   THE COMPOSED CELL IS NOT FLOORED, ON EITHER CHANNEL. Its registered class is the one-structure
#   bound plus a step bound the cell's own algorithm SATISFIES (a tie is allowed there, because a
#   row holding one structure cannot be the task however long it runs). T2 removes the first; what
#   is left admits the task. Even with the tie refused the class holds the both-maps replay with
#   one event dropped, which ``s5_bind_v3_pad_reach`` measures at n = 1000 at 0.719 / 0.742 /
#   0.604 on the k=6 composed cells at L = 48 / 64 / 96 against plain-protocol floors of 0.201 /
#   0.200 / 0.209 — 2.9x to 3.7x — and 0.826 at L = 16 against 0.516. On the exact 128 items the
#   guided read scores it is 0.719 / 0.758 / 0.562 against 0.234 / 0.227 / 0.211, and the
#   partial-carry member at W = 12, exactly the k + m slots the format forces the model to write,
#   reads 0.609 at composed@48. ``s5_bind_v3_operative_floor(..., guided=True)`` returns None
#   there rather than a max over a class that contains the task.
#
# SO A COMPOSED-CELL SCORE UNDER THIS PROTOCOL IS A WITHIN-RUN COMPARISON AND NEVER A CLEARED
# FLOOR, on the trace channel and on the answer channel alike. The comparison is a real object —
# the same seeds, the same item count, matched depth and matched cost — and the DOWNWARD
# separation it carries does not need a floor, because a floor bounds what a cheap policy scores
# and a deficit is not a claim about cheapness. What it cannot support is the other direction: no
# "the composed cell clears its floor" reading is available under this protocol at any registered
# length.
#
# IT IS UNPAIRED. The composed cell and its work-matched component are different SPECS drawing
# different item streams at different lengths, matched on seed, item count and forward-pass cost
# and on nothing else, so a difference between them is a difference of two independent
# proportions. "On the same items" is false of this comparison and is false of every three-cell
# table; only the two READS of one cell (plain and guided) score the same items.
#
# THE CHEAP CHECKPOINT-SHAPED POLICIES ARE PRICED HERE and none of them raises a floor:
# ``ckpt_copy_prev`` (emit the previous checkpoint, so the trace never moves) is exactly the
# stated initial value and the query gate — the queried slot must move at least twice and end
# different from its stated value — forces it to 0.000 at every cell; the two one-hop reads of
# the LAST event are at or below informed chance everywhere. The oracle-assisted rows are
# reported and excluded: ``s5_bind_v3_ckpt_lag`` (run the algorithm and stop j events early) is
# the copier with the true trace handed to it, and it is depth L - j on a state or composed cell
# and costs the whole scan on a retrieval one.
# ===========================================================================================
S5_BIND_V3_CKPT_ROWS = ("ckpt_copy_prev", "ckpt_last_event_operand", "ckpt_last_event_target")
S5_BIND_V3_CKPT_LAG = (1, 2, 3, 5, 9, 17, 33, 65)
S5_BIND_V3_PAD_WIDTHS = (0.02, 0.05, 0.10, 0.25)


def s5_bind_v3_trace_slot(example, k: int, m: int, agents, objs) -> str | None:
    """The FINAL checkpoint's value for the queried slot — the quantity the trace read scores.

    Read off ``meta["trace"]``, which is the per-event checkpoint stream the guided protocol
    teacher-forces the events of and the model generates the slots of: ``k + m`` tokens per
    event, the whole of P in ``agents`` order then the whole of B in ``objs`` order. None where
    the cell carries no trace or the queried slot is not in it.
    """
    tr = example.meta.get("trace")
    if not tr:
        return None
    toks = tr.split()
    per = k + m
    if per <= 0 or len(toks) < per or len(toks) % per:
        return None
    final = toks[-per:]
    qs, qb = example.meta.get("q_state"), example.meta.get("q_bind")
    if qs is not None and qs in agents:
        return final[list(agents).index(qs)]
    if qb is not None and qb in objs:
        return final[k + list(objs).index(qb)]
    return None


def s5_bind_v3_trace_is_answer(examples, k: int, m: int, agents, objs) -> tuple[int, int]:
    """``(agreements, n)`` for T1: the final checkpoint's queried slot against the gold answer.

    The whole trace read rests on this identity, so it is measured on the exact items rather than
    argued from the sampler: if it ever fails, the trace read is scoring something else and its
    floor argument is about a different quantity than the answer read's.
    """
    agree = n = 0
    for e in examples:
        v = s5_bind_v3_trace_slot(e, k, m, agents, objs)
        if v is None:
            continue
        n += 1
        agree += int(f"{v}." == e.answer)
    return agree, n


def s5_bind_v3_slot_moves(examples, k: int, m: int, agents, objs) -> dict:
    """How many times the QUERIED slot moves, per item — the copier's whole story.

    The query gate gives the copier its floor: a state query needs the queried agent's pointer to
    have moved at least twice and to end different from its stated target, a bind query the same
    of the queried object's holder. So ``ckpt_copy_prev`` — a trace that never moves — is wrong
    on every item by construction, and the distribution below says how far from a fixpoint the
    queried slot actually is rather than leaving it at "at least two".

    Returns ``{"counts": {moves: items}, "min", "max", "mean", "median", "n"}``, counting the
    first write off the STATED value, which is the move a copier misses first.
    """
    from .composition import read

    counts: Counter = Counter()
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        v = s5_bind_v3_trace_slot(e, k, m, agents, objs)
        if v is None:
            continue
        tr = e.meta["trace"].split()
        per = k + m
        qs, qb = e.meta.get("q_state"), e.meta.get("q_bind")
        if qs is not None and qs in agents:
            idx, start = list(agents).index(qs), rec["P0"].get(qs)
        else:
            idx, start = k + list(objs).index(qb), rec["B0"].get(qb)
        seq = [start] + [tr[i * per + idx] for i in range(len(tr) // per)]
        counts[sum(1 for a, b in zip(seq, seq[1:]) if a != b)] += 1
    if not counts:
        return {"counts": {}, "n": 0}
    flat = sorted(x for v, c in counts.items() for x in [v] * c)
    return {"counts": dict(sorted(counts.items())), "n": len(flat), "min": flat[0],
            "max": flat[-1], "median": flat[len(flat) // 2],
            "mean": round(sum(flat) / len(flat), 3)}


def s5_bind_v3_ckpt_lag(examples, j: int) -> float | None:
    """ORACLE-ASSISTED: the true value of the queried slot ``j`` events before the end.

    This is "copy the previous checkpoint's slot" with the TRUE trace handed to the copier, which
    is the only form of it that can score above zero — the self-referential form never moves and
    is ``ckpt_copy_prev``. It is reported and never admitted: on a state or composed cell it
    replays L - j events and is depth L - j, and on a retrieval cell it pays the whole scan to
    the pinned write and then some, so it is at or above that cell's own algorithm's cost.

    Its shape is the point. On the retrieval component the sampler pins the resolving write at or
    below 0.75L, so the queried slot has been constant for the last quarter of the stream and
    every small j reads 1.000; on the state and composed cells the gate pulls the last move into
    the final tenth, so j = 1 already loses a quarter of the items.
    """
    from .composition import answer_of, read, replay

    hits = n = 0
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        L = len(rec["events"])
        if j >= L:
            continue
        n += 1
        hits += int(answer_of(rec, replay(rec, drop=(L - j, L))) == e.answer)
    return hits / n if n else None


def s5_bind_v3_ckpt_preds(prompt: str) -> dict[str, str | None]:
    """The CHECKPOINT-SHAPED cheap policies' answers for one prompt, in rendered form.

    These are the policies a checkpoint-slot read makes available that an answer read does not
    have to price, written as predictions of the same gold (T1):

      ckpt_copy_prev           emit the previous checkpoint at every event, so the trace never
                               moves; its final queried slot is the STATED initial value. Depth
                               0, and the query gate forces it to 0.000.
      ckpt_last_event_operand  the LAST event's second operand, resolved one hop through the
                               stated block. The last event needs no search (D2: it is located
                               positionally, not by another event's output), so this is the
                               cheapest one-hop row on the cell.
      ckpt_last_event_target   the LAST event's first operand — the slot it writes — emitted as
                               the answer.
    """
    from .composition import read

    rec = read(prompt)
    if rec is None or not rec["events"]:
        return {n: None for n in S5_BIND_V3_CKPT_ROWS}
    kind, target = rec["query"]
    if kind == "state":
        init = rec["P0"].get(target)
    elif kind == "bind":
        init = rec["B0"].get(target)
    else:
        init = None
    ekind, etgt, eref, esrc = rec["events"][-1]
    if esrc == "N":
        operand = eref
    else:
        operand = (rec["P0"] if esrc == "P" else rec["B0"]).get(eref)
    return {"ckpt_copy_prev": None if init is None else f"{init}.",
            "ckpt_last_event_operand": None if operand is None else f"{operand}.",
            "ckpt_last_event_target": None if etgt is None else f"{etgt}."}


def s5_bind_v3_ckpt_floors(examples) -> dict[str, float]:
    """Every checkpoint-shaped row's accuracy on a cell's exact items, keyed by row name.

    Merged with ``s5_bind_v3_floors`` by every caller measuring a GUIDED floor; each row is then
    admitted or excluded on its own cost like any other. Kept in its own tuple rather than added
    to ``S5_BIND_V3_ROWS`` so the PLAIN protocol's floors are untouched by rows that were added
    for a protocol which did not exist when they were measured.
    """
    n = len(examples)
    if not n:
        return {}
    hits: Counter = Counter()
    defined: Counter = Counter()
    for e in examples:
        preds = s5_bind_v3_ckpt_preds(e.prompt)
        for nm in S5_BIND_V3_CKPT_ROWS:
            if preds[nm] is not None:
                defined[nm] += 1
                hits[nm] += int(preds[nm] == e.answer)
    return {nm: hits[nm] / n for nm in S5_BIND_V3_CKPT_ROWS if defined[nm]}


def s5_bind_v3_guided_admits(row: str, k: int, m: int, n_swap: int, n_give: int,
                             named: bool = False, query: str = "state", weights=None) -> bool:
    """Whether ONE row may set a floor under a SCRATCHPAD protocol — the class rule with the W
    axis removed. It is the class for BOTH of that protocol's reads, because what removes the axis
    is the format and not the channel the prediction is read from.

    T2: the guided protocol requires the whole of P and B to be emitted at every event, so a
    bound on live slots prices a resource every policy has been handed. T3: depth and steps are
    unchanged by a pad. So this is ``s5_bind_v3_admits`` with the first check dropped.

    On a COMPONENT cell that changes nothing — every depth <= 1 row already costs W = 2 — and the
    function is the registered rule. On a COMPOSED cell the registered class is the W bound plus
    a step bound the task satisfies, so dropping the W bound admits everything, this returns True
    for every row, and the floor is not a floor: ``s5_bind_v3_operative_floor(..., guided=True)``
    returns None there.
    """
    if not named:
        return True
    if row == "surface_ranker":
        impl = s5_bind_v3_surface_price(k, m, n_swap, n_give, named, query, weights)
        return (impl["S"] < s5_bind_v3_task_cost_min(k, m, n_swap, n_give, named, query)
                and s5_bind_v3_surface_depth(weights) <= S5_BIND_V3_MAX_DEPTH)
    _w, s = s5_bind_v3_row_cost(row, k, m, n_swap, n_give, query, named, weights)
    depth = s5_bind_v3_row_depth(row, query, n_swap + n_give, weights)
    return (depth <= S5_BIND_V3_MAX_DEPTH
            and s < s5_bind_v3_task_cost_min(k, m, n_swap, n_give, named, query))


def s5_bind_v3_trace_operative_floor(floors: dict[str, float], k: int, m: int,
                                     n_swap: int, n_give: int, named: bool = False,
                                     query: str = "state") -> float | None:
    """The number a TRACE score has to clear at a COMPONENT cell, or None at a composed one.

    The trace read exists under no protocol but the guided one, so this is
    ``s5_bind_v3_operative_floor`` with ``guided=True`` fixed and is kept as a named entry point
    rather than as a second rule. The guided ANSWER read is the same call with the same flag.

    None is not "no rows were measured": it is the composed cell's answer, and it is returned so
    that ``clears()`` refuses the cell rather than reading a max over a class that contains the
    task. ``s5_bind_v3_trace_floor_basis`` says which case a None is, and ``s5_bind_v3_pad_reach``
    measures how far above the plain protocol's floor the unfloorable class reaches, so the gap is
    a number rather than an absence.
    """
    return s5_bind_v3_operative_floor(floors, k, m, n_swap, n_give, named, query, guided=True)


def s5_bind_v3_trace_floor_basis(k: int, m: int, n_swap: int, n_give: int, named: bool = False,
                                 query: str = "state") -> str:
    """'measured' / 'chance' / 'unfloorable' for the TRACE read at one cell —
    ``s5_bind_v3_floor_basis`` with ``guided=True`` fixed, for the same reason the floor above is.

    'unfloorable' is the COMPOSED cell's case and the reason is T2 above: the protocol supplies
    the live slots its floor argument is made of. The other two labels mean what they mean under
    the plain protocol, because on a component cell the guided class IS the plain class — and,
    like that one, this label is about which REGISTERED rows the rule lets set the number. The
    swept families and the checkpoint-shaped rows are measured beside them and enter the max; they
    do not move the label, so the retrieval component's 'chance' stays the proof it is (no
    admitted budget reaches the write the sampler pinned) rather than flipping to 'measured'
    because a row that reads the stream's last event was added and came out under a guess.
    """
    return s5_bind_v3_floor_basis(k, m, n_swap, n_give, named, query, guided=True)


def s5_bind_v3_pad_reach(examples, widths=S5_BIND_V3_PAD_WIDTHS,
                         positions=(0.0, 0.2, 0.4, 0.6, 0.8, 0.95)) -> float | None:
    """What the UNFLOORABLE class reaches on a composed cell, measured rather than left blank.

    The best member of the block-drop continuum: carry both maps, skip one block of events, play
    the rest exactly. Every member holds W = k + m + 1 and is excluded from the plain protocol's
    floor on exactly that; a scratchpad protocol hands those slots out (T2), so every member is
    available to a policy decoding under it and each costs strictly fewer steps than the cell's
    own algorithm. It is the same number on both of that protocol's reads, because the policies
    it maxes over are the same policies.

    IT IS NOT A FLOOR AND MUST NOT BE PRINTED AS ONE — the class it comes from also contains the
    task, so its max is 1.000 and this is a lower bound on that max. It exists so the distance
    between the plain protocol's floor and the unfloorable class is a number.
    """
    if not examples:
        return None
    vals = [s5_bind_v3_block_drop(examples, w, p) for w in widths for p in positions]
    return max(vals) if vals else None


def s5_bind_v3_ckpt_copy_per_slot(examples, k: int, m: int, agents, objs) -> float | None:
    """The per-SLOT accuracy of emitting the previous checkpoint at every event.

    The reference the per-slot checkpoint diagnostic has to be read against, and it is nowhere
    near uniform: a swap moves 2 of the k + m slots and a give moves 1, so a copier is right on
    ``1 - (2 n_swap + n_give) / ((k + m) L)`` of them — 0.83 on a state cell, 0.92 on a
    retrieval one and 0.89 on the k=6 composed cell, against 1/k = 0.167. A per-slot number below
    that reference is BELOW a policy that never updates anything.

    The first checkpoint has no predecessor in the emitted stream and is scored against the
    STATED block, which is the pad a copier actually starts from; a component cell states only
    the half it moves, so the other half's first checkpoint counts against it.
    """
    from .composition import read

    hits = tot = 0
    per = k + m
    for e in examples:
        rec = read(e.prompt)
        tr = e.meta.get("trace")
        if rec is None or not tr:
            continue
        toks = tr.split()
        if not toks or len(toks) % per:
            continue
        blocks = [toks[i * per:(i + 1) * per] for i in range(len(toks) // per)]
        stated = [rec["P0"].get(a) for a in agents] + [rec["B0"].get(o) for o in objs]
        for i, b in enumerate(blocks):
            prev = blocks[i - 1] if i else stated
            for j, v in enumerate(b):
                tot += 1
                hits += int(prev[j] == v)
    return hits / tot if tot else None


# ===========================================================================================
# THE PAD-WRITE READ — the same class rule, scored on the PAD instead of on the answer
# ===========================================================================================
# WHY IT EXISTS. Under a bounded pad the composed ANSWER is generated from the model's own pad, and
# a per-item perfect pad makes that context byte-identical to the gold-pad one: pad good implies
# answer clears, so the answer axis registers the PAD WRITE and cannot register a composition gap.
# The quantity that can still separate the cells is therefore the pad write itself, per SLOT, and a
# per-slot score means nothing without a class of cheap policies to read it against.
#
# THE CLASS IS THE REGISTERED ONE, transferred to the new quantity rather than invented for it:
#
#   LIVE SLOTS   W - pad <= max(k, m) + 1, exactly as ``s5_bind_v3_pad_admits`` prices the answer
#                read. A pad of ``pad`` slots is ``pad`` free live slots (W3), so at k = m = 6 and
#                pad 2 a row may hold 8 of the 12 map cells and the composed cell's own algorithm
#                (12 cells + scratch) is excluded. THIS IS WHAT THE BOUNDED PAD COSTS: on the
#                ANSWER a partial carry buys almost nothing (the pad-2 answer floor is the plain
#                one, 1.05-1.17x chance), while on the PAD most tokens are one-hop reads of cells a
#                partial carry holds, so the same rule prices the two reads completely differently.
#   STEPS        S <= the cost of the cell's own algorithm PRODUCING THE PAD
#                (``s5_bind_v3_pad_write_task_cost``), which is the answer algorithm plus the one
#                read per give that the displaced value costs. This is the conjunct that excludes
#                a per-event BACKWARD SCAN: ``pad_scan_last_write`` recovers the true holder of a
#                cross swap's reference exactly, and its per-event cost is L-INDEPENDENT (the last
#                give writing one of m objects sits ~m / p_give events back), so an
#                L-independence rule alone would admit it. It is excluded because 2 m / p_give
#                steps per swap is ~3x what the algorithm pays per swap, and it is measured and
#                printed rather than argued away (``s5_bind_v3_pad_scan_last_write``).
#   HOPS         reported at two settings, because a per-slot read is scored token by token and
#                one of the four tokens is the composition. ``max_hops=None`` is the registered
#                composed-cell rule (no depth conjunct: the gap axis there is W) and is the
#                OPERATIVE floor, since it is the LARGER class and so the higher bar.
#                ``max_hops=1`` is the sub-class that may not itself resolve-then-read, and it is
#                the bar the two-hop token would have to clear if the depth conjunct were carried
#                over from the component rule. Excluding rows lowers a floor, so the smaller class
#                is never the operative one.
#
# WHAT A ROW IS. A carry ``pad_carry_P{jP}B{jB}_{evict}`` holds jP pointer cells and jB holder
# cells live, allocated on first write (``first``, the registry's rule for ``partial_carry_j``) or
# with the least-recently-written cell evicted (``recent``, which is what a policy spending its
# slots on the pad's own adjacent block does); every uncached read hits the STATED block (W4). A
# row emits one token per pad slot and may choose that emission per (event kind, block position) —
# both are on the surface and neither costs a slot — so the emission codes are swept per cell and
# the row's score is the best admitted code there. ``const`` and ``copy_prev`` are available to
# every carry for the same reason, which is the most permissive reading and therefore the
# conservative one for a floor.
#
# THE EMISSION IS CHOSEN PER (kind, position, SOURCE), and the source conjunct is the one this
# read turns on. Which structure resolves an event's operand is on the SURFACE — "the agent o5
# belongs to" and "the agent g3 points to" are different clauses — so switching emission on it
# costs a row nothing it does not already pay to read the event, and a class that may not switch
# is a class with an admitted policy left out of it. Leaving one out lowers a floor, which is the
# direction that invalidates a cleared reading, so the finer partition is the registered one and
# the coarse (kind, position) figure is reported beside it as the lower number it is.
#
# IT ALSO SPLITS THE TWO-HOP TOKEN INTO THE TWO THINGS IT WAS POOLING. On a composed stream a swap
# resolves its operand through the HOLDER map (cross) or through the POINTER map (same), and
# ``swap_p0`` is two dependent reads either way — but the same-source one is two reads of P alone,
# which a row holding P alone performs exactly, and it is the STATE component's own carrier depth
# rather than anything composed. The cross one needs both maps and is the only token on this rung
# whose write is the composition. They are floored and reported apart for that reason.
#
# THE CHEAP POLICIES A PAD-WRITE READ MAKES AVAILABLE, and where each one is:
#   copy the previous checkpoint's slot      ``copy_prev`` — the row emits its own last block, so
#                                            the pad never moves; the first block is its stated-map
#                                            read, which is the strongest reading of it.
#   apply only the one-hop part of the event ``operand`` at swap_p0 — the resolved operand written
#                                            without the second read — and every code the
#                                            ``max_hops=1`` sub-class admits.
#   resolve against the STATED map           ``stated_ref_val`` / ``stated_target_val`` /
#                                            ``stated_operand_val``, and every carry with cells
#                                            left over, which reads the rest of both maps out of
#                                            the header block (W4).
#   emit the identity map                    ``operand`` at position 0 with ``target_sym`` at
#                                            position 1: under P = identity a swap of (tgt, x)
#                                            writes (x, tgt), and a row picks its emission per
#                                            position, so the combination is one admitted policy.
S5_BIND_V3_PAD_CELLS = ("give_p0", "give_p1", "swap_p0", "swap_p1")

# THE SOURCE CLASSES a pad cell is partitioned by, and they are the event's own, read off the
# clause: 'named' the operand is written out, 'same' it is resolved through the structure this
# event WRITES, 'cross' through the other one (``composition.is_cross``, the registered reading).
S5_BIND_V3_PAD_SOURCES = ("named", "same", "cross")

S5_BIND_V3_PAD_CODES = ("own_gold", "operand", "target_val", "target_sym", "ref_sym",
                        "stated_ref_val", "stated_target_val", "stated_operand_val",
                        "copy_prev", "const", "uniform")

# HOPS PER (code, cell), AS THE TWO READS THEY ARE MADE OF, so the count is a property of the
# EVENT and not of the cell's name: ``(resolves_the_operand, map_reads)``. An emission's depth is
# ``map_reads`` plus the operand resolve where the event HAS one — a named operand is on the
# surface and is not a read — which is why the same ``own_gold`` at ``swap_p0`` is TWO hops on a
# composed stream (resolve, then read the pointer map) and ONE on a component stream, whose
# operands are all named. Charging the composed number on a component cell would exclude that
# cell's own one-hop policy from its own floor and manufacture a clear out of the bookkeeping.
S5_BIND_V3_PAD_HOP_PARTS = {
    "own_gold": {"give_p0": (1, 0), "give_p1": (0, 1), "swap_p0": (1, 1), "swap_p1": (0, 1)},
    "operand": {c: (1, 0) for c in S5_BIND_V3_PAD_CELLS},
    "target_val": {c: (0, 1) for c in S5_BIND_V3_PAD_CELLS},
    "stated_ref_val": {c: (1, 0) for c in S5_BIND_V3_PAD_CELLS},
    "stated_target_val": {c: (0, 1) for c in S5_BIND_V3_PAD_CELLS},
    "stated_operand_val": {c: (1, 1) for c in S5_BIND_V3_PAD_CELLS},
    "target_sym": {c: (0, 0) for c in S5_BIND_V3_PAD_CELLS},
    "ref_sym": {c: (0, 0) for c in S5_BIND_V3_PAD_CELLS},
    "copy_prev": {c: (0, 0) for c in S5_BIND_V3_PAD_CELLS},
    "const": {c: (0, 0) for c in S5_BIND_V3_PAD_CELLS},
    "uniform": {c: (0, 0) for c in S5_BIND_V3_PAD_CELLS},
}


def s5_bind_v3_pad_hops(code: str, cell: str, source: str = "cross") -> int:
    """The emission's COMPOSITION DEPTH at one pad cell on an event of this source class.

    ``S5_BIND_V3_PAD_HOP_PARTS`` split into the operand resolve and the map reads that follow it;
    a NAMED operand costs no resolve. This is the quantity the component cells' depth conjunct is
    applied to, per emitted token (``s5_bind_v3_pad_write_admits``).
    """
    if code not in S5_BIND_V3_PAD_HOP_PARTS or cell not in S5_BIND_V3_PAD_CELLS:
        raise KeyError(f"{code!r}/{cell!r} names no priced pad-write emission")
    if source not in S5_BIND_V3_PAD_SOURCES:
        raise KeyError(f"{source!r} is not one of {S5_BIND_V3_PAD_SOURCES}")
    res, reads = S5_BIND_V3_PAD_HOP_PARTS[code][cell]
    return reads + (res if source != "named" else 0)


def s5_bind_v3_pad_event_source(kind: str, src: str) -> str:
    """The source class of one event, from its kind and the structure that resolves its operand."""
    from .composition import is_cross

    if src == "N":
        return "named"
    return "cross" if is_cross((kind, None, None, src)) else "same"


# HOPS PER (code, cell) on a COMPOSED stream, where every operand is resolved. Kept as the table
# the reports print; it is DERIVED from the parts above rather than written down twice.
S5_BIND_V3_PAD_HOPS = {code: {cell: s5_bind_v3_pad_hops(code, cell, "cross")
                              for cell in S5_BIND_V3_PAD_CELLS}
                       for code in S5_BIND_V3_PAD_CODES}
S5_BIND_V3_PAD_EVICT = ("first", "recent")


def s5_bind_v3_pad_gold(rec) -> list[tuple[str, str]] | None:
    """The gold ``moved2`` pad for one item: the post-event values of the slots each event MOVED.

    A swap of ``(tgt, x)`` writes ``[P[tgt], P[x]]`` after the swap — token 0 is the operand read
    through the pointer map (two hops on a composed stream) and token 1 is the displaced pointer
    cell (one). A give of ``(o, x)`` writes ``[x, old B[o]]`` — the resolved operand and the
    displaced holder, one hop each.

    Replayed off the PROMPT and sharing no code with the format the training documents are built
    from, so the two are a check on each other rather than one function read twice.
    """
    from .composition import SWAP

    Pm, Bm = dict(rec["P0"]), dict(rec["B0"])
    out = []
    for kind, tgt, ref, src in rec["events"]:
        x = ref if src == "N" else (Pm.get(ref) if src == "P" else Bm.get(ref))
        if x is None:
            return None
        if kind == SWAP:
            if tgt not in Pm or x not in Pm:
                return None
            Pm[tgt], Pm[x] = Pm[x], Pm[tgt]
            out.append((Pm[tgt], Pm[x]))
        else:
            disp = Bm.get(tgt)
            if disp is None:
                return None
            Bm[tgt] = x
            out.append((x, disp))
    return out


def s5_bind_v3_pad_carry_rows(k: int, m: int, pad: int) -> tuple[str, ...]:
    """Every carry the live-slot rule admits at this cell and pad width, both eviction rules.

    ``W = 1 + jP + jB`` (the scratch register and the cells held), so the rule
    ``W - pad <= max(k, m) + 1`` is ``jP + jB <= max(k, m) + pad``. The sweep is the whole
    admitted grid and not a chosen point: at k = m = 6 and pad 2 that is every split of 8 cells
    between the two maps, including the registry's own ``partial_carry_j`` line (jP = k).
    """
    lim = one_structure_bound(k, m) + pad - 1
    return tuple(f"pad_carry_P{jp}B{jb}_{ev}"
                 for jp in range(k + 1) for jb in range(m + 1)
                 if jp + jb <= lim for ev in S5_BIND_V3_PAD_EVICT)


def s5_bind_v3_pad_carry_parse(row: str) -> tuple[int, int, str]:
    """``(jP, jB, evict)`` for a carry row name. Raises on anything else."""
    import re as _re

    mt = _re.fullmatch(r"pad_carry_P(\d+)B(\d+)_(first|recent)", row)
    if mt is None:
        raise KeyError(f"{row!r} is not a pad-write carry row")
    return int(mt.group(1)), int(mt.group(2)), mt.group(3)


def s5_bind_v3_pad_write_task_cost(k: int, m: int, n_swap: int, n_give: int) -> tuple[int, int]:
    """``(W, S)`` for the composed cell's own algorithm PRODUCING THE PAD.

    The answer algorithm (``s5_bind_v3_task_cost``) plus the one read per give that the displaced
    holder costs: the pad asks for a value the write is about to overwrite, and carrying both maps
    does not hand it over for free. The swap term is unchanged — the two tokens a swap emits are
    the two cells it has just written.
    """
    w, s = s5_bind_v3_task_cost(k, m, n_swap, n_give, named=False, query="state")
    return w, s + n_give


def s5_bind_v3_pad_write_cost(row: str, k: int, m: int, n_swap: int, n_give: int) -> tuple[int, int]:
    """``(W, S)`` for one pad-write row, under the same step convention as every other row.

    A carry pays the header block once, then per event: one E, one R to resolve the operand, at
    most two R for the two values it emits and at most two M for the cells it writes. That is 6 on
    a swap and 4 on a give, against the pad-producing algorithm's 6 and 4 — the carries tie on
    steps and are separated from the task on live slots alone, which is the axis the composed
    cell's rule puts the gap on.

    ``pad_scan_last_write`` is priced here and admitted by nothing: it pays a backward scan per
    cross event, and although that scan's length is L-INDEPENDENT it is ~2 m / p_give steps where
    the algorithm pays 6.
    """
    if row == "pad_scan_last_write":
        L = max(1, n_swap + n_give)
        reach = 2 * int(round(m * L / max(1, n_give)))     # E + C per event the scan passes
        return 1 + k, (k + m) + (6 + reach) * n_swap + 4 * n_give + 1
    if row in ("pad_uniform", "pad_const", "pad_copy_prev"):
        return 1, (k + m) + 2 * (n_swap + n_give) + 1
    jp, jb, _ev = s5_bind_v3_pad_carry_parse(row)
    return 1 + jp + jb, (k + m) + 6 * n_swap + 4 * n_give + 1


def s5_bind_v3_pad_write_admits(row: str, code: str, cell: str, k: int, m: int,
                                n_swap: int, n_give: int, pad: int = 0,
                                max_hops: int | None = None, source: str = "cross") -> bool:
    """Whether one ``(row, code)`` may set the floor for one pad cell on one source class.

    ``max_hops=None`` is the composed cell's own W-axis rule — live slots and steps, no depth
    conjunct — and it is the floor for the THREE ONE-HOP TOKENS. An integer adds the DEPTH
    conjunct per emitted token: ``s5_bind_v3_pad_hops(code, cell, source) <= max_hops``, which is
    ``floor_eligible``'s ``depth_row <= depth_max`` — the conjunct that sets both COMPONENT cells'
    floors (``s5_bind_v3_admits``, named branch) — read on the emission instead of on the whole
    policy. At ``max_hops = S5_BIND_V3_MAX_DEPTH`` it is the class that may not itself perform the
    two-hop resolve, and that is the registered floor for the two-hop token
    (``s5_bind_v3_pad_two_hop_floor``).
    """
    if max_hops is not None and s5_bind_v3_pad_hops(code, cell, source) > max_hops:
        return False
    w, s = s5_bind_v3_pad_write_cost(row, k, m, n_swap, n_give)
    _wt, st = s5_bind_v3_pad_write_task_cost(k, m, n_swap, n_give)
    return floor_eligible(w - pad, s, one_structure_bound(k, m), st)


def _pad_write_item(rec, gold, jp, jb, evict, acc, agents, forced=False):
    """One item, one carry: every emission code scored against the gold pad, per (cell, source).

    ``acc`` is keyed by ``(cell, source)`` because a row may switch emission on the source class,
    which is on the surface of the event line it already reads.

    The carry's beliefs are its own — an uncached cell reads the STATED block — so a free-running
    row is closed-loop exactly as the model's free-running read is. ``forced`` models the
    TEACHER-FORCED read: the gold block adjacent to the next event is readable at O(1) (W5), so
    after each event the row refreshes the slot that event NAMES to its true value where it holds
    it. Only that one: a swap's second gold token is the value now at the RESOLVED operand, and
    which slot that is is exactly the two-hop fact the row does not have.
    """
    from .composition import SWAP

    P0, B0 = rec["P0"], rec["B0"]
    Pp = dict(P0) if jp >= len(P0) else {}
    Bp = dict(B0) if jb >= len(B0) else {}
    ageP: dict = {}
    ageB: dict = {}
    clock = [0]
    prev = None

    def hold(store, age, cap, key, val):
        """Write ``key`` if the row holds it, allocating or evicting under its own rule."""
        clock[0] += 1
        if key not in store:
            if len(store) < cap:
                store[key] = val
            elif evict == "recent" and cap > 0:
                old = min(store, key=lambda z: age.get(z, -1))
                store.pop(old, None)
                age.pop(old, None)
                store[key] = val
            else:
                return
        else:
            store[key] = val
        age[key] = clock[0]

    for i, (kind, tgt, ref, src) in enumerate(rec["events"]):
        bel_p = lambda a: Pp.get(a, P0.get(a))            # noqa: E731
        bel_b = lambda o: Bp.get(o, B0.get(o))            # noqa: E731
        x = ref if src == "N" else (bel_p(ref) if src == "P" else bel_b(ref))
        x_stated = ref if src == "N" else (P0.get(ref) if src == "P" else B0.get(ref))
        if kind == SWAP:
            names = ("swap_p0", "swap_p1")
            em = {"own_gold": (bel_p(x), bel_p(tgt)),
                  "operand": (x, x),
                  "target_val": (bel_p(tgt), bel_p(tgt)),
                  "target_sym": (tgt, tgt),
                  "ref_sym": (ref, ref) if src == "P" else (None, None),
                  "stated_ref_val": (x_stated, x_stated),
                  "stated_target_val": (P0.get(tgt), P0.get(tgt)),
                  "stated_operand_val": (P0.get(x_stated), P0.get(x_stated))}
        else:
            names = ("give_p0", "give_p1")
            em = {"own_gold": (x, bel_b(tgt)),
                  "operand": (x, x),
                  "target_val": (bel_b(tgt), bel_b(tgt)),
                  "target_sym": (None, None),
                  "ref_sym": (ref, ref) if src == "P" else (None, None),
                  "stated_ref_val": (x_stated, x_stated),
                  "stated_target_val": (B0.get(tgt), B0.get(tgt)),
                  "stated_operand_val": (None, None)}
        em["copy_prev"] = prev if prev is not None else em["own_gold"]
        prev = em["copy_prev"]
        g = gold[i]
        source = s5_bind_v3_pad_event_source(kind, src)
        for p in (0, 1):
            slot = acc[(names[p], source)]
            for code, vals in em.items():
                if vals[p] is not None and vals[p] == g[p]:
                    slot[code][0] += 1
                slot[code][1] += 1
            for a in agents:
                slot["const"][a][0] += int(a == g[p])
                slot["const"][a][1] += 1
        # THE ROW'S OWN UPDATE, then — teacher-forced only — the adjacent gold block. Only the
        # slot the surface NAMES is refreshed: a swap's gold block is (value now at tgt, value now
        # at the resolved operand), and which slot the second token belongs to is exactly the
        # two-hop fact the row does not have. A give names the object it writes, so both of its
        # tokens are attributable and the first is the new holder.
        if kind == SWAP:
            vt, vx = bel_p(tgt), bel_p(x)
            hold(Pp, ageP, jp, tgt, vx)
            hold(Pp, ageP, jp, x, vt)
            if forced and tgt in Pp:
                Pp[tgt] = g[0]
        else:
            hold(Bp, ageB, jb, tgt, x)
            if forced and tgt in Bp:
                Bp[tgt] = g[0]


def s5_bind_v3_pad_write_scores(examples, k: int, m: int, pad: int = 2, rows=None,
                                forced: bool = False) -> dict:
    """Every admitted pad-write row's accuracy on one cell's exact items, per pad cell and source.

    Returns ``{"n", "counts": {cell: slots}, "rows": {row: {cell: {code: acc}}}}`` — pooled over
    source classes, which is what the coarse (kind, position) reading needs — plus
    ``{"counts_src": {"cell|source": slots}, "rows_src": {row: {"cell|source": {code: acc}}}}``,
    which is the partition the floor is registered on. ``const`` is already reduced to the best
    fixed agent IN EACH PARTITION and ``uniform`` carries 1 / k. Rows are
    ``s5_bind_v3_pad_carry_rows`` unless given.

    ``forced`` PUTS THE CLASS ON THE SAME READ AS THE MODEL, and which read that is decides what
    the comparison measures. Free-running, both the model and every row are closed-loop: one wrong
    slot corrupts the context, so the pooled number is a per-event error rate COMPOUNDED over the
    stream and falls with L on a model whose per-event behaviour is flat. That comparison answers
    "can this hold its own pad for 2L tokens", which is state capacity. Teacher-forced, each event
    is scored once against the true history — the model reads the gold pad, and a row refreshes
    the slot the surface NAMES from the adjacent gold block — and the comparison answers "given
    the state, is the two-hop write performed", which is the composition question.

    WHAT THE TEACHER-FORCED NUMBER THEREFORE CANNOT CLAIM: anything about behaviour on the model's
    own writes, at any length, end to end. A cell that clears teacher-forced and floors
    free-running is a model that computes the composed update and cannot survive its own errors.
    Both are measured and the protocol registers the first as the score and the second as the
    tracking diagnostic; neither substitutes for the other.
    """
    from .composition import SWAP, read

    rows = tuple(rows) if rows is not None else s5_bind_v3_pad_carry_rows(k, m, pad)
    prepped = []
    agents = None
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        g = s5_bind_v3_pad_gold(rec)
        if g is None:
            continue
        prepped.append((rec, g))
        if agents is None:
            # the pad's alphabet: both maps take AGENT values, and a component cell states only
            # the structure it moves, so the pool is read off the values and not off P0's keys
            agents = sorted(set(rec["P0"]) | set(rec["P0"].values()) | set(rec["B0"].values()))
    if not prepped:
        return {"n": 0, "counts": {}, "counts_src": {}, "rows": {}, "rows_src": {}}
    counts, counts_src = Counter(), Counter()
    for rec, _g in prepped:
        for kind, _t, _r, src in rec["events"]:
            pos = ("swap_p0", "swap_p1") if kind == SWAP else ("give_p0", "give_p1")
            counts["swap" if kind == SWAP else "give"] += 1
            for c in pos:
                counts_src[f"{c}|{s5_bind_v3_pad_event_source(kind, src)}"] += 1
    parts = tuple((c, s) for c in S5_BIND_V3_PAD_CELLS for s in S5_BIND_V3_PAD_SOURCES)
    out = {"n": len(prepped), "forced": bool(forced),
           "counts": {"give_p0": counts["give"], "give_p1": counts["give"],
                      "swap_p0": counts["swap"], "swap_p1": counts["swap"]},
           "counts_src": {f"{c}|{s}": counts_src[f"{c}|{s}"] for c, s in parts},
           "rows": {}, "rows_src": {}}
    for row in rows:
        jp, jb, ev = s5_bind_v3_pad_carry_parse(row)
        acc = {p: {code: [0, 0] for code in S5_BIND_V3_PAD_CODES if code != "const"}
               for p in parts}
        for p in acc:
            acc[p]["const"] = {a: [0, 0] for a in agents}
        for rec, g in prepped:
            _pad_write_item(rec, g, jp, jb, ev, acc, agents, forced=forced)

        def reduce(pairs):
            """``{code: accuracy}`` over one or more partitions' raw counts."""
            d = {}
            for code in S5_BIND_V3_PAD_CODES:
                if code == "uniform":
                    d["uniform"] = 1.0 / k
                elif code == "const":
                    per = {a: [0, 0] for a in agents}
                    for p in pairs:
                        for a, v in acc[p]["const"].items():
                            per[a][0] += v[0]
                            per[a][1] += v[1]
                    best = max(per.values(), key=lambda z: (z[0] / z[1]) if z[1] else -1.0)
                    d["const"] = (best[0] / best[1]) if best[1] else None
                else:
                    h = sum(acc[p][code][0] for p in pairs)
                    t = sum(acc[p][code][1] for p in pairs)
                    d[code] = (h / t) if t else None
            return d

        out["rows"][row] = {c: reduce([(c, s) for s in S5_BIND_V3_PAD_SOURCES])
                            for c in S5_BIND_V3_PAD_CELLS}
        out["rows_src"][row] = {f"{c}|{s}": reduce([(c, s)]) for c, s in parts}
    return out


def s5_bind_v3_pad_write_floor(scores: dict, k: int, m: int, n_swap: int, n_give: int,
                               pad: int = 2, max_hops: int | None = None,
                               by_source: bool = True) -> dict:
    """The number a pad-write score has to clear, per partition, per cell and pooled over slots.

    Every figure is a max over ROWS of that row's own count-weighted mean, never a max per cell
    recombined across rows: two rows whose cells were combined would hold both structures between
    them, which is the policy the class exists to exclude.

    ``by_source`` partitions each cell by the event's source class and lets a row choose its
    emission per partition, which the surface pays for and no slot does. It is the REGISTERED
    reading and it can only raise a cell's floor; ``by_source=False`` reproduces the coarse
    (kind, position) figure, which is reported as the lower number it is.

    Returns the per-cell and pooled floors plus ``parts``/``part_rows`` keyed ``"cell|source"``,
    which is where the two-hop token's cross and same halves are read apart.
    """
    per_cell: dict = {c: (None, None) for c in S5_BIND_V3_PAD_CELLS}
    per_part: dict = {}
    pooled = (None, None)
    tot = sum(scores["counts"].values())
    src_counts = scores.get("counts_src") or {}
    groups = {c: ([f"{c}|{s}" for s in S5_BIND_V3_PAD_SOURCES if src_counts.get(f"{c}|{s}")]
                  if by_source and src_counts else [])
              for c in S5_BIND_V3_PAD_CELLS}
    for row in scores["rows"]:
        w_sum, ok = 0.0, True
        for c in S5_BIND_V3_PAD_CELLS:
            if not scores["counts"].get(c):
                continue                          # a cell this stream has no events for
            keys = groups[c] or [c]
            cell_hits = 0.0
            picked = []
            for key in keys:
                if "|" in key:
                    cell_, source = key.split("|")
                    vals = scores["rows_src"][row][key]
                    n_key = src_counts[key]
                else:
                    # the coarse reading has no partition to price against, so it charges the
                    # CHEAPEST source class the stream actually contains: a named operand costs no
                    # resolve, and charging a resolve the events do not pay would exclude an
                    # admitted policy, which lowers a floor.
                    cell_, vals = c, scores["rows"][row][c]
                    source = ("named" if src_counts.get(f"{c}|named") else "cross")
                    n_key = scores["counts"][c]
                cand = [(v, code) for code, v in vals.items() if v is not None
                        and s5_bind_v3_pad_write_admits(row, code, cell_, k, m, n_swap, n_give,
                                                        pad, max_hops, source)]
                if not cand:
                    ok = False
                    continue
                best = max(cand)
                if "|" in key and (key not in per_part or best[0] > per_part[key][0]):
                    per_part[key] = (best[0], f"{row}:{best[1]}")
                picked.append(f"{source}:{best[1]}" if "|" in key else best[1])
                cell_hits += n_key * best[0]
            if not ok:
                continue
            cell_acc = cell_hits / scores["counts"][c]
            if per_cell[c][0] is None or cell_acc > per_cell[c][0]:
                per_cell[c] = (cell_acc, f"{row}:{'/'.join(picked)}")
            w_sum += cell_hits
        if ok and tot and (pooled[0] is None or w_sum / tot > pooled[0]):
            pooled = (w_sum / tot, row)
    return {"per_slot": pooled[0], "per_slot_row": pooled[1],
            "cells": {c: per_cell[c][0] for c in S5_BIND_V3_PAD_CELLS},
            "cell_rows": {c: per_cell[c][1] for c in S5_BIND_V3_PAD_CELLS},
            "parts": {p: v[0] for p, v in sorted(per_part.items())},
            "part_rows": {p: v[1] for p, v in sorted(per_part.items())},
            "chance": 1.0 / k, "pad": pad, "max_hops": max_hops, "by_source": bool(by_source),
            "n": scores.get("n"), "forced": scores.get("forced")}


S5_BIND_V3_TWO_HOP_CELL = "swap_p0"        # the only pad token whose write is two dependent reads
S5_BIND_V3_TWO_HOP_SOURCE = "cross"        # ... resolved through the OTHER structure


def s5_bind_v3_pad_two_hop_floor(scores: dict, k: int, m: int, n_swap: int, n_give: int,
                                 pad: int = 2) -> dict:
    """THE REGISTERED FLOOR FOR THE TWO-HOP TOKEN: the one-hop sub-class on the cross partition.

    WHICH CONJUNCT DOES THE WORK, and it is borrowed and not invented. ``floor_eligible`` carries
    two rules: on a COMPOSED cell the gap is on live slots (``w_row <= w_max``, steps tied to the
    task's) and on a COMPONENT cell it is on ``depth_row <= depth_max`` with
    ``depth_max = S5_BIND_V3_MAX_DEPTH = 1`` — a floor row may chain at most ONE event's contents.
    That second conjunct is what makes both component cells floorable, and it is the one applied
    here, per EMITTED TOKEN rather than per policy: an admitted row may hold up to
    ``max(k, m) + 1 + pad`` map cells and pay what the algorithm pays, and may still not itself
    perform the resolve-then-read that this token IS
    (``s5_bind_v3_pad_hops(code, cell, source) <= S5_BIND_V3_MAX_DEPTH``).

    WHY THE CROSS PARTITION AND NOT ALL OF ``swap_p0``. Both source classes of a swap write two
    dependent reads, so the depth conjunct excludes ``own_gold`` on both and the two floors come
    out close. What differs is what CLEARING one means: the same-source write is P read twice,
    which a row holding P alone does exactly and which the STATE component's carrier walk already
    chains eleven deep, so a score on it is a depth result the component cell has already
    registered. Only the cross write needs a value out of the holder map and then the pointer map,
    and pooling the two lets a model that does the first and floors on the second read as though
    it did both.

    THE DEPTH CONJUNCT IS ADDITIONAL AND NOT A REPLACEMENT: an admitted row still satisfies the
    live-slot and step conjuncts. That is what excludes the one policy a depth rule alone would
    let through — carrying the COMPOSED map ``C[o] = P[B[o]]``, which answers this token with a
    single read of C at the referenced key. C is m cells and would fit, but C cannot be maintained
    from C: a give moves one entry to the new holder's pointer and a swap moves every entry whose
    object points into the swapped pair, so keeping it needs P and B as well and the row is over
    the bound. It is not a swept row, and its exclusion rests on that argument rather than on a
    measured number.

    Returns the cross-partition floor, the same-source one beside it, and the whole
    ``max_hops=1`` reading, all on the exact items ``scores`` was measured on.
    """
    one = s5_bind_v3_pad_write_floor(scores, k, m, n_swap, n_give, pad,
                                     max_hops=S5_BIND_V3_MAX_DEPTH)
    full = s5_bind_v3_pad_write_floor(scores, k, m, n_swap, n_give, pad)
    cross = f"{S5_BIND_V3_TWO_HOP_CELL}|{S5_BIND_V3_TWO_HOP_SOURCE}"
    same = f"{S5_BIND_V3_TWO_HOP_CELL}|same"
    return {"floor": one["parts"].get(cross), "row": one["part_rows"].get(cross),
            "same_source_floor": one["parts"].get(same),
            "same_source_row": one["part_rows"].get(same),
            "pooled_swap_p0": one["cells"].get(S5_BIND_V3_TWO_HOP_CELL),
            "unrestricted_cross": full["parts"].get(cross),
            "unrestricted_cross_row": full["part_rows"].get(cross),
            "unrestricted_same": full["parts"].get(same),
            "n_cross": (scores.get("counts_src") or {}).get(cross),
            "n_same": (scores.get("counts_src") or {}).get(same),
            "max_hops": S5_BIND_V3_MAX_DEPTH, "chance": 1.0 / k,
            "one_hop": one, "unrestricted": full}


def s5_bind_v3_pad_write_chance(examples, k: int) -> dict:
    """The chance baseline for a PER-SLOT read, derived rather than borrowed from the answer.

    Every pad token is an AGENT name — the pad carries values of P (agents) and of B (agents) and
    never a key — so an uninformed guess is uniform over the k agents at 1 / k, and NOT the answer
    read's 1 / (k - 1). That number comes from the QUERY GATE, which excludes the queried slot's
    stated value; the gate constrains one slot at the end of the stream and says nothing about the
    2 L tokens of the pad. Returned with the measured marginal so a non-uniform target
    distribution is visible rather than assumed away: ``best_const`` is the best fixed agent, which
    is an admitted row at every pad width.
    """
    from .composition import read

    hits: Counter = Counter()
    tot = 0
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        g = s5_bind_v3_pad_gold(rec)
        if g is None:
            continue
        for blk in g:
            for v in blk:
                hits[v] += 1
                tot += 1
    if not tot:
        return {"uniform": 1.0 / k, "best_const": None, "n_slots": 0}
    return {"uniform": 1.0 / k, "best_const": max(hits.values()) / tot, "n_slots": tot,
            "marginal": {a: c / tot for a, c in sorted(hits.items())}}


def s5_bind_v3_pad_scan_last_write(examples, k: int, m: int) -> dict:
    """THE EXCLUDED ROW, measured: carry P in full and recover a cross swap's operand by scanning
    back to the last give that wrote the referenced object.

    It is the strongest cheap attack on the two-hop token and it is not admitted. Its per-event
    cost is L-INDEPENDENT — the last write to one of m objects sits ~m / p_give events back — so a
    rule that only asked for L-independence would let it in; ``s5_bind_v3_pad_write_cost`` prices
    the scan at 2 m / p_give steps per swap against the algorithm's 6, and the step conjunct
    excludes it. What it scores is reported so the exclusion is a judgement about cost and not
    about the number.

    The scan recovers the last give's RECIPIENT, which is itself a live reference on a
    source-structure stream, so it is resolved with the row's current pointer map and is exact only
    where that map has not moved since.
    """
    from .composition import SWAP, read

    hit = {c: [0, 0] for c in S5_BIND_V3_PAD_CELLS}
    for e in examples:
        rec = read(e.prompt)
        if rec is None:
            continue
        g = s5_bind_v3_pad_gold(rec)
        if g is None:
            continue
        P0, B0 = rec["P0"], rec["B0"]
        Pm = dict(P0)
        evs = rec["events"]
        for i, (kind, tgt, ref, src) in enumerate(evs):
            if src == "B":
                x = B0.get(ref)
                for j in range(i - 1, -1, -1):
                    kj, tj, rj, sj = evs[j]
                    if kj == SWAP or tj != ref:
                        continue
                    x = rj if sj == "N" else (Pm.get(rj) if sj == "P" else B0.get(rj))
                    break
            else:
                x = ref if src == "N" else Pm.get(ref)
            if kind == SWAP:
                pred = (Pm.get(x), Pm.get(tgt))
                cells = ("swap_p0", "swap_p1")
            else:
                pred = (x, B0.get(tgt))
                cells = ("give_p0", "give_p1")
            for p in (0, 1):
                hit[cells[p]][0] += int(pred[p] is not None and pred[p] == g[i][p])
                hit[cells[p]][1] += 1
            if kind == SWAP and x in Pm and tgt in Pm:
                Pm[tgt], Pm[x] = Pm[x], Pm[tgt]
    return {c: (v[0] / v[1] if v[1] else None) for c, v in hit.items()}


if __name__ == "__main__":
    print(_fmt(run_gate()))
