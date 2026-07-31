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
# RECENCY WINDOW (window_50 / window_75 / window_90): simulate the task EXACTLY, honouring every
# temporal phrase, but from the stated maps over only the last T = f*L events. This is a
# bounded-HORIZON policy rather than a bounded-state one, and it is registered because it is the
# largest floor the construct has: at k=12/L=64 it reads 0.294 / 0.235 / 0.180 for f =
# 0.9/0.75/0.5 against 0.143 for one_leg_B and 0.0909 for uniform-over-non-initial, and it does
# not decay with length (0.294/0.215/0.203/0.199 at L=64/128/192/256 for f=0.9). Two properties
# fix how it may be read:
#   - the family is MONOTONE in f, not exchangeable: f=1 is the oracle by construction, so the
#     max over any registered set is always its largest member and is not a selection statistic.
#     What that costs is that the registered cut is a design choice — 0.9 is registered, and a
#     policy that reads 90% of the stream is doing 90% of the work, so the resulting floor is
#     deliberately conservative.
#   - the smaller cuts are registered too and are what make the row informative: f=0.5 at 0.180
#     is what a genuinely truncated reader gets, and the gap between f=0.5 and f=0.9 is how much
#     of the stream is load-bearing.
#
# ONE-HOP AND STATED:
#   initial_only      — answer the stated initial role / holder (the no-op policy).
#   last_swap_1hop    — the stated role of the other operand of the last swap naming the queried
#                       agent, resolved off the stated holder map (state query only).
#
# CHANCE, not shortcuts: uniform_non_initial = 1/(k-1), since the query gates force the answer
# to differ from the stated one, and uniform = 1/k. For the whole-map readout the answer is a
# permutation of the k roles, so its chance row is 1/k! and there is no non-initial variant.
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
                "initial_only", "last_swap_1hop", "uniform_non_initial", "uniform")
# The rows that may SET a cell's floor. All of them: each is a named policy rather than a member
# of an exchangeable family, and the window rows are monotone in their cut (see above), so the
# max is never a selection statistic.
S5_BIND_ADVERSARIES = S5_BIND_ROWS
S5_BIND_CHANCE_ROWS = ("uniform_non_initial", "uniform")
# Rows defined only where some event is rendered "at this point". Under a fully decoupled
# rendering each of them reproduces the oracle on the query it is defined for — the first four
# by resolving against the stated maps, which IS the decoupled semantics, and pin_chain because
# a static give's recipient is the stated holder of the named role, i.e. exactly the retrieval
# component's answer. Printing 1.000 as a floor would be a correctness check wearing a floor's
# clothes, so they are dropped rather than reported.
S5_BIND_COUPLED_ONLY_ROWS = ("stale_resolution", "one_leg_B", "one_leg_P",
                             "final_state_resolution", "pin_chain")
# The registered recency-window cuts, as fractions of the stream length.
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


def _sb_run(read: dict, mode: str = "surface", start: int = 0, final=None):
    """Play ``events[start:]`` from the stated maps and return the resulting (P, B).

    mode 'surface' honours each event's rendered temporal phrase (the exact semantics);
    'stale' resolves every reference against the stated maps; 'B_only' feeds B into P but not
    P into B; 'P_only' the mirror; 'final' resolves dynamic references against ``final``, the
    true final maps.
    """
    P0, B0 = read["P0"], read["B0"]
    P, B = dict(P0), dict(B0)
    P0inv = {v: k for k, v in P0.items()}
    Pinv = dict(P0inv)
    for kind, x, y, dyn in read["events"][start:]:
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
        T = max(1, int(round(f * L)))
        out[f"window_{int(round(f * 100))}"] = _sb_answer(read, _sb_run(read, "surface",
                                                                        start=max(0, L - T)))
    out["last_swap_1hop"] = None
    if read["query"][0] == "state":
        for kind, x, y, _dyn in reversed(events):
            if kind == "swap" and x == read["query"][1]:
                partner = read["B0"].get(y)
                if partner is not None and partner in read["P0"]:
                    out["last_swap_1hop"] = f"{read['P0'][partner]}."
                break
    return {n: out.get(n) for n in names}


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

    ``coupled=False`` drops the recency-window rows. A floor row has to be CHEAPER than the
    task, and a windowed policy still maintains both maps over 0.9L events: on the coupled
    rendering that is cheaper than the cell's own forward pass and the row is a shortcut, but on
    a decoupled rendering the state component costs a sparse backward walk (90 steps at k=12,
    L=192) and the retrieval component costs three, so a windowed pass is an order of magnitude
    MORE expensive than the task and its accuracy is work, not a shortcut. On the decoupled
    retrieval arm it reads 1.000 for exactly that reason, and reporting that as the number a
    score is read against would make the component arm unreadable. The rows stay registered and
    printed on both renderings; they enter this max only where they are shortcuts, which is the
    same rule scripts/validate_suite.py gates on.
    """
    registered = S5_BIND_ADVERSARIES if coupled else tuple(
        r for r in S5_BIND_ADVERSARIES if not r.startswith("window_"))
    return operative_floor(floors, registered)


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


if __name__ == "__main__":
    print(_fmt(run_gate()))
