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
# s5_bind_v3 — the SOURCE-STRUCTURE family, and the Pareto floor class.
#
# WHAT A FLOOR ROW IS HERE. The construct claims the composed cell forces a forward pass
# carrying BOTH structures. A floor row is a named way of not doing that. Which rows may set a
# floor is decided by COST, not by accuracy — and not by live slots alone, which the earlier
# rule used and which is unsound in both directions:
#
#   a W-only rule ADMITS a demand-driven resolver that carries no map — it resolves the query's
#   dependencies recursively, re-walking the event list for each one, holding only a stack of
#   frames. On the temporal construct that resolver was exactly correct (1.0000 at n=1000) with
#   at most 25/18/14 live frames at k=12/32/48, FALLING in k, so a W-only rule set that cell's
#   floor to 1.000. It separated from the task only on STEPS, by 5,100x. Measured here it is
#   weaker but the argument is the same shape: memo-free and budgeted it reads 0.15-0.67x chance
#   at every budget up to 25x the task's own steps at both scored cells, and only reaches 0.94 at
#   the local cell at ~480x (scripts/probe_s5bind_v3_construct_20260731.py --stage demand).
#   a W-only rule with a "no structure-sized state" reading EXCLUDES the block-drop continuum
#   correctly, but only because every member carries both maps — it has nothing to say about a
#   member that carried one.
#
# THE PARETO RULE, stated so it is checkable:
#
#     A ROW MAY SET A FLOOR IFF IT IS STRICTLY CHEAPER THAN THE CELL'S CHEAPEST CORRECT
#     ALGORITHM IN LIVE SLOTS W, AND NO MORE EXPENSIVE IN STEPS S.
#
# Both halves are load-bearing and the tie-breaking is deliberate. A row that TIES on W is
# holding everything the task holds; whatever else it does, it is the task at a constant-factor
# discount, and that is exactly what the whole (position, width) block-drop continuum is — every
# member carries P and B across its budget, so W = k + m for all of them and the continuum goes
# out by ONE argument rather than member by member. A row that is cheaper in W but pays more in
# S is buying its slots with time; the demand resolver is that case, and it is out.
#
# Costs are counted by ``factworld.composition`` against the convention stated there (one step =
# a keyed header read, an event read, a map resolution, a map write, or a comparison — and a
# backward walk IS charged for the events it scans and rejects).
# ---------------------------------------------------------------------------
S5_BIND_V3_ROWS = ("stated_reference", "one_structure_P", "one_structure_B",
                   "final_state", "last_write_1hop", "initial_only",
                   "window_90", "window_75", "window_50",
                   "prefix_90", "prefix_75", "prefix_50",
                   "uniform_non_initial", "uniform")
S5_BIND_V3_CHANCE_ROWS = ("uniform_non_initial", "uniform")
S5_BIND_V3_TRUNCATION_ROWS = tuple(r for r in S5_BIND_V3_ROWS
                                   if r.startswith(("window_", "prefix_")))
# Rows defined only where some event carries a REFERENCE. On a component cell every operand is
# named, so each of these reproduces the oracle exactly and is dropped rather than reported.
S5_BIND_V3_REFERENCE_ROWS = ("stated_reference", "one_structure_P", "one_structure_B",
                             "final_state")
S5_BIND_V3_WINDOWS = (0.9, 0.75, 0.5)


def s5_bind_v3_row_cost(row: str, k: int, m: int, n_swap: int, n_give: int) -> tuple[int, int]:
    """``(W, S)`` for one registered row, under the cost convention in factworld.composition.

    The task's own cheapest correct algorithm is the forward pass carrying P and B:
    ``W = k + m``, ``S = (k + m) + 6 n_swap + 3 n_give + 1``.
    """
    L = n_swap + n_give
    if row in ("uniform", "uniform_non_initial", "initial_only"):
        return 1, 2
    if row == "last_write_1hop":
        return 2, 2 * L + 2
    if row in ("stated_reference", "one_structure_P"):
        # carries P alone: a cross reference is a keyed read of the stated holder block
        return k + 1, k + 6 * n_swap + n_give + 1
    if row == "one_structure_B":
        return m + 1, m + 3 * n_give + n_swap + 1
    if row == "final_state":
        # needs the true final maps before it can start: two passes, both maps, twice over
        return 2 * (k + m), 2 * ((k + m) + 6 * n_swap + 3 * n_give + 1)
    if row.startswith(("window_", "prefix_")):
        f = int(row.split("_")[1]) / 100.0
        return k + m, int((k + m) + f * (6 * n_swap + 3 * n_give) + 1)
    raise KeyError(row)


def pareto_eligible(w_row: int, s_row: int, w_task: int, s_task: int) -> bool:
    """The class rule: strictly cheaper in live slots, no more expensive in steps."""
    return w_row < w_task and s_row <= s_task


def s5_bind_v3_task_cost(k: int, m: int, n_swap: int, n_give: int,
                         named: bool = False) -> tuple[int, int]:
    """``(W, S)`` for THIS CELL's cheapest correct algorithm.

    The composed cell's is the forward pass carrying both maps. A COMPONENT cell renders its
    second operand by name, so every event's identity is fixed on the surface and the cheapest
    correct algorithm is the sparse backward carrier walk over one register — which is the whole
    point of the component arms, and why the class rule has to be evaluated against the cell in
    front of it rather than against the composed cell's cost everywhere.
    """
    if named:
        L = n_swap + n_give
        return 2, 2 * L + 2
    return k + m, (k + m) + 6 * n_swap + 3 * n_give + 1


def s5_bind_v3_classify(k: int, m: int, n_swap: int, n_give: int,
                        named: bool = False) -> dict[str, bool]:
    """Every registered row, classified by the Pareto rule at this cell's shape."""
    wt, st = s5_bind_v3_task_cost(k, m, n_swap, n_give, named)
    out = {}
    for row in S5_BIND_V3_ROWS:
        w, s = s5_bind_v3_row_cost(row, k, m, n_swap, n_give)
        out[row] = pareto_eligible(w, s, wt, st)
    return out


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
    cell's floor is the Pareto class rule (``s5_bind_v3_operative_floor``); the rest are printed
    as diagnostics.

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
        out["uniform"] = 1.0 / k
    return {nm: out[nm] for nm in S5_BIND_V3_ROWS if nm in out}


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


def s5_bind_v3_operative_floor(floors: dict[str, float], k: int, m: int,
                               n_swap: int, n_give: int, named: bool = False) -> float | None:
    """The number a source-structure cell has to clear: the max over the rows the Pareto class
    rule ADMITS at this cell's shape. Rows that tie the task on live slots, or that pay more
    steps, are diagnostics and never enter this max."""
    ok = s5_bind_v3_classify(k, m, n_swap, n_give, named)
    vals = [v for nm, v in floors.items() if ok.get(nm) and v is not None]
    return max(vals) if vals else None


def s5_bind_v3_block_drop(examples, width: float, pos: float) -> float:
    """One member of the block-drop family: skip ``width * L`` events starting at
    ``pos * (L - width * L)`` and play the rest exactly.

    Every member carries both maps, so no member can be a floor row under the Pareto rule; this
    exists to MEASURE what the excluded continuum actually reads, so the exclusion can be judged
    rather than taken on faith.
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


if __name__ == "__main__":
    print(_fmt(run_gate()))
