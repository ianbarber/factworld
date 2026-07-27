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


def operative_floor(floors: dict[str, float]) -> float | None:
    """The number a cell has to clear: the max over whichever adversaries are registered.

    Reading a score against one named row understates the floor wherever that row is not the
    largest, which is most of the low-k end of the local grid.

    ``uniform_non_start`` is always among the registered rows, so this max can never fall below
    the fixed-offset partition's common expectation 1/(k-1). A member of that partition
    therefore sets the floor only where it beats uniform_non_start, and where it does not it
    contributes nothing — which is what keeps the floor from tracking the family's selection
    noise. Rows outside S5_CHAIN_ADVERSARIES (currently ``initial_map_backhop``) are reported
    for inspection and are ignored here.
    """
    vals = [v for name, v in floors.items() if name in S5_CHAIN_ADVERSARIES and v is not None]
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
