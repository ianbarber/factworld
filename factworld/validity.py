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
