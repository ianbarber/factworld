"""THE THREE-CELL COMPARISON — the protocol, the reading rule, and the frontier scout price.

WHAT THIS DECIDES
    s5_bind_v3 puts two maps over one event stream: P (agents -> agents, rewritten by swaps)
    and B (objects -> agents, last-write-wins, rewritten by gives). Three cells sit on that one
    basis: a STATE component (named swaps — the S5 word problem), a RETRIEVAL component (named
    gives — last-write-wins), and the COMPOSED cell where every event's second operand is named
    live through the other map. The floors are closed. What has never been measured, in either
    regime, is the only thing the instrument is for: do the two components COMPOSE — is the
    composed cell harder than its components by more than the extra work it makes a solver do?

    This module is the reading rule, written before any solver number exists. It is imported by
    the runners so the verdict is applied mechanically rather than argued after the fact:

        scripts/experiment_s5bind_v3_three_cell_local_20260731.py   the from-scratch arm

    Run ``--register`` to write the pre-registration record (cells, lengths, costs, floors,
    thresholds, verdict table) BEFORE the run. Run ``--read PATH`` afterwards to re-apply the
    rule to a runner's results file — the verdict is a function of (accuracies, floors,
    thresholds) and nothing else, so a verdict that moves because a floor was re-measured is
    visible as that. Run ``--scout`` to price the frontier scout, the roster run it gates, and
    the stop rules.

THE CELLS, AND WHY EACH IS READ AGAINST ITS OWN FLOOR
    Every one of the three has a DIFFERENT chance-level policy available, so a single shared
    baseline would mis-price two of them. The floor is the max over the rows the class rule
    admits at that cell's own shape (factworld.validity.s5_bind_v3_operative_floor), recomputed
    from that cell's own items:

      state component     the one-hop read, the state-free surface read, and the fitted
                          25-feature surface ranker; the truncated carrier walk is excluded on
                          composition DEPTH (one hop against the carrier chain's 2*n_swap/k)
      retrieval component every admitted give-scan resolves nothing, because the sampler pins
                          the queried object's resolving write into [L/10, 0.75L] and no budget
                          under the algorithm's per-item minimum reaches it; the floor is
                          informed chance, proved rather than defined
      composed cell       the one-structure bound W <= max(k,m)+1 = 7 against the task's 13; the
                          fitted surface ranker sets the number

    Informed chance is 1/(k-1) — the stated initial answer is never the gold one — so 0.200 at
    the k=6 local operating point. Ratios below are to that.

THE STEP MULTIPLIER, IN BOTH COST MODELS, because they disagree and the disagreement matters
    CHARGED STEPS (factworld.composition's convention, W1-W5: the stated fact block is
    content-addressed and re-readable at one step, the event stream is not addressable) is what
    a solver with a scratchpad pays. The composed cell costs 1.82 / 1.81 / 1.78 times the state
    component and 5.6 / 6.1 / 6.6 times the retrieval component at L = 48 / 64 / 96.

    FORWARD-PASS TOKENS is what a streaming model with no scratchpad pays, and it is the cost
    model of the from-scratch arm: 1.64 / 1.66 / 1.65 against the state component and 2.45 /
    2.51 / 2.54 against the retrieval component. The two disagree most on the retrieval
    component, whose ALGORITHM is a short scan to a pinned window while its PROMPT is the whole
    stream. Reporting one number and calling it "the step multiplier" would hide that, so both
    are registered and the control is run in the cost model of the regime being measured.

    The control that separates "harder because composed" from "harder because longer" is the
    MATCHED-COST read: each component is also evaluated at the length whose cost equals the
    composed cell's at L, in that regime's cost model. On the retrieval side that control mostly
    does not exist, and the reason is the sampler rather than a choice: the window it pins the
    resolving write into gets exponentially harder to satisfy as the stream grows (3.1 ms per
    item at L=96, 167 ms at L=172, no admissible item at all at L=176), so the retrieval
    component cannot be run long enough to cost what the composed cell costs at L=64 or L=96.
    The control is registered as ABSENT there, not approximated by a shorter length that does
    not match the cost.

THE TWO READS, and both are registered because they are two different regimes
    PLAIN (no scratchpad). The answer is read off the plain prompt in one token. A streaming
    model with O(1) live state IS the class the one-structure bound prices, so this is the read
    against which the floor's W axis has force. It is also the read the frontier arm cannot
    give, because a frontier model with a visible trace is not register-bounded.

    GUIDED (scratchpad). Events teacher-forced, every per-event checkpoint and the answer
    generated by the model — the protocol under which S5-style state tracking formed locally in
    this repo at all (``scripts/experiment_dense_supervision.py``, and
    ``scripts/sweep.py::guided_free_run_eval``). It is registered because the PLAIN read has a
    documented prior: ``s5_v1`` carries "not reliably trainable in this harness (answer-only
    floors in distribution)". A protocol whose only read is one that is already known to floor
    would confirm a prior rather than measure the composition.

    Both reads score the SAME items against the SAME floors. The scratchpad caveat is stated
    where it bites: against the GUIDED read the W axis of the floor profile has no force, so
    that read is against the ADMITTED END of the profile, which at k=6 is 1.00-1.02x informed
    chance on the components and 1.16-1.21x on the composed cell — i.e. the two reads happen to
    be read against nearly the same numbers here, and the difference is in what clearing means.

    A composition claim requires the components and the composed cell to be judged on the SAME
    read. Mixing them — components on GUIDED, composed on PLAIN — would manufacture a gap out
    of the eval mode, so ``verdict()`` is applied to each read separately and both are printed.

THE READING RULE (pre-registered; every threshold below is fixed before any result exists)
    Metric: match, the canonical evaluator, on N_EVAL held-out items per (cell, length) for the
    PLAIN read and N_GUIDED for the GUIDED read (the guided decode is ~(k+m)*L sequential
    batched forwards per item, so its sample is smaller and its z is computed at its own n).

    CLEARS. A cell at (arch, seed, L) clears its floor iff BOTH hold:
        z = (a - f) / sqrt(f (1 - f) / N_EVAL)  >  Z_CLEAR   (= 3.0)
        a - f                                   >= MARGIN    (= 0.15)
    Two conjuncts because either alone is wrong here. At N_EVAL = 1000 and f = 0.2 a lift of
    0.04 is already z = 3.2, and a policy 0.04 above the floor is a policy the floor did not
    price, not a formed circuit; conversely a 0.15 lift on 50 items is noise. The margin is set
    at 0.15 = three quarters of informed chance, which is the smallest gap that cannot be
    produced by any surface family measured on this rung (the widest is the fitted ranker at
    1.14x chance, i.e. 0.028 over the floor).

    FORMS. A cell forms for an arch iff it CLEARS on at least SEEDS_CLEAR (= 2) of the seeds at
    every registered length. Seeds are counted, never averaged: this family is bimodal at the
    emergence threshold and a mean over one converged and two floored seeds is a number no seed
    produced. Per-seed values are reported in every table.

    POSITIVE CONTROL, and it gates the whole run. Every model is also read on the state
    component at L = 16 — the shortest TRAINED length, in distribution. If that is at floor the
    run is void: the harness is not training this family at this width and budget, and no cell
    downstream is interpretable. This is the one result that is about the harness rather than
    about the instrument, and it is checked first.

THE VERDICT TABLE (mechanical; ``verdict()`` returns exactly one of these)
    V5 HARNESS NULL          the L=16 state control is at floor. Nothing is claimable. Next move
                             is the training recipe, not the instrument.
    V4 COMPONENT UNREADABLE  a component does not FORM at its own registered lengths. The
                             composed cell cannot be read against it, because a composed failure
                             is then explained by the component that failed. Next move is that
                             component's budget or curriculum, not the composition.
    V3 GAP IS THE COST       the composed cell is at floor at L, and a component is ALSO at
                             floor at its matched-cost length. The composed cell's failure is
                             accounted for by how much longer it is, and no composition claim is
                             available from this run.
    V1 COMPOSITION GAP       both components FORM, including at their matched-cost lengths, and
                             the composed cell does not clear at any registered length. The
                             composed cell is harder than its components beyond the multiplier.
                             This is the reading the instrument was built to produce.
    V2 NO GAP HERE           the composed cell FORMS. Composition is not a separate difficulty
                             at this operating point; the registered lengths or k must move
                             before the cell is worth buying on the frontier.

    Nothing in this table is derived from the within-cell statistic. theta_c is an
    identification impossibility on this rendering — within a kind the class label IS the
    printed clause, and a solver that cannot hold B fails on exactly the antisymmetric direction
    the kind-balancing annihilates (factworld.composition) — so it is a structure-SWITCH
    diagnostic and no verdict here reads it. The composition evidence is the three-cell
    comparison and only that.

WHAT WOULD STOP A FRONTIER SPEND — see ``scout_plan()`` for the priced version
    The repo's standing rule is that scout data showing a ceiling stops or redesigns a paid run
    rather than merely re-budgeting it. The scout is bought first; the roster run is bought only
    if the scout separates. Both stop conditions and the buy condition are fixed here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import tasks as TK          # noqa: E402
from factworld import validity as V        # noqa: E402

# ---- the registered cells and lengths ------------------------------------------------------
LOCAL_CELLS = {"state": "s5_bind_local_v3_state",
               "bind": "s5_bind_local_v3_bind",
               "composed": "s5_bind_local_v3"}
FRONTIER_CELLS = {"state": "s5_bind_v3_state",
                  "bind": "s5_bind_v3_bind",
                  "composed": "s5_bind_v3"}
LOCAL_LENGTHS = (48, 64, 96)              # the k=6 grid the specs register
CONTROL_LENGTH = 16                       # the in-distribution positive control (state cell)
TRAIN_LENGTHS = (16, 32, 48, 64, 96)      # eval grid is IN distribution: this run is about
                                          # composition, not length extrapolation

# The retrieval component's sampler at k=6. The window it pins the resolving write into gets
# harder to satisfy as the stream grows, and the cost is measured, not assumed: 3.1 / 18.6 /
# 33.9 / 88.2 / 167.1 ms per item at L = 96 / 128 / 144 / 160 / 172, and at L = 176 no
# admissible item exists at all inside the restart cap. BIND_MAX_LENGTH is that hard ceiling;
# BIND_MATCHED_MAX is the length past which building the floor's own item pool (N_EVAL +
# N_FIT + N_SCORE = 7000 items) costs more than the control is worth — 4 minutes at 144
# against 20 at 172 — so a matched-cost control past it is registered as ABSENT rather than
# bought. Both numbers are properties of the instrument and both are reported.
BIND_MAX_LENGTH = 176
BIND_MATCHED_MAX = 144

# ---- the thresholds ------------------------------------------------------------------------
N_EVAL = 1000                             # PLAIN read
N_GUIDED = 128                            # GUIDED read: ~(k+m)*L batched forwards per item
GUIDED_LENGTHS = (48,)                    # its decode is O(n L^2); the full grid is the PLAIN read
N_FIT = 2000                              # surface-ranker fit sample; the attack showed the
N_SCORE = 4000                            # published 1.14x was a fit-budget artifact at 500
Z_CLEAR = 3.0
MARGIN = 0.15
SEEDS_CLEAR = 2

# ---- the frontier scout --------------------------------------------------------------------
SCOUT_COMPOSED_LENGTHS = (128, 256)       # the composed cell carries the length axis
SCOUT_COMPONENT_LENGTHS = (256,)          # a component that holds at 256 holds at 128
SCOUT_N = 40
SCOUT_MAX_NEW_TOKENS = 8192               # the protocol budget for every reasoning-on cell
# three models spanning the roster's measured range, not three near neighbours: the top of the
# owner's ranking, the mid-tier reasoner that sits surprisingly high on it, and the bottom.
SCOUT_MODELS = ("openai/gpt-5.5", "z-ai/glm-5.2", "nvidia/nemotron-3-ultra-550b-a55b")
ROSTER_N = 100                            # what a RANKING needs, against the scout's 40
SCOUT_CEILING = 0.90                      # composed cell at/above this for the top model -> STOP
SCOUT_SEPARATION = 0.20                   # composed-cell spread across the scout -> BUY
SCOUT_COMPONENT_MIN = 0.80                # components must be this high for the top model


# ---- costs ---------------------------------------------------------------------------------
def cell_cost(spec, L, tok=None, n_probe=200):
    """``(charged_steps, prompt_tokens)`` for one cell at one length.

    ``charged_steps`` is that cell's own cheapest correct algorithm under the W convention
    (validity.s5_bind_v3_task_cost); ``prompt_tokens`` is what a streaming model's forward pass
    actually costs. They are different cost models, not two estimates of one.
    """
    ex = TK.generate(spec, "test", n=n_probe, length=L)
    ns, ng = V.s5_bind_v3_shape(ex)
    named = V.s5_bind_v3_is_named(ex)
    query = V.s5_bind_v3_query_kind(ex)
    _w, s = V.s5_bind_v3_task_cost(spec.k, spec.n_objects_active, ns, ng, named, query)
    ptok = None
    if tok is not None:
        probe = ex[:20]
        ptok = sum(len(tok.encode(e.prompt)) for e in probe) // len(probe)
    return s, ptok


def matched_lengths(tok, cells=LOCAL_CELLS, lengths=LOCAL_LENGTHS, axis="tokens",
                    n_needed=N_EVAL + N_FIT + N_SCORE):
    """For each composed length, the component length whose cost matches it, in one cost model.

    ``axis="tokens"`` is the from-scratch regime's cost (the forward pass); ``axis="steps"`` is
    the charged-step convention a scratchpad solver pays. Both costs are affine in L, so the
    candidate is solved from two probes and then VERIFIED — including that the sampler can build
    the whole item pool the floor needs there. Unreachable returns None rather than the nearest
    feasible length, so the missing control is visible in the record instead of silently
    replaced by a shorter one that does not match the cost.
    """
    comp = TK.CANONICAL[cells["composed"]]
    out = {}
    for L in lengths:
        s, t = cell_cost(comp, L, tok)
        target = t if axis == "tokens" else s
        row = {"composed_L": L, "composed_steps": s, "composed_tokens": t, "axis": axis,
               "target_cost": target}
        for key in ("state", "bind"):
            spec = TK.CANONICAL[cells[key]]
            cap = BIND_MATCHED_MAX if key == "bind" else 16 * L
            (s1, t1), (s2, t2) = cell_cost(spec, 32, tok, 8), cell_cost(spec, 128, tok, 8)
            c1, c2 = (t1, t2) if axis == "tokens" else (s1, s2)
            slope = (c2 - c1) / (128 - 32)
            cand = int(round(32 + (target - c1) / max(1e-9, slope)))
            cand = max(8, min(cap, cand - cand % 4))
            best = None
            for probe in sorted({max(8, min(cap, cand + d)) for d in (0, -4, 4, -8, 8)},
                                key=lambda x: abs(x - cand)):
                try:
                    cs, ct = cell_cost(spec, probe, tok, n_probe=8)
                except Exception:                     # sampler cannot build this length at all
                    continue
                cost = ct if axis == "tokens" else cs
                if cost < 0.95 * target:
                    continue        # cheap check first: a length the cap holds short of the
                                    # target is not a matched control however feasible it is
                try:
                    # the floor needs the whole pool here, so feasibility is checked at the
                    # size it will actually be asked for, not at a probe size
                    TK.generate(spec, "test", n=n_needed, length=probe)
                except Exception:                     # sampler cannot fill the pool here
                    continue
                best = (probe, cost)
                break                                 # first feasible candidate, closest first
            reachable = best is not None and best[1] >= 0.95 * target
            row[key] = {"L": best[0] if reachable else None,
                        "cost": None if best is None else best[1],
                        "target": target, "reachable": bool(reachable),
                        "cap": cap, "solved_candidate": cand}
        out[L] = row
    return out


# ---- floors ---------------------------------------------------------------------------------
def cell_floor(spec, L, n_eval=N_EVAL, n_fit=N_FIT, n_score=N_SCORE):
    """The operative floor at (cell, length), and everything needed to audit it.

    Measured on ``n_score`` items drawn from the SAME deterministic test stream as the items a
    solver is scored on and DISJOINT from them, because the max over admitted rows carries an
    upward selection bias at small n (the published 1.30x on the state component was a high draw
    at n=500 and is 0.98x at n=4000) and because the fitted ranker has to be scored out of
    sample. The same rows on the exact scored items are reported beside it as the house-rule
    check; where the two differ the larger is the number a score must clear.
    """
    k, m = spec.k, spec.n_objects_active
    pool = TK.generate(spec, "test", n=n_eval + n_fit + n_score, length=L)
    scored, fit, big = pool[:n_eval], pool[n_eval:n_eval + n_fit], pool[n_eval + n_fit:]
    named = V.s5_bind_v3_is_named(big)
    query = V.s5_bind_v3_query_kind(big)
    ns, ng = V.s5_bind_v3_shape(big)
    keep = tuple(r for r in V.s5_bind_v3_family_rows(k, m, ns, ng, named, query)
                 if V.s5_bind_v3_admits(r, k, m, ns, ng, named, query))
    fl = dict(V.s5_bind_v3_floors(big, k, m))
    fl.update(V.s5_bind_v3_family_floors(big, k, m, named, query, rows=keep))
    op = V.s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, query)
    sb = V.s5_bind_v3_surface_bound(fit, k, held_out=big)
    if sb is not None and not V.s5_bind_v3_admits("surface_ranker", k, m, ns, ng, named, query):
        sb = None
    if sb is not None and (op is None or sb["held_out"] > op):
        op = sb["held_out"]
    nss, ngs = V.s5_bind_v3_shape(scored)
    keeps = tuple(r for r in V.s5_bind_v3_family_rows(k, m, nss, ngs, named, query)
                  if V.s5_bind_v3_admits(r, k, m, nss, ngs, named, query))
    fls = dict(V.s5_bind_v3_floors(scored, k, m))
    fls.update(V.s5_bind_v3_family_floors(scored, k, m, named, query, rows=keeps))
    op_scored = V.s5_bind_v3_operative_floor(fls, k, m, nss, ngs, named, query)
    floor = max(x for x in (op, op_scored) if x is not None)
    _w, s = V.s5_bind_v3_task_cost(k, m, ns, ng, named, query)
    return {"cell": spec.name, "L": L, "k": k, "m": m, "query": query, "named": named,
            "chance": 1.0 / (k - 1), "floor": floor,
            "floor_disjoint": op, "floor_on_scored_items": op_scored,
            "surface_held_out": None if sb is None else sb["held_out"],
            "basis": V.s5_bind_v3_floor_basis(k, m, ns, ng, named, query),
            "admitted_rows": {r: round(v, 4) for r, v in
                              sorted(fl.items(), key=lambda x: -x[1])
                              if V.s5_bind_v3_admits(r, k, m, ns, ng, named, query)},
            "charged_steps": s, "n_swap": ns, "n_give": ng}


# ---- the rule -------------------------------------------------------------------------------
def clears(acc, floor, n=N_EVAL, z_min=Z_CLEAR, margin=MARGIN):
    """CLEARS, exactly as registered: significant AND large enough to be a circuit."""
    if acc is None or floor is None:
        return False, None
    se = math.sqrt(max(1e-12, floor * (1.0 - floor) / n))
    z = (acc - floor) / se
    return bool(z > z_min and (acc - floor) >= margin), z


def forms(per_seed, floors, lengths, n=N_EVAL, seeds_clear=SEEDS_CLEAR):
    """FORMS: clears on >= seeds_clear seeds at EVERY registered length.

    ``per_seed`` is {seed: {L: accuracy}}; ``floors`` is {L: floor}. Returns the verdict and the
    per-length seed counts, so a cell that clears at 48 and not at 96 is visible as that rather
    than as a bare False.
    """
    counts = {}
    for L in lengths:
        counts[L] = sum(1 for s in per_seed
                        if clears(per_seed[s].get(L), floors.get(L), n)[0])
    return bool(lengths) and all(c >= seeds_clear for c in counts.values()), counts


def verdict(state_ctrl, comp_forms, comp_counts, matched_forms):
    """The verdict table, applied mechanically.

    Args:
        state_ctrl: seed count clearing the state component at CONTROL_LENGTH.
        comp_forms: {"state": bool, "bind": bool, "composed": bool} at registered lengths.
        comp_counts: {cell: {L: seeds clearing}} for the report.
        matched_forms: {"state": bool|None, "bind": bool|None} at the matched-cost lengths;
            None where the sampler cannot reach the matched length.
    """
    if state_ctrl < SEEDS_CLEAR:
        return "V5_HARNESS_NULL", (
            f"the state component is at floor at the shortest trained length L={CONTROL_LENGTH} "
            f"({state_ctrl}/{SEEDS_CLEAR} seeds). The harness is not training this family at "
            f"this width and budget; no cell downstream is interpretable.")
    if not comp_forms["state"] or not comp_forms["bind"]:
        bad = [c for c in ("state", "bind") if not comp_forms[c]]
        return "V4_COMPONENT_UNREADABLE", (
            f"component(s) {bad} do not form at their own registered lengths "
            f"{ {c: comp_counts[c] for c in bad} }. A composed failure would be explained by the "
            f"component that failed, so no composition claim is available.")
    if comp_forms["composed"]:
        return "V2_NO_GAP_HERE", (
            "the composed cell forms. Composition is not a separate difficulty at k=6 / "
            f"L<={max(LOCAL_LENGTHS)} in this regime; the lengths or k must move before the cell "
            "is worth buying on the frontier.")
    unmatched = [c for c, ok in matched_forms.items() if ok is False]
    if unmatched:
        return "V3_GAP_IS_THE_COST", (
            f"the composed cell is at floor, but component(s) {unmatched} are also at floor at "
            "their matched-cost lengths. The composed cell's failure is accounted for by cost, "
            "not by composition.")
    return "V1_COMPOSITION_GAP", (
        "both components form, including at the matched-cost lengths, and the composed cell "
        "clears nowhere. The composition is harder than its components beyond the step "
        "multiplier.")


def verdict_repaired(state_ctrl, any_ctrl, comp_forms, comp_counts, matched_forms,
                     matched_measured):
    """THE SAME RULE WITH THREE PRE-REGISTRATION DEFECTS REPAIRED, reported BESIDE ``verdict()``
    and never in place of it.

    The registered rule was run first and its answer stands as the pre-registered answer. What
    the run showed is that three of its clauses do not say what they were meant to say, and each
    was found by a result the clause mislabels rather than by taste:

    D1  THE POSITIVE CONTROL IS ONE COMPONENT, AND IT SHOULD BE A DISJUNCTION. ``verdict``
        voids the run when the STATE component is at floor at L=16, on the reasoning that a
        harness which cannot train the shortest in-distribution cell cannot be read. The
        retrieval component reads 1.000 at every length on the same models. So the harness
        trains, the run is readable, and V5's sentence — "the harness is not training this
        family" — is false where it fires. The control has to be "SOME component clears
        somewhere", which is what actually licenses reading the rest.

    D2  THE CONTROL LENGTH IS NOT IN EVERY READ'S GRID. The GUIDED read is registered at
        GUIDED_LENGTHS = (48,), which does not contain CONTROL_LENGTH = 16, so on that read the
        control is not at floor — it is not measured, and V5 fires on an absence. A control has
        to be evaluated on a grid the read covers.

    D3  V1 TREATS AN ABSENT MATCHED-COST CONTROL AS A PASS. Its branch is "no component FAILED
        at its matched length", and a control that was never measured does not fail. But V1's
        whole claim is "beyond the step multiplier", and that claim is exactly what the matched
        control establishes. Where the control is absent the claim is not available, however the
        cells came out, so the repaired rule returns V1_UNCONTROLLED instead.

    Args:
        any_ctrl: seeds clearing ANY component at ANY registered length (D1/D2).
        matched_measured: {cell: bool} — whether a matched-cost control was measured at all.
    """
    if any_ctrl < SEEDS_CLEAR:
        return "V5_HARNESS_NULL", (
            f"no component clears anywhere ({any_ctrl} seeds). Nothing downstream is "
            "interpretable; the next move is the training recipe, not the instrument.")
    if not comp_forms["state"] or not comp_forms["bind"]:
        bad = [c for c in ("state", "bind") if not comp_forms[c]]
        return "V4_COMPONENT_UNREADABLE", (
            f"component(s) {bad} do not form at their own registered lengths "
            f"{ {c: comp_counts[c] for c in bad} }, while the other one does. A composed "
            "failure would be explained by the component that failed, so no composition claim "
            "is available — and the dissociation between the components is the result.")
    if comp_forms["composed"]:
        return "V2_NO_GAP_HERE", (
            "the composed cell forms wherever both components do. Composition is not a separate "
            "difficulty at this operating point.")
    unmatched = [c for c, ok in matched_forms.items() if ok is False]
    if unmatched:
        return "V3_GAP_IS_THE_COST", (
            f"the composed cell is at floor and component(s) {unmatched} are also at floor at "
            "their matched-cost lengths. The failure is accounted for by cost.")
    if not all(matched_measured.get(c) for c in ("state", "bind")):
        missing = [c for c in ("state", "bind") if not matched_measured.get(c)]
        return "V1_UNCONTROLLED", (
            "both components form and the composed cell does not — the V1 pattern — but the "
            f"matched-cost control is absent for {missing}, so 'beyond the step multiplier' is "
            "not established. The cells separate; the cause does not.")
    return "V1_COMPOSITION_GAP", (
        "both components form, including at the matched-cost lengths, and the composed cell "
        "clears nowhere. The composition is harder than its components beyond the multiplier.")


# ---- the frontier scout ----------------------------------------------------------------------
def scout_plan(models=SCOUT_MODELS, n=SCOUT_N):
    """The priced scout, and the decision rule for whether the roster run is worth buying.

    Cells: the composed k=12 cell at both scout lengths — it carries the length axis, which is
    where separation would come from — and each component at L=256 only, because a component
    that holds at the deepest registered length holds at the shallower one and the component
    reading is a GATE rather than a ranking. Reasoning ON at the protocol budget throughout.

    n is set at SCOUT_N because the scout has to answer a SEPARATION question, not rank the
    roster: the smallest gap it must resolve is SCOUT_SEPARATION = 0.20, which at n = 40 is
    z = 2.1 against a two-sided binomial at p = 0.5 — enough to see a spread that large, and
    deliberately not enough to order two models 0.05 apart, which is what the roster run is for.
    """
    from factworld.benchmark import MODELS, cost_estimate

    cells = ([{"task": FRONTIER_CELLS["composed"], "length": L, "n": n,
               "settings": {"effort": "high", "max_new_tokens": SCOUT_MAX_NEW_TOKENS}}
              for L in SCOUT_COMPOSED_LENGTHS]
             + [{"task": FRONTIER_CELLS[c], "length": L, "n": n,
                 "settings": {"effort": "high", "max_new_tokens": SCOUT_MAX_NEW_TOKENS}}
                for c in ("state", "bind") for L in SCOUT_COMPONENT_LENGTHS])
    rows, total = [], 0.0
    for slug in models:
        if slug not in MODELS:
            rows.append({"model": slug, "error": "not in MODELS"})
            continue
        est = cost_estimate(slug, cells, assumed_output_tokens=4000)
        total += est["cost_usd"]
        rows.append({"model": slug, **est})
    # what the scout is gating: the same three cells over the whole registered grid and the
    # whole roster, at the n a ranking needs. Both numbers belong in the decision, because the
    # scout is only worth its own price against what it stops.
    roster_cells = [{"task": FRONTIER_CELLS[c], "length": L, "n": ROSTER_N,
                     "settings": {"effort": "high", "max_new_tokens": SCOUT_MAX_NEW_TOKENS}}
                    for c in ("state", "bind", "composed")
                    for L in TK.CANONICAL[FRONTIER_CELLS["composed"]].eval_lengths]
    roster, roster_total = [], 0.0
    for slug in MODELS:
        est = cost_estimate(slug, roster_cells, assumed_output_tokens=4000)
        roster_total += est["cost_usd"]
        roster.append({"model": slug, "cost_usd": est["cost_usd"]})
    return {"cells": cells, "n_cells": len(cells), "per_model": rows,
            "total_usd": round(total, 2),
            "roster_run_if_bought": {
                "models": len(MODELS), "cells_per_model": len(roster_cells), "n": ROSTER_N,
                "per_model": sorted(roster, key=lambda r: -r["cost_usd"]),
                "total_usd": round(roster_total, 2)},
            "stop_rules": [
                f"STOP (ceiling) if the top scout model's COMPOSED match >= {SCOUT_CEILING} at "
                f"L={max(SCOUT_COMPOSED_LENGTHS)}. A ceiling cannot rank the roster, so the "
                "roster run buys "
                f"nothing; redesign (raise k or L) before spending.",
                "STOP (floor) if the composed cell is at its operative floor for ALL scout "
                "models. Zero separation across the roster's range means zero expected pairwise "
                "separations in the full run — the same reading that stopped the s5_chain "
                "ranking.",
                f"STOP (component) if either COMPONENT is below {SCOUT_COMPONENT_MIN} for the "
                "top scout model. The composed cell is then unreadable on the frontier for the "
                "same reason it is unreadable locally when a component does not form.",
            ],
            "buy_rule": (
                f"BUY the roster run iff the composed cell's spread across the three scout "
                f"models is >= {SCOUT_SEPARATION} match AND both components are >= "
                f"{SCOUT_COMPONENT_MIN} for every scout model — the composed cell discriminates "
                f"inside the roster's range while the components do not."),
            "budget_note": (
                f"Priced at {SCOUT_MAX_NEW_TOKENS}-token reasoning cells with 4000 assumed "
                "completion tokens per call. The 8192 budget is the protocol minimum for a "
                "reasoning-on cell; smaller budgets manufactured the published s5 L64 cliff and "
                "chain floor as truncation artifacts, so a cheaper scout is not available."),
            }


# ---- registration ---------------------------------------------------------------------------
def register(out_prefix, axis="tokens", with_floors=True):
    """Write the pre-registration record: cells, lengths, costs, matched lengths, floors and
    every threshold, before any solver number exists."""
    from factworld.tokenizer import Tokenizer

    base = TK.CANONICAL[LOCAL_CELLS["composed"]]
    w, r = TK.build_world(base)
    tok = Tokenizer.build([w], r)
    costs = {}
    for key, nm in LOCAL_CELLS.items():
        spec = TK.CANONICAL[nm]
        for L in (CONTROL_LENGTH,) + LOCAL_LENGTHS:
            s, t = cell_cost(spec, L, tok)
            costs[f"{key}@{L}"] = {"charged_steps": s, "prompt_tokens": t}
    ml = matched_lengths(tok, axis=axis)
    floors = {}
    if with_floors:
        wanted = {("state", CONTROL_LENGTH)}
        for L in LOCAL_LENGTHS:
            for key in LOCAL_CELLS:
                wanted.add((key, L))
            for key in ("state", "bind"):
                mlen = ml[L][key]["L"]
                if mlen:
                    wanted.add((key, mlen))
        for key, L in sorted(wanted):
            spec = TK.CANONICAL[LOCAL_CELLS[key]]
            floors[f"{key}@{L}"] = cell_floor(spec, L)
            print(f"  floor {key}@{L}: {floors[f'{key}@{L}']['floor']:.4f} "
                  f"({floors[f'{key}@{L}']['floor'] / floors[f'{key}@{L}']['chance']:.2f}x chance)",
                  flush=True)
    rec = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": os.path.basename(__file__),
        "local_cells": LOCAL_CELLS, "frontier_cells": FRONTIER_CELLS,
        "local_lengths": LOCAL_LENGTHS, "control_length": CONTROL_LENGTH,
        "train_lengths": TRAIN_LENGTHS,
        "thresholds": {"n_eval": N_EVAL, "z_clear": Z_CLEAR, "margin": MARGIN,
                       "seeds_clear": SEEDS_CLEAR, "n_fit": N_FIT, "n_score": N_SCORE},
        "costs": costs, "matched_lengths": ml, "matched_axis": axis,
        "floors": floors,
        "verdicts": ["V5_HARNESS_NULL", "V4_COMPONENT_UNREADABLE", "V3_GAP_IS_THE_COST",
                     "V1_COMPOSITION_GAP", "V2_NO_GAP_HERE"],
        "scout": scout_plan(),
    }
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {out_prefix}.json")
    return rec


def read_results(path, floors_path=None):
    """Re-apply the rule to a runner's results JSON — e.g. after a floor is re-measured.

    The verdict is a function of (accuracies, floors, thresholds) and nothing else, so it can
    always be recomputed; a verdict that changed because a floor moved should be visible as
    that, not buried in a stale results file.
    """
    import experiment_s5bind_v3_three_cell_local_20260731 as E

    res = json.load(open(path))
    floors = json.load(open(floors_path))["floors"] if floors_path else res["floors"]
    cfg = res["cfg"]
    grid = {k: [int(x) for x in v] for k, v in cfg["grid"].items()}
    return E.apply_rule(res["runs"], floors, grid, cfg["eval_n"], cfg.get("guided_n", N_GUIDED))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--register", action="store_true", help="write the pre-registration record")
    ap.add_argument("--read", default=None, help="re-apply the rule to a runner results JSON")
    ap.add_argument("--read_floors", default=None, help="floors JSON to re-read --read against")
    ap.add_argument("--no_floors", action="store_true", help="skip the floor pass (fast)")
    ap.add_argument("--axis", default="tokens", choices=["tokens", "steps"])
    ap.add_argument("--scout", action="store_true", help="print the priced frontier scout")
    ap.add_argument("--out_prefix",
                    default="results/s5bind_v3_three_cell_preregistration_20260731")
    a = ap.parse_args()
    if a.scout:
        p = scout_plan()
        print(json.dumps(p, indent=2))
    if a.read:
        for arch, reads in sorted(read_results(a.read, a.read_floors).items()):
            for read, v in sorted(reads.items()):
                print(f"{arch} / {read}: {v['verdict']} — {v['why']}")
                print(f"    seeds clearing {v['seed_counts']}; control "
                      f"{v['control_seeds']}; matched {v['matched_forms']}")
    if a.register:
        os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
        register(a.out_prefix, axis=a.axis, with_floors=not a.no_floors)


if __name__ == "__main__":
    main()
