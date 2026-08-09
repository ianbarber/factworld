"""THE THREE-CELL COMPARISON — the protocol, the reading rule, and the frontier scout price.

WHAT THIS DECIDES
    s5_bind_v3 puts two maps over one event stream: P (agents -> agents, rewritten by swaps)
    and B (objects -> agents, last-write-wins, rewritten by gives). Three cells sit on that one
    basis: a STATE component (named swaps — the S5 word problem), a RETRIEVAL component (named
    gives — last-write-wins), and the COMPOSED cell where every event's second operand is named
    live through the other map. The floors are closed. What has never been measured, in either
    regime, is the only thing the instrument is for: do the two components COMPOSE — is the
    composed cell harder than its components by more than the extra work it makes a solver do?

    "The extra work it makes a solver do" is what the PAIRING fixes, and it is the reason the
    three cells are not simply read at one length. A composed stream of length L holds p_swap L
    swaps and (1 - p_swap) L gives, so a component read at the composed cell's own L is doing
    1/p_swap or 1/(1 - p_swap) times the work: composed@48 contains 17 swaps and 31 gives against
    a component@48's 48. Each component is therefore read at its WORK-MATCHED length, and also at
    its TOKEN-MATCHED one as the cost control (see PAIRINGS).

    THE COMPOSED QUERY IS A STATE QUERY ON EVERY ITEM, so retrieval enters it as an operand
    resolver and never as the answer. That asymmetry is not fixable under this sampler and no
    retrieval-query arm is registered: the bind query does need both maps (both one-structure
    replays fall to informed chance on it) but not the stream's TAIL, because the pin that proves
    the retrieval component's floor puts the resolving write at or below 0.75L. The measurement
    and the conflict are written out where the cells are declared, below.

    This module is the reading rule, written before any solver number exists. It is imported by
    the runners so the verdict is applied mechanically rather than argued after the fact:

        scripts/experiment_s5bind_v3_three_cell_local_20260731.py   the from-scratch arm

    Run ``--register`` to write the pre-registration record (cells, lengths, costs, floors,
    thresholds, verdict table) BEFORE the run. Run ``--read PATH`` afterwards to re-apply the
    rule to a runner's results file — the verdict is a function of (accuracies, floors,
    thresholds) and nothing else, so a verdict that moves because a floor was re-measured is
    visible as that. Run ``--add_trace_read PATH`` to write the trace read's declaration into an
    existing record without redrawing the rest of it. Run ``--scout`` to price the frontier
    scout, the roster run it gates, and the stop rules.

THE CELLS, AND WHY EACH IS READ AGAINST ITS OWN FLOOR
    Every one of the three has a DIFFERENT chance-level policy available, so a single shared
    baseline would mis-price two of them. The floor is the max over the rows the class rule
    admits at that cell's own shape (factworld.validity.s5_bind_v3_operative_floor), recomputed
    from that cell's own items:

      state component     the one-hop read and the state-free surface reads; the truncated
                          carrier walk is excluded on composition DEPTH (one hop against the
                          carrier chain's 2*n_swap/k)
      retrieval component every admitted give-scan resolves nothing, because the sampler pins
                          the queried object's resolving write into [L/10, 0.75L] and no budget
                          under the algorithm's per-item minimum reaches it; the floor is
                          informed chance, proved rather than defined
      composed cell       the one-structure bound W <= max(k,m)+1 = 7 against the task's 13;
                          ``last_swap_ref`` and the uniform rows set the number. THIS FLOOR IS
                          THE PLAIN PROTOCOL'S ONLY. The bound prices live slots, and the GUIDED
                          protocol's format hands out k + m of them at every event, so the cell
                          is UNFLOORABLE there on both channels and the run reports ``pad_reach``
                          — what the excluded both-maps class actually scores — in its place.

    THE FITTED 25-FEATURE SURFACE RANKER IS NOT IN ANY OF THESE FLOORS. Six of its features are
    per-candidate accumulators, so one pass over the k candidates holds W = 1 + 7k registers (43
    at k=6) and the register-lean implementation pays k passes, S = 2kL — over the composed
    cell's own algorithm at every registered length. No implementation is admitted at any cell
    (validity.s5_bind_v3_surface_impls), so it is measured, printed and read as a DIAGNOSTIC: it
    says what the state-free surface information supports, not what a cheap policy can extract.

    Informed chance is 1/(k-1) — the stated initial answer is never the gold one — so 0.200 at
    the k=6 local operating point. Ratios below are to that.

THE STEP MULTIPLIER IS THREE NUMBERS, NOT ONE, and which one is quoted depends on the PAIRING
    CHARGED STEPS (factworld.composition's convention, W1-W5: the stated fact block is
    content-addressed and re-readable at one step, the event stream is not addressable) is what a
    solver with a scratchpad pays; FORWARD-PASS TOKENS is what a streaming model with no
    scratchpad pays, and it is the cost model of the from-scratch arm. The two disagree most on
    the retrieval component, whose ALGORITHM is a short scan to a pinned window while its PROMPT
    is the whole stream. On tokens, at k=6 and L = 48 / 64 / 96:

                              vs state component      vs retrieval component
      equal length            1.65 / 1.65 / 1.65      2.46 / 2.51 / 2.54
      equal WORK              3.83 / 3.96 / 4.17      3.48 / 3.63 / 3.73
      equal tokens            1.04 / 1.02 / 1.01      1.01 /  —  /  —

    On charged steps the same three rows are 1.82/1.81/1.78, 4.95/4.89/4.98 and 1.10/1.08/1.07
    against the state component, and 5.62/6.09/6.61, 7.17/7.83/8.96 and 2.63/—/— against the
    retrieval one. The EQUAL-LENGTH row is the one the previous run reported, and it is the row
    that compares a composed cell holding 17 swaps against a state cell holding 48.

    The control that separates "harder because composed" from "harder because longer" is the
    MATCHED-COST read: each component is also evaluated at the length whose cost equals the
    composed cell's at L, in that regime's cost model — the equal-tokens row, 1.00 by
    construction. On the retrieval side that control mostly does not exist, and the reason is the
    sampler rather than a choice: the window it pins the resolving write into gets exponentially
    harder to satisfy as the stream grows (3.1 ms per item at L=96, 167 ms at L=172, no
    admissible item at all at L=176), so the retrieval component cannot be run long enough to
    cost what the composed cell costs at L=64 or L=96. The control is registered as ABSENT there,
    not approximated by a shorter length that does not match the cost.

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

    THE TWO READS SCORE THE SAME ITEMS AND NOT THE SAME FLOORS, because the GUIDED protocol
    voids the profile's W axis: its format requires the whole of P then B at every event, which
    hands out the k + m live slots the one-structure bound prices. The rule that applies there is
    ``validity.s5_bind_v3_slot_profile``'s — a model under a scratchpad protocol must clear the
    TOP of the profile, not its admitted end. On the COMPONENT cells nothing moves, since their
    class is depth <= 1 and cost under the cell's own algorithm's minimum and a pad buys neither.
    On the COMPOSED cell the W bound is the whole of the class's first conjunct, so the surviving
    class contains the task and the cell is UNFLOORABLE on both of that protocol's channels;
    ``cell_floor(..., guided=True)`` returns no floor and reports ``pad_reach`` instead.

    A composition claim requires the components and the composed cell to be judged on the SAME
    read. Mixing them — components on GUIDED, composed on PLAIN — would manufacture a gap out
    of the eval mode, so ``verdict()`` is applied to each read separately and both are printed.

THE TRACE READ IS A FROM-SCRATCH-ARM INSTRUMENT AND IS NOT SCORED ON THE FRONTIER
    The TRACE read takes the model's own FINAL CHECKPOINT's value for the queried slot instead of
    the answer token. It exists because the answer channel is separable from the state: decoding
    this run's saved checkpoints, one seed writes the CORRECT value on 100% of two state cells
    and then emits a different token on 81% and 86% of them, so two published nulls were
    emission verdicts rather than architecture ones.

    IT IS DEFINED ONLY UNDER THE GUIDED PROTOCOL, which teacher-forces the events and generates
    every checkpoint, and the GUIDED protocol is the from-scratch arm's. ``TRACE_READ_ARMS`` is
    ``("local",)`` and ``assert_trace_read`` RAISES on any other arm rather than returning a
    number, because a frontier model has no checkpoint stream this harness generates: its trace
    is prose it chose to write, under its own budget, and reading a slot out of it would score a
    different object per model. FRONTIER CELLS ARE SCORED ON THE ANSWER AND MUST STAY SO.

    WHAT IT IS FLOORED BY, and the two cell kinds differ (factworld.validity, "THE GUIDED
    PROTOCOL"). The floors are the GUIDED protocol's, so they are the same numbers the guided
    ANSWER read is judged against — the discriminator is the protocol, not the channel:
      * the final checkpoint's queried slot IS the gold answer — 4000/4000 on the disjoint pool
        and 128/128 on the scored items at every registered cell — so a floor row's trace score
        is its answer score and the numbers transfer;
      * the COMPONENT cells are floored, at the plain protocol's floor: their rule is depth <= 1
        and cost under the cell's own algorithm's minimum, and a scratchpad buys neither;
      * the COMPOSED cell is NOT floored. Its registered class is the one-structure bound plus a
        step bound its own algorithm satisfies, and the guided protocol REQUIRES the whole of P
        and B to be written out at every event — so the k + m live slots that bound prices are
        handed to every policy. ``s5_bind_v3_operative_floor(..., guided=True)`` returns None
        there on both channels. ``s5_bind_v3_pad_reach`` measures how far the unfloorable class
        reaches: 0.719 on the 128 scored composed@48 items and 0.734 on the disjoint pool,
        against the plain protocol's floor of 0.234 and 0.200.
    So a composed-cell score under this protocol is a WITHIN-RUN COMPARISON — same seeds, same
    item count, matched depth and matched cost — and never a cleared floor. The DOWNWARD
    separation it carries does not need one; the other direction is not available at any
    registered length.

    IT IS NOT A PAIRED COMPARISON AND MUST NOT BE WRITTEN AS ONE. The two legs are different
    SPECS — ``s5_bind_local_v3`` and ``s5_bind_local_v3_state`` — drawing different item streams
    from different rng namespaces, at different lengths by construction (the pairing is what puts
    them at different lengths). What is matched is the seed, the item count and the forward-pass
    cost. A per-seed difference between them is therefore an unpaired difference of two
    proportions, and every interval quoted on one is the two cells' sampling error and never a
    within-item one.

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
    produced by any surface family measured on this rung — including the ones the class rule
    EXCLUDES, whose readings a floor does not cover: the widest is the fitted 25-feature ranker
    at 1.21x chance on the k=6 composed cell, i.e. 0.042 over informed chance.

    ON THE PAD READ THE MARGIN IS FRACTIONAL, and it is the same margin transferred rather than a
    second threshold: (a - f) / (1 - f) >= MARGIN_FRAC (= 0.1875 = 0.15 / (1 - 0.2), the additive
    margin re-expressed at the answer floor it was calibrated on). An additive 0.15 is undefined
    where a floor runs to 0.93 — the bar it sets there is above 1.0, so no score clears it — and
    per-slot pad floors do (``clears_headroom``).

    FORMS. A cell forms for an arch iff it CLEARS on at least SEEDS_CLEAR (= 2) of the seeds at
    every registered length. Seeds are counted, never averaged: this family is bimodal at the
    emergence threshold and a mean over one converged and two floored seeds is a number no seed
    produced. Per-seed values are reported in every table.

    AND GATES ARE CONJOINED PER SEED, never counted apart. forms, pad_tracks and readout_alive are
    each a count over seeds, and three independent counts are satisfied by a run in which no
    single seed satisfies two of them. A seed counts toward a composition claim only where all
    three hold FOR THAT SEED (``seeds_carrying``), and a verdict that interprets the composed cell
    without that conjunction RAISES (``SeedsNotConjoined``).

    POSITIVE CONTROL, and it gates the whole run. It is a DISJUNCTION over the components —
    some component clears somewhere on the grid THIS READ COVERS — and the (read, cell, length)
    pairs it requires are declared by ``control_grid()`` and evaluated by ``evaluate_control()``.
    If no required pair was measured the control RAISES (``ControlNotEvaluable``) and no verdict
    is returned at all.

    WHY A DISJUNCTION AND WHY A RAISE, both measured rather than chosen. A single-cell control on
    the state component voids a run in which the RETRIEVAL component reads 1.000 at every length
    on the same models — so the harness demonstrably trains and "the harness is not training this
    family" is false where such an abort fires. And a control fixed at L=16 is not on the GUIDED
    read's grid at all, so a bare seed count of 0 there reports an unevaluated cell and a floored
    one with the same number. A missing cell is not a failed control, and the difference is the
    whole verdict.

    MATCHED-COST CONTROL, and V1 is not available without it. Each component is also evaluated at
    the length whose FORWARD-PASS cost equals the composed cell's at L (``matched_lengths`` on
    MATCHED_AXIS), so "harder than its components beyond the step multiplier" is tested rather
    than assumed: without it, a composed cell at floor is equally explained by the cells being
    longer. ``matched_required()`` declares the (cell, length) pairs; where the sampler cannot
    reach one it is registered ABSENT, and an absent control is NOT a pass — V1 becomes
    V1_UNCONTROLLED. On the retrieval side the control mostly does not exist and the reason is
    the sampler (see BIND_MATCHED_MAX above), so V1 is unreachable at L=64 and L=96 by
    construction and the run should be read knowing that before it starts. The GUIDED read buys
    the control at GUIDED_MATCHED_FROM only (``guided_grid``): its decode is O(n L^2), so one
    operating point is affordable and three are not, and without that one purchase the guided
    read cannot reach V1 at all — every matched-cost cell is LONGER than any length it covers.

THE VERDICT TABLE (mechanical; ``verdict()`` returns exactly one of these, or raises)
    V5 HARNESS NULL          no component clears anywhere on this read's grid. Nothing is
                             claimable. Next move is the training recipe, not the instrument.
    V0 COMPOSED UNFLOORABLE  both components form, and the composed cell has NO FLOOR under this
                             protocol, so neither "clears" nor "does not clear" is available for
                             it. It is not a null and must never be read as one: the composed
                             cell's floor argument is a bound on live slots and this protocol's
                             format hands those slots out. The number reported in its place is
                             ``pad_reach``, what the excluded both-maps class scores on the exact
                             items, and it is a lower bound on that class's max rather than a
                             floor. The verdict this replaces is V2_NO_GAP_HERE, which the
                             GUIDED read reached off that floor.
    V4 COMPONENT UNREADABLE  a component does not FORM at its own registered lengths. The
                             composed cell cannot be read against it, because a composed failure
                             is then explained by the component that failed. Next move is that
                             component's budget or curriculum, not the composition.
    V3 GAP IS THE COST       the composed cell is at floor at L, and a component is ALSO at
                             floor at its matched-cost length. The composed cell's failure is
                             accounted for by how much longer it is, and no composition claim is
                             available from this run.
    V6 TRACKING GAP          both components FORM, the composed cell is floored and at floor, and
                             the composed cell DOES NOT WRITE THE SCRATCHPAD the protocol assumes:
                             its pad accuracy is far below the level the components reach on the
                             same read. A pad protocol scores the answer a model gives FROM ITS
                             OWN PAD, so a broken pad makes "the composition is hard" and "the
                             model cannot hold the intermediate state it was handed room for"
                             predict the same floored answer, and only the second is supported.
                             It is a gate and not a null: the composed cell's answer number is
                             real and reported, it just does not carry a composition reading. The
                             verdict this replaces is V1, and without the gate V1 is what a rule
                             reading only the answer returns.
    V8 NO SEED CARRIES IT    every gate passes on its own count over seeds, and no SEEDS_CLEAR
                             seeds pass all of them AT ONCE. forms, pad_tracks and readout_alive
                             are three counts, and a run whose components form on seeds 0/1, whose
                             pad is written on seed 2 and whose readout is alive on seeds 0/1
                             satisfies every one of them while describing no trained model — the
                             seed whose pad is good is the seed whose readout is dead. The claim is
                             conjoined PER SEED (``seeds_carrying``) before any composition verdict
                             is available, and this is what a failed conjunction returns.
    V1 UNCONTROLLED          both components FORM and the composed cell clears nowhere — the V1
                             pattern — but a matched-cost control was never measured, so "beyond
                             the step multiplier" is not established. The cells separate; the
                             cause does not.
    V1 COMPOSITION GAP       both components FORM, including at their matched-cost lengths, and
                             the composed cell does not clear at any registered length. The
                             composed cell is harder than its components beyond the multiplier.
                             This is the reading the instrument was built to produce.
    V2 NO GAP HERE           the composed cell FORMS. Composition is not a separate difficulty
                             at this operating point; the registered lengths or k must move
                             before the cell is worth buying on the frontier. Reachable on the
                             PLAIN read only — a formed composed cell needs a floor it cleared,
                             and the guided protocol leaves it none.

    Nothing in this table is derived from the within-cell statistic. theta_c is an
    identification impossibility on this rendering — within a kind the class label IS the
    printed clause, and a solver that cannot hold B fails on exactly the antisymmetric direction
    the kind-balancing annihilates (factworld.composition) — so it is a structure-SWITCH
    diagnostic and no verdict here reads it. The composition evidence is the three-cell
    comparison and only that.

WHAT WOULD STOP A FRONTIER SPEND — ``scout_plan()`` prices it, ``scout_verdict()`` applies it
    The repo's standing rule is that scout data showing a ceiling stops or redesigns a paid run
    rather than merely re-budgeting it. The scout is bought first; the roster run is bought only
    if the scout separates. Every threshold and THE ORDER THEY ARE EVALUATED IN are fixed here,
    before any frontier number exists:

      1. VOID (truncation) first. A cell over 10% finish=length or empty answers is a budget
         measurement, not a model one; it is re-run at a raised budget and enters no decision.
         Evaluated anywhere but first, it makes STOP(floor) fire on truncation — which is
         exactly how the published s5 L64 cliff was manufactured.
      2. STOP (ceiling) on the top model's composed@256.
      3. STOP (floor) against INFORMED CHANCE 1/(k-1) on composed@128, never against an
         operative floor. A frontier model reasons in visible tokens, which is a scratchpad, and
         the composed cell's floor argument prices LIVE SLOTS — so the cell is unfloorable there
         for the same reason it is unfloorable under the guided protocol locally.
      4. STOP (component) on either component for the top model.
      5. BUY on the composed SPREAD across models, which needs no baseline and therefore
         survives the composed cell's floor retraction unchanged.

    NO "CLEARS THE FLOOR" LANGUAGE MAY BE USED OF THE COMPOSED CELL in the scout or roster
    report (``SCOUT_COMPOSED_FLOOR_LANGUAGE``). The frontier cells are scored on the ANSWER only
    (``FRONTIER_READS``); the frontier spec carries no event_trace and ``assert_trace_read``
    raises on that arm.
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

# NO RETRIEVAL-QUERY COMPOSED ARM IS REGISTERED, and the reason is measured. The three cells put
# a STATE query on the composed stream, so retrieval enters as an operand resolver and never as
# the answer. The same stream read with a BIND query does need both maps — both one-structure
# replays fall to informed chance on it, 0.217 and 0.203 against 0.200 at k=6/L=96 — but its
# answer does not depend on the stream's TAIL: the sampler pins the queried object's resolving
# write into [0.1L, 0.75L], which is the pin that PROVES the retrieval component's floor, and
# under it no event past 0.75L can move a bind answer. A solver carrying both maps and replaying
# only the first 90% of the stream scores 1.000 on that arm and 0.927 replaying the first 75%,
# against 0.097 and 0.143 on the state query, whose own gate puts the queried agent's last move
# inside the final 10%. A floor-proved retrieval component needs the resolving write far from the
# end; a query that reads the whole stream needs it near the end. ``scripts/validate_suite.py``
# flags a registered bind-query arm on exactly that row.
LOCAL_LENGTHS = (48, 64, 96)              # the k=6 grid the COMPOSED cell registers; each
                                          # component's own grid is DERIVED from it (see PAIRINGS)
CONTROL_LENGTH = 16                       # the in-distribution positive control (state cell)
TRAIN_LENGTHS = (16, 32, 48, 64, 96)      # eval grid is IN distribution: this run is about
                                          # composition, not length extrapolation

# THE TWO PAIRINGS, and neither is optional, because they answer different questions.
#
#   work    each component is read at the length that carries the SAME AMOUNT OF ITS OWN WORK as
#           the composed stream: the composed cell at L holds p_swap L swaps and (1 - p_swap) L
#           gives, so the state component's partner is that swap count and the retrieval
#           component's is that give count (validity.s5_bind_v3_work_match). This is each
#           component's REGISTERED grid — the lengths its FORMS verdict is read at — because
#           reading a component at the composed cell's own L compares 1/p_swap times the state
#           work and 1/(1 - p_swap) times the retrieval work, and a shallower cell scoring like a
#           deeper one is equally consistent with zero composition cost and with a composition
#           cost worth the depth difference.
#   tokens  each component is read at the length whose FORWARD PASS costs what the composed cell's
#           costs. This is the MATCHED-COST control — "harder because composed" against "harder
#           because longer" — and it is a different question: its multiplier is 1.00 by
#           construction, where the work pairing's is whatever the composed cell's extra structure
#           costs.
#
# The step multiplier is not one number and is registered under both conventions plus the
# equal-length one the tables print (``step_multipliers``).
PAIRINGS = ("work", "tokens")
REGISTERED_PAIRING = "work"               # which pairing sets each component's FORMS grid
MATCHED_PAIRING = "tokens"                # which pairing is the matched-COST control

# The two pairings AS REGISTERED, at the k=6 operating point. Both are measured — the work
# pairing off the composed streams' own event counts, the token pairing by solving each
# component's affine token cost for the composed cell's — and ``register()`` re-measures both and
# RAISES on any disagreement, so a sampler change cannot silently move the comparison.
#
# WORK_PROBE_N IS PART OF THE REGISTRATION. The event count is a sample mean over a deterministic
# stream, and the sampler's query gate biases it above p_swap L (the state gate needs the queried
# agent to have moved at least twice and recently, which selects swap-heavier streams): the
# composed cell at k=6/L=48 carries 17.0 swaps against 16.0 from p_swap alone. Its standard error
# at n = 200 is a third of an event, so rounding it is ambiguous at a different n — k=12/L=128
# rounds to 43 at n = 200 and 42 at n = 120. Fixing the probe makes the registered partner a
# reproducible number rather than a draw.
WORK_PROBE_N = 200
WORK_MATCHED = {48: {"state": 17, "bind": 31},
                64: {"state": 23, "bind": 41},
                96: {"state": 34, "bind": 62}}
TOKEN_MATCHED = {48: {"state": 80, "bind": 132},
                 64: {"state": 108, "bind": None},
                 96: {"state": 160, "bind": None}}


# THE STATE COMPONENT'S DEPTH LADDER. It is the leg with a depth axis at all — the retrieval
# component's own algorithm chains ONE hop at every (k, m, density) its spec can take — so its
# grid carries rungs ABOVE its work-matched ones and the run reports the profile across them.
# Carrier hops at k=6 are L/3: 5.7 / 7.7 / 11.3 / 16.0 / 26.7 / 42.7. The ladder is cut at the
# bottom by the FLOOR and not by the sampler: 1.26x informed chance at L=8 (2.7 hops, where a
# one-hop read still has traction) against 1.00-1.09x from L=12 to L=256, while the sampler costs
# 1.00-1.01 stream restarts per item over that whole range. FORMS is still read at the
# work-matched subset only — a ladder built to span a component's range necessarily contains
# rungs it is not required to clear, and requiring them would make V4 the standing verdict.
PROFILE_LENGTHS = {"state": (17, 23, 34, 48, 80, 128), "bind": (31, 41, 62)}


def registered_lengths(cell, lengths=LOCAL_LENGTHS, work=None):
    """The lengths a cell's own FORMS verdict is read at.

    The composed cell's are ``lengths``; each component's are the WORK-MATCHED partners of those,
    which is the whole point of the pairing — a component judged at the composed cell's L is
    judged on 1/p_swap (state) or 1/(1 - p_swap) (retrieval) times the work the composed cell
    actually makes it do.
    """
    work = WORK_MATCHED if work is None else work
    if cell == "composed":
        return tuple(lengths)
    return tuple(sorted({work[L][cell] for L in lengths if work.get(L, {}).get(cell)}))


def matched_lengths_for(cell, lengths=LOCAL_LENGTHS, token=None):
    """The MATCHED-COST control lengths for one component: the token pairing, unreachable ones
    dropped rather than replaced by a shorter cell that does not match the cost."""
    token = TOKEN_MATCHED if token is None else token
    return tuple(sorted({token[L][cell] for L in lengths if token.get(L, {}).get(cell)}))

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
GUIDED_MATCHED_FROM = 48                  # the one composed length whose MATCHED-COST control the
                                          # guided read also buys (``guided_grid``), so V1 is
                                          # reachable on that read at one operating point
MATCHED_AXIS = "tokens"                   # the FORWARD-PASS cost model, which is the from-scratch
                                          # regime's; "steps" is the scratchpad solver's
N_FIT = 2000                              # surface-ranker fit sample PER BLOCK: its held-out
N_FIT_BLOCKS = V.S5_BIND_V3_SURFACE_BLOCKS  # curve has flattened by 1000 and the fit runs on
N_SCORE = 4000                            # N_FIT_BLOCKS * N_FIT pooled, with the block-to-block
                                          # spread at N_FIT reported beside it
Z_CLEAR = 3.0
MARGIN = 0.15
SEEDS_CLEAR = 2

# THE MARGIN ON THE PAD READ, and it is the registered one transferred rather than a second
# threshold. MARGIN is ADDITIVE and was set where the ANSWER floor sits just above informed chance
# 1/(k-1) = 0.2 at k=6: 0.15 is three quarters of that chance, and every floor it was calibrated
# against reads 1.05-1.17x it. A per-SLOT pad floor sits at 2.3-5.6x its own chance and runs to
# 0.93, where the same 0.15 is not a bar but an impossibility — at composed@16 it asks for 1.077,
# so no score whatever clears it, at exactly the length where the pad is nearly written (0.9836
# measured). A bar above 1.0 is not a strict rule; it is an undefined one.
#
# WHAT REPLACES IT: the same lift, read as a share of the HEADROOM the floor leaves.
#     (a - f) / (1 - f)  >=  MARGIN_FRAC
# It is defined at every floor below 1, it is monotone in a, and it asks the same question the
# additive rule asks — how much of what the floor does not already explain does the model explain.
#
# IT IS NOT FITTED TO THIS DATA, and the constant says so: MARGIN_FRAC is MARGIN re-expressed at
# the operating point MARGIN was registered at, 0.15 / (1 - 0.2) = 0.1875. At the answer floors it
# was set against it reproduces the registered bar to within a point; nothing about the pad
# numbers enters it, and it was written down before the pad scores were confronted with it. The
# free parameter it removes is the one the additive rule had all along: which floor the 0.15 was
# for.
ANSWER_CHANCE_AT_REGISTRATION = 0.2       # 1/(k-1) at k=6 — the floor MARGIN was calibrated on
MARGIN_FRAC = MARGIN / (1.0 - ANSWER_CHANCE_AT_REGISTRATION)

# ---- the trace read ---------------------------------------------------------------------------
# The read is the model's own FINAL CHECKPOINT's value for the queried slot. It is defined only
# where the harness generates that checkpoint stream, which is the GUIDED protocol, which is the
# from-scratch arm's. A frontier model's visible trace is prose under its own budget and holds no
# slot this harness can index, so no frontier cell is scored on it — see ``assert_trace_read``.
TRACE_READ = "trace"
TRACE_READ_ARMS = ("local",)
TRACE_READ_REQUIRES = "guided"            # the only protocol that generates the checkpoints
FRONTIER_READS = ("answer",)              # what a frontier cell is scored on, and all it is


class TraceReadNotAvailable(RuntimeError):
    """The trace read was asked for on an arm that has no harness-generated checkpoint stream.

    Deliberately an exception and not a fallback to the answer read: a silent fallback would put
    two different quantities in one column, and the whole point of the trace read is that the two
    quantities come apart.
    """


def assert_trace_read(arm, read=TRACE_READ_REQUIRES):
    """Raise unless the trace read is available on this arm under this protocol.

    Args:
        arm: "local" for the from-scratch arm, anything else for a frontier one.
        read: the protocol the score came from; the trace read needs ``TRACE_READ_REQUIRES``.

    Raises:
        TraceReadNotAvailable: the arm is not in ``TRACE_READ_ARMS``, or the read is not the
            guided one. Frontier cells are scored on ``FRONTIER_READS`` and must stay so.
    """
    if arm not in TRACE_READ_ARMS:
        raise TraceReadNotAvailable(
            f"the trace read is a from-scratch-arm instrument ({TRACE_READ_ARMS}); arm={arm!r} "
            f"is scored on {FRONTIER_READS}. A frontier model's visible trace is prose under its "
            "own budget, not a checkpoint stream this harness generates, so a slot read out of "
            "it is a different quantity per model.")
    if read != TRACE_READ_REQUIRES:
        raise TraceReadNotAvailable(
            f"the trace read needs the {TRACE_READ_REQUIRES!r} protocol, which generates the "
            f"per-event checkpoints; read={read!r} has no checkpoint stream to index.")
    return True

# ---- the frontier scout --------------------------------------------------------------------
SCOUT_COMPOSED_LENGTHS = (128, 256)       # the composed cell carries the length axis
# The k=12 work pairing, measured the same way as WORK_MATCHED: composed@128/192/256 carry
# 43/64/85 swaps and 85/128/171 gives. The scout gates on the components at the partners of its
# DEEPEST composed length, because a component that holds the work the composed cell makes it do
# at 256 holds the smaller amount it makes it do at 128 — the old gate set the bar at state@256
# (42.7 carrier hops) for a composed cell that never has to clear more than 14.2.
FRONTIER_WORK_MATCHED = {128: {"state": 43, "bind": 85},
                         192: {"state": 64, "bind": 128},
                         256: {"state": 85, "bind": 171}}
SCOUT_COMPONENT_LENGTHS = {c: (FRONTIER_WORK_MATCHED[max(SCOUT_COMPOSED_LENGTHS)][c],)
                           for c in ("state", "bind")}
SCOUT_N = 40
# PER-CELL COMPLETION BUDGETS, and the 8192 they replace was a VALIDITY defect rather than a
# price one. ``benchmark.cost_estimate`` prices ``assumed_output_tokens`` and never reads
# ``max_new_tokens``, so the cheap budget bought nothing; what it did buy was truncation, scored
# as wrong (``run_frontier_benchmark._run_attempt`` empties a finish=length reply). The composed
# PROMPT alone is 2,984 tokens at L=128 and 5,583 at L=256 before a single reasoning token, and
# ``benchmark.py``'s s5_concrete facet already carries 16,384/32,768 for exactly this reason
# (opus and sonnet finished at length with no visible answer at 8,192). The budgets are keyed by
# (cell, length) so a length that is not registered has no budget rather than a default one.
SCOUT_BUDGETS = {("composed", 128): 16384, ("composed", 256): 32768,
                 ("state", 85): 16384, ("bind", 171): 16384}
SCOUT_MAX_NEW_TOKENS = min(SCOUT_BUDGETS.values())   # the floor across the scout's cells
# three models spanning the roster's measured range, not three near neighbours: the top of the
# owner's ranking, the mid-tier reasoner that sits surprisingly high on it, and the bottom.
SCOUT_MODELS = ("openai/gpt-5.5", "z-ai/glm-5.2", "nvidia/nemotron-3-ultra-550b-a55b")
SCOUT_EFFORT = "high"
ROSTER_N = 100                            # what a RANKING needs, against the scout's 40
SCOUT_CEILING = 0.90                      # composed cell at/above this for the top model -> STOP
SCOUT_SEPARATION = 0.20                   # composed-cell spread across the scout -> BUY
SCOUT_COMPONENT_MIN = 0.80                # components must be this high for the top model
SCOUT_TRUNCATION_MAX = 0.10               # per-cell finish=length OR empty rate above this -> VOID
SCOUT_FLOOR_SE = 2.0                      # "within this many se of informed chance" -> STOP(floor)
SCOUT_STOP_FLOOR_LENGTH = 128             # the composed length STOP(floor) is read at
SCOUT_STOP_CEILING_LENGTH = 256           # the composed length STOP(ceiling) is read at


class ScoutCellVoid(RuntimeError):
    """A scout cell whose truncation or empty rate is over ``SCOUT_TRUNCATION_MAX``.

    Deliberately an exception rather than a verdict. A truncated cell is not a measurement of
    the model, and letting it reach the stop table is how the published "s5 L64 cliff" was
    manufactured the first time: a budget too small to hold the answer reads as a floor. The
    cell is re-run at a raised budget and may not enter any stop or buy decision until it is.
    """


# THE COMPOSED CELL HAS NO FLOOR ON THE FRONTIER, and every threshold below is written so that
# nothing needs one. A frontier model reasons in visible tokens, which is a SCRATCHPAD, and the
# composed cell's floor argument is the one-structure live-slot bound W <= max(k, m) + 1 against
# the task's k + m + 1 — the same bound the guided protocol's format voids locally, measured at
# composed@48 as a both-maps replay reaching 0.719 against a printed floor of 0.234. So:
#   * STOP(floor) is stated against INFORMED CHANCE, 1/(k-1) — the initial map is stated and the
#     gold answer is never the queried agent's own starting value — and never against "the
#     operative floor", which does not exist here;
#   * BUY is a SPREAD rule across models on one cell, which needs no baseline at all and is why
#     it survives the retraction unchanged;
#   * NO "CLEARS THE FLOOR" LANGUAGE MAY APPEAR for the composed cell anywhere in the scout or
#     roster report. ``SCOUT_COMPOSED_FLOOR_LANGUAGE`` is the banned phrasing, kept as data so
#     the report generator can assert on its own output rather than rely on a reviewer.
SCOUT_COMPOSED_FLOOR_LANGUAGE = ("clears the floor", "clears its floor", "above the floor",
                                 "off the floor", "cleared the floor", "beats the floor")


def scout_informed_chance(cells=None):
    """Informed chance on the frontier cells: 1/(k-1) at the k=12 operating point.

    The stated initial map means a guesser knows the queried agent's own starting value is not
    the answer, so chance is over k-1 candidates and not k. It is the number STOP(floor) is
    read against, and it is not a floor: no policy class is being priced, only a guess.
    """
    cells = FRONTIER_CELLS if cells is None else cells
    k = TK.CANONICAL[cells["composed"]].k
    return 1.0 / (k - 1)


def scout_cells(models=SCOUT_MODELS, n=SCOUT_N, effort=SCOUT_EFFORT):
    """The scout's (cell, length, n, budget) plan — the single source of truth for the runner
    AND the price, so a cell cannot be run at a budget it was not priced at.

    Four cells: the composed cell at both scout lengths, since it carries the length axis and
    that is where separation would come from, and each component at the WORK-MATCHED partner of
    the DEEPEST composed length (state@85, bind@171) — a component that holds the work the
    composed cell makes it do at 256 holds the smaller amount it makes it do at 128, and the
    component reading is a GATE, not a ranking.
    """
    out = []
    for L in SCOUT_COMPOSED_LENGTHS:
        out.append({"cell": "composed", "task": FRONTIER_CELLS["composed"], "length": L, "n": n,
                    "settings": {"effort": effort,
                                 "max_new_tokens": SCOUT_BUDGETS[("composed", L)]}})
    for c in ("state", "bind"):
        for L in SCOUT_COMPONENT_LENGTHS[c]:
            out.append({"cell": c, "task": FRONTIER_CELLS[c], "length": L, "n": n,
                        "settings": {"effort": effort,
                                     "max_new_tokens": SCOUT_BUDGETS[(c, L)]}})
    return out


def scout_verdict(scores, n=SCOUT_N, chance=None, models=SCOUT_MODELS):
    """The scout's decision rule, applied mechanically and IN THE REGISTERED ORDER.

    Fixed before any scout result exists. ``scores`` is
    ``{model: {"composed@128": row, "composed@256": row, "state@85": row, "bind@171": row}}``
    where a row carries ``match`` and the two validity rates ``length_rate`` (finish=length) and
    ``empty_rate``. Returns ``(code, why, detail)``.

    The order is the rule. Evaluated in any other order, a truncated cell reads as a floor and
    STOP(floor) fires on a budget defect — which is how the published s5 L64 cliff was made.

      1. VOID_TRUNCATION   any cell over SCOUT_TRUNCATION_MAX on finish=length OR empty answers.
                           That cell is re-run at a raised budget and enters no decision. Raises
                           ScoutCellVoid rather than returning, so a void cannot be read as a
                           result.
      2. STOP_CEILING      the top model's composed@256 >= SCOUT_CEILING. A ceiling cannot rank
                           a roster, so the roster run buys nothing; redesign (raise k or L).
      3. STOP_FLOOR        every scout model within SCOUT_FLOOR_SE standard errors of INFORMED
                           CHANCE on composed@128. Redesign, do not re-budget: the frontier is a
                           scratchpad regime, so the composed cell is unfloorable there for the
                           same reason it is unfloorable on the guided read, and this is a
                           statement about a guess baseline and not about a cleared floor.
      4. STOP_COMPONENT    either component below SCOUT_COMPONENT_MIN for the top model. The
                           composed cell is then state-limited or retrieval-limited and its
                           number is unreadable for the same reason a local V4 is.
      5. BUY / NO_BUY      BUY iff the composed spread across the three models is >=
                           SCOUT_SEPARATION at EITHER composed length AND both components are >=
                           SCOUT_COMPONENT_MIN for EVERY scout model. The spread is computed per
                           length and both are reported; "either" is registered here, before any
                           number exists, because two composed lengths are registered and a rule
                           that named one would discard the other's information after the fact.
    """
    chance = scout_informed_chance() if chance is None else chance
    ceiling_key = f"composed@{SCOUT_STOP_CEILING_LENGTH}"
    floor_key = f"composed@{SCOUT_STOP_FLOOR_LENGTH}"
    models = [m for m in models if m in scores]
    detail = {"chance": chance, "n": n, "models": models}

    void = [f"{m} {key} (length {row.get('length_rate'):.2f} / empty "
            f"{row.get('empty_rate'):.2f})"
            for m in models for key, row in sorted(scores[m].items())
            if max(row.get("length_rate") or 0.0, row.get("empty_rate") or 0.0)
            > SCOUT_TRUNCATION_MAX]
    detail["void_cells"] = void
    if void:
        raise ScoutCellVoid(
            f"{len(void)} cell(s) over the {SCOUT_TRUNCATION_MAX:.0%} truncation/empty bar: "
            f"{'; '.join(void)}. Each is re-run at a raised budget and enters no stop or buy "
            "decision until it is; a truncated cell is a budget measurement, not a model one.")

    ranked = sorted(models, key=lambda m: -(scores[m].get(ceiling_key, {}).get("match") or 0.0))
    top = ranked[0] if ranked else None
    detail["top_model"] = top
    comps = {c: {m: scores[m].get(f"{c}@{L}", {}).get("match")
                 for m in models}
             for c in ("state", "bind") for L in SCOUT_COMPONENT_LENGTHS[c]}
    spreads = {}
    for L in SCOUT_COMPOSED_LENGTHS:
        vals = [scores[m].get(f"composed@{L}", {}).get("match") for m in models]
        vals = [v for v in vals if v is not None]
        spreads[L] = (max(vals) - min(vals)) if len(vals) > 1 else None
    detail["spreads"] = spreads
    detail["components"] = comps
    # DESCRIPTIVE, computed unconditionally so an early stop still reports what the later rules
    # would have read. No branch below reads these; each recomputes its own condition.
    se_c = math.sqrt(max(1e-12, chance * (1.0 - chance) / n))
    detail["informed_chance_band"] = [round(chance - SCOUT_FLOOR_SE * se_c, 4),
                                      round(chance + SCOUT_FLOOR_SE * se_c, 4)]
    detail["components_all_above_min"] = all(
        v is not None and v >= SCOUT_COMPONENT_MIN for c in comps for v in comps[c].values())
    detail["best_spread"] = max((s for s in spreads.values() if s is not None), default=None)

    top_ceiling = scores.get(top, {}).get(ceiling_key, {}).get("match")
    if top_ceiling is not None and top_ceiling >= SCOUT_CEILING:
        return "STOP_CEILING", (
            f"the top scout model ({top}) reads {top_ceiling:.3f} on {ceiling_key}, at or above "
            f"{SCOUT_CEILING}. A ceiling cannot rank the roster, so the roster run buys nothing; "
            "redesign (raise k or L) before spending."), detail

    se = math.sqrt(max(1e-12, chance * (1.0 - chance) / n))
    at_chance = {m: scores[m].get(floor_key, {}).get("match") for m in models}
    if at_chance and all(v is not None and v <= chance + SCOUT_FLOOR_SE * se
                         for v in at_chance.values()):
        return "STOP_FLOOR", (
            f"every scout model is within {SCOUT_FLOOR_SE:.0f} se of informed chance "
            f"({chance:.4f} = 1/(k-1), se {se:.4f}) on {floor_key}: {at_chance}. Zero separation "
            "across the roster's range means zero expected pairwise separations in the full run. "
            "Redesign, do not re-budget. Stated against informed chance and not against an "
            "operative floor, because the frontier is a scratchpad regime and the composed cell "
            "has no floor there."), detail

    if top is not None:
        low = {c: v[top] for c, v in comps.items()
               if v.get(top) is None or v[top] < SCOUT_COMPONENT_MIN}
        if low:
            return "STOP_COMPONENT", (
                f"component(s) {sorted(low)} read {low} for the top model ({top}), below "
                f"{SCOUT_COMPONENT_MIN}. The composed cell is state-limited or "
                "retrieval-limited, so its number does not measure the composition; redesign "
                "the component that failed before buying a ranking."), detail

    comp_ok = detail["components_all_above_min"]
    best = detail["best_spread"]
    if best is not None and best >= SCOUT_SEPARATION and comp_ok:
        return "BUY", (
            f"the composed spread across the three scout models reaches {best:.3f} "
            f"({ {L: (None if s is None else round(s, 3)) for L, s in spreads.items()} }), at or "
            f"above {SCOUT_SEPARATION}, while both components sit at or above "
            f"{SCOUT_COMPONENT_MIN} for every scout model. The composed cell discriminates "
            "inside the roster's range where the components do not."), detail
    return "NO_BUY", (
        f"no stop rule fires and the buy rule is not met: composed spread "
        f"{ {L: (None if s is None else round(s, 3)) for L, s in spreads.items()} } against "
        f"{SCOUT_SEPARATION}, components at or above {SCOUT_COMPONENT_MIN} for every model: "
        f"{comp_ok}. The roster run is not bought."), detail


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


def matched_lengths(tok, cells=LOCAL_CELLS, lengths=LOCAL_LENGTHS, axis=MATCHED_AXIS,
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


def work_matched_lengths(tok, cells=LOCAL_CELLS, lengths=LOCAL_LENGTHS, n_probe=WORK_PROBE_N,
                         n_needed=N_EVAL + N_FIT + N_SCORE):
    """For each composed length, the component length carrying the SAME AMOUNT OF ITS OWN WORK.

    The composed stream at L is counted, not modelled: it holds ``n_swap`` swaps and ``n_give``
    gives, and the component streams are all of one kind, so the partners are those two counts
    (validity.s5_bind_v3_work_match). Both cells then have the same carrier chain
    (``validity.s5_bind_v3_carrier_hops``) and the same write count in the leg being compared.

    The row shape matches ``matched_lengths`` so the two pairings can be read side by side, and
    feasibility is checked at the pool size the floor will ask for, for the same reason.
    """
    comp = TK.CANONICAL[cells["composed"]]
    out = {}
    for L in lengths:
        ex = TK.generate(comp, "test", n=n_probe, length=L)
        ns, ng = V.s5_bind_v3_shape(ex)
        s, t = cell_cost(comp, L, tok, n_probe)
        row = {"composed_L": L, "composed_steps": s, "composed_tokens": t, "axis": "work",
               "composed_n_swap": ns, "composed_n_give": ng,
               "composed_carrier_hops": round(V.s5_bind_v3_carrier_hops(comp.k, ns), 2),
               "target_cost": None}
        want = V.s5_bind_v3_work_match(ns, ng)
        for key in ("state", "bind"):
            spec = TK.CANONICAL[cells[key]]
            cand = want[key]
            reachable, cs, ct, hops = False, None, None, None
            try:
                TK.generate(spec, "test", n=n_needed, length=cand)
                cs, ct = cell_cost(spec, cand, tok, 24)
                cex = TK.generate(spec, "test", n=24, length=cand)
                cns, _cng = V.s5_bind_v3_shape(cex)
                hops = round(V.s5_bind_v3_carrier_hops(spec.k, cns), 2)
                reachable = True
            except Exception:            # the sampler cannot fill the pool at the matched length
                pass
            row[key] = {"L": cand if reachable else None, "cost": ct, "steps": cs,
                        "carrier_hops": hops, "target": cand, "reachable": bool(reachable),
                        "work": "swaps" if key == "state" else "gives",
                        "work_count": ns if key == "state" else ng}
        out[L] = row
    return out


def pairings(tok, cells=LOCAL_CELLS, lengths=LOCAL_LENGTHS, which=PAIRINGS):
    """Both registered pairings, keyed by name — ``{"work": ..., "tokens": ...}``."""
    out = {}
    for name in which:
        out[name] = (work_matched_lengths(tok, cells, lengths) if name == "work"
                     else matched_lengths(tok, cells, lengths, axis=name))
    return out


def as_pairings(ml):
    """Read a stored ``matched_lengths`` field under either shape.

    Records written before the work pairing existed hold one flat ``{composed_L: row}`` dict; that
    is the TOKEN pairing and is labelled as such rather than left unnamed.
    """
    if not ml:
        return {}
    if all(k in PAIRINGS for k in ml):
        return ml
    return {MATCHED_PAIRING: {int(k): v for k, v in ml.items()}}


def step_multipliers(tok, paired, cells=LOCAL_CELLS, lengths=LOCAL_LENGTHS):
    """The composed cell's cost over each component's, under every convention that is registered.

    THREE numbers per (component, length), because they answer three questions and quoting one
    as "the step multiplier" hides the other two:

      equal_length  the component read at the composed cell's own L — what the flagship tables
                    print, and the number that is confounded, since the two cells then carry
                    different amounts of the component's work;
      work          the component read at its WORK-MATCHED length — the extra cost the composed
                    cell's structure imposes at equal component work, which is the multiplier the
                    depth-matched comparison is against;
      tokens        the component read at its TOKEN-MATCHED length — 1.00 by construction, which
                    is what makes that pairing a control rather than a comparison.

    Each in both cost models: ``steps`` is the charged-step convention a scratchpad solver pays,
    ``tokens`` the forward pass a streaming model pays.
    """
    out = {}
    for L in lengths:
        s, t = cell_cost(TK.CANONICAL[cells["composed"]], L, tok)
        row = {"composed_L": L, "composed_steps": s, "composed_tokens": t}
        for key in ("state", "bind"):
            spec = TK.CANONICAL[cells[key]]
            row[key] = {}
            for name, cl in (("equal_length", L),
                             ("work", paired.get("work", {}).get(L, {}).get(key, {}).get("L")),
                             ("tokens", paired.get("tokens", {}).get(L, {}).get(key, {})
                              .get("L"))):
                if cl is None:
                    row[key][name] = {"L": None, "steps": None, "tokens": None}
                    continue
                cs, ct = cell_cost(spec, cl, tok, 24)
                row[key][name] = {"L": cl, "steps": round(s / cs, 2),
                                  "tokens": round(t / ct, 2)}
        out[L] = row
    return out


# ---- floors ---------------------------------------------------------------------------------
def cell_floor(spec, L, n_eval=N_EVAL, n_fit=N_FIT, n_score=N_SCORE, n_blocks=N_FIT_BLOCKS,
               guided=False):
    """The operative floor at (cell, length) under one PROTOCOL, and everything needed to audit it.

    Measured on ``n_score`` items drawn from the SAME deterministic test stream as the items a
    solver is scored on and DISJOINT from them, because the max over admitted rows carries an
    upward selection bias at small n (the published 1.30x on the state component was a high draw
    at n=500 and is 0.98x at n=4000) and because the fitted ranker has to be scored out of
    sample. The same rows on the exact scored items are reported beside it as the house-rule
    check; where the two differ the larger is the number a score must clear.

    ``guided=True`` prices the SCRATCHPAD protocol, which is the one both the guided answer read
    and the trace read decode under. It drops the live-slot conjunct, because the format hands
    out the k + m slots that conjunct prices, and it adds the checkpoint-shaped rows, which that
    format makes available. On a COMPONENT cell the floor is unchanged in kind and can only rise
    (one extra family of admitted rows). On the COMPOSED cell there is no floor at all: ``floor``
    is None, ``basis`` is 'unfloorable', and ``pad_reach`` carries what the excluded both-maps
    class scores on the exact items, so the retraction leaves a number. ``floor_plain`` keeps the
    plain protocol's number beside it as the reference it is — it is what the PLAIN read scores
    against, and it must never be read as a bar the guided score cleared.

    THE FITTED RANKER IS MEASURED AND REPORTED BUT NEVER ADDED TO THE FLOOR: no implementation of
    it achieves a price the class rule admits at any cell (validity.s5_bind_v3_surface_price),
    and its price is recomputed here from the weights the fit actually produced, so the exclusion
    is a property of the measured policy rather than of the row's name. It is fitted on
    ``n_blocks * n_fit`` items pooled and refitted on each ``n_fit`` block, so the number ships
    with the spread its fit budget leaves.
    """
    k, m = spec.k, spec.n_objects_active
    n_fit_total = n_fit * max(1, n_blocks)
    pool = TK.generate(spec, "test", n=n_eval + n_fit_total + n_score, length=L)
    scored, fit = pool[:n_eval], pool[n_eval:n_eval + n_fit_total]
    big = pool[n_eval + n_fit_total:]
    named = V.s5_bind_v3_is_named(big)
    query = V.s5_bind_v3_query_kind(big)
    unfloorable = bool(guided) and not named
    admits = V.s5_bind_v3_guided_admits if guided else V.s5_bind_v3_admits

    def rows_on(items, adm, ckpt):
        ns, ng = V.s5_bind_v3_shape(items)
        keep = tuple(r for r in V.s5_bind_v3_family_rows(k, m, ns, ng, named, query)
                     if adm(r, k, m, ns, ng, named, query))
        fl = dict(V.s5_bind_v3_floors(items, k, m))
        fl.update(V.s5_bind_v3_family_floors(items, k, m, named, query, rows=keep))
        if ckpt:
            fl.update(V.s5_bind_v3_ckpt_floors(items))
        return fl, ns, ng

    fl, ns, ng = rows_on(big, admits, guided)
    op = V.s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, query, guided=guided)
    sb = V.s5_bind_v3_surface_bound(fit, k, held_out=big, blocks=max(1, n_blocks))
    price = V.s5_bind_v3_surface_price(k, m, ns, ng, named, query,
                                       None if sb is None else sb["weights"])
    # the ranker can only ever RAISE a floor that exists; where the protocol leaves none it has
    # nothing to raise, and adding it would put a number back where the retraction removed one.
    if (price["admitted"] and sb is not None and not unfloorable
            and (op is None or sb["held_out"] > op)):
        op = sb["held_out"]
    fls, nss, ngs = rows_on(scored, admits, guided)
    op_scored = V.s5_bind_v3_operative_floor(fls, k, m, nss, ngs, named, query, guided=guided)
    have = [x for x in (op, op_scored) if x is not None]
    floor = max(have) if have else None
    _w, s = V.s5_bind_v3_task_cost(k, m, ns, ng, named, query)
    if guided:
        pb, _ns, _ng = rows_on(big, V.s5_bind_v3_admits, False)
        ps, _ns, _ng = rows_on(scored, V.s5_bind_v3_admits, False)
        plain = max(x for x in
                    (V.s5_bind_v3_operative_floor(pb, k, m, ns, ng, named, query),
                     V.s5_bind_v3_operative_floor(ps, k, m, nss, ngs, named, query))
                    if x is not None)
    else:
        plain = floor
    return {"cell": spec.name, "L": L, "k": k, "m": m, "query": query, "named": named,
            "protocol": "guided" if guided else "plain",
            "chance": 1.0 / (k - 1), "floor": floor,
            "floor_disjoint": op, "floor_on_scored_items": op_scored,
            "floor_plain": plain,
            "pad_reach": (V.s5_bind_v3_pad_reach(scored) if guided and not named else None),
            "surface_held_out": None if sb is None else sb["held_out"],
            "surface_n_fit": None if sb is None else sb["n_fit"],
            "surface_block_spread": None if sb is None else sb["block_spread"],
            "surface_blocks": None if sb is None else sb["blocks"],
            "surface_price": price, "surface_in_floor": bool(price["admitted"]),
            "basis": V.s5_bind_v3_floor_basis(k, m, ns, ng, named, query, guided=guided),
            "admitted_rows": {r: round(v, 4) for r, v in
                              sorted(fl.items(), key=lambda x: -x[1])
                              if admits(r, k, m, ns, ng, named, query)},
            "charged_steps": s, "n_swap": ns, "n_give": ng}


def trace_floor(spec, L, n_scored=N_GUIDED, n_big=N_SCORE, arm="local"):
    """The GUIDED protocol's floor at (cell, length), for BOTH of its channels, plus the
    plain protocol's number as a reference and the checkpoint diagnostics.

    Same pool discipline as ``cell_floor`` — rows measured on a DISJOINT pool because the max
    over rows carries an upward selection bias at small n, and on the exact scored items because
    a floor must be recomputed from the items it is read against, with the larger of the two
    operative. ``n_scored`` defaults to the GUIDED read's own sample, since that is the only
    protocol the trace read is defined under.

    ``answer_floor`` AND ``trace_floor`` ARE THE SAME NUMBER, and both are None on the COMPOSED
    cell. That is the cell's answer and not a missing measurement: the guided protocol hands out
    the k + m live slots the one-structure bound prices, whichever token the prediction is finally
    read from, and what is left of the class contains the task. ``pad_reach`` measures how far the
    unfloorable class gets (the best both-maps policy strictly cheaper than the task) so the
    distance from the plain protocol's floor is a number rather than a blank, and
    ``answer_floor_plain`` carries that plain number — as a reference, never as a bar the guided
    score cleared.
    """
    assert_trace_read(arm)
    k, m = spec.k, spec.n_objects_active
    pool = TK.generate(spec, "test", n=n_scored + n_big, length=L)
    scored, big = pool[:n_scored], pool[n_scored:]

    def rows_on(items):
        ns, ng = V.s5_bind_v3_shape(items)
        named = V.s5_bind_v3_is_named(items)
        query = V.s5_bind_v3_query_kind(items)
        fam = tuple(r for r in V.s5_bind_v3_family_rows(k, m, ns, ng, named, query)
                    if V.s5_bind_v3_admits(r, k, m, ns, ng, named, query)
                    or V.s5_bind_v3_guided_admits(r, k, m, ns, ng, named, query))
        fl = dict(V.s5_bind_v3_floors(items, k, m))
        fl.update(V.s5_bind_v3_family_floors(items, k, m, named, query, rows=fam))
        return fl, V.s5_bind_v3_ckpt_floors(items), (ns, ng, named, query)

    fs, cs, sh_s = rows_on(scored)
    fb, cb, sh_b = rows_on(big)
    plain = max(x for x in (V.s5_bind_v3_operative_floor(fs, k, m, *sh_s),
                            V.s5_bind_v3_operative_floor(fb, k, m, *sh_b)) if x is not None)
    gd = [V.s5_bind_v3_operative_floor({**fs, **cs}, k, m, *sh_s, guided=True),
          V.s5_bind_v3_operative_floor({**fb, **cb}, k, m, *sh_b, guided=True)]
    guided = None if all(x is None for x in gd) else max(x for x in gd if x is not None)
    world, _r = TK.build_world(spec)
    agents, objs = list(world.agents[:k]), list(world.objects[:m])
    agree_s, n_s = V.s5_bind_v3_trace_is_answer(scored, k, m, agents, objs)
    agree_b, n_b = V.s5_bind_v3_trace_is_answer(big, k, m, agents, objs)
    return {"cell": spec.name, "L": L, "k": k, "m": m, "chance": 1.0 / (k - 1),
            "n_scored": n_scored, "n_disjoint": n_big, "protocol": "guided",
            "answer_floor": guided, "trace_floor": guided, "answer_floor_plain": plain,
            "trace_basis": V.s5_bind_v3_trace_floor_basis(k, m, *sh_s),
            "slot_is_gold_scored": f"{agree_s}/{n_s}",
            "slot_is_gold_disjoint": f"{agree_b}/{n_b}",
            "ckpt_rows_scored": {r: round(v, 4) for r, v in cs.items()},
            "ckpt_rows_disjoint": {r: round(v, 4) for r, v in cb.items()},
            "copy_per_slot": V.s5_bind_v3_ckpt_copy_per_slot(scored, k, m, agents, objs),
            "slot_moves": V.s5_bind_v3_slot_moves(scored, k, m, agents, objs),
            "ckpt_lag": {j: V.s5_bind_v3_ckpt_lag(scored, j)
                         for j in V.S5_BIND_V3_CKPT_LAG if j < L},
            "pad_reach": (None if sh_s[2] else V.s5_bind_v3_pad_reach(scored)),
            "trace_admitted": {r: round(v, 4) for r, v in
                               sorted({**fs, **cs}.items(), key=lambda x: -x[1])
                               if V.s5_bind_v3_guided_admits(r, k, m, *sh_s)}}


def trace_floor_table(cells=LOCAL_CELLS, lengths=None, n_scored=N_GUIDED):
    """``trace_floor`` over the cells and lengths the GUIDED read covers, keyed ``cell@L``."""
    from factworld.tokenizer import Tokenizer

    base = TK.CANONICAL[cells["composed"]]
    w, r = TK.build_world(base)
    tok = Tokenizer.build([w], r)
    grid = lengths if lengths is not None else guided_grid(matched_lengths(tok, cells))
    if lengths is None:
        grid = dict(grid)
        grid["composed"] = list(GUIDED_LENGTHS)
    out = {}
    for key in ("state", "bind", "composed"):
        for L in grid.get(key, ()):
            out[f"{key}@{L}"] = trace_floor(TK.CANONICAL[cells[key]], L, n_scored=n_scored)
            row = out[f"{key}@{L}"]
            gf = ("unfloorable (pad reach "
                  + (f"{row['pad_reach']:.4f})" if row.get("pad_reach") is not None else "—)")
                  ) if row["trace_floor"] is None else f"{row['trace_floor']:.4f}"
            print(f"  {key}@{L}: guided (both channels) {gf} [{row['trace_basis']}] | plain "
                  f"{row['answer_floor_plain']:.4f} "
                  f"({row['answer_floor_plain'] / row['chance']:.2f}x)  slot==gold "
                  f"{row['slot_is_gold_scored']} / {row['slot_is_gold_disjoint']}", flush=True)
    return out


# ---- the rule -------------------------------------------------------------------------------
def clears(acc, floor, n=N_EVAL, z_min=Z_CLEAR, margin=MARGIN):
    """CLEARS, exactly as registered: significant AND large enough to be a circuit.

    The additive margin. It is the ANSWER read's rule and stays that read's rule unchanged; it is
    undefined where a floor plus the margin exceeds 1, which is where ``clears_headroom`` is
    registered instead (see MARGIN_FRAC).
    """
    if acc is None or floor is None:
        return False, None
    se = math.sqrt(max(1e-12, floor * (1.0 - floor) / n))
    z = (acc - floor) / se
    return bool(z > z_min and (acc - floor) >= margin), z


def clears_headroom(acc, floor, n=N_EVAL, z_min=Z_CLEAR, frac=MARGIN_FRAC):
    """CLEARS on the PAD read: significant, and closing ``frac`` of what the floor leaves.

    ``(a - f) / (1 - f) >= frac`` with the same z conjunct. Registered for the per-slot pad read,
    where floors run from 0.18 to 0.93 and an additive margin is not a bar at the top of that
    range but an impossibility (MARGIN_FRAC above). At f = 0.2 it asks for a - f >= 0.15, which is
    the additive rule at the operating point the additive rule was set at.
    """
    if acc is None or floor is None:
        return False, None
    se = math.sqrt(max(1e-12, floor * (1.0 - floor) / n))
    z = (acc - floor) / se
    head = 1.0 - floor
    lift = (acc - floor) / head if head > 1e-12 else 0.0
    return bool(z > z_min and lift >= frac), z


def bar_for(floor, frac=MARGIN_FRAC):
    """The score ``clears_headroom`` requires at this floor — always in (floor, 1]."""
    return None if floor is None else floor + frac * (1.0 - floor)


def per_seed_clears(per_seed, floors, lengths, n=N_EVAL, rule=clears):
    """{seed: does THIS seed clear at EVERY registered length}.

    The per-seed map the conjunction is taken on. A seed with a missing length, or a length with
    no floor, is False for that seed: the conjunction may only be built out of measurements.
    """
    return {s: bool(lengths) and all(rule(per_seed[s].get(L), floors.get(L), n)[0]
                                     for L in lengths)
            for s in per_seed}


def forms(per_seed, floors, lengths, n=N_EVAL, seeds_clear=SEEDS_CLEAR, rule=clears):
    """FORMS: clears on >= seeds_clear seeds at EVERY registered length.

    ``per_seed`` is {seed: {L: accuracy}}; ``floors`` is {L: floor}. Returns the verdict, the
    per-length seed counts — so a cell that clears at 48 and not at 96 is visible as that rather
    than as a bare False — and the PER-SEED map, which is what a claim combining this gate with
    another has to be built on (``seeds_carrying``).

    A LENGTH WITH NO FLOOR COUNTS None, NOT 0. Where the protocol leaves the cell unfloorable
    there is nothing for a seed to clear, and a 0 there would report an unfloorable cell and a
    floored one with the same number — the same substitution ``evaluate_control`` refuses for a
    missing cell. FORMS is False either way, and the caller reads the None: a False whose counts
    are None is not a null.
    """
    counts = {}
    for L in lengths:
        counts[L] = (None if floors.get(L) is None else
                     sum(1 for s in per_seed
                         if rule(per_seed[s].get(L), floors.get(L), n)[0]))
    return (bool(lengths) and all(c is not None and c >= seeds_clear
                                  for c in counts.values()), counts,
            per_seed_clears(per_seed, floors, lengths, n, rule))


# THE LEVEL A PAD PROTOCOL'S SCRATCHPAD HAS TO REACH BEFORE ITS ANSWER MEANS ANYTHING, and it is
# a MEASURED reference rather than a chosen bar: on the bounded pad both components free-run their
# own pad PERFECTLY ON EVERY ITEM at every registered length on every seed, so a cell writing its
# pad correctly on this protocol writes it essentially perfectly. A composed cell two thirds of the
# way there is not a slightly worse tracker; it is a model whose scratchpad is wrong by the middle
# of the stream, and the answer it reads off that pad prices something the composition rule does
# not model.
#
# THE UNIT IS THE ITEM, because the answer is. A per-TOKEN bar is not the same gate on a long
# stream as on a short one: at L = 96 the pad is 192 tokens, so 0.99 per token admits a cell whose
# items are perfect 14% of the time, and the composed cell's own numbers separate by an order of
# magnitude under the two units (per slot 0.9836 -> items perfect 0.799 at L = 16, and
# 0.7992 -> 0.098 at L = 48). What the answer is generated from is one item's whole pad, so one
# wrong token in that item is a corrupted context whatever the other 191 do.
PAD_TRACKS_MIN = 0.99


def pad_tracks(per_seed, lengths, seeds_clear=SEEDS_CLEAR, level=PAD_TRACKS_MIN):
    """Whether the composed cell WRITES the pad the protocol scores it against.

    ``per_seed`` is {seed: {L: ITEMS-PERFECT fraction}} from the SAME free-running read the answer
    comes from — the model's own pad, fed back, not a teacher-forced one, and counted per ITEM
    rather than per token for the reason above. Shaped like ``forms`` so the three gates are read
    the same way: it holds iff at least ``seeds_clear`` seeds reach ``level`` at EVERY registered
    length, and it returns the per-length counts (a cell that tracks at 48 and not at 96 is
    visible as that) AND the per-seed map the conjunction is taken on.
    """
    counts = {L: sum(1 for s in per_seed
                     if (per_seed[s].get(L) is not None and per_seed[s][L] >= level))
              for L in lengths}
    per = {s: bool(lengths) and all(per_seed[s].get(L) is not None and per_seed[s][L] >= level
                                    for L in lengths)
           for s in per_seed}
    return bool(lengths) and all(c >= seeds_clear for c in counts.values()), counts, per


class ReadoutNotEvaluable(RuntimeError):
    """A composition claim was read off a floored answer with no GOLD-PAD answer column.

    Raised rather than defaulted for the reason ``ControlNotEvaluable`` is: a read whose readout
    was never measured and a read whose readout is dead produce the same floored answer, and
    substituting one for the other reports a model that cannot read its own scratchpad as a
    composition gap.
    """


def readout_alive(gold_per_seed, floors, lengths, n, seeds_clear=SEEDS_CLEAR):
    """Whether the composed cell can READ OUT at all, measured with the GOLD pad in its context.

    ``gold_per_seed`` is {seed: {L: match}} from a read whose pad tokens are the gold ones and
    whose answer is generated — the same items, decode and floors as the scored read, differing
    only in whose pad is in the context. It holds iff the gold-pad answer CLEARS the cell's floor
    on at least ``seeds_clear`` seeds at every length.

    IT IS THE OTHER HALF OF THE PAD GATE. ``pad_tracks`` says the model's own pad is good enough
    for its answer to mean something; this says an answer would be there to mean it. A seed whose
    pad is byte-perfect and whose gold-pad answer is still at floor has a dead readout, and its
    floored composed answer is a fact about the readout and not about the composition.

    IT IS PER SEED, and the aggregate alone is not the gate. A readout is a property of ONE model:
    with three seeds and ``seeds_clear`` 2, an aggregate count returns True while the seed that
    writes the pad is the dead one, which is the exact configuration this exists to catch. The
    third return value is the per-seed map, and ``seeds_carrying`` is what a claim is built on.
    """
    counts = {L: (None if floors.get(L) is None else
                  sum(1 for s in gold_per_seed
                      if clears(gold_per_seed[s].get(L), floors.get(L), n)[0]))
              for L in lengths}
    return (bool(lengths) and all(c is not None and c >= seeds_clear for c in counts.values()),
            counts, per_seed_clears(gold_per_seed, floors, lengths, n))


class SeedsNotConjoined(RuntimeError):
    """A composition verdict was asked for without the per-seed conjunction of its gates.

    Raised rather than defaulted for the reason ``ControlNotEvaluable`` is. Three gates counted
    independently over seeds pass on a run whose pad-writing seed and whose reading-out seed are
    DIFFERENT MODELS, and the claim that comes out is a claim about no model that was trained.
    """


def seeds_carrying(gates: dict, seeds_clear=SEEDS_CLEAR):
    """The seeds on which EVERY gate holds — the conjunction a composition claim is built on.

    ``gates`` is {gate name: {seed: bool}}, the per-seed maps ``forms``, ``pad_tracks`` and
    ``readout_alive`` return. A seed counts only where all of them hold FOR THAT SEED.

    THIS IS THE RULE, and the counted-separately version is not a weaker form of it but a
    different claim. Components forming on seeds 0 and 1, a pad written on seed 2 and a readout
    alive on seeds 0 and 1 satisfies every 2-of-3 count and describes no model: the cell whose pad
    is good is the cell whose readout is dead. Returns ``(ok, per_seed, n_seeds, {gate: seeds})``.
    """
    seeds = sorted({s for g in gates.values() for s in g})
    per = {s: all(bool(g.get(s)) for g in gates.values()) for s in seeds}
    n_ok = sum(per.values())
    return (n_ok >= seeds_clear, per, n_ok,
            {name: sorted(s for s in g if g[s]) for name, g in gates.items()})


class ControlNotEvaluable(RuntimeError):
    """A control was applied at an (arm, cell, length) the arm never evaluated.

    It is deliberately not a verdict. A control that was not measured says nothing about the
    model, and folding it into an abort — as a bare seed count of 0 does — reports a MISSING CELL
    as a model at floor. Raising stops the read at the arm whose grid is wrong.
    """


CONTROL_CELLS = ("state", "bind")     # the control is a DISJUNCTION over the components


def control_grid(read, grid, guided_lengths=GUIDED_LENGTHS, control_length=CONTROL_LENGTH,
                 work=None):
    """The (cell, length) pairs the positive control REQUIRES on THIS read.

    A read is only ever controlled on lengths it covers: the PLAIN read runs the whole grid and
    is controlled at the shortest trained length, the GUIDED read runs the composed cell's
    ``guided_lengths`` and each component at ITS WORK-MATCHED PARTNER of those — which is where
    the guided read evaluates that component at all, since the components' registered grid is the
    work pairing. ``grid`` is {cell: lengths this run evaluated}, so a pair the run's own grid
    does not contain is not required and is not silently treated as a failure either.
    """
    work = WORK_MATCHED if work is None else work
    if read == "guided":
        want = {c: tuple(work[L][c] for L in guided_lengths if work.get(L, {}).get(c))
                for c in CONTROL_CELLS}
    else:
        want = {c: (control_length,) for c in CONTROL_CELLS}
    return tuple((c, L) for c in CONTROL_CELLS for L in want[c]
                 if L in tuple(grid.get(c, ())))


def evaluate_control(read, acc, floors, grid, n):
    """Evaluate the positive control on the pairs ``control_grid`` requires, or RAISE.

    Args:
        acc: {cell: {seed: {L: accuracy}}} for this read.
        floors: {cell: {L: floor}}.
        grid: {cell: lengths this run evaluated}.
        n: the items behind each accuracy on this read.

    Returns:
        {"seeds": the most seeds any required pair clears, "cleared_on": that pair,
         "required": every required pair, "per_pair": {(cell, L): seeds}}.

    Raises:
        ControlNotEvaluable: no required pair has both a floor and an accuracy for every seed.
    """
    required = control_grid(read, grid)
    if not required:
        raise ControlNotEvaluable(
            f"{read}: no control cell is on this read's grid {dict(grid)}; the control is "
            f"declared over {CONTROL_CELLS} at "
            f"{ {c: registered_lengths(c) for c in CONTROL_CELLS} if read == 'guided' else (CONTROL_LENGTH,)}")
    per_pair, best = {}, None
    for cell, L in required:
        if floors.get(cell, {}).get(L) is None:
            continue
        if not acc.get(cell) or not all(L in acc[cell][s] for s in acc[cell]):
            continue
        seeds = sum(1 for s in acc[cell] if clears(acc[cell][s].get(L), floors[cell][L], n)[0])
        per_pair[f"{cell}@{L}"] = seeds
        if best is None or seeds > best[0]:
            best = (seeds, f"{cell}@{L}")
    if best is None:
        raise ControlNotEvaluable(
            f"{read}: the control requires one of {[f'{c}@{L}' for c, L in required]} and this "
            "arm evaluated none of them. A missing cell is not a model at floor.")
    return {"seeds": best[0], "cleared_on": best[1], "per_pair": per_pair,
            "required": [f"{c}@{L}" for c, L in required]}


def matched_required(matched, cells=("state", "bind"), lengths=LOCAL_LENGTHS):
    """The (cell, length) pairs the MATCHED-COST control requires, from ``matched_lengths``.

    Returns {cell: [lengths]}. A cell whose list is empty has NO matched-cost control anywhere on
    the grid — the sampler cannot build it (BIND_MATCHED_MAX) — and V1 is then unreachable for
    that cell by construction rather than by how the run came out.
    """
    out = {c: [] for c in cells}
    for L in lengths:
        for c in cells:
            ml = matched.get(L, {}).get(c, {}).get("L")
            if ml and ml not in out[c]:
                out[c].append(ml)
    return {c: sorted(v) for c, v in out.items()}


def guided_grid(matched, cells=LOCAL_CELLS, lengths=GUIDED_LENGTHS,
                matched_from=GUIDED_MATCHED_FROM, work=None):
    """The per-cell lengths the GUIDED read runs.

    The composed cell runs ``lengths``. Each component runs its WORK-MATCHED partner of
    ``matched_from`` — the length its own registered grid puts against that composed cell — plus
    its TOKEN-MATCHED length, without which the guided read cannot reach V1 at all: that control
    is a component at a LONGER length than any the read covers, so "beyond the step multiplier"
    would be unevaluable there however the cells came out, which is exactly what the first run
    hit. The guided decode is O(n L^2), so only the shortest composed length's pair is bought.
    """
    work = WORK_MATCHED if work is None else work
    out = {c: list(lengths) for c in cells}
    for c in ("state", "bind"):
        out[c] = []                       # a component is never read at the COMPOSED cell's L
        wl = work.get(matched_from, {}).get(c)
        if wl and wl not in out[c]:
            out[c].append(wl)
        ml = matched.get(matched_from, {}).get(c, {}).get("L")
        if ml and ml not in out[c]:
            out[c].append(ml)
    return {c: sorted(v) for c, v in out.items()}


def verdict(control, comp_forms, comp_counts, matched_forms, matched_measured,
            composed_floored=True, pad_reach=None, pad_tracked=None, pad_counts=None,
            readout=None, readout_counts=None, seed_gates=None):
    """The verdict table, applied mechanically. Raises rather than aborting on a missing control.

    Args:
        control: the dict ``evaluate_control`` returns. A bare seed count is refused: it reports
            an unevaluated cell and a floored one with the same number, which is the defect this
            rule exists to remove.
        comp_forms: {"state": bool, "bind": bool, "composed": bool} at each cell's OWN registered
            lengths — the composed grid for the composed cell, the WORK-MATCHED partners of it
            for the components (``registered_lengths``). Reading a component at the composed
            cell's own L is what made V2 unreadable: at p_swap = 1/3 that compares 3x the state
            work and 1.5x the retrieval work.
        comp_counts: {cell: {L: seeds clearing}} for the report.
        matched_forms: {"state": bool|None, "bind": bool|None} at the TOKEN-matched lengths;
            None where no matched-cost control was measured.
        matched_measured: {cell: bool} — whether a matched-cost control was measured at all.
        composed_floored: whether the composed cell HAS a floor at every registered length on
            this read's protocol. False under a scratchpad protocol, where the format hands out
            the live slots the composed cell's floor argument is made of. It gates a verdict of
            its own rather than falling through: with no floor the cell cannot clear, and letting
            a cannot-clear reach V1 would read "the composition is harder than its components"
            off a number that was never measured.
        pad_reach: what the excluded both-maps class scores on the exact items, printed with the
            V0 reason so the retraction leaves a number. Not a floor.
        pad_tracked: whether the composed cell writes its own pad at component level
            (``pad_tracks``, on ITEMS PERFECT). None where the read has no pad at all — the PLAIN
            and DENSE protocols — and the gate then does not apply, which is why it defaults to
            None rather than to True. Under a BOUNDED-PAD read it is required: the answer is
            generated from the model's own pad, so a pad that is wrong by mid-stream makes the
            composed cell's floored answer uninterpretable as composition.
        pad_counts: the per-length seed counts ``pad_tracks`` returns, printed with the V6 reason.
        readout: whether the composed cell answers at all when it is HANDED the gold pad
            (``readout_alive``). It is REQUIRED on every path that INTERPRETS the composed cell's
            floored answer — V3 and both V1s — because a floored answer from a dead readout and a
            floored answer from a failed composition are the same number. The two verdicts that
            refuse to interpret it, V0 and V6, are returned before this is read. Absent on an
            interpreting path, this raises rather than defaulting.
        readout_counts: the per-length seed counts ``readout_alive`` returns, printed with V7.
        seed_gates: {gate name: {seed: bool}} — the PER-SEED maps of the three gates, which is
            what the claim is conjoined on (``seeds_carrying``). REQUIRED on every path that
            interprets the composed cell under a pad protocol, for the same reason the gold-pad
            column is: three gates counted independently over seeds are satisfied by a run whose
            pad-writing seeds and reading-out seeds are disjoint, and the composition claim that
            comes out is a claim about no trained model. Absent there, this raises.

    Raises:
        ControlNotEvaluable: ``control`` is not an evaluated control.
        ReadoutNotEvaluable: the run reaches a composition verdict under a pad protocol with no
            gold-pad answer column.
        SeedsNotConjoined: it reaches one with no per-seed conjunction of the gates.
    """
    if not isinstance(control, dict) or "seeds" not in control:
        raise ControlNotEvaluable(
            "verdict() takes the control evaluate_control() returns, not a seed count: a count "
            "cannot distinguish a cell at floor from a cell the arm never ran.")
    if control["seeds"] < SEEDS_CLEAR:
        return "V5_HARNESS_NULL", (
            f"no component clears on this read's control grid {control['per_pair']} "
            f"({control['seeds']}/{SEEDS_CLEAR} seeds at best, on {control['cleared_on']}). "
            "Nothing downstream is interpretable; the next move is the training recipe, not the "
            "instrument.")
    if not comp_forms["state"] or not comp_forms["bind"]:
        bad = [c for c in ("state", "bind") if not comp_forms[c]]
        return "V4_COMPONENT_UNREADABLE", (
            f"component(s) {bad} do not form at their own registered lengths "
            f"{ {c: comp_counts[c] for c in bad} }, while the other one does. A composed "
            "failure would be explained by the component that failed, so no composition claim "
            "is available — and the dissociation between the components is the result.")
    if not composed_floored:
        pr = "not measured" if pad_reach is None else f"{pad_reach:.3f}"
        return "V0_COMPOSED_UNFLOORABLE", (
            "both components form at their own registered lengths, and the composed cell has NO "
            "FLOOR on this read's protocol: its floor argument is the one-structure bound "
            "W <= max(k, m) + 1, and this protocol's format requires the whole of P then B at "
            "every event, which hands those k + m slots to every policy including the task's own "
            "algorithm. Neither 'clears' nor 'does not clear' is available for the composed cell "
            f"here, so no composition verdict is. What the excluded both-maps class reaches on "
            f"the exact scored items is {pr}, and it is a lower bound on that class's max rather "
            "than a floor. The comparison this read does support is WITHIN-RUN and UNPAIRED — "
            "the composed cell against its work-matched component on the same seeds and the same "
            "item count, drawn from different specs and so never item by item.")
    if comp_forms["composed"]:
        return "V2_NO_GAP_HERE", (
            "the composed cell forms, and so does each component at the length carrying the same "
            "amount of that component's own work. Composition is not a separate difficulty at "
            f"k=6 / L<={max(LOCAL_LENGTHS)} in this regime; the lengths or k must move before the "
            "cell is worth buying on the frontier.")
    if pad_tracked is False:
        return "V6_TRACKING_GAP", (
            "both components form and the composed cell is at floor, but the composed cell does "
            f"not write its own pad at the level the components reach: items perfect >= "
            f"{PAD_TRACKS_MIN} on {pad_counts if pad_counts is not None else 'no'} seeds per "
            "length, against components whose pad is perfect on every item everywhere. This "
            "read scores the answer the model gives FROM ITS OWN PAD, so a floored answer here is "
            "equally consistent with the composition being hard and with the model being unable "
            "to hold the state the pad gave it room for, and only the second is measured. No "
            "composition claim is available until the composed pad reaches component level with "
            "the answer still at floor.")
    if pad_tracked is not None and readout is None:
        raise ReadoutNotEvaluable(
            "this run reaches a composition verdict under a pad protocol and carries no GOLD-PAD "
            "answer column. Handed a perfect pad the composed answer either clears or does not, "
            "and a run that never measured it cannot tell a failed composition from a model that "
            "cannot read its own scratchpad: both are a floored answer. Measure the composed cell "
            "with the gold pad in context on the same items and decode, and pass "
            "readout_alive(...).")
    if readout is False:
        return "V7_READOUT_DEAD", (
            "both components form, the composed cell writes its pad, and the composed cell is at "
            "floor — but handed the GOLD pad on the same items and decode it is still at floor "
            f"({readout_counts if readout_counts is not None else 'no'} seeds clearing per "
            "length). The answer this read scores is downstream of a readout that does not work, "
            "so its floored value is a fact about the readout and not about the composition.")
    if pad_tracked is not None:
        if seed_gates is None:
            raise SeedsNotConjoined(
                "this run reaches a composition verdict under a pad protocol and carries no "
                "PER-SEED conjunction of its gates. forms, pad_tracks and readout_alive are three "
                "counts over seeds, and a run whose pad-writing seeds and reading-out seeds are "
                "disjoint satisfies all three while no single model satisfies any two. Pass "
                "seed_gates={'forms': ..., 'pad': ..., 'readout': ...} from the maps those "
                "functions return.")
        conj_ok, per_seed, n_seeds, by_gate = seeds_carrying(seed_gates)
        if not conj_ok:
            return "V8_NO_SEED_CARRIES_THE_CLAIM", (
                f"every gate passes on its own count, and no {SEEDS_CLEAR} seeds pass all of them "
                f"together: {by_gate} gives {n_seeds} seed(s) carrying the whole claim "
                f"({per_seed}). The gates hold on DIFFERENT MODELS, so the composition reading "
                "they would license is a reading of no model that was trained. It is not a null: "
                "each cell's number is real and reported, and the next move is a seed on which "
                "the components form, the pad is written and the readout is alive at once.")
    unmatched = [c for c, ok in matched_forms.items() if ok is False]
    if unmatched:
        return "V3_GAP_IS_THE_COST", (
            f"the composed cell is at floor and component(s) {unmatched} are also at floor at "
            "their matched-cost lengths. The failure is accounted for by cost, not composition.")
    missing = [c for c in ("state", "bind") if not matched_measured.get(c)]
    if missing:
        return "V1_UNCONTROLLED", (
            "both components form and the composed cell does not — the V1 pattern — but the "
            f"matched-cost control is absent for {missing}, so 'beyond the step multiplier' is "
            "not established. The cells separate; the cause does not.")
    return "V1_COMPOSITION_GAP", (
        "both components form, including at the matched-cost lengths, and the composed cell "
        "clears nowhere. The composition is harder than its components beyond the step "
        "multiplier.")


# ---- the frontier scout ----------------------------------------------------------------------
def scout_plan(models=SCOUT_MODELS, n=SCOUT_N):
    """The priced scout, and the decision rule for whether the roster run is worth buying.

    Cells: the composed k=12 cell at both scout lengths — it carries the length axis, which is
    where separation would come from — and each component at the WORK-MATCHED partner of the
    deepest of them (state@85, bind@171), because a component that holds the work the composed
    cell makes it do at 256 holds the smaller amount it makes it do at 128, and the component
    reading is a GATE rather than a ranking. Reasoning ON at the protocol budget throughout.

    n is set at SCOUT_N because the scout has to answer a SEPARATION question, not rank the
    roster: the smallest gap it must resolve is SCOUT_SEPARATION = 0.20, which at n = 40 is
    z = 2.1 against a two-sided binomial at p = 0.5 — enough to see a spread that large, and
    deliberately not enough to order two models 0.05 apart, which is what the roster run is for.
    """
    from factworld.benchmark import MODELS, cell_dollar_cap, cost_estimate

    cells = scout_cells(models=models, n=n)
    rows, total, worst = [], 0.0, 0.0
    for slug in models:
        if slug not in MODELS:
            rows.append({"model": slug, "error": "not in MODELS"})
            continue
        est = cost_estimate(slug, cells, assumed_output_tokens=4000)
        # WORST CASE, which the 4000-token estimate does not price: every call spending its whole
        # registered budget. It is what the per-cell dollar caps actually bound the run to, and it
        # is the number the spend is approved against.
        caps = {f"{c['cell']}@{c['length']}":
                cell_dollar_cap(slug, c["n"], c["settings"]["max_new_tokens"]) for c in cells}
        full = sum(c["n"] * c["settings"]["max_new_tokens"] for c in cells)
        w = (est["prompt_tokens"] / 1e6 * MODELS[slug]["prompt_price_per_M"]
             + full / 1e6 * MODELS[slug]["completion_price_per_M"])
        total += est["cost_usd"]
        worst += w
        rows.append({"model": slug, **est, "worst_case_usd": round(w, 2),
                     "cell_dollar_caps": caps})
    # what the scout is gating: the same three cells over their OWN registered grids — the
    # composed cell's lengths and each component's work-matched partners of them — and the whole
    # roster, at the n a ranking needs. Both numbers belong in the decision, because the scout is
    # only worth its own price against what it stops.
    roster_cells = [{"task": FRONTIER_CELLS[c], "length": L, "n": ROSTER_N,
                     "settings": {"effort": SCOUT_EFFORT,
                                  "max_new_tokens": SCOUT_BUDGETS.get((c, L),
                                                                      SCOUT_MAX_NEW_TOKENS)}}
                    for c in ("state", "bind", "composed")
                    for L in TK.CANONICAL[FRONTIER_CELLS[c]].eval_lengths]
    roster, roster_total = [], 0.0
    for slug in MODELS:
        est = cost_estimate(slug, roster_cells, assumed_output_tokens=4000)
        roster_total += est["cost_usd"]
        roster.append({"model": slug, "cost_usd": est["cost_usd"]})
    return {"cells": cells, "n_cells": len(cells), "per_model": rows,
            "total_usd": round(total, 2), "worst_case_usd": round(worst, 2),
            "budgets": {f"{c['cell']}@{c['length']}": c["settings"]["max_new_tokens"]
                        for c in cells},
            "informed_chance": round(scout_informed_chance(), 4),
            "roster_run_if_bought": {
                "models": len(MODELS), "cells_per_model": len(roster_cells), "n": ROSTER_N,
                "per_model": sorted(roster, key=lambda r: -r["cost_usd"]),
                "total_usd": round(roster_total, 2)},
            # THE ORDER IS PART OF THE RULE — see scout_verdict, which applies it.
            "stop_rules": [
                f"1. VOID (truncation), FIRST: any cell with more than "
                f"{SCOUT_TRUNCATION_MAX:.0%} finish=length or empty answers is VOID — re-run "
                "that cell at a raised budget, and it may not enter any stop or buy decision. "
                "Without this, STOP(floor) fires on truncation and manufactures the published "
                "s5 L64 cliff a second time.",
                f"2. STOP (ceiling) if the top scout model's COMPOSED match >= {SCOUT_CEILING} "
                f"at L={SCOUT_STOP_CEILING_LENGTH}. A ceiling cannot rank the roster, so the "
                "roster run buys nothing; redesign (raise k or L) before spending.",
                f"3. STOP (floor) if all three models are within {SCOUT_FLOOR_SE:.0f} se of "
                f"INFORMED CHANCE 1/(k-1) = {scout_informed_chance():.4f} on "
                f"composed@{SCOUT_STOP_FLOOR_LENGTH}. Stated against informed chance and NOT "
                "against an operative floor: the frontier is a scratchpad regime, so the "
                "composed cell is unfloorable there for exactly the reason it is unfloorable on "
                "the guided read. Redesign, do not re-budget.",
                f"4. STOP (component) if either COMPONENT is below {SCOUT_COMPONENT_MIN} for the "
                "top scout model. The composed cell is then state-limited or retrieval-limited "
                "and unreadable for the same reason it is unreadable locally when a component "
                "does not form.",
            ],
            "buy_rule": (
                f"BUY the roster run iff the composed cell's spread across the three scout "
                f"models is >= {SCOUT_SEPARATION} match at either composed length AND both "
                f"components are >= {SCOUT_COMPONENT_MIN} for every scout model — the composed "
                f"cell discriminates inside the roster's range while the components do not. It "
                f"is a SPREAD rule and needs no floor, which is why it survives the composed "
                f"cell's floor retraction; floor-clearing language is banned for the composed "
                f"cell anywhere in the scout or roster report "
                f"(SCOUT_COMPOSED_FLOOR_LANGUAGE, asserted on the report text)."),
            "budget_note": (
                f"Per-cell budgets {dict((f'{c}@{L}', b) for (c, L), b in SCOUT_BUDGETS.items())} "
                "and 4000 assumed completion tokens per call for the estimate; worst_case_usd "
                "prices every call at its full budget, which is what the per-cell dollar caps "
                "bound the run to. The old 8192 was a VALIDITY defect and not a price one: "
                "cost_estimate never reads max_new_tokens, the composed prompt alone is 2,984 "
                "tokens at L=128 and 5,583 at L=256, and a truncated reply is scored as wrong — "
                "which is how the published s5 L64 cliff and chain floor were manufactured."),
            }


# ---- registration ---------------------------------------------------------------------------
def trace_read_block(matched, with_floors=True):
    """The TRACE READ's declaration for the pre-registration record: its arm restriction, stated
    as data rather than described in prose, and the GUIDED protocol's floors.

    THOSE FLOORS ARE BOTH CHANNELS'. The trace read and this protocol's answer read are scored
    against one number per cell, because what a floor is priced against is the protocol: the
    components' are the plain protocol's floors unchanged, and the composed cell's is None, since
    the format hands out the live slots that cell's floor argument is made of. ``pad_reach``
    carries what the excluded both-maps class reaches there, so the record holds a number where
    the floor was retracted.
    """
    return {
        "read": TRACE_READ, "arms": list(TRACE_READ_ARMS),
        "requires": TRACE_READ_REQUIRES, "frontier_reads": list(FRONTIER_READS),
        "rule": "the model's own FINAL CHECKPOINT's value for the queried slot; a frontier cell "
                "is scored on the answer and assert_trace_read() raises on any other arm",
        "floors_are": "the GUIDED protocol's, for BOTH of its channels: answer_floor == "
                      "trace_floor at every cell, None on the composed cell, with pad_reach "
                      "beside it and answer_floor_plain as the plain protocol's reference",
        "floors": (trace_floor_table(
            lengths={**guided_grid(matched), "composed": list(GUIDED_LENGTHS)})
            if with_floors else {}),
    }


# ---- the pad-write read ----------------------------------------------------------------------
# WHAT IS SCORED ON IT, registered here because the two reads it can be taken under measure
# different things and only one of them is a score.
#
# THE SCORED QUANTITY IS THE TEACHER-FORCED PER-EVENT READ of the two-hop token: the gold pad is
# in context and the model emits the next pad block, so each event is scored ONCE, against the
# true history. The justification is what teacher-forcing REMOVES, not that it reads higher.
# A free-running pad is fed back, so a per-event error rate e shows up in the pooled number as
# 1 - (1 - e)^(events since the error mattered): the free-running score is the per-event quantity
# COMPOUNDED over the stream, and it falls with L on a model whose per-event behaviour is flat
# (0.902/0.900/0.897/0.895/0.891 teacher-forced at L=16..96 against 0.773/0.627/0.525/0.466/0.377
# free-running, same checkpoints). Against a floor whose class is scored the same way, a
# free-running comparison is therefore a comparison of two compounding rates and answers "can this
# model hold its own pad for 2L tokens", which is a state-capacity question. The teacher-forced
# comparison answers "given the state, does this model perform the two-hop write", which is the
# composition question the cell was built for.
#
# WHAT IT THEREFORE CANNOT CLAIM, and this is the whole of it: nothing about the model's behaviour
# on its own writes. A cell that clears here and floors free-running is a model that computes the
# composed update and cannot survive its own errors — which is a real and reportable failure, and
# it is NOT this measurement's. No end-to-end claim, no answer claim, and no claim at any length
# the free-running read was not also taken at. The free-running read stays measured and printed
# beside it as the TRACKING DIAGNOSTIC it is.
#
# THE FLOOR IT IS SCORED AGAINST is ``validity.s5_bind_v3_pad_two_hop_floor``: the one-hop
# sub-class — the component cells' registered depth conjunct applied per emitted token — on the
# CROSS partition of ``swap_p0``, measured on the exact scored items and on a disjoint pool with
# the larger operative, under the SAME read (a teacher-forced score against a teacher-forced
# class, a free-running score against a free-running one).
# WHICH READ CARRIES A CLEAR, AND WHICH INTERPRETS A FLOOR — registered 2026-08-09, before any
# per-partition solver number under the ``before2`` format was read.
#
# The two reads have different floors and the difference is structural, not a preference.
# FREE-RUNNING the model holds only what it emitted, so the gold pad is not an address space and
# the depth-<=1 closure has only the event lines and the header to read: measured held-out at
# k = 6 under ``before2`` it is 0.164 / 0.156 / 0.162 at L = 16 / 32 / 48 against 1/k = 0.1667.
# THE FREE-RUNNING FLOOR IS INFORMED CHANCE. No admitted policy is above it, so a cleared
# free-running reading of the two-hop token cannot be anything but the two-hop write, and that is
# the read a CLEAR is registered on.
# TEACHER-FORCED the gold pad IS the context, the closure reads 2.1x chance off it, and the
# question the read answers is the narrower "given the true state, is the write performed". It is
# what a FLOOR is interpreted with: a cell that clears teacher-forced and floors free-running is a
# model that computes the composed update and cannot survive its own errors, and the two readings
# are not interchangeable.
#
# THE CONJUNCTION IS NOT REQUIRED, and the reason is arithmetic rather than generosity: the two
# floors differ by 2x, so a model at a fixed accuracy can clear the fractional margin against
# chance and miss it against 0.35 — an outcome produced by the floors and not by the model.
# Both are reported, each against its OWN held-out floor, and the weaker one is never written up
# as the stronger.
PAD_WRITE_SCORED_READ = "free_run"
PAD_WRITE_FLOOR_READ = "teacher_forced"          # the read a FLOORED cell is interpreted with
PAD_WRITE_DIAGNOSTIC_READ = "teacher_forced"
PAD_WRITE_TOKEN = f"{V.S5_BIND_V3_TWO_HOP_CELL}|{V.S5_BIND_V3_TWO_HOP_SOURCE}"


def pad_write_read_block():
    """The pad-write read's declaration for the pre-registration record, as data."""
    return {
        "scored_read": PAD_WRITE_SCORED_READ,
        "diagnostic_read": PAD_WRITE_DIAGNOSTIC_READ,
        "scored_token": PAD_WRITE_TOKEN,
        "floor": "validity.s5_bind_v3_pad_two_hop_floor: the one-hop sub-class "
                 f"(depth <= {V.S5_BIND_V3_MAX_DEPTH}, the component cells' conjunct, applied per "
                 "emitted token) on the cross partition of swap_p0",
        "chance": "1/k, a per-slot pad read's own — every pad token is an agent name",
        "clears_rule": f"clears_headroom: z > {Z_CLEAR} and (a - f)/(1 - f) >= {MARGIN_FRAC:g}",
        "floor_read": PAD_WRITE_FLOOR_READ,
        "floor_selection": "held out: the closure's member is chosen on a disjoint pool and "
                           "scored on the read's own items, because a max over the family is "
                           "selection-inflated at a few hundred items",
        "cannot_claim": "a teacher-forced clear says nothing about the model on its own writes; "
                        "it is the narrower 'given the true state, is the write performed' and "
                        "is reported as that",
    }


def register(out_prefix, axis=MATCHED_AXIS, with_floors=True):
    """Write the pre-registration record: cells, both pairings, costs, floors and every
    threshold, before any solver number exists.

    Both pairings are re-measured here and checked against the registered constants; a
    disagreement RAISES rather than being written, because a sampler change that moved the
    work-matched partner would silently move what the composition comparison compares.
    """
    from factworld.tokenizer import Tokenizer

    base = TK.CANONICAL[LOCAL_CELLS["composed"]]
    w, r = TK.build_world(base)
    tok = Tokenizer.build([w], r)
    paired = pairings(tok)
    for name, want in (("work", WORK_MATCHED), ("tokens", TOKEN_MATCHED)):
        got = {L: {c: paired[name][L][c]["L"] for c in ("state", "bind")} for L in LOCAL_LENGTHS}
        if got != {L: want[L] for L in LOCAL_LENGTHS}:
            raise ValueError(f"{name} pairing drifted from the registered constants: "
                             f"measured {got}, registered {want}")
    ml = paired[MATCHED_PAIRING]
    grid = {"composed": [CONTROL_LENGTH, *LOCAL_LENGTHS]}
    for c in ("state", "bind"):
        grid[c] = sorted({CONTROL_LENGTH, *registered_lengths(c), *PROFILE_LENGTHS[c],
                          *matched_lengths_for(c)})
    costs = {}
    for key, nm in LOCAL_CELLS.items():
        spec = TK.CANONICAL[nm]
        for L in grid[key]:
            s, t = cell_cost(spec, L, tok)
            costs[f"{key}@{L}"] = {"charged_steps": s, "prompt_tokens": t}
    mult = step_multipliers(tok, paired)
    floors = {}
    if with_floors:
        wanted = {(c, L) for c in grid for L in grid[c]}
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
        "registered_lengths": {c: list(registered_lengths(c)) for c in LOCAL_CELLS},
        "profile_lengths": {c: list(v) for c, v in PROFILE_LENGTHS.items()},
        "thresholds": {"n_eval": N_EVAL, "n_guided": N_GUIDED, "z_clear": Z_CLEAR,
                       "margin": MARGIN, "margin_frac": MARGIN_FRAC,
                       "seeds_clear": SEEDS_CLEAR, "n_fit": N_FIT,
                       "n_fit_blocks": N_FIT_BLOCKS, "n_score": N_SCORE},
        "pad_write_read": pad_write_read_block(),
        "costs": costs,
        # BOTH PAIRINGS, side by side, because they answer different questions: the work pairing
        # is what each component's FORMS verdict is read at, the token pairing is the
        # matched-COST control. The multiplier is different under each and both are printed.
        "pairings": paired, "registered_pairing": REGISTERED_PAIRING,
        "matched_pairing": MATCHED_PAIRING, "step_multipliers": mult,
        "matched_lengths": ml, "matched_axis": axis,
        "guided_grid": guided_grid(ml),
        # THE TWO CONTROLS, declared as the (read, cell, length) pairs they require. A control
        # applied where its arm has no cell RAISES; it never abstains and never aborts.
        "controls": {
            "positive": {"cells": CONTROL_CELLS, "rule": "some component clears somewhere on "
                                                         "the grid the read covers",
                         "required": {r: [f"{c}@{L}" for c, L in control_grid(r, grid)]
                                      for r in ("plain", "guided")}},
            "matched_cost": {"axis": axis, "pairing": MATCHED_PAIRING,
                             "required": matched_required(ml),
                             "rule": "V1 is unavailable where a matched-cost control was never "
                                     "measured; the verdict is V1_UNCONTROLLED"},
        },
        "eval_grid": grid,
        "floors": floors,
        "trace_read": trace_read_block(ml, with_floors=with_floors),
        "verdicts": ["V5_HARNESS_NULL", "V4_COMPONENT_UNREADABLE", "V0_COMPOSED_UNFLOORABLE",
                     "V6_TRACKING_GAP", "V7_READOUT_DEAD", "V8_NO_SEED_CARRIES_THE_CLAIM",
                     "V3_GAP_IS_THE_COST", "V1_UNCONTROLLED", "V1_COMPOSITION_GAP",
                     "V2_NO_GAP_HERE"],
        "scout": scout_plan(),
    }
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {out_prefix}.json")
    return rec


def add_trace_read(record_path):
    """Write the ``trace_read`` declaration into an EXISTING pre-registration record.

    The record was written before the trace read existed, and re-running ``--register`` to add one
    block would redraw every other block with it. This adds exactly that key, stamps it with its
    own ``written_utc`` so the record says when each part of it was declared, and leaves the rest
    byte-identical. It declares a reading rule and carries no result.
    """
    from factworld.tokenizer import Tokenizer

    rec = json.load(open(record_path))
    base = TK.CANONICAL[LOCAL_CELLS["composed"]]
    w, r = TK.build_world(base)
    ml = pairings(Tokenizer.build([w], r))[MATCHED_PAIRING]
    rec["trace_read"] = {
        **trace_read_block(ml),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(record_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote trace_read into {record_path}")
    return rec["trace_read"]


def read_results(path, floors_path=None):
    """Re-apply the rule to a runner's results JSON — e.g. after a floor is re-measured.

    The verdict is a function of (accuracies, floors, thresholds) and nothing else, so it can
    always be recomputed; a verdict that changed because a floor moved should be visible as
    that, not buried in a stale results file.

    THE GUIDED FLOORS ARE PASSED TOO, and that is not optional: each read is judged under its own
    protocol, and omitting them here read the guided score against the plain protocol's floor —
    which is how the retracted V2_NO_GAP_HERE survived in this entry point after the runner's own
    had dropped it.
    """
    import experiment_s5bind_v3_three_cell_local_20260731 as E

    res = json.load(open(path))
    floors = json.load(open(floors_path))["floors"] if floors_path else res["floors"]
    cfg = res["cfg"]
    grid = {k: [int(x) for x in v] for k, v in cfg["grid"].items()}
    return E.apply_rule(res["runs"], floors, grid, cfg["eval_n"], cfg.get("guided_n", N_GUIDED),
                        res.get("guided_floors"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--register", action="store_true", help="write the pre-registration record")
    ap.add_argument("--read", default=None, help="re-apply the rule to a runner results JSON")
    ap.add_argument("--read_floors", default=None, help="floors JSON to re-read --read against")
    ap.add_argument("--no_floors", action="store_true", help="skip the floor pass (fast)")
    ap.add_argument("--axis", default=MATCHED_AXIS, choices=["tokens", "steps"])
    ap.add_argument("--scout", action="store_true", help="print the priced frontier scout")
    ap.add_argument("--add_trace_read", default=None,
                    help="write the trace_read declaration into an existing pre-registration "
                         "record (JSON path), leaving every other block untouched")
    ap.add_argument("--trace_floors", default=None,
                    help="write the TRACE read's floor table (JSON path) and exit; the guided "
                         "grid only, since the trace read is defined under no other protocol")
    ap.add_argument("--out_prefix",
                    default="results/s5bind_v3_three_cell_preregistration_20260731")
    a = ap.parse_args()
    if a.scout:
        p = scout_plan()
        print(json.dumps(p, indent=2))
    if a.add_trace_read:
        add_trace_read(a.add_trace_read)
    if a.trace_floors:
        os.makedirs(os.path.dirname(a.trace_floors) or ".", exist_ok=True)
        tbl = trace_floor_table()
        with open(a.trace_floors, "w") as f:
            json.dump({"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "protocol": os.path.basename(__file__), "read": TRACE_READ,
                       "arms": list(TRACE_READ_ARMS), "requires": TRACE_READ_REQUIRES,
                       "frontier_reads": list(FRONTIER_READS),
                       "n_scored": N_GUIDED, "n_disjoint": N_SCORE, "floors": tbl}, f, indent=2)
        print(f"wrote {a.trace_floors}")
    if a.read:
        for arch, reads in sorted(read_results(a.read, a.read_floors).items()):
            for read, v in sorted(reads.items()):
                print(f"{arch} / {read}: {v['verdict']} — {v['why']}")
                print(f"    seeds clearing {v['seed_counts']}; positive control "
                      f"{v.get('control', {}).get('per_pair')}; matched "
                      f"{v['matched_forms']} (measured {v.get('matched_measured')})")
    if a.register:
        os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
        register(a.out_prefix, axis=a.axis, with_floors=not a.no_floors)


if __name__ == "__main__":
    main()
