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
    visible as that. Run ``--scout`` to price the frontier scout, the roster run it gates, and
    the stop rules.

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
                          ``last_swap_ref`` and the uniform rows set the number

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

    Both reads score the SAME items against the SAME floors. The scratchpad caveat is stated
    where it bites: against the GUIDED read the W axis of the floor profile has no force, so
    that read is against the ADMITTED END of the profile, which at k=6 is 1.00-1.05x informed
    chance on the components and 1.02-1.05x on the composed cell — i.e. the two reads happen to
    be read against nearly the same numbers here, and the difference is in what clearing means.

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

    WHAT IT IS FLOORED BY, and the two cell kinds differ (factworld.validity, "THE TRACE READ"):
      * the final checkpoint's queried slot IS the gold answer — 4000/4000 on the disjoint pool
        and 128/128 on the scored items at every registered cell — so a floor row's trace score
        is its answer score and the numbers transfer;
      * the COMPONENT cells are floored, at the answer floor: their rule is depth <= 1 and cost
        under the cell's own algorithm's minimum, and a scratchpad buys neither;
      * the COMPOSED cell is NOT floored. Its registered class is the one-structure bound plus a
        step bound its own algorithm satisfies, and the guided protocol REQUIRES the whole of P
        and B to be written out at every event — so the k + m live slots that bound prices are
        handed to every policy. ``s5_bind_v3_trace_operative_floor`` returns None there.
        ``s5_bind_v3_trace_pad_floor`` measures how far the unfloorable class reaches: 0.719 on
        the 128 scored composed@48 items and 0.734 on the disjoint pool, against an answer floor
        of 0.234 and 0.200.
    So a composed-cell trace score is a WITHIN-RUN COMPARISON — same seeds, same items, matched
    depth and matched cost — and never a cleared floor. The DOWNWARD separation it carries does
    not need one; the other direction is not available at any registered length.

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

    FORMS. A cell forms for an arch iff it CLEARS on at least SEEDS_CLEAR (= 2) of the seeds at
    every registered length. Seeds are counted, never averaged: this family is bimodal at the
    emergence threshold and a mean over one converged and two floored seeds is a number no seed
    produced. Per-seed values are reported in every table.

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
    V4 COMPONENT UNREADABLE  a component does not FORM at its own registered lengths. The
                             composed cell cannot be read against it, because a composed failure
                             is then explained by the component that failed. Next move is that
                             component's budget or curriculum, not the composition.
    V3 GAP IS THE COST       the composed cell is at floor at L, and a component is ALSO at
                             floor at its matched-cost length. The composed cell's failure is
                             accounted for by how much longer it is, and no composition claim is
                             available from this run.
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
def cell_floor(spec, L, n_eval=N_EVAL, n_fit=N_FIT, n_score=N_SCORE, n_blocks=N_FIT_BLOCKS):
    """The operative floor at (cell, length), and everything needed to audit it.

    Measured on ``n_score`` items drawn from the SAME deterministic test stream as the items a
    solver is scored on and DISJOINT from them, because the max over admitted rows carries an
    upward selection bias at small n (the published 1.30x on the state component was a high draw
    at n=500 and is 0.98x at n=4000) and because the fitted ranker has to be scored out of
    sample. The same rows on the exact scored items are reported beside it as the house-rule
    check; where the two differ the larger is the number a score must clear.

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
    ns, ng = V.s5_bind_v3_shape(big)
    keep = tuple(r for r in V.s5_bind_v3_family_rows(k, m, ns, ng, named, query)
                 if V.s5_bind_v3_admits(r, k, m, ns, ng, named, query))
    fl = dict(V.s5_bind_v3_floors(big, k, m))
    fl.update(V.s5_bind_v3_family_floors(big, k, m, named, query, rows=keep))
    op = V.s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, query)
    sb = V.s5_bind_v3_surface_bound(fit, k, held_out=big, blocks=max(1, n_blocks))
    price = V.s5_bind_v3_surface_price(k, m, ns, ng, named, query,
                                       None if sb is None else sb["weights"])
    if price["admitted"] and sb is not None and (op is None or sb["held_out"] > op):
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
            "surface_n_fit": None if sb is None else sb["n_fit"],
            "surface_block_spread": None if sb is None else sb["block_spread"],
            "surface_blocks": None if sb is None else sb["blocks"],
            "surface_price": price, "surface_in_floor": bool(price["admitted"]),
            "basis": V.s5_bind_v3_floor_basis(k, m, ns, ng, named, query),
            "admitted_rows": {r: round(v, 4) for r, v in
                              sorted(fl.items(), key=lambda x: -x[1])
                              if V.s5_bind_v3_admits(r, k, m, ns, ng, named, query)},
            "charged_steps": s, "n_swap": ns, "n_give": ng}


def trace_floor(spec, L, n_scored=N_GUIDED, n_big=N_SCORE, arm="local"):
    """The TRACE read's floor at (cell, length), and the answer floor on the same items.

    Same pool discipline as ``cell_floor`` — rows measured on a DISJOINT pool because the max
    over rows carries an upward selection bias at small n, and on the exact scored items because
    a floor must be recomputed from the items it is read against, with the larger of the two
    operative. ``n_scored`` defaults to the GUIDED read's own sample, since that is the only
    protocol the trace read is defined under.

    ``trace_floor`` is None on the COMPOSED cell and that is the cell's answer, not a missing
    measurement: the guided protocol hands out the k + m live slots the one-structure bound
    prices, and what is left of the class contains the task. ``pad_reach`` measures how far the
    unfloorable class gets (the best both-maps policy strictly cheaper than the task) so the
    distance from the answer floor is a number rather than a blank.
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
                    or V.s5_bind_v3_trace_admits(r, k, m, ns, ng, named, query))
        fl = dict(V.s5_bind_v3_floors(items, k, m))
        fl.update(V.s5_bind_v3_family_floors(items, k, m, named, query, rows=fam))
        return fl, V.s5_bind_v3_ckpt_floors(items), (ns, ng, named, query)

    fs, cs, sh_s = rows_on(scored)
    fb, cb, sh_b = rows_on(big)
    ans = max(x for x in (V.s5_bind_v3_operative_floor(fs, k, m, *sh_s),
                          V.s5_bind_v3_operative_floor(fb, k, m, *sh_b)) if x is not None)
    tr = [V.s5_bind_v3_trace_operative_floor({**fs, **cs}, k, m, *sh_s),
          V.s5_bind_v3_trace_operative_floor({**fb, **cb}, k, m, *sh_b)]
    trace = None if all(x is None for x in tr) else max(x for x in tr if x is not None)
    world, _r = TK.build_world(spec)
    agents, objs = list(world.agents[:k]), list(world.objects[:m])
    agree_s, n_s = V.s5_bind_v3_trace_is_answer(scored, k, m, agents, objs)
    agree_b, n_b = V.s5_bind_v3_trace_is_answer(big, k, m, agents, objs)
    return {"cell": spec.name, "L": L, "k": k, "m": m, "chance": 1.0 / (k - 1),
            "n_scored": n_scored, "n_disjoint": n_big,
            "answer_floor": ans, "trace_floor": trace,
            "trace_basis": V.s5_bind_v3_trace_floor_basis(k, m, *sh_s),
            "slot_is_gold_scored": f"{agree_s}/{n_s}",
            "slot_is_gold_disjoint": f"{agree_b}/{n_b}",
            "ckpt_rows_scored": {r: round(v, 4) for r, v in cs.items()},
            "ckpt_rows_disjoint": {r: round(v, 4) for r, v in cb.items()},
            "copy_per_slot": V.s5_bind_v3_ckpt_copy_per_slot(scored, k, m, agents, objs),
            "slot_moves": V.s5_bind_v3_slot_moves(scored, k, m, agents, objs),
            "ckpt_lag": {j: V.s5_bind_v3_ckpt_lag(scored, j)
                         for j in V.S5_BIND_V3_CKPT_LAG if j < L},
            "pad_reach": (None if sh_s[2] else V.s5_bind_v3_trace_pad_floor(scored)),
            "trace_admitted": {r: round(v, 4) for r, v in
                               sorted({**fs, **cs}.items(), key=lambda x: -x[1])
                               if V.s5_bind_v3_trace_admits(r, k, m, *sh_s)}}


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
            tf = "unfloorable" if row["trace_floor"] is None else f"{row['trace_floor']:.4f}"
            print(f"  {key}@{L}: answer {row['answer_floor']:.4f} "
                  f"({row['answer_floor'] / row['chance']:.2f}x) | trace {tf} "
                  f"[{row['trace_basis']}]  slot==gold {row['slot_is_gold_scored']} / "
                  f"{row['slot_is_gold_disjoint']}", flush=True)
    return out


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


def verdict(control, comp_forms, comp_counts, matched_forms, matched_measured):
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

    Raises:
        ControlNotEvaluable: ``control`` is not an evaluated control.
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
    if comp_forms["composed"]:
        return "V2_NO_GAP_HERE", (
            "the composed cell forms, and so does each component at the length carrying the same "
            "amount of that component's own work. Composition is not a separate difficulty at "
            f"k=6 / L<={max(LOCAL_LENGTHS)} in this regime; the lengths or k must move before the "
            "cell is worth buying on the frontier.")
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
    from factworld.benchmark import MODELS, cost_estimate

    cells = ([{"task": FRONTIER_CELLS["composed"], "length": L, "n": n,
               "settings": {"effort": "high", "max_new_tokens": SCOUT_MAX_NEW_TOKENS}}
              for L in SCOUT_COMPOSED_LENGTHS]
             + [{"task": FRONTIER_CELLS[c], "length": L, "n": n,
                 "settings": {"effort": "high", "max_new_tokens": SCOUT_MAX_NEW_TOKENS}}
                for c in ("state", "bind") for L in SCOUT_COMPONENT_LENGTHS[c]])
    rows, total = [], 0.0
    for slug in models:
        if slug not in MODELS:
            rows.append({"model": slug, "error": "not in MODELS"})
            continue
        est = cost_estimate(slug, cells, assumed_output_tokens=4000)
        total += est["cost_usd"]
        rows.append({"model": slug, **est})
    # what the scout is gating: the same three cells over their OWN registered grids — the
    # composed cell's lengths and each component's work-matched partners of them — and the whole
    # roster, at the n a ranking needs. Both numbers belong in the decision, because the scout is
    # only worth its own price against what it stops.
    roster_cells = [{"task": FRONTIER_CELLS[c], "length": L, "n": ROSTER_N,
                     "settings": {"effort": "high", "max_new_tokens": SCOUT_MAX_NEW_TOKENS}}
                    for c in ("state", "bind", "composed")
                    for L in TK.CANONICAL[FRONTIER_CELLS[c]].eval_lengths]
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
                       "margin": MARGIN, "seeds_clear": SEEDS_CLEAR, "n_fit": N_FIT,
                       "n_fit_blocks": N_FIT_BLOCKS, "n_score": N_SCORE},
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
        # THE TRACE READ, declared with its arm restriction rather than described in prose. The
        # floors it carries are its own (validity.s5_bind_v3_trace_*): the components' are the
        # answer floors unchanged, and the composed cell's is None because the guided protocol
        # hands out the live slots that cell's floor argument is made of.
        "trace_read": {
            "read": TRACE_READ, "arms": list(TRACE_READ_ARMS),
            "requires": TRACE_READ_REQUIRES, "frontier_reads": list(FRONTIER_READS),
            "rule": "the model's own FINAL CHECKPOINT's value for the queried slot; a frontier "
                    "cell is scored on the answer and assert_trace_read() raises on any other "
                    "arm",
            "floors": (trace_floor_table(
                lengths={**guided_grid(ml), "composed": list(GUIDED_LENGTHS)})
                if with_floors else {}),
        },
        "verdicts": ["V5_HARNESS_NULL", "V4_COMPONENT_UNREADABLE", "V3_GAP_IS_THE_COST",
                     "V1_UNCONTROLLED", "V1_COMPOSITION_GAP", "V2_NO_GAP_HERE"],
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
    ap.add_argument("--axis", default=MATCHED_AXIS, choices=["tokens", "steps"])
    ap.add_argument("--scout", action="store_true", help="print the priced frontier scout")
    ap.add_argument("--trace_floors", default=None,
                    help="write the TRACE read's floor table (JSON path) and exit; the guided "
                         "grid only, since the trace read is defined under no other protocol")
    ap.add_argument("--out_prefix",
                    default="results/s5bind_v3_three_cell_preregistration_20260731")
    a = ap.parse_args()
    if a.scout:
        p = scout_plan()
        print(json.dumps(p, indent=2))
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
