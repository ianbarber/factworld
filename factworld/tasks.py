"""FactWorld task suite — frozen, versioned, scalable benchmark tasks (the reusable "body").

This is the layer that turns the bespoke experiment scripts into a *benchmark*: each task is a FROZEN
`TaskSpec` (pinned seed, version, explicit difficulty knobs and train/OOD-length splits), examples are
generated deterministically, and there is ONE canonical metric — **relaxed match** of the answer span
(whitespace / trailing-period invariant; see ``score_relaxed``). Exact match, semantic containment,
and last-*n* extraction are reported as diagnostics, not headline scores. Difficulty knobs (k,
n_objects, recall pool, lengths) are exposed so a task can be scaled to genuinely stress larger
models; the canonical registry pins reference instances.

Label discipline (inherited from the instrument): every example's gold answer comes from the symbolic
**oracle**, never from parsing rendered text — so labels cannot leak. This module is torch-free (data
generation needs no GPU).

Suite 1.1 adds the v2 binding/composite specs (``binding_v2`` / ``composite_copy_v2``): same knobs as
their v1 counterparts but with ``last_write_uniform=True``, which places the queried object's resolving
write uniformly over the stream instead of letting it cluster near the end (the v1 recency shortcut —
see the TaskSpec field and the RETIRED registry annotations). The recency-defective v1 family
(binding_v1 / binding_load_v1 / composite_v1 / composite_copy_v1 / composite_copy_scale_v1) is RETIRED
(issue #11): the scored registry (``CANONICAL`` / ``REPORTED``) carries ONE clean version per task; the
retired specs stay generable — and byte-identical (frozen-spec immutability; regression-pinned by
tests/goldens_prechange.json) — in the ``RETIRED`` dict for historical reproduction and the
defect-documentation tests, but are never scored.

The ``s5_bind`` family is the COMPOSED task: two structures over one event stream, each event
naming its second operand LIVE through one of them, so the composed query cannot be answered by
either component's algorithm. The registered version ablates the SOURCE STRUCTURE a reference
reads (``TaskSpec.source_ablation``); its within-cell op-type contrast is a STRUCTURE-SWITCH
diagnostic and not a composition measure — within a kind the class label IS the printed clause,
so a solver holding one structure is invisible to it at any n (``factworld.composition``).
Composition evidence comes from the three-cell comparison: the state component, the retrieval
component and the composed cell, each read against its own floor. The temporal version, which
ablated the time index instead, is RETIRED: its class was by construction the overwritten-cell
class, so a read-history failure loaded on the contrast as hard as a real deficit.

  from factworld.tasks import CANONICAL, generate, score_exact
  spec = CANONICAL["composite_copy_v2"]
  train = generate(spec, "train", n=8000)
  test  = generate(spec, "test", length=64)          # held-out OOD-length split, fixed seed
  acc   = sum(score_exact(pred, ex.answer) for pred, ex in zip(preds, test)) / len(test)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from .config import WorldConfig
from .oracle import Oracle
from .render import Renderer
from .world import Event, World

SUITE_VERSION = "1.1"

# The generation-stream version baked into every spec's RNG key at its introduction. This is
# deliberately DECOUPLED from SUITE_VERSION: `spec.version` feeds `_rng`, so retying it to the
# suite version would silently reshuffle every already-published spec's examples on a suite bump —
# violating "same (spec, split, length, idx) -> identical example, forever". v1 specs were
# introduced at "1.0" and stay there; new specs pin the version current at their introduction.
_STREAM_V1 = "1.0"

# The one canonical metric reported as the headline score (see ``score_relaxed``). The other scorers
# (exact / contains / last_n) are diagnostics. ``runner.evaluate_task`` reads this so the runner,
# the CLI, and the reports all agree on what "the score" means.
CANONICAL_METRIC = "relaxed"


@dataclass(frozen=True)
class TaskSpec:
    """A frozen, reproducible benchmark task. Difficulty knobs are explicit and scalable."""
    name: str
    family: str                       # 'recall' | 'binding' | 'composite' | 's5' | 'commutative' | 'conflict' | 'chain' | 's5_chain' | 's5_bind'
    version: str = _STREAM_V1         # RNG-stream version: frozen at spec introduction (see _STREAM_V1)
    chain_depth: int = 8              # s5_chain: number of a0 hops in the final query
    seed: int = 0
    # 'benchmark' = a scored, discriminating task; 'control' = a positive control / isolation task that is
    # degenerate as a capability score (e.g. memorized-map recall); 'experimental' = correct construct but
    # not reliably trainable in this harness yet (see s5_v1); 'retired' = superseded/defective spec kept
    # generable in RETIRED (never scored). Only 'benchmark' tasks are in REPORTED.
    kind: str = "benchmark"
    # world / breadth knobs
    k: int = 5                         # agents (= roles for s5); also the recall pool unless recall_pool set
    n_objects: int = 8
    value_vocab_size: int = 64
    n_objects_active: int = 4          # objects actually used in a give-stream (binding working set)
    # recall knobs
    recall_pool: int | None = None     # composite/recall: # facts presented (1-of-N); None -> use k
    memorized_recall: bool = True      # fixed agent->value map (memorizable) vs random per example (copy)
    # length / horizon knobs (the extrapolation axis)
    train_lengths: tuple = (4, 8, 16)  # binding-chain / permutation-history lengths in training
    eval_lengths: tuple = (16, 32, 64) # held-out OOD lengths
    worked_trace: bool = False         # s5/composite: emit the oracle state trajectory as a scratchpad
    # binding/composite: v2 give-stream sampler. The v1 sampler draws every event's object uniformly
    # from the active set, so the queried object's resolving (last) write sits ~Geometric(1/m) events
    # from the stream END regardless of L (median distance 1 at L16) — a strong recency heuristic
    # ("last event's recipient" [+ that holder's fact]) scores ~0.34@L16/0.21@L64, and L adds
    # distractor volume, not binding depth. With last_write_uniform=True the queried object is chosen
    # FIRST and its last write is placed uniformly over [floor(0.1*L), L-2] (never the final event,
    # never degenerate-early); later events draw only from the OTHER active objects, earlier events
    # from all of them (genuine overwrite history). Distance-from-end of the resolving write is then
    # ~Uniform and scales linearly with L, and the recency heuristic drops to ~chance.
    #
    # RESIDUAL DOCUMENTED FLOOR (v2, adversarial verification 2026-07-09): heuristics that FILTER
    # events by the queried object and guess among its give-recipients (uniformly / mode /
    # first-write) score E[mult(holder)/w] where w = the object's write count — measured ~0.52@L16 /
    # 0.20@L64 / 0.13@L128 on composite_copy_v2 (chance 1/16), ~0.58/0.34/0.27 on binding_v2
    # (chance 1/5). This is information-theoretic for last-write-wins, not a sampler defect: the
    # resolving write is exchangeable among the queried object's w writes, so an order-blind
    # adversary's hit rate cannot be pushed below ~E[1/w] by ANY sampler that keeps w << L
    # (multi-object interference). Unlike the v1 position shortcut (flat in L), this floor decays
    # ~1/L, so the L axis stays a genuine binding-depth axis; small-L cells (L<=16) should be read
    # against this floor, not against 1/pool.
    last_write_uniform: bool = False

    # commutative-only: dial positions mod k_positions (answer set p0..p{k_positions-1}; chance
    # 1/k_positions). n_objects_active is REUSED as the active-entity count for the commutative
    # family (the working set of dials that turn — per-entity filtering load, mirroring binding).
    # Appended + defaulted: _rng does not key on it, so no existing stream is perturbed.
    k_positions: int = 5

    # chain-only: explicit opt-in to depth >= k WRAP semantics. The pointer map is a single k-cycle, so
    # at depth >= k gold collapses to nxt^(depth mod k)(start) — depth is NOT being measured (depth ≡ 0
    # mod k is the identity). Off by default: generate() raises ValueError. The no-wrap deep-chain
    # protocol is spec.scaled(k=depth + 2).
    chain_allow_wrap: bool = False

    # s5_chain-only. distinct_path: VALIDITY GATE — require the query start to sit on a
    # final-map cycle of length >= chain_depth+1, so all depth+1 path nodes are distinct.
    # Without it the final permutation has cycles whose length divides the depth, and the
    # degenerate "answer the queried agent" (echo) strategy scores far above chance
    # (measured 0.16-0.32 on the v2 item streams vs 1/16 nominal chance); items whose
    # paths collapse onto short cycles are also individually easier, adding item-level
    # variance. Gated items hold echo and every fixed-hop heuristic at exactly 0 and make
    # item difficulty uniform (the path always visits depth+1 distinct agents).
    # event_trace: per-EVENT dense supervision — the trace prefixes the full a0 map (k
    # values in fixed agent order) after every event, i.e. a state checkpoint per event,
    # the supervision density that formed s5 locally; worked_trace alone emits only the
    # final query path, which supervises the dereference but NOT the map tracking.
    # start_trace: s5-SHAPED dense supervision — after each event emit only the query start's
    # current pointer nxt[start] (one token per event), the exact single-quantity checkpoint
    # shape that formed s5, rather than event_trace's full-map dump. Sufficient signal for
    # depth-1 readout; depth>=2 still requires the joint map (hop 2's identity is unknown
    # during the stream), so the start_trace d1-vs-d2 contrast isolates single-slot tracking
    # from joint-map maintenance.
    # compact_events: LOCAL-ONLY rendering ablation (issue #31) — events render in the
    # s5-style compact grammar ("s0 swaps a0: g1 and g3." / "s1 cycles a0: g1 -> g3 -> g5.",
    # ~4-5x fewer tokens per event than the canonical explicit-value sentences), to test
    # whether the wordy rendering is what blocks local circuit formation. The arrow cycle
    # form was retired for FRONTIER cells as ambiguity-confounded English; a from-scratch
    # model has no English prior — the training grammar's semantics are defined by the
    # generator — so the ablation is valid locally and never scored over the API.
    # Appended + defaulted: _rng does not key on either, so no existing stream is perturbed.
    distinct_path: bool = False
    event_trace: bool = False
    start_trace: bool = False
    compact_events: bool = False

    # s5_chain-only, depth 1 only: TYPED-VALUE ablation. In s5 the tracked values (roles
    # r0..r4) are a different token type from the slots they sit in (agents g0..g4); in
    # s5_chain the pointer values ARE agents, so every agent token in the stream is
    # ambiguous between "slot being written" and "value being moved" and the distinction
    # is recoverable only from syntactic position. With typed_values the a0 map sends
    # agents to ROLES, restoring the s5 type split while keeping the s5_chain event
    # grammar and stream. See _ex_s5_chain_typed and CANONICAL["s5_chain_typed_v1"].
    # Appended + defaulted: _rng does not key on it, and typed items are built by a
    # SEPARATE builder, so no existing stream is perturbed.
    typed_values: bool = False

    # s5_chain-only: the fraction of events that name their second operand BY REFERENCE TO
    # THE RUNNING STATE — "s7 swaps the values of g4's a0 and the a0 of the agent whose a0
    # is currently g11", i.e. the second slot is f^{-1}(g11) under the map as it stands when
    # the event fires. The a0 map is a bijection at every step (the initial map is a k-cycle
    # and every event composes it with a permutation), so that description names exactly one
    # agent.
    #
    # WHAT IT CHANGES. Unconditional swap/cycle events permute the DOMAIN of the map:
    # a swap of a and b sets f'(a)=f(b) and f'(b)=f(a), i.e. f' = f∘(a b), so after L events
    # f_L = f_0∘σ_1∘…∘σ_L and f_L(x) = f_0(σ_1(…σ_L(x))). One symbol pushed BACKWARD through
    # the event list answers the query, carrying log2(k) bits of state and never forming the
    # map. A referenced operand is unknown until f has been evaluated FORWARD to that event,
    # and resolving f^{-1}(gX) reads an arbitrary slot, so the forward pass has to carry the
    # whole map — log2(k!) bits. The backward walk is available to an attention model over
    # the full context and not to a streaming recurrent model; the forward walk is available
    # to both, which is what makes one construct measurable in both regimes.
    #
    # The referenced operand is drawn exactly as an unconditional swap's is (two distinct
    # uniform agents), so raising the rate changes the swap:cycle mix and nothing else about
    # the permutation stream. Appended + defaulted to 0.0: `_rng` does not key on it and the
    # draw is short-circuited at 0.0, so every existing example stream is untouched.
    conditional_rate: float = 0.0

    # s5_bind-only. The family runs TWO structures over ONE event stream and has every event
    # name its second operand through one of them.
    #
    #   p_swap      P(an event is a swap); the rest are gives. On a source-structure spec this
    #               is NOT a taste knob — see TaskSpec.source_ablation: a swap moves two cells
    #               of the first structure and a give writes one of the second, so p_swap is
    #               what equalises the two structures' write rates, and equal write rates are
    #               what make the CROSS and SAME op classes matched on read history.
    #   query_arm   which query is scored: 'state' (the queried agent's final pointer/role),
    #               'bind' (the queried object's final holder), or 'state_all' (the whole-map
    #               readout, which prices capacity separately from composition).
    #   rho_p       RETIRED temporal family only: fraction of swaps rendered "at this point".
    #   rho_b       RETIRED temporal family only: the same for gives.
    #   coupled     RETIRED temporal family only: the time-index rendering toggle.
    #
    # m, the number of objects, is n_objects_active (reused as this family's working set the
    # way the commutative rung reuses it for active dials); m <= k is required so the stated
    # holder map is injective. All five fields are appended and defaulted, `_rng` does not
    # key on them, so no pre-existing stream moves.
    p_swap: float = 0.5
    rho_p: float = 1.0
    rho_b: float = 1.0
    coupled: bool = True
    query_arm: str = "state"

    # The RNG-stream identity, defaulting to `name`. Two specs that share it generate the
    # SAME items and differ only in what is rendered and asked, which is what makes the
    # s5_bind arms exactly paired: the composed cell and its components are one item stream
    # read three ways, so a per-item difference between arms is a within-item comparison and
    # not a difference of samples. Appended and defaulted to None, and `_rng` resolves
    # `stream_name or name`, so every existing spec keys exactly as it did.
    stream_name: str | None = None

    # s5_bind-only. Forbid a swap whose referenced object still carries a LIVE PIN — the
    # sampler constraint that closes the family's state-free reset channel.
    #
    # THE CHANNEL. A dynamic give, "give o to the agent whose role at this point is r", writes
    # B[o] <- Pinv[r]; from that event until o is written again its holder is pinned, in the
    # sense that whoever holds o has role r. A later dynamic swap, "swap the roles of a and the
    # holder of o at this point", then sets P[a] <- r EXACTLY, because selecting an agent by
    # its role and reading that role back returns the role. The two references cancel the
    # state, so on such items the answer to a state query is two surface retrievals and no map
    # is carried. It is not recency: the channel is length-free, so the policy that reads it
    # (validity's ``pin_chain`` row) does not decay with L, and the recency-window rows were
    # reading the same plateau — which is why the operative floor stopped falling with length
    # instead of reaching chance.
    #
    # THE CONSTRAINT. With no_pin the sampler rejects a candidate dynamic swap whose referenced
    # object's live pin still matches its holder's current role, and redraws. Pin density goes
    # to zero, the pin_chain row and the window family both land at informed chance, and the
    # cheapest correct algorithm is untouched — the coupled cell still costs a forward pass
    # carrying both maps, so the step multiplier does not move.
    #
    # Appended and defaulted to False, and the rejection is short-circuited at the default, so
    # every pre-existing stream is byte-identical. The s5_bind specs set it True.
    no_pin: bool = False

    # s5_bind-only. The set of doses ONE skeleton must be simultaneously admissible under —
    # the field that makes a dose ladder item-paired instead of five independent samples.
    #
    # THE DEFECT IT FIXES. In the default sampler the per-event coupling coin is drawn INSIDE
    # the rejection loop, so a rejected candidate consumes it and the RNG stream diverges: two
    # specs differing only in rho draw different worlds, different events and different queries
    # at the same index. Dose comparisons are then between-item (0 of 60 prompts match at
    # k=12/L=64), so a dose response confounds the dose with the sample.
    #
    # THE SKELETON-FIRST SAMPLER. With rho_ladder set, the coupling variate u_i is drawn ONCE
    # per event SLOT, before the operands, and a candidate event is admitted only if it is
    # non-degenerate under EVERY dose in the ladder (no self-swap, no no-op write, and with
    # no_pin no live-pin cancellation); the query gates likewise hold under every dose. Nothing
    # in the draw sequence depends on the spec's own rho, so every rung of the ladder builds
    # the SAME world, the SAME event list and the SAME two queries, and the renderings differ
    # only in which events carry "at this point" — identical whitespace-token counts, and gold
    # is the one thing that moves. Event i is dynamic at dose d iff u_i < d, so the doses are
    # nested: the referenced set at 0.25 is a subset of the set at 0.5, and dose 0.0 IS the
    # all-static reading.
    #
    # Appended and defaulted to (), which selects the original sampler, so every pre-existing
    # stream is byte-identical. A spec that sets it must have rho_p == rho_b (the ladder is
    # one-dimensional) and that value must be one of its rungs.
    rho_ladder: tuple[float, ...] = ()

    # s5_bind-only. The queried agent's LAST carrier event must sit in the final ``q_tail``
    # fraction of the stream — the query gate that makes the stream's TAIL load-bearing, and the
    # mirror of the band the queried object's resolving write already sits in.
    #
    # THE HOLE IT CLOSES. Truncation is a two-sided family: a policy may drop the last T events
    # or the first T, and at matched T both pay the same price. Only the suffix half was gated.
    # The state gate asked that the queried agent MOVE (touched twice, final role different from
    # its stated one) but never said WHEN, so its carrier chain typically finished mid-stream and
    # a policy that simulated the task exactly and then stopped 10% early — validity's
    # ``prefix_90`` — was simply right: 0.4527 / 0.3747 / 0.2920 at k=12, L=128/192/256 against
    # an operative floor of 0.098-0.117. Nothing about it is shallow, which is the point: the
    # last decile of the stream carried no information about the answer, so the events were
    # priced into the prompt without being priced into the task.
    #
    # THE CONSTRAINT. With q_tail = f the sampler keeps only agents whose last carrier event
    # (as the named agent OR as the referenced holder) has index >= L - round(f*L) under EVERY
    # simulated dose, which is exactly the set of events a prefix_(1-f) policy discards. It is a
    # gate on the QUERY, not on the stream: no event distribution moves, the cheapest correct
    # algorithm is untouched, and it costs items only where no agent qualifies.
    #
    # Appended and defaulted to 0.0, which disables the gate, so every pre-existing stream is
    # byte-identical. The s5_bind specs set it to 0.1, matching the tightest registered
    # truncation budget (validity.S5_BIND_WINDOWS).
    q_tail: float = 0.0


    # ---- s5_bind_v3: the SOURCE-STRUCTURE ablation ----------------------------------------
    # ``source_ablation`` selects the v3 builder (_ex_s5_bind_v3) and nothing else in this
    # dataclass changes meaning; it is appended, defaulted False and read before the first RNG
    # draw, so every pre-existing stream is byte-identical.
    #
    # WHY IT EXISTS. The v2 construct ablated the TIME INDEX: the same reference was rendered
    # "at this point" (resolve against the running map) or "at the start" (against the stated
    # one). But the stated structure is BY DEFINITION the one before any write, so a reference
    # witnesses composition exactly when its referenced cell has been written since the start —
    # measured on the v2 generator, P(the two readings differ | write count of the read cell =
    # 0) = 0.000 and = 1.000 at one write, at both scored cells. The composition class IS the
    # overwritten-cell class, so a composition-free solver whose failures concentrate on
    # overwritten cells (bounded capacity with LRU eviction; stale-value interference) loads on
    # the contrast at type-I rates at or above the real deficit's power. The temporal ablation
    # cannot identify composition, and no repair on the write count fixes it because there is no
    # support at zero writes.
    #
    # WHAT REPLACES IT. Every reference stays DYNAMIC; only the SOURCE STRUCTURE varies. Two maps
    # into agents run over one stream — P: agents -> agents (a permutation, rewritten by swaps)
    # and B: objects -> agents (last-write-wins, rewritten by gives) — and each event names its
    # second operand through one of them:
    #     swap (writes P)  CROSS "the agent {o} belongs to at this point"  -> reads B
    #                      SAME  "the agent {g} points to at this point"   -> reads P
    #     give (writes B)  CROSS "the agent {g} points to at this point"   -> reads P
    #                      SAME  "the agent {o} belongs to at this point"  -> reads B
    # Both classes are live reads of overwritten cells, at matched write counts and matched
    # retrieval distances, so interference is common to them and cancels in theta_cross -
    # theta_same, while a solver that cannot hold the other structure fails only on CROSS.
    #
    # THE ONE CONFOUND THE SURFACES CARRY, and where it is handled. Within an event kind the
    # class IS the reference clause, so a solver that is simply worse at one clause slips on
    # swap-CROSS and give-SAME. The clause-to-class map FLIPS between the kinds, so the effect
    # lives entirely in the ANTI-symmetric combination of the two kinds' class differences, and
    # the diagnostic's default form (factworld.composition.DIAGNOSTIC_STAT) is built to be
    # uncorrelated with that direction: one RAW mass column per stratum, so any hazard that is a
    # function of the stratum sits in the model's span with a zero contrast, and a
    # precision-weighted contrast column, so the clause direction projects onto it at zero. That
    # is a statement about the COLUMNS; a solver that holds one structure and not the other still
    # fails on exactly the antisymmetric direction the kind-balancing annihilates, which is why
    # the contrast is a structure-SWITCH diagnostic and never a composition measure.
    #
    #   p_cross       P(an event's reference reads the OTHER structure).
    #                 The per-slot cross coin is drawn BEFORE the operands and both candidate
    #                 operands are drawn on every slot, so which class a slot lands in does not
    #                 depend on which operands the rejection loop happened to admit.
    #
    #                 THERE IS NO p_cross = 0 ARM. A SAME give copies another object's holder,
    #                 so with no cross-structure injection the holder map is a pure coalescent:
    #                 it collapses onto one agent, every further same-structure give is a no-op,
    #                 and the sampler cannot fill the stream (measured at k = m = 6, L = 64:
    #                 200 of 200 draws starve). The all-SAME reading is not a cell, and it is not
    #                 needed — the ablation is WITHIN the composed cell, one op class against the
    #                 other, so the control is the theta_same class itself.
    #   event_kinds   'both' (the composed stream), 'swap' (P events only) or 'give' (B events
    #                 only) — the two COMPONENT arms.
    #   named_operands  render the second operand by NAME instead of by reference. The component
    #                 arms set it: a named-operand swap stream is the S5 word problem (a sparse
    #                 backward carrier walk answers it), a named-operand give stream is
    #                 last-write-wins retrieval. Reference-rendered components are not the
    #                 components — they cost a forward pass over their own structure.
    #   match_reads   WITHIN-KIND CLASS MATCHING, as a sampler draw rule: the number of matched
    #                 candidate draws a slot takes the best of, or 0 for no matching.
    #
    #                 WHAT IT FIXES. p_swap = 1/3 equalises the two structures' write RATES, so
    #                 the two classes are matched when pooled — but the diagnostic is KIND-BALANCED,
    #                 and within a kind they were not. Measured on the shipped stream at
    #                 k=6/L=64, a swap's CROSS read was 28.5% staler than its SAME read (age
    #                 10.16 vs 7.91) with 13.0% fewer writes behind it; at k=12/L=192, 20.0%
    #                 staler. The cause is the ``no_pin`` gate: it fires only on the CROSS
    #                 reading, and the objects it refuses are exactly the recently written ones,
    #                 so it selects the CROSS class toward older cells. Pooled the gap hides
    #                 (18.64 vs 16.72) because the gives run the other way. A composition-free
    #                 solver that is merely RECENCY-BOUNDED then reads as composition: a hard
    #                 forgetting horizon at H=18, holding both structures identically, produced a
    #                 larger contrast than a real cross-only deficit at the same accuracy.
    #
    #                 THE RULE. Every slot chooses BOTH candidate reference cells — the object
    #                 ``ref_o`` (a B cell) and the agent ``ref_a`` (a P cell) — before the cross
    #                 coin is consulted, and chooses them MATCHED: a fair coin picks which pool
    #                 leads, the leader is drawn uniformly over its admissible cells, and the
    #                 other pool supplies the cell nearest the leader's read history (_nearest).
    #                 Leading always with the same pool does not work — the gated pool is offset
    #                 in age, so a one-directional projection inherits the whole offset (measured
    #                 at k=6: +17.1% leading with the objects, +7.5% with the leader randomised)
    #                 — so the direction is randomised and the two projection errors cancel. The
    #                 slot then takes the BEST of ``match_reads`` such draws. Best-of-N and not a
    #                 tolerance: a tolerance rejects, and a per-slot rejection costs item yield
    #                 like (1 - p)^L — a (3, 2) tolerance costs 34 item restarts at k=6/L=64 and
    #                 completes none at L=96 — while also conditioning the whole stream on a
    #                 property of every one of its slots.
    #
    #                 N IS PER OPERATING POINT, and measured. At k=12 ONE matched draw already
    #                 equalises the two classes' read histories to within 1% of each other and
    #                 further draws only cost yield. At k=6 the pools are six cells wide: one
    #                 draw leaves +4.6% / -10.6% on the two kinds, two close the swap kind
    #                 (+4.5%, and P(distance > 18) 0.0549 against 0.0545), and three or more
    #                 starve the L=96 cell — tighter matching feeds the holder map's coalescence,
    #                 the give pool empties, and the item restarts. The residual that leaves is
    #                 measured and reported, not assumed away (scripts/probe_s5bind_v3_statistic).
    #
    #                 IT IS NOT ENOUGH ON ITS OWN: the admission must also be coin-INDEPENDENT,
    #                 or the rule that admits a slot is itself a class-conditional selection.
    #                 Degeneracy is therefore filtered out of BOTH candidate pools, and the
    #                 ``no_pin`` gate is applied to the object pool on EVERY slot rather than
    #                 only on the ones the coin sent CROSS. Appended and defaulted to 0, so no
    #                 pre-existing stream moves.
    source_ablation: bool = False
    p_cross: float = 0.5
    event_kinds: str = "both"
    named_operands: bool = False

    # s5_bind_v3-only. The queried agent's LAST naming swap must not name the answer on the
    # surface — the query gate that kills the state-free surface read.
    #
    # THE LEAK. Scan back to the last swap whose FIRST operand is the queried agent. On a SAME
    # swap that sentence names an AGENT ("the agent g7 points to at this point"), and the swap
    # writes P[a] <- P(P(g7)), which is g7 exactly when g7 sits on a cycle of P of length 1 or 2
    # — probability 2/k with no other structure, i.e. 1.83x the informed chance 1/(k-1) at
    # k = 12 before the rest of the stream dilutes it. Emitting that slot is one backward scan
    # over two registers and carries no map at all. Measured on the ungated stream at n = 3000,
    # conditional on the branch: 0.1279 = 1.41x, z = +4.96 at k=12/L=128 and 0.2503 = 1.25x,
    # z = +4.90 at k=6/L=64. On a CROSS swap the same sentence names an OBJECT, which is not a
    # candidate answer; reading it through the stated holder map is the mirror policy and that
    # branch sits BELOW chance (0.84x / 0.86x). The component arms name their operand outright
    # and read 0.94x / 0.74x, so they do not set the gate.
    #
    # THE CONSTRAINT. With q_no_surface the sampler keeps only agents for which that reading is
    # WRONG under every simulated dose, on BOTH branches — the policy has to answer every item,
    # so it is the completed policy that is gated and not the elevated half of it. It is a gate
    # on the QUERY, not on the stream: no event distribution moves and the cheapest correct
    # algorithm is untouched. Retention is not what it costs (item restarts go 1.027 -> 1.030 at
    # k=12/L=128 and 2.403 -> 2.442 at k=6/L=96); FLOOR is. Striking an answer the sampler has
    # emptied hands a guesser the mass it carried, which is the registered
    # ``uniform_anti_surface`` row (factworld.validity) and the honest price of the gate.
    #
    # Appended and defaulted to False, which disables the gate, so every pre-existing stream is
    # byte-identical.
    q_no_surface: bool = False
    match_reads: int = 0

    def scaled(self, **knobs) -> "TaskSpec":
        """Return a harder/easier variant (e.g. spec.scaled(k=64, recall_pool=64, eval_lengths=(32,128)))."""
        return replace(self, **knobs)


@dataclass
class Example:
    prompt: str        # the model input (the full query, ending in '?')
    answer: str        # the exact expected continuation (space-separated atomic tokens)
    length: int        # the difficulty coordinate this example was drawn at
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# deterministic RNG: keyed by (spec, split, length, index) so train/test never overlap and are fixed
# ---------------------------------------------------------------------------
def _rng(spec: TaskSpec, split: str, length: int, idx: int) -> random.Random:
    # `stream_name or name`: specs that share a stream_name draw the SAME items (the s5_bind
    # arms). It defaults to None everywhere else, so every pre-existing spec keys on its name.
    return random.Random(
        f"factworld|task|{spec.stream_name or spec.name}|{spec.version}|{spec.seed}|{split}|{length}|{idx}")


def _world(spec: TaskSpec) -> tuple[World, Renderer, Oracle]:
    wc = WorldConfig(seed=spec.seed, n_entities=8, n_attributes=2,
                     value_vocab_size=spec.value_vocab_size, n_objects=spec.n_objects,
                     n_locations=6, k=spec.k, k_positions=spec.k_positions)
    w = World(wc)
    return w, Renderer(), Oracle(w)


def build_world(spec: TaskSpec) -> tuple[World, Renderer]:
    """Public: the (World, Renderer) a task is built on — e.g. to build a tokenizer covering it."""
    w, r, _ = _world(spec)
    return w, r


def _fixed_origins(spec: TaskSpec, w: World) -> dict:
    rng = random.Random(f"factworld|origins|{spec.name}|{spec.seed}")
    return dict(zip(w.agents, rng.sample(list(w.value_vocab), spec.k)))


def _param_map(spec: TaskSpec, w: World) -> dict:
    """The 'in-weights' agent→value map for the conflict family — constant across training so a model
    memorizes it, and the in-context value at test deliberately contradicts it."""
    rng = random.Random(f"factworld|parammap|{spec.name}|{spec.seed}")
    return {a: rng.choice(list(w.value_vocab)) for a in w.agents}


def _render_answer(core: str) -> str:
    """Render the answer span with attached punctuation (e.g. `g4.`, `g0 v109.`) to match the
    natural-language prompt style."""
    return f"{core}."


# ---------------------------------------------------------------------------
# per-family example builders
# ---------------------------------------------------------------------------
def _ex_recall(spec, w, r, fixed_origins, rng, split, length, idx):
    """Associative recall (1-of-N). memorized -> fixed map (fixed pool); in-context-copy -> the pool size
    is the LENGTH axis, so `eval_lengths` grows the distractor set (a real recall-extrapolation axis)."""
    if spec.memorized_recall:
        pool = spec.recall_pool or spec.k
        chosen = list(w.agents[:pool]); origins = fixed_origins
    else:
        pool = min(length, len(w.agents))                          # 1-of-N where N = length (the difficulty axis)
        chosen = rng.sample(list(w.agents), pool)
        origins = dict(zip(chosen, rng.sample(list(w.value_vocab), pool)))
    g = rng.choice(chosen)
    facts = " ".join(r.render_fact(a, "a0", origins[a], key=f"{a}|{idx}|{rng.random()}") for a in chosen)
    q = r.render_query("recall", entity=g, attribute="a0")
    return Example(f"{facts} {q}", _render_answer(origins[g]), length)


def _uniform_last_write_stream(rng, length, objs, recipients):
    """v2 give-stream sampler (see TaskSpec.last_write_uniform). Chooses the queried object FIRST,
    places its last write at position p ~ Uniform[floor(0.1*L), L-2] (never the final event, never
    degenerate-early), then fills the stream: events after p draw only from the OTHER active objects
    (interference continues to the end), events before p draw from ALL of them (the queried object
    may be overwritten earlier — genuine history). Returns (obj, p, events). No post-hoc filtering,
    so determinism stays simple and item counts exact."""
    assert length >= 2 and len(objs) >= 2, "uniform-last-write stream needs L>=2 and >=2 active objects"
    obj = rng.choice(objs)
    p = rng.randint(length // 10, length - 2)          # length // 10 == floor(0.1 * L)
    others = [o for o in objs if o != obj]
    ev = []
    for i in range(length):
        o = obj if i == p else rng.choice(objs if i < p else others)
        ev.append(Event("give", (o, rng.choice(recipients))))
    return obj, p, ev


def _ex_binding(spec, w, r, oracle, rng, length):
    """Last-write-wins binding (no recall): resolve the current holder of an object, answer the agent."""
    objs = list(w.objects[:spec.n_objects_active])
    if spec.last_write_uniform:                        # v2 sampler: uniform resolving-write position
        obj, p, ev = _uniform_last_write_stream(rng, length, objs, list(w.agents))
        meta = {"obj": obj, "last_write_pos": p}
    else:                                              # v1 sampler (frozen): uniform object draws
        ev = [Event("give", (rng.choice(objs), rng.choice(w.agents))) for _ in range(length)]
        obj = rng.choice(sorted({e.args[0] for e in ev}))
        meta = {"obj": obj}
    holder = oracle.easy_holder(ev, obj)
    hist = " ".join(r.render_history(tuple(ev), with_steps=True))
    q = r.render_query("state_easy", target=obj)
    return Example(f"{hist} {q}", _render_answer(holder), length, meta)


def _ex_composite(spec, w, r, oracle, fixed_origins, rng, length, idx):
    """binding × recall: resolve holder (oracle) then recall its a0. memorized or in-context-copy recall."""
    pool = spec.recall_pool or spec.k
    chosen = list(w.agents[:pool]) if spec.memorized_recall else rng.sample(list(w.agents), pool)
    origins = fixed_origins if spec.memorized_recall else dict(zip(chosen, rng.sample(list(w.value_vocab), pool)))
    objs = list(w.objects[:spec.n_objects_active])
    if spec.last_write_uniform:                        # v2 sampler: uniform resolving-write position
        obj, p, ev = _uniform_last_write_stream(rng, length, objs, chosen)
    else:                                              # v1 sampler (frozen): uniform object draws
        ev = [Event("give", (rng.choice(objs), rng.choice(chosen))) for _ in range(length)]
        obj = rng.choice(sorted({e.args[0] for e in ev}))
        p = None
    holder = oracle.easy_holder(ev, obj)                          # gold via the oracle
    value = origins[holder]
    facts = " ".join(r.render_fact(a, "a0", origins[a], key=f"{a}|{idx}|{rng.random()}") for a in chosen)
    hist = " ".join(r.render_history(tuple(ev), with_steps=True))
    q = r.render_query("recall", attribute="a0", entity=f"the holder of {obj}")
    prompt = f"{facts} {hist} {q}"
    meta = {"holder": holder, "obj": obj}
    if p is not None:
        meta["last_write_pos"] = p
    if spec.worked_trace:    # oracle worked-trace = optional TRAINING signal, not part of the scored answer
        meta["trace"] = " ".join(oracle.easy_holder(ev, obj, t=t) for t in range(1, length + 1))
    return Example(prompt, _render_answer(f"{holder} {value}"), length, meta)


def _ex_commutative(spec, w, r, oracle, rng, length):
    """Per-entity commutative accumulation mod k_positions (the abelian rung between
    last-write-wins binding and non-abelian S_k).

    Each event turns one active agent's dial by a NONZERO amount in {1..k_positions-1}
    (never 0 — every event changes state; never constant — count-of-events mod k is wrong).
    The answer is the queried agent's final dial position. Order is irrelevant (addition
    mod k is abelian) but EVERY matching event is load-bearing, so length L reads as
    aggregation depth (expected target turns = 2 + (L-2)/m grows linearly in L) — there is
    no recency structure for a sampler to leak (a commutative fold has no "resolving" event).

    Design guards against the shallow adversaries (gated in factworld.validity):
      - initial positions are PER-EXAMPLE random -> gold is EXACTLY uniform (majority = floor)
        and the stated-initial baseline decays as (-1/(k-1))^w toward chance;
      - >= 2 turns on the queried agent are FORCED -> the last-turn-only cheat is depressed
        BELOW chance at short L (a single leftover nonzero amount is never ≡ 0 mod k);
      - m = n_objects_active distractor agents stay active -> the entity-blind total sits
        at chance (per-entity filtering is required, mirroring binding).
    """
    assert length >= 2, "commutative stream needs L>=2 (two forced turns on the queried agent)"
    ents = list(w.agents[:spec.n_objects_active])
    initial = {g: rng.choice(w.positions) for g in ents}   # per-example random (forces reading it)
    target = rng.choice(ents)
    forced = set(rng.sample(range(length), 2))             # guarantee w_q >= 2
    events = [Event("turn_dial", ((target if i in forced else rng.choice(ents)),
                                  str(rng.randint(1, spec.k_positions - 1))))
              for i in range(length)]
    gold = oracle.comm_position(initial, events, target)
    init_block = " ".join(r.render_dial(g, initial[g]) for g in ents)
    hist = " ".join(r.render_history(tuple(events), with_steps=True))
    q = r.render_query("state_comm", target=target)
    meta = {"target": target, "initial": initial[target],
            "w": sum(1 for e in events if e.args[0] == target)}
    if spec.worked_trace:    # oracle position-trajectory = optional TRAINING signal, not the scored answer
        meta["trace"] = " ".join(oracle.comm_trace(initial, events, target)[1:])
    return Example(f"{init_block} {hist} {q}", _render_answer(gold), length, meta)


def _ex_s5(spec, w, r, oracle, rng, length, idx):
    """Sₖ state tracking: role of an agent after a swap/cycle history (worked-trace = the role trajectory)."""
    events = w.sample_hard_chain(length, episode_seed=f"{spec.name}|{idx}")
    agent = rng.choice(w.agents)
    role = oracle.hard_role(events, agent)
    # State the initial role assignment explicitly. The answer depends on it, but the g_k=r_k
    # convention is otherwise unstated — a presentation defect (see Appendix A.5: stating it
    # recovers a few points but does not crack the wall). The other families state their initial
    # conditions (recall/conflict/chain via the facts, binding/composite via the give-stream), so
    # this makes s5 consistent with the rest of the suite. Rendered via ``render_role`` so it is
    # part of the canonical grammar and applies uniformly to train and test.
    init = " ".join(r.render_role(a, w.initial_assignment[a]) for a in w.agents)
    hist = " ".join(r.render_history(tuple(events), with_steps=True))
    q = r.render_query("state_hard", target=agent)
    prompt = f"{init} {hist} {q}"
    meta = {}
    if spec.worked_trace:    # oracle role-trajectory = optional TRAINING signal, not the scored answer
        meta["trace"] = " ".join(oracle.hard_role(events, agent, t=t) for t in range(1, length + 1))
    return Example(prompt, _render_answer(role), length, meta)


def _compact_a0_event(e: Event) -> str:
    """The s5-style compact rendering of one pointer-map event (TaskSpec.compact_events).

    Local-only: ~4-5x fewer tokens per event than the canonical explicit-value sentences.
    The referenced operand keeps the relative clause ("whose a0 is g11") because that IS
    the construct — the compact form drops words, never the state reference."""
    if e.kind == "swap_a0":
        return f"swaps a0: {e.args[0]} and {e.args[1]}."
    if e.kind == "swap_a0_ref":
        return f"swaps a0: {e.args[0]} and whose a0 is {e.args[1]}."
    return "cycles a0: " + " -> ".join(e.args) + "."


def _ex_s5_chain(spec, w, r, rng, length, idx):
    """s5_chain: non-abelian pointer-map state + serial dereference in composition.

    World: k agents, initial a0 map = a single k-cycle (as in chain). Stream: length
    swap/cycle events on the a0 targets (order-sensitive, so the map must be tracked).
    Query: apply a0 `spec.chain_depth` times starting from a random agent. Gold = the
    agent reached under the FINAL map. The task composes s5-style permutation tracking
    with chain-style serial dereference; depth is kept < k so the cycle never wraps.
    `length` is the number of permutation events; chain depth is a spec knob.

    With ``spec.conditional_rate`` > 0 that fraction of events names its second operand
    by reference to the running map ("the agent whose a0 is currently g11") instead of
    by name, which fixes the event's identity only after the map has been evaluated
    forward to it. See the TaskSpec field for what that changes about the computation.
    """
    depth = spec.chain_depth
    if depth >= spec.k:
        raise ValueError(
            f"{spec.name}: chain depth {depth} >= k={spec.k} wraps the {spec.k}-cycle; "
            f"use spec.scaled(k={depth + 2}) for a no-wrap protocol."
        )
    cyc = rng.sample(list(w.agents), spec.k)
    nxt0 = {cyc[i]: cyc[(i + 1) % spec.k] for i in range(spec.k)}
    present = cyc[:]; rng.shuffle(present)
    facts = " ".join(r.render_fact(a, "a0", nxt0[a], key=f"{a}|{idx}|{rng.random()}") for a in present)

    def _cycle_len(mapping, a):
        n, x = 1, mapping[a]
        while x != a:
            n, x = n + 1, mapping[x]
        return n

    for _attempt in range(100):
        nxt = dict(nxt0)
        events: list[Event] = []
        maps: list[dict] = []                # per-event map snapshots (trace construction)
        for _ in range(length):
            # `spec.conditional_rate and ...` short-circuits at the 0.0 default, so a spec
            # without referenced operands draws exactly the numbers it drew before the knob
            # existed and its stream stays byte-identical.
            if spec.conditional_rate and rng.random() < spec.conditional_rate:
                # The second operand is named by its CURRENT value, f^{-1}(nxt[b]) = b; the
                # rendered event therefore carries (a, nxt[b]) and only the running map
                # resolves it back to b. Operands are drawn exactly as the unconditional
                # swap's are, so the permutation stream's distribution is unchanged.
                a, b = rng.sample(cyc, 2)
                events.append(Event("swap_a0_ref", (a, nxt[b])))
                nxt[a], nxt[b] = nxt[b], nxt[a]
            elif rng.random() < 0.5:
                a, b = rng.sample(cyc, 2)
                events.append(Event("swap_a0", (a, b)))
                nxt[a], nxt[b] = nxt[b], nxt[a]
            else:
                a, b, c = rng.sample(cyc, 3)
                events.append(Event("cycle_a0", (a, b, c)))
                nxt[a], nxt[b], nxt[c] = nxt[b], nxt[c], nxt[a]
            if spec.event_trace or spec.start_trace:
                maps.append(dict(nxt))
        if not spec.distinct_path:
            starts = cyc
        else:
            # gate: only starts whose final-map cycle is longer than the query depth,
            # so the depth+1 path nodes are all distinct (echo/fixed-hop floors = 0).
            starts = [a for a in cyc if _cycle_len(nxt, a) > depth]
            if not starts:                   # no long cycle formed: resample the events
                continue
        break
    else:
        raise RuntimeError(f"{spec.name}: no length->{depth+1} cycle in 100 event resamples")
    if spec.compact_events:
        ev_txts = [f"s{i} {_compact_a0_event(e)}" for i, e in enumerate(events)]
    else:
        ev_txts = [r.render_event(e, step=f"s{i}", key=f"h|{i}|{e.kind}|{'|'.join(e.args)}")
                   for i, e in enumerate(events)]
    hist = " ".join(ev_txts)

    start = rng.choice(starts)
    path = [start]
    for _ in range(depth):
        path.append(nxt[path[-1]])
    gold = path[-1]
    query = "what is " + "a0 of " * depth + f"{start}? ({depth} hops)"
    meta = {"depth": depth, "start": start, "path": path}
    if spec.conditional_rate:
        # positions of the state-referencing events: the last one is the point up to which
        # the map has to be evaluated forward before any backward walk can start.
        meta["ref_positions"] = tuple(i for i, e in enumerate(events) if e.kind == "swap_a0_ref")
    if spec.start_trace:
        meta["trace"] = " ".join([m[start] for m in maps] + path[:-1])
        # Interleaved variant of the same supervision (the protocol that formed s5): the
        # checkpoint token follows its event INSIDE the stream, so credit assignment is
        # local. Training docs use this; evaluation is free-running on the plain prompt.
        meta["interleaved_prompt"] = (
            f"{facts} " + " ".join(f"{t} {m[start]}" for t, m in zip(ev_txts, maps)) + f" {query}")
    elif spec.event_trace:
        meta["trace"] = " ".join(" ".join(m[a] for a in cyc) for m in maps) + " " + " ".join(path[:-1])
    elif spec.worked_trace:
        meta["trace"] = " ".join(path[:-1])
    return Example(f"{facts} {hist} {query}", _render_answer(gold), length, meta)


def _ex_s5_chain_typed(spec, w, r, rng, length, idx):
    """s5_chain with TYPED pointer values — the key/value type-ambiguity ablation (depth 1).

    The a0 map sends agents to ROLES instead of to agents: facts read ``g3's a0 is r5.``, the
    events are the same swap_a0/cycle_a0 stream over the same agent slots, and the query is a
    single dereference ``what is a0 of g3? (1 hops)`` answered with a role. Everything else
    follows ``_ex_s5_chain`` apart from the two consequences enumerated below.

    WHAT THE CONTRAST TESTS. Against the depth-1 untyped arm (``s5_chain_local_v2`` at
    ``chain_depth=1``) — same k, same lengths, same event distribution, same supervision, same
    rendering — the intended difference is whether a token in value position can also appear in
    slot position. In the untyped task every agent id occurs in both roles, so a from-scratch
    model must resolve "is this the slot being written or the value being moved?" from syntax
    alone, with no lexical cue; a pretrained model gets that distinction from English, which is
    one reason the frontier and local regimes may not be measuring the same thing here. In s5,
    which DOES form locally, the split is lexical: slots are agents, values are roles.

    IT IS NOT A ONE-KNOB CONTRAST. Two further things differ, both entailed by the type split
    rather than chosen, and both of which move the floors:

      INITIAL-MAP STRUCTURE. ``_ex_s5_chain`` builds its initial map as a single k-cycle over
      the k sampled agents — no fixed points, every agent reachable from every other. Values
      here come from a disjoint pool, so the map is instead a uniform random agent -> role
      bijection and "cycle" is undefined for it. The two arms therefore start from differently
      distributed maps, which is visible in the initial-map-chase floor.

      distinct_path. The untyped arm carries the gate; this one does not (and
      ``CANONICAL["s5_chain_typed_v1"]`` leaves it False). At depth 1 the gate restricts the
      queried start to a non-fixed-point of the FINAL map, resampling the event stream in the
      rare case where no such start exists. Typing buys what the gate buys — a role can never
      equal an agent, so echo scores 0 by construction — but not the same conditioning: the
      untyped answer space excludes the queried agent and so has chance 1/(k-1), while the
      typed answer space is the k roles at chance 1/k.

    A spec-level diff of the two arms is pinned in tests/test_s5_chain.py; only ``name``,
    ``typed_values`` and ``distinct_path`` may differ.

    READING THE OUTCOMES (both arms at the same k, depth 1, over >= 8 seeds, each against its
    OWN operative floor — ``validity.operative_floor``, the max over the registered shallow
    adversaries — never against 1/k and never against the other arm's floor):
      typed forms, untyped floors   — key/value type ambiguity is the binding constraint at
                                      depth 1. The s5_chain null is then about representation,
                                      not about composing tracking with dereference, and the
                                      informative next step is a typed depth-2 construct.
      both form                     — depth 1 is not where s5_chain fails; the null belongs to
                                      composition depth and the depth-2 arms are the experiment.
      both floor                    — the depth-1 readout itself does not form under this
                                      budget, so no depth-2 result is interpretable and the
                                      budget-matched arm has to come first.
      typed floors, untyped forms   — the ablation is confounded (a role-typed answer space is
                                      not harder in any intended sense); inspect the predictions
                                      before reading anything else in the family.

    Depth is restricted to 1: values are not keys here, so there is nothing to chase twice.
    """
    if spec.chain_depth != 1:
        raise ValueError(
            f"{spec.name}: typed_values is a depth-1 construct (chain_depth={spec.chain_depth}); "
            f"the pointer values are roles, not agents, so a second hop has no defined start.")
    if spec.conditional_rate:
        raise ValueError(
            f"{spec.name}: typed_values and conditional_rate are not combined "
            f"(conditional_rate={spec.conditional_rate}); the typed arm is the depth-1 "
            f"key/value ablation and its floors are measured without state references.")
    if spec.k > len(w.roles):
        raise ValueError(f"{spec.name}: k={spec.k} exceeds the {len(w.roles)}-role value pool")
    slots = rng.sample(list(w.agents), spec.k)
    vals = rng.sample(list(w.roles), spec.k)
    # a uniformly random agent -> role bijection; _ex_s5_chain uses a single k-cycle instead,
    # which is not expressible across disjoint pools (see the docstring)
    cur = dict(zip(slots, vals))
    present = slots[:]; rng.shuffle(present)
    facts = " ".join(r.render_fact(a, "a0", cur[a], key=f"{a}|{idx}|{rng.random()}") for a in present)

    events: list[Event] = []
    maps: list[dict] = []
    for _ in range(length):
        if rng.random() < 0.5:
            a, b = rng.sample(slots, 2)
            events.append(Event("swap_a0", (a, b)))
            cur[a], cur[b] = cur[b], cur[a]
        else:
            a, b, c = rng.sample(slots, 3)
            events.append(Event("cycle_a0", (a, b, c)))
            cur[a], cur[b], cur[c] = cur[b], cur[c], cur[a]
        if spec.event_trace or spec.start_trace:
            maps.append(dict(cur))

    if spec.compact_events:
        ev_txts = [f"s{i} {_compact_a0_event(e)}" for i, e in enumerate(events)]
    else:
        ev_txts = [r.render_event(e, step=f"s{i}", key=f"h|{i}|{e.kind}|{'|'.join(e.args)}")
                   for i, e in enumerate(events)]
    hist = " ".join(ev_txts)

    start = rng.choice(slots)
    gold = cur[start]
    path = [start, gold]
    query = f"what is a0 of {start}? (1 hops)"
    meta = {"depth": 1, "start": start, "path": path, "typed_values": True}
    if spec.start_trace:
        meta["trace"] = " ".join([m[start] for m in maps] + path[:-1])
        meta["interleaved_prompt"] = (
            f"{facts} " + " ".join(f"{t} {m[start]}" for t, m in zip(ev_txts, maps)) + f" {query}")
    elif spec.event_trace:
        meta["trace"] = " ".join(" ".join(m[a] for a in slots) for m in maps) + " " + " ".join(path[:-1])
    elif spec.worked_trace:
        meta["trace"] = " ".join(path[:-1])
    return Example(f"{facts} {hist} {query}", _render_answer(gold), length, meta)


def _invert(mapping: dict) -> dict:
    return {v: k for k, v in mapping.items()}


def _s5_bind_tail_lo(spec, length: int) -> int:
    """The first event index the queried agent's last carrier event may have (TaskSpec.q_tail).

    ``L - round(f*L)`` is the first index a prefix policy on budget 1-f discards, so an agent
    that clears this gate is one whose answer that policy cannot have. -1 disables the gate.
    """
    if not spec.q_tail:
        return -1
    return length - max(1, int(round(spec.q_tail * length)))


def _s5_bind_lanes(spec) -> tuple[float, ...]:
    """The doses one skeleton has to be simultaneously admissible under. 0.0 is always a lane:
    it is the all-static reading, which every arm of the family is also read under."""
    return tuple(sorted(set(spec.rho_ladder) | {0.0}))


def _s5_bind_stream(spec, agents, roles, objs, rng, length, idx):
    """One s5_bind item's world, event stream and queries — the part of the sampler that must
    not consult the rendering.

    Returns ``(P0, B0, events, writes, last_give, q_state, q_bind, fact_roles, fact_holds,
    finals)``, where ``finals`` maps each simulated dose (and ``None``, the all-static reading)
    to that dose's ``(P, B, touch)``. Events carry ``dyn`` under the spec's OWN dose.

    Two samplers, selected by ``spec.rho_ladder``:

      DEFAULT      each event's coupling coin is drawn inside the rejection loop, and the item
                   is checked for degeneracy under the spec's own dose and under the all-static
                   reading. Two doses, two lanes.
      SKELETON-FIRST (rho_ladder set)
                   the coupling variate is drawn once per event SLOT, before the operands, and
                   the item is checked under every dose in the ladder. The draw sequence is then
                   independent of the spec's own dose, so every rung of the ladder is the same
                   item — see TaskSpec.rho_ladder.
    """
    k, m = spec.k, spec.n_objects_active
    if not spec.rho_ladder:
        for _outer in range(200):
            P0 = dict(zip(agents, rng.sample(roles, k)))
            B0 = dict(zip(objs, rng.sample(agents, m)))
            P0inv = _invert(P0)
            Pc, Bc = dict(P0), dict(B0)          # the coupled ("at this point") trajectory
            Pd, Bd = dict(P0), dict(B0)          # the decoupled ("at the start") trajectory
            Pc_inv = _invert(Pc)
            events, touch_c, touch_d = [], {a: 0 for a in agents}, {a: 0 for a in agents}
            # a -> the index of the last event that MOVED a, per trajectory (TaskSpec.q_tail)
            last_c, last_d = {a: -1 for a in agents}, {a: -1 for a in agents}
            writes = {o: 0 for o in objs}
            # o -> the role its last DYNAMIC give named, while that give still stands: the live
            # pin (see TaskSpec.no_pin). A static give leaves no pin, and any give clears it.
            pin: dict[str, str | None] = {}
            for _i in range(length):
                swap = rng.random() < spec.p_swap
                ok = False
                for _try in range(200):
                    if swap:
                        a, o = rng.choice(agents), rng.choice(objs)
                        dyn = rng.random() < spec.rho_p
                        bc = Bc[o] if dyn else B0[o]
                        bd = B0[o]
                        if bc != a and bd != a:          # no self-swap under EITHER semantics
                            if spec.no_pin and dyn and pin.get(o) is not None and Pc[bc] == pin[o]:
                                continue                 # the references would cancel the state
                            ok = True
                            break
                    else:
                        o, rl = rng.choice(objs), rng.choice(roles)
                        dyn = rng.random() < spec.rho_b
                        hc = (Pc_inv if dyn else P0inv)[rl]
                        hd = P0inv[rl]
                        if hc != Bc[o] and hd != Bd[o]:  # no no-op write under EITHER semantics
                            ok = True
                            break
                if not ok:
                    break
                if swap:
                    events.append({"kind": "swap", "a": a, "ref": o, "dyn": dyn})
                    Pc[a], Pc[bc] = Pc[bc], Pc[a]
                    Pd[a], Pd[bd] = Pd[bd], Pd[a]
                    Pc_inv = _invert(Pc)
                    touch_c[a] += 1
                    touch_c[bc] += 1
                    touch_d[a] += 1
                    touch_d[bd] += 1
                    last_c[a] = last_c[bc] = last_d[a] = last_d[bd] = len(events) - 1
                else:
                    events.append({"kind": "give", "o": o, "ref": rl, "dyn": dyn})
                    Bc[o], Bd[o] = hc, hd
                    pin[o] = rl if dyn else None
                    writes[o] += 1
            if len(events) < length:
                continue
            tail_lo = _s5_bind_tail_lo(spec, length)
            cand_s = [a for a in agents if touch_c[a] >= 2 and touch_d[a] >= 2
                      and Pc[a] != P0[a] and Pd[a] != P0[a]
                      and last_c[a] >= tail_lo and last_d[a] >= tail_lo]
            last_give = {}
            for j, e in enumerate(events):
                if e["kind"] == "give":
                    last_give[e["o"]] = j
            lo, hi = length // 10, int(0.75 * length)
            cand_b = [o for o in objs if writes[o] >= 2 and Bc[o] != B0[o] and Bd[o] != B0[o]
                      and lo <= last_give.get(o, -1) <= hi]
            if not cand_s or not cand_b:
                continue
            q_state, q_bind = rng.choice(cand_s), rng.choice(cand_b)
            fact_roles, fact_holds = agents[:], objs[:]
            rng.shuffle(fact_roles)
            rng.shuffle(fact_holds)
            finals = {spec.rho_p: (Pc, Bc, touch_c), None: (Pd, Bd, touch_d)}
            return (P0, B0, events, writes, last_give, q_state, q_bind,
                    fact_roles, fact_holds, finals)
        raise RuntimeError(f"{spec.name}: no admissible item at idx={idx} "
                           f"(k={k}, m={m}, L={length})")

    lanes = _s5_bind_lanes(spec)
    if spec.rho_p != spec.rho_b:
        raise ValueError(f"{spec.name}: a rho_ladder spec is one-dimensional, so rho_p "
                         f"({spec.rho_p}) must equal rho_b ({spec.rho_b}).")
    if spec.rho_p not in lanes:
        raise ValueError(f"{spec.name}: rho_p={spec.rho_p} is not a rung of "
                         f"rho_ladder={lanes}.")
    for _outer in range(200):
        P0 = dict(zip(agents, rng.sample(roles, k)))
        B0 = dict(zip(objs, rng.sample(agents, m)))
        P0inv = _invert(P0)
        # one trajectory per rung; the all-static reading IS the 0.0 rung
        st = {d: {"P": dict(P0), "Pinv": _invert(P0), "B": dict(B0), "pin": {},
                  "touch": {a: 0 for a in agents}, "last": {a: -1 for a in agents}}
              for d in lanes}
        events, writes = [], {o: 0 for o in objs}
        for _i in range(length):
            swap = rng.random() < spec.p_swap
            u = rng.random()                     # THE SKELETON DRAW: one coin per event slot,
            ok = False                           # before the operands and outside the retries
            for _try in range(200):
                if swap:
                    a, o = rng.choice(agents), rng.choice(objs)
                else:
                    o, rl = rng.choice(objs), rng.choice(roles)
                res, good = {}, True
                for d in lanes:
                    s = st[d]
                    dyn = u < d
                    if swap:
                        x = (s["B"] if dyn else B0)[o]
                        if x == a:                             # self-swap: a no-op event
                            good = False
                            break
                        if (spec.no_pin and dyn and s["pin"].get(o) is not None
                                and s["P"][x] == s["pin"][o]):
                            good = False                       # state-free reset channel
                            break
                    else:
                        x = (s["Pinv"] if dyn else P0inv)[rl]
                        if x == s["B"][o]:                     # no-op write
                            good = False
                            break
                    res[d] = x
                if good:
                    ok = True
                    break
            if not ok:
                break
            events.append({"kind": "swap", "a": a, "ref": o, "dyn": u < spec.rho_p} if swap
                          else {"kind": "give", "o": o, "ref": rl, "dyn": u < spec.rho_p})
            for d in lanes:
                s, x = st[d], res[d]
                if swap:
                    s["P"][a], s["P"][x] = s["P"][x], s["P"][a]
                    s["Pinv"] = _invert(s["P"])
                    s["touch"][a] += 1
                    s["touch"][x] += 1
                    s["last"][a] = s["last"][x] = len(events) - 1
                else:
                    s["B"][o] = x
                    s["pin"][o] = rl if u < d else None
            if not swap:
                writes[o] += 1
        if len(events) < length:
            continue
        tail_lo = _s5_bind_tail_lo(spec, length)
        cand_s = [a for a in agents
                  if all(st[d]["touch"][a] >= 2 and st[d]["P"][a] != P0[a]
                         and st[d]["last"][a] >= tail_lo for d in lanes)]
        last_give = {}
        for j, e in enumerate(events):
            if e["kind"] == "give":
                last_give[e["o"]] = j
        lo, hi = length // 10, int(0.75 * length)
        cand_b = [o for o in objs
                  if writes[o] >= 2 and lo <= last_give.get(o, -1) <= hi
                  and all(st[d]["B"][o] != B0[o] for d in lanes)]
        if not cand_s or not cand_b:
            continue
        q_state, q_bind = rng.choice(cand_s), rng.choice(cand_b)
        fact_roles, fact_holds = agents[:], objs[:]
        rng.shuffle(fact_roles)
        rng.shuffle(fact_holds)
        finals = {d: (st[d]["P"], st[d]["B"], st[d]["touch"]) for d in lanes}
        finals[None] = finals[0.0]
        return (P0, B0, events, writes, last_give, q_state, q_bind,
                fact_roles, fact_holds, finals)
    raise RuntimeError(f"{spec.name}: no admissible item at idx={idx} "
                       f"(k={k}, m={m}, L={length}, rho_ladder={lanes})")


def _ex_s5_bind(spec, w, r, rng, length, idx):
    """The RETIRED temporal construct (tasks.RETIRED), kept generable.

    WORLD. k agents, k roles, m <= k objects. P maps agents to roles and is a bijection permuted
    by swap events; B maps objects to agents under last-write-wins. Both initial maps are STATED
    ("g3 has role r1 at the start." / "g7 holds o2 at the start.") in scrambled order.

    STREAM. Each event is a swap with probability ``p_swap``, else a give, and each names its
    second operand through the OTHER structure:
      swap — "s0 swaps the roles of g4 and the agent who holds o2 {when}."
      give — "s1 gives o3 to the agent whose role {when} is r5."
    ``when`` is "at this point" on a ``rho_p`` / ``rho_b`` fraction of swaps / gives when
    ``spec.coupled``, and "at the start" otherwise, at identical whitespace-token counts.

    WHY IT IS RETIRED. That temporal pair cannot identify composition. "At the start" is by
    definition the structure before any write, so a reference witnesses composition exactly when
    its referenced cell has been written since the start; the composition class IS the
    overwritten-cell class, and a composition-free solver whose failures concentrate on
    overwritten cells loads on the contrast as hard as a real deficit. The construct is
    superseded by the SOURCE-STRUCTURE version (``_ex_s5_bind_v3``), where every reference is
    live and only the structure it reads varies. The retirement note on tasks.RETIRED carries
    the measurements.

    NO_PIN and Q_TAIL are still set on the retired specs and still do what they say: no_pin
    closes the give -> swap reset channel (two dynamic references cancel the state, so a
    2-retrieval policy carrying no map answers a state query), and q_tail makes the stream's
    tail load-bearing. Their floor rows live in ``factworld.validity`` and still run.
    """
    k, m = spec.k, spec.n_objects_active
    if m > k:
        raise ValueError(f"{spec.name}: m={m} objects > k={k} agents; the stated holder map "
                         f"must be injective, so n_objects_active <= k.")
    if m > len(w.objects):
        raise ValueError(f"{spec.name}: m={m} exceeds the {len(w.objects)}-object pool "
                         f"(raise n_objects).")
    if spec.query_arm not in ("state", "bind", "state_all"):
        raise ValueError(f"{spec.name}: query_arm={spec.query_arm!r} not in "
                         f"{{'state','bind','state_all'}}")
    agents, roles = list(w.agents[:k]), list(w.roles[:k])
    objs = list(w.objects[:m])
    coupled = spec.coupled

    (P0, B0, events, writes, last_give, q_state, q_bind,
     fact_roles, fact_holds, finals) = _s5_bind_stream(spec, agents, roles, objs,
                                                       rng, length, idx)
    P0inv = _invert(P0)
    Pc, Bc, touch_c = finals[spec.rho_p]      # the coupled ("at this point") trajectory
    Pd, Bd, touch_d = finals[None]            # the decoupled ("at the start") trajectory

    facts = [r.render_role(a, P0[a], when=Renderer.AT_START) for a in fact_roles]
    facts += [r.render_holder(o, B0[o], when=Renderer.AT_START) for o in fact_holds]
    ev_txts = []
    for i, e in enumerate(events):
        dyn = e["dyn"] and coupled
        if e["kind"] == "swap":
            evt = Event("swap_roles_now" if dyn else "swap_roles_start", (e["a"], e["ref"]))
        else:
            evt = Event("give_role_now" if dyn else "give_role_start", (e["o"], e["ref"]))
        ev_txts.append(r.render_event(evt, step=f"s{i}"))

    P_fin, B_fin = (Pc, Bc) if coupled else (Pd, Bd)
    if spec.query_arm == "state":
        query = r.render_query("s5bind_state", target=q_state)
        gold = P_fin[q_state]
    elif spec.query_arm == "bind":
        query = r.render_query("s5bind_bind", target=q_bind)
        gold = B_fin[q_bind]
    else:
        query = r.render_query("s5bind_state_all", targets=agents)
        gold = " ".join(P_fin[a] for a in agents)

    meta = {"q_state": q_state, "q_bind": q_bind, "coupled": coupled,
            "n_swap": sum(1 for e in events if e["kind"] == "swap"),
            "n_ref": sum(1 for e in events if e["dyn"]) if coupled else 0,
            "touch": touch_c[q_state] if coupled else touch_d[q_state],
            "writes": writes[q_bind],
            "last_write_pos": last_give[q_bind]}
    if spec.event_trace or spec.worked_trace:
        # Replayed under the ACTIVE semantics: full_map is the s5 recipe (the whole state after
        # every event, P in agent order then B in object order); resolved is the single-quantity
        # checkpoint (each event's resolved operand), the shape that did NOT form locally on
        # s5_chain and so is carried as the supervision-density contrast arm.
        P, B = dict(P0), dict(B0)
        Pinv = _invert(P)
        snaps, resolved = [], []
        for e in events:
            dyn = e["dyn"] and coupled
            if e["kind"] == "swap":
                b = (B if dyn else B0)[e["ref"]]
                a = e["a"]
                P[a], P[b] = P[b], P[a]
                Pinv = _invert(P)
                resolved.append(b)
            else:
                h = (Pinv if dyn else P0inv)[e["ref"]]
                B[e["o"]] = h
                resolved.append(h)
            snaps.append(" ".join(P[a] for a in agents) + " " + " ".join(B[o] for o in objs))
        if spec.event_trace:
            meta["trace"] = " ".join(snaps)
            # Interleaved variant of the same supervision (the protocol that formed s5): each
            # checkpoint follows its event INSIDE the stream, so credit assignment is local.
            # Training docs use this; evaluation is free-running on the plain prompt.
            meta["interleaved_prompt"] = (
                " ".join(facts) + " " + " ".join(f"{t} {s}" for t, s in zip(ev_txts, snaps))
                + f" {query}")
        else:
            meta["trace"] = " ".join(resolved)
    return Example(" ".join(facts + ev_txts + [query]), _render_answer(gold), length, meta)


def _s5_bind_v3_lanes(spec) -> tuple[float, ...]:
    """The cross-doses one skeleton must be admissible under. One lane: the spec's own dose.

    Kept as a tuple because the sampler simulates a dict of trajectories keyed by dose, which is
    what lets a future dose comparison be item-paired without moving the draw sequence.
    """
    return (spec.p_cross,)


S5_BIND_V3_BIND_TAIL = 0.75
# THE SOURCE-STRUCTURE RETRIEVAL WINDOW, and the floor argument that rests on it. The queried
# object's RESOLVING (last) write must land in [L // 10, S5_BIND_V3_BIND_TAIL * L], so its
# distance from the end of the stream is at least L - 1 - floor(0.75 L) — a quarter of the
# stream. A policy that reads fewer events than that cannot have read the write, so on the
# retrieval component every bounded scan is exactly a guess, which is what makes that cell's
# floor informed chance rather than an accident of which rows were written down
# (factworld.validity, the component rule).
#
# WHAT IT COSTS, AND WHY L IS NOT A FREE AXIS HERE. The window is a REJECTION, and its acceptance
# rate falls exponentially in (1 - tail) L / m: an object qualifies only if it takes no write in
# the last quarter of the stream, which is ~(1 - 1/m)^{L/4} per object. Measured acceptance, as a
# fraction of drawn streams: 0.62 / 0.19 / 0.047 at m = 12 and L = 128/192/256, and 0.54 / 0.079
# at m = 6 and L = 48/96. The retrieval component therefore costs 22 stream restarts per item at
# its longest cell against 1.6 at its shortest, and the same window at L = 512 would admit
# roughly one stream in 500. The restart cap below is sized against that rate rather than a
# typical one: at 200 it fails one item in ~4000 at L = 256 (idx 1771 of the s5_bind_v3_bind
# test split), which is a crash and not a degradation.
S5_BIND_V3_STREAM_RESTARTS = 4000


def s5_bind_v3_bind_window(length: int) -> tuple[int, int]:
    """The event indices the queried object's resolving write may occupy, ``(lo, hi)`` inclusive.

    Stated once and read by both the sampler and ``factworld.validity``, so the cost model and
    the stream cannot drift: the retrieval component's own algorithm is a backward scan whose
    length this window sets, and the floor rule prices every row against it.
    """
    return length // 10, int(S5_BIND_V3_BIND_TAIL * length)


def _nearest(pool, key, target, rng):
    """The pool cell whose read history is closest to ``target``, ties broken uniformly.

    ``key`` maps a cell to ``(last-write index, write count)`` and the distance is that pair's
    componentwise absolute difference, ordered lexicographically: recency first, because it is
    the nuisance the bounded-memory executors read, and the write count as the tiebreak.
    """
    best, best_d = [], None
    for c in pool:
        d = (abs(key[c][0] - target[0]), abs(key[c][1] - target[1]))
        if best_d is None or d < best_d:
            best, best_d = [c], d
        elif d == best_d:
            best.append(c)
    return rng.choice(best)


def _derangement(items, rng):
    """A permutation of ``items`` with no fixed point, by rejection — so no stated pointer fact
    is ``g4 points to g4``, which would be a degenerate reference target."""
    for _ in range(200):
        perm = rng.sample(items, len(items))
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))
    raise RuntimeError("no derangement drawn")


# What the admission rules cost, as counters a diagnostic can read: operand draws attempted per
# admitted slot, and item restarts per item returned. Generation is deterministic, so these are
# a pure measurement of the sampler's rejection rate and nothing reads them back.
ADMISSION = {"items": 0, "slots": 0, "draws": 0, "restarts": 0, "short": 0}


def _s5_bind_v3_stream(spec, agents, objs, rng, length, idx):
    """One source-structure item's world, event skeleton and queries.

    SKELETON-FIRST, always. Each slot draws its kind, then the cross coin ``u``, then BOTH
    candidate reference operands (an object and an agent) and the named operand — before any
    dose is consulted. A slot is admitted only if it is non-degenerate under EVERY dose in
    ``_s5_bind_v3_lanes`` (no self-swap, no no-op write, and with ``no_pin`` no reference whose
    value a solver carrying the OTHER structure alone could already compute). Nothing in the
    draw sequence depends on the spec's own ``p_cross``, so two specs differing only in the dose
    build the SAME world, the SAME event kinds and the SAME queries: the composed cell and its
    same-structure control are one item stream read twice.

    Returns ``(P0, B0, events, moves, writes, last_move, last_write, q_state, q_bind,
    fact_agents, fact_objs, finals)``; ``finals[d] = (P, B)`` per dose.

    ``no_pin`` closes the REDUCTION channel on the object side, and is applied to the SLOT. A
    CROSS give writes B[o] <- P[b], so until P[b] moves again the equality B[o] == P[b] is live
    and a later CROSS swap naming o resolves to a value a P-only solver already has: the op is
    rendered CROSS and costs no composition. The provenance propagates through SAME gives
    (B[o] <- B[o2] inherits o2's) and dies the moment the grounding agent's pointer moves. The
    gate refuses such an object as a reference on EVERY slot, not only on the ones the coin sent
    CROSS, because a gate that fires on one class is a class-conditional selection — and this
    one selects on recency, since a pinned object is by construction a recently written one.

    THE MIRROR CHANNEL IS LEFT OPEN, and that is a power cost, not a size one. The same live
    equality is readable from the agent side: a CROSS give naming b resolves to a value a B-only
    solver already has. Refusing those agents too closes the channel (measured reducible CROSS
    density 0.156 -> 0.000 at k=12) but restricts the agent pool by ANTI-recency — an agent stays
    pinned exactly while it does not move — so the two candidate pools are then offset in
    opposite directions and no per-slot matching can bring them together: the within-kind age gap
    goes to +6.2%/-4.0% at k=12 and +20.0%/-14.3% at k=6 against +1.0%/+0.7% and +7.5%/-8.2% with
    the object gate alone. A diluted CROSS class costs power on the give kind; an unmatched one
    costs the null.

    ``match_reads`` matches the two classes WITHIN kind at the draw (see TaskSpec.match_reads), and
    every degeneracy filter runs over both candidate pools, so the set of admitted slots does not
    depend on the cross coin and the coin is the only thing that decides the class.
    """
    k, m = spec.k, spec.n_objects_active
    lanes = _s5_bind_v3_lanes(spec)
    named = spec.named_operands
    kinds = spec.event_kinds
    if kinds not in ("both", "swap", "give"):
        raise ValueError(f"{spec.name}: event_kinds={kinds!r} not in {{'both','swap','give'}}")
    for _outer in range(S5_BIND_V3_STREAM_RESTARTS):
        ADMISSION["restarts"] += 1
        P0 = _derangement(agents, rng)
        B0 = dict(zip(objs, rng.sample(agents, m)))
        st = {d: {"P": dict(P0), "B": dict(B0),
                  "prov": {o: None for o in objs},          # o -> b with B[o] == P[b], or None
                  "moves": {a: 0 for a in agents},
                  "last": {a: -1 for a in agents},
                  # a -> the agent the LAST swap naming a as its first operand names on the
                  # surface: the reference slot itself on a SAME swap, that object's stated
                  # holder on a CROSS one, the named partner on a component swap. The
                  # state-free read TaskSpec.q_no_surface gates on.
                  "surf": {}} for d in lanes}
        events, writes, last_write = [], {o: 0 for o in objs}, {}
        for _i in range(length):
            swap = kinds == "swap" or (kinds == "both" and rng.random() < spec.p_swap)
            u = rng.random()                       # THE SKELETON DRAW: before the operands
            ok, best, n_good = False, None, 0
            for _try in range(200):
                ADMISSION["draws"] += 1
                if named:
                    a = rng.choice(agents)         # swap: the named first operand
                    o = rng.choice(objs)           # give: the object written
                    ref_o = rng.choice(objs)
                    ref_a = rng.choice(agents)
                    nmd = rng.choice(agents)       # the named rendering's second operand
                    good = True
                    for d in lanes:
                        s = st[d]
                        if (nmd == a) if swap else (nmd == s["B"][o]):
                            good = False           # self-swap / no-op write
                            break
                    res = dict.fromkeys(lanes, nmd)
                    if good:
                        ok = True
                        break
                    continue
                a = rng.choice(agents)
                o = rng.choice(objs)
                nmd = None
                # THE MATCHED DRAW. Both candidate reference cells are chosen on every slot,
                # against the same read history, so the cross coin decides the class and nothing
                # else does. One pool leads on a fair coin and the other supplies its nearest
                # cell, which symmetrises the projection error the leading direction carries.
                anchor = rng.random()          # which pool leads this matched draw
                res, good, gap = {}, True, (0, 0)
                for d in lanes:
                    s = st[d]
                    if swap:                       # no self-swap under either reading
                        pool_o = [c for c in objs if s["B"][c] != a]
                        pool_a = [c for c in agents if s["P"][c] != a]
                    else:                          # no "gives o to whoever holds o", no no-op
                        pool_o = [c for c in objs if c != o and s["B"][c] != s["B"][o]]
                        pool_a = [c for c in agents if s["P"][c] != s["B"][o]]
                    if spec.no_pin:
                        pool_o = [c for c in pool_o if s["prov"][c] is None]
                    if not pool_o or not pool_a:
                        good = False               # every candidate is degenerate at this slot
                        break
                    key_o = {c: (last_write.get(c, -1), writes[c]) for c in pool_o}
                    key_a = {c: (s["last"][c], s["moves"][c]) for c in pool_a}
                    if not spec.match_reads:       # the unmatched control: two free draws
                        ref_o, ref_a = rng.choice(pool_o), rng.choice(pool_a)
                    elif anchor < 0.5:
                        ref_o = rng.choice(pool_o)
                        ref_a = _nearest(pool_a, key_a, key_o[ref_o], rng)
                    else:
                        ref_a = rng.choice(pool_a)
                        ref_o = _nearest(pool_o, key_o, key_a[ref_a], rng)
                    gap = max(gap, (abs(key_o[ref_o][0] - key_a[ref_a][0]),
                                    abs(key_o[ref_o][1] - key_a[ref_a][1])))
                    res[d] = (s["B"][ref_o] if swap else s["P"][ref_a]) if u < d \
                        else (s["P"][ref_a] if swap else s["B"][ref_o])
                if not good:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, a, o, ref_o, ref_a, res)
                n_good += 1
                if n_good >= max(1, spec.match_reads):
                    ok = True
                    break
            if not ok and best is None:
                ADMISSION["short"] += 1            # every candidate pool was empty at this slot
                break
            # THE MATCH IS BEST-OF-N, NOT A TOLERANCE. Rejecting a slot whose pair misses a
            # stated tolerance makes the item yield fall like (1 - p)^L: at k=6 a tolerance of
            # (3, 2) costs 34 item restarts at L=64 and cannot complete one at L=96, and it
            # conditions the whole stream on a property of every one of its slots. Taking the
            # best of N independent matched draws tightens the same quantity with no rejection
            # at all, and the draw does not read the cross coin, so it selects no class.
            if not named:
                _gap, a, o, ref_o, ref_a, res = best
            ADMISSION["slots"] += 1
            events.append({"kind": "swap" if swap else "give", "u": u, "a": a, "o": o,
                           "ref_o": ref_o, "ref_a": ref_a, "named": nmd})
            for d in lanes:
                s, x = st[d], res[d]
                if swap:
                    s["surf"][a] = (nmd if named else
                                    (B0[ref_o] if u < d else ref_a))
                    s["P"][a], s["P"][x] = s["P"][x], s["P"][a]
                    for g in (a, x):
                        s["moves"][g] += 1
                        s["last"][g] = len(events) - 1
                        for oo in objs:                          # the equality B[oo]==P[g] dies
                            if s["prov"][oo] == g:
                                s["prov"][oo] = None
                else:
                    s["B"][o] = x
                    cross = u < d
                    s["prov"][o] = (ref_a if cross else s["prov"][ref_o]) if not named else None
            if not swap:
                writes[o] += 1
                last_write[o] = len(events) - 1
        if len(events) < length:
            continue
        tail_lo = _s5_bind_tail_lo(spec, length)
        cand_s = [a for a in agents
                  if all(st[d]["moves"][a] >= 2 and st[d]["P"][a] != P0[a]
                         and st[d]["last"][a] >= tail_lo
                         and (not spec.q_no_surface
                              or st[d]["P"][a] != st[d]["surf"].get(a))
                         for d in lanes)]
        lo, hi = s5_bind_v3_bind_window(length)
        cand_b = [o for o in objs
                  if writes[o] >= 2 and lo <= last_write.get(o, -1) <= hi
                  and all(st[d]["B"][o] != B0[o] for d in lanes)]
        need_s = spec.query_arm in ("state", "state_all")
        if (need_s and not cand_s) or (spec.query_arm == "bind" and not cand_b):
            continue
        q_state = rng.choice(cand_s) if cand_s else None
        q_bind = rng.choice(cand_b) if cand_b else None
        fact_agents, fact_objs = agents[:], objs[:]
        rng.shuffle(fact_agents)
        rng.shuffle(fact_objs)
        finals = {d: (st[d]["P"], st[d]["B"]) for d in lanes}
        moves = {a: st[spec.p_cross]["moves"][a] for a in agents}
        last_move = {a: st[spec.p_cross]["last"][a] for a in agents}
        ADMISSION["items"] += 1
        return (P0, B0, events, moves, writes, last_move, last_write, q_state, q_bind,
                fact_agents, fact_objs, finals)
    raise RuntimeError(f"{spec.name}: no admissible item at idx={idx} "
                       f"(k={k}, m={m}, L={length}, kinds={kinds}, lanes={lanes})")


def _s5_bind_v3_event(spec, e, cross: bool) -> Event:
    """The rendered Event one skeleton slot becomes at a given dose. The four reference forms
    plus the two named-operand forms; the named give reuses the suite's plain ``give``."""
    if spec.named_operands:
        return (Event("swap_ptr_named", (e["a"], e["named"])) if e["kind"] == "swap"
                else Event("give", (e["o"], e["named"])))
    if e["kind"] == "swap":
        return (Event("swap_ptr_by_b", (e["a"], e["ref_o"])) if cross
                else Event("swap_ptr_by_p", (e["a"], e["ref_a"])))
    return (Event("give_ptr_by_p", (e["o"], e["ref_a"])) if cross
            else Event("give_ptr_by_b", (e["o"], e["ref_o"])))


def _ex_s5_bind_v3(spec, w, r, rng, length, idx):
    """s5_bind_v3: two structures over one event stream, ablated on the SOURCE STRUCTURE.

    WORLD. k agents, m <= k objects. P: agents -> agents is a derangement at the start and is
    rewritten by swaps; B: objects -> agents is stated and rewritten by gives under
    last-write-wins. Both maps land in the SAME type, which is what lets the two reference
    clauses have the same shape and the same length:

        g4 points to g9 at the start.          o2 belongs to g7 at the start.
        s0 swaps the pointers of g4 and the agent o2 belongs to at this point.   (CROSS: reads B)
        s0 swaps the pointers of g4 and the agent g7 points to at this point.    (SAME:  reads P)
        s1 gives o3 to the agent g7 points to at this point.                     (CROSS: reads P)
        s1 gives o3 to the agent o7 belongs to at this point.                    (SAME:  reads B)

    Within an event kind the two classes are identical in whitespace-token count (16 for a swap,
    13 for a give) and in register — "the agent {slot} {verb} to at this point" — so the ablation
    moves the SOURCE and nothing a token counter can see. EVERY reference is live; there is no
    static reading, which is the point (see TaskSpec.source_ablation).

    WRITE-COUNT MATCHING IS A SAMPLER PROPERTY, not an accident. A swap moves TWO agents'
    pointers and a give writes ONE object, so the two structures' cells accumulate writes at
    equal rates iff 2 p_swap / k = (1 - p_swap) / m. At m = k that is p_swap = 1/3, which is what
    the registered specs set: the cell a CROSS reference reads and the cell a SAME reference
    reads then have the same write-count and the same retrieval-distance distribution, and an
    interference effect that depends on either is common to the two classes.

    QUERIES. ``state`` "which agent does {g} point to at the end?", ``bind`` "which agent does
    {o} belong to at the end?", ``state_all`` the whole pointer map. The two single-slot forms
    are 10 whitespace tokens each. Gates, applied under every dose so a ladder's rungs condition
    on the same items: the queried agent's pointer moved at least twice, ends different from its
    stated target, and last moved inside the final ``q_tail`` of the stream; the queried object
    is written at least twice, ends different, and its resolving write sits in [0.1L, 0.75L].

    COST. The composed cell's cheapest correct algorithm is a single forward pass carrying P and
    B: W = k + m live slots, S = (k + m) + 6 n_swap + 3 n_give + 1 steps under the convention in
    ``factworld.composition`` (P is read FORWARD, so no inverse is needed — unlike v2). The two
    component arms render their second operand by name, which is what makes each of them a
    sparse backward walk over ONE carrier register; the composed cell admits neither walk,
    because no event's operand is known until the other structure has been evaluated forward to
    it. ``factworld.composition.cost_report`` counts all of this against the stated rule.
    """
    k, m = spec.k, spec.n_objects_active
    if m > k:
        raise ValueError(f"{spec.name}: m={m} objects > k={k} agents; the stated holder map "
                         f"must be injective, so n_objects_active <= k.")
    if m > len(w.objects):
        raise ValueError(f"{spec.name}: m={m} exceeds the {len(w.objects)}-object pool "
                         f"(raise n_objects).")
    if spec.query_arm not in ("state", "bind", "state_all"):
        raise ValueError(f"{spec.name}: query_arm={spec.query_arm!r} not in "
                         f"{{'state','bind','state_all'}}")
    agents, objs = list(w.agents[:k]), list(w.objects[:m])

    (P0, B0, events, moves, writes, last_move, last_write, q_state, q_bind,
     fact_agents, fact_objs, finals) = _s5_bind_v3_stream(spec, agents, objs, rng, length, idx)
    P_fin, B_fin = finals[spec.p_cross]

    # Only the structures this arm touches are stated. A component arm is defined by what it
    # does NOT contain, so stating the other map's facts would put a retrieval load on the
    # state component (and a state load on the retrieval one) that the arm never uses.
    need_p = spec.event_kinds != "give" or spec.query_arm in ("state", "state_all")
    need_b = spec.event_kinds != "swap" or spec.query_arm == "bind"
    facts = [r.render_pointer(a, P0[a]) for a in fact_agents] if need_p else []
    facts += [r.render_belongs(o, B0[o]) for o in fact_objs] if need_b else []
    ev_txts, cross_flags = [], []
    for i, e in enumerate(events):
        cross = (not spec.named_operands) and e["u"] < spec.p_cross
        cross_flags.append(cross)
        ev_txts.append(r.render_event(_s5_bind_v3_event(spec, e, cross), step=f"s{i}"))

    if spec.query_arm == "state":
        query = r.render_query("s5bind3_state", target=q_state)
        gold = P_fin[q_state]
    elif spec.query_arm == "bind":
        query = r.render_query("s5bind3_bind", target=q_bind)
        gold = B_fin[q_bind]
    else:
        query = r.render_query("s5bind3_state_all", targets=agents)
        gold = " ".join(P_fin[a] for a in agents)

    meta = {"q_state": q_state, "q_bind": q_bind, "p_cross": spec.p_cross,
            "n_swap": sum(1 for e in events if e["kind"] == "swap"),
            "n_cross": sum(cross_flags),
            "moves": moves.get(q_state), "writes": writes.get(q_bind),
            "last_write_pos": last_write.get(q_bind)}
    if spec.event_trace or spec.worked_trace:
        # Per-EVENT state checkpoints (the whole of P in agent order then B in object order),
        # replayed under this spec's own dose — the supervision density that formed s5 locally.
        P, B = dict(P0), dict(B0)
        snaps, resolved = [], []
        for e, cross in zip(events, cross_flags):
            if spec.named_operands:
                x = e["named"]
            elif e["kind"] == "swap":
                x = B[e["ref_o"]] if cross else P[e["ref_a"]]
            else:
                x = P[e["ref_a"]] if cross else B[e["ref_o"]]
            if e["kind"] == "swap":
                P[e["a"]], P[x] = P[x], P[e["a"]]
            else:
                B[e["o"]] = x
            resolved.append(x)
            snaps.append(" ".join(P[a] for a in agents) + " " + " ".join(B[o] for o in objs))
        meta["trace"] = " ".join(snaps if spec.event_trace else resolved)
        if spec.event_trace:
            meta["interleaved_prompt"] = (
                " ".join(facts) + " " + " ".join(f"{t} {s}" for t, s in zip(ev_txts, snaps))
                + f" {query}")
    return Example(" ".join(facts + ev_txts + [query]), _render_answer(gold), length, meta)


def _ex_conflict(spec, w, r, pmap, rng, length, idx):
    """In-weights ↔ in-context CONFLICT: the prompt states a value for the queried agent that DIFFERS from
    the value the model memorized (`pmap`) during training; the correct answer is the IN-CONTEXT value.
    A model that defaults to its weights answers `pmap[g]` (wrong); one that reads context answers v_ctx.
    `length` = number of facts in the prompt (the queried fact + distractors)."""
    pool = rng.sample(list(w.agents), min(length, len(w.agents)))
    g = rng.choice(pool)                                              # queried agent (NOT positionally fixed)
    v_ctx = rng.choice([v for v in w.value_vocab if v != pmap[g]])     # in-context value, ≠ the memorized one
    ctx = {a: (v_ctx if a == g else rng.choice(list(w.value_vocab))) for a in pool}
    present = pool[:]; rng.shuffle(present)                           # scramble order: no first-position shortcut
    facts = " ".join(r.render_fact(a, "a0", ctx[a], key=f"{a}|{idx}|{rng.random()}") for a in present)
    q = r.render_query("recall", entity=g, attribute="a0")
    return Example(f"{facts} {q}", _render_answer(v_ctx), length,
                   {"param_value": pmap[g], "in_context_value": v_ctx})


def _ex_chain(spec, w, r, rng, depth, idx):
    """Depth-k pointer chase (composition DEPTH is the difficulty axis — the knob that stays hard as
    models scale). A pointer map `nxt` (a single random cycle over `k` agents, resampled per example so
    it is genuinely in-context) is presented as a0-facts pointing agent→agent; the query nests `a0 of`
    `depth` times. Gold = nxt^depth(start). Facts are presented in scrambled order so adjacency does not
    leak the chain, and the cycle has no fixed points so every hop is load-bearing for any depth<k.
    `length` is reinterpreted as the chain DEPTH for this family.

    VALIDITY GATE (enforced): depth >= k wraps the cycle and gold collapses to nxt^(depth mod k)(start)
    — at depth ≡ 0 (mod k) the task degenerates to the identity — so such an example does not measure
    depth. Raises ValueError unless the caller explicitly opts into wrap semantics via
    spec.scaled(chain_allow_wrap=True). The no-wrap protocol for deep chains is spec.scaled(k=depth+2)."""
    if depth >= spec.k and not spec.chain_allow_wrap:
        raise ValueError(
            f"{spec.name}: chain depth {depth} >= k={spec.k} wraps the single {spec.k}-cycle, so gold "
            f"collapses to nxt^({depth} mod {spec.k})(start) and depth is no longer measured (design "
            f"gate: depths stay < k so every hop is load-bearing and the cycle never wraps). For a "
            f"no-wrap deep chain use spec.scaled(k={depth + 2}); to accept wrap semantics explicitly "
            f"use spec.scaled(chain_allow_wrap=True)."
        )
    cyc = rng.sample(list(w.agents), spec.k)                          # the hidden cycle order
    nxt = {cyc[i]: cyc[(i + 1) % len(cyc)] for i in range(len(cyc))}  # single k-cycle (no fixed point)
    present = cyc[:]; rng.shuffle(present)                            # render facts in scrambled order
    facts = " ".join(r.render_fact(a, "a0", nxt[a], key=f"{a}|{idx}|{rng.random()}") for a in present)
    start = rng.choice(cyc)
    path = [start]
    for _ in range(depth):
        path.append(nxt[path[-1]])
    gold = path[-1]
    query = "what is " + "a0 of " * depth + f"{start}?"
    # chain_v2 adds an explicit hop count to avoid models miscounting 128 repetitions
    # of "a0 of" in the nested query (a prompt-format confound at depth ≥ 64).
    if spec.name == "chain_v2":
        query += f" ({depth} hops)"
    meta = {"depth": depth, "start": start, "path": path}
    # Dense supervision: emit every intermediate node (excluding the final answer).
    if spec.worked_trace:
        meta["trace"] = " ".join(path[:-1])
    return Example(f"{facts} {query}", _render_answer(gold), depth, meta)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def generate(spec: TaskSpec, split: str, n: int = 1000, length: int | None = None) -> list[Example]:
    """Deterministic examples. split in {'train','test'}. train mixes train_lengths; test uses `length`
    (one OOD/ID coordinate). Same (spec,split,length,idx) -> identical example, forever."""
    assert split in ("train", "test")
    w, r, oracle = _world(spec)
    # fixed origins sample spec.k values from the value vocab, so only build them where they are USED —
    # a chain scaled to k > value_vocab_size (the no-wrap deep protocol) must not crash here, and neither
    # must a non-memorized recall/composite spec scaled likewise (the breadth-rung protocol
    # scaled(k=2*B, recall_pool=B) reaches k=256 > the 128-value vocab at B=128). _ex_recall/_ex_composite
    # read fixed origins only when spec.memorized_recall; _fixed_origins draws from its OWN rng namespace
    # ("factworld|origins|..."), so skipping it never perturbs any example stream.
    fixed = (_fixed_origins(spec, w)
             if spec.family in ("recall", "composite") and spec.memorized_recall else None)

    if spec.family == "conflict":     # special train protocol: reinforce the in-weights map, then conflict
        pmap = _param_map(spec, w)
        if split == "train":
            out, agents = [], list(w.agents)
            for j in range(n // 2):   # reinforce g→pmap[g] as standalone facts so the model memorizes it
                a = agents[j % len(agents)]
                out.append(Example("", f"{a} a0 {pmap[a]}.", 0, {"reinforce": True}))
            for i in range(n - n // 2):
                L = spec.train_lengths[i % len(spec.train_lengths)]
                out.append(_ex_conflict(spec, w, r, pmap, _rng(spec, "train", L, i), L, i))
            return out
        L = length or spec.eval_lengths[0]
        return [_ex_conflict(spec, w, r, pmap, _rng(spec, "test", L, i), L, i) for i in range(n)]

    lengths = spec.train_lengths if split == "train" else (length or spec.eval_lengths[0],)
    out = []
    for idx in range(n):
        rng = _rng(spec, split, lengths[idx % len(lengths)], idx)
        L = lengths[idx % len(lengths)] if split == "train" else lengths[0]
        if spec.family == "recall":
            out.append(_ex_recall(spec, w, r, fixed, rng, split, L, idx))
        elif spec.family == "binding":
            out.append(_ex_binding(spec, w, r, oracle, rng, L))
        elif spec.family == "composite":
            out.append(_ex_composite(spec, w, r, oracle, fixed, rng, L, idx))
        elif spec.family == "commutative":
            out.append(_ex_commutative(spec, w, r, oracle, rng, L))
        elif spec.family == "s5":
            out.append(_ex_s5(spec, w, r, oracle, rng, L, idx))
        elif spec.family == "s5_chain":
            # typed items are built by a separate function so the untyped stream's sequence of
            # rng draws — and therefore every published s5_chain example — is untouched.
            build = _ex_s5_chain_typed if spec.typed_values else _ex_s5_chain
            out.append(build(spec, w, r, rng, L, idx))               # L = permutation events
        elif spec.family == "s5_bind":
            # source_ablation selects the v3 builder BEFORE any draw, so the temporal builder's
            # streams are untouched.
            build = _ex_s5_bind_v3 if spec.source_ablation else _ex_s5_bind
            out.append(build(spec, w, r, rng, L, idx))               # L = interleaved events
        elif spec.family == "chain":
            out.append(_ex_chain(spec, w, r, rng, L, idx))           # L = chain depth
        else:
            raise ValueError(spec.family)
    return out


def score_exact(pred: str, gold: str) -> int:
    """Position-strict exact match of the full answer span (a diagnostic, NOT the canonical metric).

    ``pred`` is the model's continuation after the prompt; it must match ``gold`` token-for-token over
    gold's length (extra trailing generation is ignored, so '.'-termination is not required of the
    model). The canonical headline metric is ``score_relaxed``; this is reported alongside it to expose
    pure formatting differences (e.g. a chat model emitting ``v56.`` instead of ``v56 .``)."""
    g = gold.split()
    p = pred.split()[:len(g)]
    return int(p == g)


_COMMIT_LEADINS = ("answer", "final", "the", "so", "is", "therefore", "thus", "result",
                   "=", "**answer", "**final")
_EMPHASIS_SPAN = None  # compiled lazily (module import order: re is stdlib, cheap)


def committed_answer(pred: str) -> str:
    """The answer span a multi-line emission COMMITS to — else ``pred`` unchanged.

    Some reasoning endpoints spill their working into the visible completion (hop-by-hop
    traces, map dumps) and state the answer at the END. Prefix scoring then reads working,
    not the commitment: sonnet's xhigh s5_chain cells measured match 0.56 with contains
    0.92 this way. The commitment is located structurally — never by demanding the model
    emit exactly one token (rigid output-format contracts are a repeat source of scoring
    artifacts here). On the last non-empty, non-code-fence line:

      1. If the line reduces to one content token after stripping markdown edges and an
         answer-statement lead-in ("Answer:", "The answer is", ...), that token — covers
         "**g10**", "Answer: g11", and a lone token inside a trailing code fence.
      2. Else, if the line carries markdown emphasis and its LAST emphasized span is one
         content token, that token — covers prose commitments ("... ends at **g15**.").

    A last line with neither shape (map-dump rows, truncated working) commits to nothing
    and the prediction is scored as-is. Both rules apply to single-line predictions too —
    sonnet emits its whole dereference as one line ending in the bolded answer
    ("g5 → g8 → ... → **g7**") — and are inert for clean answers: a bare "g5." reduces to
    its own token, and multi-token spans commit nothing, so the canonical metric is
    unchanged wherever the old behavior was correct. (Local eval never routes here:
    ``extract_commit`` is a reasoning-arm setting.)"""
    import re
    from .render import Renderer
    body = pred.strip()
    lines = [ln for ln in body.splitlines()
             if ln.strip() and not ln.strip().startswith("```")]
    if not lines:
        return pred
    last = lines[-1]
    toks = [t for t in Renderer.normalize(last).split() if t not in (".", ":", ",")]
    while toks and toks[0].lower().rstrip(":") in _COMMIT_LEADINS:
        toks = toks[1:]
    if len(toks) == 1:
        return toks[0]
    spans = re.findall(r"\*\*(.+?)\*\*|`(.+?)`|__(.+?)__", last)
    if spans:
        span = next(s for s in spans[-1] if s)
        stoks = [t for t in Renderer.normalize(span).split() if t not in (".", ":", ",")]
        while stoks and stoks[0].lower().rstrip(":") in _COMMIT_LEADINS:
            stoks = stoks[1:]
        if len(stoks) == 1:
            return stoks[0]
    return pred


def score_relaxed(pred: str, gold: str) -> int:
    """THE canonical metric: whitespace / trailing-period invariant match of the answer span.

    Strips a trailing period from each side and compares the first ``len(gold)`` whitespace tokens.
    This is the fair cross-regime metric: it handles API models that omit the trailing period and local
    models that emit the correct answer and then continue generating, so chat-model tokenizers that
    glue punctuation (``v56.`` vs the atomic ``v56 .``) do not change the score. See
    ``CANONICAL_METRIC``."""
    g = gold.strip().rstrip(".").split()
    p = pred.strip().rstrip(".").split()
    return int(p[: len(g)] == g)


def score_contains(pred: str, gold: str) -> int:
    """Semantic containment: every non-punctuation token in `gold` appears somewhere in `pred`.
    This is intentionally forgiving — it separates whether the model knows the answer from whether
    it guessed the exact output format."""
    g = [t for t in gold.split() if t != "."]
    p = pred.split()
    return int(all(t in p for t in g))


def score_last_n(pred: str, gold: str) -> int:
    """Match the last len(gold) tokens of `pred` to `gold` (ignoring trailing period).
    Handles common chat-model prefixes like 'The answer is ...'."""
    g = gold.strip().rstrip(".").split()
    p = pred.strip().rstrip(".").split()
    if len(p) < len(g):
        return 0
    return int(p[-len(g) :] == g)


def content_tokens(text: str) -> list[str]:
    """Normalized answer tokens with punctuation stripped: the semantic span.

    Used by the composition decomposition (holder leg vs value leg) and by trace
    scoring, so both ignore attached/legacy punctuation.
    """
    from .render import Renderer
    return [t for t in Renderer.normalize(text).split() if t != "."]


def decompose_composite(pred: str, gold: str) -> dict:
    """Per-leg accuracy for a 2-content-token composite answer (holder, value).

    Returns:
        holder_ok: first content token matches (the state-tracking / binding leg)
        value_ok:  second content token matches, among 2-token answers (the recall leg)
        prefix:    longest matching prefix length (0/1/2) — a direct read of where
                   composition breaks (0=neither, 1=holder-only, 2=both)
    """
    g = content_tokens(gold)
    p = content_tokens(pred)
    k = 0
    while k < len(g) and k < len(p) and p[k] == g[k]:
        k += 1
    return {
        "holder_ok": int(len(g) >= 1 and len(p) >= 1 and p[0] == g[0]),
        "value_ok":  int(len(g) >= 2 and len(p) >= 2 and p[1] == g[1]),
        "prefix":    k,
    }


def trace_accuracy(pred_trace: str, gold_trace: str) -> dict:
    """Token-level agreement of a self-generated/unrolled trace against the oracle trajectory.

    Used by the autoregressive experiments (E3): for composite/s5 tasks the gold
    `meta["trace"]` is the oracle's per-step state (holder / role). This scores how
    far a model's self-produced trace follows the correct trajectory.

    Returns:
        token_acc: fraction of gold trace tokens matched at the right position
        first_diverge: index of the first mismatched token (len(gold) if all match)
        full_match: token_acc == 1.0
    """
    g = content_tokens(gold_trace)
    p = content_tokens(pred_trace)
    if not g:
        return {"token_acc": 1.0, "first_diverge": 0, "full_match": True}
    matched = 0
    first_div = len(g)
    for i, gt in enumerate(g):
        pt = p[i] if i < len(p) else None
        if pt == gt:
            matched += 1
        else:
            first_div = min(first_div, i)
            break
    return {
        "token_acc": matched / len(g),
        "first_diverge": first_div,
        "full_match": matched == len(g),
    }


# canonical frozen reference instances (scale via .scaled(...)). `kind` separates scored benchmark tasks
# from controls/experimental tasks (see the kind field: benchmark|control|experimental|retired).
# ONE version per task: superseded specs live in RETIRED below, never here.
CANONICAL = {
    # control: memorized 5-entry map shared train/test -> in-weights lookup, not retrieval. Use as a
    # positive control / floor-check, not a recall score. The honest recall task is recall_copy_v1.
    "recall_v1":        TaskSpec("recall_v1", "recall", k=5, kind="control"),
    # genuine in-context-copy recall (random map, no memorization). `length` grows the distractor POOL
    # (1-of-pool), so eval_lengths is a real recall-extrapolation axis. Like chain_v1 this separates the
    # learnable regime from the cliff: a learnability probe found in-context recall solvable at small pool
    # (pool 5: hybrid 1.00 vs transformer 0.19, §4) but flooring as the pool grows (pool 16 ≈ 0.28, pool 64
    # ≈ 0.01) — a binding-load cliff. So the scored default trains at learnable pools (2–5) and evaluates
    # the binding-load extrapolation (pools 6, 8); the harder large-pool regime is a .scaled(k=…) variant.
    "recall_copy_v1":   TaskSpec("recall_copy_v1", "recall", k=8, memorized_recall=False,
                                 value_vocab_size=64, train_lengths=(2, 3, 4, 5), eval_lengths=(6, 8)),
    # NOTE: last-write-wins binding *is* the delta-rule update, so delta-rule recurrences have a structural
    # prior here — this measures last-write-wins tracking, not a neutral cross-architecture state score.
    # binding_v1 (the recency-defective sampler) is RETIRED; this is the one registered binding task.
    # The v2 give-stream sampler: the resolving write's distance-from-end is ~Uniform and scales with
    # L (L is a genuine binding-depth axis); strong recency sits at ~1/k chance.
    "binding_v2":       TaskSpec("binding_v2", "binding", version="1.1", n_objects_active=4,
                                 last_write_uniform=True),
    # the flagship: genuine 2-hop binding × in-context-copy. BIMODAL at the emergence threshold -> report
    # p(converge) over >=5 seeds, not a mean. composite_copy_v1 (recency-defective sampler) is RETIRED;
    # this is the one registered composition task. The v2 give-stream sampler: the resolving write's
    # distance-from-end is ~Uniform in [1, ~0.9L] so L is a genuine binding-depth axis, and the strong
    # recency heuristic drops to ~1/recall_pool chance (read small-L cells against the residual
    # filter-by-object floor E[1/w] documented on TaskSpec.last_write_uniform, not against 1/pool).
    "composite_copy_v2": TaskSpec("composite_copy_v2", "composite", version="1.1", k=32,
                                  recall_pool=16, memorized_recall=False, value_vocab_size=128,
                                  train_lengths=(4, 8, 16), eval_lengths=(16, 32, 64),
                                  last_write_uniform=True),
    # experimental (v3.1): the COMMUTATIVE rung of the state-tracking ladder — per-entity dial
    # accumulation mod k_positions=5, where EVERY event matters but ORDER does not (retrieval <
    # last-write < commutative < non-abelian). Operating point mirrors binding_v2: chance 1/5,
    # working set m=4 (n_objects_active reused as the active-dial count). Length = aggregation
    # depth (the fold must absorb ~2+(L-2)/4 amounts) — no recency analog exists for a
    # commutative fold. Shallow-adversary floors (initial-only / last-turn-only /
    # entity-blind-sum / count-mod-k) are gated in factworld.validity + scripts/validate_suite.py.
    # kind=experimental until calibrated (local 3-arch + frontier probes), like s5_v1.
    "commutative_v1":   TaskSpec("commutative_v1", "commutative", version="1.1", kind="experimental",
                                 k=5, n_objects_active=4, k_positions=5,
                                 train_lengths=(4, 8, 16), eval_lengths=(16, 32, 64)),
    # experimental: correct non-abelian S5 construct, but not reliably trainable in this harness (answer-only
    # floors in-distribution; worked-trace learns train length but compounds at generation). Needs the
    # dense-per-step regime before it is a scored task. Excluded from REPORTED.
    "s5_v1":            TaskSpec("s5_v1", "s5", k=5, worked_trace=True, kind="experimental",
                                 train_lengths=(8, 16, 32), eval_lengths=(32, 64, 128)),
    # in-weights ↔ in-context CONFLICT: the model memorizes a fixed agent→value map (reinforced in train),
    # then must OVERRIDE it from a contradicting in-context fact. Operationalizes the parametric-vs-in-context
    # axis as a measured construct: answer=in-context value; a weight-defaulting model answers the memorized
    # one. `length` = #facts in the prompt (distractor pool).
    # conflict couples in-context recall with overriding a memorized map, so it inherits recall's
    # binding-load cliff. With focused small-pool training the override IS cleanly solved (gdp 1.00/0.935 at
    # pool 2/3, vs transformer 0.53) and decays as the pool grows (0.40/0.31 at pool 4/5) — a binding-load
    # extrapolation axis. Centered on the learnable edge like recall_copy_v1/chain_v1.
    "conflict_v1":      TaskSpec("conflict_v1", "conflict", k=16, value_vocab_size=64,
                                 train_lengths=(2, 3), eval_lengths=(4, 5)),
    # composition DEPTH: a depth-k pointer chase where `length` is the chain depth — train shallow, eval
    # deeper. Depth is the axis that STAYS hard as models grow (the composition-depth axis). Two knobs are
    # deliberately separated: POOL SIZE k = binding load (a cliff — a learnability probe found k=6 trains
    # to ~1.0 in-distribution at the baseline scale while k=16 floors even in-distribution, like the
    # n_objects_active cliff), and DEPTH = composition (a sharp extrapolation cliff: in-distribution depths
    # solve, depth+1 floors). The scored default fixes binding low (k=6) so depth is read cleanly; the
    # harder, scale-gated k>=16 variant is available via .scaled(k=16). Depths stay < k so the cycle never
    # wraps to a recency shortcut (validity gate confirms majority/recency at floor). ENFORCED: generating
    # at depth >= k raises ValueError (gold collapses to nxt^(depth mod k)); deep chains must use the
    # no-wrap protocol .scaled(k=depth+2), or opt into wrap explicitly via .scaled(chain_allow_wrap=True).
    # chain_v2: same pointer-chase task as chain_v1, but the query appends an explicit
    # hop count (e.g. "... of g246? (128 hops)") to remove the depth-counting confound
    # that caused models to miscount 128 nested "a0 of" phrases.
    "chain_v2":         TaskSpec("chain_v2", "chain", version="1.1", k=6, train_lengths=(2, 3), eval_lengths=(4, 5)),
    # s5_chain: THE composite stressor — order-sensitive events on the a0 pointer map
    # (non-abelian state tracking), followed by a chain_depth-hop serial dereference query
    # over the final map. length = number of permutation events; chain_depth = hops (kept < k).
    # v4 is the single scored spec. Two gates make it one computation rather than two:
    #
    #   distinct_path — every item's query path visits depth+1 distinct agents, so echo and
    #   every fixed-hop heuristic score exactly 0 and item difficulty is uniform. (The v1/v2
    #   streams admitted an echo floor of 0.16-0.32 because final-map cycles whose length
    #   divides the depth make the start its own answer; see RETIRED.)
    #
    #   conditional_rate — a quarter of the events name their second operand by reference to
    #   the running map, which is what stops the query being answerable by pushing one symbol
    #   backward through the event list (the v3 defect; see RETIRED). Measured on this
    #   stream, n=2000 per cell: the last referenced event sits at 0.909 / 0.953 / 0.969 /
    #   0.977 of the stream at L=32/64/96/128, so essentially the whole map has to be
    #   evaluated forward before any backward walk can start. Rate 0.5 raises that to
    #   0.968/0.985/0.990/0.992 — a hundredth of the stream, for twice the reference
    #   sentences; rate 0.125 leaves the initial-ref adversary at 0.146 (4.7x chance) at L=32,
    #   where four references in a barely-drifted map mostly resolve correctly.
    #
    # Operating point: k=32 with depth 16 against v3's k=16/depth 8. Depth costs nothing in
    # prompt tokens (46 chars per 8 extra hops at L=96) and breadth costs one fact sentence
    # per agent (+4.5% at L=96), while L costs a full event sentence per unit — so difficulty
    # goes on the two cheap axes and the length grid is v3's. The distinct_path gate needs a
    # final-map cycle longer than the depth, whose acceptance rate depends only on the ratio:
    # 0.65-0.69 of first draws at depth/k = 1/2 for every k in {16, 24, 32, 48} (about 1.5
    # event draws per item, against the 100 the builder allows), falling to 0.44 at depth 20
    # and 0.28 at depth 24. Conditioning on it leaves every prompt-visible statistic alone: gold,
    # start and event-operand distributions are uniform to within finite-sample bias (KL excess
    # <= 0.0013 nats against the suite's 0.02 threshold), the event mix is the drawn
    # 0.25/0.375/0.375, and references are uniform across stream quarters — all identical to
    # the same sampler with the gate off. Chance is 1/31 = 0.0323, the answer space being the
    # non-start agents; the operative floor — the max over the registered shallow adversaries
    # (initial-map chase, initial-ref resolution, echo, and the two chance rows;
    # validity.S5_CHAIN_ADVERSARIES) — is 0.0398 at L=32 and 0.0323-0.0334 at L=64/96/128 at
    # n=5000, so no shallow policy reaches 1.25x chance at any scored length, let alone the 2x
    # the suite gates on. The row supplying L=32 is initial-ref resolution, which four events of
    # drift is all a reference has to survive.
    #
    # LENGTH GRID. L=32 carries the highest floor of the four and stays scored, but not on the
    # strength of the gap: the initial-map rows are members of a fixed-offset partition whose
    # k-1 accuracies sum to exactly 1 (validity, the s5_chain adversary block), so comparing a
    # per-length max across lengths compares selection draws — against the full 31-member family
    # the L=32 max exceeds the longer lengths' by 0.0020, not by the 0.0068 the registered rows
    # suggest. What keeps L=32 is the gate margin: its 0.0398 is 1.23x chance against a 2x bound,
    # so the shortest scored length is as far from being shallow-solvable as the rest of the grid.
    "s5_chain_v4":      TaskSpec("s5_chain_v4", "s5_chain", version="2.1", k=32, chain_depth=16,
                                  distinct_path=True, conditional_rate=0.25,
                                  train_lengths=(8, 16), eval_lengths=(32, 64, 96)),
    # Local calibration variants (experimental, never scored as the frontier task).
    # local_v4: the scored construct at a from-scratch operating point — the same forward
    # evaluation, at k=8 with a 2-hop dereference and per-EVENT map checkpoints. The rate and
    # the lengths are set by the floors, not copied from the frontier spec: a reference is
    # mis-resolvable only once the map has drifted at the referenced slot, so at short
    # streams the initial-ref adversary is the operative floor (n=2000, chance 0.125: 0.519
    # at L=8 and 0.245 at L=16 for rate 0.25). At rate 0.5 it falls to 0.101 at L=16 and
    # 0.110 at L=24, below the initial-map chase, which leaves the operative floor at 0.143
    # (chance, 1/(k-1)) at L=16 and 0.156 (the chase) at L=24 — the same two rows that supply
    # it across the local v2 family, so the two families' cells are read against floors of the
    # same kind.
    # local_v4_path: same items, path-only trace — the supervision-density contrast arm.
    "s5_chain_local_v4": TaskSpec("s5_chain_local_v4", "s5_chain", version="2.1", k=8,
                                   chain_depth=2, distinct_path=True, conditional_rate=0.5,
                                   event_trace=True, worked_trace=True,
                                   train_lengths=(8, 16), eval_lengths=(16, 24),
                                   kind="experimental"),
    "s5_chain_local_v4_path": TaskSpec("s5_chain_local_v4_path", "s5_chain", version="2.1", k=8,
                                        chain_depth=2, distinct_path=True, conditional_rate=0.5,
                                        worked_trace=True,
                                        train_lengths=(8, 16), eval_lengths=(16, 24),
                                        kind="experimental"),
    # local_v2: gated items + per-EVENT map checkpoints (event_trace) — the dense per-step
    # supervision that formed s5 locally, which the retired local_v1 lacked (its path-only
    # trace supervised the dereference but not the map tracking through events).
    # local_v2_path: same gated items, path-only trace — the supervision-density contrast arm.
    "s5_chain_local_v2": TaskSpec("s5_chain_local_v2", "s5_chain", version="2.0", k=8, chain_depth=2,
                                   distinct_path=True, event_trace=True, worked_trace=True,
                                   train_lengths=(2, 4), eval_lengths=(4, 8), kind="experimental"),
    "s5_chain_local_v2_path": TaskSpec("s5_chain_local_v2_path", "s5_chain", version="2.0", k=8,
                                        chain_depth=2, distinct_path=True, worked_trace=True,
                                        train_lengths=(2, 4), eval_lengths=(4, 8), kind="experimental"),
    # experimental: the TYPED-VALUE ablation against s5_chain_local_v2 at chain_depth=1. The a0
    # map sends agents to ROLES, so a token in value position can never appear in slot position;
    # this probes key/value type ambiguity — the structural difference between s5 (which forms
    # locally) and s5_chain (which does not) that is not composition depth. Three spec fields
    # differ from the untyped arm — name, typed_values and distinct_path — and the builders
    # differ in initial-map structure; _ex_s5_chain_typed documents all of it and
    # tests/test_s5_chain.py pins the field-level diff. Read each arm against its own operative
    # floor (validity.operative_floor), never against 1/k or against the other arm's floor.
    "s5_chain_typed_v1": TaskSpec("s5_chain_typed_v1", "s5_chain", version="2.0", k=8,
                                   chain_depth=1, typed_values=True,
                                   event_trace=True, worked_trace=True,
                                   train_lengths=(2, 4), eval_lengths=(4, 8), kind="experimental"),
    # ---- s5_bind_v3: the source-structure composition (see _ex_s5_bind_v3) ---------------
    # The COMPOSED task the instrument's goal asks for, plus its two components, on one basis.
    # Two maps into agents run over one event stream — P: agents -> agents, rewritten by swaps,
    # and B: objects -> agents under last-write-wins, rewritten by gives — and every event
    # names its second operand LIVE through one of them. The ablation is which structure the
    # reference reads (TaskSpec.source_ablation), not when it reads it.
    #
    #   s5_bind_v3        COMPOSED   p_cross=0.5, state query. Both structures feed each other.
    #   s5_bind_v3_state  COMPONENT  swaps only, second operand NAMED: the S5 word problem.
    #   s5_bind_v3_bind   COMPONENT  gives only, recipient NAMED: last-write-wins retrieval.
    #
    # The two components are their own streams: a component is defined by what it does NOT
    # contain, so there is no skeleton it could share with the composed cell. The composition
    # contrast does not need one — it is an op-type contrast INSIDE the composed cell.
    #
    # p_swap = 1/3 IS THE WRITE-COUNT MATCHING. A swap moves two agents' pointers and a give
    # writes one object, so at m = k the two structures' cells accumulate writes at equal rates
    # exactly at p_swap = 1/3. That is what makes theta_cross - theta_same a contrast between
    # two op classes at matched read-history load rather than between two structures at
    # different ones (factworld.composition).
    #
    # THE COMPONENT GRIDS ARE WORK-MATCHED TO THE COMPOSED ONE. A composed stream of length L
    # holds p_swap L swaps and (1 - p_swap) L gives, so a component read at the composed cell's
    # own L is doing 1/p_swap (state) or 1/(1 - p_swap) (retrieval) times the work: composed@48
    # contains 17 swaps and 31 gives against a component@48's 48 of its own kind. Each
    # component's eval_lengths are therefore the composed cell's own event counts — (17, 23, 34)
    # and (31, 41, 62) at k=6 against composed (48, 64, 96), (43, 64, 85) and (85, 128, 171) at
    # k=12 against composed (128, 192, 256) — which equalises the carrier chain
    # (validity.s5_bind_v3_carrier_hops: 5.67 hops at composed@48 and at state@17) and the write
    # count in the leg being compared. The state component's grid also carries the rungs above
    # its work-matched ones, because it is the leg with a depth axis at all. The TOKEN-matched
    # pairing is kept beside this one as the matched-COST control and the two give different
    # multipliers: at k=6/L=48 the composed cell costs 3.83x the state component's forward pass
    # at equal WORK and 1.00x at equal TOKENS, against the 1.65x that reading both at L=48
    # reports (scripts/protocol_s5bind_v3_three_cell_20260731.py, step_multipliers).
    #
    # RAISING p_swap IS NOT AN ALTERNATIVE TO THAT PAIRING, and none is registered. No p_swap
    # makes the two legs' lengths equal — both are strictly under L — so the pairing is needed at
    # every p_swap; what p_swap moves is the composed cell's own depth per token, and that price
    # is measured. Over p_swap = 1/3 .. 2/3 at k=6 the operative floor does not move (1.00-1.11x
    # informed chance at L=48 and L=96) and the CROSS/SAME class balance does not move
    # (0.49-0.51), but the WITHIN-KIND retrieval-distance gap between the two candidate pools —
    # what the matched draw and the write-count matching exist to close — opens from 1.3% (swap)
    # and 12% (give) at L=96 to 35% and 62%, and the B leg thins from 6.2 to 2.8 minimum writes
    # per object. What it buys is 126 -> 72 prompt tokens per carrier hop. Depth per token is
    # available; matched read history is the price.
    #
    # THE RETRIEVAL COMPONENT IS A GATE AND NOT A GRADED CELL, at every setting its spec can
    # take. Swept over k in {6, 8, 12, 16}, m in {2..16} and L in {16, 48, 96}: the cell's own
    # algorithm has composition depth 1 and W = 2 live slots at EVERY point and the operative
    # floor is informed chance at every point, on the 'chance' basis. What moves is the answer
    # space (0.200 -> 0.067), the scan the sampler's window forces (4.2 -> 35.3 events) and the
    # writes the queried object takes (2.3 -> 18.2) — recall difficulty, not composition depth.
    # Below m = 4 the sampler cannot fill the window (m = 2 fails at L = 48, m = 3 at L = 96) and
    # m = 4 / L = 96 costs 224 stream restarts per item. The from-scratch arm reads the cell
    # 1.000 at every length to L = 132 and on every seed, so it is registered as the gate the
    # protocol's positive control uses and never as a difficulty axis.
    #
    # THE STATE QUERY IS THE ONLY ONE THAT REQUIRES BOTH STRUCTURES OVER THE WHOLE STREAM, and
    # the reason is a conflict inside the sampler rather than a preference. NO RETRIEVAL-QUERY
    # COMPOSED ARM IS REGISTERED; what was measured is below.
    #
    # The bind query does need both maps. On the composed stream read with a bind query BOTH
    # one-structure replays fall to informed chance — a P-only solver reads 0.217 and a B-only
    # one 0.203 against 0.200 at k=6/L=96, and 0.107 / 0.103 against 0.0909 at k=12/L=256 — just
    # as they do on the state query (0.243/0.190/0.210 and 0.173/0.157/0.187 at k=6/L=48/64/96).
    # The B-only read decays with length (0.370/0.297/0.257/0.203 at k=6/L=48/64/80/96 and
    # 0.303/0.170/0.103 at k=12/L=128/192/256) and ``one_structure_B`` is ADMITTED on a bind
    # query — it holds m + 1 slots there, not k + m + 1 — so it would set that cell's floor.
    #
    # WHAT DISQUALIFIES IT IS THE TAIL. The queried object's resolving write is pinned into
    # [0.1L, 0.75L], and that pin is exactly what PROVES the retrieval component's floor: no
    # bounded backward scan reaches it. On a bind query the same pin means no event after 0.75L
    # can move the answer, so a solver carrying both maps and replaying only the first 90% of the
    # stream scores 1.000 and one replaying the first 75% scores 0.927 (k=6) / 0.958 (k=12),
    # against 0.097 and 0.143 for the state query on the same cell — the state query's own gate
    # puts the queried agent's last move inside the final 10%. A floor-PROVED retrieval component
    # needs the resolving write far from the end; a query that depends on the whole stream needs
    # it near the end. One sampler cannot do both, and scripts/validate_suite.py flags the
    # bind-query arm on exactly that row (prefix_90).
    #
    # Operating point k = 12, m = 12: the answer space is the 12 agents, informed chance
    # 1/(k-1) = 0.0909. Floors are recomputed per cell by scripts/validate_suite.py from
    # factworld.validity.s5_bind_v3_floors, under the one-structure class rule.
    #
    # THE FLOOR IS A PROFILE, NOT A NUMBER, and the length grid is cut so the admitted end of it
    # sits at informed chance. The composed cells are measured at n=500 on the scored items
    # (validity.s5_bind_v3_*, scripts/probe_s5bind_v3_floor_20260731.py) and the component cells
    # at n=4000, because the max over admitted rows carries an upward selection bias at small n:
    # the state component's ``last_write_1hop`` reads 1.30x at n=500 and 0.98x at n=4000, and the
    # published 1.30x was a high draw. As ratios to informed chance 1/(k-1):
    #
    #   cell                        L   operative floor   what sets it
    #   s5_bind_v3                256       1.09x        uniform_anti_surface, the gate's price
    #   s5_bind_local_v3           96       1.05x        one_structure_P and last_swap_ref, the
    #                                                    admitted one-structure and surface rows
    #   s5_bind_v3_state           85       1.04x        informed chance: last_write_1hop, the
    #   s5_bind_local_v3_state    128       1.00x        carrier walk cut after one hop, against
    #                                                    the state-free surface read. The k=6
    #                                                    grid carries the rungs above its
    #                                                    work-matched ones (48, 80, 128 = 16.0,
    #                                                    26.7, 42.7 hops) because that is the
    #                                                    leg with a depth axis; the k=12 grid is
    #                                                    a gate and stops at the pairing
    #   s5_bind_v3_bind           171       1.00x        informed chance, PROVED: the sampler
    #   s5_bind_local_v3_bind      62       1.00x        pins the queried object's resolving
    #                                                    write out of reach of every admitted
    #                                                    budget, and each reads exactly 0.000
    #
    # The state component's own grid is cut at L = 12 rather than lower: its floor is 1.26x at
    # L = 8 (2.7 hops), where a one-hop read still has traction, and 1.00-1.09x from L = 12
    # (4.0 hops) to L = 256 (85.3). Its cost is flat over that whole range — 1.00-1.01 stream
    # restarts per item — so the grid is cut by the floor and not by the sampler.
    #
    # THE FITTED 25-FEATURE SURFACE RANKER IS NOT IN THOSE NUMBERS. It is measured beside them —
    # 1.04x at k=12/L=256 and 1.24x at k=6/L=96, on 4000 fit / 4000 held-out items, with a
    # block-to-block spread of 0.000-0.013 between two disjoint 2000-item fits — but no
    # implementation of it achieves a price the class rule admits, because six of its features
    # are per-candidate accumulators: one pass over the k candidates costs W = 1 + 7k live slots
    # and the register-lean implementation costs k passes, S = 2kL
    # (validity.s5_bind_v3_surface_impls). It is a DIAGNOSTIC of what the state-free surface
    # information supports, not a bound on what a cheap policy extracts. Pricing it W = 2 put the
    # k=6 composed floor at 1.14x; the corrected number is 1.05x, and the k=12 composed floor
    # does not move because the ranker never set it there. Its held-out accuracy is still
    # climbing below ~1000 fit items, so the fit budget is registered at 2000 per block.
    #
    # What the class rule EXCLUDES is measured beside it, since the exclusion is a cost argument.
    # The partial-carry family (carry P and j of the m holder cells, W = k+j+1) climbs to 6.12x
    # at j=11 at k=12/L=128 and 3.17x at j=5 at k=6/L=48; the block-drop family (both maps, one
    # block dropped) reaches 9.57x at a 0.01L block and decays to chance by 0.25L. Both sit above
    # the one-structure bound and neither is a floor. On the component cells the excluded family
    # is the cell's own algorithm truncated: the carrier walk with c events left unread reads
    # 9.28x / 7.62x / 6.41x / 4.74x / 2.79x / 1.31x / 0.96x at c = 1 2 3 5 9 17 33 (k=12, L=256)
    # and the truncated give-scan jumps from 0.000 to a resolved fraction the moment its budget
    # reaches the sampler's window. Both are excluded by the component rule — the first on depth,
    # the second on the algorithm's per-item minimum cost — and neither is a floor.
    #
    # The k=12 COMPOSED grid starts at 128 and the k=6 one at 48 because below that a short
    # stream against k gives the queried agent too few carrier events for the second leg to
    # matter. The step multiplier holds flat across each grid under every convention, so the cut
    # is a floor cut and not a cost cut. On forward-pass tokens against the state component it is
    # 1.65/1.65/1.65 reading both cells at the composed cell's own length, 3.83/3.96/4.17 at
    # equal WORK and 1.04/1.02/1.01 at equal tokens (k=6, L=48/64/96); against the retrieval
    # component 2.46/2.51/2.54, 3.48/3.63/3.73 and 1.01/—/— . At k=12 the same three columns are
    # 1.65/1.64/1.64 and 4.21/4.41/4.54 against the state component and 2.51/2.54/2.55 and
    # 3.55/3.64/3.70 against the retrieval one. Quoting one of the three as "the step multiplier"
    # is what hides the pairing it was computed under.
    #
    # kind=experimental until the calibration lands, so none of the six is in REPORTED.
    # q_no_surface IS SET AT k=12 AND NOT AT k=6, and the reason is measured on both. The gate
    # empties the state-free surface read (TaskSpec.q_no_surface) and hands a guesser the mass
    # that read used to carry, which the closed-form ``uniform_anti_surface`` row prices. At
    # k=12 that trade is bought — the operative floor falls from 1.11x/1.18x chance at L=128/256
    # to 1.09x — and at k=6 it is not: the ungated row reads 1.01x/1.02x there, because the
    # CROSS branch sits below chance and drags the whole-item policy down, while striking one of
    # five candidate answers is worth 1.20x on its own. Retention is not what decides it; the
    # gate costs 1.027 -> 1.030 item restarts at k=12/L=128 and 2.403 -> 2.442 at k=6/L=96.
    "s5_bind_v3":       TaskSpec("s5_bind_v3", "s5_bind", version="3.0", kind="experimental",
                                  source_ablation=True, k=12, n_objects=12, n_objects_active=12,
                                  p_swap=1.0 / 3.0, p_cross=0.5,
                                  query_arm="state", stream_name="s5_bind_v3",
                                  no_pin=True, q_tail=0.1, q_no_surface=True, match_reads=1,
                                  train_lengths=(16, 32), eval_lengths=(128, 192, 256)),
    "s5_bind_v3_state": TaskSpec("s5_bind_v3_state", "s5_bind", version="3.0", kind="experimental",
                                  source_ablation=True, k=12, n_objects=12, n_objects_active=12,
                                  event_kinds="swap", named_operands=True,
                                  query_arm="state", stream_name="s5_bind_v3_state", q_tail=0.1,
                                  train_lengths=(16, 32), eval_lengths=(43, 64, 85)),
    "s5_bind_v3_bind":  TaskSpec("s5_bind_v3_bind", "s5_bind", version="3.0", kind="experimental",
                                  source_ablation=True, k=12, n_objects=12, n_objects_active=12,
                                  event_kinds="give", named_operands=True,
                                  query_arm="bind", stream_name="s5_bind_v3_bind",
                                  train_lengths=(16, 32), eval_lengths=(85, 128, 171)),
    # The from-scratch operating point: k=6, m=6, shorter streams, and per-EVENT state
    # checkpoints (event_trace — the whole of P then B after every event, the supervision
    # density that formed s5 locally). A streaming model has no scratchpad, so its cost model
    # IS the forward pass, which is the algorithm this construct forces in both regimes.
    "s5_bind_local_v3": TaskSpec("s5_bind_local_v3", "s5_bind", version="3.0", kind="experimental",
                                  source_ablation=True, k=6, n_objects=6, n_objects_active=6,
                                  p_swap=1.0 / 3.0, p_cross=0.5,
                                  query_arm="state", stream_name="s5_bind_local_v3",
                                  event_trace=True, no_pin=True, q_tail=0.1, match_reads=2,
                                  train_lengths=(16, 32), eval_lengths=(48, 64, 96)),
    "s5_bind_local_v3_state": TaskSpec("s5_bind_local_v3_state", "s5_bind", version="3.0",
                                  kind="experimental",
                                  source_ablation=True, k=6, n_objects=6, n_objects_active=6,
                                  event_kinds="swap", named_operands=True, event_trace=True,
                                  query_arm="state", stream_name="s5_bind_local_v3_state",
                                  q_tail=0.1,
                                  train_lengths=(16, 32),
                                  eval_lengths=(17, 23, 34, 48, 80, 128)),
    "s5_bind_local_v3_bind": TaskSpec("s5_bind_local_v3_bind", "s5_bind", version="3.0",
                                  kind="experimental",
                                  source_ablation=True, k=6, n_objects=6, n_objects_active=6,
                                  event_kinds="give", named_operands=True, event_trace=True,
                                  query_arm="bind", stream_name="s5_bind_local_v3_bind",
                                  train_lengths=(16, 32), eval_lengths=(31, 41, 62)),
}

# the scored benchmark set (controls + experimental tasks excluded from headline reporting)
REPORTED = tuple(name for name, spec in CANONICAL.items() if spec.kind == "benchmark")

# RETIRED specs (issue #11, owner decision 2026-07-09: kill v1 — one clean version per task in the
# scored registry). These are the recency-defective v1 give-stream family: the v1 sampler draws every
# event's object uniformly from the active set, so the queried object's resolving (last) write sits
# ~Geometric(1/m) events from the stream END at every L (median distance 1 at L16). The one-line
# STRONG recency heuristic ("last give-event's recipient" [+ that holder's stated a0 fact]) therefore
# scores far above chance — measured ~0.34@L16 / 0.21@L64 on composite_copy_v1 (chance ~1/16) and
# ~0.4 on binding_v1 (chance 1/5) — and the L axis measures distractor volume, not binding depth.
# Superseded by the last_write_uniform v2 specs in CANONICAL (binding_v2 / composite_copy_v2), which
# hold that heuristic at ~chance.
#
# RETIRED specs are NEVER scored: kind='retired' keeps them out of REPORTED, scripts/validate_suite.py
# skips them (the known-shortcut annotation above replaces its per-task exemption), and the scored
# registry lookups (eval CLIs, the frontier benchmark) resolve CANONICAL names. They remain here —
# frozen and byte-identical (version pinned at _STREAM_V1; goldens in tests/goldens_prechange.json /
# tests/test_composite_v2.py) — ONLY for historical reproduction of published results and for the
# defect-documentation tests that pin the v1-vs-v2 shortcut contrast.
RETIRED = {
    # last-write-wins binding under the defective v1 sampler (published binding numbers reference it).
    "binding_v1":       TaskSpec("binding_v1", "binding", n_objects_active=4, kind="retired"),
    # binding under a LARGER working set (m=8 active objects; interference-cliff probe). Was
    # kind=experimental (never a score); retired with the rest of the v1 sampler family.
    "binding_load_v1":  TaskSpec("binding_load_v1", "binding", n_objects_active=8, kind="retired"),
    # control: recall leg = the memorized 5-map, isolating the BINDING leg (recall saturated). Was
    # kind=control; the binding leg still came from the defective v1 stream.
    "composite_v1":     TaskSpec("composite_v1", "composite", k=5, memorized_recall=True, kind="retired"),
    # the pre-fix flagship: 2-hop binding × in-context-copy under the v1 sampler (the published
    # composite grid/benchmark columns). Superseded knob-for-knob by composite_copy_v2.
    "composite_copy_v1": TaskSpec("composite_copy_v1", "composite", k=32, recall_pool=16,
                                  memorized_recall=False, value_vocab_size=128, kind="retired",
                                  train_lengths=(4, 8, 16), eval_lengths=(16, 32, 64)),
    # the EXACT §5 scale-experiment difficulty point (k=5, 1-of-5 recall so a composite floor is
    # attributable to composition, not recall capacity): what scale_wall2.py/scale_confirm.py ran.
    "composite_copy_scale_v1": TaskSpec("composite_copy_scale_v1", "composite", k=5, recall_pool=5,
                                        memorized_recall=False, value_vocab_size=128, kind="retired",
                                        train_lengths=(4, 8, 16), eval_lengths=(16, 64)),
    # s5_chain v1/v2 (issue #30 follow-up, retired 2026-07-18): both streams lack the
    # distinct_path gate, so final-map cycles whose length divides chain_depth=8 make the
    # degenerate echo strategy ("answer the queried agent") score 0.16-0.32 vs 1/16 chance,
    # and roughly half the items have degenerate (<9 distinct-agent) query paths — item
    # difficulty varies and sub-0.4 scores are uninterpretable. v1 additionally used the
    # pre-explicit event rendering (its pilot scores sat AT the echo floor). Superseded
    # knob-for-knob by s5_chain_v3 (distinct_path=True, version 2.0). Note the shared
    # cycle_a0 wording was made simultaneity-explicit on 2026-07-18 (render.py _CYCLE_A0),
    # so regenerated v1/v2 prompts are NOT byte-identical to the published runs; the runs
    # are preserved in results/benchmark/history.jsonl.
    "s5_chain_v1":      TaskSpec("s5_chain_v1", "s5_chain", version="1.1", k=16, chain_depth=8,
                                  train_lengths=(8, 16), eval_lengths=(32, 64), kind="retired"),
    "s5_chain_v2":      TaskSpec("s5_chain_v2", "s5_chain", version="1.3", k=16, chain_depth=8,
                                  train_lengths=(8, 16), eval_lengths=(32, 64, 96), kind="retired"),
    # s5_chain v3 (issue #37, retired 2026-07-27): the events permute the DOMAIN of the
    # pointer map — a swap of a and b sets f'(a)=f(b) and f'(b)=f(a), i.e. f' = f∘(a b) — so
    # after L events f_L = f_0∘σ_1∘…∘σ_L and f_L(x) = f_0(σ_1(…σ_L(x))). Pushing ONE symbol
    # backward through the event list and then applying the stated initial map answers the
    # query exactly: 1.000 on the v3 test stream at L96 (and on v1/v2/local_v2), carrying
    # log2(k) = 4 bits per hop rather than the log2(16!) = 44 of an S16 permutation. The
    # instrument-level defect is what that algorithm needs: the walk runs the event list
    # backward, which an attention model over the full context can do and a streaming
    # recurrent model cannot, so the two regimes the task exists to bridge were not running
    # the same computation. Superseded knob-for-knob by s5_chain_v4, where a fraction of the
    # events name an operand by reference to the running map, no event's identity is fixed
    # until the map has been evaluated forward to it, and both regimes run the forward pass.
    "s5_chain_v3":      TaskSpec("s5_chain_v3", "s5_chain", version="2.0", k=16, chain_depth=8,
                                  distinct_path=True, kind="retired",
                                  train_lengths=(8, 16), eval_lengths=(32, 64, 96)),
    # Ungated local pilot with path-only traces; superseded by s5_chain_local_v2[/_path].
    "s5_chain_local_v1": TaskSpec("s5_chain_local_v1", "s5_chain", version="1.3", k=8, chain_depth=2,
                                   train_lengths=(2, 4), eval_lengths=(4, 8), worked_trace=True,
                                   kind="retired"),
    # Original chain pointer-chase. Retired because the nested "a0 of a0 of ..." query becomes a
    # hop-counting confound at depth 64/128; superseded by chain_v2, which appends an explicit
    # depth annotation ("(128 hops)") to the same query.
    "chain_v1":         TaskSpec("chain_v1", "chain", k=6, train_lengths=(2, 3), eval_lengths=(4, 5),
                                  kind="retired"),
    # ---- the s5_bind TEMPORAL family (retired 2026-07-31) ------------------------------
    # Two structures over one stream, each event naming its second operand through the other,
    # ablated on the TIME INDEX: "at this point" resolves against the running map, "at the
    # start" against the stated one. RETIRED because THE ABLATION CANNOT IDENTIFY COMPOSITION.
    # The stated structure is by definition the one before any write, so a reference witnesses
    # composition exactly when its referenced cell has been written since the start — measured
    # on these streams, P(coupled reading != decoupled reading | write count of the read cell
    # = 0) = 0.000 and = 1.000 at one write, over 16,386/31,276 and 8,825/19,933 dependency-slice
    # resolutions at the two scored cells. The composition class IS the overwritten-cell class,
    # so a composition-free bounded-capacity solver (M = k/2 slots, LRU by last write, a miss
    # re-reads the stated fact, no composition deficit anywhere in it) rejects the op-type
    # contrast at 0.180/0.657/0.623 where the real deficit's power is 0.152/0.520/0.526, and a
    # stale-to-previous-value solver inflates it to 0.260. The one contrast clean of read
    # history — dynamic-but-unmoved references against static ones — has exactly zero power.
    # Superseded by the SOURCE-STRUCTURE family (s5_bind_v3), where every reference is live and
    # only the structure it reads varies.
    #
    # These specs were never scored and never published, and they no longer carry the
    # ``chain_max_gap`` steer, which was removed with the field (it could not close the
    # block-drop family and it charged the construct's own comparison — the steer follows the
    # coupled trajectory only, so the coupled arm's carrier chain ran 11.8 -> 27.6 events while
    # the decoupled reading of the same stream stayed at 12.2). Their streams are therefore
    # pinned at retirement, not at their pre-retirement values; nothing reproduces against them.
    "s5_bind_v2":       TaskSpec("s5_bind_v2", "s5_bind", kind="retired",
                                  k=12, n_objects=12, n_objects_active=12,
                                  coupled=True, query_arm="state", stream_name="s5_bind_v2",
                                  no_pin=True, q_tail=0.1,
                                  train_lengths=(16, 32), eval_lengths=(128, 192, 256)),
    "s5_bind_v2_state": TaskSpec("s5_bind_v2_state", "s5_bind", kind="retired",
                                  k=12, n_objects=12, n_objects_active=12,
                                  coupled=False, query_arm="state", stream_name="s5_bind_v2",
                                  no_pin=True, q_tail=0.1,
                                  train_lengths=(16, 32), eval_lengths=(128, 192, 256)),
    "s5_bind_v2_bind":  TaskSpec("s5_bind_v2_bind", "s5_bind", kind="retired",
                                  k=12, n_objects=12, n_objects_active=12,
                                  coupled=False, query_arm="bind", stream_name="s5_bind_v2",
                                  no_pin=True, q_tail=0.1,
                                  train_lengths=(16, 32), eval_lengths=(128, 192, 256)),
    "s5_bind_v2_map":   TaskSpec("s5_bind_v2_map", "s5_bind", kind="retired",
                                  k=12, n_objects=12, n_objects_active=12,
                                  coupled=False, query_arm="state_all", stream_name="s5_bind_v2",
                                  no_pin=True, q_tail=0.1,
                                  train_lengths=(16, 32), eval_lengths=(128, 192, 256)),
    "s5_bind_local_v2": TaskSpec("s5_bind_local_v2", "s5_bind", kind="retired",
                                  k=6, n_objects=6, n_objects_active=6,
                                  coupled=True, query_arm="state",
                                  stream_name="s5_bind_local_v2", event_trace=True,
                                  no_pin=True, q_tail=0.1,
                                  train_lengths=(16, 32), eval_lengths=(48, 64)),
    "s5_bind_local_v2_state": TaskSpec("s5_bind_local_v2_state", "s5_bind", kind="retired",
                                        k=6, n_objects=6, n_objects_active=6,
                                        coupled=False, query_arm="state",
                                        stream_name="s5_bind_local_v2", event_trace=True,
                                        no_pin=True, q_tail=0.1,
                                        train_lengths=(16, 32), eval_lengths=(48, 64)),
    # The temporal family's coupling-DOSE ladder: rho in {0, .25, .5, .75, 1} on one skeleton.
    # Retired with the family it calibrates — a dose in the temporal reading is a dose of
    # "resolve against the running map", i.e. of the read-history predicate, so the ladder
    # measures the same confound at five strengths.
    **{f"s5_bind_v2_lad{int(r * 100):02d}":
       TaskSpec(f"s5_bind_v2_lad{int(r * 100):02d}", "s5_bind", kind="retired",
                k=12, n_objects=12, n_objects_active=12, no_pin=True, q_tail=0.1,
                rho_p=r, rho_b=r, rho_ladder=(0.0, 0.25, 0.5, 0.75, 1.0),
                coupled=True, query_arm="state", stream_name="s5_bind_v2_lad",
                train_lengths=(16, 32), eval_lengths=(192,))
       for r in (0.0, 0.25, 0.5, 0.75, 1.0)},
}

# CALIBRATION specs: cells that measure how a construct behaves rather than how a model scores.
# They are generable and named like any other task, and they are NOT scored — nothing in
# REPORTED, nothing in the benchmark roster, and outside the CANONICAL validity gate, which
# fails a cell whose strongest registered shallow policy reads 0.5 or more (a calibration cell
# is allowed to be mostly floor; that is often the thing being measured).
CALIBRATION: dict[str, TaskSpec] = {}


def spec_for(name: str) -> TaskSpec:
    """Resolve a task name against CANONICAL, falling back to CALIBRATION and then RETIRED.

    The RETIRED fallback exists ONLY so historical runs remain reproducible (retired specs
    generate byte-identically forever); CALIBRATION cells measure a construct and are never
    scored. Anything reporting a score should stick to CANONICAL names.
    """
    if name in CANONICAL:
        return CANONICAL[name]
    if name in CALIBRATION:
        return CALIBRATION[name]
    if name in RETIRED:
        return RETIRED[name]
    raise KeyError(f"unknown task {name!r} (not in CANONICAL, CALIBRATION or RETIRED)")


if __name__ == "__main__":  # self-test: every canonical task generates + round-trips through the oracle
    for name, spec in CANONICAL.items():
        tr = generate(spec, "train", n=50)
        te = generate(spec, "test", n=50, length=spec.eval_lengths[-1])
        assert len({e.prompt for e in tr}) > 1 and all(e.answer for e in te)
        # determinism: regenerating gives identical examples
        assert generate(spec, "test", n=5, length=spec.eval_lengths[-1])[0].prompt == \
               generate(spec, "test", n=5, length=spec.eval_lengths[-1])[0].prompt
        ex = te[0]
        print(f"{name:<18} train={len(tr)} test@{spec.eval_lengths[-1]}={len(te)}  "
              f"e.g. ...{ex.prompt[-44:]!r} -> {ex.answer!r}")
    # metric sanity
    assert score_exact("g3 v9 . extra", "g3 v9 .") == 1 and score_exact("g3 v8 .", "g3 v9 .") == 0
    print("tasks self-test OK")
