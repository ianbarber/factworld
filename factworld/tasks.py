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

The ``s5_bind`` family runs two structures over one interleaved event stream and has every event
name its second operand through the other one, so the composed query cannot be answered by either
component's algorithm. Its arms — the composed cell, the two components and the whole-map capacity
control — share a ``stream_name`` and therefore ONE item stream, so they are exactly paired and
the coupling ablation is a within-item comparison at identical prompt length (``s5_bind_arms``).

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

    # s5_bind-only. The family runs TWO structures over ONE interleaved event stream —
    # P: agents -> roles, a bijection permuted by the swaps (state tracking), and
    # B: objects -> agents, rewritten by the gives under last-write-wins (retrieval under
    # overwrite) — and every event names its second operand THROUGH the other structure.
    #
    #   p_swap      P(an event is a swap); the rest are gives.
    #   rho_p       fraction of swaps whose holder reference is rendered "at this point"
    #   rho_b       fraction of gives whose role reference is rendered "at this point"
    #   coupled     THE RENDERING TOGGLE. True renders those references "at this point", so
    #               they resolve against the running maps; False renders every reference
    #               "at the start", so the same item's events resolve against the stated
    #               maps and the two structures never touch. The phrases are the same
    #               length, so the ablation moves two tokens per referenced event and NOT
    #               the prompt length.
    #   query_arm   which of the three paired queries is scored: 'state' (the queried
    #               agent's final role), 'bind' (the queried object's final holder), or
    #               'state_all' (every agent's final role — the whole-map readout that
    #               prices capacity separately from composition).
    #
    # m, the number of objects, is n_objects_active (reused as this family's working set the
    # way the commutative rung reuses it for active dials); m <= k is required so the stated
    # holder map is injective. All four fields are appended and defaulted, `_rng` does not
    # key on them, and no existing spec is in this family, so no stream moves.
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

    # s5_bind-only. The largest RUN of consecutive events allowed OFF the queried agent's
    # dependency chain, as a fraction of L — the gate that closes the block-drop family.
    #
    # THE FAMILY IT CLOSES. window_f (keep the last f*L events) and prefix_f (keep the first
    # f*L) are the two endpoints of one family: drop a block of width w at position p and play
    # everything else. The family is continuous in (p, w), non-monotone in both, and every
    # member costs ~0.9x the task, so the max over any finite registered subset of it is a
    # selection statistic over an effectively exchangeable set. Registering members cannot close
    # it; three rounds of registration each lost to a neighbour.
    #
    # WHY THE FAMILY WORKED. A block-drop is wrong only when it drops an event on the queried
    # agent's DEPENDENCY CHAIN — the events that can change the answer. That chain is the
    # backward carrier walk of the answer's role: at each swap both operands are on it (the
    # answer's role moves from one to the other), and under the coupled rendering the second
    # operand is itself resolved through B, so the chain closes over both structures. Under the
    # free sampler the chain holds ~2 p_swap L / k of the L events, so most blocks miss it: the
    # longest off-chain run is 43 events at k=12/L=128 and 17 at k=6/L=48, and a width-0.1L
    # block dropped in the late interior read 2.2x the floor.
    #
    # THE CONSTRAINT. With chain_max_gap = f the sampler holds every off-chain run FROM THE
    # CHAIN'S FIRST EVENT ONWARD to at most w_min - 1 events, w_min = round(f*L), the run after
    # the last chain event included. Every block of width >= w_min that starts at or after that
    # first event therefore contains an event that can change the answer. Two mechanisms:
    #   - STEERING. When the run reaches w_min - 1 the swap the coin has already produced is
    #     re-targeted onto the answer's current carrier; only if that slot drew a give as well
    #     is a swap forced into the stream. Re-targeting rather than inserting is what keeps the
    #     swap:give mix — the quantity both the composed pass and the component walk are priced
    #     on — from moving, so the step multiplier does not fall (measured: 4% or less, and the
    #     composed step count RISES). Which operand carries the steer is drawn per event: through
    #     the referenced one the sentence names an object the carrier holds, through the named
    #     one it names the carrier.
    #   - QUERY GATE. The queried agent is drawn from the agents whose final role's chain
    #     satisfies the bound. In practice that is the steered role, so the bound is a property
    #     of the item and not a selection over roles.
    #
    # THE HEAD IS NOT BOUNDED, AND MAY NOT BE. The run BEFORE the chain's first event is left
    # exactly as the free sampler draws it. Bounding it looks symmetric and is not: the first
    # chain event names the agent holding the answer's role AT THE START, whose STATED role is
    # the answer, so forcing that event into the first w_min events makes "the stated role of an
    # operand of the j-th swap" — one retrieval, no state — a shortcut. Measured with the head
    # bounded at k=12/L=128: that policy reads 1.6x chance at the best offset and 2.7x when the
    # steer also fixed which operand carried it, against 1.03x on the free stream. Widening
    # w_min does not fix it, because the leak is ~chance / P(the first chain event falls inside
    # w_min) and only reaches chance at w_min ~ 3k events, by which point the gate no longer
    # covers the widths the family actually lives at. The tail has no such leak — the last chain
    # event's other operand is the previous carrier, whose stated role is unrelated to the
    # answer — which is why q_tail could be bought and head coverage cannot.
    #
    # WHAT THAT LEAVES. Blocks lying entirely inside the leading run are not closed by
    # construction. They are the ones that discard the stream's HEAD, and dropping the head
    # perturbs P and B for every later resolution, so the answer's trajectory diverges even when
    # the block misses the chain. Measured over the full (position, width) scan at n=1500 and
    # widths >= w_min, the position-0 column reads 0.71-1.11x chance on the free stream and
    # 0.70-1.07x on the gated one, and the drops that DID miss the chain — 0-10% of the
    # (item, position) pairs at k >= 8 — read 0.7-1.5x chance conditional on missing. That is
    # measured, not constructed, and it is reported as such.
    #
    # Appended and defaulted to 0.0, which disables both mechanisms, so every pre-existing
    # stream is byte-identical. The CANONICAL specs set 0.05 at k=12 (w_min = 6/10/13 events at
    # L=128/192/256) and 0.10 at k=6 (w_min = 5/6 at L=48/64) — in both cases the smallest
    # fraction whose w_min at the cell's shortest length is still around five events. Below that
    # the sampler has to put a chain event on almost every slot and the stream degenerates into
    # alternating kinds.
    #
    # NOT AVAILABLE ON A rho_ladder SPEC. The steer follows ONE coupled trajectory, and a ladder
    # spec's whole construction is one skeleton read at five doses whose trajectories differ, so
    # no single steered event is on the chain at every rung. Setting both raises. The ladder cells
    # are CALIBRATION and are never scored; their block-drop family stays open and is reported as
    # the diagnostic it is.
    chain_max_gap: float = 0.0

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


def _s5_bind_gap_limit(spec, length: int) -> int:
    """The largest off-chain run the stream may contain AFTER the chain's first event
    (TaskSpec.chain_max_gap), in events.

    ``w_min = round(chain_max_gap * length)`` is the smallest block width the gate makes
    unavoidable, so the run may be at most ``w_min - 1``. -1 disables the gate.
    """
    if not spec.chain_max_gap:
        return -1
    return max(1, int(round(spec.chain_max_gap * length))) - 1


def _s5_bind_runs(idxs, length: int) -> tuple[int, int]:
    """``(leading run, longest run after it)`` — the runs of consecutive event indices NOT in
    ``idxs``, split at the first index because the two ends of the stream are not symmetric.

    A block of width w misses ``idxs`` iff it fits inside one of these runs. The gate bounds the
    SECOND number only; see TaskSpec.chain_max_gap on why bounding the first one hands a
    zero-state policy the answer.
    """
    if not idxs:
        return length, 0
    rest = [length - 1 - idxs[-1]]
    rest += [idxs[j + 1] - idxs[j] - 1 for j in range(len(idxs) - 1)]
    return idxs[0], max(rest)


def _s5_bind_stream(spec, agents, roles, objs, rng, length, idx):
    """One s5_bind item's world, event stream and queries — the part of the sampler that must
    not consult the rendering.

    Returns ``(P0, B0, events, writes, last_give, q_state, q_bind, fact_roles, fact_holds,
    finals)``, where ``finals`` maps each simulated dose (and ``None``, the all-static reading)
    to that dose's ``(P, B, touch)``. Events carry ``dyn`` under the spec's OWN dose.

    Two samplers, selected by ``spec.rho_ladder``:

      DEFAULT      each event's coupling coin is drawn inside the rejection loop, and the item
                   is checked for degeneracy under the spec's own dose and under the all-static
                   reading. Two doses, two lanes. This is also the sampler that can carry the
                   chain gate (``spec.chain_max_gap``): one designated role's carrier is tracked
                   under the coupled trajectory, and once it has moved once, a swap is steered
                   onto it whenever the run of events since its last move would otherwise reach
                   ``w_min``.
      SKELETON-FIRST (rho_ladder set)
                   the coupling variate is drawn once per event SLOT, before the operands, and
                   the item is checked under every dose in the ladder. The draw sequence is then
                   independent of the spec's own dose, so every rung of the ladder is the same
                   item — see TaskSpec.rho_ladder.
    """
    k, m = spec.k, spec.n_objects_active
    if spec.chain_max_gap and spec.rho_ladder:
        raise ValueError(f"{spec.name}: chain_max_gap steers ONE coupled trajectory and a "
                         f"rho_ladder spec has one per rung; the two cannot both be set.")
    if not spec.rho_ladder:
        gap_limit = _s5_bind_gap_limit(spec, length)
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
            # role -> the indices of the events that MOVE it, under the coupled trajectory: the
            # dependency chain of whichever agent ends up holding it (TaskSpec.chain_max_gap).
            chain_c: dict[str, list[int]] = {rl: [] for rl in roles}
            # the steered role's current holder, the run of events since it last moved, and
            # whether it has moved at all yet (the leading run is not steered — see below)
            carrier = rng.choice(agents) if gap_limit >= 0 else None
            run, started = 0, False
            for _i in range(length):
                # STEER the swap that the coin already gave us one slot before the run runs
                # out, and only FORCE a swap into the stream when the coin gave a give at that
                # slot too. Steering re-targets an event the stream was going to contain
                # anyway, so the swap:give mix — which is what both the composed pass and the
                # component walk are priced on — barely moves.
                #
                # NOTHING IS STEERED BEFORE THE FIRST CHAIN EVENT. That event names the agent
                # holding the answer's role at the start, so its stated role IS the answer; a
                # steered one would put that agent at a predictable place on the surface and
                # hand a zero-state policy the answer in two retrievals. Left free it lands
                # where the free stream puts it, on either operand, and the leading run is
                # bounded by REJECTION at the query gate instead — which selects items rather
                # than shaping them.
                steer = gap_limit >= 0 and started and run >= gap_limit - 1
                force = gap_limit >= 0 and started and run >= gap_limit
                swap = True if force else rng.random() < spec.p_swap
                ok = False
                for _try in range(200):
                    if swap:
                        if steer:
                            # WHICH SIDE carries the steer is drawn per try. Through the
                            # REFERENCED operand the sentence names an object the carrier
                            # happens to hold; through the NAMED one it names the carrier. A
                            # fixed side would make the answer's first carrier readable off the
                            # surface at a fixed place — see chain_max_gap on why the head is
                            # where that matters — so neither side is fixed, and the naming form
                            # is also the fallback where the carrier holds nothing or where every
                            # object it holds is refused by the other constraints (no_pin above
                            # all), which is what keeps the retry loop from starving.
                            dyn = rng.random() < spec.rho_p
                            held = ([x for x in objs if (Bc if dyn else B0)[x] == carrier]
                                    if rng.random() < 0.5 else [])
                            if held:
                                o, a = rng.choice(held), rng.choice(agents)
                            else:
                                a, o = carrier, rng.choice(objs)
                        else:
                            a, o = rng.choice(agents), rng.choice(objs)
                            dyn = rng.random() < spec.rho_p
                        bc = Bc[o] if dyn else B0[o]
                        bd = B0[o]
                        if bc != a and bd != a:          # no self-swap under EITHER semantics
                            if spec.no_pin and dyn and pin.get(o) is not None and Pc[bc] == pin[o]:
                                continue                 # the references would cancel the state
                            if steer and carrier not in (a, bc):
                                continue                 # the steered event must move the carrier
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
                    chain_c[Pc[a]].append(len(events) - 1)
                    chain_c[Pc[bc]].append(len(events) - 1)
                    Pc[a], Pc[bc] = Pc[bc], Pc[a]
                    Pd[a], Pd[bd] = Pd[bd], Pd[a]
                    Pc_inv = _invert(Pc)
                    touch_c[a] += 1
                    touch_c[bc] += 1
                    touch_d[a] += 1
                    touch_d[bd] += 1
                    last_c[a] = last_c[bc] = last_d[a] = last_d[bd] = len(events) - 1
                    if carrier in (a, bc):
                        carrier = bc if carrier == a else a
                        run, started = 0, True
                    else:
                        run += 1
                else:
                    events.append({"kind": "give", "o": o, "ref": rl, "dyn": dyn})
                    Bc[o], Bd[o] = hc, hd
                    pin[o] = rl if dyn else None
                    writes[o] += 1
                    run += 1
            if len(events) < length:
                continue
            tail_lo = _s5_bind_tail_lo(spec, length)
            cand_s = [a for a in agents if touch_c[a] >= 2 and touch_d[a] >= 2
                      and Pc[a] != P0[a] and Pd[a] != P0[a]
                      and last_c[a] >= tail_lo and last_d[a] >= tail_lo]
            if gap_limit >= 0:
                # every block of width >= w_min that starts at or after the chain's first event
                # hits the chain by construction
                cand_s = [a for a in cand_s
                          if _s5_bind_runs(chain_c[Pc[a]], length)[1] <= gap_limit]
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
    """s5_bind: two structures over one event stream, each naming its operands through the other.

    WORLD. k agents, k roles, m <= k objects. P maps agents to roles and is a bijection
    permuted by swap events; B maps objects to agents under last-write-wins. Both initial maps
    are STATED ("g3 has role r1 at the start." / "g7 holds o2 at the start.") in scrambled
    order, so nothing about either is conventional.

    STREAM. Each event is a swap with probability ``p_swap``, else a give, and each names its
    second operand through the OTHER structure:
      swap — "s0 swaps the roles of g4 and the agent who holds o2 {when}."
      give — "s1 gives o3 to the agent whose role {when} is r5."
    ``when`` is "at this point" (resolve against the running map) on a ``rho_p`` / ``rho_b``
    fraction of swaps / gives when ``spec.coupled``, and "at the start" (resolve against the
    stated map) otherwise. The decoupled rendering of the same item is the same sentences with
    that phrase replaced, so the two arms are token-for-token identical in length.

    WHY THE COUPLING IS THE WHOLE CONSTRUCT. Decoupled, the two legs are separable and cheap:
    the queried agent's role is a SPARSE backward walk over the swaps that name it (one live
    symbol), and the queried object's holder is one content-addressed retrieval of its last
    give plus one fact lookup. Coupled, neither is available — a swap's second operand is not
    known until B has been evaluated forward to that event, and a give's recipient is not known
    until P has — so the cheapest correct algorithm is a single forward pass carrying P, its
    inverse and B. Measured on this generator (step-counted register machine with free
    content-addressed retrieval, cheapest algorithm correct on EVERY item of the cell):
    at k=12/m=12/L=192 the composed cell costs 674.6 steps and 36 live cells against 90.4/2 for
    the state component (a sparse backward walk) and 3.0/3 for the retrieval component — a step
    multiplier of 7.46. Across the scored grid it reads 7.15 / 7.46 / 7.28 at L=128/192/256 and
    3.45 / 3.60 at k=6/L=48/64. The demand-driven serialisation (resolve only what the query
    needs, memoizing every event's operand) and the iterate-to-a-fixed-point serialisation are
    both more expensive, the latter by an order of magnitude, so the multiplier is not an
    artifact of forbidding a cheaper schedule.

    QUERY GATES, applied identically under both semantics so the arms condition on the same
    items: the queried agent is moved at least twice, ends on a role other than its stated one,
    and its LAST carrier event sits in the final ``q_tail`` of the stream; the queried object is
    written at least twice, ends with a holder other than its stated one, and its resolving
    write sits in [0.1L, 0.75L]. Each positional clause answers a truncation policy. Without the
    object's upper bound its resolving write lands near the stream end, where the map has not
    moved since, and "resolve every reference against the FINAL map" — a wrong-TIME policy, not
    a shallow one — reads 0.33-0.44 at every L. Without ``q_tail`` the queried agent's carrier
    chain finishes mid-stream, and simulating the task exactly and stopping 10% early reads
    0.45/0.37/0.29 at L=128/192/256 against a 0.10-0.12 floor: the two clauses are the two ends
    of one family (see TaskSpec.q_tail and validity's window_/prefix_ rows). The queried agent's
    chain also has no gap of ``chain_max_gap`` events or more after its first event, which is
    what makes every block-drop of that width or wider land on it.

    NO_PIN. Two dynamic references compose into a state-free reset channel: a give writes
    B[o] <- Pinv[r], pinning o's holder to role r, and a later swap naming o then writes r onto
    its own agent, because selecting an agent by its role and reading that role back returns
    the role. On such items a 2-retrieval policy carrying no map answers the state query, and
    because the channel is length-free that policy does not decay with L. ``spec.no_pin``
    rejects the second event of such a pair at sampling time; see TaskSpec.no_pin.

    FLOORS. The registered shallow policies are in ``factworld.validity.s5_bind_floors``; the
    operative floor is the max over the rows that carry NO map (``s5_bind_operative_floor``) and
    is what a score is read against. Registration is by resource class, not by accuracy: the
    cheapest correct algorithm on a coupled cell carries P, its inverse and B — 2k + m live
    slots — so a policy that carries a map is doing the task's own work at a constant-factor
    discount and is reported as a diagnostic instead. With no_pin, q_tail and chain_max_gap set
    and L/k >= 8, that floor lands within 1.03-1.14x the informed chance 1/(k-1) across k=6..16
    (n=1500); at shorter L the one-leg rows are still well above it, which is what the length
    grid in CANONICAL is cut on.

    THE BLOCK-DROP FAMILY. window_f and prefix_f are two positions of one continuum — drop a
    block of width w at position p, play everything else — which is continuous in (p, w) and
    non-monotone in both, so the max over any finite registered subset of it is a selection
    statistic and no set of registered members could ever have been its floor. Every member
    carries both maps, so the class rule excludes the whole continuum at once. The class rule is
    not left to carry it alone: ``spec.chain_max_gap`` bounds the off-chain runs so that every
    block of width >= w_min drops an event that can change the answer, and over the full
    19-position x 8-width scan (n=1500, independent parser and simulator,
    scripts/probe_s5bind_block_drop_20260730.py) the best member at any width >= w_min reads
    1.00-1.33x chance on every scored cell, against 1.7-5.1x before the gate. Blocks NARROWER
    than w_min, and blocks lying inside the stream's un-gated leading run, are excluded by class
    alone; the residual is measured (see TaskSpec.chain_max_gap) and is not folded into the
    floor.
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
            out.append(_ex_s5_bind(spec, w, r, rng, L, idx))         # L = interleaved events
        elif spec.family == "chain":
            out.append(_ex_chain(spec, w, r, rng, L, idx))           # L = chain depth
        else:
            raise ValueError(spec.family)
    return out


def s5_bind_arms(spec: TaskSpec, split: str = "test", n: int = 200, length: int | None = None,
                 arms=((True, "state"), (False, "state"), (False, "bind"), (False, "state_all"))
                 ) -> dict[tuple[bool, str], list[Example]]:
    """The SAME s5_bind items read under several (coupled, query_arm) settings.

    Item generation does not consult either knob — the sampler rejects self-swaps and no-op
    writes under BOTH semantics and the query gates hold under both — so index i is one world,
    one event stream and one pair of queries throughout, and the returned lists are aligned.

    This is what makes the coupling ablation a within-item comparison: the coupled and
    decoupled renderings of item i are the same sentences with "at this point" replaced by "at
    the start", identical in whitespace-token count, so a difference between the two arms is
    not a difference of samples, of prompt lengths, or of query difficulty. A per-step error
    rate that scales with prompt length is common to both and cancels — which a normalised gap
    against the whole-map readout does not do: a single error rate fitted on a component
    predicts a large state_all-to-composed gap with no composition ability present at all.
    """
    base = spec if spec.stream_name is not None else replace(spec, stream_name=spec.name)
    return {(c, q): generate(replace(base, coupled=c, query_arm=q), split, n, length)
            for c, q in arms}


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
    # ---- s5_bind: the mutual-reference composition (see _ex_s5_bind) --------------------
    # Four specs, ONE item stream. They share ``stream_name="s5_bind_v2"``, so item i is the
    # same world, the same events and the same two queries in all four; they differ only in
    # what is rendered ("at this point" vs "at the start") and what is asked. That is the
    # pairing the family exists for — the composed cell and its two components are read off
    # the same items, at identical prompt lengths, so the coupling ablation is within-item.
    #
    #   s5_bind_v2        COMPOSED   coupled rendering, single-slot state query
    #   s5_bind_v2_state  COMPONENT  decoupled, same query — permutation tracking alone
    #   s5_bind_v2_bind   COMPONENT  decoupled, holder query — retrieval under overwrite alone
    #   s5_bind_v2_map    CONTROL    decoupled, whole-map readout — capacity, not composition
    #
    # The control is registered because the composed cell needs k slots live and the state
    # component needs one; without a k-slot readout at the same length, a composed-minus-
    # component gap is not separable from carrying more state at all. It is a CONTROL, not a
    # null: min(component, control) is a ceiling on the composed cell, never a floor.
    #
    # Operating point k=12, m=12: the answer space is the 12 roles and the whole-map readout
    # is a 12-slot permutation. The name carries the construct version, ``version`` the RNG
    # stream (frozen at introduction); v1 of the construct never reached the registry — it
    # admitted an iterate-to-a-fixed-point serialisation and its coupling ablation moved
    # prompt length, both of which this rendering excludes by construction.
    #
    # LENGTH GRID. The operative floor (validity.s5_bind_operative_floor — the max over the
    # registered policies that carry no map) is what sets the shortest scored length, and with
    # the pin channel closed and the chain gate on it reaches the informed chance
    # 1/(k-1) = 0.0909 rather than plateauing above it. Measured on this stream at n=3000,
    # k=12, coupled/state, as a ratio to that chance: L=64 1.38, L=96 1.06, L=128 1.03,
    # L=192 1.03, L=256 1.15. The residual at L=64 is not the pin channel (pin density is 0.000
    # at every length) but stream length against k: with 12 agents and 12 objects a 64-event
    # stream gives the queried agent a short carrier chain and each object ~2.7 writes, so
    # feeding B into P but not P into B still lands 1.4x chance. From L/k >= 8 it falls to
    # chance, so the grid starts at 128. The floor per cell is recomputed and printed by
    # scripts/validate_suite.py; a rescaled cell carries its own.
    #
    # chain_max_gap = 0.05 here: w_min = 6 / 10 / 13 events at L = 128 / 192 / 256, so every
    # block-drop that wide or wider lands on the queried agent's dependency chain. Over the full
    # 19-position x 8-width scan at n=1500 the best member reads 1.08-1.20x chance, against
    # 2.36-4.73x on the un-gated stream (scripts/probe_s5bind_block_drop_20260730.py).
    #
    # kind=experimental until the calibration lands, so none of the four is in REPORTED.
    "s5_bind_v2":       TaskSpec("s5_bind_v2", "s5_bind", kind="experimental",
                                  k=12, n_objects=12, n_objects_active=12,
                                  coupled=True, query_arm="state", stream_name="s5_bind_v2",
                                  no_pin=True, q_tail=0.1, chain_max_gap=0.05,
                                  train_lengths=(16, 32), eval_lengths=(128, 192, 256)),
    "s5_bind_v2_state": TaskSpec("s5_bind_v2_state", "s5_bind", kind="experimental",
                                  k=12, n_objects=12, n_objects_active=12,
                                  coupled=False, query_arm="state", stream_name="s5_bind_v2",
                                  no_pin=True, q_tail=0.1, chain_max_gap=0.05,
                                  train_lengths=(16, 32), eval_lengths=(128, 192, 256)),
    "s5_bind_v2_bind":  TaskSpec("s5_bind_v2_bind", "s5_bind", kind="experimental",
                                  k=12, n_objects=12, n_objects_active=12,
                                  coupled=False, query_arm="bind", stream_name="s5_bind_v2",
                                  no_pin=True, q_tail=0.1, chain_max_gap=0.05,
                                  train_lengths=(16, 32), eval_lengths=(128, 192, 256)),
    "s5_bind_v2_map":   TaskSpec("s5_bind_v2_map", "s5_bind", kind="experimental",
                                  k=12, n_objects=12, n_objects_active=12,
                                  coupled=False, query_arm="state_all", stream_name="s5_bind_v2",
                                  no_pin=True, q_tail=0.1, chain_max_gap=0.05,
                                  train_lengths=(16, 32), eval_lengths=(128, 192, 256)),
    # The from-scratch operating point: k=6, m=6, shorter streams, and per-EVENT state
    # checkpoints (event_trace — the whole state after every event, the supervision density
    # that formed s5 locally). A streaming model has no scratchpad, so its cost model is the
    # forward pass, which is the algorithm this construct forces in both regimes. Paired the
    # same way, on their own stream. Its operative floor reaches the 0.200 informed chance at
    # the same L/k as the frontier point (measured n=3000, as a ratio to chance: L=32 1.28,
    # L=48 1.11, L=64 1.09), so the scored lengths start at 48.
    #
    # chain_max_gap = 0.1 here, not the 0.05 the k=12 point carries: at L=48 a 0.05 fraction is
    # a two-event w_min, and holding the chain that dense leaves the sampler no free swap:give
    # mix — the event kinds start alternating. At 0.1 w_min is 5 / 6 events at L = 48 / 64 and
    # the best drop that wide, over the whole position scan, reads 1.08x and 1.03x chance;
    # drops of 2-3 events are not closed, reading 1.97x and 1.53x, and are excluded by resource
    # class alone.
    "s5_bind_local_v2": TaskSpec("s5_bind_local_v2", "s5_bind", kind="experimental",
                                  k=6, n_objects=6, n_objects_active=6,
                                  coupled=True, query_arm="state",
                                  stream_name="s5_bind_local_v2", event_trace=True,
                                  no_pin=True, q_tail=0.1, chain_max_gap=0.1,
                                  train_lengths=(16, 32), eval_lengths=(48, 64)),
    "s5_bind_local_v2_state": TaskSpec("s5_bind_local_v2_state", "s5_bind", kind="experimental",
                                        k=6, n_objects=6, n_objects_active=6,
                                        coupled=False, query_arm="state",
                                        stream_name="s5_bind_local_v2", event_trace=True,
                                        no_pin=True, q_tail=0.1, chain_max_gap=0.1,
                                        train_lengths=(16, 32), eval_lengths=(48, 64)),
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
}

# CALIBRATION specs: cells that measure how a construct behaves rather than how a model scores.
# They are generable and named like any other task, and they are NOT scored — nothing in
# REPORTED, nothing in the benchmark roster, and outside the CANONICAL validity gate, which
# fails a cell whose strongest registered shallow policy reads 0.5 or more (a calibration cell
# is allowed to be mostly floor; that is often the thing being measured).
#
# ---- s5_bind: the coupling-DOSE ladder ---------------------------------------------------
# rho_p = rho_b in {0, 0.25, 0.5, 0.75, 1} on the composed arm of the k=12 operating point:
# the fraction of events whose second operand is named "at this point" rather than "at the
# start".
#
# THE LADDER IS ONE ITEM STREAM READ FIVE WAYS. All five rungs set the same rho_ladder and
# share stream_name="s5_bind_v2_lad", so the skeleton-first sampler (TaskSpec.rho_ladder)
# gives index i the same world, the same events and the same two queries at every dose; the
# five prompts differ only in which sentences say "at this point", at equal whitespace-token
# counts, and gold is the only other thing that moves. A dose contrast is therefore
# within-item. The bottom rung renders every reference statically, so the origin of a dose
# response is an identity control — at rho=0 the coupled and decoupled readings of an item are
# the same string — rather than an assumption. The top rung carries the composed cell's knobs
# but is a different draw from s5_bind_v2, whose stream predates the paired sampler; the
# ladder is read within itself, never against that cell item by item.
#
# Two measured properties fix what the ladder can be read for.
#
#   THE FLOOR MOVES WITH THE DOSE, and at the low end it takes most of the cell. Operative
#   floor (factworld.validity.s5_bind_floors, n=500 paired test items) against 1/(k-1) = 0.091:
#       rho        0.0    0.25     0.5    0.75     1.0
#       L=192    0.250   0.558   0.144   0.108   0.104
#   and pin density is 0.0000 at every rung. The low rungs are set by one_leg_B — feed B into P
#   but never P into B — because when a quarter of the events are referenced, half the coupling
#   already reproduces the answer; the top two are set by stale_resolution at chance.
#   Lowering the dose walks the cell continuously into its own decoupled reading, so headroom
#   is a function of the dose. Only L=192 is registered: at L=64 four of the five rungs read
#   0.496 or more, and a rung with 0.06 of headroom measures its floor.
#
#   THE CHEAPEST CORRECT ALGORITHM IS A FUNCTION OF THE DOSE. On a step-counted register
#   machine with free content-addressed retrieval, k=12/L=64, the composed cell costs
#   29 / 127 / 178 / 249 / 289 steps at rho = 0 / 0.25 / 0.5 / 0.75 / 1: the all-static reading
#   is a sparse backward walk over one live symbol, the intermediate doses are cheapest under
#   demand-driven resolution, and only rho=1 forces the full forward pass. An executor with no
#   composition-specific failure at all — one per-step slip rate, nothing else — therefore
#   walks down the ladder: 0.80 / 0.74 / 0.65 / 0.60 at rho = 0.25 / 0.5 / 0.75 / 1, a slope of
#   0.27 per unit dose, at a slip rate that leaves the decoupled component at 0.95. A slope in
#   rho measures the cost of the cheapest algorithm; it is not by itself evidence about
#   composition. What the paired ladder adds is the per-ITEM contrast the between-item ladder
#   could not support: at a fixed dose the skeleton fixes WHICH of an item's own references
#   cross structures, so the number of load-bearing cross-structure resolutions varies at
#   fixed load-bearing depth.
#
# WHAT SEPARATES COMPOSITION FROM STEP COUNT, AND WHAT DOES NOT.
#
#   MATCHED STEP COUNT DOES NOT. Split the answer's backward dataflow slice under the cheapest
#   correct algorithm into the ops it actually depends on: at k=12/L=192/rho=1 that is 213 of
#   the 676 steps, because the forward pass must materialise B before it knows which of it the
#   query needs. The decoupled whole-map arm's slice is ALL of its steps. Matched on steps at
#   ~700, an executor with no composition-specific failure — one per-op slip rate, nothing
#   else — already reads acc(map) - acc(composed) = -0.385 scoring the map whole and +0.178
#   scoring it per slot: the null offset is larger than any composition effect and its SIGN is
#   a scoring choice. Read against the SLICE instead, one accuracy-vs-slice law fits both arms:
#   at a 0.002 slip rate it misses the composed arm by -0.022 and the whole-map arm by -0.026,
#   so the arm difference it leaves is 0.004. The two matchings cannot be satisfied at once —
#   the composed arm's slice/steps ratio is 0.31 and every uncomposed arm of this family is at
#   1.0 — so an arm difference prices whichever cost variable was matched, not composition. A
#   within-arm regression of accuracy on steps says the same: at one slip rate the per-step
#   log-survival slope is -6.8e-4 on the composed arm and -21.8e-4 on the uncomposed one, 3.2x
#   apart with no composition deficit anywhere. The ladder makes
#   the same point without a second arm: every rung costs 677.1 steps, and the
#   composition-free executor still walks 0.901 / 0.832 / 0.761 / 0.675 across
#   rho = 0.25 / 0.5 / 0.75 / 1.
#
#   AN OP-TYPE CONTRAST WITHIN ONE CELL DOES. Fit P(correct) = q + (1-q)/k with
#   q = exp(-(theta_w w + theta_z z + theta_x x)) over items, where w counts the slice's writes,
#   z its resolutions that did NOT need the other structure's running state, and x those that
#   did. z and x are the same operation in the same position of the same algorithm at the same
#   cost; they differ only in where the value came from, so theta_x - theta_z is zero for any
#   executor whose slip rate does not depend on that. Measured over 24 composition-free
#   executor configurations (k=12 rho=1/0.75 L=192, k=6 L=64; slips propagating / read-only /
#   first-slip-fatal / per-item-heterogeneous; two slip rates each) the statistic reads -0.021
#   to +0.0004 and the one-sided likelihood-ratio test's type-I is 0.000-0.090 against a
#   nominal 0.05. It is blind by design to a failure that garbles every dynamic reference
#   alike, including the ones whose referenced cell has not moved: those ask nothing about
#   composition, and a uniform surface failure on them is not evidence about it.
#
#   ITS POWER IS THE PROBLEM, AND IT SPLITS THE TWO REGIMES. Against an executor that resolves
#   a cross-structure reference against the STATED map with probability gamma, power at
#   alpha=0.05 is
#       k=12, L=192, n=500 :  0.09 / 0.27 / 0.35 for deficits costing 0.00 / 0.36 / 0.54
#       k=6,  L=64,  n=500 :  0.03 / 0.20 / 0.34 for deficits costing 0.00 / 0.12 / 0.22
#       k=6,  L=64,  n=5000:  0.02 / 0.81 / 0.99 for the same three
#   Nothing in the construct lifts the k=12 curve: greedy selection of the most informative
#   (item, query) pairs — the strongest lever that still conditions on the item alone — moves
#   the contrast's standard error from 0.0054 to 0.0026, worth about 4x the sample size. So the
#   statistic is an instrument for the from-scratch regime, where thousands of items are free,
#   and not for the frontier regime, where a few hundred is the budget.
#
# The k=6 operating point has no ladder: its longest scored stream is 48 events, where the five
# rungs read 0.488 / 0.852 / 0.604 / 0.320 / 0.228 against a 0.200 chance level. The op-type
# contrast needs no ladder — it runs on any coupled cell, including s5_bind_local_v2 itself.
_S5_BIND_LADDER = (0.0, 0.25, 0.5, 0.75, 1.0)

CALIBRATION = {
    "s5_bind_v2_lad00":  TaskSpec("s5_bind_v2_lad00", "s5_bind", kind="experimental",
                                   k=12, n_objects=12, n_objects_active=12, no_pin=True, q_tail=0.1,
                                   rho_p=0.0, rho_b=0.0, rho_ladder=_S5_BIND_LADDER,
                                   coupled=True, query_arm="state",
                                   stream_name="s5_bind_v2_lad",
                                   train_lengths=(16, 32), eval_lengths=(192,)),
    "s5_bind_v2_lad25":  TaskSpec("s5_bind_v2_lad25", "s5_bind", kind="experimental",
                                   k=12, n_objects=12, n_objects_active=12, no_pin=True, q_tail=0.1,
                                   rho_p=0.25, rho_b=0.25, rho_ladder=_S5_BIND_LADDER,
                                   coupled=True, query_arm="state",
                                   stream_name="s5_bind_v2_lad",
                                   train_lengths=(16, 32), eval_lengths=(192,)),
    "s5_bind_v2_lad50":  TaskSpec("s5_bind_v2_lad50", "s5_bind", kind="experimental",
                                   k=12, n_objects=12, n_objects_active=12, no_pin=True, q_tail=0.1,
                                   rho_p=0.5, rho_b=0.5, rho_ladder=_S5_BIND_LADDER,
                                   coupled=True, query_arm="state",
                                   stream_name="s5_bind_v2_lad",
                                   train_lengths=(16, 32), eval_lengths=(192,)),
    "s5_bind_v2_lad75":  TaskSpec("s5_bind_v2_lad75", "s5_bind", kind="experimental",
                                   k=12, n_objects=12, n_objects_active=12, no_pin=True, q_tail=0.1,
                                   rho_p=0.75, rho_b=0.75, rho_ladder=_S5_BIND_LADDER,
                                   coupled=True, query_arm="state",
                                   stream_name="s5_bind_v2_lad",
                                   train_lengths=(16, 32), eval_lengths=(192,)),
    "s5_bind_v2_lad100": TaskSpec("s5_bind_v2_lad100", "s5_bind", kind="experimental",
                                   k=12, n_objects=12, n_objects_active=12, no_pin=True, q_tail=0.1,
                                   rho_p=1.0, rho_b=1.0, rho_ladder=_S5_BIND_LADDER,
                                   coupled=True, query_arm="state",
                                   stream_name="s5_bind_v2_lad",
                                   train_lengths=(16, 32), eval_lengths=(192,)),
}


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
