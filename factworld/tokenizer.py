"""Atomic / closed-vocabulary tokenizer for FactWorld (milestone M4).

No BPE, no merges, no external dependencies — pure Python stdlib. The renderer
(`factworld.render.Renderer`) emits documents as *space-separated atomic tokens*
(content IDs like ``e17 a3 v42 o2 loc1 g4 s5`` plus shared structural/function
words). Tokenization is therefore exactly whitespace splitting, and the contract
is a perfect round-trip::

    decode(encode(text)) == text

for any rendered string whose tokens are all in the vocabulary (i.e. when no
``<unk>`` substitution occurs and ``add_bos``/``add_eos`` are both False).

Vocabulary
----------
The vocabulary is the union of:

(a) Special tokens at FIXED low ids::

        <pad>=0, <bos>=1, <eos>=2, <unk>=3

(b) Every content token enumerable from each ``World`` passed to ``build`` (all
    worlds are included so auxiliary operator-world namespaces are covered):
    ``entities, value_vocab, attribute_names, objects, locations, agents, roles``.

(c) Step tokens ``s0 .. s{max_step-1}`` (``max_step`` default 256) — emitted by
    ``render_history(with_steps=True)`` and used as as-of-t query labels. The same
    integer range supplies the chain/s5_chain hop annotation ``(N`` … ``hops)``.

(d) Structural / function tokens. These are derived *robustly* by rendering a
    probe of every statement type the renderer supports (``render_fact``;
    ``render_history`` with ``with_steps`` False AND True over an easy and a hard
    chain; the pointer-map events ``swap_a0``/``swap_a0_ref``/``cycle_a0``; the
    dial event and assertion; ``render_query`` for families ``recall``/
    ``state_easy``/``state_hard``/``state_comm`` with ``t=None`` AND ``t`` set) and
    collecting every token that is NOT a content/step token. A fixed seed set of
    structural words is additionally unioned in: it covers the paraphrase slots a
    single probe may not select, and the compact event grammar, which
    ``tasks._compact_a0_event`` emits without going through the renderer at all and
    which the probe therefore cannot reach.

Coverage: vocabulary, not encode-time normalisation
---------------------------------------------------
Some surface forms glue punctuation to a content id (``g5's``, ``a0,``, ``o3.``) or
to an integer (``(2``, ``hops)``). Two ways to cover them exist, and the contract
``decode(encode(x)) == x`` decides between them:

* **Normalise at encode time** — split ``a0,`` into ``a0`` + ``,``. This breaks the
  round trip: ``decode`` joins with single spaces and has no way to know which of
  the two tokens were originally glued, so ``x`` cannot be reconstructed. Making
  decode re-attach by rule would bake renderer-specific grammar into the tokenizer
  and still be ambiguous (``.`` follows both glued and free tokens).
* **Extend the vocabulary** — enumerate exactly the glued forms the renderer and
  the task suite can emit. Whitespace splitting stays the whole of tokenization,
  so the round trip is exact by construction.

The second is what this module does. The cost is vocabulary size, so the
enumeration is deliberately minimal: each content token is combined only with the
suffixes its TYPE can occupy (see ``_NAT_SUFFIX_BY_TYPE``) rather than with the
full punctuation product, and the hop annotation reuses the step-token integer
range instead of a second independent range.

Id-ordering scheme
------------------
Deterministic and stable. The four specials occupy ids 0..3 in their fixed
order. Every other token (content + step + structural) is gathered into one set
and assigned ids 4.. in plain ``sorted()`` order of the token string. Because the
input is a set sorted by string, identical worlds always yield identical
``token_to_id`` maps, independent of insertion order.

Stability is *within* a vocabulary, not across changes to it. The 256 ``(N`` hop
forms are added unconditionally, for every world, so extending the vocabulary
reorders the sorted set: ids shift from the first added string onward and
``vocab_size`` grows even for tasks that emit none of the new tokens. Local numbers
predating an extension are therefore not bit-reproducible, since both the
embedding/output shapes and the seeded initialisation that fills them depend on the
vocabulary. Re-running a pre-extension configuration reproduces its protocol, not
its exact value. Task items are unaffected — ``tasks.generate`` never consults the
tokenizer, so the item streams and every floor computed from them are stable.
"""
from __future__ import annotations

from .render import Renderer, classify
from .world import Event, World

# Special tokens at fixed low ids.
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
_SPECIALS = (PAD, BOS, EOS, UNK)

# Structural / function words the corpus reuses in role/holder ASSERTION lines
# (these reuse renderer words but a single rendered probe might not surface every
# paraphrase slot, so we union them in explicitly).
_STRUCTURAL_SEED = {
    "'s", "is", "the", "of", "has", "move", "to", "moved", "give", "given",
    "swap", "cycle", "and", "roles", "role", "where", "what", "does", "have",
    "at", "held", "by", ":", ".", "?",
    # scenario-id binding: a marker + a shared 10-token digit vocab (compositional id, no
    # per-scenario embedding) — see Renderer.render_scenario.
    "scn", *(f"#{d}" for d in range(10)),
    # composite pointer-chasing query: "... a0 of the holder of o3 ?"
    "holder",
    # clean natural-language renderer v2 (fixed subject-verb templates + arrow cycles)
    "gives", "receives", "moves", "swaps", "cycles", "holds", "->", "→",
    # commutative-state rung (turn_dial events, dial assertions, state_comm query):
    # "s0 turns g3's dial 2 clicks." / "g3's dial is at p2." / "what position is g3's dial?"
    # Amounts are bare digit tokens (1..k_positions-1); digits 0-9 cover k_positions <= 10.
    "turn", "turns", "dial", "dial?", "position", "click", "clicks", "click.", "clicks.",
    *(str(d) for d in range(10)),
    # state-referencing pointer-map event (Renderer._SWAP_A0_REF, s5_chain conditional_rate):
    # "s3 swaps the values of g4's a0 and the a0 of the agent whose a0 is currently g11."
    # The renderer probe covers all three words, but the COMPACT grammar is emitted by
    # tasks._compact_a0_event rather than by the renderer ("s3 swaps a0: g4 and whose a0 is
    # g11."), and the probe cannot reach it; "whose" is the token the two forms share.
    "agent", "whose", "currently",
    # mutual-reference family (s5_bind): the temporal phrases that say WHICH map resolves an
    # event's reference, and the query anchor. "at this point" / "at the start" end a sentence
    # in the two event forms and the two initial-condition lines, so the glued forms are
    # emitted too; "at the start"/"at this point" also occur mid-sentence in the give form
    # ("... whose role at the start is r5."), so the bare words are needed as well.
    "this", "point", "start", "end", "each", "who", "point.", "start.", "end?",
}


class Tokenizer:
    """Closed-vocabulary atomic tokenizer with an exact whitespace round-trip."""

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id: dict[str, int] = dict(token_to_id)
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]
        self.unk_id = self.token_to_id[UNK]

    # ----- construction -----
    @classmethod
    def build(cls, worlds: list[World], renderer: Renderer, max_step: int = 256) -> "Tokenizer":
        """Build a tokenizer covering every token reachable from ``worlds``.

        Specials get fixed ids 0..3; all remaining tokens (content from every
        world, step tokens ``s0..s{max_step-1}``, and structural tokens probed
        from the renderer) are assigned ids 4.. in sorted-by-string order.
        """
        tokens: set[str] = set()

        # (b) content tokens from every world (covers aux namespaces).
        for w in worlds:
            tokens.update(w.entities)
            tokens.update(w.value_vocab)
            tokens.update(w.attribute_names)
            tokens.update(w.objects)
            tokens.update(w.locations)
            tokens.update(w.agents)
            tokens.update(w.roles)
            tokens.update(w.positions)

        # (c) step tokens.
        tokens.update(f"s{i}" for i in range(max_step))

        # (d) structural tokens — explicit seed set ...
        tokens.update(_STRUCTURAL_SEED)
        # ... plus everything non-content probed from the renderer over every world.
        for w in worlds:
            for piece in cls._probe(w, renderer):
                for tk in piece.split():
                    if classify(tk) is None:  # not a content/step ID
                        tokens.add(tk)

        # Attached-punctuation surface forms (e.g. "g5's", "v107.", "o3?") are emitted by
        # the natural renderer; include the minimal set so every rendered document tokenizes
        # without <unk>. As-of-t queries ("who holds o0 at s0?") glue '?' to a step token, so
        # cover those too. See ``_natural_surface_forms`` (type -> suffix map) for why we do not
        # use the full content x {'s, ., ?} product.
        tokens.update(cls._natural_surface_forms(worlds, renderer))
        tokens.update(f"s{i}?" for i in range(max_step))
        # Hop annotation on the chain / s5_chain query ("... of g246? (128 hops)"). The count is
        # glued to the opening paren, so it is a surface form like the ones above; it ranges over
        # the same integers as the step labels.
        tokens.update(f"({i}" for i in range(max_step))
        tokens.add("hops)")

        # Specials are reserved at fixed ids; never let a probe shadow them.
        tokens.difference_update(_SPECIALS)

        token_to_id: dict[str, int] = {t: i for i, t in enumerate(_SPECIALS)}
        for i, tk in enumerate(sorted(tokens), start=len(_SPECIALS)):
            token_to_id[tk] = i
        return cls(token_to_id)

    # Attached-punctuation suffixes each content TYPE can take in a natural template
    # (derived from the natural templates in render.py). Adding the full
    # content x {'s, ., ?} product instead inflates the vocab 3-4x with dead tokens
    # (e.g. "v107's", "g5?" when g5 is never queried) and measurably hurts length
    # generalization (binding L64: bloated 0.69 -> minimal 0.81, mean over 3 seeds).
    _NAT_SUFFIX_BY_TYPE = {
        # agent: fact-subject ('s), recall entity (?), give-dest / answer (.), and one
        # slot of the mutual-reference whole-map readout, which enumerates its k slots
        # ("what role does each of g0, g1, ... have at the end?") so the answer's ORDER is
        # stated rather than conventional (,)
        "g":   ("'s", "?", ".", ","),
        "e":   ("'s", "?", "."),   # entity: same slots as agent
        "v":   (".",),              # value: fact value + answer
        "o":   ("?", "."),          # object: query target (?), holder assertion "g3 holds o0." (.)
        "r":   (".",),              # role: s5 answer
        "loc": (".",),              # location: move destination
        "p":   (".",),              # dial position: dial assertion + commutative answer
        # attribute: the pointer-map events glue punctuation to the attribute name —
        # "... takes g4's old a0," / "... old a0." (cycle_a0) and "swaps a0:" / "cycles a0:"
        # (the compact s5-style event grammar, tasks.TaskSpec.compact_events).
        "a":   (",", ".", ":"),
    }

    @staticmethod
    def _natural_surface_forms(worlds, renderer):
        """Exact, minimal set of attached-punctuation surface forms for the natural renderer.

        Each content token is combined only with the suffixes its TYPE can occupy in a natural
        template (see ``_NAT_SUFFIX_BY_TYPE``), so the vocab stays small while every rendered
        natural document tokenizes without ``<unk>``."""
        forms: set[str] = set()
        for w in worlds:
            for bucket in (w.entities, w.agents, w.value_vocab, w.objects, w.locations, w.roles,
                           w.positions, w.attribute_names):
                for tk in bucket:
                    c = classify(tk)
                    for suf in Tokenizer._NAT_SUFFIX_BY_TYPE.get(c, ()):
                        forms.add(tk + suf)
        return forms


    @staticmethod
    def _probe(world: World, renderer: Renderer):
        """Yield a rendered string of every statement type the renderer supports.

        Defensive about partial worlds: the minimal auxiliary operator-worlds have no recall
        side (``n_entities=0``), so each section is guarded by symbol availability. Structural
        tokens are world-independent, so unioning probes across all worlds covers everything
        (and ``_STRUCTURAL_SEED`` is a backstop).
        """
        # facts + recall query — need the recall symbols.
        if world.entities and world.attribute_names and world.value_vocab:
            ent, attr, val = world.entities[0], world.attribute_names[0], world.value_vocab[0]
            for key in ("k0", "k1", "k2", f"fact|{ent}|{attr}"):
                yield renderer.render_fact(ent, attr, val, key=key)
            yield renderer.render_query("recall", entity=ent, attribute=attr)

        # easy-state — needs objects and a non-empty holder domain.
        if world.objects and world.holders:
            easy = world.sample_easy_chain(40, "tok_probe_easy")
            for with_steps in (False, True):
                yield from renderer.render_history(easy, with_steps=with_steps)
            for e in easy:
                yield renderer.render_event(e, key="alt0")
                yield renderer.render_event(e, key="alt1")
            obj = world.objects[0]
            yield renderer.render_query("state_easy", target=obj, t=None)
            yield renderer.render_query("state_easy", target=obj, t=5)

        # hard-state — needs at least two agents (for a transposition).
        if len(world.agents) >= 2:
            hard = world.sample_hard_chain(40, "tok_probe_hard")
            for with_steps in (False, True):
                yield from renderer.render_history(hard, with_steps=with_steps)
            for e in hard:
                yield renderer.render_event(e, key="alt0")
                yield renderer.render_event(e, key="alt1")
            agent = world.agents[0]
            yield renderer.render_query("state_hard", target=agent, t=None)
            yield renderer.render_query("state_hard", target=agent, t=5)

        # pointer-map (a0) events — the s5_chain grammar. swap_a0 needs two agents,
        # cycle_a0 three. These carry the only tokens in the suite that glue punctuation
        # to the attribute name ("old a0," / "old a0.") plus "values"/"simultaneously:"/
        # "takes"/"old", none of which appear in any other statement type.
        # swap_a0_ref is the state-referencing form (TaskSpec.conditional_rate): its second
        # operand is a VALUE the running map holds, so the sentence carries a relative clause
        # ("the agent whose a0 is currently g11") and with it the only occurrences of
        # "agent"/"whose"/"currently" in the suite.
        if len(world.agents) >= 2:
            a, b = world.agents[0], world.agents[1]
            yield renderer.render_event(Event("swap_a0", (a, b)), step="s0")
            yield renderer.render_event(Event("swap_a0_ref", (a, b)), step="s0")
        if len(world.agents) >= 3:
            a, b, c = world.agents[0], world.agents[1], world.agents[2]
            yield renderer.render_event(Event("cycle_a0", (a, b, c)), step="s0")

        # mutual-reference (s5_bind) — the two referenced event forms under both temporal
        # readings, the two temporally-anchored initial-condition lines, and the three
        # queries. Nothing else in the suite emits "this"/"point"/"start"/"end"/"each", the
        # glued "point."/"start."/"end?" forms, or the comma-separated slot enumeration.
        if world.agents and world.objects and world.roles:
            a, o, rl = world.agents[0], world.objects[0], world.roles[0]
            for kind, args in (("swap_roles_now", (a, o)), ("swap_roles_start", (a, o)),
                               ("give_role_now", (o, rl)), ("give_role_start", (o, rl))):
                yield renderer.render_event(Event(kind, args), step="s0")
            yield renderer.render_role(a, rl, when=renderer.AT_START)
            yield renderer.render_holder(o, a, when=renderer.AT_START)
            yield renderer.render_query("s5bind_state", target=a)
            yield renderer.render_query("s5bind_bind", target=o)
            yield renderer.render_query("s5bind_state_all", targets=list(world.agents))

        # commutative-state — needs agents and dial positions.
        if world.agents and world.positions:
            agent = world.agents[0]
            yield renderer.render_dial(agent, world.positions[0])
            for amount in range(1, len(world.positions)):
                yield renderer.render_event(Event("turn_dial", (agent, str(amount))), step="s0")
            yield renderer.render_query("state_comm", target=agent, t=None)
            yield renderer.render_query("state_comm", target=agent, t=5)

    # ----- properties -----
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ----- encode / decode -----
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Whitespace-split ``text`` to ids; unknown tokens map to ``unk_id``."""
        ids = [self.token_to_id.get(tk, self.unk_id) for tk in text.split()]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Join tokens with single spaces, skipping ``<pad>``.

        ``decode(encode(text)) == text`` whenever every token of ``text`` is in
        vocab and ``add_bos``/``add_eos`` were False.
        """
        toks = []
        for i in ids:
            tk = self.id_to_token.get(i, UNK)
            if tk == PAD:
                continue
            toks.append(tk)
        return " ".join(toks)
