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
    ``render_history(with_steps=True)`` and used as as-of-t query labels.

(d) Structural / function tokens. These are derived *robustly* by rendering a
    probe of every statement type the renderer supports (``render_fact``;
    ``render_history`` with ``with_steps`` False AND True over an easy and a hard
    chain; ``render_query`` for families ``recall``/``state_easy``/``state_hard``
    with ``t=None`` AND ``t`` set) and collecting every token that is NOT a
    content/step token. A fixed seed set of structural words used by the corpus's
    role/holder assertion lines is additionally unioned in so the vocab does not
    depend on which paraphrase slots a given seed happens to select.

Id-ordering scheme
------------------
Deterministic and stable. The four specials occupy ids 0..3 in their fixed
order. Every other token (content + step + structural) is gathered into one set
and assigned ids 4.. in plain ``sorted()`` order of the token string. Because the
input is a set sorted by string, identical worlds always yield identical
``token_to_id`` maps, independent of insertion order.
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
        "g":   ("'s", "?", "."),   # agent: fact-subject ('s), recall entity (?), give-dest / answer (.)
        "e":   ("'s", "?", "."),   # entity: same slots as agent
        "v":   (".",),              # value: fact value + answer
        "o":   ("?",),              # object: state_easy / composite-holder query target
        "r":   (".",),              # role: s5 answer
        "loc": (".",),              # location: move destination
    }

    @staticmethod
    def _natural_surface_forms(worlds, renderer):
        """Exact, minimal set of attached-punctuation surface forms for the natural renderer.

        Each content token is combined only with the suffixes its TYPE can occupy in a natural
        template (see ``_NAT_SUFFIX_BY_TYPE``), so the vocab stays small while every rendered
        natural document tokenizes without ``<unk>``."""
        forms: set[str] = set()
        for w in worlds:
            for bucket in (w.entities, w.agents, w.value_vocab, w.objects, w.locations, w.roles):
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
