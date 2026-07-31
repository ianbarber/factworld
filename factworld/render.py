"""Deterministic natural-language renderer + its inverse parser.

FactWorld renders every statement as **clean natural language with attached
punctuation**: facts as ``g0's a0 is v18.``, events as ``s0 gives o0 to g0.``,
cycles as ``s0 cycles roles: g0 -> g1 -> g2.``  One fixed phrasing per statement
type (no paraphrase variety) so the model sees a uniform grammar — this is the
single canonical format; the earlier space-separated "atomic-token" v1 format
lives in git history.

One statement type names an operand by STATE rather than by name: ``s7 swaps the
values of g4's a0 and the a0 of the agent whose a0 is currently g11.`` Such a
sentence encodes (named slot, referenced value); which slot the value belongs to
is a property of the map at that point in the stream, so parsing recovers the
pair and not the resolution.

The mutual-reference family carries that further: its events name their second
operand through the OTHER of the two structures the stream maintains, and a
temporal phrase says which map resolves the description — ``s0 swaps the roles of
g4 and the agent who holds o2 at this point.`` against the running holder map,
``... at the start.`` against the stated one. The phrase is part of the record, so
the four event kinds ``swap_roles_now`` / ``swap_roles_start`` / ``give_role_now``
/ ``give_role_start`` round-trip it, and the two readings occupy the same number
of whitespace tokens.

Content tokens are still atomic IDs (``e17 a3 v42 o2 loc1 g4 r0 s5``); the step
label ``sN`` is the event subject. The render <-> parse round-trip is a contract
(the ground-truth re-parse check — every rendered document must parse back to the
KB record it encodes). ``normalize()`` detaches attached punctuation
(``v109.`` -> ``v109 .``, ``g0's`` -> ``g0 's``) so scoring and parsing operate on
a canonical whitespace-token form regardless of how a model glued its output.
"""
from __future__ import annotations

import re
import zlib

from .world import Event

# Classify an atomic token by its type prefix (optionally namespaced, e.g. "aux1_g0").
# 'loc' must precede the single-char alternatives so "loc3" isn't read as an object.
# 'p' = dial position (the commutative-state answer set p0..p{k-1}; no other pN token exists).
_TOK = re.compile(r"^(?:[A-Za-z0-9]+_)?(loc|[eavogrsp])(\d+)$")

# Markdown emphasis / inline-code characters stripped from token EDGES by ``normalize``:
# chat models decorate answers ("**g22**", "`g22`", "_g22_") and the tokens-path scorers
# must treat those as the bare token. Edge-only, so namespaced ids with an INTERNAL
# underscore ("aux1_g0") are untouched.
_MD_EDGE_CHARS = "*_`"


def classify(token: str) -> str | None:
    m = _TOK.match(token)
    return m.group(1) if m else None


def _is_content_id(token: str) -> bool:
    return classify(token) is not None


class Renderer:
    """The single FactWorld renderer: clean natural language, attached punctuation.

    Each statement type has exactly one phrasing (a 1-tuple, kept as a tuple so a
    future paraphrase pass can slot in variants without touching call sites). The
    step label is the event subject: ``s0 gives o0 to g0.``.
    """

    _FACT = ("{e}'s {a} is {v}.",)
    _MOVE = ("moves {o} to {h}.",)
    _GIVE = ("gives {o} to {h}.",)
    _SWAP = ("swaps {a} and {b}.",)
    _CYCLE = ("cycles roles: {flows}.",)
    _SWAP_A0 = ("swaps the values of {a}'s a0 and {b}'s a0.",)
    # The second operand is named by its CURRENT value rather than by name: {v} is a value
    # the a0 map holds when the event fires, and the map is a bijection, so exactly one agent
    # answers to the description. "currently" is what separates it from the stated initial
    # facts — the referenced agent is the one whose a0 is {v} after every preceding event,
    # not the one the fact block gives {v} to.
    _SWAP_A0_REF = ("swaps the values of {a}'s a0 and the a0 of the agent "
                    "whose a0 is currently {v}.",)
    # The three assignments are SIMULTANEOUS: "{c}'s a0 takes {a}'s old a0" reads against
    # a's pre-event value, not the value a was just assigned. The pre-2026-07-18 wording
    # ("{a}'s a0 becomes {b}'s a0, ...") admitted a sequential-assignment misreading in
    # which the last leg reads an already-updated value.
    _CYCLE_A0 = ("cycles a0 simultaneously: {a}'s a0 takes {b}'s old a0, "
                 "{b}'s a0 takes {c}'s old a0, and {c}'s a0 takes {a}'s old a0.",)
    _ROLE = ("{g} has role {r}.",)
    _HOLDER = ("{h} holds {o}.",)
    _TURN = ("turns {g}'s dial {n} {clicks}.",)
    _DIAL = ("{g}'s dial is at {p}.",)

    # --- the mutual-reference (s5_bind) surfaces --------------------------------------
    # Two structures run over one event stream — agents->roles, permuted by the swaps, and
    # objects->agents, rewritten by the gives — and every event names its SECOND operand
    # through the other structure. A temporal phrase says which map resolves that
    # description: ``at this point`` reads the map as it stands when the event fires,
    # ``at the start`` reads the stated initial one. The two phrases are the same number of
    # whitespace tokens, so the coupled and decoupled renderings of ONE item differ by two
    # tokens per referenced event and never in length — the coupling can therefore be
    # ablated without moving prompt length, which is what makes the paired comparison exact.
    # The four event kinds are the two structural forms x the two temporal readings, so the
    # reading is part of the record a sentence encodes and the render/parse round-trip
    # recovers it.
    AT_POINT = "at this point"
    AT_START = "at the start"
    AT_END = "at the end"
    _ROLE_AT = ("{g} has role {r} {when}.",)
    _HOLDER_AT = ("{h} holds {o} {when}.",)
    _SWAP_BY_HOLDER = ("swaps the roles of {a} and the agent who holds {o} {when}.",)
    _GIVE_BY_ROLE = ("gives {o} to the agent whose role {when} is {r}.",)

    # Role-flow arrow in the compact cycle notation: "g0 -> g1 -> g2" means g0's role
    # passes to g1, g1's to g2, and g2's back to g0 (the canonical cycle_roles args).
    _CYCLE_ARROW = " -> "

    @staticmethod
    def _pick(options: tuple[str, ...], key: str) -> str:
        return options[zlib.crc32(key.encode()) % len(options)]  # deterministic across runs

    # ----- render -----
    def render_fact(self, entity: str, attribute: str, value: str, key: str | None = None) -> str:
        return self._pick(self._FACT, key or f"fact|{entity}|{attribute}").format(
            e=entity, a=attribute, v=value
        )

    def render_event(self, event: Event, step: str | None = None, key: str | None = None) -> str:
        """Render an event. ``step`` (e.g. "s0") is the subject; when omitted the bare
        predicate is returned (used only by tokenizer probing)."""
        k = key or f"ev|{event.kind}|{'|'.join(event.args)}"
        if event.kind == "move":
            s = self._pick(self._MOVE, k).format(o=event.args[0], h=event.args[1])
        elif event.kind == "give":
            s = self._pick(self._GIVE, k).format(o=event.args[0], h=event.args[1])
        elif event.kind == "swap_role":
            s = self._pick(self._SWAP, k).format(a=event.args[0], b=event.args[1])
        elif event.kind == "cycle_roles":
            s = self._pick(self._CYCLE, k).format(flows=self._CYCLE_ARROW.join(event.args))
        elif event.kind == "swap_a0":
            s = self._pick(self._SWAP_A0, k).format(a=event.args[0], b=event.args[1])
        elif event.kind == "swap_a0_ref":
            s = self._pick(self._SWAP_A0_REF, k).format(a=event.args[0], v=event.args[1])
        elif event.kind == "cycle_a0":
            s = self._pick(self._CYCLE_A0, k).format(a=event.args[0], b=event.args[1], c=event.args[2])
        elif event.kind in ("swap_roles_now", "swap_roles_start"):
            when = self.AT_POINT if event.kind.endswith("now") else self.AT_START
            s = self._pick(self._SWAP_BY_HOLDER, k).format(a=event.args[0], o=event.args[1], when=when)
        elif event.kind in ("give_role_now", "give_role_start"):
            when = self.AT_POINT if event.kind.endswith("now") else self.AT_START
            s = self._pick(self._GIVE_BY_ROLE, k).format(o=event.args[0], r=event.args[1], when=when)
        elif event.kind == "turn_dial":
            clicks = "click" if event.args[1] == "1" else "clicks"
            s = self._pick(self._TURN, k).format(g=event.args[0], n=event.args[1], clicks=clicks)
        else:
            raise ValueError(f"unknown event kind {event.kind!r}")
        return f"{step} {s}" if step is not None else s

    def render_history(self, events, with_steps: bool = True) -> list[str]:
        # with_steps is kept for API stability but events always carry a subject; when
        # False we still emit the step label because the natural grammar requires one.
        return [
            self.render_event(e, step=f"s{i}",
                              key=f"h|{i}|{e.kind}|{'|'.join(e.args)}")
            for i, e in enumerate(events)
        ]

    def render_scenario(self, idx: int, width: int = 4) -> str:
        """Scenario id as a marker + shared digit tokens, e.g. 'scn #0 #0 #4 #2'. Binding is
        compositional over a 10-token digit vocab (no unique per-scenario embedding); zero-padded
        to a fixed width for clean positional reading."""
        return "scn " + " ".join(f"#{c}" for c in str(idx).zfill(width))

    # The three assertion renderers take ``step`` and prefix it exactly as ``render_event`` does.
    # They previously accepted the argument and discarded it, so a stepped assertion parsed back
    # with step=None and the render/parse round-trip contract failed on every aux corpus document
    # (issue #38). The scored task streams are unaffected: tasks.py calls render_role and
    # render_dial without a step, and factworld.corpus is the only caller that passes one.
    def render_role(self, agent: str, role: str, step: str | None = None, key: str | None = None,
                    when: str | None = None) -> str:
        """``g3 has role r1.`` — or, with ``when``, the temporally-anchored form the
        mutual-reference family states its initial conditions in (``... r1 at the start.``).
        Appended keyword defaulting to None, so every existing call renders as before."""
        if when is None:
            s = self._pick(self._ROLE, key or f"role|{agent}").format(g=agent, r=role)
        else:
            s = self._pick(self._ROLE_AT, key or f"role|{agent}|{when}").format(g=agent, r=role, when=when)
        return f"{step} {s}" if step is not None else s

    def render_holder(self, obj: str, holder: str, step: str | None = None, key: str | None = None,
                      when: str | None = None) -> str:
        """``g3 holds o0.`` — or, with ``when``, the temporally-anchored form (see
        ``render_role``)."""
        if when is None:
            s = self._pick(self._HOLDER, key or f"holder|{obj}").format(o=obj, h=holder)
        else:
            s = self._pick(self._HOLDER_AT, key or f"holder|{obj}|{when}").format(o=obj, h=holder, when=when)
        return f"{step} {s}" if step is not None else s

    def render_dial(self, agent: str, position: str, step: str | None = None, key: str | None = None) -> str:
        """Commutative-state initial-condition line: ``g3's dial is at p2.``"""
        s = self._pick(self._DIAL, key or f"dial|{agent}").format(g=agent, p=position)
        return f"{step} {s}" if step is not None else s

    def render_query(self, family: str, *, entity=None, attribute=None, target=None, t=None,
                     targets=None) -> str:
        # as-of-t references the (t-1)-th event label; t=None means the final state
        step = None if t is None else f"s{t - 1}"
        # The mutual-reference queries. All three end "at the end", which is what separates
        # them from the single-structure families' queries and what the parser routes on;
        # the whole-map readout names its slots explicitly so the answer's ORDER is stated
        # in the prompt rather than conventional.
        if family == "s5bind_state":
            return f"what role does {target} have {self.AT_END}?"
        if family == "s5bind_bind":
            return f"who is the holder of {target} {self.AT_END}?"
        if family == "s5bind_state_all":
            return f"what role does each of {', '.join(targets)} have {self.AT_END}?"
        if family == "recall":
            return f"what is {attribute} of {entity}?"
        if family == "state_easy":
            # "where is X?" invites a list of locations; be explicit about the final holder.
            return (f"who is the final holder of {target}?" if step is None
                    else f"who holds {target} at {step}?")
        if family == "state_hard":
            return (f"what role does {target} have?" if step is None
                    else f"what role does {target} have at {step}?")
        if family == "state_comm":
            return (f"what position is {target}'s dial?" if step is None
                    else f"what position is {target}'s dial at {step}?")
        raise ValueError(f"unknown query family {family!r}")

    # ----- attached-punctuation -> canonical whitespace normalization -----
    @staticmethod
    def normalize(text: str) -> str:
        """Detach attached punctuation so scoring/parsing work on canonical whitespace tokens.

        Also strips markdown emphasis / inline-code from token edges ("**g22**", "`g22`",
        "_g22_" -> "g22") so a chat model's decoration cannot flip a correct answer to 0.
        Edge-only: namespaced ids with an internal underscore ("aux1_g0") are untouched,
        and matching stays positional over whitespace tokens (prefix-commit), so a correct
        answer buried mid-prose still scores 0.

        Examples:
            "g9's a0 is v26." -> "g9 's a0 is v26 ."
            "s1 gives o0 to g0." -> "s1 gives o0 to g0 ."
            "what is a0 of g7?" -> "what is a0 of g7 ?"
            "**g22**." -> "g22 ."
        """
        text = text.strip()
        text = re.sub(r"([a-zA-Z0-9]+)'s\b", r"\1 's", text)   # g9's -> g9 's
        text = re.sub(r"(?<=\S)([.,?!])", r" \1", text)          # v26. -> v26 .
        # markdown emphasis off token edges; tokens that were PURE markdown ("**") vanish
        toks = (t.strip(_MD_EDGE_CHARS) for t in text.split())
        return " ".join(t for t in toks if t)

    # ----- parse (exact inverse) -----
    def _typed(self, text: str):
        buckets: dict[str, list[str]] = {t: [] for t in ("e", "a", "v", "o", "loc", "g", "r", "s", "p")}
        toks = text.split()
        for tk in toks:
            c = classify(tk)
            if c:
                buckets[c].append(tk)
        return buckets, toks

    def _parse_s5_bind(self, toks: list[str], typed: dict, step: str | None) -> dict | None:
        """The five mutual-reference surfaces, or None when the text is not one of them.

        Tried before every other shape because two of the five would otherwise be swallowed
        by an earlier branch: the referenced give carries ``whose`` (the pointer-map
        reference clause) and the referenced swap carries ``swaps`` (the plain role swap).
        Each shape is recognised by a token combination no other statement type in the suite
        emits — ``at the end`` for the queries, ``swaps``+``holds`` and ``gives``+``whose``
        for the two events, and a bare ``at the start`` for the two initial-condition lines
        (checked last, since both events can also carry it).
        """
        if "end" in toks:
            if "each" in toks:
                return {"type": "query", "family": "s5bind_state_all",
                        "targets": tuple(typed["g"]), "step": step}
            if "holder" in toks:
                return {"type": "query", "family": "s5bind_bind",
                        "target": typed["o"][0], "step": step}
            return {"type": "query", "family": "s5bind_state", "target": typed["g"][0], "step": step}
        now = "point" in toks
        if "swaps" in toks and "holds" in toks:
            kind = "swap_roles_now" if now else "swap_roles_start"
            return {"type": "event", "event": Event(kind, (typed["g"][0], typed["o"][0])),
                    "step": step}
        if "gives" in toks and "whose" in toks:
            kind = "give_role_now" if now else "give_role_start"
            return {"type": "event", "event": Event(kind, (typed["o"][0], typed["r"][0])),
                    "step": step}
        if "start" in toks:
            if "holds" in toks:
                return {"type": "holder", "object": typed["o"][0], "holder": typed["g"][0],
                        "step": step, "when": self.AT_START}
            if "role" in toks and typed["r"]:
                return {"type": "role", "agent": typed["g"][0], "role": typed["r"][0],
                        "step": step, "when": self.AT_START}
        return None

    def parse(self, text: str) -> dict:
        # Normalize attached punctuation back to canonical whitespace tokens before parsing.
        text = self.normalize(text)
        typed, toks = self._typed(text)
        step = typed["s"][0] if typed["s"] else None
        rec = self._parse_s5_bind(toks, typed, step)
        if rec is not None:
            return rec
        if "?" in toks:
            # state_comm FIRST: the dial query ("what position is g3 's dial ?") contains no
            # where/who/role/e-token and would otherwise fall through to the recall fallback.
            if "dial" in toks:
                return {"type": "query", "family": "state_comm", "target": typed["g"][0], "step": step}
            # state_easy queries are interrogated with 'where'/'who'. Do NOT key on the words
            # 'holder'/'holds': the composite recall query 'what is a0 of the holder of o3 ?'
            # also contains 'holder' and would be misrouted, breaking the round-trip.
            if "where" in toks or "who" in toks:
                return {"type": "query", "family": "state_easy", "target": typed["o"][0], "step": step}
            if "role" in toks:
                return {"type": "query", "family": "state_hard", "target": typed["g"][0], "step": step}
            # recall: a plain entity (e-token) or the composite 'holder of {obj}' phrase.
            if typed["e"]:
                return {"type": "query", "family": "recall",
                        "entity": typed["e"][0], "attribute": typed["a"][0] if typed["a"] else None}
            obj = typed["o"][0] if typed["o"] else None
            return {"type": "query", "family": "recall",
                    "entity": f"the holder of {obj}" if obj else None,
                    "attribute": typed["a"][0] if typed["a"] else None, "object": obj}
        if "whose" in toks:
            # a0 swap with a state-REFERENCED operand. The record the sentence encodes is
            # (named slot, referenced value): the second slot is f^{-1}(value) under the map
            # as it stands at this event, which no parse of the sentence alone can resolve.
            return {"type": "event", "event": Event("swap_a0_ref", (typed["g"][0], typed["g"][1])),
                    "step": step}
        if "swap" in toks or "swaps" in toks:
            return {"type": "event", "event": Event("swap_role", tuple(typed["g"])), "step": step}
        if "cycle" in toks or "cycles" in toks:
            return {"type": "event", "event": Event("cycle_roles", tuple(typed["g"])), "step": step}
        if "turn" in toks or "turns" in toks:                          # commutative dial event
            amount = next(t for t in toks if t.isdigit())              # the bare click count
            return {"type": "event", "event": Event("turn_dial", (typed["g"][0], amount)), "step": step}
        if any(w in toks for w in ("move", "moved", "moves", "give", "given", "gives", "receives")):
            kind = "move" if any(w in toks for w in ("move", "moved", "moves")) else "give"
            holder = (typed["loc"] + typed["g"])[0]  # the single non-object holder (location or agent)
            return {"type": "event", "event": Event(kind, (typed["o"][0], holder)), "step": step}
        if "dial" in toks and typed["p"]:                              # dial assertion (initial-condition line)
            return {"type": "dial", "agent": typed["g"][0], "position": typed["p"][0], "step": step}
        if typed["g"] and typed["r"]:                                  # role assertion (worked-trace line)
            return {"type": "role", "agent": typed["g"][0], "role": typed["r"][0], "step": step}
        if typed["o"] and (typed["loc"] or typed["g"]):                # holder assertion (easy answer line)
            return {"type": "holder", "object": typed["o"][0], "holder": (typed["loc"] + typed["g"])[0], "step": step}
        return {"type": "fact", "entity": typed["e"][0], "attribute": typed["a"][0], "value": typed["v"][0]}

    def parse_history(self, lines) -> list[Event]:
        return [self.parse(line)["event"] for line in lines]
