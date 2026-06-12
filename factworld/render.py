"""Deterministic template renderer + its exact inverse parser.

Kept in one module on purpose: the render <-> parse round-trip is a contract (the
ground-truth re-parse check — every rendered document must parse back to the KB record it
encodes, or it is dropped). Templates expose controlled paraphrase slots (a few variants per
statement type, chosen deterministically) so the renderer can later be A/B'd against an
LLM-paraphrase pass. Content tokens are atomic IDs (`e17`, `a3`, `v42`, `o2`, `loc1`, `g4`,
`r0`, `s5`); everything else is a shared structural/function word that carries no answer signal.

Role/holder *assertion* statements (used by the auxiliary operator-world worked traces and
final answers) deliberately reuse only words already in the fact/event/query templates, so they
introduce no new vocabulary.
"""
from __future__ import annotations

import re
import zlib

from .world import Event

# Classify an atomic token by its type prefix (optionally namespaced, e.g. "aux1_g0").
# 'loc' must precede the single-char alternatives so "loc3" isn't read as an object.
_TOK = re.compile(r"^(?:[A-Za-z0-9]+_)?(loc|[eavogrs])(\d+)$")


def classify(token: str) -> str | None:
    m = _TOK.match(token)
    return m.group(1) if m else None


class Renderer:
    _FACT = ("{e} 's {a} is {v} .", "the {a} of {e} is {v} .", "{e} has {a} {v} .")
    _MOVE = ("move {o} to {h} .", "{o} is moved to {h} .")
    _GIVE = ("give {o} to {h} .", "{o} is given to {h} .")
    _SWAP = ("swap {a} {b} .", "swap the roles of {a} and {b} .")
    _CYCLE = ("cycle {ags} .", "cycle the roles of {ags} .")
    _ROLE = ("{g} has role {r} .", "the role of {g} is {r} .")
    _HOLDER = ("{o} is at {h} .",)

    @staticmethod
    def _pick(options: tuple[str, ...], key: str) -> str:
        return options[zlib.crc32(key.encode()) % len(options)]  # deterministic across runs

    # ----- render -----
    def render_fact(self, entity: str, attribute: str, value: str, key: str | None = None) -> str:
        return self._pick(self._FACT, key or f"fact|{entity}|{attribute}").format(e=entity, a=attribute, v=value)

    def render_event(self, event: Event, step: str | None = None, key: str | None = None) -> str:
        k = key or f"ev|{event.kind}|{'|'.join(event.args)}"
        if event.kind == "move":
            s = self._pick(self._MOVE, k).format(o=event.args[0], h=event.args[1])
        elif event.kind == "give":
            s = self._pick(self._GIVE, k).format(o=event.args[0], h=event.args[1])
        elif event.kind == "swap_role":
            s = self._pick(self._SWAP, k).format(a=event.args[0], b=event.args[1])
        elif event.kind == "cycle_roles":
            s = self._pick(self._CYCLE, k).format(ags=" ".join(event.args))
        else:
            raise ValueError(f"unknown event kind {event.kind!r}")
        return f"{step} : {s}" if step is not None else s

    def render_history(self, events, with_steps: bool = False) -> list[str]:
        return [
            self.render_event(e, step=f"s{i}" if with_steps else None,
                              key=f"h|{i}|{e.kind}|{'|'.join(e.args)}")
            for i, e in enumerate(events)
        ]

    def render_scenario(self, idx: int, width: int = 4) -> str:
        """Scenario id as a marker + shared digit tokens, e.g. 'scn #0 #0 #4 #2'. Binding is
        compositional over a 10-token digit vocab (no unique per-scenario embedding); zero-padded
        to a fixed width for clean positional reading. Used to bind an IWL query to a stored history."""
        return "scn " + " ".join(f"#{c}" for c in str(idx).zfill(width))

    def render_role(self, agent: str, role: str, step: str | None = None, key: str | None = None) -> str:
        s = self._pick(self._ROLE, key or f"role|{agent}").format(g=agent, r=role)
        return f"{step} : {s}" if step is not None else s

    def render_holder(self, obj: str, holder: str, step: str | None = None, key: str | None = None) -> str:
        s = self._pick(self._HOLDER, key or f"holder|{obj}").format(o=obj, h=holder)
        return f"{step} : {s}" if step is not None else s

    def render_query(self, family: str, *, entity=None, attribute=None, target=None, t=None) -> str:
        # as-of-t references the (t-1)-th event label; t=None means the final state
        step = None if t is None else f"s{t - 1}"
        if family == "recall":
            return f"what is {attribute} of {entity} ?"
        if family == "state_easy":
            return f"where is {target} ?" if step is None else f"where is {target} at {step} ?"
        if family == "state_hard":
            return f"what role does {target} have ?" if step is None else f"what role does {target} have at {step} ?"
        raise ValueError(f"unknown query family {family!r}")

    # ----- parse (exact inverse) -----
    def _typed(self, text: str):
        buckets: dict[str, list[str]] = {t: [] for t in ("e", "a", "v", "o", "loc", "g", "r", "s")}
        toks = text.split()
        for tk in toks:
            c = classify(tk)
            if c:
                buckets[c].append(tk)
        return buckets, toks

    def parse(self, text: str) -> dict:
        typed, toks = self._typed(text)
        step = typed["s"][0] if typed["s"] else None
        if "?" in toks:
            if "where" in toks:
                return {"type": "query", "family": "state_easy", "target": typed["o"][0], "step": step}
            if "role" in toks:
                return {"type": "query", "family": "state_hard", "target": typed["g"][0], "step": step}
            return {"type": "query", "family": "recall", "entity": typed["e"][0], "attribute": typed["a"][0]}
        if "swap" in toks:
            return {"type": "event", "event": Event("swap_role", tuple(typed["g"])), "step": step}
        if "cycle" in toks:
            return {"type": "event", "event": Event("cycle_roles", tuple(typed["g"])), "step": step}
        if any(w in toks for w in ("move", "moved", "give", "given")):
            kind = "move" if ("move" in toks or "moved" in toks) else "give"
            holder = (typed["loc"] + typed["g"])[0]  # the single non-object holder (location or agent)
            return {"type": "event", "event": Event(kind, (typed["o"][0], holder)), "step": step}
        if typed["g"] and typed["r"]:                                  # role assertion (worked-trace line)
            return {"type": "role", "agent": typed["g"][0], "role": typed["r"][0], "step": step}
        if typed["o"] and (typed["loc"] or typed["g"]):                # holder assertion (easy answer line)
            return {"type": "holder", "object": typed["o"][0], "holder": (typed["loc"] + typed["g"])[0], "step": step}
        return {"type": "fact", "entity": typed["e"][0], "attribute": typed["a"][0], "value": typed["v"][0]}

    def parse_history(self, lines) -> list[Event]:
        return [self.parse(line)["event"] for line in lines]
