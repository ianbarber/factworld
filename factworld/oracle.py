"""The symbolic ground-truth solver.

Every eval item resolves through the oracle: it is correct *by construction* and is the 100%
reference that the validity gate demands (if the oracle is ever <100%, the eval is wrong).
It also emits worked traces (intermediate states) used for the aux operator-world supervision
(Option A) and for as-of-t queries.
"""
from __future__ import annotations

from .world import Event, World


class Oracle:
    def __init__(self, world: World):
        self.world = world

    # --- recall (static attribute lookup) ---
    def recall(self, entity: str, attribute: str) -> str:
        return self.world.attrs[entity][attribute]

    # --- easy-state (last-write-wins over a single holder slot) ---
    def easy_holder(self, events: list[Event], obj: str, t: int | None = None) -> str:
        """Holder of `obj` after the first `t` events (t=None → the whole chain)."""
        holder = self.world.initial_holder[obj]
        for e in events if t is None else events[:t]:
            if e.kind in ("move", "give") and e.args[0] == obj:
                holder = e.args[1]
        return holder

    def easy_trace(self, events: list[Event], obj: str) -> list[str]:
        """Holder of `obj` after each prefix, including the initial state (length len+1)."""
        out = [self.world.initial_holder[obj]]
        for e in events:
            if e.kind in ("move", "give") and e.args[0] == obj:
                out.append(e.args[1])
            else:
                out.append(out[-1])
        return out

    # --- commutative-state (per-agent dial accumulation mod k_positions; abelian rung) ---
    def comm_position(self, initial: dict[str, str], events, agent: str, t: int | None = None) -> str:
        """Dial position of `agent` after the first `t` events (t=None → the whole chain).

        `initial` maps agent -> starting position token (per-example, stated in the prompt).
        Addition mod k_positions is commutative: EVERY matching event contributes to the sum,
        but their order does not matter — the abelian rung between last-write-wins and S_k.
        """
        pos = self.world.positions.index(initial[agent])
        for e in (events if t is None else events[:t]):
            if e.kind == "turn_dial" and e.args[0] == agent:
                pos = (pos + int(e.args[1])) % len(self.world.positions)
        return self.world.positions[pos]

    def comm_trace(self, initial: dict[str, str], events, agent: str) -> list[str]:
        """Position of `agent` after each prefix, including the initial state (length len+1).

        Worked-trace supervision parity with `easy_trace`/`hard_trace`.
        """
        out = [initial[agent]]
        k = len(self.world.positions)
        for e in events:
            if e.kind == "turn_dial" and e.args[0] == agent:
                idx = (self.world.positions.index(out[-1]) + int(e.args[1])) % k
                out.append(self.world.positions[idx])
            else:
                out.append(out[-1])
        return out

    # --- hard-state (composition of S_k role permutations) ---
    @staticmethod
    def _apply(assignment: dict[str, str], e: Event) -> dict[str, str]:
        a = dict(assignment)
        if e.kind == "swap_role":
            x, y = e.args
            a[x], a[y] = a[y], a[x]
        elif e.kind == "cycle_roles":
            cyc = e.args
            old = [assignment[ag] for ag in cyc]
            for i, ag in enumerate(cyc):
                a[ag] = old[(i - 1) % len(cyc)]
        else:
            raise ValueError(f"not a role-permutation event: {e.kind!r}")
        return a

    def hard_assignment(self, events: list[Event], t: int | None = None) -> dict[str, str]:
        a = dict(self.world.initial_assignment)
        for e in events if t is None else events[:t]:
            a = self._apply(a, e)
        return a

    def hard_role(self, events: list[Event], agent: str, t: int | None = None) -> str:
        return self.hard_assignment(events, t)[agent]

    def hard_trace(self, events: list[Event]) -> list[dict[str, str]]:
        """Full assignment after each prefix, including the initial state (length len+1).

        This is the worked-trace supervision for the auxiliary operator worlds.
        """
        states = [dict(self.world.initial_assignment)]
        for e in events:
            states.append(self._apply(states[-1], e))
        return states
