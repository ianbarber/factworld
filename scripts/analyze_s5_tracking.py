"""Diagnose how many S₅ ops a frontier model can track.

For each stored per-example prediction from an s5 length sweep, regenerate the example's exact
queried-agent role trajectory (init + per-step, via the oracle) and measure:

  - acc       : pred == final role (the canonical relaxed score)
  - horizon   : the latest step t (0..L) at which the agent's role equaled the model's pred.
                If the model tracked faithfully to the end, horizon == L. If it stalled at an
                intermediate state, horizon < L. (Noisy when a role recurs late by chance —
                read the trend across lengths, not single cells.)
  - missed    : L - horizon, the trailing ops the model's answer failed to absorb.

The trajectory is recomputed from the same deterministic seeds the task uses (``_world`` /
``_rng`` / ``sample_hard_chain``), aligned to the stored examples by index — no prompt parsing.

Usage:
    python scripts/analyze_s5_tracking.py docs/openrouter/s5-length-sweep.jsonl
"""
from __future__ import annotations

import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld.tasks import CANONICAL, _world, _rng, content_tokens  # noqa: E402


def trajectory(spec, w, oracle, length, idx):
    """The queried agent and its role over [init, s1, ..., s_length] for example idx."""
    events = w.sample_hard_chain(length, episode_seed=f"{spec.name}|{idx}")
    agent = _rng(spec, "test", length, idx).choice(w.agents)
    roles = [oracle.hard_assignment(events, t)[agent] for t in range(length + 1)]
    return agent, roles


def pred_role(pred_text):
    ct = content_tokens(pred_text)
    return ct[0] if ct else None


def main(jsonl):
    spec = CANONICAL["s5_v1"]
    w, _r, oracle = _world(spec)
    rows = [json.loads(line) for line in open(jsonl) if line.strip()]
    # s5 rows only, grouped by model then length
    rows = [r for r in rows if r.get("task") == "s5_v1"]
    rows.sort(key=lambda r: (r["model"], r["length"]))

    print(f"{'model':<30} {'L':>4} {'acc':>5} {'horizon':>8} {'missed':>7} {'frac≥L-1':>9} {'n':>3}")
    print("-" * 70)
    for row in rows:
        L = row["length"]
        n = len(row.get("examples", [])) or row.get("n", 0)
        acc, horizons, missed, near = [], [], [], []
        for i, e in enumerate(row["examples"]):
            _agent, roles = trajectory(spec, w, oracle, L, i)
            gold = roles[-1]
            pred = pred_role(e.get("pred", ""))
            acc.append(pred == gold)
            if pred is None:
                horizons.append(0); missed.append(L); near.append(False); continue
            hits = [t for t, rl in enumerate(roles) if rl == pred]
            h = max(hits) if hits else -1
            horizons.append(h)
            missed.append(L - h if h >= 0 else L)
            near.append(h >= L - 1)  # correct or off by only the last op
        print(f"{row['model']:<30} {L:>4} {sum(acc)/n:>5.2f} {statistics.mean(horizons):>8.1f} "
              f"{statistics.mean(missed):>7.1f} {sum(near)/n*100:>8.0f}% {n:>3}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/openrouter/s5-length-sweep.jsonl")
