"""Resolve the RL flicker: is s1's holder-emergence a slow climb or noise? Power it up properly.

The n=3 GRPO run (rl_grpo.py) gave RL value ≈ floor with ONE seed (s1) showing holder emergence (0.15-0.23) —
unresolvable at that power. Here: 5 seeds, 5× steps (7500 GRPO), with the reward TRAJECTORY and MID-TRAINING
evals logged, so we can see whether reward/holder climb over training (slow bootstrap) or stay flat (noise).
  - reward/holder rising across 7500 steps on multiple seeds -> RL IS climbing; escalate (positive headline).
  - flat/floor with at most isolated flickers -> defensible negative at this scale/recipe.

Same setup as rl_grpo (d256, answer-only SFT warmup then GRPO, outcome 0/1 reward, free scratchpad).

  .venv/bin/python followups/non-abelian-state/rl_flicker.py
"""
from __future__ import annotations

import os
import statistics
import sys
from collections import defaultdict
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import rl_grpo as RG
from rl_grpo import _world, make_prompt, make_warmup_docs, generate_group, reward_and_state, grpo_loss, evaluate

SEEDS = [0, 1, 2, 3, 4]
STEPS = 7500
EVAL_AT = [0, 1500, 3750, 7500]   # mid-training eval checkpoints (does value/holder climb?)
EVAL_LEN = [16, 64]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_flicker.md")


def train_traj(model: Any, tok: Any, w: Any, r: Any, origins: dict, oracle: Any, seed: int,
               evs: dict, device: str = "cuda") -> tuple[list, dict]:
    """GRPO with reward-trajectory + mid-training eval. Returns (traj, ckpt_evals).

    Args:
        model: the SFT-warmed ``HybridLM`` (mutated in place by GRPO).
        tok: the atomic tokenizer.
        w: the FactWorld ``World``.
        r: the ``Renderer``.
        origins: the fixed agent -> a0-value map (parametric recall).
        oracle: the symbolic ``Oracle``.
        seed: RNG seed for prompt sampling.
        evs: maps eval length -> a list of pre-built eval prompts.
        device: torch device.

    Returns:
        A ``(traj, ckpt)`` pair: ``traj`` is a list of ``(step, mean_reward)`` points and
        ``ckpt`` maps checkpoint step -> {length: (value, holder)} evals.
    """
    import random
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=RG.LR, weight_decay=0.0)
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    ag_ids = {tok.token_to_id[g] for g in w.agents}
    rng = random.Random(9000 + seed)
    traj: list = []
    ckpt: dict = {}
    if 0 in EVAL_AT:
        ckpt[0] = {L: evaluate(model, tok, w, evs[L], device) for L in EVAL_LEN}
    running = []
    for step in range(1, STEPS + 1):
        batch = [make_prompt(w, r, origins, oracle, rng.choice(RG.TRAIN_LEN), rng) for _ in range(RG.PROMPTS_PER_STEP)]
        opt.zero_grad(); step_rew = []
        for prompt, holder, value in batch:
            pids = list(tok.encode(prompt))
            model.eval()
            comps = generate_group(model, tok, pids, RG.GROUP, device)
            rew = [reward_and_state(c, tok, val_ids, ag_ids, holder, value)[0] for c in comps]
            step_rew.extend(rew)
            m, sd = sum(rew) / len(rew), (statistics.pstdev(rew) if len(set(rew)) > 1 else 0.0)
            if sd == 0:
                continue
            adv = [(x - m) / (sd + 1e-6) for x in rew]
            model.train()
            grpo_loss(model, tok, pids, comps, adv, device).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(sum(step_rew) / len(step_rew))
        if step % 250 == 0:
            mr = sum(running[-250:]) / len(running[-250:])
            traj.append((step, round(mr, 3)))
            print(f"    s{seed} step {step}: reward {mr:.3f}", flush=True)
        if step in EVAL_AT:
            ckpt[step] = {L: evaluate(model, tok, w, evs[L], device) for L in EVAL_LEN}
            print(f"    s{seed} EVAL@{step}: " + " ".join(
                f"L{L} v={ckpt[step][L][0]:.3f} h={ckpt[step][L][1]:.3f}" for L in EVAL_LEN), flush=True)
    return traj, ckpt


def main() -> None:
    """Warm-start per seed, run 7500-step GRPO with trajectory + mid-eval logging, and tabulate the climb."""
    import random
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    warm_docs = make_warmup_docs(w, r, origins, oracle, 8000, 2)
    tok, docs, _ = T.prepare(warm_docs, [], [w])
    evs = {L: [make_prompt(w, r, origins, oracle, L, random.Random(900 + L + j)) for j in range(150)]
           for L in EVAL_LEN}
    agg = defaultdict(dict)
    print("=== RL FLICKER RESOLUTION: 5 seeds x 7500 GRPO steps, trajectory + mid-eval ===", flush=True)
    for s in SEEDS:
        model = T.run("gdp_hybrid", tok, docs, [], steps=2500, batch=32, d_model=256, n_layers=4,
                      d_ff=1024, seed=s, return_model=True)["model"]
        traj, ckpt = train_traj(model, tok, w, r, origins, oracle, s, evs)
        agg[s] = {"traj": traj, "ckpt": ckpt}
        del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("rl_flicker done.", flush=True)


def write_md(agg: dict) -> None:
    """Write the per-seed value/holder/reward-trajectory table to ``OUT``.

    Args:
        agg: maps seed -> {"traj": [...], "ckpt": {...}}.
    """
    lines = [
        "# RL flicker resolution — slow climb or noise? (5 seeds × 7500 GRPO steps)\n",
        "`followups/non-abelian-state/rl_flicker.py`. gdp_hybrid d256, answer-only SFT warmup then GRPO 7500 "
        "steps, outcome 0/1 reward. Per-seed composite value `v` / holder `h` at mid-training checkpoints, and "
        "the reward trajectory. Climb across checkpoints on multiple seeds = RL bootstraps; flat = negative. "
        "Floor = 0.20.\n",
        "| seed | v@0 | v@3750 | v@7500 | h@7500 (L16) | reward 0→7500 |",
        "|---|---|---|---|---|---|",
    ]
    for s in sorted(agg):
        c = agg[s].get("ckpt", {}); tr = agg[s].get("traj", [])
        def v(step: int) -> str:
            """Format L16 composite value at ``step`` (``…`` if not checkpointed)."""
            return f"{c[step][16][0]:.2f}" if step in c else "…"
        h7 = f"{c[7500][16][1]:.2f}" if 7500 in c else "…"
        rwd = f"{tr[0][1]:.2f}→{tr[-1][1]:.2f}" if tr else "…"
        lines.append(f"| s{s} | {v(0)} | {v(3750)} | {v(7500)} | {h7} | {rwd} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
