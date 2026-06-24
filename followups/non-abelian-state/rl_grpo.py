"""RL vs static (the big rigor thread): does OUTCOME-ONLY reward climb the supervision cliff that SFT can't?

Our cliff (supervision_sweep.py) is a STATIC-SFT result: answer-only next-token training floors on the
non-abelian composite. But agentic training is RL/outcome-reward, whose credit assignment differs — exploration
+ advantage-weighting can route a sparse terminal reward back to the intermediate computation in ways SFT
cannot. So "process supervision is required" may be an artifact of static SFT, not a property of the task.

This is the decisive test: GRPO from scratch, reward = the composite answer is correct (0/1), the model free to
generate its own scratchpad before answering. Same architecture/world as the SFT runs. Baselines (static SFT,
train len 16): answer-only = 0.20 floor; dense per-step = ~1.0.
  - RL >> 0.20  -> outcome reward climbs the cliff; "process supervision needed" was an SFT artifact.
  - RL ~ 0.20   -> the cliff is fundamental to the task at this scale, not specific to SFT.
We also log the holder-resolution rate (state leg) to see whether RL learns the state even without rewarding it.

Two experiments live here:
  default  — the GRPO core: 3 seeds, 1500 GRPO steps, within-seed SFT-vs-RL contrast (main()).
  --flicker — the powered-up flicker-resolution variant: 5 seeds, 7500 GRPO steps, with the reward TRAJECTORY
             and MID-TRAINING evals logged, so we can tell whether reward/holder climb over training (slow
             bootstrap) or stay flat (noise). Resolves the n=3 run's lone holder-emergence flicker on s1.

  .venv/bin/python followups/non-abelian-state/rl_grpo.py            # GRPO core (3 seeds, 1500 steps)
  .venv/bin/python followups/non-abelian-state/rl_grpo.py --flicker  # flicker resolution (5 seeds, 7500 steps)
"""
from __future__ import annotations

import os
import random
import statistics
import sys
from collections import defaultdict
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dense_capstone import _world

SEEDS = [0, 1, 2]
TRAIN_LEN = (4, 8, 16)
EVAL_LEN = [16, 64]
STEPS = 1500
PROMPTS_PER_STEP = 16
GROUP = 8            # GRPO completions per prompt
MAX_NEW = 10         # room for a short scratchpad + answer
TEMP = 1.0
LR = 1e-4
ENT_COEF = 0.01
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_grpo.md")

# --- --flicker variant: powered-up trajectory + mid-training eval ---
FLICKER_SEEDS = [0, 1, 2, 3, 4]
FLICKER_STEPS = 7500
EVAL_AT = [0, 1500, 3750, 7500]   # mid-training eval checkpoints (does value/holder climb?)
FLICKER_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_flicker.md")


def make_prompt(w: Any, r: Any, origins: dict, oracle: Any, L: int, rng: Any) -> tuple[str, str, str]:
    """Non-abelian composite, parametric recall. Returns (prompt_str, gold_holder, gold_value).

    Args:
        w: the FactWorld ``World``.
        r: the ``Renderer``.
        origins: the fixed agent -> a0-value map (parametric recall).
        oracle: the symbolic ``Oracle``.
        L: chain length.
        rng: a ``random.Random`` for sampling the episode and role.

    Returns:
        A ``(prompt_str, gold_holder, gold_value)`` triple.
    """
    ev = w.sample_hard_chain(L, episode_seed=f"rl|{rng.random()}")
    inv = {ro: ag for ag, ro in oracle.hard_assignment(ev).items()}
    role = rng.choice(w.roles)
    holder = inv[role]
    hist = " ".join(r.render_history(tuple(ev), with_steps=True))
    return f"{hist} what is a0 of the holder of role {role} ? : ", holder, origins[holder]


def make_warmup_docs(w: Any, r: Any, origins: dict, oracle: Any, n: int, seed: int) -> list[str]:
    """Answer-only SFT docs (the static baseline that floors): prompt + gold value. RL starts from this model,
    so the comparison is within-seed: does outcome-reward RL push past the static answer-only SFT it begins at?

    Args:
        w: the FactWorld ``World``.
        r: the ``Renderer``.
        origins: the fixed agent -> a0-value map (parametric recall).
        oracle: the symbolic ``Oracle``.
        n: number of docs to generate.
        seed: RNG seed for sampling.

    Returns:
        A list of ``n`` answer-only training strings.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        prompt, _holder, value = make_prompt(w, r, origins, oracle, rng.choice(TRAIN_LEN), rng)
        out.append(f"{prompt}{value} .")
    return out


def generate_group(model: Any, tok: Any, prompt_ids: list, G: int, device: str) -> list:
    """Sample G completions for one prompt (no grad). Returns list of completion-id lists (no prompt).

    Args:
        model: the policy model.
        tok: the atomic tokenizer.
        prompt_ids: the prompt token ids.
        G: number of completions to sample.
        device: torch device.

    Returns:
        A list of ``G`` completion-id lists (each excludes the prompt).
    """
    import torch
    dot = tok.token_to_id["."]
    base = torch.tensor([prompt_ids] * G, device=device)            # (G, P) identical prompt
    comps = [[] for _ in range(G)]
    done = [False] * G
    cur = base
    with torch.no_grad():
        for _ in range(MAX_NEW):
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(cur)[:, -1].float()
            probs = torch.softmax(logits / TEMP, dim=-1)
            nxt = torch.multinomial(probs, 1)                        # (G,1)
            cur = torch.cat([cur, nxt], dim=1)
            for g in range(G):
                if done[g]:
                    continue
                t = int(nxt[g, 0])
                comps[g].append(t)
                if t == dot:
                    done[g] = True
            if all(done):
                break
    return comps


def reward_and_state(comp: list, tok: Any, val_ids: Any, ag_ids: Any, gold_holder: str,
                     gold_value: str) -> tuple[float, float]:
    """(reward, holder_resolved): reward=1 if first emitted value == gold value (the composite answer).

    Args:
        comp: one completion's token ids.
        tok: the atomic tokenizer.
        val_ids: the set of value-token ids.
        ag_ids: the set of agent-token ids.
        gold_holder: the gold holder agent token.
        gold_value: the gold composite value token.

    Returns:
        A ``(reward, holder_resolved)`` pair of 0/1 floats.
    """
    first_val = next((t for t in comp if t in val_ids), None)
    first_ag = next((t for t in comp if t in ag_ids), None)
    rew = 1.0 if first_val is not None and first_val == tok.token_to_id[gold_value] else 0.0
    holder_ok = 1.0 if first_ag is not None and first_ag == tok.token_to_id[gold_holder] else 0.0
    return rew, holder_ok


def grpo_loss(model: Any, tok: Any, prompt_ids: list, comps: list, adv: list, device: str) -> Any:
    """Policy-gradient loss over one group: -sum_t logprob(comp_t) * advantage, + entropy bonus.

    Args:
        model: the policy model.
        tok: the atomic tokenizer.
        prompt_ids: the prompt token ids.
        comps: the group's completion-id lists.
        adv: per-completion advantages.
        device: torch device.

    Returns:
        The scalar GRPO loss tensor (backward-able).
    """
    import torch
    import torch.nn.functional as F
    seqs = [prompt_ids + c for c in comps]
    ml = max(len(s) for s in seqs)
    pad = tok.pad_id
    inp = torch.full((len(seqs), ml), pad, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        inp[i, : len(s)] = torch.tensor(s, device=device)
    P = len(prompt_ids)
    with torch.autocast(device, dtype=torch.bfloat16):
        logits = model(inp)[:, :-1].float()                          # predict token t+1 from t
    logp = F.log_softmax(logits, dim=-1)
    probs = logp.exp()
    ent = -(probs * logp).sum(-1)                                    # (G, ml-1) per-position entropy
    total = 0.0
    ent_total = 0.0
    ntok = 0
    for i, c in enumerate(comps):
        for j in range(len(c)):                                      # completion token j sits at position P+j
            pos = P + j - 1                                          # predicted from previous position
            total = total - logp[i, pos, c[j]] * adv[i]
            ent_total = ent_total + ent[i, pos]
            ntok += 1
    return (total - ENT_COEF * ent_total) / max(1, ntok)


def evaluate(model: Any, tok: Any, w: Any, exs: list, device: str) -> tuple[float, float]:
    """Greedy composite value-accuracy + holder-resolution rate.

    Args:
        model: the model to evaluate.
        tok: the atomic tokenizer.
        w: the FactWorld ``World``.
        exs: a list of ``(prompt, holder, value)`` eval examples.
        device: torch device.

    Returns:
        A ``(value_accuracy, holder_resolution_rate)`` pair.
    """
    import torch
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    ag_ids = {tok.token_to_id[g] for g in w.agents}
    dot = tok.token_to_id["."]
    model.eval()
    v_ok = h_ok = 0
    with torch.no_grad():
        for prompt, holder, value in exs:
            ids = list(tok.encode(prompt)); fa = fv = None
            for _ in range(MAX_NEW):
                with torch.autocast(device, dtype=torch.bfloat16):
                    nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                if fa is None and nx in ag_ids: fa = nx
                if fv is None and nx in val_ids: fv = nx
                ids.append(nx)
                if nx == dot: break
            v_ok += int(fv == tok.token_to_id[value]); h_ok += int(fa == tok.token_to_id[holder])
    return v_ok / len(exs), h_ok / len(exs)


def train_rl(model: Any, tok: Any, w: Any, r: Any, origins: dict, oracle: Any, seed: int,
             device: str = "cuda") -> Any:
    """GRPO on a pre-warmed model (answer-only SFT). Returns the RL-updated model.

    Args:
        model: the SFT-warmed model (mutated in place).
        tok: the atomic tokenizer.
        w: the FactWorld ``World``.
        r: the ``Renderer``.
        origins: the fixed agent -> a0-value map (parametric recall).
        oracle: the symbolic ``Oracle``.
        seed: RNG seed for prompt sampling.
        device: torch device.

    Returns:
        The same ``model``, RL-updated in place.
    """
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    ag_ids = {tok.token_to_id[g] for g in w.agents}
    rng = random.Random(9000 + seed)
    running = []
    for step in range(STEPS):
        batch = [make_prompt(w, r, origins, oracle, rng.choice(TRAIN_LEN), rng) for _ in range(PROMPTS_PER_STEP)]
        opt.zero_grad()
        step_rew = []
        for prompt, holder, value in batch:
            pids = list(tok.encode(prompt))
            model.eval()                                             # rollouts: no_grad, fused_recurrent decode
            comps = generate_group(model, tok, pids, GROUP, device)
            rs = [reward_and_state(c, tok, val_ids, ag_ids, holder, value) for c in comps]
            rew = [x[0] for x in rs]
            step_rew.extend(rew)
            m, sd = (sum(rew) / len(rew)), (statistics.pstdev(rew) if len(set(rew)) > 1 else 0.0)
            if sd == 0:
                continue                                             # no learning signal in this group
            adv = [(x - m) / (sd + 1e-6) for x in rew]
            model.train()                                            # loss: chunk kernel (backward-able)
            loss = grpo_loss(model, tok, pids, comps, adv, device)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(sum(step_rew) / len(step_rew))
        if step % 100 == 0:
            print(f"    s{seed} step {step}: mean reward {sum(running[-100:]) / len(running[-100:]):.3f}", flush=True)
    return model


def train_traj(model: Any, tok: Any, w: Any, r: Any, origins: dict, oracle: Any, seed: int,
               evs: dict, device: str = "cuda") -> tuple[list, dict]:
    """GRPO (the --flicker variant) with reward-trajectory + mid-training eval. Returns (traj, ckpt_evals).

    Same GRPO update as ``train_rl`` (group advantage, entropy-bonus loss, lr LR), but runs ``FLICKER_STEPS``
    and logs the reward trajectory (every 250 steps) and greedy value/holder evals at the ``EVAL_AT``
    checkpoints, so a slow bootstrap can be distinguished from noise.

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
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    ag_ids = {tok.token_to_id[g] for g in w.agents}
    rng = random.Random(9000 + seed)
    traj: list = []
    ckpt: dict = {}
    if 0 in EVAL_AT:
        ckpt[0] = {L: evaluate(model, tok, w, evs[L], device) for L in EVAL_LEN}
    running = []
    for step in range(1, FLICKER_STEPS + 1):
        batch = [make_prompt(w, r, origins, oracle, rng.choice(TRAIN_LEN), rng) for _ in range(PROMPTS_PER_STEP)]
        opt.zero_grad(); step_rew = []
        for prompt, holder, value in batch:
            pids = list(tok.encode(prompt))
            model.eval()
            comps = generate_group(model, tok, pids, GROUP, device)
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
    """SFT-warm per seed, run GRPO from that model, and tabulate within-seed SFT vs RL composite accuracy."""
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
    print("=== RL (GRPO, outcome-only reward) vs static SFT on the non-abelian composite ===", flush=True)
    print("    within-seed contrast: answer-only SFT warmup, then GRPO from that model", flush=True)
    for s in SEEDS:
        # static baseline: answer-only SFT (the regime that floors)
        model = T.run("gdp_hybrid", tok, docs, [], steps=2500, batch=32, d_model=256, n_layers=4,
                      d_ff=1024, seed=s, return_model=True)["model"]
        for L in EVAL_LEN:
            agg[L][s] = {"sft": evaluate(model, tok, w, evs[L], "cuda")}
        print(f"  SFT s{s} :: " + "  ".join(f"L{L} v={agg[L][s]['sft'][0]:.3f}" for L in EVAL_LEN), flush=True)
        # then GRPO from that same model
        model = train_rl(model, tok, w, r, origins, oracle, s)
        for L in EVAL_LEN:
            agg[L][s]["rl"] = evaluate(model, tok, w, evs[L], "cuda")
        print(f"  RL  s{s} :: " + "  ".join(
            f"L{L} v={agg[L][s]['rl'][0]:.3f} h={agg[L][s]['rl'][1]:.3f}" for L in EVAL_LEN), flush=True)
        del model; torch.cuda.empty_cache()
        write_md(agg)
    write_md(agg)
    print("rl_grpo done.", flush=True)


def write_md(agg: dict) -> None:
    """Write the SFT-vs-RL value/holder table (mean ± pstdev over seeds) to ``OUT``.

    Args:
        agg: maps eval length -> seed -> {"sft": (...), "rl": (...)}.
    """
    lines = [
        "# RL (GRPO, outcome-only reward) vs static SFT — does reward climb the supervision cliff?\n",
        "`followups/non-abelian-state/rl_grpo.py`. gdp_hybrid d256x4, 3 seeds. Within-seed contrast: answer-only "
        "SFT (2500 steps) — the static regime that floors — then GRPO (group 8, outcome 0/1 reward on the "
        "composite answer, free scratchpad, 1500 steps) from that same model. `value` = composite accuracy "
        "(greedy); `holder` = state-leg resolution rate (never rewarded directly). Floor = 0.20. RL value >> SFT "
        "value would mean outcome reward climbs the cliff static SFT cannot.\n",
        "| eval | SFT value | RL value | RL holder |",
        "|---|---|---|---|",
    ]
    for L in EVAL_LEN:
        d = agg.get(L)
        if not d:
            continue
        sv = [x["sft"][0] for x in d.values() if "sft" in x]
        rv = [x["rl"][0] for x in d.values() if "rl" in x]
        rh = [x["rl"][1] for x in d.values() if "rl" in x]
        sft = f"{statistics.mean(sv):.2f}±{statistics.pstdev(sv):.2f}" if sv else "…"
        rlv = f"{statistics.mean(rv):.2f}±{statistics.pstdev(rv):.2f}" if rv else "…"
        rlh = f"{statistics.mean(rh):.2f}±{statistics.pstdev(rh):.2f}" if rh else "…"
        lines.append(f"| L{L} | {sft} | {rlv} | {rlh} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


def main_flicker() -> None:
    """Warm-start per seed, run 7500-step GRPO with trajectory + mid-eval logging, and tabulate the climb."""
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
    for s in FLICKER_SEEDS:
        model = T.run("gdp_hybrid", tok, docs, [], steps=2500, batch=32, d_model=256, n_layers=4,
                      d_ff=1024, seed=s, return_model=True)["model"]
        traj, ckpt = train_traj(model, tok, w, r, origins, oracle, s, evs)
        agg[s] = {"traj": traj, "ckpt": ckpt}
        del model; torch.cuda.empty_cache()
        write_md_flicker(agg)
    write_md_flicker(agg)
    print("rl_flicker done.", flush=True)


def write_md_flicker(agg: dict) -> None:
    """Write the per-seed value/holder/reward-trajectory table to ``FLICKER_OUT``.

    Args:
        agg: maps seed -> {"traj": [...], "ckpt": {...}}.
    """
    lines = [
        "# RL flicker resolution — slow climb or noise? (5 seeds × 7500 GRPO steps)\n",
        "`followups/non-abelian-state/rl_grpo.py --flicker`. gdp_hybrid d256, answer-only SFT warmup then GRPO "
        "7500 steps, outcome 0/1 reward. Per-seed composite value `v` / holder `h` at mid-training checkpoints, "
        "and the reward trajectory. Climb across checkpoints on multiple seeds = RL bootstraps; flat = negative. "
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
    with open(FLICKER_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if "--flicker" in sys.argv:
        main_flicker()
    else:
        main()
