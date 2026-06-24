"""Scheduled sampling (free-running exposure) in BASE training: can it raise the clean-base rate?

base_reliability.py found dense_h16 (teacher-forced per-step holder) saturated at 1.00 for ALL bases -> the
fragility is FREE-RUNNING autoregressive stability (exposure bias), not learning the map. Attack it directly:
during base training, at holder-checkpoint slots, with probability p(step) replace the gold holder token in the
CONTEXT with the model's own argmax, so the recurrence learns to stay stable on its own (sometimes-wrong) state.
Loss targets stay gold throughout. p ramps 0 -> p_max over the first half of training.

Arms (K seeds), metric = clean-base fraction (free-running answer-only L16 e2e >= 0.95), + EMA readout + L128
(native length-gen check):
  baseline   p=0 (== default; the 1/8 anchor)
  ss         p_max=0.5, default gate
  ss+gate    p_max=0.5 + GDP forget-gate retention-init (the base_reliability modest lever, stacked)

  .venv/bin/python followups/non-abelian-state/schedsample.py
"""
from __future__ import annotations

import copy
import math
import os
import random
import statistics
import sys
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dense_capstone import _world, e2e_eval
from supervision_sweep import build as build_ck
from post_state import make_pool
from base_reliability import apply_gate_init     # reuse the gate-retention-init

SEEDS = list(range(6))
ARMS = [("baseline", 0.0, "default"), ("ss", 0.5, "default"), ("ss+gate", 0.5, "slow")]
STEPS, CLEAN_THR, N_EVAL = 6000, 0.95, 150
D_MODEL, N_LAYERS, D_FF = 384, 6, 1536
EVAL_LEN = [16, 128]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schedsample.md")


def train_ss(tok: Any, pool: list[str], seed: int, p_max: float, gate: str,
             device: str = "cuda") -> tuple[Any, Any]:
    """Train a gdp_hybrid base with scheduled-sampling (free-running) holder exposure.

    Args:
        tok: the atomic tokenizer.
        pool: the short in-distribution training strings.
        seed: RNG seed for init and minibatch sampling.
        p_max: peak scheduled-sampling probability (ramped 0 -> p_max over the first half).
        gate: the GDP forget-gate retention-init mode passed to ``apply_gate_init``.
        device: torch device.

    Returns:
        A ``(model, ema_model)`` pair: the final model and its cosine-tail weight average.
    """
    import torch
    import torch.nn.functional as F
    from factworld.models import build_model
    torch.manual_seed(seed)
    model = build_model("gdp_hybrid", tok.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=4,
                        d_ff=D_FF, num_householder=4, allow_neg_eigval=True).to(device)
    apply_gate_init(model, gate)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warmup, pad = 1000, tok.pad_id
    holder_id = tok.token_to_id["holder"]
    rng = random.Random(7000 + seed)
    model.train()
    for step in range(STEPS):
        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * (min(1.0, (step + 1) / warmup) if step < warmup else
                               0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, STEPS - warmup))))
        p = p_max * min(1.0, step / (STEPS * 0.5))           # ramp 0 -> p_max over first half
        seqs = [tok.encode(d) for d in rng.sample(pool, 32)]
        ml = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), ml), pad, dtype=torch.long, device=device)
        for ri, s in enumerate(seqs):
            inp[ri, : len(s)] = torch.tensor(s, device=device)
        x = inp[:, :-1].clone(); tgt = inp[:, 1:]
        if p > 0:                                            # scheduled sampling: splice model's own holders
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                pred = model(x).argmax(-1)                   # pred[:,t] = predicted inp[:,t+1]
            do = (x == holder_id) & (torch.rand_like(x, dtype=torch.float) < p)   # at 'holder' marker positions
            bb, tt = do.nonzero(as_tuple=True)
            keep = (tt + 1) < x.shape[1]
            bb, tt = bb[keep], tt[keep]
            x[bb, tt + 1] = pred[bb, tt]                     # replace the holder agent in context with the model's
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(x)
            ce = F.cross_entropy(logits.reshape(-1, tok.vocab_size), tgt.reshape(-1), reduction="none")
            mask = (tgt != pad).float().reshape(-1)
            loss = (ce * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        d = 0.999 if step > STEPS // 2 else 0.99
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if ema[k].dtype.is_floating_point:
                    ema[k].mul_(d).add_(v.float(), alpha=1 - d)
    ema_model = copy.deepcopy(model)
    ema_model.load_state_dict({k: (ema[k].to(v.dtype) if ema[k].dtype.is_floating_point else v)
                               for k, v in ema_model.state_dict().items()})
    return model, ema_model


def main() -> None:
    """Train each scheduled-sampling arm over seeds and tabulate clean-base / EMA / L128 rates."""
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    short_pool = make_pool("short", w, r, origins, oracle, seed=2)
    tok, _, _ = T.prepare(short_pool, [], [w])
    assert "holder" in tok.token_to_id, "scheduled sampling needs the 'holder' marker token"
    evs = {L: [build_ck(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(N_EVAL)]
           for L in EVAL_LEN}
    res = {}
    print(f"=== SCHEDULED SAMPLING: free-running exposure in base training (clean L16>={CLEAN_THR}) ===",
          flush=True)
    for name, p_max, gate in ARMS:
        rows = []
        for s in SEEDS:
            model, ema_model = train_ss(tok, short_pool, s, p_max, gate)
            raw = {L: e2e_eval(model, tok, w, evs[L])[1] for L in EVAL_LEN}
            em = {L: e2e_eval(ema_model, tok, w, evs[L])[1] for L in EVAL_LEN}
            rows.append({"seed": s, "raw": raw, "ema": em})
            print(f"  {name:<9} s{s} :: L16 raw={raw[16]:.2f} ema={em[16]:.2f} | "
                  f"L128 raw={raw[128]:.2f} ema={em[128]:.2f}", flush=True)
            del model, ema_model; torch.cuda.empty_cache()
        res[name] = rows
        cr = sum(x["raw"][16] >= CLEAN_THR for x in rows)
        ce = sum(x["ema"][16] >= CLEAN_THR for x in rows)
        print(f"  >>> {name}: clean(L16>={CLEAN_THR}) raw={cr}/{len(SEEDS)} ema={ce}/{len(SEEDS)}", flush=True)
        write_md(res)
    write_md(res)
    print("schedsample done.", flush=True)


def write_md(res: dict) -> None:
    """Write the scheduled-sampling clean-base / EMA / L128 table to ``OUT``."""
    lines = [
        "# Scheduled sampling — free-running exposure in base training (`schedsample.py`, 18.5M)\n",
        f"At holder slots, with prob p(ramp 0->p_max) replace the gold holder in CONTEXT with the model's own "
        f"argmax (targets stay gold). Metric: clean-base fraction (free-running L16 e2e >= {CLEAN_THR}); baseline "
        "~1/8. EMA = weight-avg over the cosine tail. L128 flags native length-gen. Floor = 0.20.\n",
        "| arm | clean L16 raw | clean L16 ema | max L128 (raw / ema) | L16 per seed (raw) |",
        "|---|---|---|---|---|",
    ]
    for name, _p, _g in ARMS:
        if name not in res:
            continue
        rows = res[name]
        cr = sum(x["raw"][16] >= CLEAN_THR for x in rows)
        ce = sum(x["ema"][16] >= CLEAN_THR for x in rows)
        mx_raw = max(x["raw"][128] for x in rows)
        mx_ema = max(x["ema"][128] for x in rows)
        l16s = " ".join(f"{x['raw'][16]:.2f}" for x in sorted(rows, key=lambda d: -d["raw"][16]))
        lines.append(f"| {name} | {cr}/{len(rows)} | {ce}/{len(rows)} | {mx_raw:.2f} / {mx_ema:.2f} | {l16s} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
