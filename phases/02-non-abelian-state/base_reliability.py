"""Raise the clean-base rate (and check native length-gen): short-conv + gate-retention-init, many seeds.

The post-training lever is reliable GIVEN a clean base (free-running answer-only L16 e2e >= 0.95), but clean
bases are rare (~1/8 seeds; base L16 dist 0.97,0.86,0.77,0.62,0.37,0.30,0.27,0.26). dense_h16 (teacher-forced)
is saturated 1.00 for ALL bases -> the fragility is FREE-RUNNING autoregressive stability, not learning the map.

Two convergent leads for that:
  1. SHORT-CONV (use_short_conv=True): fla warns "ShortConvolution is crucial... do not turn it off" and we have
     had it OFF the whole study; FPRM (Orvieto, arXiv:2606.18206) finds the 1D causal conv is THE state-tracking
     lever (S5 45%->97%). A translation-equivariant scan primitive -> plausibly stabilizes the free-running circuit.
  2. GATE-RETENTION-INIT: GDP forget gate g = -exp(A_log)*softplus(a_proj+dt_bias), A~U(0,16), dt log-uniform
     [1e-3,0.1] -> random decay at init. Permutation composition wants near-norm-preserving (g~0). Bias dt_bias/
     A_log toward retention so every seed starts near the stable basin.

Arms (one knob each + combo) x K seeds. Metric: clean-base fraction (#{L16 e2e >= 0.95}). Also eval L128 (raw &
EMA) to catch any NATIVE length-generalization (would be a big surprise -> short-conv alone fixes the wall).

  .venv/bin/python followups/non-abelian-state/base_reliability.py
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

SEEDS = list(range(8))
ARMS = [("default", False, "default"),     # current setup (no conv, default gate) -- the 1/8 baseline
        ("shortconv", True, "default"),    # the dual-endorsed lead
        ("gate_slow", False, "slow"),      # retention init
        ("conv+gate", True, "slow")]       # combo
STEPS, CLEAN_THR, N_EVAL = 6000, 0.95, 150
D_MODEL, N_LAYERS, D_FF = 384, 6, 1536
EVAL_LEN = [16, 128]                        # L16 = clean-base predictor; L128 = native length-gen check
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_reliability.md")


def inv_softplus(y: float) -> float:
    """Inverse softplus.

    Args:
        y: the softplus output value.

    Returns:
        The pre-softplus value x such that ``softplus(x) == y``.
    """
    return math.log(math.expm1(y))


def apply_gate_init(model: Any, mode: str) -> None:
    """Bias the GDP forget gate toward retention (slow gate) for retention-init arms.

    Args:
        model: a built ``HybridLM`` (mutated in place).
        mode: ``"default"`` (no-op) or ``"slow"`` (retention init on GDP layers).
    """
    if mode == "default":
        return
    for b in model.blocks:
        lyr = getattr(getattr(b, "mix", None), "layer", None)
        if lyr is not None and hasattr(lyr, "dt_bias"):      # GDP layers only (attn skipped)
            lyr.dt_bias.data.fill_(inv_softplus(1e-3))       # slow gate -> retention
            if hasattr(lyr, "A_log"):
                lyr.A_log.data.fill_(math.log(0.5))


def train_emma(tok: Any, pool: list[str], seed: int, use_conv: bool, gate: str,
               device: str = "cuda") -> tuple[Any, Any]:
    """Train one gdp_hybrid base while accumulating an EMA copy of the weights.

    Args:
        tok: the atomic tokenizer.
        pool: training strings (the short in-distribution pool).
        seed: RNG/init seed.
        use_conv: whether to enable ShortConvolution (``use_short_conv``).
        gate: gate-init mode passed to ``apply_gate_init`` (``"default"`` or ``"slow"``).
        device: torch device.

    Returns:
        A ``(model, ema_model)`` pair — the raw trained model and its EMA-weight copy.
    """
    import torch
    import torch.nn.functional as F
    from factworld.models import build_model
    torch.manual_seed(seed)
    model = build_model("gdp_hybrid", tok.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=4,
                        d_ff=D_FF, num_householder=4, allow_neg_eigval=True, use_short_conv=use_conv).to(device)
    apply_gate_init(model, gate)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warmup, pad, rng = 1000, tok.pad_id, random.Random(7000 + seed)
    model.train()
    for step in range(STEPS):
        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * (min(1.0, (step + 1) / warmup) if step < warmup else
                               0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, STEPS - warmup))))
        seqs = [tok.encode(d) for d in rng.sample(pool, 32)]
        ml = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), ml), pad, dtype=torch.long, device=device)
        for ri, s in enumerate(seqs):
            inp[ri, : len(s)] = torch.tensor(s, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(inp[:, :-1]); tgt = inp[:, 1:]
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
    """Train K=8 bases per arm (short-conv x gate-init) and tabulate clean-base fraction and L128 length-gen."""
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    short_pool = make_pool("short", w, r, origins, oracle, seed=2)
    tok, _, _ = T.prepare(short_pool, [], [w])
    evs = {L: [build_ck(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(N_EVAL)]
           for L in EVAL_LEN}
    res = {}
    print(f"=== BASE-RELIABILITY: short-conv x gate-init, clean-base fraction (L16>={CLEAN_THR}) ===", flush=True)
    for name, use_conv, gate in ARMS:
        rows = []
        for s in SEEDS:
            model, ema_model = train_emma(tok, short_pool, s, use_conv, gate)
            raw = {L: e2e_eval(model, tok, w, evs[L])[1] for L in EVAL_LEN}
            em = {L: e2e_eval(ema_model, tok, w, evs[L])[1] for L in EVAL_LEN}
            rows.append({"seed": s, "raw": raw, "ema": em})
            print(f"  {name:<10} s{s} :: L16 raw={raw[16]:.2f} ema={em[16]:.2f} | "
                  f"L128 raw={raw[128]:.2f} ema={em[128]:.2f}", flush=True)
            del model, ema_model; torch.cuda.empty_cache()
        res[name] = rows
        cr = sum(x["raw"][16] >= CLEAN_THR for x in rows)
        ce = sum(x["ema"][16] >= CLEAN_THR for x in rows)
        print(f"  >>> {name}: clean(L16>={CLEAN_THR}) raw={cr}/{len(SEEDS)} ema={ce}/{len(SEEDS)}", flush=True)
        write_md(res)
    write_md(res)
    print("base_reliability done.", flush=True)


def write_md(res: dict) -> None:
    """Write the clean-base-fraction and L128 length-gen table to ``OUT``."""
    lines = [
        "# Base-reliability — short-conv x gate-init, clean-base fraction (`base_reliability.py`, 18.5M)\n",
        f"Free-running answer-only L16 e2e (clean = >= {CLEAN_THR}); baseline ~1/8. `shortconv` = use_short_conv "
        "(fla 'crucial'; FPRM's state-tracking lever); `gate_slow` = GDP forget-gate retention-init. EMA = "
        "weight-avg over the cosine tail. L128 raw/ema flags any NATIVE length-generalization (no post-training). "
        "Floor = 0.20.\n",
        "| arm | clean L16 raw | clean L16 ema | max L128 (raw / ema) | L16 per seed (raw) |",
        "|---|---|---|---|---|",
    ]
    for name, _c, _g in ARMS:
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
