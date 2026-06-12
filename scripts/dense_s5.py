"""Canonical group word-problem (S5 / A5) state-tracking probe — dense per-token supervision.

A from-scratch reimplementation of the standard automaton/group word problem (Liu et al., 2023): the model
reads a sequence of group elements g_1…g_t and must output the running prefix product g_1·…·g_t at EVERY
position (dense supervision; loss on all positions but the first). Evaluation is a single forward pass with
a masked argmax (no autoregressive generation), exact-match at train length and at OOD lengths.

This is the isolation control FactWorld's main suite lacks: FactWorld's hard-state composite is single-query
/ answer-only (which floors — §4), whereas non-abelian state-tracking is learnable under the canonical
dense regime. We re-derive the prior-art mechanism (Grazzi et al., 2024; Siems et al., 2025) at our scale,
in our own repo: the GatedDeltaProduct product structure (n_h=4, neg-eig) flat-extrapolates; the single-
reflection null decays; diagonal/attention baselines shortcut.

  .venv/bin/python scripts/dense_s5.py --smoke      # quick 1-seed validation (gdp n_h=4 vs n_h=1 null)
  .venv/bin/python scripts/dense_s5.py --group s5   # full matrix
"""
import argparse
import itertools
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from factworld.models import build_model


# ---------------------------------------------------------------------------
# group tables (S5 = all 120 perms of 5; A5 = the 60 even perms) — precomputed once
# ---------------------------------------------------------------------------
def _build_group(which: str):
    perms = list(itertools.permutations(range(5)))                 # lexicographic, |S5|=120
    index = {p: i for i, p in enumerate(perms)}

    def compose(a, b):                                             # (a then b): apply a, then b
        return tuple(b[a[i]] for i in range(5))

    n = len(perms)
    mul = [[0] * n for _ in range(n)]
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            mul[i][j] = index[compose(a, b)]
    mul = torch.tensor(mul, dtype=torch.long)                     # (120,120) multiplication table

    if which == "s5":
        # 5 standard generators of S5 (4 adjacent transpositions + the 5-cycle)
        def perm_from(swaps=None, cyc=False):
            p = list(range(5))
            if cyc:
                p = [1, 2, 3, 4, 0]
            elif swaps:
                i, j = swaps
                p[i], p[j] = p[j], p[i]
            return index[tuple(p)]
        gens = [perm_from(swaps=(0, 1)), perm_from(swaps=(1, 2)), perm_from(swaps=(2, 3)),
                perm_from(swaps=(3, 4)), perm_from(cyc=True)]
    else:  # a5: generators of the alternating group (3-cycles); verify closure == 60
        def perm(mapping):
            return index[tuple(mapping)]
        gens = [perm([1, 2, 0, 3, 4]), perm([0, 2, 3, 1, 4]), perm([0, 1, 3, 4, 2])]  # 3-cycles
    gens = sorted(set(gens))
    # verify the generated subgroup size
    seen, frontier = set(gens) | {index[tuple(range(5))]}, list(gens)
    while frontier:
        a = frontier.pop()
        for g in gens:
            c = mul[a][g].item()
            if c not in seen:
                seen.add(c); frontier.append(c)
    return mul, gens, len(seen)


def sample_batch(mul, gens, batch, length, device, gen: torch.Generator):
    """input g_0..g_{L-1}; target_t = prefix product g_0·…·g_t; loss on positions 1..L-1."""
    gt = torch.tensor(gens)
    g = gt[torch.randint(0, len(gens), (batch, length), generator=gen)]   # (B,L) group elements
    cum = torch.empty_like(g)
    cum[:, 0] = g[:, 0]
    for t in range(1, length):
        cum[:, t] = mul[cum[:, t - 1], g[:, t]]
    mask = torch.ones(batch, length, dtype=torch.bool); mask[:, 0] = False
    return g.to(device), cum.to(device), mask.to(device)


# ---------------------------------------------------------------------------
# train + single-pass masked-argmax eval
# ---------------------------------------------------------------------------
def run_config(label, arch, mul, gens, *, n_h=4, neg=True, d_model=288, n_layers=6, n_heads=4,
               d_ff=None, steps=30000, batch=64, train_len=32, eval_lens=(32, 64, 128), lr=1e-3,
               warmup=1000, seed=0, device="cuda"):
    torch.manual_seed(seed)
    vocab = 121  # 120 group elements + PAD (unused here but keeps a clean head)
    d_ff = d_ff or 4 * d_model
    model = build_model(arch, vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        d_ff=d_ff, num_householder=n_h, allow_neg_eigval=neg).to(device)
    nparam = model.num_params()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1000 + seed)

    model.train()
    for step in range(steps):
        for pg in opt.param_groups:                              # cosine schedule with linear warmup
            pg["lr"] = lr * (min(1.0, (step + 1) / warmup) if step < warmup
                             else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        g, tgt, mask = sample_batch(mul, gens, batch, train_len, device, gen)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(g)
        ce = F.cross_entropy(logits.reshape(-1, vocab).float(), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        loss = (ce * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # eval: single forward pass, masked argmax, exact-match = ALL supervised positions correct
    model.eval()
    out = {}
    egen = torch.Generator().manual_seed(7777)
    with torch.no_grad():
        for L in eval_lens:
            g, tgt, mask = sample_batch(mul, gens, 256, L, device, egen)
            with torch.autocast(device, dtype=torch.bfloat16):
                pred = model(g).argmax(-1)
            correct = (pred == tgt) | ~mask
            seq_exact = correct.all(dim=1).float().mean().item()        # whole-sequence exact match
            tok_acc = (correct & mask).sum().item() / mask.sum().item() # per-token accuracy
            out[L] = (seq_exact, tok_acc)
    del model; torch.cuda.empty_cache()
    return nparam, out


CORE = [
    # (label, arch, n_h, neg, d_model, n_layers, d_ff)
    ("gdp_nh4",      "gdp_pure",    4, True,  288, 6, None),   # the product-structure winner
    ("gdp_nh1_null", "gdp_pure",    1, True,  288, 6, None),   # single-reflection null (neg-eig on)
    ("gdn",          "gdn_pure",    4, True,  288, 6, None),   # diagonal delta baseline
    ("transformer",  "transformer", 4, True,  288, 6, None),   # softmax baseline
    ("gru",          "gru",         4, True,  288, 6, None),   # NC1 reference
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="s5", choices=["s5", "a5"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--configs", type=int, default=0, help="run only the first N configs (0 = all)")
    a = ap.parse_args()
    mul, gens, order = _build_group(a.group)
    print(f"=== DENSE {a.group.upper()} word problem (order={order}, {len(gens)} generators); "
          f"train L32 -> eval 32/64/128; exact-match (seq) / token-acc ===", flush=True)
    configs = CORE[:2] if a.smoke else (CORE[:a.configs] if a.configs else CORE)
    seeds = [0] if a.smoke else list(range(a.seeds))
    steps = 3000 if a.smoke else a.steps
    for (label, arch, n_h, neg, d_model, n_layers, d_ff) in configs:
        rows = []
        for s in seeds:
            nparam, out = run_config(label, arch, mul, gens, n_h=n_h, neg=neg, d_model=d_model,
                                     n_layers=n_layers, d_ff=d_ff, steps=steps, seed=s)
            rows.append(out)
            cells = "  ".join(f"L{L}={out[L][0]:.3f}/{out[L][1]:.3f}" for L in out)
            print(f"  {label:<14} ({nparam/1e6:.2f}M) s{s} :: {cells}", flush=True)
        # mean exact-match across seeds
        Ls = list(rows[0])
        mean = {L: sum(r[L][0] for r in rows) / len(rows) for L in Ls}
        print(f"  {label:<14} MEAN exact :: " + "  ".join(f"L{L}={mean[L]:.3f}" for L in Ls), flush=True)
    print("dense_s5 done.", flush=True)


if __name__ == "__main__":
    main()
