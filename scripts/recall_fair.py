"""Decisive recall differential: isolate FORMAT (1-hop vs deferred read-out) from RECIPE (n_heads) from
the PAD pathology, on a clean MQAR (pool 2 = 1-of-2 copy). All atomic-vocab, same optimizer/steps.

formats (how the value is supervised for a query on key k with value v):
  onehop : [... k v]              target v at the key position (Zoology: value = next token after key)   [1-hop]
  defsep : [... k SEP]            target v at the SEP position (key is the immediately-prior token)      [2-hop, fair]
  defpad : [... QUERY k ANS PAD]  target v at the PAD slot (our recall_variants format)                 [2-hop, pathological]
"""
import os
import sys
import math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from factworld.models import build_model

NKEY, NVAL = 64, 64
QUERY, ANS, SEP, PAD = 1, 2, 3, 0
K0 = 4
V0 = K0 + NKEY
VOCAB = V0 + NVAL


def make_batch(batch, pool, nq, fmt, device, gen):
    X, T = [], []
    for _ in range(batch):
        keys = torch.randperm(NKEY, generator=gen)[:pool] + K0
        vals = torch.randint(0, NVAL, (pool,), generator=gen) + V0
        toks, tgt = [], []
        for k, v in zip(keys.tolist(), vals.tolist()):
            toks += [k, v]; tgt += [-100, -100]
        for j in torch.randint(0, pool, (nq,), generator=gen).tolist():
            k, v = int(keys[j]), int(vals[j])
            if fmt == "onehop":
                toks += [k, v]; tgt += [v, -100]
            elif fmt == "defsep":
                toks += [k, SEP]; tgt += [-100, v]
            else:  # defpad
                toks += [QUERY, k, ANS, PAD]; tgt += [-100, -100, -100, v]
        X.append(torch.tensor(toks)); T.append(torch.tensor(tgt))
    L = max(len(s) for s in X)
    XX = torch.full((batch, L), PAD, dtype=torch.long)
    TT = torch.full((batch, L), -100, dtype=torch.long)
    for i, (s, t) in enumerate(zip(X, T)):
        XX[i, :len(s)] = s; TT[i, :len(t)] = t
    return XX.to(device), TT.to(device), (TT != -100).to(device)


def run(arch, fmt, heads, pool=2, d=320, nl=4, steps=30000, seed=0, device="cuda"):
    torch.manual_seed(seed)
    m = build_model(arch, VOCAB, d_model=d, n_layers=nl, n_heads=heads, d_ff=4 * d).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(100 + seed)
    m.train()
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * (min(1.0, (step + 1) / 1000) if step < 1000
                               else 0.5 * (1 + math.cos(math.pi * (step - 1000) / (steps - 1000))))
        X, T, M = make_batch(64, pool, 8, fmt, device, gen)
        with torch.autocast(device, dtype=torch.bfloat16):
            lo = m(X)
        loss = F.cross_entropy(lo.reshape(-1, VOCAB).float(), T.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    eg = torch.Generator().manual_seed(9999)
    with torch.no_grad():
        X, T, M = make_batch(512, pool, 8, fmt, device, eg)
        with torch.autocast(device, dtype=torch.bfloat16):
            pred = m(X).argmax(-1)
        acc = ((pred == T) & M).sum().item() / M.sum().item()
    del m; torch.cuda.empty_cache()
    return acc


CELLS = [  # (arch, fmt, heads)
    ("transformer", "onehop", 4), ("transformer", "onehop", 8),
    ("transformer", "defsep", 8), ("transformer", "defpad", 8),
    ("gdp_pure", "onehop", 4), ("gdp_pure", "defpad", 4),
]

if __name__ == "__main__":
    print("=== RECALL FAIR DIFFERENTIAL (pool 2 = 1-of-2; floor 1/64=0.016; '1/pool'=0.5) ===", flush=True)
    for arch, fmt, h in CELLS:
        accs = [run(arch, fmt, h, seed=s) for s in range(2)]
        print(f"  {arch:<11} {fmt:<7} h{h} :: acc={sum(accs) / len(accs):.3f} (seeds {accs[0]:.2f}/{accs[1]:.2f})", flush=True)
    print("recall_fair done.", flush=True)
