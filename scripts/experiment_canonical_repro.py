"""Canonical-result reproduction on the natural-language instrument.

Validates that FactWorld reproduces the field's established single-capability dissociations
on the natural-language format. If these hold, the instrument is sound and downstream
architecture claims are trustworthy. If the 1-hop MQAR positive control fails for the
transformer, we have a setup bug.

Three reproductions, each set up appropriately within FactWorld's task families:

  (R1) 1-hop associative recall (the "attention solves recall" regime; Arora/Zoology).
       Value is read adjacent to the key. Expected: transformer ≈ 1.0 (positive control).

  (R2) Deferred read-out recall (the composition regime; value read at an arbitrary
       later position). Expected: the recurrent hybrid wins; the transformer is weak.
       This is what `recall_copy_v1` actually tests.

  (R3) S5 dense-supervision extrapolation (Liu et al. 2023; Siems et al. 2025).
       Expected: product recurrence extrapolates past training length; transformer shortcuts.

The point is not to re-derive the papers -- it is to show the instrument gates the same
dissociations they report, on the natural format, so our downstream results inherit their
validity.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

from factworld import tasks as TK, train as T
from factworld.backends import LocalBackend
from factworld.render import Renderer
from factworld.runner import evaluate_task
from factworld.tokenizer import Tokenizer


# ----- R1: canonical 1-hop MQAR (synthetic, value adjacent to key) -----
# This is the Zoology/Arora regime: the transformer is *expected* to ace it.
def r1_onehop_mqar(arch, *, d=256, nl=4, steps=20000, seed=0, pool=16, device="cuda"):
    """On-the-fly 1-hop MQAR. Each doc: [k0 v0 k1 v1 ...] then queries [k_q v_q] where the
    value is the token immediately after the (repeated) key. Pure attention/induction task."""
    from factworld.models import build_model
    NKEY, NVAL, VOCAB = 64, 64, 4 + 64 + 64
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    K0, V0 = 4, 4 + 64
    torch.manual_seed(seed)
    m = build_model(arch, VOCAB, d_model=d, n_layers=nl, n_heads=4, d_ff=4 * d).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(100 + seed)

    def make_batch(bs, pool, nq):
        X, T = [], []
        for _ in range(bs):
            keys = torch.randperm(NKEY, generator=gen)[:pool] + K0
            vals = torch.randint(0, NVAL, (pool,), generator=gen) + V0
            toks, tgt = [], []
            for k, v in zip(keys.tolist(), vals.tolist()):
                toks += [k, v]; tgt += [-100, -100]
            for j in torch.randint(0, pool, (nq,), generator=gen).tolist():
                k, v = int(keys[j]), int(vals[j])
                toks += [k, v]; tgt += [-100, v]   # 1-hop: value is next token after the (repeated) key
            X.append(torch.tensor(toks)); T.append(torch.tensor(tgt))
        L = max(len(s) for s in X)
        XX = torch.full((bs, L), PAD, dtype=torch.long); TT = torch.full((bs, L), -100, dtype=torch.long)
        for i, (s, t) in enumerate(zip(X, T)):
            XX[i, :len(s)] = s; TT[i, :len(t)] = t
        return XX.to(device), TT.to(device), (TT != -100).to(device)

    m.train()
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * (min(1.0, (step + 1) / 1000) if step < 1000
                               else 0.5 * (1 + math.cos(math.pi * (step - 1000) / (steps - 1000))))
        X, Tg, M = make_batch(64, pool, 8)
        with torch.autocast(device, dtype=torch.bfloat16):
            lo = m(X)
        loss = F.cross_entropy(lo.reshape(-1, VOCAB).float(), Tg.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        X, Tg, M = make_batch(512, pool, 8)
        with torch.autocast(device, dtype=torch.bfloat16):
            pred = m(X).argmax(-1)
        acc = ((pred == Tg) & M).sum().item() / M.sum().item()
    del m; torch.cuda.empty_cache()
    return acc


# ----- R2: deferred read-out (FactWorld's recall_copy_v1, the composition regime) -----
def r2_deferred_recall(arch, *, d=256, nl=4, steps=8000, seed=0, device="cuda"):
    """The FactWorld deferred read-out: facts up front, query at the end. This is the harder
    regime composition needs. Expected: recurrent hybrid wins, transformer weak."""
    spec = TK.CANONICAL["recall_copy_v1"]
    w, r = TK.build_world(spec)
    tr = TK.generate(spec, "train", n=8000)
    docs = [f"{e.prompt} {e.answer}" for e in tr]
    tok, dset, _ = T.prepare(docs, [], [w], renderer=r)
    run = T.run(arch, tok, dset, [], steps=steps, batch=32, d_model=d, n_layers=nl,
                d_ff=4 * d, seed=seed, return_model=True, device=device)
    be = LocalBackend([w], arch=arch, model=run["model"], tokenizer=tok, device=device)
    acc = {L: evaluate_task(be, spec, split="test", n=100, length=L)["overall"] for L in (5, 6, 8)}
    del run["model"]; torch.cuda.empty_cache()
    return acc


# ----- R3: S5 dense-supervision extrapolation (Liu/Siems regime) -----
def r3_s5_extrapolation(arch, *, d=256, nl=4, steps=4000, seed=0, device="cuda"):
    """Train dense (K=1) on S5, eval free-running past training length.
    Expected: product recurrence extrapolates; transformer shortcuts (floors past L16)."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import importlib.util
    _s = importlib.util.spec_from_file_location("eds", os.path.join(REPO, "scripts", "experiment_dense_supervision.py"))
    eds = importlib.util.module_from_spec(_s); _s.loader.exec_module(eds)
    spec = TK.RETIRED["composite_copy_scale_v1"].scaled(k=5)
    model, tok, w, r, oracle, origins = eds.run_K(spec, 1, seed, steps=steps, batch=32, d_model=d,
                                                   n_layers=nl, train_n=8000, device=device, arch=arch,
                                                   eval_lengths=(16, 64, 128))
    out = {}
    for L in (16, 64, 128):
        exs = eds.build_eval(spec, w, r, oracle, origins, L, 1, 100)
        fh, fv = eds.e2e_eval(model, tok, w, exs, device=device)
        out[L] = fv
    del model; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description="Canonical-result reproduction on the natural instrument.")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--steps_r1", type=int, default=20000)
    ap.add_argument("--steps_r2", type=int, default=8000)
    ap.add_argument("--steps_r3", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/canonical_repro.jsonl")
    a = ap.parse_args()

    print("=" * 70, flush=True)
    print("CANONICAL-RESULT REPRODUCTION (natural-language instrument)", flush=True)
    print("=" * 70, flush=True)
    archs = ["gdp_hybrid", "fprm", "transformer"]

    for label, fn, kwargs in [
        ("R1 1-hop MQAR (pool 16)", r1_onehop_mqar, {"d": a.d, "steps": a.steps_r1, "pool": 16}),
        ("R2 deferred read-out (FactWorld recall_copy)", r2_deferred_recall, {"d": a.d, "steps": a.steps_r2}),
        ("R3 S5 dense extrapolation (K=1)", r3_s5_extrapolation, {"d": a.d, "steps": a.steps_r3}),
    ]:
        print(f"\n### {label}", flush=True)
        for arch in archs:
            for seed in a.seeds:
                try:
                    res = fn(arch, seed=seed, **kwargs)
                    rec = {"repro": label, "arch": arch, "seed": seed, "result": res}
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"  {arch:12s} seed{seed}: {res}", flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    print(f"  {arch:12s} seed{seed}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
