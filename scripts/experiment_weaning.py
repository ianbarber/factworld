"""Weaning bridge — can a dense-learned s5 circuit survive weaning to answer-only?

The biggest open question for the agentic regime: dense per-step supervision moves the s5 wall, but
you can't deploy with dense supervision. Can you train dense, then WEAN to answer-only (no per-step
labels) and keep the circuit? Phase 2 said yes via a mixed-density curriculum on the atomic format;
this reproduces/tests it on the natural format.

Arms (gdp_hybrid, s5/composite):
  - dense_only   : K=1 throughout (reference — circuit forms, but needs labels at deploy).
  - answer_only  : K=inf throughout (reference floor — circuit never forms).
  - wean_linear  : dense K=1, then linearly sparsify K each epoch 1->2->4->inf.
  - wean_mixed   : dense K=1, then fine-tune on a MIX of densities (K in {1,2,4,inf}) — Phase 2's winner.

Eval: FREE-RUN end-to-end value accuracy (no holder labels at eval) at L16/32/64. If wean_mixed >
answer_only and approaches dense_only, the circuit survives weaning -> the bridge to agentic works.

Example:
    .venv-train/bin/python scripts/experiment_weaning.py --seeds 0 1 2 3 4 5 6 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld import train as T
from factworld.render import Renderer
from factworld.oracle import Oracle
from factworld.tokenizer import Tokenizer
from factworld.config import WorldConfig
from factworld.world import World
import importlib.util

# reuse the dense-supervision stream builder + eval
_spec = importlib.util.spec_from_file_location("eds", os.path.join(REPO, "scripts", "experiment_dense_supervision.py"))
eds = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eds)


def make_world(spec):
    wc = WorldConfig(seed=spec.seed, n_entities=8, n_attributes=2,
                     value_vocab_size=spec.value_vocab_size, n_objects=spec.n_objects,
                     n_locations=6, k=spec.k)
    w = World(wc)
    return w, Renderer(), Oracle(w)


def build_docs_at_K(spec, w, r, oracle, origins, n, K, seed):
    docs = []
    for j in range(n):
        L = [4, 8, 16][j % 3]
        words, *_ = eds.build_stream(spec, w, r, oracle, origins, L, K, random.Random(seed * 1000 + j))
        docs.append(" ".join(words))
    return docs


def train_phase(model, tok, docs, *, steps, batch, d_model, n_layers, d_ff, seed, device):
    """Continue training an existing model on new docs (returns same model)."""
    enc = [tok.encode(t, add_eos=True) for t in docs]
    enc.sort(key=len)
    # reuse T.run but it builds a fresh model; instead do a manual continuation loop on the passed model
    import torch, math
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    gen = torch.Generator(device=device).manual_seed(seed)
    pad = tok.pad_id
    ndoc = len(enc)
    model.train()
    for step in range(steps):
        # lr warmup+cosine mirroring T.run
        warmup = 1000
        lr_mult = (step + 1) / max(1, warmup) if step < warmup else 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, steps - warmup))))
        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * lr_mult
        start = int(torch.randint(0, max(1, ndoc - batch), (1,), generator=gen, device=device).item())
        chunk = enc[start:start + batch]
        ml = max(len(s) for s in chunk)
        inp = torch.full((len(chunk), ml), pad, dtype=torch.long, device=device)
        for ri, s in enumerate(chunk):
            inp[ri, : len(s)] = torch.tensor(s, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(inp[:, :-1])
            tgt = inp[:, 1:]
            import torch.nn.functional as F
            ce = F.cross_entropy(logits.reshape(-1, tok.vocab_size), tgt.reshape(-1), reduction="none")
            mask = (tgt != pad).float().reshape(-1)
            loss = (ce * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def run_arm(spec, arm, seed, *, steps, wean_steps, batch, d_model, n_layers, train_n, device, mix_default="1,2,4,inf"):
    """Train one (arm, seed). Returns the trained model + assets."""
    w, r, oracle = make_world(spec)
    origins = TK._fixed_origins(spec, w)
    tok = Tokenizer.build([w], r)
    d_ff = 4 * d_model

    if arm == "dense_only":
        docs = build_docs_at_K(spec, w, r, oracle, origins, train_n, 1, seed)
        run = T.run("gdp_hybrid", tok, [tok.encode(t, add_eos=True) for t in docs], [], steps=steps,
                    batch=batch, d_model=d_model, n_layers=n_layers, d_ff=d_ff, seed=seed,
                    return_model=True, device=device)
        return run["model"], tok, w, r, oracle, origins

    if arm == "answer_only":
        docs = build_docs_at_K(spec, w, r, oracle, origins, train_n, 10 ** 9, seed)
        run = T.run("gdp_hybrid", tok, [tok.encode(t, add_eos=True) for t in docs], [], steps=steps,
                    batch=batch, d_model=d_model, n_layers=n_layers, d_ff=d_ff, seed=seed,
                    return_model=True, device=device)
        return run["model"], tok, w, r, oracle, origins

    # both weaning arms: start from a dense K=1 model
    docs_dense = build_docs_at_K(spec, w, r, oracle, origins, train_n, 1, seed)
    run = T.run("gdp_hybrid", tok, [tok.encode(t, add_eos=True) for t in docs_dense], [], steps=steps,
                batch=batch, d_model=d_model, n_layers=n_layers, d_ff=d_ff, seed=seed,
                return_model=True, device=device)
    model = run["model"]

    if arm == "wean_linear":
        # sparsify K across weaning phases: 1 -> 2 -> 4 -> inf, equal step budget each
        per = max(1, wean_steps // 4)
        for K in [2, 4, 8, 10 ** 9]:
            docs = build_docs_at_K(spec, w, r, oracle, origins, train_n, K, seed + hash(K) % 1000)
            model = train_phase(model, tok, docs, steps=per, batch=batch, d_model=d_model,
                                n_layers=n_layers, d_ff=d_ff, seed=seed, device=device)
        return model, tok, w, r, oracle, origins

    if arm.startswith("wean_mixed"):
        # Phase 2's winner: fine-tune on a MIX of densities. The K-set is configurable via the arm
        # name (e.g. 'wean_mixed:1,2,4,inf' or 'wean_mixed:1,inf') or the --mix default.
        ks_str = arm.split(":", 1)[1] if ":" in arm else mix_default
        ks = [int(k) if k != "inf" else 10 ** 9 for k in ks_str.split(",")]
        docs = []
        for j in range(train_n):
            L = [4, 8, 16][j % 3]
            K = random.Random(seed * 7 + j).choice(ks)
            words, *_ = eds.build_stream(spec, w, r, oracle, origins, L, K, random.Random(seed * 1000 + j))
            docs.append(" ".join(words))
        model = train_phase(model, tok, docs, steps=wean_steps, batch=batch, d_model=d_model,
                            n_layers=n_layers, d_ff=d_ff, seed=seed, device=device)
        return model, tok, w, r, oracle, origins

    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser(description="s5 weaning bridge: dense -> answer-only.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--arms", nargs="+", default=["dense_only", "answer_only", "wean_linear", "wean_mixed"])
    ap.add_argument("--mix", default="1,2,4,inf",
                    help="Default density K-set for wean_mixed (comma-sep, 'inf'=answer-only). Override per-arm with wean_mixed:1,4,inf.")
    ap.add_argument("--steps", type=int, default=4000, help="Dense + answer-only training steps.")
    ap.add_argument("--wean_steps", type=int, default=4000, help="Total weaning fine-tune steps.")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--eval_lengths", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    spec = TK.CANONICAL["composite_copy_scale_v1"].scaled(k=5)   # k=5 non-abelian composite (s5-style)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    from pathlib import Path
    prefix = Path(a.out_prefix or f"results/weaning_{ts}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl = Path(f"{prefix}.jsonl"); md = Path(f"{prefix}.md")

    print(f"=== weaning bridge -> {jsonl} ===", flush=True)
    # FREE-RUN eval at K=1 stream shape (holder slots present) but the model generates them with no
    # teacher forcing -> measures whether the circuit survives without label supervision at eval.
    agg = defaultdict(lambda: defaultdict(list))
    for arm in a.arms:
        for seed in a.seeds:
            print(f"\n--- {arm} seed={seed} ---", flush=True)
            model, tok, w, r, oracle, origins = run_arm(
                spec, arm, seed, steps=a.steps, wean_steps=a.wean_steps, batch=a.batch,
                d_model=a.d_model, n_layers=a.n_layers, train_n=a.train_n, device=a.device,
                mix_default=a.mix)
            row = {"arm": arm, "seed": seed}
            for L in a.eval_lengths:
                exs = eds.build_eval(spec, w, r, oracle, origins, L, 1, a.eval_n)
                fh, fv = eds.e2e_eval(model, tok, w, exs, device=a.device)
                row[f"L{L}_holder"] = fh; row[f"L{L}_value"] = fv
                agg[arm][L].append(fv)
                print(f"    L{L}: free-run holder={fh:.2f} value={fv:.2f}", flush=True)
            with jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")
            import torch; del model; torch.cuda.empty_cache()

    lines = ["# Weaning bridge — does a dense-learned s5 circuit survive weaning to answer-only?", "",
             f"`scripts/experiment_weaning.py`. gdp_hybrid d{a.d_model}x{a.n_layers}, dense {a.steps} steps "
             f"(+{a.wean_steps} wean), seeds {a.seeds}. FREE-RUN eval (no holder labels at eval; the model "
             f"generates its own state). value = end-to-end accuracy. Floor = 0.20.", "",
             "| arm | " + " | ".join(f"value @L{L}" for L in a.eval_lengths) + " | conv @L16 |",
             "|---|" + "---|" * (len(a.eval_lengths) + 1)]
    for arm in a.arms:
        cells = []
        for L in a.eval_lengths:
            xs = agg[arm][L]
            cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "-")
        conv = sum(1 for x in agg[arm].get(a.eval_lengths[0], []) if x > 0.5)
        cells.append(f"{conv}/{len(a.seeds)}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "_dense_only = needs labels at deploy (reference ceiling); answer_only = circuit never "
              "forms (floor); wean_linear = sparsify K 1->2->4->inf; wean_mixed = fine-tune on a mix of "
              "densities (Phase 2's winner). If a wean arm >> answer_only and approaches dense_only, the "
              "circuit survives weaning -> the bridge to the agentic (label-free) regime works._"]
    md.write_text("\n".join(lines))
    print(f"\n=== wrote {md} ===", flush=True)


if __name__ == "__main__":
    main()
