"""M7/M8 training + eval harness.

Efficiency: tokenize/pack the corpus and pre-encode the probes ONCE via `prepare()`, then reuse
across every arch x seed run; eval is BATCHED (sorted by length, right-padded, one forward per
batch — right-padding is safe because every mixer here is causal/left-to-right, so the answer
position never sees the pad). Training is next-token LM under the standard recipe (AdamW wd 0.01,
lr 1e-3, clip 1.0, warmup + cosine, bf16 autocast).
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from .models import build_model
from .render import Renderer
from .tokenizer import Tokenizer


def prepare(texts, probes, worlds, max_len=1280, renderer=None):
    """Tokenizer + PER-DOCUMENT training sequences (length-sorted) + pre-encoded probes, once.

    Each doc is its own training sequence (not concatenated into fixed windows): the previous packed
    scheme trained answers inside a cross-document context that eval — which presents each prompt in
    isolation — never sees, so state-tracking could not even be overfit. Docs are length-sorted so
    dynamic per-batch padding wastes little.

    Args:
        renderer: optional ``Renderer`` instance (defaults to a fresh ``Renderer()``).
            The renderer is only used to probe structural tokens; pass it explicitly
            when the caller already holds one (e.g. the task's renderer).
    """
    tok = Tokenizer.build(worlds, renderer or Renderer())
    docs = [tok.encode(t, add_eos=True)[:max_len] for t in texts]
    docs.sort(key=len)
    encoded = [(tok.encode(p.prompt), tok.token_to_id.get(p.gold, tok.unk_id),
                (p.condition, p.family, p.length)) for p in probes]
    return tok, docs, encoded


@torch.no_grad()
def evaluate(model, tok, encoded, device, token_budget=8192, max_bs=256):
    """Batched single-answer-token exact-match. TOKEN-BUDGET batching (sorted by length): a batch
    grows until batch_size*maxlen exceeds the budget, so long L128 prompts get small batches that
    fit mamba2's memory-heavy naive scan while short probes still pack into big batches."""
    model.eval()
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    per: dict = defaultdict(list)
    i, total = 0, len(order)
    while i < total:
        maxlen, j = len(encoded[order[i]][0]), i + 1
        while j < total:
            ml = max(maxlen, len(encoded[order[j]][0]))
            if (j - i + 1) > max_bs or (j - i + 1) * ml > token_budget:
                break
            maxlen, j = ml, j + 1
        chunk = order[i:j]
        i = j
        seqs = [encoded[c][0] for c in chunk]
        ml = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), ml), tok.pad_id, dtype=torch.long, device=device)
        last = torch.empty(len(seqs), dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            inp[r, : len(s)] = torch.tensor(s, device=device)
            last[r] = len(s) - 1
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(inp)
        pred = logits[torch.arange(len(seqs), device=device), last].float().argmax(-1)
        for r, c in enumerate(chunk):
            per[encoded[c][2]].append(int(pred[r].item() == encoded[c][1]))
    return {k: sum(v) / len(v) for k, v in per.items()}, {k: len(v) for k, v in per.items()}


def run(arch, tok, docs, encoded, *, device="cuda", steps=20000, batch=16, lr=1e-3, warmup=1000,
        weight_decay=0.01, clip=1.0, d_model=320, n_layers=4, n_heads=4, d_ff=1280, use_forget_gate=True,
        seed=0, return_model=False, num_householder=4, allow_neg_eigval=True,
        use_short_conv=False, resid_init=False):
    torch.manual_seed(seed)
    model = build_model(arch, tok.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                        use_forget_gate=use_forget_gate, num_householder=num_householder,
                        allow_neg_eigval=allow_neg_eigval, use_short_conv=use_short_conv,
                        resid_init=resid_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_mult(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    gen = torch.Generator(device=device).manual_seed(seed)
    ndoc, pad = len(docs), tok.pad_id
    model.train()
    last = float("nan")
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr * lr_mult(step)
        start = int(torch.randint(0, max(1, ndoc - batch), (1,), generator=gen, device=device).item())
        chunk = docs[start:start + batch]                       # contiguous in length-sorted order -> low padding
        ml = max(len(s) for s in chunk)
        inp = torch.full((len(chunk), ml), pad, dtype=torch.long, device=device)
        for ri, s in enumerate(chunk):
            inp[ri, : len(s)] = torch.tensor(s, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(inp[:, :-1])
            tgt = inp[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, tok.vocab_size), tgt.reshape(-1), reduction="none")
            mask = (tgt != pad).float().reshape(-1)             # next-token loss on real tokens only
            loss = (ce * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        last = loss.detach().item()

    acc, n = evaluate(model, tok, encoded, device) if encoded else ({}, {})
    if return_model:
        return {"arch": arch, "final_loss": last, "acc": acc, "n": n, "model": model}
    del model
    torch.cuda.empty_cache()
    return {"arch": arch, "final_loss": last, "acc": acc, "n": n}
