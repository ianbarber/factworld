"""Bench a small pretrained model (Qwen3-1.7B) on FactWorld — find the RL-liftable tasks.

The MOPD pivot: instead of a from-scratch model, use a real pretrained model as the base,
RL-specialise it per domain, then MOPD-distil. The paper's own setting is a pretrained
Qwen3; we use the smallest cached member (Qwen3-1.7B) that fits a 3090.

This script establishes the base's per-task accuracy so we can pick two domains where it
is PARTIAL (RL headroom) rather than at ceiling (recall/conflict for strong models) or floor
(chain/s5 — the walls). We report several metrics because a chat model's output format
varies: `relaxed` (strict span), `contains` (gold token appears), `last_n` (gold is the
tail). The reward used for RL will be whichever is both robust and hard to game.

  .venv/bin/python experiments/mopd/bench_qwen.py --n 40 --think 0
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld.render import Renderer

MODEL = "Qwen/Qwen3-1.7B"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_qwen.md")

INSTRUCTION = ("Read the statements and answer the question. "
               "Respond with ONLY the answer token(s) (e.g. `g4` or `g4 v56`), nothing else.")

# (task, eval length). Span the ladder: easy recall/conflict -> binding -> composite/chain/s5.
BENCH = [
    ("recall_copy_v1", 6), ("conflict_v1", 4), ("binding_v2", 16),
    ("composite_copy_v2", 16), ("chain_v1", 4), ("s5_v1", 32),
]


def build_chat(tok, task_prompt: str, think: bool) -> str:
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\n{task_prompt}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=think)


def strip_think(text: str) -> str:
    """Drop a Qwen <think>...</think> block, keep the final answer."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--think", type=int, default=0, help="1 = enable Qwen thinking mode")
    ap.add_argument("--max_new", type=int, default=None)
    ap.add_argument("--show", type=int, default=3, help="print this many raw samples per task")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        print("no GPU"); return

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
    max_new = a.max_new or (512 if a.think else 24)
    print(f"loaded {MODEL}  think={a.think}  max_new={max_new}", flush=True)

    rows = []
    for name, L in BENCH:
        spec = TK.spec_for(name)  # CANONICAL names only (v2 uniform-last-write sampler)
        exs = TK.generate(spec, "test", n=a.n, length=L)
        chats = [build_chat(tok, e.prompt, bool(a.think)) for e in exs]
        preds = []
        bs = 16
        for i in range(0, len(chats), bs):
            batch = chats[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, enc.input_ids.shape[1]:]
            preds += [strip_think(t) for t in tok.batch_decode(gen, skip_special_tokens=True)]
        rel = sum(TK.score_relaxed(Renderer.normalize(p), Renderer.normalize(e.answer))
                  for p, e in zip(preds, exs)) / len(exs)
        con = sum(TK.score_contains(Renderer.normalize(p), Renderer.normalize(e.answer))
                  for p, e in zip(preds, exs)) / len(exs)
        lstn = sum(TK.score_last_n(Renderer.normalize(p), Renderer.normalize(e.answer))
                   for p, e in zip(preds, exs)) / len(exs)
        rows.append((name, L, rel, con, lstn))
        print(f"\n=== {name}@L{L} ===  relaxed={rel:.3f} contains={con:.3f} last_n={lstn:.3f}", flush=True)
        for e, p in list(zip(exs, preds))[:a.show]:
            print(f"  gold={e.answer!r:12} pred={p[:80]!r}", flush=True)

    write_md(a, max_new, rows)
    print("\nbench_qwen done.", flush=True)


def write_md(a, max_new, rows) -> None:
    lines = [
        "# Bench — Qwen3-1.7B on FactWorld (base, before RL)\n",
        f"`experiments/mopd/bench_qwen.py`. {MODEL}, chat template, think={a.think}, "
        f"max_new={max_new}, greedy, n={a.n}. `relaxed` = strict answer span; `contains` = gold "
        "token appears; `last_n` = gold is the tail. Pick RL-teacher domains where the base is "
        "PARTIAL (headroom), not at ceiling or floor.\n",
        "| task | L | relaxed | contains | last_n |", "|---|---|---|---|---|",
    ]
    for name, L, rel, con, lstn in rows:
        lines.append(f"| {name} | {L} | {rel:.3f} | {con:.3f} | {lstn:.3f} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
