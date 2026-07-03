"""Pretrain on world dynamics, then finetune on tasks with optional output-only masking.

The hope is that pretraining on raw world sequences (facts + history, no queries)
lets the model learn the fact-map and event dynamics as a pure LM problem, before
switching to producing answers for task queries.

Example:
    .venv-train/bin/python scripts/experiment_pretrain_finetune.py \
        --seeds 0 1 2 --d_model 768 --n_layers 8 --batch 64 \
        --pretrain_steps 15000 --finetune_steps 10000 \
        --pretrain_n 40000 --finetune_n 40000 \
        --finetune_output_only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK, train as T
from factworld.backends import LocalBackend
from experiment_curriculum_staged import (
    build_training_docs, eval_grid, flatten_eval, staged_specs,
)


def _strip_query(prompt: str) -> str:
    """Remove the final query sentence (ends with '?') from a prompt."""
    if "?" in prompt:
        prefix = prompt.rsplit("?", 1)[0].rstrip()
        return prefix
    return prompt


def build_pretrain_docs(specs: dict, n_per_arm: int):
    """Generate world-dynamics sequences without task queries."""
    docs = []
    # Composite: facts + history, drop the final query.
    for e in TK.generate(specs["composite_p16"], "train", n=n_per_arm):
        docs.append(f"{_strip_query(e.prompt)}.")

    # Recall: just the facts.
    for e in TK.generate(specs["recall_hard"], "train", n=n_per_arm // 2):
        docs.append(f"{_strip_query(e.prompt)}.")

    # Binding: just the history.
    for e in TK.generate(specs["binding"], "train", n=n_per_arm // 2):
        docs.append(f"{_strip_query(e.prompt)}.")
    return docs


def run_pretrain_finetune(arch: str, seed: int, *, d_model, n_layers, batch,
                          pretrain_steps, finetune_steps, pretrain_n, finetune_n,
                          eval_n, device, finetune_output_only, use_staged_finetune,
                          loss_log_interval):
    import torch

    specs = staged_specs()
    base = specs["composite_p16"]
    w, r = TK.build_world(base)
    tok, _, _ = T.prepare([], [], [w], renderer=r)
    d_ff = 4 * d_model

    # ---- Pretrain on world dynamics ----
    pretrain_texts = build_pretrain_docs(specs, n_per_arm=pretrain_n // 3)
    _, pretrain_docs, _ = T.prepare(pretrain_texts, [], [w], renderer=r)
    pre = T.run(
        arch, tok, pretrain_docs, [], steps=pretrain_steps, batch=batch,
        d_model=d_model, n_layers=n_layers, d_ff=d_ff, seed=seed,
        return_model=True, device=device, loss_log_interval=loss_log_interval,
    )
    model = pre["model"]

    # ---- Finetune on tasks ----
    if use_staged_finetune:
        # Three-stage curriculum with output-only masking on composite arms.
        from experiment_curriculum_staged import train_stages
        stage_records, model = train_stages(
            arch, seed,
            [({"binding": 0.5, "recall_easy": 0.5}, int(finetune_steps * 0.4)),
             ({"binding": 0.25, "recall_med": 0.35, "composite_p5": 0.4}, int(finetune_steps * 0.3)),
             ({"binding": 0.15, "recall_hard": 0.25, "composite_p5": 0.25, "composite_p16": 0.35},
              finetune_steps - int(finetune_steps * 0.4) - int(finetune_steps * 0.3))],
            tok, specs, d_model=d_model, n_layers=n_layers, batch=batch,
            train_n=finetune_n, eval_n=eval_n, device=device,
            loss_log_interval=loss_log_interval,
        )
        final_eval = stage_records[-1]["eval"]
    else:
        # Single mix of all arms.
        weights = {"binding": 0.15, "recall_easy": 0.10, "recall_med": 0.10,
                   "recall_hard": 0.15, "composite_p5": 0.25, "composite_p16": 0.25}
        finetune_texts, _ = build_training_docs(specs, weights, finetune_n)
        paired = []
        for text in finetune_texts:
            if "?" in text:
                prompt = text.rsplit("?", 1)[0] + "?"
            else:
                prompt = ""
            full = tok.encode(text, add_eos=True)[:1280]
            p_len = len(tok.encode(prompt))
            paired.append((len(full), full, p_len))
        paired.sort(key=lambda x: x[0])
        finetune_docs = [p[1] for p in paired]
        prompt_lens = [p[2] for p in paired]

        fin = T.run(
            arch, tok, finetune_docs, [], steps=finetune_steps, batch=batch,
            d_model=d_model, n_layers=n_layers, d_ff=d_ff, seed=seed,
            return_model=True, device=device, model=model,
            loss_log_interval=loss_log_interval,
            prompt_lens=prompt_lens if finetune_output_only else None,
        )
        model = fin["model"]
        backend = LocalBackend([w], arch=arch, model=model, tokenizer=tok, device=device)
        final_eval = eval_grid(backend, specs, eval_n=eval_n)

    flat = flatten_eval(final_eval)
    # Strip the model object before JSON serialization.
    pre_serializable = {k: v for k, v in pre.items() if k != "model"}
    del model
    torch.cuda.empty_cache()
    return {"pretrain": pre_serializable, "final_eval": final_eval, "flat_final": flat}


def main():
    ap = argparse.ArgumentParser(description="Pretrain on dynamics, finetune on tasks.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--archs", default="gdp_hybrid")
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pretrain_steps", type=int, default=15000)
    ap.add_argument("--finetune_steps", type=int, default=10000)
    ap.add_argument("--pretrain_n", type=int, default=40000)
    ap.add_argument("--finetune_n", type=int, default=40000)
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loss_log_interval", type=int, default=200)
    ap.add_argument("--finetune_output_only", action="store_true")
    ap.add_argument("--use_staged_finetune", action="store_true")
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    archs = [x.strip() for x in a.archs.split(",")]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = a.out_prefix or f"results/pretrain_finetune_{ts}"
    log_path = Path(f"{prefix}.jsonl")
    md_path = Path(f"{prefix}.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "archs": archs, "d_model": a.d_model, "n_layers": a.n_layers, "batch": a.batch,
        "pretrain_steps": a.pretrain_steps, "finetune_steps": a.finetune_steps,
        "pretrain_n": a.pretrain_n, "finetune_n": a.finetune_n,
        "finetune_output_only": a.finetune_output_only,
        "use_staged_finetune": a.use_staged_finetune,
    }

    lines = [
        "# Pretrain on dynamics + finetune on tasks",
        "",
        f"d_model={a.d_model} n_layers={a.n_layers} batch={a.batch} "
        f"pretrain_steps={a.pretrain_steps} finetune_steps={a.finetune_steps} "
        f"pretrain_n={a.pretrain_n} finetune_n={a.finetune_n}",
        f"finetune_output_only={a.finetune_output_only} use_staged_finetune={a.use_staged_finetune}",
        "",
        "| arch | seed | bind | p5 exact | p16 exact | p16 holder | p16 value | scaffold | pre loss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for arch in archs:
        for seed in a.seeds:
            print(f"\n=== {arch} seed {seed} ===", flush=True)
            try:
                res = run_pretrain_finetune(
                    arch, seed, d_model=a.d_model, n_layers=a.n_layers, batch=a.batch,
                    pretrain_steps=a.pretrain_steps, finetune_steps=a.finetune_steps,
                    pretrain_n=a.pretrain_n, finetune_n=a.finetune_n,
                    eval_n=a.eval_n, device=a.device,
                    finetune_output_only=a.finetune_output_only,
                    use_staged_finetune=a.use_staged_finetune,
                    loss_log_interval=a.loss_log_interval,
                )
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                res = {"error": str(e)}
            rec = {"arch": arch, "seed": seed, **cfg, **res}
            with log_path.open("a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            flat = res.get("flat_final", {})
            pre_loss = res.get("pretrain", {}).get("final_loss", float("nan"))
            lines.append(
                f"| {arch} | {seed} | {flat.get('binding_L16_overall', 0):.2f} | "
                f"{flat.get('composite_p5_L16_overall', 0):.2f} | "
                f"{flat.get('composite_p16_L16_overall', 0):.2f} | "
                f"{flat.get('composite_p16_L16_holder', 0):.2f} | "
                f"{flat.get('composite_p16_L16_value', 0):.2f} | "
                f"{flat.get('composite_p16_scaffolded', 0):.2f} | {pre_loss:.3f} |"
            )
            md_path.write_text("\n".join(lines))
            print(f"    -> {flat}  pre_loss={pre_loss}", flush=True)

    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()
