"""Staged curriculum training for FactWorld composition.

The static mix in ``experiment_curriculum.py`` did not produce signal: splitting
optimization pressure across four arms at once gave worse per-leg performance than
single-task training. This script tries a *staged* curriculum: the model first
specializes on easy sub-circuits (binding + tiny recall pools), then sees larger
recall pools, then small-pool composites, and only at the end faces the full
pool-16 composite. Each stage continues from the previous model.

Default 3-stage schedule (25k steps total):
  phase1: binding:0.5, recall_easy:0.5  (10k steps)
  phase2: binding:0.25, recall_med:0.35, composite_p5:0.4  (7.5k steps)
  phase3: binding:0.15, recall_hard:0.25, composite_p5:0.3, composite_p16:0.3  (7.5k steps)

Recall arms split the pool-size axis:
  recall_easy  pools 2-4
  recall_med   pools 5-8
  recall_hard  pools 12-16

Example smoke test:
    .venv-train/bin/python scripts/experiment_curriculum_staged.py \
        --seeds 0 --steps 3000 --d_model 256 --n_layers 4 --train_n 400

Full run:
    .venv-train/bin/python scripts/experiment_curriculum_staged.py \
        --seeds 0 1 2 --steps 25000 --d_model 512 --n_layers 8
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK, train as T
from factworld.backends import LocalBackend
from factworld.runner import evaluate_task
from sweep import build_docs, prefix_decomp  # noqa: E402

# Shared world so every arm uses the same agent/value vocabulary.
_SHARED = dict(k=32, value_vocab_size=128, seed=0)


def staged_specs():
    """Task specs for training arms and eval.

    v2 port (issue #11 re-measure, 2026-07-10): the binding/composite arms now derive from
    the CANONICAL v2 specs (binding_v2 / composite_copy_v2, last_write_uniform=True) instead
    of the RETIRED recency-defective v1 samplers. Knob parity is automatic: composite_copy_v2
    is k=32/pool16 like v1, and _SHARED re-applies the same k/value_vocab_size/seed. The
    published §5 flagship numbers (gdp 0.747 / fprm 0.253 / transformer 0.005 composite k=32)
    were produced by the v1 arms; re-runs on these v2 arms are the re-measurement.
    """
    binding = TK.CANONICAL["binding_v2"].scaled(**_SHARED, name="curriculum_binding")
    recall_easy = TK.CANONICAL["recall_copy_v1"].scaled(
        **_SHARED,
        memorized_recall=False,
        train_lengths=(2, 3, 4),
        eval_lengths=(4,),
        name="curriculum_recall_easy",
    )
    recall_med = TK.CANONICAL["recall_copy_v1"].scaled(
        **_SHARED,
        memorized_recall=False,
        train_lengths=(5, 8),
        eval_lengths=(8,),
        name="curriculum_recall_med",
    )
    recall_hard = TK.CANONICAL["recall_copy_v1"].scaled(
        **_SHARED,
        memorized_recall=False,
        train_lengths=(12, 16),
        eval_lengths=(16,),
        name="curriculum_recall_hard",
    )
    # Composite arms carry the oracle holder trajectory so we can train with
    # dense per-step supervision via --use_trace.
    composite_p5 = TK.CANONICAL["composite_copy_v2"].scaled(
        **_SHARED, recall_pool=5, worked_trace=True, name="curriculum_composite_p5",
    )
    composite_p16 = TK.CANONICAL["composite_copy_v2"].scaled(
        **_SHARED, recall_pool=16, worked_trace=True, name="curriculum_composite_p16",
    )
    return {
        "binding": binding,
        "recall_easy": recall_easy,
        "recall_med": recall_med,
        "recall_hard": recall_hard,
        "composite_p5": composite_p5,
        "composite_p16": composite_p16,
    }


def parse_schedule(s: str) -> list[tuple[dict[str, float], int]]:
    """Parse a schedule string into [(weights, steps), ...].

    Format: "arm:w,arm:w:steps;arm:w,arm:w:steps;..."
    The last ``:steps`` in each phase gives the phase length.
    Weights within a phase must sum to 1.0.
    """
    phases = []
    known = set(staged_specs())
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        items = [it.strip() for it in part.split(",")]
        if len(items) < 1:
            raise ValueError(f"phase needs at least one arm:steps, got: {part}")
        weights = {}
        if len(items) == 1:
            # Single-arm phase: "arm:w:steps"
            last = items[0]
            if ":" not in last:
                raise ValueError(f"entry must be 'arm:w:steps', got: {last}")
            *weight_toks, steps_str = last.split(":")
            if len(weight_toks) != 2:
                raise ValueError(f"entry must be 'arm:w:steps', got: {last}")
            weights[weight_toks[0].strip()] = float(weight_toks[1].strip())
        else:
            for item in items[:-1]:
                if ":" not in item:
                    raise ValueError(f"arm weight must be 'arm:w', got: {item}")
                key, val = item.rsplit(":", 1)
                weights[key.strip()] = float(val.strip())
            # Last item is the final arm weight plus ':steps'
            last = items[-1]
            if ":" not in last:
                raise ValueError(f"last entry must be 'arm:w:steps', got: {last}")
            *weight_toks, steps_str = last.split(":")
            if len(weight_toks) != 2:
                raise ValueError(f"last entry must be 'arm:w:steps', got: {last}")
            weights[weight_toks[0].strip()] = float(weight_toks[1].strip())
        steps = int(steps_str.strip())
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"phase weights must sum to 1.0, got {total}: {part}")
        unknown = set(weights) - known
        if unknown:
            raise ValueError(f"unknown arms: {unknown}")
        phases.append((weights, steps))
    if not phases:
        raise ValueError("schedule must contain at least one phase")
    return phases


def _build_interleaved_doc(example, renderer):
    """Convert a composite example with a worked trace into explicit per-step Q&A."""
    obj = example.meta.get("obj")
    trace_tokens = TK.content_tokens(example.meta.get("trace", ""))
    if not obj or not trace_tokens:
        return f"{example.prompt} {example.answer}"
    # Reconstruct the composite query and the facts+history prefix.
    query = renderer.render_query("recall", attribute="a0", entity=f"the holder of {obj}")
    if example.prompt.endswith(query):
        facts_history = example.prompt[: -len(query)].rstrip()
    else:
        # Fallback: split at the last '?' if query reconstruction differs.
        facts_history = example.prompt.rsplit("?", 1)[0].rstrip()
        query = renderer.render_query("recall", attribute="a0", entity=f"the holder of {obj}")
    parts = [facts_history]
    for t, holder in enumerate(trace_tokens, start=1):
        q = renderer.render_query("state_easy", target=obj, t=t)
        parts.append(f"{q} {holder}.")
    final_holder = trace_tokens[-1]
    final_value = TK.content_tokens(example.answer)[-1]
    parts.append(f"{query} {final_holder} {final_value}.")
    return " ".join(parts)


def _build_marker_doc(example):
    """Composite trace followed by an explicit marker before the value."""
    trace_tokens = TK.content_tokens(example.meta.get("trace", ""))
    if not trace_tokens:
        return f"{example.prompt} {example.answer}"
    holder = trace_tokens[-1]
    value = TK.content_tokens(example.answer)[-1]
    trace_str = " ".join(trace_tokens)
    return f"{example.prompt} {trace_str} the value is {value}."


def build_training_docs(specs: dict[str, TK.TaskSpec], weights: dict[str, float], train_n: int,
                        use_trace: bool = False, dense_format: str = "trace",
                        renderer=None):
    """Concatenate per-arm docs with counts proportional to mix weights."""
    docs = []
    arm_counts = {}
    for arm, w in weights.items():
        n = max(1, int(round(train_n * w)))
        arm_counts[arm] = n
        examples = TK.generate(specs[arm], "train", n=n)
        if use_trace and arm.startswith("composite"):
            if dense_format == "interleaved":
                docs.extend(_build_interleaved_doc(e, renderer) for e in examples)
            elif dense_format == "marker":
                docs.extend(_build_marker_doc(e) for e in examples)
            else:
                docs.extend(build_docs(examples, use_trace=True))
        else:
            docs.extend(build_docs(examples, use_trace=False))
    return docs, arm_counts


def scaffold_prompt(prompt: str, holder: str | None) -> str:
    if holder is None:
        return prompt
    return f"{prompt} (the holder is {holder})"


def eval_task(backend, spec, *, eval_n: int, length: int, scaffolded: bool = False,
              use_trace: bool = False):
    if not scaffolded:
        # In trace mode the model emits the worked trajectory before the answer,
        # so give it enough generation budget and score the committed tail.
        max_new = (length + 6) if use_trace else None
        res = evaluate_task(backend, spec, split="test", n=eval_n, length=length,
                            max_new_tokens=max_new)
        dec = prefix_decomp(res["examples"], trace_mode=use_trace)
        out = {"overall": res["overall"], **dec}
        for name in ("last_n", "relaxed", "contains"):
            if name in res.get("metrics", {}):
                out[name] = res["metrics"][name]["overall"]
        return out
    examples = TK.generate(spec, "test", n=eval_n, length=length)
    prompts = [scaffold_prompt(e.prompt, e.meta.get("holder")) for e in examples]
    max_new = (length + 6) if use_trace else 8
    preds = backend.generate(prompts, max_new_tokens=max_new, stop_at=".")
    value_hits = n_two = 0
    for e, pred in zip(examples, preds):
        gold_ct = TK.content_tokens(e.answer)
        if len(gold_ct) < 2:
            continue
        n_two += 1
        pred_ct = TK.content_tokens(pred)
        if use_trace and len(pred_ct) >= len(gold_ct):
            pred_ct = pred_ct[-len(gold_ct):]
        if gold_ct[1] in pred_ct:
            value_hits += 1
    return {"scaffolded_value": value_hits / max(1, n_two)}


def eval_grid(backend, specs: dict[str, TK.TaskSpec], *, eval_n: int, use_trace: bool = False):
    """Independent per-task eval."""
    g = {}
    g["binding_L16"] = eval_task(backend, specs["binding"], eval_n=eval_n, length=16,
                                  use_trace=use_trace)
    g["recall_easy_L4"] = eval_task(backend, specs["recall_easy"], eval_n=eval_n, length=4,
                                    use_trace=use_trace)
    g["recall_med_L8"] = eval_task(backend, specs["recall_med"], eval_n=eval_n, length=8,
                                   use_trace=use_trace)
    g["recall_hard_L16"] = eval_task(backend, specs["recall_hard"], eval_n=eval_n, length=16,
                                     use_trace=use_trace)
    g["composite_p5_L16"] = eval_task(backend, specs["composite_p5"], eval_n=eval_n, length=16,
                                      use_trace=use_trace)
    g["composite_p16_L16"] = eval_task(backend, specs["composite_p16"], eval_n=eval_n, length=16,
                                       use_trace=use_trace)
    g["composite_p16_scaffolded"] = eval_task(
        backend, specs["composite_p16"], eval_n=eval_n, length=16, scaffolded=True,
        use_trace=use_trace,
    )
    return g


def train_stages(arch: str, seed: int, schedule, tok, specs, *, d_model, n_layers, n_heads,
                 batch, train_n, eval_n, device, loss_log_interval, lr=1e-3, wandb_project=None,
                 wandb_log_every=1, use_trace: bool = False, dense_format: str = "trace"):
    """Train one model through all curriculum stages, continuing from the previous model."""
    import torch

    d_ff = 4 * d_model
    base = specs["composite_p16"]
    w, r = TK.build_world(base)
    model = None
    stage_records = []
    cumulative_steps = 0
    for phase_idx, (weights, steps) in enumerate(schedule):
        texts, arm_counts = build_training_docs(
            specs, weights, train_n, use_trace=use_trace, dense_format=dense_format,
            renderer=r,
        )
        # Re-tokenize each phase; vocab is identical because world is shared.
        _, docs, _ = T.prepare(texts, [], [w], renderer=r)
        run_name = f"{arch}_seed{seed}_phase{phase_idx}"
        run = T.run(
            arch, tok, docs, [], steps=steps, batch=batch, lr=lr, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_ff=d_ff, seed=seed, return_model=True, device=device,
            model=model, loss_log_interval=loss_log_interval,
            wandb_project=wandb_project, wandb_run_name=run_name, wandb_log_every=wandb_log_every,
            wandb_config={"phase": phase_idx, "phase_weights": weights,
                          "cumulative_steps_before": cumulative_steps,
                          "use_trace": use_trace, "dense_format": dense_format},
        )
        model = run["model"]
        backend = LocalBackend([w], arch=arch, model=model, tokenizer=tok, device=device)
        evals = eval_grid(backend, specs, eval_n=eval_n, use_trace=use_trace)
        stage_records.append({
            "phase": phase_idx,
            "weights": weights,
            "steps": steps,
            "arm_counts": arm_counts,
            "final_loss": run["final_loss"],
            "loss_curve": run.get("loss_curve", []),
            "eval": evals,
        })
        cumulative_steps += steps
    # Final eval is already in the last stage record; return model so caller can clean up.
    return stage_records, model


def flatten_eval(evals: dict) -> dict[str, float]:
    flat = {}
    for key, val in evals.items():
        if "scaffolded" in key:
            flat[key] = val["scaffolded_value"]
        else:
            flat[f"{key}_overall"] = val["overall"]
            flat[f"{key}_holder"] = val.get("holder_acc", 0.0)
            flat[f"{key}_value"] = val.get("value_acc", 0.0)
            for name in ("last_n", "relaxed", "contains"):
                if name in val:
                    flat[f"{key}_{name}"] = val[name]
    return flat


def aggregate(runs):
    by_arch = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for k, v in r["flat_final"].items():
            by_arch[r["arch"]][k].append(v)
    summary = {}
    for arch, metrics in by_arch.items():
        summary[arch] = {
            m: {"mean": statistics.mean(v), "std": statistics.pstdev(v) if len(v) > 1 else 0.0, "n": len(v)}
            for m, v in metrics.items()
        }
    return summary


def write_markdown(summary, cfg, path: Path):
    trace_note = " (dense supervision / use_trace)" if cfg.get("use_trace") else ""
    fmt_note = f" format={cfg.get('dense_format', 'trace')}" if cfg.get("use_trace") else ""
    lines = [
        f"# Staged curriculum{trace_note} — one model, progressive difficulty, per-task eval",
        "",
        f"schedule={cfg['schedule_str']} d_model={cfg['d_model']} n_layers={cfg['n_layers']} "
        f"batch={cfg['batch']} seeds={cfg['seeds']} train_n={cfg['train_n']} "
        f"use_trace={cfg.get('use_trace', False)}{fmt_note}",
        "",
        "## Final per-task eval (mean over seeds)",
        "",
        "| arch | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 last-N | p16 holder | p16 value | p16 scaffold |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arch, m in sorted(summary.items()):
        def g(key):
            return m.get(key, {}).get("mean", float("nan"))
        lines.append(
            f"| {arch} | {g('binding_L16_overall'):.2f} | {g('recall_easy_L4_overall'):.2f} | "
            f"{g('recall_med_L8_overall'):.2f} | {g('recall_hard_L16_overall'):.2f} | "
            f"{g('composite_p5_L16_overall'):.2f} | {g('composite_p16_L16_overall'):.2f} | "
            f"{g('composite_p16_L16_last_n'):.2f} | {g('composite_p16_L16_holder'):.2f} | "
            f"{g('composite_p16_L16_value'):.2f} | {g('composite_p16_scaffolded'):.2f} |"
        )
    lines += [
        "",
        "## Stage details",
        "",
        cfg["schedule_str"],
    ]
    path.write_text("\n".join(lines))


def default_schedule(total_steps: int) -> list[tuple[dict[str, float], int]]:
    """Three-stage default: easy -> medium -> full composite."""
    p1 = int(round(total_steps * 0.40))
    p2 = int(round(total_steps * 0.30))
    p3 = total_steps - p1 - p2
    return [
        ({"binding": 0.5, "recall_easy": 0.5}, p1),
        ({"binding": 0.25, "recall_med": 0.35, "composite_p5": 0.4}, p2),
        ({"binding": 0.15, "recall_hard": 0.25, "composite_p5": 0.3, "composite_p16": 0.3}, p3),
    ]


def schedule_to_str(schedule) -> str:
    return ";".join(
        ",".join(f"{k}:{v}" for k, v in w.items()) + f":{s}" for w, s in schedule
    )


def main():
    # Default schedule scales with --steps, so smoke tests get short phases.
    default_sched = default_schedule(25000)
    default_sched_str = schedule_to_str(default_sched)
    ap = argparse.ArgumentParser(description="Staged curriculum training for composition.")
    ap.add_argument("--schedule", default=default_sched_str,
                    help="Semicolon-separated phases. Each phase: arm:w,arm:w:steps. "
                         "Weights sum to 1 per phase. Use ':steps' to set phase length.")
    ap.add_argument("--phase3", default=None,
                    help="Override phase-3 weights while keeping phases 1-2 and lengths fixed. "
                         "Format: arm:w,arm:w  (e.g. binding:0.25,recall_hard:0.25,composite_p5:0.25,composite_p16:0.25). "
                         "Weights are normalized to sum to 1.")
    ap.add_argument("--archs", default="gdp_hybrid,fprm,transformer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=4,
                    help="Number of attention heads (must divide d_model).")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Peak learning rate for AdamW.")
    ap.add_argument("--train_n", type=int, default=8000,
                    help="Total training examples per phase, split across arms by mix weights.")
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loss_log_interval", type=int, default=200,
                    help="Record training loss every N steps within each phase.")
    ap.add_argument("--wandb_project", default=None,
                    help="Log live training curves to this Weights & Biases project.")
    ap.add_argument("--wandb_log_every", type=int, default=1,
                    help="Log to wandb every N steps (1 = every step).")
    ap.add_argument("--use_trace", action="store_true",
                    help="Train composite arms with oracle holder traces (dense supervision).")
    ap.add_argument("--dense_format", default="trace", choices=["trace", "interleaved", "marker"],
                    help="How to format dense supervision. 'trace' = holder sequence; "
                         "'interleaved' = explicit per-step Q&A; 'marker' = holder sequence + marker + value.")
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    # If the user took the default schedule, rescale phase lengths to match --steps.
    if a.schedule == default_sched_str:
        a.schedule = schedule_to_str(default_schedule(a.steps))
    schedule = parse_schedule(a.schedule)

    if a.phase3 is not None:
        # Override phase-3 weights while preserving phase lengths.
        weights = {}
        for item in a.phase3.split(","):
            key, val = item.strip().split(":")
            weights[key.strip()] = float(val.strip())
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            weights = {k: v / total for k, v in weights.items()}
        phase3_steps = schedule[-1][1]
        schedule[-1] = (weights, phase3_steps)
        a.schedule = schedule_to_str(schedule)

    archs = [x.strip() for x in a.archs.split(",")]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = a.out_prefix or f"results/curriculum_staged_{ts}"
    log_path = Path(f"{prefix}.jsonl")
    md_path = Path(f"{prefix}.md")
    json_path = Path(f"{prefix}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    specs = staged_specs()
    base = specs["composite_p16"]
    w, r = TK.build_world(base)
    tok, _, _ = T.prepare([], [], [w], renderer=r)

    cfg = {
        "schedule_str": a.schedule, "schedule": [(w, s) for w, s in schedule],
        "archs": archs, "seeds": a.seeds, "d_model": a.d_model, "n_layers": a.n_layers,
        "batch": a.batch, "train_n": a.train_n, "eval_n": a.eval_n,
        "loss_log_interval": a.loss_log_interval, "use_trace": a.use_trace,
        "dense_format": a.dense_format,
    }
    runs = []
    total = len(archs) * len(a.seeds)
    print(f"=== staged curriculum: {total} runs, schedule={a.schedule} -> {log_path} ===", flush=True)

    for i, arch in enumerate(archs):
        for seed in a.seeds:
            tag = f"{arch} seed {seed}"
            print(f"\n--- [{len(runs)+1}/{total}] {tag} ---", flush=True)
            try:
                stage_records, model = train_stages(
                    arch, seed, schedule, tok, specs, d_model=a.d_model, n_layers=a.n_layers,
                    n_heads=a.n_heads, batch=a.batch, lr=a.lr, train_n=a.train_n, eval_n=a.eval_n,
                    device=a.device, loss_log_interval=a.loss_log_interval,
                    wandb_project=a.wandb_project, wandb_log_every=a.wandb_log_every,
                    use_trace=a.use_trace, dense_format=a.dense_format,
                )
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                result = {"error": str(e)}
            else:
                import torch
                final_eval = stage_records[-1]["eval"]
                flat_final = flatten_eval(final_eval)
                result = {
                    "stage_records": stage_records,
                    "final_loss": stage_records[-1]["final_loss"],
                    "final_eval": final_eval,
                    "flat_final": flat_final,
                }
                del model
                torch.cuda.empty_cache()

            rec = {"arch": arch, "seed": seed, **cfg, **result}
            with log_path.open("a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            if "flat_final" in result:
                runs.append(rec)
                print(f"    -> {result['flat_final']}  loss={result.get('final_loss', 'n/a')}", flush=True)

            summary = aggregate(runs)
            write_markdown(summary, cfg, md_path)
            json_path.write_text(json.dumps({"cfg": cfg, "summary": summary, "runs": runs}, indent=2))

    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()
