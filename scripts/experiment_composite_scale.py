"""Compute-matched scale sweep for FactWorld composition.

The headline local result (reports/factworld-consolidated.md §5) compared `gdp_hybrid`, `fprm`,
and `transformer` at matched `(d_model, n_layers)` and called all three "~40M params". That label
is misleading on two counts this sweep corrects:

  1. **The comparison is compute-matched, not param-matched.** `fprm` is a weight-tied looped block
     (`factworld/models.py:239`): one `FPRMBlock` applied `n_loops` times. At matched `(d_model,
     depth)` it does roughly the same number of block-applications as the others, so its per-token
     FLOPs match — but its *parameters* are far fewer (the weights are reused). Measured at
     d=256/depth=4: transformer 9.59M FLOPs/token & 4.47M params vs fprm 9.60M FLOPs/token &
     1.33M params. Same compute, ~3.4× fewer params — a property of the architecture, not an
     unfairness. The honest axis is FLOPs.
  2. **The §5 "~40M" was actually 10M / 76M / 101M** (fprm / transformer / gdp at d=768/L=8). We
     report the measured params AND FLOPs for every cell so neither number is hidden.

So this sweep varies model size by scaling `(d_model, depth)` together for all three architectures
— which empirically keeps per-token FLOPs within ~1.25× across archs (GDP's Householder product is
the only meaningful add) — and asks whether the §5 ranking survives scale. It reuses the staged-
curriculum machinery from ``experiment_curriculum_staged`` verbatim (same specs, schedule, per-leg
decomposition); only model size varies.

The scored number is the canonical relaxed match on `composite_p16@L16`, with the holder (binding)
and value (recall) legs reported alongside (the routing decomposition).

Crash-safe: each completed (scale, arch, seed) run is appended to a JSONL log as it finishes, and
the markdown + JSON summary are rewritten after every aggregation pass, so an interrupt or an OOM
at the largest scale keeps the smaller-scale results.

Example smoke test (proves the wiring):
    .venv-train/bin/python scripts/experiment_composite_scale.py \\
        --scales smoke --archs transformer --seeds 0

Real sweep (compute-matched ladder, multi-seed):
    .venv-train/bin/python scripts/experiment_composite_scale.py \\
        --scales small medium large --archs gdp_hybrid,fprm,transformer --seeds 0 1
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

from factworld import tasks as TK, train as T  # noqa: E402

# Reuse the frozen staged-curriculum machinery so this sweep differs from §5 ONLY in model size.
from experiment_curriculum_staged import (  # noqa: E402
    aggregate,
    default_schedule,
    flatten_eval,
    parse_schedule,
    schedule_to_str,
    staged_specs,
    train_stages,
)

# Parameter buckets. (d_model, n_layers, batch, steps). d_model must be divisible by n_heads (4).
# batch=128 matches the §5 recipe (reports/factworld-consolidated.md §5) for small/medium; large uses
# 64 to stay memory-safe on the 268M gdp model. The script prints the MEASURED param count + FLOPs per
# arch so the buckets are auditable. `steps` is the total curriculum budget (phase lengths scale off it).
SCALES = {
    "smoke":   dict(d_model=48,  n_layers=2,  batch=8,   steps=240,  train_n=400,  eval_n=16),
    "xsmall":  dict(d_model=256, n_layers=4,  batch=128, steps=12000, train_n=8000, eval_n=100),
    "small":   dict(d_model=384, n_layers=6,  batch=128, steps=20000, train_n=80000, eval_n=200),
    "medium":  dict(d_model=768, n_layers=8,  batch=128, steps=25000, train_n=80000, eval_n=200),
    "large":   dict(d_model=1024, n_layers=12, batch=64, steps=25000, train_n=80000, eval_n=200),
}


def measured_params(arch: str, d_model: int, n_layers: int, n_heads: int, vocab_size: int, device: str) -> int:
    """Build a fresh model at this size only to count parameters (no training)."""
    import torch
    from factworld.models import build_model

    d_ff = 4 * d_model
    torch.manual_seed(0)
    m = build_model(
        arch, vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
    ).to(device)
    n = m.num_params()
    del m
    torch.cuda.empty_cache()
    return n


# Representative sequence length for the FLOPs measurement (composite prompts at pool-16 / L16).
FLOPS_SEQ_LEN = 256


def measured_flops(arch: str, d_model: int, n_layers: int, n_heads: int, vocab_size: int,
                   device: str, seq_len: int = FLOPS_SEQ_LEN) -> int:
    """Forward FLOPs/token via torch's flop counter (fla layers counted; autocast bf16 — GDP/GDN
    chunk kernels require bf16). This is the compute axis that is actually matched at fixed
    (d_model, depth) across the weight-tied fprm and the untied stacks."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from factworld.models import build_model

    d_ff = 4 * d_model
    torch.manual_seed(0)
    m = build_model(
        arch, vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
    ).to(device).eval()
    ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.bfloat16):
            m(ids)  # warmup (kernel compile / autotune)
    fcm = FlopCounterMode(display=False)
    with fcm:
        with torch.no_grad():
            with torch.autocast(device, dtype=torch.bfloat16):
                m(ids)
    total = fcm.get_total_flops()
    del m
    torch.cuda.empty_cache()
    return total // seq_len  # per-token forward FLOPs


def measured_size(arch, d_model, n_layers, n_heads, vocab_size, device):
    """(params, per-token forward FLOPs) — built once per (scale, arch), reused for every seed."""
    return (
        measured_params(arch, d_model, n_layers, n_heads, vocab_size, device),
        measured_flops(arch, d_model, n_layers, n_heads, vocab_size, device),
    )


def run_one(scale_name, scale_cfg, arch, seed, schedule, tok, specs, *, n_heads, lr, device,
            loss_log_interval):
    import torch

    d_model = scale_cfg["d_model"]
    n_layers = scale_cfg["n_layers"]
    batch = scale_cfg["batch"]
    steps = scale_cfg["steps"]
    train_n = scale_cfg["train_n"]
    eval_n = scale_cfg["eval_n"]

    stage_records, model = train_stages(
        arch, seed, schedule, tok, specs,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, batch=batch, lr=lr,
        train_n=train_n, eval_n=eval_n, device=device, loss_log_interval=loss_log_interval,
    )
    nparams = model.num_params()
    flat_final = flatten_eval(stage_records[-1]["eval"])
    del model
    torch.cuda.empty_cache()
    return {
        "n_params": nparams,
        "final_loss": stage_records[-1]["final_loss"],
        "flat_final": flat_final,
        "stage_records": stage_records,
    }


def summary_by_scale_arch(runs):
    """mean ± std (across seeds) for the key cells, keyed by (scale, arch)."""
    bucket = defaultdict(lambda: defaultdict(list))   # [(scale,arch)][metric] -> [values]
    params = {}
    for r in runs:
        key = (r["scale"], r["arch"])
        params[key] = r["n_params"]
        for m, v in r["flat_final"].items():
            bucket[key][m].append(v)
    out = {}
    for key, metrics in bucket.items():
        out[key] = {
            m: {"mean": statistics.mean(v), "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
                "n": len(v)}
            for m, v in metrics.items()
        }
    return out, params


def _stringify_keys(d):
    """json.dumps rejects tuple keys; collapse (scale, arch) -> 'scale|arch'."""
    return {f"{k[0]}|{k[1]}": v for k, v in d.items()}


def write_markdown(summary, size_map, cfg, path: Path):
    scales = cfg["scales"]
    archs = cfg["archs"]
    headline = "composite_p16_L16_overall"   # canonical relaxed match on the flagship cell

    def cell(scale, arch, metric=headline):
        s = summary.get((scale, arch), {})
        if metric not in s:
            return "—"
        d = s[metric]
        return f"{d['mean']:.2f}±{d['std']:.2f}"

    lines = [
        "# FactWorld composition — compute-matched scale sweep",
        "",
        "Same staged curriculum and eval as `reports/factworld-consolidated.md` §5, varied ONLY in "
        "model size. **The match is on compute, not parameters.** All architectures share "
        "(d_model, depth); `fprm` is weight-tied (one block looped `n_loops` times), so at matched "
        "(d_model, depth) its per-token FLOPs equal the transformer's while its parameter count is "
        "~5–11× lower across scales (≈8× at medium). The size table below reports both axes so neither is hidden. Score = canonical "
        "**relaxed match** on `composite_copy_v2` pool-16 @L16, with the holder (binding) and value "
        "(recall) legs reported alongside (the routing decomposition).",
        "",
        f"seeds={cfg['seeds']}  n_heads={cfg['n_heads']}  "
        f"schedule=default_schedule(<scale steps>) — 3 phases: binding+recall_easy → "
        f"+recall_med+composite_p5 → +recall_hard+composite_p16. Per-scale "
        f"(d_model, n_layers, batch, steps, train_n, eval_n):",
        "",
    ]
    for s in scales:
        c = SCALES[s]
        lines.append(
            f"  - **{s}**: d_model={c['d_model']} n_layers={c['n_layers']} batch={c['batch']} "
            f"steps={c['steps']} train_n={c['train_n']} eval_n={c['eval_n']}"
        )
    lines += [
        "",
        f"## Measured size — params / per-token forward FLOPs (seq_len={FLOPS_SEQ_LEN}; bf16)",
        "",
        "Params counted with tied head/embed counted once. Per-token FLOPs from torch's flop counter "
        "(captures the `fla` layers; GDP runs ~1.2× the transformer from the Householder product). "
        "Read each cell as `params · FLOPs/token`. Within a scale column the FLOPs are matched; the "
        "param differences are architectural (fprm weight-tied).",
        "",
        "| arch | " + " | ".join(scales) + " |",
        "| " + " | ".join(["---"] * (len(scales) + 1)) + " |",
    ]
    for arch in archs:
        cells = []
        for s in scales:
            sz = size_map.get((s, arch))
            cells.append(f"{sz[0]/1e6:.1f}M · {sz[1]/1e6:.2f}G" if sz else "—")
        lines.append(f"| {arch} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## `composite_copy_v2` pool-16 @L16 — relaxed match (mean±std over seeds)",
        "",
        "| arch | " + " | ".join(scales) + " |",
        "| " + " | ".join(["---"] * (len(scales) + 1)) + " |",
    ]
    for arch in archs:
        lines.append(f"| {arch} | " + " | ".join(cell(s, arch) for s in scales) + " |")

    lines += [
        "",
        "### holder leg (binding) — mean±std",
        "",
        "| arch | " + " | ".join(scales) + " |",
        "| " + " | ".join(["---"] * (len(scales) + 1)) + " |",
    ]
    for arch in archs:
        lines.append(
            f"| {arch} | "
            + " | ".join(cell(s, arch, "composite_p16_L16_holder") for s in scales) + " |"
        )

    lines += [
        "",
        "### value leg (recall of the resolved holder) — mean±std",
        "",
        "| arch | " + " | ".join(scales) + " |",
        "| " + " | ".join(["---"] * (len(scales) + 1)) + " |",
    ]
    for arch in archs:
        lines.append(
            f"| {arch} | "
            + " | ".join(cell(s, arch, "composite_p16_L16_value") for s in scales) + " |"
        )

    lines += [
        "",
        "## Reading it",
        "",
        "- This is a **compute-matched** comparison (matched d_model + depth ≈ matched per-token "
        "FLOPs; see the size table). Parameter counts differ by design — `fprm` is weight-tied.",
        "- If `gdp_hybrid` stays ahead as compute grows, the §5 ranking is scale-robust.",
        "- If the transformer closes the gap at the largest size, §5 is regime-scoped to small "
        "models and the headline should say so.",
        "- The holder/value split localizes any change: a scale effect on `overall` that tracks "
        "`value` is a recall-capacity gain; one that tracks `holder` is a state-tracking gain.",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote markdown to {path}")


def main():
    ap = argparse.ArgumentParser(description="Scale sweep for FactWorld composition.")
    ap.add_argument("--scales", default="small,medium,large",
                    help=f"Comma-separated scale buckets. One of: {sorted(SCALES)}.")
    ap.add_argument("--archs", default="gdp_hybrid,fprm,transformer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n_heads", type=int, default=4, help="Attention heads (must divide d_model).")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override total curriculum steps for ALL scales (phase lengths rescale).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loss_log_interval", type=int, default=200)
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    scales = [s.strip() for s in a.scales.split(",")]
    unknown = [s for s in scales if s not in SCALES]
    if unknown:
        raise SystemExit(f"unknown scales {unknown}; choose from {sorted(SCALES)}")
    archs = [x.strip() for x in a.archs.split(",")]

    # The 3-stage schedule is built PER scale off default_schedule(scale_steps), so each bucket's
    # phase lengths match its own step budget — medium's default_schedule(25000) reproduces the §5
    # schedule (10000 / 7500 / 7500) exactly. A global --steps override replaces every bucket's steps.
    specs = staged_specs()
    base = specs["composite_p16"]
    w, r = TK.build_world(base)
    tok, _, _ = T.prepare([], [], [w], renderer=r)

    def scale_cfg_for(s):
        sc = dict(SCALES[s])
        if a.steps is not None:
            sc["steps"] = a.steps
        return sc

    cfg = {
        "scales": scales, "archs": archs, "seeds": a.seeds, "n_heads": a.n_heads,
        "lr": a.lr, "scale_cfg": {s: scale_cfg_for(s) for s in scales},
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = a.out_prefix or f"results/composite_scale_{ts}"
    log_path = Path(f"{prefix}.jsonl")
    md_path = Path(f"{prefix}.md")
    json_path = Path(f"{prefix}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Measure params + per-token FLOPs once per (scale, arch). The buckets are COMPUTE-matched
    # (shared d_model + depth); fprm is weight-tied so its params are far lower for ~equal FLOPs.
    # Reporting both is the whole point — neither axis is hidden.
    size_map: dict[tuple[str, str], tuple[int, int]] = {}
    print("=== measured size: params / per-token forward FLOPs (the compute axis) ===", flush=True)
    for s in scales:
        sc = SCALES[s]
        if a.steps is not None:
            sc = {**sc, "steps": a.steps}
        parts = []
        for arch in archs:
            p, f = measured_size(arch, sc["d_model"], sc["n_layers"], a.n_heads, tok.vocab_size, a.device)
            size_map[(s, arch)] = (p, f)
            parts.append(f"{arch}={p/1e6:.1f}M/{f/1e6:.2f}G")
        print(f"  {s} (d={sc['d_model']} L={sc['n_layers']}): " + "  ".join(parts), flush=True)

    runs = []
    total = len(scales) * len(archs) * len(a.seeds)
    print(f"\n=== compute-matched scale sweep: {total} runs -> {log_path} ===", flush=True)

    for s in scales:
        sc = scale_cfg_for(s)
        schedule = parse_schedule(schedule_to_str(default_schedule(sc["steps"])))
        for arch in archs:
            for seed in a.seeds:
                tag = f"{s}/{arch}/seed{seed}"
                print(f"\n--- [{len(runs)+1}/{total}] {tag} ---", flush=True)
                try:
                    result = run_one(
                        s, sc, arch, seed, schedule, tok, specs, n_heads=a.n_heads, lr=a.lr,
                        device=a.device, loss_log_interval=a.loss_log_interval,
                    )
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    result = {"error": str(e)}
                else:
                    result["flops_per_tok"] = size_map[(s, arch)][1]
                    runs.append({"scale": s, "arch": arch, "seed": seed, **result})
                    print(
                        f"    -> params={result['n_params']/1e6:.1f}M  "
                        f"flops={result['flops_per_tok']/1e6:.2f}G/tok  "
                        f"comp_p16@L16 relaxed={result['flat_final'].get('composite_p16_L16_overall'):.3f}  "
                        f"holder={result['flat_final'].get('composite_p16_L16_holder'):.3f}  "
                        f"value={result['flat_final'].get('composite_p16_L16_value'):.3f}  "
                        f"loss={result.get('final_loss', 'n/a')}",
                        flush=True,
                    )

                rec = {"scale": s, "arch": arch, "seed": seed, **cfg, **result}
                with log_path.open("a") as f:
                    f.write(json.dumps(rec, default=float) + "\n")

                summary, _ = summary_by_scale_arch([r for r in runs if "flat_final" in r])
                write_markdown(summary, size_map, cfg, md_path)
                json_path.write_text(
                    json.dumps({"cfg": cfg, "summary": _stringify_keys(summary),
                                "size": _stringify_keys(size_map),
                                "runs": runs}, indent=2, default=float)
                )

    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()
